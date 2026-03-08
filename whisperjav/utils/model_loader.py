"""Resilient HuggingFace Hub downloads with network fallback.

Monkeypatches huggingface_hub.snapshot_download() to gracefully handle
network errors (SSL failures, timeouts, proxy issues) by falling back
to locally cached models when available.

This is critical for users behind corporate proxies or Chinese VPN
services (e.g., v2rayN) where SSL certificate validation often fails
even though the model cache is already complete.

Architecture: A single monkeypatch applied once at startup protects ALL
code paths that download from HuggingFace — faster-whisper, stable-ts,
speech enhancement models, etc. No per-call-site wrappers needed.

Usage:
    # Call once at application startup (cli.py, webview_gui/main.py)
    from whisperjav.utils.model_loader import patch_hf_hub_downloads
    patch_hf_hub_downloads()

See: GitHub issue #204
"""

import ssl

from whisperjav.utils.logger import logger

_patched = False

# Error strings that indicate a network/SSL problem (not a model/CUDA problem)
_NETWORK_ERROR_INDICATORS = [
    "ssl",
    "certificate",
    "urlopen error",
    "connection",
    "timeout",
    "network",
    "socket",
    "getaddrinfo",
    "name or service not known",
    "temporary failure in name resolution",
    "connectionreseterror",
    "remotedisconnected",
]


def _is_network_error(error: Exception) -> bool:
    """Check if an exception is caused by a network/SSL problem."""
    _NON_NETWORK_OS_ERRORS = (
        FileNotFoundError,
        FileExistsError,
        IsADirectoryError,
        NotADirectoryError,
        PermissionError,
    )

    current = error
    seen = set()
    while current and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _NON_NETWORK_OS_ERRORS):
            current = current.__cause__ or current.__context__
            continue
        if isinstance(current, (ssl.SSLError, ConnectionError, TimeoutError)):
            return True
        if isinstance(current, OSError):
            error_str = str(current).lower()
            if any(ind in error_str for ind in _NETWORK_ERROR_INDICATORS):
                return True
            current = current.__cause__ or current.__context__
            continue
        error_str = str(current).lower()
        if any(ind in error_str for ind in _NETWORK_ERROR_INDICATORS):
            return True
        current = current.__cause__ or current.__context__
    return False


def patch_hf_hub_downloads():
    """Monkeypatch huggingface_hub.snapshot_download with network resilience.

    On SSL/connection/timeout errors, automatically retries with
    local_files_only=True to use cached models. Provides clear user
    messages for each scenario:

    1. SSL fail + cache hit  -> warning + continues normally
    2. SSL fail + no cache   -> error with actionable guidance
    """
    global _patched
    if _patched:
        return

    try:
        import huggingface_hub
    except ImportError:
        return  # HF Hub not installed, nothing to patch

    _original_snapshot_download = huggingface_hub.snapshot_download

    def _resilient_snapshot_download(*args, **kwargs):
        # If already requesting local-only, don't wrap
        if kwargs.get("local_files_only"):
            return _original_snapshot_download(*args, **kwargs)

        try:
            return _original_snapshot_download(*args, **kwargs)
        except Exception as e:
            if not _is_network_error(e):
                raise

            # Extract repo_id for messaging (first positional arg or kwarg)
            repo_id = args[0] if args else kwargs.get("repo_id", "unknown")

            logger.warning(
                "Network/SSL error while checking '%s' on HuggingFace: %s",
                repo_id, e,
            )
            logger.warning(
                "Attempting to load from local cache..."
            )

            try:
                result = _original_snapshot_download(
                    *args, **{**kwargs, "local_files_only": True}
                )
                logger.info(
                    "Loaded '%s' from local cache. "
                    "Network issues did not affect model loading.",
                    repo_id,
                )
                return result
            except Exception:
                logger.error(
                    "Model '%s' not found in local cache. "
                    "A working internet connection is required for first-time "
                    "model download. Please check your VPN/proxy settings, "
                    "or try disconnecting your VPN and downloading the model "
                    "once with a direct connection.",
                    repo_id,
                )
                raise e  # Re-raise original network error

    # Apply the patch to the module AND to faster_whisper.utils which
    # may have already imported snapshot_download at module load time
    huggingface_hub.snapshot_download = _resilient_snapshot_download

    try:
        import faster_whisper.utils as fw_utils
        if hasattr(fw_utils, "huggingface_hub"):
            fw_utils.huggingface_hub.snapshot_download = _resilient_snapshot_download
    except ImportError:
        pass

    _patched = True
    logger.debug("HuggingFace Hub download resilience patch applied")
