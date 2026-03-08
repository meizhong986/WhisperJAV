"""Resilient HuggingFace Hub downloads with network fallback.

Monkeypatches huggingface_hub.snapshot_download() and hf_hub_download()
to gracefully handle network errors (SSL failures, timeouts, proxy issues)
with a 3-step fallback strategy:

    1. Normal download from huggingface.co
    2. Load from local cache (local_files_only=True)
    3. Download from hf-mirror.com (official China mirror)

This is critical for users behind corporate proxies or Chinese VPN
services (e.g., v2rayN, Clash) where SSL certificate validation often
fails even though the model cache is already complete.

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

# Widely-used community mirror for HuggingFace in China
_HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

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


def _get_cache_dir():
    """Get the HuggingFace Hub cache directory path for diagnostics."""
    try:
        from huggingface_hub import constants
        return constants.HF_HUB_CACHE
    except Exception:
        return "(unknown)"


def _make_resilient_wrapper(original_fn, fn_name):
    """Create a resilient wrapper for a HuggingFace Hub download function.

    3-step fallback on SSL/network errors:
      1. Try normal download from huggingface.co
      2. Try loading from local cache (local_files_only=True)
      3. Try downloading from hf-mirror.com (official China mirror)

    Works for both snapshot_download and hf_hub_download.
    """

    def _resilient_wrapper(*args, **kwargs):
        # If already requesting local-only, don't wrap
        if kwargs.get("local_files_only"):
            return original_fn(*args, **kwargs)

        try:
            return original_fn(*args, **kwargs)
        except Exception as e:
            if not _is_network_error(e):
                raise

            # Extract identifier for messaging (first positional arg or kwarg)
            resource_id = args[0] if args else kwargs.get("repo_id", "unknown")
            hf_url = f"https://huggingface.co/{resource_id}"
            cache_dir = _get_cache_dir()

            logger.warning(
                "[HF Download] Step 1 FAILED — network/SSL error downloading "
                "'%s' from %s",
                resource_id, hf_url,
            )
            logger.warning(
                "[HF Download] Error: %s: %s",
                type(e).__name__, e,
            )

            # --- Step 2: Try local cache ---
            logger.info(
                "[HF Download] Step 2 — checking local cache at: %s",
                cache_dir,
            )
            try:
                result = original_fn(
                    *args, **{**kwargs, "local_files_only": True}
                )
                logger.info(
                    "[HF Download] Step 2 OK — loaded '%s' from local cache. "
                    "Network issues did not affect model loading.",
                    resource_id,
                )
                return result
            except Exception:
                logger.warning(
                    "[HF Download] Step 2 FAILED — '%s' not found in local "
                    "cache.",
                    resource_id,
                )

            # --- Step 3: Try China mirror ---
            mirror_url = f"{_HF_MIRROR_ENDPOINT}/{resource_id}"
            logger.info(
                "[HF Download] Step 3 — trying China mirror: %s",
                mirror_url,
            )
            try:
                result = original_fn(
                    *args,
                    **{**kwargs, "endpoint": _HF_MIRROR_ENDPOINT},
                )
                logger.info(
                    "[HF Download] Step 3 OK — downloaded '%s' from mirror "
                    "(%s). Model is now cached locally for future use.",
                    resource_id, _HF_MIRROR_ENDPOINT,
                )
                return result
            except Exception as mirror_err:
                logger.error(
                    "[HF Download] Step 3 FAILED — mirror download also "
                    "failed: %s: %s",
                    type(mirror_err).__name__, mirror_err,
                )

            # --- All steps failed: comprehensive diagnostics ---
            logger.error(
                "[HF Download] All download methods failed for '%s'. "
                "Diagnostic summary:",
                resource_id,
            )
            logger.error(
                "  Model:      %s", resource_id,
            )
            logger.error(
                "  Source URL:  %s", hf_url,
            )
            logger.error(
                "  Mirror URL: %s", mirror_url,
            )
            logger.error(
                "  Cache dir:  %s", cache_dir,
            )
            logger.error(
                "  Error type: %s", type(e).__name__,
            )
            logger.error(
                "  To download manually, visit: %s", mirror_url,
            )
            logger.error(
                "  Then place the downloaded files in: %s", cache_dir,
            )
            logger.error(
                "  Or set environment variable: "
                "HF_ENDPOINT=https://hf-mirror.com"
            )
            raise e  # Re-raise original network error

    _resilient_wrapper.__name__ = f"_resilient_{fn_name}"
    _resilient_wrapper.__qualname__ = f"_resilient_{fn_name}"
    return _resilient_wrapper


def patch_hf_hub_downloads():
    """Monkeypatch huggingface_hub download functions with network resilience.

    Patches both snapshot_download (used by faster-whisper for model repos)
    and hf_hub_download (used for individual file downloads like NeMo, LLM
    GGUF files, classifiers).

    On SSL/connection/timeout errors, uses a 3-step fallback with full
    diagnostic logging at each step:

    1. Normal download fails     -> log error type and source URL
    2. Try local cache           -> success: continue; fail: try mirror
    3. Try hf-mirror.com (China) -> success: model cached for future use
    4. All failed                -> comprehensive diagnostic summary with
                                    manual download instructions
    """
    global _patched
    if _patched:
        return

    try:
        import huggingface_hub
    except ImportError:
        return  # HF Hub not installed, nothing to patch

    # Patch snapshot_download (used by faster-whisper, stable-ts)
    _patched_snapshot = _make_resilient_wrapper(
        huggingface_hub.snapshot_download, "snapshot_download"
    )
    huggingface_hub.snapshot_download = _patched_snapshot

    # Patch hf_hub_download (used by NeMo VAD, local LLM backend, classifiers)
    _patched_hf_hub = _make_resilient_wrapper(
        huggingface_hub.hf_hub_download, "hf_hub_download"
    )
    huggingface_hub.hf_hub_download = _patched_hf_hub

    # Also patch faster_whisper.utils which imports huggingface_hub at module level
    try:
        import faster_whisper.utils as fw_utils
        if hasattr(fw_utils, "huggingface_hub"):
            fw_utils.huggingface_hub.snapshot_download = _patched_snapshot
            fw_utils.huggingface_hub.hf_hub_download = _patched_hf_hub
    except ImportError:
        pass

    _patched = True
    logger.debug("HuggingFace Hub download resilience patch applied")
