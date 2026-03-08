"""Resilient model loading with network fallback.

Wraps faster-whisper's WhisperModel to gracefully handle network errors
(SSL failures, timeouts, proxy issues) by falling back to locally cached
models when available. This is critical for users behind corporate proxies
or Chinese VPN services (e.g., v2rayN) where SSL certificate validation
often fails even though the model cache is already complete.

See: GitHub issue #204
"""

import ssl
from typing import Optional, Union

from whisperjav.utils.logger import logger


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
    # Walk the exception chain (including __cause__ and __context__)
    current = error
    seen = set()
    while current and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ssl.SSLError, ConnectionError, TimeoutError, OSError)):
            return True
        error_str = str(current).lower()
        if any(indicator in error_str for indicator in _NETWORK_ERROR_INDICATORS):
            return True
        # Check wrapped exceptions
        current = current.__cause__ or current.__context__
    return False


def load_faster_whisper_model(
    model_size_or_path: str,
    device: str = "auto",
    compute_type: str = "default",
    cpu_threads: int = 0,
    num_workers: int = 1,
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
):
    """Load a faster-whisper WhisperModel with network error fallback.

    On network/SSL errors, automatically retries with local_files_only=True
    to use cached models. This prevents proxy/VPN SSL issues from blocking
    users who already have the model downloaded.

    Args:
        Same as faster_whisper.WhisperModel.__init__

    Returns:
        A WhisperModel instance.

    Raises:
        The original exception if both online and offline loading fail.
    """
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(
            model_size_or_path=model_size_or_path,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            download_root=download_root,
            local_files_only=local_files_only,
            **kwargs,
        )
    except Exception as e:
        if local_files_only or not _is_network_error(e):
            raise

        logger.warning(
            f"Network error during model download: {e}. "
            "Attempting to load from local cache..."
        )

        try:
            model = WhisperModel(
                model_size_or_path=model_size_or_path,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
                download_root=download_root,
                local_files_only=True,
                **kwargs,
            )
            logger.info(
                "Model loaded from local cache (offline). "
                "Network issues did not affect model loading."
            )
            return model
        except Exception:
            # Cache miss or corrupted cache — re-raise the original network error
            # so the user sees the actual problem (SSL/proxy) rather than a
            # confusing "file not found in cache" message
            logger.error(
                "Model not found in local cache. The network error prevents downloading. "
                "Please check your internet connection or proxy settings, "
                "or download the model manually."
            )
            raise e
