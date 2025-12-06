"""GPU memory management utilities."""
import os

from whisperjav.utils.logger import logger


def safe_cuda_cleanup():
    """
    Clear CUDA cache if safe to do so.

    Skips in subprocess workers where:
    - Process exit will free GPU memory anyway
    - torch.cuda.empty_cache() can crash on Windows

    This function centralizes CUDA cache cleanup logic to avoid
    duplicating subprocess detection across multiple ASR modules.
    """
    if os.environ.get('WHISPERJAV_SUBPROCESS_WORKER') == '1':
        logger.debug("Skipping CUDA cache clear in subprocess (freed on exit)")
        return

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
    except Exception as e:
        logger.debug(f"CUDA cache clear failed (non-fatal): {e}")
