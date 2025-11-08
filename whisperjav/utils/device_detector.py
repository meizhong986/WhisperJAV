#!/usr/bin/env python3
"""
Smart device detection for multi-platform GPU acceleration.

Automatically detects and selects the best available compute device:
- NVIDIA CUDA (highest priority for compatibility)
- Apple Silicon MPS (Metal Performance Shaders)
- AMD ROCm (detection only, limited support)
- CPU (fallback)

This module enables WhisperJAV to run on:
- NVIDIA GPUs (RTX 20/30/40/50 series, Blackwell, etc.)
- Apple M1/M2/M3/M4/M5 chips
- AMD GPUs (detection only, defer to CPU)
- CPU-only systems
"""

import sys
import platform
from typing import Dict, Optional, Tuple
import logging

from whisperjav.utils.logger import logger


def _check_cuda_available() -> Tuple[bool, Optional[str]]:
    """
    Check if CUDA is available and get GPU name.

    Returns:
        (is_available, gpu_name)
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, gpu_name
        return False, None
    except Exception as e:
        logger.debug(f"CUDA check failed: {e}")
        return False, None


def _check_mps_available() -> Tuple[bool, Optional[str]]:
    """
    Check if Apple Metal Performance Shaders (MPS) is available.

    Returns:
        (is_available, chip_name)
    """
    # MPS only available on macOS
    if platform.system() != 'Darwin':
        return False, None

    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Try to get chip name from platform
            try:
                chip_info = platform.processor() or platform.machine()
                return True, f"Apple Silicon ({chip_info})"
            except:
                return True, "Apple Silicon"
        return False, None
    except Exception as e:
        logger.debug(f"MPS check failed: {e}")
        return False, None


def _check_rocm_available() -> Tuple[bool, Optional[str]]:
    """
    Check if AMD ROCm is available.

    Note: ROCm detection only. WhisperJAV currently defers to CPU
    due to CTranslate2 dependency limitations.

    Returns:
        (is_available, gpu_name)
    """
    try:
        import torch
        # ROCm builds of PyTorch use 'cuda' backend but with AMD GPUs
        if torch.cuda.is_available():
            # Check if this is actually ROCm (not NVIDIA CUDA)
            gpu_name = torch.cuda.get_device_name(0)
            if 'AMD' in gpu_name.upper() or 'RADEON' in gpu_name.upper():
                return True, gpu_name
        return False, None
    except Exception as e:
        logger.debug(f"ROCm check failed: {e}")
        return False, None


def get_best_device(prefer_cpu: bool = False) -> str:
    """
    Auto-detect and return the best available compute device.

    Priority order:
    1. CUDA (NVIDIA GPUs) - highest compatibility
    2. MPS (Apple Silicon) - native macOS GPU
    3. CPU (fallback or explicit preference)

    Note: ROCm (AMD GPUs) detected but deferred to CPU due to
    CTranslate2 dependency limitations in faster-whisper pipeline.

    Args:
        prefer_cpu: Force CPU mode even if GPU available

    Returns:
        Device string: "cuda", "mps", or "cpu"

    Example:
        >>> device = get_best_device()
        >>> model.to(device)
    """
    if prefer_cpu:
        logger.debug("CPU mode explicitly requested")
        return "cpu"

    # Priority 1: NVIDIA CUDA
    cuda_available, cuda_name = _check_cuda_available()
    if cuda_available:
        logger.debug(f"CUDA device detected: {cuda_name}")
        return "cuda"

    # Priority 2: Apple MPS
    mps_available, mps_name = _check_mps_available()
    if mps_available:
        logger.debug(f"MPS device detected: {mps_name}")
        return "mps"

    # ROCm detection (informational only)
    rocm_available, rocm_name = _check_rocm_available()
    if rocm_available:
        logger.warning(
            f"AMD GPU detected ({rocm_name}), but ROCm support is limited. "
            "Using CPU mode. See documentation for details."
        )

    # Fallback: CPU
    logger.debug("No compatible GPU detected, using CPU")
    return "cpu"


def get_device_info() -> Dict[str, any]:
    """
    Get detailed information about available compute devices.

    Returns:
        Dictionary with device availability and details:
        {
            'best_device': 'cuda' | 'mps' | 'cpu',
            'cuda': {'available': bool, 'name': str, 'count': int},
            'mps': {'available': bool, 'name': str},
            'rocm': {'available': bool, 'name': str},
            'cpu': {'cores': int},
            'platform': str
        }

    Example:
        >>> info = get_device_info()
        >>> print(f"Running on {info['best_device']}")
        >>> if info['cuda']['available']:
        >>>     print(f"GPU: {info['cuda']['name']}")
    """
    info = {
        'best_device': get_best_device(),
        'platform': platform.system(),
        'cuda': {'available': False, 'name': None, 'count': 0},
        'mps': {'available': False, 'name': None},
        'rocm': {'available': False, 'name': None},
        'cpu': {'cores': 0}
    }

    # CUDA info
    cuda_available, cuda_name = _check_cuda_available()
    if cuda_available:
        try:
            import torch
            info['cuda'] = {
                'available': True,
                'name': cuda_name,
                'count': torch.cuda.device_count()
            }
        except:
            info['cuda'] = {'available': True, 'name': cuda_name, 'count': 1}

    # MPS info
    mps_available, mps_name = _check_mps_available()
    if mps_available:
        info['mps'] = {'available': True, 'name': mps_name}

    # ROCm info
    rocm_available, rocm_name = _check_rocm_available()
    if rocm_available:
        info['rocm'] = {'available': True, 'name': rocm_name}

    # CPU info
    try:
        import multiprocessing
        info['cpu']['cores'] = multiprocessing.cpu_count()
    except:
        info['cpu']['cores'] = 1

    return info


def log_device_info():
    """
    Log comprehensive device information for debugging.

    Useful for troubleshooting platform-specific issues.
    Call during application startup or with --verbose flag.
    """
    info = get_device_info()

    logger.info("=" * 60)
    logger.info("Device Detection Report")
    logger.info("=" * 60)
    logger.info(f"Platform: {info['platform']}")
    logger.info(f"Best Device: {info['best_device']}")
    logger.info("")

    if info['cuda']['available']:
        logger.info(f"✓ NVIDIA CUDA: {info['cuda']['name']} ({info['cuda']['count']} GPU(s))")
    else:
        logger.info("✗ NVIDIA CUDA: Not available")

    if info['mps']['available']:
        logger.info(f"✓ Apple MPS: {info['mps']['name']}")
    else:
        logger.info("✗ Apple MPS: Not available")

    if info['rocm']['available']:
        logger.info(f"⚠ AMD ROCm: {info['rocm']['name']} (detected but unsupported)")
    else:
        logger.info("✗ AMD ROCm: Not available")

    logger.info(f"✓ CPU: {info['cpu']['cores']} cores")
    logger.info("=" * 60)


def is_gpu_available() -> bool:
    """
    Check if any GPU (CUDA or MPS) is available.

    Returns:
        True if CUDA or MPS available, False otherwise

    Example:
        >>> if is_gpu_available():
        >>>     print("GPU acceleration enabled")
    """
    device = get_best_device()
    return device in ('cuda', 'mps')


if __name__ == "__main__":
    # CLI usage: python -m whisperjav.utils.device_detector
    logging.basicConfig(level=logging.INFO)
    log_device_info()

    print("\nQuick Check:")
    print(f"  Best device: {get_best_device()}")
    print(f"  GPU available: {is_gpu_available()}")
