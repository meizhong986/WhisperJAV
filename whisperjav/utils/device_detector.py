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


# Set when the CUDA device is present but the installed PyTorch build has no
# kernels for it (#411: GTX 1060, sm_61, cu128 wheel built for sm_75 and up).
# Read by the start-up gate to say why the GPU is not used.
CUDA_UNUSABLE_REASON: Optional[str] = None
# The compute capability of that card, so the start-up gate can tell a card older than
# 1.10 supports (below 7.5: 1.9.5 still runs it) from one newer than the build.
CUDA_UNUSABLE_CAPABILITY: Optional[Tuple[int, int]] = None


def cuda_build_supports_device(capability: Tuple[int, int], arch_list) -> bool:
    """Can a PyTorch build compiled for ``arch_list`` run on a card of ``capability``?

    This is NVIDIA's per-entry cubin/PTX rule: an ``sm_XY`` cubin runs on hardware
    of the same major version with minor >= Y; a ``compute_XY`` PTX is JIT-compiled
    for any hardware with capability >= X.Y. Architecture-specific suffixes
    (``sm_90a``, ``sm_100f``) are stripped the way ``torch.cuda._extract_arch_version``
    does. Recent PyTorch applies the same idea in its start-up warning
    (``_warn_unsupported_code``) with family exceptions such as 8.7 and 10.1, which
    this rule does not model and therefore treats as supported; older PyTorch only
    warned on the list's min/max (``_check_capability``) and on the major version
    (``_check_cubins``). Where those disagree, this rule errs toward reporting the
    card as usable. A list with no ``sm``/``compute`` entry cannot be judged and is
    treated as supported.
    """
    major, minor = int(capability[0]), int(capability[1])
    judged = False
    for arch in arch_list or []:
        if not isinstance(arch, str) or "_" not in arch:
            continue
        kind, _, num = arch.partition("_")
        num = num.removesuffix("a").removesuffix("f")
        if kind not in ("sm", "compute") or not num.isdigit() or len(num) < 2:
            continue
        judged = True
        a_major, a_minor = int(num[:-1]), int(num[-1])
        if kind == "sm" and a_major == major and a_minor <= minor:
            return True
        if kind == "compute" and (a_major, a_minor) <= (major, minor):
            return True
    return not judged


def _check_cuda_available() -> Tuple[bool, Optional[str]]:
    """
    Check if CUDA is available AND usable by this PyTorch build, and get the GPU name.

    Handles CUDA driver version mismatch errors gracefully. A card the build has
    no kernels for (#411) is reported as not available, with the reason kept in
    ``CUDA_UNUSABLE_REASON`` for the start-up gate and ``--check``.

    Returns:
        (is_available, gpu_name)
    """
    global CUDA_UNUSABLE_REASON, CUDA_UNUSABLE_CAPABILITY
    CUDA_UNUSABLE_REASON = None
    CUDA_UNUSABLE_CAPABILITY = None
    try:
        import torch
        if torch.cuda.is_available():
            # get_device_name(0) can throw RuntimeError if driver is incompatible
            try:
                gpu_name = torch.cuda.get_device_name(0)
                try:
                    capability = tuple(torch.cuda.get_device_capability(0))
                    arch_list = list(torch.cuda.get_arch_list())
                except Exception as e:  # noqa: BLE001 - cannot judge, keep the old answer
                    logger.debug(f"CUDA capability check skipped: {e}")
                    capability, arch_list = None, []
                if capability and not cuda_build_supports_device(capability, arch_list):
                    CUDA_UNUSABLE_REASON = (
                        f"{gpu_name} has compute capability {capability[0]}.{capability[1]}, "
                        f"but this PyTorch build has kernels only for "
                        f"{', '.join(arch_list)}. The GPU cannot be used by this build."
                    )
                    CUDA_UNUSABLE_CAPABILITY = (int(capability[0]), int(capability[1]))
                    logger.warning(CUDA_UNUSABLE_REASON)
                    return False, None
                return True, gpu_name
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "driver version is insufficient" in error_msg:
                    logger.warning(
                        "CUDA driver version is too old for PyTorch CUDA runtime. "
                        "Falling back to CPU mode. Update NVIDIA drivers to enable GPU acceleration."
                    )
                else:
                    logger.warning(f"CUDA initialization failed: {e}")
                return False, None
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
            try:
                gpu_name = torch.cuda.get_device_name(0)
                if 'AMD' in gpu_name.upper() or 'RADEON' in gpu_name.upper():
                    return True, gpu_name
            except RuntimeError:
                # Driver version mismatch - not ROCm
                pass
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


def pass_device(run_device: Optional[str], pass_setting: Optional[str] = None) -> str:
    """The device a pipeline or pass runs on: --device governs every pipeline (design 22.3.1, owner W237 1).

    --device cpu is absolute ("Run on the CPU this time" must reach every part). Otherwise a
    pass's own explicit setting (its Customize Device) stands; else the run's --device; else auto.
    """
    if run_device == "cpu":
        return "cpu"
    if pass_setting and pass_setting != "auto":
        return pass_setting
    return run_device or "auto"


def resolve_run_device(device: Optional[str], hf_device: Optional[str],
                       qwen_device: Optional[str]) -> Tuple[Optional[str], list]:
    """The run's --device, with --hf-device and --qwen-device as hidden names for it (design 22.3.1).

    Returns (device, notes): a note for each old option that --device overrides.
    """
    notes = []
    for flag, value in (("--hf-device", hf_device), ("--qwen-device", qwen_device)):
        if not value:
            continue
        if device is None:
            device = value
        elif value != device:
            notes.append(f"Note: {flag} {value} is ignored: --device {device} sets every pipeline's device.")
    return device, notes


def ctranslate2_device(requested: Optional[str], best: Optional[str] = None) -> str:
    """The device for a CTranslate2 pipeline (balanced, fast, faster): design 22.3.3.

    "auto" becomes "cuda" only when CTranslate2 itself reports a CUDA device. PyTorch's answer is
    not enough: a ROCm PyTorch (an Own PyTorch install on an AMD GPU) reports CUDA, and
    CTranslate2 cannot use that GPU. An explicit request is passed as given. ``best`` is
    get_best_device()'s answer when the caller already has it.
    """
    if requested and requested != "auto":
        return requested
    best = best or get_best_device()
    if best != "cuda":
        return best  # "mps" is turned into the CPU by the modules themselves (T53)
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception as e:  # noqa: BLE001 - no answer from CTranslate2 means it cannot use the GPU
        logger.debug(f"CTranslate2 CUDA check failed: {e}")
    if _own_pytorch_install():
        logger.warning(OWN_PYTORCH_CTRANSLATE2_TEXT)
    return "cpu"


# T67 (design 22.10), only on an Own PyTorch install.
OWN_PYTORCH_CTRANSLATE2_TEXT = (
    "balanced, fast and faster run on the CPU (not accelerated): their engine (CTranslate2) can use "
    "only NVIDIA GPUs. fidelity, transformers and ChronosJAV ask for your GPU through your PyTorch "
    "(not tested).")


def _own_pytorch_install() -> bool:
    try:
        from whisperjav.installer.core.accelerators import read_record
        return read_record()[1] == "own-pytorch"
    except Exception:  # noqa: BLE001
        return False


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
