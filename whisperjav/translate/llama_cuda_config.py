"""
Centralized CUDA Configuration for llama-cpp-python.

SINGLE SOURCE OF TRUTH for CUDA version support.

=============================================================================
IMPORTANT: Only 4 CUDA versions are officially supported.
Adding more creates maintenance burden without clear value.
=============================================================================

Official Support:
    - cu128: Primary (standalone installer, driver 570+)
    - cu118: Fallback (standalone installer, driver 450+)
    - cu126: Google Colab (driver 560+)
    - cu130: Development only (driver 575+)

NOTE: post_install.py.template has a COPY of this config because it runs
before whisperjav is installed. If you change this file, you MUST also
update post_install.py.template to match!
"""

from typing import Dict, List, Optional, Tuple


# =============================================================================
# Official CUDA Version Configuration
# =============================================================================

OFFICIAL_CUDA_VERSIONS: Dict[str, Dict] = {
    "cu130": {
        "description": "CUDA 13.0 (development only)",
        "min_driver": (575, 0, 0),
        "on_huggingface": False,  # Dev only, uses JamePeng cascade
        "standalone_installer": False,
    },
    "cu128": {
        "description": "CUDA 12.8 (primary)",
        "min_driver": (570, 0, 0),
        "on_huggingface": True,
        "standalone_installer": True,
    },
    "cu126": {
        "description": "CUDA 12.6 (Colab)",
        "min_driver": (560, 0, 0),
        "on_huggingface": True,
        "standalone_installer": False,
    },
    "cu118": {
        "description": "CUDA 11.8 (legacy fallback)",
        "min_driver": (450, 0, 0),
        "on_huggingface": True,  # Will be uploaded
        "standalone_installer": True,
    },
}

# Backward compatibility fallback order (newest to oldest)
# CUDA is backward compatible: newer driver can run older wheels
# Example: cu130 system can use cu128 wheel
FALLBACK_ORDER: List[str] = ["cu130", "cu128", "cu126", "cu118"]

# Standalone installer (.exe) ONLY supports these two
STANDALONE_CUDA_VERSIONS: List[str] = ["cu128", "cu118"]

# Google Colab uses this specific version
COLAB_CUDA_VERSION: str = "cu126"

# HuggingFace wheel repository
HUGGINGFACE_REPO_ID: str = "mei986/whisperjav-wheels"
HUGGINGFACE_WHEEL_VERSION: str = "0.3.21"  # llama-cpp-python version


# =============================================================================
# Helper Functions
# =============================================================================

def get_compatible_cuda_versions(detected_cuda: str) -> List[str]:
    """
    Get CUDA versions compatible with the detected version.

    CUDA is backward compatible: a newer driver can run wheels built
    for older CUDA versions. This returns all compatible versions
    in preference order (exact match first, then older versions).

    Args:
        detected_cuda: Detected CUDA version (e.g., "cu128")

    Returns:
        List of compatible CUDA versions in preference order

    Example:
        get_compatible_cuda_versions("cu130") -> ["cu130", "cu128", "cu126", "cu118"]
        get_compatible_cuda_versions("cu118") -> ["cu118"]
    """
    if not detected_cuda or not detected_cuda.startswith("cu"):
        return []

    try:
        detected_num = int(detected_cuda[2:])  # cu128 -> 128
    except (ValueError, IndexError):
        return []

    compatible = []
    for cuda_ver in FALLBACK_ORDER:
        try:
            ver_num = int(cuda_ver[2:])
            if ver_num <= detected_num:
                compatible.append(cuda_ver)
        except (ValueError, IndexError):
            continue

    return compatible


def get_min_driver_for_cuda(cuda_version: str) -> Optional[Tuple[int, int, int]]:
    """
    Get minimum driver version required for a CUDA version.

    Args:
        cuda_version: CUDA version (e.g., "cu128")

    Returns:
        Minimum driver version tuple (major, minor, patch) or None
    """
    config = OFFICIAL_CUDA_VERSIONS.get(cuda_version)
    if config:
        return config["min_driver"]
    return None


def is_on_huggingface(cuda_version: str) -> bool:
    """Check if wheels for this CUDA version are on HuggingFace."""
    config = OFFICIAL_CUDA_VERSIONS.get(cuda_version)
    return config.get("on_huggingface", False) if config else False


def is_standalone_supported(cuda_version: str) -> bool:
    """Check if this CUDA version is supported by standalone installer."""
    return cuda_version in STANDALONE_CUDA_VERSIONS


def get_cuda_description(cuda_version: str) -> str:
    """Get human-readable description for a CUDA version."""
    config = OFFICIAL_CUDA_VERSIONS.get(cuda_version)
    return config.get("description", f"CUDA {cuda_version}") if config else f"Unknown ({cuda_version})"


def cuda_version_from_driver(driver_version: Tuple[int, int, int]) -> Optional[str]:
    """
    Determine the best CUDA version for a given driver version.

    Args:
        driver_version: Driver version tuple (major, minor, patch)

    Returns:
        Best matching CUDA version or None if driver too old
    """
    # Check from newest to oldest
    for cuda_ver in FALLBACK_ORDER:
        min_driver = get_min_driver_for_cuda(cuda_ver)
        if min_driver and driver_version >= min_driver:
            return cuda_ver
    return None
