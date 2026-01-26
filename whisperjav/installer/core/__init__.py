"""
WhisperJAV Installer Core Module
================================

This subpackage contains the core installation infrastructure:

- config.py: Centralized constants (CUDA matrix, timeouts, etc.)
- registry.py: Single Source of Truth for all package definitions
- executor.py: Robust package installation with retry/timeout handling
- detector.py: GPU and platform detection
- standalone.py: Self-contained version for standalone installer

ARCHITECTURAL DECISION:
----------------------
These modules are designed to have MINIMAL dependencies on the rest of
WhisperJAV. This is intentional:

1. install.py (root) may run before whisperjav is installed
2. standalone.py MUST work in a fresh conda environment
3. Circular imports would break the installation process

IMPORT RULES:
------------
- Core modules may import from Python stdlib
- Core modules may import from each other (no circles)
- Core modules MUST NOT import from whisperjav.* (except this package)

Author: Senior Architect
Date: 2026-01-26
"""

# Executor exports
from .executor import (
    StepExecutor,
    ExecutionResult,
)

# Detector exports
from .detector import (
    DetectedPlatform,
    GPUInfo,
    PrerequisiteResult,
    detect_platform,
    detect_gpu,
    get_platform_name,
    get_torch_index_url,
    check_python_version,
    check_ffmpeg,
    check_git,
    check_webview2,
    check_vcredist,
    check_prerequisites,
    print_prerequisites_report,
)

# Registry exports
from .registry import (
    Package,
    Extra,
    Platform,
    InstallSource,
    PACKAGES,
    get_packages_in_install_order,
    get_packages_for_platform,
    get_packages_by_extra,
    get_import_map,
    generate_pyproject_extras,
    generate_core_dependencies,
    generate_requirements_txt,
    get_package_by_name,
    get_all_package_names,
)

# Config exports
from .config import (
    # CUDA configuration
    CUDA_DRIVER_MATRIX,
    CUDADriverEntry,
    CPU_TORCH_INDEX,
    MIN_TORCH_CUDA_VERSION,

    # Retry/timeout configuration
    DEFAULT_RETRY_COUNT,
    DEFAULT_RETRY_DELAY,
    DEFAULT_TIMEOUT,
    GIT_INSTALL_TIMEOUT,
    GIT_TIMEOUT_CONFIGS,
    GIT_TIMEOUT_PATTERNS,

    # Platform requirements
    PYTHON_MIN_VERSION,
    PYTHON_MAX_VERSION,
    DISK_SPACE_REQUIRED_GB,
    NETWORK_CHECK_TIMEOUT,

    # Package manager
    UV_TIMEOUT_ENV,
    UV_TIMEOUT_VALUE,
    PIP_SPECIFIC_ARGS,

    # Installation order
    ORDER_PYTORCH,
    ORDER_SCIENTIFIC,
    ORDER_WHISPER,
    ORDER_AUDIO,
    ORDER_GUI,
    ORDER_TRANSLATE,
    ORDER_ENHANCE,
    ORDER_HUGGINGFACE,
    ORDER_COMPAT,

    # URL templates
    PYTORCH_INDEX_TEMPLATE,
    HUGGINGFACE_WHEEL_BASE,
)

__all__ = [
    # Executor exports
    "StepExecutor",
    "ExecutionResult",

    # Detector exports
    "DetectedPlatform",
    "GPUInfo",
    "PrerequisiteResult",
    "detect_platform",
    "detect_gpu",
    "get_platform_name",
    "get_torch_index_url",
    "check_python_version",
    "check_ffmpeg",
    "check_git",
    "check_webview2",
    "check_vcredist",
    "check_prerequisites",
    "print_prerequisites_report",

    # Registry exports
    "Package",
    "Extra",
    "Platform",
    "InstallSource",
    "PACKAGES",
    "get_packages_in_install_order",
    "get_packages_for_platform",
    "get_packages_by_extra",
    "get_import_map",
    "generate_pyproject_extras",
    "generate_core_dependencies",
    "generate_requirements_txt",
    "get_package_by_name",
    "get_all_package_names",

    # Config exports
    "CUDA_DRIVER_MATRIX",
    "CUDADriverEntry",
    "CPU_TORCH_INDEX",
    "MIN_TORCH_CUDA_VERSION",
    "DEFAULT_RETRY_COUNT",
    "DEFAULT_RETRY_DELAY",
    "DEFAULT_TIMEOUT",
    "GIT_INSTALL_TIMEOUT",
    "GIT_TIMEOUT_CONFIGS",
    "GIT_TIMEOUT_PATTERNS",
    "PYTHON_MIN_VERSION",
    "PYTHON_MAX_VERSION",
    "DISK_SPACE_REQUIRED_GB",
    "NETWORK_CHECK_TIMEOUT",
    "UV_TIMEOUT_ENV",
    "UV_TIMEOUT_VALUE",
    "PIP_SPECIFIC_ARGS",
    "ORDER_PYTORCH",
    "ORDER_SCIENTIFIC",
    "ORDER_WHISPER",
    "ORDER_AUDIO",
    "ORDER_GUI",
    "ORDER_TRANSLATE",
    "ORDER_ENHANCE",
    "ORDER_HUGGINGFACE",
    "ORDER_COMPAT",
    "PYTORCH_INDEX_TEMPLATE",
    "HUGGINGFACE_WHEEL_BASE",
]
