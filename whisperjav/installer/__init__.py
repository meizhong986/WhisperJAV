"""
WhisperJAV Installation System
==============================

This module provides the unified installation infrastructure for WhisperJAV.
All installation scripts (install.py, shell wrappers, post_install.py) use
this module to ensure consistent behavior.

ARCHITECTURAL OVERVIEW:
----------------------

    ┌─────────────────────────────────────────────────────────────────┐
    │                    PACKAGE REGISTRY (registry.py)               │
    │                     SINGLE SOURCE OF TRUTH                      │
    │                                                                 │
    │  Defines: packages, versions, sources, order, extras            │
    │  Generates: pyproject.toml sections, requirements.txt           │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           ▼                       ▼                       ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   DETECTOR      │   │   EXECUTOR      │   │   VALIDATION    │
    │   (detector.py) │   │   (executor.py) │   │   (sync.py)     │
    │                 │   │                 │   │                 │
    │  - GPU detect   │   │  - pip/uv       │   │  - pyproject    │
    │  - Driver ver   │   │  - Retry (3x)   │   │    sync check   │
    │  - CUDA select  │   │  - Git timeout  │   │  - Import scan  │
    │  - Platform     │   │  - Logging      │   │  - CI/CD hook   │
    └─────────────────┘   └─────────────────┘   └─────────────────┘


WHY THIS ARCHITECTURE:
---------------------

1. SINGLE SOURCE OF TRUTH (registry.py):
   - Before: Package versions defined in 4+ places
   - After: One place to update, generates everything else
   - Eliminates version drift between files

2. UNIFIED EXECUTOR (executor.py):
   - Before: Retry logic only in .bat, missing in .py and .sh
   - After: Same retry/timeout logic everywhere
   - Users get consistent experience regardless of install method

3. CENTRALIZED DETECTION (detector.py):
   - Before: GPU detection duplicated 3 times
   - After: One implementation, tested thoroughly
   - CUDA version selection is consistent

4. VALIDATION (validation/):
   - Before: No way to catch drift before release
   - After: CI blocks merges with undeclared dependencies
   - "Ghost dependency" bugs are caught at commit time


USAGE:
-----

For installation scripts (install.py, post_install.py):

    from whisperjav.installer import (
        # Detection
        detect_gpu,
        detect_platform,
        check_prerequisites,

        # Registry
        get_packages_in_install_order,
        get_packages_by_extra,

        # Execution
        StepExecutor,

        # Config
        PYTHON_MIN_VERSION,
        PYTHON_MAX_VERSION,
    )

    # Detect GPU and select CUDA version
    gpu_info = detect_gpu()
    cuda_version = gpu_info.cuda_version or "cpu"

    # Get packages to install
    packages = get_packages_in_install_order()

    # Install with retry/timeout handling
    executor = StepExecutor(log_file=Path("install.log"))
    for pkg in packages:
        result = executor.install_package(pkg, cuda_version=cuda_version)
        if not result.success and pkg.required:
            raise InstallationError(f"Failed to install {pkg.name}")


For validation (CI/CD):

    python -m whisperjav.installer.validation


INSTITUTIONAL KNOWLEDGE:
-----------------------

Q: Why not just use pyproject.toml for everything?
A: pyproject.toml cannot express:
   - Installation ORDER (torch before whisper)
   - Conditional index URLs (GPU vs CPU)
   - Runtime GPU detection
   See: .docs/INSTALLATION_ARCHITECTURE_IMPLEMENTATION_PLAN_V2.md Section 2.1

Q: Why separate registry.py from executor.py?
A: Separation of concerns:
   - Registry: WHAT to install (data)
   - Executor: HOW to install (logic)
   This allows testing each independently.

Q: Why is standalone.py separate?
A: standalone.py runs in conda environment where whisperjav may not be
   installed yet. It must have ZERO imports from whisperjav.*.
   See: Gemini Review watchpoint #1


Author: Senior Architect
Date: 2026-01-26
"""

# =============================================================================
# Executor - Robust Package Installation
# =============================================================================
#
# WHY EXPORT EXECUTOR:
# The executor handles retry logic, Git timeout detection, and uv support.
# All installation scripts should use this for consistent behavior.
#

from .core.executor import (
    StepExecutor,
    ExecutionResult,
)


# =============================================================================
# Detector - GPU and Platform Detection
# =============================================================================
#
# WHY EXPORT DETECTOR:
# GPU detection determines CUDA version for PyTorch installation.
# Unified detection ensures consistent behavior across all install paths.
#

from .core.detector import (
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


# =============================================================================
# Registry - Single Source of Truth for Packages
# =============================================================================
#
# WHY EXPORT REGISTRY:
# Installation scripts need access to package definitions.
# The registry is THE authoritative source for what packages to install.
#

from .core.registry import (
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


# =============================================================================
# Configuration Constants
# =============================================================================
#
# WHY EXPORT CONFIG:
# Installation scripts need access to timeouts, CUDA matrix, etc.
# Centralizing here ensures consistency.
#

from .core.config import (
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

    # Installation order ranges
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


# =============================================================================
# Validation - CI/CD Checks
# =============================================================================
#
# WHY EXPORT VALIDATION:
# Validation functions can be run in CI/CD to catch drift.
# Also useful for debugging installation issues.
#

from .validation import (
    validate_standalone_self_containment,
    run_all_validations,
)


# =============================================================================
# Public API Exports
# =============================================================================
#
# NOTE: Registry, Executor, Detector exports will be added as modules are
# implemented. For now, we export config only.
#

__all__ = [
    # Executor - Installation
    "StepExecutor",
    "ExecutionResult",

    # Detector - GPU and Platform
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

    # Registry - Package definitions
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

    # Config - CUDA
    "CUDA_DRIVER_MATRIX",
    "CUDADriverEntry",
    "CPU_TORCH_INDEX",
    "MIN_TORCH_CUDA_VERSION",

    # Config - Retry/Timeout
    "DEFAULT_RETRY_COUNT",
    "DEFAULT_RETRY_DELAY",
    "DEFAULT_TIMEOUT",
    "GIT_INSTALL_TIMEOUT",
    "GIT_TIMEOUT_CONFIGS",
    "GIT_TIMEOUT_PATTERNS",

    # Config - Platform
    "PYTHON_MIN_VERSION",
    "PYTHON_MAX_VERSION",
    "DISK_SPACE_REQUIRED_GB",
    "NETWORK_CHECK_TIMEOUT",

    # Config - Package Manager
    "UV_TIMEOUT_ENV",
    "UV_TIMEOUT_VALUE",
    "PIP_SPECIFIC_ARGS",

    # Config - Installation Order
    "ORDER_PYTORCH",
    "ORDER_SCIENTIFIC",
    "ORDER_WHISPER",
    "ORDER_AUDIO",
    "ORDER_GUI",
    "ORDER_TRANSLATE",
    "ORDER_ENHANCE",
    "ORDER_HUGGINGFACE",
    "ORDER_COMPAT",

    # Config - URLs
    "PYTORCH_INDEX_TEMPLATE",
    "HUGGINGFACE_WHEEL_BASE",

    # Validation
    "validate_standalone_self_containment",
    "run_all_validations",
]


# =============================================================================
# Version
# =============================================================================

__version__ = "2.0.0"
"""
Installer module version.

Separate from WhisperJAV version to track installer changes independently.
"""
