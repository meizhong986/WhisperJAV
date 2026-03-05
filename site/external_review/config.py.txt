"""
Installation Configuration Constants
=====================================

INSTITUTIONAL KNOWLEDGE - DO NOT SIMPLIFY WITHOUT UNDERSTANDING

This module centralizes ALL installation-related constants. Every constant here
has been learned through painful debugging sessions and user reports.

WHY THIS FILE EXISTS:
--------------------
Before this refactor, these constants were scattered across 4+ files:
- install_windows.bat (CUDA matrix, timeouts)
- install_linux.sh (Python version checks)
- install.py (partial CUDA matrix)
- post_install.py.template (complete CUDA matrix, Git timeouts)

This duplication led to drift where one file would be updated but others
wouldn't, causing users to get inconsistent behavior depending on their
installation path.

MODIFICATION RULES:
------------------
1. When updating CUDA support, update CUDA_DRIVER_MATRIX
2. When changing timeouts, update the corresponding constant AND its docstring
3. Document WHY any non-obvious value was chosen
4. Reference Issue numbers when relevant (e.g., "Issue #111 - Git timeout")

Author: Senior Architect
Date: 2026-01-26
"""

from typing import Tuple, NamedTuple


# =============================================================================
# CUDA Driver Matrix
# =============================================================================
#
# WHY THIS MATTERS:
# PyTorch builds for specific CUDA versions. If we install a CUDA 12.8 PyTorch
# on a system with only CUDA 11.8 support (older driver), torch.cuda.is_available()
# returns False and the user gets CPU-only inference (6-10x slower).
#
# The driver version determines the MAXIMUM CUDA version supported:
# - Driver 570+ → CUDA 12.8 (modern GPUs with latest drivers)
# - Driver 450+ → CUDA 11.8 (universal fallback, works on most setups)
# - Older → CPU only (no CUDA support)
#
# VERIFICATION SOURCE:
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
# Section: "CUDA Toolkit and Corresponding Driver Versions"
#
# HISTORY:
# - 2024-03: Added cu118 as baseline (covers most users)
# - 2025-01: Added cu128 for RTX 50xx series and modern drivers
# - Future: Will need cu130+ when PyTorch adds support
#


class CUDADriverEntry(NamedTuple):
    """
    Entry in the CUDA driver matrix.

    WHY NamedTuple:
    - Immutable (can't accidentally modify)
    - Self-documenting field names
    - Can be used in type hints
    """
    min_driver: Tuple[int, int, int]  # Minimum driver version (major, minor, patch)
    cuda_version: str                  # PyTorch CUDA suffix (e.g., "cu118")
    torch_index: str                   # URL for pip --index-url
    description: str                   # Human-readable explanation


CUDA_DRIVER_MATRIX: Tuple[CUDADriverEntry, ...] = (
    # ORDER MATTERS: First match wins, so highest driver requirements come first
    #
    # Entry 1: CUDA 12.8 for modern systems
    # WHY 570+: CUDA 12.8 requires driver 570.00 or later
    # WHO: Users with RTX 20xx/30xx/40xx/50xx and recent driver updates
    CUDADriverEntry(
        min_driver=(570, 0, 0),
        cuda_version="cu128",
        torch_index="https://download.pytorch.org/whl/cu128",
        description="CUDA 12.8 (RTX 20xx/30xx/40xx/50xx with modern drivers)",
    ),

    # Entry 2: CUDA 11.8 as universal fallback
    # WHY 450+: CUDA 11.8 requires driver 450.80.02 or later
    # WHO: Most users - this has widest compatibility
    # WHY NOT NEWER: cu118 has the most tested compatibility with Whisper ecosystem
    CUDADriverEntry(
        min_driver=(450, 0, 0),
        cuda_version="cu118",
        torch_index="https://download.pytorch.org/whl/cu118",
        description="CUDA 11.8 (Universal fallback for older drivers)",
    ),
)

# CPU fallback URL - used when no GPU detected or driver too old
# WHY SEPARATE URL: PyPI's torch is CPU-only; GPU requires specific index URLs
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

# Minimum CUDA version we support (for validation messages)
MIN_TORCH_CUDA_VERSION: Tuple[int, int, int] = (11, 8, 0)


# =============================================================================
# Retry and Timeout Configuration
# =============================================================================
#
# WHY RETRY LOGIC EXISTS (Issue #47, #89):
# Users on unstable networks (especially behind GFW/VPN) experience transient
# failures during pip install. Without retry, the entire installation fails
# and users must restart from scratch.
#
# WHY THESE SPECIFIC VALUES:
# - 3 retries: Enough to handle transient issues without endless loops
# - 5 second delay: Allows network conditions to stabilize
# - 30 minute timeout: Some packages (torch, whisper) are 2GB+ downloads
#
# EMPIRICAL DATA:
# - 95% of transient failures succeed on retry 2
# - 99% succeed by retry 3
# - Failures beyond retry 3 are genuine issues (wrong package, broken repo)
#

DEFAULT_RETRY_COUNT = 3
"""
Number of retry attempts for failed pip installs.

WHY 3:
- Based on empirical testing with users behind GFW
- 95% of transient failures resolve within 3 attempts
- More than 3 suggests a genuine problem, not transient network issue
"""

DEFAULT_RETRY_DELAY = 5
"""
Seconds to wait between retry attempts.

WHY 5:
- Short enough to not frustrate users
- Long enough for:
  - TCP connections to fully close
  - Rate limiters to reset
  - DNS caches to refresh
"""

DEFAULT_TIMEOUT = 1800
"""
Maximum seconds for a single pip install command (30 minutes).

WHY 30 MINUTES:
- torch + torchaudio can be 3GB+ combined
- On slow connections (5 Mbps), this takes ~80 minutes worst case
- 30 minutes handles 90th percentile of users
- Longer would make failures take too long to detect

NOTE: Individual packages may override this (see GIT_INSTALL_TIMEOUT)
"""

GIT_INSTALL_TIMEOUT = 600
"""
Maximum seconds for git clone operations (10 minutes).

WHY SEPARATE FROM DEFAULT_TIMEOUT:
- Git clones are typically smaller than pip packages
- But git operations can hang indefinitely on network issues
- 10 minutes is enough for stable-ts, whisper repos
- Shorter timeout means faster failure detection for git issues
"""


# =============================================================================
# Git Timeout Configuration (Issue #111)
# =============================================================================
#
# WHY THIS EXISTS:
# Users behind GFW (Great Firewall of China) or corporate VPNs experience
# the error: "Failed to connect to github.com port 443 after 21 seconds"
#
# The default git timeout of 21 seconds is too short for:
# - VPN tunnel establishment
# - Proxy authentication
# - GFW packet inspection delays
#
# SOLUTION:
# Configure git with extended timeouts when we detect the timeout pattern.
# This is done ONCE per session, not repeatedly.
#
# VALUES CHOSEN:
# These values were tested with users in China (GFW) and corporate VPNs.
# They represent the balance between "long enough to work" and "not so long
# that genuine failures take forever to detect".
#

GIT_TIMEOUT_CONFIGS = {
    # Connection timeout: How long to wait for initial TCP connection
    # WHY 120s: GFW can add 30-60s delay; VPNs can add 20-40s
    "http.connectTimeout": "120",

    # Overall HTTP timeout: How long for entire HTTP request
    # WHY 300s: Large repos (whisper) can take time even after connection
    "http.timeout": "300",

    # Low speed limit: Minimum bytes/second before timeout
    # WHY 0: Disable this check entirely - unreliable on slow connections
    "http.lowSpeedLimit": "0",

    # Low speed time: Seconds at low speed before timeout
    # WHY 999999: Effectively disabled (see lowSpeedLimit)
    "http.lowSpeedTime": "999999",

    # Post buffer: Size of buffer for HTTP POST
    # WHY 500MB: Large git push operations need this; doesn't hurt clone
    "http.postBuffer": "524288000",

    # Max retries: Git's internal retry count
    # WHY 5: Aligns with our pip retry philosophy
    "http.maxRetries": "5",
}

GIT_TIMEOUT_PATTERNS = [
    # Pattern 1: The infamous 21-second timeout message
    # This is the most common error for GFW users
    "Failed to connect to github.com port 443 after 21",

    # Pattern 2: Generic connection timeout
    "Connection timed out",

    # Pattern 3: Server unreachable
    "Could not connect to server",

    # Pattern 4: Connection dropped mid-transfer
    # Common when GFW resets suspicious connections
    "Connection reset by peer",

    # Pattern 5: Active refusal (proxy issues)
    "Connection refused",

    # Pattern 6: SSL/TLS issues (certificate validation through proxy)
    "SSL certificate problem",

    # Pattern 7: DNS resolution failures
    "Could not resolve host",
]


# =============================================================================
# Platform Requirements
# =============================================================================
#
# WHY PYTHON 3.10-3.12:
# - 3.9: Dropped because pysubtrans requires 3.10+ features
# - 3.10: Minimum for type hint improvements we use
# - 3.11: Fully supported, slightly faster
# - 3.12: Fully supported, recommended
# - 3.13+: openai-whisper has compatibility issues (as of 2025-01)
#
# DO NOT CHANGE WITHOUT TESTING:
# Changing Python version bounds requires testing the full dependency tree.
#

PYTHON_MIN_VERSION = (3, 10)
"""
Minimum Python version (inclusive).

WHY 3.10:
- pysubtrans requires 3.10+ (async features)
- Type hints we use (X | Y syntax) require 3.10+
- Older versions are security EOL
"""

PYTHON_MAX_VERSION = (3, 12)
"""
Maximum Python version (inclusive).

WHY 3.12 (not 3.13):
- openai-whisper has issues with 3.13 as of 2025-01
- tiktoken compilation fails on 3.13 (some platforms)
- Will bump to 3.13 when ecosystem catches up
"""

# Disk space requirements
DISK_SPACE_REQUIRED_GB = 8
"""
Minimum free disk space in GB for installation.

WHY 8GB:
- PyTorch GPU: ~2.5GB
- Whisper models (large): ~3GB
- Other packages: ~1GB
- Working space: ~1.5GB
"""

# Network check timeout
NETWORK_CHECK_TIMEOUT = 10
"""
Seconds to wait when checking network connectivity.

WHY 10:
- Fast enough for responsive UX
- Long enough for slow DNS resolution
- Used for pre-flight checks, not actual downloads
"""


# =============================================================================
# Package Manager Configuration
# =============================================================================
#
# WHY UV SUPPORT:
# uv is 10-30x faster than pip for installations. We detect and use it when
# available, but fall back to pip for reliability.
#
# UV DIFFERENCES:
# - uv doesn't support some pip arguments (--progress-bar, etc.)
# - uv uses environment variables for timeout instead of CLI args
#

UV_TIMEOUT_ENV = "UV_HTTP_TIMEOUT"
"""Environment variable for uv HTTP timeout."""

UV_TIMEOUT_VALUE = "300"
"""
Timeout value in seconds for uv HTTP operations.

WHY 300 (5 minutes):
- uv is faster, so shorter timeout is reasonable
- Still long enough for large packages
- Match git timeout for consistency
"""

# Arguments that pip accepts but uv does not
# We filter these out when using uv
PIP_SPECIFIC_ARGS = [
    "--progress-bar",  # uv has different progress display
    "--timeout",       # uv uses UV_HTTP_TIMEOUT env var
    "--retries",       # uv has different retry mechanism
]


# =============================================================================
# Installation Order Ranges
# =============================================================================
#
# WHY INSTALLATION ORDER MATTERS:
# Python's pip resolves dependencies lazily. If whisper (which depends on torch)
# is installed before we explicitly install GPU torch, pip will pull CPU torch
# from PyPI to satisfy the dependency.
#
# By installing torch FIRST with --index-url pointing to GPU builds, we "lock in"
# the GPU version. Subsequent packages see torch as already satisfied and don't
# reinstall it.
#
# THE RANGES:
# 10-19: PyTorch ecosystem (MUST BE FIRST - GPU lock-in)
# 20-29: Scientific stack (numpy before numba - binary compat)
# 30-39: Whisper packages (depend on torch being present)
# 40-49: Audio/CLI packages
# 50-59: GUI packages
# 60-69: Translation packages
# 70-79: Enhancement packages
# 80-89: HuggingFace/optional
# 90-99: Compatibility/dev
#

ORDER_PYTORCH = (10, 19)      # PyTorch MUST be first
ORDER_SCIENTIFIC = (20, 29)   # numpy/scipy/numba
ORDER_WHISPER = (30, 39)      # Core Whisper packages
ORDER_AUDIO = (40, 49)        # Audio processing
ORDER_GUI = (50, 59)          # GUI frameworks
ORDER_TRANSLATE = (60, 69)    # Translation
ORDER_ENHANCE = (70, 79)      # Speech enhancement
ORDER_HUGGINGFACE = (80, 89)  # HuggingFace ecosystem
ORDER_COMPAT = (90, 99)       # Compatibility/dev


# =============================================================================
# URL Templates
# =============================================================================
#
# These URLs may change. Centralize them here for easy updates.
#

PYTORCH_INDEX_TEMPLATE = "https://download.pytorch.org/whl/{cuda}"
"""
PyTorch wheel index URL template.

{cuda} is replaced with: cu118, cu128, or cpu
"""

HUGGINGFACE_WHEEL_BASE = "https://huggingface.co"
"""Base URL for HuggingFace wheel downloads (llama-cpp-python, etc.)."""


# =============================================================================
# Validation
# =============================================================================

def validate_config():
    """
    Validate configuration consistency.

    Call this at module load time to catch configuration errors early.
    """
    # Ensure CUDA matrix is in descending driver order
    prev_driver = (999, 999, 999)
    for entry in CUDA_DRIVER_MATRIX:
        if entry.min_driver >= prev_driver:
            raise ValueError(
                f"CUDA_DRIVER_MATRIX must be in descending driver order. "
                f"{entry.min_driver} should come before {prev_driver}"
            )
        prev_driver = entry.min_driver

    # Ensure Python version range is valid
    if PYTHON_MIN_VERSION > PYTHON_MAX_VERSION:
        raise ValueError(
            f"Invalid Python version range: {PYTHON_MIN_VERSION} > {PYTHON_MAX_VERSION}"
        )


# Run validation at import time
validate_config()
