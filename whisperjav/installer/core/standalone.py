"""
WhisperJAV Standalone Installation Utilities
=============================================

# ==============================================================================
#                    ⚠️ CRITICAL: SELF-CONTAINMENT REQUIREMENT ⚠️
# ==============================================================================
#
# This file MUST remain completely self-contained with ZERO imports from
# whisperjav.* modules.
#
# WHY THIS CONSTRAINT:
# --------------------
# This file is designed to be:
# 1. Bundled with standalone installers (conda-constructor)
# 2. Run BEFORE whisperjav is installed
# 3. Copied directly to installer bundles without modification
#
# If ANY import from whisperjav.* is added, the standalone installer will
# break on fresh machines where WhisperJAV is not yet installed.
#
# FORBIDDEN IMPORTS:
# ------------------
# from whisperjav import ...          # ❌ BREAKS STANDALONE
# from whisperjav.installer import ... # ❌ BREAKS STANDALONE
# import whisperjav                    # ❌ BREAKS STANDALONE
#
# ALLOWED IMPORTS:
# ----------------
# - Python standard library (os, sys, subprocess, etc.)
# - External packages that are always installed (pynvml, etc.)
# - Relative imports from this file's siblings (.config, etc.) are NOT allowed
#   because this file may be copied standalone
#
# CI VALIDATION:
# --------------
# A CI check scans this file for forbidden imports. Commits that add
# 'from whisperjav' or 'import whisperjav' will be blocked.
#
# HISTORY (Gemini Review Watchpoint #1):
# --------------------------------------
# This constraint was identified as the most "brittle" aspect of the
# architecture refactoring. A developer accidentally importing whisperjav
# would silently break the standalone installer on fresh machines.
#
# ==============================================================================
"""

import os
import re
import sys
import shutil
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)


# =============================================================================
# Type Definitions
# =============================================================================
#
# WHY NAMEDTUPLES:
# - Immutable, hashable, and self-documenting
# - Can be easily serialized/deserialized
# - Work with or without type hints
#

class CUDADriverEntry(NamedTuple):
    """
    Mapping from NVIDIA driver version to CUDA toolkit version.

    WHY THIS MAPPING:
    - NVIDIA drivers support specific CUDA versions
    - Driver 570+ required for CUDA 12.8
    - Driver 450+ required for CUDA 11.8
    - This matrix ensures we install compatible PyTorch

    MAINTENANCE:
    When NVIDIA releases new CUDA versions, update this matrix.
    Check pytorch.org/get-started/locally/ for supported versions.
    """
    min_driver: Tuple[int, int, int]
    cuda_version: str
    torch_index: str
    description: str = ""


class GPUInfo(NamedTuple):
    """
    Information about detected NVIDIA GPU.

    WHY CAPTURE ALL THIS:
    - name: For display to user
    - driver_version: For CUDA version selection
    - cuda_version: Determined from driver_version
    - memory_mb: For model size recommendations
    - source: For debugging detection issues
    """
    name: Optional[str] = None
    driver_version: Optional[Tuple[int, int, int]] = None
    cuda_version: Optional[str] = None
    memory_mb: Optional[int] = None
    source: str = "none"
    message: str = ""


class InstallResult(NamedTuple):
    """
    Result of an installation attempt.

    WHY STRUCTURED RESULT:
    - success: Primary indicator
    - stdout/stderr: For logging and debugging
    - return_code: For programmatic handling
    - duration: For performance analysis
    - attempts: For retry tracking
    """
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    duration_seconds: float = 0.0
    attempts: int = 1
    command: str = ""


# =============================================================================
# CUDA Driver Matrix
# =============================================================================
#
# WHY THIS MATRIX:
# PyTorch with CUDA requires matching driver versions. If we install
# CUDA 12.8 on a system with driver 450, PyTorch CUDA won't work.
#
# SIMPLIFIED MATRIX (v1.8.0+):
# - PyTorch 2.7.x dropped CUDA 12.1/12.4 support
# - Only CUDA 12.8 and 11.8 are now available
# - This simplifies driver compatibility
#
# DRIVER REQUIREMENTS:
# - CUDA 12.8: Driver 570+ (RTX 20xx/30xx/40xx/50xx with modern drivers)
# - CUDA 11.8: Driver 450+ (Universal fallback for older drivers)
#

CUDA_DRIVER_MATRIX: Tuple[CUDADriverEntry, ...] = (
    CUDADriverEntry(
        min_driver=(570, 0, 0),
        cuda_version="cu128",
        torch_index="https://download.pytorch.org/whl/cu128",
        description="CUDA 12.8 (RTX 20xx/30xx/40xx/50xx with modern drivers)",
    ),
    CUDADriverEntry(
        min_driver=(450, 0, 0),
        cuda_version="cu118",
        torch_index="https://download.pytorch.org/whl/cu118",
        description="CUDA 11.8 (Universal fallback for older drivers)",
    ),
)

CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


# =============================================================================
# Retry Configuration
# =============================================================================
#
# WHY RETRY LOGIC:
# Issue #47, #89: Users in China (behind GFW) and on slow VPNs experience
# transient network failures. Without retry, installation fails completely.
#
# RETRY STRATEGY:
# - 3 attempts with 5-second delay
# - Exponential backoff not used (pip already has its own retry)
# - Git timeout detection triggers automatic reconfiguration
#

DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 5  # seconds
DEFAULT_TIMEOUT = 1800  # 30 minutes for large package installs

# Extended timeout for Git operations (cloning repositories)
GIT_INSTALL_TIMEOUT = 3600  # 60 minutes

# Git timeout patterns to detect connection issues
GIT_TIMEOUT_PATTERNS = (
    "Failed to connect to github.com",
    "Connection timed out",
    "after 21",  # "...after 21 ms"
    "Connection refused",
    "Could not resolve host",
)


# =============================================================================
# Utility Functions
# =============================================================================

def parse_version_string(version_str: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse version string into tuple.

    WHY CUSTOM PARSER:
    - nvidia-smi outputs "570.86.10"
    - Need to compare as numeric tuples
    - packaging.version is overkill for this use case

    Args:
        version_str: Version string like "570.86.10"

    Returns:
        Tuple (major, minor, patch) or None if parse fails
    """
    if not version_str:
        return None
    digits = [int(piece) for piece in re.split(r"[^0-9]", version_str) if piece.isdigit()]
    if not digits:
        return None
    while len(digits) < 3:
        digits.append(0)
    return tuple(digits[:3])  # type: ignore[return-value]


def format_version_tuple(version: Optional[Tuple[int, int, int]]) -> str:
    """Format version tuple as string."""
    if not version:
        return "unknown"
    return ".".join(str(part) for part in version)


# =============================================================================
# GPU Detection
# =============================================================================
#
# WHY MULTIPLE DETECTION METHODS:
# Different environments have different tools available:
# 1. nvidia-smi: Best option, but requires NVIDIA driver tools
# 2. pynvml: Python library, works even if nvidia-smi not in PATH
# 3. /proc/driver/nvidia/version: Linux fallback
#
# DETECTION ORDER:
# 1. Try nvidia-smi first (most reliable, gives GPU name)
# 2. Try pynvml (works in Python environments)
# 3. Try /proc fallback (Linux only)
# 4. Return "no GPU" if all fail
#

def detect_gpu_nvidia_smi() -> Optional[GPUInfo]:
    """
    Detect NVIDIA GPU using nvidia-smi.

    WHY NVIDIA-SMI:
    - Most reliable method when available
    - Returns both driver version and GPU name
    - Works on Windows, Linux, and macOS with NVIDIA drivers
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        line = result.stdout.strip().splitlines()[0]
        if "," in line:
            driver_str, gpu_name = line.split(",", 1)
        else:
            driver_str, gpu_name = line, "Unknown GPU"

        driver_tuple = parse_version_string(driver_str.strip())
        if driver_tuple:
            cuda_version = select_cuda_version(driver_tuple)
            return GPUInfo(
                name=gpu_name.strip(),
                driver_version=driver_tuple,
                cuda_version=cuda_version,
                source="nvidia-smi",
                message=f"Detected via nvidia-smi: {gpu_name.strip()}",
            )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
        pass
    return None


def detect_gpu_pynvml() -> Optional[GPUInfo]:
    """
    Detect NVIDIA GPU using pynvml library.

    WHY PYNVML:
    - Works when nvidia-smi is not in PATH
    - Pure Python implementation
    - Can get more detailed GPU info (memory, compute capability)
    """
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            driver_str = pynvml.nvmlSystemGetDriverVersion()
            gpu_name = pynvml.nvmlDeviceGetName(handle)

            if isinstance(driver_str, bytes):
                driver_str = driver_str.decode("utf-8")
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")

            # Try to get memory info
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_mb = mem_info.total // (1024 * 1024)
            except Exception:
                memory_mb = None

            driver_tuple = parse_version_string(str(driver_str))
            if driver_tuple:
                cuda_version = select_cuda_version(driver_tuple)
                return GPUInfo(
                    name=str(gpu_name),
                    driver_version=driver_tuple,
                    cuda_version=cuda_version,
                    memory_mb=memory_mb,
                    source="pynvml",
                    message=f"Detected via pynvml: {gpu_name}",
                )
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except (ImportError, Exception):
        pass
    return None


def detect_gpu_proc() -> Optional[GPUInfo]:
    """
    Detect NVIDIA GPU using /proc filesystem (Linux only).

    WHY /PROC FALLBACK:
    - Works on Linux when nvidia-smi/pynvml unavailable
    - Kernel module exposes driver version
    - Last resort before giving up
    """
    proc_path = Path("/proc/driver/nvidia/version")
    if not proc_path.exists():
        return None

    try:
        content = proc_path.read_text()
        # Pattern: "NVRM version: NVIDIA UNIX x86_64 Kernel Module  570.86.10  ..."
        match = re.search(r"Kernel Module\s+(\d+(?:\.\d+)*)", content)
        if match:
            driver_tuple = parse_version_string(match.group(1))
            if driver_tuple:
                cuda_version = select_cuda_version(driver_tuple)
                return GPUInfo(
                    name="NVIDIA GPU (detected via /proc)",
                    driver_version=driver_tuple,
                    cuda_version=cuda_version,
                    source="/proc",
                    message="Detected via /proc/driver/nvidia/version",
                )
    except Exception:
        pass
    return None


def select_cuda_version(driver_version: Tuple[int, int, int]) -> Optional[str]:
    """
    Select appropriate CUDA version based on driver version.

    WHY DRIVER-BASED SELECTION:
    Different CUDA versions require different minimum driver versions.
    Installing incompatible versions causes silent failures.

    Args:
        driver_version: Tuple like (570, 86, 10)

    Returns:
        CUDA version string like "cu128" or None if driver too old
    """
    for entry in CUDA_DRIVER_MATRIX:
        if driver_version >= entry.min_driver:
            return entry.cuda_version
    return None


def detect_gpu() -> GPUInfo:
    """
    Detect NVIDIA GPU using best available method.

    WHY PRIORITY ORDER:
    1. nvidia-smi: Most reliable, gives GPU name
    2. pynvml: Python library fallback
    3. /proc: Linux kernel fallback
    4. No GPU: Clean failure

    Returns:
        GPUInfo with detection results
    """
    # Try nvidia-smi first (most reliable)
    result = detect_gpu_nvidia_smi()
    if result:
        return result

    # Try pynvml (Python library)
    result = detect_gpu_pynvml()
    if result:
        return result

    # Try /proc fallback (Linux)
    result = detect_gpu_proc()
    if result:
        return result

    # No GPU detected
    return GPUInfo(
        source="none",
        message="No NVIDIA GPU detected. Will use CPU-only installation.",
    )


def get_torch_index_url(cuda_version: Optional[str]) -> str:
    """
    Get PyTorch index URL for given CUDA version.

    Args:
        cuda_version: CUDA version like "cu128" or None for CPU

    Returns:
        Index URL for pip install --index-url
    """
    if not cuda_version:
        return CPU_TORCH_INDEX

    for entry in CUDA_DRIVER_MATRIX:
        if entry.cuda_version == cuda_version:
            return entry.torch_index

    return CPU_TORCH_INDEX


# =============================================================================
# Git Timeout Configuration
# =============================================================================
#
# WHY GIT TIMEOUT HANDLING:
# Issue #111: Users behind GFW or on slow VPNs experience Git timeouts.
# Default Git timeout is 21 seconds, which is too short for large repos.
#
# SOLUTION:
# Detect timeout errors and automatically configure Git with extended timeouts.
#

def is_git_timeout_error(error_output: str) -> bool:
    """
    Check if error output indicates a Git timeout.

    Args:
        error_output: stderr from failed command

    Returns:
        True if this looks like a Git timeout
    """
    error_lower = error_output.lower()
    return any(pattern.lower() in error_lower for pattern in GIT_TIMEOUT_PATTERNS)


def configure_git_timeouts() -> bool:
    """
    Configure Git with extended timeouts for slow connections.

    WHY THESE VALUES:
    - http.connectTimeout: 120s (default 21s is too short for GFW)
    - http.timeout: 300s for slow downloads
    - http.lowSpeedLimit: 0 disables early abort
    - http.lowSpeedTime: 999999 effectively disabled
    - http.postBuffer: 500MB for large pushes
    - http.maxRetries: 5 for transient failures

    Returns:
        True if configuration succeeded
    """
    configs = [
        ("http.connectTimeout", "120"),
        ("http.timeout", "300"),
        ("http.lowSpeedLimit", "0"),
        ("http.lowSpeedTime", "999999"),
        ("http.postBuffer", "524288000"),
        ("http.maxRetries", "5"),
    ]

    success = True
    for key, value in configs:
        try:
            subprocess.run(
                ["git", "config", "--global", key, value],
                check=True,
                capture_output=True,
                timeout=10,
            )
        except Exception:
            success = False

    return success


# =============================================================================
# Package Installation with Retry
# =============================================================================
#
# WHY RETRY LOGIC:
# Network issues are transient. A single failure shouldn't abort installation.
#
# RETRY STRATEGY:
# - 3 attempts (configurable)
# - 5 second delay between attempts
# - Git timeout detection with auto-configuration
# - Detailed logging for debugging
#

def run_pip_command(
    args: List[str],
    description: str = "",
    timeout: int = DEFAULT_TIMEOUT,
    retry_count: int = DEFAULT_RETRY_COUNT,
    retry_delay: int = DEFAULT_RETRY_DELAY,
    uv_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    env: Optional[Dict[str, str]] = None,
) -> InstallResult:
    """
    Run pip/uv command with retry logic and Git timeout handling.

    WHY STRUCTURED APPROACH:
    - Retry handles transient network failures
    - Git timeout detection auto-configures for GFW users
    - uv support for faster installation
    - Detailed results for debugging

    Args:
        args: Arguments for pip (without "pip install")
        description: Human-readable description for logging
        timeout: Command timeout in seconds
        retry_count: Number of retry attempts
        retry_delay: Delay between retries in seconds
        uv_path: Path to uv executable (if available)
        logger: Logger for output (optional)
        env: Environment variables (optional)

    Returns:
        InstallResult with success status and details
    """
    import time as time_module

    # Build command
    if uv_path and uv_path.exists():
        python_exe = sys.executable
        cmd = [str(uv_path), "pip", "install", "--python", python_exe] + args
    else:
        cmd = [sys.executable, "-m", "pip", "install"] + args

    # Merge environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    # Set uv timeout if using uv
    if uv_path:
        run_env["UV_HTTP_TIMEOUT"] = "300"

    git_timeouts_configured = False
    last_stdout = ""
    last_stderr = ""
    last_return_code = 0
    start_time = time_module.time()

    for attempt in range(1, retry_count + 1):
        if logger:
            logger.info(f"Attempt {attempt}/{retry_count}: {description}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env,
            )
            last_stdout = result.stdout
            last_stderr = result.stderr
            last_return_code = result.returncode

            if result.returncode == 0:
                duration = time_module.time() - start_time
                return InstallResult(
                    success=True,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    return_code=0,
                    duration_seconds=duration,
                    attempts=attempt,
                    command=" ".join(cmd),
                )

            # Check for Git timeout
            if is_git_timeout_error(result.stderr) and not git_timeouts_configured:
                if logger:
                    logger.warning("Git timeout detected, configuring extended timeouts...")
                configure_git_timeouts()
                git_timeouts_configured = True

        except subprocess.TimeoutExpired:
            last_stderr = f"Command timed out after {timeout} seconds"
            last_return_code = -1

        except Exception as e:
            last_stderr = str(e)
            last_return_code = -1

        # Wait before retry (except on last attempt)
        if attempt < retry_count:
            if logger:
                logger.info(f"Retrying in {retry_delay} seconds...")
            time_module.sleep(retry_delay)

    duration = time_module.time() - start_time
    return InstallResult(
        success=False,
        stdout=last_stdout,
        stderr=last_stderr,
        return_code=last_return_code,
        duration_seconds=duration,
        attempts=retry_count,
        command=" ".join(cmd),
    )


# =============================================================================
# Package Manager Detection
# =============================================================================
#
# WHY UV SUPPORT:
# uv is 10-30x faster than pip for large installs.
# It's bundled with conda-constructor installers.
#

def detect_uv() -> Optional[Path]:
    """
    Detect uv package manager.

    WHY UV:
    - 10-30x faster than pip
    - Better dependency resolution
    - Built-in caching

    Returns:
        Path to uv executable or None
    """
    # Check in current environment
    env_uv = Path(sys.prefix) / "uv.exe"
    if env_uv.exists():
        return env_uv

    # Check in PATH
    uv_path = shutil.which("uv")
    if uv_path:
        return Path(uv_path)

    return None


# =============================================================================
# Module Exports
# =============================================================================
#
# NOTE: This file is designed to be self-contained.
# These exports are for documentation purposes.
#

__all__ = [
    # Types
    "CUDADriverEntry",
    "GPUInfo",
    "InstallResult",
    # Constants
    "CUDA_DRIVER_MATRIX",
    "CPU_TORCH_INDEX",
    "DEFAULT_RETRY_COUNT",
    "DEFAULT_RETRY_DELAY",
    "DEFAULT_TIMEOUT",
    "GIT_INSTALL_TIMEOUT",
    "GIT_TIMEOUT_PATTERNS",
    # Utilities
    "parse_version_string",
    "format_version_tuple",
    # GPU Detection
    "detect_gpu",
    "detect_gpu_nvidia_smi",
    "detect_gpu_pynvml",
    "detect_gpu_proc",
    "select_cuda_version",
    "get_torch_index_url",
    # Git Timeout
    "is_git_timeout_error",
    "configure_git_timeouts",
    # Installation
    "run_pip_command",
    "detect_uv",
]
