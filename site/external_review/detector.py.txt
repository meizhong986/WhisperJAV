"""
Platform and GPU Detector
==========================

INSTITUTIONAL KNOWLEDGE - GPU DETECTION IS CRITICAL

This module provides unified detection of:
- Operating system and architecture
- NVIDIA GPU presence and driver version
- Recommended CUDA version based on driver
- Apple Silicon Metal support
- Installation prerequisites (FFmpeg, Git, WebView2, VC++)

WHY THIS MODULE EXISTS:
----------------------
Before this refactor, GPU detection was duplicated across:
- install_windows.bat (Windows-specific nvidia-smi parsing)
- post_install.py.template (detect_nvidia_driver function)
- install.py (partial detection)

This duplication caused:
1. Inconsistent detection logic (different thresholds)
2. Some paths missing GPU detection entirely
3. Hard-to-maintain platform-specific code scattered everywhere

GPU DETECTION STRATEGY:
----------------------
We use multiple detection methods in order of preference:
1. nvidia-smi command (most reliable, requires NVIDIA drivers)
2. pynvml library (Python bindings, may not be installed)
3. /proc/driver/nvidia/version (Linux fallback)

Each method is tried in order; first success wins.

CUDA VERSION SELECTION:
----------------------
The driver version determines the MAXIMUM CUDA version supported:
- Driver 570+ → CUDA 12.8 (cu128) - RTX 20xx/30xx/40xx/50xx modern drivers
- Driver 450+ → CUDA 11.8 (cu118) - Universal fallback
- Older → CPU only

VERIFICATION SOURCE:
https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
Section: "CUDA Toolkit and Corresponding Driver Versions"

WHY WE DON'T ALWAYS USE NEWEST CUDA:
- cu118 has widest ecosystem compatibility
- Some packages don't have cu128 builds yet
- cu118 is safe default that works for most users

Author: Senior Architect
Date: 2026-01-26
Issue References: GPU detection consistency across all install paths
"""

import os
import sys
import subprocess
import shutil
import re
import platform
from typing import Optional, Tuple, NamedTuple, List
from enum import Enum, auto

from .config import (
    CUDA_DRIVER_MATRIX,
    CPU_TORCH_INDEX,
    PYTHON_MIN_VERSION,
    PYTHON_MAX_VERSION,
)


# =============================================================================
# Enums and Types
# =============================================================================


class DetectedPlatform(Enum):
    """
    Detected operating system and architecture.

    WHY SEPARATE FROM registry.Platform:
    - registry.Platform is for package filtering (declarative)
    - DetectedPlatform is for runtime detection (descriptive)
    - Different purposes, so separate types
    """
    WINDOWS = auto()
    LINUX = auto()
    MACOS_INTEL = auto()
    MACOS_SILICON = auto()
    UNKNOWN = auto()


class GPUInfo(NamedTuple):
    """
    GPU detection result.

    WHY NamedTuple:
    - Immutable (can't accidentally modify detection results)
    - Self-documenting field names
    - Easy to serialize/log

    FIELDS:
    - detected: Was a compatible GPU found?
    - name: GPU model name (e.g., "NVIDIA GeForce RTX 4090")
    - driver_version: Tuple (major, minor, patch) or None
    - cuda_version: Selected CUDA version ("cu118", "cu128") or None for CPU
    - torch_index: URL for pip --index-url
    - detection_method: How we detected ("nvidia-smi", "nvml", "proc", "none")
    - message: Human-readable status message
    """
    detected: bool
    name: Optional[str]
    driver_version: Optional[Tuple[int, int, int]]
    cuda_version: Optional[str]
    torch_index: str
    detection_method: str
    message: str


class PrerequisiteResult(NamedTuple):
    """
    Result of prerequisite check.

    FIELDS:
    - name: Prerequisite name (e.g., "FFmpeg", "Git")
    - found: Was it found?
    - version: Version string if found
    - path: Path to executable if found
    - message: Human-readable status
    """
    name: str
    found: bool
    version: Optional[str] = None
    path: Optional[str] = None
    message: str = ""


# =============================================================================
# Platform Detection
# =============================================================================


def detect_platform() -> DetectedPlatform:
    """
    Detect current operating system and architecture.

    WHY DETECT ARCHITECTURE:
    - macOS on Apple Silicon needs different handling (Metal instead of CUDA)
    - Windows/Linux x64 need NVIDIA GPU detection
    - Platform-specific packages need filtering

    Returns:
        DetectedPlatform enum value
    """
    system = sys.platform
    machine = platform.machine().lower()

    if system == "win32":
        return DetectedPlatform.WINDOWS

    elif system.startswith("linux"):
        return DetectedPlatform.LINUX

    elif system == "darwin":
        # macOS - distinguish Intel from Apple Silicon
        if machine in ("arm64", "aarch64"):
            return DetectedPlatform.MACOS_SILICON
        else:
            return DetectedPlatform.MACOS_INTEL

    else:
        return DetectedPlatform.UNKNOWN


def get_platform_name() -> str:
    """
    Get human-readable platform name.

    Returns:
        String like "Windows", "Linux", "macOS (Apple Silicon)"
    """
    plat = detect_platform()
    return {
        DetectedPlatform.WINDOWS: "Windows",
        DetectedPlatform.LINUX: "Linux",
        DetectedPlatform.MACOS_INTEL: "macOS (Intel)",
        DetectedPlatform.MACOS_SILICON: "macOS (Apple Silicon)",
        DetectedPlatform.UNKNOWN: f"Unknown ({sys.platform})",
    }[plat]


# =============================================================================
# GPU Detection
# =============================================================================


def detect_gpu() -> GPUInfo:
    """
    Detect NVIDIA GPU and select appropriate CUDA version.

    Detection methods (in order):
    1. nvidia-smi command
    2. /proc/driver/nvidia/version (Linux fallback)
    3. Registry check (Windows fallback - not implemented yet)

    CUDA selection based on driver version:
    - Driver 570+ → cu128 (CUDA 12.8)
    - Driver 450+ → cu118 (CUDA 11.8)
    - Otherwise → CPU

    Returns:
        GPUInfo with detection results

    WHY THIS ORDER:
    - nvidia-smi is most reliable and gives GPU name
    - /proc is fast fallback on Linux
    - Registry is last resort on Windows
    """
    # Check if we're on a platform that could have NVIDIA GPU
    plat = detect_platform()
    if plat == DetectedPlatform.MACOS_SILICON:
        return GPUInfo(
            detected=False,
            name=None,
            driver_version=None,
            cuda_version=None,
            torch_index=CPU_TORCH_INDEX,  # macOS uses CPU/Metal, not CUDA
            detection_method="platform",
            message="Apple Silicon detected - CUDA not supported, using Metal",
        )

    # Try nvidia-smi first (most reliable)
    result = _detect_via_nvidia_smi()
    if result.detected:
        return result

    # Try /proc on Linux
    if plat == DetectedPlatform.LINUX:
        result = _detect_via_proc()
        if result.detected:
            return result

    # No GPU detected
    return GPUInfo(
        detected=False,
        name=None,
        driver_version=None,
        cuda_version=None,
        torch_index=CPU_TORCH_INDEX,
        detection_method="none",
        message="No NVIDIA GPU detected - using CPU",
    )


def _detect_via_nvidia_smi() -> GPUInfo:
    """
    Detect GPU using nvidia-smi command.

    nvidia-smi output parsing:
    - Driver version: "Driver Version: 570.00.00"
    - GPU name: "NVIDIA GeForce RTX 4090"

    WHY nvidia-smi:
    - Most reliable detection method
    - Works on Windows and Linux
    - Gives both driver version and GPU name
    - Fails cleanly if NVIDIA drivers not installed
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return GPUInfo(
            detected=False,
            name=None,
            driver_version=None,
            cuda_version=None,
            torch_index=CPU_TORCH_INDEX,
            detection_method="nvidia-smi",
            message="nvidia-smi not found in PATH",
        )

    try:
        # Get driver version
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return GPUInfo(
                detected=False,
                name=None,
                driver_version=None,
                cuda_version=None,
                torch_index=CPU_TORCH_INDEX,
                detection_method="nvidia-smi",
                message=f"nvidia-smi failed: {result.stderr}",
            )

        driver_str = result.stdout.strip().split("\n")[0]  # First GPU
        driver_version = _parse_driver_version(driver_str)

        # Get GPU name
        name_result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_name = name_result.stdout.strip().split("\n")[0] if name_result.returncode == 0 else None

        # Select CUDA version based on driver
        cuda_version, torch_index, message = _select_cuda_version(driver_version)

        return GPUInfo(
            detected=True,
            name=gpu_name,
            driver_version=driver_version,
            cuda_version=cuda_version,
            torch_index=torch_index,
            detection_method="nvidia-smi",
            message=message,
        )

    except subprocess.TimeoutExpired:
        return GPUInfo(
            detected=False,
            name=None,
            driver_version=None,
            cuda_version=None,
            torch_index=CPU_TORCH_INDEX,
            detection_method="nvidia-smi",
            message="nvidia-smi timed out",
        )
    except Exception as e:
        return GPUInfo(
            detected=False,
            name=None,
            driver_version=None,
            cuda_version=None,
            torch_index=CPU_TORCH_INDEX,
            detection_method="nvidia-smi",
            message=f"nvidia-smi error: {e}",
        )


def _detect_via_proc() -> GPUInfo:
    """
    Detect GPU using /proc filesystem (Linux only).

    WHY /proc:
    - Faster than nvidia-smi
    - Works even if nvidia-smi is not in PATH
    - Reliable fallback on Linux systems

    FILE: /proc/driver/nvidia/version
    FORMAT: "NVRM version: NVIDIA UNIX x86_64 Kernel Module  570.00.00  ..."
    """
    proc_path = "/proc/driver/nvidia/version"

    if not os.path.exists(proc_path):
        return GPUInfo(
            detected=False,
            name=None,
            driver_version=None,
            cuda_version=None,
            torch_index=CPU_TORCH_INDEX,
            detection_method="proc",
            message="/proc/driver/nvidia/version not found",
        )

    try:
        with open(proc_path, "r") as f:
            content = f.read()

        # Parse driver version from content
        # Example: "NVRM version: NVIDIA UNIX x86_64 Kernel Module  570.00.00  ..."
        match = re.search(r"Module\s+(\d+)\.(\d+)\.(\d+)", content)
        if not match:
            # Try alternate format
            match = re.search(r"(\d+)\.(\d+)\.(\d+)", content)

        if match:
            driver_version = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            cuda_version, torch_index, message = _select_cuda_version(driver_version)

            return GPUInfo(
                detected=True,
                name=None,  # /proc doesn't give GPU name
                driver_version=driver_version,
                cuda_version=cuda_version,
                torch_index=torch_index,
                detection_method="proc",
                message=message,
            )
        else:
            return GPUInfo(
                detected=False,
                name=None,
                driver_version=None,
                cuda_version=None,
                torch_index=CPU_TORCH_INDEX,
                detection_method="proc",
                message="Could not parse driver version from /proc",
            )

    except Exception as e:
        return GPUInfo(
            detected=False,
            name=None,
            driver_version=None,
            cuda_version=None,
            torch_index=CPU_TORCH_INDEX,
            detection_method="proc",
            message=f"/proc error: {e}",
        )


def _parse_driver_version(version_str: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse driver version string into tuple.

    Examples:
    - "570.00.00" → (570, 0, 0)
    - "450.80.02" → (450, 80, 2)
    - "550.54" → (550, 54, 0)

    WHY TUPLE:
    - Easy version comparison: (570, 0, 0) > (450, 0, 0)
    - Handles major.minor.patch consistently
    """
    if not version_str:
        return None

    # Remove any non-version characters
    version_str = version_str.strip()

    # Try to parse
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version_str)
    if match:
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3)) if match.group(3) else 0
        return (major, minor, patch)

    return None


def _select_cuda_version(
    driver_version: Optional[Tuple[int, int, int]]
) -> Tuple[Optional[str], str, str]:
    """
    Select CUDA version based on driver version.

    Uses CUDA_DRIVER_MATRIX from config.py (single source of truth).

    Args:
        driver_version: Tuple (major, minor, patch) or None

    Returns:
        (cuda_version, torch_index_url, message)
    """
    if driver_version is None:
        return None, CPU_TORCH_INDEX, "Driver version unknown - using CPU"

    # Check against CUDA matrix (highest driver requirements first)
    for entry in CUDA_DRIVER_MATRIX:
        if driver_version >= entry.min_driver:
            return (
                entry.cuda_version,
                entry.torch_index,
                f"{entry.description} (driver {driver_version[0]}.{driver_version[1]})",
            )

    # Driver too old for any CUDA version
    return (
        None,
        CPU_TORCH_INDEX,
        f"Driver {driver_version[0]}.{driver_version[1]} too old for CUDA - using CPU",
    )


def get_torch_index_url(cuda_version: Optional[str] = None) -> str:
    """
    Get PyTorch index URL for CUDA version or auto-detect.

    Args:
        cuda_version: "cu118", "cu128", "cpu", or None for auto-detect

    Returns:
        PyTorch index URL
    """
    if cuda_version is None:
        gpu_info = detect_gpu()
        return gpu_info.torch_index

    if cuda_version == "cpu":
        return CPU_TORCH_INDEX

    # Find in CUDA matrix
    for entry in CUDA_DRIVER_MATRIX:
        if entry.cuda_version == cuda_version:
            return entry.torch_index

    # Unknown CUDA version - return CPU as safe fallback
    return CPU_TORCH_INDEX


# =============================================================================
# Prerequisites Checking
# =============================================================================


def check_python_version() -> PrerequisiteResult:
    """
    Check if Python version is compatible.

    Returns:
        PrerequisiteResult with version check status
    """
    version = sys.version_info[:2]
    version_str = f"{version[0]}.{version[1]}"

    if version < PYTHON_MIN_VERSION:
        return PrerequisiteResult(
            name="Python",
            found=True,
            version=version_str,
            path=sys.executable,
            message=f"Python {version_str} is too old. Requires {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}+",
        )

    if version > PYTHON_MAX_VERSION:
        return PrerequisiteResult(
            name="Python",
            found=True,
            version=version_str,
            path=sys.executable,
            message=f"Python {version_str} is too new. Maximum supported is {PYTHON_MAX_VERSION[0]}.{PYTHON_MAX_VERSION[1]}",
        )

    return PrerequisiteResult(
        name="Python",
        found=True,
        version=version_str,
        path=sys.executable,
        message=f"Python {version_str} - OK",
    )


def check_ffmpeg() -> PrerequisiteResult:
    """
    Check if FFmpeg is available.

    WHY FFMPEG:
    - Required for audio extraction from video
    - Used by ffmpeg-python, pydub, imageio
    - Must be in PATH for all these libraries
    """
    ffmpeg_path = shutil.which("ffmpeg")

    if not ffmpeg_path:
        return PrerequisiteResult(
            name="FFmpeg",
            found=False,
            message="FFmpeg not found in PATH. Install from https://ffmpeg.org/download.html",
        )

    # Try to get version
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # First line usually has version: "ffmpeg version 5.1.2 ..."
        version_line = result.stdout.split("\n")[0] if result.stdout else ""
        version_match = re.search(r"version\s+(\S+)", version_line)
        version = version_match.group(1) if version_match else "unknown"
    except Exception:
        version = "unknown"

    return PrerequisiteResult(
        name="FFmpeg",
        found=True,
        version=version,
        path=ffmpeg_path,
        message=f"FFmpeg {version} - OK",
    )


def check_git() -> PrerequisiteResult:
    """
    Check if Git is available.

    WHY GIT:
    - Required for pip install git+https://... packages
    - openai-whisper, stable-ts, ffmpeg-python all install from git
    """
    git_path = shutil.which("git")

    if not git_path:
        return PrerequisiteResult(
            name="Git",
            found=False,
            message="Git not found in PATH. Install from https://git-scm.com/downloads",
        )

    # Get version
    try:
        result = subprocess.run(
            [git_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # "git version 2.39.0"
        version = result.stdout.strip().replace("git version ", "") if result.stdout else "unknown"
    except Exception:
        version = "unknown"

    return PrerequisiteResult(
        name="Git",
        found=True,
        version=version,
        path=git_path,
        message=f"Git {version} - OK",
    )


def check_webview2() -> PrerequisiteResult:
    """
    Check if Microsoft Edge WebView2 Runtime is installed (Windows only).

    WHY WEBVIEW2:
    - Required by pywebview on Windows for modern web rendering
    - Pre-installed on Windows 11, needs manual install on Windows 10
    - Without it, pywebview falls back to IE engine (terrible)

    DETECTION:
    - Check registry for WebView2 installation
    """
    if detect_platform() != DetectedPlatform.WINDOWS:
        return PrerequisiteResult(
            name="WebView2",
            found=True,
            message="Not required on this platform",
        )

    try:
        import winreg

        # Check both per-user and per-machine installations
        paths = [
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"),
        ]

        for hkey, path in paths:
            try:
                with winreg.OpenKey(hkey, path) as key:
                    version, _ = winreg.QueryValueEx(key, "pv")
                    if version:
                        return PrerequisiteResult(
                            name="WebView2",
                            found=True,
                            version=version,
                            message=f"WebView2 {version} - OK",
                        )
            except FileNotFoundError:
                continue

        return PrerequisiteResult(
            name="WebView2",
            found=False,
            message="WebView2 not found. Download from https://developer.microsoft.com/microsoft-edge/webview2/",
        )

    except ImportError:
        return PrerequisiteResult(
            name="WebView2",
            found=False,
            message="Cannot check WebView2 (winreg not available)",
        )


def check_vcredist() -> PrerequisiteResult:
    """
    Check if Visual C++ Redistributable is installed (Windows only).

    WHY VC++:
    - Required by many Python packages with C extensions
    - torch, numpy, scipy all need VC++ runtime
    - Missing VC++ causes cryptic DLL errors

    DETECTION:
    - Check registry for VC++ 2015-2022 (x64)
    """
    if detect_platform() != DetectedPlatform.WINDOWS:
        return PrerequisiteResult(
            name="VC++ Redist",
            found=True,
            message="Not required on this platform",
        )

    try:
        import winreg

        # VC++ 2015-2022 Redistributable (x64)
        key_path = r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                installed, _ = winreg.QueryValueEx(key, "Installed")
                if installed:
                    try:
                        version, _ = winreg.QueryValueEx(key, "Version")
                    except FileNotFoundError:
                        version = "unknown"

                    return PrerequisiteResult(
                        name="VC++ Redist",
                        found=True,
                        version=version,
                        message=f"VC++ Redistributable {version} - OK",
                    )
        except FileNotFoundError:
            pass

        return PrerequisiteResult(
            name="VC++ Redist",
            found=False,
            message="VC++ Redistributable not found. Download from https://aka.ms/vs/17/release/vc_redist.x64.exe",
        )

    except ImportError:
        return PrerequisiteResult(
            name="VC++ Redist",
            found=False,
            message="Cannot check VC++ (winreg not available)",
        )


def check_prerequisites() -> dict:
    """
    Check all installation prerequisites.

    Returns comprehensive dict with:
    - python: PrerequisiteResult
    - ffmpeg: PrerequisiteResult
    - git: PrerequisiteResult
    - webview2: PrerequisiteResult (Windows only)
    - vc_redist: PrerequisiteResult (Windows only)
    - gpu: GPUInfo
    - platform: str
    - all_ok: bool
    """
    results = {
        "python": check_python_version(),
        "ffmpeg": check_ffmpeg(),
        "git": check_git(),
        "gpu": detect_gpu(),
        "platform": get_platform_name(),
    }

    # Windows-specific checks
    if detect_platform() == DetectedPlatform.WINDOWS:
        results["webview2"] = check_webview2()
        results["vc_redist"] = check_vcredist()

    # Calculate all_ok (excluding optional GPU)
    required_checks = ["python", "ffmpeg", "git"]
    results["all_ok"] = all(
        results[check].found
        for check in required_checks
        if check in results
    )

    return results


def print_prerequisites_report(results: dict = None):
    """
    Print a formatted prerequisites report.

    Args:
        results: Output from check_prerequisites(), or None to run checks
    """
    if results is None:
        results = check_prerequisites()

    print("\n" + "=" * 60)
    print("  WhisperJAV Prerequisites Check")
    print("=" * 60)

    # Platform
    print(f"\n  Platform: {results['platform']}")

    # Required checks
    for name in ["python", "ffmpeg", "git"]:
        if name in results:
            r = results[name]
            status = "[OK]" if r.found else "[!!]"
            print(f"  {status} {r.message}")

    # Windows-specific
    if "webview2" in results:
        r = results["webview2"]
        status = "[OK]" if r.found else "[!!]"
        print(f"  {status} {r.message}")

    if "vc_redist" in results:
        r = results["vc_redist"]
        status = "[OK]" if r.found else "[!!]"
        print(f"  {status} {r.message}")

    # GPU
    gpu = results["gpu"]
    if gpu.detected:
        print(f"\n  GPU: {gpu.name or 'NVIDIA GPU'}")
        print(f"  Driver: {gpu.driver_version[0]}.{gpu.driver_version[1]}")
        print(f"  CUDA: {gpu.cuda_version}")
        print(f"  {gpu.message}")
    else:
        print(f"\n  GPU: {gpu.message}")

    # Summary
    print("\n" + "-" * 60)
    if results["all_ok"]:
        print("  [OK] All prerequisites satisfied")
    else:
        print("  [!!] Some prerequisites missing - see above")
    print("=" * 60 + "\n")
