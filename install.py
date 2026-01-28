#!/usr/bin/env python3
"""
WhisperJAV Source Installation Script
=====================================

This script handles staged installation of WhisperJAV from source,
working around pip dependency resolution conflicts (Issue #90).

ARCHITECTURAL NOTE:
------------------
This script uses the unified installer module (whisperjav/installer/)
for detection and execution, ensuring consistent behavior across all
installation paths (install.py, .bat, .sh, post_install.py).

WHY STAGED INSTALLATION:
-----------------------
Standard `pip install whisperjav` would fail because:
1. pip resolves all dependencies at once
2. PyTorch on PyPI is CPU-only
3. Whisper packages depend on torch
4. Result: User gets CPU torch even with RTX 4090

Our solution:
1. Install torch FIRST with --index-url for GPU version
2. torch is now "locked in" with CUDA support
3. Install whisper packages - they see torch satisfied
4. Result: User gets GPU inference (correct)

Prerequisites:
    - Python 3.10-3.12 (3.9 dropped due to pysubtrans dependency)
    - FFmpeg in system PATH
    - Git

Linux System Dependencies (Issue #33):
    Before running this script on Linux, install system packages:

    Debian/Ubuntu:
        sudo apt-get update
        sudo apt-get install -y python3-dev build-essential ffmpeg libsndfile1

    Fedora/RHEL:
        sudo dnf install python3-devel gcc ffmpeg libsndfile

    If you encounter audio package build errors, also install:
        sudo apt-get install -y portaudio19-dev

Usage:
    python install.py [options]

Options:
    --cpu-only              Install CPU-only PyTorch (no CUDA)
    --cuda118               Install PyTorch for CUDA 11.8 (driver 450+)
    --cuda128               Install PyTorch for CUDA 12.8 (driver 570+, default)
    --no-speech-enhancement Skip speech enhancement packages
    --minimal               Minimal install (transcription only)
    --dev                   Install in development/editable mode
    --local-llm             Install local LLM (tries prebuilt wheel first)
    --local-llm-build       Install local LLM (builds from source)
    --help                  Show this help message

Examples:
    python install.py                    # Standard install with CUDA 12.8
    python install.py --cuda118          # Install with CUDA 11.8 (older drivers)
    python install.py --cpu-only         # CPU-only install
    python install.py --minimal --dev    # Minimal dev install
    python install.py --local-llm        # Include local LLM (uses prebuilt wheel)
    python install.py --local-llm-build  # Include local LLM (builds if no wheel)

Author: Senior Architect
Date: 2026-01-26
Refactored: Uses whisperjav.installer module for unified behavior
"""

import os
import sys
import argparse
import shutil
import time
import platform as platform_module
from datetime import datetime
from pathlib import Path

# =============================================================================
# Bootstrap: Import installer module
# =============================================================================
#
# WHY BOOTSTRAP:
# This script may run before whisperjav is installed. We need to import
# the installer module directly from the source tree, not from site-packages.
#
# IMPORTANT: We only import detection and config - NOT the registry.
# The registry is for validation and generation, not direct installation.
# This script maintains its own step-based installation order for clarity.
#

# Add source directory to path for imports
_source_dir = Path(__file__).parent
if str(_source_dir) not in sys.path:
    sys.path.insert(0, str(_source_dir))

try:
    # Import from installer module
    from whisperjav.installer import (
        # Detection
        detect_gpu,
        detect_platform,
        check_python_version as _check_python,
        check_ffmpeg as _check_ffmpeg,
        check_git as _check_git,
        DetectedPlatform,

        # Execution
        StepExecutor,
        ExecutionResult,

        # Config
        PYTHON_MIN_VERSION,
        PYTHON_MAX_VERSION,
        DEFAULT_RETRY_COUNT,
        CPU_TORCH_INDEX,
    )
    from whisperjav.installer.core.registry import Package, InstallSource, Extra
    _INSTALLER_AVAILABLE = True
except ImportError as e:
    # Fallback if installer module not available (shouldn't happen)
    print(f"WARNING: Could not import installer module: {e}")
    print("         Using legacy fallback functions.")
    _INSTALLER_AVAILABLE = False


# =============================================================================
# Llama-cpp build utilities
# =============================================================================
#
# WHY SEPARATE IMPORT:
# llama_build_utils.py handles platform-specific wheel selection for
# llama-cpp-python. It needs special import because it may run before
# dependencies are installed.
#

try:
    import importlib.util
    _llama_utils_path = _source_dir / "whisperjav" / "translate" / "llama_build_utils.py"
    _spec = importlib.util.spec_from_file_location("llama_build_utils", _llama_utils_path)
    _llama_build_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_llama_build_utils)
    get_llama_cpp_source_info = _llama_build_utils.get_llama_cpp_source_info
    get_prebuilt_wheel_url = _llama_build_utils.get_prebuilt_wheel_url
    # Alias for install_linux.sh compatibility
    get_llama_cpp_prebuilt_wheel = get_prebuilt_wheel_url
except Exception as e:
    print(f"WARNING: Could not import llama_build_utils: {e}")
    get_llama_cpp_source_info = None
    get_prebuilt_wheel_url = None
    get_llama_cpp_prebuilt_wheel = None


# =============================================================================
# Logging Infrastructure
# =============================================================================
#
# WHY LOGGING TO FILE:
# - Console output scrolls away during long installations
# - Users need logs for troubleshooting failed installations
# - Matches standalone installer behavior
#

_LOG_FILE: Path = None  # Set in main() after we know source_dir


def _init_logging(source_dir: Path):
    """Initialize log file for installation."""
    global _LOG_FILE
    _LOG_FILE = source_dir / "install_log.txt"
    # Clear previous log
    try:
        with open(_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"WhisperJAV Installation Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    except Exception:
        pass


def log(message: str):
    """Log message to console and file with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {message}"
    print(line)
    if _LOG_FILE:
        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def log_section(title: str):
    """Log a section header."""
    log("")
    log("=" * 60)
    log(f"  {title}")
    log("=" * 60)


# =============================================================================
# Preflight Checks (Critical for user experience)
# =============================================================================
#
# WHY PREFLIGHT CHECKS:
# - Fail fast with clear messages before starting long installation
# - Match standalone installer behavior
# - Prevent confusing failures mid-installation
#


def check_disk_space(min_gb: int = 8) -> bool:
    """
    Check if sufficient disk space is available.

    Args:
        min_gb: Minimum required free space in GB

    Returns:
        True if sufficient space, False otherwise
    """
    try:
        total, used, free = shutil.disk_usage(_source_dir)
        free_gb = free / (1024**3)
        log(f"  Disk free space: {free_gb:.1f} GB (minimum: {min_gb} GB)")
        if free_gb < min_gb:
            log(f"  [!!] ERROR: Not enough disk space!")
            log(f"       Please free up at least {min_gb} GB and retry.")
            return False
        log(f"  [OK] Disk space sufficient")
        return True
    except Exception as e:
        log(f"  [WARN] Could not check disk space: {e}")
        return True  # Non-fatal, proceed with caution


def check_network(timeout: int = 10) -> bool:
    """
    Check if network connectivity to PyPI is available.

    Args:
        timeout: Connection timeout in seconds

    Returns:
        True if network available, False otherwise
    """
    log("  Checking network connectivity...")
    try:
        import urllib.request
        urllib.request.urlopen("https://pypi.org", timeout=timeout)
        log("  [OK] Network connectivity to PyPI")
        return True
    except Exception as e:
        log(f"  [!!] ERROR: Cannot reach PyPI: {e}")
        log("       Internet connection is required for package downloads.")
        return False


def check_webview2_windows() -> bool:
    """
    Check if Microsoft Edge WebView2 runtime is installed (Windows only).

    Returns:
        True if detected or not Windows, False if missing on Windows
    """
    if sys.platform != "win32":
        return True

    log("  Checking for Microsoft Edge WebView2...")
    try:
        import winreg
        key_paths = [
            r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
            r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        ]
        for key_path in key_paths:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                winreg.CloseKey(key)
                log("  [OK] WebView2 runtime detected")
                return True
            except FileNotFoundError:
                continue
        log("  [WARN] WebView2 runtime NOT detected")
        log("         The GUI (whisperjav-gui) requires WebView2.")
        log("         Download from: https://go.microsoft.com/fwlink/p/?LinkId=2124703")
        return False
    except Exception as e:
        log(f"  [WARN] Could not check WebView2: {e}")
        return True  # Assume OK if can't check


def check_vc_redist_windows() -> bool:
    """
    Check if Visual C++ 2015-2022 Redistributable is installed (Windows only).

    Returns:
        True if detected or not Windows, False if missing on Windows
    """
    if sys.platform != "win32":
        return True

    log("  Checking for Visual C++ Redistributable...")
    try:
        import winreg
        key_paths = [
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        ]
        for key_path in key_paths:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                winreg.CloseKey(key)
                log("  [OK] Visual C++ Redistributable detected")
                return True
            except FileNotFoundError:
                continue
        log("  [WARN] Visual C++ Redistributable NOT detected")
        log("         PyTorch and native libraries require VC++ 2015-2022.")
        log("         Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        return False
    except Exception as e:
        log(f"  [WARN] Could not check VC++ Redistributable: {e}")
        return True  # Assume OK if can't check


def run_preflight_checks() -> bool:
    """
    Run all preflight checks before starting installation.

    Returns:
        True if all critical checks pass, False otherwise
    """
    log_section("Preflight Checks")

    all_passed = True

    # Critical: Disk space
    if not check_disk_space(8):
        all_passed = False

    # Critical: Network
    if not check_network():
        all_passed = False

    # Windows-specific (non-fatal but important warnings)
    if sys.platform == "win32":
        check_webview2_windows()
        check_vc_redist_windows()

    if not all_passed:
        log("")
        log("=" * 60)
        log("  PREFLIGHT CHECKS FAILED")
        log("=" * 60)
        log("  Please address the issues above before continuing.")
        return False

    log("")
    log("  All preflight checks passed!")
    return True


# =============================================================================
# Git Timeout Detection and Configuration (Issue #111)
# =============================================================================
#
# WHY THIS MATTERS:
# Users behind GFW (Great Firewall) or slow VPN connections experience
# 21-second TCP timeouts when connecting to GitHub. This is the Windows
# TCP retransmission default. We detect this pattern and auto-configure
# Git with extended timeouts.
#

_git_timeouts_configured = False


def is_git_timeout_error(error_output: str) -> bool:
    """
    Detect if error was caused by Git connection timeout.

    The 21-second timeout is Windows TCP retransmission default.
    """
    timeout_patterns = [
        "Failed to connect to github.com port 443 after 21",
        "Connection timed out after",
        "Could not connect to server",
        "Connection reset by peer",
        "Connection refused",
        "error: RPC failed",
        "fatal: unable to access",
    ]
    error_lower = error_output.lower()
    return any(pattern.lower() in error_lower for pattern in timeout_patterns)


def configure_git_for_slow_connections() -> bool:
    """
    Configure Git with extended timeouts for users behind GFW or slow VPN.

    Returns:
        True if configuration succeeded, False otherwise
    """
    global _git_timeouts_configured

    if _git_timeouts_configured:
        return True

    log("")
    log("=" * 60)
    log("  Configuring Git for slow/unstable connections")
    log("=" * 60)
    log("Detected connection timeout. This often happens with VPN or")
    log("behind the Great Firewall. Configuring extended timeouts...")
    log("")

    # Find Git executable
    git_exe = shutil.which("git")
    if not git_exe:
        log("WARNING: Git executable not found. Cannot configure timeouts.")
        return False

    log(f"Using Git: {git_exe}")

    import subprocess

    # Git timeout configurations
    git_configs = [
        ("http.connectTimeout", "120"),
        ("http.timeout", "300"),
        ("http.lowSpeedLimit", "0"),
        ("http.lowSpeedTime", "999999"),
        ("http.postBuffer", "524288000"),
        ("http.maxRetries", "5"),
    ]

    success = True
    for key, value in git_configs:
        try:
            result = subprocess.run(
                [git_exe, "config", "--global", key, value],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                log(f"  + {key} = {value}")
            else:
                log(f"  x Failed to set {key}")
                success = False
        except Exception as e:
            log(f"  x Error setting {key}: {e}")
            success = False

    # Set environment variables for current process
    os.environ["GIT_HTTP_CONNECT_TIMEOUT"] = "120"
    os.environ["GIT_HTTP_TIMEOUT"] = "300"
    log("")
    log("Environment variables set:")
    log("  GIT_HTTP_CONNECT_TIMEOUT=120")
    log("  GIT_HTTP_TIMEOUT=300")

    _git_timeouts_configured = True
    log("")
    log("Git configured for slow connections. Retrying...")
    return success


def create_failure_file(error_message: str):
    """Create a failure marker file with troubleshooting info."""
    if not _source_dir:
        return

    failure_file = _source_dir / "INSTALLATION_FAILED.txt"
    try:
        with open(failure_file, "w", encoding="utf-8") as f:
            f.write("WhisperJAV Installation Failed\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Error: {error_message}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Troubleshooting:\n")
            f.write(f"- Check install_log.txt for details\n")
            f.write("- Ensure Python 3.10-3.12, FFmpeg, and Git are in PATH\n")
            f.write("- Try: pip cache purge && pip install --upgrade pip\n")
            f.write("- For network issues, check firewall/proxy settings\n\n")
            f.write("Support: https://github.com/meizhong986/WhisperJAV/issues\n")
        log(f"Failure details written to: {failure_file}")
    except Exception:
        pass


# =============================================================================
# Installation Functions
# =============================================================================


def print_header(text: str, step: str = None):
    """Print a formatted section header."""
    log("\n" + "=" * 60)
    if step:
        log(f"  {step}: {text}")
    else:
        log(f"  {text}")
    log("=" * 60)


def check_prerequisites():
    """
    Check all prerequisites using the detector module.

    Returns:
        dict with prerequisite check results
    """
    results = {}

    # Python version
    #
    # NOTE ON PrerequisiteResult.found FOR PYTHON:
    # For Python version checks, 'found' is always True (we're running Python).
    # The COMPATIBILITY result is encoded in the message:
    # - "Python 3.11 - OK" = compatible
    # - "Python 3.8 is too old..." = incompatible
    # - "Python 3.14 is too new..." = incompatible
    #
    # We check "- OK" in message to determine actual compatibility.
    #
    if _INSTALLER_AVAILABLE:
        py_result = _check_python()
        results["python"] = py_result
        is_compatible = "- OK" in py_result.message
        status = "[OK]" if is_compatible else "[!!]"
        log(f"  {status} {py_result.message}")

        # Exit on incompatible Python (too old OR too new)
        if not is_compatible:
            log("\n  ERROR: Python version is not compatible with WhisperJAV.")
            log("         Requires Python 3.10-3.12")
            create_failure_file("Python version not compatible (requires 3.10-3.12)")
            sys.exit(1)
    else:
        # Fallback
        major, minor = sys.version_info[:2]
        if major < 3 or (major == 3 and minor < 10):
            log(f"  [!!] Python {major}.{minor} is too old. Requires 3.10+")
            create_failure_file(f"Python {major}.{minor} too old (requires 3.10+)")
            sys.exit(1)
        if minor > 12:
            log(f"  [WARN] Python {major}.{minor} may have compatibility issues")
        log(f"  [OK] Python {major}.{minor}")
        results["python"] = type("Result", (), {"found": True, "version": f"{major}.{minor}"})()

    # FFmpeg
    if _INSTALLER_AVAILABLE:
        ff_result = _check_ffmpeg()
        results["ffmpeg"] = ff_result
        status = "[OK]" if ff_result.found else "[WARN]"
        log(f"  {status} {ff_result.message}")
        if not ff_result.found:
            _print_ffmpeg_instructions()
    else:
        # Fallback
        import subprocess
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            log(f"  [OK] FFmpeg found")
            results["ffmpeg"] = type("Result", (), {"found": True})()
        else:
            log(f"  [WARN] FFmpeg not found")
            _print_ffmpeg_instructions()
            results["ffmpeg"] = type("Result", (), {"found": False})()

    # Git
    if _INSTALLER_AVAILABLE:
        git_result = _check_git()
        results["git"] = git_result
        status = "[OK]" if git_result.found else "[!!]"
        log(f"  {status} {git_result.message}")
    else:
        git_path = shutil.which("git")
        if git_path:
            log(f"  [OK] Git found")
            results["git"] = type("Result", (), {"found": True})()
        else:
            log(f"  [!!] Git not found - required for installation")
            results["git"] = type("Result", (), {"found": False})()

    return results


def _print_ffmpeg_instructions():
    """Print FFmpeg installation instructions."""
    log("         FFmpeg is required for audio/video processing.")
    log("         Install FFmpeg and add it to your PATH before using WhisperJAV.")
    if sys.platform == "linux":
        log("         Linux: sudo apt-get install ffmpeg")
    elif sys.platform == "darwin":
        log("         macOS: brew install ffmpeg")
    else:
        log("         Windows: Download from https://www.gyan.dev/ffmpeg/builds/")


def detect_cuda_version(args) -> str:
    """
    Determine CUDA version based on arguments and GPU detection.

    WHY THIS LOGIC:
    1. User can explicitly request CPU/CUDA version via args
    2. If GPU requested but none detected, fall back to CPU
    3. Auto-detection uses driver version to select optimal CUDA

    Args:
        args: Parsed command-line arguments

    Returns:
        "cpu", "cu118", or "cu128"
    """
    # Explicit user request
    if args.cpu_only:
        return "cpu"
    if args.cuda118:
        return "cu118"
    if args.cuda128:
        return "cu128"

    # Auto-detect GPU
    if _INSTALLER_AVAILABLE:
        gpu_info = detect_gpu()
        if gpu_info.detected:
            log(f"\n  GPU detected: {gpu_info.name}")
            log(f"  Driver: {gpu_info.driver_version[0]}.{gpu_info.driver_version[1]}")
            log(f"  Selected: {gpu_info.cuda_version or 'CPU'}")
            return gpu_info.cuda_version or "cpu"
        else:
            log(f"\n  {gpu_info.message}")
            return "cpu"
    else:
        # Fallback: basic nvidia-smi check
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            gpu_name = result.stdout.strip().split('\n')[0]
            if gpu_name:
                log(f"\n  GPU detected: {gpu_name}")
                # Default to cu118 when GPU detected but can't determine driver
                # (safer than cu128 which requires driver 570+)
                return "cu118"
        except Exception:
            pass
        log("\n  No NVIDIA GPU detected - using CPU")
        return "cpu"


def get_torch_index_url(cuda_version: str) -> str:
    """
    Get PyTorch index URL for specified CUDA version.

    Args:
        cuda_version: "cpu", "cu118", or "cu128"

    Returns:
        PyTorch wheel index URL
    """
    urls = {
        "cpu": "https://download.pytorch.org/whl/cpu",
        "cu118": "https://download.pytorch.org/whl/cu118",
        "cu128": "https://download.pytorch.org/whl/cu128",
    }
    return urls.get(cuda_version, urls["cu128"])


def create_executor(log_file: Path = None) -> StepExecutor:
    """
    Create a StepExecutor for package installation.

    WHY StepExecutor:
    - Provides retry logic (3 attempts by default)
    - Detects Git timeout and auto-configures extended timeouts
    - Supports uv for faster installation
    - Logs all operations for debugging

    Args:
        log_file: Optional path for detailed logging

    Returns:
        Configured StepExecutor instance
    """
    if _INSTALLER_AVAILABLE:
        return StepExecutor(
            log_file=log_file,
            max_retries=DEFAULT_RETRY_COUNT,
        )
    else:
        return None


def run_pip(executor: StepExecutor, args: list, description: str, allow_fail: bool = False) -> bool:
    """
    Run pip command with retry logic and Git timeout detection.

    WHY THIS IMPLEMENTATION:
    - Retry logic (3 attempts by default)
    - Captures output for Git timeout detection
    - Auto-configures Git with extended timeouts when timeout detected
    - Logs all operations for troubleshooting

    Args:
        executor: StepExecutor instance or None (currently unused, kept for API compat)
        args: pip arguments (without "pip install")
        description: Human-readable description
        allow_fail: If True, don't raise on failure

    Returns:
        True if successful, False if failed (when allow_fail=True)
    """
    import subprocess
    global _git_timeouts_configured

    # Build command
    cmd = [sys.executable, "-m", "pip"] + args

    # Check if this is a git+https install (more timeout-sensitive)
    is_git_install = any("git+" in str(arg) for arg in args)

    # Add timeout flag for git installs
    if is_git_install and "--timeout" not in str(args):
        # Insert timeout after "install" if present
        if "install" in args:
            idx = args.index("install") + 1
            args = args[:idx] + ["--timeout=120"] + args[idx:]
            cmd = [sys.executable, "-m", "pip"] + args

    log(f"\n>>> {description}")
    cmd_display = ' '.join(cmd[:8])
    if len(cmd) > 8:
        cmd_display += "..."
    log(f"    {cmd_display}")

    retry_count = DEFAULT_RETRY_COUNT if _INSTALLER_AVAILABLE else 3

    for attempt in range(1, retry_count + 1):
        try:
            log(f"    [Attempt {attempt}/{retry_count}]")

            # Capture output for timeout detection
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                timeout=1800  # 30 minute timeout for large downloads
            )

            # Log last few lines of output
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        log(f"    {line}")

            log(f"    [OK] {description}")
            return True

        except subprocess.TimeoutExpired:
            log(f"    [!] Timeout (30 min) - attempt {attempt}")
            if attempt < retry_count:
                log("    Retrying in 10 seconds...")
                time.sleep(10)

        except subprocess.CalledProcessError as e:
            error_output = ""
            if e.stdout:
                error_output += e.stdout
            if e.stderr:
                error_output += e.stderr

            # Log error output (last 10 lines)
            if error_output:
                lines = error_output.strip().split('\n')
                for line in lines[-10:]:
                    if line.strip():
                        log(f"    {line}")

            # Check for Git timeout pattern (Issue #111)
            if is_git_install and is_git_timeout_error(error_output):
                if not _git_timeouts_configured:
                    log("")
                    log("    Detected Git connection timeout!")
                    configure_git_for_slow_connections()
                    # Don't count this as a failed attempt
                    log("    Retrying with extended timeouts...")
                    time.sleep(5)
                    continue

            if attempt < retry_count:
                log(f"    [!] Attempt {attempt} failed, retrying in 10s...")
                time.sleep(10)
            elif allow_fail:
                log(f"    [WARN] {description} - failed (optional)")
                return False
            else:
                log(f"    [ERROR] {description} - failed after {retry_count} attempts")
                raise

        except Exception as e:
            log(f"    [!] Unexpected error: {e}")
            if attempt < retry_count:
                log("    Retrying in 10 seconds...")
                time.sleep(10)
            elif allow_fail:
                log(f"    [WARN] {description} - failed (optional)")
                return False
            else:
                log(f"    [ERROR] {description} - failed")
                raise

    return False


# =============================================================================
# Main Installation Logic
# =============================================================================


def main():
    """
    Main installation function.

    INSTALLATION ORDER (CRITICAL):
    1. Preflight checks (disk, network, WebView2, VC++)
    2. Upgrade pip (ensure latest features)
    3. PyTorch (MUST BE FIRST - GPU lock-in)
    4. Core dependencies (numpy before numba)
    5. Whisper packages (see torch as satisfied)
    6. Optional packages (HuggingFace, translation, VAD, etc.)
    7. WhisperJAV (with --no-deps to preserve our torch)
    """
    # -------------------------------------------------------------------------
    # Argument Parsing
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="WhisperJAV Source Installation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--cpu-only", action="store_true",
                        help="Install CPU-only PyTorch")
    parser.add_argument("--cuda118", action="store_true",
                        help="Install PyTorch for CUDA 11.8 (driver 450+)")
    parser.add_argument("--cuda128", action="store_true",
                        help="Install PyTorch for CUDA 12.8 (driver 570+, default)")
    parser.add_argument("--no-speech-enhancement", action="store_true",
                        help="Skip speech enhancement packages")
    parser.add_argument("--minimal", action="store_true",
                        help="Minimal install (transcription only)")
    parser.add_argument("--dev", action="store_true",
                        help="Install in development/editable mode")
    parser.add_argument("--local-llm", action="store_true",
                        help="Install local LLM support (fast - prebuilt wheel only)")
    parser.add_argument("--local-llm-build", action="store_true",
                        help="Install local LLM support (slow - build from source if needed)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip preflight checks (disk space, network, etc.)")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Check source directory
    # -------------------------------------------------------------------------
    if not (_source_dir / "pyproject.toml").exists():
        print("ERROR: pyproject.toml not found.")
        print("       Run this script from the WhisperJAV source directory.")
        print("       git clone https://github.com/meizhong986/whisperjav.git")
        print("       cd whisperjav")
        print("       python install.py")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Initialize logging
    # -------------------------------------------------------------------------
    _init_logging(_source_dir)
    log_section("WhisperJAV Source Installation")
    log(f"Python: {sys.version}")
    log(f"Platform: {sys.platform}")
    log(f"Source directory: {_source_dir}")

    # -------------------------------------------------------------------------
    # Preflight checks (disk space, network, etc.)
    # -------------------------------------------------------------------------
    if not args.skip_preflight:
        if not run_preflight_checks():
            create_failure_file("Preflight checks failed")
            sys.exit(1)
    else:
        log("Skipping preflight checks (--skip-preflight)")

    # -------------------------------------------------------------------------
    # Prerequisites and GPU detection
    # -------------------------------------------------------------------------
    log_section("Checking Prerequisites")
    prereqs = check_prerequisites()

    # Check for critical failures
    #
    # WHY GIT IS REQUIRED:
    # Several packages are installed from Git URLs (stable-ts, faster-whisper, etc.)
    # Without Git, pip cannot clone these repositories.
    #
    if not prereqs["git"].found:
        log("\nERROR: Git is required for installation.")
        log("       Install Git from: https://git-scm.com/download/")
        create_failure_file("Git not found (required for installation)")
        sys.exit(1)

    cuda_version = detect_cuda_version(args)

    # -------------------------------------------------------------------------
    # Print installation plan
    # -------------------------------------------------------------------------
    log_section("Installation Plan")
    log(f"  PyTorch: {cuda_version}")
    log(f"  Speech Enhancement: {'No' if args.no_speech_enhancement or args.minimal else 'Yes'}")
    log(f"  Local LLM: {'Yes' if args.local_llm or args.local_llm_build else 'No'}")
    log(f"  Mode: {'Development' if args.dev else 'Standard'}")
    log(f"  Log file: {_LOG_FILE}")

    # Create executor for retry/timeout handling
    log_file = _source_dir / "install_log.txt"
    executor = create_executor(log_file)

    # -------------------------------------------------------------------------
    # Step 1: Upgrade pip
    # -------------------------------------------------------------------------
    #
    # WHY FIRST:
    # Older pip versions have dependency resolution bugs that cause failures.
    # Upgrading first ensures we have the latest resolver.
    #
    print_header("Upgrading pip", "Step 1/6")
    run_pip(executor, ["install", "--upgrade", "pip"], "Upgrade pip")

    # -------------------------------------------------------------------------
    # Step 2: Install PyTorch
    # -------------------------------------------------------------------------
    #
    # WHY SECOND (BEFORE EVERYTHING):
    # This is the GPU lock-in step. By installing torch with --index-url
    # pointing to CUDA wheels, we ensure GPU version is installed.
    # All subsequent packages that depend on torch will see it as satisfied.
    #
    print_header("Installing PyTorch", "Step 2/6")
    torch_url = get_torch_index_url(cuda_version)
    run_pip(
        executor,
        ["install", "torch", "torchaudio", "--index-url", torch_url],
        f"Install PyTorch ({cuda_version})"
    )

    # -------------------------------------------------------------------------
    # Step 3: Install core dependencies
    # -------------------------------------------------------------------------
    #
    # WHY THIS ORDER:
    # - numpy MUST be installed before numba (binary ABI compatibility)
    # - scipy depends on numpy
    # - These are the foundation for all audio processing
    #
    print_header("Installing core dependencies", "Step 3/6")
    core_deps = [
        # Scientific stack (order matters: numpy before numba)
        "numpy>=1.26.0,<2.0",  # NumPy 1.26.x for pyvideotrans compatibility
        "scipy>=1.10.1",
        "numba>=0.58.0",  # 0.58.0+ supports NumPy 1.22-2.0

        # Audio processing
        "librosa>=0.10.0",
        "soundfile",
        "pydub",
        "pyloudnorm",

        # Subtitle processing
        "pysrt",
        "srt",

        # Utilities
        "tqdm",
        "colorama",
        "requests",
        "regex",
        "aiofiles",
        "jsonschema",

        # Configuration
        "pydantic>=2.0,<3.0",
        "PyYAML>=6.0",

        # pyvideotrans compatibility (Phase 1 prep)
        "av>=13.0.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.9",
        "httpx>=0.27.0",
        "websockets>=13.0",
        "soxr>=0.3.0",
    ]
    run_pip(executor, ["install"] + core_deps, "Install core dependencies")

    # -------------------------------------------------------------------------
    # Step 4: Install Whisper packages
    # -------------------------------------------------------------------------
    #
    # WHY GIT INSTALL:
    # - openai-whisper: Main branch has latest fixes not on PyPI
    # - stable-ts: Custom fork with setup.py fixes
    # - ffmpeg-python: PyPI tarball has build issues
    #
    # WHY AFTER PYTORCH:
    # These packages depend on torch. Since torch is already installed with
    # GPU support, pip will NOT try to install CPU torch to satisfy deps.
    #
    print_header("Installing Whisper packages", "Step 4/6")
    run_pip(
        executor,
        ["install", "git+https://github.com/openai/whisper@main"],
        "Install openai-whisper from GitHub"
    )
    run_pip(
        executor,
        ["install", "git+https://github.com/meizhong986/stable-ts-fix-setup.git@main"],
        "Install stable-ts from GitHub"
    )
    run_pip(
        executor,
        ["install", "git+https://github.com/kkroening/ffmpeg-python.git"],
        "Install ffmpeg-python from GitHub"
    )
    run_pip(executor, ["install", "faster-whisper>=1.1.0"], "Install faster-whisper")

    # -------------------------------------------------------------------------
    # Step 5: Install optional packages
    # -------------------------------------------------------------------------
    print_header("Installing optional packages", "Step 5/6")

    # HuggingFace / Transformers
    run_pip(
        executor,
        ["install", "huggingface-hub>=0.25.0", "transformers>=4.40.0", "accelerate>=0.26.0"],
        "Install HuggingFace packages"
    )

    # Translation (pysubtrans requires Python 3.10+)
    run_pip(
        executor,
        ["install", "pysubtrans>=1.5.0", "openai>=1.35.0", "google-genai>=1.39.0"],
        "Install translation packages"
    )

    # -------------------------------------------------------------------------
    # Local LLM (llama-cpp-python) - OPTIONAL
    # -------------------------------------------------------------------------
    #
    # WHY SPECIAL HANDLING:
    # llama-cpp-python requires platform-specific builds:
    # - Apple Silicon: Build from source with Metal (fast ~10min)
    # - Intel Mac: CPU only (no Metal)
    # - Windows/Linux: Prebuilt CUDA wheels available from HuggingFace
    #
    # We use get_prebuilt_wheel_url() from llama_build_utils.py to select
    # the correct wheel based on platform and CUDA version.
    #
    if args.local_llm or args.local_llm_build:
        _install_local_llm(executor, args.local_llm_build)
    else:
        log("\n    Skipping local LLM (use --local-llm or --local-llm-build to install)")

    # VAD packages
    run_pip(executor, ["install", "silero-vad>=6.0", "auditok"], "Install VAD packages")

    if not args.minimal:
        run_pip(executor, ["install", "ten-vad"], "Install TEN VAD", allow_fail=True)
        run_pip(executor, ["install", "scikit-learn>=1.3.0"], "Install scikit-learn")

    # -------------------------------------------------------------------------
    # Speech Enhancement - OPTIONAL
    # -------------------------------------------------------------------------
    #
    # WHY OPTIONAL:
    # These packages can be tricky to install (compilation, dependencies).
    # We use allow_fail=True so failures don't stop the entire installation.
    #
    if not args.no_speech_enhancement and not args.minimal:
        _install_speech_enhancement(executor)

    # GUI dependencies
    run_pip(executor, ["install", "pywebview>=5.0.0"], "Install PyWebView", allow_fail=True)
    if sys.platform == "win32":
        run_pip(
            executor,
            ["install", "pythonnet>=3.0", "pywin32>=305"],
            "Install Windows GUI deps",
            allow_fail=True
        )

    # -------------------------------------------------------------------------
    # Step 6: Install WhisperJAV
    # -------------------------------------------------------------------------
    #
    # WHY --no-deps:
    # We've carefully installed all dependencies in the correct order.
    # Using --no-deps prevents pip from re-resolving and potentially
    # overwriting our GPU PyTorch with CPU version.
    #
    print_header("Installing WhisperJAV", "Step 6/6")
    if args.dev:
        run_pip(executor, ["install", "--no-deps", "-e", "."], "Install WhisperJAV (editable)")
    else:
        run_pip(executor, ["install", "--no-deps", "."], "Install WhisperJAV")

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------
    print_header("Verifying Installation")
    _verify_installation()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    _print_summary(args)


def _install_local_llm(executor: StepExecutor, build_from_source: bool):
    """
    Install llama-cpp-python with platform-specific handling.

    WHY COMPLEX:
    Different platforms need different builds:
    - Apple Silicon: Metal backend (build from source)
    - Intel Mac: CPU only
    - Windows/Linux with NVIDIA: CUDA wheels from HuggingFace
    - Windows/Linux without NVIDIA: CPU build

    Args:
        executor: StepExecutor for installation
        build_from_source: Whether to build from source if no wheel
    """
    log(f"\n    Installing llama-cpp-python for local LLM translation...")

    is_apple_silicon = (sys.platform == "darwin" and platform_module.machine() == "arm64")
    is_intel_mac = (sys.platform == "darwin" and platform_module.machine() != "arm64")

    if is_apple_silicon:
        # Apple Silicon: build from source with Metal
        log("    Apple Silicon detected - building from source with Metal support.")
        if get_llama_cpp_source_info:
            git_url, backend, cmake_args, env_vars = get_llama_cpp_source_info()
            log(f"    Backend: {backend}")
            for key, value in env_vars.items():
                log(f"    Setting {key}={value}")
                os.environ[key] = value
            if cmake_args:
                log(f"    Setting CMAKE_ARGS={cmake_args}")
                os.environ["CMAKE_ARGS"] = cmake_args
            run_pip(
                executor,
                ["install", git_url],
                f"Install llama-cpp-python ({backend})",
                allow_fail=True
            )
        else:
            log("    ERROR: llama_build_utils not available")

    elif is_intel_mac:
        # Intel Mac: CPU only
        if build_from_source and get_llama_cpp_source_info:
            log("    Intel Mac detected - building CPU-only version.")
            git_url, backend, cmake_args, env_vars = get_llama_cpp_source_info()
            log(f"    Backend: {backend}")
            for key, value in env_vars.items():
                log(f"    Setting {key}={value}")
                os.environ[key] = value
            run_pip(
                executor,
                ["install", git_url],
                f"Install llama-cpp-python ({backend})",
                allow_fail=True
            )
        else:
            log("    Intel Mac detected - no prebuilt wheels available.")
            log("    To build CPU-only version, use --local-llm-build.")
            log("    Skipping local LLM installation.")

    else:
        # Windows/Linux: try prebuilt wheel first
        if get_prebuilt_wheel_url:
            wheel_url, wheel_backend = get_prebuilt_wheel_url(verbose=True)

            if wheel_url:
                log(f"    Backend: {wheel_backend}")
                run_pip(
                    executor,
                    ["install", wheel_url],
                    f"Install llama-cpp-python ({wheel_backend})",
                    allow_fail=True
                )
                run_pip(
                    executor,
                    ["install", "llama-cpp-python[server]"],
                    "Install llama-cpp-python server extras",
                    allow_fail=True
                )
            elif build_from_source and get_llama_cpp_source_info:
                log("    No prebuilt wheel - building from source...")
                git_url, backend, cmake_args, env_vars = get_llama_cpp_source_info()
                log(f"    Backend: {backend}")
                for key, value in env_vars.items():
                    log(f"    Setting {key}={value}")
                    os.environ[key] = value
                if cmake_args:
                    log(f"    Setting CMAKE_ARGS={cmake_args}")
                    os.environ["CMAKE_ARGS"] = cmake_args
                run_pip(
                    executor,
                    ["install", git_url],
                    f"Install llama-cpp-python ({backend})",
                    allow_fail=True
                )
            else:
                log("    No prebuilt wheel available for your platform.")
                log("    To build from source, use --local-llm-build.")
                log("    Skipping local LLM installation.")
        else:
            log("    ERROR: llama_build_utils not available")


def _install_speech_enhancement(executor: StepExecutor):
    """
    Install speech enhancement packages.

    WHY SEPARATE FUNCTION:
    These packages are complex and may fail. We isolate them so failures
    don't affect the rest of the installation.
    """
    log("\n    Installing speech enhancement packages...")
    log("    (These can be tricky - failures here are non-fatal)")

    # ModelScope dependencies (including oss2 required for ZipEnhancer)
    run_pip(
        executor,
        ["install", "addict", "simplejson", "sortedcontainers", "packaging", "oss2"],
        "Install ModelScope dependencies (including oss2)",
        allow_fail=True
    )
    run_pip(
        executor,
        ["install", "datasets>=2.14.0,<4.0"],
        "Install datasets",
        allow_fail=True
    )
    run_pip(
        executor,
        ["install", "modelscope>=1.20"],
        "Install ModelScope",
        allow_fail=True
    )

    # ClearVoice (custom fork with relaxed librosa dependency)
    run_pip(
        executor,
        ["install", "git+https://github.com/meizhong986/ClearerVoice-Studio.git#subdirectory=clearvoice"],
        "Install ClearVoice (forked)",
        allow_fail=True
    )

    # BS-RoFormer vocal isolation
    run_pip(
        executor,
        ["install", "bs-roformer-infer"],
        "Install BS-RoFormer",
        allow_fail=True
    )

    # ONNX Runtime
    run_pip(
        executor,
        ["install", "onnxruntime>=1.16.0"],
        "Install ONNX Runtime",
        allow_fail=True
    )


def _verify_installation():
    """Verify WhisperJAV was installed correctly."""
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "-c", "import whisperjav; print(f'WhisperJAV {whisperjav.__version__}')"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip()
        log(f"    [OK] {version} installed successfully!")

        # Also check torch CUDA availability
        if _INSTALLER_AVAILABLE:
            from whisperjav.installer.core.executor import StepExecutor
            temp_executor = StepExecutor()
            cuda_ok, cuda_msg = temp_executor.verify_torch_cuda()
            status = "[OK]" if cuda_ok else "[INFO]"
            log(f"    {status} {cuda_msg}")

    except subprocess.CalledProcessError:
        log("    [WARN] Could not verify installation")


def _print_summary(args):
    """Print installation summary and next steps."""
    print_header("Installation Complete!")

    log("")
    log("  To run WhisperJAV:")
    log("    whisperjav video.mp4 --mode balanced")
    log("")
    log("  To run with GUI:")
    log("    whisperjav-gui")
    log("")
    log("  For help:")
    log("    whisperjav --help")

    if args.local_llm or args.local_llm_build:
        log("")
        log("  To translate with local LLM (no API key needed):")
        log("    whisperjav video.mp4 --translate --translate-provider local")
    else:
        log("")
        log("  To enable local LLM translation, re-install with:")
        log("    python install.py --local-llm-build    (builds from source)")
        log("    python install.py --local-llm          (uses prebuilt wheel if available)")

    log("")
    log("  If you encounter issues with speech enhancement, re-run with:")
    log("    python install.py --no-speech-enhancement")
    log("")
    if _LOG_FILE:
        log(f"  Installation log saved to: {_LOG_FILE}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
