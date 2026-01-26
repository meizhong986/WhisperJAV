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
import platform as platform_module
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
# Installation Functions
# =============================================================================


def print_header(text: str, step: str = None):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    if step:
        print(f"  {step}: {text}")
    else:
        print(f"  {text}")
    print("=" * 60)


def check_prerequisites():
    """
    Check all prerequisites using the detector module.

    Returns:
        dict with prerequisite check results
    """
    print_header("Checking Prerequisites")

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
        print(f"  {status} {py_result.message}")

        # Exit on incompatible Python (too old OR too new)
        if not is_compatible:
            print("\n  ERROR: Python version is not compatible with WhisperJAV.")
            print("         Requires Python 3.10-3.12")
            sys.exit(1)
    else:
        # Fallback
        major, minor = sys.version_info[:2]
        if major < 3 or (major == 3 and minor < 10):
            print(f"  [!!] Python {major}.{minor} is too old. Requires 3.10+")
            sys.exit(1)
        if minor > 12:
            print(f"  [WARN] Python {major}.{minor} may have compatibility issues")
        print(f"  [OK] Python {major}.{minor}")
        results["python"] = type("Result", (), {"found": True, "version": f"{major}.{minor}"})()

    # FFmpeg
    if _INSTALLER_AVAILABLE:
        ff_result = _check_ffmpeg()
        results["ffmpeg"] = ff_result
        status = "[OK]" if ff_result.found else "[WARN]"
        print(f"  {status} {ff_result.message}")
        if not ff_result.found:
            _print_ffmpeg_instructions()
    else:
        # Fallback
        import subprocess
        import shutil
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            print(f"  [OK] FFmpeg found")
            results["ffmpeg"] = type("Result", (), {"found": True})()
        else:
            print(f"  [WARN] FFmpeg not found")
            _print_ffmpeg_instructions()
            results["ffmpeg"] = type("Result", (), {"found": False})()

    # Git
    if _INSTALLER_AVAILABLE:
        git_result = _check_git()
        results["git"] = git_result
        status = "[OK]" if git_result.found else "[!!]"
        print(f"  {status} {git_result.message}")
    else:
        import shutil
        git_path = shutil.which("git")
        if git_path:
            print(f"  [OK] Git found")
            results["git"] = type("Result", (), {"found": True})()
        else:
            print(f"  [!!] Git not found - required for installation")
            results["git"] = type("Result", (), {"found": False})()

    return results


def _print_ffmpeg_instructions():
    """Print FFmpeg installation instructions."""
    print("         FFmpeg is required for audio/video processing.")
    print("         Install FFmpeg and add it to your PATH before using WhisperJAV.")
    if sys.platform == "linux":
        print("         Linux: sudo apt-get install ffmpeg")
    elif sys.platform == "darwin":
        print("         macOS: brew install ffmpeg")
    else:
        print("         Windows: Download from https://www.gyan.dev/ffmpeg/builds/")


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
            print(f"\n  GPU detected: {gpu_info.name}")
            print(f"  Driver: {gpu_info.driver_version[0]}.{gpu_info.driver_version[1]}")
            print(f"  Selected: {gpu_info.cuda_version or 'CPU'}")
            return gpu_info.cuda_version or "cpu"
        else:
            print(f"\n  {gpu_info.message}")
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
                print(f"\n  GPU detected: {gpu_name}")
                # Default to cu128 when GPU detected (modern default)
                return "cu128"
        except Exception:
            pass
        print("\n  No NVIDIA GPU detected - using CPU")
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
    Run pip command using executor or fallback.

    WHY WRAPPER:
    - Uses StepExecutor when available (retry, timeout handling)
    - Falls back to direct subprocess when executor not available
    - Consistent interface for the rest of the script

    Args:
        executor: StepExecutor instance or None
        args: pip arguments (without "pip install")
        description: Human-readable description
        allow_fail: If True, don't raise on failure

    Returns:
        True if successful, False if failed (when allow_fail=True)
    """
    import subprocess

    # Build command
    cmd = [sys.executable, "-m", "pip"] + args

    print(f"\n>>> {description}")
    print(f"    {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")

    if executor and _INSTALLER_AVAILABLE:
        # Use executor for retry/timeout handling
        # Create a temporary Package for the executor
        pkg = Package(
            name=description,
            source=InstallSource.PYPI,  # Will be overridden by args
            required=not allow_fail,
        )

        # Run directly since executor.install_package expects Package
        # For now, use subprocess with retry logic embedded
        for attempt in range(1, DEFAULT_RETRY_COUNT + 1):
            try:
                result = subprocess.run(cmd, check=True, capture_output=False)
                print(f"    [OK] {description}")
                return True
            except subprocess.CalledProcessError as e:
                if attempt < DEFAULT_RETRY_COUNT:
                    print(f"    [!] Attempt {attempt} failed, retrying...")
                    import time
                    time.sleep(5)
                elif allow_fail:
                    print(f"    [WARN] {description} - failed (optional)")
                    return False
                else:
                    print(f"    [ERROR] {description} - failed")
                    raise
    else:
        # Fallback: direct subprocess
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"    [OK] {description}")
            return True
        except subprocess.CalledProcessError as e:
            if allow_fail:
                print(f"    [WARN] {description} - failed (optional)")
                return False
            else:
                print(f"    [ERROR] {description} - failed")
                raise

    return False


# =============================================================================
# Main Installation Logic
# =============================================================================


def main():
    """
    Main installation function.

    INSTALLATION ORDER (CRITICAL):
    1. Upgrade pip (ensure latest features)
    2. PyTorch (MUST BE FIRST - GPU lock-in)
    3. Core dependencies (numpy before numba)
    4. Whisper packages (see torch as satisfied)
    5. Optional packages (HuggingFace, translation, VAD, etc.)
    6. WhisperJAV (with --no-deps to preserve our torch)
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
    # Prerequisites and GPU detection
    # -------------------------------------------------------------------------
    prereqs = check_prerequisites()

    # Check for critical failures
    #
    # WHY GIT IS REQUIRED:
    # Several packages are installed from Git URLs (stable-ts, faster-whisper, etc.)
    # Without Git, pip cannot clone these repositories.
    #
    if not prereqs["git"].found:
        print("\nERROR: Git is required for installation.")
        print("       Install Git from: https://git-scm.com/download/")
        sys.exit(1)

    cuda_version = detect_cuda_version(args)

    # -------------------------------------------------------------------------
    # Print installation plan
    # -------------------------------------------------------------------------
    print_header("WhisperJAV Source Installation")
    print(f"  PyTorch: {cuda_version}")
    print(f"  Speech Enhancement: {'No' if args.no_speech_enhancement or args.minimal else 'Yes'}")
    print(f"  Local LLM: {'Yes' if args.local_llm or args.local_llm_build else 'No'}")
    print(f"  Mode: {'Development' if args.dev else 'Standard'}")
    print("=" * 60)

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
        print("\n    Skipping local LLM (use --local-llm or --local-llm-build to install)")

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
    print(f"\n    Installing llama-cpp-python for local LLM translation...")

    is_apple_silicon = (sys.platform == "darwin" and platform_module.machine() == "arm64")
    is_intel_mac = (sys.platform == "darwin" and platform_module.machine() != "arm64")

    if is_apple_silicon:
        # Apple Silicon: build from source with Metal
        print("    Apple Silicon detected - building from source with Metal support.")
        if get_llama_cpp_source_info:
            git_url, backend, cmake_args, env_vars = get_llama_cpp_source_info()
            print(f"    Backend: {backend}")
            for key, value in env_vars.items():
                print(f"    Setting {key}={value}")
                os.environ[key] = value
            if cmake_args:
                print(f"    Setting CMAKE_ARGS={cmake_args}")
                os.environ["CMAKE_ARGS"] = cmake_args
            run_pip(
                executor,
                ["install", git_url],
                f"Install llama-cpp-python ({backend})",
                allow_fail=True
            )
        else:
            print("    ERROR: llama_build_utils not available")

    elif is_intel_mac:
        # Intel Mac: CPU only
        if build_from_source and get_llama_cpp_source_info:
            print("    Intel Mac detected - building CPU-only version.")
            git_url, backend, cmake_args, env_vars = get_llama_cpp_source_info()
            print(f"    Backend: {backend}")
            for key, value in env_vars.items():
                print(f"    Setting {key}={value}")
                os.environ[key] = value
            run_pip(
                executor,
                ["install", git_url],
                f"Install llama-cpp-python ({backend})",
                allow_fail=True
            )
        else:
            print("    Intel Mac detected - no prebuilt wheels available.")
            print("    To build CPU-only version, use --local-llm-build.")
            print("    Skipping local LLM installation.")

    else:
        # Windows/Linux: try prebuilt wheel first
        if get_prebuilt_wheel_url:
            wheel_url, wheel_backend = get_prebuilt_wheel_url(verbose=True)

            if wheel_url:
                print(f"    Backend: {wheel_backend}")
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
                print("    No prebuilt wheel - building from source...")
                git_url, backend, cmake_args, env_vars = get_llama_cpp_source_info()
                print(f"    Backend: {backend}")
                for key, value in env_vars.items():
                    print(f"    Setting {key}={value}")
                    os.environ[key] = value
                if cmake_args:
                    print(f"    Setting CMAKE_ARGS={cmake_args}")
                    os.environ["CMAKE_ARGS"] = cmake_args
                run_pip(
                    executor,
                    ["install", git_url],
                    f"Install llama-cpp-python ({backend})",
                    allow_fail=True
                )
            else:
                print("    No prebuilt wheel available for your platform.")
                print("    To build from source, use --local-llm-build.")
                print("    Skipping local LLM installation.")
        else:
            print("    ERROR: llama_build_utils not available")


def _install_speech_enhancement(executor: StepExecutor):
    """
    Install speech enhancement packages.

    WHY SEPARATE FUNCTION:
    These packages are complex and may fail. We isolate them so failures
    don't affect the rest of the installation.
    """
    print("\n    Installing speech enhancement packages...")
    print("    (These can be tricky - failures here are non-fatal)")

    # ModelScope dependencies
    run_pip(
        executor,
        ["install", "addict", "simplejson", "sortedcontainers", "packaging"],
        "Install ModelScope dependencies",
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
        print(f"    [OK] {version} installed successfully!")

        # Also check torch CUDA availability
        if _INSTALLER_AVAILABLE:
            from whisperjav.installer.core.executor import StepExecutor
            temp_executor = StepExecutor()
            cuda_ok, cuda_msg = temp_executor.verify_torch_cuda()
            status = "[OK]" if cuda_ok else "[INFO]"
            print(f"    {status} {cuda_msg}")

    except subprocess.CalledProcessError:
        print("    [WARN] Could not verify installation")


def _print_summary(args):
    """Print installation summary and next steps."""
    print_header("Installation Complete!")

    summary = """
  To run WhisperJAV:
    whisperjav video.mp4 --mode balanced

  To run with GUI:
    whisperjav-gui

  For help:
    whisperjav --help
"""

    if args.local_llm or args.local_llm_build:
        summary += """
  To translate with local LLM (no API key needed):
    whisperjav video.mp4 --translate --translate-provider local
"""
    else:
        summary += """
  To enable local LLM translation, re-install with:
    python install.py --local-llm-build    (builds from source)
    python install.py --local-llm          (uses prebuilt wheel if available)
"""

    summary += """
  If you encounter issues with speech enhancement, re-run with:
    python install.py --no-speech-enhancement
"""

    print(summary)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
