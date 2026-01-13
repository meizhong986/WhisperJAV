#!/usr/bin/env python3
"""
WhisperJAV Source Installation Script
=====================================

This script handles staged installation of WhisperJAV from source,
working around pip dependency resolution conflicts (Issue #90).

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
    --cuda118               Install PyTorch for CUDA 11.8
    --cuda121               Install PyTorch for CUDA 12.1
    --cuda124               Install PyTorch for CUDA 12.4 (default)
    --no-speech-enhancement Skip speech enhancement packages
    --minimal               Minimal install (transcription only)
    --dev                   Install in development/editable mode
    --local-llm             Install local LLM (fast - prebuilt wheel only)
    --local-llm-build       Install local LLM (slow - builds from source if needed)
    --help                  Show this help message

Examples:
    python install.py                    # Standard install with CUDA 12.4
    python install.py --cpu-only         # CPU-only install
    python install.py --minimal --dev    # Minimal dev install
    python install.py --local-llm        # Include local LLM (uses prebuilt wheel)
    python install.py --local-llm-build  # Include local LLM (builds if no wheel)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_pip(args, description, allow_fail=False):
    """Run pip command with error handling."""
    cmd = [sys.executable, "-m", "pip"] + args
    print(f"\n>>> {description}")
    print(f"    {' '.join(cmd)}")

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


def check_python_version():
    """Verify Python version is compatible."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"ERROR: Python 3.10+ required. Found: {major}.{minor}")
        print("       Python 3.9 is no longer supported due to pysubtrans dependency.")
        sys.exit(1)
    if minor > 12:
        print(f"WARNING: Python 3.13+ may have compatibility issues with openai-whisper")
    print(f"Python {major}.{minor} detected")


def detect_nvidia_gpu():
    """Detect if NVIDIA GPU is available."""
    # Try nvidia-smi first (most reliable)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_name = result.stdout.strip().split('\n')[0]
        if gpu_name:
            print(f"NVIDIA GPU detected: {gpu_name}")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: check /proc/driver/nvidia (Linux)
    if sys.platform == "linux":
        try:
            if Path("/proc/driver/nvidia/version").exists():
                print("NVIDIA driver detected")
                return True
        except Exception:
            pass

    print("No NVIDIA GPU detected")
    return False


def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
        version_line = result.stdout.split('\n')[0]
        print(f"FFmpeg found: {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: FFmpeg not found in PATH")
        print("  FFmpeg is required for audio/video processing.")
        print("  Install FFmpeg and add it to your PATH before using WhisperJAV.")
        if sys.platform == "linux":
            print("  Linux: sudo apt-get install ffmpeg")
        elif sys.platform == "darwin":
            print("  macOS: brew install ffmpeg")
        else:
            print("  Windows: Download from https://www.gyan.dev/ffmpeg/builds/")
        return False


def get_torch_index_url(cuda_version):
    """Get PyTorch index URL for specified CUDA version."""
    urls = {
        "cpu": "https://download.pytorch.org/whl/cpu",
        "cuda118": "https://download.pytorch.org/whl/cu118",
        "cuda121": "https://download.pytorch.org/whl/cu121",
        "cuda124": "https://download.pytorch.org/whl/cu124",
    }
    return urls.get(cuda_version, urls["cuda121"])


def get_system_cuda_version():
    """Detect CUDA version from nvcc or nvidia-smi."""
    import re

    # Try nvcc first
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            match = re.search(r"release (\d+)\.(\d+)", result.stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
    except Exception:
        pass

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # nvidia-smi doesn't directly give CUDA version, but presence means CUDA capable
            # Try to get CUDA version from nvidia-smi output
            result2 = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, text=True, timeout=10
            )
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result2.stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
    except Exception:
        pass

    return None, None


def get_llama_cpp_prebuilt_wheel():
    """
    Try to find a prebuilt wheel from JamePeng's releases.

    Auto-detects platform, Python version, and CUDA version.
    Queries GitHub API to find matching wheel dynamically.

    Release naming conventions (from JamePeng/llama-cpp-python):
    - CUDA: v{ver}-cu{cuda}-Basic-{os}-{date} (e.g., v0.3.21-cu130-Basic-win-20260111)
    - Metal: v{ver}-metal-{date} (when available)
    - Wheel: llama_cpp_python-{ver}-cp{py}-cp{py}-{platform}.whl

    Returns:
        tuple: (wheel_url, backend_desc) or (None, None) if no suitable wheel found
    """
    import json
    import platform as platform_module
    import urllib.request
    import urllib.error

    # Determine platform identifiers
    if sys.platform == "win32":
        os_tag = "win"
        wheel_platform = "win_amd64"
    elif sys.platform == "linux":
        os_tag = "linux"
        wheel_platform = "linux_x86_64"
    elif sys.platform == "darwin":
        os_tag = "metal"  # JamePeng uses -metal- tag for macOS
        if platform_module.machine() == "arm64":
            wheel_platform = "arm64"  # Matches macosx_*_arm64
        else:
            wheel_platform = "x86_64"
    else:
        print(f"    Unknown platform: {sys.platform}")
        return None, None

    # Python version
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"

    # Detect CUDA version for Windows/Linux
    target_cudas = []
    if sys.platform in ("win32", "linux"):
        cuda_major, cuda_minor = get_system_cuda_version()
        if cuda_major:
            system_cuda = cuda_major * 10 + (cuda_minor if cuda_minor < 10 else cuda_minor // 10)
            # Build list of compatible CUDA versions (highest first)
            for cu in [130, 128, 126, 124]:
                if cu <= system_cuda:
                    target_cudas.append(f"cu{cu}")
            print(f"    Detected CUDA {cuda_major}.{cuda_minor}, compatible: {target_cudas}")
        else:
            print("    No CUDA detected")

    print(f"    Searching for prebuilt wheel: {os_tag}, {py_ver}, platform={wheel_platform}")

    # Query GitHub API for releases
    try:
        api_url = "https://api.github.com/repos/JamePeng/llama-cpp-python/releases?per_page=50"
        req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=15) as response:
            releases = json.loads(response.read().decode())
    except Exception as e:
        print(f"    Could not fetch releases: {e}")
        return None, None

    # Search strategy:
    # 1. For Windows/Linux with CUDA: find cu{version} release matching OS
    # 2. For macOS: find metal release
    # 3. Match wheel file by Python version and platform

    if os_tag == "metal":
        # macOS: look for -metal- releases
        for release in releases:
            tag = release.get("tag_name", "")
            if "-metal-" not in tag.lower():
                continue
            for asset in release.get("assets", []):
                name = asset.get("name", "")
                if not name.endswith(".whl"):
                    continue
                if py_ver not in name:
                    continue
                if wheel_platform in name:
                    wheel_url = asset.get("browser_download_url")
                    print(f"    Found prebuilt wheel: {name}")
                    return wheel_url, "Metal (prebuilt wheel)"
    else:
        # Windows/Linux: look for CUDA releases
        for cuda_tag in target_cudas:
            for release in releases:
                tag = release.get("tag_name", "")
                # Match: v{ver}-cu{cuda}-Basic-{os}-{date}
                if f"-{cuda_tag}-" not in tag:
                    continue
                if f"-{os_tag}-" not in tag:
                    continue
                for asset in release.get("assets", []):
                    name = asset.get("name", "")
                    if not name.endswith(".whl"):
                        continue
                    if py_ver not in name:
                        continue
                    if wheel_platform in name:
                        wheel_url = asset.get("browser_download_url")
                        print(f"    Found prebuilt wheel: {name}")
                        return wheel_url, f"CUDA ({cuda_tag} prebuilt wheel)"

    print(f"    No matching prebuilt wheel found for {py_ver}/{os_tag}/{wheel_platform}")
    return None, None


def get_llama_cpp_source_info():
    """
    Get llama-cpp-python source build info based on platform.

    Auto-detects GPU/Metal/CPU and returns appropriate build configuration.
    Uses JamePeng's fork which has active maintenance.

    Returns:
        tuple: (git_url, backend_desc, cmake_args)
            - git_url: Git URL for pip install
            - backend_desc: Human-readable description
            - cmake_args: CMAKE_ARGS value or None

    See: https://github.com/JamePeng/llama-cpp-python
    """
    import platform as platform_module

    git_url = "llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git"
    cmake_args = None

    if sys.platform == "darwin":
        chip = platform_module.processor() or platform_module.machine()
        if "arm" in chip.lower() or "apple" in chip.lower():
            backend = "Metal (Apple Silicon) - building from source"
            cmake_args = "-DGGML_METAL=on"
        else:
            backend = "CPU (Intel Mac) - building from source"
    elif sys.platform == "win32":
        cuda_major, _ = get_system_cuda_version()
        if cuda_major:
            backend = "CUDA - building from source (~45 min)"
            cmake_args = "-DGGML_CUDA=on"
        else:
            backend = "CPU - building from source"
    else:
        # Linux
        cuda_major, _ = get_system_cuda_version()
        if cuda_major:
            backend = "CUDA - building from source (~45 min)"
            cmake_args = "-DGGML_CUDA=on"
        else:
            backend = "CPU - building from source"

    return git_url, backend, cmake_args


def main():
    parser = argparse.ArgumentParser(
        description="WhisperJAV Source Installation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--cpu-only", action="store_true",
                        help="Install CPU-only PyTorch")
    parser.add_argument("--cuda118", action="store_true",
                        help="Install PyTorch for CUDA 11.8")
    parser.add_argument("--cuda121", action="store_true",
                        help="Install PyTorch for CUDA 12.1")
    parser.add_argument("--cuda124", action="store_true",
                        help="Install PyTorch for CUDA 12.4 (default)")
    parser.add_argument("--no-speech-enhancement", action="store_true",
                        help="Skip speech enhancement packages")
    parser.add_argument("--minimal", action="store_true",
                        help="Minimal install (transcription only)")
    parser.add_argument("--dev", action="store_true",
                        help="Install in development/editable mode")

    # Local LLM options
    parser.add_argument("--local-llm", action="store_true",
                        help="Install local LLM support (fast install - prebuilt wheel only, skip if unavailable)")
    parser.add_argument("--local-llm-build", action="store_true",
                        help="Install local LLM support (slow install - build from source if no prebuilt wheel)")

    args = parser.parse_args()

    # Determine CUDA version
    if args.cpu_only:
        cuda_version = "cpu"
    elif args.cuda118:
        cuda_version = "cuda118"
    elif args.cuda121:
        cuda_version = "cuda121"
    else:
        cuda_version = "cuda124"  # default (enables prebuilt llama-cpp wheels)

    # Auto-detect GPU and warn if installing CUDA version without GPU
    if cuda_version != "cpu":
        has_gpu = detect_nvidia_gpu()
        if not has_gpu:
            print("\n" + "!" * 60)
            print("  WARNING: No NVIDIA GPU detected!")
            print("  You selected CUDA installation but no GPU was found.")
            print("  Switching to CPU-only installation automatically.")
            print("  (To force CUDA install anyway, edit this script)")
            print("!" * 60 + "\n")
            cuda_version = "cpu"

    # Check we're in the right directory
    setup_py = Path("setup.py")
    if not setup_py.exists():
        print("ERROR: setup.py not found. Run this script from the WhisperJAV source directory.")
        print("       git clone https://github.com/meizhong986/whisperjav.git")
        print("       cd whisperjav")
        print("       python install.py")
        sys.exit(1)

    print("=" * 60)
    print("  WhisperJAV Source Installation")
    print("=" * 60)
    print(f"  PyTorch: {cuda_version}")
    print(f"  Speech Enhancement: {'No' if args.no_speech_enhancement or args.minimal else 'Yes'}")
    print(f"  Mode: {'Development' if args.dev else 'Standard'}")
    print("=" * 60)

    check_python_version()
    check_ffmpeg()

    # Step 1: Upgrade pip
    print("\n" + "=" * 60)
    print("  Step 1/6: Upgrading pip")
    print("=" * 60)
    run_pip(["install", "--upgrade", "pip"], "Upgrade pip")

    # Step 2: Install PyTorch
    print("\n" + "=" * 60)
    print("  Step 2/6: Installing PyTorch")
    print("=" * 60)
    torch_url = get_torch_index_url(cuda_version)
    run_pip(
        ["install", "torch", "torchaudio", "--index-url", torch_url],
        f"Install PyTorch ({cuda_version})"
    )

    # Step 3: Install core dependencies
    print("\n" + "=" * 60)
    print("  Step 3/6: Installing core dependencies")
    print("=" * 60)
    core_deps = [
        "numpy>=2.0", "scipy>=1.14.0", "librosa>=0.11.0",  # scipy 1.14.0+ for NumPy 2.0 ABI
        "soundfile", "pydub", "tqdm", "colorama", "requests", "regex",
        "pysrt", "srt", "aiofiles", "jsonschema", "pyloudnorm",
        "pydantic>=2.0,<3.0", "PyYAML>=6.0", "numba>=0.60.0",  # numba 0.60.0+ for NumPy 2.0
    ]
    run_pip(["install"] + core_deps, "Install core dependencies")

    # Step 4: Install Whisper packages (git-based to avoid PyPI conflicts)
    print("\n" + "=" * 60)
    print("  Step 4/6: Installing Whisper packages")
    print("=" * 60)
    run_pip(
        ["install", "git+https://github.com/openai/whisper@main"],
        "Install openai-whisper from GitHub"
    )
    run_pip(
        ["install", "git+https://github.com/meizhong986/stable-ts-fix-setup.git@main"],
        "Install stable-ts from GitHub"
    )
    run_pip(
        ["install", "git+https://github.com/kkroening/ffmpeg-python.git"],
        "Install ffmpeg-python from GitHub"
    )
    run_pip(["install", "faster-whisper>=1.1.0"], "Install faster-whisper")

    # Step 5: Install optional packages
    print("\n" + "=" * 60)
    print("  Step 5/6: Installing optional packages")
    print("=" * 60)

    # HuggingFace / Transformers
    run_pip(
        ["install", "huggingface-hub>=0.25.0", "transformers>=4.40.0", "accelerate>=0.26.0"],
        "Install HuggingFace packages"
    )

    # Translation (pysubtrans requires Python 3.10+)
    run_pip(
        ["install", "pysubtrans>=1.5.0", "openai>=1.35.0", "google-genai>=1.39.0"],
        "Install translation packages"
    )

    # Local LLM translation (llama-cpp-python) - OPTIONAL
    # Uses JamePeng's fork with active maintenance and multi-platform support
    # Only installed if --local-llm or --local-llm-build is specified
    # Note: Apple Silicon builds from source with Metal (~10min, fast)
    #       Intel Mac uses CPU build (no Metal/GPU acceleration)
    #       Windows/Linux uses prebuilt wheels (requires CUDA 12.4+)
    if args.local_llm or args.local_llm_build:
        print(f"\n    Installing llama-cpp-python for local LLM translation...")

        import platform as platform_module
        is_apple_silicon = (sys.platform == "darwin" and platform_module.machine() == "arm64")
        is_intel_mac = (sys.platform == "darwin" and platform_module.machine() != "arm64")

        if is_apple_silicon:
            # Apple Silicon: build from source with Metal (fast ~10min)
            print("    Apple Silicon detected - building from source with Metal support.")
            git_url, backend, cmake_args = get_llama_cpp_source_info()
            print(f"    Backend: {backend}")
            if cmake_args:
                print(f"    Setting CMAKE_ARGS={cmake_args}")
                os.environ["CMAKE_ARGS"] = cmake_args
            run_pip(
                ["install", git_url],
                f"Install llama-cpp-python ({backend})",
                allow_fail=True
            )
        elif is_intel_mac:
            # Intel Mac: CPU-only build (no Metal support)
            if args.local_llm_build:
                print("    Intel Mac detected - building CPU-only version.")
                git_url, backend, cmake_args = get_llama_cpp_source_info()
                print(f"    Backend: {backend}")
                # cmake_args will be None for Intel Mac (CPU build)
                run_pip(
                    ["install", git_url],
                    f"Install llama-cpp-python ({backend})",
                    allow_fail=True
                )
            else:
                print("    Intel Mac detected - no prebuilt wheels available.")
                print("    To build CPU-only version, use --local-llm-build.")
                print("    Skipping local LLM installation.")
        else:
            # Windows/Linux: try prebuilt wheel first (requires CUDA 12.4+)
            wheel_url, wheel_backend = get_llama_cpp_prebuilt_wheel()

            if wheel_url:
                # Prebuilt wheel available - use it (fast install)
                print(f"    Backend: {wheel_backend}")
                run_pip(
                    ["install", wheel_url],
                    f"Install llama-cpp-python ({wheel_backend})",
                    allow_fail=True
                )
                # Install [server] extras
                run_pip(
                    ["install", "llama-cpp-python[server]"],
                    "Install llama-cpp-python server extras",
                    allow_fail=True
                )
            elif args.local_llm_build:
                # No prebuilt wheel, but user opted for source build
                git_url, backend, cmake_args = get_llama_cpp_source_info()
                print(f"    Backend: {backend}")
                if cmake_args:
                    print(f"    Setting CMAKE_ARGS={cmake_args}")
                    os.environ["CMAKE_ARGS"] = cmake_args
                run_pip(
                    ["install", git_url],
                    f"Install llama-cpp-python ({backend})",
                    allow_fail=True
                )
            else:
                # --local-llm specified but no prebuilt wheel available
                print("    No prebuilt wheel available for your platform.")
                print("    Prebuilt wheels require CUDA 12.4+. To build from source, use --local-llm-build.")
                print("    Skipping local LLM installation.")
    else:
        print("\n    Skipping local LLM (use --local-llm or --local-llm-build to install)")

    # VAD
    run_pip(["install", "silero-vad>=6.0", "auditok"], "Install VAD packages")

    if not args.minimal:
        run_pip(["install", "ten-vad"], "Install TEN VAD", allow_fail=True)
        run_pip(["install", "scikit-learn>=1.5.0"], "Install scikit-learn")  # 1.5.0+ for NumPy 2.0

    # Speech Enhancement
    if not args.no_speech_enhancement and not args.minimal:
        print("\n    Installing speech enhancement packages...")
        print("    (These can be tricky - failures here are non-fatal)")

        run_pip(
            ["install", "addict", "simplejson", "sortedcontainers", "packaging"],
            "Install ModelScope dependencies",
            allow_fail=True
        )
        run_pip(
            ["install", "datasets>=2.14.0,<4.0"],
            "Install datasets",
            allow_fail=True
        )
        run_pip(
            ["install", "modelscope>=1.20"],
            "Install ModelScope",
            allow_fail=True
        )
        run_pip(
            ["install", "git+https://github.com/meizhong986/ClearerVoice-Studio.git#subdirectory=clearvoice"],
            "Install ClearVoice (forked)",
            allow_fail=True
        )
        run_pip(
            ["install", "bs-roformer-infer"],
            "Install BS-RoFormer",
            allow_fail=True
        )
        run_pip(
            ["install", "onnxruntime>=1.16.0"],
            "Install ONNX Runtime",
            allow_fail=True
        )

    # GUI dependencies (platform-specific)
    run_pip(["install", "pywebview>=5.0.0"], "Install PyWebView", allow_fail=True)
    if sys.platform == "win32":
        run_pip(["install", "pythonnet>=3.0", "pywin32>=305"], "Install Windows GUI deps", allow_fail=True)

    # Step 6: Install WhisperJAV
    print("\n" + "=" * 60)
    print("  Step 6/6: Installing WhisperJAV")
    print("=" * 60)
    if args.dev:
        run_pip(["install", "--no-deps", "-e", "."], "Install WhisperJAV (editable)")
    else:
        run_pip(["install", "--no-deps", "."], "Install WhisperJAV")

    # Verify installation
    print("\n" + "=" * 60)
    print("  Verifying Installation")
    print("=" * 60)
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import whisperjav; print(f'WhisperJAV {whisperjav.__version__}')"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip()
        print(f"    [OK] {version} installed successfully!")
    except subprocess.CalledProcessError:
        print("    [WARN] Could not verify installation")

    # Summary
    print("\n" + "=" * 60)
    print("  Installation Complete!")
    print("=" * 60)

    # Build summary message based on what was installed
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
    python install.py --local-llm          (fast - prebuilt wheel only)
    python install.py --local-llm-build    (slow - builds from source if needed)
"""

    summary += """
  If you encounter issues with speech enhancement, re-run with:
    python install.py --no-speech-enhancement
"""
    print(summary)


if __name__ == "__main__":
    main()
