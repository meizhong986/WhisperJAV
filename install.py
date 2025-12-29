#!/usr/bin/env python3
"""
WhisperJAV Source Installation Script
=====================================

This script handles staged installation of WhisperJAV from source,
working around pip dependency resolution conflicts (Issue #90).

Prerequisites:
    - Python 3.9-3.12
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
    --cuda121               Install PyTorch for CUDA 12.1 (default)
    --cuda124               Install PyTorch for CUDA 12.4
    --no-speech-enhancement Skip speech enhancement packages
    --minimal               Minimal install (transcription only)
    --dev                   Install in development/editable mode
    --help                  Show this help message

Examples:
    python install.py                    # Standard install with CUDA 12.1
    python install.py --cpu-only         # CPU-only install
    python install.py --minimal --dev    # Minimal dev install
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
    if major < 3 or (major == 3 and minor < 9):
        print(f"ERROR: Python 3.9+ required. Found: {major}.{minor}")
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
                        help="Install PyTorch for CUDA 12.1 (default)")
    parser.add_argument("--cuda124", action="store_true",
                        help="Install PyTorch for CUDA 12.4")
    parser.add_argument("--no-speech-enhancement", action="store_true",
                        help="Skip speech enhancement packages")
    parser.add_argument("--minimal", action="store_true",
                        help="Minimal install (transcription only)")
    parser.add_argument("--dev", action="store_true",
                        help="Install in development/editable mode")

    args = parser.parse_args()

    # Determine CUDA version
    if args.cpu_only:
        cuda_version = "cpu"
    elif args.cuda118:
        cuda_version = "cuda118"
    elif args.cuda124:
        cuda_version = "cuda124"
    else:
        cuda_version = "cuda121"  # default

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
        "numpy>=2.0", "scipy>=1.10.1", "librosa>=0.11.0",
        "soundfile", "pydub", "tqdm", "colorama", "requests", "regex",
        "pysrt", "srt", "aiofiles", "jsonschema", "pyloudnorm",
        "pydantic>=2.0,<3.0", "PyYAML>=6.0", "numba",
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

    # Translation
    run_pip(
        ["install", "PySubtrans>=0.7.0", "openai>=1.35.0", "google-genai>=1.39.0"],
        "Install translation packages"
    )

    # VAD
    run_pip(["install", "silero-vad>=6.0", "auditok"], "Install VAD packages")

    if not args.minimal:
        run_pip(["install", "ten-vad"], "Install TEN VAD", allow_fail=True)
        run_pip(["install", "scikit-learn>=1.3.0"], "Install scikit-learn")

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
    print("""
  To run WhisperJAV:
    whisperjav video.mp4 --mode balanced

  To run with GUI:
    whisperjav-gui

  For help:
    whisperjav --help

  If you encounter issues with speech enhancement, re-run with:
    python install.py --no-speech-enhancement
""")


if __name__ == "__main__":
    main()
