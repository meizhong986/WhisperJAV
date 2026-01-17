#!/usr/bin/env python3
"""
Shared utilities for building and installing llama-cpp-python.

This module provides helper functions for:
- Detecting GPU architecture and CUDA version
- Finding and downloading prebuilt wheels
- Building llama-cpp-python from source

Used by:
- install.py
- local_backend.py
- post_install.py (installer)
"""

import json
import os
import sys
import subprocess
import platform
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple, Dict


def get_cuda_architecture() -> Optional[str]:
    """
    Detect CUDA compute capability for the installed GPU.

    Returns:
        str or None: CUDA architecture (e.g., "89" for RTX 40 series) or None if not detected
    """
    # Try nvidia-smi first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Format is like "8.9" -> we want "89"
            cap = result.stdout.strip().split('\n')[0].strip()
            if '.' in cap:
                major, minor = cap.split('.')
                return f"{major}{minor}"
    except Exception:
        pass

    # Fallback: try to infer from GPU name
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip().lower()
            # Map GPU series to compute capability
            if any(x in gpu_name for x in ["rtx 40", "rtx 50", "ada", "l40"]):
                return "89"  # Ada Lovelace
            elif any(x in gpu_name for x in ["rtx 30", "a100", "a10", "a30", "a40"]):
                return "86"  # Ampere
            elif any(x in gpu_name for x in ["rtx 20", "gtx 16", "t4", "quadro rtx"]):
                return "75"  # Turing
            elif any(x in gpu_name for x in ["v100", "titan v"]):
                return "70"  # Volta
            elif any(x in gpu_name for x in ["gtx 10", "p100", "p40", "p4"]):
                return "61"  # Pascal
    except Exception:
        pass

    return None


def detect_cuda_version() -> Optional[str]:
    """
    Detect CUDA version from nvidia-smi or torch.

    Returns:
        CUDA version string like "cu128" or None if not available
    """
    # Try torch first (most reliable)
    try:
        import torch
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda
            if cuda_ver:
                # Convert "12.8" to "cu128"
                parts = cuda_ver.split(".")
                if len(parts) >= 2:
                    return f"cu{parts[0]}{parts[1]}"
    except Exception:
        pass

    # Fall back to nvidia-smi driver version
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            driver_ver = result.stdout.strip().split("\n")[0]
            # Map driver version to CUDA version (simplified: cu128 or cu118)
            major = int(driver_ver.split(".")[0])
            if major >= 570:
                return "cu128"
            elif major >= 450:
                return "cu118"
    except Exception:
        pass

    return None


def get_prebuilt_wheel_url(cuda_version: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to find a prebuilt wheel URL from JamePeng's GitHub releases.

    Auto-detects platform, Python version, and CUDA version.
    Queries GitHub API to find matching wheel dynamically.

    Args:
        cuda_version: CUDA version like "cu128" or None to auto-detect

    Returns:
        tuple: (wheel_url, backend_desc) or (None, None) if no suitable wheel found
    """
    # Auto-detect CUDA if not provided
    if cuda_version is None and sys.platform in ("win32", "linux"):
        cuda_version = detect_cuda_version()

    # Determine platform identifiers
    if sys.platform == "win32":
        os_tag = "win"
        wheel_platform = "win_amd64"
    elif sys.platform == "linux":
        os_tag = "linux"
        wheel_platform = "linux_x86_64"
    elif sys.platform == "darwin":
        os_tag = "metal"
        if platform.machine() == "arm64":
            wheel_platform = "arm64"
        else:
            wheel_platform = "x86_64"
    else:
        return None, None

    # Python version
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"

    # Build search criteria for CUDA versions
    target_cudas = []
    if cuda_version and sys.platform in ("win32", "linux"):
        # Simplified: only cu128 or cu118
        if cuda_version >= "cu128":
            target_cudas = ["cu128", "cu118"]
        elif cuda_version >= "cu118":
            target_cudas = ["cu118"]

    # Query GitHub API for releases
    try:
        api_url = "https://api.github.com/repos/JamePeng/llama-cpp-python/releases?per_page=50"
        req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=15) as response:
            releases = json.loads(response.read().decode())
    except Exception:
        return None, None

    # Search for matching wheel
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
                    return wheel_url, "Metal (prebuilt wheel)"
    else:
        # Windows/Linux: look for CUDA releases
        for cuda_tag in target_cudas:
            for release in releases:
                tag = release.get("tag_name", "")
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
                        return wheel_url, f"CUDA ({cuda_tag} prebuilt wheel)"

    return None, None


def download_wheel(url: str, filename: Optional[str] = None) -> Optional[Path]:
    """
    Download wheel from URL to temp directory.

    Args:
        url: URL to download from
        filename: Optional filename, extracted from URL if not provided

    Returns:
        Path to downloaded wheel, or None if failed
    """
    if filename is None:
        filename = url.split("/")[-1]

    try:
        temp_dir = Path(tempfile.gettempdir()) / "whisperjav_wheels"
        temp_dir.mkdir(exist_ok=True)
        dest_path = temp_dir / filename

        # Use cached wheel if exists
        if dest_path.exists():
            return dest_path

        urllib.request.urlretrieve(url, dest_path)
        return dest_path

    except Exception:
        return None


def install_wheel(wheel_path: Path, verbose: bool = True) -> bool:
    """
    Install wheel using pip.

    Args:
        wheel_path: Path to the wheel file
        verbose: If True, print progress messages

    Returns:
        True if installation succeeded, False otherwise
    """
    try:
        if verbose:
            print(f"Installing llama-cpp-python from wheel...")

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", str(wheel_path), "--no-deps"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            # Also install server extras
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python[server]"],
                capture_output=True,
                timeout=60
            )
            if verbose:
                print("  Successfully installed!")
            return True
        else:
            if verbose and result.stderr:
                print(f"  Installation failed: {result.stderr[:200]}")
            return False

    except Exception as e:
        if verbose:
            print(f"  Installation failed: {e}")
        return False


def install_from_prebuilt(verbose: bool = True) -> bool:
    """
    Try to install llama-cpp-python from a prebuilt wheel.

    Args:
        verbose: If True, print progress messages

    Returns:
        True if installation succeeded, False otherwise
    """
    if verbose:
        print("Searching for prebuilt wheel...")

    wheel_url, backend_desc = get_prebuilt_wheel_url()

    if not wheel_url:
        if verbose:
            print("  No prebuilt wheel found for this platform/CUDA version")
        return False

    if verbose:
        print(f"  Found: {backend_desc}")
        print(f"  Downloading...")

    wheel_path = download_wheel(wheel_url)
    if not wheel_path:
        if verbose:
            print("  Download failed")
        return False

    return install_wheel(wheel_path, verbose=verbose)


def get_build_parallel_level() -> int:
    """
    Get optimal CMAKE_BUILD_PARALLEL_LEVEL based on CPU cores.

    Returns:
        int: Number of parallel build jobs (min 2, max 16, default ~75% of cores)
    """
    try:
        import multiprocessing
        cores = multiprocessing.cpu_count()
        # Use ~75% of cores, min 2, max 16
        parallel = max(2, min(16, int(cores * 0.75)))
        return parallel
    except Exception:
        return 4  # Safe default


def get_llama_cpp_source_info() -> Tuple[str, str, Optional[str], Dict[str, str]]:
    """
    Get llama-cpp-python source build info based on platform.

    Auto-detects GPU/Metal/CPU and returns appropriate build configuration.

    Optimizations:
    - CMAKE_BUILD_PARALLEL_LEVEL: Uses ~75% of CPU cores for faster builds
    - CMAKE_CUDA_ARCHITECTURES: Targets specific GPU for faster builds and smaller binary

    Returns:
        tuple: (git_url, backend_desc, cmake_args, env_vars)
            - git_url: Git URL for pip install
            - backend_desc: Human-readable description
            - cmake_args: CMAKE_ARGS value or None
            - env_vars: Dict of additional environment variables to set
    """
    git_url = "llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git"
    cmake_args = None
    env_vars = {}

    # Set parallel build level for faster compilation
    parallel_level = get_build_parallel_level()
    env_vars["CMAKE_BUILD_PARALLEL_LEVEL"] = str(parallel_level)

    if sys.platform == "darwin":
        chip = platform.processor() or platform.machine()
        if "arm" in chip.lower() or "apple" in chip.lower():
            backend = f"Metal (Apple Silicon) - building from source ({parallel_level} jobs)"
            cmake_args = "-DGGML_METAL=on"
        else:
            backend = f"CPU (Intel Mac) - building from source ({parallel_level} jobs)"
    elif sys.platform == "win32":
        cuda_arch = get_cuda_architecture()
        if cuda_arch:
            cmake_args = f"-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"
            backend = f"CUDA (sm_{cuda_arch}) - building from source ({parallel_level} jobs)"
        else:
            backend = f"CPU - building from source ({parallel_level} jobs)"
    else:
        # Linux
        cuda_arch = get_cuda_architecture()
        if cuda_arch:
            cmake_args = f"-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"
            backend = f"CUDA (sm_{cuda_arch}) - building from source ({parallel_level} jobs)"
        else:
            backend = f"CPU - building from source ({parallel_level} jobs)"

    return git_url, backend, cmake_args, env_vars


def build_from_source(verbose: bool = True) -> bool:
    """
    Build llama-cpp-python from source with GPU support.

    Args:
        verbose: If True, print progress messages

    Returns:
        True if build succeeded, False otherwise
    """
    git_url, backend, cmake_args, env_vars = get_llama_cpp_source_info()

    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        if verbose:
            print(f"  Setting {key}={value}")

    if cmake_args:
        os.environ["CMAKE_ARGS"] = cmake_args
        if verbose:
            print(f"  Setting CMAKE_ARGS={cmake_args}")

    if verbose:
        parallel_level = env_vars.get("CMAKE_BUILD_PARALLEL_LEVEL", "?")
        print(f"\n  Building from source ({parallel_level} parallel jobs)...")
        print("  This may take ~10 minutes...\n")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", git_url],
            capture_output=not verbose,
            text=True
        )

        if result.returncode == 0:
            return True
        else:
            if not verbose and result.stderr:
                print(result.stderr)
            return False

    except Exception as e:
        if verbose:
            print(f"  Build failed: {e}")
        return False
