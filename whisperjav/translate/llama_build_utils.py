#!/usr/bin/env python3
"""
Shared utilities for building llama-cpp-python.

This module provides helper functions for detecting GPU architecture,
setting up build environments, and building llama-cpp-python from source.

Used by:
- install.py
- local_backend.py
- post_install.py (installer)
"""

import os
import sys
import subprocess
import platform
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
