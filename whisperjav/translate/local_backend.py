"""
Local LLM Server Manager for translation.

This module manages a local llama-cpp-python OpenAI-compatible server,
allowing PySubtrans to use local models through the same code path as cloud APIs.

Features:
- Automatic VRAM detection and model selection
- Pre-quantized GGUF models (no license acceptance required)
- OpenAI-compatible API server for seamless PySubtrans integration
- Server lifecycle management (start/stop)

Available Models:
- llama-8b:  Llama 3.1 8B (Q4) - 6GB+ VRAM (default)
- gemma-9b:  Gemma 2 9B (Q4_K_M) - 8GB+ VRAM (alternative)
- llama-3b:  Llama 3.2 3B (Q4_K_M) - 3GB+ VRAM (basic, low VRAM only)
- auto:      Auto-select based on available VRAM

Usage:
    # CLI
    whisperjav-translate -i input.srt --provider local --model auto
    whisperjav-translate -i input.srt --provider local --model llama-3b

    # Programmatic
    from whisperjav.translate.local_backend import start_local_server, stop_local_server
    api_base, port = start_local_server(model="auto")
    # ... use api_base with OpenAI-compatible client ...
    stop_local_server()
"""

import atexit
import gc
import logging
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Import shared build utilities
from .llama_build_utils import (
    build_from_source as _build_from_source,
    detect_cuda_version,
    get_prebuilt_wheel_url,
    download_wheel,
    install_wheel,
)

logger = logging.getLogger(__name__)

# HuggingFace wheel repository for lazy download
WHEEL_REPO_ID = "mei986/whisperjav-wheels"
WHEEL_VERSION = "0.3.21"  # llama-cpp-python version in our wheel repo

# Global server process reference
_server_process: Optional[subprocess.Popen] = None
_server_port: Optional[int] = None

# Note: atexit handler registered after stop_local_server is defined (see end of module)

# Model registry - uncensored GGUF models for translation
# These models have content filters removed for unrestricted translation
# VRAM estimates include model + 8K context KV cache
MODEL_REGISTRY = {
    'llama-3b': {
        'repo': 'mradermacher/Llama-3.2-3B-Instruct-uncensored-GGUF',
        'file': 'Llama-3.2-3B-Instruct-uncensored.Q4_K_M.gguf',
        'vram': 3.0,
        'desc': 'Llama 3.2 3B - Basic quality, for low VRAM systems only'
    },
    'llama-8b': {
        'repo': 'Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF',
        'file': 'Llama-3.1-8B-Lexi-Uncensored_V2_Q4.gguf',
        'vram': 6.0,
        'desc': 'Llama 3.1 8B - Default, requires 6GB+ VRAM'
    },
    'gemma-9b': {
        'repo': 'bartowski/gemma-2-9b-it-abliterated-GGUF',
        'file': 'gemma-2-9b-it-abliterated-Q4_K_M.gguf',
        'vram': 8.0,
        'desc': 'Gemma 2 9B - Alternative model, requires 8GB+ VRAM'
    },
}


# =============================================================================
# Lazy Download: Install llama-cpp-python on first use
# =============================================================================

# Cache for llama-cpp installation status to avoid repeated import attempts
_llama_cpp_status_cache: Optional[Tuple[bool, Optional[str]]] = None


def _check_llama_cpp_status() -> Tuple[bool, Optional[str]]:
    """
    Check if llama-cpp-python is installed and functional.

    Returns:
        Tuple of (is_functional, error_message)
        - (True, None): llama-cpp-python is installed and working
        - (False, None): llama-cpp-python is not installed (ImportError)
        - (False, error_msg): llama-cpp-python is installed but broken (DLL issues)
    """
    global _llama_cpp_status_cache

    # Return cached result if available
    if _llama_cpp_status_cache is not None:
        return _llama_cpp_status_cache

    try:
        import llama_cpp  # noqa: F401
        _llama_cpp_status_cache = (True, None)
        return (True, None)
    except ImportError:
        # Not installed at all
        _llama_cpp_status_cache = (False, None)
        return (False, None)
    except (RuntimeError, OSError) as e:
        # Installed but DLL loading failed
        # RuntimeError: "Failed to load shared library 'ggml.dll'"
        # OSError: Alternative DLL loading error on some systems
        error_msg = str(e)
        logger.warning(f"llama-cpp-python installed but not functional: {error_msg}")
        _llama_cpp_status_cache = (False, error_msg)
        return (False, error_msg)
    except Exception as e:
        # Unexpected error during import
        error_msg = f"Unexpected error: {type(e).__name__}: {e}"
        logger.warning(f"llama-cpp-python check failed: {error_msg}")
        _llama_cpp_status_cache = (False, error_msg)
        return (False, error_msg)


def _clear_llama_cpp_status_cache():
    """Clear the status cache to force re-check after reinstallation."""
    global _llama_cpp_status_cache
    _llama_cpp_status_cache = None


def _is_llama_cpp_installed() -> bool:
    """Check if llama-cpp-python is installed and functional."""
    is_functional, _ = _check_llama_cpp_status()
    return is_functional


def _diagnose_dll_failure(error_msg: str) -> str:
    """
    Diagnose DLL loading failure and provide actionable guidance.

    Args:
        error_msg: The error message from the failed import

    Returns:
        Diagnostic message with suggested fixes
    """
    diagnosis_parts = []

    # Check for common error patterns
    error_lower = error_msg.lower()

    if "ggml.dll" in error_lower or "llama.dll" in error_lower:
        diagnosis_parts.append("DLL Loading Failure Detected")
        diagnosis_parts.append("-" * 40)

        # Check CUDA version
        cuda_version = detect_cuda_version()
        if cuda_version:
            diagnosis_parts.append(f"Current CUDA version (from PyTorch): {cuda_version}")
        else:
            diagnosis_parts.append("CUDA: Not detected (CPU-only mode)")

        # Check if CUDA toolkit is installed (not just driver)
        cuda_toolkit_installed = _check_cuda_toolkit_installed()
        if cuda_toolkit_installed:
            diagnosis_parts.append(f"CUDA Toolkit: Installed ({cuda_toolkit_installed})")
        else:
            diagnosis_parts.append("CUDA Toolkit: NOT FOUND (only driver installed?)")
            diagnosis_parts.append("")
            diagnosis_parts.append("LIKELY CAUSE: CUDA toolkit is not installed.")
            diagnosis_parts.append("The llama-cpp-python DLLs need CUDA runtime libraries")
            diagnosis_parts.append("(cublas64_*.dll, cudart64_*.dll) which come with")
            diagnosis_parts.append("the CUDA Toolkit, not just the NVIDIA driver.")

        # Check VC++ runtime
        vc_installed = _check_vc_runtime_installed()
        if vc_installed:
            diagnosis_parts.append("Visual C++ Runtime: Installed")
        else:
            diagnosis_parts.append("Visual C++ Runtime: NOT FOUND")
            diagnosis_parts.append("")
            diagnosis_parts.append("POSSIBLE CAUSE: Visual C++ 2015-2022 Redistributable missing.")
            diagnosis_parts.append("Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")

        diagnosis_parts.append("")
        diagnosis_parts.append("RECOMMENDED FIXES:")
        diagnosis_parts.append("1. Reinstall llama-cpp-python with correct CUDA version:")
        diagnosis_parts.append("   pip uninstall llama-cpp-python")
        diagnosis_parts.append("   pip install llama-cpp-python  # Will auto-download correct wheel")
        diagnosis_parts.append("")
        diagnosis_parts.append("2. Or use CPU-only mode (slower but always works):")
        diagnosis_parts.append("   whisperjav-translate --translate-gpu-layers 0")
        diagnosis_parts.append("")
        diagnosis_parts.append("3. Or use cloud translation providers:")
        diagnosis_parts.append("   whisperjav-translate --provider deepseek")

    elif "one of its dependencies" in error_lower:
        diagnosis_parts.append("Missing Dependency Detected")
        diagnosis_parts.append("A required DLL dependency could not be found.")
        diagnosis_parts.append("")
        diagnosis_parts.append("Common causes:")
        diagnosis_parts.append("- CUDA toolkit not installed (only driver)")
        diagnosis_parts.append("- CUDA version mismatch")
        diagnosis_parts.append("- Missing Visual C++ Runtime")

    else:
        diagnosis_parts.append("Unknown DLL Error")
        diagnosis_parts.append(f"Error: {error_msg}")

    return "\n".join(diagnosis_parts)


def _check_cuda_toolkit_installed() -> Optional[str]:
    """
    Check if CUDA toolkit is installed (not just driver).

    Returns:
        CUDA toolkit version string if found, None otherwise
    """
    try:
        # Try nvcc (CUDA compiler - only present with toolkit)
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Parse version from output like "Cuda compilation tools, release 12.4, V12.4.131"
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    import re
                    match = re.search(r'release\s+(\d+\.\d+)', line, re.IGNORECASE)
                    if match:
                        return match.group(1)
            return "installed (version unknown)"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check common CUDA toolkit installation paths on Windows
    if sys.platform == "win32":
        import os
        cuda_paths = [
            os.environ.get("CUDA_PATH", ""),
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        ]
        for base_path in cuda_paths:
            if base_path and os.path.exists(base_path):
                # Check for version directories
                if os.path.isdir(base_path):
                    for item in os.listdir(base_path):
                        if item.startswith("v") and os.path.isdir(os.path.join(base_path, item)):
                            return item[1:]  # Remove 'v' prefix
                    # CUDA_PATH points directly to versioned dir
                    nvcc_path = os.path.join(base_path, "bin", "nvcc.exe")
                    if os.path.exists(nvcc_path):
                        return "installed"

    return None


def _check_vc_runtime_installed() -> bool:
    """Check if Visual C++ 2015-2022 Redistributable is installed."""
    if sys.platform != "win32":
        return True  # Not needed on non-Windows

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
                return True
            except FileNotFoundError:
                continue
        return False
    except Exception:
        return True  # Assume OK if we can't check


def _uninstall_llama_cpp() -> bool:
    """
    Uninstall llama-cpp-python to allow clean reinstallation.

    Returns:
        True if uninstallation succeeded or package wasn't installed
    """
    print("Uninstalling broken llama-cpp-python installation...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Clear the status cache to force re-check
        _clear_llama_cpp_status_cache()

        if result.returncode == 0:
            print("  Uninstalled successfully.")
            return True
        elif "not installed" in result.stdout.lower() or "not installed" in result.stderr.lower():
            print("  Package was not installed.")
            return True
        else:
            logger.warning(f"Uninstall may have failed: {result.stderr[:200]}")
            return True  # Proceed anyway
    except Exception as e:
        logger.warning(f"Error during uninstall: {e}")
        return True  # Proceed anyway


def _are_server_deps_installed() -> bool:
    """Check if server dependencies (uvicorn, fastapi, etc.) are installed."""
    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
        import pydantic_settings  # noqa: F401
        import sse_starlette  # noqa: F401
        import starlette_context  # noqa: F401
        return True
    except ImportError:
        return False


def _install_server_deps() -> bool:
    """
    Install whisperjav[local-llm] server dependencies.

    These are platform-agnostic deps (uvicorn, fastapi, etc.) that must be
    installed before llama-cpp-python.

    Returns:
        True if installation succeeded or deps already installed, False otherwise
    """
    if _are_server_deps_installed():
        logger.debug("Server dependencies already installed")
        return True

    print("Installing server dependencies (uvicorn, fastapi, etc.)...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "whisperjav[local-llm]"],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        if result.returncode == 0:
            print("  Server dependencies installed successfully!")
            return True
        else:
            logger.warning(f"Failed to install server deps: {result.stderr[:200]}")
            return False
    except Exception as e:
        logger.warning(f"Failed to install server deps: {e}")
        return False


def _wait_with_cancel_option(seconds: int = 10) -> bool:
    """
    Wait for specified seconds, allowing user to cancel with Ctrl+C.

    Args:
        seconds: Number of seconds to wait (default 10)

    Returns:
        True if user waited (proceed with build), False if user cancelled
    """
    print(f"\nPress Ctrl+C within {seconds}s to cancel, or wait to continue automatically...")

    try:
        for i in range(seconds, 0, -1):
            print(f"  Starting in {i}s...", end='\r')
            time.sleep(1)
        print("  Starting build...       ")  # Extra spaces to clear countdown
        return True
    except KeyboardInterrupt:
        print("\n  Build cancelled by user.")
        return False


def _get_wheel_filenames(cuda_version: Optional[str] = None) -> Tuple[list, str]:
    """
    Build possible wheel filenames based on platform, Python version, and CUDA version.

    Returns:
        Tuple of (list_of_wheel_filenames, backend_subfolder)
        Multiple filenames are returned for Linux to handle different platform tags.
    """
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"

    # Determine platform tags (may have multiple variants for Linux)
    if sys.platform == "win32":
        plat_tags = ["win_amd64"]
    elif sys.platform == "linux":
        # Try multiple platform tags - wheels may use different conventions
        # Order: most specific first
        plat_tags = [
            "manylinux_2_17_x86_64.manylinux2014_x86_64",  # Standard manylinux
            "linux_x86_64",  # Simplified tag (used by some builds)
        ]
    elif sys.platform == "darwin":
        if platform.machine() == "arm64":
            plat_tags = ["macosx_11_0_arm64"]
        else:
            plat_tags = ["macosx_10_15_x86_64"]
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    # Determine backend subfolder
    if sys.platform == "darwin" and platform.machine() == "arm64":
        backend = "metal"
    elif cuda_version:
        backend = cuda_version
    else:
        backend = "cpu"

    # Build wheel filenames for all platform tag variants
    # Format: llama_cpp_python-{version}-{pyver}-{pyver}-{platform}.whl
    wheel_names = [
        f"llama_cpp_python-{WHEEL_VERSION}-{py_ver}-{py_ver}-{plat_tag}.whl"
        for plat_tag in plat_tags
    ]

    return wheel_names, backend


def _get_wheel_filename(cuda_version: Optional[str] = None) -> Tuple[str, str]:
    """
    Build wheel filename based on platform, Python version, and CUDA version.
    Returns the first (preferred) filename variant.

    Returns:
        Tuple of (wheel_filename, backend_subfolder)
    """
    wheel_names, backend = _get_wheel_filenames(cuda_version)
    return wheel_names[0], backend


def _download_wheel_from_huggingface(cuda_version: Optional[str] = None) -> Optional[Path]:
    """
    Download llama-cpp-python wheel from HuggingFace.

    Tries multiple wheel filename variants for Linux (manylinux vs linux_x86_64).

    Returns:
        Path to downloaded wheel, or None if not found
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        wheel_names, backend = _get_wheel_filenames(cuda_version)

        print(f"Downloading llama-cpp-python from HuggingFace...")
        print(f"  Backend: {backend}")

        # Try each wheel filename variant
        for wheel_name in wheel_names:
            wheel_path = f"llama-cpp-python/{backend}/{wheel_name}"
            logger.info(f"Trying HuggingFace: {WHEEL_REPO_ID}/{wheel_path}")

            try:
                local_path = hf_hub_download(
                    repo_id=WHEEL_REPO_ID,
                    filename=wheel_path,
                    repo_type="dataset"
                )
                print(f"  Found: {wheel_name}")
                logger.info(f"Downloaded wheel to: {local_path}")
                return Path(local_path)
            except EntryNotFoundError:
                logger.debug(f"Not found: {wheel_path}")
                continue

        # None of the variants found
        logger.warning(f"Wheel not found on HuggingFace for backend {backend}")
        print(f"  Not found on HuggingFace (tried {len(wheel_names)} variants)")
        return None

    except Exception as e:
        logger.warning(f"Failed to download from HuggingFace: {e}")
        return None



def ensure_llama_cpp_installed() -> bool:
    """
    Ensure llama-cpp-python is installed, downloading if necessary.

    This implements lazy download: on first use of `--provider local`,
    the wheel is automatically downloaded and installed.

    Installation flow:
    1. Check if already installed and functional
    2. If broken (DLL issues), diagnose and offer reinstall
    3. Install server deps first (whisperjav[local-llm]) - platform agnostic
    4. Detect CUDA version
    5. Try prebuilt wheel (HuggingFace, then GitHub)
    6. Fall back to source build (with 10s user cancel window)

    Returns:
        True if llama-cpp-python is available, False otherwise
    """
    # Step 0: Check current status (installed, not installed, or broken)
    is_functional, error_msg = _check_llama_cpp_status()

    if is_functional:
        logger.debug("llama-cpp-python already installed and functional")
        return True

    # Handle broken installation (DLL loading failed)
    if error_msg is not None:
        print("\n" + "!" * 60)
        print("  LLAMA-CPP-PYTHON INSTALLATION BROKEN")
        print("!" * 60)
        print("\nllama-cpp-python is installed but cannot load its native libraries.")
        print("")

        # Show diagnosis
        diagnosis = _diagnose_dll_failure(error_msg)
        print(diagnosis)
        print("")

        # Offer to reinstall
        print("Would you like to uninstall and reinstall llama-cpp-python?")
        print("This may fix the issue by downloading the correct version.")
        print("")

        try:
            response = input("Reinstall? (Y/n): ").strip().lower()
            if response in ('', 'y', 'yes'):
                _uninstall_llama_cpp()
                _clear_llama_cpp_status_cache()
                # Continue to fresh installation below
            else:
                print("\nSkipping reinstall. You can:")
                print("  1. Fix the issue manually (see diagnosis above)")
                print("  2. Use cloud translation: --provider deepseek")
                return False
        except (EOFError, KeyboardInterrupt):
            # Non-interactive mode - try reinstall automatically
            print("\nNon-interactive mode: attempting automatic reinstall...")
            _uninstall_llama_cpp()
            _clear_llama_cpp_status_cache()

    # Fresh installation
    print("\n" + "=" * 60)
    print("  FIRST-TIME SETUP: Installing llama-cpp-python")
    print("=" * 60)
    print("\nllama-cpp-python is required for local LLM translation.")
    print("This is a one-time download (~700MB).\n")

    # Step 1: Install server dependencies first (uvicorn, fastapi, etc.)
    # These are platform-agnostic and declared in setup.py extras_require['local']
    if not _install_server_deps():
        print("WARNING: Could not install server dependencies.")
        print("         Server may not start correctly.\n")

    # Step 2: Detect CUDA version
    cuda_version = detect_cuda_version()
    if cuda_version:
        print(f"Detected CUDA: {cuda_version}")

        # Validate CUDA toolkit availability for CUDA wheels
        cuda_toolkit = _check_cuda_toolkit_installed()
        if not cuda_toolkit:
            print("")
            print("WARNING: CUDA toolkit not detected (only driver installed).")
            print("         GPU wheels require CUDA toolkit runtime libraries.")
            print("         Will attempt installation anyway - may fall back to CPU.")
            print("")
    else:
        print("CUDA not detected (will use CPU or Metal)")

    # Step 3: Try HuggingFace first
    wheel_path = _download_wheel_from_huggingface(cuda_version)

    # Fall back to JamePeng GitHub (using shared utility)
    if not wheel_path:
        print("\nHuggingFace wheel not found, trying JamePeng GitHub...")
        wheel_url, backend_desc = get_prebuilt_wheel_url(
            cuda_version, verbose=True, version=WHEEL_VERSION
        )
        if wheel_url:
            print(f"  Backend: {backend_desc}")
            print(f"  URL: {wheel_url}")
            wheel_path = download_wheel(wheel_url)

    # Install if we got a wheel
    if wheel_path and install_wheel(wheel_path):
        # Clear cache and verify installation
        _clear_llama_cpp_status_cache()
        is_functional, error_msg = _check_llama_cpp_status()

        if is_functional:
            print("\n✓ llama-cpp-python installed successfully!")
            print("  Future runs will start immediately.\n")
            return True
        elif error_msg:
            # Installation succeeded but DLL loading failed
            print("\n" + "!" * 60)
            print("  INSTALLATION COMPLETED BUT DLL LOADING FAILED")
            print("!" * 60)
            print("")
            print(_diagnose_dll_failure(error_msg))
            print("")
            print("The wheel was installed but native libraries cannot load.")
            print("This usually means CUDA version mismatch or missing dependencies.")
            print("")
            print("Options:")
            print("  1. Install CUDA toolkit matching your PyTorch version")
            print("  2. Use CPU-only mode: --translate-gpu-layers 0")
            print("  3. Use cloud translation: --provider deepseek")
            return False

    # Step 4: Fall back to building from source (with 10s cancel window)
    print("\nNo prebuilt wheel found. Building from source... (~10 min)")

    # Give user 10 seconds to cancel before starting long build
    if not _wait_with_cancel_option(seconds=10):
        print("\nYou can manually install later with:")
        print("  python install.py --local-llm-build")
        print("\nOr use cloud translation providers:")
        print("  whisperjav-translate -i file.srt --provider deepseek")
        return False

    if _build_from_source():
        _clear_llama_cpp_status_cache()
        is_functional, error_msg = _check_llama_cpp_status()

        if is_functional:
            print("\n✓ llama-cpp-python built and installed successfully!")
            print("  Future runs will start immediately.\n")
            return True
        elif error_msg:
            print("\n" + "!" * 60)
            print("  BUILD COMPLETED BUT DLL LOADING FAILED")
            print("!" * 60)
            print("")
            print(_diagnose_dll_failure(error_msg))
            return False

    # All methods failed
    print("\n" + "!" * 60)
    print("  INSTALLATION FAILED")
    print("!" * 60)
    print("\nCould not install llama-cpp-python automatically.")
    print("\nManual installation options:")
    print("  1. pip install whisperjav[local-llm]  # Install server deps")
    print("  2. python install.py --local-llm-build")
    print("\nAlternatively, use cloud translation providers:")
    print("  whisperjav-translate -i file.srt --provider deepseek")
    print("")

    return False


def get_available_vram_gb() -> float:
    """Detect available VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info()
            return free_mem / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def get_best_model_for_vram(vram_gb: float) -> str:
    """Select best model based on available VRAM.

    Priority: gemma-9b (best) > llama-8b (good) > llama-3b (basic)
    """
    if vram_gb >= 8:
        return 'gemma-9b'
    if vram_gb >= 6:
        return 'llama-8b'
    if vram_gb >= 3:
        return 'llama-3b'
    # Very low VRAM - still try llama-3b, may use CPU offload
    return 'llama-3b'


def ensure_model_downloaded(model_id: str) -> Path:
    """Download GGUF model if not already cached.

    Provides user feedback about download progress since models are large (2-6GB).
    """
    if model_id not in MODEL_REGISTRY:
        valid = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_id}. Valid: {valid}")

    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    info = MODEL_REGISTRY[model_id]
    repo_id = info['repo']
    filename = info['file']

    # Check if model is already cached
    cached_path = try_to_load_from_cache(repo_id=repo_id, filename=filename)
    if cached_path is not None:
        logger.info(f"Model already cached: {filename}")
        return Path(cached_path)

    # Model needs to be downloaded - inform user
    # Estimate sizes based on quantization (Q4_K_M ≈ 0.5 bytes/param)
    size_estimates = {
        'llama-3b': '~2.0 GB',
        'llama-8b': '~4.7 GB',
        'gemma-9b': '~5.5 GB',
    }
    size_str = size_estimates.get(model_id, 'several GB')

    print(f"\n{'='*60}")
    print(f"  DOWNLOADING MODEL: {model_id}")
    print(f"{'='*60}")
    print(f"  File: {filename}")
    print(f"  Size: {size_str}")
    print(f"  This is a one-time download.")
    print(f"{'='*60}\n")

    logger.info(f"Downloading model: {filename} ({size_str})")

    # Download with progress bar (tqdm will show in console)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename
    )

    print(f"\nModel downloaded successfully: {filename}\n")
    logger.info(f"Model downloaded: {path}")

    return Path(path)


def _find_free_port() -> int:
    """Find a free port for the server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, max_wait: int = 300) -> bool:
    """Wait for server to be ready.

    Args:
        port: Server port to check
        max_wait: Maximum wait time in seconds (default 5 minutes for large models)

    Returns:
        True if server is ready, False if server died or timeout
    """
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/v1/models"
    start_time = time.time()
    last_log_time = start_time

    while True:
        elapsed = time.time() - start_time

        # Check timeout
        if elapsed > max_wait:
            logger.warning(f"Server startup timed out after {max_wait}s")
            return False

        # Check if process died
        if _server_process is not None and _server_process.poll() is not None:
            return False

        # Periodic progress logging (every 30s)
        if time.time() - last_log_time > 30:
            logger.info(f"Still waiting for server... ({int(elapsed)}s elapsed, loading model)")
            last_log_time = time.time()

        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            pass
        time.sleep(0.5)


def _release_gpu_memory():
    """Release GPU memory before starting local LLM server.

    This is critical when running after Whisper transcription, which may
    still hold GPU memory. Without cleanup, llama-cpp-python may fail
    to initialize with cryptic errors.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Released GPU memory (torch.cuda.empty_cache)")
    except ImportError:
        pass  # torch not available, skip CUDA cleanup


def _check_existing_llama_servers() -> list:
    """Check for existing llama-cpp server processes.

    Returns list of (pid, cmdline) tuples for any found servers.
    """
    found = []
    try:
        if platform.system() == 'Windows':
            # Use tasklist to find python processes, then filter
            result = subprocess.run(
                ['tasklist', '/fi', 'imagename eq python.exe', '/fo', 'csv', '/nh'],
                capture_output=True, text=True, timeout=5
            )
            # This gives us PIDs but not cmdlines on Windows
            # Just check if llama_cpp.server might be running via wmic
            result = subprocess.run(
                ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline'],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split('\n'):
                if 'llama_cpp.server' in line or 'llama-cpp-python' in line:
                    # Extract PID (last number on the line)
                    parts = line.strip().split()
                    if parts:
                        try:
                            pid = int(parts[-1])
                            found.append((pid, line.strip()[:100]))
                        except ValueError:
                            pass
        else:
            # Unix: use ps
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split('\n'):
                if 'llama_cpp.server' in line or 'llama-cpp-python' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            found.append((pid, line[:100]))
                        except ValueError:
                            pass
    except Exception as e:
        logger.debug(f"Could not check for existing servers: {e}")

    return found


def start_local_server(
    model: str = "auto",
    n_gpu_layers: int = -1,
    n_ctx: int = 8192
) -> Tuple[str, int]:
    """
    Start the local LLM server.

    Args:
        model: Model ID from MODEL_REGISTRY or 'auto'
        n_gpu_layers: GPU layers to offload (-1 = all, 0 = CPU only)
        n_ctx: Context window size

    Returns:
        Tuple of (api_base_url, port)

    Raises:
        RuntimeError: If server fails to start or llama-cpp-python unavailable
    """
    global _server_process, _server_port

    # Release GPU memory from previous operations (e.g., Whisper transcription)
    # This helps prevent llama-cpp-python initialization failures
    _release_gpu_memory()

    # Ensure llama-cpp-python is installed (lazy download on first use)
    if not ensure_llama_cpp_installed():
        raise RuntimeError(
            "llama-cpp-python is not installed and could not be downloaded automatically.\n"
            "Install manually with: pip install llama-cpp-python[server]\n"
            "Or use cloud providers: whisperjav-translate -i file.srt --provider deepseek"
        )

    # Stop existing server if running (in THIS process)
    stop_local_server()

    # Check for other llama-cpp servers (in OTHER processes)
    existing_servers = _check_existing_llama_servers()
    if existing_servers:
        logger.warning(f"Found {len(existing_servers)} existing llama-cpp server(s) running!")
        for pid, cmdline in existing_servers:
            logger.warning(f"  PID {pid}: {cmdline}")
        logger.warning("These may be using GPU memory. If startup fails, close them first.")
        print(f"\n{'!'*60}")
        print(f"  WARNING: Found {len(existing_servers)} existing llama-cpp server(s)")
        print(f"  These may be using GPU memory and cause startup failure.")
        print(f"  If this fails, close other terminals running translations.")
        print(f"{'!'*60}\n")

    # Model selection
    if model == "auto":
        vram = get_available_vram_gb()
        model = get_best_model_for_vram(vram)
        logger.info(f"VRAM: {vram:.1f}GB, selected model: {model}")

    # Download model
    try:
        model_path = ensure_model_downloaded(model)
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")

    logger.info(f"Starting local server with {model_path.name}...")

    # Find free port
    port = _find_free_port()

    # Start server using llama-cpp-python's built-in server
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--n_gpu_layers", str(n_gpu_layers),
        "--n_ctx", str(n_ctx),
    ]

    # Use temp file for stderr to avoid blocking (pipes can fill up and block the server)
    import tempfile
    stderr_file = tempfile.NamedTemporaryFile(mode='w+', suffix='_llm_server.log', delete=False)
    stderr_path = stderr_file.name

    try:
        _server_process = subprocess.Popen(
            cmd,
            stderr=stderr_file,  # Write stderr to temp file (non-blocking)
        )
        _server_port = port
    except Exception as e:
        stderr_file.close()
        raise RuntimeError(f"Failed to start server: {e}")

    # Wait for server to be ready (5 min timeout for large models)
    logger.info(f"Waiting for server on port {port} (this may take 1-2 minutes for large models)...")
    if not _wait_for_server(port):
        # Get exit code and stderr if process died
        exit_code = _server_process.poll() if _server_process else None
        stderr_output = ""
        try:
            stderr_file.close()
            with open(stderr_path, 'r', encoding='utf-8', errors='replace') as f:
                stderr_output = f.read()
        except Exception:
            pass

        stop_local_server()

        # Provide specific guidance for common errors
        error_msg = f"Failed to start local server: Server process exited (code: {exit_code})"

        # Always show stderr content first for debugging
        if stderr_output:
            stderr_lines = stderr_output.strip().split('\n')[-15:]
            error_msg += f"\n\nServer stderr:\n" + "\n".join(stderr_lines)

        # Add specific guidance based on error type
        stderr_lower = stderr_output.lower()
        if "0xc000001d" in stderr_output or "illegal instruction" in stderr_lower:
            error_msg += (
                "\n\n[Diagnosis: Binary compatibility issue]"
                "\nThis error can be caused by:"
                "\n  1. CUDA version mismatch - llama-cpp-python was built for a different CUDA version"
                "\n  2. CPU instruction issue - binary requires AVX2/AVX512 instructions"
                "\n"
                "\nFixes to try:"
                "\n  - Reinstall llama-cpp-python matching your CUDA version:"
                "\n      pip uninstall llama-cpp-python"
                "\n      pip install llama-cpp-python  # Let lazy download get correct wheel"
                "\n  - Use CPU-only mode: --translate-gpu-layers 0"
                "\n  - Build from source for your CUDA version:"
                "\n      CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --no-binary llama-cpp-python"
            )
        elif "out of memory" in stderr_lower or ("cuda" in stderr_lower and "memory" in stderr_lower):
            error_msg += (
                "\n\n[Diagnosis: GPU memory issue]"
                "\nNot enough GPU memory to load the model."
                "\nCheck if another llama-cpp server is already running (e.g., from another terminal)."
                "\nTry: taskkill /f /im python.exe (WARNING: kills ALL Python processes)"
                "\nOr use a smaller model: --translate-model llama-3b"
            )
        elif not stderr_output:
            # No stderr captured - check for common issues
            error_msg += (
                "\n\n[No error details captured]"
                "\nPossible causes:"
                "\n- CUDA version mismatch (llama-cpp-python built for different CUDA)"
                "\n- Another llama-cpp server is using GPU memory"
                "\n- GPU driver issue"
                "\n- Model file corrupted"
                "\n"
                "\nTry:"
                "\n- Close other terminals running llama-cpp servers"
                "\n- Reinstall: pip uninstall llama-cpp-python && pip install llama-cpp-python"
                "\n- CPU-only mode: --translate-gpu-layers 0"
            )

        raise RuntimeError(error_msg)

    api_base = f"http://127.0.0.1:{port}/v1"
    logger.info(f"Local server ready at {api_base}")

    return api_base, port


def stop_local_server():
    """Stop the local LLM server if running."""
    global _server_process, _server_port

    if _server_process is not None:
        logger.info("Stopping local server...")
        try:
            _server_process.terminate()
            _server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_process.kill()
            _server_process.wait()
        except Exception as e:
            logger.warning(f"Error stopping server: {e}")
        finally:
            _server_process = None
            _server_port = None

    # Cleanup GPU memory
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def is_server_running() -> bool:
    """Check if the local server is running."""
    global _server_process
    return _server_process is not None and _server_process.poll() is None


def get_server_info() -> Optional[Tuple[str, int]]:
    """Get info about the running server."""
    global _server_port
    if is_server_running() and _server_port:
        return f"http://127.0.0.1:{_server_port}/v1", _server_port
    return None


def list_models() -> dict:
    """List available models with descriptions."""
    return {k: v['desc'] for k, v in MODEL_REGISTRY.items()}


# Register atexit handler to prevent orphan llama-cpp server processes if Python crashes
atexit.register(stop_local_server)
