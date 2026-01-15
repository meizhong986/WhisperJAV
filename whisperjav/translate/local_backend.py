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


def _is_llama_cpp_installed() -> bool:
    """Check if llama-cpp-python is installed and functional."""
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_cuda_version() -> Optional[str]:
    """
    Detect CUDA version from nvidia-smi or torch.

    Returns:
        CUDA version string like "cu124" or None if not available
    """
    # Try torch first (most reliable)
    try:
        import torch
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda
            if cuda_ver:
                # Convert "12.4" to "cu124"
                parts = cuda_ver.split(".")
                if len(parts) >= 2:
                    return f"cu{parts[0]}{parts[1]}"
    except Exception:
        pass

    # Fall back to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            driver_ver = result.stdout.strip().split("\n")[0]
            # Map driver version to CUDA version
            # Reference: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
            major = int(driver_ver.split(".")[0])
            if major >= 570:
                return "cu128"
            elif major >= 560:
                return "cu126"
            elif major >= 551:
                return "cu124"
            elif major >= 531:
                return "cu121"
            elif major >= 520:
                return "cu118"
    except Exception:
        pass

    return None


def _get_wheel_filename(cuda_version: Optional[str] = None) -> Tuple[str, str]:
    """
    Build wheel filename based on platform, Python version, and CUDA version.

    Returns:
        Tuple of (wheel_filename, backend_subfolder)
    """
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"

    # Determine platform tag
    if sys.platform == "win32":
        plat_tag = "win_amd64"
    elif sys.platform == "linux":
        plat_tag = "manylinux_2_17_x86_64.manylinux2014_x86_64"
    elif sys.platform == "darwin":
        if platform.machine() == "arm64":
            plat_tag = "macosx_11_0_arm64"
        else:
            plat_tag = "macosx_10_15_x86_64"
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    # Determine backend subfolder
    if sys.platform == "darwin" and platform.machine() == "arm64":
        backend = "metal"
    elif cuda_version:
        backend = cuda_version
    else:
        backend = "cpu"

    # Build wheel filename
    # Format: llama_cpp_python-{version}-{pyver}-{pyver}-{platform}.whl
    wheel_name = f"llama_cpp_python-{WHEEL_VERSION}-{py_ver}-{py_ver}-{plat_tag}.whl"

    return wheel_name, backend


def _download_wheel_from_huggingface(cuda_version: Optional[str] = None) -> Optional[Path]:
    """
    Download llama-cpp-python wheel from HuggingFace.

    Returns:
        Path to downloaded wheel, or None if not found
    """
    try:
        from huggingface_hub import hf_hub_download, HfFileSystemResolvedPath
        from huggingface_hub.utils import EntryNotFoundError

        wheel_name, backend = _get_wheel_filename(cuda_version)
        wheel_path = f"llama-cpp-python/{backend}/{wheel_name}"

        logger.info(f"Downloading wheel from HuggingFace: {WHEEL_REPO_ID}/{wheel_path}")
        print(f"Downloading llama-cpp-python from HuggingFace...")
        print(f"  Wheel: {wheel_name}")
        print(f"  Backend: {backend}")

        local_path = hf_hub_download(
            repo_id=WHEEL_REPO_ID,
            filename=wheel_path,
            repo_type="dataset"
        )

        logger.info(f"Downloaded wheel to: {local_path}")
        return Path(local_path)

    except EntryNotFoundError:
        logger.warning(f"Wheel not found on HuggingFace: {wheel_path}")
        return None
    except Exception as e:
        logger.warning(f"Failed to download from HuggingFace: {e}")
        return None


def _download_wheel_from_github(cuda_version: Optional[str] = None) -> Optional[Path]:
    """
    Download llama-cpp-python wheel from JamePeng GitHub releases (fallback).

    Returns:
        Path to downloaded wheel, or None if not found
    """
    import json
    import tempfile
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
        os_tag = "metal"
        if platform.machine() == "arm64":
            wheel_platform = "arm64"
        else:
            wheel_platform = "x86_64"
    else:
        return None

    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"

    # Build search criteria
    # Note: CUDA 13.x (cu130) is NOT supported - no PyTorch/llama-cpp wheels available
    target_cudas = []
    if cuda_version and sys.platform in ("win32", "linux"):
        # Search compatible CUDA versions in order (highest supported first)
        if cuda_version >= "cu128":
            target_cudas = ["cu128", "cu126", "cu124"]
        elif cuda_version >= "cu126":
            target_cudas = ["cu126", "cu124"]
        elif cuda_version >= "cu124":
            target_cudas = ["cu124"]
        elif cuda_version >= "cu121":
            target_cudas = ["cu121"]

    try:
        logger.info("Searching JamePeng GitHub releases...")
        print("Searching JamePeng GitHub releases for prebuilt wheel...")

        api_url = "https://api.github.com/repos/JamePeng/llama-cpp-python/releases?per_page=50"
        req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=15) as response:
            releases = json.loads(response.read().decode())

        # Search for matching wheel
        if os_tag == "metal":
            # macOS: look for metal releases
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
                        return _download_wheel_url(wheel_url, name)
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
                            return _download_wheel_url(wheel_url, name)

        logger.warning("No matching wheel found on JamePeng GitHub")
        return None

    except Exception as e:
        logger.warning(f"Failed to search JamePeng GitHub: {e}")
        return None


def _download_wheel_url(url: str, filename: str) -> Optional[Path]:
    """Download wheel from URL to temp directory."""
    import tempfile
    import urllib.request

    try:
        logger.info(f"Downloading: {filename}")
        print(f"  Found: {filename}")
        print(f"  Downloading...")

        temp_dir = Path(tempfile.gettempdir()) / "whisperjav_wheels"
        temp_dir.mkdir(exist_ok=True)
        dest_path = temp_dir / filename

        if dest_path.exists():
            logger.info(f"Using cached wheel: {dest_path}")
            return dest_path

        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Downloaded to: {dest_path}")
        return dest_path

    except Exception as e:
        logger.warning(f"Failed to download wheel: {e}")
        return None


def _install_wheel(wheel_path: Path) -> bool:
    """Install wheel using pip."""
    try:
        logger.info(f"Installing wheel: {wheel_path}")
        print(f"Installing llama-cpp-python...")

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
            logger.info("Successfully installed llama-cpp-python")
            print("  Successfully installed!")
            return True
        else:
            logger.error(f"pip install failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Failed to install wheel: {e}")
        return False


def ensure_llama_cpp_installed() -> bool:
    """
    Ensure llama-cpp-python is installed, downloading if necessary.

    This implements lazy download: on first use of `--provider local`,
    the wheel is automatically downloaded and installed.

    Returns:
        True if llama-cpp-python is available, False otherwise
    """
    if _is_llama_cpp_installed():
        logger.debug("llama-cpp-python already installed")
        return True

    print("\n" + "=" * 60)
    print("  FIRST-TIME SETUP: Installing llama-cpp-python")
    print("=" * 60)
    print("\nllama-cpp-python is required for local LLM translation.")
    print("This is a one-time download (~700MB).\n")

    # Detect CUDA version
    cuda_version = _detect_cuda_version()
    if cuda_version:
        print(f"Detected CUDA: {cuda_version}")
    else:
        print("CUDA not detected (will use CPU or Metal)")

    # Try HuggingFace first
    wheel_path = _download_wheel_from_huggingface(cuda_version)

    # Fall back to JamePeng GitHub
    if not wheel_path:
        print("\nHuggingFace wheel not found, trying JamePeng GitHub...")
        wheel_path = _download_wheel_from_github(cuda_version)

    # Install if we got a wheel
    if wheel_path and _install_wheel(wheel_path):
        # Verify installation
        if _is_llama_cpp_installed():
            print("\n✓ llama-cpp-python installed successfully!")
            print("  Future runs will start immediately.\n")
            return True

    # All methods failed
    print("\n" + "!" * 60)
    print("  INSTALLATION FAILED")
    print("!" * 60)
    print("\nCould not install llama-cpp-python automatically.")
    print("\nManual installation options:")
    print("  1. pip install llama-cpp-python[server]")
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
    """Download GGUF model if not already cached."""
    if model_id not in MODEL_REGISTRY:
        valid = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_id}. Valid: {valid}")

    from huggingface_hub import hf_hub_download

    info = MODEL_REGISTRY[model_id]
    logger.info(f"Ensuring model downloaded: {info['file']}")

    path = hf_hub_download(
        repo_id=info['repo'],
        filename=info['file']
    )
    return Path(path)


def _find_free_port() -> int:
    """Find a free port for the server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _wait_for_server(port: int) -> bool:
    """Wait for server to be ready (no timeout - models can take a while to load)."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/v1/models"

    while True:
        # Check if process died
        if _server_process is not None and _server_process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            pass
        time.sleep(0.5)


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

    # Ensure llama-cpp-python is installed (lazy download on first use)
    if not ensure_llama_cpp_installed():
        raise RuntimeError(
            "llama-cpp-python is not installed and could not be downloaded automatically.\n"
            "Install manually with: pip install llama-cpp-python[server]\n"
            "Or use cloud providers: whisperjav-translate -i file.srt --provider deepseek"
        )

    # Stop existing server if running
    stop_local_server()

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

    try:
        _server_process = subprocess.Popen(
            cmd,
            # Don't capture output - let user see server logs (model loading progress, errors)
        )
        _server_port = port
    except Exception as e:
        raise RuntimeError(f"Failed to start server: {e}")

    # Wait for server to be ready (no timeout - large models can take minutes)
    logger.info(f"Waiting for server on port {port}...")
    if not _wait_for_server(port):
        # Get exit code if process died
        exit_code = _server_process.poll() if _server_process else None
        stop_local_server()
        raise RuntimeError(f"Server process exited unexpectedly (code: {exit_code})")

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
