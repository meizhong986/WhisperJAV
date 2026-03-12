#!/usr/bin/env python3
"""
WhisperJAV Source Installation Script — uv-based
=================================================

Thin wrapper around ``uv sync`` that handles GPU detection and
PyTorch index selection.  The old 6-stage pip orchestration is no
longer needed because uv resolves per-package indexes natively via
``[tool.uv.sources]`` in pyproject.toml.

Prerequisites:
    - Python 3.10-3.12 (with pip — used to auto-install uv if missing)
    - FFmpeg in system PATH
    - Git

Usage:
    python install.py [options]

Options:
    --cpu-only              Install CPU-only PyTorch (no CUDA)
    --cuda VERSION          Install PyTorch for specific CUDA version (e.g. cu118, cu124, cu128)
    --no-speech-enhancement Skip speech enhancement packages
    --minimal               Minimal install (transcription only)
    --dev                   Install in development/editable mode
    --local-llm             Install local LLM support (prebuilt wheel)
    --local-llm-build       Install local LLM support (build from source if needed)
    --no-local-llm          Skip local LLM installation
    --skip-preflight        Skip preflight checks (disk space, network, etc.)
    --log-file PATH         Custom log file path
    --help                  Show this help message

Examples:
    python install.py                    # Standard install (auto-detect GPU)
    python install.py --cpu-only         # CPU-only install
    python install.py --cuda cu118       # Explicit CUDA 11.8
    python install.py --minimal --dev    # Minimal dev install
    python install.py --local-llm        # Include local LLM (no prompt)
"""

import os
import sys
import argparse
import shutil
import platform as platform_module
from datetime import datetime
from pathlib import Path

# =============================================================================
# Constants
# =============================================================================

_source_dir = Path(__file__).parent

# PyTorch index URLs by CUDA version
PYTORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu128": "https://download.pytorch.org/whl/cu128",
}

# =============================================================================
# Bootstrap: Import installer module (for GPU detection + preflight)
# =============================================================================

if str(_source_dir) not in sys.path:
    sys.path.insert(0, str(_source_dir))

try:
    from whisperjav.installer import (
        detect_gpu,
        check_python_version as _check_python,
        check_ffmpeg as _check_ffmpeg,
        check_git as _check_git,
    )
    _INSTALLER_AVAILABLE = True
except ImportError:
    _INSTALLER_AVAILABLE = False

# Bootstrap llama_build_utils for --local-llm
try:
    import importlib.util
    import types

    _translate_dir = _source_dir / "whisperjav" / "translate"

    for _pkg_name, _pkg_path in [
        ("whisperjav", _source_dir / "whisperjav"),
        ("whisperjav.translate", _translate_dir),
    ]:
        if _pkg_name not in sys.modules:
            _pkg = types.ModuleType(_pkg_name)
            _pkg.__path__ = [str(_pkg_path)]
            _pkg.__package__ = _pkg_name
            sys.modules[_pkg_name] = _pkg

    _cuda_config_path = _translate_dir / "llama_cuda_config.py"
    _cuda_spec = importlib.util.spec_from_file_location(
        "whisperjav.translate.llama_cuda_config", _cuda_config_path)
    _cuda_mod = importlib.util.module_from_spec(_cuda_spec)
    sys.modules["whisperjav.translate.llama_cuda_config"] = _cuda_mod
    _cuda_spec.loader.exec_module(_cuda_mod)

    _utils_path = _translate_dir / "llama_build_utils.py"
    _utils_spec = importlib.util.spec_from_file_location(
        "whisperjav.translate.llama_build_utils", _utils_path)
    _utils_mod = importlib.util.module_from_spec(_utils_spec)
    _utils_mod.__package__ = "whisperjav.translate"
    sys.modules["whisperjav.translate.llama_build_utils"] = _utils_mod
    _utils_spec.loader.exec_module(_utils_mod)

    get_llama_cpp_source_info = _utils_mod.get_llama_cpp_source_info
    get_prebuilt_wheel_url = _utils_mod.get_prebuilt_wheel_url
except Exception:
    get_llama_cpp_source_info = None
    get_prebuilt_wheel_url = None


# =============================================================================
# Logging
# =============================================================================

_LOG_FILE: Path = None


def _init_logging(log_path: Path):
    global _LOG_FILE
    _LOG_FILE = log_path
    try:
        with open(_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"WhisperJAV Installation Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    except Exception:
        pass


def log(message: str):
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
    log("")
    log("=" * 60)
    log(f"  {title}")
    log("=" * 60)


# =============================================================================
# uv command resolution
# =============================================================================

def _uv_cmd() -> list:
    """Return the command prefix to invoke uv.

    Prefers the bare ``uv`` binary on PATH.  Falls back to
    ``python -m uv`` which always works when uv is pip-installed
    (even if Scripts/ isn't on PATH yet — common on Windows).
    """
    if shutil.which("uv"):
        return ["uv"]
    return [sys.executable, "-m", "uv"]


# =============================================================================
# Preflight Checks
# =============================================================================

def check_uv() -> bool:
    """Check that uv is installed and reachable. Auto-install if missing."""
    import subprocess

    # Check if uv is already available (on PATH or via python -m)
    for cmd in [["uv", "--version"], [sys.executable, "-m", "uv", "--version"]]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                log(f"  [OK] uv found: {version}")
                return True
        except (FileNotFoundError, Exception):
            continue

    # uv not found — attempt auto-bootstrap via pip
    log("  [INFO] uv not found — installing automatically via pip...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "uv"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            # Verify uv is now reachable (on PATH or via python -m)
            for verify_cmd in [["uv", "--version"],
                               [sys.executable, "-m", "uv", "--version"]]:
                try:
                    verify = subprocess.run(
                        verify_cmd, capture_output=True, text=True, timeout=10
                    )
                    if verify.returncode == 0:
                        log(f"  [OK] uv installed successfully")
                        return True
                except (FileNotFoundError, Exception):
                    continue
            log("  [WARN] uv was installed but is not reachable.")
            log("         Try restarting your terminal, then re-run install.py.")
            return False
    except FileNotFoundError:
        log("  [WARN] pip not available — cannot auto-install uv")
    except subprocess.TimeoutExpired:
        log("  [WARN] pip install uv timed out")
    except Exception as e:
        log(f"  [WARN] Failed to auto-install uv: {e}")

    log("  [!!] Could not install uv automatically.")
    log("       Manual install: pip install uv")
    log("       Or: https://docs.astral.sh/uv/getting-started/installation/")
    if sys.platform == "win32":
        log("       Windows:     powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
    return False


def check_disk_space(min_gb: int = 8) -> bool:
    try:
        total, used, free = shutil.disk_usage(_source_dir)
        free_gb = free / (1024**3)
        log(f"  Disk free space: {free_gb:.1f} GB (minimum: {min_gb} GB)")
        if free_gb < min_gb:
            log(f"  [!!] ERROR: Not enough disk space!")
            return False
        log(f"  [OK] Disk space sufficient")
        return True
    except Exception as e:
        log(f"  [WARN] Could not check disk space: {e}")
        return True


def check_network(timeout: int = 10) -> bool:
    log("  Checking network connectivity...")
    try:
        import urllib.request
        urllib.request.urlopen("https://pypi.org", timeout=timeout)
        log("  [OK] Network connectivity to PyPI")
        return True
    except Exception as e:
        log(f"  [!!] ERROR: Cannot reach PyPI: {e}")
        return False


def run_preflight_checks() -> bool:
    log_section("Preflight Checks")
    all_passed = True

    if not check_uv():
        all_passed = False
    if not check_disk_space(8):
        all_passed = False
    if not check_network():
        all_passed = False

    # Check prerequisites via installer module
    if _INSTALLER_AVAILABLE:
        py_result = _check_python()
        is_compatible = "- OK" in py_result.message
        status = "[OK]" if is_compatible else "[!!]"
        log(f"  {status} {py_result.message}")
        if not is_compatible:
            log("  ERROR: Python version not compatible (requires 3.10-3.12)")
            all_passed = False

        ff_result = _check_ffmpeg()
        status = "[OK]" if ff_result.found else "[WARN]"
        log(f"  {status} {ff_result.message}")

        git_result = _check_git()
        status = "[OK]" if git_result.found else "[!!]"
        log(f"  {status} {git_result.message}")
        if not git_result.found:
            all_passed = False
    else:
        # Fallback checks
        major, minor = sys.version_info[:2]
        if major < 3 or (major == 3 and minor < 10) or (major == 3 and minor > 12):
            log(f"  [!!] Python {major}.{minor} not compatible (requires 3.10-3.12)")
            all_passed = False
        else:
            log(f"  [OK] Python {major}.{minor}")

        if not shutil.which("ffmpeg"):
            log("  [WARN] FFmpeg not found in PATH")
        else:
            log("  [OK] FFmpeg found")

        if not shutil.which("git"):
            log("  [!!] Git not found in PATH")
            all_passed = False
        else:
            log("  [OK] Git found")

    if not all_passed:
        log("")
        log("  PREFLIGHT CHECKS FAILED — please fix the issues above.")
    else:
        log("")
        log("  All preflight checks passed!")
    return all_passed


# =============================================================================
# GPU Detection
# =============================================================================

def detect_cuda_version(args) -> str:
    """Determine CUDA version from args + GPU detection.

    Returns: "cpu", "cu118", "cu124", "cu128", or "metal" (Apple Silicon).
    """
    if args.cpu_only:
        return "cpu"
    if args.cuda:
        cuda = args.cuda.lower().replace("cuda", "").replace("cu", "")
        # Normalize: "118" -> "cu118", "12.8" -> "cu128"
        cuda = cuda.replace(".", "")
        key = f"cu{cuda}"
        if key in PYTORCH_INDEXES:
            return key
        log(f"  [WARN] Unknown CUDA version '{args.cuda}', defaulting to cu128")
        return "cu128"

    # Auto-detect
    if _INSTALLER_AVAILABLE:
        gpu_info = detect_gpu()
        if gpu_info.detected:
            log(f"  GPU detected: {gpu_info.name}")
            if gpu_info.driver_version:
                log(f"  Driver: {gpu_info.driver_version[0]}.{gpu_info.driver_version[1]}")
            cuda = gpu_info.cuda_version or "cpu"
            log(f"  Selected: {cuda}")
            return cuda
        else:
            log(f"  {gpu_info.message}")

    # Apple Silicon check
    if sys.platform == "darwin" and platform_module.machine() == "arm64":
        log("  Apple Silicon detected — using MPS (PyPI torch)")
        return "metal"

    # Basic nvidia-smi fallback
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        gpu_name = result.stdout.strip().split('\n')[0]
        if gpu_name:
            log(f"  GPU detected: {gpu_name}")
            return "cu128"
    except Exception:
        pass

    log("  No NVIDIA GPU detected — using CPU")
    return "cpu"


# =============================================================================
# uv sync wrapper
# =============================================================================

def run_uv_sync(extras: list, cuda_version: str, dev: bool = False) -> bool:
    """Run ``uv sync`` with the given extras and pytorch index override.

    Args:
        extras: List of extra names to install (e.g. ["cli", "gui"]).
        cuda_version: "cpu", "cu118", "cu128", or "metal".
        dev: If True, include dev dependencies.

    Returns:
        True on success, False on failure.
    """
    import subprocess

    cmd = _uv_cmd() + ["sync"]

    # Pin to the running Python interpreter so uv doesn't follow
    # .python-version (which may request a different version).
    cmd.extend(["--python", sys.executable])

    for extra in extras:
        cmd.extend(["--extra", extra])

    # Override pytorch index based on GPU detection
    # For macOS ("metal"), don't override — pyproject.toml sources already
    # route macOS to default PyPI via platform marker.
    if cuda_version != "metal" and cuda_version in PYTORCH_INDEXES:
        cmd.extend(["--index", f"pytorch={PYTORCH_INDEXES[cuda_version]}"])

    if dev:
        # dev deps are defined as [project.optional-dependencies] extra,
        # not as a [dependency-groups] group.
        cmd.extend(["--extra", "dev"])

    # --verbose makes uv show download progress and resolution details
    cmd.append("--verbose")

    log(f"\n>>> Running: {' '.join(cmd)}")

    try:
        # Stream output live so the user sees download progress in real time.
        # Also tee to the log file for post-mortem debugging.
        process = subprocess.Popen(
            cmd,
            cwd=str(_source_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout
            text=True,
            bufsize=1,  # line-buffered
        )

        for line in process.stdout:
            line = line.rstrip('\n')
            if line:
                print(f"    {line}", flush=True)
                if _LOG_FILE:
                    try:
                        with open(_LOG_FILE, "a", encoding="utf-8") as f:
                            f.write(f"    {line}\n")
                    except Exception:
                        pass

        process.wait(timeout=1800)

        if process.returncode != 0:
            log(f"    [ERROR] uv sync failed (exit code {process.returncode})")
            return False

        log("    [OK] uv sync completed successfully")
        return True

    except subprocess.TimeoutExpired:
        process.kill()
        log("    [ERROR] uv sync timed out after 30 minutes")
        return False
    except FileNotFoundError:
        log("    [ERROR] uv not found — install from https://docs.astral.sh/uv/")
        return False
    except Exception as e:
        log(f"    [ERROR] Unexpected error: {e}")
        return False


# =============================================================================
# llama-cpp-python (special handling — CUDA wheel URL logic)
# =============================================================================

def _install_local_llm(build_from_source: bool):
    """Install llama-cpp-python with platform-specific handling."""
    import subprocess

    log("\n    Installing llama-cpp-python for local LLM translation...")

    is_apple_silicon = (sys.platform == "darwin" and platform_module.machine() == "arm64")
    is_intel_mac = (sys.platform == "darwin" and platform_module.machine() != "arm64")

    def _uv_pip_install(args: list, description: str, allow_fail: bool = False) -> bool:
        cmd = _uv_cmd() + ["pip", "install"] + args
        log(f"\n>>> {description}")
        log(f"    {' '.join(cmd[:8])}{'...' if len(cmd) > 8 else ''}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                if allow_fail:
                    log(f"    [WARN] {description} — failed (optional)")
                    return False
                log(f"    [ERROR] {description} — failed")
                if result.stderr:
                    for line in result.stderr.strip().split('\n')[-5:]:
                        log(f"    {line}")
                return False
            log(f"    [OK] {description}")
            return True
        except subprocess.TimeoutExpired:
            log(f"    [WARN] {description} — timed out")
            return False
        except Exception as e:
            log(f"    [WARN] {description} — {e}")
            return False

    if is_apple_silicon:
        log("    Apple Silicon detected — checking for prebuilt Metal wheel...")
        prebuilt_installed = False
        if get_prebuilt_wheel_url:
            wheel_url, wheel_backend = get_prebuilt_wheel_url(verbose=True)
            if wheel_url:
                log(f"    Found prebuilt wheel: {wheel_backend}")
                prebuilt_installed = _uv_pip_install(
                    [wheel_url], f"Install llama-cpp-python ({wheel_backend})", allow_fail=True
                )
                if prebuilt_installed:
                    _uv_pip_install(
                        ["llama-cpp-python[server]"],
                        "Install llama-cpp-python server extras", allow_fail=True
                    )

        if not prebuilt_installed:
            log("    Building from source with Metal support...")
            if get_llama_cpp_source_info:
                git_url, backend, cmake_args, env_vars = get_llama_cpp_source_info()
                for key, value in env_vars.items():
                    os.environ[key] = value
                if cmake_args:
                    os.environ["CMAKE_ARGS"] = cmake_args
                _uv_pip_install([git_url], f"Install llama-cpp-python ({backend})", allow_fail=True)

    elif is_intel_mac:
        if build_from_source and get_llama_cpp_source_info:
            log("    Intel Mac detected — building CPU-only version.")
            git_url, backend, cmake_args, env_vars = get_llama_cpp_source_info()
            for key, value in env_vars.items():
                os.environ[key] = value
            _uv_pip_install([git_url], f"Install llama-cpp-python ({backend})", allow_fail=True)
        else:
            log("    Intel Mac — no prebuilt wheels. Use --local-llm-build to build from source.")

    else:
        # Windows/Linux
        if get_prebuilt_wheel_url:
            wheel_url, wheel_backend = get_prebuilt_wheel_url(verbose=True)
            if wheel_url:
                log(f"    Backend: {wheel_backend}")
                _uv_pip_install([wheel_url], f"Install llama-cpp-python ({wheel_backend})", allow_fail=True)
                _uv_pip_install(
                    ["llama-cpp-python[server]"],
                    "Install llama-cpp-python server extras", allow_fail=True
                )
            elif build_from_source and get_llama_cpp_source_info:
                log("    No prebuilt wheel — building from source...")
                git_url, backend, cmake_args, env_vars = get_llama_cpp_source_info()
                for key, value in env_vars.items():
                    os.environ[key] = value
                if cmake_args:
                    os.environ["CMAKE_ARGS"] = cmake_args
                _uv_pip_install([git_url], f"Install llama-cpp-python ({backend})", allow_fail=True)
            else:
                log("    No prebuilt wheel available. Use --local-llm-build to build from source.")
        else:
            log("    ERROR: llama_build_utils not available")


# =============================================================================
# Timed User Input
# =============================================================================

def timed_input(prompt: str, timeout_seconds: int, default_response: str) -> str:
    import threading

    print(prompt, end='', flush=True)
    result = [default_response]

    def get_input():
        try:
            result[0] = input()
        except (EOFError, KeyboardInterrupt):
            result[0] = default_response

    input_thread = threading.Thread(target=get_input, daemon=True)
    input_thread.start()
    input_thread.join(timeout=timeout_seconds)

    if input_thread.is_alive():
        print(f"\n[Auto-continuing after {timeout_seconds}s timeout — using default: '{default_response}']")
        return default_response
    return result[0]


# =============================================================================
# Verification
# =============================================================================

def _venv_python() -> str:
    """Return the path to the venv Python created by uv sync."""
    if sys.platform == "win32":
        venv_py = _source_dir / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = _source_dir / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    # Fallback to the running interpreter (e.g. if uv reused it)
    return sys.executable


def _verify_installation():
    """Verify WhisperJAV was installed correctly."""
    import subprocess

    python = _venv_python()
    log(f"    Verifying with: {python}")

    try:
        result = subprocess.run(
            [python, "-c", "import whisperjav; print(f'WhisperJAV {whisperjav.__version__}')"],
            capture_output=True, text=True, check=True
        )
        log(f"    [OK] {result.stdout.strip()} installed successfully!")
    except subprocess.CalledProcessError:
        log("    [WARN] Could not verify WhisperJAV import")

    # Check torch CUDA
    try:
        result = subprocess.run(
            [python, "-c",
             "import torch; cuda = torch.cuda.is_available(); "
             "dev = torch.cuda.get_device_name(0) if cuda else 'N/A'; "
             "print(f'torch {torch.__version__}, CUDA={cuda}, device={dev}')"],
            capture_output=True, text=True, check=True
        )
        log(f"    [OK] {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        log("    [INFO] Could not verify torch CUDA status")


def create_failure_file(error_message: str):
    failure_file = _source_dir / "INSTALLATION_FAILED.txt"
    try:
        with open(failure_file, "w", encoding="utf-8") as f:
            f.write("WhisperJAV Installation Failed\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Error: {error_message}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Troubleshooting:\n")
            f.write("- Check install_log.txt for details\n")
            f.write("- Ensure uv, Python 3.10-3.12, FFmpeg, and Git are in PATH\n")
            f.write("- Try: uv cache clean\n")
            f.write("- For network issues, check firewall/proxy settings\n\n")
            f.write("Support: https://github.com/meizhong986/WhisperJAV/issues\n")
    except Exception:
        pass


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WhisperJAV Source Installation Script (uv-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--cpu-only", action="store_true",
                        help="Install CPU-only PyTorch")
    parser.add_argument("--cuda", type=str, default=None, metavar="VERSION",
                        help="CUDA version (e.g. cu118, cu124, cu128)")
    parser.add_argument("--no-speech-enhancement", action="store_true",
                        help="Skip speech enhancement packages")
    parser.add_argument("--minimal", action="store_true",
                        help="Minimal install (transcription only)")
    parser.add_argument("--dev", action="store_true",
                        help="Install in development/editable mode")
    parser.add_argument("--local-llm", action="store_true",
                        help="Install local LLM support (fast — prebuilt wheel)")
    parser.add_argument("--local-llm-build", action="store_true",
                        help="Install local LLM support (build from source if needed)")
    parser.add_argument("--no-local-llm", action="store_true",
                        help="Skip local LLM installation")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip preflight checks")
    parser.add_argument("--log-file", type=str, default=None, metavar="PATH",
                        help="Custom log file path")
    # Legacy compat flags (still accepted, mapped internally)
    parser.add_argument("--cuda118", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cuda128", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Map legacy flags
    if args.cuda118 and not args.cuda:
        args.cuda = "cu118"
    if args.cuda128 and not args.cuda:
        args.cuda = "cu128"

    # -------------------------------------------------------------------------
    # Validate source directory
    # -------------------------------------------------------------------------
    if not (_source_dir / "pyproject.toml").exists():
        print("ERROR: pyproject.toml not found.")
        print("       Run this script from the WhisperJAV source directory.")
        print("       git clone https://github.com/meizhong986/whisperjav.git")
        print("       cd whisperjav && python install.py")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Initialize logging
    # -------------------------------------------------------------------------
    log_path = Path(args.log_file) if args.log_file else _source_dir / "install_log.txt"
    _init_logging(log_path)
    log_section("WhisperJAV Source Installation (uv)")
    log(f"Python: {sys.version}")
    log(f"Platform: {sys.platform}")
    log(f"Source: {_source_dir}")

    # -------------------------------------------------------------------------
    # Preflight checks
    # -------------------------------------------------------------------------
    if not args.skip_preflight:
        if not run_preflight_checks():
            create_failure_file("Preflight checks failed")
            sys.exit(1)
    else:
        log("Skipping preflight checks (--skip-preflight)")

    # -------------------------------------------------------------------------
    # GPU detection
    # -------------------------------------------------------------------------
    log_section("GPU Detection")
    cuda_version = detect_cuda_version(args)

    # -------------------------------------------------------------------------
    # Determine extras
    # -------------------------------------------------------------------------
    if args.minimal:
        extras = ["cli"]
    else:
        extras = ["cli", "gui", "translate", "llm", "huggingface", "qwen",
                   "analysis", "compatibility"]
        if not args.no_speech_enhancement:
            extras.append("enhance")

    log_section("Installation Plan")
    log(f"  PyTorch: {cuda_version}")
    log(f"  Extras: {', '.join(extras)}")
    log(f"  Speech Enhancement: {'No' if args.no_speech_enhancement or args.minimal else 'Yes'}")
    log(f"  Mode: {'Development' if args.dev else 'Standard'}")
    log(f"  Log file: {_LOG_FILE}")

    # -------------------------------------------------------------------------
    # Run uv sync (single command replaces 6-stage pip pipeline)
    # -------------------------------------------------------------------------
    log_section("Installing Dependencies (uv sync)")
    success = run_uv_sync(extras, cuda_version, dev=args.dev)

    if not success:
        # Retry without enhance (most common failure point)
        if "enhance" in extras:
            log("")
            log("    Retrying without speech enhancement...")
            extras_retry = [e for e in extras if e != "enhance"]
            success = run_uv_sync(extras_retry, cuda_version, dev=args.dev)
            if success:
                log("    [OK] Installation succeeded without speech enhancement.")
                log("    You can install it later: uv sync --extra enhance")

    if not success:
        log("")
        log("    INSTALLATION FAILED")
        create_failure_file("uv sync failed — check install_log.txt for details")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Local LLM (llama-cpp-python) — handled separately
    # -------------------------------------------------------------------------
    if args.local_llm or args.local_llm_build:
        _install_local_llm(args.local_llm_build)
    elif args.no_local_llm:
        log("\n    Skipping local LLM (--no-local-llm specified)")
    else:
        log("")
        log("=" * 60)
        log("  LOCAL LLM TRANSLATION")
        log("=" * 60)
        log("")
        log("WhisperJAV can translate subtitles using local AI models,")
        log("allowing offline translation without API keys.")
        log("")
        response = timed_input("Install local LLM translation? (Y/n): ", 30, "y").strip().lower()
        if response in ('', 'y', 'yes'):
            _install_local_llm(build_from_source=False)
        else:
            log("Skipping local LLM. Install later: pip install llama-cpp-python[server]")

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------
    log_section("Verifying Installation")
    _verify_installation()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    log_section("Installation Complete!")
    log("")
    # Guide user to activate the venv created by uv sync
    venv_dir = _source_dir / ".venv"
    if venv_dir.exists():
        log("  Packages were installed into the project's .venv.")
        log("  Activate it before running WhisperJAV:")
        log("")
        if sys.platform == "win32":
            log(f"    .venv\\Scripts\\activate")
        else:
            log(f"    source .venv/bin/activate")
        log("")
        log("  Or use 'uv run' to run commands in the venv:")
        log("    uv run whisperjav video.mp4 --mode balanced")
        log("    uv run whisperjav-gui")
    log("")
    log("  To run WhisperJAV:")
    log("    whisperjav video.mp4 --mode balanced")
    log("")
    log("  To run with GUI:")
    log("    whisperjav-gui")
    log("")
    log("  For help:")
    log("    whisperjav --help")
    log("")
    if _LOG_FILE:
        log(f"  Installation log: {_LOG_FILE}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
