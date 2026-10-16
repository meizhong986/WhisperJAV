#!/usr/bin/env python3
"""
WhisperJAV Source Installation Script — uv-based
=================================================

Thin wrapper around ``uv sync`` that handles GPU detection and
PyTorch index selection.  The old 6-stage pip orchestration is no
longer needed because uv resolves per-package indexes natively via
``[tool.uv.sources]`` in pyproject.toml.

Prerequisites:
    - Python 3.13 (with pip — used to auto-install uv if missing)
    - FFmpeg in system PATH
    - Git

Usage:
    python install.py [options]

Options:
    --cpu-only              Install CPU-only PyTorch (no CUDA)
    --cuda VERSION          Install PyTorch for a specific CUDA build: cu130 (default) or cu128
    --no-speech-enhancement Skip speech enhancement packages
    --minimal               Minimal install (transcription only)
    --dev                   Install in development/editable mode
    --local                 Force local .venv even if conda/venv is active
    --skip-preflight        Skip preflight checks (disk space, network, etc.)
    --log-file PATH         Custom log file path
    --help                  Show this help message

Examples:
    python install.py                    # Standard install (auto-detect GPU)
    python install.py --cpu-only         # CPU-only install
    python install.py --cuda cu128       # CUDA 12.8 build instead of CUDA 13.0
    python install.py --minimal --dev    # Minimal dev install
"""

import os
import re
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

# PyTorch builds. v1.10.0 (REQ2): cu130 is the stack; cu128 is the per-PC fallback when
# cuBLAS 12 cannot be loaded next to CUDA 13 (item 2 spec D). Their indexes are in
# pyproject.toml ([tool.uv.index]); uv.lock holds all three (item 2f).
CUDA_BUILDS = ("cpu", "cu128", "cu130")
# Linux stays on CUDA 12.8 for 1.10.0 (owner, W45): on Linux, pip-installed cuBLAS 12 is found
# only with LD_LIBRARY_PATH set before Python starts (faster-whisper's README), so with CUDA 13
# nothing would load it for CTranslate2; with CUDA 12.8, PyTorch loads it, as in 1.9.5.
IS_LINUX = sys.platform.startswith("linux")
DEFAULT_CUDA = "cu128" if IS_LINUX else "cu130"
# v1.10.0 (item 2f): the uv.lock extra of each PyTorch build this script can choose.
LOCKED_EXTRAS = {"cu130": "cu130", "cu128": "cu128", "cpu": "cpu", "metal": "cpu"}

# The owner's texts (W43) and the links they name; the same as the Windows installer's.
ISSUES_URL = "https://github.com/meizhong986/WhisperJAV/issues"
RELEASE_195_URL = "https://github.com/meizhong986/WhisperJAV/releases/tag/v1.9.5"
NVIDIA_DRIVER_URL = "https://www.nvidia.com/Download/index.aspx"
MIN_COMPUTE_CAPABILITY = (7, 5)
MIN_DRIVER_CUDA13 = (580, 0, 0)
MIN_DRIVER_CUDA128 = (570, 0, 0)
MIN_DRIVER = MIN_DRIVER_CUDA128 if IS_LINUX else MIN_DRIVER_CUDA13

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
# Environment Detection
# =============================================================================

def _check_env_python_version(env_path: str) -> bool:
    """Check if the Python in an environment is compatible (3.13).

    Returns True if compatible or if check fails (fail-open — preflight
    will catch hard failures later).
    """
    import subprocess

    if sys.platform == "win32":
        py = Path(env_path) / "Scripts" / "python.exe"
    else:
        py = Path(env_path) / "bin" / "python"

    if not py.exists():
        return True  # Can't check — let preflight handle it

    try:
        result = subprocess.run(
            [str(py), "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            major, minor = map(int, result.stdout.strip().split())
            if major == 3 and minor == 13:
                return True
            log(f"  [WARN] Active environment Python {major}.{minor} is outside "
                f"the supported version (3.13)")
            log(f"         Environment: {env_path}")
            log(f"         Falling back to local .venv instead.")
            return False
    except Exception:
        pass
    return True  # Fail-open


def _detect_target_env(force_local: bool = False) -> tuple:
    """Detect active environment for uv sync targeting.

    Args:
        force_local: If True, skip detection and use .venv (--local flag).

    Returns:
        (env_path, is_external): env_path is the target environment path,
        or None to let uv create .venv. is_external indicates whether the
        env is one we didn't create (conda/venv) — used to decide --inexact.
    """
    if force_local:
        return None, False  # Let uv create .venv

    # VIRTUAL_ENV first — more specific (user explicitly activated)
    venv = os.environ.get("VIRTUAL_ENV")
    if venv and Path(venv).is_dir():
        if _check_env_python_version(venv):
            return venv, True
        return None, False  # Incompatible Python → fall back to .venv

    # CONDA_PREFIX second — may be base env
    conda = os.environ.get("CONDA_PREFIX")
    if conda and Path(conda).is_dir():
        if _check_env_python_version(conda):
            return conda, True
        return None, False  # Incompatible Python → fall back to .venv

    return None, False  # No active env → uv creates .venv (safe default)


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
            log("  ERROR: Python version not compatible (requires 3.13)")
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
        if (major, minor) != (3, 13):
            log(f"  [!!] Python {major}.{minor} not compatible (requires 3.13)")
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

    Returns: "cpu", "cu130", "cu128", "metal" (Apple Silicon), or "stop" (an old card
    or driver, item 2 spec E/F; the reason has been shown).
    """
    if args.cpu_only:
        return "cpu"
    if args.cuda:
        cuda = args.cuda.lower().replace("cuda", "").replace("cu", "")
        # Normalize: "118" -> "cu118", "12.8" -> "cu128"
        cuda = cuda.replace(".", "")
        key = f"cu{cuda}"
        if key in CUDA_BUILDS:
            return key
        log(f"  [WARN] Unknown CUDA version '{args.cuda}', defaulting to {DEFAULT_CUDA}")
        return DEFAULT_CUDA

    # --force-cuda: manual override when GPU detection fails (#200 Optimus laptops)
    if hasattr(args, 'force_cuda') and args.force_cuda:
        cuda = args.force_cuda.lower().replace("cuda", "").replace("cu", "").replace(".", "")
        key = f"cu{cuda}"
        if key in CUDA_BUILDS:
            log(f"  --force-cuda: using {key} (bypassing GPU detection)")
            return key
        log(f"  [WARN] --force-cuda '{args.force_cuda}' not recognized, using {DEFAULT_CUDA}")
        return DEFAULT_CUDA

    # Auto-detect
    if _INSTALLER_AVAILABLE:
        gpu_info = detect_gpu()
        if gpu_info.detected and gpu_info.cuda_version != "metal":
            log(f"  GPU detected: {gpu_info.name}")
            if gpu_info.driver_version:
                log(f"  Driver: {gpu_info.driver_version[0]}.{gpu_info.driver_version[1]:02d}")
            gate = gpu_support_gate(gpu_info)
            if gate != "gpu":
                return gate
            # The detector chooses by the CUDA 13 rule (driver 580+); the gate has already
            # checked this platform's own floor, so the platform's build is used.
            cuda = DEFAULT_CUDA
            log(f"  Selected: {cuda}")
            return cuda
        elif gpu_info.detected:
            log(f"  {gpu_info.message}")
            return "metal"
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
            return DEFAULT_CUDA
    except Exception:
        pass

    log("  No NVIDIA GPU detected — using CPU")
    return "cpu"


def _release_version() -> str:
    match = re.search(r'^__version__\s*=\s*"([^"]+)"',
                      (_source_dir / "whisperjav" / "__version__.py").read_text(encoding="utf-8"), re.M)
    return match.group(1) if match else "1.10"


def _wheel_fits_this_pc(platform_tag: str) -> bool:
    """Does a wheel's platform tag (win_amd64, manylinux_2_28_x86_64, ...) fit this PC?"""
    machine = platform_module.machine().lower()
    arches = {"amd64": ("x86_64", "amd64"), "x86_64": ("x86_64", "amd64"),
              "arm64": ("arm64", "aarch64"), "aarch64": ("arm64", "aarch64")}.get(machine, (machine,))
    if sys.platform == "win32":
        system = platform_tag.startswith("win")
    elif IS_LINUX:
        system = platform_tag.startswith(("manylinux", "musllinux", "linux"))
    elif sys.platform == "darwin":
        system = platform_tag.startswith("macosx")
        if system and platform_tag.endswith("universal2"):
            return True
    else:
        system = False
    return system and platform_tag.endswith(tuple(f"_{a}" for a in arches))


def lock_has_pytorch_for_this_pc(cuda_version: str) -> bool:
    """D3 (owner W57): does uv.lock hold a PyTorch wheel of this build for this PC?

    Each build's torch comes from its own index (pyproject.toml); the wheels uv.lock
    records under that index are every platform the build can be installed on.
    """
    index = f'source = {{ registry = "https://download.pytorch.org/whl/{LOCKED_EXTRAS[cuda_version]}" }}'
    text = (_source_dir / "uv.lock").read_text(encoding="utf-8")
    for block in text.split("[[package]]"):
        if not block.lstrip().startswith('name = "torch"\n') or index not in block:
            continue
        for wheel in re.findall(r'url = "[^"]+/([^/"]+)\.whl"', block):
            if _wheel_fits_this_pc(wheel.rsplit("-", 1)[-1]):
                return True
    return False


def unsupported_build_message(cuda_version: str) -> str:
    """The reason this PyTorch build cannot be installed on this PC, or "" (D3, owner W57)."""
    version = _release_version()
    if IS_LINUX and cuda_version == "cu130":
        return (f"CUDA 13.0 is not offered on Linux in WhisperJAV {version}. "
                f"Use --cuda cu128 (the Linux default) or --cpu-only.")
    if sys.platform == "darwin" and cuda_version in ("cu130", "cu128"):
        return "macOS has no CUDA builds. Run without --cuda; WhisperJAV installs the Apple build."
    if not lock_has_pytorch_for_this_pc(cuda_version):
        label = {"cu130": "CUDA 13.0", "cu128": "CUDA 12.8"}.get(cuda_version, "CPU")
        return (f"PyTorch has no {label} build for this computer ({sys.platform}, "
                f"{platform_module.machine()}) in WhisperJAV {version}, so it cannot be installed here. "
                f"Nothing was installed.")
    return ""


def wait_for_answer(prompt: str, valid, on_no_console: str) -> str:
    """Ask until one of `valid` is typed. No time limit (owner, W37 (2))."""
    while True:
        try:
            answer = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            log(f"  (no answer could be read; using '{on_no_console}')")
            return on_no_console
        if answer in valid:
            return answer


def gpu_support_gate(gpu_info) -> str:
    """Old card or old driver, before anything is installed (item 2 spec E, F).

    Returns "gpu", "cpu" (the user chose it) or "stop". Texts: the owner's (W43).
    """
    cc = getattr(gpu_info, "compute_capability", None)
    name = gpu_info.name or "Unknown GPU"
    if cc is not None and tuple(cc) < MIN_COMPUTE_CAPABILITY:
        log("")
        log("!" * 60)
        log(f"Your graphics card, {name} (compute capability {cc[0]}.{cc[1]}), is older than "
            "WhisperJAV 1.10 supports. 1.10 needs a GTX 16-series, RTX 20-series or newer card.")
        log("Choose: 1 = my card is identified wrongly (show where to report it, then stop); "
            "2 = install for the CPU only (much slower); 3 = stop.")
        log(f"WhisperJAV 1.9.5 still supports this card: {RELEASE_195_URL}.")
        log("!" * 60)
        answer = wait_for_answer("Type 1, 2 or 3 and press Enter: ", ("1", "2", "3"), "3")
        if answer == "2":
            log("  Installing for the CPU only, as chosen.")
            return "cpu"
        if answer == "1":
            log(f"Please report it here: {ISSUES_URL} (attach install_log.txt).")
        log("The installer will now stop.")
        return "stop"

    if gpu_info.driver_version and tuple(gpu_info.driver_version) < MIN_DRIVER:
        v = gpu_info.driver_version
        log("")
        log("!" * 60)
        log(f"Your NVIDIA driver ({v[0]}.{v[1]:02d}) is older than WhisperJAV 1.10 needs "
            f"({MIN_DRIVER[0]} or newer). "
            f"Update it from {NVIDIA_DRIVER_URL}, restart the PC, and run this installer again. "
            "Nothing has been installed. Press Enter to close.")
        log("!" * 60)
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        return "stop"
    return "gpu"


# =============================================================================
# uv sync wrapper
# =============================================================================

def run_uv_sync(extras: list, cuda_version: str, dev: bool = False,
                target_env: str = None, is_external: bool = False) -> bool:
    """Run ``uv sync --frozen`` with the given extras and the PyTorch build's extra.

    v1.10.0 (item 2f): uv.lock holds every PyTorch build (extras cu130, cu128, cpu), so
    an install takes the locked versions as they are and never rewrites uv.lock. The
    old ``--index pytorch=<url>`` override made uv re-resolve on cu128 and CPU.

    Args:
        extras: List of extra names to install (e.g. ["cli", "gui"]).
        cuda_version: "cpu", "cu130", "cu128", or "metal".
        dev: If True, include dev dependencies.
        target_env: Path to target environment (conda/venv), or None for .venv.
        is_external: If True, the target env is one we didn't create — add --inexact.

    Returns:
        True on success, False on failure.
    """
    import subprocess

    # Build environment with UV_PROJECT_ENVIRONMENT if targeting an active env
    env = dict(os.environ)
    if target_env:
        env["UV_PROJECT_ENVIRONMENT"] = target_env
        log(f"  Installing into active environment: {target_env}")
    else:
        log(f"  No active environment — installing into .venv")

    cmd = _uv_cmd() + ["sync", "--frozen"]

    # --inexact for external envs: don't remove packages we didn't install
    if is_external:
        cmd.append("--inexact")

    # Pin to the running Python interpreter so uv doesn't download a different one.
    # Only when NOT targeting an external env (--python conflicts with env targeting).
    if not target_env:
        cmd.extend(["--python", sys.executable])

    for extra in extras:
        cmd.extend(["--extra", extra])

    # The PyTorch build. macOS ("metal") takes the cpu extra: that index carries
    # the Apple Silicon wheels (pyproject.toml, [tool.uv.sources]).
    cmd.extend(["--extra", LOCKED_EXTRAS[cuda_version]])

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
            env=env,
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
# One-off package installs (cuBLAS 12 for faster-whisper)
# =============================================================================

def run_uv_pip_install(args: list, description: str, allow_fail: bool = False) -> bool:
    """Run uv pip install for one-off package installs.

    Used for packages that are not in pyproject.toml (nvidia-cublas-cu12, which only
    a CUDA 13.0 install on Windows needs).

    Args:
        args: Arguments to pass after 'uv pip install'.
        description: Human-readable description for logging.
        allow_fail: If True, log warning instead of error on failure.

    Returns:
        True on success.
    """
    import subprocess

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
# cuBLAS 12 for faster-whisper on a CUDA 13.0 install (item 2 spec C, D)
# =============================================================================

CUBLAS12_PACKAGE = "nvidia-cublas-cu12==12.8.4.1"


def _run_in_env(python: str, *args: str) -> tuple:
    """Run `python args...` in the target environment; (ok, first output line)."""
    import subprocess
    try:
        r = subprocess.run([python, *args], capture_output=True, text=True, timeout=600,
                           cwd=str(Path.home()))  # not the source folder: test the installed package
    except Exception as e:
        return False, str(e)
    out = (r.stdout or "").strip().splitlines()
    err = (r.stderr or "").strip().splitlines()
    return r.returncode == 0, (out[0] if out else (err[-1] if err else f"exit code {r.returncode}"))


def _write_stack(python: str, stack: str):
    ok, detail = _run_in_env(python, "-c",
                             f"from whisperjav.installer.cublas12 import write_stack; write_stack({stack!r})")
    log(f"  Stack recorded: {stack}" if ok else f"  [WARN] Could not record the stack: {detail}")


def setup_cublas12(target_env: str, extras: list, dev: bool, is_external: bool) -> bool:
    """Windows, CUDA 13.0: place cuBLAS 12 for CTranslate2; fall back to CUDA 12.8.

    Returns False if installing or placing cuBLAS 12 failed (an ordinary install
    failure, not a reason to change the stack) or if the fallback itself failed. Only a
    failed load check moves the PC to CUDA 12.8 (spec D; review 2026-10-14).
    """
    python = _venv_python(target_env)
    log_section("CUDA 12 library for faster-whisper")
    if not run_uv_pip_install(["--python", python, CUBLAS12_PACKAGE],
                              "Install cuBLAS 12 (for faster-whisper next to CUDA 13)"):
        return False
    ok, detail = _run_in_env(python, "-m", "whisperjav.installer.cublas12", "place")
    log(f"  Place cuBLAS 12 in CTranslate2's folder: {detail}")
    if not ok:
        return False
    passed, detail = _run_in_env(python, "-m", "whisperjav.installer.cublas12", "check")
    log(f"  cuBLAS 12 load check (fresh process): {detail}")
    if passed:
        _write_stack(python, "cu130")
        return True

    log(f"  cuBLAS 12 check FAILED: {detail}")
    log("")
    log("!" * 60)
    log("WARNING: This PC could not load the CUDA 12 library (cublas64_12.dll) that Balanced, "
        "Fast, Faster and Kotoba need next to CUDA 13. WhisperJAV will now install its CUDA 12.8 "
        f"build instead of the standard CUDA 13 build. Please report this so it can be fixed: "
        f"{ISSUES_URL} — attach install_log.txt. Press Enter to continue.")
    log("!" * 60)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    if not run_uv_sync(extras, "cu128", dev=dev, target_env=target_env, is_external=is_external):
        return False
    # Found without importing ctranslate2: importing it would load the DLLs being removed.
    _run_in_env(python, "-c",
                "import os, importlib.util; "
                "d = list(importlib.util.find_spec('ctranslate2').submodule_search_locations)[0]; "
                "[os.remove(os.path.join(d, n)) for n in ('cublas64_12.dll', 'cublasLt64_12.dll') "
                "if os.path.exists(os.path.join(d, n))]")
    run_uv_pip_install(["--python", python, "--reinstall", "--no-deps", "ctranslate2==4.8.3"],
                       "Reinstall CTranslate2 as shipped", allow_fail=True)
    import subprocess
    subprocess.run(_uv_cmd() + ["pip", "uninstall", "--python", python, "nvidia-cublas-cu12"],
                   capture_output=True, text=True)
    _write_stack(python, "cu128")
    return True


# =============================================================================
# Verification
# =============================================================================

def _venv_python(target_env: str = None) -> str:
    """Return the Python path for the target environment."""
    if target_env:
        # Active conda or venv
        if sys.platform == "win32":
            py = Path(target_env) / "Scripts" / "python.exe"
        else:
            py = Path(target_env) / "bin" / "python"
        if py.exists():
            return str(py)
    else:
        # .venv created by uv
        if sys.platform == "win32":
            py = _source_dir / ".venv" / "Scripts" / "python.exe"
        else:
            py = _source_dir / ".venv" / "bin" / "python"
        if py.exists():
            return str(py)
    return sys.executable


def _verify_installation(target_env: str = None):
    """Verify WhisperJAV was installed correctly."""
    import subprocess

    python = _venv_python(target_env)
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
            f.write("- Ensure uv, Python 3.13, FFmpeg, and Git are in PATH\n")
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
                        help="CUDA build: cu130 (default) or cu128")
    parser.add_argument("--no-speech-enhancement", action="store_true",
                        help="Skip speech enhancement packages")
    parser.add_argument("--minimal", action="store_true",
                        help="Minimal install (transcription only)")
    parser.add_argument("--dev", action="store_true",
                        help="Install in development/editable mode")
    # llama-cpp local translation was removed in v1.10.0 (REQ2). The flags are still
    # accepted so that existing commands do not stop; hidden from --help.
    parser.add_argument("--local-llm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--local-llm-build", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-local-llm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip preflight checks")
    parser.add_argument("--local", action="store_true",
                        help="Force local .venv even if conda/venv is active")
    parser.add_argument("--force-cuda", type=str, default=None, metavar="VERSION",
                        help="Force CUDA build even if GPU detection fails (cu130 or cu128). "
                             "Useful for Optimus laptops where NVML/nvidia-smi may not detect the GPU.")
    parser.add_argument("--log-file", type=str, default=None, metavar="PATH",
                        help="Custom log file path")
    # Internal flags (used by upgrade.py)
    parser.add_argument("--upgrade", action="store_true",
                        help=argparse.SUPPRESS)
    # Legacy compat flags (still accepted, mapped internally)
    parser.add_argument("--cuda118", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cuda128", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Map legacy flags. cu118 is gone in v1.10.0; --cuda118 gets the CUDA 12.8 build.
    if args.cuda118 and not args.cuda:
        args.cuda = "cu128"
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
    # Environment detection
    # -------------------------------------------------------------------------
    target_env, is_external = _detect_target_env(force_local=args.local)

    # -------------------------------------------------------------------------
    # GPU detection
    # -------------------------------------------------------------------------
    log_section("GPU Detection")
    cuda_version = detect_cuda_version(args)
    if cuda_version == "stop":
        create_failure_file("Stopped before installing: graphics card or driver older than 1.10 supports")
        sys.exit(1)
    # D3 (owner W57): a build 1.10 does not offer on this PC stops here, before anything
    # is installed, instead of failing inside uv sync.
    unsupported = unsupported_build_message(cuda_version)
    if unsupported:
        log(f"  {unsupported}")
        create_failure_file(unsupported)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Determine extras
    # -------------------------------------------------------------------------
    if args.minimal:
        extras = ["cli"]
    else:
        extras = ["cli", "gui", "translate", "huggingface", "qwen",
                   "analysis", "compatibility"]
        if not args.no_speech_enhancement:
            extras.append("enhance")

    log_section("Installation Plan")
    if target_env:
        env_type = "conda" if os.environ.get("CONDA_PREFIX") == target_env else "venv"
        log(f"  Target: {env_type} environment at {target_env}")
    else:
        log(f"  Target: project .venv (local)")
    log(f"  PyTorch: {cuda_version}")
    log(f"  Extras: {', '.join(extras)}")
    log(f"  Speech Enhancement: {'No' if args.no_speech_enhancement or args.minimal else 'Yes'}")
    log(f"  Mode: {'Development' if args.dev else 'Standard'}")
    if is_external:
        log(f"  --inexact: Yes (preserving existing packages)")
    log(f"  Log file: {_LOG_FILE}")

    # -------------------------------------------------------------------------
    # Run uv sync (single command replaces 6-stage pip pipeline)
    # -------------------------------------------------------------------------
    log_section("Installing Dependencies (uv sync)")
    success = run_uv_sync(extras, cuda_version, dev=args.dev,
                          target_env=target_env, is_external=is_external)

    if not success:
        # Retry without enhance (most common failure point)
        if "enhance" in extras:
            log("")
            log("    Retrying without speech enhancement...")
            extras_retry = [e for e in extras if e != "enhance"]
            success = run_uv_sync(extras_retry, cuda_version, dev=args.dev,
                                  target_env=target_env, is_external=is_external)
            if success:
                log("    [OK] Installation succeeded without speech enhancement.")
                # The build actually chosen: without it uv would put PyPI's PyTorch in.
                log(f"    You can install it later: uv sync --frozen --extra all "
                    f"--extra {LOCKED_EXTRAS[cuda_version]}")

    if not success:
        log("")
        log("    INSTALLATION FAILED")
        create_failure_file("uv sync failed — check install_log.txt for details")
        sys.exit(1)

    if args.local_llm or args.local_llm_build:
        log("")
        log("  llama-cpp local translation was removed in WhisperJAV 1.10; --local-llm does "
            "nothing. Use Ollama for local translation.")

    # -------------------------------------------------------------------------
    # cuBLAS 12 for faster-whisper (Windows, CUDA 13.0); stack record
    # -------------------------------------------------------------------------
    if cuda_version == "cu130" and sys.platform == "win32":
        if not setup_cublas12(target_env, extras, args.dev, is_external):
            log("    INSTALLATION FAILED (cuBLAS 12 for faster-whisper)")
            create_failure_file("cuBLAS 12 install, placing or CUDA 12.8 fallback failed — check install_log.txt")
            sys.exit(1)
    elif cuda_version in ("cu130", "cu128"):
        _write_stack(_venv_python(target_env), cuda_version)
    else:
        _write_stack(_venv_python(target_env), "cpu")

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------
    log_section("Verifying Installation")
    _verify_installation(target_env)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    log_section("Installation Complete!")
    log("")
    if target_env:
        env_type = "conda" if os.environ.get("CONDA_PREFIX") == target_env else "venv"
        log(f"  Packages installed into your {env_type} environment at {target_env}")
        log(f"  Run `whisperjav --help` to verify.")
    else:
        venv_dir = _source_dir / ".venv"
        if venv_dir.exists():
            log("  Packages installed into .venv.")
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
    # Check for GUI launcher (created by platform shell wrapper after install.py returns)
    if sys.platform == "darwin":
        launcher = _source_dir / "WhisperJAV.command"
        if launcher.exists():
            log("  To launch the GUI:")
            log("    Double-click WhisperJAV.command in Finder")
            log("    (Tip: drag it to your Dock for easy access)")
            log("")
    elif sys.platform == "win32":
        launcher = _source_dir / "WhisperJAV.bat"
        if launcher.exists():
            log("  To launch the GUI:")
            log("    Double-click WhisperJAV.bat")
            log("")
    else:
        launcher = _source_dir / "WhisperJAV.sh"
        if launcher.exists():
            log("  To launch the GUI:")
            log("    Run: ./WhisperJAV.sh")
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
