"""
WhisperJAV v1.7.3 Post-Install Script
======================================

This script runs after the conda environment is created and:
1. Performs comprehensive preflight checks (disk space, network, WebView2)
2. Detects NVIDIA GPU and installs appropriate PyTorch build
3. Offers CPU-only fallback for systems without NVIDIA GPU
4. Installs all Python dependencies from requirements_v1.7.3.txt
5. Installs WhisperJAV from GitHub
6. Creates desktop shortcut
7. Provides detailed installation summary

All output is logged to install_log_v1.7.3.txt
"""

import os
import sys
import re
import time
import shlex
import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Tuple

try:
    import pynvml  # type: ignore
    PYNVML_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore
    PYNVML_AVAILABLE = False

LOG_FILE = os.path.join(sys.prefix, "install_log_v1.7.3.txt")

MIN_TORCH_CUDA_VERSION: Tuple[int, int, int] = (11, 8, 0)
CPU_FALLBACK_COMMAND = "pip3 install torch torchaudio"


class DriverMatrixEntry(NamedTuple):
    min_driver: Tuple[int, int, int]
    cuda_version: str
    pip_command: str


class DriverInfo(NamedTuple):
    driver_version: Optional[Tuple[int, int, int]]
    gpu_name: Optional[str]
    source: str


class TorchInstallPlan(NamedTuple):
    pip_args: list[str]
    pip_command: str
    description: str
    uses_gpu: bool
    target_label: str
    driver_requirement: Optional[Tuple[int, int, int]]
    driver_detected: Optional[Tuple[int, int, int]]
    gpu_name: Optional[str]
    reason: str


# IMPORTANT: ctranslate2 (faster-whisper) does NOT support CUDA 13.
# Even for CUDA 13+ drivers, we must use cu128 wheels.
# PyTorch 2.9.0 is the maximum version available for cu128.
TORCH_DRIVER_MATRIX: Sequence[DriverMatrixEntry] = (
    # CUDA 13.x drivers (580+) - use cu128 due to ctranslate2 compatibility
    DriverMatrixEntry(
        (580, 65, 0),
        "CUDA 12.8 (cu128)",  # Label reflects actual wheels, not driver CUDA
        "pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128"
    ),
    # CUDA 12.8 drivers (570+)
    DriverMatrixEntry(
        (570, 65, 0),
        "CUDA 12.8 (cu128)",
        "pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128"
    ),
    # CUDA 12.6 drivers (560+)
    DriverMatrixEntry(
        (560, 76, 0),
        "CUDA 12.6 (cu126)",
        "pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126"
    ),
    # CUDA 12.4 drivers (551+)
    DriverMatrixEntry(
        (551, 61, 0),
        "CUDA 12.4 (cu124)",
        "pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
    ),
    # CUDA 12.1 drivers (531+)
    DriverMatrixEntry(
        (531, 14, 0),
        "CUDA 12.1 (cu121)",
        "pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121"
    ),
    # CUDA 11.8 drivers (520+) - oldest supported
    DriverMatrixEntry(
        (520, 6, 0),
        "CUDA 11.8 (cu118)",
        "pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118"
    ),
)


def log(message: str):
    """Log message to console and file with timestamp"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def log_section(title: str):
    """Log a section header for better readability"""
    log("\n" + "=" * 80)
    log(f"  {title}")
    log("=" * 80)


def format_version_tuple(version: Optional[Tuple[int, int, int]]) -> str:
    if not version:
        return "unknown"
    return ".".join(str(part) for part in version)


def parse_version_string(version_str: str) -> Optional[Tuple[int, int, int]]:
    if not version_str:
        return None
    digits = [int(piece) for piece in re.split(r"[^0-9]", version_str) if piece.isdigit()]
    if not digits:
        return None
    while len(digits) < 3:
        digits.append(0)
    return tuple(digits[:3])  # type: ignore[return-value]


def ensure_pynvml_loaded() -> bool:
    """Attempt to import or install pynvml for driver detection."""
    global PYNVML_AVAILABLE, pynvml

    if PYNVML_AVAILABLE:
        return True

    try:
        import importlib

        pynvml = importlib.import_module("pynvml")  # type: ignore
        PYNVML_AVAILABLE = True
        return True
    except ImportError:
        log("Installing nvidia-ml-py for NVIDIA driver detection...")
        if run_pip([
            "install",
            "nvidia-ml-py",
            "--progress-bar",
            "on"
        ], "Install NVML helper"):
            try:
                import importlib

                pynvml = importlib.import_module("pynvml")  # type: ignore
                PYNVML_AVAILABLE = True
                return True
            except ImportError:
                log("WARNING: pynvml module still unavailable after installation attempt.")
    return False


def detect_nvidia_driver() -> DriverInfo:
    """Detect NVIDIA driver version using nvidia-smi or NVML."""
    smi_cmd = [
        "nvidia-smi",
        "--query-gpu=driver_version,name",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            smi_cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            timeout=10
        )
        line = result.stdout.strip().splitlines()[0]
        if "," in line:
            driver_str, gpu_name = line.split(",", 1)
        else:
            driver_str, gpu_name = line, "Unknown GPU"
        driver_tuple = parse_version_string(driver_str.strip())
        if driver_tuple:
            return DriverInfo(driver_tuple, gpu_name.strip(), "nvidia-smi")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
        pass

    if ensure_pynvml_loaded():
        try:
            pynvml.nvmlInit()  # type: ignore[attr-defined]
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # type: ignore[attr-defined]
            driver_str = pynvml.nvmlSystemGetDriverVersion()  # type: ignore[attr-defined]
            gpu_name = pynvml.nvmlDeviceGetName(handle)  # type: ignore[attr-defined]
            if isinstance(driver_str, bytes):
                driver_str = driver_str.decode('utf-8')
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            driver_tuple = parse_version_string(str(driver_str))
            if driver_tuple:
                return DriverInfo(driver_tuple, str(gpu_name), "pynvml")
        except Exception as exc:  # pragma: no cover - hardware specific
            log(f"WARNING: NVML detection failed: {exc}")
        finally:
            try:
                pynvml.nvmlShutdown()  # type: ignore[attr-defined]
            except Exception:
                pass

    return DriverInfo(None, None, "none")


def get_installed_torch_info() -> Optional[dict]:
    """
    Get information about the currently installed PyTorch.

    Returns:
        dict with 'torch_version', 'torchaudio_version', 'cuda_version', 'cuda_available'
        or None if PyTorch is not installed.
    """
    try:
        import torch
    except ImportError:
        return None

    info = {
        'torch_version': torch.__version__,
        'torchaudio_version': None,
        'cuda_version': None,
        'cuda_available': torch.cuda.is_available(),
    }

    try:
        import torchaudio
        info['torchaudio_version'] = torchaudio.__version__
    except ImportError:
        pass

    if info['cuda_available']:
        cuda_str = getattr(torch.version, "cuda", "") or ""
        info['cuda_version'] = parse_version_string(cuda_str)

    return info


def extract_versions_from_plan(plan: 'TorchInstallPlan') -> dict:
    """
    Extract PyTorch version and CUDA tag from install plan.

    Returns:
        dict with 'torch_version' (tuple or None), 'cuda_tag' (str or None), 'cuda_version' (tuple or None)
    """
    import re
    result = {
        'torch_version': None,
        'cuda_tag': None,
        'cuda_version': None,
    }

    if not plan.uses_gpu:
        return result

    # Extract CUDA tag from URL (e.g., ".../whl/cu128")
    cuda_match = re.search(r'/whl/(cu\d+)', plan.pip_command)
    if cuda_match:
        result['cuda_tag'] = cuda_match.group(1)
        result['cuda_version'] = cuda_tag_to_version(result['cuda_tag'])

    # Extract PyTorch version if specified (e.g., "torch==2.6.0")
    torch_match = re.search(r'torch==(\d+\.\d+\.\d+)', plan.pip_command)
    if torch_match:
        result['torch_version'] = parse_version_string(torch_match.group(1))

    return result


def cuda_tag_to_version(cuda_tag: str) -> Optional[Tuple[int, int, int]]:
    """
    Convert CUDA tag (e.g., 'cu121', 'cu128') to version tuple.

    Examples:
        'cu118' -> (11, 8, 0)
        'cu121' -> (12, 1, 0)
        'cu124' -> (12, 4, 0)
        'cu126' -> (12, 6, 0)
        'cu128' -> (12, 8, 0)
    """
    if not cuda_tag or not cuda_tag.startswith('cu'):
        return None

    digits = cuda_tag[2:]  # Remove 'cu' prefix
    if len(digits) == 3:  # cu121 -> 12.1, cu128 -> 12.8
        major = int(digits[:2])
        minor = int(digits[2:])
        return (major, minor, 0)
    elif len(digits) == 2:  # cu11 -> 11.0 (unlikely but handle)
        return (int(digits), 0, 0)
    return None


def verify_existing_torch_matches_plan(plan: 'TorchInstallPlan') -> bool:
    """
    Check if installed PyTorch matches what the install plan would install.

    Checks BOTH PyTorch version AND CUDA version:
    - If installed PyTorch >= plan's PyTorch (or plan has no version = latest)
    - AND installed CUDA >= plan's CUDA
    - THEN skip installation

    This enables smart skipping for upgrade users who already have the correct
    PyTorch version installed, avoiding unnecessary reinstallation.

    Returns:
        True if installed PyTorch satisfies the plan (can skip installation).
        False if installation is needed.
    """
    log("Checking for existing PyTorch installation...")

    installed = get_installed_torch_info()
    if not installed:
        log("PyTorch not installed. Installation required.")
        return False

    log(f"Found PyTorch {installed['torch_version']}")
    if installed['torchaudio_version']:
        log(f"Found torchaudio {installed['torchaudio_version']}")
    else:
        log("torchaudio not installed.")

    # Case 1: Plan is CPU-only
    if not plan.uses_gpu:
        if installed['cuda_available']:
            log("Plan is CPU-only but CUDA PyTorch is installed. Keeping existing CUDA version.")
            return True  # Keep the better version
        else:
            log("CPU-only PyTorch already installed. Skipping reinstall.")
            return True

    # Case 2: Plan requires GPU but installed is CPU-only
    if not installed['cuda_available']:
        log("Plan requires CUDA but installed PyTorch is CPU-only. Installation required.")
        return False

    # Case 3: Both are CUDA - compare BOTH PyTorch and CUDA versions
    installed_cuda = installed['cuda_version']
    installed_torch = parse_version_string(installed['torch_version'].split('+')[0])  # Remove +cu128 suffix

    plan_info = extract_versions_from_plan(plan)
    plan_cuda = plan_info['cuda_version']
    plan_torch = plan_info['torch_version']

    log(f"Installed: PyTorch {format_version_tuple(installed_torch)}, CUDA {format_version_tuple(installed_cuda)}")
    log(f"Plan targets: PyTorch {format_version_tuple(plan_torch) if plan_torch else 'latest'}, CUDA {plan_info['cuda_tag']} ({format_version_tuple(plan_cuda)})")

    # Validate we can determine installed versions
    if not installed_cuda:
        log("Cannot determine installed CUDA version. Installation required.")
        return False

    if not installed_torch:
        log("Cannot determine installed PyTorch version. Installation required.")
        return False

    if not plan_cuda:
        log("Cannot determine plan's target CUDA version. Will proceed with installation.")
        return False

    # Compare CUDA versions (major.minor)
    installed_cuda_mm = (installed_cuda[0], installed_cuda[1])
    plan_cuda_mm = (plan_cuda[0], plan_cuda[1])

    cuda_ok = installed_cuda_mm >= plan_cuda_mm

    # Compare PyTorch versions
    # If plan has no specific version (uses latest), any recent version is OK
    if plan_torch:
        torch_ok = installed_torch >= plan_torch
    else:
        # Plan uses latest - if user has 2.0+, consider it OK
        torch_ok = installed_torch >= (2, 0, 0)

    log(f"CUDA check: {installed_cuda_mm} >= {plan_cuda_mm} = {cuda_ok}")
    log(f"PyTorch check: {format_version_tuple(installed_torch)} >= {format_version_tuple(plan_torch) if plan_torch else '2.0.0'} = {torch_ok}")

    if cuda_ok and torch_ok:
        log("Both PyTorch and CUDA versions satisfy requirements. Skipping reinstall.")
        return True

    if not cuda_ok:
        log(f"Installed CUDA {installed_cuda[0]}.{installed_cuda[1]} < plan's {plan_cuda[0]}.{plan_cuda[1]}.")
    if not torch_ok:
        log(f"Installed PyTorch {format_version_tuple(installed_torch)} < plan's {format_version_tuple(plan_torch)}.")

    log("Upgrade recommended.")
    return False


def normalize_pip_command(command: str) -> list[str]:
    parts = shlex.split(command)
    if len(parts) < 2:
        raise ValueError(f"Invalid pip command: {command}")
    head = parts[0].lower()
    if head not in {"pip", "pip3"}:
        raise ValueError(f"Unsupported pip executable '{parts[0]}' in command: {command}")
    return parts[1:]


def select_torch_install_plan(driver_info: DriverInfo) -> TorchInstallPlan:
    if driver_info.driver_version:
        for entry in TORCH_DRIVER_MATRIX:
            if driver_info.driver_version >= entry.min_driver:
                return TorchInstallPlan(
                    pip_args=normalize_pip_command(entry.pip_command),
                    pip_command=entry.pip_command,
                    description=f"Install PyTorch/Torchaudio ({entry.cuda_version})",
                    uses_gpu=True,
                    target_label=entry.cuda_version,
                    driver_requirement=entry.min_driver,
                    driver_detected=driver_info.driver_version,
                    gpu_name=driver_info.gpu_name,
                    reason=""
                )

        min_required = TORCH_DRIVER_MATRIX[-1].min_driver
        reason = (
            f"Detected driver {format_version_tuple(driver_info.driver_version)} "
            f"on {driver_info.gpu_name or 'GPU'} is below minimum "
            f"{format_version_tuple(min_required)} for CUDA wheels."
        )
    else:
        reason = "No NVIDIA GPU or driver detected."

    return TorchInstallPlan(
        pip_args=normalize_pip_command(CPU_FALLBACK_COMMAND),
        pip_command=CPU_FALLBACK_COMMAND,
        description="Install PyTorch/Torchaudio (CPU-only)",
        uses_gpu=False,
        target_label="CPU",
        driver_requirement=None,
        driver_detected=driver_info.driver_version,
        gpu_name=driver_info.gpu_name,
        reason=reason
    )


def install_pytorch_and_torchaudio(plan: TorchInstallPlan) -> bool:
    log(f"Selected pip command: {plan.pip_command}")
    pip_preview = f"{sys.executable} -m pip {' '.join(plan.pip_args)}"
    log(f"Executing pip command: {pip_preview}")
    return run_pip(plan.pip_args, plan.description)


def timed_input(prompt: str, timeout_seconds: int, default_response: str = "") -> str:
    """
    Get user input with timeout, returning default if timeout expires.
    Allows unattended installation while preserving interactive option.

    Args:
        prompt: The prompt to display to user
        timeout_seconds: Seconds to wait before using default
        default_response: Value to return if timeout expires

    Returns:
        User input or default_response if timeout
    """
    import threading

    print(prompt, end='', flush=True)

    # Use threading for cross-platform timeout support
    result = [default_response]  # Mutable container for thread communication

    def get_input():
        try:
            result[0] = input()
        except (EOFError, KeyboardInterrupt):
            result[0] = default_response

    input_thread = threading.Thread(target=get_input, daemon=True)
    input_thread.start()
    input_thread.join(timeout=timeout_seconds)

    if input_thread.is_alive():
        # Timeout occurred
        print(f"\n[Auto-continuing after {timeout_seconds}s timeout - using default: '{default_response}']")
        return default_response
    else:
        return result[0]


def check_disk_space(min_gb: int = 8) -> bool:
    """Check if sufficient disk space is available"""
    try:
        total, used, free = shutil.disk_usage(sys.prefix)
        free_gb = free / (1024**3)
        log(f"Disk free space: {free_gb:.1f} GB (minimum required: {min_gb} GB)")
        if free_gb < min_gb:
            log("ERROR: Not enough free disk space for installation.")
            log(f"       Please free up at least {min_gb} GB and retry.")
            return False
        return True
    except Exception as e:
        log(f"WARNING: Could not determine disk space: {e}")
        return True  # Non-fatal, proceed with caution


def check_network(timeout: int = 10) -> bool:
    """Check if network connectivity to PyPI is available"""
    log("Checking network connectivity to PyPI...")
    try:
        import urllib.request
        urllib.request.urlopen("https://pypi.org", timeout=timeout)
        log("Network check: OK")
        return True
    except Exception as e:
        log(f"ERROR: Network check failed: {e}")
        log("       Internet connection is required for downloading dependencies.")
        return False


def check_webview2_windows() -> bool:
    """Check if Microsoft Edge WebView2 runtime is installed (Windows only)"""
    import platform
    if platform.system() != 'Windows':
        return True  # Only needed on Windows

    log("Checking for Microsoft Edge WebView2 runtime...")
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
                log("WebView2 runtime: Detected")
                return True
            except FileNotFoundError:
                continue
        log("WebView2 runtime: NOT DETECTED")
        return False
    except Exception as e:
        log(f"WARNING: Could not check WebView2 status: {e}")
        return True  # Assume OK if can't check


def prompt_webview2_install():
    """Prompt user to install WebView2 and open download page"""
    log("\n" + "!" * 80)
    log("  IMPORTANT: WebView2 Runtime Required for GUI")
    log("!" * 80)
    log("")
    log("WhisperJAV uses Microsoft Edge WebView2 for its modern web-based interface.")
    log("WebView2 is not currently installed on this system.")
    log("")
    log("The installer will now open your browser to download WebView2.")
    log("You have 30 seconds to install it, or the installer will continue automatically.")
    log("")

    download_url = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"
    try:
        import webbrowser
        webbrowser.open(download_url)
        log(f"Opening: {download_url}")
    except Exception:
        log(f"Please manually download from: {download_url}")

    log("")
    timed_input("Press Enter after installing WebView2 (auto-continues in 30s): ", 30, "")

    # Re-check after user confirms
    if check_webview2_windows():
        log("WebView2 detected! Installation will continue.")
    else:
        log("WARNING: WebView2 still not detected. You can install it later.")
        log("         The application will not launch without WebView2.")


def check_cuda_driver() -> DriverInfo:
    """Log and return detected NVIDIA driver information."""
    log("Checking for NVIDIA GPU and driver version...")
    driver_info = detect_nvidia_driver()

    if driver_info.driver_version:
        log(
            f"Detected NVIDIA driver {format_version_tuple(driver_info.driver_version)} "
            f"for GPU {driver_info.gpu_name or 'Unknown GPU'} "
            f"via {driver_info.source}."
        )
    else:
        log("No NVIDIA GPU detected or driver information unavailable.")

    return driver_info


def install_pytorch(driver_info: Optional[DriverInfo] = None) -> bool:
    """
    Install PyTorch/Torchaudio with best available acceleration.

    Smart detection for upgrade users:
    - Detects currently installed PyTorch version and CUDA
    - Compares against what would be installed based on driver
    - Skips installation if existing version is compatible
    - Only installs/upgrades when necessary
    """
    log_section("PyTorch Installation")

    # First, detect driver and determine what we WOULD install
    driver_info = driver_info or detect_nvidia_driver()
    plan = select_torch_install_plan(driver_info)

    log(f"Installation plan: {plan.description}")
    if plan.uses_gpu:
        log(f"Target: {plan.target_label} for {plan.gpu_name or 'GPU'}")
    else:
        log(f"Target: CPU-only ({plan.reason})")

    # Check if existing installation satisfies the plan
    if verify_existing_torch_matches_plan(plan):
        log("")
        log("=" * 60)
        log("  Existing PyTorch installation is compatible!")
        log("  Skipping PyTorch reinstallation (saves time & bandwidth)")
        log("=" * 60)
        return True

    # Need to install - confirm with user for CPU-only
    if plan.uses_gpu:
        log(
            f"Preparing to install CUDA-enabled PyTorch ({plan.target_label}) "
            f"for GPU {plan.gpu_name or 'Unknown GPU'} with driver "
            f"{format_version_tuple(plan.driver_detected)}"
        )
    else:
        log("\n" + "!" * 80)
        log("  GPU ACCELERATION NOT AVAILABLE")
        log("!" * 80)
        if plan.reason:
            log(plan.reason)
        log("\nWithout CUDA this installer will configure CPU-only PyTorch.")
        log("Processing will be significantly slower (6-10x).")
        response = timed_input("Install CPU-only PyTorch? (Y/n): ", 20, "y").strip().lower()
        if response != 'y':
            log("User declined CPU-only PyTorch installation.")
            return False

    if not install_pytorch_and_torchaudio(plan):
        return False

    try:
        import torch

        log(f"PyTorch {torch.__version__} installed successfully!")
        if torch.cuda.is_available():
            log(f"CUDA acceleration: ENABLED (devices: {torch.cuda.device_count()})")
        else:
            log("CUDA acceleration: DISABLED (CPU-only mode)")
    except Exception as exc:
        log(f"WARNING: PyTorch installed but verification failed: {exc}")

    return True


def run_pip(args: list, description: str, retries: int = 3) -> bool:
    """
    Run pip command with retries

    Args:
        args: Pip arguments (e.g., ["install", "package"])
        description: Human-readable description for logging
        retries: Number of retry attempts (default 3)

    Returns:
        True if successful, False otherwise
    """
    log(f"Starting: {description}")
    pip_cmd = [os.path.join(sys.prefix, 'python.exe'), '-m', 'pip'] + args

    for attempt in range(retries):
        log(f"Attempt {attempt+1}/{retries}: {' '.join(pip_cmd)}")
        try:
            result = subprocess.run(
                pip_cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                timeout=1800  # 30 minute timeout for large downloads
            )
            if result.stdout:
                # Only log last 20 lines to avoid clutter
                lines = result.stdout.strip().split('\n')
                for line in lines[-20:]:
                    log(f"  {line}")
            log(f"SUCCESS: {description}")
            return True

        except subprocess.TimeoutExpired:
            log(f"ERROR: {description} timed out (30 minutes)")
            if attempt < retries - 1:
                log("Retrying in 10 seconds...")
                time.sleep(10)

        except subprocess.CalledProcessError as e:
            log(f"ERROR: {description} failed (rc={e.returncode})")
            if e.stdout:
                lines = e.stdout.strip().split('\n')
                for line in lines[-20:]:
                    log(f"  {line}")
            if e.stderr:
                lines = e.stderr.strip().split('\n')
                for line in lines[-20:]:
                    log(f"  {line}")
            if attempt < retries - 1:
                log("Retrying in 10 seconds...")
                time.sleep(10)

        except Exception as e:
            log(f"ERROR: Unexpected error during pip execution: {e}")
            if attempt < retries - 1:
                log("Retrying in 10 seconds...")
                time.sleep(10)

    log(f"FATAL: {description} failed after {retries} attempts")
    return False


def create_failure_file(error_message: str):
    """Create a failure marker file with troubleshooting info"""
    failure_file = os.path.join(sys.prefix, "INSTALLATION_FAILED_v1.7.3.txt")
    try:
        with open(failure_file, "w", encoding="utf-8") as f:
            f.write("WhisperJAV v1.7.3 Installation Failed\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Error: {error_message}\n\n")
            f.write("Manual Cleanup Steps:\n")
            f.write(f"1. Delete installation directory: {sys.prefix}\n")
            f.write("2. Delete desktop shortcut: WhisperJAV v1.7.3.lnk\n")
            f.write(f"3. Check install_log_v1.7.3.txt for details\n\n")
            f.write("Common Solutions:\n")
            f.write("- Out of disk space: Free up 8GB and retry\n")
            f.write("- Network error: Check internet connection and firewall\n")
            f.write("- CUDA error: Update NVIDIA drivers or try CPU mode\n")
            f.write("- WebView2 missing: Install from https://go.microsoft.com/fwlink/p/?LinkId=2124703\n\n")
            f.write("Support: https://github.com/meizhong986/WhisperJAV/issues\n")
        log(f"Failure details written to: {failure_file}")
    except Exception:
        pass


def print_installation_summary(install_start_time: float):
    """Print a comprehensive installation summary"""
    install_duration = int(time.time() - install_start_time)
    minutes = install_duration // 60
    seconds = install_duration % 60

    log("\n\n")
    log("=" * 80)
    log(" " * 20 + "WhisperJAV v1.7.3 Installation Complete!")
    log("=" * 80)
    log("")
    log(f"Installation Summary:")
    log(f"  ✓ Installation directory: {sys.prefix}")
    log(f"  ✓ Python version: {sys.version.split()[0]}")

    try:
        import torch
        if torch.cuda.is_available():
            log(f"  ✓ PyTorch: {torch.__version__} with CUDA {torch.version.cuda}")
            log(f"  ✓ GPU acceleration: ENABLED ({torch.cuda.get_device_name(0)})")
        else:
            log(f"  ✓ PyTorch: {torch.__version__} (CPU-only mode)")
            log(f"  ⚠ GPU acceleration: DISABLED (processing will be slower)")
    except Exception:
        log(f"  ? PyTorch: Status unknown")

    try:
        import numpy as np
        log(f"  ✓ NumPy: {np.__version__}")
    except Exception:
        log(f"  ? NumPy: Status unknown")

    if check_webview2_windows():
        log(f"  ✓ WebView2 runtime: Detected")
    else:
        log(f"  ⚠ WebView2 runtime: NOT DETECTED (GUI will not work)")
        log(f"    Install from: https://go.microsoft.com/fwlink/p/?LinkId=2124703")

    log(f"  ✓ Desktop shortcut: Created")
    log(f"  ✓ Installation time: {minutes}m {seconds}s")
    log("")
    log("New in v1.7.3:")
    log("  ✓ Speech enhancement backends (ZipEnhancer, ClearVoice, BS-RoFormer)")
    log("")
    log("Next Steps:")
    log("  1. Launch WhisperJAV from the desktop shortcut")
    log("  2. On first run, AI models will download (~3GB, 5-10 minutes)")
    log("  3. Select your video files and start processing!")
    log("")
    log(f"Logs saved to: {LOG_FILE}")
    log("=" * 80)
    log("")


def embed_icon_in_exe(exe_path: str, icon_path: str) -> bool:
    """
    Embed icon into executable file using LIEF library.

    Args:
        exe_path: Path to the .exe file
        icon_path: Path to the .ico file

    Returns:
        True if successful, False otherwise
    """
    try:
        log(f"Embedding icon into executable...")

        # Try to import lief, install if not available
        try:
            import lief
        except ImportError:
            log(f"Installing LIEF library for icon embedding...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "lief"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode != 0:
                log(f"INFO: Could not install LIEF library")
                return False
            import lief

        # Load the executable
        binary = lief.parse(exe_path)
        if not binary:
            log(f"INFO: Could not parse executable with LIEF")
            return False

        # Load and add the icon
        icon_manager = lief.PE.ResourcesManager.from_file(exe_path)
        icon_manager.change_icon(icon_path)

        # Save the modified executable
        builder = lief.PE.Builder(binary)
        builder.build_resources(True)
        builder.build()
        builder.write(exe_path)

        log(f"✓ Icon embedded successfully")
        return True

    except Exception as e:
        log(f"INFO: Could not embed icon: {e}")
        log(f"      The executable will use default Python icon")
        log(f"      The desktop shortcut will still display the correct icon")
        return False


def copy_launcher_to_root() -> str:
    """
    Copy Scripts/whisperjav-gui.exe to installation root as WhisperJAV-GUI.exe
    and embed the whisperjav icon into it.

    Returns:
        Path to copied .exe or None if failed
    """
    scripts_exe = os.path.join(sys.prefix, "Scripts", "whisperjav-gui.exe")
    root_exe = os.path.join(sys.prefix, "WhisperJAV-GUI.exe")
    icon_file = os.path.join(sys.prefix, "whisperjav_icon.ico")

    if not os.path.exists(scripts_exe):
        log(f"INFO: Scripts launcher not found: {scripts_exe}")
        log(f"      This is normal - the shortcut will use pythonw.exe instead")
        log(f"      (Both methods work equally well)")
        return None

    try:
        log(f"Copying launcher to root directory...")
        shutil.copy2(scripts_exe, root_exe)

        if os.path.exists(root_exe):
            log(f"✓ Launcher created: WhisperJAV-GUI.exe")
            log(f"  Users can double-click this file to launch the GUI")

            # Try to embed icon if icon file exists
            if os.path.exists(icon_file):
                embed_icon_in_exe(root_exe, icon_file)
            else:
                log(f"INFO: Icon file not found: {icon_file}")
                log(f"      Desktop shortcut will still show correct icon")

            return root_exe
        else:
            log(f"INFO: Could not create launcher in root (will use pythonw.exe fallback)")
            return None

    except Exception as e:
        log(f"INFO: Could not copy launcher: {e}")
        log(f"      The shortcut will use pythonw.exe fallback instead")
        return None


def main() -> int:
    """Main installation workflow"""
    install_start_time = time.time()

    log_section("WhisperJAV v1.7.3 Post-Install Started")
    log(f"Installation prefix: {sys.prefix}")
    log(f"Python executable: {sys.executable}")
    log(f"Python version: {sys.version}")

    # === Phase 1: Preflight Checks ===
    log_section("Phase 1: Preflight Checks")

    if not check_disk_space(8):
        create_failure_file("Insufficient disk space (8GB required)")
        return 1

    if not check_network():
        create_failure_file("Network connectivity required")
        return 1

    # Check WebView2 (non-fatal, but prompt user)
    if not check_webview2_windows():
        prompt_webview2_install()

    # === Phase 2: GPU and CUDA Detection ===
    log_section("Phase 2: GPU and CUDA Detection")
    driver_info = check_cuda_driver()

    # === Phase 3: PyTorch Installation ===
    log_section("Phase 3: PyTorch Installation")
    if not install_pytorch(driver_info):
        create_failure_file("PyTorch installation failed")
        return 1

    # === Phase 4: Python Dependencies ===
    log_section("Phase 4: Python Dependencies Installation")

    req_path = os.path.join(sys.prefix, "requirements_v1.7.3.txt")
    constraints_path = os.path.join(sys.prefix, "constraints_v1.7.3.txt")

    if not os.path.exists(req_path):
        log(f"ERROR: requirements_v1.7.3.txt not found at {req_path}")
        create_failure_file(f"Missing requirements file: {req_path}")
        return 1

    # === Phase 4a: Install git-based packages separately ===
    # These packages are installed first to avoid pip resolution complexity.
    # Installing them individually prevents combinatorial explosion in dependency resolution.
    log_section("Phase 4a: Git-based Packages (installed separately)")

    git_packages = [
        ("ffmpeg-python", "git+https://github.com/kkroening/ffmpeg-python.git"),
        ("openai-whisper", "git+https://github.com/openai/whisper@main"),
        ("stable-ts", "git+https://github.com/meizhong986/stable-ts-fix-setup.git@main"),
    ]

    for pkg_name, pkg_url in git_packages:
        log(f"Installing {pkg_name} from GitHub...")
        if not run_pip(
            ["install", pkg_url, "--progress-bar", "on"],
            f"Install {pkg_name}"
        ):
            log(f"WARNING: {pkg_name} installation failed, continuing...")
            # Non-fatal for ffmpeg-python, but whisper packages are important
            if pkg_name in ("openai-whisper", "stable-ts"):
                log(f"ERROR: {pkg_name} is required. Installation may be incomplete.")

    # === Phase 4b: Install clearvoice first (with its stated deps) ===
    # clearvoice's metadata declares numpy<2.0 and librosa==0.10.0, but it actually works
    # with numpy 2.0+ and librosa 0.11.0+ (tested). We install it first to get all its
    # dependencies, then upgrade numpy/librosa to the versions we want.
    log_section("Phase 4b: ClearVoice (install with all dependencies)")
    log("Installing clearvoice (this brings in numpy<2.0 temporarily)...")
    if not run_pip(
        ["install", "clearvoice", "--progress-bar", "on"],
        "Install clearvoice"
    ):
        log("WARNING: clearvoice installation failed, continuing...")

    # === Phase 4c: Fix package versions ===
    # Some packages get downgraded during installation due to dependency conflicts.
    # This phase ensures we have the correct versions for WhisperJAV.
    log_section("Phase 4c: Fix Package Versions")

    # Packages to upgrade to latest (clearvoice/modelscope install older versions)
    upgrade_packages = [
        ("numpy>=2.0", "NumPy 2.x for modelscope/zipenhancer"),
        ("librosa>=0.11.0", "librosa 0.11+ for NumPy 2.x support"),
        ("fsspec", "latest fsspec"),
        ("huggingface-hub", "latest huggingface-hub"),
        ("opencv-python", "latest opencv-python"),
    ]

    log("Upgrading packages to target versions...")
    for pkg_spec, description in upgrade_packages:
        log(f"  - {description}")

    upgrade_list = [pkg for pkg, _ in upgrade_packages]
    if not run_pip(
        ["install"] + upgrade_list + ["--upgrade", "--progress-bar", "on"],
        "Upgrade packages to target versions"
    ):
        log("WARNING: Package upgrade failed, continuing...")

    # Pin datasets to 2.18.0 (modelscope requires this exact version)
    # datasets 4.x removes HubDatasetModuleFactoryWithoutScript which modelscope needs
    log("Pinning datasets==2.18.0 (required by modelscope)...")
    if not run_pip(
        ["install", "datasets==2.18.0", "--progress-bar", "on"],
        "Pin datasets version for modelscope"
    ):
        log("WARNING: datasets pinning failed - modelscope may not work correctly")

    # === Phase 4d: Main requirements ===
    log_section("Phase 4d: PyPI Dependencies")
    log(f"Installing dependencies from: {req_path}")
    log("This will download ~500MB of packages. Please wait...")
    log("(v1.7.3 includes speech enhancement backends)")

    # Use constraints file if available to protect PyTorch version
    if os.path.exists(constraints_path):
        log(f"Using constraints file to protect PyTorch: {constraints_path}")
        if not run_pip(
            ["install", "-c", constraints_path, "-r", req_path, "--progress-bar", "on"],
            "Install Python dependencies (with constraints)"
        ):
            create_failure_file("Dependencies installation failed")
            return 1
    else:
        if not run_pip(
            ["install", "-r", req_path, "--progress-bar", "on"],
            "Install Python dependencies"
        ):
            create_failure_file("Dependencies installation failed")
            return 1

    # === Phase 5: WhisperJAV Application ===
    log_section("Phase 5: WhisperJAV Application Installation")

    # Find WhisperJAV wheel in installation directory
    import glob
    wheel_pattern = os.path.join(sys.prefix, "whisperjav-*.whl")
    wheels = glob.glob(wheel_pattern)

    if not wheels:
        log(f"ERROR: No WhisperJAV wheel found matching: {wheel_pattern}")
        log("ERROR: The installer package may be corrupted or incomplete")
        create_failure_file(f"Missing local wheel (pattern: {wheel_pattern})")
        return 1

    if len(wheels) > 1:
        log(f"WARNING: Multiple wheels found, using first: {wheels[0]}")

    local_wheel = wheels[0]
    log(f"Installing WhisperJAV from local wheel: {local_wheel}")
    log("Using --no-deps to avoid reinstalling dependencies...")

    if not run_pip(
        ["install", "--no-deps", local_wheel, "--progress-bar", "on"],
        "Install WhisperJAV application"
    ):
        create_failure_file("WhisperJAV application installation failed")
        return 1

    # === Phase 5.5: Copy Launcher to Root ===
    log_section("Phase 5.5: User-Friendly Launcher Setup")
    launcher_exe = copy_launcher_to_root()

    # === Phase 5.8: Verify Icon File ===
    log_section("Phase 5.8: Icon File Verification")
    icon_path = os.path.join(sys.prefix, 'whisperjav_icon.ico')
    if os.path.exists(icon_path):
        icon_size = os.path.getsize(icon_path)
        log(f"✓ Icon file found: {icon_path}")
        log(f"  Size: {icon_size:,} bytes")
        if icon_size < 1000:
            log(f"  WARNING: Icon file may be corrupted (too small)")
    else:
        log(f"✗ WARNING: Icon file NOT found at: {icon_path}")
        log(f"  This may cause the application icon to not display correctly.")
        log(f"  Checking alternative locations...")

        # Check if it's in the package
        site_packages_icon = os.path.join(sys.prefix, 'Lib', 'site-packages', 'whisperjav', 'webview_gui', 'assets', 'whisperjav_icon.ico')
        if os.path.exists(site_packages_icon):
            log(f"  Found icon in package: {site_packages_icon}")
        else:
            log(f"  Icon not found in package either")

    # === Installation Complete ===
    # Note: Desktop shortcut is now created by NSIS installer during installation
    # The shortcut launches: pythonw.exe -m whisperjav.webview_gui.main
    # Working directory is set to the installation folder ($INSTDIR)
    # Icon: whisperjav_icon.ico in installation folder
    print_installation_summary(install_start_time)

    log("\nInstallation completed successfully!")
    log("You may now close this window.")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()

        if exit_code != 0:
            log("\n" + "!" * 80)
            log("  INSTALLATION FAILED")
            log("!" * 80)
            log(f"Check {LOG_FILE} for details.")
            log("This window will close in 60 seconds...")
            time.sleep(60)
        else:
            log("\nInstallation complete! Window will close in 15 seconds...")
            log("(Press Enter to close immediately)")
            timed_input("", 15, "")

        sys.exit(exit_code)

    except KeyboardInterrupt:
        log("\nInstallation interrupted by user.")
        sys.exit(1)

    except Exception as e:
        log(f"\nFATAL: Unhandled exception occurred:")
        log(str(e))
        log("\nFull traceback:")
        log(traceback.format_exc())
        log("\nThis window will close in 60 seconds...")
        time.sleep(60)
        sys.exit(1)
