"""
WhisperJAV v1.7.5 Post-Install Script
======================================

This script runs after the conda environment is created and:
1. Performs comprehensive preflight checks (disk space, network, WebView2)
2. Detects NVIDIA GPU and installs appropriate PyTorch build
3. Offers CPU-only fallback for systems without NVIDIA GPU
4. Installs all Python dependencies from requirements_v1.7.5.txt
5. Installs WhisperJAV from GitHub
6. Creates desktop shortcut
7. Provides detailed installation summary

All output is logged to install_log_v1.7.5.txt
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

LOG_FILE = os.path.join(sys.prefix, "install_log_v1.7.5.txt")

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


TORCH_DRIVER_MATRIX: Sequence[DriverMatrixEntry] = (
    DriverMatrixEntry(
        (580, 65, 0),
        "CUDA 13.0",
        "pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu128"
    ),
    DriverMatrixEntry(
        (570, 65, 0),
        "CUDA 12.8",
        "pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu128"
    ),
    DriverMatrixEntry(
        (560, 76, 0),
        "CUDA 12.6",
        "pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu126"
    ),
    DriverMatrixEntry(
        (551, 61, 0),
        "CUDA 12.4",
        "pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
    ),
    DriverMatrixEntry(
        (531, 14, 0),
        "CUDA 12.1",
        "pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121"
    ),
    DriverMatrixEntry(
        (520, 6, 0),
        "CUDA 11.8",
        "pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118"
    ),
)


# =============================================================================
# TeeLogger: Capture ALL console output to log file
# =============================================================================
# This ensures that even crashes, pip output, and library warnings are logged.

class TeeLogger:
    """
    Duplicates all output to both the terminal and a log file simultaneously.

    This captures EVERYTHING written to stdout/stderr including:
    - print() statements
    - Library warnings (e.g., PyTorch's NumPy warning)
    - Subprocess output when piped through
    - Crash tracebacks

    Uses line buffering so output is saved even if the script crashes.
    """

    def __init__(self, log_path: str, original_stream):
        self.terminal = original_stream
        self.log_path = log_path
        self._log_file = None
        self._open_log()

    def _open_log(self):
        """Open or reopen the log file"""
        try:
            self._log_file = open(
                self.log_path, "a",
                encoding="utf-8",
                buffering=1,  # Line buffering for immediate writes
                errors="replace"  # Don't crash on encoding errors
            )
        except Exception:
            self._log_file = None

    def write(self, message: str):
        """Write to both terminal and log file"""
        # Always write to terminal
        if self.terminal:
            try:
                self.terminal.write(message)
            except Exception:
                pass  # Terminal write failed, continue anyway

        # Write to log file
        if self._log_file:
            try:
                self._log_file.write(message)
                self._log_file.flush()  # Flush immediately for crash safety
            except Exception:
                # Try to reopen if file was closed
                self._open_log()
                if self._log_file:
                    try:
                        self._log_file.write(message)
                        self._log_file.flush()
                    except Exception:
                        pass

    def flush(self):
        """Flush both streams"""
        if self.terminal:
            try:
                self.terminal.flush()
            except Exception:
                pass
        if self._log_file:
            try:
                self._log_file.flush()
            except Exception:
                pass

    def close(self):
        """Close the log file (terminal stays open)"""
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    # Support fileno() for subprocess compatibility
    def fileno(self):
        if self.terminal:
            return self.terminal.fileno()
        return -1

    # Support isatty() for color detection
    def isatty(self):
        if self.terminal:
            try:
                return self.terminal.isatty()
            except Exception:
                pass
        return False


# Global TeeLogger instance (set by setup_tee_logging)
_tee_logger: Optional[TeeLogger] = None


def setup_tee_logging(install_dir: str) -> bool:
    """
    Setup TeeLogger to capture ALL console output to a comprehensive log file.

    This should be called at the very start of main() before any other code runs.

    Args:
        install_dir: Directory to store the log file (usually sys.prefix)

    Returns:
        True if setup successful, False otherwise
    """
    global _tee_logger

    log_path = os.path.join(install_dir, "install_log_full_v1.7.5.txt")

    try:
        # Create TeeLogger for stdout
        _tee_logger = TeeLogger(log_path, sys.stdout)

        # Redirect both stdout and stderr through TeeLogger
        sys.stdout = _tee_logger
        sys.stderr = _tee_logger

        # Write initial header
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*80}")
        print(f"  WhisperJAV v1.7.5 Installation Log (Full Console Capture)")
        print(f"  Started: {ts}")
        print(f"  Log file: {log_path}")
        print(f"{'='*80}\n")

        return True

    except Exception as e:
        # If TeeLogger setup fails, continue anyway with normal logging
        print(f"WARNING: Could not setup TeeLogger: {e}")
        return False


def setup_crash_handler():
    """
    Setup exception hook to log crashes before the console window closes.

    This ensures that even unexpected crashes are captured in the log file.
    """
    original_excepthook = sys.excepthook

    def crash_handler(exc_type, exc_value, exc_tb):
        """Log the crash, then call the original exception handler"""
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'='*80}")
            print(f"  FATAL CRASH at {ts}")
            print(f"{'='*80}")
            print(f"Exception Type: {exc_type.__name__}")
            print(f"Exception Value: {exc_value}")
            print("\nFull Traceback:")
            traceback.print_exception(exc_type, exc_value, exc_tb)
            print(f"{'='*80}\n")

            # Flush to ensure it's written
            if hasattr(sys.stdout, 'flush'):
                sys.stdout.flush()
            if hasattr(sys.stderr, 'flush'):
                sys.stderr.flush()

        except Exception:
            pass  # Don't let the crash handler crash

        # Call original handler
        original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = crash_handler


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


def verify_existing_torch_stack() -> bool:
    """Return True if CUDA-enabled torch and torchaudio meet requirements."""
    log("Checking for existing CUDA-enabled PyTorch and torchaudio...")
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        log(f"PyTorch stack missing: {exc}")
        return False

    log(f"Found PyTorch {torch.__version__}")
    try:
        log(f"Found torchaudio {torchaudio.__version__}")
    except Exception:
        log("Unable to determine torchaudio version.")
    cuda_available = torch.cuda.is_available()
    log(f"PyTorch CUDA available: {'YES' if cuda_available else 'NO'}")

    if not cuda_available:
        return False

    cuda_str = getattr(torch.version, "cuda", "") or ""
    cuda_tuple = parse_version_string(cuda_str)
    if not cuda_tuple:
        log("Unable to determine PyTorch CUDA version; reinstall required.")
        return False

    if cuda_tuple >= MIN_TORCH_CUDA_VERSION:
        log(f"Existing PyTorch satisfies CUDA >= {format_version_tuple(MIN_TORCH_CUDA_VERSION)}. Skipping reinstall.")
        return True

    log(
        "Installed PyTorch CUDA version "
        f"{format_version_tuple(cuda_tuple)} is below required "
        f"{format_version_tuple(MIN_TORCH_CUDA_VERSION)}. Reinstalling..."
    )
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
    """Install PyTorch/Torchaudio with best available acceleration."""
    log_section("PyTorch Installation")

    if verify_existing_torch_stack():
        return True

    driver_info = driver_info or detect_nvidia_driver()
    plan = select_torch_install_plan(driver_info)

    if plan.uses_gpu:
        # Extract the actual CUDA binary version from the pip command URL
        # e.g., cu128 -> CUDA 12.8, cu126 -> CUDA 12.6, cu121 -> CUDA 12.1
        cuda_binary_version = plan.target_label  # e.g., "CUDA 12.8"
        driver_version_str = format_version_tuple(plan.driver_detected)

        log(
            f"Preparing to install PyTorch ({cuda_binary_version} build) "
            f"for GPU: {plan.gpu_name or 'Unknown GPU'}"
        )
        log(f"  Detected driver: {driver_version_str}")
        log(f"  Binary compatibility: {cuda_binary_version} builds work with driver {driver_version_str}")
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

    # Verify PyTorch installation using subprocess to avoid numpy warning
    # (numpy is not installed yet at this phase, and torch tries to import it)
    log("Verifying PyTorch installation...")
    try:
        # Run verification in subprocess with warnings suppressed
        verify_cmd = [
            os.path.join(sys.prefix, 'python.exe'),
            '-c',
            'import warnings; warnings.filterwarnings("ignore"); '
            'import torch; '
            'print(f"VERSION:{torch.__version__}"); '
            'print(f"CUDA:{torch.cuda.is_available()}"); '
            'print(f"DEVICES:{torch.cuda.device_count() if torch.cuda.is_available() else 0}")'
        ]
        result = subprocess.run(
            verify_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            version = cuda_avail = devices = None
            for line in output_lines:
                if line.startswith("VERSION:"):
                    version = line.split(":", 1)[1]
                elif line.startswith("CUDA:"):
                    cuda_avail = line.split(":", 1)[1] == "True"
                elif line.startswith("DEVICES:"):
                    devices = int(line.split(":", 1)[1])

            if version:
                log(f"PyTorch {version} installed successfully!")
                if cuda_avail:
                    log(f"CUDA acceleration: ENABLED (devices: {devices})")
                else:
                    log("CUDA acceleration: DISABLED (CPU-only mode)")
            else:
                log("PyTorch installed (version check pending)")
        else:
            log(f"WARNING: PyTorch verification returned non-zero exit code")
            if result.stderr:
                log(f"  stderr: {result.stderr.strip()[:200]}")
    except subprocess.TimeoutExpired:
        log("WARNING: PyTorch verification timed out (but installation may be OK)")
    except Exception as exc:
        log(f"WARNING: PyTorch verification failed: {exc}")
        log("         (This may be OK - full verification will happen after Phase 4)")

    return True


def run_pip(args: list, description: str, retries: int = 3, stream_output: bool = True) -> bool:
    """
    Run pip command with retries and real-time output streaming.

    Args:
        args: Pip arguments (e.g., ["install", "package"])
        description: Human-readable description for logging
        retries: Number of retry attempts (default 3)
        stream_output: If True, stream output in real-time through TeeLogger (default True)

    Returns:
        True if successful, False otherwise
    """
    log(f"Starting: {description}")

    # Add progress bar flag to install commands for better UX
    if 'install' in args and '--progress-bar' not in args:
        args = args + ['--progress-bar', 'on']

    pip_cmd = [os.path.join(sys.prefix, 'python.exe'), '-m', 'pip'] + args

    for attempt in range(retries):
        log(f"Attempt {attempt+1}/{retries}: {' '.join(pip_cmd)}")

        if stream_output:
            # Stream output in real-time through TeeLogger
            # This ensures ALL pip output (including warnings, errors) is captured
            process = None
            try:
                process = subprocess.Popen(
                    pip_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1  # Line buffered
                )

                # Track progress for user feedback
                start_time = time.time()
                last_status_time = start_time
                packages_downloaded = 0
                packages_installed = 0
                current_package = ""

                # Stream output line by line with progress tracking
                if process.stdout:
                    for line in process.stdout:
                        line = line.rstrip('\n\r')
                        if not line:
                            continue

                        # Parse pip output for progress information
                        line_lower = line.lower()

                        # Detect package downloads
                        if 'downloading' in line_lower:
                            packages_downloaded += 1
                            # Extract package name if present
                            if ' ' in line:
                                parts = line.split()
                                for part in parts:
                                    if '-' in part and not part.startswith('-'):
                                        current_package = part.split('-')[0][:30]
                                        break

                        # Detect package installations
                        if 'installing collected packages' in line_lower:
                            # Count packages being installed
                            if ':' in line:
                                pkg_list = line.split(':')[1] if ':' in line else ''
                                packages_installed = len([p for p in pkg_list.split(',') if p.strip()])

                        # Detect successful install
                        if 'successfully installed' in line_lower:
                            packages_installed = line.count(',') + 1

                        # Print line to log
                        print(f"  [pip] {line}")
                        sys.stdout.flush()

                        # Show periodic status update every 30 seconds during long operations
                        current_time = time.time()
                        if current_time - last_status_time > 30:
                            elapsed = int(current_time - start_time)
                            status_parts = [f"  ... still working ({elapsed}s elapsed)"]
                            if packages_downloaded > 0:
                                status_parts.append(f"{packages_downloaded} packages downloaded")
                            if current_package:
                                status_parts.append(f"current: {current_package}")
                            log(" | ".join(status_parts))
                            last_status_time = current_time

                # Wait for process to complete
                return_code = process.wait(timeout=1800)  # 30 minute timeout

                # Show final summary
                elapsed = int(time.time() - start_time)
                if return_code == 0:
                    summary_parts = [f"SUCCESS: {description}"]
                    if elapsed > 10:
                        summary_parts.append(f"({elapsed}s)")
                    if packages_installed > 0:
                        summary_parts.append(f"({packages_installed} packages)")
                    log(" ".join(summary_parts))
                    return True
                else:
                    log(f"ERROR: {description} failed (rc={return_code}) after {elapsed}s")
                    if attempt < retries - 1:
                        log("Retrying in 10 seconds...")
                        time.sleep(10)

            except subprocess.TimeoutExpired:
                log(f"ERROR: {description} timed out (30 minutes)")
                if process:
                    process.kill()
                    process.wait()
                if attempt < retries - 1:
                    log("Retrying in 10 seconds...")
                    time.sleep(10)

            except Exception as e:
                log(f"ERROR: Unexpected error during pip execution: {e}")
                traceback.print_exc()  # Full traceback goes through TeeLogger
                if attempt < retries - 1:
                    log("Retrying in 10 seconds...")
                    time.sleep(10)

        else:
            # Original behavior: capture output (for quick commands)
            try:
                result = subprocess.run(
                    pip_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    encoding='utf-8',
                    timeout=1800
                )
                if result.stdout:
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
    failure_file = os.path.join(sys.prefix, "INSTALLATION_FAILED_v1.7.5.txt")
    try:
        with open(failure_file, "w", encoding="utf-8") as f:
            f.write("WhisperJAV v1.7.5 Installation Failed\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Error: {error_message}\n\n")
            f.write("Manual Cleanup Steps:\n")
            f.write(f"1. Delete installation directory: {sys.prefix}\n")
            f.write("2. Delete desktop shortcut: WhisperJAV v1.7.5.lnk\n")
            f.write(f"3. Check install_log_v1.7.5.txt for details\n\n")
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
    log(" " * 20 + "WhisperJAV v1.7.5 Installation Complete!")
    log("=" * 80)
    log("")
    log("INSTALLATION SUMMARY")
    log("-" * 40)
    log(f"  Installation directory: {sys.prefix}")
    log(f"  Python version: {sys.version.split()[0]}")
    log(f"  Installation time: {minutes}m {seconds}s")
    log("")

    # PyTorch and CUDA status
    log("GPU/CUDA STATUS")
    log("-" * 40)
    try:
        import torch
        if torch.cuda.is_available():
            log(f"  ✓ PyTorch: {torch.__version__}")
            log(f"  ✓ CUDA version: {torch.version.cuda}")
            log(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            log(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            log(f"  ✓ Status: GPU ACCELERATION ENABLED")
        else:
            log(f"  ⚠ PyTorch: {torch.__version__} (CPU-only)")
            log(f"  ⚠ Status: CPU MODE (processing will be slower)")
    except Exception as e:
        log(f"  ? PyTorch: Could not determine status ({e})")
    log("")

    # GUI status
    log("GUI STATUS")
    log("-" * 40)
    if check_webview2_windows():
        log(f"  ✓ WebView2 runtime: Detected")
    else:
        log(f"  ✗ WebView2 runtime: NOT DETECTED")
        log(f"    Install from: https://go.microsoft.com/fwlink/p/?LinkId=2124703")
    log("")

    # Key packages verification using importlib.metadata (more reliable than imports)
    # Using metadata avoids import failures from missing DLLs or heavy dependencies
    log("INSTALLED PACKAGES (Key Components)")
    log("-" * 40)

    # Map: (pip/distribution name, display name)
    # Note: Distribution names may differ from import names
    key_packages = [
        ("whisperjav", "WhisperJAV"),
        ("faster-whisper", "Faster Whisper"),
        ("openai-whisper", "OpenAI Whisper"),
        ("stable-ts", "Stable-TS"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("modelscope", "ModelScope (ZipEnhancer)"),
        ("numpy", "NumPy"),
        ("librosa", "Librosa"),
        ("pywebview", "PyWebView"),
    ]

    for pkg_dist_name, pkg_display in key_packages:
        try:
            # Use importlib.metadata.version() - checks distribution, not import
            result = subprocess.run(
                [os.path.join(sys.prefix, 'python.exe'), '-c',
                 f'from importlib.metadata import version; print(version("{pkg_dist_name}"))'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                version = result.stdout.strip()
                log(f"  ✓ {pkg_display}: {version}")
            else:
                # Fallback: try import method for packages with different dist names
                log(f"  ✗ {pkg_display}: NOT INSTALLED")
        except Exception:
            log(f"  ? {pkg_display}: Unknown")
    log("")

    # Shortcuts status
    log("SHORTCUTS")
    log("-" * 40)
    log("  ✓ Desktop shortcut: WhisperJAV v1.7.5.lnk")
    log("  ✓ Start Menu: WhisperJAV folder")
    log("  Note: Shortcuts are created by the NSIS installer")
    log("")

    # Next steps
    log("NEXT STEPS")
    log("-" * 40)
    log("  1. Launch WhisperJAV from the desktop shortcut")
    log("  2. On first run, AI models will download (~3GB)")
    log("  3. Select your video files and start processing!")
    log("")
    log("TROUBLESHOOTING")
    log("-" * 40)
    log(f"  Log file: {LOG_FILE}")
    log("  Support: https://github.com/meizhong986/WhisperJAV/issues")
    log("")
    log("=" * 80)
    log("")


def embed_icon_in_exe(exe_path: str, icon_path: str) -> bool:
    """
    Embed icon into executable file using LIEF library.

    This is a best-effort operation - the desktop shortcut will display
    the correct icon regardless of whether this succeeds.

    Args:
        exe_path: Path to the .exe file
        icon_path: Path to the .ico file

    Returns:
        True if successful, False otherwise

    Note: LIEF API varies between versions. This function tries multiple approaches.
    """
    log(f"Attempting to embed icon into executable (optional)...")
    log(f"  Executable: {exe_path}")
    log(f"  Icon: {icon_path}")

    try:
        # Try to import lief, install if not available
        try:
            import lief
        except ImportError:
            log(f"  Installing LIEF library...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "lief"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode != 0:
                log(f"  INFO: Could not install LIEF library - skipping icon embedding")
                return False
            import lief

        # Load the executable
        binary = lief.parse(exe_path)
        if not binary:
            log(f"  INFO: Could not parse executable - skipping icon embedding")
            return False

        # Check for resources section
        if not binary.has_resources:
            log(f"  INFO: Executable has no resources section - skipping icon embedding")
            return False

        # Get resources manager
        manager = binary.resources_manager
        if manager is None:
            log(f"  INFO: Could not get resources manager - skipping icon embedding")
            return False

        # Try different LIEF API approaches for icon changing
        icon_changed = False

        # Approach 1: Try change_icon with file path (older LIEF versions)
        try:
            manager.change_icon(icon_path)
            icon_changed = True
            log(f"  Icon changed using path method")
        except (TypeError, AttributeError) as e:
            log(f"  Path method failed: {e}")

        # Approach 2: Try loading icon as PE and extracting resources (newer LIEF)
        if not icon_changed:
            try:
                # Load icon file as binary (some .ico files can be parsed as PE)
                icon_binary = lief.parse(icon_path)
                if icon_binary and hasattr(icon_binary, 'resources_manager'):
                    icon_manager = icon_binary.resources_manager
                    if icon_manager:
                        # Try to get icons from icon binary and add to exe
                        if hasattr(icon_manager, 'icons') and icon_manager.icons:
                            for icon in icon_manager.icons:
                                try:
                                    manager.add_icon(icon)
                                    icon_changed = True
                                except Exception:
                                    pass
                            if icon_changed:
                                log(f"  Icon changed using PE parse method")
            except Exception as e:
                log(f"  PE parse method failed: {e}")

        if not icon_changed:
            log(f"  INFO: Icon embedding not supported by this LIEF version")
            log(f"        The desktop shortcut will still display the correct icon")
            return False

        # Build and write the modified executable
        try:
            builder = lief.PE.Builder(binary)
            builder.build_resources(True)
            builder.build()
            builder.write(exe_path)
            log(f"  ✓ Icon embedded successfully")
            return True
        except Exception as e:
            log(f"  INFO: Could not write modified executable: {e}")
            return False

    except Exception as e:
        log(f"  INFO: Icon embedding failed: {e}")
        log(f"        This is optional - the desktop shortcut will display the correct icon")
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


def log_environment_info():
    """Log detailed environment information for debugging"""
    import platform
    log_section("Environment Information (for debugging)")

    # Resolve paths to clean up any ".." or relative components
    script_path = Path(__file__).resolve() if '__file__' in dir() else Path(sys.argv[0]).resolve()
    install_prefix = Path(sys.prefix).resolve()
    python_exe = Path(sys.executable).resolve()

    log(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Script location: {script_path}")
    log(f"Platform: {platform.platform()}")
    log(f"Architecture: {platform.machine()}")
    log(f"Python version: {platform.python_version()}")
    log(f"Python implementation: {platform.python_implementation()}")
    log("")
    log(f"Installation prefix: {install_prefix}")
    log(f"Python executable: {python_exe}")
    log(f"Sys.path[0]: {Path(sys.path[0]).resolve() if sys.path else 'N/A'}")
    log("")

    # Log PATH (truncated)
    path_env = os.environ.get("PATH", "")
    path_entries = path_env.split(os.pathsep)[:10]  # First 10 entries
    log(f"PATH (first {len(path_entries)} entries):")
    for p in path_entries:
        log(f"  - {p}")
    if len(path_env.split(os.pathsep)) > 10:
        log(f"  ... and {len(path_env.split(os.pathsep)) - 10} more entries")
    log("")

    # Log disk space
    try:
        total, used, free = shutil.disk_usage(sys.prefix)
        log(f"Disk space at {sys.prefix}:")
        log(f"  Total: {total / (1024**3):.1f} GB")
        log(f"  Used:  {used / (1024**3):.1f} GB")
        log(f"  Free:  {free / (1024**3):.1f} GB")
    except Exception as e:
        log(f"Could not determine disk space: {e}")
    log("")

    # Log critical executables
    # Note: In conda environments, git and ffmpeg are in Library/bin, not Scripts
    log("Checking critical executables:")

    # Define search paths for each executable type
    # Conda installs some tools in Library/bin, others in Scripts or root
    exe_search_paths = {
        "python.exe": [sys.prefix, os.path.join(sys.prefix, "Scripts")],
        "pythonw.exe": [sys.prefix, os.path.join(sys.prefix, "Scripts")],
        "pip.exe": [os.path.join(sys.prefix, "Scripts"), sys.prefix],
        "git.exe": [
            os.path.join(sys.prefix, "Library", "bin"),  # Conda git location!
            os.path.join(sys.prefix, "Scripts"),
            sys.prefix,
        ],
        "ffmpeg.exe": [
            os.path.join(sys.prefix, "Library", "bin"),  # Conda ffmpeg location
            os.path.join(sys.prefix, "Scripts"),
            sys.prefix,
        ],
    }

    for exe_name, search_paths in exe_search_paths.items():
        found = False
        for search_path in search_paths:
            exe_path = os.path.join(search_path, exe_name)
            if os.path.exists(exe_path):
                log(f"  ✓ {exe_name}: {exe_path}")
                found = True
                break
        if not found:
            # Also check if it's in system PATH (pip can find it even if we can't)
            system_path = shutil.which(exe_name.replace(".exe", ""))
            if system_path:
                log(f"  ✓ {exe_name}: {system_path} (system PATH)")
            else:
                # Only warn for git - it's critical but pip may find it anyway
                if exe_name == "git.exe":
                    log(f"  ⚠ {exe_name}: Not found in conda env (pip may still find it)")
                else:
                    log(f"  ✗ {exe_name}: NOT FOUND")
    log("")


def main() -> int:
    """Main installation workflow"""
    # === CRITICAL: Setup TeeLogger FIRST to capture ALL output ===
    # This must be before ANY other code to ensure crashes are logged
    setup_tee_logging(sys.prefix)
    setup_crash_handler()

    install_start_time = time.time()

    log_section("WhisperJAV v1.7.5 Post-Install Started")
    log(f"Installation prefix: {sys.prefix}")
    log(f"Python executable: {sys.executable}")
    log(f"Python version: {sys.version}")
    log(f"Full log: {os.path.join(sys.prefix, 'install_log_full_v1.7.5.txt')}")

    # Log detailed environment info for debugging
    log_environment_info()

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

    req_path = os.path.join(sys.prefix, "requirements_v1.7.5.txt")
    constraints_path = os.path.join(sys.prefix, "constraints_v1.7.5.txt")

    if not os.path.exists(req_path):
        log(f"ERROR: requirements_v1.7.5.txt not found at {req_path}")
        create_failure_file(f"Missing requirements file: {req_path}")
        return 1

    # Check for constraints file (optional but recommended)
    use_constraints = os.path.exists(constraints_path)
    if use_constraints:
        log(f"✓ Constraints file found: {constraints_path}")
        log("  Using constraints to prevent version conflicts with speech enhancement packages")
    else:
        log(f"⚠ Constraints file not found: {constraints_path}")
        log("  Proceeding without constraints (may cause version conflicts)")

    log(f"Installing dependencies from: {req_path}")
    log("This will download ~800MB of packages including speech enhancement backends.")
    log("Please wait... (this may take 10-20 minutes depending on network speed)")
    log("")

    # Build pip install command with constraints if available
    pip_args = ["install", "-r", req_path, "--progress-bar", "on"]
    if use_constraints:
        pip_args.extend(["-c", constraints_path])

    # Log the exact command for debugging
    pip_cmd_str = f"pip {' '.join(pip_args)}"
    log(f"DEBUG: Pip command: {pip_cmd_str}")

    if not run_pip(pip_args, "Install Python dependencies"):
        log("")
        log("=" * 80)
        log("  DEPENDENCY INSTALLATION TROUBLESHOOTING")
        log("=" * 80)
        log("Common issues and solutions:")
        log("1. Network timeout: Check internet connection, retry installation")
        log("2. Version conflict: modelscope/clearvoice may conflict with numpy>=2.0")
        log("   Solution: Try running: pip install numpy>=2.0 --force-reinstall")
        log("3. Git not in PATH: Ensure git is installed and accessible")
        log(f"4. Check detailed errors in: {LOG_FILE}")
        log("=" * 80)
        create_failure_file("Dependencies installation failed")
        return 1

    # === Phase 4.5: Verify Critical Dependencies ===
    log_section("Phase 4.5: Verifying Critical Dependencies")
    verification_passed = True

    # List of critical packages to verify
    # Format: (distribution_name, version_spec) - using pip/dist names, not import names
    critical_packages = [
        ("numpy", ">=2.0"),
        ("scipy", ">=1.10.1"),
        ("librosa", ">=0.11.0"),
        ("datasets", ">=2.14.0,<4.0"),
        ("modelscope", ">=1.20"),
        ("faster-whisper", ">=1.1.0"),  # Note: dist name uses hyphen
        ("transformers", ">=4.40.0"),
    ]

    log("Verifying critical package installations using importlib.metadata...")
    log("(This is more reliable than import-based checks)")
    for pkg_dist_name, version_spec in critical_packages:
        try:
            # Use importlib.metadata.version() instead of import
            # This avoids false negatives from import failures
            result = subprocess.run(
                [os.path.join(sys.prefix, 'python.exe'), '-c',
                 f'from importlib.metadata import version; print(version("{pkg_dist_name}"))'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                version = result.stdout.strip()
                log(f"  ✓ {pkg_dist_name}: {version}")
            else:
                log(f"  ✗ {pkg_dist_name}: NOT FOUND IN METADATA")
                if result.stderr:
                    log(f"    Error: {result.stderr.strip()[:100]}")
                verification_passed = False
        except Exception as e:
            log(f"  ✗ {pkg_dist_name}: ERROR ({e})")
            verification_passed = False

    if not verification_passed:
        log("")
        log("WARNING: Some critical packages failed verification.")
        log("The application may not function correctly.")
        log("Consider reinstalling or checking the log for errors.")

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

    # NOTE: Desktop shortcut creation is handled by NSIS after post_install completes.
    # NSIS is the appropriate tool for Windows shell integration (shortcuts, registry, uninstaller).
    # See custom_template_v1.7.5.nsi.tmpl lines 1433-1511.

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
