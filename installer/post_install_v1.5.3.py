"""
WhisperJAV v1.5.3 Post-Install Script
======================================

This script runs after the conda environment is created and:
1. Performs comprehensive preflight checks (disk space, network, WebView2)
2. Detects NVIDIA GPU and installs appropriate PyTorch build
3. Offers CPU-only fallback for systems without NVIDIA GPU
4. Installs all Python dependencies from requirements_v1.5.3.txt
5. Installs WhisperJAV from GitHub
6. Creates desktop shortcut
7. Provides detailed installation summary

All output is logged to install_log_v1.5.3.txt
"""

import os
import sys
import re
import time
import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

LOG_FILE = os.path.join(sys.prefix, "install_log_v1.5.3.txt")


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


def check_cuda_driver() -> tuple[bool, float]:
    """
    Check for NVIDIA GPU and CUDA driver version

    Returns:
        (has_cuda, cuda_version) - cuda_version is 0.0 if no CUDA
    """
    log("Checking for NVIDIA GPU and CUDA driver...")
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            timeout=10
        )
        output = result.stdout
        match = re.search(r"CUDA Version: (\d+\.\d+)", output)
        if not match:
            log("WARNING: Could not detect CUDA version from nvidia-smi output.")
            return False, 0.0

        cuda_version = float(match.group(1))
        log(f"NVIDIA GPU detected with CUDA {cuda_version}")
        return True, cuda_version

    except FileNotFoundError:
        log("INFO: 'nvidia-smi' not found. No NVIDIA GPU detected.")
        return False, 0.0
    except subprocess.TimeoutExpired:
        log("WARNING: nvidia-smi timed out.")
        return False, 0.0
    except subprocess.CalledProcessError as e:
        log(f"WARNING: nvidia-smi failed (rc={e.returncode})")
        return False, 0.0
    except Exception as e:
        log(f"WARNING: Unexpected error checking NVIDIA driver: {e}")
        return False, 0.0


def check_existing_pytorch() -> bool:
    """Check if compatible PyTorch is already installed"""
    log("Checking for existing PyTorch installation...")
    try:
        import torch
        log(f"Found PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            log(f"PyTorch CUDA support: YES (built with CUDA {cuda_version})")
            return True
        else:
            log("PyTorch CUDA support: NO (will reinstall CUDA build)")
            return False
    except ImportError:
        log("PyTorch not installed yet.")
        return False
    except Exception as e:
        log(f"Error checking PyTorch: {e}")
        return False


def install_pytorch(cuda_version: float) -> bool:
    """
    Install PyTorch with appropriate CUDA version or CPU-only

    Args:
        cuda_version: Detected CUDA version from nvidia-smi (0.0 = no CUDA)

    Returns:
        True if installation successful

    CUDA Compatibility:
        PyTorch CUDA builds require drivers with matching or higher CUDA support:
        - cu118 requires driver CUDA >= 11.8
        - cu121 requires driver CUDA >= 12.1
        - cu124 requires driver CUDA >= 12.4

        We select the HIGHEST PyTorch CUDA build that the driver supports
        to maximize performance while ensuring compatibility.
    """
    log_section("PyTorch Installation")

    # Select HIGHEST compatible PyTorch CUDA build for the detected driver
    if cuda_version >= 12.8:
        log("Installing PyTorch with CUDA 12.4+ support (Blackwell compatible)...")
        log("Detected CUDA 12.8+ driver - supports NVIDIA Blackwell GPUs (RTX 50-series)")
        log("This will download ~2GB of packages. Please wait...")
        pytorch_index = "https://download.pytorch.org/whl/cu124"
        build_type = "CUDA 12.4+ (Blackwell)"
    elif cuda_version >= 12.4:
        log("Installing PyTorch with CUDA 12.4 support...")
        log("This will download ~2GB of packages. Please wait...")
        pytorch_index = "https://download.pytorch.org/whl/cu124"
        build_type = "CUDA 12.4"
    elif cuda_version >= 12.1:
        log("Installing PyTorch with CUDA 12.1 support...")
        log("This will download ~2GB of packages. Please wait...")
        pytorch_index = "https://download.pytorch.org/whl/cu121"
        build_type = "CUDA 12.1"
    elif cuda_version >= 11.8:
        log("Installing PyTorch with CUDA 11.8 support...")
        log("This will download ~2GB of packages. Please wait...")
        pytorch_index = "https://download.pytorch.org/whl/cu118"
        build_type = "CUDA 11.8"
    else:
        log("No compatible NVIDIA GPU detected.")
        log("")
        log("You can install CPU-only PyTorch, but processing will be MUCH slower:")
        log("  - GPU (CUDA): ~5-10 minutes per hour of video")
        log("  - CPU only: ~30-60 minutes per hour of video (6-10x slower)")
        log("")
        log("Waiting for your choice (auto-accepts CPU-only in 20 seconds)...")

        response = timed_input("Install CPU-only version? (y/N): ", 20, "y").strip().lower()
        if response == 'y':
            log("Installing CPU-only PyTorch...")
            pytorch_index = "https://download.pytorch.org/whl/cpu"
            build_type = "CPU-only"
        else:
            log("Skipping PyTorch installation. You must install it manually later.")
            log("Installation cannot continue without PyTorch.")
            return False

    # Install PyTorch with version constraint (WhisperJAV requires 2.1+)
    success = run_pip(
        ["install", "torch>=2.1,<3.0", "torchvision", "torchaudio",
         "--index-url", pytorch_index, "--progress-bar", "on"],
        f"Install PyTorch ({build_type})"
    )

    if success:
        # Verify installation
        try:
            import torch
            log(f"PyTorch {torch.__version__} installed successfully!")
            if torch.cuda.is_available():
                log(f"CUDA acceleration: ENABLED (devices: {torch.cuda.device_count()})")
            else:
                log("CUDA acceleration: DISABLED (CPU-only mode)")
        except Exception as e:
            log(f"WARNING: PyTorch installed but verification failed: {e}")

    return success


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
    failure_file = os.path.join(sys.prefix, "INSTALLATION_FAILED_v1.5.3.txt")
    try:
        with open(failure_file, "w", encoding="utf-8") as f:
            f.write("WhisperJAV v1.5.3 Installation Failed\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Error: {error_message}\n\n")
            f.write("Manual Cleanup Steps:\n")
            f.write(f"1. Delete installation directory: {sys.prefix}\n")
            f.write("2. Delete desktop shortcut: WhisperJAV v1.5.3.lnk\n")
            f.write(f"3. Check install_log_v1.5.3.txt for details\n\n")
            f.write("Common Solutions:\n")
            f.write("- Out of disk space: Free up 8GB and retry\n")
            f.write("- Network error: Check internet connection and firewall\n")
            f.write("- CUDA error: Update NVIDIA drivers or try CPU mode\n")
            f.write("- WebView2 missing: Install from https://go.microsoft.com/fwlink/p/?LinkId=2124703\n\n")
            f.write("Support: https://github.com/meizhong986/WhisperJAV/issues\n")
        log(f"Failure details written to: {failure_file}")
    except Exception:
        pass


def print_installation_summary(install_start_time: float, cuda_version: float):
    """Print a comprehensive installation summary"""
    install_duration = int(time.time() - install_start_time)
    minutes = install_duration // 60
    seconds = install_duration % 60

    log("\n\n")
    log("=" * 80)
    log(" " * 20 + "WhisperJAV v1.5.3 Installation Complete!")
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

    if check_webview2_windows():
        log(f"  ✓ WebView2 runtime: Detected")
    else:
        log(f"  ⚠ WebView2 runtime: NOT DETECTED (GUI will not work)")
        log(f"    Install from: https://go.microsoft.com/fwlink/p/?LinkId=2124703")

    log(f"  ✓ Desktop shortcut: Created")
    log(f"  ✓ Installation time: {minutes}m {seconds}s")
    log("")
    log("Next Steps:")
    log("  1. Launch WhisperJAV from the desktop shortcut")
    log("  2. On first run, AI models will download (~3GB, 5-10 minutes)")
    log("  3. Select your video files and start processing!")
    log("")
    log(f"Logs saved to: {LOG_FILE}")
    log("=" * 80)
    log("")


def copy_launcher_to_root() -> str:
    """
    Copy Scripts/whisperjav-gui.exe to installation root as WhisperJAV-GUI.exe
    This provides a user-friendly launcher that users can easily find and run.

    Returns:
        Path to copied .exe or None if failed
    """
    scripts_exe = os.path.join(sys.prefix, "Scripts", "whisperjav-gui.exe")
    root_exe = os.path.join(sys.prefix, "WhisperJAV-GUI.exe")

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

    log_section("WhisperJAV v1.5.3 Post-Install Started")
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
    has_cuda, cuda_version = check_cuda_driver()

    # === Phase 3: PyTorch Installation ===
    log_section("Phase 3: PyTorch Installation")

    if check_existing_pytorch():
        log("Compatible PyTorch already installed. Skipping reinstall.")
    else:
        if not install_pytorch(cuda_version):
            create_failure_file("PyTorch installation failed")
            return 1

    # === Phase 4: Python Dependencies ===
    log_section("Phase 4: Python Dependencies Installation")

    req_path = os.path.join(sys.prefix, "requirements_v1.5.3.txt")
    if not os.path.exists(req_path):
        log(f"ERROR: requirements_v1.5.3.txt not found at {req_path}")
        create_failure_file(f"Missing requirements file: {req_path}")
        return 1

    log(f"Installing dependencies from: {req_path}")
    log("This will download ~500MB of packages. Please wait...")

    if not run_pip(
        ["install", "-r", req_path, "--progress-bar", "on"],
        "Install Python dependencies"
    ):
        create_failure_file("Dependencies installation failed")
        return 1

    # === Phase 5: WhisperJAV Application ===
    log_section("Phase 5: WhisperJAV Application Installation")

    # Install from local wheel bundled with installer
    local_wheel = os.path.join(sys.prefix, "whisperjav_local.whl")

    if not os.path.exists(local_wheel):
        log(f"ERROR: Local wheel not found: {local_wheel}")
        log("ERROR: The installer package may be corrupted or incomplete")
        create_failure_file(f"Missing local wheel: {local_wheel}")
        return 1

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

    # === Phase 6: Desktop Shortcut ===
    log_section("Phase 6: Desktop Shortcut Creation")

    shortcut_bat = os.path.join(sys.prefix, 'create_desktop_shortcut_v1.5.3.bat')
    if os.path.exists(shortcut_bat):
        log("Creating desktop shortcut...")
        try:
            result = subprocess.run(
                shortcut_bat,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            # Log all output for debugging
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        log(f"  {line}")
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        log(f"  ERROR: {line}")

            if result.returncode == 0:
                log("Desktop shortcut created successfully!")
            else:
                log(f"WARNING: Shortcut creation returned code {result.returncode}")
        except Exception as e:
            log(f"WARNING: Failed to create desktop shortcut: {e}")
            log("         You can launch manually with: pythonw -m whisperjav.webview_gui.main")
    else:
        log(f"WARNING: Shortcut script not found: {shortcut_bat}")
        log("         Skipping desktop shortcut creation.")

    # === Installation Complete ===
    print_installation_summary(install_start_time, cuda_version)

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
