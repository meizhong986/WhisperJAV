import subprocess
import sys
import os
import re
import time
import traceback
from datetime import datetime

LOG_FILE = os.path.join(sys.prefix, "install_log.txt")

def log(message):
    """Logs a message to the console and the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(full_message + "\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")

def check_cuda_driver():
    """Checks if a compatible NVIDIA driver is installed on the system."""
    log("Checking for compatible NVIDIA GPU driver...")
    min_cuda_version = 11.8
    try:
        # The 'nvidia-smi' command is the most reliable way to check for the driver
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True, encoding='utf-8'
        )
        output = result.stdout
        
        # Look for the CUDA Version reported by the driver
        match = re.search(r"CUDA Version: (\d+\.\d+)", output)
        if not match:
            log("ERROR: Could not find CUDA version in 'nvidia-smi' output.")
            log("Please ensure you have a modern NVIDIA driver installed.")
            return False

        driver_cuda_version = float(match.group(1))
        log(f"Found NVIDIA driver with support for CUDA {driver_cuda_version}.")

        if driver_cuda_version < min_cuda_version:
            log(f"ERROR: Your NVIDIA driver is too old. WhisperJAV requires a driver supporting CUDA {min_cuda_version} or newer.")
            log("Please update your NVIDIA Game Ready or Studio driver from the NVIDIA website.")
            return False
        
        log("NVIDIA driver check passed.")
        return True

    except FileNotFoundError:
        log("ERROR: 'nvidia-smi' command not found.")
        log("This means the NVIDIA driver is not installed or not in your system's PATH.")
        log("Please install the latest driver for your NVIDIA GPU.")
        return False
    except subprocess.CalledProcessError as e:
        log(f"ERROR: 'nvidia-smi' failed to run. Return code: {e.returncode}")
        log(f"Output:\n{e.stdout}\n{e.stderr}")
        return False
    except Exception as e:
        log(f"An unexpected error occurred during the NVIDIA driver check: {e}")
        return False


def check_existing_pytorch():
    """Checks if a compatible version of PyTorch is already installed."""
    log("Checking for existing PyTorch installation...")
    min_cuda_version = 11.8
    required_packages = ["torch", "torchvision", "torchaudio"]
    
    try:
        # Check if all required packages are installed
        import importlib
        for pkg in required_packages:
            importlib.import_module(pkg)
        log("Found existing torch, torchvision, and torchaudio.")

        # Now check CUDA availability and version
        import torch
        if not torch.cuda.is_available():
            log("WARNING: PyTorch is installed but CUDA is not available. This may be due to:")
            log("1. Missing NVIDIA drivers")
            log("2. PyTorch installed without CUDA support")
            log("Will proceed with reinstall.")
            return False

        # Check CUDA toolkit version compatibility
        torch_cuda_version = torch.version.cuda
        if torch_cuda_version:
            log(f"PyTorch was built with CUDA version: {torch_cuda_version}")
            if float(torch_cuda_version) >= min_cuda_version:
                log("Existing PyTorch installation is compatible.")
                return True
            else:
                log(f"PyTorch CUDA version is older than required {min_cuda_version}.")
        else:
            log("PyTorch was built without CUDA support.")

        return False

    except ImportError as e:
        log(f"Missing PyTorch package: {e}. Will install.")
        return False
    except Exception as e:
        log(f"Error checking PyTorch: {e}. Will reinstall.")
        return False

def run_command(cmd, description):
    # ... (this function remains unchanged from the previous version) ...
    log(f"Starting: {description}")
    pip_cmd = [os.path.join(sys.prefix, 'python.exe'), '-m', 'pip'] + cmd
    
    for attempt in range(3):
        log(f"Attempt {attempt + 1}/3: {' '.join(pip_cmd)}")
        try:
            result = subprocess.run(
                pip_cmd, capture_output=True, text=True, check=True, encoding='utf-8'
            )
            log(f"Success: {description}")
            if result.stdout:
                log(f"STDOUT:\n{result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            log(f"ERROR on attempt {attempt + 1}: {description} failed!")
            log(f"Return code: {e.returncode}")
            if e.stdout:
                log(f"STDOUT:\n{e.stdout.strip()}")
            if e.stderr:
                log(f"STDERR:\n{e.stderr.strip()}")
        except Exception as e:
            log(f"A non-subprocess exception occurred: {e}")
            log(traceback.format_exc())

        if attempt < 2:
            log("Retrying in 5 seconds...")
            time.sleep(5)
            
    log(f"FATAL: Command failed after 3 attempts: {description}")
    return False

def main():
    log("="*60)
    log("WhisperJAV Post-Install Script Started")
    log("="*60)

    # --- Step 1: Verify system prerequisites (NVIDIA Driver) ---
    if not check_cuda_driver():
        log("FATAL: CUDA missing. WhisperJAV requires GPU with CUDA and CUDNN. Halting installation.")
        # Writing a user-facing error file can be helpful
        with open(os.path.join(sys.prefix, "INSTALLATION_FAILED.txt"), "w") as f:
            f.write("Installation failed because a compatible NVIDIA driver was not found.\n")
            f.write("Please update your GPU drivers from the NVIDIA website and run the installer again.\n")
            f.write("Check the install_log.txt file for more details.\n")
        return 1


    # --- Step 2: Check for and install PyTorch ---
    if check_existing_pytorch():
        log("Pytorch exist. Skipping PyTorch installation.")
    else:
        log("Didn't find cuda enabled pytorch. Proceeding with new PyTorch installation.")
        if not run_command([
            "install", "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ], "PyTorch for CUDA 12.4"):
            return 1
            


    # --- Step 3: Install all other dependencies from requirements.txt ---
    req_path = os.path.join(sys.prefix, "requirements.txt")
    if not run_command(["install", "-r", req_path], "dependencies from requirements.txt"):
        return 1


    # --- Step 4: Install WhisperJAV from the GitHub repository ---
    # Make sure to replace YOUR_USERNAME/YOUR_REPONAME with your actual details
    repo_url = "git+https://github.com/meizhong986/WhisperJAV.git"
    log(f"Installing WhisperJAV from GitHub: {repo_url}...")
    if not run_command(["install", "--no-deps", repo_url], "WhisperJAV application from GitHub"):
        return 1
    
    # --- Step 5: Desktop integration without building an EXE ---
    try:
        # Remove any previously built EXE to avoid AV false positives
        exe_path = os.path.join(sys.prefix, 'WhisperJAV.exe')
        if os.path.exists(exe_path):
            try:
                os.remove(exe_path)
                log("Removed existing WhisperJAV.exe to avoid antivirus false positives.")
            except Exception as e:
                log(f"WARNING: Failed to remove WhisperJAV.exe: {e}")

        # Create desktop shortcut (the script now points to pythonw.exe -m whisperjav.gui.whisperjav_gui)
        shortcut_bat = os.path.join(sys.prefix, 'create_desktop_shortcut.bat')
        if os.path.exists(shortcut_bat):
            log("Creating desktop shortcut...")
            sc = subprocess.run(shortcut_bat, shell=True)
            if sc.returncode == 0:
                log("Desktop shortcut created.")
            else:
                log(f"WARNING: Desktop shortcut script returned code {sc.returncode}.")
        else:
            log("WARNING: create_desktop_shortcut.bat not found; skipping shortcut creation.")
    except Exception as e:
        log(f"WARNING: Exception during desktop shortcut creation: {e}")
        log(traceback.format_exc())


    log("\n" + "="*60)
    log("Post-install script completed successfully.")
    log("="*60)
    return 0

if __name__ == "__main__":
    try:
        # A final pause can help users see the error message in the console before it closes
        exit_code = main()
        if exit_code != 0:
            log("Installation failed. Please review the log. The window will close in 30 seconds.")
            time.sleep(30)
        sys.exit(exit_code)
    except Exception as e:
        log(f"FATAL SCRIPT ERROR: An unexpected error occurred: {e}")
        log(traceback.format_exc())
        time.sleep(30)
        sys.exit(1)