import subprocess
import sys
import os
import traceback
from datetime import datetime

# Create log file in the installation directory
LOG_FILE = os.path.join(sys.prefix, "install_log.txt")

def log(message):
    """Log to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(full_message + "\n")
    except Exception:
        pass

def run_command(cmd, description):
    """Run command with full logging"""
    log(f"Running: {description}")
    log(f"Command: {' '.join(cmd)}")
    
    try:
        # Use the python.exe from the new environment to run pip
        pip_cmd = [os.path.join(sys.prefix, 'python.exe'), '-m', 'pip'] + cmd
        result = subprocess.run(pip_cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        log(f"Success: {description}")
        if result.stdout:
            log(f"STDOUT: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"ERROR: {description} failed!")
        log(f"Return code: {e.returncode}")
        log(f"STDOUT: {e.stdout}")
        log(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        log(f"EXCEPTION: {str(e)}")
        log(traceback.format_exc())
        return False

def main():
    log("="*60)
    log("WhisperJAV Post-Install Script Started")
    log("="*60)
    
    # --- 1. Install PyTorch from its special index ---
    log("Installing PyTorch with CUDA 12.4...")
    if not run_command([
        "install", 
        "torch==2.6.0", 
        "torchvision==0.21.0", 
        "torchaudio==2.6.0",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    ], "PyTorch installation"):
        log("FATAL: PyTorch installation failed. Aborting.")
        return 1
    
    # --- 2. Install regular packages ---
    packages = [
        "numba==0.61.2",
        "soundfile==0.13.1", 
        "ffmpeg-python==0.2.0",
        "pysrt==1.1.2",
        "srt==3.5.3",
        "auditok==0.3.0",
        "faster-whisper==1.1.1",
    ]
    
    log("\n" + "="*40)
    log("Installing pip packages...")
    for pkg in packages:
        if not run_command(["install", pkg], f"Installing {pkg}"):
            log(f"WARNING: Failed to install {pkg}, continuing...")
    
    # --- 3. Install git packages ---
    git_packages = [
        "git+https://github.com/openai/whisper.git@v20231117", # Pinned to a specific commit for stability
        "git+https://github.com/meizhong986/stable-ts-fix-setup.git@main",
        "git+https://github.com/meizhong986/WhisperJAV.git@v1.1.2"
    ]
    
    log("\n" + "="*40)
    log("Installing git packages...")
    for pkg in git_packages:
        if not run_command(["install", pkg], f"Installing {pkg}"):
            log(f"ERROR: Failed to install git package {pkg}. Aborting.")
            return 1
            
    log("\n" + "="*60)
    log("Post-install script completed successfully.")
    log("="*60)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"FATAL SCRIPT ERROR: {str(e)}")
        log(traceback.format_exc())
        sys.exit(1)