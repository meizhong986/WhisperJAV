# WhisperJAV Linux Installation Guide

**Version:** 1.8.3
**Last Updated:** 2026-02-10
**Platforms:** Ubuntu, Debian, Fedora, RHEL, Arch Linux, Google Colab, Kaggle

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
   - [Ubuntu / Debian](#ubuntu--debian)
   - [Fedora / RHEL / CentOS Stream](#fedora--rhel--centos-stream)
   - [Arch Linux / Manjaro](#arch-linux--manjaro)
3. [NVIDIA Driver and CUDA Setup](#nvidia-driver-and-cuda-setup)
4. [Installation Methods](#installation-methods)
   - [Method 1: Source Installation (Recommended)](#method-1-source-installation-recommended)
   - [Method 2: pip Install with Extras](#method-2-pip-install-with-extras)
   - [Method 3: Conda Environment](#method-3-conda-environment)
5. [GPU Verification](#gpu-verification)
6. [Installing Specific Extras](#installing-specific-extras)
7. [Headless Server Setup](#headless-server-setup)
8. [Google Colab Setup](#google-colab-setup)
9. [Kaggle Setup](#kaggle-setup)
10. [Running the Application](#running-the-application)
11. [Systemd Service Setup](#systemd-service-setup)
12. [Troubleshooting](#troubleshooting)
13. [Performance Tuning](#performance-tuning)
14. [Uninstallation](#uninstallation)

---

## System Requirements

### Hardware

| Component | Minimum | Recommended | Qwen3-ASR |
|-----------|---------|-------------|-----------|
| CPU | 4 cores (x86_64) | 8+ cores | 8+ cores |
| RAM | 8 GB | 16 GB | 32 GB |
| GPU VRAM | 4 GB (basic) | 8 GB | 16+ GB |
| Disk Space | 15 GB (install) | 50 GB (install + models + temp) | 50+ GB |
| Network | Required for install | Broadband for model downloads | 3-10 GB model downloads |

### Supported GPUs

| GPU Family | VRAM | Recommended Mode | Notes |
|-----------|------|-----------------|-------|
| RTX 4090/4080/4070 | 12-24 GB | All modes, Qwen3-ASR | Best performance |
| RTX 3090/3080/3070 | 8-24 GB | All modes, Qwen3-ASR | Excellent |
| RTX 3060/3050 | 6-12 GB | Balanced, Fast | Qwen possible with 12 GB |
| RTX 2080/2070/2060 | 6-11 GB | Balanced, Fast | Good |
| GTX 1080 Ti/1070 | 8-11 GB | Balanced, Fast | Adequate |
| Tesla V100/A100 | 16-80 GB | All modes | Data center GPUs |
| No GPU (CPU only) | N/A | Faster mode only | 10-50x slower |

### Software

| Component | Requirement | Notes |
|-----------|------------|-------|
| Linux Kernel | 4.15+ | 5.4+ recommended for modern NVIDIA drivers |
| Python | 3.10, 3.11, or 3.12 | 3.9 and 3.13+ are NOT supported |
| NVIDIA Driver | 450+ (cu118) or 570+ (cu128) | Required for GPU acceleration |
| FFmpeg | 4.0+ | Required for audio/video processing |
| Git | 2.0+ | Required for installing packages from GitHub |
| GCC / build-essential | Any recent version | Required for compiled extensions |

---

## Prerequisites

Install the following system packages BEFORE running the WhisperJAV installer. These are system-level libraries that cannot be installed by pip.

### Ubuntu / Debian

```bash
# Update package lists
sudo apt-get update

# Essential: Python, build tools, FFmpeg, Git
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ffmpeg \
    git

# Audio processing libraries
sudo apt-get install -y \
    libsndfile1 \
    libsndfile1-dev

# Optional: For TEN VAD native library
sudo apt-get install -y libc++1 libc++abi1

# Optional: For PyAudio/auditok (microphone input)
sudo apt-get install -y portaudio19-dev

# Optional: For GUI (whisperjav-gui)
sudo apt-get install -y \
    libwebkit2gtk-4.0-dev \
    libgtk-3-dev \
    gir1.2-webkit2-4.0
```

**Ubuntu 20.04 (Focal) users:** The default Python is 3.8, which is too old. Install Python 3.10+ from the deadsnakes PPA:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
# Use python3.11 instead of python3 in all subsequent commands
```

### Fedora / RHEL / CentOS Stream

```bash
# Essential: Python, build tools, FFmpeg, Git
sudo dnf install -y \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    ffmpeg \
    git

# Audio processing libraries
sudo dnf install -y libsndfile libsndfile-devel

# Optional: For PyAudio/auditok
sudo dnf install -y portaudio-devel

# Optional: For GUI
sudo dnf install -y \
    webkit2gtk4.0-devel \
    gtk3-devel
```

**RHEL/CentOS:** FFmpeg is not in the default repos. Enable RPM Fusion first:

```bash
# RHEL 9 / CentOS Stream 9
sudo dnf install -y \
    https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-9.noarch.rpm \
    https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-9.noarch.rpm
sudo dnf install -y ffmpeg
```

### Arch Linux / Manjaro

```bash
# Essential: Python, build tools, FFmpeg, Git
sudo pacman -S --noconfirm \
    python \
    python-pip \
    base-devel \
    ffmpeg \
    git

# Audio processing libraries
sudo pacman -S --noconfirm libsndfile

# Optional: For PyAudio/auditok
sudo pacman -S --noconfirm portaudio

# Optional: For GUI
sudo pacman -S --noconfirm webkit2gtk gtk3
```

---

## NVIDIA Driver and CUDA Setup

WhisperJAV uses PyTorch for GPU inference. You need NVIDIA drivers but do NOT need to install the CUDA Toolkit separately -- PyTorch bundles its own CUDA runtime.

### Check Current Driver

```bash
# Check if NVIDIA driver is installed
nvidia-smi
```

If `nvidia-smi` is not found, you need to install NVIDIA drivers.

### Install NVIDIA Drivers

**Ubuntu / Debian:**

```bash
# Method 1: Ubuntu's recommended driver tool (easiest)
sudo ubuntu-drivers autoinstall
sudo reboot

# Method 2: Specific driver version
sudo apt-get install -y nvidia-driver-570
sudo reboot
```

**Fedora:**

```bash
# Enable RPM Fusion repos first (see above), then:
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda
sudo reboot
```

**Arch Linux:**

```bash
sudo pacman -S nvidia nvidia-utils
sudo reboot
```

### Verify Driver Version

After installation and reboot:

```bash
nvidia-smi
```

Look for the driver version in the output. This determines which CUDA version PyTorch will use:

| Driver Version | CUDA Support | PyTorch Index |
|---------------|-------------|---------------|
| 570+ | CUDA 12.8 | Best performance (default) |
| 450-569 | CUDA 11.8 | Universal fallback |
| < 450 | None | CPU only (update drivers!) |

### Data Center / Cloud GPUs

For Tesla, A100, H100, or other data center GPUs, install the data center driver:

```bash
# Ubuntu
sudo apt-get install -y nvidia-headless-570-server nvidia-utils-570-server
sudo reboot
```

---

## Installation Methods

### Method 1: Source Installation (Recommended)

This method uses the automated installer that handles GPU detection, installation ordering, and retry logic.

```bash
# Step 1: Clone the repository
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# Step 2: Create and activate a virtual environment
python3 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate

# Step 3: Run the installer
python install.py
```

The installer will:
1. Check Python version, FFmpeg, Git, disk space, and network
2. Detect your GPU and select the optimal CUDA version
3. Install PyTorch with GPU support (or CPU fallback)
4. Install all dependencies in the correct order
5. Install WhisperJAV
6. Verify the installation

**Installer Options:**

```bash
# CPU-only (no GPU)
python install.py --cpu-only

# Force specific CUDA version
python install.py --cuda118     # For older drivers (450+)
python install.py --cuda128     # For modern drivers (570+)

# Skip optional features
python install.py --no-speech-enhancement
python install.py --minimal     # Transcription only

# Include local LLM translation
python install.py --local-llm          # Prebuilt wheel (fast)
python install.py --local-llm-build    # Build from source (slow)
python install.py --no-local-llm       # Skip without prompting

# Development mode (editable install)
python install.py --dev

# Skip preflight checks
python install.py --skip-preflight
```

**Alternative: Use the shell wrapper:**

```bash
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

The shell wrapper checks for PEP 668 (externally-managed Python) and delegates to `install.py`.

### Method 2: pip Install with Extras

If you want more control over what gets installed, use pip directly. You MUST install PyTorch first.

```bash
# Step 1: Create and activate a virtual environment
python3 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate

# Step 2: Upgrade pip
pip install --upgrade pip

# Step 3: Install PyTorch with CUDA (MUST BE FIRST!)
# For driver 570+:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# For driver 450-569:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 4: Install WhisperJAV with desired extras
pip install "whisperjav[cli] @ git+https://github.com/meizhong986/whisperjav.git"

# Or install from local clone:
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
pip install -e ".[cli]"
```

**IMPORTANT:** Always install PyTorch FIRST with `--index-url` before installing WhisperJAV. If you skip this step, pip will install CPU-only PyTorch and you will get 10-50x slower performance.

### Method 3: Conda Environment

```bash
# Step 1: Create conda environment
conda create -n whisperjav python=3.11 -y
conda activate whisperjav

# Step 2: Install PyTorch via conda (handles CUDA automatically)
conda install pytorch torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
# Or for CUDA 11.8:
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Step 3: Install system deps that conda doesn't provide
conda install ffmpeg -c conda-forge -y

# Step 4: Install WhisperJAV
pip install "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git"
# Or from local clone:
cd whisperjav
pip install -e ".[all]"
```

---

## GPU Verification

After installation, verify GPU support is working:

```bash
# Quick check: Is CUDA available?
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No GPU')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Full diagnostic:
python3 -m whisperjav.utils.preflight_check -v

# Device detection report:
python3 -m whisperjav.utils.device_detector
```

Expected output for a working GPU setup:

```
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
CUDA version: 12.8
```

If CUDA shows False, see [Troubleshooting: CUDA Not Detected](#cuda-not-detected).

---

## Installing Specific Extras

WhisperJAV uses a modular extras system. Install only what you need:

```bash
# After activating your venv and installing PyTorch first:

# CLI only (transcription, no GUI)
pip install "whisperjav[cli] @ git+https://github.com/meizhong986/whisperjav.git"

# CLI + Translation
pip install "whisperjav[cli,translate] @ git+https://github.com/meizhong986/whisperjav.git"

# CLI + GUI
pip install "whisperjav[cli,gui] @ git+https://github.com/meizhong986/whisperjav.git"

# CLI + Qwen3-ASR (large model, needs 8+ GB VRAM)
pip install "whisperjav[cli,qwen] @ git+https://github.com/meizhong986/whisperjav.git"

# Unix-optimized (CLI + translate + enhance + huggingface, no GUI)
pip install "whisperjav[unix] @ git+https://github.com/meizhong986/whisperjav.git"

# Everything
pip install "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git"
```

### Available Extras

| Extra | Description | System Deps Required |
|-------|-------------|---------------------|
| `cli` | Audio processing, VAD, scene detection | libsndfile |
| `gui` | PyWebView GUI interface | libwebkit2gtk-4.0-dev, libgtk-3-dev |
| `translate` | AI subtitle translation (cloud APIs) | None |
| `llm` | Local LLM server (FastAPI) | None |
| `enhance` | Speech enhancement (ClearVoice, BS-RoFormer) | libsndfile |
| `huggingface` | HuggingFace Transformers integration | None |
| `qwen` | Qwen3-ASR pipeline (requires huggingface) | None (8+ GB VRAM recommended) |
| `analysis` | Visualization and analysis tools | None |
| `compatibility` | pyvideotrans integration | None |
| `all` | Everything combined | All of the above |
| `unix` | CLI + translate + enhance + huggingface + analysis + compatibility | libsndfile |
| `colab` | Optimized for Google Colab | N/A (Colab pre-installs most) |
| `kaggle` | Optimized for Kaggle | N/A |
| `dev` | Development tools (pytest, ruff) | None |

---

## Headless Server Setup

For servers without a display (SSH-only, cloud VMs, CI/CD):

```bash
# Step 1: Install prerequisites (no GUI packages needed)
sudo apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential ffmpeg git libsndfile1

# Step 2: Create venv
python3 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate

# Step 3: Install PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Step 4: Install WhisperJAV (unix extra = no GUI deps)
pip install "whisperjav[unix] @ git+https://github.com/meizhong986/whisperjav.git"

# Or use the installer with minimal flag:
python install.py --minimal
```

**Key points for headless operation:**
- Use the `[unix]` extra or `[cli]` extra instead of `[all]` to skip GUI dependencies
- The GUI (`whisperjav-gui`) requires a display server and WebKit2GTK -- skip it on servers
- CLI mode (`whisperjav`) works fully headless
- Set `MPLBACKEND=Agg` if matplotlib warnings appear (no display for plots)

---

## Google Colab Setup

WhisperJAV includes a dedicated Colab installer that handles all setup automatically.

### Quick Start

In a Colab notebook cell:

```python
# Cell 1: Clone and install
!git clone https://github.com/meizhong986/WhisperJAV.git
!bash WhisperJAV/installer/install_colab.sh
```

```python
# Cell 2: Upload or mount your video
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 3: Transcribe
!MPLBACKEND=Agg /content/whisperjav_env/bin/whisperjav \
    /content/drive/MyDrive/video.mp4 \
    --mode balanced \
    --sensitivity aggressive
```

### What the Colab Installer Does

1. Installs `uv` package manager (10-100x faster than pip)
2. Creates an isolated virtual environment at `/content/whisperjav_env`
3. Installs PyTorch with CUDA support matching Colab's GPU
4. Installs system libraries (portaudio, libsndfile, ffmpeg, libc++)
5. Installs WhisperJAV with all extras including Qwen3-ASR
6. Attempts to install llama-cpp-python from prebuilt wheels (optional)

### Colab Tips

- Use `MPLBACKEND=Agg` to avoid matplotlib display errors
- Mount Google Drive to save output subtitles persistently
- Source the aliases file for shorter commands:
  ```bash
  !source /content/whisperjav_aliases.sh
  ```
- Debug mode: `!bash WhisperJAV/installer/install_colab.sh --debug`

---

## Kaggle Setup

Similar to Colab, but use the pip-based approach:

```python
# Cell 1: Install
!pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
!pip install "whisperjav[kaggle] @ git+https://github.com/meizhong986/whisperjav.git"

# Cell 2: Verify
!python -c "import whisperjav; print(whisperjav.__version__)"

# Cell 3: Transcribe
!whisperjav /kaggle/input/your-dataset/video.mp4 --mode balanced
```

---

## Running the Application

### CLI Usage

```bash
# Activate your virtual environment first
source ~/.venv/whisperjav/bin/activate

# Basic transcription
whisperjav video.mp4

# With mode and sensitivity
whisperjav video.mp4 --mode balanced --sensitivity aggressive

# Faster mode (less accurate, quick)
whisperjav video.mp4 --mode faster

# With speech enhancement
whisperjav video.mp4 --mode balanced --enhance

# With Qwen3-ASR pipeline (requires [qwen] extra)
whisperjav video.mp4 --mode qwen --input-mode assembly

# With translation
whisperjav video.mp4 --translate --translate-provider deepseek

# Batch processing (all .mp4 files in directory)
whisperjav /path/to/videos/ --mode balanced

# Specify output directory
whisperjav video.mp4 --output-dir /path/to/subtitles/

# Help
whisperjav --help

# Pre-flight environment check
whisperjav --check
```

### GUI Usage

```bash
# Requires [gui] extra and WebKit2GTK
source ~/.venv/whisperjav/bin/activate
whisperjav-gui
```

**Note:** The GUI requires a display server (X11 or Wayland) and WebKit2GTK. It will not work over SSH unless you use X11 forwarding or VNC.

### Translation

```bash
# Translate existing subtitles
whisperjav-translate -i subtitles.srt --provider deepseek

# Translate with specific instructions
whisperjav-translate -i subtitles.srt --provider gemini --instructions standard
```

---

## Systemd Service Setup

For automated/scheduled transcription on a server:

### Create a Service File

```bash
sudo tee /etc/systemd/system/whisperjav-batch.service << 'EOF'
[Unit]
Description=WhisperJAV Batch Transcription
After=network.target

[Service]
Type=oneshot
User=your-username
Group=your-group
WorkingDirectory=/home/your-username
Environment="PATH=/home/your-username/.venv/whisperjav/bin:/usr/local/bin:/usr/bin"
Environment="MPLBACKEND=Agg"
ExecStart=/home/your-username/.venv/whisperjav/bin/whisperjav \
    /data/incoming/ \
    --mode balanced \
    --output-dir /data/subtitles/
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

### Create a Timer for Scheduled Runs

```bash
sudo tee /etc/systemd/system/whisperjav-batch.timer << 'EOF'
[Unit]
Description=Run WhisperJAV batch transcription hourly

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
EOF
```

### Enable and Start

```bash
sudo systemctl daemon-reload
sudo systemctl enable whisperjav-batch.timer
sudo systemctl start whisperjav-batch.timer

# Check status
sudo systemctl status whisperjav-batch.timer

# View logs
journalctl -u whisperjav-batch.service -f
```

---

## Troubleshooting

### CUDA Not Detected

**Symptom:** `torch.cuda.is_available()` returns `False`

**Diagnosis:**

```bash
# Step 1: Check if NVIDIA driver is loaded
nvidia-smi

# Step 2: Check if PyTorch was installed with CUDA
python3 -c "import torch; print(torch.version.cuda)"
# Should print "12.8" or "11.8", NOT "None"

# Step 3: Check driver compatibility
python3 -c "import torch; print(torch.__version__)"
nvidia-smi | head -3
# Compare driver version with CUDA requirements
```

**Common Causes and Fixes:**

| Cause | Fix |
|-------|-----|
| CPU-only PyTorch installed | `pip uninstall torch torchaudio && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128` |
| Driver too old for CUDA 12.8 | Update driver: `sudo apt install nvidia-driver-570` or use `--cuda118` |
| NVIDIA driver not installed | Install driver (see [NVIDIA Driver Setup](#nvidia-driver-and-cuda-setup)) |
| Running in container without GPU passthrough | Pass `--gpus all` to Docker: `docker run --gpus all ...` |
| Nouveau driver loaded instead of nvidia | Blacklist nouveau: `echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf && sudo update-initramfs -u && sudo reboot` |

### Library Not Found Errors

**`OSError: sndfile library not found`**

```bash
# Ubuntu/Debian
sudo apt-get install -y libsndfile1 libsndfile1-dev

# Fedora/RHEL
sudo dnf install -y libsndfile libsndfile-devel

# Arch
sudo pacman -S libsndfile
```

**`ModuleNotFoundError: No module named '_tkinter'`**

```bash
# Ubuntu/Debian
sudo apt-get install -y python3-tk

# Fedora
sudo dnf install -y python3-tkinter
```

**`ImportError: libwebkit2gtk-4.0.so: cannot open shared object file`**

The GUI requires WebKit2GTK. For CLI-only use, this is not needed.

```bash
# Ubuntu/Debian
sudo apt-get install -y libwebkit2gtk-4.0-dev

# Fedora
sudo dnf install -y webkit2gtk4.0-devel
```

### Permission Denied

**`ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied`**

You are trying to install to the system Python without a virtual environment. Create one:

```bash
python3 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate
# Now retry installation
```

**`error: externally-managed-environment`** (PEP 668)

Same solution -- create and activate a virtual environment. This error appears on Debian 12+, Ubuntu 24.04+, and similar modern distributions.

If `python3 -m venv` fails with "No module named venv":

```bash
# Ubuntu/Debian
sudo apt-get install -y python3-venv
# Or for a specific version:
sudo apt-get install -y python3.11-venv
```

### Git Timeout / Network Issues

**Symptom:** `Failed to connect to github.com port 443 after 21 ms`

This commonly occurs behind the Great Firewall (GFW) or slow VPN connections.

```bash
# Option 1: The installer auto-configures Git timeouts on retry
# Just run the installer again -- it detects and handles this

# Option 2: Manually configure Git
git config --global http.connectTimeout 120
git config --global http.timeout 300
git config --global http.maxRetries 5

# Option 3: Use a proxy
export https_proxy=http://your-proxy:port
export http_proxy=http://your-proxy:port
```

### PyTorch Version Mismatch

**Symptom:** `RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED` or similar

```bash
# Check current versions
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Reinstall matching versions
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### numba / llvmlite Errors

**Symptom:** `ImportError: numba needs NumPy 1.x` or `Cannot import llvmlite`

```bash
# Reinstall numpy and numba
pip install "numpy>=1.26.0"
pip install --force-reinstall "numba>=0.59.0"
```

### Speech Enhancement Failures

**Symptom:** modelscope / clearvoice installation fails

```bash
# These packages are optional. Reinstall without them:
python install.py --no-speech-enhancement

# Or install specific backends:
pip install clearvoice       # ClearVoice only
pip install bs-roformer-infer # BS-RoFormer only
```

### Out of Memory (OOM)

**Symptom:** `CUDA out of memory` during transcription

```bash
# Use a smaller model
whisperjav video.mp4 --mode faster

# Reduce batch size for Qwen pipeline
whisperjav video.mp4 --mode qwen --input-mode vad_slicing

# Monitor GPU memory
watch -n 1 nvidia-smi
```

---

## Performance Tuning

### VRAM Management

| GPU VRAM | Recommended Settings |
|----------|---------------------|
| 4 GB | `--mode faster` only |
| 6 GB | `--mode fast` or `--mode balanced` with small model |
| 8 GB | `--mode balanced --sensitivity balanced` |
| 12 GB | `--mode balanced --sensitivity aggressive` |
| 16+ GB | All modes including `--mode qwen --input-mode assembly` |
| 24+ GB | All modes, large batch sizes |

### Qwen3-ASR Specific Tuning

Qwen3-ASR requires significant VRAM. Use input modes based on your GPU:

| Input Mode | VRAM Usage | Quality | Speed |
|-----------|-----------|---------|-------|
| `assembly` | Highest (text gen + alignment separately) | Best for long scenes | Moderate |
| `context_aware` | High (coupled ASR + alignment) | Best for dialogue | Slower |
| `vad_slicing` | Lower (short segments) | Good for noisy audio | Fastest |

```bash
# Assembly mode (recommended for 16+ GB VRAM)
whisperjav video.mp4 --mode qwen --input-mode assembly

# VAD slicing mode (for 8 GB VRAM)
whisperjav video.mp4 --mode qwen --input-mode vad_slicing
```

### Environment Variables

```bash
# Limit GPU memory usage (fraction of total VRAM)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use specific GPU in multi-GPU systems
export CUDA_VISIBLE_DEVICES=0

# Disable matplotlib display (headless servers)
export MPLBACKEND=Agg

# Enable TF32 for faster inference on Ampere+ GPUs
export TORCH_ALLOW_TF32=1
```

### Batch Processing Optimization

For processing many files:

```bash
# Process all .mp4 files
whisperjav /path/to/videos/ --mode balanced

# Use screen/tmux for long-running jobs
tmux new -s whisperjav
whisperjav /path/to/videos/ --mode balanced --sensitivity aggressive
# Ctrl+B, D to detach; tmux attach -t whisperjav to reattach
```

---

## Uninstallation

```bash
# Remove the virtual environment
rm -rf ~/.venv/whisperjav

# Remove cached models (optional, saves disk space)
rm -rf ~/.cache/whisper
rm -rf ~/.cache/huggingface

# Remove desktop entry (if created)
rm -f ~/.local/share/applications/whisperjav.desktop

# Remove source code (if cloned)
rm -rf ~/whisperjav  # Adjust path as needed
```

---

## Appendix: Architecture Overview

### Installation Flow

```
install_linux.sh (thin wrapper)
    |
    v
install.py (orchestrator)
    |
    +-- Preflight checks (disk, network)
    +-- detect_gpu() --> CUDA version selection
    +-- Step 1: pip upgrade
    +-- Step 2: PyTorch (GPU lock-in via --index-url)
    +-- Step 3: Core deps (numpy, scipy, numba, audio libs)
    +-- Step 4: Whisper packages (openai-whisper, stable-ts, faster-whisper)
    +-- Step 5: Optional (HuggingFace, Qwen, translation, VAD, enhancement, GUI)
    +-- Step 6: WhisperJAV (--no-deps to preserve GPU torch)
    +-- Verification
```

### Why PyTorch Must Be Installed First

PyTorch on PyPI is CPU-only. If you run `pip install whisperjav` directly, pip resolves torch from PyPI and you get CPU-only inference (10-50x slower). By installing torch FIRST with `--index-url https://download.pytorch.org/whl/cu128`, the GPU version is "locked in" and subsequent packages see it as already satisfied.

### Package Registry

All package definitions live in `whisperjav/installer/core/registry.py`. This is the single source of truth for:
- Package names and versions
- Installation order (PyTorch first, numba after numpy, etc.)
- Which extras each package belongs to
- Platform-specific packages (Windows-only, Linux-only)
- Import name mapping (e.g., `opencv-python` imports as `cv2`)

When adding or modifying dependencies, update the registry and run validation:

```bash
python -m whisperjav.installer.validation
```
