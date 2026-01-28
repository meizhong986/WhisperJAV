# WhisperJAV v1.8.2rc1 Installation Guide

## Windows Users

**Recommended: Fresh Installation**

1. If you have a previous version, uninstall it first:
   - Settings → Apps → WhisperJAV → Uninstall

2. Download the installer:
   - [WhisperJAV-1.8.2rc1-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/download/v1.8.2rc1/WhisperJAV-1.8.2rc1-Windows-x86_64.exe)

3. Run the installer and follow the prompts

4. When asked "Install local LLM translation? (Y/n)" → Press Enter or type Y

5. Launch from the desktop shortcut

---

## Expert Users (Windows / macOS / Linux)

### Prerequisites

| Platform | Requirements |
|----------|--------------|
| Windows | Python 3.10-3.12, FFmpeg, Git |
| macOS | Xcode Command Line Tools, Homebrew, Python 3.10-3.12, FFmpeg |
| Linux | Python 3.10-3.12, FFmpeg, build-essential, libsndfile1 |

### Installation Steps

**1. Clone the repository**

```bash
git clone https://github.com/meizhong986/WhisperJAV.git
cd WhisperJAV
git checkout v1.8.2rc1
```

**2. Run the installation script**

Windows (Command Prompt or PowerShell):
```cmd
installer\install_windows.bat
```

macOS:
```bash
# Install system dependencies first
xcode-select --install
brew install python@3.11 ffmpeg git

# Run installer
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

Linux (Debian/Ubuntu):
```bash
# Install system dependencies first
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip python3-venv build-essential ffmpeg libsndfile1 git

# Create and activate virtual environment (required on Ubuntu 24.04+)
python3 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate

# Run installer
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

Linux (Fedora/RHEL):
```bash
# Install system dependencies first
sudo dnf install python3-devel gcc ffmpeg libsndfile git

# Run installer
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

**3. Installation options**

The install scripts accept these flags:

```bash
--cpu-only              # Force CPU-only PyTorch (no GPU)
--cuda118               # Use CUDA 11.8 (older drivers)
--cuda128               # Use CUDA 12.8 (default for NVIDIA)
--local-llm             # Include local LLM translation (prebuilt wheel)
--local-llm-build       # Include local LLM translation (build from source)
--minimal               # Minimal install (transcription only)
--dev                   # Development/editable mode
```

Example:
```bash
./installer/install_linux.sh --local-llm
```

---

## Google Colab / Kaggle

Use the updated notebooks from the v1.8.2rc1 release:

- [WhisperJAV Colab Edition](https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/v1.8.2rc1/notebook/WhisperJAV_colab_edition.ipynb)
- [WhisperJAV Colab Expert](https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/v1.8.2rc1/notebook/WhisperJAV_colab_edition_expert.ipynb)
- [WhisperJAV Two-Pass](https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/v1.8.2rc1/notebook/WhisperJAV_colab_parallel.ipynb)
- [WhisperJAV Two-Pass Expert](https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/v1.8.2rc1/notebook/WhisperJAV_colab_parallel_expert.ipynb)

---

## pip Install (Advanced)

For users who want direct pip installation:

```bash
# Create virtual environment
python -m venv whisperjav-env
source whisperjav-env/bin/activate  # Linux/macOS
# whisperjav-env\Scripts\activate   # Windows

# Install PyTorch first (critical for GPU support)
# NVIDIA GPU:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Apple Silicon:
pip install torch torchaudio

# CPU only:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install WhisperJAV
pip install git+https://github.com/meizhong986/WhisperJAV.git@v1.8.2rc1
```

---

## Verifying Installation

After installation, verify it works:

```bash
# Check version
whisperjav --version

# Test transcription (replace with your file)
whisperjav video.mp4 --mode fast

# Test local LLM translation
whisperjav-translate -i subtitles.srt --provider local
```

---

## Troubleshooting

**Local LLM not working?**
- Ensure you said "Y" during installation when prompted for local LLM
- Or reinstall with: `pip install llama-cpp-python[server]`

**DLL errors on Windows?**
- Fresh install is recommended for v1.8.2rc1
- The installer now properly sets up all dependencies

**GPU not detected?**
- Update NVIDIA drivers to 450+ (for CUDA 11.8) or 570+ (for CUDA 12.8)
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`
