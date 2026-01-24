# WhisperJAV v1.8.1 Release Notes

**Release Date:** January 24, 2026
**Type:** Hotfix Release
**Issue:** [#141 - Local LLM DLL Loading Failures](https://github.com/meizhong986/WhisperJAV/issues/141)

---

## Bug Fixes

This hotfix addresses critical issues with local LLM translation:

| Fix | Description | Affected Users |
|-----|-------------|----------------|
| **Linux CUDA library loading** | `libcudart.so.12: cannot open shared object file` even with PyTorch CUDA installed | Linux users with NVIDIA GPUs |
| **Function name mismatch** | `get_llama_cpp_prebuilt_wheel` not found in install.py | Users running `install_linux.sh` or `install_windows.bat` |
| **Edge case handling** | "Successfully installed!" immediately followed by "INSTALLATION FAILED" | Windows users in rare pip/import race conditions |

---

## Installation Guide

### Section 1: Upgrading from v1.8.0 to v1.8.1

> **If you already have v1.8.0 installed, this is all you need.**

The v1.8.1 fixes are in the runtime code (the wheel). A simple upgrade is sufficient - no need for a complete reinstall.

#### Option A: Command Line Upgrade (Recommended)

```bash
whisperjav-upgrade --wheel-only
```

This takes ~30 seconds and updates only the WhisperJAV package while preserving your environment.

#### Option B: GUI Upgrade

1. Open WhisperJAV GUI
2. Go to **Menu → Check for Updates**
3. Click **"Update Now"**

#### Option C: Manual pip Upgrade

```bash
pip install -U git+https://github.com/meizhong986/whisperjav.git
```

**That's it!** Your existing Python environment, PyTorch, and all dependencies remain unchanged.

---

### Section 2: Fresh Installation (New Users & Users from v1.7.x or Earlier)

> **If you are a new user OR upgrading from any version before v1.8.0, follow this complete installation guide.**

v1.8.1 is a hotfix release with the same installation requirements as v1.8.0. Due to breaking dependency changes introduced in v1.8.0 (NumPy 1.x, new PyTorch builds, installer overhaul), a complete fresh installation is required.

---

#### Windows (Recommended - Standalone Installer)

**Best for:** Most users, beginners, and those who want a GUI.

1. **Download the Installer:**
   [**WhisperJAV-1.8.1-Windows-x86_64.exe**](https://github.com/meizhong986/WhisperJAV/releases/download/v1.8.1/WhisperJAV-1.8.1-Windows-x86_64.exe)

2. **Run the Installer:**
   Double-click the downloaded `.exe` file. No admin rights required.

3. **Follow the Prompts:**
   The installer handles all dependencies automatically:
   - Python 3.10.18 with conda
   - FFmpeg for audio/video processing
   - Git for GitHub package installs
   - PyTorch with CUDA support (auto-detected)

4. **Launch:**
   Open "WhisperJAV v1.8.1" from your Desktop shortcut.

**Installation Time:** ~10-20 minutes (internet-dependent)
**First Run:** AI models download (~3GB, 5-10 minutes additional)

<details>
<summary><strong>Upgrading from v1.7.x?</strong> (Click to expand)</summary>

A fresh installation is required due to breaking dependency changes:

1. **Uninstall v1.7.x first:**
   - Open **Settings** → **Apps** → Search "WhisperJAV" → **Uninstall**
   - Or run: `%LOCALAPPDATA%\WhisperJAV\Uninstall-WhisperJAV.exe`

2. **Install v1.8.1:** Run the new installer.

**What's preserved:** Your AI models (`%USERPROFILE%\.cache\huggingface\`), transcription outputs, and custom files are stored outside the installation directory and will not be deleted.

</details>

---

#### Windows (Expert - Source Install)

**Best for:** Developers and advanced users who want more control.

**Prerequisites:**
- Python 3.10-3.12 ([python.org](https://www.python.org/downloads/))
- FFmpeg in PATH ([gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/))
- Git ([git-scm.com](https://git-scm.com/download/win))

```batch
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

:: Standard install (auto-detects GPU)
installer\install_windows.bat

:: Or with options:
installer\install_windows.bat --cpu-only     # Force CPU only
installer\install_windows.bat --cuda118      # Force CUDA 11.8
installer\install_windows.bat --local-llm    # Include local LLM support
```

---

#### macOS (Apple Silicon & Intel)

**Best for:** M1/M2/M3/M4 Macs and Intel Mac users. CLI + GUI supported.

**1. Install Prerequisites**

```bash
# Install Xcode Command Line Tools (required for GUI)
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system tools
brew install python@3.11 ffmpeg git
```

**2. Install WhisperJAV**

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh

# Run the installer (auto-detects Mac architecture)
./installer/install_linux.sh

# With local LLM support (Apple Silicon builds with Metal):
./installer/install_linux.sh --local-llm-build
```

> **Apple Silicon:** Native MPS acceleration enabled automatically.
> **Intel Macs:** CPU-only mode (5-10x slower than Apple Silicon).

---

#### Linux (Ubuntu/Debian/Fedora)

**Best for:** Servers, desktops with NVIDIA GPUs.

**1. Install System Dependencies**

```bash
# Debian / Ubuntu
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip build-essential ffmpeg libsndfile1 git

# Fedora / RHEL
sudo dnf install python3-devel gcc ffmpeg libsndfile git
```

**2. Install WhisperJAV**

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh

# Standard install (auto-detects GPU)
./installer/install_linux.sh

# With options:
./installer/install_linux.sh --cpu-only        # Force CPU only
./installer/install_linux.sh --local-llm       # Include local LLM (prebuilt wheel)
./installer/install_linux.sh --local-llm-build # Include local LLM (build from source)
```

---

#### Google Colab / Kaggle

Use the one-click notebooks:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_parallel_expert.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_parallel_expert.ipynb)

---

#### pip Install (Advanced)

> **Warning:** Manual pip install requires careful dependency management. We strongly recommend using the installer scripts above.

```bash
# 1. Create virtual environment
python -m venv whisperjav-env
source whisperjav-env/bin/activate  # Linux/Mac
# whisperjav-env\Scripts\activate   # Windows

# 2. Install PyTorch FIRST (critical for GPU support)
# NVIDIA GPU:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# Apple Silicon:
pip install torch torchaudio
# CPU only:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git
```

---

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Windows 10/11, macOS 11+, Linux | Latest versions |
| **Python** | 3.10 | 3.11 or 3.12 |
| **RAM** | 8 GB | 16 GB |
| **Disk Space** | 8 GB | 15 GB (with models) |
| **GPU** | Optional | NVIDIA RTX 2060+ or Apple Silicon |

**Supported GPU Acceleration:**
- NVIDIA CUDA 11.8+ (Windows/Linux)
- Apple MPS (macOS Apple Silicon)
- CPU fallback (all platforms, 5-10x slower)

---

## Enabling Local LLM Translation

Local LLM translation allows offline subtitle translation without API keys.

### Automatic Setup (Recommended)

Simply use local translation - llama-cpp-python installs automatically:

```bash
whisperjav video.mp4 --translate --translate-provider local
```

### Troubleshooting

#### Linux: `libcudart.so.12: cannot open shared object file`

The v1.8.1 fix handles this automatically by preloading CUDA libraries from PyTorch. If issues persist:

```bash
# Verify PyTorch CUDA is working
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

#### Windows: DLL Load Failed

The v1.8.1 fix adds PyTorch's CUDA DLLs to the search path. If issues persist:

1. Update NVIDIA drivers to 570+ for CUDA 12.8
2. Or try CPU mode: `--device cpu`

---

## Downloads

| File | Description |
|------|-------------|
| [WhisperJAV-1.8.1-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/download/v1.8.1/WhisperJAV-1.8.1-Windows-x86_64.exe) | Windows Standalone Installer (~292 MB) |
| [whisperjav-1.8.1-py3-none-any.whl](https://github.com/meizhong986/WhisperJAV/releases/download/v1.8.1/whisperjav-1.8.1-py3-none-any.whl) | Python Wheel (for upgrades) |

---

## Acknowledgments

Thanks to the users who reported these issues:
- **@parheliamm** - Linux CUDA library issues and detailed logs
- **@leefowan** - Windows edge case report
- **@justantopair-ai** - Documentation feedback

---

**Full Changelog:** [v1.8.0...v1.8.1](https://github.com/meizhong986/WhisperJAV/compare/v1.8.0...v1.8.1)
