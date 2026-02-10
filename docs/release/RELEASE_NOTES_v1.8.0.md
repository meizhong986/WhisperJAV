# WhisperJAV v1.8.0 Release Notes

## Table of Contents

- [What's New in v1.8.0](#whats-new-in-v180)
- [Important: Upgrade Notice](#important-upgrade-notice)
- [Installation](#installation)
  - [Windows (Recommended)](#windows-recommended)
  - [Windows (Expert)](#windows-expert)
  - [macOS (CLI Only)](#macos-cli-only)
  - [Linux](#linux)
  - [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Local LLM Translation](#local-llm-translation)
- [License](#license)

---

## What's New in v1.8.0

This is a **major release** with significant infrastructure changes and new feature. Please make sure you install completely. My recommendation is to uninstall previous version(s) and install this one.

- **Local LLM Translation** — [Thanks to @hyiip] For local translation using llama-cpp. Current version supports unfiltered versions of Llama 3.1 8B, Gemma 2 9B, and llama-3b.

- **Installer Overhaul** — Revamped the installer to use uv.exe. It reduces installation time and simplifies the dependency stack. Cleaner upgrade path for future releases.

- **CUDA 12.8 Default** — Updated to PyTorch with CUDA 12.8 for speed and new accelerators. Fallback to CUDA 11.8 for compatibility to older series.

- **Scene Detection Optimization** — Refactored scene detection pipeline for better memory efficiency and faster processing on long videos.

- **Additional Debug Output** — Making it easier to diagnose issues with specific content.

- **Downgrade to numpy 1.x** — The complexity and reliance to modelscope for future roadmap is easier with numpy 1.x.

---

## Important: Upgrade Notice

> **v1.8.0 is a major release. If you are upgrading from v1.7.x, a complete installation is required.**

This release includes breaking changes to the dependency stack (NumPy 1.x, new PyTorch builds, restructured speech enhancement backends). In-place upgrades will fail.

**To upgrade:**

1. **Uninstall the previous version first**
   - Windows: Settings → Apps → WhisperJAV → Uninstall
   - Mac/Linux: Delete your virtual environment (`rm -rf whisperjav-env`)

2. **Install v1.8.0** using the instructions below

**What's preserved:** Your previously downloaded whisper models and vad models will be preserved. No need for new download.

---

## Installation

Choose the installation method that matches your platform.

### Windows (Recommended)

*Best for: Most users, beginners, GUI users*

1. **Download:** [WhisperJAV-1.8.0-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/latest)
2. **Run** the installer and follow the prompts
3. **Launch** from the Desktop shortcut

The installer handles everything: Python, FFmpeg, Git, PyTorch with GPU support, and all dependencies.

> **First launch:** Takes a few minutes to initialize. GPU is auto-detected; falls back to CPU if no compatible GPU is found.

> **GPU Support:** CUDA 12.8 for RTX 30/40 series, CUDA 11.8 for GTX 10/16 and RTX 20 series, CPU fallback otherwise.

---

### Windows (Expert)

```batch
:: Install prerequisites: Python 3.10-3.12, FFmpeg, Git (add to PATH)

git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
installer\install_windows.bat
```

---

### macOS (CLI Only)

> **Note:** GUI is not available on macOS. Use command line only.

```bash
# Prerequisites
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11 ffmpeg git

# Install
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

- **Apple Silicon (M1/M2/M3/M4):** GPU acceleration via MPS
- **Intel Mac:** CPU only

---

### Linux

```bash
# Prerequisites (Debian/Ubuntu)
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip build-essential ffmpeg libsndfile1 git

# Prerequisites (Fedora/RHEL)
sudo dnf install python3-devel gcc ffmpeg libsndfile git

# Install
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

---

### Prerequisites

| Requirement | Details |
|-------------|---------|
| Python | 3.10–3.12 (3.9 dropped, 3.13+ not yet supported) |
| FFmpeg | Must be in system PATH |
| Disk Space | 8GB+ for installation |
| GPU (optional) | NVIDIA CUDA, Apple MPS, or AMD ROCm |

---

## Quick Start

### GUI (Windows only)

```bash
whisperjav-gui
```

A window opens. Add files, select a mode, click Start.

### Command Line

```bash
# Basic usage
whisperjav video.mp4

# With options
whisperjav video.mp4 --mode balanced --sensitivity aggressive

# Process a folder
whisperjav /path/to/folder --output-dir ./subtitles
```

---

## Local LLM Translation

*New in v1.8.0* — Run translation entirely on your GPU with no cloud API:

```bash
whisperjav-translate -i subtitles.srt --provider local
```

**Zero-config setup:** On first use, WhisperJAV automatically downloads llama-cpp-python (~700MB). No manual installation.

| Model | VRAM | Notes |
|-------|------|-------|
| `llama-8b` | 6GB+ | **Default** — Llama 3.1 8B |
| `gemma-9b` | 8GB+ | Gemma 2 9B |
| `llama-3b` | 3GB+ | Llama 3.2 3B (low VRAM) |
| `auto` | varies | Auto-selects based on VRAM |

```bash
# Use specific model
whisperjav-translate -i subtitles.srt --provider local --model gemma-9b

# Control GPU layers
whisperjav-translate -i subtitles.srt --provider local --translate-gpu-layers 32
```

**Resume support:** If translation is interrupted, run the same command again. It resumes from where it left off.

---

## License

MIT License
