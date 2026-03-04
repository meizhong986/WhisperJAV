v1.8.6 is a feature + stability release. Highlights: VTT output format, M4B audiobook support, encoding fixes for non-English Windows, auto-tuned LLM batch sizes, serial ensemble mode, anime-whisper backend with Kotoba model support, TEN VAD tuning, ensemble presets, and Linux/macOS upgrade support.

---
### What's Changed

| Enhancement | Description |
|------|-------------|
| **VTT output format (#143)** | `--output-format srt/vtt/both` CLI option + GUI dropdown in Advanced Options. Converts SRT to WebVTT for HTML5 video players. Works with all pipelines including ensemble. |
| **M4B audiobook support (#194)** | `.m4b` added to allowed input extensions. FFmpeg handles the format natively — no extra dependencies. |
| **Ensemble serial mode (#179)** | `--ensemble-serial` completes each file (Pass 1 → Pass 2 → Merge) before starting the next. Delivers results incrementally instead of batching all files. GUI checkbox in merge strategy row. |
| **Anime-Whisper backend** | New generator for the anime-whisper HuggingFace model. Greedy decoding, safe token cap, TEN VAD segmenter. GUI integration with ChronosJAV branding. Now also supports **Kotoba v2.0** and **Kotoba v2.1** as alternative model options in the GUI dropdown. |
| **TEN VAD tuning** | Retuned TEN VAD defaults for better segment granularity: threshold 0.20→0.26, end padding 200→100ms, min speech duration 100→81ms. Fixes oversized 15-36s subtitle segments observed with Kotoba models while preserving sensitivity for short utterances. |
| **Ensemble parameter presets** | Save, load, and delete named ensemble configurations. Cross-pipeline loading, badge names, persistent storage in settings file. |
| **GUI settings persistence** | Initial framework for persisting pipeline and ensemble settings across sessions. Backup rotation, atomic writes. Also adds CLI quality knobs and longest merge strategy. |
| **Linux/macOS upgrade tool** | `whisperjav-upgrade` now detects pip-based installations on Linux/macOS. Platform-appropriate error messages and step labels instead of Windows-only assumptions. |

---

### Bug Fixes

| Issue | What Happened | Fixed |
|-------|---------------|-------|
| #190 | GBK codec crash on Chinese/Japanese Windows during local LLM translation | Process-wide UTF-8 mode (`PYTHONIOENCODING=utf-8`) covers PySubtrans internals |
| #183 (secondary) | "API token limit" + "No matches" with local LLM translation | Auto-cap batch size when model context window is small |
| #183 (upgrade) | `whisperjav-upgrade --wheel-only` shows `[X] installation not found` on Linux | Cross-platform Python/pip detection helpers |
| Anime-whisper | Generator crash, forced max_new_tokens, wrong decoding defaults | Rewritten with low-level API, greedy decoding, safe token cap |

### Breaking Changes

None.

---

## Installation Guide

### Upgrading from v1.8.5

Same dependency set — safe upgrade:

```bash
whisperjav-upgrade
```

Or wheel-only (code changes only, no dependency reinstall):

```bash
whisperjav-upgrade --wheel-only
```

**New:** `whisperjav-upgrade` now works on Linux and macOS pip-based installations.

---

### Windows -- Standalone Installer (Most Users)

The easiest way. No Python knowledge needed.

Recommended: Uninstall the old version first (Settings > Apps > WhisperJAV), then install fresh. Your models and output files are stored separately and won't be lost.

1. **Download:** WhisperJAV-1.8.6-Windows-x86_64.exe from below
2. **Run the installer.** No admin rights required. Installs to `%LOCALAPPDATA%\WhisperJAV`.
3. **Wait 10-20 minutes.** It downloads and configures Python, PyTorch, FFmpeg, and all dependencies.
4. **Launch** from the Desktop shortcut.
5. **First run** downloads models (~3 GB, another several minutes).

**GPU auto-detection:** The installer checks your NVIDIA driver version and picks the right PyTorch:
- Driver 570+ gets CUDA 12.8 (optimal for RTX 20/30/40/50-series)
- Driver 450-569 gets CUDA 11.8 (broad compatibility)
- No NVIDIA GPU gets CPU-only mode

---

### Windows -- Source Install (Developers)

For people who manage their own Python environments.

**Prerequisites:** Python 3.10-3.12, Git, FFmpeg in PATH.

```batch
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

:: Full automated install (auto-detects GPU)
installer\install_windows.bat

:: Or with options:
installer\install_windows.bat --cpu-only        :: Force CPU
installer\install_windows.bat --cuda118         :: Force CUDA 11.8
installer\install_windows.bat --cuda128         :: Force CUDA 12.8
installer\install_windows.bat --local-llm       :: Include local LLM translation
```

The installer runs in 5 phases: PyTorch first (with GPU detection), then scientific stack, Whisper packages, audio/CLI tools, and optional extras. This order matters -- PyTorch must be installed before anything that depends on it, or you end up with CPU-only wheels.

For the full walkthrough, see [docs/guides/installation_windows_python.md](docs/guides/installation_windows_python.md).

---

### macOS (Apple Silicon)

**Prerequisites:**
```bash
xcode-select --install                    # Xcode Command Line Tools
brew install python@3.12 ffmpeg git       # Or python@3.11
```

**Install:**
```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# Create a virtual environment (required for Homebrew Python)
python3 -m venv ~/venvs/whisperjav
source ~/venvs/whisperjav/bin/activate

# Run the macOS installer
chmod +x installer/install_mac.sh
./installer/install_mac.sh
```

**GPU acceleration:** Apple Silicon (M1/M2/M3/M4/M5) gets MPS acceleration automatically for Whisper pipelines. Use `--mode transformers` for best performance. The `balanced`, `fast`, and `faster` modes use CTranslate2 which doesn't support MPS, so those fall back to CPU.

**Qwen pipeline on Mac:** Currently runs on CPU only. The forced aligner doesn't detect MPS yet. This is a known limitation we plan to fix.

**Intel Macs:** CPU-only. No GPU acceleration available.

For the full walkthrough, see [docs/guides/installation_mac_apple_silicon.md](docs/guides/installation_mac_apple_silicon.md).

---

### Linux (Ubuntu, Debian, Fedora, Arch)

**1. Install system packages first** -- these can't come from pip:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv python3-dev \
    build-essential ffmpeg git libsndfile1 libsndfile1-dev

# Fedora / RHEL
sudo dnf install -y python3 python3-pip python3-devel gcc gcc-c++ \
    ffmpeg git libsndfile libsndfile-devel

# Arch
sudo pacman -S --noconfirm python python-pip base-devel ffmpeg git libsndfile
```

For the GUI, you'll also need WebKit2GTK (`libwebkit2gtk-4.0-dev` on Ubuntu, `webkit2gtk4.0-devel` on Fedora).

**2. Install WhisperJAV:**

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# Recommended: use the install script
chmod +x installer/install_linux.sh
./installer/install_linux.sh

# With options:
./installer/install_linux.sh --cpu-only
./installer/install_linux.sh --local-llm
```

**NVIDIA GPU:** You need the NVIDIA driver (450+ or 570+) but NOT the CUDA Toolkit -- PyTorch bundles its own CUDA runtime.

**PEP 668 note:** If your distro's Python is "externally managed" (Ubuntu 24.04+, Fedora 38+), you'll need a virtual environment. The install script detects this and tells you what to do.

For the full walkthrough including Colab/Kaggle setup, headless servers, and systemd services, see [docs/guides/installation_linux.md](docs/guides/installation_linux.md).

---

### Google Colab / Kaggle

The notebooks are not updated yet for this release.

---

## Full Changelog

**17 commits since v1.8.5**

[View full comparison](https://github.com/meizhong986/WhisperJAV/compare/v1.8.5...v1.8.6)
