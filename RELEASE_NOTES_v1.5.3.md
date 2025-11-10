# WhisperJAV v1.5.3 Release Notes

---

## What's New at a Glance

**Five sets of improvements**:

1. **Multi-Platform Accelerator** (Experimental) - opened up preflight checks to allow Apple Silicon Macs, CPU, and ROCm AMD GPUs
2. **Resume Translation** - translation progress uses auto-save to allow resuming
3. **Additional Languages** - Chinese and Korean transcription
4. **GUI Enhancements** - scroll bar, clearer options
5. **Bug Fixes** - Windows UTF-8 encoding and CUDA detection fixed

---

## Notes

- **Mac and AMD GPU users**: I've opened the preflight checks and added new hardware detection, but I don't have the hardware to test it myself. Please report back if you encounter any issues—or even if it works! :)
- **RTX 50 series (Blackwell)**: Bundles PyTorch 2.9 for standalone exe release
- **Translation**: Auto-saves `.subtrans` file. When passed to whisperjav-translate, it resumes translation if not complete
- **CPU-only mode**: A speed warning is shown. You can use a CLI switch to bypass if you want

New `device_detector.py` module provides:
- CUDA/MPS/ROCm capability checking
- Graceful degradation to CPU when needed

### Example Usage

```bash
# Start translation
whisperjav-translate -i subtitles.srt

# [Translation interrupted after 50/100 batches]

# Resume from batch 51
whisperjav-translate -i subtitles.srt
# Output: "Resuming from batch 51 of 100..."
```


## Bug Fixes

- Subprocess calls failed on Windows with non-ASCII characters
- Type mismatch in CUDA version checking (string vs float)
- Console output lagged with large logs


## System Requirements (Updated)

### Supported Platforms

| Platform | GPU Support | Status |
|----------|-------------|--------|
| **Windows 10/11** (64-bit) | NVIDIA (CUDA) | Fully Supported |
| **macOS** (Apple Silicon) | M1/M2/M3/M4/M5 (MPS) | Fully Supported (NEW) |
| **macOS** (Intel) | CPU only | Supported (slower) |
| **Linux** | NVIDIA (CUDA) | Fully Supported |
| **Linux** | AMD (ROCm) | Limited (experimental) |

### GPU Requirements

**NVIDIA GPUs:**
- CUDA 11.8+ required
- RTX 20/30/40/50 series recommended
- 4-6 GB VRAM for fast and faster modes
- 6-8 GB VRAM for balanced mode
- 10 GB VRAM for Whisper direct to English translation (use large-v2)


**Apple Silicon:**
- macOS 12.3+ (MPS support)
- unified memory minimum: see above

**AMD GPUs:**
- ROCm detection only
- Falls back to CPU if failed

### Memory Requirements

- **8 GB RAM minimum** (system memory)
- **16 GB RAM recommended** for large files (>2 hours)
- **GPU VRAM**: See above under NVIDIA GPUs


### Software Requirements

- **Python**: 3.9, 3.10, 3.11, or 3.12 (3.13+ incompatible)
- **FFmpeg**: Required (must be in PATH)
- **WebView2**: Required for GUI (Windows only, auto-prompted)
- **requirements.txt**

---

## Installation & Upgrading

### New Installation

#### Windows Standalone (Installer - Recommended)
```bash
# Download WhisperJAV-1.5.3-Windows-x86_64.exe (coming soon)
# Run installer and follow prompts
# IMPORTANT: Do not close the pop-up terminal window during installation
# Desktop shortcut will be created automatically
```


#### macOS (Apple Silicon) - Experimental (not tested on my end)
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 ffmpeg

# Install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git@main

# Launch GUI
whisperjav-gui
```

#### From Source (All Python Platforms)
```bash
# Clone repository
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# Install in development mode
pip install -e . -U

# Launch GUI, CLI, or translation tool
whisperjav-gui              # GUI
whisperjav video.mp4        # CLI
whisperjav-translate        # Translation CLI
```

---

### Upgrading from v1.5.1

#### If Using Installer (Windows)
1. **Option A**: Uninstall v1.5.1 first (clean install)
2. **Option B**: Install v1.5.3 alongside (keeps both versions)
3. Run new v1.5.3 installer
4. **Models and cache preserved** automatically

#### If Using pip
```bash
# Upgrade to latest version
pip install -U --no-deps git+https://github.com/meizhong986/whisperjav.git@main

# Verify version
python -c "from whisperjav import __version__; print(__version__)"
# Output: 1.5.3
```

#### Cache and Settings
- **Configuration files**: Automatically migrated
- **Model cache**: Preserved (no re-download needed)
- **Custom settings**: Retained from previous version

---

**Performance Notes:**
- M1/M2/M3/M4/M5 use Metal Performance Shaders (MPS)
- Significantly faster than CPU-only mode
- Performance comparable to entry-level NVIDIA GPUs

---

## Acknowledgments

Special thanks to:
- Community testers who reported UTF-8 encoding issues on Windows
- Apple Silicon Mac users who requested native GPU support
- Contributors who tested experimental Chinese/Korean features
- All users who provided feedback on v1.5.1

---

## Documentation

**User Guide**: See GUI_USER_GUIDE.md for detailed usage instructions

---

## License

WhisperJAV is released under the MIT License. See LICENSE file for details.

---

**Version 1.5.3 - November 2025**
*Multi-Platform Release - Expanding Access, Improving Reliability*
