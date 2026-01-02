# WhisperJAV v1.7.5 Release Notes

**Release Date:** January 2026

---

## New Features

### In-App Upgrade System (Experimental)

A complete upgrade system that checks for new versions and allows one-click updates.

| Interface | Command |
|-----------|---------|
| **GUI** | Notification appears on startup when updates are available |
| **CLI** | `whisperjav-upgrade --check` |
| **CLI** | `whisperjav-upgrade --yes` (auto-confirm) |
| **CLI** | `whisperjav-upgrade --wheel-only` (skip dependencies) |

**Update Notification Intervals:**
- Critical/security updates: Always shown
- Major updates: Shown every launch
- Minor updates: Shown monthly
- Patch updates: Shown weekly (dismissable)

> **Note:** This feature is experimental. Backup important files before upgrading.

---

### Resumable Processing

| Feature | Description |
|---------|-------------|
| **Translation Resume** | Interrupted translations continue from where they stopped |
| **Batch Skip** | `--skip-existing` flag skips files that already have subtitles |

---

### GPU Auto-Detection

Installer and runtime now automatically detect GPU availability and fall back to CPU-only mode when no compatible GPU is found.

---

### Installation Scripts

New standalone scripts for source builds with automatic CUDA detection:

| Script | Platform | CUDA Support |
|--------|----------|--------------|
| `install_windows.bat` | Windows | Auto-detect (11.8, 12.1, 12.4, 12.6, 12.8) |
| `install_linux.sh` | Linux | Auto-detect (11.8, 12.1, 12.4, 12.6, 12.8) |

---

## Bug Fixes

| Issue | Description |
|-------|-------------|
| [#97](https://github.com/meizhong986/WhisperJAV/issues/97) | Fixed FFmpeg path not found in installer |
| [#96](https://github.com/meizhong986/WhisperJAV/issues/96) | Added GUI documentation for tab relationships and settings persistence |
| [#95](https://github.com/meizhong986/WhisperJAV/issues/95) | Fixed CUDA driver compatibility - falls back to CPU if incompatible |
| [#94](https://github.com/meizhong986/WhisperJAV/issues/94) | Fixed Semantic options in Advanced tab |
| [#92](https://github.com/meizhong986/WhisperJAV/issues/92) | Fixed version number display in GUI |
| - | Fixed installer shortcut not launching application |

---

## Installer Improvements

- Improved Windows installer robustness and completeness
- Added `fsspec>=2025.3.0` constraint to prevent unnecessary downgrades
- Added progress indication during pip install phase
- NSIS now handles all shortcut creation (proper separation of concerns)
- Added support for CUDA 12.8 (driver 570+)
- Fixed desktop shortcut pointing to correct executable (`WhisperJAV-GUI.exe`)

---

## Documentation Updates

- GUI documentation for tab relationships and settings persistence
- Speech enhancement CLI documentation
- Updated installation section with new scripts and features

---

## Installation

> **Note for v1.7.4 users:** Version 1.7.4 had incomplete dependencies. Please follow the instructions below to ensure a clean upgrade.

Choose your platform below:

---

### I'm a Windows User

1. Download: **[WhisperJAV-1.7.5-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/tag/v1.7.5)**
2. Run the installer
3. If upgrading, it will guide you to uninstall the old version first

> **What gets preserved during upgrade:** AI models (~3GB), downloaded packages (pip cache), and settings. Reinstalls are much faster than first-time installs.

No Python knowledge required. The installer includes everything.

---

### I'm on Mac or Linux

Use the install script (recommended over pip):

```bash
# Install system dependencies first (see README for details)
# macOS: brew install python@3.11 ffmpeg git
# Linux: apt-get install python3-dev ffmpeg libsndfile1 git

# Clone and install
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

> **Why use the script instead of pip?** The script auto-detects your GPU, selects the correct CUDA/MPS version, and installs dependencies in the right order to avoid conflicts.

**Upgrading from v1.7.4 or earlier:**
```bash
cd whisperjav
git pull
./installer/install_linux.sh
```

---

### I'm a Python Developer

<details>
<summary><b>Manual pip install</b> (only if you know what you're doing)</summary>

⚠️ **Warning:** Manual pip often fails due to dependency conflicts. Use the install scripts unless you have a specific reason not to.

```bash
# Install PyTorch first (choose your platform)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124  # NVIDIA
pip install torch torchaudio  # Apple Silicon
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu  # CPU only

# Then install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git@main
```

</details>

<details>
<summary><b>Windows source install</b></summary>

```batch
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
installer\install_windows.bat
```

</details>

---

## Technical Details

### New Files
- `whisperjav/version_checker.py` - Version checking and update notifications
- `whisperjav/upgrade.py` - Self-upgrade CLI tool
- `tests/test_upgrade_version_checker.py` - 44 tests for version checker
- `tests/test_upgrade_cli.py` - 36 tests for upgrade CLI

### Test Coverage
- 80 automated tests for upgrade system

---

## Known Issues

- Self-upgrade on Windows may require Administrator privileges if installed in Program Files
- Update notifications require GUI restart after upgrade

---

## Contributors

- MeiZhong - Development and testing
- Claude (Anthropic) - Code assistance and documentation

---

**Questions or Issues?** [Open a GitHub issue](https://github.com/meizhong986/WhisperJAV/issues)
