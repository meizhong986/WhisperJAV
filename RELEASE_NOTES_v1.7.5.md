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

### Upgrading from v1.7.4?

If you are upgrading from version 1.7.4, simply run the new installer.

- **Do not uninstall manually.** The installer will detect your previous version.
- **Data Safety:** Your AI models (~3GB), settings, and cached downloads will be **preserved**.
- **Clean Install:** The installer will automatically clean up old dependency conflicts.

---

### Windows (Recommended)

*Best for: Most users, beginners, and those who want a GUI.*

1. **Download the Installer:**
   **[Download WhisperJAV-1.7.5-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/tag/v1.7.5)**
2. **Run the File:** Double-click the downloaded `.exe`.
3. **Follow the Prompts:** The installer handles all dependencies (Python, FFmpeg, Git) automatically.
4. **Launch:** Open "WhisperJAV" from your Desktop shortcut.

> **Note:** The first launch may take a few minutes as it initializes the engine. GPU is auto-detected; CPU-only mode is used if no compatible GPU is found.

---

### macOS (Apple Silicon & Intel)

*Best for: M1/M2/M3/M4 users and Intel Mac users.*

The install script auto-detects your Mac architecture and handles PyTorch dependencies automatically.

**1. Install Prerequisites**

```bash
# Install Xcode Command Line Tools (required for GUI)
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system tools
brew install python@3.11 ffmpeg git
```

> **GUI Requirement:** The Xcode Command Line Tools are required to compile `pyobjc`, which enables the GUI. Without it, only CLI mode will work.

**2. Install WhisperJAV**

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh

# Run the installer (auto-detects Mac architecture)
./installer/install_linux.sh
```

> **Intel Macs:** The script automatically uses CPU-only mode. Expect slower processing (5-10x) compared to Apple Silicon with MPS acceleration.

---

### Linux (Ubuntu/Debian/Fedora)

*Best for: Servers, desktops with NVIDIA GPUs.*

The install script auto-detects NVIDIA GPUs and installs the matching CUDA version.

**1. Install System Dependencies**

```bash
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y python3-dev python3-pip build-essential ffmpeg libsndfile1 git

# Fedora / RHEL
sudo dnf install python3-devel gcc ffmpeg libsndfile git
```

**2. Install WhisperJAV**

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh

# Standard Install (auto-detects GPU)
./installer/install_linux.sh

# Or force CPU-only (for servers without GPU)
./installer/install_linux.sh --cpu-only
```

> **Performance:** A 2-hour video takes ~5-10 minutes on GPU vs ~30-60 minutes on CPU.

---

### Advanced / Developer

*Best for: Contributors and Python experts.*

<details>
<summary><b>Manual pip install</b></summary>

> **Warning:** Manual `pip install` is risky due to dependency conflicts (NumPy 2.x vs SciPy). We strongly recommend using the scripts above.

**1. Create Environment**

```bash
python -m venv whisperjav-env
source whisperjav-env/bin/activate   # Linux/Mac
# whisperjav-env\Scripts\activate    # Windows
```

**2. Install PyTorch First (Critical)**

You must install PyTorch *before* the main package to ensure hardware acceleration works.

- **NVIDIA GPU:** `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`
- **Apple Silicon:** `pip install torch torchaudio`
- **CPU only:** `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`

**3. Install WhisperJAV**

```bash
pip install git+https://github.com/meizhong986/whisperjav.git@main
```

</details>

<details>
<summary><b>Editable / Dev install</b></summary>

Use this if you plan to modify the code.

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# Windows
installer\install_windows.bat --dev

# Mac/Linux
./installer/install_linux.sh --dev

# Or manual
pip install -e ".[dev]"
```

</details>

<details>
<summary><b>Windows source install</b></summary>

```batch
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
installer\install_windows.bat              # Auto-detects GPU
installer\install_windows.bat --cpu-only   # Force CPU only
installer\install_windows.bat --cuda118    # Force CUDA 11.8
installer\install_windows.bat --cuda124    # Force CUDA 12.4
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
