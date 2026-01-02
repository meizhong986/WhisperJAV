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

> **Note for v1.7.4 users:** Version 1.7.4 had incomplete dependencies. Please follow the instructions below for your platform to ensure a clean upgrade.

---

### I'm a Windows User (GUI)

**If you installed WhisperJAV using the Windows installer (.exe):**

1. Download the new installer: **[WhisperJAV-1.7.5-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/tag/v1.7.5)**
2. Run it - it will install fresh to `%LOCALAPPDATA%\WhisperJAV`
3. Your AI models are preserved (no re-download needed)

---

### I'm a Mac or Linux User

**First time installing:**
```bash
pip install whisperjav
```

**Upgrading from any previous version:**
```bash
pip uninstall whisperjav -y
pip install whisperjav
```

> We recommend uninstall + reinstall to ensure all dependencies are correct.

---

### I'm a Developer / Power User

**Windows - from source:**
```bash
install_windows.bat
```

**Linux - from source:**
```bash
./install_linux.sh
```

**Manual pip with GPU auto-detection:**
```bash
pip install -U --force-reinstall whisperjav
```

These scripts auto-detect your GPU and install the appropriate CUDA version (11.8, 12.1, 12.4, 12.6, or 12.8).

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
