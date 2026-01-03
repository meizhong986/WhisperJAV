# WhisperJAV Installer Build System

This directory contains the automated build system for creating WhisperJAV Windows installers.

## Quick Start

To build an installer for the current version:

```bash
cd installer
python build_release.py --clean   # Clean previous build
python build_release.py           # Generate all files
cd generated
build_installer_v{VERSION}.bat    # Build the .exe
```

## Directory Structure

```
installer/
├── VERSION                 # Single source of truth for version numbers
├── build_release.py        # Main build orchestrator (run this!)
├── templates/              # Template files with {{VERSION}} placeholders
│   ├── construct.yaml.template
│   ├── post_install.py.template
│   ├── build_installer.bat.template
│   └── ... (11 templates total)
├── generated/              # Output directory (auto-generated, gitignored)
├── LICENSE.txt             # License for installer
├── whisperjav_icon.ico     # Application icon
├── install_linux.sh        # Linux/macOS installation script
├── install_windows.bat     # Windows batch installation script
└── README.md               # This file
```

## How It Works

### 1. VERSION File (Source of Truth)

Edit `VERSION` to set the release version:

```ini
[version]
major = 1
minor = 7
patch = 6
prerelease =          # a0, b0, rc0 for pre-releases (PEP 440)
display_label =       # alpha, beta, rc for UI display
architecture = v4.4

[metadata]
app_name = WhisperJAV
description = Japanese AV Subtitle Generator with AI-powered transcription
author = MeiZhong
# ...
```

### 2. Build Orchestrator

`build_release.py` runs 5 phases:

| Phase | Action |
|-------|--------|
| 1 | Update `whisperjav/__version__.py` from VERSION |
| 2 | Generate 11 version-stamped files from templates |
| 3 | Build wheel package (`whisperjav-{version}-py3-none-any.whl`) |
| 4 | Copy static files (LICENSE.txt, icon) |
| 5 | Run validation script |

### 3. Templates

Templates in `templates/` use placeholders that get replaced:

| Placeholder | Example Value |
|-------------|---------------|
| `{{VERSION}}` | 1.7.6 |
| `{{DISPLAY_VERSION}}` | 1.7.6-beta |
| `{{VERSION_MAJOR}}` | 1 |
| `{{VERSION_MINOR}}` | 7 |
| `{{VERSION_PATCH}}` | 6 |
| `{{APP_NAME}}` | WhisperJAV |
| `{{PYTHON_VERSION}}` | 3.10.18 |
| `{{SHORTCUT_NAME}}` | WhisperJAV v1.7.6 |

## Build Commands

```bash
# Full build (recommended)
python build_release.py --clean && python build_release.py

# Preview without making changes
python build_release.py --dry-run

# Clean generated files only
python build_release.py --clean

# Run validation only
python build_release.py --validate
```

## Release Workflow

### Step 1: Update Version

Edit `installer/VERSION`:
```ini
patch = 6  # Increment for new release
```

### Step 2: Generate Files

```bash
cd installer
python build_release.py --clean
python build_release.py
```

### Step 3: Build Installer

```bash
cd generated
build_installer_v1.7.6.bat
```

This creates: `WhisperJAV-1.7.6-Windows-x86_64.exe`

### Step 4: Distribute

Upload the `.exe` to GitHub Releases. Do NOT commit `.exe` files.

## Prerequisites

- **Python 3.10+** with pip
- **Anaconda/Miniconda** with constructor: `conda install constructor -c conda-forge`
- **NSIS** (for Windows installer packaging)

## Installation Scripts

For users who prefer script-based installation:

| Script | Platform | Description |
|--------|----------|-------------|
| `install_linux.sh` | Linux/macOS | Full installation with venv |
| `install_windows.bat` | Windows | Full installation with venv |

Both scripts:
- Create isolated Python virtual environment
- Install PyTorch with appropriate CUDA version
- Install WhisperJAV from GitHub (latest version)
- Create desktop shortcuts

## Troubleshooting

### "Template not found" error
Ensure all 11 template files exist in `templates/`.

### Wheel build fails
Run from project root: `python setup.py bdist_wheel`

### Constructor fails
Install constructor: `conda install constructor -c conda-forge`

### Validation fails
Check generated files match VERSION. Run `python build_release.py --validate`.

## Generated Files Reference

When you run `build_release.py`, these files are created in `generated/`:

| File | Purpose |
|------|---------|
| `construct_v{VERSION}.yaml` | conda-constructor configuration |
| `post_install_v{VERSION}.py` | Post-installation script |
| `post_install_v{VERSION}.bat` | Post-install entry point |
| `build_installer_v{VERSION}.bat` | Runs constructor to build .exe |
| `validate_installer_v{VERSION}.py` | Pre-build validation |
| `requirements_v{VERSION}.txt` | Python dependencies list |
| `WhisperJAV_Launcher_v{VERSION}.py` | GUI launcher script |
| `create_desktop_shortcut_v{VERSION}.bat` | Shortcut creation |
| `uninstall_v{VERSION}.bat` | Clean uninstaller |
| `README_INSTALLER_v{VERSION}.txt` | User documentation |
| `custom_template_v{VERSION}.nsi.tmpl` | NSIS template |
| `whisperjav-{VERSION}-py3-none-any.whl` | Python wheel package |

## Important Notes

1. **Never edit files in `generated/`** - they are overwritten on each build
2. **Edit templates instead** - changes propagate to all future builds
3. **VERSION is the single source of truth** - all version numbers derive from it
4. **Run `--clean` before builds** - prevents stale file issues
