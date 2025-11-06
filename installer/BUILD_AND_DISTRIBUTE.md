# Build and Distribution Guide - WhisperJAV GUI

## New: One-command exe builds (requires conda env WJ)

If you already have the WJ conda environment active and PyInstaller installed:

```powershell
# Build both CLI and GUI exes
conda activate WJ
pwsh -File installer/build_all.ps1 -Clean

# Or individually
pwsh -File installer/build_exe_cli.ps1 -Clean
pwsh -File installer/build_exe_gui.ps1 -Clean
```

Outputs:
- `dist/whisperjav/whisperjav.exe` (CLI)
- `dist/whisperjav-gui/whisperjav-gui.exe` (GUI)

These scripts also try to copy `ffmpeg.exe` from your conda env into each `dist` folder for convenience.

## Quick Reference

This guide covers two distribution methods for WhisperJAV v1.5.1:
1. **Conda-Constructor Installer** (Recommended) - Full installer with all dependencies
2. **PyInstaller Standalone Executables** - Portable .exe files

---

## Prerequisites

### Required Software
- **Python 3.9-3.12** (3.13+ not supported)
- **PyInstaller:** `pip install pyinstaller`
- **PyWebView:** `pip install pywebview`
- **WhisperJAV dependencies:** `pip install -e .` from repository root

### Recommended
- **Git** for version control
- **7-Zip** or WinRAR for creating distribution archives
- **Clean Windows VM** for testing final executable

---

## Distribution Method 1: Conda-Constructor Installer (Recommended)

The v1.5.1 installer uses conda-constructor to create a self-contained Windows installer that:
- Installs Python 3.10.18, conda, git, and ffmpeg
- Downloads PyTorch with CUDA support during post-install
- Installs all Python dependencies
- Creates WhisperJAV-GUI.exe launcher in installation root
- Creates desktop shortcut
- Supports automated/unattended installation

### Building the Conda-Constructor Installer

```batch
cd C:\BIN\git\WhisperJav_V1_Minami_Edition\installer

# Build the installer
build_installer_v1.5.1.bat

# This creates:
# WhisperJAV-1.5.1-Windows-x86_64.exe (~250-300MB)
```

### Testing the Installer

1. Run the installer on a clean Windows VM
2. Choose installation directory (default: `%LOCALAPPDATA%\WhisperJAV`)
3. Wait for post-install (downloads PyTorch and dependencies)
4. Verify desktop shortcut created
5. Launch via shortcut or WhisperJAV-GUI.exe in install folder
6. Test full workflow

### Conda-Constructor Files

- `construct_v1.5.1.yaml` - Main configuration
- `post_install_v1.5.1.bat` - Post-install wrapper
- `post_install_v1.5.1.py` - Main post-install script
- `requirements_v1.5.1.txt` - Python dependencies
- `create_desktop_shortcut_v1.5.1.bat` - Shortcut creation
- `WhisperJAV_Launcher_v1.5.1.py` - GUI launcher
- `README_INSTALLER_v1.5.1.txt` - User documentation
- `whisperjav_icon.ico` - Application icon

---

## Distribution Method 2: PyInstaller Standalone Executables

This method creates portable .exe files without requiring a full installation.

### Step 1: Verify Installation

```batch
# Check Python version
python --version
# Should output: Python 3.9.x, 3.10.x, 3.11.x, or 3.12.x

# Check PyInstaller
pyinstaller --version
# Should output version number

# Check WhisperJAV installation
python -c "import whisperjav; print(whisperjav.__version__)"
# Should output: 1.5.1
```

### Step 2: Clean Previous Builds

```batch
cd C:\BIN\git\WhisperJav_V1_Minami_Edition\installer

# Remove old build artifacts
if exist build rmdir /s /q build
if exist dist\whisperjav-gui-web rmdir /s /q dist\whisperjav-gui-web
```

### Step 3: Run Build Script

```powershell
# Build both CLI and GUI executables (recommended)
pwsh -File installer/build_all.ps1 -Clean

# Or build individually:
pwsh -File installer/build_exe_cli.ps1 -Clean     # CLI only
pwsh -File installer/build_exe_gui.ps1 -Clean     # GUI only
```

**Output locations:**
- CLI: `dist/whisperjav/whisperjav.exe`
- GUI: `dist/whisperjav-gui/whisperjav-gui.exe`

The build scripts automatically:
1. Clean previous builds (with -Clean flag)
2. Run PyInstaller with spec files
3. Bundle all assets (HTML, CSS, JS, icon)
4. Copy ffmpeg.exe from conda environment (if available)

### Step 4: Verify Build

```batch
# Navigate to GUI output directory
cd dist\whisperjav-gui

# Check executable exists
dir whisperjav-gui.exe

# Check file size (should be several MB)
# Check icon is visible in Explorer

# Right-click exe → Properties → Details
# Verify:
# - File version: 1.5.1
# - Product name: WhisperJAV GUI
# - Description: WhisperJAV - Japanese AV Subtitle Generator (GUI)
```

---

## Testing the Executable

### Local Testing

```batch
# Run from dist folder
cd C:\BIN\git\WhisperJav_V1_Minami_Edition\installer\dist\whisperjav-gui
whisperjav-gui.exe

# Should launch without console window
# Window should display with icon
# No errors should appear
```

### Quick Functional Test

1. Launch executable
2. Click "Add File(s)" - file dialog should open
3. Add a test video file
4. Click "Start" - process should begin
5. Watch console for logs
6. Wait for completion
7. Verify SRT file created in output folder

### Clean Machine Testing

**Important:** Test on a machine that has never had Python or WhisperJAV installed.

**Option A: Virtual Machine**
1. Create Windows 10/11 VM
2. Copy entire `whisperjav-gui` folder to VM
3. Run executable
4. If WebView2 error appears, follow link to install
5. Relaunch and test full workflow

**Option B: Friend's Computer**
1. Find someone with a clean Windows machine
2. Copy folder via USB or network
3. Have them run executable
4. Guide them through one video processing
5. Collect feedback on UX

---

## Packaging for Distribution

### Primary Distribution: Conda-Constructor Installer

The recommended distribution method for v1.5.1 is the conda-constructor installer:
- File: `WhisperJAV-1.5.1-Windows-x86_64.exe` (~250-300MB)
- Contains: Complete Python environment, all dependencies
- Post-install: Downloads PyTorch and creates launcher
- Ready to distribute as-is

### Alternative: ZIP Archive of PyInstaller Build

```batch
# Navigate to dist folder
cd C:\BIN\git\WhisperJav_V1_Minami_Edition\installer\dist

# Create ZIP with 7-Zip
7z a -tzip WhisperJAV-GUI-v1.5.1-Windows.zip whisperjav-gui\

# Or use Windows built-in compression:
# Right-click whisperjav-gui folder
# Send to → Compressed (zipped) folder
# Rename to WhisperJAV-GUI-v1.5.1-Windows.zip
```

### Option 2: Installer (Advanced)

For a professional installer, consider using:
- **Inno Setup** (free, lightweight)
- **NSIS** (free, flexible)
- **WiX Toolset** (free, MSI packages)

**Not covered in this guide** - see installer tool documentation.

### What to Include

```
WhisperJAV-GUI-v1.5.1-Windows.zip
│
├── whisperjav-gui/
│   ├── whisperjav-gui.exe      (main executable)
│   ├── *.dll                   (dependencies)
│   ├── _internal/              (PyInstaller bundled files)
│   └── ... (other PyInstaller files)
│
├── README.txt                  (quick start - create this)
└── LICENSE.txt                 (if open source)
```

Note: For conda-constructor installer, just distribute the .exe file directly.

### Create README.txt

```txt
WhisperJAV GUI v1.5.1
=====================

Thank you for downloading WhisperJAV!

QUICK START:
1. Extract this ZIP to a folder (e.g., C:\WhisperJAV)
2. Run whisperjav-gui-web.exe
3. If WebView2 error appears, install from the provided link
4. Click "Add File(s)" to select videos
5. Click "Start" to generate subtitles

REQUIREMENTS:
- Windows 10 or later (64-bit)
- 8GB RAM minimum (16GB recommended)
- Microsoft Edge WebView2 Runtime (will prompt to install if missing)

DOCUMENTATION:
See USER_GUIDE.md for complete instructions, troubleshooting, and FAQ.

SUPPORT:
GitHub: https://github.com/meizhong986/WhisperJAV/issues

LICENSE:
MIT License - See LICENSE.txt
```

---

## Creating a GitHub Release

### Step 1: Prepare Release Assets

1. Build executable (as above)
2. Create distribution ZIP
3. Prepare release notes

### Step 2: Tag Release in Git

```batch
# From repository root
cd C:\BIN\git\WhisperJav_V1_Minami_Edition

# Create and push tag
git tag -a v1.5.1 -m "Release v1.5.1 - PyWebView GUI Phase 5 Complete"
git push origin v1.5.1
```

### Step 3: Create GitHub Release

1. Go to repository on GitHub
2. Click "Releases" → "Draft a new release"
3. **Tag:** v1.5.1
4. **Title:** WhisperJAV GUI v1.5.1 - Production Ready
5. **Description:**

```markdown
## WhisperJAV v1.5.1 - Complete PyWebView GUI Takeover

This release completes the transition to PyWebView GUI and introduces a comprehensive conda-constructor installer.

### Major Changes
- Complete PyWebView GUI takeover - removed legacy Tkinter GUI
- WhisperJAV-GUI.exe launcher in installation root for easy access
- Conda-constructor installer with automated dependency management
- Enhanced file management and progress tracking
- Improved stability and error handling

### Installer Features
- Self-contained Windows installer (~250-300MB)
- Automatic CUDA version detection and PyTorch installation
- Desktop shortcut creation with icon
- Automated/unattended installation support
- Comprehensive post-install validation

### GUI Enhancements
- Modern web-based interface with PyWebView
- Real-time progress monitoring and log streaming
- Enhanced file management with better UX
- WebView2 detection with installation guidance
- Professional appearance and responsiveness

### Requirements
- Windows 10 or later (64-bit)
- 8GB RAM minimum (16GB recommended)
- Microsoft Edge WebView2 Runtime (will prompt if missing)

### Installation

**Recommended: Conda-Constructor Installer**
1. Download `WhisperJAV-1.5.1-Windows-x86_64.exe` (Full Installer)
2. Run the installer
3. Follow installation wizard
4. Launch via desktop shortcut or WhisperJAV-GUI.exe in install folder

**Alternative: Portable Executable**
1. Download `WhisperJAV-GUI-v1.5.1-Windows.zip` (Portable)
2. Extract to a folder
3. Run `whisperjav-gui.exe`
4. Install WebView2 if prompted

### Documentation
- README_INSTALLER_v1.5.1.txt (included in installer)
- GitHub README for usage instructions

### Known Limitations
- Windows-only
- Requires WebView2 Runtime (included in Windows 11, downloadable for Windows 10)
- CUDA support requires NVIDIA GPU with compatible driver

**Full Changelog:** See commit history for details.
```

6. **Upload Assets:**
   - WhisperJAV-1.5.1-Windows-x86_64.exe (Conda-constructor installer)
   - WhisperJAV-GUI-v1.5.1-Windows.zip (Portable PyInstaller build - optional)
   - README_INSTALLER_v1.5.1.txt (included in installer, can also attach separately)

7. **Publish Release**

---

## Distribution Checklist

Before releasing to users:

### Build Quality
- [ ] Executable built successfully
- [ ] No build errors or warnings
- [ ] Icon displays in executable
- [ ] Version metadata correct (1.5.1)
- [ ] File size reasonable (not bloated)

### Testing
- [ ] Executable launches on development machine
- [ ] Full workflow test (add files → process → complete)
- [ ] Tested on clean Windows 10 machine
- [ ] Tested on clean Windows 11 machine
- [ ] WebView2 detection works (tested without WebView2)
- [ ] All Phase 5 features work (loading states, shortcuts, etc.)

### Documentation
- [ ] USER_GUIDE.md complete and accurate
- [ ] README.txt created for distribution package
- [ ] Release notes written
- [ ] Known issues documented

### Packaging
- [ ] ZIP archive created
- [ ] USER_GUIDE.md included
- [ ] README.txt included
- [ ] Archive extracts correctly
- [ ] Folder structure logical

### Release
- [ ] Git tag created (v1.5.1)
- [ ] GitHub release created
- [ ] Assets uploaded
- [ ] Release published

---

## Post-Release

### Monitor
- Watch GitHub Issues for bug reports
- Check user feedback
- Monitor download counts

### Support
- Respond to issues promptly
- Update FAQ based on common questions
- Collect feature requests for future versions

### Iterate
- Plan patch releases for critical bugs (v1.4.6, etc.)
- Plan minor releases for enhancements (v1.5.0, etc.)
- Track feedback for major releases (v2.0.0, etc.)

---

## Troubleshooting Build Issues

### "PyInstaller not found"
```batch
pip install pyinstaller
```

### "Module not found" during build
```batch
# Reinstall WhisperJAV with all dependencies
pip install -e .[gui] -U
```

### "Asset file not found"
- Verify `whisperjav/webview_gui/assets/` contains all files
- Check spec file paths are correct
- Ensure icon file exists

### Executable crashes on launch
- Run with console mode enabled (edit spec: `console=True`)
- Check for Python errors in console output
- Verify all dependencies bundled

### Icon doesn't appear
- Check icon file exists at `whisperjav/webview_gui/assets/icon.ico`
- Verify spec file icon path is correct
- Rebuild after fixing icon path

---

## Contact

**Issues:** https://github.com/meizhong986/WhisperJAV/issues
**Discussions:** https://github.com/meizhong986/WhisperJAV/discussions

---

**Version:** 1.5.1
**Last Updated:** 2025
**Maintainer:** MeiZhong
