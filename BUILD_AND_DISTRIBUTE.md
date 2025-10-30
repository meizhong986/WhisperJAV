# Build and Distribution Guide - WhisperJAV GUI

## Quick Reference

This guide walks you through building and distributing the WhisperJAV GUI executable for Windows.

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

## Build Process

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
# Should output: 1.4.5
```

### Step 2: Clean Previous Builds

```batch
cd C:\BIN\git\WhisperJav_V1_Minami_Edition\installer

# Remove old build artifacts
if exist build rmdir /s /q build
if exist dist\whisperjav-gui-web rmdir /s /q dist\whisperjav-gui-web
```

### Step 3: Run Build Script

```batch
# Execute the build script
build_whisperjav_installer_web.bat

# This will:
# 1. Check for PyInstaller
# 2. Clean previous builds
# 3. Run PyInstaller with spec file
# 4. Bundle all assets (HTML, CSS, JS, icon)
# 5. Create executable in dist/whisperjav-gui-web/
```

**Expected Output:**
```
========================================
WhisperJAV PyWebView GUI Builder
========================================

[1/4] Checking PyInstaller... OK
[2/4] Checking spec file... OK
[3/4] Cleaning previous builds...
[4/4] Building executable...

========================================
Build Complete!
========================================

Executable location:
  C:\BIN\git\WhisperJav_V1_Minami_Edition\installer\dist\whisperjav-gui-web\whisperjav-gui-web.exe
```

### Step 4: Verify Build

```batch
# Navigate to output directory
cd dist\whisperjav-gui-web

# Check executable exists
dir whisperjav-gui-web.exe

# Check file size (should be several MB)
# Check icon is visible in Explorer

# Right-click exe → Properties → Details
# Verify:
# - File version: 1.4.5
# - Product name: WhisperJAV GUI
# - Description: WhisperJAV GUI - Japanese Adult Video Subtitle Generator
```

---

## Testing the Executable

### Local Testing

```batch
# Run from dist folder
cd C:\BIN\git\WhisperJav_V1_Minami_Edition\installer\dist\whisperjav-gui-web
whisperjav-gui-web.exe

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
2. Copy entire `whisperjav-gui-web` folder to VM
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

### Option 1: ZIP Archive (Recommended)

```batch
# Navigate to dist folder
cd C:\BIN\git\WhisperJav_V1_Minami_Edition\installer\dist

# Create ZIP with 7-Zip
7z a -tzip WhisperJAV-GUI-v1.4.5-Windows.zip whisperjav-gui-web\

# Or use Windows built-in compression:
# Right-click whisperjav-gui-web folder
# Send to → Compressed (zipped) folder
# Rename to WhisperJAV-GUI-v1.4.5-Windows.zip
```

### Option 2: Installer (Advanced)

For a professional installer, consider using:
- **Inno Setup** (free, lightweight)
- **NSIS** (free, flexible)
- **WiX Toolset** (free, MSI packages)

**Not covered in this guide** - see installer tool documentation.

### What to Include

```
WhisperJAV-GUI-v1.4.5-Windows.zip
│
├── whisperjav-gui-web/
│   ├── whisperjav-gui-web.exe  (main executable)
│   ├── *.dll                   (dependencies)
│   ├── webview_gui_assets/     (HTML/CSS/JS)
│   └── ... (other PyInstaller files)
│
├── USER_GUIDE.md               (include this!)
├── README.txt                  (quick start - create this)
└── LICENSE.txt                 (if open source)
```

### Create README.txt

```txt
WhisperJAV GUI v1.4.5
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
git tag -a v1.4.5 -m "Release v1.4.5 - PyWebView GUI Phase 5 Complete"
git push origin v1.4.5
```

### Step 3: Create GitHub Release

1. Go to repository on GitHub
2. Click "Releases" → "Draft a new release"
3. **Tag:** v1.4.5
4. **Title:** WhisperJAV GUI v1.4.5 - Production Ready
5. **Description:**

```markdown
## WhisperJAV GUI v1.4.5 - Production Ready

This release brings the new PyWebView-based GUI to production quality with professional polish and comprehensive documentation.

### New Features
- Modern web-based GUI built with PyWebView
- Real-time progress monitoring and log streaming
- Drag-and-drop file support
- Keyboard shortcuts for power users
- Professional About dialog with help
- WebView2 detection with user-friendly error handling

### Enhancements
- Loading states for better user feedback
- Professional Windows executable with icon and metadata
- Comprehensive user guide and troubleshooting documentation
- Batch processing with async support

### Requirements
- Windows 10 or later (64-bit)
- 8GB RAM minimum (16GB recommended)
- Microsoft Edge WebView2 Runtime (will prompt if missing)

### Installation
1. Download `WhisperJAV-GUI-v1.4.5-Windows.zip`
2. Extract to a folder
3. Run `whisperjav-gui-web.exe`
4. See `USER_GUIDE.md` for complete instructions

### Documentation
- [User Guide](USER_GUIDE.md) - Complete usage instructions
- [Testing Checklist](PHASE5_FINAL_TESTING.md) - For QA and testing

### Known Limitations
- Windows-only (for now)
- Requires WebView2 Runtime
- No settings persistence yet

**Full Changelog:** See commit history for details.
```

6. **Upload Assets:**
   - WhisperJAV-GUI-v1.4.5-Windows.zip
   - USER_GUIDE.md (optional, can reference repository file)

7. **Publish Release**

---

## Distribution Checklist

Before releasing to users:

### Build Quality
- [ ] Executable built successfully
- [ ] No build errors or warnings
- [ ] Icon displays in executable
- [ ] Version metadata correct (1.4.5)
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
- [ ] Git tag created (v1.4.5)
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

**Version:** 1.4.5
**Last Updated:** 2025
**Maintainer:** MeiZhong
