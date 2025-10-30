# Phase 1 Quick Start Guide

## TL;DR - Get Testing in 3 Steps

### Step 1: Install with GUI Dependencies
```bash
cd C:\BIN\git\WhisperJav_V1_Minami_Edition
pip install -e .[gui]
```

### Step 2: Run the GUI
```bash
whisperjav-gui-web
```

### Step 3: Test All Features
1. Click **"Test Hello World"** → Should see green success message
2. Click **"Start Log Stream"** → Should see colored log messages appear
3. Click **"Select Video File"** → Native file picker should open

**If all 3 work, Phase 1 is successful!**

---

## Build Executable (Optional)

```bash
pip install pyinstaller
cd installer
build_whisperjav_installer_web.bat
```

Executable will be at:
```
installer\dist\whisperjav-gui-web\whisperjav-gui-web.exe
```

---

## What Was Implemented

### Files Created:
```
whisperjav/webview_gui/
├── __init__.py          # Module initialization
├── main.py              # Entry point + PyWebView window
├── api.py               # Python API for JavaScript
└── assets/
    └── index.html       # Phase 1 test interface

installer/
├── whisperjav-gui-web.spec        # PyInstaller config
└── build_whisperjav_installer_web.bat  # Build script

PHASE1_TESTING.md        # Detailed testing instructions
PHASE1_QUICK_START.md    # This file
```

### Files Modified:
```
setup.py                 # Added [gui] extra + whisperjav-gui-web entry point
```

### Features Tested:
- ✅ Python ↔ JavaScript bidirectional bridge
- ✅ Async log streaming (Python → JavaScript via evaluate_js)
- ✅ Native file/folder dialogs (PyWebView API)
- ✅ Asset bundling (dev + PyInstaller modes)

---

## Troubleshooting One-Liners

**GUI won't start:**
```bash
pip install pywebview>=5.0.0
```

**Entry point not found:**
```bash
pip uninstall whisperjav && pip install -e .[gui]
```

**Asset file not found:**
```bash
ls whisperjav/webview_gui/assets/index.html  # Should exist
```

**Build failed:**
```bash
pip install pyinstaller
cd installer
pyinstaller whisperjav-gui-web.spec --clean --noconfirm
```

---

## Success Criteria

Phase 1 passes when:
- [ ] GUI launches from `whisperjav-gui-web` command
- [ ] All 3 test buttons work without errors
- [ ] Executable builds successfully
- [ ] Executable runs with same functionality

**See PHASE1_TESTING.md for comprehensive testing instructions.**
