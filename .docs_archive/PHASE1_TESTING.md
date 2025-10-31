# Phase 1 Testing Instructions - PyWebView GUI

This document provides step-by-step instructions for testing the Phase 1 PyWebView GUI proof-of-concept.

## Overview

Phase 1 implements a minimal proof-of-concept with the following test scenarios:

1. **Python ↔ JavaScript Bridge**: Verify bidirectional communication works
2. **Asynchronous Log Streaming**: Test real-time message passing from Python to JavaScript
3. **Native File Dialogs**: Verify PyWebView's cross-platform file dialog integration

---

## Prerequisites

### Required Software
- Python 3.9-3.12
- Git (for version control)

### Required Python Packages
The GUI requires PyWebView, which will be installed in the next section.

---

## Installation & Setup

### Step 1: Navigate to Repository
```bash
cd C:\BIN\git\WhisperJav_V1_Minami_Edition
```

### Step 2: Install WhisperJAV with GUI Dependencies
```bash
pip install -e .[gui]
```

This installs:
- WhisperJAV in development mode (`-e` flag)
- PyWebView 5.0.0+ (the `[gui]` extra)
- All other dependencies from `setup.py`

**Expected Output:**
```
Successfully installed pywebview-5.x.x whisperjav-x.x.x ...
```

### Step 3: Verify Installation
```bash
whisperjav-gui-web --help
```

**If this fails**, the entry point wasn't registered. Try:
```bash
pip install -e .[gui] --force-reinstall --no-deps
```

Or run directly:
```bash
python -m whisperjav.webview_gui.main
```

---

## Test Scenario 1: Running from Source (Development Mode)

### Run the GUI
```bash
whisperjav-gui-web
```

**Expected Console Output:**
```
WhisperJAV PyWebView GUI - Phase 1
==================================================
Window created successfully
Asset path: C:\BIN\git\WhisperJav_V1_Minami_Edition\whisperjav\webview_gui\assets\index.html
Starting PyWebView...
```

### GUI Window Should Appear
- **Title**: "WhisperJAV - Web GUI (Phase 1 Test)"
- **Size**: 1000x700 pixels
- **Content**: Three test sections with buttons

### Test 1: Python ↔ JavaScript Bridge
1. Click **"Test Hello World"** button
2. **Expected Result**:
   - Output box shows green success message
   - Displays message from Python
   - Shows timestamp

**Success Criteria:**
- No errors in console
- Output box shows: "Message: Hello from Python! PyWebView bridge is working."

### Test 2: Asynchronous Log Streaming
1. Click **"Start Log Stream"** button
2. **Expected Result**:
   - Messages appear one by one (simulated logs)
   - Color-coded messages:
     - Blue: `[INFO]`
     - Orange: `[PROGRESS]`
     - Green: `[SUCCESS]`
   - Auto-scrolls to bottom
3. Click **"Stop Log Stream"** to stop early (optional)
4. Click **"Clear Logs"** to clear output

**Success Criteria:**
- Messages appear sequentially (not all at once)
- Python thread successfully calls JavaScript function
- Color coding applied correctly

### Test 3: Native File Dialogs
1. Click **"Select Video File"** button
2. **Expected Result**:
   - Native Windows file picker opens
   - File type filter shows: "Video Files (*.mp4;*.mkv;...)"
3. Select any file (or cancel)
4. **Expected Result**:
   - Selected file path appears in output box
   - OR "No Selection" message if cancelled

5. Click **"Select Output Directory"** button
6. **Expected Result**:
   - Native Windows folder picker opens
7. Select any folder (or cancel)
8. **Expected Result**:
   - Selected folder path appears
   - OR "No Selection" message if cancelled

**Success Criteria:**
- Native dialogs open (not web-based file pickers)
- Paths are displayed correctly
- Cancel action handled gracefully

---

## Test Scenario 2: Building Executable (Production Mode)

### Step 1: Install PyInstaller
```bash
pip install pyinstaller
```

### Step 2: Build Executable
```bash
cd installer
build_whisperjav_installer_web.bat
```

**Expected Console Output:**
```
========================================
WhisperJAV PyWebView GUI Builder
========================================

[1/4] Checking PyInstaller... OK
[2/4] Checking spec file... OK
[3/4] Cleaning previous builds...
[4/4] Building executable...
      This may take several minutes...
```

**Build Time:** 2-5 minutes (depending on system)

### Step 3: Verify Build Success
Look for:
```
========================================
Build Complete!
========================================

Executable location:
  C:\BIN\git\WhisperJav_V1_Minami_Edition\installer\dist\whisperjav-gui-web\whisperjav-gui-web.exe
```

### Step 4: Test Executable
```bash
cd dist\whisperjav-gui-web
whisperjav-gui-web.exe
```

**Expected Behavior:**
- GUI launches (same as development mode)
- Console shows asset path from bundled location:
  ```
  Asset path: C:\Users\...\AppData\Local\Temp\_MEIxxxxxx\webview_gui_assets\index.html
  ```

### Step 5: Repeat All Tests
Run all three tests from Scenario 1 with the executable:
- Test 1: Hello World
- Test 2: Log Streaming
- Test 3: File Dialogs

**Success Criteria:**
- All tests pass identically to development mode
- Assets are found in bundled location
- No missing file errors

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'webview'"
**Solution:**
```bash
pip install pywebview>=5.0.0
```

### Issue: "FileNotFoundError: Asset file not found"
**Cause:** Assets not bundled correctly

**Solution (Development Mode):**
- Check that `whisperjav/webview_gui/assets/index.html` exists
- Reinstall: `pip install -e .[gui]`

**Solution (Executable Mode):**
- Check spec file `datas` section
- Rebuild with clean flag: `pyinstaller whisperjav-gui-web.spec --clean --noconfirm`

### Issue: "Entry point not found"
**Solution:**
```bash
pip uninstall whisperjav
pip install -e .[gui]
```

### Issue: GUI window appears blank
**Cause:** HTML file loaded but JavaScript not executing

**Solution:**
- Open browser DevTools (if available on your platform)
- Check console for JavaScript errors
- Verify `pywebviewready` event fires

### Issue: File dialogs don't open
**Cause:** Platform-specific backend not loaded

**Solution:**
- Windows: Ensure .NET Framework 4.7.2+ installed
- macOS: Ensure running on macOS 10.15+
- Linux: Install GTK dependencies: `sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.0`

---

## Success Checklist

Phase 1 is considered successful when:

- [ ] Development mode: All 3 tests pass
- [ ] Executable builds without errors
- [ ] Executable mode: All 3 tests pass
- [ ] Assets found in both dev and bundled modes
- [ ] Python → JavaScript calls work (hello_world)
- [ ] JavaScript ← Python calls work (window.evaluate_js in log stream)
- [ ] Native file/folder dialogs functional
- [ ] No console errors during normal operation

---

## Next Steps

After Phase 1 passes:
1. Report results (which tests passed/failed)
2. Proceed to Phase 2 (real WhisperJAV integration)
3. Migrate Tkinter GUI features to PyWebView

---

## File Structure Reference

```
whisperjav/
├── webview_gui/
│   ├── __init__.py          # Module initialization
│   ├── main.py              # Entry point + window setup
│   ├── api.py               # Python API exposed to JavaScript
│   └── assets/
│       └── index.html       # Phase 1 test interface

installer/
├── whisperjav-gui-web.spec        # PyInstaller configuration
└── build_whisperjav_installer_web.bat  # Build script
```

---

## Support

If you encounter issues not covered in Troubleshooting:

1. Check console output for detailed error messages
2. Verify Python version: `python --version` (must be 3.9-3.12)
3. Verify PyWebView version: `pip show pywebview` (must be ≥5.0.0)
4. Check asset files exist in expected locations
5. Run with debug flag: Edit `main.py` and ensure `webview.start(debug=True)`

---

**Phase 1 Testing Document**
*WhisperJAV PyWebView GUI Migration*
*Version 1.0*
