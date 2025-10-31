# Migration Plan: PyWebView GUI Complete Takeover

**Document Version:** 1.0
**Date:** 2025-10-31
**Status:** APPROVED - READY FOR IMPLEMENTATION
**Branch:** `pywebview_dev` → merge to `main`

---

## Executive Summary

This document provides complete, step-by-step instructions for migrating WhisperJAV from the legacy Tkinter GUI to the new PyWebView GUI as the primary and only GUI interface. This is a **clean cutover migration** with complete removal of all Tkinter GUI code and infrastructure.

**Migration Strategy:** Complete removal with no deprecation period
**Target Version:** v1.5.0 (or next major release)
**Estimated Time:** 2-4 hours for implementation + testing
**Risk Level:** Medium (breaking change, but PyWebView GUI is production-tested)

---

## Migration Decisions Summary

| Decision Point | Choice | Rationale |
|---|---|---|
| **Legacy Tkinter GUI** | Complete removal | PyWebView GUI is mature, tested, and superior in all aspects |
| **Command alias** | Remove `whisperjav-gui-web` | Single clean entry point, no confusion |
| **Executable naming** | `whisperjav-gui.exe` | Standard naming matching command |
| **Installer launcher** | Update to PyWebView | Complete migration, no mixed implementations |
| **Deprecation period** | None | Clean break, announce in release notes |

---

## Current State vs. Target State

### Current State (v1.4.x)

**Entry Points:**
```python
"whisperjav=whisperjav.main:main",                      # CLI ✅
"whisperjav-gui=whisperjav.gui.whisperjav_gui:main",   # Tkinter GUI ❌
"whisperjav-gui-web=whisperjav.webview_gui.main:main", # PyWebView GUI ✅
"whisperjav-translate=whisperjav.translate.cli:main",  # Translation ✅
```

**Directory Structure:**
```
whisperjav/
├── gui/                          # Tkinter GUI (TO BE REMOVED)
│   ├── __init__.py
│   ├── whisperjav_gui.py        # 573 lines
│   ├── config_integration.py
│   └── assets/icon.ico
├── webview_gui/                  # PyWebView GUI (PRIMARY)
│   ├── __init__.py
│   ├── main.py
│   ├── api.py
│   ├── assets/                   # HTML/CSS/JS
│   └── *.md                      # Documentation
└── gui_launcher.py               # Tkinter launcher (TO BE REMOVED)

installer/
├── whisperjav-gui-web.spec       # PyInstaller spec (TO BE RENAMED)
├── build_whisperjav_installer_web.bat  # Build script (TO BE RENAMED)
└── WhisperJAV_Launcher.py        # Conda launcher (TO BE UPDATED)
```

### Target State (v1.5.0)

**Entry Points:**
```python
"whisperjav=whisperjav.main:main",                   # CLI ✅
"whisperjav-gui=whisperjav.webview_gui.main:main",  # PyWebView GUI (ONLY)
"whisperjav-translate=whisperjav.translate.cli:main", # Translation ✅
```

**Directory Structure:**
```
whisperjav/
├── webview_gui/                  # ONLY GUI
│   ├── __init__.py
│   ├── main.py
│   ├── api.py
│   ├── assets/
│   └── *.md

installer/
├── whisperjav-gui.spec           # RENAMED from whisperjav-gui-web.spec
├── build_whisperjav_installer_gui.bat  # RENAMED from build_whisperjav_installer_web.bat
└── WhisperJAV_Launcher.py        # UPDATED to use webview_gui
```

---

## Implementation Phases

### PHASE 0: Pre-Migration Safety ⚠️

**Purpose:** Create safety net for rollback if needed

**Steps:**

1. **Create git tag for current state:**
   ```bash
   git tag -a v1.4.5-pre-migration -m "State before PyWebView GUI takeover"
   git push origin v1.4.5-pre-migration
   ```

2. **Ensure clean working directory:**
   ```bash
   git status
   # Should show: nothing to commit, working tree clean
   ```

3. **Create migration branch (if not already on pywebview_dev):**
   ```bash
   # If needed:
   git checkout -b migration/pywebview-takeover
   ```

4. **Document current Tkinter functionality:**
   - Take screenshots of Tkinter GUI
   - Note any Tkinter-specific features
   - Verify PyWebView has feature parity

5. **Verify PyWebView GUI works:**
   ```bash
   whisperjav-gui-web
   # Test: file selection, processing, console output, all tabs
   ```

**Validation:**
- [ ] Git tag created
- [ ] Working directory clean
- [ ] On correct branch
- [ ] PyWebView GUI tested and functional

---

### PHASE 1: Update Entry Points (setup.py)

**Purpose:** Make `whisperjav-gui` point to PyWebView GUI, remove `whisperjav-gui-web`

**File:** `setup.py`

**Current Code (Lines 71-78):**
```python
entry_points={
    "console_scripts": [
        "whisperjav=whisperjav.main:main",
        "whisperjav-gui=whisperjav.gui.whisperjav_gui:main",       # OLD
        "whisperjav-gui-web=whisperjav.webview_gui.main:main",     # NEW (temp)
        "whisperjav-translate=whisperjav.translate.cli:main",
    ],
},
```

**New Code:**
```python
entry_points={
    "console_scripts": [
        "whisperjav=whisperjav.main:main",
        "whisperjav-gui=whisperjav.webview_gui.main:main",         # UPDATED
        "whisperjav-translate=whisperjav.translate.cli:main",
    ],
},
```

**Implementation:**
1. Open `setup.py`
2. Locate `entry_points` dictionary (around line 71)
3. Replace the `whisperjav-gui` line to point to `whisperjav.webview_gui.main:main`
4. Remove the `whisperjav-gui-web` line completely
5. Save file

**Validation:**
- [ ] `setup.py` modified correctly
- [ ] Only 3 entry points remain (whisperjav, whisperjav-gui, whisperjav-translate)
- [ ] `whisperjav-gui` points to `whisperjav.webview_gui.main:main`

**Testing (after this phase):**
```bash
# Reinstall in development mode
pip install -e .

# Verify command now points to PyWebView
whisperjav-gui
# Should launch PyWebView GUI (not Tkinter)

# Verify old command no longer exists
whisperjav-gui-web
# Should show: command not found or error
```

---

### PHASE 2: Remove Tkinter GUI Directory

**Purpose:** Complete removal of legacy Tkinter GUI code

**Files to Delete:**
```
whisperjav/gui/                    # ENTIRE DIRECTORY
├── __init__.py
├── whisperjav_gui.py
├── config_integration.py
├── assets/
│   └── icon.ico
└── screen/                        # Documentation subfolder
    ├── GUI-Design-Proposed-NEW-UI-TABS.pdf
    ├── GUI-Design-Proposed-NEW-UI-TABS.pptx
    ├── GUI-NEW-DESIGN_IMPLEMENTATION-PLAN-CLAUDE.md
    ├── tabs.png
    └── Screenshot_basic.png
```

**Implementation:**

**Option A: Using Bash (recommended):**
```bash
# From repository root
rm -rf whisperjav/gui
```

**Option B: Manual deletion:**
1. Navigate to `whisperjav/` directory
2. Delete `gui/` folder completely
3. Verify it's gone: `ls whisperjav/gui` should error

**Validation:**
- [ ] `whisperjav/gui/` directory no longer exists
- [ ] `ls whisperjav/gui` returns error
- [ ] No Tkinter GUI code remains in project

**Note:** Do NOT commit yet - we'll batch all changes together

---

### PHASE 3: Remove GUI Launcher File

**Purpose:** Remove standalone Tkinter launcher

**File to Delete:**
```
whisperjav/gui_launcher.py          # Root level file
```

**Implementation:**
```bash
# From repository root
rm whisperjav/gui_launcher.py
```

**Validation:**
- [ ] `whisperjav/gui_launcher.py` no longer exists
- [ ] File deleted from repository root

---

### PHASE 4: Update Build Infrastructure - Spec File

**Purpose:** Rename and update PyInstaller spec for standard naming

**Current File:** `installer/whisperjav-gui-web.spec` (120 lines)
**New File:** `installer/whisperjav-gui.spec`

**Implementation Steps:**

1. **Rename the file:**
   ```bash
   cd installer
   mv whisperjav-gui-web.spec whisperjav-gui.spec
   ```

2. **Update exe name inside spec file:**

   **Find (around line 90-100):**
   ```python
   exe = EXE(
       pyz,
       a.scripts,
       [],
       exclude_binaries=True,
       name='whisperjav-gui-web',           # OLD NAME
       debug=False,
       # ...
   )
   ```

   **Replace with:**
   ```python
   exe = EXE(
       pyz,
       a.scripts,
       [],
       exclude_binaries=True,
       name='whisperjav-gui',                # NEW NAME
       debug=False,
       # ...
   )
   ```

3. **Verify icon path is correct:**
   ```python
   icon='whisperjav/webview_gui/assets/icon.ico',  # Should be correct already
   ```

4. **Verify datas section bundles assets:**
   ```python
   datas=[
       ('../whisperjav/webview_gui/assets', 'whisperjav/webview_gui/assets'),
   ],
   ```

**Validation:**
- [ ] File renamed to `whisperjav-gui.spec`
- [ ] exe `name=` parameter updated to `'whisperjav-gui'`
- [ ] Icon path correct
- [ ] Assets bundling correct

---

### PHASE 5: Update Build Infrastructure - Build Script

**Purpose:** Rename and update build script for new spec file

**Current File:** `installer/build_whisperjav_installer_web.bat` (93 lines)
**New File:** `installer/build_whisperjav_installer_gui.bat`

**Implementation Steps:**

1. **Rename the file:**
   ```bash
   cd installer
   mv build_whisperjav_installer_web.bat build_whisperjav_installer_gui.bat
   ```

2. **Update spec file reference inside script:**

   **Find (around line 30-40):**
   ```batch
   set SPEC_FILE=whisperjav-gui-web.spec
   ```

   **Replace with:**
   ```batch
   set SPEC_FILE=whisperjav-gui.spec
   ```

3. **Update any comments or echo statements:**

   **Find:**
   ```batch
   echo Building WhisperJAV Web GUI executable...
   ```

   **Replace with:**
   ```batch
   echo Building WhisperJAV GUI executable...
   ```

4. **Verify output path references:**
   ```batch
   echo Executable created: dist\whisperjav-gui\whisperjav-gui.exe
   ```

**Validation:**
- [ ] File renamed to `build_whisperjav_installer_gui.bat`
- [ ] SPEC_FILE variable updated
- [ ] Echo messages updated
- [ ] Output paths correct

---

### PHASE 6: Update Installer Launcher

**Purpose:** Make Conda installer launch PyWebView GUI

**File:** `installer/WhisperJAV_Launcher.py` (82 lines)

**Current Code (Lines 60-70):**
```python
def main():
    """Launch WhisperJAV GUI"""
    # ... setup code ...

    # Launch GUI using pythonw (no console window)
    subprocess.Popen(
        [pythonw_path, "-X", "utf8", "-m", "whisperjav.gui.whisperjav_gui"],  # OLD
        cwd=str(script_dir),
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    )
```

**New Code:**
```python
def main():
    """Launch WhisperJAV GUI"""
    # ... setup code ...

    # Launch GUI using pythonw (no console window)
    subprocess.Popen(
        [pythonw_path, "-X", "utf8", "-m", "whisperjav.webview_gui.main"],   # NEW
        cwd=str(script_dir),
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    )
```

**Implementation:**
1. Open `installer/WhisperJAV_Launcher.py`
2. Find the subprocess.Popen call (around line 67)
3. Change module from `whisperjav.gui.whisperjav_gui` to `whisperjav.webview_gui.main`
4. Save file

**Additional Changes (Optional but Recommended):**

Update docstrings and comments:
```python
"""
WhisperJAV GUI Launcher

Launches the WhisperJAV PyWebView GUI with proper environment setup.
Handles first-run model downloads and PATH configuration.
"""
```

**Validation:**
- [ ] Module path updated to `whisperjav.webview_gui.main`
- [ ] Docstrings updated (optional)
- [ ] No other Tkinter references remain

**Testing (if Conda installer available):**
```bash
python installer/WhisperJAV_Launcher.py
# Should launch PyWebView GUI
```

---

### PHASE 7: Update Documentation - README.md

**Purpose:** Update all GUI references to reflect PyWebView as primary

**File:** `README.md`

**Changes Required:**

#### 1. **Line 105-106: GUI Launch Command**

**Find:**
```markdown
```bash
whisperjav-gui
```
**Note:** Currently launches Tkinter GUI
```

**Replace with:**
```markdown
```bash
whisperjav-gui
```
**Note:** Launches modern PyWebView-based GUI
```

#### 2. **Lines 420-440: GUI Interface Section**

**Find:**
```markdown
### GUI Interface (Work in Progress)

The graphical interface provides:
...
```

**Replace with:**
```markdown
### GUI Interface

The PyWebView-based graphical interface provides:
- Modern, responsive HTML/CSS/JS interface
- Drag-and-drop file and folder selection
- Real-time progress monitoring and log streaming
- Visual mode/sensitivity selection with descriptions
- Advanced settings in tabbed interface
- Keyboard shortcuts (Ctrl+O, Ctrl+R, F1, Esc, F5)
- Professional look and feel for non-technical users

**System Requirements:**
- Windows: Requires WebView2 runtime (automatically installed with Edge browser)
- macOS: Uses native WebKit
- Linux: Uses GTK WebKit2
```

#### 3. **Lines 444-449: Quick Start Steps**

**Find:**
```markdown
1. Launch: `python whisperjav_gui.py` or `whisperjav-gui`
```

**Replace with:**
```markdown
1. Launch: `whisperjav-gui`
```

#### 4. **Add GUI Documentation Reference (after Quick Start)**

**Add:**
```markdown
For detailed GUI usage instructions, see [GUI_USER_GUIDE.md](GUI_USER_GUIDE.md).
```

**Implementation:**
1. Open `README.md`
2. Make all 4 changes listed above
3. Search for any remaining "Tkinter" references and update/remove
4. Search for "whisperjav-gui-web" and update to "whisperjav-gui"
5. Save file

**Validation:**
- [ ] Launch command updated
- [ ] GUI Interface section updated with PyWebView features
- [ ] Quick Start updated
- [ ] GUI documentation reference added
- [ ] No "Tkinter" references remain
- [ ] No "whisperjav-gui-web" references remain

---

### PHASE 8: Update Documentation - CLAUDE.md

**Purpose:** Update development documentation for Claude Code

**File:** `CLAUDE.md` (Project instructions)

**Changes Required:**

#### 1. **Lines 20-28: Running the Application**

**Find:**
```markdown
# GUI application
whisperjav-gui
# or
python -m whisperjav.gui.whisperjav_gui
```

**Replace with:**
```markdown
# GUI application
whisperjav-gui
# or
python -m whisperjav.webview_gui.main
```

#### 2. **Lines 10-13: Entry Points Section**

**Find:**
```markdown
**Entry Points**
- `whisperjav/gui/whisperjav_gui.py`: Tkinter GUI application
```

**Replace with:**
```markdown
**Entry Points**
- `whisperjav/webview_gui/main.py`: PyWebView GUI application
```

#### 3. **Add PyWebView GUI Section (after Entry Points)**

**Add:**
```markdown
**GUI Application** (`whisperjav/webview_gui/`)
- `main.py`: PyWebView window creation, WebView2 detection, icon loading
- `api.py`: Backend API class with subprocess management
- `assets/`: HTML/CSS/JS frontend interface
- System requirements: WebView2 (Windows), WebKit (macOS/Linux)
```

**Implementation:**
1. Open `CLAUDE.md`
2. Make all 3 changes listed above
3. Search for any remaining Tkinter references
4. Save file

**Validation:**
- [ ] Running the Application section updated
- [ ] Entry Points section updated
- [ ] PyWebView GUI section added
- [ ] No Tkinter references in core sections

---

### PHASE 9: Update BUILD_AND_DISTRIBUTE.md (Optional)

**Purpose:** Update build documentation if needed

**File:** `BUILD_AND_DISTRIBUTE.md`

**Changes Required:**

Search and replace throughout file:
- `whisperjav-gui-web.spec` → `whisperjav-gui.spec`
- `build_whisperjav_installer_web.bat` → `build_whisperjav_installer_gui.bat`
- `whisperjav-gui-web.exe` → `whisperjav-gui.exe`
- `whisperjav-gui-web` command → `whisperjav-gui`

**Implementation:**
1. Open `BUILD_AND_DISTRIBUTE.md`
2. Use find/replace for all 4 patterns above
3. Verify all references updated
4. Save file

**Validation:**
- [ ] All file names updated
- [ ] All command names updated
- [ ] Instructions still accurate

---

### PHASE 10: Remove Tkinter Test Files (Optional Cleanup)

**Purpose:** Remove any Tkinter-specific test files

**Files to Check:**
```
tests/test_tab_spacing.py          # Tkinter GUI test (DELETE if exists)
tests/test_simple_gap.py           # Tkinter GUI test (DELETE if exists)
```

**Implementation:**
```bash
# Check if files exist
ls tests/test_tab_spacing.py tests/test_simple_gap.py 2>/dev/null

# If they exist, delete them
rm tests/test_tab_spacing.py tests/test_simple_gap.py 2>/dev/null
```

**Validation:**
- [ ] Tkinter test files removed (if they existed)
- [ ] PyWebView test file `whisperjav/webview_gui/test_api.py` still present

---

## PHASE 11: Testing & Validation

**Purpose:** Comprehensive testing before committing changes

### Test 1: Entry Point Installation

```bash
# Reinstall package with new entry points
pip install -e .

# Verify installation
pip show whisperjav
```

**Expected Result:**
- Installation succeeds with no errors
- Package version updated (if applicable)

**Validation:**
- [ ] `pip install -e .` succeeds
- [ ] No error messages

---

### Test 2: CLI Entry Point (Should Not Be Affected)

```bash
# Test CLI still works
whisperjav --help
```

**Expected Result:**
- Help message displays
- No errors

**Validation:**
- [ ] `whisperjav --help` works
- [ ] CLI functionality unaffected

---

### Test 3: GUI Entry Point (PRIMARY TEST)

```bash
# Test GUI launches PyWebView
whisperjav-gui
```

**Expected Result:**
- PyWebView GUI window opens (NOT Tkinter)
- Window title: "WhisperJAV"
- Modern HTML/CSS interface visible
- No console errors

**Validation:**
- [ ] `whisperjav-gui` command works
- [ ] PyWebView GUI launches (verify HTML interface, not Tkinter)
- [ ] Window icon displays
- [ ] No errors in console

---

### Test 4: Verify Old Command Removed

```bash
# This should fail
whisperjav-gui-web
```

**Expected Result:**
- Error: command not found (Windows) or "No such file or directory"
- This is CORRECT behavior

**Validation:**
- [ ] `whisperjav-gui-web` command no longer exists
- [ ] Error message displayed (expected)

---

### Test 5: GUI Functionality Test

**Test in the GUI:**
1. Click "Add Files" - file dialog opens
2. Select test video file
3. Click "Add Folder" - folder dialog opens
4. Verify files appear in list
5. Click "Start" - processing begins (or error if no valid file)
6. Verify console output appears in real-time
7. Test keyboard shortcuts:
   - Ctrl+O (Add Files)
   - Ctrl+R (Remove Selected)
   - F1 (Help)
   - Esc (Close dialogs)
8. Switch between "Transcription Mode" and "Advanced Options" tabs

**Validation:**
- [ ] File selection works
- [ ] Folder selection works
- [ ] Files display in list
- [ ] Processing starts correctly
- [ ] Console output streams in real-time
- [ ] Keyboard shortcuts work
- [ ] Tabs switch correctly
- [ ] All features functional

---

### Test 6: Build Executable

```bash
# Navigate to installer directory
cd installer

# Run build script
build_whisperjav_installer_gui.bat
```

**Expected Result:**
- Build completes successfully
- Output: `dist/whisperjav-gui/whisperjav-gui.exe`
- No errors during build

**Validation:**
- [ ] Build script runs without errors
- [ ] Executable created at `dist/whisperjav-gui/whisperjav-gui.exe`
- [ ] Executable size reasonable (50-150MB depending on bundling)

---

### Test 7: Test Built Executable

```bash
# Run the built executable
cd dist/whisperjav-gui
whisperjav-gui.exe
```

**Expected Result:**
- GUI launches from executable
- Same functionality as development version
- No console window (if configured)
- WebView2 detection works (Windows)

**Validation:**
- [ ] Executable launches successfully
- [ ] GUI fully functional
- [ ] WebView2 detection dialog appears if WebView2 missing (Windows)
- [ ] No crashes or errors

---

### Test 8: Clean Machine Test (Recommended)

**Setup:**
- Use Windows VM or clean machine
- Do NOT have Python or development environment installed
- Copy only the `dist/whisperjav-gui/` folder

**Test:**
1. Run `whisperjav-gui.exe`
2. If WebView2 missing, should show friendly error dialog
3. If WebView2 present, GUI should launch normally
4. Test full processing workflow

**Validation:**
- [ ] Executable runs on clean machine
- [ ] WebView2 detection works correctly
- [ ] GUI fully functional without development environment

---

### Test 9: Installer Launcher Test (If Using Conda Installer)

```bash
# If you have Conda installer built
python installer/WhisperJAV_Launcher.py
```

**Expected Result:**
- PyWebView GUI launches
- First-run dialog appears if needed
- No Tkinter GUI

**Validation:**
- [ ] Launcher works with PyWebView GUI
- [ ] No errors referencing Tkinter or missing modules

---

## PHASE 12: Final Validation Checklist

**Before committing, verify ALL items:**

### Entry Points
- [ ] `whisperjav` (CLI) works
- [ ] `whisperjav-gui` launches PyWebView GUI
- [ ] `whisperjav-gui-web` does NOT exist (command not found)
- [ ] `whisperjav-translate` works (unaffected)

### File Structure
- [ ] `whisperjav/gui/` directory deleted
- [ ] `whisperjav/gui_launcher.py` deleted
- [ ] `whisperjav/webview_gui/` directory intact
- [ ] `installer/whisperjav-gui.spec` exists (renamed)
- [ ] `installer/whisperjav-gui-web.spec` does NOT exist
- [ ] `installer/build_whisperjav_installer_gui.bat` exists (renamed)
- [ ] `installer/build_whisperjav_installer_web.bat` does NOT exist

### Documentation
- [ ] `README.md` updated with PyWebView references
- [ ] `CLAUDE.md` updated with new entry points
- [ ] No "Tkinter" references in core documentation (except git history)
- [ ] No "whisperjav-gui-web" references in user-facing docs

### Build System
- [ ] `setup.py` entry points correct
- [ ] Spec file exe name is `whisperjav-gui`
- [ ] Build script references correct spec file
- [ ] Executable builds successfully
- [ ] Executable runs on test machine

### Functionality
- [ ] All GUI features work (file selection, processing, logs)
- [ ] Keyboard shortcuts work
- [ ] Tabs switch correctly
- [ ] WebView2 detection works (Windows)
- [ ] No console errors
- [ ] Processing completes successfully

### Code Quality
- [ ] No broken imports
- [ ] No references to deleted files
- [ ] No Tkinter imports (except in webview_gui/main.py for error fallback)
- [ ] All tests pass (if applicable)

---

## PHASE 13: Commit Changes

**Purpose:** Commit all migration changes as atomic commit

### Implementation:

```bash
# Check status
git status

# Expected changes:
# - Modified: setup.py
# - Modified: README.md
# - Modified: CLAUDE.md
# - Modified: installer/WhisperJAV_Launcher.py
# - Deleted: whisperjav/gui/ (entire directory)
# - Deleted: whisperjav/gui_launcher.py
# - Renamed: installer/whisperjav-gui-web.spec → installer/whisperjav-gui.spec
# - Renamed: installer/build_whisperjav_installer_web.bat → installer/build_whisperjav_installer_gui.bat
# - New: MIGRATION_PLAN_PYWEBVIEW_GUI_TAKEOVER.md

# Stage all changes
git add -A

# Verify staged changes
git status

# Commit with descriptive message
git commit -m "Complete PyWebView GUI takeover - remove Tkinter GUI

BREAKING CHANGE: whisperjav-gui now launches PyWebView GUI (was Tkinter)

Changes:
- Updated setup.py entry points: whisperjav-gui → PyWebView
- Removed whisperjav-gui-web entry point (consolidated to whisperjav-gui)
- Deleted whisperjav/gui/ directory (Tkinter GUI)
- Deleted whisperjav/gui_launcher.py
- Renamed installer/whisperjav-gui-web.spec → whisperjav-gui.spec
- Renamed installer/build_whisperjav_installer_web.bat → build_whisperjav_installer_gui.bat
- Updated installer/WhisperJAV_Launcher.py to launch PyWebView GUI
- Updated README.md and CLAUDE.md with new commands
- Executable now built as whisperjav-gui.exe (was whisperjav-gui-web.exe)

Rationale:
- PyWebView GUI is production-ready, tested, and superior
- Modern HTML/CSS/JS interface with better UX
- Drag-and-drop, keyboard shortcuts, real-time log streaming
- Clean architecture with thin GUI wrapper
- Tested and validated on Windows 10/11

Migration:
- Users: Run 'whisperjav-gui' as before (launches new GUI automatically)
- Developers: See MIGRATION_PLAN_PYWEBVIEW_GUI_TAKEOVER.md for details
- Rollback: Git tag v1.4.5-pre-migration available if needed

Resolves: GUI migration to PyWebView
Ref: pywebview_dev branch, 5-phase PyWebView implementation"
```

**Validation:**
- [ ] All changes staged
- [ ] Commit message descriptive
- [ ] Commit created successfully

---

## PHASE 14: Post-Commit Actions

### 1. Push to Repository (If Ready)

```bash
# Push to remote
git push origin pywebview_dev

# Or if merging to main:
# git checkout main
# git merge pywebview_dev
# git push origin main
```

### 2. Create GitHub Release (When Ready)

**Version:** v1.5.0

**Title:** WhisperJAV v1.5.0 - PyWebView GUI Launch

**Release Notes Template:**
```markdown
# WhisperJAV v1.5.0 - PyWebView GUI Launch

## 🚀 Major Changes

### New PyWebView GUI (Default)

The `whisperjav-gui` command now launches a modern PyWebView-based interface!

**What's New:**
- 🎨 Modern HTML/CSS/JS interface (goodbye Tkinter!)
- 🖱️ Drag-and-drop file/folder support
- ⌨️ Keyboard shortcuts (Ctrl+O, Ctrl+R, F1, Esc, F5)
- 📊 Real-time log streaming and progress updates
- 🎯 Professional, responsive design
- 🚀 Better performance and reliability

**Migration:**
- ✅ No action required! Just run `whisperjav-gui` as usual
- ✅ Same functionality, better experience
- ✅ All your workflows work the same

**System Requirements:**
- Windows: WebView2 runtime (included with Edge browser)
- macOS: Native WebKit (built-in)
- Linux: GTK WebKit2

## ⚠️ Breaking Changes

- `whisperjav-gui-web` command removed (consolidated to `whisperjav-gui`)
- Tkinter GUI completely removed
- Rollback: Use v1.4.5 if you need the old Tkinter GUI

## 📦 Installation

**Method 1: PyPI (when released)**
```bash
pip install -U whisperjav[gui]
```

**Method 2: Windows Executable**
Download `whisperjav-gui.exe` from releases below.

## 📚 Documentation

- [GUI User Guide](GUI_USER_GUIDE.md)
- [Build & Distribute Guide](BUILD_AND_DISTRIBUTE.md)
- [Migration Plan](MIGRATION_PLAN_PYWEBVIEW_GUI_TAKEOVER.md)

## 🐛 Bug Fixes

- Improved error handling for WebView2 detection
- Fixed Unicode handling in console output
- Better subprocess management

## 💾 Downloads

- `whisperjav-gui.exe` - Windows executable (recommended for most users)
- Source code (zip)
- Source code (tar.gz)

---

**Full Changelog:** v1.4.5...v1.5.0
```

### 3. Update Version Number (If Not Done Yet)

**Files to update:**
- `setup.py` - version parameter
- `whisperjav/__init__.py` - `__version__` constant
- `installer/version_info.txt` - Windows version metadata

**Example:**
```python
# setup.py
version="1.5.0",

# whisperjav/__init__.py
__version__ = "1.5.0"

# installer/version_info.txt
filevers=(1, 5, 0, 0),
prodvers=(1, 5, 0, 0),
```

### 4. Announcement and Communication

**Where to announce:**
- GitHub Releases (primary)
- Project README.md (update "Latest Release" section)
- Discord/Slack/forum if applicable
- PyPI release notes (if publishing)

**Key messaging:**
- Emphasize improvement, not just change
- Highlight new features
- Reassure users (no action needed)
- Provide rollback option (v1.4.5) if needed

---

## Rollback Plan

**If critical issues arise after migration:**

### Option 1: Revert to Git Tag (Recommended)

```bash
# View available tags
git tag -l

# Checkout pre-migration state
git checkout v1.4.5-pre-migration

# Create new branch from this point
git checkout -b hotfix/tkinter-restore

# If needed, push as emergency release
```

### Option 2: Revert Specific Commit

```bash
# Find the migration commit hash
git log --oneline

# Revert the commit (creates new commit that undoes changes)
git revert <commit-hash>

# Push revert
git push origin main
```

### Option 3: Cherry-Pick Fixes Forward

```bash
# If only specific issues, fix them forward rather than reverting
git checkout -b fix/pywebview-issue
# Make fixes
git commit -m "Fix: ..."
git push origin fix/pywebview-issue
```

**Rollback Testing:**
- [ ] Tkinter GUI works from rolled-back state
- [ ] `whisperjav-gui` launches Tkinter
- [ ] All features functional
- [ ] Documentation matches rolled-back state

---

## Troubleshooting

### Issue 1: `whisperjav-gui` command not found

**Symptoms:**
```
'whisperjav-gui' is not recognized as an internal or external command
```

**Solution:**
```bash
# Reinstall in development mode
pip install -e .

# Or reinstall from source
pip install -U .

# Verify installation
pip show whisperjav
```

### Issue 2: GUI launches but shows blank window

**Symptoms:**
- Window opens but interface doesn't load
- Console shows asset loading errors

**Solution:**
1. Verify assets bundled correctly:
   ```bash
   # Check spec file datas section
   grep -A 3 "datas=" installer/whisperjav-gui.spec
   ```

2. Rebuild with clean flag:
   ```bash
   cd installer
   pyinstaller --clean --noconfirm whisperjav-gui.spec
   ```

3. Check file permissions on assets directory

### Issue 3: WebView2 Error on Windows

**Symptoms:**
```
WebView2 runtime not detected
```

**Solution:**
1. Install WebView2 Runtime:
   - Download from: https://developer.microsoft.com/en-us/microsoft-edge/webview2/
   - Or install Microsoft Edge browser

2. Verify installation:
   - Check registry keys (see webview_gui/main.py:check_webview2_windows())
   - Restart application

### Issue 4: Import Errors After Migration

**Symptoms:**
```
ModuleNotFoundError: No module named 'whisperjav.gui'
```

**Solution:**
1. Clear Python cache:
   ```bash
   find . -type d -name "__pycache__" -exec rm -rf {} +
   find . -name "*.pyc" -delete
   ```

2. Reinstall package:
   ```bash
   pip uninstall whisperjav
   pip install -e .
   ```

### Issue 5: Executable Doesn't Run

**Symptoms:**
- Double-click does nothing
- Or crashes immediately

**Solution:**
1. Run from command line to see errors:
   ```bash
   cd dist/whisperjav-gui
   whisperjav-gui.exe
   ```

2. Check dependencies bundled:
   ```bash
   # View spec file excludes - might be excluding needed modules
   grep "excludes=" installer/whisperjav-gui.spec
   ```

3. Rebuild without excludes:
   ```python
   # In spec file, comment out excludes temporarily
   # excludes=['tkinter', 'matplotlib', ...],
   ```

---

## Success Criteria

**Migration is considered successful when:**

✅ **Functionality:**
- [ ] `whisperjav-gui` launches PyWebView GUI on first try
- [ ] All features work (file selection, processing, console output)
- [ ] Keyboard shortcuts functional
- [ ] Tabs switch correctly
- [ ] Processing completes successfully
- [ ] WebView2 detection works (Windows)

✅ **Build & Distribution:**
- [ ] Executable builds without errors
- [ ] Executable runs on clean Windows machine
- [ ] Executable size reasonable (<150MB)
- [ ] Icon displays correctly in taskbar/window

✅ **Code Quality:**
- [ ] No Tkinter imports remain (except error fallback)
- [ ] No broken imports or missing modules
- [ ] Documentation accurate and up-to-date
- [ ] No orphaned files or references

✅ **User Experience:**
- [ ] Launch command simple and memorable (`whisperjav-gui`)
- [ ] No confusing command variants
- [ ] GUI looks professional and polished
- [ ] Error messages clear and helpful

✅ **Release Readiness:**
- [ ] Version number updated
- [ ] Release notes prepared
- [ ] GitHub release created
- [ ] Rollback plan documented and tested

---

## Appendix A: File Modification Summary

| File | Action | Description |
|---|---|---|
| `setup.py` | Modified | Entry points: whisperjav-gui → webview_gui, removed whisperjav-gui-web |
| `whisperjav/gui/` | Deleted | Entire Tkinter GUI directory removed |
| `whisperjav/gui_launcher.py` | Deleted | Standalone Tkinter launcher removed |
| `installer/whisperjav-gui-web.spec` | Renamed | → `whisperjav-gui.spec`, exe name updated |
| `installer/build_whisperjav_installer_web.bat` | Renamed | → `build_whisperjav_installer_gui.bat`, references updated |
| `installer/WhisperJAV_Launcher.py` | Modified | Module path: whisperjav.gui → whisperjav.webview_gui |
| `README.md` | Modified | GUI commands, features, references updated |
| `CLAUDE.md` | Modified | Entry points, commands updated |
| `BUILD_AND_DISTRIBUTE.md` | Modified | File names, commands updated (optional) |
| `MIGRATION_PLAN_PYWEBVIEW_GUI_TAKEOVER.md` | Created | This document |

---

## Appendix B: Command Reference

### Before Migration (v1.4.x)

```bash
whisperjav                # CLI (unchanged)
whisperjav-gui            # Tkinter GUI (OLD)
whisperjav-gui-web        # PyWebView GUI (NEW, temp name)
whisperjav-translate      # Translation CLI (unchanged)
```

### After Migration (v1.5.0)

```bash
whisperjav                # CLI (unchanged)
whisperjav-gui            # PyWebView GUI (PRIMARY)
whisperjav-translate      # Translation CLI (unchanged)
```

---

## Appendix C: Testing Checklist (Quick Reference)

**Pre-Flight:**
- [ ] Git tag created
- [ ] Working directory clean
- [ ] Branch correct

**Implementation:**
- [ ] setup.py updated
- [ ] whisperjav/gui/ deleted
- [ ] gui_launcher.py deleted
- [ ] Spec file renamed and updated
- [ ] Build script renamed and updated
- [ ] Launcher updated
- [ ] README.md updated
- [ ] CLAUDE.md updated

**Testing:**
- [ ] `pip install -e .` succeeds
- [ ] `whisperjav --help` works (CLI)
- [ ] `whisperjav-gui` launches PyWebView
- [ ] `whisperjav-gui-web` fails (expected)
- [ ] GUI all features work
- [ ] Executable builds
- [ ] Executable runs
- [ ] Clean machine test passes

**Commit:**
- [ ] All changes staged
- [ ] Descriptive commit message
- [ ] Pushed to repository

---

## Appendix D: Useful Commands

```bash
# Check entry points installed
pip show -f whisperjav | grep console_scripts

# Find Tkinter references
grep -r "tkinter" --include="*.py" --include="*.md" .

# Find whisperjav-gui-web references
grep -r "whisperjav-gui-web" --include="*.py" --include="*.md" --include="*.bat" .

# Check GUI module import
python -c "from whisperjav.webview_gui import main; print('OK')"

# Test API import
python -c "from whisperjav.webview_gui.api import WhisperJAVAPI; print('OK')"

# Build executable with verbose output
cd installer
pyinstaller --clean --noconfirm --log-level DEBUG whisperjav-gui.spec

# Check executable dependencies (Windows)
cd dist/whisperjav-gui
dumpbin /dependents whisperjav-gui.exe

# Test launcher directly
python installer/WhisperJAV_Launcher.py
```

---

## Document History

| Version | Date | Changes | Author |
|---|---|---|---|
| 1.0 | 2025-10-31 | Initial comprehensive migration plan | Claude Code |

---

**END OF MIGRATION PLAN**

This document is APPROVED and READY FOR IMPLEMENTATION.

For questions or issues during implementation, refer to:
- PyWebView documentation: `whisperjav/webview_gui/*.md`
- Original implementation: Phase 1-5 documentation in `.docs_archive/`
- Rollback plan: Section in this document

Good luck with the migration! 🚀
