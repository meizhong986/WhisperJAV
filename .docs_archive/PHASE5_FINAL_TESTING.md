# Phase 5: Final Testing Checklist

## Pre-Build Testing (Development Mode)

Run the application in development mode: `python -m whisperjav.webview_gui.main`

### Core Functionality
- [ ] Application launches without errors
- [ ] Window displays with correct title "WhisperJAV GUI"
- [ ] Window icon displays correctly (if available)
- [ ] All UI sections render properly (Source, Destination, Tabs, Console)

### File Management
- [ ] "Add File(s)" button opens file dialog
- [ ] "Add Folder" button opens folder dialog
- [ ] Files are added to the list correctly
- [ ] Folders are added to the list correctly
- [ ] File selection works (single click)
- [ ] Multi-select works (Ctrl+Click)
- [ ] Range select works (Shift+Click)
- [ ] "Remove Selected" button works
- [ ] "Clear" button works
- [ ] Drag-and-drop files works (if supported by browser backend)

### Output Directory
- [ ] Default output directory loads on startup
- [ ] "Browse" button opens directory dialog
- [ ] Selected directory updates in the text field
- [ ] "Open" button opens directory in file explorer

### Tab Navigation
- [ ] Tab 1 (Transcription Mode) displays correctly
- [ ] Tab 2 (Advanced Options) displays correctly
- [ ] Tab switching works via mouse click
- [ ] Tab switching works via keyboard (Arrow keys)

### Form Options
- [ ] Mode selection (Balanced/Fast/Faster) works
- [ ] Sensitivity dropdown works
- [ ] Output language dropdown works
- [ ] Model override checkbox enables/disables dropdown
- [ ] Async processing checkbox works
- [ ] Max workers input accepts numbers
- [ ] Opening credit text input works
- [ ] Keep temp files checkbox works
- [ ] Temp directory browse works

### Process Execution
- [ ] "Start" button is disabled when no files selected
- [ ] "Start" button is enabled when files are selected
- [ ] Process starts successfully
- [ ] Progress bar shows indeterminate animation
- [ ] Status label updates to "Running..."
- [ ] Console shows command being executed
- [ ] Console streams log output in real-time
- [ ] "Cancel" button is enabled during execution
- [ ] "Cancel" button terminates process
- [ ] Process completes successfully (test with small file)
- [ ] Completion message shows in console
- [ ] Status updates to "Completed"

### Loading States
- [ ] "Add File(s)" button shows loading spinner during dialog
- [ ] "Add Folder" button shows loading spinner during dialog
- [ ] Buttons are properly disabled during loading

### Keyboard Shortcuts
- [ ] Ctrl+O opens file selection dialog
- [ ] Ctrl+R starts processing (when files selected)
- [ ] Escape cancels process (when running)
- [ ] F1 opens About dialog
- [ ] F5 warns before refreshing (when process running)

### About Dialog
- [ ] Footer "About" link opens modal
- [ ] About dialog displays correctly
- [ ] Version number shows (1.4.5)
- [ ] All content displays properly
- [ ] Close button works
- [ ] Clicking outside modal closes it
- [ ] Escape key closes modal

### Error Handling
- [ ] No JavaScript errors in console
- [ ] No Python errors in console
- [ ] Proper error messages for invalid operations

### WebView2 Detection (Windows Only)
- [ ] Application checks for WebView2 on Windows
- [ ] Error dialog shows if WebView2 missing (test on clean VM if possible)
- [ ] Error message provides download link

---

## Build Testing

Build the executable: `cd installer && build_whisperjav_installer_web.bat`

### Build Process
- [ ] PyInstaller is installed
- [ ] Build script runs without errors
- [ ] No missing module warnings
- [ ] Icon is bundled correctly
- [ ] All assets (HTML/CSS/JS) are bundled
- [ ] Executable created at `installer/dist/whisperjav-gui-web/whisperjav-gui-web.exe`

### Executable Metadata (Windows)
- [ ] Right-click exe → Properties shows icon
- [ ] Details tab shows correct version (1.4.5)
- [ ] Details tab shows correct product name
- [ ] Details tab shows correct description
- [ ] Details tab shows copyright information

---

## Packaged Executable Testing

Run the built executable: `installer\dist\whisperjav-gui-web\whisperjav-gui-web.exe`

### Startup
- [ ] Executable launches without console window (unless debugging)
- [ ] No Python installation required
- [ ] Window displays with correct icon
- [ ] Application initializes properly

### Full Workflow Test
- [ ] Add a test video file
- [ ] Select output directory
- [ ] Choose transcription mode (Balanced recommended)
- [ ] Start processing
- [ ] Monitor progress and logs
- [ ] Process completes successfully
- [ ] Output SRT file is created
- [ ] Can open output folder and verify file
- [ ] Can run multiple consecutive processes

### Functionality Re-Test
Repeat all "Core Functionality" tests from Pre-Build section with the executable.

### Edge Cases
- [ ] Works with Japanese filenames (e.g., テスト.mp4)
- [ ] Works with paths containing spaces
- [ ] Works with long file paths (>260 characters if possible)
- [ ] Handles disk full gracefully
- [ ] Handles permission errors gracefully
- [ ] Shows meaningful error if video codec not supported
- [ ] Shows meaningful error if file doesn't exist
- [ ] Multiple consecutive runs work without restart

---

## Clean Machine Testing (Critical)

Test on a Windows machine that:
- Has never had Python installed
- Has never had WhisperJAV installed
- Is a fresh Windows install (or VM)

### Installation & First Run
- [ ] Copy `whisperjav-gui-web` folder to clean machine
- [ ] Run executable directly
- [ ] WebView2 check works correctly
- [ ] If WebView2 missing, error dialog appears with instructions
- [ ] After installing WebView2, application launches
- [ ] All functionality works on clean machine

---

## Performance Testing

### Responsiveness
- [ ] UI remains responsive during file selection
- [ ] UI remains responsive during process execution
- [ ] Log scrolling is smooth
- [ ] Tab switching is instant
- [ ] No lag when adding many files (test with 10+ files)

### Memory Usage
- [ ] Application doesn't leak memory during execution
- [ ] Memory usage is reasonable (check Task Manager)
- [ ] Application closes cleanly without hanging

---

## Documentation Testing

### User Guide
- [ ] USER_GUIDE.md is clear and comprehensive
- [ ] Installation instructions are correct
- [ ] Screenshots (if any) match current UI
- [ ] Troubleshooting section covers common issues

### Testing Checklist
- [ ] This checklist is complete
- [ ] All tests are reproducible

---

## Sign-off

**Tested by:** ___________________
**Date:** ___________________
**Build Version:** 1.4.5
**Platform:** Windows 10/11

**Overall Status:**
- [ ] All critical tests passed
- [ ] All non-critical tests passed
- [ ] Known issues documented
- [ ] Ready for distribution

**Notes:**
_______________________________________________________
_______________________________________________________
_______________________________________________________
