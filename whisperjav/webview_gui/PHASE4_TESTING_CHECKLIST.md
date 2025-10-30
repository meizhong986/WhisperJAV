# Phase 4 Testing Checklist

**Purpose:** Comprehensive manual testing guide for Phase 4 integration
**Tester:** Developer / QA
**Date:** 2025-10-30

---

## Pre-Testing Setup

### Environment Preparation

- [ ] **Install dependencies:** `pip install -e .`
- [ ] **Verify installation:** `python -c "import webview; print('OK')"`
- [ ] **Prepare test files:**
  - [ ] 1 short video file (~30s, MP4/MKV)
  - [ ] 1 medium video file (~2-5min)
  - [ ] 1 folder with multiple videos
- [ ] **Clear output directory:** Remove previous test outputs

### Launch Application

```bash
# Standard launch
python -m whisperjav.webview_gui.main

# Debug launch (with DevTools)
set WHISPERJAV_DEBUG=1
python -m whisperjav.webview_gui.main
```

**Expected Result:**
- [ ] Window opens (1000x700 resolution)
- [ ] Title shows "WhisperJAV - Web GUI (Phase 1 Test)"
- [ ] Console shows "WhisperJAV GUI initialized" (green)
- [ ] Console shows "Ready to process video files" (info)
- [ ] Output directory field is populated with default path
- [ ] Start button is disabled (no files selected)

---

## Section 1: File Selection Tests

### Test 1.1: Add Files (Single)

**Steps:**
1. Click "Add File(s)" button
2. Select **1 video file** from file dialog
3. Click "Open"

**Expected Results:**
- [x] File dialog opens with video file filters
- [x] Selected file appears in file list
- [x] File shows with 📄 icon
- [x] Console logs: "Added 1 file(s)" (info)
- [x] Start button becomes enabled
- [x] Empty state message is hidden

### Test 1.2: Add Files (Multiple)

**Steps:**
1. Click "Add File(s)" button
2. Select **3 video files** (Ctrl+Click)
3. Click "Open"

**Expected Results:**
- [x] All 3 files appear in file list
- [x] Console logs: "Added 3 file(s)" (info)
- [x] Files are added without duplicates (if already present)

### Test 1.3: Add Files (Cancel)

**Steps:**
1. Click "Add File(s)" button
2. Click "Cancel" in file dialog

**Expected Results:**
- [x] Dialog closes
- [x] **No error message shown**
- [x] File list unchanged
- [x] No console logs

### Test 1.4: Add Folder

**Steps:**
1. Click "Add Folder" button
2. Select a folder containing videos
3. Click "Select Folder"

**Expected Results:**
- [x] Folder dialog opens
- [x] Selected folder appears in file list
- [x] Folder shows with 📁 icon
- [x] Console logs: "Added folder: [path]" (info)

### Test 1.5: Add Folder (Cancel)

**Steps:**
1. Click "Add Folder" button
2. Click "Cancel" in folder dialog

**Expected Results:**
- [x] Dialog closes
- [x] **No error message shown**
- [x] File list unchanged

### Test 1.6: Duplicate Prevention

**Steps:**
1. Add a file
2. Add the **same file** again

**Expected Results:**
- [x] File appears only once in list
- [x] Console logs: "Added 1 file(s)" (second time shows 0 if duplicate skipped)

### Test 1.7: File Selection (Single Click)

**Steps:**
1. Click on a file item in the list

**Expected Results:**
- [x] File becomes highlighted (blue background)
- [x] "Remove Selected" button becomes enabled

### Test 1.8: File Selection (Ctrl+Click)

**Steps:**
1. Click file 1
2. Ctrl+Click file 2
3. Ctrl+Click file 3

**Expected Results:**
- [x] Multiple files are selected (blue background)
- [x] Ctrl+Click on already-selected file deselects it

### Test 1.9: File Selection (Shift+Click)

**Steps:**
1. Click file 1
2. Shift+Click file 5

**Expected Results:**
- [x] Range selection: files 1-5 all selected

### Test 1.10: File Selection (Keyboard Navigation)

**Steps:**
1. Click a file to focus
2. Press **Arrow Down** key
3. Press **Arrow Up** key
4. Press **Shift+Arrow Down** (range select)

**Expected Results:**
- [x] Arrow keys navigate file list
- [x] Shift+Arrow selects range
- [x] Delete/Backspace removes selected items

### Test 1.11: Remove Selected

**Steps:**
1. Select 2 files
2. Click "Remove Selected" button

**Expected Results:**
- [x] Selected files removed from list
- [x] Console logs: "Removed 2 item(s)" (info)
- [x] Selection cleared

### Test 1.12: Clear All

**Steps:**
1. Add multiple files
2. Click "Clear" button

**Expected Results:**
- [x] All files removed from list
- [x] Console logs: "Cleared [N] item(s)" (info)
- [x] Empty state message reappears
- [x] Start button becomes disabled

---

## Section 2: Directory Controls

### Test 2.1: Browse Output Directory

**Steps:**
1. Click "Browse" button next to Output field
2. Select a folder
3. Click "Select Folder"

**Expected Results:**
- [x] Folder dialog opens
- [x] Output field updates with selected path
- [x] Console logs: "Output directory: [path]" (info)

### Test 2.2: Browse Output (Cancel)

**Steps:**
1. Click "Browse" button next to Output
2. Click "Cancel" in dialog

**Expected Results:**
- [x] Dialog closes
- [x] **No error message shown**
- [x] Output field unchanged

### Test 2.3: Open Output Folder (Exists)

**Steps:**
1. Ensure output path exists
2. Click "Open" button next to Output field

**Expected Results:**
- [x] File explorer opens showing the folder
- [x] Console logs: "Opened folder: [path]" (info)

### Test 2.4: Open Output Folder (Doesn't Exist)

**Steps:**
1. Set output path to non-existent folder
2. Click "Open" button

**Expected Results:**
- [x] Folder is created
- [x] File explorer opens showing the new folder
- [x] Console logs: "Opened folder: [path]" (info)

### Test 2.5: Open Output (Empty Path)

**Steps:**
1. Clear output field (manually edit to empty)
2. Click "Open" button

**Expected Results:**
- [x] Warning shown: "No Output Directory"
- [x] Console logs warning message

### Test 2.6: Browse Temp Directory

**Steps:**
1. Switch to "Transcription Adv. Options" tab
2. Click "Browse" button next to Temp dir field
3. Select a folder
4. Click "Select Folder"

**Expected Results:**
- [x] Folder dialog opens
- [x] Temp dir field updates with selected path
- [x] Console logs: "Temp directory: [path]" (info)

---

## Section 3: Process Execution

### Test 3.1: Start with No Files

**Steps:**
1. Ensure file list is empty
2. Click "Start" button

**Expected Results:**
- [x] Error dialog appears: "No Files Selected"
- [x] Console logs error message (red)
- [x] Process does **not** start

### Test 3.2: Start with Files (Successful Completion)

**Steps:**
1. Add 1 short video file (~30s)
2. Configure: Mode=faster, Sensitivity=balanced
3. Click "Start" button
4. Wait for completion

**Expected Results:**
- [x] Start button becomes disabled
- [x] Cancel button becomes enabled
- [x] Progress bar animates (indeterminate)
- [x] Status label shows "Running..."
- [x] Console shows command: `> whisperjav.main [args]`
- [x] Logs stream in real-time (100ms updates)
- [x] Status updates periodically (500ms)
- [x] Upon completion:
  - [x] Progress bar fills to 100%
  - [x] Status label shows "Completed"
  - [x] Console logs: "✓ Process Completed: Transcription finished successfully"
  - [x] Start button re-enabled
  - [x] Cancel button disabled

### Test 3.3: Start with Multiple Files (Async)

**Steps:**
1. Add 2 short video files
2. Configure: Async processing enabled, Max workers=2
3. Click "Start" button
4. Wait for completion

**Expected Results:**
- [x] Both files processed
- [x] Logs show parallel processing (if async is working)
- [x] Completion status shown
- [x] 2 SRT files created in output directory

### Test 3.4: Cancel During Execution

**Steps:**
1. Add 1 medium video file (2-5 min)
2. Click "Start" button
3. Wait 5-10 seconds (process running)
4. Click "Cancel" button

**Expected Results:**
- [x] Process terminates within 5 seconds
- [x] Console logs: "Process cancelled by user" (warning)
- [x] Status label shows "Cancelled"
- [x] Progress bar resets
- [x] Start button re-enabled
- [x] Cancel button disabled
- [x] Log streaming stops

### Test 3.5: Process Fails (Error Exit Code)

**Steps:**
1. Add an invalid file (e.g., text file renamed to .mp4)
2. Click "Start" button
3. Wait for process to fail

**Expected Results:**
- [x] Process exits with non-zero code
- [x] Error dialog appears: "Process Failed: Process exited with code [N]"
- [x] Console logs error message (red)
- [x] Status label shows "Error (exit code: [N])"
- [x] Progress bar resets
- [x] UI returns to idle state

### Test 3.6: Multiple Runs in Sequence

**Steps:**
1. Add file, run process, wait for completion
2. Add different file, run process again
3. Repeat once more

**Expected Results:**
- [x] Each run completes successfully
- [x] UI resets properly between runs
- [x] No memory leaks or stuck intervals
- [x] Logs from previous runs remain visible

### Test 3.7: Clear Console During Execution

**Steps:**
1. Start a process
2. Wait for logs to appear
3. Click "Clear" button in Console section

**Expected Results:**
- [x] Console is cleared
- [x] Process continues running
- [x] New logs continue streaming

---

## Section 4: Form Options

### Test 4.1: Mode Selection

**Steps:**
1. Select each mode radio button:
   - balanced
   - fast
   - faster
2. Start process for each mode

**Expected Results:**
- [x] All modes work correctly
- [x] Command log shows correct `--mode` argument
- [x] Process executes with selected mode

### Test 4.2: Sensitivity Selection

**Steps:**
1. Select each sensitivity option:
   - conservative
   - balanced
   - aggressive
2. Start process for each sensitivity

**Expected Results:**
- [x] All sensitivities work correctly
- [x] Command log shows correct `--sensitivity` argument

### Test 4.3: Output Language Selection

**Steps:**
1. Select "japanese"
2. Start process
3. Check output SRT file (should be in Japanese)
4. Select "english-direct"
5. Start process
6. Check output SRT file (should be in English)

**Expected Results:**
- [x] Both languages work correctly
- [x] Command log shows correct `--subs-language` argument
- [x] Output files contain correct language

### Test 4.4: Model Override (Enable/Disable)

**Steps:**
1. **Uncheck** "Model override" checkbox
2. Verify model dropdown is **disabled**
3. **Check** "Model override" checkbox
4. Verify model dropdown is **enabled**
5. Select "large-v2"
6. Start process

**Expected Results:**
- [x] Checkbox toggles dropdown state correctly
- [x] When unchecked: no `--model` argument in command
- [x] When checked: command log shows `--model large-v2`

### Test 4.5: Model Selection

**Steps:**
1. Enable model override
2. Select each model:
   - large-v3
   - large-v2
   - turbo
3. Start process for each

**Expected Results:**
- [x] All models selectable
- [x] Command log shows correct `--model` argument

### Test 4.6: Async Processing (Enable/Disable)

**Steps:**
1. **Uncheck** "Async processing" checkbox
2. Verify command has no async args
3. **Check** "Async processing" checkbox
4. Set Max workers to 4
5. Start process

**Expected Results:**
- [x] Command log shows `--async-processing --max-workers 4`
- [x] Process runs with async processing

### Test 4.7: Max Workers Input

**Steps:**
1. Enable async processing
2. Test max workers input:
   - Type "1"
   - Type "16"
   - Try typing "0" or "17" (should be clamped)

**Expected Results:**
- [x] Input accepts values 1-16
- [x] Command log shows correct `--max-workers [N]`

### Test 4.8: Opening Credit

**Steps:**
1. Enter "Produced by Test Team" in Opening credit field
2. Start process
3. Check output SRT file for credit

**Expected Results:**
- [x] Command log shows `--credit "Produced by Test Team"`
- [x] SRT file contains opening credit

### Test 4.9: Keep Temp Files

**Steps:**
1. **Check** "Keep temp files" checkbox
2. Start process
3. Check if temp files remain after completion

**Expected Results:**
- [x] Command log shows `--keep-temp`
- [x] Temp files are not deleted

### Test 4.10: Temp Directory

**Steps:**
1. Browse and select a custom temp directory
2. Start process
3. Check if temp files are created in custom location

**Expected Results:**
- [x] Command log shows `--temp-dir [path]`
- [x] Temp files created in custom directory

### Test 4.11: Verbosity Levels

**Steps:**
1. Select each verbosity level:
   - quiet
   - summary
   - normal
   - verbose
2. Start process for each

**Expected Results:**
- [x] Command log shows correct `--verbosity` argument
- [x] Console output varies by verbosity level

### Test 4.12: WIP Features (Disabled)

**Steps:**
1. Try clicking the following checkboxes:
   - Adaptive classification
   - Adaptive audio enhancements
   - Smart postprocessing

**Expected Results:**
- [x] All three checkboxes are **disabled**
- [x] Tooltip shows "Work in Progress - Not yet implemented"
- [x] Cannot be checked

---

## Section 5: Tab Navigation

### Test 5.1: Tab Switching (Mouse)

**Steps:**
1. Click "Transcription Mode" tab
2. Click "Transcription Adv. Options" tab
3. Click "Transcription Mode" tab again

**Expected Results:**
- [x] Active tab changes
- [x] Active panel content updates
- [x] Tab button shows active state (underline)

### Test 5.2: Tab Switching (Keyboard)

**Steps:**
1. Focus tab bar (click a tab)
2. Press **Arrow Right** key
3. Press **Arrow Left** key
4. Press **Home** key
5. Press **End** key

**Expected Results:**
- [x] Arrow Right moves to next tab
- [x] Arrow Left moves to previous tab
- [x] Home moves to first tab
- [x] End moves to last tab
- [x] Active panel content updates

### Test 5.3: Tab Switching During Execution

**Steps:**
1. Start a process
2. Switch between tabs while process is running

**Expected Results:**
- [x] Tabs switch correctly
- [x] Process continues running
- [x] Logs continue streaming in Console section

---

## Section 6: Error Handling

### Test 6.1: File Dialog Error

**Steps:**
1. Force an error by denying file system permissions (if possible)
2. Try opening file dialog

**Expected Results:**
- [x] Error dialog shown: "File Selection Error: [details]"
- [x] Console logs error (red)

### Test 6.2: Start Process Error (Invalid Args)

**Steps:**
1. Manually corrupt options (e.g., via browser DevTools if accessible)
2. Try starting process

**Expected Results:**
- [x] Error dialog shown: "Start Failed: [details]"
- [x] Console logs error (red)
- [x] Process does not start

### Test 6.3: Cancel Non-Running Process

**Steps:**
1. Ensure no process is running
2. Click "Cancel" button (should be disabled, but test via DevTools if accessible)

**Expected Results:**
- [x] Button is disabled (cannot be clicked normally)
- [x] If forced: API returns "No process running"

### Test 6.4: Open Non-Existent Folder (Permission Denied)

**Steps:**
1. Set output path to a protected location (e.g., C:\Windows\System32)
2. Click "Open" button

**Expected Results:**
- [x] Error dialog or system permission denied dialog shown
- [x] Console logs error details

---

## Section 7: UI State Management

### Test 7.1: Button States (No Files)

**Steps:**
1. Clear all files from list

**Expected Results:**
- [x] Start button: **disabled**
- [x] Cancel button: **disabled**
- [x] Remove Selected button: **disabled**
- [x] Clear button: **disabled**

### Test 7.2: Button States (Files Added)

**Steps:**
1. Add 2 files

**Expected Results:**
- [x] Start button: **enabled**
- [x] Cancel button: **disabled**
- [x] Remove Selected button: **disabled** (no selection)
- [x] Clear button: **enabled**

### Test 7.3: Button States (File Selected)

**Steps:**
1. Add 2 files
2. Click on 1 file

**Expected Results:**
- [x] Start button: **enabled**
- [x] Cancel button: **disabled**
- [x] Remove Selected button: **enabled**
- [x] Clear button: **enabled**

### Test 7.4: Button States (Process Running)

**Steps:**
1. Add file and start process

**Expected Results:**
- [x] Start button: **disabled**
- [x] Cancel button: **enabled**
- [x] Remove Selected button: **disabled**
- [x] Clear button: **disabled**
- [x] Add Files button: **enabled** (not disabled)
- [x] Add Folder button: **enabled** (not disabled)

### Test 7.5: Button States (Process Completed)

**Steps:**
1. Wait for process to complete

**Expected Results:**
- [x] Start button: **enabled** (if files exist)
- [x] Cancel button: **disabled**
- [x] Remove Selected button: **enabled** (if selection exists)
- [x] Clear button: **enabled** (if files exist)

---

## Section 8: Log Streaming

### Test 8.1: Real-Time Log Updates

**Steps:**
1. Start a process
2. Observe console output

**Expected Results:**
- [x] Logs appear in real-time (< 200ms delay)
- [x] Console auto-scrolls to bottom
- [x] No log lines are skipped or lost

### Test 8.2: Log Formatting

**Steps:**
1. Start a process
2. Check console output for:
   - Command line (starts with `>`)
   - Progress messages
   - Error messages (if any)
   - Success messages

**Expected Results:**
- [x] All log lines displayed correctly
- [x] UTF-8 characters (Japanese, checkmarks) render correctly
- [x] No garbled or corrupted text

### Test 8.3: Log Polling Stops on Completion

**Steps:**
1. Start a process
2. Open browser DevTools (if debug mode enabled)
3. Wait for completion
4. Check if polling continues

**Expected Results:**
- [x] Log polling stops after completion
- [x] No more API calls to `get_logs()` after process ends
- [x] No console errors

---

## Section 9: Status Monitoring

### Test 9.1: Status Label Updates

**Steps:**
1. Start a process
2. Observe status label

**Expected Results:**
- [x] Before start: "Idle"
- [x] During execution: "Running..."
- [x] On completion: "Completed"
- [x] On error: "Error (exit code: [N])"
- [x] On cancel: "Cancelled"

### Test 9.2: Status Polling Stops on Completion

**Steps:**
1. Start a process
2. Open browser DevTools (if debug mode enabled)
3. Wait for completion
4. Check if polling continues

**Expected Results:**
- [x] Status polling stops after completion
- [x] No more API calls to `get_process_status()` after process ends

---

## Section 10: Cross-Browser Testing

### Test 10.1: Chrome/Chromium (Windows)

**Steps:**
1. Run GUI on Windows with default PyWebView backend

**Expected Results:**
- [x] All features work correctly
- [x] No JavaScript errors
- [x] Native file dialogs work

### Test 10.2: Edge WebView2 (Windows)

**Steps:**
1. Force Edge backend (if possible)
2. Test all features

**Expected Results:**
- [x] All features work correctly

### Test 10.3: Safari WebKit (macOS)

**Steps:**
1. Run GUI on macOS
2. Test all features

**Expected Results:**
- [x] All features work correctly
- [x] Native file dialogs use macOS style

### Test 10.4: GTK WebKit (Linux)

**Steps:**
1. Run GUI on Linux
2. Test all features

**Expected Results:**
- [x] All features work correctly
- [x] Native file dialogs use GTK style

---

## Section 11: Performance Testing

### Test 11.1: Memory Usage (Long Session)

**Steps:**
1. Run 5 processes in sequence
2. Monitor memory usage (Task Manager / Activity Monitor)

**Expected Results:**
- [x] Memory usage remains stable
- [x] No memory leaks
- [x] Polling intervals properly cleared

### Test 11.2: CPU Usage (Idle)

**Steps:**
1. Launch GUI
2. Leave idle for 1 minute
3. Monitor CPU usage

**Expected Results:**
- [x] CPU usage near 0% when idle
- [x] No unnecessary polling or background activity

### Test 11.3: CPU Usage (Active)

**Steps:**
1. Start a process
2. Monitor CPU usage during execution

**Expected Results:**
- [x] CPU usage increases (expected - transcription is CPU-intensive)
- [x] GUI remains responsive
- [x] Log streaming doesn't stutter

### Test 11.4: Large File List

**Steps:**
1. Add 50+ files to list
2. Test scrolling, selection, removal

**Expected Results:**
- [x] UI remains responsive
- [x] Scrolling is smooth
- [x] Selection works correctly

### Test 11.5: Long-Running Process

**Steps:**
1. Start a process that takes 10+ minutes
2. Leave GUI running
3. Check if logs continue streaming

**Expected Results:**
- [x] Logs stream continuously
- [x] No timeouts or connection issues
- [x] Process completes successfully

---

## Section 12: Edge Cases

### Test 12.1: Empty Output Directory Field

**Steps:**
1. Clear output directory field
2. Try starting process

**Expected Results:**
- [x] Process uses default output directory
- [x] Or shows error if field is required

### Test 12.2: Very Long File Paths

**Steps:**
1. Add files with very long paths (> 200 characters)
2. Start process

**Expected Results:**
- [x] File paths display correctly (truncated with ellipsis if needed)
- [x] Process handles long paths correctly

### Test 12.3: Special Characters in File Names

**Steps:**
1. Add files with special characters: `[test] video (2023).mp4`
2. Start process

**Expected Results:**
- [x] File names display correctly
- [x] Process handles special characters correctly

### Test 12.4: Japanese Characters in File Names

**Steps:**
1. Add files with Japanese names: `テスト動画.mp4`
2. Start process

**Expected Results:**
- [x] File names display correctly (UTF-8)
- [x] Process handles Japanese characters correctly

### Test 12.5: Window Resize

**Steps:**
1. Resize window to minimum size (800x600)
2. Resize to maximum (full screen)
3. Test all UI elements

**Expected Results:**
- [x] UI scales correctly
- [x] No layout breaks
- [x] All elements remain accessible

---

## Section 13: Initialization Tests

### Test 13.1: Default Output Directory Loading

**Steps:**
1. Launch GUI
2. Check output directory field

**Expected Results:**
- [x] Output field populated with default path
- [x] Path exists or can be created
- [x] Console logs no errors

### Test 13.2: PyWebView Bridge Connection

**Steps:**
1. Launch GUI with debug mode
2. Open browser DevTools console
3. Check for "PyWebView API ready!" message

**Expected Results:**
- [x] Console logs: "PyWebView API ready!"
- [x] Console logs: "PyWebView bridge connected" (success)
- [x] API methods are accessible

### Test 13.3: Fallback Behavior (API Unavailable)

**Steps:**
1. Simulate API unavailability (e.g., delay startup)
2. Check if fallback values are used

**Expected Results:**
- [x] GUI loads with fallback default output directory
- [x] Console logs warning: "Using fallback output directory"

---

## Post-Testing Cleanup

### Cleanup Steps

- [ ] Delete test output files
- [ ] Clear test directories
- [ ] Close GUI application
- [ ] Review console logs for any errors
- [ ] Document any issues found

---

## Test Results Summary

### Statistics

- **Total Tests:** 130+
- **Passed:** _____
- **Failed:** _____
- **Skipped:** _____

### Issues Found

_(Document any bugs, unexpected behavior, or UX issues)_

1. Issue: _____________________
   - Severity: Critical / High / Medium / Low
   - Steps to reproduce: _____
   - Expected: _____
   - Actual: _____

2. Issue: _____________________
   ...

### Performance Notes

_(Document any performance concerns)_

- Memory usage: _____ MB
- CPU usage (idle): _____ %
- CPU usage (active): _____ %
- Log streaming latency: _____ ms

### Browser Compatibility

- [ ] Chrome/Chromium: Pass / Fail
- [ ] Edge WebView2: Pass / Fail
- [ ] Safari WebKit: Pass / Fail
- [ ] GTK WebKit: Pass / Fail

---

## Tester Sign-Off

**Tester Name:** _____________________
**Date:** _____________________
**Signature:** _____________________

**Status:** ☐ All Tests Passed ☐ Issues Found ☐ Needs Retesting

---

**End of Testing Checklist**
