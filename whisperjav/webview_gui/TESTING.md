# Phase 3 Frontend - Testing Guide

## Quick Start

### Option 1: Development Server (Recommended for Frontend Work)

```bash
python -m whisperjav.webview_gui.dev_server
```

**Features:**
- Hot reload: Edit HTML/CSS/JS → Save → Refresh (Ctrl+R)
- Debug mode enabled (F12 for developer console)
- 1000x700 window
- No API backend (mock data only)

**Expected Output:**
```
============================================================
WhisperJAV Web GUI - Development Server
============================================================
Serving: C:\...\whisperjav\webview_gui\assets\index.html
Window size: 1000x700
Debug mode: ENABLED

Usage:
  1. Edit HTML/CSS/JS files in assets/ directory
  2. Save your changes
  3. Refresh the window (Ctrl+R / Cmd+R) to see updates

Note: API calls will fail in dev mode (Phase 3)
      For full testing, use: python -m whisperjav.webview_gui.main
============================================================
```

### Option 2: Main Application (Phase 1-2 API integration)

```bash
python -m whisperjav.webview_gui.main
```

**Features:**
- Full PyWebView API integration
- Backend API available (but not wired to UI yet in Phase 3)
- Production mode

---

## Testing Checklist

### Visual Appearance

Open the dev server and verify:

- [ ] **Header**
  - Deep blue background (#024873)
  - White text: "WhisperJAV" and "Simple Runner"

- [ ] **Source Section**
  - White background with border
  - Empty state shows: 📂 "No files selected" with hint
  - 4 buttons: Add File(s), Add Folder, Remove Selected, Clear
  - Remove/Clear buttons are disabled (no files yet)

- [ ] **Destination Section**
  - Output path field (read-only, gray background)
  - Browse and Open buttons

- [ ] **Tabs Section**
  - Tab bar with white background
  - Two tabs: "Transcription Mode" and "Transcription Adv. Options"
  - First tab is active (darker background #F0F2F5, bold blue text)
  - Second tab is inactive (lighter background #F7F8FA, gray text)
  - Active tab appears "lifted" and connected to content area

- [ ] **Tab 1 Content**
  - Mode radio buttons: balanced (checked), fast, faster
  - Sensitivity dropdown: balanced (selected)
  - Output language dropdown: japanese (selected)
  - Two info boxes with blue left border explaining options

- [ ] **Tab 2 Content**
  - 3 disabled checkboxes (WIP features) with gray text
  - Verbosity dropdown: summary (selected)
  - Model override checkbox (unchecked)
  - Model dropdown (disabled, grayed out)
  - Async processing checkbox (unchecked)
  - Max workers number input: 1
  - Opening credit text input (empty, with placeholder)
  - Keep temp files checkbox (unchecked)
  - Temp dir input (empty, with placeholder) + Browse button

- [ ] **Run Controls**
  - Progress bar (empty, gray)
  - Status label: "Idle"
  - Start button (disabled - no files)
  - Cancel button (disabled - not running)

- [ ] **Console**
  - Dark background (#1E1E1E)
  - Light gray text (#D4D4D4)
  - Shows: "Ready."
  - Clear button in header

- [ ] **Footer**
  - Gray text: "WhisperJAV - Subtitle generation for Japanese Adult Videos"

---

## Functional Testing

### Test 1: Tab Switching

1. Click "Transcription Adv. Options" tab
   - ✅ Tab becomes active (bold blue text, darker background)
   - ✅ First tab becomes inactive (lighter background)
   - ✅ Content switches to show advanced options
   - ✅ URL does NOT change (no page reload)

2. Click "Transcription Mode" tab
   - ✅ Switches back to first tab
   - ✅ Content shows mode/sensitivity/language options

3. **Keyboard navigation:**
   - Click on tab bar to focus
   - Press Arrow Right → Second tab activates
   - Press Arrow Left → First tab activates
   - Press Home → First tab
   - Press End → Last tab

**Expected:** Smooth transitions, no flickering, active tab clearly visible

---

### Test 2: File List Management

1. Click "Add File(s)" button
   - ✅ Console shows: "Added 3 mock files (Phase 3 test)"
   - ✅ File list shows 3 items with 📄 icons
   - ✅ Files: `C:\Videos\sample1.mp4`, `sample2.mkv`, `sample3.avi`
   - ✅ Remove Selected and Clear buttons become enabled
   - ✅ Empty state disappears

2. Click "Add Folder" button
   - ✅ Console shows: "Added mock folder (Phase 3 test)"
   - ✅ File list shows folder with 📁 icon
   - ✅ Folder: `C:\Videos\JAV_Collection`

3. **Single selection:**
   - Click on first file
   - ✅ File becomes selected (blue background, white text)
   - ✅ Remove Selected button enabled

4. **Multi-select (Ctrl+Click):**
   - Hold Ctrl, click on second file
   - ✅ Both files selected
   - Hold Ctrl, click on second file again
   - ✅ Second file deselected (toggle)

5. **Range select (Shift+Click):**
   - Click on first file
   - Hold Shift, click on third file
   - ✅ Files 1, 2, 3 all selected

6. Click "Remove Selected" button
   - ✅ Selected files removed from list
   - ✅ Console shows: "Removed X item(s)"

7. Click "Clear" button
   - ✅ All files removed
   - ✅ Empty state reappears
   - ✅ Remove/Clear buttons disabled
   - ✅ Console shows: "Cleared all files"

**Keyboard navigation:**
- Click in file list to focus
- Press Arrow Down → Next file selected
- Press Arrow Up → Previous file selected
- Press Delete → Selected file(s) removed

---

### Test 3: Form Controls

1. **Model Override:**
   - Model dropdown is disabled (grayed out)
   - Check "Model override" checkbox
   - ✅ Model dropdown becomes enabled
   - ✅ Can select different models (large-v3, large-v2, turbo)
   - Uncheck "Model override"
   - ✅ Model dropdown disabled again

2. **Mode Selection:**
   - Click "fast" radio button
   - ✅ "balanced" deselects
   - ✅ "fast" becomes selected

3. **Dropdowns:**
   - Change sensitivity to "aggressive"
   - Change language to "english-direct"
   - Change verbosity to "verbose"
   - ✅ All changes reflected

4. **Number Input:**
   - Change max workers to 4
   - Try to set to 0 → Min is 1
   - Try to set to 20 → Max is 16

5. **Text Inputs:**
   - Type in opening credit: "Produced by Test"
   - Type in temp dir: "C:\Temp\WhisperJAV"
   - ✅ Text appears in fields

---

### Test 4: Run Controls

1. **Initial State:**
   - Start button is disabled (no files)
   - Cancel button is disabled (not running)

2. Add mock files (Add File(s) button)
   - ✅ Start button becomes enabled

3. Click "Start" button
   - ✅ Progress bar shows indeterminate animation (moving bar)
   - ✅ Status changes to "Running..."
   - ✅ Console shows: "> Starting transcription (Phase 3 mock)"
   - ✅ Console shows: "Mode: balanced, Sensitivity: balanced"
   - ✅ Console shows: "Files: 3"
   - ✅ Start button becomes disabled
   - ✅ Cancel button becomes enabled

4. Wait 3 seconds
   - ✅ Progress bar fills to 100%
   - ✅ Status changes to "Done"
   - ✅ Console shows: "Process completed successfully (Phase 3 mock)" in green
   - ✅ Start button re-enabled
   - ✅ Cancel button disabled

5. Click "Start" again, then click "Cancel"
   - ✅ Progress bar resets to 0%
   - ✅ Status changes to "Cancelled"
   - ✅ Console shows: "Process cancelled by user" in yellow
   - ✅ Buttons return to normal state

---

### Test 5: Console

1. **Auto-scroll:**
   - Add many files (click Add Files multiple times)
   - ✅ Console auto-scrolls to show latest message

2. **Color coding:**
   - Info messages: Light blue (#4FC3F7)
   - Success messages: Green (#66BB6A)
   - Warning messages: Orange/Yellow (#FFA726)
   - Error messages: Red (#EF5350)
   - Command messages: Gold (#FFD700)

3. **Clear console:**
   - Click "Clear" button in console header
   - ✅ Console clears to show only "Ready."

---

### Test 6: Hover States

1. Hover over buttons
   - ✅ Background color darkens slightly
   - ✅ Cursor changes to pointer

2. Hover over inactive tab
   - ✅ Background lightens to #F2F4FA (between inactive/active)
   - ✅ Text color darkens slightly

3. Hover over file list item
   - ✅ Background lightens
   - ✅ Border changes to blue

---

### Test 7: Focus States (Keyboard Navigation)

1. Press Tab repeatedly
   - ✅ Focus moves through all interactive elements
   - ✅ Blue outline visible on focused element

2. Focus on button, press Enter or Space
   - ✅ Button activates

3. Focus on tab, press Arrow keys
   - ✅ Tabs switch

4. Focus on file list, press Arrow keys
   - ✅ Selection changes

---

### Test 8: Responsive Behavior

1. Resize window smaller (< 900px wide)
   - ✅ Form rows stack vertically
   - ✅ Tabs adjust width
   - ✅ No horizontal scrolling (except console)

2. Resize window very small
   - ✅ Layout remains usable
   - ✅ Buttons wrap to new lines if needed

---

## Browser Testing (Optional)

For CSS-only testing, open `index.html` directly in browser:

```bash
# Open in default browser
start whisperjav/webview_gui/assets/index.html
# or
open whisperjav/webview_gui/assets/index.html  # macOS
```

**Note:** JavaScript functionality will be limited without PyWebView bridge.

---

## Known Issues (Phase 3)

These are expected and will be fixed in Phase 4:

1. **File dialogs don't open** - Mock data used instead
2. **Browse buttons do nothing** - Just log to console
3. **No real subprocess** - 3-second mock simulation
4. **No real progress** - Indeterminate animation only
5. **No error handling** - Basic validation only

---

## Console Testing (Developer Tools)

Press F12 (or Cmd+Opt+I on Mac) to open developer console:

### Check State

```javascript
// View current state
AppState

// Check file list
AppState.selectedFiles

// Check selected indices
AppState.selectedIndices

// Check active tab
AppState.activeTab
```

### Manual Testing

```javascript
// Add test file
AppState.selectedFiles.push('C:\\Test\\video.mp4')
FileListManager.render()

// Switch tab
TabManager.switchTab('tab2')

// Log message
ConsoleManager.log('Test message', 'success')

// Set progress
ProgressManager.setProgress(50)

// Get form data
FormManager.getFormData()
```

---

## Performance Testing

1. **Large file list:**
   - Add 100+ files manually in console:
     ```javascript
     for (let i = 0; i < 100; i++) {
         AppState.selectedFiles.push(`C:\\Videos\\file${i}.mp4`);
     }
     FileListManager.render();
     ```
   - ✅ Rendering should be smooth
   - ✅ Selection should be responsive
   - ✅ Scrolling should be smooth

2. **Rapid tab switching:**
   - Click tabs rapidly
   - ✅ No lag or flickering

3. **Console spam:**
   - Log many messages:
     ```javascript
     for (let i = 0; i < 1000; i++) {
         ConsoleManager.log(`Message ${i}`, 'info');
     }
     ```
   - ✅ Auto-scroll works
   - ✅ No memory issues

---

## Accessibility Testing

1. **Keyboard-only navigation:**
   - Unplug mouse
   - Navigate entire UI with keyboard
   - ✅ All functions accessible

2. **Screen reader (optional):**
   - Enable Windows Narrator or macOS VoiceOver
   - ✅ ARIA attributes read correctly
   - ✅ Button labels make sense

3. **High contrast mode:**
   - Enable Windows High Contrast
   - ✅ UI remains usable
   - ✅ Focus indicators visible

---

## Bug Reporting Template

If you find an issue, report it with this format:

```
**Issue:** [Brief description]

**Steps to Reproduce:**
1. Open dev server
2. Click "Add Files"
3. [etc.]

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Screenshot:** [If applicable]

**Browser/OS:** Windows 11, Edge WebView2

**Console Errors:** [Copy from F12 console]
```

---

## Success Criteria

Phase 3 is successful if:

- ✅ All Tkinter GUI features are present
- ✅ Tab design matches file folder metaphor
- ✅ File list supports multi-select
- ✅ Form controls work correctly
- ✅ Console auto-scrolls and shows colored logs
- ✅ Progress bar animates
- ✅ Mock testing demonstrates all UI interactions
- ✅ No console errors
- ✅ Professional appearance
- ✅ Keyboard navigation works
- ✅ Focus indicators visible

**All criteria met = Ready for Phase 4 (API Integration)**

---

## Next Steps After Testing

Once testing is complete:

1. **Document any issues** in GitHub issues or notes
2. **Fix critical bugs** (if any)
3. **Proceed to Phase 4** - Backend API integration
4. **Test again** with real file dialogs and subprocess execution

---

## Contact

For questions or issues:
- Check `PHASE3_FRONTEND_GUIDE.md` for detailed documentation
- Review `whisperjav/gui/whisperjav_gui.py` for Tkinter reference
- Test with `dev_server.py` for rapid iteration
