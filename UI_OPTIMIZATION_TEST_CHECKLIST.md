# WhisperJAV GUI - UI Optimization Test Checklist

**Date:** 2025-10-30
**Tester:** _______________
**Test Environment:** _______________
**Screen Resolution:** _______________

---

## Pre-Test Setup

### Prerequisites
- [ ] GUI can be launched: `python -m whisperjav.webview_gui.main`
- [ ] Browser DevTools available (F12 key)
- [ ] Sample video files ready for testing
- [ ] Backup CSS verified: `style.backup.css` exists
- [ ] Git status clean (or changes committed)

### Test Environment Details
- **OS:** Windows _____ (10/11)
- **Screen Size:** _____ inches
- **Resolution:** _____ x _____
- **Scaling:** _____ % (100/125/150/200)
- **Window Size:** _____ x _____ (should default to 1000x700)

---

## Section 1: Visual Verification Tests

### 1.1 Overall Layout (CRITICAL)
- [ ] **No vertical scrollbar** appears at default 700px window height
- [ ] All sections visible without scrolling
- [ ] Console section takes up significant space (largest visible section)
- [ ] Footer visible at bottom (not cut off)
- [ ] Header visible at top
- [ ] No overlapping elements
- [ ] No horizontal scrollbar (width = 1000px should be sufficient)

**If FAILED:** Check window height, verify CSS loaded correctly

---

### 1.2 Header Section (~45px)
- [ ] "WhisperJAV" title clearly visible
- [ ] "Simple Runner" subtitle readable (small but legible)
- [ ] Adequate padding around text (not cramped)
- [ ] Background color: Blue (#024873)
- [ ] Text color: White

**Measured Height:** _____ px (DevTools)
**Expected:** ~40-50px
**Pass/Fail:** _____

---

### 1.3 Source Section (~135px)
- [ ] "Source" header clearly visible
- [ ] File list container displays correctly
- [ ] Empty state shows when no files added:
  - [ ] Folder icon visible
  - [ ] "No files selected" text readable
  - [ ] Hint text readable
- [ ] File list shows 2-3 files comfortably when populated
- [ ] Scrollbar appears when more than 3 files added
- [ ] All 4 buttons visible and properly sized:
  - [ ] "Add File(s)" button
  - [ ] "Add Folder" button
  - [ ] "Remove Selected" button
  - [ ] "Clear" button
- [ ] Button text fully readable (not truncated)

**Measured Height:** _____ px (DevTools)
**Expected:** ~130-140px
**Pass/Fail:** _____

---

### 1.4 Destination Section (~60px)
- [ ] "Destination" header visible
- [ ] "Output:" label visible
- [ ] Output path field displays correctly (truncates long paths with ellipsis)
- [ ] "Browse" button visible and sized appropriately
- [ ] "Open" button visible and sized appropriately
- [ ] All elements on single row (no wrapping on 1000px width)

**Measured Height:** _____ px (DevTools)
**Expected:** ~55-65px
**Pass/Fail:** _____

---

### 1.5 Tabs Section (~235px)

#### Tab Bar
- [ ] Both tab buttons visible:
  - [ ] "Transcription Mode"
  - [ ] "Transcription Adv. Options"
- [ ] Active tab appears "lifted" (file folder effect)
- [ ] Active tab has darker text color
- [ ] Inactive tab has lighter background
- [ ] Tab border styling correct (top/left/right borders)

#### Tab Content - Transcription Mode
- [ ] All 3 radio buttons visible (balanced, fast, faster)
- [ ] Sensitivity dropdown visible
- [ ] Output language dropdown visible
- [ ] Both info boxes visible at bottom:
  - [ ] "Speed vs. Accuracy" info
  - [ ] "Details vs. Noise" info
- [ ] Info text readable (small but legible)
- [ ] All form controls properly spaced (not overlapping)

#### Tab Content - Advanced Options
- [ ] All checkboxes visible:
  - [ ] Adaptive classification (disabled)
  - [ ] Adaptive audio enhancements (disabled)
  - [ ] Smart postprocessing (disabled)
  - [ ] Model override checkbox
  - [ ] Async processing checkbox
  - [ ] Keep temp files checkbox
- [ ] All dropdowns/inputs visible:
  - [ ] Verbosity dropdown
  - [ ] Model selection dropdown (disabled by default)
  - [ ] Max workers number input
  - [ ] Opening credit text input
  - [ ] Temp dir input with Browse button
- [ ] Form controls properly aligned

**Measured Height:** _____ px (DevTools)
**Expected:** ~230-240px
**Pass/Fail:** _____

---

### 1.6 Run Controls Section (~60px)
- [ ] Progress bar visible and properly sized
- [ ] Progress bar height: ~6px (thin but visible)
- [ ] Status label visible ("Idle" initially)
- [ ] "Start" button visible and properly sized
- [ ] "Cancel" button visible (disabled initially)
- [ ] Button text fully readable

**Measured Height:** _____ px (DevTools)
**Expected:** ~55-65px
**Pass/Fail:** _____

---

### 1.7 Console Section (~300-400px) - MOST IMPORTANT
- [ ] "Console" header with "Clear" button visible
- [ ] Console output area clearly visible
- [ ] Console has dark background (#1E1E1E)
- [ ] Console text color: Light gray (#D4D4D4)
- [ ] "Ready." initial message visible
- [ ] Console takes up SIGNIFICANT vertical space
- [ ] Console is LARGEST section on screen
- [ ] Scrollbar appears in console when output exceeds height
- [ ] Monospace font used (Consolas/Monaco)

**Measured Height:** _____ px (DevTools)
**Expected:** ~250-400px (should be largest section)
**Pass/Fail:** _____

**Visual Check:** Is console obviously larger than tabs section?
- [ ] Yes (PASS)
- [ ] No (FAIL - check CSS flex-grow)

---

### 1.8 Footer Section (~25px)
- [ ] Copyright text visible
- [ ] "About" link visible
- [ ] Text readable (small but legible)
- [ ] Centered alignment

**Measured Height:** _____ px (DevTools)
**Expected:** ~20-30px
**Pass/Fail:** _____

---

## Section 2: Readability Tests

### 2.1 Font Sizes
Test with user sitting at normal distance from screen (~24 inches for laptop).

- [ ] All text is readable without squinting
- [ ] Header title (18px) clearly readable
- [ ] Section headers (15px) clearly readable
- [ ] Body text (13px) clearly readable
- [ ] Form labels (12px) clearly readable
- [ ] Small text (11px) still legible (info boxes, footer)
- [ ] Console output (12px) readable
- [ ] No text appears blurry or pixelated

**Readability Rating:** _____/10
**Pass/Fail:** _____ (Pass if ≥7/10)

---

### 2.2 Spacing & Clarity
- [ ] Visual hierarchy clear (important elements stand out)
- [ ] Sections clearly separated (borders/shadows visible)
- [ ] No cramped feeling (whitespace adequate)
- [ ] No elements touching/overlapping
- [ ] Tab separation clear (active vs inactive obvious)
- [ ] Button spacing adequate (easy to click correct button)

---

## Section 3: Functional Tests

### 3.1 File Management
- [ ] "Add File(s)" button opens file dialog
- [ ] Can select single file successfully
- [ ] Can select multiple files successfully
- [ ] Files appear in file list with correct names
- [ ] File paths use monospace font
- [ ] Long file paths truncate with ellipsis
- [ ] "Add Folder" button opens folder dialog
- [ ] Can select folder successfully
- [ ] All video files in folder appear in list
- [ ] File items can be clicked to select
- [ ] Selected files highlight (blue background, white text)
- [ ] Multiple files can be selected (Ctrl+click)
- [ ] "Remove Selected" button removes selected files
- [ ] "Remove Selected" disabled when no selection
- [ ] "Clear" button removes all files
- [ ] "Clear" disabled when list empty
- [ ] File list scrolls when more than 3 files added

**File List Capacity Test:**
Add 10 files. Can you:
- [ ] See 2-3 files without scrolling
- [ ] Scroll to see remaining files
- [ ] Select any file easily

---

### 3.2 Destination Controls
- [ ] "Browse" button opens folder dialog
- [ ] Selected folder path appears in output field
- [ ] Long paths truncate correctly
- [ ] "Open" button opens folder in file explorer
- [ ] "Open" button disabled when no path set

---

### 3.3 Tab Navigation
- [ ] Click on "Transcription Mode" tab activates it
- [ ] Click on "Transcription Adv. Options" tab activates it
- [ ] Tab switches smoothly (no flicker)
- [ ] Active tab appears "lifted" (file folder effect preserved)
- [ ] Tab content changes correctly
- [ ] Previous tab content hidden when switching
- [ ] Active tab font weight bolder than inactive
- [ ] Active tab slightly larger than inactive

**Tab Animation Test:**
- [ ] Tab switch has smooth transition (~0.2s)
- [ ] No janky/jumpy animation
- [ ] Background colors transition smoothly

---

### 3.4 Form Controls - Transcription Mode Tab

#### Mode Selection
- [ ] Can select "balanced" radio button
- [ ] Can select "fast" radio button
- [ ] Can select "faster" radio button
- [ ] Only one can be selected at a time
- [ ] Default selection: "balanced"

#### Sensitivity Dropdown
- [ ] Dropdown opens on click
- [ ] Shows 3 options: conservative, balanced, aggressive
- [ ] Can select each option
- [ ] Default selection: "balanced"
- [ ] Selected value displays correctly

#### Output Language Dropdown
- [ ] Dropdown opens on click
- [ ] Shows 2 options: japanese, english-direct
- [ ] Can select each option
- [ ] Default selection: "japanese"
- [ ] Selected value displays correctly

---

### 3.5 Form Controls - Advanced Options Tab

#### Checkboxes
- [ ] "Adaptive classification" checkbox visible (disabled)
- [ ] "Adaptive audio enhancements" checkbox visible (disabled)
- [ ] "Smart postprocessing" checkbox visible (disabled)
- [ ] "Model override" checkbox toggleable
- [ ] "Async processing" checkbox toggleable
- [ ] "Keep temp files" checkbox toggleable
- [ ] Checkboxes have visible check mark when selected
- [ ] Disabled checkboxes appear grayed out

#### Model Override
- [ ] Model dropdown disabled by default
- [ ] Model dropdown enables when "Model override" checked
- [ ] Model dropdown shows: large-v3, large-v2, turbo
- [ ] Can select each model option
- [ ] Default: large-v3

#### Async Processing
- [ ] Max workers input accepts numbers
- [ ] Default value: 1
- [ ] Min value: 1
- [ ] Max value: 16
- [ ] Can type in values
- [ ] Can use up/down arrows

#### Other Inputs
- [ ] Opening credit text input accepts text
- [ ] Temp dir input accepts text
- [ ] Temp dir "Browse" button opens folder dialog
- [ ] Selected temp dir path appears in input

---

### 3.6 Run Controls

#### Progress Bar
- [ ] Progress bar visible
- [ ] Progress bar width spans available space
- [ ] Progress bar height: ~6px (thin but visible)
- [ ] Progress fill color: Blue (#024873)
- [ ] Status label updates (test during actual run)

#### Buttons
- [ ] "Start" button enabled when files added
- [ ] "Start" button disabled when no files
- [ ] "Cancel" button disabled initially
- [ ] Buttons have hover effect (darker background)
- [ ] Buttons have focus effect (blue outline when tabbed to)

---

### 3.7 Console

#### Basic Display
- [ ] Console shows "Ready." on startup
- [ ] Console has dark background (#1E1E1E)
- [ ] Console text is light gray (#D4D4D4)
- [ ] "Clear" button visible in header
- [ ] "Clear" button removes all console output
- [ ] Console output uses monospace font

#### During Processing (Run a test file)
- [ ] Console output appears in real-time
- [ ] Color coding works:
  - [ ] Info messages: Light blue (#4FC3F7)
  - [ ] Success messages: Green (#66BB6A)
  - [ ] Warning messages: Orange (#FFA726)
  - [ ] Error messages: Red (#EF5350)
  - [ ] Commands: Gold (#FFD700)
- [ ] Console auto-scrolls to bottom as new messages arrive
- [ ] Can manually scroll up to see previous messages
- [ ] Scrollbar appears when content exceeds height
- [ ] Long lines wrap correctly (no horizontal scroll)

---

### 3.8 About Dialog
- [ ] Click "About" link in footer opens modal
- [ ] Modal appears centered on screen
- [ ] Modal has semi-transparent dark overlay
- [ ] Modal content readable:
  - [ ] WhisperJAV logo/icon
  - [ ] Version number
  - [ ] Description text
  - [ ] Key Features list
  - [ ] Technology section
  - [ ] Keyboard shortcuts
  - [ ] GitHub link
- [ ] "Close" button closes modal
- [ ] "X" button in header closes modal
- [ ] Click outside modal closes it
- [ ] ESC key closes modal

---

## Section 4: Interaction Tests

### 4.1 Hover States
Test mouse hover on each element:

- [ ] Buttons change color on hover
- [ ] File items highlight on hover
- [ ] Tab buttons highlight on hover (except active)
- [ ] Links underline on hover
- [ ] Hover transitions smooth (~0.15s)

---

### 4.2 Focus States (Keyboard Navigation)
Use TAB key to navigate:

- [ ] Can tab through all interactive elements
- [ ] Focused elements show blue outline
- [ ] Tab order is logical (top to bottom, left to right)
- [ ] Can activate buttons with ENTER/SPACE
- [ ] Can toggle checkboxes with SPACE
- [ ] Can open dropdowns with ENTER
- [ ] Can navigate file list with arrow keys
- [ ] Focus indicator clearly visible

---

### 4.3 Keyboard Shortcuts
- [ ] **Ctrl+O** - Opens "Add Files" dialog
- [ ] **Ctrl+R** - Starts processing (when files added)
- [ ] **Escape** - Cancels processing (during run)
- [ ] **F1** - Opens About dialog

---

## Section 5: Responsive Tests

### 5.1 Window Resize

#### Increase Height (700px → 800px)
- [ ] Console expands to use additional space
- [ ] Other sections stay same height
- [ ] No visual glitches

**Console Height at 800px:** _____ px
**Expected:** ~400px (100px more than at 700px)
**Pass/Fail:** _____

#### Decrease Height (700px → 600px)
- [ ] Console shrinks but maintains minimum (~250px)
- [ ] Other sections stay same height
- [ ] May require slight vertical scroll
- [ ] No visual glitches

**Console Height at 600px:** _____ px
**Expected:** ~200-250px (respects minimum)
**Pass/Fail:** _____

#### Increase Width (1000px → 1200px)
- [ ] Elements scale appropriately
- [ ] No weird gaps or stretching
- [ ] Console width increases
- [ ] Forms use additional width

#### Decrease Width (1000px → 800px, minimum)
- [ ] Layout adapts gracefully
- [ ] No horizontal scrollbar
- [ ] Text doesn't overflow
- [ ] Buttons may wrap but remain accessible

---

### 5.2 High DPI Displays
If testing on high DPI display (150%, 200% scaling):

- [ ] UI scales correctly with OS scaling
- [ ] Text remains crisp (no blur)
- [ ] Icons/borders not pixelated
- [ ] Proportions maintained
- [ ] No layout breakage

**OS Scaling:** _____ %
**Pass/Fail:** _____

---

## Section 6: Performance Tests

### 6.1 Initial Load
- [ ] GUI loads in < 2 seconds
- [ ] No visible flicker during load
- [ ] CSS applied immediately (no FOUC - Flash of Unstyled Content)
- [ ] All elements render correctly

**Load Time:** _____ seconds

---

### 6.2 During Operation
Add 20+ files and process them:

- [ ] Console updates smoothly (no lag)
- [ ] Progress bar animates smoothly
- [ ] UI remains responsive (can still click buttons)
- [ ] No freezing or hanging
- [ ] Console scrolling smooth

---

## Section 7: Cross-Browser Tests (Windows Only)

### 7.1 Edge WebView2 (Primary)
- [ ] All visual tests pass
- [ ] All functional tests pass
- [ ] All interaction tests pass
- [ ] Performance acceptable

---

### 7.2 Chrome (Fallback, if available)
- [ ] All visual tests pass
- [ ] All functional tests pass
- [ ] All interaction tests pass
- [ ] Performance acceptable

---

## Section 8: Regression Tests

### 8.1 Features Not Changed
Verify these work exactly as before:

- [ ] File selection (single/multiple)
- [ ] Folder selection
- [ ] Output directory selection
- [ ] Mode selection (balanced/fast/faster)
- [ ] All advanced options
- [ ] Processing execution
- [ ] Progress reporting
- [ ] Console logging
- [ ] Error handling
- [ ] Keyboard shortcuts

---

### 8.2 Visual Elements Preserved
- [ ] Color scheme identical to before
- [ ] Tab "lift" effect still works
- [ ] Active tab still prominent
- [ ] Shadows and borders intact
- [ ] Hover effects same as before
- [ ] Focus effects same as before
- [ ] Professional appearance maintained

---

## Section 9: Accessibility Tests

### 9.1 Screen Reader (Optional)
If using screen reader (NVDA, JAWS):

- [ ] All labels read correctly
- [ ] Buttons announced properly
- [ ] Form controls accessible
- [ ] Console output readable

---

### 9.2 Keyboard-Only Navigation
Navigate entire UI using only keyboard:

- [ ] Can access all features
- [ ] Tab order logical
- [ ] No keyboard traps
- [ ] All actions possible (no mouse required)

---

### 9.3 High Contrast Mode (Windows)
Enable Windows High Contrast mode:

- [ ] UI remains usable
- [ ] Text readable
- [ ] Borders visible
- [ ] Focus indicators clear

---

## Section 10: Edge Cases

### 10.1 Long Text Handling
- [ ] Long file paths truncate with ellipsis
- [ ] Long output paths truncate correctly
- [ ] Long console messages wrap (no horizontal scroll)
- [ ] Long opening credit text fits in input

---

### 10.2 Many Files
Add 50+ files:

- [ ] File list scrolls correctly
- [ ] Performance acceptable
- [ ] Selection works
- [ ] Clear button works

---

### 10.3 Rapid Tab Switching
Switch tabs rapidly 10+ times:

- [ ] No visual glitches
- [ ] No lag
- [ ] Content switches correctly
- [ ] No console errors

---

## Section 11: Measurement Verification

### 11.1 Section Heights (Use DevTools)

Measure actual heights and compare to expected:

| Section | Expected | Actual | Pass/Fail |
|---------|----------|--------|-----------|
| Header | ~45px | ___px | ___ |
| Source | ~135px | ___px | ___ |
| Destination | ~60px | ___px | ___ |
| Tabs | ~235px | ___px | ___ |
| Run Controls | ~60px | ___px | ___ |
| Console | ~300px+ | ___px | ___ |
| Footer | ~25px | ___px | ___ |
| **Total** | **~700px** | **___px** | ___ |

**Tolerance:** ±10px acceptable

---

### 11.2 Font Sizes (Use DevTools)

Verify CSS variables applied:

| Element | Expected | Actual | Pass/Fail |
|---------|----------|--------|-----------|
| --font-size-base | 13px | ___px | ___ |
| --font-size-sm | 11px | ___px | ___ |
| --font-size-lg | 15px | ___px | ___ |
| Header H1 | 18px | ___px | ___ |
| Section headers | 15px | ___px | ___ |

---

## Section 12: User Experience Assessment

### 12.1 Subjective Ratings (1-10 scale)

Rate the following aspects:

- **Readability:** ___/10 (Can you read all text comfortably?)
- **Visual Clarity:** ___/10 (Are elements clearly distinguishable?)
- **Spacing:** ___/10 (Does spacing feel comfortable, not cramped?)
- **Professional Appearance:** ___/10 (Does it look polished?)
- **Ease of Use:** ___/10 (Are controls easy to find and use?)
- **Console Visibility:** ___/10 (Is console prominent enough?)
- **Overall Experience:** ___/10 (General impression?)

**Average Rating:** ___/10
**Acceptable if:** ≥7/10 average

---

### 12.2 Comparison to Original
If you have access to original version (style.backup.css):

**Which version do you prefer overall?**
- [ ] Original (more spacious)
- [ ] Optimized (more compact)
- [ ] No strong preference

**Advantages of optimized version:**
- ___________________________________
- ___________________________________

**Disadvantages of optimized version:**
- ___________________________________
- ___________________________________

---

## Section 13: Issues & Notes

### Critical Issues (Blocking)
List any issues that prevent basic functionality:

1. ___________________________________
2. ___________________________________

---

### Minor Issues (Non-blocking)
List any cosmetic or minor usability issues:

1. ___________________________________
2. ___________________________________

---

### Observations & Suggestions
Any other notes or recommendations:

1. ___________________________________
2. ___________________________________

---

## Final Verdict

### Test Summary
- **Total Tests:** ~150+
- **Tests Passed:** ___
- **Tests Failed:** ___
- **Pass Rate:** ____%

### Overall Result
- [ ] **PASS** - Ready for deployment (≥95% pass rate, no critical issues)
- [ ] **PASS WITH MINOR ISSUES** - Deploy with noted issues (90-95% pass rate)
- [ ] **CONDITIONAL PASS** - Fix specific issues before deploy (80-90% pass rate)
- [ ] **FAIL** - Significant issues, needs rework (<80% pass rate)

### Recommendation
```
□ APPROVE for immediate deployment
□ APPROVE with minor fixes
□ REVISE and retest
□ ROLLBACK to original version
```

---

### Tester Sign-off

**Tester Name:** _______________
**Date Tested:** _______________
**Test Duration:** _____ hours
**Environment:** _______________

**Signature:** _______________

---

## Appendix: How to Rollback

If testing reveals critical issues:

1. **Stop the GUI** (close window)
2. **Restore backup:**
   ```bash
   cd C:\BIN\git\WhisperJav_V1_Minami_Edition\whisperjav\webview_gui\assets
   copy style.backup.css style.css
   ```
3. **Restart GUI:**
   ```bash
   python -m whisperjav.webview_gui.main
   ```
4. **Verify original version restored**
5. **Report issues to developer**

---

## Quick Pass/Fail Summary

Use this for rapid smoke testing:

- [ ] ✅ No vertical scrollbar at 700px height
- [ ] ✅ All sections visible
- [ ] ✅ Console is largest section
- [ ] ✅ All text readable
- [ ] ✅ All buttons clickable
- [ ] ✅ Tabs switch correctly
- [ ] ✅ File management works
- [ ] ✅ Form controls work
- [ ] ✅ Professional appearance maintained
- [ ] ✅ No visual glitches

**If all checked: APPROVE for deployment ✅**

