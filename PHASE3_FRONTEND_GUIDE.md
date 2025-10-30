# Phase 3: Frontend Build - Complete Guide

## Overview

Phase 3 delivers a complete, professional HTML/CSS/JS frontend that replicates all features of the Tkinter GUI with modern web technologies. The UI is optimized for non-technical users with a clean, intuitive design.

**Status:** ✅ Complete
**Files Created:**
- `whisperjav/webview_gui/assets/index.html` - Semantic HTML structure
- `whisperjav/webview_gui/assets/style.css` - Professional styling
- `whisperjav/webview_gui/assets/app.js` - UI interactions (no API yet)
- `whisperjav/webview_gui/dev_server.py` - Hot reload development server

---

## UI Structure

### Complete Feature Parity with Tkinter

The web GUI implements **100% feature parity** with the Tkinter GUI (`whisperjav/gui/whisperjav_gui.py`):

```
WhisperJAV GUI
├── Header
│   ├── Title: "WhisperJAV"
│   └── Subtitle: "Simple Runner"
│
├── Source Section
│   ├── File list (multi-select with Shift/Ctrl)
│   ├── Empty state: "No files selected"
│   └── Buttons: Add File(s), Add Folder, Remove Selected, Clear
│
├── Destination Section
│   ├── Output directory (read-only input)
│   └── Buttons: Browse, Open
│
├── Tabbed Options (File folder design)
│   ├── Tab 1: Transcription Mode
│   │   ├── Mode: Radio buttons (balanced/fast/faster)
│   │   ├── Sensitivity: Dropdown (conservative/balanced/aggressive)
│   │   ├── Output language: Dropdown (japanese/english-direct)
│   │   └── Info labels explaining options
│   │
│   └── Tab 2: Transcription Adv. Options
│       ├── Row 1: Adaptive features (WIP, disabled) + Verbosity
│       ├── Row 2: Model override + Async processing
│       ├── Row 3: Opening credit text
│       └── Row 4: Keep temp files + Temp directory
│
├── Run Controls
│   ├── Progress bar (indeterminate animation)
│   ├── Status label (Idle/Running/Done)
│   └── Buttons: Start, Cancel
│
├── Console
│   ├── Log output (dark theme, monospace)
│   ├── Auto-scroll to bottom
│   ├── Color coding (info/success/warning/error)
│   └── Clear button
│
└── Footer
    └── App description
```

---

## Design Philosophy

### 1. File Folder Tab Metaphor

The tab design follows the Tkinter implementation's "physical folder" metaphor:

**Active Tab:**
- Background: `#F0F2F5` (matches content area)
- Text: `#024873` (deep blue, bold, 15px)
- Border: 2px on top/sides, seamless bottom connection
- Visual: "Lifts" to connect with content area

**Inactive Tabs:**
- Background: `#F7F8FA` (lighter than active)
- Text: `#4B5563` (gray, 14px)
- Visual: Sits in background, distinct from active

**Implementation:**
```css
/* Active tab breaks through tab bar border to connect with content */
.tab-button.active {
    background: var(--active-tab-bg);      /* #F0F2F5 */
    color: var(--active-tab-text);         /* #024873 */
    font-weight: 600;
    border-bottom: 2px solid var(--active-tab-bg);  /* Seamless connection */
    z-index: 10;
}

.tab-content-container {
    background: var(--active-tab-bg);      /* Same as active tab */
}
```

### 2. Color Palette

```css
/* Tab System */
--tab-bar-bg: #FFFFFF          /* Tab bar background */
--content-area-bg: #F0F2F5     /* Content area & active tab */
--active-tab-bg: #F0F2F5       /* Active tab (matches content) */
--inactive-tab-bg: #F7F8FA     /* Inactive tabs (lighter) */
--hover-tab-bg: #F2F4FA        /* Hover state (between inactive/active) */

/* Text */
--active-tab-text: #024873     /* Active tab text (deep blue, bold) */
--inactive-tab-text: #4B5563   /* Inactive tab text (gray) */
--text-primary: #212529        /* Body text */
--text-secondary: #6C757D      /* Secondary text */
--text-muted: #ADB5BD          /* Muted text */

/* UI Elements */
--primary-color: #024873       /* Primary buttons, focus states */
--secondary-color: #6C757D     /* Secondary buttons */
--border-color: #D1D5DB        /* Borders */
--disabled-bg: #E9ECEF         /* Disabled inputs */
```

### 3. Typography

```css
/* System Fonts for Native Feel */
--font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', ...
--font-mono: 'Consolas', 'Monaco', 'Courier New', monospace

/* Sizes */
--font-size-base: 14px         /* Body text */
--font-size-sm: 12px           /* Labels, hints */
--font-size-lg: 16px           /* Section headers */
```

### 4. Spacing & Layout

- **Window size:** 1000x700 (optimized)
- **Section padding:** 16px
- **Element gap:** 12px (sections), 8px (buttons), 6px (form groups)
- **Border radius:** 6px (standard), 8px (large sections)

---

## Component Details

### 1. File List Component

**Features:**
- Multi-select support (Shift+Click, Ctrl+Click)
- Keyboard navigation (Arrow keys, Delete)
- File/folder icons (📄 📁)
- Empty state with helpful message
- Auto-scroll to selection

**HTML Structure:**
```html
<div class="file-list" id="fileList">
    <!-- Empty state -->
    <div class="empty-state" id="emptyState">
        <span class="empty-icon">📂</span>
        <p>No files selected</p>
        <p class="empty-hint">Click "Add File(s)" or "Add Folder" to get started</p>
    </div>

    <!-- File items (dynamically added) -->
    <div class="file-item" data-path="..." data-index="0">
        <span class="file-icon">📄</span>
        <span class="file-path">C:\Videos\sample.mp4</span>
    </div>
</div>
```

**JavaScript:**
```javascript
// Selection handling
handleItemClick(item, event) {
    if (event.ctrlKey) {
        this.toggleSelection(index);      // Ctrl: Toggle
    } else if (event.shiftKey) {
        this.selectRange(last, index);     // Shift: Range
    } else {
        this.selectSingle(index);          // Normal: Single
    }
}
```

### 2. Tab Component

**Keyboard Navigation:**
- Arrow Left/Right: Switch tabs
- Home/End: First/last tab
- Tab key: Focus next element

**ARIA Attributes:**
```html
<button
    class="tab-button active"
    role="tab"
    aria-selected="true"
    aria-controls="tab1-panel"
    data-tab="tab1"
>
    Transcription Mode
</button>
```

**JavaScript:**
```javascript
switchTab(tabId) {
    // Update button states
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
        btn.setAttribute('aria-selected', btn.dataset.tab === tabId);
    });

    // Show/hide panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === `${tabId}-panel`);
    });
}
```

### 3. Form Controls

**Conditional Enable/Disable:**
```javascript
// Model dropdown enabled only when override checkbox checked
modelOverrideCheckbox.addEventListener('change', () => {
    modelSelect.disabled = !modelOverrideCheckbox.checked;
});

// Start button disabled if no files selected
FormManager.validateForm() {
    const hasFiles = AppState.selectedFiles.length > 0;
    startBtn.disabled = !hasFiles || AppState.isRunning;
    return hasFiles;
}
```

**Default Values:**
- Mode: `balanced`
- Sensitivity: `balanced`
- Language: `japanese`
- Verbosity: `summary`
- Model: `large-v3`
- Max workers: `1`

### 4. Console Component

**Features:**
- Auto-scroll to bottom on new logs
- Color coding by log type
- Dark theme (VS Code-inspired)
- Monospace font for technical output

**CSS:**
```css
.console-output {
    background: #1E1E1E;
    color: #D4D4D4;
    font-family: var(--font-mono);
    white-space: pre-wrap;
    word-wrap: break-word;
}

.console-line.info { color: #4FC3F7; }
.console-line.success { color: #66BB6A; }
.console-line.warning { color: #FFA726; }
.console-line.error { color: #EF5350; }
```

**JavaScript:**
```javascript
log(message, type = 'info') {
    const line = document.createElement('div');
    line.className = `console-line ${type}`;
    line.textContent = message;
    output.appendChild(line);

    // Auto-scroll
    output.scrollTop = output.scrollHeight;
}
```

### 5. Progress Component

**Indeterminate Animation:**
```css
@keyframes indeterminate-progress {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(400%); }
}

.progress-bar.indeterminate .progress-fill {
    width: 30%;
    animation: indeterminate-progress 1.5s ease-in-out infinite;
}
```

**JavaScript:**
```javascript
setIndeterminate(active) {
    this.progressBar.classList.toggle('indeterminate', active);
}

setProgress(percent) {
    this.progressBar.classList.remove('indeterminate');
    this.progressFill.style.width = `${percent}%`;
}
```

---

## State Management

### AppState Object

```javascript
const AppState = {
    // File list
    selectedFiles: [],           // Array of file paths
    selectedIndices: new Set(),  // Set of selected indices

    // UI state
    activeTab: 'tab1',          // Current tab ID
    isRunning: false,           // Process running state

    // Form values
    outputDir: '',              // Output directory path

    init() {
        this.loadDefaultOutputDir();
    }
};
```

### Component Managers

```javascript
TabManager.init()          // Tab switching, keyboard nav
FileListManager.init()     // File list, selection, rendering
FormManager.init()         // Form validation, data collection
ConsoleManager.init()      // Log display, auto-scroll
ProgressManager.init()     // Progress bar, status label
RunControls.init()         // Start/cancel buttons
```

---

## Testing & Development

### Development Server (Hot Reload)

```bash
# Launch dev server with hot reload
python -m whisperjav.webview_gui.dev_server

# Edit HTML/CSS/JS → Save → Refresh window to see changes
```

**Features:**
- Debug mode enabled (shows developer console)
- 1000x700 window size (same as production)
- No API backend needed for frontend work
- F12/Cmd+Opt+I: Open developer tools

### Preview in Browser

```bash
# Open in browser for CSS testing
# File: C:\BIN\git\WhisperJav_V1_Minami_Edition\whisperjav\webview_gui\assets\index.html

# Note: JavaScript functionality limited without PyWebView bridge
```

### Testing Checklist

**Visual:**
- [ ] All sections present and properly styled
- [ ] Tab switching works smoothly
- [ ] File folder tab design matches Tkinter
- [ ] Active tab clearly distinguished from inactive
- [ ] Colors match palette (active: #F0F2F5, inactive: #F7F8FA)
- [ ] Typography is readable and hierarchical
- [ ] Buttons have hover states
- [ ] Focus indicators visible (keyboard navigation)

**Functionality:**
- [ ] Tab switching (mouse & keyboard)
- [ ] File list selection (single, Ctrl+Click, Shift+Click)
- [ ] File list keyboard navigation (arrows, delete)
- [ ] Model dropdown enables when override checked
- [ ] Start button disabled when no files
- [ ] Console auto-scrolls on new logs
- [ ] Progress bar indeterminate animation works
- [ ] Clear console button works
- [ ] Mock file add/remove works

**Accessibility:**
- [ ] Tab navigation with keyboard (Tab, Shift+Tab)
- [ ] ARIA attributes present
- [ ] Focus visible indicators
- [ ] Disabled controls properly marked
- [ ] Tooltips on WIP features

---

## Mock Testing (Phase 3)

Since API integration is Phase 4, the UI includes mock functionality for testing:

### Mock File Operations

```javascript
// Add Files button
FileListManager.addFilesMock() {
    const mockFiles = [
        'C:\\Videos\\sample1.mp4',
        'C:\\Videos\\sample2.mkv',
        'C:\\Videos\\sample3.avi'
    ];
    AppState.selectedFiles.push(...mockFiles);
    this.render();
}

// Add Folder button
FileListManager.addFolderMock() {
    const mockFolder = 'C:\\Videos\\JAV_Collection';
    AppState.selectedFiles.push(mockFolder);
    this.render();
}
```

### Mock Process Run

```javascript
// Start button
RunControls.start() {
    ProgressManager.setIndeterminate(true);
    ProgressManager.setStatus('Running...');

    // Simulate 3-second run
    setTimeout(() => {
        ProgressManager.setProgress(100);
        ProgressManager.setStatus('Done');
        ConsoleManager.log('Process completed successfully', 'success');
    }, 3000);
}
```

**Test Workflow:**
1. Click "Add File(s)" → 3 mock files added
2. Click "Add Folder" → 1 mock folder added
3. Select files (Shift+Click, Ctrl+Click)
4. Click "Remove Selected" → Files removed
5. Click "Start" → Mock 3-second run
6. Click "Cancel" → Process stops

---

## Phase 4 Integration Points

Phase 3 is **frontend only**. Phase 4 will wire up the backend API.

### API Calls to Implement (Phase 4)

```javascript
// File dialogs
FileListManager.addFiles() {
    const response = await pywebview.api.select_files();
    if (response.success) {
        AppState.selectedFiles.push(...response.paths);
        this.render();
    }
}

FileListManager.addFolder() {
    const response = await pywebview.api.select_folder();
    if (response.success) {
        AppState.selectedFiles.push(response.path);
        this.render();
    }
}

// Process management
RunControls.start() {
    const options = FormManager.getFormData();
    const response = await pywebview.api.start_process(options);
    if (response.success) {
        this.startLogPolling();
    }
}

// Log polling
startLogPolling() {
    this.logPollInterval = setInterval(async () => {
        const logs = await pywebview.api.get_logs();
        logs.forEach(line => ConsoleManager.appendRaw(line));

        const status = await pywebview.api.get_process_status();
        if (status.status !== 'running') {
            clearInterval(this.logPollInterval);
            this.handleComplete(status);
        }
    }, 100);
}
```

### Data Flow (Phase 4)

```
User Action → JavaScript Event → API Call → Python Backend → Subprocess → Response
                                                                              ↓
User sees result ← UI Update ← JavaScript ←──────────────────────────────────┘
```

---

## File Structure

```
whisperjav/webview_gui/
├── main.py                 # Production launcher (Phase 1-2)
├── api.py                  # Backend API (Phase 2)
├── dev_server.py           # Development server (Phase 3)
└── assets/
    ├── index.html          # HTML structure (Phase 3)
    ├── style.css           # Professional styling (Phase 3)
    └── app.js              # UI interactions (Phase 3, API in Phase 4)
```

---

## CSS Architecture

### CSS Variables

All colors, fonts, and spacing defined in `:root` for easy customization:

```css
:root {
    /* Colors */
    --primary-color: #024873;
    --border-radius: 6px;

    /* Typography */
    --font-family: ...;
    --font-size-base: 14px;

    /* Shadows */
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.12);
}
```

### Component Classes

```css
/* Layout */
.app-container, .app-header, .app-main, .app-footer

/* Sections */
.section, .section-header, .section-content

/* File list */
.file-list, .file-item, .empty-state

/* Tabs */
.tabs-container, .tab-bar, .tab-button, .tab-panel

/* Forms */
.form-row, .form-group, .form-label, .form-input, .form-select

/* Buttons */
.btn, .btn-primary, .btn-secondary, .btn-text, .btn-sm, .btn-lg

/* Console */
.console-output, .console-line
```

### Modular Structure

```
style.css
├── CSS Variables (lines 1-60)
├── Reset & Base (lines 61-80)
├── Layout (lines 81-140)
├── Sections (lines 141-180)
├── Source Section (lines 181-250)
├── Destination Section (lines 251-280)
├── Tabs Section (lines 281-360)
├── Form Elements (lines 361-480)
├── Buttons (lines 481-540)
├── Run Section (lines 541-600)
├── Console Section (lines 601-650)
├── Scrollbar (lines 651-680)
├── Responsive (lines 681-700)
└── Accessibility (lines 701-730)
```

---

## Browser Compatibility

### Target Browsers

- **Windows:** Edge (Chromium), Chrome
- **macOS:** Safari (WebKit), Chrome

### PyWebView Rendering Engines

- **Windows:** Edge WebView2 (Chromium)
- **macOS:** WKWebView (WebKit)
- **Linux:** GTK WebKit2

### CSS Features Used

- CSS Variables (supported in all modern browsers)
- Flexbox (full support)
- Grid (full support)
- CSS Animations (full support)
- `:focus-visible` (modern browsers, graceful degradation)

---

## Performance Considerations

### Optimizations

1. **No external dependencies** - All CSS/JS inline or local
2. **Minimal DOM updates** - Only re-render changed items
3. **Efficient selectors** - Class-based, not complex queries
4. **Auto-scroll throttling** - Only on new logs
5. **Event delegation** - Single listener for file list

### File Size

- `index.html`: ~10 KB
- `style.css`: ~15 KB
- `app.js`: ~12 KB
- **Total:** ~37 KB (very lightweight)

---

## Accessibility Features

### ARIA Attributes

```html
<!-- Tabs -->
<button role="tab" aria-selected="true" aria-controls="tab1-panel">

<!-- Panels -->
<div role="tabpanel" aria-labelledby="tab1-button">
```

### Keyboard Navigation

- **Tab:** Focus next element
- **Shift+Tab:** Focus previous element
- **Arrow Left/Right:** Switch tabs
- **Arrow Up/Down:** Navigate file list
- **Delete/Backspace:** Remove selected files
- **Enter/Space:** Activate button

### Focus Indicators

```css
:focus-visible {
    outline: 2px solid var(--primary-color);
    outline-offset: 2px;
}

.btn:focus-visible {
    box-shadow: 0 0 0 3px rgba(2, 72, 115, 0.3);
}
```

---

## Known Limitations (Phase 3)

1. **No API integration** - Mock data only
2. **No real file dialogs** - Placeholder functions
3. **No subprocess execution** - Simulated progress
4. **No log streaming** - Mock console output
5. **No error handling** - Basic validation only

**All limitations will be resolved in Phase 4.**

---

## Next Steps: Phase 4

Phase 4 will integrate the frontend with the backend API:

1. **Wire up file dialogs** - Replace mocks with `pywebview.api.select_files()`, etc.
2. **Process management** - Call `start_process()`, `cancel_process()`, `get_process_status()`
3. **Log streaming** - Poll `get_logs()` every 100ms and update console
4. **Error handling** - Display API errors in console
5. **Validation** - Validate form data before submission
6. **State persistence** - Save/restore user preferences (optional)

---

## Troubleshooting

### Issue: Tabs not switching

**Solution:** Check browser console for JavaScript errors. Ensure `app.js` is loaded.

### Issue: File list not rendering

**Solution:** Check `AppState.selectedFiles` in console. Call `FileListManager.render()` manually.

### Issue: Styles not applying

**Solution:** Verify `style.css` is linked in `index.html`. Check for CSS syntax errors.

### Issue: Dev server won't start

**Solution:** Ensure `pywebview` is installed (`pip install pywebview`). Check Python version (3.9-3.12).

### Issue: Console not scrolling

**Solution:** Check `ConsoleManager.log()` is calling `output.scrollTop = output.scrollHeight`.

---

## Resources

### Tkinter GUI Reference
- File: `whisperjav/gui/whisperjav_gui.py`
- Tab styling: Lines 83-183
- UI structure: Lines 185-360

### Backend API Reference
- File: `whisperjav/webview_gui/api.py`
- Methods: 11 total (process management, file dialogs, configuration)

### PyWebView Documentation
- Docs: https://pywebview.flowrl.com/
- File dialogs: `window.create_file_dialog()`
- JavaScript bridge: `pywebview.api.*`

---

## Summary

Phase 3 delivers a **complete, professional frontend** with:

- ✅ 100% feature parity with Tkinter GUI
- ✅ File folder tab design matching Tkinter
- ✅ Professional color palette and typography
- ✅ Multi-select file list with keyboard navigation
- ✅ Form validation and conditional controls
- ✅ Console with auto-scroll and color coding
- ✅ Progress bar with indeterminate animation
- ✅ Accessibility features (ARIA, keyboard nav, focus indicators)
- ✅ Mock testing for all UI interactions
- ✅ Development server for hot reload

**Phase 4 Task:** Wire frontend to backend API for full functionality.

---

**Ready for Phase 4: Backend Integration**
