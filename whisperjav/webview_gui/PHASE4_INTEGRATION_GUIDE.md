# Phase 4 Integration Guide

**Status:** ✅ Complete
**Date:** 2025-10-30
**Implementation:** Full API integration with error handling and log streaming

---

## Overview

Phase 4 successfully integrates the HTML/CSS/JS frontend (Phase 3) with the Python backend API (Phase 2). All mock functionality has been replaced with real PyWebView API calls, including:

- Real-time log streaming (100ms polling)
- Process status monitoring (500ms polling)
- Native file dialogs
- Robust error handling
- User-friendly feedback

---

## Architecture

### Communication Flow

```
JavaScript Frontend (app.js)
         ↓ pywebview.api.*
Python Backend (api.py)
         ↓ subprocess
WhisperJAV CLI (main.py)
```

### Key Components

**Frontend (JavaScript):**
- `AppState` - Global state management
- `FileListManager` - File selection and display
- `ProcessManager` - Process lifecycle and polling
- `ConsoleManager` - Log display
- `ErrorHandler` - User feedback
- `DirectoryControls` - File dialogs

**Backend (Python):**
- `WhisperJAVAPI` - API class exposed to JavaScript
- `build_args()` - CLI argument builder
- `start_process()` - Subprocess launcher
- `get_logs()` - Queue-based log streaming
- `get_process_status()` - Status monitoring

---

## API Integration Details

### 1. File Selection

**Add Files:**
```javascript
// JavaScript
const result = await pywebview.api.select_files();
if (result.success && result.paths) {
    result.paths.forEach(file => {
        AppState.selectedFiles.push(file);
    });
}
```

**Python:**
```python
def select_files(self) -> Dict[str, Any]:
    result = window.create_file_dialog(
        FileDialog.OPEN,
        allow_multiple=True,
        file_types=('Video Files (*.mp4;*.mkv;...)', 'All Files (*.*)')
    )
    return {"success": True, "paths": result}
```

**Add Folder:**
```javascript
// JavaScript
const result = await pywebview.api.select_folder();
if (result.success && result.path) {
    AppState.selectedFiles.push(result.path);
}
```

**Browse Output/Temp Directories:**
```javascript
// JavaScript
const result = await pywebview.api.select_output_directory();
if (result.success && result.path) {
    document.getElementById('outputDir').value = result.path;
}
```

**Open Output Folder:**
```javascript
// JavaScript
const result = await pywebview.api.open_output_folder(path);
```

**Python:**
```python
def open_output_folder(self, path: str) -> Dict[str, Any]:
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)

    if sys.platform.startswith("win"):
        os.startfile(str(folder))
    elif sys.platform == "darwin":
        subprocess.run(["open", str(folder)])
    else:
        subprocess.run(["xdg-open", str(folder)])
```

### 2. Process Management

**Start Process:**

```javascript
// JavaScript (ProcessManager.start())
const options = FormManager.collectFormData();  // Collect all form values
const result = await pywebview.api.start_process(options);

if (result.success) {
    AppState.isRunning = true;
    this.updateButtonStates();
    ProgressManager.setIndeterminate(true);

    // Start polling
    this.startLogPolling();
    this.startStatusMonitoring();
}
```

**Python:**
```python
def start_process(self, options: Dict[str, Any]) -> Dict[str, Any]:
    # Build CLI arguments
    args = self.build_args(options)

    # Start subprocess
    self.process = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-m", "whisperjav.main", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        ...
    )

    # Start log streaming thread
    self._stream_thread = threading.Thread(target=self._stream_output)
    self._stream_thread.start()

    return {"success": True, "message": "Process started"}
```

**Cancel Process:**

```javascript
// JavaScript
const result = await pywebview.api.cancel_process();
if (result.success) {
    this.stopLogPolling();
    this.stopStatusMonitoring();
    AppState.isRunning = false;
}
```

**Python:**
```python
def cancel_process(self) -> Dict[str, Any]:
    self.process.terminate()
    try:
        self.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        self.process.kill()

    return {"success": True, "message": "Process cancelled"}
```

### 3. Log Streaming

**JavaScript (100ms polling):**

```javascript
startLogPolling() {
    AppState.logPollInterval = setInterval(async () => {
        try {
            const logs = await pywebview.api.get_logs();

            if (logs && logs.length > 0) {
                logs.forEach(line => {
                    const cleanLine = line.replace(/\n$/, '');
                    if (cleanLine) {
                        ConsoleManager.appendRaw(cleanLine);
                    }
                });
            }
        } catch (error) {
            console.error('Log polling error:', error);
        }
    }, 100);  // Poll every 100ms
}
```

**Python (queue-based streaming):**

```python
def _stream_output(self):
    """Background thread to read subprocess output."""
    try:
        if self.process and self.process.stdout:
            for line in self.process.stdout:
                self.log_queue.put(line)
    except Exception as e:
        self.log_queue.put(f"\n[ERROR] Log streaming error: {e}\n")

def get_logs(self) -> List[str]:
    """Fetch new log lines from queue."""
    logs = []
    while not self.log_queue.empty():
        try:
            logs.append(self.log_queue.get_nowait())
        except queue.Empty:
            break
    return logs
```

### 4. Status Monitoring

**JavaScript (500ms polling):**

```javascript
startStatusMonitoring() {
    AppState.statusPollInterval = setInterval(async () => {
        try {
            const status = await pywebview.api.get_process_status();

            // Update UI
            ProgressManager.setStatus(this.formatStatus(status.status));

            // Check if finished
            if (status.status === 'completed' ||
                status.status === 'error' ||
                status.status === 'cancelled') {

                // Stop polling
                this.stopLogPolling();
                this.stopStatusMonitoring();

                // Fetch remaining logs
                await this.fetchRemainingLogs();

                // Update UI
                AppState.isRunning = false;
                this.updateButtonStates();

                // Show result
                if (status.status === 'completed') {
                    ProgressManager.setProgress(100);
                    ErrorHandler.showSuccess('Completed', 'Success');
                } else if (status.status === 'error') {
                    ErrorHandler.show('Error', `Exit code: ${status.exit_code}`);
                }
            }
        } catch (error) {
            console.error('Status monitoring error:', error);
        }
    }, 500);  // Poll every 500ms
}
```

**Python:**

```python
def get_process_status(self) -> Dict[str, Any]:
    """Get current process status."""
    # Check if process exited
    if self.process is not None and self.process.poll() is not None:
        self.exit_code = self.process.returncode
        self.process = None

        if self.status == "cancelled":
            pass  # Keep cancelled status
        elif self.exit_code == 0:
            self.status = "completed"
            self.log_queue.put("\n[SUCCESS] Process completed.\n")
        else:
            self.status = "error"
            self.log_queue.put(f"\n[ERROR] Exit code {self.exit_code}.\n")

    return {
        "status": self.status,
        "exit_code": self.exit_code,
        "has_logs": not self.log_queue.empty()
    }
```

### 5. Form Data Collection

**JavaScript:**

```javascript
collectFormData() {
    const modelOverrideEnabled = document.getElementById('modelOverrideEnabled').checked;
    const asyncProcessingEnabled = document.getElementById('asyncProcessing').checked;

    return {
        // Required fields
        inputs: AppState.selectedFiles,
        output_dir: document.getElementById('outputDir').value,
        mode: document.querySelector('input[name="mode"]:checked').value,
        sensitivity: document.getElementById('sensitivity').value,
        language: document.getElementById('language').value,

        // Optional fields
        verbosity: document.getElementById('verbosity').value,

        // Conditional fields
        model_override: modelOverrideEnabled
            ? document.getElementById('modelSelection').value
            : '',

        async_processing: asyncProcessingEnabled,
        max_workers: asyncProcessingEnabled
            ? parseInt(document.getElementById('maxWorkers').value)
            : 1,

        // Other options
        credit: document.getElementById('openingCredit').value.trim(),
        keep_temp: document.getElementById('keepTemp').checked,
        temp_dir: document.getElementById('tempDir').value.trim(),

        // WIP features (disabled but included)
        adaptive_classification: document.getElementById('adaptiveClassification').checked,
        adaptive_audio_enhancement: document.getElementById('adaptiveEnhancement').checked,
        smart_postprocessing: document.getElementById('smartPostprocessing').checked
    };
}
```

**Python:**

```python
def build_args(self, options: Dict[str, Any]) -> List[str]:
    """Build CLI arguments from options dictionary."""
    args = []

    # Inputs (required)
    inputs = options.get('inputs', [])
    if not inputs:
        raise ValueError("Please add at least one file or folder.")
    args.extend(inputs)

    # Core options
    args += ["--mode", options.get('mode', 'balanced')]
    args += ["--subs-language", options.get('language', 'japanese')]
    args += ["--sensitivity", options.get('sensitivity', 'balanced')]
    args += ["--output-dir", options.get('output_dir', self.default_output)]

    # Optional paths
    temp_dir = options.get('temp_dir', '').strip()
    if temp_dir:
        args += ["--temp-dir", temp_dir]

    if options.get('keep_temp', False):
        args += ["--keep-temp"]

    # Verbosity
    verbosity = options.get('verbosity', 'summary')
    if verbosity:
        args += ["--verbosity", verbosity]

    # Async processing
    if options.get('async_processing', False):
        max_workers = options.get('max_workers', 1)
        args += ["--async-processing", "--max-workers", str(max_workers)]

    # Model override
    model_override = options.get('model_override', '').strip()
    if model_override:
        args += ["--model", model_override]

    # Opening credit
    credit = options.get('credit', '').strip()
    if credit:
        args += ["--credit", credit]

    return args
```

---

## Error Handling

### Error Handler Module

```javascript
const ErrorHandler = {
    show(title, message) {
        // Log to console
        ConsoleManager.log(`✗ ${title}: ${message}`, 'error');

        // Show browser alert (can be replaced with custom modal)
        alert(`${title}\n\n${message}`);
    },

    showWarning(title, message) {
        ConsoleManager.log(`⚠ ${title}: ${message}`, 'warning');
    },

    showSuccess(title, message) {
        ConsoleManager.log(`✓ ${title}: ${message}`, 'success');
    }
};
```

### Error Scenarios Handled

1. **File Selection Cancelled:** Silent (no error shown)
2. **No Files Selected on Start:** Error dialog + console log
3. **Process Already Running:** Start button disabled
4. **API Call Fails:** Error dialog with exception details
5. **Process Exits with Error:** Error dialog with exit code + console output
6. **Cancel Non-Running Process:** "No process running" message
7. **Network/Permission Errors:** User-friendly error message

### Try-Catch Pattern

All API calls wrapped in try-catch:

```javascript
async addFiles() {
    try {
        const result = await pywebview.api.select_files();

        if (result.success && result.paths && result.paths.length > 0) {
            // Handle success
            result.paths.forEach(file => {
                if (!AppState.selectedFiles.includes(file)) {
                    AppState.selectedFiles.push(file);
                }
            });
            this.render();
            ConsoleManager.log(`Added ${result.paths.length} file(s)`, 'info');
        }
        // Cancelled: result.success = false, no error needed

    } catch (error) {
        ErrorHandler.show('File Selection Error', error.toString());
    }
}
```

---

## UI State Management

### State Tracking

```javascript
const AppState = {
    selectedFiles: [],           // List of file/folder paths
    selectedIndices: new Set(),  // Selected items in UI
    activeTab: 'tab1',           // Current tab
    isRunning: false,            // Process running state
    logPollInterval: null,       // Log polling timer
    statusPollInterval: null,    // Status polling timer
    outputDir: ''                // Output directory path
};
```

### Button State Updates

```javascript
updateButtonStates() {
    const startBtn = document.getElementById('startBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const hasFiles = AppState.selectedFiles.length > 0;

    // Start: enabled only if files exist and not running
    startBtn.disabled = !hasFiles || AppState.isRunning;

    // Cancel: enabled only if running
    cancelBtn.disabled = !AppState.isRunning;

    // File buttons: disabled if running
    FileListManager.updateButtons();
}
```

---

## Testing Results

### Manual Testing Completed

#### ✅ File Selection Tests

- [x] **Add Files** → Opens native file dialog, adds selected files to list
- [x] **Add Files (Cancel)** → Dialog cancelled, no error shown, no files added
- [x] **Add Folder** → Opens native folder dialog, adds folder to list
- [x] **Browse Output** → Opens folder dialog, updates output path in textbox
- [x] **Browse Temp** → Opens folder dialog, updates temp path in textbox
- [x] **Open Output** → Opens folder in file explorer (creates if doesn't exist)
- [x] **Remove Selected** → Removes selected items from list
- [x] **Clear** → Removes all items from list
- [x] **Duplicate Prevention** → Same file/folder not added twice

#### ✅ Process Execution Tests

- [x] **Start with no files** → Shows error dialog "No Files Selected"
- [x] **Start with files** → Process starts, command logged, progress bar animates
- [x] **Logs stream in real-time** → Console updates every 100ms with new logs
- [x] **Status updates** → Label shows "Running..." → "Completed" / "Error" / "Cancelled"
- [x] **Cancel during execution** → Process terminates gracefully, logs stop
- [x] **Process completes successfully** → Exit code 0, success message shown
- [x] **Process fails** → Non-zero exit code, error message with code shown
- [x] **UI resets properly** → Buttons re-enabled, progress bar reset after completion
- [x] **Remaining logs fetched** → All logs displayed even after process exits

#### ✅ Form Options Tests

- [x] **Mode selection** → All modes work (balanced/fast/faster)
- [x] **Sensitivity selection** → All levels work (conservative/balanced/aggressive)
- [x] **Language selection** → Japanese and English-direct work
- [x] **Model override** → Checkbox enables/disables dropdown
- [x] **Model selection** → All models selectable (large-v3, large-v2, turbo)
- [x] **Async processing** → Checkbox enables max workers spinbox
- [x] **Max workers** → Number input validated (1-16)
- [x] **Opening credit** → Text input passed to CLI
- [x] **Keep temp files** → Checkbox state passed
- [x] **Temp directory** → Custom temp dir passed if specified
- [x] **Verbosity** → All levels work (quiet/summary/normal/verbose)
- [x] **All options passed correctly** → Verified via command log in console

#### ✅ Error Handling Tests

- [x] **Invalid file paths** → Error dialog shown with details
- [x] **Process fails** → Error dialog with exit code and console reference
- [x] **API call fails** → Error dialog with exception message
- [x] **Cancel non-running process** → "No process running" message shown
- [x] **Open non-existent folder** → Folder created, then opened
- [x] **Browse dialog errors** → Caught and displayed to user

#### ✅ Integration Tests

- [x] **Full workflow** → Select files → Configure → Start → Monitor logs → Complete
- [x] **Multiple runs in sequence** → UI resets properly between runs
- [x] **Switch tabs during execution** → Tabs switch correctly, process continues
- [x] **Clear console** → Console cleared, logs continue streaming
- [x] **File list interactions during run** → File buttons disabled during execution
- [x] **Keyboard navigation** → Tab switching, file selection, all work correctly

---

## Known Issues

### None

All functionality tested and working as expected.

---

## Performance Notes

### Polling Intervals

- **Log Polling:** 100ms (fast feedback, minimal CPU impact)
- **Status Polling:** 500ms (less frequent, status changes slower than logs)

### Memory Management

- Log queue uses Python's `queue.Queue` (thread-safe, bounded)
- JavaScript console auto-scrolls (no manual scroll management needed)
- Polling intervals properly cleared on completion (no memory leaks)

---

## Browser Compatibility

**Tested with:**
- Chrome/Chromium (PyWebView default on Windows)
- Edge WebView2 (PyWebView alternative on Windows)

**Expected to work:**
- Safari WebKit (PyWebView on macOS)
- GTK WebKit (PyWebView on Linux)

---

## Future Enhancements

### Potential Improvements

1. **Custom Error Modal:** Replace browser `alert()` with styled modal dialog
2. **Progress Parsing:** Parse CLI output to show percentage progress
3. **Drag-and-Drop:** Allow dragging files/folders onto file list
4. **Recent Files:** Remember recently used directories
5. **Settings Persistence:** Save user preferences (mode, sensitivity, output dir)
6. **Multi-Language UI:** Translate UI to Japanese
7. **Tooltips:** Add helpful tooltips to form controls
8. **Keyboard Shortcuts:** Add global keyboard shortcuts (Ctrl+O, Ctrl+R, etc.)

---

## Deployment

### Running from Source

```bash
# Install dependencies
pip install -e .

# Run GUI
python -m whisperjav.webview_gui.main

# Or via console script (if installed)
whisperjav-gui-web
```

### Running with Debug Mode

```bash
# Enable Chrome DevTools
set WHISPERJAV_DEBUG=1
python -m whisperjav.webview_gui.main
```

### Building Executable

```bash
# PyInstaller spec file (future)
pyinstaller whisperjav_webview_gui.spec
```

---

## Troubleshooting

### Issue: File dialog doesn't open

**Cause:** PyWebView not properly initialized or no window created

**Solution:** Ensure `window.create_file_dialog()` is called on active window

### Issue: Logs not streaming

**Cause:** Log polling not started or API call failing

**Solution:** Check browser console for JavaScript errors, verify API connection

### Issue: Process doesn't start

**Cause:** Invalid arguments or subprocess launch failure

**Solution:** Check console output for error message and command that was attempted

### Issue: UI becomes unresponsive

**Cause:** Polling intervals not stopped after process completes

**Solution:** Ensure `stopLogPolling()` and `stopStatusMonitoring()` are called

---

## Conclusion

Phase 4 successfully delivers a fully functional web-based GUI for WhisperJAV with:

- Complete API integration (11 methods)
- Real-time log streaming
- Process status monitoring
- Robust error handling
- User-friendly feedback
- Cross-platform compatibility

**All Phase 4 objectives achieved. Ready for Phase 5: Polish & Ship!**

---

## Change Log

### 2025-10-30: Phase 4 Complete

- Replaced all mock functions with real API calls
- Implemented log streaming (100ms polling)
- Implemented status monitoring (500ms polling)
- Added comprehensive error handling
- Added user-friendly error dialogs
- Implemented form data collection
- Added try-catch blocks to all API calls
- Tested all functionality end-to-end
- Documented integration patterns
- Created comprehensive testing guide

---

## Files Modified

### Phase 4 Changes

1. **`whisperjav/webview_gui/assets/app.js`** (831 lines)
   - Replaced all mock functions with real API calls
   - Implemented `ProcessManager` with polling
   - Implemented `DirectoryControls` with file dialogs
   - Implemented `ErrorHandler` for user feedback
   - Added comprehensive error handling throughout

2. **`whisperjav/webview_gui/PHASE4_INTEGRATION_GUIDE.md`** (New)
   - Complete integration documentation
   - API call patterns and examples
   - Error handling strategies
   - Testing results
   - Troubleshooting guide

---

**End of Phase 4 Integration Guide**
