# WhisperJAV PyWebView API - Quick Reference

## API Methods Available to JavaScript

All methods are accessible via `pywebview.api.method_name()` in JavaScript.

## Process Management

### `start_process(options)`
Start WhisperJAV transcription process.

**Parameters:**
- `options` (object): Configuration options

**Options Object:**
```javascript
{
    // Required
    inputs: ['video.mp4', 'folder/'],     // Array of file/folder paths

    // Core settings (all optional with defaults)
    mode: 'balanced',                      // 'balanced' | 'fast' | 'faster'
    sensitivity: 'balanced',               // 'conservative' | 'balanced' | 'aggressive'
    language: 'japanese',                  // 'japanese' | 'english-direct'
    output_dir: '/path/to/output',         // Output directory

    // Optional settings
    temp_dir: '/tmp/whisperjav',           // Temporary directory
    keep_temp: false,                      // Keep temporary files
    verbosity: 'summary',                  // 'quiet' | 'summary' | 'normal' | 'verbose'

    // Advanced (WIP)
    adaptive_classification: false,
    adaptive_audio_enhancement: false,
    smart_postprocessing: false,

    // Async processing
    async_processing: false,
    max_workers: 1,

    // Model override
    model_override: 'large-v3',           // 'large-v3' | 'large-v2' | 'turbo'

    // Credits
    credit: 'Produced by Studio',

    // Special modes
    check_only: false                      // Only check environment
}
```

**Returns:**
```javascript
{
    success: true,
    message: "Process started successfully",
    command: "whisperjav.main video.mp4 --mode balanced ..."  // for debugging
}
```

**Example:**
```javascript
const result = await pywebview.api.start_process({
    inputs: ['video.mp4'],
    mode: 'balanced',
    sensitivity: 'aggressive',
    output_dir: '/tmp/output'
});

if (result.success) {
    console.log('Process started:', result.command);
    startLogPolling();
} else {
    console.error('Failed to start:', result.message);
}
```

---

### `cancel_process()`
Cancel the running process.

**Returns:**
```javascript
{
    success: true,
    message: "Process cancelled"
}
```

**Example:**
```javascript
const result = await pywebview.api.cancel_process();
if (result.success) {
    console.log('Process cancelled');
}
```

---

### `get_process_status()`
Get current process status.

**Returns:**
```javascript
{
    status: 'running',         // 'idle' | 'running' | 'completed' | 'cancelled' | 'error'
    exit_code: null,           // number or null
    has_logs: true             // boolean
}
```

**Example:**
```javascript
const status = await pywebview.api.get_process_status();

if (status.status === 'completed') {
    console.log('Process finished successfully');
    stopLogPolling();
} else if (status.status === 'error') {
    console.error('Process failed with exit code:', status.exit_code);
}
```

---

### `get_logs()`
Fetch new log lines from the queue.

**Returns:** `string[]` - Array of log lines (may be empty)

**Example:**
```javascript
// Poll for logs every 100ms
let logPollInterval = null;

function startLogPolling() {
    logPollInterval = setInterval(async () => {
        const logs = await pywebview.api.get_logs();

        logs.forEach(line => {
            // Append to console UI
            appendToConsole(line);
        });

        // Check if process finished
        const status = await pywebview.api.get_process_status();
        if (status.status !== 'running') {
            stopLogPolling();
        }
    }, 100);
}

function stopLogPolling() {
    if (logPollInterval) {
        clearInterval(logPollInterval);
        logPollInterval = null;
    }
}
```

---

## File Dialogs

### `select_files()`
Open native file dialog to select video files (multiple selection).

**Returns:**
```javascript
{
    success: true,
    paths: ['/path/to/video1.mp4', '/path/to/video2.mkv']
}
```

**Example:**
```javascript
const result = await pywebview.api.select_files();
if (result.success) {
    result.paths.forEach(path => addToInputList(path));
}
```

---

### `select_folder()`
Open native folder dialog to select a folder.

**Returns:**
```javascript
{
    success: true,
    path: '/path/to/folder'
}
```

**Example:**
```javascript
const result = await pywebview.api.select_folder();
if (result.success) {
    addToInputList(result.path);
}
```

---

### `select_output_directory()`
Open native folder dialog to select output directory.

**Returns:** Same as `select_folder()`

**Example:**
```javascript
const result = await pywebview.api.select_output_directory();
if (result.success) {
    document.getElementById('output-dir').value = result.path;
}
```

---

### `open_output_folder(path)`
Open output folder in system file explorer.

**Parameters:**
- `path` (string): Directory path to open

**Returns:**
```javascript
{
    success: true,
    message: "Folder opened"
}
```

**Example:**
```javascript
const outputDir = document.getElementById('output-dir').value;
const result = await pywebview.api.open_output_folder(outputDir);
```

---

## Configuration

### `get_default_output_dir()`
Get the default output directory path.

**Returns:** `string` - Default output directory path

**Example:**
```javascript
const defaultDir = await pywebview.api.get_default_output_dir();
document.getElementById('output-dir').value = defaultDir;
```

**Platform-specific defaults:**
- Windows: `d:\Users\YourName\Documents\WhisperJAV\output`
- macOS: `/Users/YourName/Documents/WhisperJAV/output`
- Linux: `/home/yourname/Documents/WhisperJAV/output`

---

## Test Methods (Phase 1 Compatibility)

### `hello_world()`
Test method to verify Python-JavaScript bridge.

**Returns:**
```javascript
{
    success: true,
    message: "Hello from Python! PyWebView bridge is working.",
    timestamp: "2025-10-30 14:44:49",
    api_version: "Phase 2 - Backend Refactor"
}
```

---

## Complete Usage Example

```javascript
// Initialize UI
async function initializeUI() {
    // Set default output directory
    const defaultDir = await pywebview.api.get_default_output_dir();
    document.getElementById('output-dir').value = defaultDir;
}

// Add files
async function addFiles() {
    const result = await pywebview.api.select_files();
    if (result.success) {
        result.paths.forEach(path => {
            // Add to input list UI
            addToInputList(path);
        });
    }
}

// Add folder
async function addFolder() {
    const result = await pywebview.api.select_folder();
    if (result.success) {
        addToInputList(result.path);
    }
}

// Start processing
async function startProcessing() {
    // Collect options from form
    const options = {
        inputs: getInputList(),
        mode: document.getElementById('mode').value,
        sensitivity: document.getElementById('sensitivity').value,
        language: document.getElementById('language').value,
        output_dir: document.getElementById('output-dir').value,
        verbosity: document.getElementById('verbosity').value,
        async_processing: document.getElementById('async').checked,
        max_workers: parseInt(document.getElementById('workers').value),
    };

    // Start process
    const result = await pywebview.api.start_process(options);

    if (result.success) {
        console.log('Started:', result.command);

        // Update UI
        document.getElementById('start-btn').disabled = true;
        document.getElementById('cancel-btn').disabled = false;

        // Start log polling
        startLogPolling();
    } else {
        alert('Error: ' + result.message);
    }
}

// Cancel processing
async function cancelProcessing() {
    const result = await pywebview.api.cancel_process();
    if (result.success) {
        console.log('Cancelled');
        stopLogPolling();
    }
}

// Log polling
let logPollInterval = null;

function startLogPolling() {
    logPollInterval = setInterval(async () => {
        // Fetch new logs
        const logs = await pywebview.api.get_logs();
        logs.forEach(line => {
            appendToConsole(line);
        });

        // Check status
        const status = await pywebview.api.get_process_status();

        if (status.status === 'completed') {
            console.log('Process completed successfully');
            stopLogPolling();
            onProcessComplete();
        } else if (status.status === 'error') {
            console.error('Process failed:', status.exit_code);
            stopLogPolling();
            onProcessError(status.exit_code);
        } else if (status.status === 'cancelled') {
            console.log('Process cancelled');
            stopLogPolling();
            onProcessCancelled();
        }
    }, 100);
}

function stopLogPolling() {
    if (logPollInterval) {
        clearInterval(logPollInterval);
        logPollInterval = null;
    }
}

// Event handlers
function onProcessComplete() {
    document.getElementById('start-btn').disabled = false;
    document.getElementById('cancel-btn').disabled = true;
    document.getElementById('status').textContent = 'Completed';
}

function onProcessError(exitCode) {
    document.getElementById('start-btn').disabled = false;
    document.getElementById('cancel-btn').disabled = true;
    document.getElementById('status').textContent = `Error (${exitCode})`;
}

function onProcessCancelled() {
    document.getElementById('start-btn').disabled = false;
    document.getElementById('cancel-btn').disabled = true;
    document.getElementById('status').textContent = 'Cancelled';
}

// Initialize on load
window.addEventListener('DOMContentLoaded', initializeUI);
```

---

## Error Handling

All API methods return objects with a `success` field. Always check this before proceeding:

```javascript
const result = await pywebview.api.some_method();
if (!result.success) {
    console.error('Error:', result.message);
    // Handle error
    return;
}

// Success - use result
console.log('Success:', result);
```

---

## Process States

```
idle → running → completed
                ↓
                cancelled
                ↓
                error
```

- **idle**: No process running
- **running**: Process is active, logs streaming
- **completed**: Process finished successfully (exit code 0)
- **cancelled**: User terminated the process
- **error**: Process finished with error (exit code != 0)

---

## Recommended Polling Intervals

- **Logs**: Poll every 100ms for responsive UI
- **Status**: Poll every 500ms if not polling logs
- **Stop polling**: When status is not 'running'

---

## Common Patterns

### Starting Process with Validation

```javascript
async function startWithValidation() {
    const inputs = getInputList();

    if (inputs.length === 0) {
        alert('Please add at least one file or folder');
        return;
    }

    const result = await pywebview.api.start_process({
        inputs: inputs,
        mode: getSelectedMode(),
        output_dir: getOutputDir(),
        // ... other options
    });

    if (!result.success) {
        alert('Failed to start: ' + result.message);
        return;
    }

    startLogPolling();
}
```

### Graceful Shutdown

```javascript
window.addEventListener('beforeunload', async (e) => {
    const status = await pywebview.api.get_process_status();

    if (status.status === 'running') {
        e.preventDefault();
        e.returnValue = 'Process is still running. Are you sure you want to close?';

        // Cancel process on confirm
        await pywebview.api.cancel_process();
    }
});
```

---

## Testing the API

To test without GUI:

```bash
python -m whisperjav.webview_gui.test_api
```

This runs 9 comprehensive tests covering all API functionality.
