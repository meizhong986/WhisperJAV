# Phase 2: Backend Refactor - Documentation

## Overview

Phase 2 implements a complete backend API for the WhisperJAV PyWebView GUI. The API is fully decoupled from the UI and can be tested standalone.

## What Changed

### 1. Complete API Refactor (`api.py`)

**Before (Phase 1):**
- Simple test methods (`hello_world`, `start_log_stream`)
- Mock log streaming with hardcoded test messages
- Basic file dialogs

**After (Phase 2):**
- Full WhisperJAV subprocess management
- Real CLI argument building from options
- Queue-based log streaming
- Complete process lifecycle management
- All features from Tkinter GUI migrated

### 2. New Capabilities

#### Process Management
- `build_args(options)` - Builds CLI arguments from options dictionary
- `start_process(options)` - Launches subprocess with proper environment
- `cancel_process()` - Terminates running subprocess
- `get_process_status()` - Returns current process state
- `get_logs()` - Fetches new log lines from queue

#### File Dialogs
- `select_files()` - Multi-file selection for video files
- `select_folder()` - Folder selection
- `select_output_directory()` - Output directory selection
- `open_output_folder(path)` - Opens folder in system file explorer

#### Configuration
- `get_default_output_dir()` - Returns default output directory path
- Platform-specific Documents folder detection (Windows/macOS/Linux)

## Architecture

### Core Design Principles

1. **Thin Wrapper Pattern**
   - API doesn't implement transcription logic
   - Delegates to CLI via subprocess: `python -m whisperjav.main`
   - Same pattern as Tkinter GUI

2. **Decoupled from UI**
   - No webview imports in business logic
   - Can be tested without GUI
   - Only file dialogs require active window

3. **Queue-Based Log Streaming**
   - Background thread reads subprocess stdout
   - Logs buffered in `queue.Queue`
   - JavaScript polls `get_logs()` for new messages
   - Prevents blocking on every line

### State Management

```python
class WhisperJAVAPI:
    def __init__(self):
        # Process state
        self.process: Optional[subprocess.Popen] = None
        self.status = "idle"  # idle, running, completed, cancelled, error
        self.exit_code: Optional[int] = None

        # Log streaming
        self.log_queue: queue.Queue = queue.Queue()
        self._stream_thread: Optional[threading.Thread] = None
```

### Process Lifecycle

```
idle -> running -> completed/cancelled/error -> idle
```

**State Transitions:**
- `idle`: No process running
- `running`: Subprocess active, streaming logs
- `completed`: Process finished successfully (exit code 0)
- `cancelled`: User terminated process
- `error`: Process finished with error (exit code != 0)

## API Reference

### Process Management

#### `build_args(options: Dict[str, Any]) -> List[str]`

Builds CLI arguments from options dictionary.

**Options:**
```python
{
    # Required
    'inputs': List[str],              # Input files/folders

    # Core options
    'mode': str,                      # balanced/fast/faster (default: balanced)
    'sensitivity': str,               # conservative/balanced/aggressive (default: balanced)
    'language': str,                  # japanese/english-direct (default: japanese)
    'output_dir': str,                # Output directory path

    # Optional paths
    'temp_dir': str,                  # Temporary directory path
    'keep_temp': bool,                # Keep temporary files (default: False)

    # Verbosity
    'verbosity': str,                 # quiet/summary/normal/verbose (default: summary)

    # Advanced options (WIP)
    'adaptive_classification': bool,  # Adaptive classification (default: False)
    'adaptive_audio_enhancement': bool, # Adaptive audio enhancements (default: False)
    'smart_postprocessing': bool,     # Smart postprocessing (default: False)

    # Async processing
    'async_processing': bool,         # Enable async processing (default: False)
    'max_workers': int,               # Max workers for async (default: 1)

    # Model override
    'model_override': str,            # large-v3/large-v2/turbo

    # Credits
    'credit': str,                    # Opening credit text

    # Special modes
    'check_only': bool,               # Check environment only (default: False)
}
```

**Returns:** `List[str]` - CLI arguments

**Raises:** `ValueError` - If inputs are missing or invalid

**Example:**
```python
api = WhisperJAVAPI()
options = {
    'inputs': ['video.mp4'],
    'mode': 'balanced',
    'sensitivity': 'aggressive',
    'output_dir': '/tmp/output'
}
args = api.build_args(options)
# Returns: ['video.mp4', '--mode', 'balanced', '--sensitivity', 'aggressive', ...]
```

#### `start_process(options: Dict[str, Any]) -> Dict[str, Any]`

Starts WhisperJAV subprocess with given options.

**Returns:**
```python
{
    "success": bool,
    "message": str,
    "command": str  # (optional, for debugging)
}
```

**Example:**
```python
result = api.start_process({
    'inputs': ['video.mp4'],
    'mode': 'balanced',
    'output_dir': '/tmp/output'
})

if result['success']:
    print(f"Started: {result['command']}")
else:
    print(f"Error: {result['message']}")
```

#### `cancel_process() -> Dict[str, Any]`

Cancels the running subprocess.

**Returns:**
```python
{
    "success": bool,
    "message": str
}
```

**Example:**
```python
result = api.cancel_process()
if result['success']:
    print("Process cancelled")
```

#### `get_process_status() -> Dict[str, Any]`

Gets current process status.

**Returns:**
```python
{
    "status": str,       # idle, running, completed, cancelled, error
    "exit_code": int,    # or None
    "has_logs": bool     # True if new logs available
}
```

**Example:**
```python
status = api.get_process_status()
print(f"Status: {status['status']}")
if status['exit_code'] is not None:
    print(f"Exit code: {status['exit_code']}")
```

#### `get_logs() -> List[str]`

Fetches new log lines from the queue.

**Returns:** `List[str]` - List of log lines (may be empty)

**Usage Pattern (JavaScript):**
```javascript
// Poll for logs every 100ms
setInterval(async () => {
    const logs = await pywebview.api.get_logs();
    logs.forEach(line => {
        console.log(line);
        // Append to UI console
    });
}, 100);
```

### File Dialogs

#### `select_files() -> Dict[str, Any]`

Opens native file dialog to select multiple video files.

**Returns:**
```python
{
    "success": bool,
    "paths": List[str],  # or None
    "message": str       # (if error)
}
```

#### `select_folder() -> Dict[str, Any]`

Opens native folder dialog to select a folder.

**Returns:**
```python
{
    "success": bool,
    "path": str,         # or None
    "message": str       # (if error)
}
```

#### `select_output_directory() -> Dict[str, Any]`

Alias for `select_folder()`. Opens native folder dialog for output directory.

#### `open_output_folder(path: str) -> Dict[str, Any]`

Opens output folder in system file explorer.

**Returns:**
```python
{
    "success": bool,
    "message": str
}
```

### Configuration

#### `get_default_output_dir() -> str`

Returns the default output directory path.

**Returns:** `str` - Default output directory path

**Platform-specific behavior:**
- **Windows**: `%USERPROFILE%\Documents\WhisperJAV\output`
- **macOS/Linux**: `~/Documents/WhisperJAV/output`

## Testing

### Standalone Test Suite

Run the test suite without GUI:

```bash
python -m whisperjav.webview_gui.test_api
```

**Tests:**
1. API Initialization
2. Build Arguments (basic, advanced, check-only, error handling)
3. Process Status
4. Log Queue
5. File Dialog Methods
6. Default Output Directory
7. Phase 1 Compatibility
8. Cancel Without Process
9. Start Without Inputs

**Expected Output:**
```
======================================================================
               WhisperJAV API Standalone Tests
======================================================================

... (test output) ...

======================================================================
                         Test Summary
======================================================================
  Total tests:  9
  Passed:       9
  Failed:       0
======================================================================

[SUCCESS] All tests passed! API is ready for Phase 3 (Frontend Integration).
```

### Testing Individual Methods

```python
from whisperjav.webview_gui.api import WhisperJAVAPI

# Create API instance
api = WhisperJAVAPI()

# Test argument building
options = {
    'inputs': ['test.mp4'],
    'mode': 'balanced',
    'output_dir': '/tmp/output'
}
args = api.build_args(options)
print(f"Args: {args}")

# Test log queue
api.log_queue.put("Test log\n")
logs = api.get_logs()
print(f"Logs: {logs}")

# Test status
status = api.get_process_status()
print(f"Status: {status}")
```

## Comparison with Tkinter GUI

### Similarities

1. **Subprocess Pattern**
   - Both use `subprocess.Popen` with same arguments
   - Both set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8:replace`
   - Both run from `REPO_ROOT` for module resolution

2. **Argument Building**
   - Same CLI flags and values
   - Same validation logic
   - Same default values

3. **Process Management**
   - Same cancellation logic (terminate → wait → kill)
   - Same exit code handling
   - Same log streaming approach

### Differences

1. **Log Streaming**
   - **Tkinter**: Direct `_append_log()` from reader thread
   - **PyWebView**: Queue-based buffering, polled by JavaScript
   - **Advantage**: Better performance, no GUI blocking

2. **State Exposure**
   - **Tkinter**: Internal state, UI updates directly
   - **PyWebView**: Explicit `get_process_status()` method
   - **Advantage**: Cleaner API, testable without UI

3. **File Dialogs**
   - **Tkinter**: `tkinter.filedialog`
   - **PyWebView**: `webview.create_file_dialog()`
   - **Advantage**: Native system dialogs

## Environment Setup

### REPO_ROOT Path Resolution

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
# whisperjav/webview_gui/api.py -> whisperjav/webview_gui -> whisperjav -> REPO_ROOT
```

### Subprocess Environment

```python
env = os.environ.copy()
env["PYTHONUTF8"] = "1"
env["PYTHONIOENCODING"] = "utf-8:replace"

cmd = [sys.executable, "-X", "utf8", "-m", "whisperjav.main", *args]
proc = subprocess.Popen(
    cmd,
    cwd=str(REPO_ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
    universal_newlines=True,
    encoding="utf-8",
    errors="replace",
    env=env,
)
```

**Key settings:**
- `PYTHONUTF8=1`: Force UTF-8 mode
- `PYTHONIOENCODING=utf-8:replace`: Handle encoding errors gracefully
- `cwd=str(REPO_ROOT)`: Module resolution from project root
- `universal_newlines=True`: Text mode (not bytes)
- `errors="replace"`: Replace undecodable characters

## Next Steps (Phase 3)

### Frontend Integration

1. **Update HTML/JavaScript**
   - Replace test UI with real form controls
   - Implement option collection
   - Add log polling and display

2. **Features to Implement**
   - File/folder list management
   - Mode and sensitivity selection
   - Advanced options toggles
   - Real-time log console
   - Progress indication
   - Start/Cancel buttons

3. **JavaScript API Usage**
   ```javascript
   // Start process
   const result = await pywebview.api.start_process({
       inputs: ['video.mp4'],
       mode: 'balanced',
       sensitivity: 'aggressive',
       output_dir: '/tmp/output'
   });

   // Poll for logs
   setInterval(async () => {
       const logs = await pywebview.api.get_logs();
       logs.forEach(appendToConsole);
   }, 100);

   // Check status
   const status = await pywebview.api.get_process_status();
   if (status.status === 'completed') {
       // Handle completion
   }

   // Cancel process
   await pywebview.api.cancel_process();
   ```

## Troubleshooting

### Issue: Process doesn't start

**Check:**
1. Are inputs provided? (`options['inputs']` must be non-empty)
2. Is another process running? (Only one process at a time)
3. Check logs with `get_logs()` for error messages

### Issue: No logs appearing

**Check:**
1. Is log polling enabled? (Call `get_logs()` periodically)
2. Is process actually running? (`get_process_status().status == 'running'`)
3. Queue may be empty if subprocess hasn't output yet

### Issue: Process can't be cancelled

**Check:**
1. Is process still running? (`get_process_status().status == 'running'`)
2. Check if process has already finished
3. Force kill is attempted after 5 second timeout

## Files Modified/Created

### Modified
- `whisperjav/webview_gui/api.py` - Complete refactor with real functionality

### Created
- `whisperjav/webview_gui/test_api.py` - Standalone test suite
- `whisperjav/webview_gui/PHASE2_DOCUMENTATION.md` - This file

## Summary

Phase 2 successfully implements a complete backend API that:
- Mirrors all Tkinter GUI functionality
- Uses the same subprocess pattern
- Is completely decoupled from UI
- Can be tested standalone
- Improves log streaming with queue-based approach
- Provides clean JavaScript-friendly API

All 9 test cases pass, confirming the API is ready for Phase 3: Frontend Integration.
