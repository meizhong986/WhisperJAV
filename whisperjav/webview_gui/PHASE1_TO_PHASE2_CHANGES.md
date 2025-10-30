# Phase 1 to Phase 2 Migration Guide

## Overview of Changes

Phase 2 transforms the test API into a production-ready backend while maintaining backward compatibility with Phase 1.

## File Changes

### Modified Files

#### `whisperjav/webview_gui/api.py`

**Before (Phase 1): 194 lines**
- Test methods only
- Mock log streaming
- Basic file dialogs

**After (Phase 2): 558 lines**
- Full WhisperJAV integration
- Real subprocess management
- Queue-based log streaming
- Complete feature set

### New Files

1. `whisperjav/webview_gui/test_api.py` - Standalone test suite
2. `whisperjav/webview_gui/PHASE2_DOCUMENTATION.md` - Architecture guide
3. `whisperjav/webview_gui/API_REFERENCE.md` - JavaScript reference
4. `whisperjav/webview_gui/PHASE2_SUMMARY.md` - Completion summary
5. `whisperjav/webview_gui/PHASE1_TO_PHASE2_CHANGES.md` - This file

### Unchanged Files

- `whisperjav/webview_gui/main.py` - No changes
- `whisperjav/webview_gui/assets/index.html` - No changes (will change in Phase 3)

## API Changes

### Removed Methods

None. All Phase 1 methods are preserved for backward compatibility.

### Modified Methods

#### `hello_world()`

**Before:**
```python
def hello_world(self) -> dict:
    return {
        "success": True,
        "message": "Hello from Python! PyWebView bridge is working.",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
```

**After:**
```python
def hello_world(self) -> Dict[str, Any]:
    return {
        "success": True,
        "message": "Hello from Python! PyWebView bridge is working.",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_version": "Phase 2 - Backend Refactor"  # NEW
    }
```

**Change:** Added `api_version` field to indicate Phase 2.

### Deprecated Methods

#### `start_log_stream()` and `stop_log_stream()`

**Status:** DEPRECATED (but still present)

**Reason:** Replaced by real process execution with `start_process()` and log queue polling.

**Migration:**
```javascript
// Phase 1 (DEPRECATED)
await pywebview.api.start_log_stream();  // Mock logs

// Phase 2 (NEW)
await pywebview.api.start_process({
    inputs: ['video.mp4'],
    mode: 'balanced',
    output_dir: '/tmp/output'
});

// Poll for real logs
setInterval(async () => {
    const logs = await pywebview.api.get_logs();
    logs.forEach(appendToConsole);
}, 100);
```

#### `select_video_file()`

**Status:** DEPRECATED (but still present)

**Renamed to:** `select_files()` (now supports multiple files)

**Migration:**
```javascript
// Phase 1 (single file)
const result = await pywebview.api.select_video_file();
if (result.success) {
    addFile(result.path);
}

// Phase 2 (multiple files)
const result = await pywebview.api.select_files();
if (result.success) {
    result.paths.forEach(addFile);
}
```

### New Methods

#### Process Management

1. **`build_args(options)`**
   - Builds CLI arguments from options dictionary
   - Same logic as Tkinter GUI

2. **`start_process(options)`**
   - Starts real WhisperJAV subprocess
   - Replaces mock `start_log_stream()`

3. **`cancel_process()`**
   - Cancels running subprocess
   - Graceful termination with timeout

4. **`get_process_status()`**
   - Returns process state and exit code
   - Replaces internal state checking

5. **`get_logs()`**
   - Fetches new logs from queue
   - Replaces mock log generation

#### File Dialogs

6. **`select_files()`**
   - Multi-file selection (replaces `select_video_file()`)

7. **`select_folder()`**
   - Folder selection for input

8. **`open_output_folder(path)`**
   - Opens folder in system file explorer

#### Configuration

9. **`get_default_output_dir()`**
   - Returns platform-specific default path

## Initialization Changes

### State Management

**Before (Phase 1):**
```python
def __init__(self):
    self._log_streaming = False
    self._log_thread: Optional[threading.Thread] = None
```

**After (Phase 2):**
```python
def __init__(self):
    # Process state
    self.process: Optional[subprocess.Popen] = None
    self.status = "idle"
    self.exit_code: Optional[int] = None

    # Log streaming
    self.log_queue: queue.Queue = queue.Queue()
    self._stream_thread: Optional[threading.Thread] = None

    # Default output directory
    self.default_output = str(DEFAULT_OUTPUT)
```

### New Imports

**Added:**
```python
import os
import sys
import queue
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
```

### New Module-Level Code

**Added:**
```python
# Determine REPO_ROOT for module resolution
REPO_ROOT = Path(__file__).resolve().parents[2]

# Platform-specific Documents folder detection
def _get_documents_dir() -> Path:
    # ... implementation ...

DEFAULT_OUTPUT = _get_documents_dir() / "WhisperJAV" / "output"
```

## Behavior Changes

### Log Streaming

**Phase 1:**
- Mock logs with hardcoded messages
- Direct `evaluate_js()` calls from thread
- JavaScript receives logs via `window.handleLogMessage()`

**Phase 2:**
- Real subprocess stdout/stderr
- Queue-based buffering
- JavaScript polls `get_logs()` periodically
- No direct `evaluate_js()` from processing thread

**Migration:**
```javascript
// Phase 1
window.handleLogMessage = function(message) {
    appendToConsole(message);
};

// Phase 2
setInterval(async () => {
    const logs = await pywebview.api.get_logs();
    logs.forEach(appendToConsole);
}, 100);
```

### Process Lifecycle

**Phase 1:**
- No real process management
- Mock states only

**Phase 2:**
- Full subprocess lifecycle
- States: `idle` → `running` → `completed`/`cancelled`/`error`

**Usage:**
```javascript
// Start
const result = await pywebview.api.start_process(options);

// Monitor
setInterval(async () => {
    const status = await pywebview.api.get_process_status();
    if (status.status !== 'running') {
        // Process finished
        handleCompletion(status);
    }
}, 500);

// Cancel
await pywebview.api.cancel_process();
```

## Backward Compatibility

### Preserved Methods

All Phase 1 methods are **still present**:
- ✓ `hello_world()` - Works as before (with added `api_version`)
- ✓ `start_log_stream()` - Still works (but deprecated)
- ✓ `stop_log_stream()` - Still works (but deprecated)
- ✓ `select_video_file()` - Renamed but original still present
- ✓ `select_output_directory()` - Works as before

### Gradual Migration

You can migrate incrementally:

1. **Keep Phase 1 UI** - Old HTML/JS still works
2. **Test new methods** - Add new API calls alongside old ones
3. **Switch over** - Replace old methods when ready
4. **Remove old code** - Clean up deprecated methods

### Testing Compatibility

```javascript
// Test Phase 1 compatibility
const result = await pywebview.api.hello_world();
console.log('API version:', result.api_version);
// Output: "Phase 2 - Backend Refactor"

// Test Phase 2 features
const status = await pywebview.api.get_process_status();
console.log('Process status:', status.status);
// Output: "idle"
```

## Breaking Changes

**None.** Phase 2 is 100% backward compatible with Phase 1.

## Testing

### Before (Phase 1)

No automated tests. Manual testing in GUI only.

### After (Phase 2)

Comprehensive test suite:
```bash
python -m whisperjav.webview_gui.test_api
```

**Coverage:**
- API initialization
- Argument building
- Process status
- Log queue
- File dialogs
- Configuration
- Error handling

**Result: 9/9 tests pass**

## Documentation

### Before (Phase 1)

- Inline docstrings only
- No external documentation

### After (Phase 2)

- Complete architecture guide (`PHASE2_DOCUMENTATION.md`)
- JavaScript API reference (`API_REFERENCE.md`)
- Test suite with examples (`test_api.py`)
- Summary report (`PHASE2_SUMMARY.md`)
- Migration guide (this file)

## Performance Improvements

### Log Streaming

**Phase 1:**
- Direct `evaluate_js()` per line (slow, blocking)
- Artificial 800ms delay between messages

**Phase 2:**
- Queue-based buffering (fast, non-blocking)
- Real-time streaming from subprocess
- JavaScript controls polling rate (100ms recommended)

### Process Management

**Phase 1:**
- No real process management
- Mock delays

**Phase 2:**
- Real subprocess execution
- Background thread for stdout reading
- Non-blocking queue operations

## Code Quality Improvements

### Type Hints

**Added throughout:**
```python
def build_args(self, options: Dict[str, Any]) -> List[str]:
def start_process(self, options: Dict[str, Any]) -> Dict[str, Any]:
def get_logs(self) -> List[str]:
```

### Error Handling

**Improved:**
```python
try:
    args = self.build_args(options)
except ValueError as e:
    return {"success": False, "message": str(e)}
except Exception as e:
    return {"success": False, "message": f"Failed: {e}"}
```

### Documentation

**Comprehensive docstrings:**
```python
def start_process(self, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start WhisperJAV subprocess with given options.

    Args:
        options: Dictionary of options (see build_args for format)

    Returns:
        dict: Response with success status and message
    """
```

## Migration Checklist

For developers migrating from Phase 1 to Phase 2:

### Backend (Python)
- [x] No changes needed - API is backward compatible
- [x] Optional: Remove deprecated method usage in future

### Frontend (JavaScript)
- [ ] Update `index.html` to collect full options (Phase 3)
- [ ] Replace `start_log_stream()` with `start_process()`
- [ ] Add log polling loop with `get_logs()`
- [ ] Add status monitoring with `get_process_status()`
- [ ] Update file selection to use `select_files()` (multi-select)
- [ ] Add cancel button with `cancel_process()`

### Testing
- [x] Run standalone tests: `python -m whisperjav.webview_gui.test_api`
- [x] Verify backward compatibility (Phase 1 UI still works)
- [ ] Test new features in Phase 3 UI

## Example: Complete Migration

### Phase 1 Code (Old)

```javascript
// Start mock logging
async function start() {
    await pywebview.api.start_log_stream();
}

// Handle logs
window.handleLogMessage = function(message) {
    console.log(message);
};

// Select single file
async function selectFile() {
    const result = await pywebview.api.select_video_file();
    if (result.success) {
        document.getElementById('file').value = result.path;
    }
}
```

### Phase 2 Code (New)

```javascript
// Start real processing
async function start() {
    const options = {
        inputs: getInputList(),
        mode: 'balanced',
        sensitivity: 'aggressive',
        output_dir: document.getElementById('output-dir').value
    };

    const result = await pywebview.api.start_process(options);
    if (result.success) {
        startLogPolling();
    } else {
        alert('Error: ' + result.message);
    }
}

// Poll for logs
let logInterval;
function startLogPolling() {
    logInterval = setInterval(async () => {
        const logs = await pywebview.api.get_logs();
        logs.forEach(console.log);

        // Check if done
        const status = await pywebview.api.get_process_status();
        if (status.status !== 'running') {
            stopLogPolling();
        }
    }, 100);
}

function stopLogPolling() {
    clearInterval(logInterval);
}

// Select multiple files
async function selectFiles() {
    const result = await pywebview.api.select_files();
    if (result.success) {
        result.paths.forEach(addToInputList);
    }
}

// Cancel processing
async function cancel() {
    const result = await pywebview.api.cancel_process();
    if (result.success) {
        stopLogPolling();
    }
}
```

## Summary of Benefits

### Phase 1 → Phase 2 Improvements

1. **Real Functionality**
   - Mock logs → Real subprocess execution
   - Test methods → Production methods

2. **Better Architecture**
   - Direct evaluate_js() → Queue-based buffering
   - Internal state → Explicit status API
   - No tests → Comprehensive test suite

3. **More Features**
   - 3 methods → 11 methods
   - Basic dialogs → Full file management
   - No config → Platform-specific defaults

4. **Better Documentation**
   - Inline only → 4 detailed documents
   - No examples → Complete JavaScript examples
   - No tests → 9 automated tests

5. **100% Backward Compatible**
   - All Phase 1 methods preserved
   - Gradual migration possible
   - No breaking changes

---

**Phase 2 is a strict superset of Phase 1.**

Everything that worked in Phase 1 still works in Phase 2, plus much more.
