# Phase 1 Architecture Overview

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Launches GUI                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  whisperjav-gui-web                              │
│              (Console Script Entry Point)                        │
│                                                                   │
│  Entry in setup.py:                                              │
│    "whisperjav-gui-web=whisperjav.webview_gui.main:main"        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│           whisperjav/webview_gui/main.py                         │
│                                                                   │
│  • get_asset_path() - Resolves HTML location                    │
│  • create_window() - Configures PyWebView                       │
│  • main() - Entry point function                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PyWebView Window                              │
│                                                                   │
│  • URL: file:///path/to/assets/index.html                       │
│  • js_api: WhisperJAVAPI instance                               │
│  • Backend: WinForms (Windows) / Cocoa (Mac) / GTK (Linux)     │
└───────────────┬────────────────────────┬────────────────────────┘
                │                        │
                ▼                        ▼
┌───────────────────────────┐  ┌──────────────────────────────────┐
│  Python Backend           │  │  JavaScript Frontend             │
│  (api.py)                 │  │  (index.html)                    │
│                           │  │                                  │
│  WhisperJAVAPI Class:     │  │  UI Components:                  │
│  ├── hello_world()        │◄─┤  ├── Test buttons               │
│  ├── start_log_stream()   │  │  ├── Output areas               │
│  ├── stop_log_stream()    │  │  └── Event handlers             │
│  ├── select_video_file()  │  │                                  │
│  └── select_output_dir()  │  │  PyWebView Bridge:               │
│                           │  │  └── pywebview.api.method()     │
│  Threading:               │  │                                  │
│  └── _stream_test_logs()  ├─►│  Callbacks:                      │
│      (background thread)  │  │  └── window.handleLogMessage()  │
└───────────────────────────┘  └──────────────────────────────────┘
```

## Data Flow Examples

### Example 1: Hello World (JavaScript → Python → JavaScript)

```
1. User clicks "Test Hello World" button
   │
   ├─► [JS] onclick="testHelloWorld()"
   │
   ├─► [JS] pywebview.api.hello_world()
   │
   ├─► [PyWebView Bridge] Serializes call, invokes Python
   │
   ├─► [Python] WhisperJAVAPI.hello_world()
   │       Returns: {"success": True, "message": "...", "timestamp": "..."}
   │
   ├─► [PyWebView Bridge] Serializes response, returns to JavaScript
   │
   └─► [JS] Response displayed in output div
```

### Example 2: Log Stream (Python → JavaScript via evaluate_js)

```
1. User clicks "Start Log Stream"
   │
   ├─► [JS] pywebview.api.start_log_stream()
   │
   ├─► [Python] WhisperJAVAPI.start_log_stream()
   │       Spawns background thread: _stream_test_logs()
   │
   ├─► [Background Thread] Loop through test messages
   │       For each message:
   │       ├─► window.evaluate_js('window.handleLogMessage("...")')
   │       │
   │       └─► [JS] window.handleLogMessage(message)
   │               Appends to log output div
   │
   └─► Messages appear one-by-one in real-time
```

### Example 3: File Dialog (PyWebView Native API)

```
1. User clicks "Select Video File"
   │
   ├─► [JS] pywebview.api.select_video_file()
   │
   ├─► [Python] WhisperJAVAPI.select_video_file()
   │       Gets window: webview.windows[0]
   │       Opens dialog: window.create_file_dialog(OPEN_DIALOG, ...)
   │
   ├─► [PyWebView] Shows native OS file picker
   │       (Windows Explorer, macOS Finder, Linux GTK dialog)
   │
   ├─► [User] Selects file or cancels
   │
   ├─► [Python] Returns: {"success": True, "path": "C:\\...\\file.mp4"}
   │
   └─► [JS] Displays file path in output div
```

## Asset Loading (Dev vs Bundled)

### Development Mode
```
Repository Structure:
whisperjav/
└── webview_gui/
    ├── main.py
    └── assets/
        └── index.html

get_asset_path("index.html"):
├── sys.frozen = False
├── base_path = Path(__file__).parent
│                = .../whisperjav/webview_gui
├── asset_path = base_path / "assets" / "index.html"
└── Returns: .../whisperjav/webview_gui/assets/index.html
```

### Bundled Mode (PyInstaller)
```
PyInstaller Bundle:
dist/whisperjav-gui-web/
├── whisperjav-gui-web.exe
└── (extracted to temp folder at runtime)
    └── _MEIxxxxxx/
        ├── webview_gui_assets/
        │   └── index.html       # Bundled by spec file
        └── (other dependencies)

get_asset_path("index.html"):
├── sys.frozen = True
├── base_path = Path(sys._MEIPASS)
│                = C:\Users\...\AppData\Local\Temp\_MEIxxxxxx
├── asset_path = base_path / "webview_gui_assets" / "index.html"
└── Returns: C:\Users\...\AppData\Local\Temp\_MEIxxxxxx\webview_gui_assets\index.html
```

## PyInstaller Bundling Process

```
build_whisperjav_installer_web.bat
├── Cleans previous builds
├── Runs: pyinstaller whisperjav-gui-web.spec
│
└─► whisperjav-gui-web.spec
    ├── Analysis Phase:
    │   ├── Entry point: whisperjav/webview_gui/main.py
    │   ├── Collect hidden imports (PyWebView backends)
    │   └── Collect data files (HTML/CSS/JS)
    │
    ├── PYZ Phase:
    │   └── Create Python archive (compressed .pyc files)
    │
    ├── EXE Phase:
    │   └── Create executable bootstrap
    │
    └── COLLECT Phase:
        └── Bundle everything into dist/whisperjav-gui-web/
            ├── whisperjav-gui-web.exe
            ├── Python DLLs
            ├── PyWebView backends
            └── Data files (webview_gui_assets/*)
```

## Thin Wrapper Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                        Design Philosophy                         │
│                                                                   │
│  GUI Layer (PyWebView):                                          │
│  ├── Minimal logic                                               │
│  ├── UI rendering only                                           │
│  ├── Input collection                                            │
│  └── Progress display                                            │
│                                                                   │
│  Business Logic (CLI):                                           │
│  ├── All transcription logic                                     │
│  ├── Subprocess execution                                        │
│  ├── File handling                                               │
│  └── Configuration management                                    │
│                                                                   │
│  Communication:                                                   │
│  └── GUI spawns: python -m whisperjav.main [args]               │
│      (Same as Tkinter GUI pattern)                               │
└─────────────────────────────────────────────────────────────────┘
```

## Next Phase Preview

### Phase 2: Real WhisperJAV Integration

```
api.py additions:
├── start_transcription(video_path, options) → subprocess.Popen()
├── get_transcription_progress() → Parse stdout
├── stop_transcription() → proc.terminate()
└── Stream real-time logs via evaluate_js()

index.html additions:
├── Video file input
├── Mode/sensitivity selectors
├── Output directory picker
├── Start/Stop buttons
└── Real-time progress bar + log viewer
```

---

**Architecture designed to maintain:**
- Separation of concerns (GUI ↔ CLI)
- Cross-platform compatibility
- Easy testing and debugging
- Professional appearance
