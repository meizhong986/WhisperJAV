"""
WhisperJAV PyWebView GUI Entry Point

Modern web-based GUI launcher for WhisperJAV.
Handles asset path resolution for both development and bundled modes.

Requires the [gui] extra: pip install whisperjav[gui]
"""

# ===========================================================================
# EARLY SETUP - Must be before any library imports
# ===========================================================================
import sys

# Console setup (UTF-8, warnings)
from whisperjav.utils.console import setup_console, print_missing_extra_error
setup_console()

# Platform detection
from whisperjav.utils.platform import is_windows

# ===========================================================================
# GUI DEPENDENCY CHECK - Before importing pywebview
# ===========================================================================
def _check_gui_dependencies():
    """Check if GUI dependencies are installed."""
    missing = []

    try:
        import webview  # noqa: F401
    except ImportError:
        missing.append("pywebview")

    # Windows-specific dependencies
    if is_windows():
        try:
            import clr  # pythonnet
        except ImportError:
            missing.append("pythonnet")

    if missing:
        print_missing_extra_error(
            extra_name="gui",
            missing_packages=missing,
            feature_description="PyWebView GUI interface"
        )
        if is_windows():
            print("Note: On Windows, WebView2 runtime is also required.")
            print("Download from: https://developer.microsoft.com/en-us/microsoft-edge/webview2/")
        elif sys.platform == "darwin":
            print("On macOS, the GUI uses WebKit which is built into the OS.")
            print("You just need the Python binding. Install it with:")
            print("  pip install pywebview")
            print()
            print("If you installed from source, re-run the installer:")
            print("  ./installer/install_mac.sh")
        sys.exit(1)

_check_gui_dependencies()

# ===========================================================================
# Standard imports (after dependency check)
# ===========================================================================
import os
import platform
from pathlib import Path
import time
import json

import webview
from webview.dom import DOMEventHandler


def _setup_conda_path():
    """
    Set up PATH for conda environment dependencies (FFmpeg, DLLs).

    When launched directly via pip entry point (WhisperJAV-GUI.exe),
    the conda environment's Library\bin is not on PATH. This causes
    FFmpeg and other bundled tools to be inaccessible.

    This function adds the necessary directories to PATH:
    - Library\bin (FFmpeg, DLLs)
    - Scripts (other conda tools)

    Safe to call multiple times - directories are only added once.
    """
    if platform.system() != 'Windows':
        return  # Only needed on Windows conda installs

    # Determine installation root
    # sys.prefix is the conda environment root (e.g., %LOCALAPPDATA%\WhisperJAV)
    install_root = Path(sys.prefix)

    # Key directories to add to PATH
    lib_bin_dir = install_root / "Library" / "bin"
    scripts_dir = install_root / "Scripts"

    # Get current PATH
    current_path = os.environ.get("PATH", "")
    path_dirs = current_path.split(os.pathsep)

    # Add directories if not already present
    dirs_to_add = []
    for dir_path in [str(lib_bin_dir), str(scripts_dir)]:
        # Case-insensitive check on Windows
        if not any(d.lower() == dir_path.lower() for d in path_dirs):
            if Path(dir_path).exists():
                dirs_to_add.append(dir_path)

    if dirs_to_add:
        new_path = os.pathsep.join(dirs_to_add + [current_path])
        os.environ["PATH"] = new_path
        print(f"Added to PATH: {', '.join(dirs_to_add)}")


def on_drop_event(e):
    """
    Handle file/folder drops from OS into WebView.

    Uses PyWebView's pywebviewFullPath to get absolute file paths,
    bypassing browser security restrictions.

    Args:
        e: DOM drop event with dataTransfer.files
    """
    files = e.get('dataTransfer', {}).get('files', [])
    if len(files) == 0:
        return

    paths = []
    unsupported = []

    for file in files:
        # Get full path using PyWebView's native property
        full_path = file.get('pywebviewFullPath')
        if full_path:
            paths.append(full_path)
        else:
            # Fallback: filename only (shouldn't happen, but defensive)
            unsupported.append(file.get('name', 'unknown'))

    # Send paths to JavaScript
    try:
        window = webview.windows[0]
        if paths:
            # Escape backslashes for JavaScript string
            paths_json = json.dumps(paths)
            window.evaluate_js(f"FileListManager.addDroppedFiles({paths_json})")

        if unsupported:
            names_str = ', '.join(unsupported)
            window.evaluate_js(
                f"ErrorHandler.showWarning('Incomplete Paths', "
                f"'Some files could not be added: {names_str}. Please use Add Files button.')"
            )
    except Exception as ex:
        print(f"Error handling drop event: {ex}")


def bind_dom_events(window):
    """
    Bind drag-drop events to window DOM after creation.

    Registers handlers for dragenter, dragover, and drop events
    to enable native file path access via pywebviewFullPath.

    Args:
        window: PyWebView window instance
    """
    try:
        # Prevent default for drag events (required for drop to work)
        # preventDefault=True, stopPropagation=True
        window.dom.document.events.dragenter += DOMEventHandler(lambda e: None, True, True)
        window.dom.document.events.dragover += DOMEventHandler(lambda e: None, True, True)

        # Handle actual file drops
        window.dom.document.events.drop += DOMEventHandler(on_drop_event, True, True)

        print("DOM drag-drop events bound successfully")
    except Exception as ex:
        print(f"Warning: Could not bind DOM events: {ex}")
        print("Drag-drop may not work correctly. Please use Add Files button.")


def get_asset_path(relative_path: str) -> Path:
    """
    Get absolute path to asset file.

    Handles both development mode and PyInstaller bundled mode.

    Args:
        relative_path: Path relative to assets directory (e.g., "index.html")

    Returns:
        Absolute path to the asset file

    Example:
        >>> get_asset_path("index.html")
        Path("/path/to/whisperjav/webview_gui/assets/index.html")
    """
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as bundled executable
        base_path = Path(sys._MEIPASS)
        asset_path = base_path / "webview_gui_assets" / relative_path
    else:
        # Running in development mode
        base_path = Path(__file__).parent
        asset_path = base_path / "assets" / relative_path

    if not asset_path.exists():
        raise FileNotFoundError(
            f"Asset file not found: {asset_path}\n"
            f"Relative path requested: {relative_path}\n"
            f"Base path: {base_path}\n"
            f"Frozen: {getattr(sys, 'frozen', False)}"
        )

    return asset_path


def check_webview2_windows():
    """
    Check if WebView2 runtime is installed on Windows.

    Returns:
        bool: True if WebView2 is available or not needed, False if missing on Windows
    """
    if platform.system() != 'Windows':
        return True  # Only needed on Windows

    try:
        import winreg
        # Check registry for WebView2 runtime
        key_paths = [
            r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
            r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
        ]

        for key_path in key_paths:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                winreg.CloseKey(key)
                return True  # Found WebView2
            except FileNotFoundError:
                continue

        return False  # Not found

    except Exception as e:
        print(f"Warning: Could not check WebView2 status: {e}")
        return True  # Assume it's there, let it fail naturally if not


def show_webview2_error():
    """Show user-friendly error dialog if WebView2 is missing."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()

        message = """WebView2 Runtime is required but not installed.

WhisperJAV GUI uses Microsoft Edge WebView2 to display its interface.

Please download and install it from:
https://go.microsoft.com/fwlink/p/?LinkId=2124703

After installation, restart WhisperJAV GUI."""

        messagebox.showerror("WebView2 Required", message)
        root.destroy()
    except Exception as e:
        # Fallback to console message if tkinter not available
        print("\n" + "=" * 60)
        print("ERROR: WebView2 Runtime Required")
        print("=" * 60)
        print("\nWhisperJAV GUI requires Microsoft Edge WebView2.")
        print("\nDownload and install from:")
        print("https://go.microsoft.com/fwlink/p/?LinkId=2124703")
        print("\nAfter installation, restart WhisperJAV GUI.")
        print("=" * 60 + "\n")


def create_window():
    """
    Create and configure the PyWebView window.

    Returns:
        webview.Window: Configured window instance
    """
    import webview
    from .api import WhisperJAVAPI

    # Get HTML file path
    html_path = get_asset_path("index.html")

    # Create API instance
    api = WhisperJAVAPI()

    # Check for icon file with multiple fallback locations
    icon_path = None

    # Allow disabling icon loading for debugging libpng/rendering issues
    # Set WHISPERJAV_NO_ICON=1 to skip icon loading entirely
    skip_icon = os.getenv('WHISPERJAV_NO_ICON', '').lower() in ('1', 'true', 'yes')
    if skip_icon:
        print("Icon loading disabled via WHISPERJAV_NO_ICON")
    else:
        # Priority 1: Check install root directory (where installer places whisperjav_icon.ico)
        # This is needed for conda-constructor installations
        install_root_icon = Path(sys.prefix) / "whisperjav_icon.ico"
        if install_root_icon.exists():
            icon_path = install_root_icon

        # Priority 2: Check bundled executable assets (PyInstaller)
        elif getattr(sys, 'frozen', False):
            try:
                icon_path = get_asset_path('icon.ico')
            except FileNotFoundError:
                pass  # No icon available

        # Priority 3: Running as script - check assets directory
        else:
            icon_file = Path(__file__).parent / "assets" / "icon.ico"
            if icon_file.exists():
                icon_path = icon_file


    width_s, height_s = int(1920 * 0.63), int(1080 * 0.85)

    # Create window with icon if available
    window_kwargs = {
        'title': "WhisperJAV GUI",
        'url': str(html_path),
        'js_api': api,
        'width': width_s,
        'height': height_s,
        'resizable': True,
        'frameless': False,
        'easy_drag': True,
        'text_select': True,
        'min_size': (800, 600)
    }

    # Only pass 'icon' if the installed pywebview supports it
    icon_used = False
    if icon_path and icon_path.exists():
        try:
            import inspect
            if 'icon' in inspect.signature(webview.create_window).parameters:
                window_kwargs['icon'] = str(icon_path)
                icon_used = True
        except Exception:
            # Be conservative: if we cannot introspect, skip setting icon to avoid TypeError
            pass

    window = webview.create_window(**window_kwargs)

    # Return the window along with icon path and whether icon was applied via kwarg
    return window, icon_path, icon_used


def _set_windows_icon(window_title: str, ico_path: Path, timeout: float = 5.0) -> bool:
    """Best-effort set taskbar/titlebar icon on Windows when pywebview lacks 'icon'.

    Finds the top-level window for this process with the given title and sends
    WM_SETICON for both small and big icons. Returns True if applied.
    """
    if not ico_path or not ico_path.exists():
        return False

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        EnumWindows = user32.EnumWindows
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        GetWindowTextW = user32.GetWindowTextW
        GetWindowTextLengthW = user32.GetWindowTextLengthW
        IsWindowVisible = user32.IsWindowVisible
        GetWindowThreadProcessId = user32.GetWindowThreadProcessId
        SendMessageW = user32.SendMessageW
        LoadImageW = user32.LoadImageW

        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x0010
        LR_DEFAULTSIZE = 0x0040

        current_pid = os.getpid()

        def _find_hwnd_by_title(target_title: str):
            found = []

            def callback(hwnd, lParam):  # noqa: N802
                if not IsWindowVisible(hwnd):
                    return True
                length = GetWindowTextLengthW(hwnd)
                if length == 0:
                    return True
                buf = ctypes.create_unicode_buffer(length + 1)
                GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                pid = wintypes.DWORD()
                GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value == current_pid and title == target_title:
                    found.append(hwnd)
                    return False  # stop enum
                return True

            EnumWindows(EnumWindowsProc(callback), 0)
            return found[0] if found else None

        # Wait for the window to actually exist
        hwnd = None
        end_time = time.time() + timeout
        while time.time() < end_time and hwnd is None:
            hwnd = _find_hwnd_by_title(window_title)
            if hwnd is None:
                time.sleep(0.05)

        if hwnd is None:
            return False

        # Load icon from file and set it
        hicon = LoadImageW(None, str(ico_path), IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE)
        if not hicon:
            return False

        SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)
        SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)
        return True

    except Exception:
        return False


def main():
    """
    Entry point for whisperjav-gui-web command.

    Initializes PyWebView and starts the GUI.
    """
    import webview
    import logging
    # Use display version for user-facing output, fallback to PEP 440 version
    try:
        from whisperjav.__version__ import __version_display__ as version
    except ImportError:
        from whisperjav.__version__ import __version__ as version

    # Suppress HTTP server logs from pywebview's bottle server
    # This prevents Chrome DevTools 404 messages from cluttering console
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('bottle').setLevel(logging.ERROR)

    print(f"WhisperJAV GUI v{version}")
    print("=" * 50)

    # Set up PATH for conda environment (FFmpeg, DLLs)
    # Must be called BEFORE any code that needs FFmpeg
    _setup_conda_path()

    # Check WebView2 on Windows
    if not check_webview2_windows():
        show_webview2_error()
        sys.exit(1)

    # Set Windows AppUserModelID for taskbar icon support
    # Must be called BEFORE creating GUI windows
    # This tells Windows this is a unique application, not just pythonw.exe
    if platform.system() == 'Windows':
        try:
            import ctypes
            # Unique identifier for WhisperJAV (company.product.subproduct.version format)
            appid = 'WhisperJAV.GUI.PyWebView.v1.x'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)
            print(f"Windows AppUserModelID set: {appid}")
        except Exception as e:
            print(f"Warning: Could not set Windows AppUserModelID: {e}")
            # Non-fatal - continue anyway

    try:
        # Create window (stored in webview's global window registry)
        window, icon_path, icon_used = create_window()
        print("Window created successfully")
        print(f"Asset path: {get_asset_path('index.html')}")

        # Start GUI
        # Use debug=True for development, False for production
        # Debug mode can be enabled via WHISPERJAV_DEBUG environment variable
        debug_mode = os.getenv('WHISPERJAV_DEBUG', '').lower() in ('1', 'true', 'yes')
        print(f"Starting PyWebView... (debug={debug_mode})")
        # If icon kwarg couldn't be used (older pywebview), set icon after start on Windows
        if platform.system() == 'Windows' and icon_path and not icon_used:
            def _after_start():
                _set_windows_icon('WhisperJAV GUI', icon_path)
                bind_dom_events(window)
            webview.start(debug=debug_mode, func=_after_start)
        else:
            webview.start(debug=debug_mode, func=lambda: bind_dom_events(window))

    except FileNotFoundError as e:
        print("\nERROR: Asset file not found!")
        print(str(e))
        print("\nIf running from source, ensure you're in the repository root.")
        print("If running from executable, ensure assets were bundled correctly.")
        sys.exit(1)

    except Exception as e:
        print("\nERROR: Failed to start GUI!")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
