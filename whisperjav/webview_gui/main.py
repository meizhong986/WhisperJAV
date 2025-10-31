"""
WhisperJAV PyWebView GUI Entry Point

Modern web-based GUI launcher for WhisperJAV.
Handles asset path resolution for both development and bundled modes.
"""

import os
import sys
import platform
from pathlib import Path
import time


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

    # Check for icon file
    icon_path = None
    if getattr(sys, 'frozen', False):
        # Running as executable
        try:
            icon_path = get_asset_path('icon.ico')
        except FileNotFoundError:
            pass  # No icon available
    else:
        # Running as script
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
    from whisperjav.__version__ import __version__

    # Suppress HTTP server logs from pywebview's bottle server
    # This prevents Chrome DevTools 404 messages from cluttering console
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('bottle').setLevel(logging.ERROR)

    print(f"WhisperJAV GUI v{__version__}")
    print("=" * 50)

    # Check WebView2 on Windows
    if not check_webview2_windows():
        show_webview2_error()
        sys.exit(1)

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
            webview.start(debug=debug_mode, func=_after_start)
        else:
            webview.start(debug=debug_mode)

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
