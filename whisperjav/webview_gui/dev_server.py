"""
WhisperJAV PyWebView GUI - Development Server

Hot reload development environment for rapid frontend iteration.
Serves the HTML/CSS/JS assets directly without the full API backend.

Usage:
    python -m whisperjav.webview_gui.dev_server

Features:
- Debug mode enabled (shows developer console)
- Live reload: Edit HTML/CSS/JS and refresh to see changes
- No API calls needed for frontend development
- Same window size as production (1000x700)

Note: This is for frontend development only. API calls will fail.
      For full integration testing, use main.py instead.
"""

import webview
from pathlib import Path


def main():
    """
    Launch development server with hot reload.

    Simply edit the HTML/CSS/JS files in the assets/ directory,
    save your changes, and refresh the browser window to see updates.
    """
    # Get path to index.html
    assets_dir = Path(__file__).parent / 'assets'
    index_path = assets_dir / 'index.html'

    if not index_path.exists():
        print(f"Error: index.html not found at {index_path}")
        return

    print("=" * 60)
    print("WhisperJAV Web GUI - Development Server")
    print("=" * 60)
    print(f"Serving: {index_path}")
    print("Window size: 1000x700")
    print("Debug mode: ENABLED")
    print()
    print("Usage:")
    print("  1. Edit HTML/CSS/JS files in assets/ directory")
    print("  2. Save your changes")
    print("  3. Refresh the window (Ctrl+R / Cmd+R) to see updates")
    print()
    print("Note: API calls will fail in dev mode (Phase 3)")
    print("      For full testing, use: python -m whisperjav.webview_gui.main")
    print("=" * 60)
    print()

    # Create window with debug mode enabled
    window = webview.create_window(
        title='WhisperJAV GUI (Dev Mode)',
        url=index_path.as_uri(),
        width=1000,
        height=700,
        resizable=True,
        min_size=(800, 600)
    )

    # Start with debug mode for developer console
    webview.start(debug=True)


if __name__ == '__main__':
    main()
