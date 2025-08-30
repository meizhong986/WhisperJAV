# whisperjav/gui/__init__.py
"""WhisperJAV GUI Package"""

# This file is kept intentionally simple to avoid the import conflicts
# that were causing a RuntimeWarning and slow performance.

def main():
    """Main entry point for GUI."""
    from whisperjav.gui.whisperjav_gui import main as gui_main
    gui_main()