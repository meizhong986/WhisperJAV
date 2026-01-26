# whisperjav/cli.py
#!/usr/bin/env python3
"""
Command line interface entry point for WhisperJAV.

This module provides the console script entry point that is installed
via pip. It handles early setup (console encoding, warning suppression)
before importing the main application.

Entry point: whisperjav
"""

# ===========================================================================
# EARLY SETUP - Must be before any library imports
# ===========================================================================
# Import console utilities first for encoding fix and warning suppression
from whisperjav.utils.console import setup_console
setup_console()

# ===========================================================================
# Standard imports (after console setup)
# ===========================================================================
import sys
import shutil


def main():
    """Console script entry point for whisperjav command."""
    # Pre-flight checks
    if sys.version_info < (3, 10):
        print("Error: WhisperJAV requires Python 3.10 or higher")
        print(f"Current version: {sys.version}")
        sys.exit(1)

    # Check for FFmpeg in PATH (required dependency)
    if shutil.which("ffmpeg") is None:
        print("Error: FFmpeg is not installed or not in PATH")
        print()
        print("Please install FFmpeg:")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        print("  Linux:   sudo apt install ffmpeg")
        print("  macOS:   brew install ffmpeg")
        sys.exit(1)

    # Check for CLI dependencies (optional but recommended)
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        print("Warning: CLI processing dependencies not installed")
        print("For full functionality, install with: pip install whisperjav[cli]")
        print()

    # Import and run the main application
    try:
        from whisperjav.main import main as whisperjav_main
        whisperjav_main()
    except ImportError as e:
        print(f"Error: Failed to import WhisperJAV: {e}")
        print()
        print("This may indicate a broken installation. Try:")
        print("  pip install --force-reinstall whisperjav[cli]")
        sys.exit(1)


if __name__ == "__main__":
    main()
