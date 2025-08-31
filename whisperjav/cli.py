# whisperjav/cli.py
#!/usr/bin/env python3
"""Command line interface entry point for WhisperJAV."""

import sys
import os
import shutil

def main():
    """Console script entry point."""
    # Pre-flight checks
    if sys.version_info < (3, 8):
        print("Error: WhisperJAV requires Python 3.8 or higher")
        sys.exit(1)
    
    # Check for FFmpeg in PATH (cross-platform)
    if shutil.which("ffmpeg") is None:
        print("Error: FFmpeg is not installed or not in PATH")
        sys.exit(1)
    
    # Now import and run the actual main function
    try:
        from whisperjav.main import main as whisperjav_main
        whisperjav_main()
    except ImportError as e:
        print(f"Error: Failed to import WhisperJAV: {e}")
        print("Please ensure WhisperJAV is properly installed")
        sys.exit(1)

if __name__ == "__main__":
    main()