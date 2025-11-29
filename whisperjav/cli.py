# whisperjav/cli.py
#!/usr/bin/env python3
"""Command line interface entry point for WhisperJAV."""

# ===========================================================================
# EARLY WARNING SUPPRESSION - Must be before any library imports
# ===========================================================================
import os
import warnings

# TensorFlow/oneDNN warnings - suppress before TF is loaded as side effect
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Suppress specific Python warnings from dependencies
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*chunk_length_s.*is very experimental.*")
warnings.filterwarnings("ignore", message=".*sparse_softmax_cross_entropy.*deprecated.*")
# ===========================================================================

import sys
import io
import shutil

# Fix stdout/stderr encoding for Windows BEFORE any imports
# This ensures unicode characters work correctly in console output
def _fix_console_encoding():
    """Ensure stdout and stderr use UTF-8 encoding on Windows."""
    if sys.stdout is not None and (not hasattr(sys.stdout, 'encoding') or sys.stdout.encoding.lower() != 'utf-8'):
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else io.BufferedWriter(io.FileIO(1, 'w')),
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )
        except (AttributeError, OSError):
            pass

    if sys.stderr is not None and (not hasattr(sys.stderr, 'encoding') or sys.stderr.encoding.lower() != 'utf-8'):
        try:
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer if hasattr(sys.stderr, 'buffer') else io.BufferedWriter(io.FileIO(2, 'w')),
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )
        except (AttributeError, OSError):
            pass

# Apply fix immediately at module level
_fix_console_encoding()

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