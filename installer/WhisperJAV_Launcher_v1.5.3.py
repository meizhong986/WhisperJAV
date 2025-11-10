#!/usr/bin/env python
"""
WhisperJAV v1.5.3 GUI Launcher
================================

This launcher script:
1. Sets up the Python environment (PATH for ffmpeg, DLLs)
2. Checks for first-run status (informational only)
3. Launches the PyWebView GUI using pythonw.exe (no console window)
4. Detaches the GUI process so the launcher can exit

The GUI handles all user interaction, including model downloads
on first transcription.

No Tkinter dependency - all messages go to console (if run from terminal)
or are handled by the GUI itself.
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def log_first_run_info(install_root: Path):
    """
    Check if this is first run and log informational message.

    Note: We don't prompt the user here. The PyWebView GUI will handle
    model downloads automatically when the user starts their first transcription.
    """
    config_path = install_root / "whisperjav_config.json"

    try:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if config.get("first_run", True):
                print("=" * 60)
                print("  WhisperJAV First Run")
                print("=" * 60)
                print("AI models (~3GB) will download on first transcription.")
                print("This is a one-time download that takes 5-10 minutes.")
                print("=" * 60)
                print()

                # Mark first run as complete
                config["first_run"] = False
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)

        else:
            # Create default config for first run
            config = {
                "first_run": True,
                "version": "1.5.3"
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)

    except Exception as e:
        # Non-fatal: if config handling fails, just continue
        print(f"Note: Could not read/write config file: {e}")
        print("This won't affect the application.")


def main():
    """Launch WhisperJAV GUI using pythonw.exe (windowless)"""
    try:
        # Determine installation root directory
        # When frozen (PyInstaller), use directory of executable
        # Otherwise, use sys.prefix (conda environment root)
        if getattr(sys, 'frozen', False):
            install_root = Path(os.path.dirname(sys.executable))
        else:
            install_root = Path(sys.prefix)

        print(f"WhisperJAV v1.5.3 Launcher")
        print(f"Installation: {install_root}")
        print()

        # Add Scripts and Library\bin to PATH for this session
        # This ensures ffmpeg and other DLLs are accessible
        scripts_dir = install_root / "Scripts"
        lib_bin_dir = install_root / "Library" / "bin"

        path_additions = [
            str(scripts_dir),
            str(lib_bin_dir),
            os.environ.get("PATH", "")
        ]
        os.environ["PATH"] = os.pathsep.join(path_additions)

        # Check for first run (informational only)
        log_first_run_info(install_root)

        # Find pythonw.exe (preferred) or python.exe (fallback)
        pythonw = install_root / "pythonw.exe"
        if not pythonw.exists():
            pythonw = install_root / "python.exe"
            if not pythonw.exists():
                print(f"ERROR: Cannot find Python executable in {install_root}")
                print("Installation may be corrupted.")
                input("Press Enter to exit...")
                sys.exit(1)

        print(f"Launching GUI...")
        print(f"Using: {pythonw}")
        print()

        # Build command to launch PyWebView GUI
        cmd = [
            str(pythonw),
            "-m",
            "whisperjav.webview_gui.main"
        ]

        # Launch as detached process (DETACHED_PROCESS flag on Windows)
        # This allows the launcher to exit while the GUI continues running
        if sys.platform == "win32":
            creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            creationflags = 0

        subprocess.Popen(
            cmd,
            cwd=str(install_root),
            creationflags=creationflags,
            close_fds=True
        )

        print("GUI launched successfully!")
        print("You can close this window.")

    except FileNotFoundError as e:
        print()
        print("=" * 60)
        print("  ERROR: File Not Found")
        print("=" * 60)
        print(f"{e}")
        print()
        print("The installation may be incomplete or corrupted.")
        print("Please try reinstalling WhisperJAV.")
        print("=" * 60)
        input("Press Enter to exit...")
        sys.exit(1)

    except Exception as e:
        print()
        print("=" * 60)
        print("  ERROR: Failed to Start WhisperJAV")
        print("=" * 60)
        print(f"{e}")
        print()
        print("Please check the installation or report this issue at:")
        print("https://github.com/meizhong986/WhisperJAV/issues")
        print("=" * 60)
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
