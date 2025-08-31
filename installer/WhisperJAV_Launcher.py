# WhisperJAV_Launcher.py
#!/usr/bin/env python
"""WhisperJAV GUI Launcher"""

import os
import sys
import json
import subprocess
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

def check_first_run(install_root: Path):
    """Check if this is first run and download models if needed"""
    config_path = install_root / "whisperjav_config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        if config.get("first_run", True):
            # Show first run dialog
            root = tk.Tk()
            root.withdraw()
            
            result = messagebox.askyesno(
                "First Run - Download Models",
                "This appears to be your first run.\n\n"
                "WhisperJAV needs to download AI models (~3GB).\n"
                "This is a one-time download.\n\n"
                "Download now?",
                icon='question'
            )
            
            root.destroy()
            
            if result:
                # Update config
                config["first_run"] = False
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print("Model download will start when you select a model in the GUI.")
            else:
                print("You can download models later from the GUI.")

def main():
    """Launch WhisperJAV GUI using the installed Python (pythonw.exe)."""
    try:
        # Determine install root: when frozen, use folder of the EXE; else use sys.prefix
        exe_dir = Path(getattr(sys, 'frozen', False) and os.path.dirname(sys.executable) or sys.prefix)

        # Add Scripts to PATH for this session
        scripts_dir = str(exe_dir / "Scripts")
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")

        # Check first run prompt
        check_first_run(exe_dir)

        # Build command to run GUI entrypoint with pythonw (no console window)
        pythonw = os.path.join(str(exe_dir), "pythonw.exe")
        if not os.path.exists(pythonw):
            # Fallback to python.exe if pythonw.exe is not present
            pythonw = os.path.join(str(exe_dir), "python.exe")

        cmd = [pythonw, "-m", "whisperjav.gui.whisperjav_gui"]
        # Launch detached so the GUI isn't tied to the launcher process
        creationflags = 0x00000008  # DETACHED_PROCESS
        subprocess.Popen(cmd, cwd=str(exe_dir), creationflags=creationflags)

    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to start WhisperJAV:\n{e}\n\n"
            "Please check the installation."
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
