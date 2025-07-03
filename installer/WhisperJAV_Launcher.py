# WhisperJAV_Launcher.py
#!/usr/bin/env python
"""WhisperJAV GUI Launcher"""

import os
import sys
import json
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

def check_first_run():
    """Check if this is first run and download models if needed"""
    config_path = Path(sys.prefix) / "whisperjav_config.json"
    
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
    """Launch WhisperJAV GUI"""
    try:
        # Add Scripts to PATH for this session
        scripts_dir = os.path.join(sys.prefix, "Scripts")
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
        
        # Check first run
        check_first_run()
        
        # Import and launch GUI
        print("Starting WhisperJAV GUI...")
        from whisperjav.gui.whisperjav_gui import main as gui_main
        gui_main()
        
    except ImportError as e:
        messagebox.showerror(
            "Import Error",
            f"Failed to import WhisperJAV:\n{e}\n\n"
            "The installation may be incomplete.\n"
            "Please reinstall WhisperJAV."
        )
        sys.exit(1)
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to start WhisperJAV:\n{e}\n\n"
            "Please check the installation."
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
