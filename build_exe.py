import PyInstaller.__main__
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent
os.chdir(project_root)  # Ensure we're in the correct directory

# Check if icon exists, skip if not
icon_path = Path('whisperjav/gui/assets/icon.ico')
icon_args = ['--icon=' + str(icon_path)] if icon_path.exists() else []

PyInstaller.__main__.run([
    'whisperjav/gui/whisperjav_gui.py',       # Entry point to your GUI
    '--name=WhisperJAV_GUI',
    '--onefile',
    '--windowed',
    *icon_args,  # Only include icon if it exists
    
    # Include config directory
    f'--add-data=whisperjav/config{os.pathsep}config',
    
    # Include essential data files
    f'--add-data=whisperjav/config/config.template.json{os.pathsep}config',
    
    # Collect dependencies that use dynamic imports
    '--collect-all=torch',
    '--collect-all=whisper',
    '--collect-all=stable_whisper',
    '--collect-all=faster_whisper',
    
    # Hidden imports for safety
    '--hidden-import=torch',
    '--hidden-import=torchaudio',
    # Remove torchvision - not used in WhisperJAV
    '--hidden-import=stable_whisper',
    '--hidden-import=faster_whisper',
    '--hidden-import=ffmpeg',
    '--hidden-import=auditok',
    
    # Exclude unnecessary modules to reduce size
    '--exclude-module=tests',
    '--exclude-module=matplotlib',
    '--exclude-module=PyQt5',
    '--exclude-module=notebook',
    
    # Optional: Add console for debugging (remove for final release)
    # '--console',
])

print("Build complete! Check the 'dist' folder for WhisperJAV_GUI.exe")