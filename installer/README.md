# WhisperJAV Installer

This directory contains the files needed to build a standalone Windows installer for WhisperJAV using conda constructor.

## Prerequisites

- Anaconda or Miniconda installed
- Constructor package: conda install constructor -c conda-forge

## Building the Installer

1. First, create the icon (if not already present):
   python create_icon.py

2. Build the installer:
   constructor . --clean
   
   Or use the batch file:
   build_installer.bat

3. The installer will be created as: WhisperJAV-1.1.2-Windows-x86_64-Setup.exe

## Files Description

- construct.yaml - Main configuration file for constructor
- post_install.py - Script that runs after installation to set up WhisperJAV
- WhisperJAV_Launcher.py - Python launcher script for the GUI
- WhisperJAV.bat - Batch file launcher
- LICENSE.txt - License text shown during installation
- README_INSTALLER.txt - Readme file included with installation
- create_icon.py - Script to generate the application icon
- whisperjav_icon.ico - Application icon (generated)

## Customization

Edit construct.yaml to:
- Change version number
- Add/remove packages
- Modify installer text
- Change installation paths

## Distribution

Upload the generated .exe file to GitHub Releases or your distribution platform.
Do NOT commit the .exe file to the repository.
