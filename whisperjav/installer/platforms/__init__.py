"""
WhisperJAV Installer Platform Support
=====================================

This subpackage contains platform-specific installation logic.

WHY PLATFORM SEPARATION:
-----------------------
Different platforms have different:
- Package managers (apt, brew, winget)
- System prerequisites (WebView2 on Windows, WebKit on Linux/macOS)
- Path conventions (backslashes vs forward slashes)
- Permission models (admin vs user installs)

By separating platform code, we:
1. Keep the main installer logic clean
2. Make it easy to add new platform support
3. Enable platform-specific testing

MODULES:
-------
- base.py: Abstract base class defining the platform interface
- windows.py: Windows-specific logic (WebView2, VC++, shortcuts)
- linux.py: Linux-specific logic (PEP 668, apt packages)
- macos.py: macOS-specific logic (Metal, brew)

Author: Senior Architect
Date: 2026-01-26
"""

# Exports will be added as modules are implemented
__all__ = []
