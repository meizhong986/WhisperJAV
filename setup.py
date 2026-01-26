#!/usr/bin/env python
"""
Minimal setup.py shim for backward compatibility.

All package configuration is now in pyproject.toml.
This file exists to support:
- Older pip versions that don't fully support PEP 517/518
- Legacy tools that expect setup.py

For the full configuration, see pyproject.toml.

Migration Notes:
- Dependencies are now modular via extras (pip install whisperjav[cli,gui,translate])
- Platform-specific markers handle Windows/Linux/macOS automatically
- The original setup.py is preserved as setup_legacy.py for reference
"""
from setuptools import setup

if __name__ == "__main__":
    setup()
