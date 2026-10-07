"""Put the installed environment's tool folders on PATH (Windows installs).

The Windows installer places FFmpeg (and DLLs) in <install>\\Library\\bin, which is on PATH only when the conda
environment is activated. The GUI adds it at start-up (webview_gui/main.py, _setup_conda_path); the command line did
not, so an installed CLI reported "FFmpeg not found" (known since the 1.8.11 notes; reported again on 1.9.2, #436).
This is the same logic for the command line, silent. Safe to call more than once.
"""
import os
import platform
import sys
from pathlib import Path
from typing import List


def ensure_conda_dirs_on_path() -> List[str]:
    """Prepend <sys.prefix>\\Library\\bin and \\Scripts to PATH when they exist and are missing.
    Returns the folders added (empty when nothing changed or not on Windows)."""
    if platform.system() != "Windows":
        return []
    root = Path(sys.prefix)
    current = os.environ.get("PATH", "")
    present = {d.lower() for d in current.split(os.pathsep) if d}
    added = [str(d) for d in (root / "Library" / "bin", root / "Scripts")
             if d.exists() and str(d).lower() not in present]
    if added:
        os.environ["PATH"] = os.pathsep.join(added + ([current] if current else []))
    return added
