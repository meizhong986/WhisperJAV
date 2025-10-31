# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for WhisperJAV PyWebView GUI

This builds a standalone executable for the PyWebView-based GUI.
HTML/CSS/JS assets are bundled into the executable.

Usage:
    pyinstaller installer/whisperjav-gui.spec

Output:
    dist/whisperjav-gui.exe (Windows)
    dist/whisperjav-gui (macOS/Linux)
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os
import sys

# Add repository root to path
repo_root = os.path.abspath(os.path.join(SPECPATH, '..'))
sys.path.insert(0, repo_root)

# Collect all submodules for pywebview to ensure platform-specific backends are included
pywebview_hiddenimports = collect_submodules('webview')

# Collect data files for bundling
datas = []

# 1. Collect HTML/CSS/JS assets from webview_gui/assets/
assets_dir = os.path.join(repo_root, 'whisperjav', 'webview_gui', 'assets')
if os.path.exists(assets_dir):
    for filename in os.listdir(assets_dir):
        file_path = os.path.join(assets_dir, filename)
        if os.path.isfile(file_path):
            # Bundle as webview_gui_assets/ in the executable
            datas.append((file_path, 'webview_gui_assets'))

# 2. Collect config files
config_dir = os.path.join(repo_root, 'whisperjav', 'config')
if os.path.exists(config_dir):
    for filename in os.listdir(config_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(config_dir, filename)
            datas.append((file_path, 'whisperjav/config'))

# 3. Collect translation defaults
translate_defaults = os.path.join(repo_root, 'whisperjav', 'translate', 'defaults')
if os.path.exists(translate_defaults):
    for filename in os.listdir(translate_defaults):
        if filename.endswith('.txt'):
            file_path = os.path.join(translate_defaults, filename)
            datas.append((file_path, 'whisperjav/translate/defaults'))

# Get icon path
icon_path = os.path.join(assets_dir, 'icon.ico')
if not os.path.exists(icon_path):
    # Fallback to installer icon
    icon_path = os.path.join(repo_root, 'installer', 'whisperjav_icon.ico')
if not os.path.exists(icon_path):
    icon_path = None  # No icon available

# Analysis phase
a = Analysis(
    ['../whisperjav/webview_gui/main.py'],
    pathex=[repo_root],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # PyWebView dependencies
        'webview',
        'webview.window',
        'webview.platforms',
        'webview.platforms.winforms',  # Windows backend
        'webview.platforms.cocoa',     # macOS backend
        'webview.platforms.gtk',       # Linux backend
        'whisperjav.webview_gui',
        'whisperjav.webview_gui.api',
        'whisperjav.webview_gui.main',
        # WhisperJAV core modules
        'whisperjav',
        'whisperjav.pipelines',
        'whisperjav.pipelines.faster_pipeline',
        'whisperjav.pipelines.fast_pipeline',
        'whisperjav.pipelines.balanced_pipeline',
        'whisperjav.modules',
        'whisperjav.modules.audio_extraction',
        'whisperjav.modules.scene_detection',
        'whisperjav.modules.audio_preprocessing',
        'whisperjav.modules.stable_ts_asr',
        'whisperjav.modules.whisper_pro_asr',
        'whisperjav.modules.srt_postprocessing',
        'whisperjav.modules.subtitle_sanitizer',
        'whisperjav.modules.media_discovery',
        'whisperjav.utils',
        'whisperjav.utils.logger',
        'whisperjav.utils.metadata_manager',
        'whisperjav.utils.progress_aggregator',
        'whisperjav.config',
        'whisperjav.config.manager',
        # Core scientific packages
        'numpy',
        'numpy.core',
        'scipy',
        'scipy.ndimage',
        'scipy.signal',
        # Whisper dependencies
        'whisper',
        'stable_whisper',
        'faster_whisper',
        # Audio processing
        'soundfile',
        'librosa',
        'auditok',
    ] + pywebview_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude old Tkinter GUI (replaced by PyWebView)
        'tkinter',
        'tkinter.ttk',
        '_tkinter',
        # Exclude all Qt bindings - PyWebView uses WebView2 on Windows, not Qt
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PySide2',
        'PySide2.QtCore',
        'PySide2.QtGui',
        'PySide2.QtWidgets',
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PyQt5.sip',
        'sip',
    ],
    # NOTE: DO NOT exclude numpy, scipy, torch, matplotlib, or PIL/Pillow
    # These are REQUIRED core dependencies for WhisperJAV!
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# PYZ (Python archive)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Version info path (Windows only)
version_info_path = os.path.join(repo_root, 'installer', 'version_info.txt')
if not os.path.exists(version_info_path):
    version_info_path = None

# EXE (executable)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='whisperjav-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # False for production (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
    version=version_info_path,  # Windows version info
)

# COLLECT (bundle everything into dist folder)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='whisperjav-gui',
)
