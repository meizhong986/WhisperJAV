# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for WhisperJAV CLI

Builds a console executable for the CLI entrypoint (whisperjav).

Usage:
    pyinstaller installer/whisperjav-cli.spec

Output:
    dist/whisperjav/whisperjav.exe (Windows)
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Add repository root to path
repo_root = os.path.abspath(os.path.join(SPECPATH, '..'))
sys.path.insert(0, repo_root)

datas = []

# Include configuration JSONs
config_dir = os.path.join(repo_root, 'whisperjav', 'config')
if os.path.exists(config_dir):
    for name in os.listdir(config_dir):
        if name.endswith('.json'):
            datas.append((os.path.join(config_dir, name), 'whisperjav/config'))

# Include instruction presets
instr_dir = os.path.join(repo_root, 'whisperjav', 'instructions')
if os.path.exists(instr_dir):
    for name in os.listdir(instr_dir):
        if name.endswith('.txt'):
            datas.append((os.path.join(instr_dir, name), 'whisperjav/instructions'))

# Include translate defaults (prompt templates)
trans_defaults = os.path.join(repo_root, 'whisperjav', 'translate', 'defaults')
if os.path.exists(trans_defaults):
    for name in os.listdir(trans_defaults):
        if name.endswith('.txt'):
            datas.append((os.path.join(trans_defaults, name), 'whisperjav/translate/defaults'))

# Hidden imports for dynamic modules
hidden = [
    # core package
    'whisperjav',
    'whisperjav.main',
    'whisperjav.cli',
    'whisperjav.config',
    'whisperjav.config.manager',
    'whisperjav.modules',
    'whisperjav.modules.audio_extraction',
    'whisperjav.modules.audio_preparation',
    'whisperjav.modules.audio_preprocessing',
    'whisperjav.modules.cross_subtitle_processor',
    'whisperjav.modules.hallucination_remover',
    'whisperjav.modules.media_discovery',
    'whisperjav.modules.repetition_cleaner',
    'whisperjav.modules.scene_detection',
    'whisperjav.modules.segment_classification',
    'whisperjav.modules.srt_postprocessing',
    'whisperjav.modules.srt_postproduction',
    # pipelines
    'whisperjav.pipelines',
    'whisperjav.pipelines.balanced_pipeline',
    'whisperjav.pipelines.fast_pipeline',
    'whisperjav.pipelines.faster_pipeline',
    # translation (invoked optionally)
    'whisperjav.translate',
    'whisperjav.translate.cli',
    'whisperjav.translate.core',
    'whisperjav.translate.providers',
    # utils
    'whisperjav.utils',
    # third-party commonly missed
    'stable_whisper',
    'faster_whisper',
    'whisper',
    'numpy',
    'scipy',
    'soundfile',
    'librosa',
    'auditok',
    'pysrt',
    'srt',
    'tqdm',
    'PIL',
    'pyloudnorm',
    # PySubtrans when translation is used
    'pysubtrans',
]

# Try to collect any submodules that PyInstaller might miss
hidden += collect_submodules('whisperjav')

a = Analysis(
    ['../whisperjav/main.py'],
    pathex=[repo_root],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # exclude legacy GUI toolkits to reduce size
        'tkinter',
        '_tkinter',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

version_info_path = os.path.join(repo_root, 'installer', 'version_info.txt')
if not os.path.exists(version_info_path):
    version_info_path = None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='whisperjav',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # CLI wants a console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    version=version_info_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='whisperjav',
)
