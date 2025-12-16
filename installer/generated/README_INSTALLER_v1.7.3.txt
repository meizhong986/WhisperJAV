===============================================================================
                            WhisperJAV v1.7.3
                Japanese AV Subtitle Generator with AI
===============================================================================

Thank you for installing WhisperJAV!

-------------------------------------------------------------------------------
QUICK START
-------------------------------------------------------------------------------
1. Double-click the "WhisperJAV v1.7.3" desktop icon
2. Select your video/audio files using "Add File(s)" or "Add Folder"
3. Choose processing mode (Balanced recommended for best quality)
4. Click "Start" to begin processing
5. Subtitles will be saved next to your media files in an output folder

FIRST RUN NOTE: On your first transcription, AI models will download (~3GB).
                This is a one-time process that takes 5-10 minutes depending
                on your internet speed. Progress is shown in the GUI.

-------------------------------------------------------------------------------
WHAT'S NEW IN v1.7.3
-------------------------------------------------------------------------------
Version 1.7.3 introduces speech enhancement backends for improved audio quality:

1. SPEECH ENHANCEMENT BACKENDS (NEW)
   - ZipEnhancer: Lightweight SOTA denoising (16kHz, ~500MB VRAM)
   - ClearVoice: High-quality MossFormer2 denoising (48kHz)
   - BS-RoFormer: Vocal isolation from music/background (44.1kHz)
   - FFmpeg DSP: Basic audio filtering without AI
   - None: Passthrough (no enhancement)

2. NUMPY 2.x SUPPORT
   - Full compatibility with NumPy 2.x
   - ModelScope/ZipEnhancer work with latest numpy

3. DEPENDENCY PROTECTION
   - Constraints file prevents PyTorch/NumPy downgrade
   - Protected installation order for ML packages

-------------------------------------------------------------------------------
WHAT THIS INSTALLER DOES
-------------------------------------------------------------------------------
This installer sets up a complete, self-contained WhisperJAV environment:

1. Base Environment Setup (~1-2 minutes)
   - Python 3.10.18 with conda package manager
   - FFmpeg for audio/video processing
   - Git for GitHub package installs

2. Post-Installation Downloads (~8-15 minutes, ~2.5 GB download)
   - PyTorch with CUDA support (for NVIDIA GPU acceleration)
     OR CPU-only PyTorch (if no compatible GPU detected)
   - Python dependencies (Whisper, audio processing, translation)
   - Speech enhancement backends (ModelScope, ClearVoice, BS-RoFormer)
   - WhisperJAV application from GitHub

3. Setup and Configuration
   - WhisperJAV-GUI.exe launcher created in installation folder
   - Desktop shortcut creation (points to WhisperJAV-GUI.exe)
   - Environment configuration
   - WebView2 runtime check (required for GUI)

TOTAL INSTALLATION TIME: ~10-20 minutes (depending on internet speed)
TOTAL DISK SPACE REQUIRED: ~8 GB (install + cache + models)

-------------------------------------------------------------------------------
SYSTEM REQUIREMENTS
-------------------------------------------------------------------------------
MINIMUM:
  - Windows 10 (64-bit) or Windows 11
  - 8 GB RAM (16 GB recommended for large models)
  - 8 GB free disk space (install + models + cache)
  - Internet connection (for downloads)
  - Microsoft Edge WebView2 Runtime (installer will prompt if missing)

RECOMMENDED FOR BEST PERFORMANCE:
  - NVIDIA GPU with 6+ GB VRAM (RTX 2060 or better)
  - NVIDIA Driver with CUDA 11.8 or newer (CUDA 12.1+ preferred)
  - 16 GB RAM
  - SSD for faster processing

SUPPORTED PLATFORMS:
  - Windows: NVIDIA GPU (RTX 20/30/40/50-series) or CPU-only
  - macOS: Apple Silicon (M1/M2/M3/M4/M5) with native MPS acceleration
  - macOS: Intel (CPU-only, slower)
  - Linux: NVIDIA GPU or CPU-only

CPU-ONLY MODE:
  WhisperJAV can run without an NVIDIA GPU using CPU-only mode, but
  processing will be significantly slower (6-10x slower than GPU mode).
  Expect 30-60 minutes per hour of video on CPU vs 5-10 minutes on GPU.

CUDA GPU ACCELERATION:
  The installer automatically detects your NVIDIA driver's CUDA version and
  installs the best matching PyTorch build for optimal performance.

-------------------------------------------------------------------------------
AFTER INSTALLATION
-------------------------------------------------------------------------------
LAUNCHING THE APPLICATION:
  - Desktop shortcut: "WhisperJAV v1.7.3.lnk" (double-click to start)
  - Manual launch: Double-click WhisperJAV-GUI.exe in installation folder
  - Alternative: Open the installation folder and run:
    pythonw.exe -m whisperjav.webview_gui.main

INSTALLATION LOCATION:
  - Default: C:\Users\[YourName]\AppData\Local\WhisperJAV
  - Or custom location chosen during installation

LOGS AND DIAGNOSTICS:
  - Installation log: install_log_v1.7.3.txt (in install folder)
  - Application logs: Shown in GUI console during processing
  - Failure marker: INSTALLATION_FAILED_v1.7.3.txt (only if install failed)

OUTPUT FILES:
  - Subtitles are saved next to your input video/audio files
  - Format: SRT (SubRip Text) - compatible with most video players
  - Filename pattern: [original_name]_output/[original_name].srt

-------------------------------------------------------------------------------
SPEECH ENHANCEMENT (NEW IN v1.7.3)
-------------------------------------------------------------------------------
Speech enhancement improves transcription quality by cleaning audio:

BACKENDS:
  - zipenhancer: Best balance of quality and speed (recommended)
    - Uses ModelScope ZipEnhancer model
    - ~500MB VRAM, native 16kHz
    - Ideal for ASR pre-processing

  - clearvoice: Highest quality denoising
    - Uses MossFormer2 at 48kHz
    - 9-16GB VRAM required
    - Best for very noisy audio

  - bs-roformer: Vocal isolation
    - Separates vocals from music/background
    - Use when background music is present

  - ffmpeg-dsp: Basic DSP filtering
    - No AI, uses FFmpeg filters
    - Good for simple audio fixes

  - none: No enhancement (passthrough)

USAGE:
  Select enhancement backend in GUI settings or via CLI:
  whisperjav video.mp4 --speech-enhancer zipenhancer

-------------------------------------------------------------------------------
TROUBLESHOOTING
-------------------------------------------------------------------------------
INSTALLATION ISSUES:

1. "NVIDIA driver not found" or "CUDA version too old":
   - Download latest driver from: https://www.nvidia.com/drivers
   - Or accept CPU-only installation when prompted (slower processing)

2. "WebView2 runtime not detected":
   - The installer will open: https://go.microsoft.com/fwlink/p/?LinkId=2124703
   - Download and install the "Evergreen Standalone Installer"
   - Most Windows 10/11 systems have this pre-installed

3. "Network connection failed":
   - Check your internet connection
   - Disable VPN or proxy temporarily
   - Check firewall settings (allow Python, pip, git)

4. "Out of disk space":
   - Free up at least 8 GB on your system drive
   - Models are downloaded to C:\Users\[Name]\.cache\whisper (~3 GB)

5. "Installation failed after retries":
   - Check install_log_v1.7.3.txt for specific error messages
   - Common causes: antivirus blocking downloads, network timeout
   - Try running the installer as Administrator (right-click > Run as admin)

RUNTIME ISSUES:

1. "GUI won't launch" or "Blank window":
   - Ensure WebView2 is installed (see issue #2 above)
   - Try running from command line to see error messages:
     python.exe -m whisperjav.webview_gui.main

2. "Processing is very slow":
   - Check if CUDA is enabled: Look for "GPU acceleration: ENABLED" in console
   - If CPU-only, consider upgrading to an NVIDIA GPU
   - For large videos, use "faster" mode for quicker results

3. "Speech enhancement backend not available":
   - Ensure modelscope, clearvoice, or bs-roformer-infer is installed
   - Try: pip install modelscope clearvoice bs-roformer-infer
   - Check GUI settings for available backends

-------------------------------------------------------------------------------
UNINSTALLING
-------------------------------------------------------------------------------
To completely remove WhisperJAV v1.7.3:

1. Delete the installation directory:
   C:\Users\[YourName]\AppData\Local\WhisperJAV (or your custom location)

2. Delete the desktop shortcut:
   Desktop\WhisperJAV v1.7.3.lnk

3. (Optional) Delete cached models to free up ~3 GB:
   C:\Users\[YourName]\.cache\whisper

4. (Optional) Delete user configuration:
   [Install Directory]\whisperjav_config.json

NOTE: An automated uninstaller (uninstall_v1.7.3.bat) is included in the
      installation directory for your convenience.

-------------------------------------------------------------------------------
UPGRADING
-------------------------------------------------------------------------------
To upgrade an existing WhisperJAV installation:

1. Run upgrade_whisperjav.bat (or upgrade_whisperjav.py)
   Located in your installation folder

2. The upgrade script will:
   - Install new dependencies (including speech enhancement backends)
   - Update WhisperJAV from GitHub
   - Preserve your configuration and models

-------------------------------------------------------------------------------
VERSION INFORMATION
-------------------------------------------------------------------------------
WhisperJAV Version: 1.7.3
Release Date: December 2025
Installer Version: v1.7.3 (conda-constructor)

Key Changes in 1.7.3:
  - Speech enhancement backends (ZipEnhancer, ClearVoice, BS-RoFormer)
  - NumPy 2.x compatibility
  - Dependency conflict protection
  - ONNX runtime support for ZipEnhancer

Previous Major Changes (1.7.2):
  - Multi-platform GPU support (Apple Silicon, Blackwell)
  - Resume translation feature (auto-save progress)
  - Multi-language support (Chinese, Korean - experimental)
  - GUI improvements (language selection, CPU mode checkbox)

For full release notes, see RELEASE_NOTES_v1.7.3.md

-------------------------------------------------------------------------------
SUPPORT & COMMUNITY
-------------------------------------------------------------------------------
GitHub Repository: https://github.com/meizhong986/WhisperJAV
Issue Tracker: https://github.com/meizhong986/WhisperJAV/issues
Documentation: See GitHub README

For bugs, feature requests, or questions, please open an issue on GitHub
with detailed information about your system and the problem.

-------------------------------------------------------------------------------
LICENSE
-------------------------------------------------------------------------------
WhisperJAV is released under the MIT License.
See LICENSE.txt in the installation directory for details.

Copyright (c) 2025 MeiZhong and contributors

===============================================================================
