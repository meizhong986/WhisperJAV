===============================================================================
                            WhisperJAV v1.5.1
                Japanese AV Subtitle Generator with AI
===============================================================================

Thank you for installing WhisperJAV!

-------------------------------------------------------------------------------
QUICK START
-------------------------------------------------------------------------------
1. Double-click the "WhisperJAV v1.5.1" desktop icon
2. Select your video/audio files using "Add File(s)" or "Add Folder"
3. Choose processing mode (Balanced recommended for best quality)
4. Click "Start" to begin processing
5. Subtitles will be saved next to your media files in an output folder

FIRST RUN NOTE: On your first transcription, AI models will download (~3GB).
                This is a one-time process that takes 5-10 minutes depending
                on your internet speed. Progress is shown in the GUI.

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

CPU-ONLY MODE:
  WhisperJAV can run without an NVIDIA GPU using CPU-only mode, but
  processing will be significantly slower (6-10x slower than GPU mode).
  Expect 30-60 minutes per hour of video on CPU vs 5-10 minutes on GPU.

CUDA GPU ACCELERATION:
  The installer automatically detects your NVIDIA driver's CUDA version and
  installs the best matching PyTorch build for optimal performance:

  - CUDA 11.8 - 12.0: PyTorch with CUDA 11.8 support
  - CUDA 12.1 - 12.3: PyTorch with CUDA 12.1 support
  - CUDA 12.4+:       PyTorch with CUDA 12.4 support

  This ensures you get the best possible GPU acceleration for your hardware
  while maintaining compatibility with your driver version.

-------------------------------------------------------------------------------
AFTER INSTALLATION
-------------------------------------------------------------------------------
LAUNCHING THE APPLICATION:
  - Desktop shortcut: "WhisperJAV v1.5.1.lnk" (double-click to start)
  - Manual launch: Double-click WhisperJAV-GUI.exe in installation folder
  - Alternative: Open the installation folder and run:
    pythonw.exe -m whisperjav.webview_gui.main

INSTALLATION LOCATION:
  - Default: C:\Users\[YourName]\AppData\Local\WhisperJAV
  - Or custom location chosen during installation

LOGS AND DIAGNOSTICS:
  - Installation log: install_log_v1.5.1.txt (in install folder)
  - Application logs: Shown in GUI console during processing
  - Failure marker: INSTALLATION_FAILED_v1.5.1.txt (only if install failed)

OUTPUT FILES:
  - Subtitles are saved next to your input video/audio files
  - Format: SRT (SubRip Text) - compatible with most video players
  - Filename pattern: [original_name]_output/[original_name].srt

-------------------------------------------------------------------------------
FEATURES
-------------------------------------------------------------------------------
PROCESSING MODES:
  - Balanced: Best quality, uses scene detection + voice activity detection
  - Fast: Good quality, faster than balanced, uses scene detection
  - Faster: Fastest processing, direct transcription without preprocessing

SENSITIVITY LEVELS:
  - Conservative: Fewer false positives, cleaner output
  - Balanced: Good balance of detail and accuracy (recommended)
  - Aggressive: Maximum detail capture, may include more background noise

OUTPUT LANGUAGES:
  - Japanese: Original Japanese transcription
  - English-Direct: Direct English transcription (for English audio/JAV)

ADVANCED FEATURES:
  - Batch processing: Process multiple files sequentially
  - Model override: Choose specific Whisper models (large-v3, large-v2, turbo)
  - Opening credits: Add custom credit lines to subtitles
  - Async processing: Process multiple files simultaneously (experimental)
  - Translation: Translate subtitles to other languages (via whisperjav-translate)

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
   - Check install_log_v1.5.1.txt for specific error messages
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

3. "Model download stuck":
   - Large models (3GB) can take 10-20 minutes on slow connections
   - Check your internet speed: https://fast.com
   - Models are cached, so this only happens once

4. "Subtitles have errors or gibberish":
   - Try "balanced" mode for better quality
   - Use "conservative" sensitivity to reduce false positives
   - For non-Japanese audio, use "english-direct" output language

5. "Application crashes during processing":
   - Check logs in the GUI console
   - Ensure you have enough RAM (16 GB recommended)
   - Try processing one file at a time instead of batch
   - Report crashes at: https://github.com/meizhong986/WhisperJAV/issues

-------------------------------------------------------------------------------
UNINSTALLING
-------------------------------------------------------------------------------
To completely remove WhisperJAV v1.5.1:

1. Delete the installation directory:
   C:\Users\[YourName]\AppData\Local\WhisperJAV (or your custom location)

2. Delete the desktop shortcut:
   Desktop\WhisperJAV v1.5.1.lnk

3. (Optional) Delete cached models to free up ~3 GB:
   C:\Users\[YourName]\.cache\whisper

4. (Optional) Delete user configuration:
   [Install Directory]\whisperjav_config.json

NOTE: An automated uninstaller (uninstall_v1.5.1.bat) is included in the
      installation directory for your convenience.

-------------------------------------------------------------------------------
PERFORMANCE TIPS
-------------------------------------------------------------------------------
1. GPU Acceleration:
   - Ensure NVIDIA drivers are up to date
   - Close other GPU-intensive applications before processing
   - Verify CUDA is enabled by checking console output

2. Processing Speed:
   - "faster" mode: ~5 minutes per hour of video (GPU)
   - "fast" mode: ~7 minutes per hour of video (GPU)
   - "balanced" mode: ~10 minutes per hour of video (GPU)

3. Quality vs Speed:
   - For best accuracy: Use "balanced" mode with "balanced" sensitivity
   - For speed: Use "faster" mode with "conservative" sensitivity
   - For maximum detail: Use "balanced" mode with "aggressive" sensitivity

4. Batch Processing:
   - Process multiple files overnight using batch mode
   - Use "async processing" (experimental) for parallel execution

-------------------------------------------------------------------------------
ADVANCED USAGE
-------------------------------------------------------------------------------
COMMAND-LINE INTERFACE:
  WhisperJAV also includes a CLI for automation and scripting:

  # Basic transcription
  whisperjav video.mp4 --mode balanced --sensitivity aggressive

  # With translation
  whisperjav video.mp4 --translate --target-lang english

  # Batch processing
  whisperjav folder/*.mp4 --mode fast --workers 2

  Run "whisperjav --help" for full CLI documentation

TRANSLATION:
  Use whisperjav-translate for AI-powered subtitle translation:

  # Translate Japanese to English
  whisperjav-translate -i subtitles.srt --target english

  # Use custom instructions
  whisperjav-translate -i subtitles.srt --custom-gist [URL]

  See translation documentation for setup and API key configuration

-------------------------------------------------------------------------------
SUPPORT & COMMUNITY
-------------------------------------------------------------------------------
GitHub Repository: https://github.com/meizhong986/WhisperJAV
Issue Tracker: https://github.com/meizhong986/WhisperJAV/issues
Documentation: See GitHub README

For bugs, feature requests, or questions, please open an issue on GitHub
with detailed information about your system and the problem.

-------------------------------------------------------------------------------
VERSION INFORMATION
-------------------------------------------------------------------------------
WhisperJAV Version: 1.5.1
Release Date: January 2025
Installer Version: v1.5.1 (conda-constructor)

Key Changes in 1.5.1:
  - Complete PyWebView GUI takeover - modern, responsive web interface
  - Removed legacy Tkinter GUI components
  - Enhanced file management and progress tracking
  - Improved stability and error handling
  - Better WebView2 detection and installation guidance
  - CPU-only fallback for systems without NVIDIA GPUs

For full release notes, visit the GitHub repository.

-------------------------------------------------------------------------------
LICENSE
-------------------------------------------------------------------------------
WhisperJAV is released under the MIT License.
See LICENSE.txt in the installation directory for details.

Copyright (c) 2025 MeiZhong and contributors

===============================================================================
