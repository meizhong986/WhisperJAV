# WhisperJAV Windows Standalone Installation Guide

**Version:** 1.8.3
**Last Updated:** 2026-02-10
**Installer Type:** Standalone .exe (no admin required)

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Before You Install](#before-you-install)
3. [Installation Steps](#installation-steps)
4. [First-Run Experience](#first-run-experience)
5. [Verifying Your Installation](#verifying-your-installation)
6. [Upgrading from a Previous Version](#upgrading-from-a-previous-version)
7. [Uninstallation](#uninstallation)
8. [Troubleshooting](#troubleshooting)
9. [Silent/Unattended Installation](#silentunattended-installation)
10. [FAQ](#faq)

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| Operating System | Windows 10 (64-bit) or Windows 11 |
| RAM | 8 GB |
| Free Disk Space | 8 GB (installation + AI models + cache) |
| Internet | Required during installation for downloading dependencies |
| Runtime | Microsoft Edge WebView2 (installer will prompt if missing) |

### Recommended for Best Performance

| Component | Recommendation |
|-----------|----------------|
| GPU | NVIDIA GPU with 6+ GB VRAM (RTX 2060 or better) |
| GPU Driver | NVIDIA Driver 570+ (for CUDA 12.8 support) |
| RAM | 16 GB |
| Storage | SSD (significantly faster processing) |
| Internet | Broadband connection (for initial model download) |

### GPU Compatibility

WhisperJAV uses NVIDIA CUDA for GPU-accelerated transcription. The installer automatically detects your GPU and installs the appropriate version:

| Your NVIDIA Driver | CUDA Version Installed | Compatible GPUs |
|--------------------|------------------------|-----------------|
| 570 or newer | CUDA 12.8 (optimal) | RTX 20xx, 30xx, 40xx, 50xx series |
| 450 - 569 | CUDA 11.8 (universal) | All CUDA-capable NVIDIA GPUs |
| No NVIDIA GPU | CPU-only (slower) | Any system (6-10x slower than GPU) |

If you do not have an NVIDIA GPU, WhisperJAV will still work in CPU-only mode. Processing will be slower (approximately 30-60 minutes per hour of video, compared to 5-10 minutes with a GPU).

### Supported NVIDIA GPUs

- **GeForce:** GTX 1060+, RTX 2060+, RTX 3060+, RTX 4060+, RTX 5070+
- **Quadro / RTX Professional:** Quadro RTX, RTX A-series
- **Data Center:** T4, A10, A100, H100, B100

---

## Before You Install

### 1. Update Your NVIDIA Driver (Recommended)

For the best performance, update your NVIDIA driver to version 570 or newer:

1. Visit [nvidia.com/drivers](https://www.nvidia.com/drivers)
2. Select your GPU model
3. Download and install the latest "Game Ready" or "Studio" driver
4. Restart your computer after installation

You can check your current driver version by:
- Right-clicking on the desktop and selecting "NVIDIA Control Panel"
- Looking at the bottom-left corner for "Driver Version"

### 2. Ensure WebView2 is Installed

WhisperJAV uses Microsoft Edge WebView2 for its graphical interface. Most Windows 10 and 11 systems have this pre-installed. If you are unsure, the installer will check for you and provide a download link if needed.

To check manually:
1. Open "Add or remove programs" in Windows Settings
2. Search for "WebView2"
3. If "Microsoft Edge WebView2 Runtime" appears, you are ready

If not installed, download it from:
[https://go.microsoft.com/fwlink/p/?LinkId=2124703](https://go.microsoft.com/fwlink/p/?LinkId=2124703)

### 3. Free Up Disk Space

Ensure you have at least 8 GB of free space on the drive where WhisperJAV will be installed. The space breakdown is approximately:

| Component | Size |
|-----------|------|
| Base installation (Python + conda) | ~500 MB |
| PyTorch with CUDA | ~2.5 GB |
| Python dependencies | ~1.5 GB |
| AI models (downloaded on first use) | ~3 GB |
| **Total** | **~7.5 GB** |

### 4. Temporary Antivirus Exclusion (If Needed)

Some antivirus software may flag the installer or post-installation downloads. If you encounter issues:

1. Temporarily disable real-time scanning, or
2. Add an exclusion for `%LOCALAPPDATA%\WhisperJAV`

---

## Installation Steps

### Step 1: Download the Installer

Download `WhisperJAV-1.8.3-Windows-x86_64.exe` from the [GitHub Releases page](https://github.com/meizhong986/whisperjav/releases).

The installer file is approximately 150 MB.

### Step 2: Run the Installer

Double-click the downloaded `.exe` file. You do **not** need to run it as Administrator.

If Windows SmartScreen shows a warning:
1. Click "More info"
2. Click "Run anyway"

### Step 3: Accept the License Agreement

Read and accept the MIT license agreement to continue.

### Step 4: Choose Installation Type

- **Just Me (recommended):** Installs for the current user only. No admin privileges needed.
- **All Users:** Installs for all users on the computer. Requires admin privileges.

### Step 5: Choose Installation Location

The default location is:
```
C:\Users\[YourName]\AppData\Local\WhisperJAV
```

You can change this to any location. Keep these guidelines in mind:
- Avoid paths with spaces (e.g., "Program Files") if possible
- Avoid paths longer than 46 characters unless you have Long Paths enabled
- Avoid paths with special characters (accented letters, CJK characters)

### Step 6: Installation Options

The installer will show optional settings:
- **Add to PATH:** Enabled by default. Allows running `whisperjav` from the command line and enables future upgrades via `pip install -U`.
- **Create shortcuts:** Creates a desktop shortcut for easy access.

Click "Install" to begin.

### Step 7: Post-Installation (Automated)

After the base environment is extracted, a post-installation script runs automatically. This is the longest part of the installation. A console window will appear showing progress:

#### Phase 1: Preflight Checks (< 1 minute)
- Disk space verification (8 GB minimum)
- Network connectivity check (to PyPI)
- Visual C++ Redistributable check (auto-installed if missing)
- WebView2 runtime check (prompts download if missing)

#### Phase 2: GPU Detection (< 1 minute)
- Detects NVIDIA GPU and driver version
- Selects appropriate CUDA version

#### Phase 3: PyTorch Installation (3-5 minutes)
- Downloads and installs PyTorch with CUDA support (~2.5 GB)
- If no GPU is detected, you will be asked whether to install CPU-only PyTorch

#### Phase 3.5: Core Whisper Packages (2-3 minutes)
- Installs OpenAI Whisper, Stable-TS, FFmpeg-Python from GitHub
- These packages use git and may trigger Git timeout handling for users behind firewalls

#### Phase 4: Python Dependencies (3-5 minutes)
- Installs all remaining Python packages from requirements file
- Approximately 500 MB of downloads

#### Phase 5: WhisperJAV Application (< 1 minute)
- Installs the WhisperJAV application from the bundled wheel

#### Phase 5.3: Local LLM Translation (Optional, Interactive)
- You will be prompted to install a local Large Language Model for offline translation
- Type "Y" to install (recommended) or "N" to skip
- If you do not respond within 30 seconds, it defaults to "Y"
- This can be installed later if you skip it now

#### Phase 5.5-5.8: Launcher and Icon Setup (< 1 minute)
- Creates `WhisperJAV-GUI.exe` launcher in the installation directory
- Verifies icon files

After all phases complete, a desktop shortcut named "WhisperJAV v1.8.3" is created.

### Step 8: Complete

The installation summary will display:
- Installation directory
- Python version
- PyTorch version and CUDA status
- WebView2 status
- Installation time

Press Enter to close the installer window.

**Total installation time:** 10-20 minutes depending on internet speed and hardware.

---

## First-Run Experience

### Launching WhisperJAV

Double-click the **"WhisperJAV v1.8.3"** shortcut on your desktop. Alternatively, double-click `WhisperJAV-GUI.exe` in the installation directory.

### First Transcription: AI Model Download

The first time you process a video, WhisperJAV needs to download AI models. This is a one-time process:

| Model | Size | Download Time |
|-------|------|---------------|
| Whisper Large-v3 | ~3 GB | 5-10 minutes |

The download progress is shown in the GUI. After the initial download, models are cached locally and no further downloads are needed.

Models are stored in:
```
C:\Users\[YourName]\.cache\whisper\
```

### Basic Workflow

1. **Add files:** Click "Add File(s)" or drag and drop video/audio files
2. **Select mode:** Choose "Balanced" for best quality, "Faster" for speed
3. **Start processing:** Click "Start"
4. **Find your subtitles:** Output SRT files are saved next to your input files in an `_output` folder

---

## Verifying Your Installation

### Check GPU Acceleration

After installation, open a command prompt from the installation directory and run:

```cmd
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output for GPU systems:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4070
```

### Check WhisperJAV Version

```cmd
python -c "from whisperjav.__version__ import __version__; print(f'WhisperJAV {__version__}')"
```

Expected output:
```
WhisperJAV 1.8.3
```

### Check Installation Log

The installation log is located at:
```
[install_directory]\install_log_v1.8.3.txt
```

This file contains detailed information about every step of the installation process and is useful for diagnosing issues.

---

## Upgrading from a Previous Version

### Automatic Upgrade Detection

If you install WhisperJAV v1.8.3 to the same directory as a previous version, the installer will:

1. Detect the existing WhisperJAV installation
2. Ask if you want to replace it
3. If you confirm, it will cleanly remove the old version before installing

Your user configuration in `%APPDATA%\WhisperJAV` is preserved during upgrades.

### Manual Upgrade via pip

If you have PATH enabled (the default), you can upgrade without re-downloading the installer:

```cmd
pip install -U "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git"
```

### Preserving Configuration

User settings are stored in:
```
%APPDATA%\WhisperJAV\whisperjav_config.json
```

This file is NOT deleted during upgrades or reinstallation. If you are upgrading from a version older than v1.8.0, the installer will automatically migrate your configuration from the old location.

---

## Uninstallation

### Method 1: Windows Settings (Recommended)

1. Open Windows Settings > Apps > Apps & features
2. Search for "WhisperJAV"
3. Click "Uninstall"
4. Follow the prompts

### Method 2: Uninstall Script

Run the `uninstall_v1.8.3.bat` file located in the installation directory. This interactive script will:

1. Remove the desktop shortcut
2. Remove the Start Menu shortcut
3. Ask whether to delete user configuration files
4. Ask whether to delete cached AI models (~3 GB)
5. Remove the installation directory

### Method 3: Manual Removal

1. **Delete the installation directory:**
   ```
   %LOCALAPPDATA%\WhisperJAV
   ```
   (or your custom installation location)

2. **Delete the desktop shortcut:**
   ```
   %USERPROFILE%\Desktop\WhisperJAV v1.8.3.lnk
   ```

3. **(Optional) Delete cached AI models** to free approximately 3 GB:
   ```
   %USERPROFILE%\.cache\whisper
   ```

4. **(Optional) Delete user configuration:**
   ```
   %APPDATA%\WhisperJAV
   ```

5. **(Optional) Clean PATH entries:**
   If "Add to PATH" was enabled during installation, remove these entries from your user PATH:
   - `[install_dir]\Scripts`
   - `[install_dir]\Library\bin`

---

## Troubleshooting

### Installation Issues

#### "NVIDIA driver not found" or "No NVIDIA GPU detected"

**Cause:** No NVIDIA GPU installed, or drivers are not installed/outdated.

**Solution:**
- If you have an NVIDIA GPU, download and install the latest driver from [nvidia.com/drivers](https://www.nvidia.com/drivers)
- If you do not have an NVIDIA GPU, accept the CPU-only installation when prompted. Processing will be slower but fully functional.

#### "WebView2 runtime not detected"

**Cause:** Microsoft Edge WebView2 is not installed. This is required for the graphical interface.

**Solution:**
1. The installer will automatically open a download page
2. Download and install the "Evergreen Standalone Installer" from [Microsoft](https://go.microsoft.com/fwlink/p/?LinkId=2124703)
3. After installation, press Enter in the installer window to continue

#### "Network connection failed"

**Cause:** No internet connection or firewall blocking downloads.

**Solution:**
- Check your internet connection
- Temporarily disable VPN or proxy
- Check firewall settings: allow `python.exe`, `pip.exe`, `git.exe`, and `uv.exe` to access the internet
- If behind a corporate firewall, contact your IT department

#### "Git connection timeout" or "Failed to connect to github.com"

**Cause:** GitHub is blocked or slow (common behind the Great Firewall of China or with some VPN configurations).

**Solution:**
- The installer automatically detects this and configures extended Git timeouts
- Wait for the automatic retry (up to 3 attempts)
- If retries fail, try:
  - Switching to a different VPN endpoint
  - Using a GitHub mirror/proxy
  - Running the installer at a different time when network congestion is lower

#### "Out of disk space"

**Cause:** Less than 8 GB of free space on the installation drive.

**Solution:**
- Free up disk space and re-run the installer
- Choose a different installation drive with more space
- Note that AI models (downloaded on first use) also require approximately 3 GB

#### "Installation failed after retries"

**Solution:**
1. Check `install_log_v1.8.3.txt` in the installation directory for the specific error
2. Common causes:
   - Antivirus blocking downloads: add an exclusion for the installation directory
   - Network timeout: try again with a better connection
   - Corrupted download: delete the installation directory and start fresh
3. Try running the installer as Administrator (right-click > Run as administrator)

#### "Post-install script failed"

**Cause:** An error occurred during the automated package installation phase.

**Solution:**
1. Check `install_log_v1.8.3.txt` for the specific phase that failed
2. If it failed at PyTorch installation: check your GPU driver version
3. If it failed at dependency installation: check network connectivity
4. Re-run the installer; it will detect the existing partial installation and offer to replace it

### Runtime Issues

#### "GUI won't launch" or blank window appears

**Solution:**
1. Ensure WebView2 is installed (see above)
2. Try launching from the command line to see error messages:
   ```cmd
   cd %LOCALAPPDATA%\WhisperJAV
   python.exe -m whisperjav.webview_gui.main
   ```
3. Check if antivirus is blocking the application

#### Processing is very slow

**Solution:**
1. Check if GPU acceleration is enabled by looking at the console output when processing starts
2. If it says "CPU-only mode", your GPU is not being used
3. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
4. Close other GPU-intensive applications (games, video editors)
5. Use "Faster" mode for quicker results at the cost of some accuracy

#### "Model download stuck"

**Solution:**
- Large models (approximately 3 GB) take 5-20 minutes depending on your connection
- The download will resume if interrupted
- If stuck for more than 30 minutes, cancel and check your internet connection
- Models are cached in `%USERPROFILE%\.cache\whisper\`

#### Application crashes during processing

**Solution:**
1. Ensure you have enough RAM (16 GB recommended for large models)
2. Try processing one file at a time
3. Use a smaller Whisper model (e.g., "medium" instead of "large-v3")
4. Check the GUI console for error messages
5. Report persistent crashes at [GitHub Issues](https://github.com/meizhong986/whisperjav/issues)

---

## Silent/Unattended Installation

The installer supports command-line options for automated deployment:

```cmd
WhisperJAV-1.8.3-Windows-x86_64.exe /S /D=C:\WhisperJAV
```

### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `/S` | Silent mode (no GUI) | Off |
| `/Q` | Quiet mode (suppress console output) | Off |
| `/D=<path>` | Installation directory (must be last parameter) | `%LOCALAPPDATA%\WhisperJAV` |
| `/InstallationType=AllUsers` | Install for all users (requires admin) | JustMe |
| `/AddToPath=0` or `1` | Add to system PATH | 1 (on) |
| `/NoShortcuts=0` or `1` | Skip shortcut creation | 0 (create shortcuts) |
| `/InstallLocalLLM=0` or `1` | Install local LLM translation | Prompt |
| `/NoRegistry=0` or `1` | Skip registry entries | 0 |
| `/?` | Show help | -- |

### Examples

Install silently to a custom directory:
```cmd
cmd /C START /WAIT WhisperJAV-1.8.3-Windows-x86_64.exe /S /D=D:\WhisperJAV
```

Install silently, skip local LLM, no PATH modification:
```cmd
cmd /C START /WAIT WhisperJAV-1.8.3-Windows-x86_64.exe /S /AddToPath=0 /InstallLocalLLM=0
```

---

## FAQ

### Q: Do I need an NVIDIA GPU?

**A:** No. WhisperJAV works on any Windows 10/11 system. Without an NVIDIA GPU, it runs in CPU-only mode, which is approximately 6-10 times slower. For occasional use, CPU mode is perfectly fine. For regular use or long videos, an NVIDIA GPU is strongly recommended.

### Q: How much disk space do I need?

**A:** Approximately 8 GB total: about 4.5 GB for the installation and about 3 GB for AI models (downloaded on first use). If you also install the local LLM for translation, add approximately 4-8 GB more.

### Q: Can I install on a network drive?

**A:** This is not recommended. WhisperJAV requires fast disk access for temporary files and model loading. Install on a local SSD for the best experience.

### Q: Is my data sent to the internet?

**A:** No. All transcription happens locally on your computer. Internet access is only needed during installation (to download packages) and on first run (to download AI models). After that, WhisperJAV works completely offline. The only exception is if you use cloud translation providers (DeepSeek, Gemini, etc.), which require API keys and send subtitle text to those services.

### Q: Can I use this alongside another Python installation?

**A:** Yes. WhisperJAV installs its own isolated Python environment. It does not interfere with any existing Python installation on your system.

### Q: How do I update to a newer version?

**A:** Download and run the new installer. If you install to the same directory, the installer will detect the previous version and offer to replace it. Your user settings are preserved. Alternatively, if PATH was enabled, run: `pip install -U "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git"`

### Q: What video formats are supported?

**A:** WhisperJAV supports any format that FFmpeg can read, which includes virtually all common video and audio formats: MP4, MKV, AVI, MOV, WMV, FLV, WebM, MP3, WAV, FLAC, AAC, OGG, and many others.

### Q: Where are the subtitle files saved?

**A:** Subtitle files (SRT format) are saved next to your input video files in an `_output` subfolder. For example, if your video is at `D:\Videos\movie.mp4`, the subtitle will be at `D:\Videos\movie_output\movie.srt`.

### Q: The installer is stuck at "Extracting packages"

**A:** This phase extracts the conda packages and can take 2-5 minutes depending on your disk speed. It may appear frozen but is working. Check disk activity in Task Manager to confirm.

### Q: Can I move the installation after it is complete?

**A:** No. Moving the installation directory will break internal paths. If you need to change the location, uninstall and reinstall to the new location.

### Q: What is the "local LLM" option during installation?

**A:** This installs llama-cpp-python, which enables translating subtitles using a local AI model on your computer, without needing API keys or internet access. It requires additional disk space (4-8 GB) and works best with an NVIDIA GPU. You can skip this during installation and add it later if needed.

### Q: I am behind a corporate firewall / Great Firewall of China

**A:** The installer has built-in detection and handling for slow or blocked connections to GitHub. It will automatically configure extended timeouts and retry failed downloads. If installation still fails, try using a VPN or proxy that allows access to github.com and pypi.org.

---

## Technical Details

### Installation Directory Structure

After installation, the directory structure looks like this:

```
%LOCALAPPDATA%\WhisperJAV\
    python.exe                    # Python interpreter
    pythonw.exe                   # Python (no console window)
    WhisperJAV-GUI.exe            # GUI launcher
    whisperjav_icon.ico           # Application icon
    install_log_v1.8.3.txt        # Installation log
    uninstall_v1.8.3.bat          # Uninstall script
    Lib\                          # Python standard library
        site-packages\
            whisperjav\           # WhisperJAV application code
            torch\                # PyTorch
            ...
    Scripts\                      # Executable scripts
        whisperjav.exe            # CLI entry point
        whisperjav-gui.exe        # GUI entry point
        whisperjav-translate.exe  # Translation CLI
        pip.exe                   # Package manager
        ...
    Library\
        bin\                      # FFmpeg, git, and other tools
            ffmpeg.exe
            git.exe
            ...
```

### Log File Locations

| Log | Path | Purpose |
|-----|------|---------|
| Installation log | `[install_dir]\install_log_v1.8.3.txt` | Detailed installation progress |
| Failure marker | `[install_dir]\INSTALLATION_FAILED_v1.8.3.txt` | Created only if installation fails |
| Uninstall log | `%TEMP%\whisperjav_uninstall_v1.8.3.txt` | Uninstallation details |

### Network Requirements

During installation, the following domains must be accessible:

| Domain | Purpose |
|--------|---------|
| `pypi.org` | Python package downloads |
| `files.pythonhosted.org` | Python package files |
| `github.com` | Git-based package downloads |
| `download.pytorch.org` | PyTorch CUDA wheels |
| `huggingface.co` | AI model downloads (first run) |
| `aka.ms` | VC++ Redistributable (if needed) |
| `go.microsoft.com` | WebView2 runtime (if needed) |

---

## Support

If you encounter issues not covered in this guide:

1. Check the installation log at `install_log_v1.8.3.txt`
2. Search existing issues at [GitHub Issues](https://github.com/meizhong986/whisperjav/issues)
3. Open a new issue with:
   - Your Windows version
   - Your GPU model and driver version
   - The installation log file
   - A description of the problem
