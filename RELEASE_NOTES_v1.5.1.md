# WhisperJAV v1.5.1 Release Notes

## What's New from v1.4.5

Three major changes in this version:

1. **whisperjav-translate** - New experimental feature for AI-powered subtitle translation (translate Japanese subtitles to English or other languages)
2. **New GUI** - Completely redesigned window interface (old Tkinter interface removed)
3. **New Windows Installer** - Different installation system that automatically sets up everything

---

## 1. AI Subtitle Translation (whisperjav-translate)

This is a new experimental feature that translates your generated Japanese subtitles into other languages using AI.

### How to Use Translation

**From Command Line:**
```bash
whisperjav-translate -i your_subtitles.srt --target english
```

**From GUI:**
Translation features are accessible through the main WhisperJAV GUI interface.

**What You Need:**
- An API key from one of these services: DeepSeek, Gemini, Claude, GPT, or OpenRouter
- The AI service you choose will have costs (check their pricing)

**Translation Providers Supported:**
- DeepSeek (recommended - lower cost)
- Google Gemini
- Anthropic Claude
- OpenAI GPT
- OpenRouter

**Note:** This is experimental. Translation quality depends on the AI service you use.

---

## 2. New GUI Interface

The window interface has been completely redesigned. If you used version 1.4.5, you'll notice it looks different.

### How to Start the GUI

**After installation, you have two ways to open the program:**

1. **Desktop Shortcut**: Double-click the "WhisperJAV v1.5.1" icon on your desktop
2. **Direct Launch**: Go to your installation folder and double-click `WhisperJAV-GUI.exe`

**Default installation folder:**
`C:\Users\[YourName]\AppData\Local\WhisperJAV`

### What Changed in the Interface
- Uses your web browser's rendering engine (requires WebView2 - see below)
- File management and progress display improved
- Same three processing modes: Balanced, Fast, Faster

---

## 3. New Windows Installer

**Important:** This version uses a different installer than v1.4.5. The installation process has changed.

### How to Install

**Download:** `WhisperJAV-1.5.1-Windows-x86_64.exe` (250-300 MB)

**Steps:**
1. Download the installer file
2. Double-click to run it
3. Choose where to install (default location: `C:\Users\[YourName]\AppData\Local\WhisperJAV`)
4. Click through the installation wizard
5. **Wait 10-20 minutes** - the installer downloads and sets up all required components
6. **Important:** Black console windows will pop up during installation - **do not close them**. They close automatically when finished.

**What the Installer Does:**
- Installs Python and required libraries
- Downloads PyTorch (AI processing library) - approximately 2.5 GB
- Detects if you have an NVIDIA graphics card and installs appropriate version
- Creates a desktop shortcut
- Creates `WhisperJAV-GUI.exe` in the installation folder

**First Time You Process a Video:**
When you process your first video, the program downloads AI models (~3 GB). This is a one-time download that takes 5-10 minutes.

### Alternative: Python Installation (Technical Users)

If you manage your own Python environment:
```bash
pip install -U git+https://github.com/meizhong986/whisperjav.git
whisperjav-gui
```
Requires: Python 3.9-3.12, PyTorch with CUDA, WebView2

---

## System Requirements

### Your Computer Needs
- Windows 10 or 11 (64-bit)
- 8 GB RAM minimum (16 GB recommended)
- 8 GB free disk space
- Internet connection (for installation)

### Graphics Card
- **NVIDIA card**: 5-10 minutes to process 1 hour of video
- **No NVIDIA card**: 30-60 minutes to process 1 hour of video (works, just slower)

Update NVIDIA drivers if you have one: https://www.nvidia.com/drivers

### WebView2 (Required for GUI)
- Windows 11: Already installed
- Windows 10: Installer will prompt you to download if needed
- Download: https://go.microsoft.com/fwlink/p/?LinkId=2124703

### Installation Notes
- Windows Defender may show a warning - this is normal, safe to proceed
- No administrator rights needed
- Installs to your user folder by default

---

## Common Problems

**"NVIDIA driver not found" during installation:**
- Update graphics driver: https://www.nvidia.com/drivers
- Or continue without GPU (slower processing)

**"WebView2 not found" when starting GUI:**
- Download: https://go.microsoft.com/fwlink/p/?LinkId=2124703
- Install "Evergreen Standalone Installer"
- Restart WhisperJAV

**Desktop shortcut missing:**
- Go to installation folder: `C:\Users\[YourName]\AppData\Local\WhisperJAV`
- Run `create_desktop_shortcut_manual.bat`
- Or create shortcut to `WhisperJAV-GUI.exe` manually

**GUI won't start:**
- Check if WebView2 is installed
- Look at `install_log_v1.5.1.txt` in installation folder for error messages

**Processing is slow:**
- Normal without NVIDIA graphics card (6-10x slower)
- Use "Faster" mode instead of "Balanced" for speed

**Subtitle timing doesn't match video:**
- Some videos have variable audio bitrate causing this
- No fix - try a different video source if possible

---

## Upgrading from v1.4.5

**The installer is completely different.** Recommended approach:

### Steps to Upgrade

1. **Uninstall old version:**
   - Delete folder: `C:\Users\[YourName]\AppData\Local\WhisperJAV`
   - Delete old desktop shortcuts

2. **Install v1.5.1:**
   - Run `WhisperJAV-1.5.1-Windows-x86_64.exe`
   - Follow installation steps (see section 3 above)

3. **Your AI models are safe:**
   - Downloaded models (~3 GB) are stored separately at:
   - `C:\Users\[YourName]\.cache\whisper`
   - They won't be deleted and will work with v1.5.1

### Key Differences from v1.4.5

- **Different GUI**: Old Tkinter interface is gone. New WebView2-based interface.
- **Different launcher**: Now `WhisperJAV-GUI.exe` instead of old launcher files
- **Desktop shortcut location changed**: Points to new launcher
- **New feature**: whisperjav-translate for AI subtitle translation

### If You Used Python Installation (Not Installer)

```bash
pip install --upgrade git+https://github.com/meizhong986/whisperjav.git
whisperjav-gui
```

---

## Processing Modes (Unchanged from v1.4.5)

Three modes available:

| Mode | Speed | Accuracy | Graphics Memory | When to Use |
|------|-------|----------|-----------------|-------------|
| Balanced | Slowest | Best | 6 GB | Most accurate subtitles |
| Fast | Medium | Very Good | 4 GB | Recommended - good balance |
| Faster | Fastest | Good | 2-3 GB | Quick results, older graphics cards |

---

## Questions or Problems?

Report issues: https://github.com/meizhong986/WhisperJAV/issues

Version 1.5.1 - January 2025
