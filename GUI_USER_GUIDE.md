# WhisperJAV GUI - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Getting Started](#getting-started)
5. [User Interface Overview](#user-interface-overview)
6. [Processing Workflow](#processing-workflow)
7. [Advanced Options](#advanced-options)
8. [Keyboard Shortcuts](#keyboard-shortcuts)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## Introduction

WhisperJAV GUI is a desktop application for generating high-quality Japanese subtitles from video files. It uses OpenAI's Whisper speech recognition with specialized enhancements for Japanese Adult Videos (JAV), including:

- **Multi-mode processing** (Balanced, Fast, Faster) for different speed/accuracy tradeoffs
- **Scene detection** to improve transcription accuracy
- **Voice Activity Detection (VAD)** to filter out background noise
- **Japanese language optimization** with specialized regrouping rules
- **Batch processing** to handle multiple files at once
- **Real-time progress monitoring** with live log output

---

## System Requirements

### Minimum Requirements
- **Operating System:** Windows 10 or later (64-bit)
- **RAM:** 8GB minimum, 16GB recommended
- **Disk Space:** 5GB free space for application and temporary files
- **GPU:** NVIDIA GPU with CUDA support recommended (CPU-only mode available but slower)

### Required Software
- **WebView2 Runtime** (automatically detected, download link provided if missing)
- No Python installation required (standalone executable)

### Recommended for Best Performance
- **GPU:** NVIDIA RTX 3060 or better with 8GB+ VRAM
- **RAM:** 16GB or more
- **Storage:** SSD for faster file I/O

---

## Installation

### Option 1: Standalone Executable (Recommended)

1. **Download** the latest release from the GitHub repository
2. **Extract** the ZIP archive to a folder of your choice (e.g., `C:\Program Files\WhisperJAV`)
3. **Locate** `whisperjav-gui-web.exe` in the extracted folder
4. **Run** the executable

### Option 2: Install from Source

If you prefer to run from source:

```bash
# Clone the repository
git clone https://github.com/meizhong986/WhisperJAV.git
cd WhisperJAV

# Install in development mode
pip install -e .[gui]

# Run the GUI
whisperjav-gui-web
```

### WebView2 Installation

If you see an error about missing WebView2 Runtime:

1. Click the download link in the error dialog, or visit:
   https://go.microsoft.com/fwlink/p/?LinkId=2124703
2. Download and install the **Evergreen Standalone Installer**
3. Restart WhisperJAV GUI

---

## Getting Started

### First Launch

1. **Double-click** `whisperjav-gui-web.exe` to launch the application
2. The main window will open with four main sections:
   - **Source:** Add files or folders to process
   - **Destination:** Choose where to save output
   - **Options:** Configure transcription settings (tabs)
   - **Console:** View real-time progress and logs

### Quick Start: Process Your First Video

1. **Add Files:**
   - Click **"Add File(s)"** button
   - Select one or more video files (.mp4, .mkv, .avi, etc.)
   - Or click **"Add Folder"** to process all videos in a directory

2. **Choose Output Location:**
   - Click **"Browse"** next to the Output field
   - Select where you want the subtitle files saved
   - Or use the default location

3. **Select Processing Mode:**
   - **Balanced** (recommended): Best accuracy, moderate speed
   - **Fast:** Good balance, uses scene detection
   - **Faster:** Fastest, minimal preprocessing

4. **Start Processing:**
   - Click the **"Start"** button
   - Monitor progress in the Console section
   - Wait for completion (time varies by video length and mode)

5. **Access Results:**
   - Click **"Open"** next to Output to view the folder
   - Subtitle files will have the same name as your video with `.srt` extension

---

## User Interface Overview

### Source Section
- **File List:** Displays selected files and folders
  - Click to select items
  - Ctrl+Click for multi-select
  - Shift+Click for range select
- **Add File(s):** Opens file picker for video files
- **Add Folder:** Opens folder picker to process all videos in directory
- **Remove Selected:** Removes selected items from list
- **Clear:** Removes all items from list

### Destination Section
- **Output:** Path where subtitle files will be saved
- **Browse:** Select a different output directory
- **Open:** Open the output folder in File Explorer

### Transcription Mode Tab
- **Mode:**
  - **Balanced:** Full preprocessing pipeline with scene detection and VAD (most accurate)
  - **Fast:** Scene detection only (good balance)
  - **Faster:** Direct transcription (fastest, least accurate)

- **Sensitivity:**
  - **Conservative:** Higher thresholds, fewer false positives, may miss quiet speech
  - **Balanced:** Default settings (recommended)
  - **Aggressive:** Lower thresholds, captures more detail but may include noise

- **Output Language:**
  - **Japanese:** Transcribe in Japanese (default)
  - **English-direct:** Transcribe directly to English (experimental)

### Transcription Advanced Options Tab
- **Adaptive Features (WIP):** Not yet implemented, disabled
- **Verbosity:** Control amount of console output (quiet/summary/normal/verbose)
- **Model Override:** Force a specific Whisper model (large-v3, large-v2, turbo)
- **Async Processing:** Process multiple files in parallel
- **Max Workers:** Number of parallel processes (1-16)
- **Opening Credit:** Add custom text to the beginning of subtitles
- **Keep Temp Files:** Preserve temporary audio files for debugging
- **Temp Dir:** Custom location for temporary files

### Run Section
- **Progress Bar:** Shows processing status
  - Indeterminate animation during processing
  - Fills to 100% on completion
- **Status Label:** Current state (Idle, Running, Completed, Error)
- **Start Button:** Begin processing
- **Cancel Button:** Stop current process

### Console Section
- **Real-time Logs:** View processing output
- **Color-coded Messages:**
  - Blue: Informational
  - Green: Success
  - Orange: Warning
  - Red: Error
- **Clear Button:** Clear console output

---

## Processing Workflow

### Understanding Processing Modes

#### Balanced Mode (Recommended)
**Best for:** High-quality transcription, JAV content with variable audio quality

**Pipeline:**
1. Audio extraction from video
2. Scene detection (splits audio into segments)
3. Voice Activity Detection (removes silence and noise)
4. Whisper transcription
5. Japanese regrouping (natural dialogue breaks)
6. Hallucination removal (filters AI artifacts)
7. Repetition cleaning
8. SRT formatting

**Pros:** Most accurate, best for complex audio
**Cons:** Slowest processing time

#### Fast Mode
**Best for:** Good-quality source videos, batch processing

**Pipeline:**
1. Audio extraction
2. Scene detection
3. Whisper transcription
4. Post-processing
5. SRT formatting

**Pros:** Good balance of speed and accuracy
**Cons:** May miss quiet speech

#### Faster Mode
**Best for:** Clean audio, quick previews, testing

**Pipeline:**
1. Audio extraction
2. Faster-Whisper transcription
3. Minimal post-processing
4. SRT formatting

**Pros:** Fastest processing
**Cons:** Least accurate, may include hallucinations

### Understanding Sensitivity

- **Conservative:** Use for clean audio with clear speech
- **Balanced:** Use for typical JAV content (default)
- **Aggressive:** Use for quiet speech, whispering, or distant voices

### Batch Processing Best Practices

1. **Group Similar Files:** Process files with similar audio quality together
2. **Use Async Processing:** Enable for 3+ files on multi-core systems
3. **Monitor Resources:** Watch CPU/GPU usage, adjust max workers if needed
4. **Test First:** Process one file with each mode to find best settings
5. **Check Results:** Review subtitles, adjust settings if needed

---

## Advanced Options

### Model Override

Force a specific Whisper model:
- **large-v3:** Newest model, best accuracy (default)
- **large-v2:** Previous generation, good compatibility
- **turbo:** Faster but less accurate

### Async Processing

Process multiple files simultaneously:
- **Benefits:** Faster total processing time for batches
- **Requirements:** Multi-core CPU, sufficient RAM
- **Recommended Workers:**
  - 4-core CPU: 2 workers
  - 8-core CPU: 3-4 workers
  - 12+ core CPU: 4-6 workers

### Opening Credit

Add custom text at the start of subtitles:
```
Example: "Produced by Your Name"
         "Translated by Translation Team"
```

### Temp Directory

By default, temporary files are stored in the system temp folder and deleted after processing.

**When to customize:**
- Insufficient space on system drive
- Faster storage available (SSD)
- Debugging audio extraction issues

**Enable "Keep Temp Files"** to preserve intermediate files for troubleshooting.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **Ctrl+O** | Open file selection dialog |
| **Ctrl+R** | Start processing (if files selected) |
| **Escape** | Cancel running process |
| **F1** | Show About dialog |
| **F5** | Refresh (warns if process running) |
| **Delete** | Remove selected files from list |
| **Ctrl+Click** | Multi-select files |
| **Shift+Click** | Range-select files |
| **Arrow Keys** | Navigate between tabs |

---

## Troubleshooting

### Application Won't Start

**Error: WebView2 Required**
- **Cause:** Missing Microsoft Edge WebView2 Runtime
- **Solution:** Download and install from the link in the error dialog

**Error: Asset file not found**
- **Cause:** Incomplete installation or corrupted files
- **Solution:** Re-extract the archive, ensure all files are present

### Processing Errors

**Process exits with code 1**
- **Cause:** Invalid video file or codec not supported
- **Solution:** Verify video plays correctly, try converting to MP4 with H.264 codec

**Out of memory error**
- **Cause:** Insufficient RAM for large video files
- **Solution:** Close other applications, reduce max workers, process files individually

**CUDA error (GPU)**
- **Cause:** GPU driver issue or insufficient VRAM
- **Solution:** Update GPU drivers, fallback to CPU mode by setting environment variable:
  ```
  set CUDA_VISIBLE_DEVICES=-1
  ```

### Quality Issues

**Missing dialogue**
- **Cause:** Sensitivity too conservative
- **Solution:** Try "Aggressive" sensitivity, or use "Balanced" mode

**Too many hallucinations (repeated phrases)**
- **Cause:** Noisy audio or aggressive settings
- **Solution:** Try "Conservative" sensitivity, or "Balanced" mode

**Incorrect timing**
- **Cause:** Scene detection too aggressive
- **Solution:** Try "Fast" or "Faster" mode

### Performance Issues

**Slow processing**
- **Cause:** CPU mode, large files, or "Balanced" mode
- **Solution:**
  - Ensure GPU is being used (check console for CUDA messages)
  - Try "Fast" or "Faster" mode
  - Update GPU drivers

**Application freezes**
- **Cause:** Processing very large files
- **Solution:** Close and restart, process files individually

---

## FAQ

### General Questions

**Q: What video formats are supported?**
A: Any format supported by FFmpeg (.mp4, .mkv, .avi, .mov, .wmv, etc.)

**Q: Do I need an internet connection?**
A: No, all processing is done locally on your computer.

**Q: How long does processing take?**
A: Varies by mode and hardware:
- Balanced mode: ~1-2x video length (GPU), 3-5x (CPU)
- Fast mode: ~0.5-1x video length (GPU), 2-3x (CPU)
- Faster mode: ~0.3-0.5x video length (GPU), 1-2x (CPU)

**Q: Can I process multiple files at once?**
A: Yes, add multiple files and enable "Async processing" in Advanced Options.

**Q: What subtitle format is generated?**
A: SRT (SubRip) format, compatible with most video players.

### Technical Questions

**Q: Can I use this with other languages?**
A: Yes, but optimization is for Japanese. For other languages, use standard Whisper.

**Q: Does it support GPU acceleration?**
A: Yes, NVIDIA GPUs with CUDA are automatically detected and used.

**Q: Can I run this on Mac or Linux?**
A: Currently Windows-only. For other platforms, install from source with Python.

**Q: How much disk space do I need?**
A: Temporary files use ~500MB per hour of video. Ensure sufficient free space.

**Q: Are the subtitles perfect?**
A: No, AI transcription is not 100% accurate. Review and edit as needed.

### Troubleshooting Questions

**Q: Why is the output in English instead of Japanese?**
A: Check "Output language" setting in Transcription Mode tab.

**Q: Why are there no subtitles for some scenes?**
A: Voice Activity Detection may have filtered them. Try "Aggressive" sensitivity.

**Q: Can I stop and resume processing?**
A: No, processing must complete or be cancelled. Results are saved per file in batch mode.

**Q: Where are my subtitle files?**
A: In the output directory specified in the Destination section. Default is usually `Documents\WhisperJAV\output`.

---

## Need More Help?

- **GitHub Issues:** https://github.com/meizhong986/WhisperJAV/issues
- **Documentation:** Check repository README.md
- **Logs:** Enable "verbose" mode for detailed diagnostic information

---

**Version:** 1.4.5
**Last Updated:** 2025
**License:** MIT
