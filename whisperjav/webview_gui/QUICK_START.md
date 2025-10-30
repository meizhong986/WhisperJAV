# WhisperJAV PyWebView GUI - Quick Start Guide

**Version:** Phase 4 - Complete Integration
**Date:** 2025-10-30

---

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# Install in development mode
pip install -e .

# Install PyWebView (if not already installed)
pip install pywebview
```

### Verify Installation

```bash
python -c "import webview; print('PyWebView OK')"
python -c "import whisperjav; print('WhisperJAV OK')"
```

---

## Running the GUI

### Standard Launch

```bash
python -m whisperjav.webview_gui.main
```

### Debug Mode (with Chrome DevTools)

```bash
# Windows
set WHISPERJAV_DEBUG=1
python -m whisperjav.webview_gui.main

# macOS/Linux
export WHISPERJAV_DEBUG=1
python -m whisperjav.webview_gui.main
```

### Console Script (if installed)

```bash
whisperjav-gui-web
```

---

## Basic Usage

### 1. Add Video Files

**Option A: Add Files**
- Click "Add File(s)" button
- Select one or more video files
- Click "Open"

**Option B: Add Folder**
- Click "Add Folder" button
- Select folder containing videos
- Click "Select Folder"

### 2. Configure Settings

**Transcription Mode Tab:**
- **Mode:** balanced (accuracy) / fast / faster (speed)
- **Sensitivity:** conservative / balanced / aggressive
- **Output language:** japanese / english-direct

**Advanced Options Tab:**
- **Model override:** Choose specific Whisper model
- **Async processing:** Enable parallel processing
- **Max workers:** Number of parallel processes
- **Opening credit:** Add custom opening subtitle
- **Keep temp files:** Retain temporary files for debugging
- **Temp directory:** Custom temporary directory
- **Verbosity:** quiet / summary / normal / verbose

### 3. Set Output Directory

- Default output: `C:\Users\[You]\Documents\WhisperJAV\output`
- Click "Browse" to change
- Click "Open" to view output folder in file explorer

### 4. Start Processing

- Click "Start" button
- Monitor progress in console output
- Wait for completion (status shows "Completed")
- Cancel anytime with "Cancel" button

### 5. Review Results

- Output SRT files saved to output directory
- Click "Open" next to Output field to view files
- Console shows full processing log

---

## Keyboard Shortcuts

### Tab Navigation

- **Arrow Left/Right:** Switch between tabs
- **Home:** First tab
- **End:** Last tab

### File List

- **Arrow Up/Down:** Navigate files
- **Shift + Arrow:** Range selection
- **Ctrl + Click:** Multi-select
- **Delete/Backspace:** Remove selected files

---

## Recommended Settings

### For Best Quality

```
Mode: balanced
Sensitivity: balanced
Language: japanese
Model override: large-v3 (if GPU available)
Verbosity: normal
```

### For Speed

```
Mode: faster
Sensitivity: conservative
Language: japanese
Async processing: enabled
Max workers: 4 (adjust based on CPU cores)
Verbosity: summary
```

### For Noisy Audio

```
Mode: balanced
Sensitivity: aggressive
Language: japanese
Verbosity: verbose
```

---

## Troubleshooting

### GUI Won't Start

**Error:** "Asset file not found"

**Solution:**
```bash
# Ensure you're in the repository root
cd C:\BIN\git\WhisperJav_V1_Minami_Edition

# Run from source
python -m whisperjav.webview_gui.main
```

### File Dialog Doesn't Open

**Error:** File dialog not appearing

**Solution:**
- Ensure PyWebView is properly installed
- Check if window is active (click on window first)
- Try debug mode to see error messages

### Process Won't Start

**Error:** "No Files Selected"

**Solution:**
- Add at least one file or folder before clicking Start

**Error:** "Process already running"

**Solution:**
- Wait for current process to complete
- Or click "Cancel" to stop current process

### Logs Not Streaming

**Error:** Console not updating

**Solution:**
- Check browser DevTools console for JavaScript errors (debug mode)
- Restart GUI
- Verify API connection: check for "PyWebView bridge connected" message

### Process Fails Immediately

**Error:** "Process exited with code [N]"

**Solution:**
- Check console output for error details
- Verify video file is valid (not corrupted)
- Ensure FFmpeg is installed and in PATH
- Check if required models are downloaded

---

## File Formats Supported

### Video Files

- **MP4** (H.264, H.265)
- **MKV** (Matroska)
- **AVI**
- **MOV** (QuickTime)
- **FLV** (Flash Video)
- **WMV** (Windows Media)
- **WEBM** (VP8, VP9)

### Output Format

- **SRT** (SubRip Subtitle)
- UTF-8 encoded
- Compatible with all major media players

---

## System Requirements

### Minimum Requirements

- **OS:** Windows 10/11, macOS 10.14+, Linux (GTK)
- **Python:** 3.9 - 3.12
- **RAM:** 4 GB
- **Disk:** 2 GB free space (for models)

### Recommended Requirements

- **OS:** Windows 11, macOS 12+, Ubuntu 22.04+
- **Python:** 3.10 or 3.11
- **RAM:** 8 GB
- **GPU:** NVIDIA GPU with CUDA support (for faster processing)
- **Disk:** 5 GB free space

### Dependencies

- **PyWebView:** Web-based GUI framework
- **FFmpeg:** Audio/video processing (must be in PATH)
- **OpenAI Whisper:** ASR engine
- **Faster-Whisper:** Optimized ASR backend (optional)
- **PyTorch:** Machine learning framework

---

## Performance Tips

### For Faster Processing

1. **Use faster mode:** Trades some accuracy for speed
2. **Enable GPU:** Ensure CUDA is installed (NVIDIA GPUs)
3. **Enable async processing:** Process multiple files in parallel
4. **Use faster-whisper backend:** Automatically used in "faster" mode
5. **Reduce sensitivity:** Use "conservative" sensitivity
6. **Use smaller model:** Use "turbo" model instead of "large-v3"

### For Better Accuracy

1. **Use balanced mode:** Full preprocessing pipeline
2. **Use aggressive sensitivity:** Captures more detail
3. **Use larger model:** Override with "large-v3" model
4. **Use verbose output:** See detailed processing logs
5. **Process one file at a time:** Disable async processing

### For GPU Users

```
Mode: faster
Model override: large-v3
Async processing: enabled
Max workers: 2-4 (depends on VRAM)
```

### For CPU-Only Users

```
Mode: faster
Model override: turbo (smaller, faster)
Async processing: disabled (unless multi-core CPU)
Keep expectations realistic: ~2-5x slower than GPU
```

---

## Output File Naming

**Pattern:** `{input_name}.srt`

**Examples:**
- Input: `video.mp4` → Output: `video.srt`
- Input: `JAV-001.mkv` → Output: `JAV-001.srt`

**Location:** `{output_dir}/{input_name}.srt`

---

## Logging

### Console Output

- Real-time streaming (100ms updates)
- Color-coded messages:
  - **Info:** Regular progress updates
  - **Success:** Completion messages (green ✓)
  - **Warning:** Non-critical issues (yellow ⚠)
  - **Error:** Critical failures (red ✗)

### File Logging

- Optional file logging: `whisperjav.log`
- Enable via CLI: `--log-file whisperjav.log`
- Not available in GUI mode (console only)

---

## Advanced Features

### Opening Credit

Add custom subtitle at the beginning:

```
Opening credit: Produced by Team WhisperJAV
```

Result: First subtitle line shows credit for 5 seconds.

### Custom Temp Directory

Specify where temporary files are stored:

```
Temp dir: D:\Temp\WhisperJAV
Keep temp files: ✓
```

Use for debugging or inspecting intermediate files.

### Model Override

Force specific Whisper model:

- **large-v3:** Best accuracy, slowest (default)
- **large-v2:** Good accuracy, slower
- **turbo:** Faster, acceptable accuracy

### Async Processing

Process multiple files in parallel:

```
Async processing: ✓
Max workers: 4
```

**Note:** Uses more RAM and CPU. Adjust workers based on system resources.

---

## FAQ

### Q: Can I process folders?

**A:** Yes! Click "Add Folder" to add a folder. All video files in the folder will be processed.

### Q: Can I process multiple files at once?

**A:** Yes! Add multiple files or use async processing to process in parallel.

### Q: How long does processing take?

**A:** Depends on mode, hardware, and video length:
- **GPU (CUDA):** ~1-3x realtime (1 hour video = 20-60 minutes)
- **CPU-only:** ~5-10x realtime (1 hour video = 5-10 hours)

### Q: Can I cancel processing?

**A:** Yes! Click the "Cancel" button anytime. Process terminates within 5 seconds.

### Q: Where are models stored?

**A:** `~/.cache/whisper/` (default PyTorch cache location)

### Q: Do I need an internet connection?

**A:** Only for first run (to download models). Offline afterward.

### Q: Can I use this for non-JAV content?

**A:** Yes! WhisperJAV works with any Japanese audio content. Name is just branding.

### Q: Does it support other languages?

**A:** Whisper supports 90+ languages, but this tool is optimized for Japanese. Other languages may work but are untested.

---

## Support

### Documentation

- **Main README:** `README.md`
- **CLI Guide:** `CLAUDE.md`
- **Phase 4 Integration Guide:** `PHASE4_INTEGRATION_GUIDE.md`
- **Testing Checklist:** `PHASE4_TESTING_CHECKLIST.md`

### Issues

- **GitHub Issues:** https://github.com/meizhong986/whisperjav/issues
- **Discussions:** https://github.com/meizhong986/whisperjav/discussions

### Contact

- **Author:** meizhong986
- **Repository:** https://github.com/meizhong986/whisperjav

---

## License

WhisperJAV is open-source software. See `LICENSE` file for details.

---

## Credits

- **OpenAI Whisper:** ASR engine
- **PyWebView:** Cross-platform GUI framework
- **Faster-Whisper:** Optimized inference backend
- **Stable-ts:** Japanese text regrouping

---

**End of Quick Start Guide**

Enjoy using WhisperJAV! 🎥🎤📝
