# WhisperJAV v1.7.1 Release Notes

**Release Date:** December 2024
**Status:** Stable

---

## About This Release

WhisperJAV v1.7.1 is the first stable release of the 1.7.x series. It includes all features introduced in the v1.7.0 beta releases, plus bug fixes and improvements based on beta testing feedback.

---

## What's New in v1.7.x

### Kotoba Model Support
Added support for the `kotoba-tech/kotoba-whisper-v2.0-faster` model, which is specifically tuned for Japanese speech. This model handles Japanese dialogue better than standard Whisper models, especially for conversational speech patterns common in JAV content.

Use it with:
```bash
whisperjav video.mp4 --mode kotoba-faster-whisper
```

### Ensemble Mode (Two-Pass Workflow)
New two-pass processing that runs your video through two different pipelines and merges the results. Different models catch different things, so combining them often gives better coverage.

```bash
whisperjav video.mp4 --ensemble --pass1-pipeline kotoba-faster-whisper --pass2-pipeline balanced
```

### Parameter Customization in GUI
The GUI now lets you tweak ASR parameters when using ensemble mode. You can adjust beam size, temperature, VAD thresholds, and other settings without touching config files.

### Debug Option
Added `--debug` flag for troubleshooting. Outputs detailed logs about what's happening at each step.

```bash
whisperjav video.mp4 --mode balanced --debug
```

### YAML-Driven Configuration (v4 Architecture)
New v4 configuration system that's YAML-driven and patchable without code changes. New models can be added by creating a YAML file without modifying Python code.

### Transformers Two-Pass Mode
Enhanced transformers pipeline with two-pass processing for improved accuracy.

---

## Changes Since v1.7.0-beta

### Bug Fixes
- Fixed issues reported during beta testing
- Improved stability in async processing
- Fixed edge cases in faster-whisper ASR module

### Improvements
- Enhanced transformers 2-pass mode
- Colab notebook improvements with new `fix_notebook.py` utility
- GUI refinements in app.js

---

## Known Limitations

### Translation is CLI-only
The translation feature (`whisperjav-translate`) is not yet integrated into the GUI. For now, run it separately:
```bash
whisperjav-translate -i subtitles.srt --provider deepseek
```

### VAD Toggle Incomplete
You can turn VAD on/off for the kotoba pipeline using `--no-vad`, but this toggle doesn't work for the other pipelines yet (balanced, fast, etc.). This will be addressed in a future update.

---

## Installation

### From Source (Recommended)
```bash
pip install git+https://github.com/meizhong986/whisperjav.git@main
```

### Windows Installer
Download and run: `WhisperJAV-1.7.1-Windows-x86_64.exe`

### Upgrading from Previous Versions
```bash
pip install -U git+https://github.com/meizhong986/whisperjav.git
```

---

## System Requirements

- Python 3.9 - 3.12 (Python 3.13+ is not compatible with openai-whisper)
- NVIDIA GPU with CUDA support (recommended) or Apple Silicon (MPS)
- FFmpeg installed and in system PATH
- 8GB+ available disk space for installation

---

## Feedback

If you encounter issues or have suggestions for improvements, please open an issue on GitHub:
https://github.com/meizhong986/whisperjav/issues
