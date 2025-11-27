# WhisperJAV v1.7.0-beta Release Notes

**Release Date:** November 2024
**Status:** Beta

---

## About This Release

This is a beta release. We're putting it out to get feedback from users before the stable release. If you run into issues or have suggestions, please let us know.

---

## What's New

### Kotoba Model Support
Added support for the `kotoba-tech/kotoba-whisper-v2.0-faster` model, which is specifically tuned for Japanese speech. In our testing, it handles Japanese dialogue better than the standard Whisper models, especially for conversational speech patterns common in JAV content.

Use it with:
```
whisperjav video.mp4 --mode kotoba-faster-whisper
```

### Ensemble Mode (Two-Pass Workflow)
New two-pass processing that runs your video through two different pipelines and merges the results. The idea is simple: different models catch different things, so combining them often gives better coverage.

```
whisperjav video.mp4 --ensemble --pass1-pipeline kotoba-faster-whisper --pass2-pipeline balanced
```

### Parameter Customization in GUI
The GUI now lets you tweak ASR parameters when using ensemble mode. You can adjust things like beam size, temperature, VAD thresholds, and other settings without touching config files.

### Debug Option
Added `--debug` flag for troubleshooting. Outputs detailed logs about what's happening at each step.

```
whisperjav video.mp4 --mode balanced --debug
```

---

## Known Limitations

### Translation is CLI-only
The translation feature (`whisperjav-translate`) is not yet integrated into the GUI. For now, run it separately:
```
whisperjav-translate -i subtitles.srt --provider deepseek
```

### VAD Toggle Incomplete
You can turn VAD on/off for the kotoba pipeline using `--no-vad`, but this toggle doesn't work for the other pipelines yet (balanced, fast, etc.). We'll add this in a future update.

---

## Upgrading

```bash
pip install -U git+https://github.com/meizhong986/whisperjav.git
```

Or use the Windows installer: `WhisperJAV-1.7.0-beta-Windows-x86_64.exe`

---

## Feedback

This is a beta. Things might break. If they do, or if you have ideas for improvements, open an issue on GitHub.
