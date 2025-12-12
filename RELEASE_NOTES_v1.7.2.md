# WhisperJAV v1.7.2 Release Notes

**Release Date:** December 2024
**Status:** Stable

---

## About This Release

v1.7.2 introduces a unified speech segmentation system and expands VAD options with three new providers. The ensemble mode UI has been reworked to make it easier to mix different configurations across passes.

---

## What's New

### Speech Segmentation System

Voice Activity Detection (VAD) providers are now grouped under a common "Speech Segmenter" abstraction. This makes it straightforward to swap between providers without changing other pipeline settings.

The segmenter groups detected speech into segments and clusters them by proximity (segments within 4 seconds are grouped together). This grouping helps Whisper process natural dialogue chunks rather than isolated fragments.

### New VAD Providers

Three new speech segmentation backends are now available:

| Provider | Notes |
|----------|-------|
| **TEN VAD** | Lightweight and fast. Works well with JAV content in testing. |
| **NVIDIA NeMo** | Two variants: NeMo Lite (standard) and NeMo Diarization (speaker-aware). Requires separate installation. |
| **Whisper Segmentation** | Uses Whisper itself for speech detection. Available in multiple sizes (tiny through medium). |

The existing Silero VAD (v3.1 and v4.0) remains available and is still the default.

### Ensemble Mode UI Rework

The ensemble mode tab now uses a grid layout where each pass has its own row of settings. You can independently configure:

- Pipeline type (Balanced, Fast, Faster, Fidelity, Transformers)
- Sensitivity level
- Scene detector
- Speech segmenter
- Model

This "mix and match" approach lets you combine different configurations. For example, run Pass 1 with Balanced + Silero, then Pass 2 with Transformers + TEN.

### Kotoba Bilingual Model

The `kotoba-tech/kotoba-whisper-bilingual-v1.0` model is now the default for Transformers pipelines. In testing, it handles mixed Japanese/English dialogue better than the Japanese-only models. Pairing it with TEN VAD has shown reasonable results for typical JAV content.

---

## Bug Fixes

### Speech Segmenter Selection Fixed

Fixed a bug where the speech segmenter selection in ensemble mode was being ignored. The GUI would show your selection (e.g., TEN or NeMo), but the backend always used Silero v4.0. Both passes now correctly use the selected segmenter.

### Silero Version Selection Fixed

Fixed an issue where selecting Silero v3.1 would still use v4.0. The version suffix is now properly passed through to the factory.

---

## GUI Changes

Several ensemble mode dropdown options have been adjusted:

| Change | Reason |
|--------|--------|
| Removed "None" from Pass 2 sensitivity | Sensitivity setting is required for consistent behavior |
| Removed "Small" and "Medium" models | These rarely improve results and increase confusion |
| Removed "NeMo Diarization" | Requires additional setup; NeMo Lite is sufficient for most cases |
| Removed "Whisper VAD (tiny)" | Quality too low to be useful |
| Renamed "Whisper VAD" to "Whisper Seg. Medium" | Clarifies which Whisper size is used |

---

## Installation

### From Source
```bash
pip install -U git+https://github.com/meizhong986/whisperjav.git
```

### Optional: TEN VAD
```bash
# See https://github.com/AgoraIO/TEN-VAD for installation
pip install ten-vad
```

### Optional: NVIDIA NeMo
```bash
pip install nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main
```

---

## CLI Examples

```bash
# Use TEN VAD with balanced pipeline
whisperjav video.mp4 --mode balanced --speech-segmenter ten

# Ensemble with different segmenters per pass
whisperjav video.mp4 --ensemble \
    --pass1-pipeline balanced --pass1-speech-segmenter silero \
    --pass2-pipeline transformers --pass2-speech-segmenter ten

# Skip VAD entirely (process full audio)
whisperjav video.mp4 --mode balanced --speech-segmenter none
```

---

## Known Issues

- NeMo models download on first use (~500MB). This can take a few minutes.
- TEN VAD requires a separate pip install (not bundled with WhisperJAV).
- Console text selection in GUI can cause brief UI lag on very long outputs.

---

## Feedback

Report issues or suggestions at:
https://github.com/meizhong986/whisperjav/issues
