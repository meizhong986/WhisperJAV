# WhisperJAV v1.7.3 Release Notes

**Release Date:** December 2024
**Status:** Stable

---

## About This Release

v1.7.3 adds a Speech Enhancement module for preprocessing audio before transcription. This release also includes stability fixes for ensemble mode and addresses several bugs reported since v1.7.2.

---

## What's New

### Speech Enhancement Module

A new optional preprocessing stage that can clean up audio before it reaches Whisper. Three backends are available:

| Backend | What It Does | When To Use |
|---------|--------------|-------------|
| **ZipEnhancer** | Removes background noise | General denoising, low VRAM systems |
| **ClearVoice** | Speech denoising (multiple models) | Heavier noise, if you have VRAM headroom |
| **BS-RoFormer** | Separates vocals from music/background | Audio with loud background music |

Enhancement is disabled by default. Enable it via the GUI dropdown or CLI flag when needed.

### When Enhancement Helps (and When It Doesn't)

Speech enhancers modify the audio waveform before Whisper processes it. This changes the mel spectrogram that Whisper was trained on, which can help or hurt depending on the source material.

**Good candidates for enhancement:**
- Recordings with constant background music drowning out dialogue
- Audio with heavy static, hiss, or ambient noise
- Videos where vocals compete with sound effects

**Skip enhancement when:**
- The audio is already reasonably clear
- Your current results are acceptable without it
- You're unsure (test on a short clip first)

If transcription is working fine, adding enhancement won't improve it and may introduce artifacts. Test before committing to a full video.

### Memory Note

ZipEnhancer (16kHz) is lightweight and works on most systems. The 48kHz ClearVoice model requires more VRAM and can cause issues on 8GB GPUs. Stick with 16kHz models if you're memory-constrained.

---

## Bug Fixes

### Direct-to-English Translation Fixed

The `--subs-language direct-to-english` CLI switch now works correctly. Previously, it would still output Japanese text. The ASR modules now honor the runtime task override.

### Transformers Default Model Fixed

When using Transformers mode without specifying a model, the default is now `kotoba-tech/kotoba-whisper-bilingual-v1.0` (as shown in the GUI) instead of `kotoba-v2.2`. This applies to both GUI and CLI.

### Ensemble Merge Strategy Default Changed

The default merge strategy is now `pass1_primary` (Pass 1 as base, fill gaps from Pass 2) instead of `smart_merge`. This produces more predictable results for typical two-pass workflows.

### NeMo Lazy Loading Fixed

NeMo toolkit no longer initializes when the GUI starts. The heavy Megatron initialization only happens when you actually select a NeMo segmenter.

---

## Stability Fixes

Several crashes in ensemble mode have been addressed:

| Issue | Fix |
|-------|-----|
| BrokenProcessPool | Fixed process pool corruption during multi-file batch processing |
| CUDA Context Corruption | Fixed cascade failure when speech enhancement hit GPU errors |
| ctranslate2 Destructor | Applied workaround to prevent crash during Python shutdown |

VAD parameters have been tuned across all sensitivity profiles. Detection thresholds and padding values were adjusted based on testing.

---

## Installation

### Upgrading from v1.7.2
```bash
pip install -U git+https://github.com/meizhong986/whisperjav.git
```

### Fresh Install
```bash
pip install git+https://github.com/meizhong986/whisperjav.git
```

### Optional: BS-RoFormer
```bash
pip install bs-roformer-infer
```

ZipEnhancer and ClearVoice are included in the default installation.

---

## CLI Examples

```bash
# Transformers mode now uses kotoba-bilingual by default
whisperjav video.mp4 --mode transformers

# Enable ZipEnhancer for noisy audio
whisperjav video.mp4 --mode balanced --pass1-speech-enhancer zipenhancer

# Ensemble with enhancement on Pass 1 only
whisperjav video.mp4 --ensemble \
    --pass1-pipeline balanced --pass1-speech-enhancer zipenhancer \
    --pass2-pipeline transformers

# Direct-to-English translation (now works)
whisperjav video.mp4 --mode balanced --subs-language direct-to-english

# BS-RoFormer for vocal isolation (loud background music)
whisperjav video.mp4 --mode balanced --pass1-speech-enhancer bs-roformer
```

---

## Known Issues

- The 48kHz ClearVoice model is hidden in the GUI due to VRAM fragmentation on 8GB GPUs. Available via CLI if you have sufficient VRAM.
- Speech enhancement adds processing time (roughly 20-40% depending on backend).
- Enhancement is applied per-scene after scene detection. Very short scenes may not benefit.

---

## Feedback

Report issues or suggestions at:
https://github.com/meizhong986/whisperjav/issues
