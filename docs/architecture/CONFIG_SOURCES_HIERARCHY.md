# WhisperJAV Configuration Sources Hierarchy

## Overview (v1.8.9+)

**As of v1.8.9, Pydantic presets are the SINGLE SOURCE OF TRUTH for all pipeline parameters.**

`asr_config.json` has been stripped to contain only `version` and `ui_preferences` (console verbosity). All ASR, VAD, scene detection, decoder, transcriber, and engine parameters come exclusively from Pydantic component presets. The old multi-source hierarchy has been eliminated.

---

## Architecture (v1.8.9+)

```
CLI args (--mode balanced --sensitivity aggressive --vad-threshold 0.1)
    |
    v
main.py
    |
    +--[legacy modes: balanced/faster/fast/fidelity/kotoba]
    |       |
    |       v
    |   resolve_legacy_pipeline(mode, sensitivity)
    |       |
    |       v
    |   resolve_config_v3(asr=..., vad=..., features=...)
    |       |
    |       v
    |   Pydantic Component Registries
    |       config/components/asr/faster_whisper.py   (ASR presets)
    |       config/components/vad/silero.py            (VAD presets)
    |       config/components/features/scene_detection.py (Scene presets)
    |       |
    |       v
    |   component.get_preset(sensitivity).model_dump()
    |       |
    |       v
    |   CLI overrides applied (--vad-threshold, --speech-pad-ms, etc.)
    |   Only non-None CLI args override Pydantic values.
    |       |
    |       v
    |   resolved_config dict → Pipeline.__init__()
    |
    +--[modern modes: transformers/qwen/decoupled]
            |
            v
        Direct CLI args → Pipeline.__init__()
        (no config resolution, no Pydantic)
```

### What asr_config.json Contains (v1.8.9+)

```json
{
  "version": "4.4",
  "ui_preferences": { "console_verbosity": "summary", ... }
}
```

Only read by `ConfigManager.get_ui_preferences()` in `main.py` for console verbosity.

---

## Pydantic Preset Files (Source of Truth)

| File | What It Controls | Sensitivity Presets |
|------|-----------------|-------------------|
| `config/components/asr/faster_whisper.py` | Decoder (beam_size, patience, temperature), Transcriber (logprob_threshold, no_speech_threshold), Engine (repetition_penalty, chunk_length) | conservative, balanced, aggressive |
| `config/components/asr/stable_ts.py` | Stable-ts ASR params (fast/faster modes) | conservative, balanced, aggressive |
| `config/components/asr/openai_whisper.py` | OpenAI Whisper params (fidelity mode) | conservative, balanced, aggressive |
| `config/components/vad/silero.py` | VAD threshold, speech_pad_ms, max_group_duration_s, chunk_threshold_s | conservative, balanced, aggressive |
| `config/components/features/scene_detection.py` | Auditok/Silero scene detection params (energy thresholds, max_duration_s, pass1/pass2) | conservative, balanced, aggressive |

---

## Parameters NOT Controlled by Sensitivity

These use fixed values regardless of `--sensitivity`:

| Parameter | Location | Value | Notes |
|-----------|----------|-------|-------|
| `MAX_SAFE_CPS` | `config/sanitization_constants.py` | 30.0 | Post-processing subtitle filter |
| `MIN_SUBTITLE_DURATION` | `config/sanitization_constants.py` | 0.3 | Post-processing timing |
| `start_pad_samples` | `modules/speech_segmentation/factory.py` | 11200 (700ms) | Silero backend only |
| `end_pad_samples` | `modules/speech_segmentation/factory.py` | 20800 (1300ms) | Silero backend only |
| Speech segmenter backend | `modules/faster_whisper_pro_asr.py` | "silero-v4.0" | Hardcoded fallback |

---

## CLI Override Behavior

CLI arguments act as **overrides only**. If the user doesn't pass a flag, the Pydantic preset value is used unchanged.

| CLI Flag | Pydantic Source | Override Behavior |
|----------|----------------|-------------------|
| `--sensitivity` | Selects which preset to load | Always has value ("aggressive" default) |
| `--vad-threshold` | `SileroVADOptions.threshold` | Overrides if non-None |
| `--speech-pad-ms` | `SileroVADOptions.speech_pad_ms` | Overrides if non-None |
| `--model` | `FasterWhisperASR.model_id` | Overrides if non-None |
| `--device` | `FasterWhisperASR.default_device` | Overrides if non-None |
| `--compute-type` | `FasterWhisperASR.default_compute_type` | Overrides if non-None |

---

## Debugging Config Issues

### Quick Check: What Values Are Actually Used?

```bash
# Dump resolved parameters for any mode/sensitivity
python -m whisperjav.main --dump-params /dev/null --mode balanced --sensitivity aggressive
```

### Trace the Source

1. **Check Pydantic preset** — `config/components/asr/faster_whisper.py` presets dict
2. **Check CLI override** — was a flag explicitly passed?
3. **Check module fallback** — `modules/faster_whisper_pro_asr.py` `.get()` defaults (should match balanced preset)

### Verify Runtime Values

```python
from whisperjav.config.legacy import resolve_legacy_pipeline
config = resolve_legacy_pipeline('balanced', 'aggressive')
print(config['params']['decoder']['beam_size'])  # Should be 5
print(config['params']['vad']['threshold'])       # Should be 0.05
```

---

## Historical Context

| Version | Config Architecture |
|---------|-------------------|
| v1.0–v1.6 | `asr_config.json` was the source of truth |
| v1.7.0 | Pydantic components introduced, but JSON still read by TranscriptionTuner |
| v1.7.3 | Multi-source conflicts caused 20% subtitle regression |
| v1.8.9 | **JSON stripped to ui_preferences only.** Pydantic is sole source of truth. |

## Audit Trail

Full per-pipeline config resolution audit: `docs/audit/config_resolution_per_pipeline.md`

---

*Document created: 2025-12-21*
*Last updated: 2026-03-16 (v1.8.9 — simplified to single source of truth)*
