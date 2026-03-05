# WhisperJAV Configuration Sources Hierarchy

## Overview

WhisperJAV has multiple configuration sources that can override each other. Understanding this hierarchy is critical for debugging parameter issues.

**Lesson Learned (v1.7.3 regression):** A `chunk_threshold_s` parameter was set to 4.0 in two config files, but the actual runtime value was 0.18 because a third config source (component presets) had higher priority.

---

## Configuration Sources (Highest to Lowest Priority)

### 1. Component Presets (HIGHEST PRIORITY)
**Location:** `whisperjav/config/components/vad/silero.py`

```python
class SileroVAD:
    presets = {
        "conservative": SileroVADOptions(chunk_threshold_s=4.0, ...),
        "balanced": SileroVADOptions(chunk_threshold_s=4.0, ...),
        "aggressive": SileroVADOptions(chunk_threshold_s=4.0, ...),
    }
```

**When used:** Pipeline initialization via `TranscriptionTuner` when sensitivity preset is selected.

**Key insight:** These Pydantic models define the ACTUAL values used at runtime. They override everything else.

---

### 2. JSON Config File (asr_config.json)
**Location:** `whisperjav/config/asr_config.json`

```json
{
  "silero_vad_options": {
    "balanced": {
      "chunk_threshold_s": 4.0,
      "threshold": 0.225,
      ...
    },
    "aggressive": { ... },
    "conservative": { ... }
  }
}
```

**When used:** Legacy pipelines, fallback values, and some code paths that read directly from JSON.

**Key insight:** May be ignored if component presets take precedence in the code path.

---

### 3. YAML Ecosystem Configs
**Location:** `whisperjav/config/v4/ecosystems/tools/*.yaml`

Example: `silero-speech-segmentation.yaml`
```yaml
defaults:
  chunk_threshold_s: 2.5
  threshold: 0.5
```

**When used:** V4 config architecture, accessed via `ConfigManager`.

**Key insight:** Intended for future extensibility. Not always active in current pipelines.

---

### 4. Backend Module Defaults (LOWEST PRIORITY)
**Location:** `whisperjav/modules/speech_segmentation/backends/silero.py`

```python
class SileroSpeechSegmenter:
    def __init__(self, ..., chunk_threshold_s=None, ...):
        if chunk_threshold_s is not None:
            self.chunk_threshold_s = chunk_threshold_s
        else:
            self.chunk_threshold_s = 4.0  # Default fallback
```

**When used:** Only when no config provides a value (rare).

**Key insight:** This is the last resort. Usually overridden by higher-priority sources.

---

## Config Flow Diagram

```
User selects: mode=balanced, sensitivity=aggressive
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  TranscriptionTuner.resolve_config()                    │
│  - Looks up sensitivity preset from component presets   │
│  - Returns SileroVADOptions with chunk_threshold_s=4.0  │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  FasterWhisperProASR.__init__()                         │
│  - Receives merged config from TranscriptionTuner       │
│  - Passes to SpeechSegmenterFactory.create()            │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  SileroSpeechSegmenter.__init__()                       │
│  - Uses chunk_threshold_s from config (4.0)             │
│  - Falls back to module default only if not provided    │
└─────────────────────────────────────────────────────────┘
```

---

## Key Files Reference

| File | Purpose | Parameters Defined |
|------|---------|-------------------|
| `config/components/vad/silero.py` | Pydantic presets for Silero VAD | threshold, min_speech_duration_ms, chunk_threshold_s, speech_pad_ms, etc. |
| `config/asr_config.json` | Legacy JSON config | silero_vad_options, common_transcriber_options, temperature, etc. |
| `config/v4/ecosystems/tools/silero-speech-segmentation.yaml` | V4 YAML config | defaults, sensitivity overrides |
| `modules/speech_segmentation/backends/silero.py` | Backend implementation | VERSION_DEFAULTS, fallback values |
| `config/transcription_tuner.py` | Config resolver | Merges all sources, applies sensitivity |

---

## Debugging Config Issues

### Step 1: Check Debug Output
Look for the actual runtime values:
```
Creating speech segmenter: silero-v3.1 with params: {
    'chunk_threshold_s': 4.0,  <-- THIS is what's actually used
    ...
}
```

### Step 2: Trace the Source
If the value is wrong, check in order:
1. `config/components/vad/silero.py` - presets dict
2. `config/asr_config.json` - silero_vad_options section
3. `modules/speech_segmentation/backends/silero.py` - __init__ defaults

### Step 3: Verify All Sources Match
When fixing a parameter, update ALL sources to avoid confusion:
```bash
grep -rn "chunk_threshold" whisperjav/config/ whisperjav/modules/
```

---

## Common Pitfalls

### Pitfall 1: Fixing the Wrong Config
**Symptom:** You change a value in `asr_config.json` but runtime still uses old value.
**Cause:** Component presets in `config/components/vad/silero.py` have higher priority.
**Solution:** Always check and update component presets first.

### Pitfall 2: Multiple Config Systems
**Symptom:** V4 YAML configs don't seem to take effect.
**Cause:** Current pipelines may use component presets instead of V4 system.
**Solution:** Check which resolver is active in the pipeline code.

### Pitfall 3: Merged Configs
**Symptom:** Some parameters work, others don't.
**Cause:** `merged_segmenter_config = {**vad_params, **speech_segmenter_config}` - later dict wins.
**Solution:** Understand the merge order in `FasterWhisperProASR.__init__()`.

---

## Version History

| Version | chunk_threshold_s | Notes |
|---------|------------------|-------|
| v1.7.1 | 4.0 | Inline in FasterWhisperProASR, worked well |
| v1.7.3 (broken) | 0.18-0.2 | Multiple sources, component presets had wrong value |
| v1.7.3 (fixed) | 4.0 | All sources aligned |

---

## Related Documentation

- `docs/adr/ADR-001-yaml-config-architecture.md` - V4 config architecture decisions
- `whisperjav/config/v4/README.md` - V4 config system guide
- `CLAUDE.md` - General codebase guide

---

*Document created: 2025-12-21*
*Last updated: 2025-12-21*
*Context: Issue investigation - v1.7.3 produced 20% fewer subtitles than v1.7.1*
