# Compute Type & Temperature Configuration Catalog

**Created:** 2025-12-21
**Context:** Investigation of v1.7.3 subtitle regression (20%+ fewer subtitles than v1.7.1)

---

## Executive Summary

This document catalogs ALL sources where `compute_type` and `temperature` are defined,
their priority, and how they flow through the config resolution system.

**Key Finding:** The config system has multiple overlapping sources with different priorities.
The `resolver_v3.py` uses **component class defaults** (not asr_config.json values),
which has caused unexpected runtime values.

---

## Part 1: compute_type Configuration Sources

### 1.1 Component Class Default (ACTIVE - Used by resolver_v3)

**File:** `whisperjav/config/components/asr/faster_whisper.py:207`
```python
default_compute_type = "float16"
```

**When used:** `resolver_v3.py` builds model config using this value (line 133):
```python
'compute_type': asr_component.default_compute_type,
```

**Impact:** This is the ACTUAL runtime value for most pipeline modes.

---

### 1.2 JSON Config Models Section (NOT USED by resolver_v3!)

**File:** `whisperjav/config/asr_config.json` (models section)
```json
"faster-whisper-large-v2-int8": {
  "compute_type": "int8_float16"
}
```

**When used:** Only when `TranscriptionTuner.resolve_params()` is called directly.
The `resolver_v3.py` path IGNORES this!

**Impact:** This value is NOT reaching the ASR module in most cases.

---

### 1.3 CLI Model Override (args.model)

**File:** `whisperjav/main.py:1124`
```python
override_model_config = {
    "compute_type": "int8",  # Default to int8 for quantized models
}
```

**When used:** Only when user specifies `--model` flag.

**Impact:** This applies int8, but only for explicit model overrides.

---

### 1.4 FasterWhisperProASR Auto-Selection (Fallback)

**File:** `whisperjav/modules/faster_whisper_pro_asr.py:54-64`
```python
requested_compute_type = model_config.get("compute_type")
if requested_compute_type is None or requested_compute_type == "auto":
    if self.device == "cuda":
        self.compute_type = "int8_float16"  # CUDA auto-default
    elif self.device == "mps":
        self.compute_type = "float16"
    else:  # cpu
        self.compute_type = "int8"
```

**When used:** Only when compute_type is None or "auto" in model_config.

**Impact:** Never reached because resolver_v3 always passes a value ("float16").

---

### 1.5 v1.7.1 vs v1.7.3 Comparison

| Source | v1.7.1 | v1.7.3 | Notes |
|--------|--------|--------|-------|
| Component class | float16 | float16 | Same |
| asr_config.json (models) | int8_float16 | int8_float16 | Same, but IGNORED |
| CLI override | int8 | int8 | Same, only for --model |
| ASR auto-select | int8 | int8_float16 (CUDA) | Changed! But rarely reached |
| **Actual runtime** | **float16** | **float16** | **Both use component default!** |

---

## Part 2: temperature Configuration Sources

### 2.1 Component Class Presets (HIGHEST PRIORITY)

**File:** `whisperjav/config/components/asr/faster_whisper.py`

| Preset | Line | Value | Status |
|--------|------|-------|--------|
| conservative | 228 | `[0.0]` | Correct |
| balanced | 268 | `[0.0, 0.1]` | ✅ RESTORED (was `[0.0]`) |
| aggressive | 308 | `[0.0, 0.3]` | ✅ RESTORED (was `[0.0]`) |

**When used:** `resolver_v3.py` uses these via the preset lookup.

---

### 2.2 JSON Config Section (LOWER PRIORITY)

**File:** `whisperjav/config/asr_config.json` (common_transcriber_options)
```json
"balanced": { "temperature": [0.0, 0.1] },
"conservative": { "temperature": [0.0] },
"aggressive": { "temperature": [0.0, 0.3] }
```

**When used:** Legacy pipelines, TranscriptionTuner direct calls.

**Impact:** May be overridden by component presets depending on code path.

---

### 2.3 Legacy Mapper Defaults

**File:** `whisperjav/config/legacy.py:229`
```python
'temperature': asr_params.get('temperature', [0.0, 0.1]),
```

**When used:** When mapping v3 config to legacy structure.

**Impact:** Provides fallback if not in asr_params.

---

## Part 3: Config Resolution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLI Entry (main.py)                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  args.mode == "balanced"                                            │
│  → resolve_legacy_pipeline() in legacy.py                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  legacy.py calls resolve_config_v3() in resolver_v3.py              │
│  This builds model_config using COMPONENT CLASS DEFAULTS:           │
│    compute_type = asr_component.default_compute_type ("float16")    │
│    temperature from preset lookup                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  _map_to_legacy_structure() in legacy.py                            │
│  Maps v3 config to legacy decoder/provider structure                │
│  Temperature: asr_params.get('temperature', [0.0, 0.1])             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BalancedPipeline receives resolved_config                          │
│  Passes model_config to FasterWhisperProASR                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FasterWhisperProASR.__init__()                                     │
│  compute_type = model_config.get("compute_type")  # Gets "float16"  │
│  Since not None/auto, uses passed value directly                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Files to Modify for v1.7.1 Parity

### To change compute_type from float16 to int8:

| File | Line | Current | Target | Purpose |
|------|------|---------|--------|---------|
| `config/components/asr/faster_whisper.py` | 207 | `"float16"` | `"int8"` | Component class default |
| `config/components/asr/stable_ts.py` | 299 | `"float16"` | Keep | Different ASR |
| `config/components/asr/openai_whisper.py` | 176 | `"float16"` | Keep | Different ASR |
| `config/components/asr/kotoba_faster_whisper.py` | 175 | `"float16"` | Keep | Different ASR |

### To ensure temperature fallbacks:

| File | Line | Current | Target | Status |
|------|------|---------|--------|--------|
| `config/components/asr/faster_whisper.py` | 268 | `[0.0, 0.1]` | `[0.0, 0.1]` | ✅ DONE |
| `config/components/asr/faster_whisper.py` | 308 | `[0.0, 0.3]` | `[0.0, 0.3]` | ✅ DONE |

---

## Part 5: Risks and Considerations

### compute_type Change Risks:

1. **int8 uses FP32 for non-quantized layers** - slower on modern GPUs with Tensor Cores
2. **int8_float16 uses FP16 for non-quantized layers** - faster but requires FP16 support
3. **float16 uses FP16 weights (no quantization)** - highest precision but most VRAM

### Recommended Approach:

For v1.7.1 parity testing:
- Change `default_compute_type` to `"int8"` in faster_whisper.py
- If this fixes the regression, the issue was compute_type precision
- If not, investigate other parameters

### Alternative: Make compute_type configurable per-sensitivity

Instead of a class-level default, add compute_type to each preset:
```python
presets = {
    "conservative": FasterWhisperOptions(
        compute_type="int8",  # More conservative
        ...
    ),
    "balanced": FasterWhisperOptions(
        compute_type="int8",  # v1.7.1 compatible
        ...
    ),
    "aggressive": FasterWhisperOptions(
        compute_type="int8_float16",  # Maximum detection
        ...
    ),
}
```

This would require modifying FasterWhisperOptions to include compute_type as a field.

---

## Part 6: Related Documentation

- `docs/architecture/CONFIG_SOURCES_HIERARCHY.md` - General config priority
- `whisperjav/config/v4/README.md` - V4 YAML config system
- `CLAUDE.md` - Config priority warning section

---

*Document created: 2025-12-21*
*Last updated: 2025-12-21*
