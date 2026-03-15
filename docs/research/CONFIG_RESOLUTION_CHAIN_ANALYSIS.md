# Config Resolution Chain Analysis — Balanced Pipeline

> Created: 2026-03-15 | Source: Deep investigation of config flow, GUI Customize panel, and parameter comparison vs XXL

## Purpose

This document validates that the v1.8.9 Quality Improvement Plan (`docs/plans/V189_QUALITY_IMPROVEMENT_PLAN.md`) can safely change balanced pipeline parameters by modifying **one file** — the Pydantic presets in `config/components/asr/faster_whisper.py`. It also identifies two additional items to include in the v1.8.9 PR.

---

## 1. Config Resolution Chain

The complete parameter resolution path for the balanced pipeline:

```
CLI args (main.py argparse)
    ↓
resolve_legacy_pipeline("balanced", sensitivity, task)     [config/legacy.py:133]
    ↓
LEGACY_PIPELINES["balanced"] → asr="faster_whisper"        [config/legacy.py:96]
    ↓
resolve_config_v3(asr="faster_whisper", sensitivity=...)   [config/resolver_v3.py:189]
    ↓
FasterWhisperASR.get_preset(sensitivity)                   [config/components/asr/faster_whisper.py:244]
    ↓
Pydantic preset dict (beam_size, best_of, temperature...)  [config/components/asr/faster_whisper.py:212-334]
    ↓
_get_compute_type_for_device("faster_whisper", "cuda")     [config/resolver_v3.py:183]
    → returns "auto" (CTranslate2 resolves to int8_float16)
    ↓
Pipeline receives resolved config dict
    ↓
CTranslate2 model.transcribe(**params)
```

### Key Finding: Pydantic Presets Are the SOLE Runtime Source of Truth

For the balanced pipeline CLI and GUI paths, the **only** place ASR parameters are defined at runtime is `FasterWhisperASR.presets` in `config/components/asr/faster_whisper.py`.

Other config sources are **NOT** in the runtime path:

| Source | Status | Explanation |
|--------|--------|-------------|
| `config/components/asr/faster_whisper.py` presets | **ACTIVE** | Sole runtime source for balanced pipeline |
| `config/resolver_v3.py` `_get_compute_type_for_device()` | **ACTIVE** | Returns `"auto"` for faster_whisper on CUDA |
| `config/asr_config.json` | **DEAD CODE** | `TranscriptionTuner` reads it, but `main.py` never calls `TranscriptionTuner` for balanced pipeline |
| `config/v4/ecosystems/` YAML | **NOT USED** | v4 YAML system exists but balanced pipeline uses v3 resolution |
| `config/transcription_tuner.py` | **DEAD CODE** | Never invoked by `main.py` for CLI/GUI balanced path |

### Implication

Changing Pydantic preset values in `faster_whisper.py` is sufficient. No other config files need to be changed for the parameter fixes to take effect at runtime.

---

## 2. Pipeline Isolation

Each pipeline uses a completely separate ASR component class:

| Pipeline | ASR Component | Provider | Presets Class |
|----------|--------------|----------|---------------|
| balanced | `FasterWhisperASR` | `faster_whisper` | `FasterWhisperOptions` |
| fidelity | `OpenAIWhisperASR` | `openai_whisper` | `OpenAIWhisperOptions` |
| faster | `StableTsASR` | `stable_ts` | `StableTsOptions` |
| fast | `StableTsASR` | `stable_ts` | `StableTsOptions` |

**Confirmed**: `fidelity_pipeline.py` line 12 imports `WhisperProASR` (NOT `FasterWhisperProASR`). Changes to `FasterWhisperASR.presets` have zero impact on fidelity, faster, or fast pipelines.

---

## 3. GUI Customize Panel Investigation

### C1: Does opening Customize → Apply without changes alter parameters?

**Answer: No.** The GUI Customize panel loads its default values from the backend API:

- `api.py:2042` — `get_pipeline_defaults()` calls `resolve_legacy_pipeline()` → reads Pydantic presets
- `app.js:1839` — `openCustomize()` fetches defaults from backend, populates form fields
- `app.js:3943` — `applyCustomization()` collects current form values, sends as JSON

The flow is: Backend (Pydantic presets) → JSON → JS form fields → User edits (or not) → JSON back to backend → CLI args. No hardcoded JS values override the backend defaults.

### C2: Do GUI default values match Pydantic presets?

**Answer: Yes.** The GUI populates form fields from the API response, which reads from Pydantic presets. There are no hardcoded JavaScript defaults that differ from Python backend values.

### C3: Do VAD values match?

**Answer: Yes.** VAD parameters follow the same resolution path through `SileroVAD.presets` in `config/components/vad/silero.py`. The GUI categorization groups some VAD params under "Audio Preprocessing" but the values are consistent.

### Conclusion

The GUI Customize panel is NOT a source of parameter drift. It faithfully reflects Pydantic preset values.

---

## 4. Parameter Comparison: WhisperJAV Balanced Aggressive vs XXL Defaults

| Parameter | WhisperJAV (balanced/aggressive) | XXL Default | Gap? |
|-----------|----------------------------------|-------------|------|
| **model** | `large-v2` (line 201) | `large-v3` | **YES — Q1 CRITICAL** |
| **compute_type** | `"auto"` → `int8_float16` (line 183) | `float16` | **YES — Q2 HIGH** |
| **beam_size** | `2` (line 298) | `5` | **YES — Q3 HIGH** |
| **best_of** | `1` (line 299) | `5` | **YES — Q3 HIGH** |
| **temperature** | `[0.0, 0.3]` (line 308) | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` | **YES — 2d** |
| patience | `2.9` (line 300) | `1.0` (default) | WhisperJAV higher (ok) |
| compression_ratio_threshold | `3.0` (line 309) | `2.4` | WhisperJAV more permissive (ok) |
| logprob_threshold | `-2.5` (line 310) | `-1.0` | WhisperJAV more permissive (ok) |
| no_speech_threshold | `0.22` (line 312) | `0.6` | WhisperJAV more aggressive (ok) |
| condition_on_previous_text | `False` (line 314) | `True` (Whisper default) | Intentional — D4 decision |
| repetition_penalty | `1.1` (line 322) | `1.0` | WhisperJAV slightly higher (ok) |
| no_repeat_ngram_size | `2` (line 323) | `0` | WhisperJAV stricter (ok) |
| hallucination_silence_threshold | `2.5` (line 332) | None | WhisperJAV has it (good) |
| vad_filter | `False` (external VAD) | `True` (internal) | Intentional — D5 decision |
| vocal separation | None | `--ff_vocal_extract mdx_kim2` | **Architectural gap** |

### Summary

- **4 parameters** account for most of the quality gap (Q1, Q2, Q3, 2d) — all fixable by changing preset values
- **Vocal separation** is an architectural difference — addressed by TH2 (XXL BYOP)
- Other parameter differences are intentional WhisperJAV design decisions (D4, D5)

---

## 5. The `--model` Compute Type Trap

**Location**: `main.py:1751`

```python
override_compute_type = args.compute_type if args.compute_type and args.compute_type != "auto" else "int8"
```

When a user overrides `--model` on the CLI, this line silently forces `compute_type="int8"` regardless of pipeline or sensitivity. This defeats the v1.8.9 compute_type fix (Q2).

**Recommendation**: Change fallback from `"int8"` to `"auto"` or call `_get_compute_type_for_device()` to respect the same resolution logic.

**Must be fixed in v1.8.9 PR alongside the preset parameter changes.**

---

## 6. `asr_config.json` Dead Code Note

`config/asr_config.json` contains ASR parameters that look authoritative but are never read at runtime for the balanced pipeline CLI/GUI path. The `TranscriptionTuner` class reads from it, but `main.py` never invokes `TranscriptionTuner` in the balanced pipeline flow.

**Recommendation**: Add a deprecation comment at the top of `asr_config.json` noting which sections are dead code vs still active (scene detection config IS still used). This prevents future developers from assuming changes to `asr_config.json` affect balanced pipeline behavior.

---

## 7. Refreshed Recommendations for V189 Implementation

Based on this analysis, the V189 Quality Improvement Plan is **safe to execute**. The config architecture is NOT a blocker.

### Changes to the V189 plan:

1. **Confirm**: Pydantic presets in `faster_whisper.py` are the only file that needs parameter changes for Q1, Q2 (partial), Q3, 2d
2. **Confirm**: `resolver_v3.py` line 183 needs change for Q2 (compute_type `"auto"` → `"float16"` for balanced)
3. **Add to V189 PR**: Fix `main.py:1751` compute_type trap (change `"int8"` fallback to `"auto"`)
4. **Add to V189 PR**: Add deprecation note to `asr_config.json` header
5. **No GUI changes needed**: Customize panel already reads from Pydantic presets
6. **Pipeline isolation confirmed**: No risk of affecting fidelity/faster/fast pipelines

### Files to modify (updated):

| File | Change | Status |
|------|--------|--------|
| `config/components/asr/faster_whisper.py` | model_id, beam_size, best_of, temperature | In original plan |
| `config/resolver_v3.py` | compute_type for balanced | In original plan |
| `config/sanitization_constants.py` | MAX_SAFE_CPS | In original plan |
| `config/asr_config.json` | pass2_max_duration_s + deprecation note | Updated |
| `modules/faster_whisper_pro_asr.py` | VAD failover | In original plan |
| **`main.py`** | **Fix line 1751 compute_type trap** | **NEW — must add** |

---

## Sources

- `whisperjav/config/components/asr/faster_whisper.py` lines 188-334 — Pydantic presets
- `whisperjav/config/resolver_v3.py` lines 170-260 — Config resolution
- `whisperjav/config/legacy.py` lines 94-164 — Legacy pipeline definitions
- `whisperjav/main.py` line 1751 — Compute type trap
- `whisperjav/webview_gui/api.py` line 2042 — GUI defaults loading
- `whisperjav/webview_gui/assets/app.js` lines 1839, 3943 — GUI Customize panel
- `whisperjav/pipelines/fidelity_pipeline.py` line 12 — Pipeline isolation proof
- `docs/research/FASTER_WHISPER_XXL_CLI_REFERENCE.md` — XXL default parameters
