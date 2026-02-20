# Qwen Pipeline — Architecture & Reference

> **Status**: Current as of v1.8.5-dev (post-Phase 4 Strangulation)
> **Source of truth**: `whisperjav/pipelines/qwen_pipeline.py`
> **Last verified against code**: 2026-02-20

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [The 9-Phase Processing Flow](#3-the-9-phase-processing-flow)
4. [Decoupled Subtitle Pipeline (ADR-006)](#4-decoupled-subtitle-pipeline-adr-006)
5. [Component Registry](#5-component-registry)
6. [Default Values Reference](#6-default-values-reference)
7. [CLI Arguments](#7-cli-arguments)
8. [GUI Wiring](#8-gui-wiring)
9. [Ensemble Mode Wiring](#9-ensemble-mode-wiring)
10. [Configuration File Locations](#10-configuration-file-locations)
11. [VRAM Management](#11-vram-management)
12. [Alignment Sentinel & Step-Down Retry](#12-alignment-sentinel--step-down-retry)
13. [Timestamp Resolution Modes](#13-timestamp-resolution-modes)
14. [Deprecated / Removed Features](#14-deprecated--removed-features)

---

## 1. Overview

The Qwen Pipeline is a dedicated 9-phase subtitle generation pipeline built around
Qwen3-ASR. It follows the "redundancy over reuse" principle — no backend switching,
no Whisper fallbacks.

Since **Phase 4 (Strangulation)**, only the **assembly mode** code path exists.
The previous coupled modes (`context_aware`, `vad_slicing`) were deleted. Legacy
mode names are still accepted as CLI/API values but are mapped to assembly mode
with appropriate framer overrides (with deprecation warnings).

**Key architectural decisions:**
- ADR-004: Dedicated Qwen pipeline (separate from Whisper pipelines)
- ADR-006: Decoupled Subtitle Pipeline (protocol-based component swapping)

---

## 2. Architecture Diagram

### High-Level: 9-Phase Flow

```
Media File
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Audio Extraction (48kHz WAV)                    │
│   AudioExtractor → {basename}_extracted.wav              │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 2: Scene Detection (default: semantic)             │
│   SceneDetectorFactory → scene_0000.wav, scene_0001.wav… │
│   Safe chunking: min=12s, max=48s                        │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 3: Speech Enhancement ◀── VRAM Block 1             │
│   (default: none) → *_enhanced.wav                       │
│   Cleanup: enhancer.cleanup() → safe_cuda_cleanup()      │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 4: Speech Segmentation / VAD (default: TEN)        │
│   SpeechSegmenterFactory → speech_regions_per_scene{}    │
│   Used by: VadGroupedFramer, Sentinel recovery           │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 5: ASR Transcription ◀── VRAM Block 2              │
│   DecoupledSubtitlePipeline orchestrator                  │
│   (see Section 4 for internal flow)                      │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 6: Scene SRT Generation                            │
│   WhisperResult.to_srt_vtt() per scene                   │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 7: SRT Stitching                                   │
│   SRTStitcher.stitch() → {basename}_stitched.srt         │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 8: Sanitisation (SKIPPED for Qwen)                 │
│   Copy stitched → final output                           │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 9: Analytics (optional)                            │
│   compute_analytics() → .analytics.json                  │
└──────────────────────────────────────────────────────────┘
    │
    ▼
{basename}.{lang}.whisperjav.srt
```

### Phase 5 Internal: Decoupled Subtitle Pipeline

```
For each scene:
    ┌─────────────────────────────────────┐
    │ Step 1: Temporal Framing            │
    │   TemporalFramer.frame(audio, sr)   │
    │   → TemporalFrame[]                 │
    │   + Audio slicing → temp WAVs       │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │ Steps 2-4: Generate + Clean         │
    │   ◀─ generator.load() [VRAM]        │
    │   TextGenerator.generate_batch()    │
    │   TextCleaner.clean_batch()         │
    │   generator.unload() ─▶             │
    │   safe_cuda_cleanup()               │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │ Steps 5-7: Alignment                │
    │   ◀─ aligner.load() [VRAM]          │
    │   TextAligner.align_batch()         │
    │   aligner.unload() ─▶               │
    │   safe_cuda_cleanup()               │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │ Step 8: Word Merging                │
    │   frame-relative → scene-relative   │
    │   offset by frame.start             │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │ Step 9: Sentinel → Reconstruct →    │
    │          Harden                      │
    │   assess_alignment_quality()        │
    │   if COLLAPSED → redistribute       │
    │   reconstruct_from_words()          │
    │   harden_scene_result()             │
    │   → WhisperResult + SceneDiagnostics│
    └─────────────────────────────────────┘

Step-Down Retry (if enabled):
    Pass 1 results → identify COLLAPSED scenes
    → reframe with tighter grouping (6.0s)
    → repeat Steps 2-9 for collapsed scenes only
    → replace Pass 1 results if improved
```

---

## 3. The 9-Phase Processing Flow

| Phase | Name | Default | Module | VRAM |
|-------|------|---------|--------|------|
| 1 | Audio Extraction | 48kHz WAV | `AudioExtractor` | — |
| 2 | Scene Detection | `semantic` | `SceneDetectorFactory` | — |
| 3 | Speech Enhancement | `none` | `create_enhancer_direct()` | Block 1 |
| 4 | Speech Segmentation | `ten` (TEN VAD) | `SpeechSegmenterFactory` | — |
| 5 | ASR Transcription | Assembly mode | `DecoupledSubtitlePipeline` | Block 2 |
| 6 | Scene SRT Gen | — | `WhisperResult.to_srt_vtt()` | — |
| 7 | SRT Stitching | — | `SRTStitcher` | — |
| 8 | Sanitisation | Skipped | (no Qwen-specific sanitizer) | — |
| 9 | Analytics | Optional | `pipeline_analytics` | — |

### Phase 2: Scene Detection Bounds (Safe Chunking)

When `safe_chunking=True` (default):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `min_duration` | 12s | Avoid too-short scenes that waste overhead |
| `max_duration` | 48s | Keep well within ForcedAligner's 180s hard limit |

### Phase 4: VAD Configuration

The speech segmenter runs on every scene and produces `speech_regions_per_scene{}`.
These regions are used by:
1. **VadGroupedFramer** — to create temporal frames based on VAD groups
2. **Alignment Sentinel** — for VAD-guided collapse recovery (Strategy C)
3. **Hardening** — for `vad_only` and `aligner_vad_fallback` timestamp modes

---

## 4. Decoupled Subtitle Pipeline (ADR-006)

### Protocol Components

| Protocol | Purpose | Current Implementations |
|----------|---------|------------------------|
| `TemporalFramer` | "WHEN does dialogue happen?" | `FullSceneFramer`, `VadGroupedFramer`, `SrtSourceFramer`, `ManualFramer` |
| `TextGenerator` | "WHAT was said?" (text-only ASR) | `Qwen3TextGenerator` |
| `TextCleaner` | Pre-alignment text cleaning | `Qwen3TextCleaner`, `PassthroughCleaner` |
| `TextAligner` | "WHEN exactly was each word said?" | `Qwen3ForcedAligner` |

### File Layout

```
whisperjav/modules/subtitle_pipeline/
├── __init__.py          # Re-exports public APIs
├── types.py             # TemporalFrame, FramingResult, TranscriptionResult,
│                        # AlignmentResult, WordTimestamp, TimestampMode,
│                        # HardeningConfig, StepDownConfig, SceneDiagnostics
├── protocols.py         # TemporalFramer, TextGenerator, TextCleaner, TextAligner
├── hardening.py         # harden_scene_result()
├── reconstruction.py    # reconstruct_from_words()
├── orchestrator.py      # DecoupledSubtitlePipeline (the wiring)
├── framers/
│   ├── factory.py       # TemporalFramerFactory
│   ├── full_scene.py    # FullSceneFramer (1 frame = entire scene)
│   ├── vad_grouped.py   # VadGroupedFramer (VAD-based dialogue chunks)
│   ├── srt_source.py    # SrtSourceFramer (use existing SRT boundaries)
│   └── manual.py        # ManualFramer (API-only, timestamps provided)
├── generators/
│   ├── factory.py       # TextGeneratorFactory
│   └── qwen3_generator.py  # Qwen3TextGenerator
├── cleaners/
│   ├── factory.py       # TextCleanerFactory
│   └── qwen3.py         # Qwen3TextCleaner
└── aligners/
    ├── factory.py       # TextAlignerFactory
    └── qwen3_aligner.py # Qwen3ForcedAligner
```

### Framer Details

| Framer | CLI Name | Behavior | Supports `reframe()` |
|--------|----------|----------|---------------------|
| `FullSceneFramer` | `full-scene` | Single frame spanning entire scene. No audio slicing needed. | Yes |
| `VadGroupedFramer` | `vad-grouped` | Groups VAD speech segments into chunks up to `max_group_duration_s`. | Yes |
| `SrtSourceFramer` | `srt-source` | Creates frames from an existing SRT file's segment boundaries. | No |
| `ManualFramer` | `manual` | Takes explicit timestamp list. Python API only. | No |

### VRAM Swap Pattern

The orchestrator explicitly manages GPU memory to allow the ASR model (1.7B)
and ForcedAligner (0.6B) to share a single GPU:

```
generator.load()    ← ~5 GB VRAM for Qwen3-ASR-1.7B
  → generate all scenes
generator.unload()  ← release VRAM
safe_cuda_cleanup() ← torch.cuda.empty_cache()

aligner.load()      ← ~2 GB VRAM for ForcedAligner-0.6B
  → align all scenes
aligner.unload()    ← release VRAM
safe_cuda_cleanup()
```

---

## 5. Component Registry

### Factories

| Factory | Location | Registry |
|---------|----------|----------|
| `TemporalFramerFactory` | `subtitle_pipeline/framers/factory.py` | `full-scene`, `vad-grouped`, `srt-source`, `manual` |
| `TextGeneratorFactory` | `subtitle_pipeline/generators/factory.py` | `qwen3` |
| `TextCleanerFactory` | `subtitle_pipeline/cleaners/factory.py` | `qwen3`, `passthrough` |
| `TextAlignerFactory` | `subtitle_pipeline/aligners/factory.py` | `qwen3` |
| `SceneDetectorFactory` | `modules/scene_detection_backends/` | `auditok`, `silero`, `semantic`, `none` |
| `SpeechSegmenterFactory` | `modules/speech_segmentation/` | `ten`, `silero`, `nemo`, `none`, etc. |
| `SpeechEnhancerFactory` | `modules/speech_enhancement/` | `none`, `clearvoice`, `bs-roformer` |

---

## 6. Default Values Reference

### Pipeline Constructor Defaults

| Parameter | Default | CLI Flag | Notes |
|-----------|---------|----------|-------|
| `qwen_input_mode` | `"assembly"` | `--qwen-input-mode` | Only active value; `context_aware`/`vad_slicing` map to assembly |
| `qwen_framer` | `"full-scene"` | `--qwen-framer` | Temporal framing strategy |
| `scene_detector` | `"semantic"` | `--qwen-scene` | Scene detection backend |
| `speech_enhancer` | `"none"` | `--qwen-enhancer` | Speech enhancement backend |
| `speech_segmenter` | `"ten"` | `--qwen-segmenter` | VAD / speech segmentation backend |
| `segmenter_max_group_duration` | `6.0` | `--qwen-max-group-duration` | Max VAD group size in seconds |
| `model_id` | `"Qwen/Qwen3-ASR-1.7B"` | `--qwen-model-id` | ASR model |
| `aligner_id` | `"Qwen/Qwen3-ForcedAligner-0.6B"` | `--qwen-aligner` | ForcedAligner model |
| `device` | `"auto"` | `--qwen-device` | `auto`/`cuda`/`cpu` |
| `dtype` | `"auto"` | `--qwen-dtype` | `auto`/`float16`/`bfloat16`/`float32` |
| `batch_size` | `1` | `--qwen-batch-size` | Inference batch size |
| `max_new_tokens` | `4096` | `--qwen-max-tokens` | Token generation budget (overridden dynamically) |
| `language` | `"Japanese"` | `--qwen-language` | ASR language |
| `timestamps` | `"word"` | `--qwen-timestamps` | `word` (ForcedAligner) or `none` |
| `context` | `""` | `--qwen-context` | Context string for accuracy |
| `attn_implementation` | `"auto"` | `--qwen-attn` | Attention backend |
| `timestamp_mode` | `"aligner_interpolation"` | `--qwen-timestamp-mode` | Timestamp resolution strategy |
| `qwen_safe_chunking` | `True` | `--qwen-safe-chunking` | Enforce scene boundaries |
| `assembly_cleaner` | `True` | `--qwen-assembly-cleaner` | Pre-alignment text cleaning |
| `stepdown_enabled` | `True` | `--qwen-stepdown` | Adaptive step-down retry |
| `stepdown_initial_group` | `6.0` | `--qwen-stepdown-initial-group` | Tier 1 group size |
| `stepdown_fallback_group` | `6.0` | `--qwen-stepdown-fallback-group` | Tier 2 fallback group size |
| `repetition_penalty` | `1.1` | `--qwen-repetition-penalty` | Generation repetition penalty |
| `max_tokens_per_audio_second` | `20.0` | `--qwen-max-tokens-per-second` | Dynamic token budget scaling |
| `japanese_postprocess` | `False` | — | **Deprecated.** Always False for Qwen3. |

### Default CLI vs Default Pipeline

Note: The CLI default for `--qwen-timestamp-mode` is `aligner_vad_fallback`,
while the pipeline constructor default is `aligner_interpolation`. The CLI value
takes precedence when invoked from the command line.

---

## 7. CLI Arguments

All Qwen-specific arguments are in the `"Qwen3-ASR Mode Options (--mode qwen)"` group.

**Usage:** `whisperjav video.mp4 --mode qwen [options]`

### Core Model

| Flag | Default | Description |
|------|---------|-------------|
| `--qwen-model-id` | `Qwen/Qwen3-ASR-1.7B` | ASR model HuggingFace ID |
| `--qwen-aligner` | `Qwen/Qwen3-ForcedAligner-0.6B` | ForcedAligner model ID |
| `--qwen-device` | `auto` | Compute device (`auto`/`cuda`/`cpu`) |
| `--qwen-dtype` | `auto` | Data type (`auto`/`float16`/`bfloat16`/`float32`) |
| `--qwen-batch-size` | `1` | Inference batch size |
| `--qwen-max-tokens` | `4096` | Max tokens to generate per utterance |
| `--qwen-language` | `Japanese` | Language for ASR (`auto` for auto-detect) |
| `--qwen-timestamps` | `word` | Timestamp mode (`word`/`none`) |
| `--qwen-attn` | `auto` | Attention implementation |

### Pipeline Control

| Flag | Default | Description |
|------|---------|-------------|
| `--qwen-input-mode` | `assembly` | Input strategy. `context_aware`/`vad_slicing` are deprecated aliases. |
| `--qwen-framer` | `full-scene` | Temporal framer (`full-scene`/`vad-grouped`/`srt-source`) |
| `--qwen-framer-srt-path` | — | SRT file for `srt-source` framer |
| `--qwen-scene` | `semantic` | Scene detection (`none`/`auditok`/`silero`/`semantic`) |
| `--qwen-safe-chunking` | `True` | Enforce 12-48s scene boundaries |
| `--qwen-timestamp-mode` | `aligner_vad_fallback` | Timestamp resolution strategy |
| `--qwen-assembly-cleaner` | `True` | Pre-alignment text cleaning |

### Speech Processing

| Flag | Default | Description |
|------|---------|-------------|
| `--qwen-enhancer` | `none` | Enhancement backend (`none`/`clearvoice`/`bs-roformer`/etc.) |
| `--qwen-enhancer-model` | — | Enhancement model variant |
| `--qwen-segmenter` | `ten` | VAD backend (`ten`/`silero`/`nemo`/`none`/etc.) |
| `--qwen-max-group-duration` | — (pipeline: `6.0`) | Max VAD group duration |

### Step-Down Retry

| Flag | Default | Description |
|------|---------|-------------|
| `--qwen-stepdown` / `--no-qwen-stepdown` | `True` | Enable/disable step-down retry |
| `--qwen-stepdown-initial-group` | — (pipeline: `6.0`) | Tier 1 group size |
| `--qwen-stepdown-fallback-group` | — (pipeline: `6.0`) | Tier 2 fallback group size |

### Generation Safety

| Flag | Default | Description |
|------|---------|-------------|
| `--qwen-repetition-penalty` | `1.1` | Token repetition penalty (1.0=off) |
| `--qwen-max-tokens-per-second` | `20.0` | Dynamic token budget per audio second |

### Context

| Flag | Default | Description |
|------|---------|-------------|
| `--qwen-context` | `""` | Inline context string (speaker names, domain terms) |
| `--qwen-context-file` | — | Path to context glossary file |

---

## 8. GUI Wiring

### Ensemble Tab (Tab 3)

The GUI uses a two-pass ensemble architecture. The Qwen pipeline is typically
used as Pass 2.

#### HTML Elements (index.html)

| Element ID | Type | Options | Default | Visible When |
|------------|------|---------|---------|-------------|
| `pass1-pipeline` | `<select>` | faster/fast/balanced/fidelity/transformers/**qwen** | Pass 1 default varies | Always |
| `pass2-pipeline` | `<select>` | faster/fast/balanced/fidelity/transformers/**qwen** | `qwen` | Always |
| `pass1-framer` | `<select>` | **full-scene**/vad-grouped | `full-scene` | When pass1 = qwen |
| `pass2-framer` | `<select>` | **full-scene**/vad-grouped | `full-scene` | When pass2 = qwen |
| `pass1-scene` / `pass2-scene` | `<select>` | auditok/silero/**semantic**/none | `semantic` | Always |
| `pass1-enhancer` / `pass2-enhancer` | `<select>` | **none**/clearvoice/bs-roformer | `none` | Always |
| `pass1-model` / `pass2-model` | `<select>` | Qwen3-ASR-1.7B/Qwen3-ASR-0.6B | `1.7B` | When qwen |

#### JavaScript State (app.js)

```javascript
state: {
    pass1: {
        pipeline: 'balanced',
        framer: 'full-scene',    // Qwen framer selection
        // ... other fields
    },
    pass2: {
        pipeline: 'qwen',
        framer: 'full-scene',    // Qwen framer selection
        // ... other fields
    }
}
```

#### Visibility Logic

When a pass's pipeline is `qwen`:
- The **sensitivity** dropdown is hidden (not applicable to Qwen)
- The **framer** dropdown is shown (replaces sensitivity position)
- The Customize modal loads YAML schemas for model/segmenter/enhancer/scene parameters

#### Data Flow: GUI → Backend

```
app.js state    →  EnsembleManager.startProcessing()
                →  pywebview.api.start_ensemble(options)
                →  api.py: _build_ensemble_args(options)
                     pass1.framer → qwen1_params['framer']
                →  pass_worker.py: prepare_qwen_params()
                     mapping: 'framer' → 'qwen_framer'
                →  _create_qwen_pipeline_subprocess()
                     → --qwen-framer {value}
                →  main.py argparse → QwenPipeline(qwen_framer=value)
```

---

## 9. Ensemble Mode Wiring

### Default Qwen Parameters (pass_worker.py)

```python
DEFAULT_QWEN_PARAMS = {
    "qwen_model_id": "Qwen/Qwen3-ASR-1.7B",
    "qwen_device": "auto",
    "qwen_dtype": "auto",
    "qwen_batch_size": 1,
    "qwen_max_tokens": 4096,
    "qwen_language": "Japanese",
    "qwen_timestamps": "word",
    "qwen_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
    "qwen_scene": "semantic",
    "qwen_context": "",
    "qwen_attn": "auto",
    "qwen_enhancer": "none",
    "qwen_segmenter": "ten",
    "qwen_input_mode": "assembly",
    "qwen_framer": "vad-grouped",     # ← NOTE: ensemble defaults to vad-grouped
    "qwen_safe_chunking": True,
    "qwen_timestamp_mode": "aligner_vad_fallback",
    "qwen_assembly_cleaner": True,
    "qwen_repetition_penalty": 1.1,
    "qwen_max_tokens_per_second": 20.0,
    "qwen_stepdown": True,
}
```

**Important difference**: The ensemble default framer is `vad-grouped` (to preserve
the behavior of the previous `vad_slicing` mode), while the CLI/pipeline default
is `full-scene`.

### Parameter Mapping (GUI key → CLI key)

```python
mapping = {
    "model_id":                "qwen_model_id",
    "device":                  "qwen_device",
    "dtype":                   "qwen_dtype",
    "batch_size":              "qwen_batch_size",
    "max_new_tokens":          "qwen_max_tokens",
    "language":                "qwen_language",
    "timestamps":              "qwen_timestamps",
    "aligner_id":              "qwen_aligner",
    "scene":                   "qwen_scene",
    "context":                 "qwen_context",
    "attn_implementation":     "qwen_attn",
    "input_mode":              "qwen_input_mode",
    "framer":                  "qwen_framer",
    "safe_chunking":           "qwen_safe_chunking",
    "timestamp_mode":          "qwen_timestamp_mode",
    "assembly_cleaner":        "qwen_assembly_cleaner",
    "repetition_penalty":      "qwen_repetition_penalty",
    "max_tokens_per_audio_second": "qwen_max_tokens_per_second",
    "max_group_duration":      "qwen_max_group_duration",
    "stepdown":                "qwen_stepdown",
    "stepdown_initial_group":  "qwen_stepdown_initial_group",
    "stepdown_fallback_group": "qwen_stepdown_fallback_group",
}
```

---

## 10. Configuration File Locations

| File | Purpose | Priority |
|------|---------|----------|
| `whisperjav/config/v4/ecosystems/qwen/models/qwen3-asr-1.7b.yaml` | YAML model config (spec, presets, GUI hints) | YAML schema source |
| `whisperjav/config/v4/ecosystems/qwen/ecosystem.yaml` | Qwen ecosystem metadata | Ecosystem registry |
| `whisperjav/config/v4/ecosystems/tools/ten-speech-segmentation.yaml` | TEN VAD config (spec, presets, GUI hints) | Tool schema source |
| `whisperjav/config/asr_config.json` | Legacy ASR settings (v1-v3) | Not used by Qwen pipeline |
| `whisperjav/pipelines/qwen_pipeline.py` | Pipeline constructor defaults | **Constructor = runtime truth** |
| `whisperjav/main.py` | CLI argparse defaults | **CLI = user-facing defaults** |
| `whisperjav/ensemble/pass_worker.py` | Ensemble defaults (`DEFAULT_QWEN_PARAMS`) | **Ensemble = GUI defaults** |

### Config Priority (highest to lowest)

1. **CLI arguments** / GUI Customize overrides — user-specified values
2. **Ensemble defaults** (`DEFAULT_QWEN_PARAMS`) — for GUI ensemble mode
3. **Pipeline constructor defaults** (`QwenPipeline.__init__`) — fallback
4. **YAML model/tool configs** — schema source for GUI Customize modal

---

## 11. VRAM Management

### Two-Block Pattern

The pipeline uses two sequential VRAM blocks to avoid GPU memory contention:

| Block | Phase | Models Loaded | Approx. VRAM |
|-------|-------|---------------|-------------|
| Block 1 | Phase 3 | Speech Enhancer (e.g., ClearVoice, BS-RoFormer) | ~2-4 GB |
| Block 2 | Phase 5 | Qwen3-ASR-1.7B (generator) + ForcedAligner-0.6B (aligner) | ~5+2 GB |

Block 1 is fully released before Block 2 loads:
```
Phase 3: enhancer.cleanup() → del enhancer → safe_cuda_cleanup()
Phase 5: generator.load() → ... → generator.unload() → safe_cuda_cleanup()
         aligner.load() → ... → aligner.unload() → safe_cuda_cleanup()
```

Within Block 2, the generator and aligner are loaded/unloaded sequentially
(never simultaneously) via the orchestrator's VRAM swap pattern.

### Minimum GPU Requirements

| Configuration | Minimum VRAM |
|---------------|-------------|
| Qwen3-ASR-1.7B + ForcedAligner (sequential) | ~6 GB |
| Qwen3-ASR-0.6B + ForcedAligner (sequential) | ~4 GB |
| With ClearVoice enhancement (Block 1) | +2 GB (released before Block 2) |
| With BS-RoFormer enhancement (Block 1) | +4 GB (released before Block 2) |

---

## 12. Alignment Sentinel & Step-Down Retry

### Alignment Sentinel

The sentinel detects **alignment collapse** — when the ForcedAligner produces
degenerate timestamps (all words clustered in a tiny time span). This can happen
with repetitive content (e.g., "うん、うん、うん") or very short audio.

**Source**: `whisperjav/modules/alignment_sentinel.py`

**Functions**:
- `assess_alignment_quality(words, scene_duration)` → status dict
- `redistribute_collapsed_words(words, duration, regions)` → corrected words

**Collapse detection criteria** (from `assess_alignment_quality`):
- Word span < 0.5s AND degenerate ratio > 40%
- A word is "degenerate" if its duration < 0.01s

**Recovery strategies**:
- **Strategy C (VAD-guided)**: When Phase 4 VAD regions are available, distributes
  words proportionally across speech regions. Used when `speech_regions` is provided.
- **Strategy B (Proportional)**: Distributes words evenly across the scene duration.
  Used when no VAD data is available.

### Step-Down Retry

When alignment collapses on a scene, the orchestrator can retry with tighter
temporal framing:

```
Pass 1: Normal framing → COLLAPSED on scene 5, 12, 17
Pass 2: Reframe collapsed scenes with fallback_max_group_s (6.0s)
         → Re-generate + Re-align → Replace if improved
```

**Configuration**:
- `stepdown_enabled` (default: `True`) — Enable/disable
- `stepdown_fallback_group` (default: `6.0`) — Tighter max group for retry
- Only works with framers that support `reframe()` (FullSceneFramer, VadGroupedFramer)

---

## 13. Timestamp Resolution Modes

Controlled by `--qwen-timestamp-mode`. Applied during hardening (Step 9).

| Mode | Value | Behavior |
|------|-------|----------|
| **Aligner + Interpolation** | `aligner_interpolation` | Default in pipeline constructor. Keeps valid aligner timestamps; interpolates null timestamps using character-length-based math between anchors. |
| **Aligner + VAD Fallback** | `aligner_vad_fallback` | Default in CLI. Keeps valid aligner timestamps; distributes null timestamps proportionally across scene duration. |
| **Aligner Only** | `aligner_only` | Raw aligner output. Null timestamps remain as zeros. For diagnostics. |
| **VAD Only** | `vad_only` | Discards aligner timestamps entirely. Distributes all segments proportionally by character count. |

---

## 14. Deprecated / Removed Features

### Removed in Phase 4 (Strangulation)

| Feature | Status | Replacement |
|---------|--------|-------------|
| `InputMode.CONTEXT_AWARE` coupled path | **Deleted** | Assembly + `full-scene` framer |
| `InputMode.VAD_SLICING` coupled path | **Deleted** | Assembly + `vad-grouped` framer |
| `QwenASR.transcribe()` via `stable_whisper.transcribe_any()` | **Deleted** | Orchestrator (generate → clean → align) |
| `_transcribe_speech_regions()` method | **Deleted** | Orchestrator handles VAD groups |
| `_offset_result_timestamps()` method | **Deleted** | Word merging in orchestrator |

### Deprecated (accepted but mapped)

| Parameter | Accepted Value | Maps To |
|-----------|---------------|---------|
| `--qwen-input-mode context_aware` | `context_aware` | `assembly` + `--qwen-framer full-scene` |
| `--qwen-input-mode vad_slicing` | `vad_slicing` | `assembly` + `--qwen-framer vad-grouped` |
| `--qwen-japanese-postprocess` | — | No effect. Always `False` for Qwen3. |

### Not Yet Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| Phase 8 Sanitisation | Skipped | No Qwen-specific sanitizer. Stitched SRT passes through directly. |
| vLLM backend for TextGenerator | Deferred | Architecture ready (protocol-based), integration not built. |
| Assembly step-down per-scene error recovery | Partial | Orchestrator retries collapsed scenes, but no per-frame error recovery. |
| WhisperSegmentFramer | Deferred | Would allow using Whisper draft boundaries as temporal frames. |

---

## Appendix: Default Path Summary

### CLI Direct Mode

```
whisperjav video.mp4 --mode qwen
```

Default wiring:
```
framer:     full-scene
scene:      semantic (min=12s, max=48s)
enhancer:   none
segmenter:  ten (max_group=6.0s)
cleaner:    qwen3
aligner:    Qwen3-ForcedAligner-0.6B
timestamp:  aligner_interpolation (CLI: aligner_vad_fallback)
stepdown:   enabled (fallback=6.0s)
```

### GUI Ensemble Mode (Pass 2 = Qwen)

```
framer:     full-scene (GUI default) / vad-grouped (ensemble DEFAULT_QWEN_PARAMS)
scene:      semantic
enhancer:   none
segmenter:  ten
timestamp:  aligner_vad_fallback
stepdown:   enabled
```

**Note**: There is a discrepancy between the GUI HTML default (`full-scene`) and
the ensemble `DEFAULT_QWEN_PARAMS` default (`vad-grouped`). The ensemble default
takes precedence because `DEFAULT_QWEN_PARAMS` is applied before GUI overrides.
If the user doesn't change the framer dropdown, the GUI sends `full-scene` which
overrides the ensemble default.
