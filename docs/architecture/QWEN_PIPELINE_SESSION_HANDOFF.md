# Qwen Pipeline Session Handoff

## Date: 2026-02-06
## Session Status: CLI Sync Complete, Ready for 3rd Flow Design

---

## Completed Work

### ✅ Investigation Complete
- Root cause identified: CLI defaults were overriding pipeline class defaults
- Confirmed working configuration: `vad_slicing` + `semantic` + `ten` + `aligner_vad_fallback`

### ✅ CLI Defaults Synchronized
All four CLI defaults now match the proven working configuration:

| Parameter | Default Value |
|-----------|---------------|
| `--qwen-input-mode` | `vad_slicing` |
| `--qwen-scene` | `semantic` |
| `--qwen-segmenter` | `ten` |
| `--qwen-timestamp-mode` | `aligner_vad_fallback` |

### ✅ Verified Working
Simple command now produces good results:
```bash
whisperjav file.wav --mode qwen --qwen-language Japanese
```

---

## Next Task: 3rd Pipeline Flow

### Current Pipeline Flows (InputMode enum in qwen_pipeline.py)

**Flow 1: VAD_SLICING (Current Default)**
```
Audio → Scene Detection (semantic) → Enhancement → VAD Segmentation (TEN) → ASR per segment → Stitch → Sanitize
```
- Chops audio into small VAD segments before ASR
- Stable timestamps via `aligner_vad_fallback`
- Proven working, now the default

**Flow 2: CONTEXT_AWARE (Experimental)**
```
Audio → Scene Detection → Enhancement → ASR on full scene (~180s) → Stitch → Sanitize
```
- Feeds larger chunks to preserve LALM context
- Requires safe chunking (150-210s) for ForcedAligner 300s limit
- Still experimental, needs refinement

**Flow 3: TBD**
- To be designed in next session
- Considerations:
  - Hybrid approach?
  - Different chunking strategies for different content types?
  - Adaptive based on audio characteristics?

---

## Key Technical Context

### Qwen3-ASR Characteristics
- Large Audio-Language Model (LALM) - benefits from longer context
- ForcedAligner has 300s architectural limit
- Word-level timestamps via aligner, fallback to VAD boundaries

### Scene Detection Methods
- `semantic` - True MERGE logic, guarantees min_duration (recommended)
- `auditok` - Silence-based, only FILTER
- `silero` - VAD-based, only FILTER

### TimestampMode Options
- `aligner_vad_fallback` - Stable, uses VAD boundaries when aligner fails (default)
- `aligner_interpolation` - Mathematically interpolates null timestamps
- `aligner_only` - No fallback, keeps null timestamps
- `vad_only` - Discards aligner timestamps entirely

### TEN VAD Parameters
- `max_group_duration_s` - Default 29.0s (CLI: `--qwen-max-group-duration`)
- Groups adjacent speech segments up to this duration

---

## Key Source Files

| File | Purpose |
|------|---------|
| `whisperjav/pipelines/qwen_pipeline.py` | Main pipeline, InputMode enum, 8-phase flow |
| `whisperjav/main.py` | CLI argument definitions (qwen args: lines 420-490) |
| `whisperjav/modules/scene_detection.py` | DynamicSceneDetector, semantic adapter |
| `whisperjav/modules/speech_segmentation/backends/ten.py` | TEN VAD implementation |
| `whisperjav/modules/qwen_asr.py` | Qwen3-ASR model wrapper |

---

## Git State

### Branch
`dev_qwenasr_transformers`

### Uncommitted Changes
```
M  whisperjav/main.py              # CLI defaults synced
M  whisperjav/pipelines/qwen_pipeline.py  # Pipeline defaults, segmenter param
M  whisperjav/modules/scene_detection.py  # Semantic adapter param fix
```

### Recent Commits
```
a18948b Add Japanese-specific post-processing for Qwen pipeline
8d3d0e4 Uninstall torechdec before installing other components
8622c2d Implement Context-Aware Chunking for Qwen3-ASR pipeline
```

---

## Session Restart Instructions

### Prompt for New Session
```
We're continuing Qwen pipeline development.

Completed:
- CLI defaults synchronized to working config (vad_slicing, semantic, ten, aligner_vad_fallback)
- Investigation complete, root cause was CLI/pipeline default mismatch

Next task:
- Design and implement 3rd pipeline flow for Qwen
- Then optimizations, enhancements, QA checks

Reference: docs/architecture/QWEN_PIPELINE_SESSION_HANDOFF.md

Please read the handoff document, then let's discuss the 3rd flow design.
```

### First Steps in New Session
1. Read this handoff document
2. Read `qwen_pipeline.py` lines 52-108 (InputMode and TimestampMode enums)
3. Discuss 3rd flow requirements and design
4. Implement after design approval

---

## Test Files
- `test_media/S01E04-scene4_extracted.wav` - Reference test file (293.7s, Japanese)
- `test_output_e2e/` - Output directory with verified good results
