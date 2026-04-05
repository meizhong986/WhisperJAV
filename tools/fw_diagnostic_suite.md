# Faster-Whisper Diagnostic Suite

## Purpose

Reproduces WhisperJAV's exact production audio flow (audio extraction, scene detection, VAD grouping via WhisperJAV modules), then calls Faster-Whisper directly on each VAD group with multiple parameter variants. Captures per-group metadata, per-segment metrics (`avg_logprob`, `no_speech_prob`, `compression_ratio`, `temperature`), and whether each transcribe() call yielded segments.

Used during v1.8.10-hf development to isolate Faster-Whisper's per-chunk behavior from WhisperJAV's post-processing layers.

## Script

`tools/fw_diagnostic_suite.py`

## Invocation

```bash
cd <WhisperJAV repo root>  # required so whisperjav.modules imports resolve
python tools/fw_diagnostic_suite.py --run all
```

### Arguments

| Flag | Default | Purpose |
|------|---------|---------|
| `--input` | `C:\BIN\git\WhisperJav_V1_Minami_Edition\test_media\293sec-S01E04-scene4.mkv` | Input media file |
| `--outdir` | `F:\MEDIA_DLNA\SONE-853\DIAG_FW` | Output directory (created at runtime) |
| `--run` | `A` | Run IDs: single (`A`), comma-separated (`A,C,E`), or `all` |
| `--model` | `large-v2` | Whisper model name |
| `--device` | `cuda` | Faster-Whisper device |
| `--compute-type` | `float16` | Faster-Whisper compute type |
| `--language` | `ja` | Transcription language |
| `--segmenter` | `silero-v6.2` | Speech segmenter backend (e.g. `silero-v6.2`, `ten`, `silero`) |
| `--skip-prep` | flag | Reuse cached scenes + vad_groups.json from outdir |

## Dependencies

- `faster_whisper` (for `WhisperModel` and `transcribe()`)
- WhisperJAV modules: `whisperjav.modules.audio_extraction`, `whisperjav.modules.scene_detection_backends`, `whisperjav.modules.speech_segmentation`
- Must run from WhisperJAV project root (or with PYTHONPATH) for imports to resolve

## Pipeline (4 Phases)

### Phase 1 — Audio Extraction
Uses `whisperjav.modules.audio_extraction.AudioExtractor` to extract 16kHz mono WAV from input media.

### Phase 2 — Scene Detection
Uses `SceneDetectorFactory.create("semantic")` (falls back to auditok if sklearn missing). Saves scene WAVs to `<outdir>/scenes/` and metadata to `<outdir>/scenes.json`.

### Phase 3 — VAD Grouping
Uses `SpeechSegmenterFactory.create(<segmenter>)` where segmenter is set by `--segmenter`. Saves VAD group metadata to `<outdir>/vad_groups.json`.

### Phase 4 — Faster-Whisper Transcription
Loads `faster_whisper.WhisperModel` ONCE with specified device/compute-type. Runs each PARAM_VARIANT sequentially against all VAD groups using the same model instance.

## Parameter Variants (8 Runs)

All runs share the same base params (mirrors WhisperJAV production kwargs, 21 fields):

```python
task="transcribe", language="ja", suppress_blank=True, without_timestamps=False,
max_initial_timestamp=0.0, condition_on_previous_text=False, word_timestamps=True,
chunk_length=30, repetition_penalty=1.3, no_repeat_ngram_size=3,
multilingual=False, log_progress=False,
vad_filter=False,              # critical: audio is pre-segmented by WhisperJAV
vad_parameters={...},           # unused when vad_filter=False, included for kwargs parity
compression_ratio_threshold=2.4
```

### Per-Variant Overrides

| Variant | beam_size | best_of | patience | temperature | log_prob_threshold | no_speech_threshold | Other |
|---------|-----------|---------|----------|-------------|-------------------|--------------------|----|
| **A1** | 4 | 3 | 2.5 | 0.0 | **-1.0** | 0.9 | — |
| **A2** | 4 | 3 | 2.5 | 0.0 | **-1.0** | **0.6** | — |
| **A** | 4 | 3 | 2.5 | 0.0 | -1.3 | 0.9 | — |
| **B** | **1** | **1** | **1.0** | 0.0 | -1.3 | 0.9 | — |
| **C** | 4 | 3 | 2.5 | **[0.0,0.2,0.4,0.6,0.8,1.0]** | -1.3 | 0.9 | Temperature fallback |
| **D** | 4 | 3 | 2.5 | 0.0 | -1.3 | 0.9 | `condition_on_previous_text=True` |
| **E** | 4 | 3 | 2.5 | 0.0 | -1.3 | **0.99** | — |
| **F** | **5** | **5** | **1.0** | **[0.0,0.2,0.4,0.6,0.8,1.0]** | -1.0 | 0.6 | Faster-Whisper library defaults |

**Execution order with `--run all`**: A1 → A2 → A → B → C → D → E → F (dictionary insertion order).

## Verbose Logging

At import time, `enable_verbose_dependencies()` sets DEBUG level for two loggers:

- `faster_whisper` — emits internal messages including `"No speech threshold is met (%f > %f)"` when the skip filter fires
- `whisperjav` — emits internal messages from WhisperJAV components

Faster-Whisper does NOT support a `verbose=True` parameter on `transcribe()`. Debug output is captured via the Python logger.

Faster-Whisper stdout/stderr are also redirected per-group to `<outdir>/run_<X>_fw_verbose.log` via `contextlib.redirect_stdout/redirect_stderr`.

## Output Files

All outputs go to `--outdir` (default `F:\MEDIA_DLNA\SONE-853\DIAG_FW`).

### Saved Once (shared across runs)

| File | Content |
|------|---------|
| `293sec-S01E04-scene4_extracted.wav` | Extracted 16kHz mono audio |
| `scenes.json` | Scene metadata from scene detector |
| `scenes/*.wav` | Per-scene audio files |
| `vad_groups.json` | VAD group list: start_abs, end_abs, duration, scene_idx, group_idx, n_sub_segments |

### Per-Variant

| File | Content |
|------|---------|
| `run_<X>_results.csv` | One row per VAD group |
| `run_<X>_summary.json` | Aggregate stats |
| `run_<X>_fw_verbose.log` | Per-group `[result]` lines + redirected FW stdout/stderr |

### CSV Columns

```
scene_idx, group_idx, start_abs, end_abs, duration,
audio_peak, audio_rms,
detected_language, language_probability, info_duration_after_vad,
n_segments_yielded,
first_seg_avg_logprob, first_seg_no_speech_prob, first_seg_compression_ratio, first_seg_temperature,
raw_text_concat, error
```

When `n_segments_yielded=0`, the `first_seg_*` columns are empty. Empty yield means Faster-Whisper's generator produced no segments for that VAD group — either the internal skip filter fired, or the model produced no output.

### Summary JSON Schema

```json
{
  "run_name": "A1",
  "params": {...},
  "total_groups": N,
  "groups_with_output": N,
  "groups_empty": N,
  "empty_pct": float,
  "mean_avg_logprob": float | null,
  "mean_no_speech_prob": float | null,
  "n_segments_total": N,
  "elapsed_sec": float
}
```

`mean_avg_logprob` and `mean_no_speech_prob` are `null` when all groups are empty.

## Example Usage

Single variant:
```bash
python tools/fw_diagnostic_suite.py --run A
```

All variants:
```bash
python tools/fw_diagnostic_suite.py --run all
```

Comma-separated:
```bash
python tools/fw_diagnostic_suite.py --run A,C,F
```

Different segmenter:
```bash
python tools/fw_diagnostic_suite.py --run all --segmenter ten
```

Reuse prepared scenes/VAD groups:
```bash
python tools/fw_diagnostic_suite.py --run all --skip-prep
```

## Observed Behavior (2026-04-05 runs, 293sec-S01E04-scene4.mkv)

| Segmenter | Groups Created | Per-Run Result |
|-----------|---------------|----------------|
| silero-v6.2 | 13 | All 8 variants: 0% empty, all groups produced output |
| ten | 23 | A1: 87% empty (3 groups captured). A2/A/B/C/D/E/F: 100% empty (0 captured) |

## Behavior Notes

1. Model loaded ONCE before the variant loop — all selected runs use the same `WhisperModel` instance
2. Scene detection defaults to semantic, falls back to auditok if sklearn missing
3. Segmenter params constructed internally with `chunk_threshold_s: 1.0, max_group_duration_s: 29.0` (may override sensitivity-based YAML defaults)
4. `enable_verbose_dependencies()` runs at import time — debug logging active before `main()` executes
