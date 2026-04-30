# VAD Ground-Truth Analyser

Developer / advanced-user utility for comparing WhisperJAV speech-segmenter
backends side-by-side on a single media file. Produces an interactive Plotly
HTML visualization plus JSON and CSV reports.

**Ground truth is optional.** When a reference SRT is provided, the tool
computes frame-level F1/precision/recall, segment IoU, boundary drift, and
missed/false-alarm percentages per backend. When no SRT is provided, the
tool still runs all backends and produces an **inter-backend agreement
matrix** (pairwise F1) as a consensus proxy.

---

## When to use it

- Evaluating a new VAD backend addition (e.g. we used this to measure
  `whisperseg` against existing backends on JAV audio)
- Triaging a suspected VAD regression on a specific media file
- Choosing a sensitivity preset for a specific audio style
- Spot-checking backend disagreement when you don't have ground truth

---

## Installation

Nothing extra if you already have the tools WhisperJAV needs at runtime
(`onnxruntime`, `soundfile`, `numpy`, `plotly`). If missing:

```bash
pip install plotly>=5.0.0 soundfile numpy scipy
# optional: whisperseg backend
pip install whisperjav[whisperseg]
```

`ffmpeg` must be on PATH for video input (audio files load directly through
`soundfile`).

---

## Quick start

### With ground truth

```bash
python -m tools.vad_groundtruth_analyser video.mp4 --ground-truth gt.srt
```

Produces three files next to `video.mp4`:

- `video.vad_analysis.html` — interactive multi-track timeline
- `video.vad_analysis.json` — full report with per-backend metrics
- `video.vad_analysis.csv` — one row per backend, easy to concatenate across runs

### Without ground truth

```bash
python -m tools.vad_groundtruth_analyser video.mp4
```

Skips per-backend metrics; still produces the HTML, JSON (with `agreement`
matrix), and CSV (metric columns empty).

### Custom backends and sensitivity

```bash
python -m tools.vad_groundtruth_analyser audio.wav \
    --backends whisperseg,silero-v6.2 \
    --sensitivity balanced
```

### CSV-only (for batch collection / CI)

```bash
python -m tools.vad_groundtruth_analyser video.mp4 -g gt.srt \
    --no-html --no-json
```

---

## CLI reference

```
python -m tools.vad_groundtruth_analyser MEDIA [options]

positional:
  MEDIA                             audio or video file (ffmpeg-extracted if video)

optional:
  -g, --ground-truth PATH           SRT; every entry = one speech segment
  -b, --backends LIST               comma-separated backend names
                                    (default: silero-v3.1,silero-v6.2,ten,whisperseg)
  -s, --sensitivity {conservative,balanced,aggressive}
                                    (default: aggressive)
  -o, --output-dir DIR              (default: alongside media)
  --frame-ms INT                    frame grid for frame-level metrics (default: 10)
  --iou-match-threshold FLOAT       min IoU to count as segment match (default: 0.1)
  --waveform-points INT             waveform downsample for HTML (default: 10000)
  --timeout SEC                     per-backend timeout (default: 300)
  --no-html / --no-json / --no-csv  skip specific outputs
  --title STR                       custom HTML title
  -v, --verbose                     DEBUG logging
  --version
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0    | success (>=1 backend produced valid output) |
| 1    | media or GT load failure |
| 2    | all backends failed / unavailable |
| 3    | CLI argument error |

---

## Output formats

### HTML

A self-contained interactive Plotly chart:

```
Row 0: Waveform (blue filled area)
Row 1: Ground truth (only if provided) — bright green bars
Row 2..N: One row per backend — distinct color per backend
Bottom: range slider for zoom / pan; shared x-axis
```

Features: zoom with range slider, hover tooltips on every segment, click
legend to toggle a track, scroll to zoom, dark theme for contrast.

### JSON

Canonical artefact. `schema_version: "1.0.0"` pinned. Top-level structure:

```json
{
  "schema_version": "1.0.0",
  "media_file": "...",
  "audio_duration_sec": 14.016,
  "sample_rate": 16000,
  "sensitivity": "aggressive",
  "frame_ms": 10,
  "generated_at": "2026-04-22T14:37:54+00:00",
  "ground_truth": {...} | null,
  "backends": {
    "whisperseg": {
      "name": "whisperseg",
      "display_name": "WhisperSeg (JA-ASMR)",
      "available": true, "success": true, "error": null,
      "processing_time_sec": 8.15,
      "parameters": {"threshold": 0.25, ...},
      "segments": [{"start_sec": ..., "end_sec": ..., "confidence": ..., "metadata": {...}}],
      "num_segments": 3,
      "coverage_ratio": 0.932,
      "metrics": {
        "frame_f1": 0.972, "frame_precision": 0.945, "frame_recall": 1.0,
        "iou_mean": ..., "iou_median": ..., "iou_per_gt": [...],
        "onset_drift_mean_ms": ..., "offset_drift_mean_ms": ...,
        "missed_speech_pct": 0.0, "false_alarm_pct": 5.5,
        "num_matched_segments": 3, "num_unmatched_gt": 0
      }
    },
    "...": "..."
  },
  "agreement": {
    "backend_order": ["silero-v3.1", "silero-v6.2", "ten", "whisperseg"],
    "pair_f1": [[1.0, ...], ...],
    "consensus_coverage_pct": 82.9
  }
}
```

`metrics` is `null` in GT-less mode or when a backend failed.

### CSV

One row per backend. Fixed schema — safe to concatenate across runs:

```
media_file, media_duration_sec, sensitivity, ground_truth_available,
backend, available, success, error, processing_time_sec,
num_segments, coverage_pct,
frame_f1, frame_precision, frame_recall,
iou_mean, iou_median,
onset_drift_mean_ms, offset_drift_mean_ms,
missed_speech_pct, false_alarm_pct
```

GT-less rows leave the metric columns empty.

---

## Metrics glossary

**Frame F1 (headline)** — harmonic mean of precision and recall on a 10 ms
frame grid. GT and backend speech regions are converted to binary masks; F1
is computed over those masks. Range `[0, 1]`.

**Frame precision** — fraction of backend-speech frames that are actually
GT-speech. High precision = few false alarms.

**Frame recall** — fraction of GT-speech frames that the backend correctly
covered. High recall = few missed speech regions.

**IoU (Intersection over Union)** — per segment pair: `overlap_seconds /
union_seconds`. `iou_per_gt` has one IoU per GT segment (0.0 if no matching
backend segment); `iou_mean` and `iou_median` aggregate across matched
pairs.

**Boundary drift** — for matched GT/backend pairs, the mean and median of
`|gt_start − backend_start|` (onset) and `|gt_end − backend_end|` (offset),
in milliseconds.

**Missed speech %** — GT-speech frames not covered by any backend segment,
as a percentage of GT-speech frames. Complementary to recall expressed from
a different angle.

**False alarm %** — backend-speech frames with no GT overlap, as a
percentage of backend-speech frames.

**Inter-backend agreement (GT-less mode)** — pairwise F1 between all
successful backends. Symmetric, diagonal = 1.0. `consensus_coverage_pct`
is the percentage of frames where **>=2** backends simultaneously claim
speech — a proxy for "reliable" speech regions when ground truth is absent.

---

## Ground-truth SRT conventions

- Every SRT entry is treated as a speech segment regardless of its text
  content.
- Text is preserved only for HTML hover tooltips; it does **not** influence
  any metric.
- Overlapping SRT entries are union-ed on the speech mask.
- Entries with `end <= start` are skipped with a warning.
- If the SRT parses to zero valid entries the tool auto-degrades to
  GT-less mode (same as if no `--ground-truth` was passed).

---

## Programmatic use (advanced)

```python
from pathlib import Path
from tools.vad_groundtruth_analyser import analyse
from tools.vad_groundtruth_analyser.reporter import write_json, write_csv, write_html

report, waveform, gt_segs = analyse(
    media_path=Path("audio.wav"),
    gt_path=Path("gt.srt"),                         # or None
    backends=["whisperseg", "silero-v6.2"],
    sensitivity="balanced",
    frame_ms=10,
    timeout_sec=120,
)
# report is an AnalysisReport dataclass — serialize or inspect directly.
f1 = report.backends["whisperseg"].metrics.frame_f1
write_json(report, Path("out.json"))
write_html(report, waveform, Path("out.html"), gt_segments=gt_segs or None)
```

Lower-level primitives (for custom metric work):

```python
from tools.vad_groundtruth_analyser.runner import BackendRunner
from tools.vad_groundtruth_analyser.metrics import (
    compute_timing_metrics, compute_agreement_matrix, segments_to_mask,
)
from tools.vad_groundtruth_analyser.loader import parse_srt, load_media_audio
```

---

## Troubleshooting

**`ffmpeg is required to extract audio from video files.`**
Install ffmpeg and ensure it's on PATH. Audio-only files (WAV/MP3/FLAC/OGG/
M4A/AAC/OPUS) skip this step.

**`WhisperSeg requires onnxruntime.`**
`pip install whisperjav[whisperseg]` (or `[whisperseg-gpu]` for CUDA).

**All backends show `N/A`.**
None of the backends have their dependencies installed in the current env.
Run `python -c "from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory;
print(SpeechSegmenterFactory.get_available_backends())"` to see availability
per backend.

**HTML is ~5 MB.**
Plotly JS is embedded for offline use. Size scales with segment count not
audio duration. Disable with `--no-html` if you only need CSV/JSON.

**Japanese text in GT SRT renders as mojibake.**
Encoding auto-detection tries UTF-8, UTF-8-BOM, CP932, Latin-1 in order.
If your SRT is in a different encoding, re-save it as UTF-8.

---

## Limitations (v1)

- Speech-enhancer chaining is **not yet supported** — backends see raw
  audio only. Reserved for a future release.
- Backends run **sequentially** (not in parallel) to avoid GPU contention
  when several backends share the same device.
- 30 s chunking in WhisperSeg has no inter-chunk overlap; boundary
  artefacts at exact 30 s multiples are possible on long files. Use
  `--waveform-points` to zoom in on suspect regions.
- Only metrics; no inference of **why** a backend disagrees with GT
  (separate forensic tool territory — see `tools/forensic_csv_generator.py`).

---

## See also

- `whisperjav/modules/speech_segmentation/` — backend implementations
- `scripts/visualization/` — DAW-style WhisperJAV-pipeline output visualizer
- `whisperjav/bench/` — subtitle-text benchmark (CER + text IoU); complementary focus
- `tools/forensic_csv_generator.py`, `tools/fw_diagnostic_suite.py` — other dev utilities
