# Qwen Pipeline Diagnostic & Benchmarking System — Requirements

## Schema Version: 1.0.0

## Purpose

Compare Qwen pipeline input modes (assembly, context_aware, vad_slicing) against
ground-truth SRT to determine which produces the best text and timing accuracy.

---

## Top-Level Questions

1. Which input mode produces the most accurate **text** (lowest CER)?
2. Which input mode produces the most accurate **timing** (highest IoU)?
3. Which **scenes** are problematic, and why?
4. What **pipeline decisions** (sentinel recovery, VAD fallback, sanitization) affected each scene?

---

## Per-Test Aggregate Metrics

| Metric | Source |
|--------|--------|
| Input mode, model, settings | `_master.json` → `stages.asr.input_mode` |
| Overall CER vs ground truth | Computed by bench utility |
| Overall timing IoU score | Computed by bench utility |
| Total subtitles (produced vs GT) | `_master.json` + SRT |
| Processing time | `_master.json` → `stages.*.time_sec` |
| Sentinel collapses/recoveries | `_master.json` → `stages.asr.alignment_*` |
| Sanitization stats | `_master.json` → `stages.sanitisation.stats` |

## Per-Scene Diagnostics

### New artifact: `scene_NNNN_diagnostics.json`

Saved in `raw_subs/` alongside existing artifacts (debug mode only).

```json
{
  "schema_version": "1.0.0",
  "scene_index": 0,
  "scene_start_sec": 0.0,
  "scene_end_sec": 77.152,
  "scene_duration_sec": 77.152,
  "input_mode": "vad_slicing",

  "sentinel": {
    "status": "OK",
    "assessment": {
      "word_count": 45,
      "char_count": 156,
      "word_span_sec": 72.3,
      "coverage_ratio": 0.937,
      "aggregate_cps": 2.2,
      "anchor_sec": 0.5
    },
    "recovery": null
  },

  "timing_sources": {
    "aligner_native": 8,
    "vad_fallback": 3,
    "interpolated": 1,
    "total_segments": 12
  },

  "vad_regions": [
    {"start": 0.5, "end": 8.2},
    {"start": 12.1, "end": 15.8}
  ],

  "asr_stats": {
    "raw_char_count": 156,
    "clean_char_count": 148,
    "token_budget": 1543
  }
}
```

### Field Definitions

- **sentinel.status**: `"OK"` or `"COLLAPSED"` — from `assess_alignment_quality()`
- **sentinel.assessment**: Full metrics dict from sentinel
- **sentinel.recovery**: Present only if collapsed. `{strategy: "proportional"|"vad_guided", words_redistributed: N}`
- **timing_sources**: Counts per timing method. `aligner_native = total - vad_fallback - interpolated`
- **vad_regions**: Phase 4 speech regions for this scene
- **asr_stats**: Character counts and token budget

---

## Cross-Test Comparison

- Side-by-side CER per scene across all tests
- Side-by-side timing IoU per scene across all tests
- Ranking of modes per scene (best → worst)
- Problem scenes: where all modes struggle (CER > threshold)
- Differentiating scenes: where modes diverge most

---

## Existing vs New Data

| Data | Available? | Source |
|------|-----------|--------|
| Input mode, model, settings | YES | `_master.json` |
| Aggregate sentinel stats | YES | `_master.json` |
| Aggregate sanitization stats | YES | `_master.json` |
| Processing times | YES | `_master.json` |
| Scene boundaries | YES | `scenes/*_semantic.json` |
| Raw ASR text per scene | YES | `raw_subs/*_assembly_raw.txt` or `*_master.txt` |
| Clean ASR text per scene | PARTIAL | `*_assembly_clean.txt` (assembly only) |
| Word-level timestamps per scene | YES | `raw_subs/*_merged.json` |
| Per-scene SRT | YES | `scene_srts/*_scene_NNNN.srt` |
| Sanitization actions | YES | `raw_subs/*_stitched.artifacts.srt` |
| Final SRT | YES | `*.ja.whisperjav.srt` |
| **Per-scene sentinel assessment** | **NEW** | `scene_NNNN_diagnostics.json` |
| **Per-scene sentinel recovery** | **NEW** | `scene_NNNN_diagnostics.json` |
| **Per-scene timing source breakdown** | **NEW** | `scene_NNNN_diagnostics.json` |
| **VAD speech regions per scene** | **NEW** | `scene_NNNN_diagnostics.json` |
