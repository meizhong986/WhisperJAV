# ADR-006: Decoupled Subtitle Pipeline Architecture

| Field        | Value                                      |
|--------------|--------------------------------------------|
| **Status**   | IMPLEMENTED (Phases 0–6 complete, 2026-02-17) |
| **Date**     | 2026-02-16                                 |
| **Authors**  | MK (product vision), Claude (architecture) |
| **Supersedes** | Extends ADR-004 (dedicated Qwen pipeline) |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Context & Problem Statement](#2-context--problem-statement)
3. [Current Architecture Audit](#3-current-architecture-audit)
4. [Architectural Comparison: Assembly vs VAD_SLICING](#4-architectural-comparison-assembly-vs-vad_slicing)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Proposed Architecture](#6-proposed-architecture)
7. [Protocol Specifications](#7-protocol-specifications)
8. [Data Types & Contracts](#8-data-types--contracts)
9. [Orchestrator Design](#9-orchestrator-design)
10. [Audit Finding Resolutions](#10-audit-finding-resolutions)
11. [Current Mode Mapping](#11-current-mode-mapping)
12. [New Workflows Enabled](#12-new-workflows-enabled)
13. [What Doesn't Change](#13-what-doesnt-change)
14. [Future Expansion](#14-future-expansion)

---

## 1. Executive Summary

This ADR proposes a **model-agnostic, component-based subtitle generation pipeline** that
separates the concerns of temporal framing, text generation, text cleaning, and text alignment
into independent, swappable protocol domains. The design is motivated by:

1. **The timestamp problem**: Many high-quality ASR models produce excellent text but no
   timestamps. To produce SRT subtitles, timestamps must come from somewhere else.
2. **Aligner unreliability**: The Qwen3 ForcedAligner frequently collapses on JAV content,
   mapping all words to a ~100ms window instead of the actual speech duration.
3. **Experimentation need**: Finding the optimal approach requires mixing and matching
   different temporal framing, transcription, and alignment strategies.
4. **Reusability**: The solution should apply to any ASR model lacking built-in timestamps,
   not just Qwen3-ASR, including transformers-based and vLLM-served models.

The architecture introduces **six protocol domains** (four existing, two new) and a
**TemporalFramer** concept that cleanly decouples "when does dialogue happen?" from "what
was said?" — enabling aligner-free workflows where reliable timestamps come from external
sources (Whisper segments, existing SRTs, VAD grouping).

---

## 2. Context & Problem Statement

### 2.1 The Fundamental Challenge

Many state-of-the-art ASR models excel at text transcription but lack built-in timestamp
output. To produce SRT subtitle files, these models need auxiliary timestamp generation
capabilities. The pipeline must therefore handle two separate concerns:

- **Text generation**: "What was said?" (high-quality transcription)
- **Timestamp assignment**: "When was each word/segment said?" (accurate timing)

### 2.2 Current Experience with Qwen3-ASR

Qwen3-ASR is a representative example of this class of models:

- **Text quality**: Excellent Japanese transcription, superior to Whisper for JAV content
- **Timestamp mechanism**: Offers a ForcedAligner model that attempts to map text to audio
  and output predicted word timestamps
- **Reliability**: The ForcedAligner is **unreliable for JAV content** — it frequently
  "collapses", mapping all words to a ~100ms window instead of spreading them across the
  actual speech duration. This is what the Alignment Sentinel exists to detect.

### 2.3 Current Pipeline Approaches (Three Input Modes)

The current Qwen pipeline (`qwen_pipeline.py`, ~2178 lines) implements three input modes:

| Mode | Approach | Strength | Weakness |
|------|----------|----------|----------|
| **ASSEMBLY** | Decoupled text generation + alignment. Batch ASR text-only → sanitize → VRAM swap → batch align → reconstruct | Mid-pipeline text cleaning, VRAM efficiency, batch processing, future vLLM socket | No timestamp resolution, no boundary clamping, no step-down retry |
| **CONTEXT_AWARE** | Coupled ASR+Aligner on full scenes (30-90s), batch_size=1 | Simple, leverages full scene context | Applies wrong post-processor (Whisper-era), hardcoded interpolation only |
| **VAD_SLICING** | VAD segments (up to 29s groups via TEN VAD), coupled ASR per group | Adaptive step-down, 4-mode timestamp resolution, double boundary clamping, rich diagnostics | No mid-pipeline text cleaning, relies on Whisper-era post-processor |

Each mode discovered important patterns, but the implementations are tangled together in
mode-specific code paths rather than composable components. The hardening features from
VAD_SLICING cannot be easily ported to Assembly because they operate in fundamentally
different coordinate systems.

### 2.4 Vision

Architect a **reusable, expandable, modular pipeline** that:

1. Works with **any ASR model** lacking built-in timestamps
2. Supports both **transformers** and **vLLM** backends
3. Allows **experimentation** with various temporal framing, generation, and alignment
   strategies
4. Makes the unreliable aligner **optional** when better timestamp sources are available
5. Follows the established **Protocol + Factory** pattern already used by three other
   domains in the codebase

---

## 3. Current Architecture Audit

### 3.1 Methodology

Code-driven analysis of four core files:
- `qwen_pipeline.py` (~2178 lines)
- `qwen_asr.py` (~1754 lines)
- `alignment_sentinel.py` (~417 lines)
- `assembly_text_cleaner.py` (~663 lines)

Every finding cites specific file:line references. Comments and docstrings were NOT trusted
— only executed code paths were analyzed.

### 3.2 Findings Summary

| # | Sev | ID | Title | File(s) | Lines | Description |
|---|-----|----|-------|---------|-------|-------------|
| 1 | CRIT | C1 | OOM retry uses stale closure | qwen_asr.py | 784, 977-997 | `_transcribe_with_regrouping` OOM handler unloads+reloads model, but the `qwen_inference` closure still references the OLD model via captured `qwen_model` variable. Old model not freed (ref held), new model loaded → double VRAM. Retry uses wrong model with old batch_size. |
| 2 | CRIT | C2 | Coupled modes apply wrong post-processor | qwen_pipeline.py, qwen_asr.py | 1780-1798, 964-973 | **Revised**: CONTEXT_AWARE and VAD_SLICING incorrectly apply the Whisper-era `JapanesePostProcessor` to Qwen3-ASR output (qwen_asr.py:964-973). Assembly is **correct** to skip it — it has its own Qwen-specific `AssemblyTextCleaner`. The Whisper-era post-processor was designed for Whisper's output characteristics, not Qwen3's. |
| 3 | HIGH | H1 | `_apply_vad_only_timestamps` produces overlapping subs | qwen_pipeline.py | 1973-1975 | All segments in a group receive identical `[0, group_duration]` range. Multiple segments stack at same timestamp. Should distribute proportionally by character count. |
| 4 | HIGH | H2 | Assembly mode doesn't apply timestamp resolution | qwen_pipeline.py | 1640-1798 | Assembly Step 8 skips `_apply_timestamp_interpolation()` and `_apply_vad_timestamp_fallback()`. Words with null timestamps (start=0, end=0) pass through unmodified. `timestamp_mode` setting silently ignored in assembly. |
| 5 | HIGH | H3 | `suppress_silence` may alter sentinel recovery timestamps | qwen_pipeline.py | 1872-1882 | `_reconstruct_from_words()` passes `suppress_silence=True` to `transcribe_any()`. stable-ts modifies word timestamps based on audio silence detection — potentially undoing sentinel's carefully computed VAD-guided redistribution. Affects both assembly and coupled sentinel recovery. |
| 6 | HIGH | H4 | Assembly diagnostics report word count not segment count | qwen_pipeline.py | 1769-1773 | `timing_sources.aligner_native` and `total_segments` both set to `n_words` (word count from merge). Measured before `_reconstruct_from_words()` so actual segment count after regrouping never recorded. |
| 7 | MED | M1 | Stale "300s" comment — actual limit is 180s | qwen_pipeline.py | 217 | Comment says "ForcedAligner's 300s architectural limit" but actual limit is 180s (`qwen_asr.py:217: MAX_FORCE_ALIGN_SECONDS = 180`). Code correctly uses 120s max. |
| 8 | MED | M2 | `_transcribe_without_aligner` returns `end=0.0` | qwen_asr.py | 1063 | Creates segment with `end: 0.0`. Downstream logic uses `seg.end <= 0.0` as "aligner returned NULL" heuristic. This segment would be incorrectly classified as collapsed. |
| 9 | MED | M3 | No `batch_size` override for assembly text generation | qwen_pipeline.py | 1505 | Assembly's stated benefit is higher batch_size (ASR-only uses less VRAM). But `QwenASR` constructed with same `_asr_config` (`batch_size=1` default). No mechanism for assembly-specific batch_size. |
| 10 | MED | M4 | Hallucination list access via private `_exact_lists` | assembly_text_cleaner.py | 576 | `getattr(remover, '_exact_lists', None)` accesses private attribute. If `HallucinationRemover` refactors, silently returns `None` → Stage 3 does nothing, no error. |
| 11 | MED | M5 | Redundant inline VRAM cleanup — not using `safe_cuda_cleanup()` | qwen_pipeline.py | 462-467, 799-804, 1542-1548, 1619-1624 | Same 6-line `gc.collect + torch.cuda.empty_cache` pattern repeated 4 times. `BasePipeline.cleanup()` uses `safe_cuda_cleanup()` from `gpu_utils`. Pipeline doesn't. |
| 12 | MED | M6 | `_check_audio_limits` runs redundantly for every VAD clip | qwen_asr.py | 674 | Called at start of every `transcribe()` call. In VAD_SLICING, runs for every speech region (50+ per scene). Each call loads librosa/soundfile to detect duration already known from segmenter. |
| 13 | MED | M7 | Timestamp modes only apply to VAD_SLICING | qwen_pipeline.py | 1202-1213, 669, 1640 | CONTEXT_AWARE hardcodes interpolation. Assembly ignores `timestamp_mode` entirely. Only VAD_SLICING respects user's `--timestamp-mode` setting. |
| 14 | MED | M8 | `suppress_silence` in coupled mode transcription | qwen_asr.py | 937-938 | Same as H3 but for primary coupled-mode path. All coupled-mode transcriptions have aligner timestamps potentially modified by stable-ts silence suppression. |
| 15 | LOW | L1 | Deprecated `_apply_vad_filter` never called | qwen_pipeline.py | 2116-2177 | Dead code. 62 lines. |
| 16 | LOW | L2 | Dead `cross_scene_context` branches | qwen_pipeline.py | 249, 593-599, 782-784, 1489-1491 | `self.cross_scene_context = False` hardcoded. All branches are dead code. |
| 17 | LOW | L3 | `_normalize_language_for_qwen` redundant second-pass | qwen_asr.py | 1655-1659 | Loop over `QWEN_SUPPORTED_LANGUAGES` is redundant — all names already in `LANGUAGE_TO_QWEN_MAP`. |
| 18 | LOW | L4 | `_map_language_code` fallback produces invalid ISO | qwen_asr.py | 1692 | "Unknown" → "un" (invalid ISO code). Cosmetic. |
| 19 | LOW | L5 | `_estimate_tokens` only used for warnings | qwen_asr.py | 481-505 | Not used for any functional logic. |
| 20 | LOW | L6 | Assembly context propagation differs from coupled | qwen_pipeline.py | 1489-1492, 593-601 | Moot — `cross_scene_context` flag is hardcoded `False`. |
| 21 | LOW | L7 | Diagnostics JSON only preserved in debug mode | qwen_pipeline.py | 975-976, 1802-1810 | Master metadata needs `--debug`. Assembly diagnostics cleaned up unless `--keep-temp`. |

**Counts by severity:**

| Severity | Count |
|----------|-------|
| CRITICAL | 2 |
| HIGH     | 4 |
| MEDIUM   | 8 |
| LOW      | 7 |
| **Total** | **21** |

---

## 4. Architectural Comparison: Assembly vs VAD_SLICING

### 4.1 Execution Flow Side-by-Side

**VAD_SLICING** (`_transcribe_speech_regions`, lines 987-1425):
```
Per scene:
  Read scene audio once (memory) → Per VAD group:
    ├─ Slice audio in-memory
    ├─ Write temp WAV per group
    ├─ asr.transcribe() [coupled: ASR+Aligner in VRAM together]
    ├─ Alignment Sentinel (assess on WhisperResult words)
    ├─ Recovery if COLLAPSED (VAD-guided or proportional)
    ├─ Timestamp resolution (4 modes)
    ├─ Clamp to group bounds [0, group_duration]
    ├─ Offset to scene-relative
    └─ Clamp to scene bounds [0, scene_duration]
  Merge all group results
  Defensive chronological sort
  Build per-scene diagnostics (with group_details)
```

**ASSEMBLY** (`_phase5_assembly`, lines 1431-1818):
```
Across ALL scenes:
  Step 1-2: Load ASR text-only → Batch transcribe_text_only()
  Step 3:   Unload ASR (VRAM swap)
  Step 4:   AssemblyTextCleaner.clean_batch()
  Step 5-6: Load aligner → Batch align_standalone()
  Step 7:   Unload aligner (VRAM cleanup)
  Step 8:   Per scene:
    ├─ merge_master_with_timestamps()
    ├─ Alignment Sentinel (assess on raw word dicts)
    ├─ Recovery if COLLAPSED (VAD-guided or proportional)
    ├─ (no timestamp resolution)
    ├─ (no boundary clamping)
    ├─ (no offset — timestamps already scene-relative)
    ├─ _reconstruct_from_words() [stable-ts regrouping]
    └─ (no Japanese post-processing — correct, see Section 5.1)
  Build per-scene diagnostics
```

### 4.2 Feature-by-Feature Comparison

| # | Feature | VAD_SLICING | ASSEMBLY | Gap? |
|---|---------|-------------|----------|------|
| A | **Adaptive Step-Down** (Tier 1→Tier 2) | Yes. Tier 1: 30s groups → detect collapse → defer. Tier 2: re-group collapsed segments at 6s → transcribe + recover. Lines 1257-1343. | No. Operates at scene level (30-120s). No mechanism to re-group at finer granularity. | **YES — Major** |
| B | **Timestamp Resolution** | Full 4-mode dispatch: `ALIGNER_WITH_INTERPOLATION`, `ALIGNER_WITH_VAD_FALLBACK`, `ALIGNER_ONLY`, `VAD_ONLY`. Lines 1202-1213. Respects `--timestamp-mode` CLI flag. | None. Null timestamps pass through unmodified. `timestamp_mode` silently ignored. | **YES — Major** |
| C | **Boundary Clamping** (double) | Two-layer clamp: (1) Group-relative `[0, group_duration]` (lines 1220-1227). (2) Scene-relative `[0, scene_duration]` after offset (lines 1234-1242). | None. Timestamps from the aligner go directly to `_reconstruct_from_words()`. Aligner timestamps > scene_duration persist into final SRT. | **YES — Major** |
| D | **Text Cleaning for Qwen** | None mid-pipeline. Relies on Whisper-era `SubtitleSanitizer` post-pipeline. Also **incorrectly** applies Whisper-era `JapanesePostProcessor` via coupled `transcribe()`. | **Correct**: `AssemblyTextCleaner` operates between text generation and alignment (4 stages: phrase repetition, char flood, hallucination filter, whitespace). Prevents aligner NULL failures from hallucinated text. | **Assembly is AHEAD** |
| E | **Timestamp Offset** | `_offset_result_timestamps(result, group_start_sec)` at line 1230. Shifts group-relative to scene-relative. | Not needed — aligner timestamps are already scene-relative (aligner operates on full scene audio). | N/A (different architecture) |
| F | **Sentinel Granularity** | Per-group (5-30s audio). Smaller audio = more precise collapse detection. VAD groups provide natural speech boundaries for Strategy C recovery. | Per-scene (30-120s audio). Coarser detection. VAD regions available but at scene level. | Partial gap |
| G | **Sentinel Step-Down Deferral** | `skip_recovery_on_collapse=True` on Tier 1: defers recovery, queues segments for Tier 2 re-grouping at tighter boundaries. Only recovers at Tier 2 (last resort). Lines 1157-1161. | No deferral. Immediately recovers collapsed scenes. No retry at finer granularity. | **YES — Major** (ties to A) |
| H | **Mid-Pipeline Text Cleaning** | None. Raw ASR text (with potential hallucinations/repetitions) goes directly to aligner via coupled `transcribe()`. | Yes — `AssemblyTextCleaner` between text generation and alignment. 4 stages: phrase repetition, char flood, hallucination filter, whitespace. | **Assembly is AHEAD** |
| I | **Dynamic Token Budget** | Applied per-group via `asr.transcribe()` → `_compute_dynamic_token_limit(audio_duration)`. | Applied per-scene via `asr.transcribe_text_only()`. | Parity |
| J | **Audio I/O Pattern** | Reads scene audio once into memory. Slices in-memory per group. Writes temp WAV per group (required by QwenASR file-path API). | Passes full scene audio paths to batch APIs. No slicing. Less I/O overhead but larger audio per call. | N/A (different by design) |
| K | **Per-Group Diagnostics** | Rich `group_details` array: tier, group_index, time_range, duration, outcome, sentinel_status, triggers, assessment_snapshot, subs_produced, recovery info. Lines 1070-1248. | No per-unit diagnostics. Per-scene only: sentinel status, asr_stats, timing_sources (using wrong metric — word count not segment count, per H4). | **YES — Major** |
| L | **Sentinel Analytics** | Full assessment_snapshot per group: word_count, char_count, coverage_ratio, aggregate_cps, zero_position_ratio, degenerate_ratio, triggers list. | Per-scene assessment dict stored but no snapshot decomposition. No triggers list in diagnostics structure. | **YES — Moderate** |
| M | **Min Duration Guard** | Groups < 0.1s are skipped (line 1086-1092). | Scenes with empty cleaned text are skipped (line 1648-1655). No minimum duration guard. | **YES — Minor** |
| N | **Error Recovery per Unit** | Per-group `try/except` (line 1116-1128). Failed group → `None`, other groups continue. | Per-scene `try/except` (line 1794-1799). But text generation and alignment are batched (Steps 2+6), so a crash there fails all scenes. | Partial gap |
| O | **Merge + Chronological Sort** | Merges tier1/tier2 results. Explicit `sort(key=s.start)` at line 1336. Defensive sort at 1367-1368 as safety net. | No merge needed (one result per scene). But no chronological sort within the result after `_reconstruct_from_words()`. | **YES — Minor** |
| P | **Sentinel Stats** | `self._sentinel_stats` accumulates across all groups. Instance-level dict. | Local counters. Assigned to `self._sentinel_stats` at end. | Parity |
| Q | **Progress Reporting** | Per-group logging with tier tag (`[T1]/[T2]`), group index, time range. | Per-scene `[DIAG]` logging. Batch-level tqdm from `transcribe_text_only()` and `align_standalone()`. | Different style, both adequate |
| R | **Debug Artifacts** | Per-group via `asr.transcribe(artifacts_dir=...)` — saves master text, timestamps JSON, merged words JSON. | Per-scene: `_assembly_raw.txt`, `_assembly_clean.txt`, `_assembly_merged.json`. Richer mid-pipeline visibility. | **Assembly is AHEAD** |

### 4.3 Summary: What Each Mode Contributes

**What VAD_SLICING has that Assembly lacks (and needs):**

| Priority | Feature | Impact if Missing |
|----------|---------|-------------------|
| P0 | Timestamp Resolution (4 modes) | Null timestamps persist in final SRT |
| P0 | Boundary Clamping (double) | Aligner timestamps can exceed scene duration |
| P1 | Adaptive Step-Down | No retry mechanism for collapsed scenes |
| P1 | Step-Down Deferral in Sentinel | Cannot defer collapse recovery to finer granularity |
| P1 | Per-Group Diagnostics | No sub-scene visibility for debugging |
| P2 | Min Duration Guard (0.1s) | Near-empty scenes waste alignment compute |
| P2 | Sentinel Assessment Snapshot | Less diagnostic data for analysis |
| P2 | Defensive Chronological Sort | Segments may be out of order after recovery |

**What Assembly has that VAD_SLICING lacks:**

| Feature | Value |
|---------|-------|
| Mid-pipeline text cleaning (AssemblyTextCleaner) | Prevents aligner NULL failures from hallucinated input |
| VRAM-exclusive phases | Higher batch_size potential, future vLLM socket |
| Raw vs clean text artifacts | Better debuggability for text quality issues |
| Batch processing | Amortized overhead across scenes |

---

## 5. Key Design Decisions

### 5.1 JapanesePostProcessor Is NOT Suitable for Qwen3-ASR

**Decision**: The Whisper-era `JapanesePostProcessor` was developed for Whisper's output
characteristics (aizuchi patterns, particle anchoring, merge/split rules tuned to Whisper
segmentation). It is **not suitable** for Qwen3-ASR output.

**Implications**:
- Assembly is **correct** to skip `JapanesePostProcessor` — it has its own Qwen-specific
  `AssemblyTextCleaner` (4 stages: phrase repetition, char flood, hallucination filter,
  whitespace)
- Coupled modes (CONTEXT_AWARE, VAD_SLICING) are **incorrect** to apply it via
  `qwen_asr.py:964-973`
- This **flips audit finding C2**: coupled modes have the bug, not assembly
- Each future ASR model may need its own text cleaner tuned to that model's artifacts

### 5.2 New Shared Method, Not Porting

**Decision**: Create a new shared hardening method rather than porting VAD_SLICING features
to Assembly.

**Rationale**: The two modes operate in fundamentally different coordinate systems and data
formats:

| Dimension | VAD_SLICING | ASSEMBLY |
|-----------|-------------|----------|
| Coordinate system | Group-relative → offset → scene-relative | Scene-relative throughout |
| Input data | WhisperResult (from coupled transcribe) | Raw word dicts (from merge) |
| Sentinel input | `extract_words_from_result(result)` adapter | Direct word dicts (native format) |
| Recovery VAD regions | Offset from scene-relative to group-relative | Scene-relative (no transform) |
| Reconstruction | Only after sentinel recovery | Always (every scene) |
| Timestamp resolution | Before reconstruction | Would need to be after reconstruction |

Porting means adapting each feature for a different coordinate system — every adaptation
risks bugs. The shared hardening method works on the common output (WhisperResult) that
both modes produce.

### 5.3 TemporalFramer as a Distinct Concern

**Decision**: Introduce a `TemporalFramer` protocol between scene detection and text
generation — architecturally distinct from `SpeechSegmenter` (VAD).

**The four orthogonal questions the pipeline answers:**

| Question | Concern | Protocol Domain |
|----------|---------|-----------------|
| WHERE is there speech? | Physical detection (VAD) | SpeechSegmenter (existing) |
| WHAT are the dialogue segments? | Semantic segmentation with time boundaries | **TemporalFramer (new)** |
| WHAT was said? | Text generation | TextGenerator (new) |
| WHEN exactly was each word said? | Sub-segment alignment | TextAligner (new) |

**Why TemporalFramer is distinct from SpeechSegmenter:**

| | SpeechSegmenter (VAD) | TemporalFramer |
|---|---|---|
| Level | Physical (speech/silence) | Semantic (dialogue units) |
| Input | Audio only | Audio, SRT, Whisper output, or other |
| Output | Speech regions | Dialogue segments with optional text |
| May use ASR internally | No | Yes (Whisper-as-framer) |
| May use no audio at all | No | Yes (SRT-as-source) |
| Granularity | Sub-second frames | Utterance-level (1-30s) |

SpeechSegmenter doesn't go away — it becomes an **internal dependency** of the
`vad-grouped` framer backend. But `whisper-segment` and `srt-source` framers bypass
SpeechSegmenter entirely.

### 5.4 TemporalFramer Backends

After evaluation, four backends represent genuinely different strategies (not parameter
variations), plus one utility backend:

| Backend | Input | How it frames | When to use |
|---------|-------|---------------|-------------|
| **`vad-grouped`** | Audio | VAD → speech regions → grouped dialogue chunks | ASR with short context windows (<=30s) |
| **`whisper-segment`** | Audio | Whisper ASR → extract segment boundaries | Cross-model workflows (Whisper timing + Qwen text) |
| **`srt-source`** | SRT file | Parse timestamps → dialogue frames | Re-transcription of existing subtitles |
| **`full-scene`** | Scene bounds | Entire scene = one frame | When aligner or coupled ASR handles all timing |
| **`manual`** | User timestamps | Direct injection | Testing, debugging, special workflows |

**Dropped: `scene-proportional`** — Naive time-based subdivision (120s → 4x30s) cuts at
arbitrary points, potentially mid-word. VAD-based framing is always superior. If a
`full-scene` frame exceeds some hard limit, fixed-window subdivision can be a fallback
**within** the `full-scene` backend, not a separate strategy.

### 5.5 The Aligner Becomes Optional

**Key architectural win**: If the TemporalFramer provides sufficiently accurate timestamps
and the TextGenerator produces text within those boundaries, the TextAligner can be
**skipped entirely**. This is significant because:

- The ForcedAligner is the **most unreliable component** (collapse is what the sentinel
  exists to detect)
- SRT subtitles need segment-level timing, not word-level — aligner precision is often
  unnecessary
- Aligner-free workflows are simpler, faster, and more robust

---

## 6. Proposed Architecture

### 6.1 Block Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    DecoupledSubtitlePipeline                      │
│                                                                  │
│  SceneDetector ─→ [SpeechEnhancer] ─→ TemporalFramer            │
│  (Protocol)       (Protocol,optional)  (Protocol - NEW)          │
│  ├─ auditok       ├─ clearvoice        ├─ vad-grouped            │
│  ├─ silero        ├─ bs-roformer       ├─ whisper-segment        │
│  ├─ semantic      ├─ none              ├─ srt-source             │
│  └─ none          └─ future            ├─ full-scene             │
│                                        └─ manual                 │
│                                                                  │
│  ─→ TextGenerator ─→ TextCleaner ─→ [TextAligner] ─→ Hardening  │
│     (Protocol)       (Protocol)     (Protocol,opt)               │
│     ├─ qwen3         ├─ assembly    ├─ qwen-forced-aligner       │
│     ├─ future-model  ├─ passthrough ├─ ctc-aligner               │
│     └─ vllm-backend  └─ future      ├─ none (skip)              │
│                                     └─ future                    │
└──────────────────────────────────────────────────────────────────┘
```

**Six protocol domains:**
1. **SceneDetector** — existing, established (auditok, silero, semantic, none)
2. **SpeechEnhancer** — existing, established (clearvoice, bs-roformer, none)
3. **TemporalFramer** — NEW (vad-grouped, whisper-segment, srt-source, full-scene, manual)
4. **TextGenerator** — NEW (qwen3-text-only, future transformers, vLLM)
5. **TextCleaner** — formalizes existing (assembly text cleaner, passthrough)
6. **TextAligner** — formalizes existing (qwen-forced-aligner, ctc-aligner, none)

Plus a **shared Hardening** stage (not a protocol — a deterministic post-processing step).

### 6.2 SpeechEnhancer Placement

SpeechEnhancer operates on scene-level audio **before** temporal framing because:
- Enhancement needs surrounding context for noise profile estimation
- Framing backends that use VAD internally (`vad-grouped`) get better accuracy on clean
  audio
- Framing backends that run ASR internally (`whisper-segment`) benefit from enhanced audio

### 6.3 Existing Protocol + Factory Pattern

The codebase already uses a consistent pattern for three domains:

| Domain | Protocol | Factory | Result type | Backends dir |
|--------|----------|---------|-------------|--------------|
| Scene Detection | `SceneDetector` | `SceneDetectorFactory` | `SceneDetectionResult` | `scene_detection_backends/` |
| Speech Segmentation | `SpeechSegmenter` | `SpeechSegmenterFactory` | `SegmentationResult` | `speech_segmentation/backends/` |
| Speech Enhancement | `SpeechEnhancer` | `SpeechEnhancerFactory` | `EnhancementResult` | `speech_enhancement/backends/` |

Each follows:
- Protocol defines the contract (input/output types, methods)
- Dataclass defines the result type
- Factory creates implementations by name
- `cleanup()` for resource management
- Implementations live in `backends/` subdirectory

New domains will follow the same pattern.

---

## 7. Protocol Specifications

### 7.1 TemporalFramer Protocol

```python
@dataclass
class TemporalFrame:
    start: float                      # Frame start (seconds, scene-relative)
    end: float                        # Frame end (seconds, scene-relative)
    text: Optional[str] = None        # Pre-existing text (from SRT, Whisper draft)
    confidence: Optional[float] = None
    source: str = ""                  # Backend that produced this frame

@dataclass
class FramingResult:
    frames: List[TemporalFrame]
    metadata: Dict[str, Any]          # Backend-specific metadata

@runtime_checkable
class TemporalFramer(Protocol):
    def frame(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> FramingResult: ...

    def cleanup(self) -> None: ...
```

**Backend implementations:**

| Backend | `frame()` behavior | Key kwargs |
|---------|-------------------|------------|
| `vad-grouped` | Runs SpeechSegmenter internally → groups speech regions into dialogue chunks | `max_group_duration_s`, `chunk_threshold_s`, segmenter config |
| `whisper-segment` | Runs faster-whisper → extracts segment boundaries | `model_size`, `language`, `compute_type` |
| `srt-source` | Parses SRT file → extracts timestamps, optionally preserves text | `srt_path`, `keep_text` |
| `full-scene` | Returns single frame spanning entire scene | (none — trivial) |
| `manual` | Returns user-provided timestamps as frames | `timestamps: List[Tuple[float, float]]` |

### 7.2 TextGenerator Protocol

```python
@dataclass
class TranscriptionResult:
    text: str                          # Transcribed text
    language: str                      # Detected or specified language
    metadata: Dict[str, Any]           # Model-specific metadata (token count, etc.)

@runtime_checkable
class TextGenerator(Protocol):
    def generate(
        self,
        audio_path: Path,
        language: str = "ja",
        context: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult: ...

    def generate_batch(
        self,
        audio_paths: List[Path],
        language: str = "ja",
        contexts: Optional[List[str]] = None,
        **kwargs
    ) -> List[TranscriptionResult]: ...

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def cleanup(self) -> None: ...
```

**Implementations:**

| Implementation | Wraps | Notes |
|----------------|-------|-------|
| `Qwen3TextGenerator` | `QwenASR.transcribe_text_only()` | Batch support via existing batch API |
| `TransformersGenerator` | Any HuggingFace ASR model | Generic transformers pipeline adapter |
| `VLLMGenerator` | OpenAI-compatible API to vLLM server | HTTP client, no VRAM management |
| `WhisperTextGenerator` | faster-whisper/openai-whisper | Text extraction (timestamps ignored) |

### 7.3 TextCleaner Protocol

```python
@runtime_checkable
class TextCleaner(Protocol):
    def clean(self, text: str, **kwargs) -> str: ...
    def clean_batch(self, texts: List[str], **kwargs) -> List[str]: ...
```

**Implementations:**

| Implementation | Wraps | When to use |
|----------------|-------|-------------|
| `AssemblyTextCleaner` (existing) | 4-stage Qwen-specific cleaning | Qwen3-ASR output |
| `PassthroughCleaner` | No-op | Models whose output needs no cleaning |
| Future model-specific cleaners | TBD | Per-model characteristics |

### 7.4 TextAligner Protocol

```python
@dataclass
class WordTimestamp:
    word: str
    start: float                       # Seconds, scene-relative
    end: float                         # Seconds, scene-relative

@dataclass
class AlignmentResult:
    words: List[WordTimestamp]
    metadata: Dict[str, Any]           # Aligner-specific (coverage, CPS, etc.)

@runtime_checkable
class TextAligner(Protocol):
    def align(
        self,
        audio_path: Path,
        text: str,
        language: str = "ja",
        **kwargs
    ) -> AlignmentResult: ...

    def align_batch(
        self,
        audio_paths: List[Path],
        texts: List[str],
        language: str = "ja",
        **kwargs
    ) -> List[AlignmentResult]: ...

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def cleanup(self) -> None: ...
```

**Implementations:**

| Implementation | Wraps | Notes |
|----------------|-------|-------|
| `Qwen3ForcedAlignerAdapter` | `QwenASR.align_standalone()` | Batch support via existing batch API |
| `CTCAligner` | wav2vec2-based CTC segmentation | Alternative neural alignment |
| `VADProportionalAligner` | No neural model — VAD + proportional char distribution | Fallback when no aligner available |
| `NoneAligner` | Skip alignment | Timestamps from TemporalFramer are sufficient |
| `HybridAligner` | Try primary, fall back to secondary on collapse | Composed from other aligners |

---

## 8. Data Types & Contracts

### 8.1 Inter-Component Data Flow

```
SceneDetector        → SceneDetectionResult (scene boundaries)
SpeechEnhancer       → EnhancementResult (enhanced audio paths)
TemporalFramer       → FramingResult (List[TemporalFrame])
TextGenerator        → TranscriptionResult (text per frame)
TextCleaner          → str (cleaned text per frame)
TextAligner          → AlignmentResult (word timestamps per frame)
Hardening            → stable_whisper.WhisperResult (Phase 5→6 contract)
```

### 8.2 Shared Types Package Location

```
whisperjav/modules/subtitle_pipeline/
├── __init__.py
├── types.py             # TemporalFrame, TranscriptionResult, AlignmentResult, WordTimestamp
├── protocols.py         # TemporalFramer, TextGenerator, TextCleaner, TextAligner protocols
├── orchestrator.py      # DecoupledSubtitlePipeline
├── hardening.py         # harden_scene_result() — shared post-reconstruction
├── reconstruction.py    # reconstruct_from_words() — extracted from current static method
├── framers/             # TemporalFramer backends
│   ├── __init__.py
│   ├── factory.py       # TemporalFramerFactory
│   ├── vad_grouped.py
│   ├── whisper_segment.py
│   ├── srt_source.py
│   ├── full_scene.py
│   └── manual.py
├── generators/          # TextGenerator backends
│   ├── __init__.py
│   ├── factory.py       # TextGeneratorFactory
│   └── qwen3.py
├── aligners/            # TextAligner backends
│   ├── __init__.py
│   ├── factory.py       # TextAlignerFactory
│   ├── qwen3.py
│   ├── vad_proportional.py
│   └── none.py
└── cleaners/            # TextCleaner backends
    ├── __init__.py
    ├── factory.py       # TextCleanerFactory
    ├── qwen3.py         # Wraps AssemblyTextCleaner
    └── passthrough.py
```

### 8.3 Phase 5 → Phase 6 Contract Preserved

The pipeline output remains `stable_whisper.WhisperResult` — the existing contract between
Phase 5 (ASR) and Phase 6 (SRT generation). The `reconstruction.py` module handles
converting `AlignmentResult` (word timestamps) into `WhisperResult` via
`stable_whisper.transcribe_any()`.

---

## 9. Orchestrator Design

### 9.1 DecoupledSubtitlePipeline

Not a pipeline class (doesn't extend `BasePipeline`). Rather, a Phase 5 replacement that
any pipeline class can use internally:

```python
class DecoupledSubtitlePipeline:
    def __init__(
        self,
        framer: TemporalFramer,
        generator: TextGenerator,
        cleaner: TextCleaner,
        aligner: Optional[TextAligner],   # None = skip alignment
        hardening_config: HardeningConfig,
        artifacts_dir: Optional[Path] = None,  # debug artifact output
    ): ...

    def process_scenes(
        self,
        scene_paths: List[Path],
        scene_durations: List[float],
        speech_regions_per_scene: Optional[List[List[Tuple[float, float]]]] = None,
    ) -> List[Tuple[Optional[WhisperResult], Dict]]:
        """
        Process all scenes through the decoupled pipeline.

        Returns list of (result, diagnostics) tuples, one per scene.
        Result is None if scene processing failed.
        """
```

### 9.2 Internal Flow

```
1.  framer.frame(scene_audio) → temporal_frames per scene
    ─── if framer produces text (srt-source, whisper-segment): skip steps 2-5
2.  generator.load()
3.  generator.generate_batch(frame_audios) → raw_texts
4.  generator.unload()                      ← VRAM swap
5.  cleaner.clean_batch(raw_texts) → clean_texts
6.  IF aligner is not None:
      aligner.load()
7.    aligner.align_batch(frame_audios, clean_texts) → alignments
8.    aligner.unload()                      ← VRAM swap
9.  Per scene/frame:
      merge text + timestamps → word dicts
      sentinel detection → recovery if collapsed
      reconstruct → WhisperResult
      harden (timestamp resolution, boundary clamp, chronological sort)
      emit diagnostics
```

**Key flow variations:**

| TemporalFramer | TextAligner | Flow |
|----------------|-------------|------|
| `full-scene` | `qwen-forced-aligner` | Current assembly flow: text → align → merge → harden |
| `vad-grouped` | `none` | VAD provides timing, text generated per frame, timestamps inherited from frames |
| `whisper-segment` | `none` | Whisper provides timing + optional draft text, Qwen re-transcribes per frame |
| `srt-source` | `none` | SRT provides timing, Qwen transcribes per timestamp window |
| `vad-grouped` | `qwen-forced-aligner` | Current VAD_SLICING flow but decoupled |

### 9.3 Hardening Stage

The shared `harden_scene_result()` function applies **identically** to all results
regardless of how they were produced:

```python
def harden_scene_result(
    result: WhisperResult,
    scene_duration_sec: float,
    timestamp_mode: TimestampMode,
    speech_regions: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[WhisperResult, Dict]:
    """
    Post-reconstruction hardening — shared by all pipeline paths.

    Steps:
    1. Timestamp resolution (interpolation/fallback/VAD-only per timestamp_mode)
    2. Boundary clamping [0, scene_duration]
    3. Chronological sort
    4. Diagnostic snapshot (segment count, timing source breakdown, quality metrics)
    """
```

This resolves:
- **H2**: Assembly gets timestamp resolution
- **H1**: VAD_ONLY distributes proportionally (not blanket assignment)
- **H4**: Diagnostics measured after reconstruction, with correct segment count
- **M7**: All modes respect `timestamp_mode` setting
- **Comparison B/C/O**: All modes get boundary clamping and chronological sort

### 9.4 Sentinel Integration (One Place, Not Four)

Currently, sentinel detection + recovery is implemented in:
1. Assembly Step 8 (lines 1693-1763) — operates on raw word dicts
2. VAD_SLICING `_transcribe_group` (lines 1129-1197) — operates on WhisperResult via adapter
3. Coupled-mode sentinel recovery (via shared code paths)

In the new architecture, sentinel integration lives **once** in the orchestrator's Step 9,
operating on word dicts from either:
- `AlignmentResult.words` (when aligner ran)
- Direct extraction from `TemporalFrame.text` timestamps (when framer provided timing)

---

## 10. Audit Finding Resolutions

| Finding | Resolution in New Architecture |
|---------|-------------------------------|
| **C1**: OOM retry stale closure | **Eliminated** — orchestrator manages model lifecycle externally via `generator.load()`/`unload()`. No closures capture model references. |
| **C2** (revised): Coupled modes apply wrong post-processor | **Resolved** — `TextCleaner` protocol makes cleaning strategy explicit per model. Qwen3 uses `AssemblyTextCleaner`. Whisper-era `JapanesePostProcessor` removed from Qwen path. |
| **H1**: VAD_ONLY overlapping subs | **Fixed** in shared `harden_scene_result()` — proportional distribution by character count. |
| **H2**: Assembly no timestamp resolution | **Fixed** — `harden_scene_result()` applies to all paths. |
| **H3**: `suppress_silence` conflicts with sentinel | **Design decision** made once in `reconstruction.py` — set `suppress_silence=False` for sentinel-recovered results, `True` for normal results. |
| **H4**: Wrong diagnostic metrics | **Fixed** — diagnostics measured after reconstruction in hardening stage, with correct segment count. |
| **M1**: Stale 300s comment | **Resolved** by code extraction — hard limit lives in aligner backend, not in pipeline comments. |
| **M2**: `end=0.0` treated as NULL | **Resolved** — sentinel operates on `AlignmentResult.words` which has explicit null representation, not overloaded `0.0`. |
| **M3**: No batch_size override for assembly | **Resolved** — `TextGenerator.load()` can accept model config including batch_size. |
| **M4**: Private `_exact_lists` access | **Resolved** — `TextCleaner` protocol encapsulates cleaning. If a cleaner needs hallucination lists, it manages that internally. |
| **M5**: Redundant VRAM cleanup | **Resolved** — orchestrator calls `safe_cuda_cleanup()` once after each `unload()`. |
| **M6**: Redundant `_check_audio_limits` | **Resolved** — duration known from `TemporalFrame.end - TemporalFrame.start`. No redundant file reads. |
| **M7**: Timestamp modes inconsistent | **Resolved** — one `harden_scene_result()`, one behavior for all modes. |
| **M8**: `suppress_silence` in coupled mode | **Resolved** — same design decision as H3, applied once. |
| **L1-L7** | **Resolved** by fresh implementation — dead code, stale comments, and cosmetic issues don't carry over. |

---

## 11. Current Mode Mapping

### 11.1 How Existing Modes Map to New Architecture

| Current Mode | TemporalFramer | TextGenerator | TextCleaner | TextAligner | Notes |
|---|---|---|---|---|---|
| **ASSEMBLY** | `full-scene` | `qwen3-text-only` | `assembly (Qwen3)` | `qwen-forced-aligner` | Closest to current assembly flow |
| **VAD_SLICING** | `vad-grouped` | `qwen3-text-only` (or coupled) | `assembly (Qwen3)` | `qwen-forced-aligner` (or `none`) | Decoupled version of current flow |
| **CONTEXT_AWARE** | `full-scene` | `qwen3-coupled` | `assembly (Qwen3)` | built-in (coupled) | May remain as legacy path initially |

### 11.2 Migration Strategy

1. **Phase 0**: Build types, protocols, hardening, reconstruction — independent of current
   pipeline
2. **Phase 1**: Create Qwen3 adapters (wrap existing `QwenASR` methods behind protocols)
3. **Phase 2**: Build orchestrator, integrate sentinel
4. **Phase 3**: Wire into `QwenPipeline` for assembly mode first; coupled modes continue
   as-is initially
5. **Future**: Deprecate mode-specific code as orchestrator proves equivalent

---

## 12. New Workflows Enabled

The component architecture trivially enables workflows that are currently impossible:

### 12.1 Whisper-Guided Qwen

```
TemporalFramer("whisper-segment") → Whisper produces segment timestamps
TextGenerator("qwen3") → Qwen transcribes within those timestamps
TextAligner(None) → Skip aligner — Whisper's segment timestamps are the final timing
```

**Value**: Uses Whisper's reliable timing with Qwen's superior Japanese transcription.
The aligner (most unreliable component) is eliminated entirely.

### 12.2 SRT Re-Transcription

```
TemporalFramer("srt-source", srt_path="existing.srt") → Extract timestamps from SRT
TextGenerator("qwen3") → Qwen re-transcribes each dialogue window
TextAligner(None) → Timestamps inherited from source SRT
```

**Value**: An existing SRT with good timing but poor transcription gets re-transcribed
with a better model. No aligner needed.

### 12.3 Two-Pass Refinement

```
Pass 1: Normal pipeline → output1.srt (may have timing issues)
Pass 2: TemporalFramer("srt-source", srt_path="output1.srt") → use Pass 1 timestamps
        TextGenerator("qwen3") → re-transcribe with context from Pass 1
        TextAligner("qwen-forced-aligner") → refine word-level timing
```

**Value**: Iterative refinement where each pass improves on the previous.

### 12.4 Aligner-Free Fast Mode

```
TemporalFramer("vad-grouped") → VAD provides speech boundaries
TextGenerator("qwen3") → Qwen transcribes per group
TextAligner(None) → Frame timestamps become segment timestamps
```

**Value**: Fastest possible pipeline — no aligner load/compute. Segment-level timing from
VAD is sufficient for many SRT use cases.

### 12.5 Cross-Model Benchmarking

```
Same audio + same TemporalFramer
→ TextGenerator("qwen3") → AlignmentResult A
→ TextGenerator("whisper-large-v3") → AlignmentResult B
→ Compare against ground-truth SRT → per-scene accuracy metrics
```

**Value**: Because all generators produce the same `TranscriptionResult` type, comparison
is trivial. The diagnostic/benchmarking utility can run multiple generators on the same
input and compare results.

---

## 13. What Doesn't Change

| Component | Status |
|-----------|--------|
| **Phases 1-4** (audio extraction, scene detection, speech enhancement, speech segmentation) | Untouched |
| **Phases 6-8** (SRT generation, stitching, sanitization, output) | Untouched |
| **Ensemble system** (`pass_worker.py`) | Continues to construct pipelines as before |
| **CLI and GUI interfaces** | Pass same parameters, wired through same paths |
| **`stable_whisper.WhisperResult`** | Remains the Phase 5→6 contract |
| **Scene detection backends** | Already refactored (Sprint 3), unchanged |
| **Speech enhancement backends** | Already established, unchanged |
| **Speech segmentation backends** | Become internal to `vad-grouped` framer |
| **SRT stitching** | Unchanged — operates on Phase 6 output |

The refactoring is **scoped to Phase 5** — replacing mode-specific tangled code with
composable components.

---

## 14. Future Expansion

### 14.1 New TextGenerator Implementations

| Model | Backend Type | Integration Path |
|-------|-------------|------------------|
| Future transformers models | `TransformersGenerator` | Generic HuggingFace pipeline adapter |
| vLLM-served models | `VLLMGenerator` | OpenAI-compatible HTTP API |
| Whisper (text extraction) | `WhisperTextGenerator` | faster-whisper, discard timestamps |

### 14.2 New TextAligner Implementations

| Aligner | Approach | Use Case |
|---------|----------|----------|
| CTC Aligner (wav2vec2) | CTC-based forced alignment | Alternative to Qwen3 aligner |
| WhisperCrossAttention | Use Whisper's attention maps for alignment | Cross-model alignment |
| HybridAligner | Try primary → fall back on collapse | Composed from other aligners |

### 14.3 Diagnostic/Benchmarking Utility

The protocol-based architecture makes benchmarking natural:

```
Same audio + same text → different TextAligner implementations → AlignmentResult[]
                       → compare against ground-truth SRT
                       → per-scene metrics (timing accuracy, coverage, CPS)
```

Because all aligners produce the same `AlignmentResult` type, the utility can:
1. Run multiple aligners on the same input
2. Compare their results against ground truth
3. Report which aligner works best for which content type
4. Identify per-scene problem areas
5. Generate comparative reports

### 14.4 vLLM Integration (per ADR-005)

Assembly mode was identified as the "vLLM socket" in ADR-005. The new architecture makes
this explicit: `VLLMGenerator` is a drop-in `TextGenerator` implementation that talks to
a vLLM server via HTTP. No VRAM management needed (the server handles it). The VRAM swap
points (Steps 4 and 8 in the orchestrator) become no-ops for VLLMGenerator.

---

## Appendix A: Alignment Sentinel Reference

The Alignment Sentinel (`alignment_sentinel.py`) detects and recovers from ForcedAligner
collapse. It operates as standalone functions (no class state):

**Detection** (`assess_alignment_quality`):
- Coverage ratio: words covering <5% of scene = collapsed
- Aggregate CPS: >50 chars/sec = physically impossible speech rate
- Word span: sub-500ms with substantial text = collapsed
- Zero position: >10% words at (0.0, 0.0) = aligner returned NULL
- Degenerate: >40% words with start==end = cluster collapse

**Recovery strategies**:
- **Strategy C** (VAD-guided): Distribute words proportionally across VAD speech regions,
  skipping silence gaps. Used when VAD data available.
- **Strategy B** (Proportional): Distribute words from anchor position at conversational
  speed (~10 CPS Japanese). Fallback when no VAD data.

**In new architecture**: Sentinel moves from 3-4 integration points to one place in
orchestrator Step 9.

## Appendix B: AssemblyTextCleaner Reference

The `AssemblyTextCleaner` (`assembly_text_cleaner.py`) is the Qwen3-ASR-specific mid-pipeline
text cleaner. It operates between text generation and alignment with four stages:

1. **Phrase repetition** (`PhraseRepetitionConfig`): Detects and collapses repeated
   multi-word phrases
2. **Character flood** (`CharFloodConfig`): Detects sequences of repeated characters
   (e.g., "ああああああ")
3. **Hallucination filter** (`HallucinationFilterConfig`): Removes known hallucination
   patterns using lists from `HallucinationRemover`
4. **Whitespace** (`WhitespaceConfig`): Normalizes whitespace artifacts

This cleaner is specifically tuned for Qwen3-ASR's hallucination characteristics and is
NOT interchangeable with the Whisper-era `JapanesePostProcessor`.

## Appendix C: Key Constants & Limits

| Constant | Value | Location | Notes |
|----------|-------|----------|-------|
| ForcedAligner hard limit | 180s | `qwen_asr.py:217` | NOT 300s as some comments claim |
| Assembly max scene duration | 120s | `qwen_pipeline.py:398` | 60s safety margin below aligner limit |
| TEN VAD max group duration | 29s | Config YAML | Stay within Whisper's 30s context window |
| Sentinel coverage threshold | 5% | `alignment_sentinel.py` | Words covering <5% of scene = collapsed |
| Sentinel CPS threshold | 50.0 | `alignment_sentinel.py` | Physically impossible speech rate |
| Sentinel degenerate threshold | 40% | `alignment_sentinel.py` | >40% zero-duration words = collapse |
| Target CPS for recovery | 10.0 | `alignment_sentinel.py` | Japanese conversational speed |
| Repetition penalty | 1.1 | Pipeline default | Applied via `_apply_generation_config()` |
| Max tokens per audio second | 20.0 | Pipeline default | Dynamic per-call token budget |

---

*End of ADR-006*
