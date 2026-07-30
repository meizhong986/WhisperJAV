# WhisperSeg VAD Tradeoff Analysis — Information Pack

## 1. CONTEXT: What is WhisperJAV?

WhisperJAV is a subtitle generation tool for Japanese Adult Videos (JAV). It uses
OpenAI Whisper with custom enhancements for Japanese language processing, scene
detection, and voice activity detection (VAD).

The **anime-whisper** ASR backend (inside the "qwen" / ChronosJAV pipeline) uses
**WhisperSeg** — a Whisper encoder-decoder model fine-tuned on ~500h Japanese ASMR audio,
exported to ONNX. It provides frame-level (20ms resolution) speech probabilities across
30-second input windows. These probabilities are post-processed by a Silero-compatible
dual-threshold hysteresis state machine to produce speech segments.

In anime-whisper mode, `timestamp_mode="vad_only"` — meaning VAD segments directly
become subtitle timing boundaries. This makes VAD segmentation quality absolutely
critical for subtitle accuracy.

---

## 2. BACKGROUND: The Mechanical Split Problem

### 2.1 Symptom

When running anime-whisper ensemble pass 1 with **aggressive** sensitivity, **all**
subtitles were exactly 4.020s long with 0.020s (one frame) gaps between them. The VAD
was operating as a fixed-interval chopper, not a probabilistic speech detector.

### 2.2 Root Cause (Verified)

Two converging issues in `whisperseg.py::_probs_to_segments()`:

**Issue A — neg_threshold cliff:**
The offset (hysteresis) threshold is derived as:
```python
neg_threshold = max(threshold - 0.15, 0.01)
```
This formula works fine at the upstream default `threshold=0.50` (gives neg_threshold=0.35).
But our aggressive preset uses `threshold=0.15` (intentionally low to capture all possible
dialogue, including quiet/whispered speech). At 0.15, the formula gives
`neg_threshold=0.01` — functionally "never end a segment naturally." The hysteresis is dead.

**Issue B — Naive force-split:**
When a speech segment exceeds `max_speech_duration_s` (was 4.0s), it's chopped at an
exact interval. After each split, `triggered=False` resets the state machine, which
immediately re-triggers onset (any probability >= 0.15 is guaranteed in continuous speech).
Result: perfectly regular 4.020s segments with mechanical 0.020s gaps. No intelligent
boundary seeking.

### 2.3 Upstream Verification

Our state machine is a faithful line-for-line port of two upstream sources:

1. **TransWithAI/Whisper-Vad-EncDec-ASMR-onnx/inference.py** — `get_speech_timestamps()`
2. **TransWithAI/Faster-Whisper-TransWithAI-ChickenRice/vad_manager.py** — `get_speech_timestamps_onnx()`

Both upstreams use the same `neg_threshold` formula, the same naive force-split, and
the same defaults (threshold=0.50, max_speech=infinity). Our problem is a parameter
interaction at the extreme low end of threshold, not a porting error.

### 2.4 ChickenRice's Solution

The ChickenRice repo has a **POST-VAD** smart splitter that our code lacked:
`infer.py::create_contiguous_chunks()` with `split_window_factor=0.4`.

How it works:
- Groups VAD speech spans into contiguous chunks up to a maximum duration
- When a chunk would exceed the maximum, searches the **last 40%** for the
  **largest silence gap** (gap between consecutive speech spans)
- Splits at the midpoint of that gap
- Falls back to exact chop if no gap > 100ms is found

Key distinction: ChickenRice's splitter operates on **speech spans** (post-VAD),
not raw probabilities (in-VAD). It looks for gaps between spans, not probability dips.

---

## 3. WHAT WE CHANGED (i2 Fix)

Three changes were made to address the mechanical split:

### 3.1 CLA1: max_speech_duration_s 4.0 → 3.5

Owner-directed. Typical JAV dialogue is ~2.7s, so 4.0s was too long. 3.5s is a tighter
ceiling that still accommodates longer natural sentences.

**File:** `whisperjav/config/anime_whisper_vad.py` line 67

### 3.2 Fix 1: Decouple neg_threshold

Added explicit `neg_threshold` parameter to WhisperSeg so onset and offset thresholds
can be set independently.

- `whisperseg.py __init__`: added `neg_threshold: Optional[float] = None` (line 83)
- `whisperseg.py _probs_to_segments`: if `self.neg_threshold is not None`, uses it
  directly; else derives from formula (lines 457-460)
- `anime_whisper_vad.py`: added `"neg_threshold": 0.10` to aggressive row (line 64)
- `factory.py`: added `"neg_threshold": (float, None, True)` to whisperseg schema (line 151)

The idea: keep `threshold=0.15` for wide-net onset (captures quiet dialogue), but use
`neg_threshold=0.10` for offset so segments can END naturally when probability drops
below 0.10. This restores hysteresis without sacrificing recall.

### 3.3 Fix 2: Smart force-split (ChickenRice-inspired)

When a segment hits `max_speech_duration_s`, instead of chopping at the exact boundary,
search the **last 40%** of the segment for the **lowest-probability frame** and split
there. This is an adaptation of ChickenRice's `split_window_factor=0.4` logic, translated
from "silence gaps between spans" (post-VAD) to "probability dips within continuous speech"
(in-VAD).

**Implementation** (`whisperseg.py` lines 492-522):

```python
if triggered and "start" in current:
    duration = i - current["start"]
    if duration > max_speech_frames:
        window_start = int(max_speech_frames * 0.6)
        window_probs = current_probs[window_start:max_speech_frames]
        if window_probs:
            min_prob = float(np.min(window_probs))
            mean_prob = float(np.mean(window_probs))
            if min_prob < mean_prob * 0.85:
                best_offset = window_start + int(np.argmin(window_probs))
            else:
                best_offset = max_speech_frames
        else:
            best_offset = max_speech_frames
        split_frame = current["start"] + best_offset
        current["end"] = split_frame
        # ... stats ...
        speeches.append(current)

        current = {"start": split_frame}
        current_probs = current_probs[best_offset:]
        temp_end = 0
        continue
```

Key design decisions:
- **`triggered` stays True** across splits — no re-onset needed, no audio gap
- **Probability array carried forward** — the remainder of the speech continues
- **Quality gate**: `min_prob < mean_prob * 0.85` — only splits at a dip if it's
  meaningfully below the window mean. With flat probabilities (all 0.8), argmin
  returns index 0, which would cause ALL splits at 60% of max_speech. The 85%
  gate prevents this, falling back to the full `max_speech_frames` ceiling instead.
- **No gap inserted** — segments are contiguous, unlike the old code which reset
  `triggered=False` and created a 0.020s gap.

### 3.4 Previous (old) parameter state for aggressive

Prior to the i2 fix, the aggressive row (set in commit `09f0eb3`, the cross-clip VAD
sweep retune) was:

```python
"aggressive": {
    "chunk_threshold_s": 0.2,
    "max_group_duration_s": 2.0,
    "threshold": 0.15,
    # neg_threshold: NOT SET → formula gives max(0.15-0.15, 0.01) = 0.01
    "start_pad_ms": 0,
    "end_pad_ms": 30,
    "max_speech_duration_s": 4.0,
}
```

### 3.5 New parameter state for aggressive

```python
"aggressive": {
    "chunk_threshold_s": 0.2,
    "max_group_duration_s": 2.0,
    "threshold": 0.15,
    "neg_threshold": 0.10,
    "start_pad_ms": 0,
    "end_pad_ms": 30,
    "max_speech_duration_s": 3.5,
}
```

---

## 4. E2E TEST RESULTS — THE TRADEOFF

The owner tested both old and new code on the same content using anime-whisper ensemble
mode with aggressive sensitivity, semantic scene detection (default), and WhisperSeg
(default).

### 4.1 Old code (threshold=0.15, no explicit neg_threshold → 0.01, max_speech=4.0, naive split)

**Strengths:**
- Better dialogue capture — more subtitles generated, including dialogue in quiet/noisy areas
- **Better transcription accuracy** — more characters captured per subtitle, fewer
  truncations at word boundaries. The dialogue content itself is more correct and complete.

**Weaknesses:**
- Timestamps start ~1 second too early
- Timestamps finish slightly too soon
- Two-way dialogues sometimes blended into a single subtitle line
- All segments exactly 4.020s (mechanical, not probabilistic)

### 4.2 New code (threshold=0.15, neg_threshold=0.10, max_speech=3.5, smart split)

**Strengths:**
- Better timestamp accuracy — segments align more naturally with speech
- No more mechanical 4.020s uniform segments

**Weaknesses (CRITICAL):**
- **Worse transcription accuracy** — characters missing at the beginning and/or end
  of transcriptions. The dialogue that IS captured is less accurate / less complete
  than the old code. **This is the most important regression.**
- Fewer captured dialogues overall — dialogue in quiet or noisy areas is missed entirely

### 4.3 Analysis

**The primary concern is transcription accuracy degradation, not just missed subtitles.**
The new code doesn't just capture fewer dialogues — the dialogues it DOES capture are
transcribed less accurately (truncated beginnings/endings). This suggests the VAD
segment boundaries are cutting into speech, not just failing to reach distant speech.

Why this happens: The old code's dead hysteresis (neg_threshold=0.01) kept the state
machine triggered through EVERYTHING — noise, silence, background music — which
accidentally created very long segments that contained all the dialogue with generous
margins. The Whisper ASR had more audio context around each speech passage, and the
segment boundaries never cut into speech (they were always in non-speech territory,
just far away from it). The segment boundaries were meaningless (all 4.020s), but the
transcription quality was high because nothing was clipped.

The new code's restored hysteresis (neg_threshold=0.10) allows the state machine to
properly end segments when probability drops, which gives accurate timestamps. But it
also ends segments during quiet passages, noise dips, and brief pauses — AND it may
be ending segments too aggressively, clipping the edges of actual speech. When a
segment boundary falls inside or too close to speech, the ASR truncates characters
at the boundary.

**The core tension:** We want the wide-net dialogue capture AND transcription accuracy
of the dead hysteresis, WITH the timestamp accuracy of the live hysteresis. These appear
contradictory with the current architecture — but note that the accuracy problem may
specifically point to neg_threshold=0.10 being too high, or min_silence_duration_ms=100
being too short, or end_pad_ms=30 being insufficient to protect speech edges.

---

## 5. GOAL

Find a solution that achieves ALL THREE:
1. **High transcription accuracy** — the dialogue that IS captured must be complete and
   correct, with no characters truncated at segment boundaries (like the old code).
   **This is the highest priority.**
2. **High dialogue capture** — don't miss dialogue in quiet/noisy areas (like the old code)
3. **Accurate timestamps** — segments should align with actual speech boundaries (like the new code)

The ideal outcome is a VAD configuration or algorithmic change that:
- Keeps `threshold=0.15` for wide-net onset (non-negotiable — owner requirement)
- Produces variable-length segments that follow natural speech boundaries
- Does NOT clip into speech at segment edges (protects transcription accuracy)
- Doesn't drop dialogue in challenging audio conditions
- Doesn't pad 1s too early or truncate endings

### Possible directions (not exhaustive — the analyst should evaluate independently):

1. **Lower neg_threshold further** (e.g., 0.05 instead of 0.10) — keeps more segments
   alive through dips but still allows SOME natural endings

2. **Longer min_silence_duration_ms** — even when prob < neg_threshold, require a longer
   sustained silence before ending. Currently 100ms. Increasing to 200-300ms would let
   brief dips pass without ending the segment.

3. **Two-pass approach** — first pass with dead hysteresis (old code) for maximum recall,
   second pass to trim/refine timestamps. Owner previously asked about this; conclusion
   was single-pass should suffice, but the E2E results reopen the question.

4. **Post-VAD merging** — run with live hysteresis (new code), then merge segments that
   are separated by very short gaps (< some threshold) into contiguous segments.

5. **ChickenRice's actual post-VAD approach** — run VAD at higher neg_threshold to get
   many small segments with accurate boundaries, then use `create_contiguous_chunks`
   logic to group them up to max_group_duration_s. This is closer to how ChickenRice
   actually works: short accurate segments → smart grouping. Currently our code groups
   at the segment level (gap-based grouping in `group_segments` from `ten.py`), but the
   grouping logic may not be optimal for this use case.

6. **Adaptive neg_threshold** — start with low neg_threshold for high recall, but if a
   segment is getting very long (approaching max_speech), temporarily raise neg_threshold
   to find a natural break before forced split.

---

## 6. REFERENCE: ChickenRice Upstream Sources

### 6.1 inference.py (TransWithAI/Whisper-Vad-EncDec-ASMR-onnx)

The original WhisperSeg inference script. Key function: `get_speech_timestamps()`.
- Dual-threshold state machine (threshold / neg_threshold via `max(threshold-0.15, 0.01)`)
- Naive force-split at max_speech_duration (exact interval chop, triggered=False after)
- Default threshold=0.50, default max_speech=infinity
- MIT License

### 6.2 vad_manager.py (Faster-Whisper-TransWithAI-ChickenRice)

ChickenRice's version of the same state machine. Nearly identical to inference.py.
Key function: `get_speech_timestamps_onnx()`.
- Same formula, same naive split, same defaults
- Additionally wraps the result in a pandas-friendly format

### 6.3 infer.py (Faster-Whisper-TransWithAI-ChickenRice)

Contains `create_contiguous_chunks()` — the POST-VAD smart splitter:

```python
def create_contiguous_chunks(speeches, max_duration, split_window_factor=0.4):
    # Groups speech spans into chunks up to max_duration
    # When a chunk exceeds max_duration:
    #   - Searches the last (split_window_factor * 100)% for the largest silence GAP
    #   - Splits at the midpoint of that gap
    #   - Falls back to exact chop if no gap > 100ms
    # Returns list of (start_sec, end_sec) chunks
```

This operates on completed speech spans (the output of `get_speech_timestamps_onnx`),
NOT on raw probabilities. It's looking for gaps BETWEEN segments, not dips WITHIN a
single continuous-speech segment.

### 6.4 download_models.py (Faster-Whisper-TransWithAI-ChickenRice)

Contains model download and configuration. Less relevant to the VAD tradeoff, but
documents the default parameters and model revision used by ChickenRice.

---

## 7. FILE LOCATIONS IN WHISPERJAV

### Modified for i2 (uncommitted):

| File | What changed |
|------|-------------|
| `whisperjav/config/anime_whisper_vad.py` | max_speech 4.0→3.5, added neg_threshold=0.10 in aggressive |
| `whisperjav/modules/speech_segmentation/backends/whisperseg.py` | neg_threshold param, decoupled offset, smart force-split |
| `whisperjav/modules/speech_segmentation/factory.py` | neg_threshold in whisperseg param schema |

### Key architecture files (unmodified, for context):

| File | Purpose |
|------|---------|
| `whisperjav/modules/speech_segmentation/backends/ten.py` | `group_segments()` — post-VAD gap/duration grouping |
| `whisperjav/modules/speech_segmentation/base.py` | `SpeechSegment`, `SegmentationResult` dataclasses |
| `whisperjav/pipelines/qwen_pipeline.py` | ChronosJAV pipeline using anime-whisper |
| `whisperjav/ensemble/pass_worker.py` | Ensemble pass orchestration |
| `whisperjav/config/v4/ecosystems/` | YAML config files |

### State machine location:

`whisperseg.py::_probs_to_segments()` — lines 434-601. This is the function that
converts the raw frame probability stream into speech segments. It contains:
- Lines 457-460: neg_threshold derivation (or explicit value)
- Lines 462-470: Duration/frame conversions
- Lines 479-555: Main state machine loop (onset, force-split, hysteresis offset)
- Lines 557-575: Post-hoc padding with overlap prevention
- Lines 577-601: Frame indices → SpeechSegment conversion

### Tests:

7 synthetic probability tests in the prior session (inline, not in a test file — they
were run as verification, not committed as permanent tests). These test:
1. neg_threshold storage when explicit
2. Formula fallback when None
3. Smart split seeks probability minimum
4. Flat-probability fallback to full ceiling
5. Natural offset with decoupled neg_threshold
6. Old behavior confirmed at neg_threshold=0.01
7. neg_threshold appears in _get_parameters()

---

## 8. PARAMETER REFERENCE

### Aggressive sensitivity — current uncommitted state:

| Parameter | Value | What it does |
|-----------|-------|-------------|
| `threshold` | 0.15 | Onset: speech starts when prob >= 0.15 |
| `neg_threshold` | 0.10 | Offset: speech CANDIDATE end when prob < 0.10 |
| `min_silence_duration_ms` | 100 | Silence confirmation: candidate end confirmed after 100ms sustained |
| `min_speech_duration_ms` | 100 | Segments shorter than 100ms are dropped |
| `max_speech_duration_s` | 3.5 | Force-split ceiling (smart split in last 40%) |
| `chunk_threshold_s` | 0.2 | Post-VAD grouping: merge segments with gap < 0.2s |
| `max_group_duration_s` | 2.0 | Post-VAD grouping: groups capped at 2.0s |
| `start_pad_ms` | 0 | No padding before speech onset |
| `end_pad_ms` | 30 | 30ms padding after speech offset |

### Conservative and balanced — unchanged:

**Conservative:** threshold=0.35, no neg_threshold (formula gives 0.20), chunk_threshold_s=0.3,
max_group_duration_s=3.0, start_pad=100, end_pad=100

**Balanced:** threshold=0.30, no neg_threshold (formula gives 0.15), chunk_threshold_s=0.25,
max_group_duration_s=2.5, start_pad=50, end_pad=50

---

## 9. CONSTRAINTS AND NON-NEGOTIABLES

1. **threshold=0.15 for aggressive is non-negotiable** — owner explicitly chose this for
   maximum dialogue recall. The low threshold is the whole point of aggressive mode.

2. **Single-pass VAD preferred** — owner was asked about two-pass and concluded single
   pass should suffice. This conclusion may be revisited given the E2E results.

3. **anime-whisper `timestamp_mode="vad_only"`** — VAD boundaries directly become subtitle
   times. Any VAD improvement directly improves subtitles; any VAD regression directly
   degrades subtitles.

4. **Changes must be confined to aggressive sensitivity** — conservative and balanced rows
   must not be touched (they have their own validated defaults).

5. **The WhisperSeg model itself is frozen** — we cannot retrain or modify the ONNX model.
   We can only change how its output probabilities are processed.

6. **Smart force-split is a good direction** — the mechanical 4.020s splits were the
   original complaint. Any solution must avoid returning to uniform-length segments.

---

## 10. WHAT THE ANALYST SHOULD DO

1. **Analyze the tradeoff deeply** — why does the new code produce worse transcription
   accuracy? Is it because segment boundaries cut into speech (clipping)? Is it because
   shorter segments give the ASR less context? Or both? Understanding the mechanism is
   essential before proposing a fix.

2. **Evaluate the possible directions** listed in Section 5, and propose others if they
   identify better approaches.

3. **Consider the interaction between parameters** — neg_threshold, min_silence_duration_ms,
   max_speech_duration_s, end_pad_ms, and chunk_threshold_s all interact. A change in one
   may compensate for another.

4. **Consider architectural alternatives** — is the fix better applied at the state machine
   level (in-VAD), or at the post-VAD grouping level (like ChickenRice's approach), or at
   the ASR input level (feeding larger audio context regardless of VAD boundaries)?

5. **Propose a specific recommendation** with exact parameter values or code changes,
   along with reasoning for why it should resolve the tradeoff.
