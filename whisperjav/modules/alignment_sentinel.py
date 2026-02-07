"""
Alignment Sentinel — detects and recovers from ForcedAligner collapse.

The ForcedAligner sometimes "collapses": it maps all words in a scene to a
~100ms window instead of spreading them across the actual speech duration.
The ASR text is correct, but the timestamps are garbage.

This module provides:
    extract_words_from_result()  — extract word dicts from WhisperResult
    assess_alignment_quality()   — detect collapse signatures
    redistribute_collapsed_words() — recover by redistributing timestamps

Architecture: standalone module-level functions (no class), same pattern as
merge_master_with_timestamps() in qwen_asr.py.  Independently testable.

Integration points:
    - Assembly mode: Phase 5 Step 8, after merge_master_with_timestamps()
    - Coupled modes: after asr.transcribe() returns a WhisperResult
"""

from typing import Any, Dict, List, Optional, Tuple

from whisperjav.utils.logger import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Detection thresholds (aggregate V1)
_MIN_CHAR_COUNT_FOR_ASSESSMENT = 10  # Below this, not enough data to judge
_COVERAGE_RATIO_THRESHOLD = 0.05     # Words covering <5% of scene = collapsed
_AGGREGATE_CPS_THRESHOLD = 50.0      # Physically impossible speech rate
_WORD_SPAN_THRESHOLD = 0.5           # Sub-500ms with substantial text

# Recovery parameters
_TARGET_CPS = 10.0  # Japanese conversational speed (~10 chars/sec)


# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------

def assess_alignment_quality(
    words: List[Dict[str, Any]],
    scene_duration_sec: float,
) -> Dict[str, Any]:
    """
    Scan merged word list for alignment collapse signatures.

    Args:
        words: Output of merge_master_with_timestamps(), each dict has
               keys 'word' (str), 'start' (float), 'end' (float).
        scene_duration_sec: Known duration of the scene audio in seconds.

    Returns:
        Assessment dict with keys:
            status: "OK" or "COLLAPSED"
            word_count, char_count, word_span_sec, scene_duration_sec,
            coverage_ratio, aggregate_cps, anchor_sec
    """
    result = {
        "status": "OK",
        "word_count": 0,
        "char_count": 0,
        "word_span_sec": 0.0,
        "scene_duration_sec": scene_duration_sec,
        "coverage_ratio": 0.0,
        "aggregate_cps": 0.0,
        "anchor_sec": 0.0,
    }

    # Guard: insufficient data
    if not words or scene_duration_sec <= 0:
        return result

    word_count = len(words)
    char_count = sum(len(w.get("word", "")) for w in words)

    result["word_count"] = word_count
    result["char_count"] = char_count

    if char_count <= _MIN_CHAR_COUNT_FOR_ASSESSMENT:
        return result

    # Compute metrics
    first_start = words[0].get("start", 0.0)
    last_end = words[-1].get("end", 0.0)
    word_span_sec = max(0.0, last_end - first_start)

    coverage_ratio = word_span_sec / scene_duration_sec if scene_duration_sec > 0 else 0.0
    aggregate_cps = char_count / word_span_sec if word_span_sec > 0 else float("inf")

    result["word_span_sec"] = word_span_sec
    result["coverage_ratio"] = coverage_ratio
    result["aggregate_cps"] = aggregate_cps
    result["anchor_sec"] = first_start

    # Detection logic (aggregate V1)
    collapsed = False
    reason = []

    if coverage_ratio < _COVERAGE_RATIO_THRESHOLD:
        collapsed = True
        reason.append(f"coverage={coverage_ratio:.3f}<{_COVERAGE_RATIO_THRESHOLD}")

    if aggregate_cps > _AGGREGATE_CPS_THRESHOLD:
        collapsed = True
        reason.append(f"CPS={aggregate_cps:.1f}>{_AGGREGATE_CPS_THRESHOLD}")

    if word_span_sec < _WORD_SPAN_THRESHOLD:
        collapsed = True
        reason.append(f"span={word_span_sec:.3f}s<{_WORD_SPAN_THRESHOLD}s")

    if collapsed:
        result["status"] = "COLLAPSED"
        logger.debug(
            "[SENTINEL] Collapse detected: %s (chars=%d, span=%.3fs, scene=%.1fs)",
            ", ".join(reason), char_count, word_span_sec, scene_duration_sec,
        )

    return result


# ---------------------------------------------------------------------------
# Word extraction (coupled-mode adapter)
# ---------------------------------------------------------------------------

def extract_words_from_result(result) -> List[Dict[str, Any]]:
    """
    Extract word dicts from a WhisperResult's segments.

    Produces the same [{word, start, end}] format as merge_master_with_timestamps(),
    enabling the existing sentinel + recovery pipeline to work on coupled-mode results
    (where only the final WhisperResult is available, not raw word dicts).

    Args:
        result: A stable_whisper.WhisperResult with .segments, each segment
                having .text, .start, .end, and optionally .words attributes.

    Returns:
        List of word dicts: [{'word': str, 'start': float, 'end': float}, ...]
    """
    words = []
    for seg in result.segments:
        if hasattr(seg, 'words') and seg.words:
            for w in seg.words:
                words.append({'word': w.word, 'start': w.start, 'end': w.end})
        else:
            # Segment has no word-level data — treat segment text as single "word"
            words.append({'word': seg.text.strip(), 'start': seg.start, 'end': seg.end})
    return words


# ---------------------------------------------------------------------------
# Recovery dispatcher
# ---------------------------------------------------------------------------

def redistribute_collapsed_words(
    words: List[Dict[str, Any]],
    scene_duration_sec: float,
    speech_regions: Optional[List[Tuple[float, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Recover from alignment collapse by redistributing word timestamps.

    Returns a NEW word list with corrected timestamps (does not mutate input).

    Args:
        words: Collapsed word list [{'word': str, 'start': float, 'end': float}, ...]
        scene_duration_sec: Known scene duration in seconds.
        speech_regions: VAD regions as [(start_sec, end_sec), ...], sorted by
                        start time. None if no VAD data available.

    Returns:
        New word list with redistributed timestamps.
    """
    if not words:
        return []

    total_chars = sum(len(w.get("word", "")) for w in words)
    if total_chars == 0:
        total_chars = len(words)  # fallback: equal distribution

    if speech_regions:
        return _distribute_words_across_regions(words, speech_regions, total_chars)
    else:
        return _distribute_words_from_anchor(words, scene_duration_sec, total_chars)


# ---------------------------------------------------------------------------
# Strategy C: VAD-guided distribution
# ---------------------------------------------------------------------------

def _distribute_words_across_regions(
    words: List[Dict[str, Any]],
    speech_regions: List[Tuple[float, float]],
    total_chars: int,
) -> List[Dict[str, Any]]:
    """
    Strategy C (Hybrid Anchor + VAD). Distribute words proportionally by
    character count across VAD speech regions, skipping silence gaps.

    Algorithm (speech timeline mapping):
        1. Sort regions by start time
        2. Calculate total_speech_dur = sum of all region durations
        3. For each word, compute fractional position in speech timeline
        4. Map timeline position to real time via _timeline_to_real()
    """
    # Sort and filter valid regions
    regions = sorted(
        [(s, e) for s, e in speech_regions if e > s],
        key=lambda r: r[0],
    )
    if not regions:
        # No valid regions — fall back to proportional
        logger.debug("[SENTINEL] No valid VAD regions, falling back to proportional")
        scene_end = max(w.get("end", 0.0) for w in words) if words else 0.0
        return _distribute_words_from_anchor(words, scene_end, total_chars)

    total_speech_dur = sum(e - s for s, e in regions)
    if total_speech_dur <= 0:
        scene_end = regions[-1][1]
        return _distribute_words_from_anchor(words, scene_end, total_chars)

    result = []
    cumulative_chars = 0

    for w in words:
        word_text = w.get("word", "")
        word_chars = len(word_text) or 1  # avoid zero-length

        # Fractional position in the speech timeline
        frac_start = cumulative_chars / total_chars
        frac_end = (cumulative_chars + word_chars) / total_chars

        # Map to real time
        timeline_start = frac_start * total_speech_dur
        timeline_end = frac_end * total_speech_dur

        real_start = _timeline_to_real(timeline_start, regions)
        real_end = _timeline_to_real(timeline_end, regions)

        # Ensure non-zero word duration (minimum 20ms)
        if real_end <= real_start:
            real_end = real_start + 0.02

        result.append({
            "word": word_text,
            "start": round(real_start, 3),
            "end": round(real_end, 3),
        })

        cumulative_chars += word_chars

    return result


# ---------------------------------------------------------------------------
# Strategy B: Proportional from anchor (fallback)
# ---------------------------------------------------------------------------

def _distribute_words_from_anchor(
    words: List[Dict[str, Any]],
    scene_duration_sec: float,
    total_chars: int,
) -> List[Dict[str, Any]]:
    """
    Strategy B (proportional fallback). When no VAD data available.

    Uses the aligner's anchor position (first word start) as a hint for
    where speech begins, then distributes words at conversational speed.
    """
    if not words:
        return []

    anchor = words[0].get("start", 0.0)

    # Estimate speech duration from character count
    estimated_duration = total_chars / _TARGET_CPS

    # Start from anchor, expand forward
    start = anchor
    end = start + estimated_duration

    # Clamp to scene boundaries
    if end > scene_duration_sec:
        end = scene_duration_sec
        # If anchor is near scene end, expand backwards
        if end - start < estimated_duration * 0.5:
            start = max(0.0, scene_duration_sec - estimated_duration)

    # Ensure we have a valid range
    if end <= start:
        start = 0.0
        end = scene_duration_sec

    span = end - start

    result = []
    cumulative_chars = 0

    for w in words:
        word_text = w.get("word", "")
        word_chars = len(word_text) or 1

        frac_start = cumulative_chars / total_chars
        frac_end = (cumulative_chars + word_chars) / total_chars

        word_start = start + frac_start * span
        word_end = start + frac_end * span

        # Minimum 20ms word duration
        if word_end <= word_start:
            word_end = word_start + 0.02

        result.append({
            "word": word_text,
            "start": round(word_start, 3),
            "end": round(word_end, 3),
        })

        cumulative_chars += word_chars

    return result


# ---------------------------------------------------------------------------
# Helper: timeline-to-real mapping
# ---------------------------------------------------------------------------

def _timeline_to_real(
    timeline_pos: float,
    regions: List[Tuple[float, float]],
) -> float:
    """
    Map a position in the flattened speech timeline to real time.

    The "speech timeline" is a continuous axis from 0 to total_speech_dur
    (sum of all region durations). This function maps a position on that
    axis back to real time, accounting for silence gaps between regions.

    Args:
        timeline_pos: Position in the flattened speech timeline (seconds).
        regions: Sorted list of (start_sec, end_sec) speech regions.

    Returns:
        Real time in seconds.
    """
    cumulative = 0.0

    for region_start, region_end in regions:
        region_dur = region_end - region_start
        if region_dur <= 0:
            continue

        if cumulative + region_dur >= timeline_pos:
            # Position falls within this region
            offset_in_region = timeline_pos - cumulative
            return region_start + offset_in_region

        cumulative += region_dur

    # Past the end of all regions — clamp to last region end
    if regions:
        return regions[-1][1]
    return 0.0
