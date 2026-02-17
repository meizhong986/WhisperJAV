"""
Post-reconstruction hardening for the Decoupled Subtitle Pipeline.

Applies timestamp resolution, boundary clamping, and chronological sorting
to a WhisperResult.  Shared by ALL pipeline paths regardless of how the
result was produced (assembly, coupled, aligner-free).

Extracted from qwen_pipeline.py (lines 1888-2109, 1220-1242, 1336/1367).
All methods converted from QwenPipeline @staticmethod to module-level
functions.

Audit findings resolved:
    H1  — VAD_ONLY now distributes proportionally (was blanket assignment)
    H2  — Assembly mode now gets timestamp resolution (via orchestrator)
    H4  — Diagnostics measured after reconstruction with correct counts
    M7  — All modes respect timestamp_mode (single code path)
"""

from whisperjav.modules.subtitle_pipeline.types import (
    HardeningConfig,
    HardeningDiagnostics,
    TimestampMode,
)
from whisperjav.utils.logger import logger

# We import stable_whisper types but avoid a hard dependency at module load.
# The WhisperResult type is only needed at call time.
try:
    import stable_whisper
except ImportError:
    stable_whisper = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def harden_scene_result(
    result,  # stable_whisper.WhisperResult
    config: HardeningConfig,
) -> HardeningDiagnostics:
    """
    Apply post-reconstruction hardening to a WhisperResult.

    Mutates ``result`` in-place.  Returns diagnostics describing what
    was applied.

    Steps:
        1. Timestamp resolution (based on config.timestamp_mode)
        2. Boundary clamping [0, scene_duration]
        3. Chronological segment sort
        4. Diagnostic snapshot

    Args:
        result: A stable_whisper.WhisperResult to harden.
        config: Hardening configuration (timestamp mode, scene duration).

    Returns:
        HardeningDiagnostics with counts of what was applied.
    """
    diag = HardeningDiagnostics(timestamp_mode=config.timestamp_mode.value)

    if result is None or not hasattr(result, "segments") or not result.segments:
        return diag

    # Step 1: Timestamp resolution
    mode = config.timestamp_mode
    duration = config.scene_duration_sec

    if mode == TimestampMode.VAD_ONLY:
        _apply_vad_only_timestamps(result, duration)
    elif mode == TimestampMode.ALIGNER_WITH_INTERPOLATION:
        diag.interpolated_count = _apply_timestamp_interpolation(result, duration)
    elif mode == TimestampMode.ALIGNER_WITH_VAD_FALLBACK:
        diag.fallback_count = _apply_vad_timestamp_fallback(result, duration)
    # ALIGNER_ONLY: no-op — keep whatever the aligner produced

    # Step 2: Boundary clamping
    if duration > 0:
        diag.clamped_count = _clamp_timestamps(result, duration)

    # Step 3: Chronological sort
    diag.sorted = _sort_segments_chronologically(result)

    # Step 4: Diagnostics
    diag.segment_count = len(result.segments) if result.segments else 0

    return diag


# ---------------------------------------------------------------------------
# Timestamp resolution: interpolation (default)
# ---------------------------------------------------------------------------
# Extracted from QwenPipeline._apply_timestamp_interpolation
# (qwen_pipeline.py:1978-2109). Literal extraction — logic preserved exactly.


def _apply_timestamp_interpolation(
    result,  # stable_whisper.WhisperResult
    group_duration: float = 0.0,
) -> int:
    """
    Interpolate timestamps for segments where the aligner returned NULL.

    Algorithm:
        1. Identify "anchor" segments with valid timestamps (end > 0)
        2. Identify "gap" segments between anchors with NULL timestamps
        3. For each gap: distribute time proportionally by character count

    Edge cases:
        - Leading NULLs (no previous anchor): start from 0.0
        - Trailing NULLs (no next anchor): estimate from char count,
          capped to group_duration
        - All NULLs (total aligner failure): return 0 (cannot interpolate)

    Args:
        result: WhisperResult with segments to process.
        group_duration: Duration cap (0 = no cap).

    Returns:
        Number of segments that received interpolated timestamps.
    """
    if not result or not result.segments:
        return 0

    segments = result.segments
    n = len(segments)
    interpolated_count = 0

    # Find all anchor indices (segments with valid timestamps)
    anchors = []
    for i, seg in enumerate(segments):
        if seg.end > 0:
            anchors.append(i)

    # No anchors → cannot interpolate
    if not anchors:
        logger.debug("No anchor timestamps found, cannot interpolate")
        return 0

    def interpolate_gap(gap_indices: list[int], start_time: float, end_time: float):
        """Distribute time proportionally by character count."""
        nonlocal interpolated_count

        if not gap_indices:
            return

        total_chars = sum(len(segments[i].text.strip()) for i in gap_indices)
        if total_chars == 0:
            total_chars = len(gap_indices)  # equal distribution fallback

        gap_duration = end_time - start_time
        if gap_duration <= 0:
            gap_duration = 0.5 * len(gap_indices)  # 0.5s per segment fallback

        current_time = start_time
        for idx in gap_indices:
            seg = segments[idx]
            seg_chars = len(seg.text.strip()) or 1
            seg_duration = gap_duration * (seg_chars / total_chars)

            seg.start = current_time
            seg.end = current_time + seg_duration
            current_time = seg.end
            interpolated_count += 1

    # Handle leading gap (segments before first anchor)
    if anchors[0] > 0:
        leading_indices = list(range(0, anchors[0]))
        leading_end = segments[anchors[0]].start
        interpolate_gap(leading_indices, 0.0, leading_end)

    # Handle gaps between anchors
    for i in range(len(anchors) - 1):
        prev_anchor_idx = anchors[i]
        next_anchor_idx = anchors[i + 1]

        gap_indices = []
        for j in range(prev_anchor_idx + 1, next_anchor_idx):
            if segments[j].end <= 0:
                gap_indices.append(j)

        if gap_indices:
            gap_start = segments[prev_anchor_idx].end
            gap_end = segments[next_anchor_idx].start
            interpolate_gap(gap_indices, gap_start, gap_end)

    # Handle trailing gap (segments after last anchor)
    if anchors[-1] < n - 1:
        trailing_indices = []
        for j in range(anchors[-1] + 1, n):
            if segments[j].end <= 0:
                trailing_indices.append(j)

        if trailing_indices:
            trailing_start = segments[anchors[-1]].end
            # Estimate trailing duration: ~50ms per character (conservative)
            total_trailing_chars = sum(len(segments[i].text.strip()) for i in trailing_indices)
            estimated_duration = max(0.5, total_trailing_chars * 0.05)

            # Cap to group_duration to prevent overflow
            if group_duration > 0:
                max_trailing = max(0.0, group_duration - trailing_start)
                if estimated_duration > max_trailing:
                    logger.debug(
                        "Interpolation: trailing gap capped from %.2fs to %.2fs "
                        "(group_duration=%.2fs, trailing_start=%.2fs)",
                        estimated_duration,
                        max_trailing,
                        group_duration,
                        trailing_start,
                    )
                    estimated_duration = max(0.1, max_trailing)

            interpolate_gap(trailing_indices, trailing_start, trailing_start + estimated_duration)

    return interpolated_count


# ---------------------------------------------------------------------------
# Timestamp resolution: VAD fallback
# ---------------------------------------------------------------------------
# Extracted from QwenPipeline._apply_vad_timestamp_fallback
# (qwen_pipeline.py:1888-1950). Literal extraction — logic preserved exactly.


def _apply_vad_timestamp_fallback(
    result,  # stable_whisper.WhisperResult
    group_duration: float,
) -> int:
    """
    Apply VAD timestamp fallback for segments with null aligner timestamps.

    Detects segments with end <= 0.0 (aligner failure) and distributes them
    proportionally by character count across the group duration.

    Operates on words if available (to avoid double-offset via stable-ts
    Segment property setters), falls back to segment-level assignment.

    Args:
        result: WhisperResult with region-relative timestamps.
        group_duration: Duration of the scene/group in seconds.

    Returns:
        Number of segments that received fallback timestamps.
    """
    if not result or not result.segments:
        return 0

    # Collect segments needing fallback
    fallback_segs = [seg for seg in result.segments if seg.end <= 0.0]
    if not fallback_segs:
        return 0

    # Proportional distribution by character count
    total_chars = sum(len(seg.text.strip()) or 1 for seg in fallback_segs)
    cumulative = 0

    for seg in fallback_segs:
        seg_chars = len(seg.text.strip()) or 1
        frac_start = cumulative / total_chars
        frac_end = (cumulative + seg_chars) / total_chars

        if hasattr(seg, "words") and seg.words:
            # Word-level assignment (avoids double-offset via property setters)
            word_total = sum(len(w.word) or 1 for w in seg.words)
            w_cum = 0
            seg_span = (frac_end - frac_start) * group_duration
            seg_base = frac_start * group_duration
            for w in seg.words:
                w_chars = len(w.word) or 1
                w.start = seg_base + (w_cum / word_total) * seg_span
                w.end = seg_base + ((w_cum + w_chars) / word_total) * seg_span
                w_cum += w_chars
        else:
            seg.start = frac_start * group_duration
            seg.end = frac_end * group_duration

        cumulative += seg_chars

    return len(fallback_segs)


# ---------------------------------------------------------------------------
# Timestamp resolution: VAD-only (FIXED — audit H1)
# ---------------------------------------------------------------------------
# Extracted from QwenPipeline._apply_vad_only_timestamps
# (qwen_pipeline.py:1952-1975).
#
# **FIX**: Original assigned ALL segments the identical range [0, duration],
# creating overlapping subtitles.  Now distributes proportionally by
# character count — same algorithm as _apply_vad_timestamp_fallback but
# applied to ALL segments, not just null-timestamp ones.


def _apply_vad_only_timestamps(
    result,  # stable_whisper.WhisperResult
    group_duration: float,
) -> None:
    """
    Force proportional timestamps on all segments, discarding aligner output.

    Used in VAD_ONLY mode where aligner timestamps are intentionally ignored
    in favour of the speech segmenter's time boundaries.

    All segments are distributed proportionally by character count across
    [0, group_duration].

    Args:
        result: WhisperResult with segments.
        group_duration: Duration to distribute across.
    """
    if not result or not result.segments:
        return

    segments = result.segments
    total_chars = sum(len(seg.text.strip()) or 1 for seg in segments)
    cumulative = 0

    for seg in segments:
        seg_chars = len(seg.text.strip()) or 1
        frac_start = cumulative / total_chars
        frac_end = (cumulative + seg_chars) / total_chars

        if hasattr(seg, "words") and seg.words:
            # Word-level assignment (avoids double-offset via property setters)
            word_total = sum(len(w.word) or 1 for w in seg.words)
            w_cum = 0
            seg_span = (frac_end - frac_start) * group_duration
            seg_base = frac_start * group_duration
            for w in seg.words:
                w_chars = len(w.word) or 1
                w.start = seg_base + (w_cum / word_total) * seg_span
                w.end = seg_base + ((w_cum + w_chars) / word_total) * seg_span
                w_cum += w_chars
        else:
            seg.start = frac_start * group_duration
            seg.end = frac_end * group_duration

        cumulative += seg_chars


# ---------------------------------------------------------------------------
# Boundary clamping
# ---------------------------------------------------------------------------
# Extracted from inline code at qwen_pipeline.py:1220-1227 and 1234-1242.
# Unified into a single function operating in scene-relative coordinates.


def _clamp_timestamps(
    result,  # stable_whisper.WhisperResult
    max_duration: float,
) -> int:
    """
    Clamp all segment/word timestamps to [0, max_duration].

    Prevents interpolation overflow or aligner drift from producing
    timestamps beyond the scene boundary.

    Operates on words when available (stable-ts Segment.start/.end property
    setters propagate to underlying words, so word-level is the safe path).

    Args:
        result: WhisperResult to clamp.
        max_duration: Maximum allowed timestamp (scene duration).

    Returns:
        Number of timestamps that were clamped.
    """
    clamped = 0

    for seg in result.segments:
        if hasattr(seg, "words") and seg.words:
            for word in seg.words:
                new_start = max(0.0, min(word.start, max_duration))
                new_end = max(new_start, min(word.end, max_duration))
                if new_start != word.start or new_end != word.end:
                    clamped += 1
                word.start = new_start
                word.end = new_end
        else:
            new_start = max(0.0, min(seg.start, max_duration))
            new_end = max(new_start, min(seg.end, max_duration))
            if new_start != seg.start or new_end != seg.end:
                clamped += 1
            seg.start = new_start
            seg.end = new_end

    return clamped


# ---------------------------------------------------------------------------
# Chronological sorting
# ---------------------------------------------------------------------------
# Extracted from inline sort at qwen_pipeline.py:1336 and 1367-1368.


def _sort_segments_chronologically(
    result,  # stable_whisper.WhisperResult
) -> bool:
    """
    Sort segments by start time.  Defensive safety net after timestamp
    resolution and sentinel recovery, which can produce out-of-order segments.

    Args:
        result: WhisperResult to sort.

    Returns:
        True if any segments were reordered, False if already sorted.
    """
    if not result.segments or len(result.segments) <= 1:
        return False

    starts_before = [seg.start for seg in result.segments]
    result.segments.sort(key=lambda s: s.start)
    starts_after = [seg.start for seg in result.segments]

    reordered = starts_before != starts_after
    if reordered:
        logger.debug(
            "[HARDENING] Segments reordered by chronological sort (%d segments)",
            len(result.segments),
        )
    return reordered
