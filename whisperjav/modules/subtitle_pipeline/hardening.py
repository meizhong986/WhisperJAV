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

Gap fixes (timestamp architecture redesign):
    G2  — aligner_vad_fallback now uses speech_regions as anchors
    G4  — vad_only hardening is a no-op (frame boundaries are the output)
"""

from typing import Optional

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
        diag.fallback_count = _apply_vad_timestamp_fallback(
            result, duration, speech_regions=config.speech_regions,
        )
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
# Timestamp resolution: VAD fallback (G2 rewrite — speech-region-aware)
# ---------------------------------------------------------------------------
# Original: distributed null-timestamp segments proportionally across the
# entire [0, group_duration] range — no VAD awareness, cruder than
# interpolation (inverted granularity problem).
#
# G2 fix: Uses anchor + gap structure (like interpolation) but constrains
# gap distribution to speech_regions — words land in speech, not silence.
# Falls back to proportional when no speech_regions available.


def _apply_vad_timestamp_fallback(
    result,  # stable_whisper.WhisperResult
    group_duration: float,
    speech_regions: Optional[list[tuple[float, float]]] = None,
) -> int:
    """
    Apply VAD-aware timestamp fallback for segments with null aligner timestamps.

    Detects segments with end <= 0.0 (aligner failure) and distributes them
    within speech regions if available, falling back to proportional
    distribution across [0, group_duration] when no regions are provided.

    Algorithm:
        1. Find anchors (segments with end > 0) and gaps (end <= 0)
        2. For each gap between anchors:
           a. Clip speech_regions to the anchor window
           b. Distribute gap segments within speech portions
        3. Handle leading/trailing gaps similarly
        4. If no speech_regions: proportional fallback (original behavior)

    Operates on words if available (to avoid double-offset via stable-ts
    Segment property setters), falls back to segment-level assignment.

    Args:
        result: WhisperResult with region-relative timestamps.
        group_duration: Duration of the scene/group in seconds.
        speech_regions: Optional VAD speech regions [(start, end), ...].

    Returns:
        Number of segments that received fallback timestamps.
    """
    if not result or not result.segments:
        return 0

    segments = result.segments

    # Collect segments needing fallback (null timestamps)
    fallback_indices = [i for i, seg in enumerate(segments) if seg.end <= 0.0]
    if not fallback_indices:
        return 0

    # Find anchor indices (valid aligner timestamps)
    anchor_indices = [i for i, seg in enumerate(segments) if seg.end > 0.0]

    # No anchors AND no speech regions → full proportional fallback
    if not anchor_indices and not speech_regions:
        return _proportional_fallback(segments, fallback_indices, 0.0, group_duration)

    # No anchors but speech regions available → distribute all in speech
    if not anchor_indices:
        return _distribute_gap_in_speech(
            segments, fallback_indices, 0.0, group_duration, speech_regions,
        )

    fallback_count = 0

    # Handle leading gap (before first anchor)
    if anchor_indices[0] > 0:
        leading_gap = [i for i in range(0, anchor_indices[0]) if segments[i].end <= 0.0]
        if leading_gap:
            window_end = segments[anchor_indices[0]].start
            fallback_count += _distribute_gap_in_speech(
                segments, leading_gap, 0.0, window_end, speech_regions,
            )

    # Handle gaps between anchors
    for k in range(len(anchor_indices) - 1):
        prev_idx = anchor_indices[k]
        next_idx = anchor_indices[k + 1]

        gap = [i for i in range(prev_idx + 1, next_idx) if segments[i].end <= 0.0]
        if gap:
            window_start = segments[prev_idx].end
            window_end = segments[next_idx].start
            fallback_count += _distribute_gap_in_speech(
                segments, gap, window_start, window_end, speech_regions,
            )

    # Handle trailing gap (after last anchor)
    last_anchor = anchor_indices[-1]
    if last_anchor < len(segments) - 1:
        trailing_gap = [i for i in range(last_anchor + 1, len(segments)) if segments[i].end <= 0.0]
        if trailing_gap:
            window_start = segments[last_anchor].end
            fallback_count += _distribute_gap_in_speech(
                segments, trailing_gap, window_start, group_duration, speech_regions,
            )

    return fallback_count


def _distribute_gap_in_speech(
    segments,
    gap_indices: list[int],
    window_start: float,
    window_end: float,
    speech_regions: Optional[list[tuple[float, float]]],
) -> int:
    """
    Distribute gap segments within speech regions of a time window.

    If speech_regions are available, clips them to [window_start, window_end]
    and distributes items proportionally by character count within speech
    portions only. Falls back to even distribution if no clipped regions.

    Args:
        segments: Full segments list (mutated in-place).
        gap_indices: Indices of null-timestamp segments to distribute.
        window_start: Start of the time window.
        window_end: End of the time window.
        speech_regions: Optional VAD speech regions.

    Returns:
        Number of segments distributed.
    """
    if not gap_indices:
        return 0

    # Clip speech regions to the window
    clipped = _clip_regions(speech_regions, window_start, window_end) if speech_regions else []

    if clipped:
        total_speech_dur = sum(e - s for s, e in clipped)
        if total_speech_dur > 0:
            # Distribute within speech timeline, then map back to real time
            total_chars = sum(len(segments[i].text.strip()) or 1 for i in gap_indices)
            cumulative = 0

            for idx in gap_indices:
                seg = segments[idx]
                seg_chars = len(seg.text.strip()) or 1
                frac_start = cumulative / total_chars
                frac_end = (cumulative + seg_chars) / total_chars

                timeline_start = frac_start * total_speech_dur
                timeline_end = frac_end * total_speech_dur

                real_start = _timeline_to_real(timeline_start, clipped)
                real_end = _timeline_to_real(timeline_end, clipped)

                # Ensure non-zero segment duration (minimum 20ms)
                if real_end <= real_start:
                    real_end = real_start + 0.02

                _assign_segment_timestamps(seg, real_start, real_end)
                cumulative += seg_chars

            return len(gap_indices)

    # No usable speech regions in window — proportional fallback
    return _proportional_fallback(segments, gap_indices, window_start, window_end)


def _proportional_fallback(
    segments,
    gap_indices: list[int],
    start: float,
    end: float,
) -> int:
    """Distribute gap segments proportionally by char count across [start, end]."""
    if not gap_indices:
        return 0

    duration = end - start
    if duration <= 0:
        duration = 0.5 * len(gap_indices)

    total_chars = sum(len(segments[i].text.strip()) or 1 for i in gap_indices)
    cumulative = 0

    for idx in gap_indices:
        seg = segments[idx]
        seg_chars = len(seg.text.strip()) or 1
        frac_start = cumulative / total_chars
        frac_end = (cumulative + seg_chars) / total_chars

        seg_start = start + frac_start * duration
        seg_end = start + frac_end * duration

        _assign_segment_timestamps(seg, seg_start, seg_end)
        cumulative += seg_chars

    return len(gap_indices)


def _assign_segment_timestamps(seg, start: float, end: float) -> None:
    """Assign timestamps to a segment, operating at word-level when possible."""
    if hasattr(seg, "words") and seg.words:
        # Word-level assignment (avoids double-offset via stable-ts property setters)
        word_total = sum(len(w.word) or 1 for w in seg.words)
        w_cum = 0
        seg_span = end - start
        for w in seg.words:
            w_chars = len(w.word) or 1
            w.start = start + (w_cum / word_total) * seg_span
            w.end = start + ((w_cum + w_chars) / word_total) * seg_span
            w_cum += w_chars
    else:
        seg.start = start
        seg.end = end


def _clip_regions(
    regions: list[tuple[float, float]],
    window_start: float,
    window_end: float,
) -> list[tuple[float, float]]:
    """Clip speech regions to a time window, returning only overlapping portions."""
    clipped = []
    for rs, re in regions:
        cs = max(rs, window_start)
        ce = min(re, window_end)
        if ce > cs:
            clipped.append((cs, ce))
    return clipped


def _timeline_to_real(
    timeline_pos: float,
    regions: list[tuple[float, float]],
) -> float:
    """
    Map a position in the flattened speech timeline to real time.

    The "speech timeline" is a continuous axis from 0 to total_speech_dur
    (sum of all region durations). This function maps a position on that
    axis back to real time, accounting for silence gaps between regions.

    Duplicated from alignment_sentinel._timeline_to_real per "redundancy
    over reuse" principle — different module, different concern.

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


# ---------------------------------------------------------------------------
# Timestamp resolution: VAD-only (G4 fix — no-op)
# ---------------------------------------------------------------------------
# With G1 in place, vad_only takes the aligner-free path (Branch B) in the
# orchestrator. Branch B produces frame-boundary-based timing. REGROUP_JAV
# creates subtitle segments that respect those boundaries. Hardening should
# NOT redistribute — the frame boundaries ARE the desired timestamps.


def _apply_vad_only_timestamps(
    result,  # stable_whisper.WhisperResult
    group_duration: float,
) -> None:
    """
    No-op: frame-boundary timing from the aligner-free path is the desired output.

    With aligner=None (set by _build_subtitle_pipeline for VAD_ONLY),
    Branch B creates word dicts with frame boundaries. REGROUP_JAV
    preserves those for short groups (<=8s). Nothing to fix here.

    Args:
        result: WhisperResult with segments (timestamps already correct).
        group_duration: Duration (unused — no redistribution needed).
    """
    return


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
