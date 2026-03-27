"""
Metrics computation module for the pipeline analysis test suite.

Pure computation — no WhisperJAV imports. Takes BackendRunResult and
optional ground truth, produces AnalysisResult and ComparisonResult.
"""

import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import (
    AnalysisResult,
    BackendRunResult,
    ComparisonResult,
    SegmentInfo,
)


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def parse_srt_file(srt_path: Path) -> List[SegmentInfo]:
    """Parse SRT subtitle file into SegmentInfo list.

    Args:
        srt_path: Path to SRT file

    Returns:
        List of SegmentInfo with start/end timestamps
    """
    segments: List[SegmentInfo] = []

    # Try multiple encodings
    content = None
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            with open(srt_path, "r", encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        print(f"  Warning: Could not decode SRT file: {srt_path}")
        return []

    for match in _TIMESTAMP_RE.finditer(content):
        groups = match.groups()
        start_h, start_m, start_s, start_ms = map(int, groups[:4])
        end_h, end_m, end_s, end_ms = map(int, groups[4:])

        start_sec = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
        end_sec = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000

        segments.append(SegmentInfo(start_sec=start_sec, end_sec=end_sec))

    return segments


# ---------------------------------------------------------------------------
# Time coverage array (boolean mask on time grid)
# ---------------------------------------------------------------------------


def compute_time_coverage_array(
    segments: List[SegmentInfo],
    duration_sec: float,
    resolution_ms: int = 10,
) -> np.ndarray:
    """Convert segment list to boolean time-grid array.

    Args:
        segments: List of SegmentInfo
        duration_sec: Total audio duration
        resolution_ms: Grid granularity in milliseconds (10ms = 100 pts/sec)

    Returns:
        Boolean numpy array of shape (num_bins,) where True = covered
    """
    num_bins = max(1, int(duration_sec * 1000 / resolution_ms))
    coverage = np.zeros(num_bins, dtype=bool)

    for seg in segments:
        start_bin = max(0, int(seg.start_sec * 1000 / resolution_ms))
        end_bin = min(num_bins, int(seg.end_sec * 1000 / resolution_ms))
        if start_bin < end_bin:
            coverage[start_bin:end_bin] = True

    return coverage


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------


def find_gaps(
    segments: List[SegmentInfo],
    duration_sec: float,
    min_gap_sec: float = 0.5,
) -> List[SegmentInfo]:
    """Find time regions not covered by any segment.

    Args:
        segments: List of SegmentInfo (need not be sorted)
        duration_sec: Total audio duration
        min_gap_sec: Minimum gap duration to report

    Returns:
        List of SegmentInfo representing gaps
    """
    if not segments:
        if duration_sec >= min_gap_sec:
            return [SegmentInfo(start_sec=0.0, end_sec=duration_sec)]
        return []

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: s.start_sec)

    gaps: List[SegmentInfo] = []

    # Gap before first segment
    if sorted_segs[0].start_sec >= min_gap_sec:
        gaps.append(
            SegmentInfo(start_sec=0.0, end_sec=sorted_segs[0].start_sec)
        )

    # Gaps between segments
    for i in range(len(sorted_segs) - 1):
        gap_start = sorted_segs[i].end_sec
        gap_end = sorted_segs[i + 1].start_sec
        if gap_end - gap_start >= min_gap_sec:
            gaps.append(SegmentInfo(start_sec=gap_start, end_sec=gap_end))

    # Gap after last segment
    last_end = sorted_segs[-1].end_sec
    if duration_sec - last_end >= min_gap_sec:
        gaps.append(SegmentInfo(start_sec=last_end, end_sec=duration_sec))

    return gaps


# ---------------------------------------------------------------------------
# Single backend analysis
# ---------------------------------------------------------------------------


def compute_metrics(
    result: BackendRunResult,
    audio_duration_sec: float,
) -> AnalysisResult:
    """Compute descriptive statistics for a single backend run.

    Args:
        result: BackendRunResult from a successful run
        audio_duration_sec: Total audio duration in seconds

    Returns:
        AnalysisResult with computed metrics
    """
    segments = result.segments

    if not segments:
        return AnalysisResult(
            backend_name=result.backend_name,
            backend_type=result.backend_type,
            num_segments=0,
            total_coverage_sec=0.0,
            coverage_ratio=0.0,
            mean_segment_duration=0.0,
            median_segment_duration=0.0,
            min_segment_duration=0.0,
            max_segment_duration=0.0,
            std_segment_duration=0.0,
            processing_time_sec=result.processing_time_sec,
        )

    durations = [s.duration_sec for s in segments]
    total_coverage = sum(durations)
    coverage_ratio = (
        total_coverage / audio_duration_sec if audio_duration_sec > 0 else 0.0
    )

    return AnalysisResult(
        backend_name=result.backend_name,
        backend_type=result.backend_type,
        num_segments=len(segments),
        total_coverage_sec=total_coverage,
        coverage_ratio=coverage_ratio,
        mean_segment_duration=statistics.mean(durations),
        median_segment_duration=statistics.median(durations),
        min_segment_duration=min(durations),
        max_segment_duration=max(durations),
        std_segment_duration=(
            statistics.stdev(durations) if len(durations) > 1 else 0.0
        ),
        processing_time_sec=result.processing_time_sec,
    )


# ---------------------------------------------------------------------------
# Ground truth comparison
# ---------------------------------------------------------------------------


def _compute_overlap(seg_a: SegmentInfo, seg_b: SegmentInfo) -> float:
    """Compute temporal overlap between two segments in seconds."""
    overlap_start = max(seg_a.start_sec, seg_b.start_sec)
    overlap_end = min(seg_a.end_sec, seg_b.end_sec)
    return max(0.0, overlap_end - overlap_start)


def compute_ground_truth_metrics(
    result: BackendRunResult,
    ground_truth: List[SegmentInfo],
    audio_duration_sec: float,
    overlap_threshold: float = 0.5,
) -> AnalysisResult:
    """Compute precision/recall/F1 against ground truth segments.

    A ground truth segment is "recalled" if any detected segment overlaps
    at least overlap_threshold of the ground truth segment's duration.

    A detected segment is "precise" if it overlaps at least overlap_threshold
    of its own duration with any ground truth segment.

    Args:
        result: BackendRunResult from a successful run
        ground_truth: List of ground truth SegmentInfo
        audio_duration_sec: Total audio duration
        overlap_threshold: Minimum overlap ratio to count as a match (0-1)

    Returns:
        AnalysisResult with ground truth metrics populated
    """
    # First compute basic metrics
    analysis = compute_metrics(result, audio_duration_sec)

    if not ground_truth or not result.segments:
        analysis.gt_recall = 0.0 if ground_truth else None
        analysis.gt_precision = 0.0 if result.segments else None
        analysis.gt_f1 = 0.0
        analysis.gt_matched_count = 0
        analysis.gt_total_count = len(ground_truth) if ground_truth else 0
        return analysis

    detected = result.segments

    # Recall: what fraction of ground truth segments are captured?
    recalled = 0
    for gt_seg in ground_truth:
        gt_dur = gt_seg.duration_sec
        if gt_dur <= 0:
            continue
        # Check if any detected segment covers enough of this GT segment
        total_overlap = 0.0
        for det_seg in detected:
            total_overlap += _compute_overlap(gt_seg, det_seg)
        if total_overlap / gt_dur >= overlap_threshold:
            recalled += 1

    recall = recalled / len(ground_truth)

    # Precision: what fraction of detected segments overlap with ground truth?
    precise = 0
    for det_seg in detected:
        det_dur = det_seg.duration_sec
        if det_dur <= 0:
            continue
        total_overlap = 0.0
        for gt_seg in ground_truth:
            total_overlap += _compute_overlap(det_seg, gt_seg)
        if total_overlap / det_dur >= overlap_threshold:
            precise += 1

    precision = precise / len(detected)

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    analysis.gt_recall = recall
    analysis.gt_precision = precision
    analysis.gt_f1 = f1
    analysis.gt_matched_count = recalled
    analysis.gt_total_count = len(ground_truth)

    return analysis


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------


def compute_pairwise_comparison(
    result_a: BackendRunResult,
    result_b: BackendRunResult,
    audio_duration_sec: float,
    resolution_ms: int = 10,
) -> ComparisonResult:
    """Compare two backend runs using time-grid IoU.

    Args:
        result_a: First backend result
        result_b: Second backend result
        audio_duration_sec: Total audio duration
        resolution_ms: Grid resolution for IoU computation

    Returns:
        ComparisonResult with IoU and difference metrics
    """
    coverage_a = compute_time_coverage_array(
        result_a.segments, audio_duration_sec, resolution_ms
    )
    coverage_b = compute_time_coverage_array(
        result_b.segments, audio_duration_sec, resolution_ms
    )

    intersection = np.sum(coverage_a & coverage_b)
    union = np.sum(coverage_a | coverage_b)
    iou = float(intersection / union) if union > 0 else 0.0

    # Coverage ratios
    ratio_a = float(np.sum(coverage_a)) / len(coverage_a) if len(coverage_a) > 0 else 0.0
    ratio_b = float(np.sum(coverage_b)) / len(coverage_b) if len(coverage_b) > 0 else 0.0

    return ComparisonResult(
        backend_a=result_a.backend_name,
        backend_b=result_b.backend_name,
        iou_score=iou,
        coverage_diff=abs(ratio_a - ratio_b),
        segment_count_diff=abs(
            len(result_a.segments) - len(result_b.segments)
        ),
    )
