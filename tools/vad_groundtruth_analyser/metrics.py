"""Timing-accuracy metrics for VAD backends.

Two sections:
    1. GT-mode metrics (frame F1, IoU matching, boundary drift, missed/false-alarm)
    2. Inter-backend agreement (pairwise F1, consensus coverage)

All functions are pure (no I/O) and deterministic. Empty inputs map to zeros,
never NaN. `compute_timing_metrics` is the composed public entry point.
"""

from __future__ import annotations

from statistics import median
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .models import (
    AgreementMatrix,
    BackendMetrics,
    BackendReport,
    GtSegment,
    VadSegment,
)


# ---------------------------------------------------------------------------
# Frame-mask primitives
# ---------------------------------------------------------------------------

def segments_to_mask(
    segments: Sequence,           # Sequence of objects with .start_sec / .end_sec
    duration_sec: float,
    frame_ms: int = 10,
) -> np.ndarray:
    """Build a 1-D bool array where True indicates speech at that frame.

    Frames are uniformly spaced at `frame_ms` milliseconds starting from 0.
    Segments are clipped to [0, duration_sec]. Overlapping segments union-ed.

    Args:
        segments: any iterable with .start_sec and .end_sec attributes.
        duration_sec: total audio duration; mask length = ceil(duration / frame_s).
        frame_ms: grid resolution (default 10 ms → 100 frames/sec).

    Returns:
        1-D numpy bool array.
    """
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0, got {duration_sec}")
    if frame_ms <= 0:
        raise ValueError(f"frame_ms must be > 0, got {frame_ms}")

    frame_s = frame_ms / 1000.0
    n_frames = int(np.ceil(duration_sec / frame_s))
    mask = np.zeros(n_frames, dtype=bool)

    for seg in segments:
        s = max(0.0, float(seg.start_sec))
        e = min(duration_sec, float(seg.end_sec))
        if e <= s:
            continue
        i0 = int(np.floor(s / frame_s))
        i1 = int(np.ceil(e / frame_s))
        i0 = max(0, min(n_frames, i0))
        i1 = max(0, min(n_frames, i1))
        if i1 > i0:
            mask[i0:i1] = True
    return mask


def compute_frame_prf1(
    gt_mask: np.ndarray,
    test_mask: np.ndarray,
) -> Tuple[float, float, float]:
    """Frame-level precision, recall, F1.

    Edge cases:
        - Both masks all-false: P = R = F1 = 0 by convention (no speech to detect).
        - test_mask all-false, gt non-empty: P=0 (no predictions), R=0, F1=0.
        - gt_mask all-false, test non-empty: P=0 (every prediction is false alarm),
          R=0 (no GT to recall), F1=0.

    Returns:
        (precision, recall, f1) — all in [0, 1], no NaN.
    """
    if gt_mask.shape != test_mask.shape:
        # Pad the shorter one with False
        n = max(len(gt_mask), len(test_mask))
        if len(gt_mask) < n:
            gt_mask = np.concatenate([gt_mask, np.zeros(n - len(gt_mask), dtype=bool)])
        if len(test_mask) < n:
            test_mask = np.concatenate([test_mask, np.zeros(n - len(test_mask), dtype=bool)])

    tp = int(np.sum(gt_mask & test_mask))
    fp = int(np.sum(~gt_mask & test_mask))
    fn = int(np.sum(gt_mask & ~test_mask))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Segment IoU + matching
# ---------------------------------------------------------------------------

def _segment_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """IoU of two closed time intervals. 0 if disjoint or degenerate."""
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if overlap <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return overlap / union if union > 0 else 0.0


def compute_iou_matrix(
    gt_segs: Sequence,
    test_segs: Sequence,
) -> np.ndarray:
    """Returns N_gt × N_test IoU matrix; zeros if either side is empty."""
    n_gt, n_test = len(gt_segs), len(test_segs)
    if n_gt == 0 or n_test == 0:
        return np.zeros((n_gt, n_test), dtype=np.float64)
    m = np.zeros((n_gt, n_test), dtype=np.float64)
    for i, g in enumerate(gt_segs):
        for j, t in enumerate(test_segs):
            m[i, j] = _segment_iou(
                g.start_sec, g.end_sec, t.start_sec, t.end_sec,
            )
    return m


def match_segments_by_iou(
    gt_segs: Sequence,
    test_segs: Sequence,
    min_iou: float = 0.1,
) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching by descending IoU.

    Returns a list of (gt_idx, test_idx, iou) triples. Each gt and each test
    appears at most once. Pairs with iou < min_iou are excluded.

    Deterministic: ties broken by (lower gt_idx, lower test_idx).
    """
    if len(gt_segs) == 0 or len(test_segs) == 0:
        return []

    iou = compute_iou_matrix(gt_segs, test_segs)
    # Collect all candidate pairs with iou >= min_iou
    candidates: List[Tuple[float, int, int]] = []
    for i in range(iou.shape[0]):
        for j in range(iou.shape[1]):
            if iou[i, j] >= min_iou:
                candidates.append((float(iou[i, j]), i, j))
    # Sort descending by IoU; ties broken by ascending (i, j)
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_gt, used_test = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for score, i, j in candidates:
        if i in used_gt or j in used_test:
            continue
        matches.append((i, j, score))
        used_gt.add(i)
        used_test.add(j)
    return matches


# ---------------------------------------------------------------------------
# Boundary drift
# ---------------------------------------------------------------------------

def _compute_drift_stats(
    gt_segs: Sequence,
    test_segs: Sequence,
    matches: List[Tuple[int, int, float]],
) -> Tuple[float, float, float, float]:
    """Return (onset_mean_abs_ms, offset_mean_abs_ms, onset_median_ms, offset_median_ms)."""
    if not matches:
        return 0.0, 0.0, 0.0, 0.0
    onset_errs, offset_errs = [], []
    for gi, ti, _ in matches:
        onset_errs.append(abs(gt_segs[gi].start_sec - test_segs[ti].start_sec) * 1000.0)
        offset_errs.append(abs(gt_segs[gi].end_sec - test_segs[ti].end_sec) * 1000.0)
    return (
        float(np.mean(onset_errs)),
        float(np.mean(offset_errs)),
        float(median(onset_errs)),
        float(median(offset_errs)),
    )


# ---------------------------------------------------------------------------
# Coverage error metrics
# ---------------------------------------------------------------------------

def compute_missed_speech_pct(gt_mask: np.ndarray, test_mask: np.ndarray) -> float:
    """% of GT-speech frames not covered by any backend segment."""
    gt_sum = int(np.sum(gt_mask))
    if gt_sum == 0:
        return 0.0
    missed = int(np.sum(gt_mask & ~test_mask))
    return 100.0 * missed / gt_sum


def compute_false_alarm_pct(gt_mask: np.ndarray, test_mask: np.ndarray) -> float:
    """% of backend-speech frames with no GT overlap."""
    test_sum = int(np.sum(test_mask))
    if test_sum == 0:
        return 0.0
    fa = int(np.sum(~gt_mask & test_mask))
    return 100.0 * fa / test_sum


# ---------------------------------------------------------------------------
# Composite public entry point — GT mode
# ---------------------------------------------------------------------------

def compute_timing_metrics(
    gt_segs: Sequence[GtSegment],
    test_segs: Sequence[VadSegment],
    duration_sec: float,
    frame_ms: int = 10,
    iou_match_threshold: float = 0.1,
) -> BackendMetrics:
    """Compose all per-backend timing metrics.

    Guarantees:
        - No NaN in output (empty inputs → zeros)
        - Deterministic for same inputs
        - iou_per_gt has exactly len(gt_segs) entries (0.0 for unmatched)
    """
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0, got {duration_sec}")

    gt_mask = segments_to_mask(gt_segs, duration_sec, frame_ms)
    test_mask = segments_to_mask(test_segs, duration_sec, frame_ms)

    precision, recall, f1 = compute_frame_prf1(gt_mask, test_mask)

    # Segment matching
    matches = match_segments_by_iou(gt_segs, test_segs, min_iou=iou_match_threshold)
    matched_gt = {m[0] for m in matches}

    iou_per_gt: List[float] = [0.0] * len(gt_segs)
    for gi, _, score in matches:
        iou_per_gt[gi] = score

    if matches:
        ious = [m[2] for m in matches]
        iou_mean = float(np.mean(ious))
        iou_median = float(median(ious))
    else:
        iou_mean = 0.0
        iou_median = 0.0

    onset_mean_ms, offset_mean_ms, onset_median_ms, offset_median_ms = _compute_drift_stats(
        gt_segs, test_segs, matches,
    )

    return BackendMetrics(
        frame_precision=float(precision),
        frame_recall=float(recall),
        frame_f1=float(f1),
        iou_mean=iou_mean,
        iou_median=iou_median,
        iou_per_gt=iou_per_gt,
        onset_drift_mean_ms=onset_mean_ms,
        offset_drift_mean_ms=offset_mean_ms,
        onset_drift_median_ms=onset_median_ms,
        offset_drift_median_ms=offset_median_ms,
        missed_speech_pct=compute_missed_speech_pct(gt_mask, test_mask),
        false_alarm_pct=compute_false_alarm_pct(gt_mask, test_mask),
        num_matched_segments=len(matches),
        num_unmatched_gt=len(gt_segs) - len(matched_gt),
    )


# ---------------------------------------------------------------------------
# Inter-backend agreement (GT-less mode & cross-check)
# ---------------------------------------------------------------------------

def compute_agreement_matrix(
    backend_reports: Dict[str, BackendReport],
    duration_sec: float,
    frame_ms: int = 10,
) -> AgreementMatrix:
    """Pairwise F1 matrix across successful backends.

    Only backends with success=True are included. Diagonal is 1.0 by
    definition. Symmetric. Also computes consensus_coverage_pct = % of
    frames where >= 2 backends simultaneously claim speech.

    Returns AgreementMatrix with empty matrix if <2 successful backends.
    """
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0, got {duration_sec}")

    # Filter to successful backends; preserve input order
    items = [(name, rep) for name, rep in backend_reports.items() if rep.success]

    # Build masks in one pass
    masks: Dict[str, np.ndarray] = {}
    for name, rep in items:
        masks[name] = segments_to_mask(rep.segments, duration_sec, frame_ms)

    order = [n for n, _ in items]
    n = len(order)
    pair_f1: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(order):
        pair_f1[i][i] = 1.0
        for j in range(i + 1, n):
            b = order[j]
            _, _, f1 = compute_frame_prf1(masks[a], masks[b])
            pair_f1[i][j] = f1
            pair_f1[j][i] = f1

    # Consensus: frames where sum of masks >= 2
    if n >= 2:
        stacked = np.stack([masks[o] for o in order], axis=0).astype(np.int8)
        counts = stacked.sum(axis=0)
        total = counts.size
        consensus = int(np.sum(counts >= 2))
        consensus_pct = 100.0 * consensus / total if total > 0 else 0.0
    else:
        consensus_pct = 0.0

    return AgreementMatrix(
        backend_order=order,
        pair_f1=pair_f1,
        consensus_coverage_pct=float(consensus_pct),
    )
