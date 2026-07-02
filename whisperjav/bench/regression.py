"""
Accuracy regression scoring and gating for WhisperJAV.

Builds on whisperjav.bench (CER, IoU, subtitle matching) to provide the
release gate that was missing: score a pipeline's SRT output against a
ground-truth corpus, store the result as a baseline, and FAIL when a new
run regresses beyond configured thresholds.

Motivating history (docs/ISSUE_TRACKER_v1.8.x.md):
- v1.8.12 silero max_speech/max_group retune shipped an input-context
  regression (commit 916dee3) - no gate caught it.
- large-v3 + aggressive preset produced catastrophic output (F4/F6:
  6-10 subs vs 68 GT) - discovered manually, forced the large-v2 revert.
- YAML/Pydantic preset mismatch (aggressive threshold 0.08 vs 0.18)
  shipped silently.

All scoring functions are pure and dependency-light (stdlib + existing
bench modules). Running pipelines requires a GPU machine; scoring and
gating run anywhere, including CI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from whisperjav.bench.loader import SubtitleEntry, _parse_srt_file, load_ground_truth
from whisperjav.bench.matcher import match_subtitles
from whisperjav.bench.metrics import (
    compute_cer_from_segments,
    compute_iou,
    normalize_text,
)

SCHEMA_VERSION = "1.0.0"

# IoU below which a matched pair counts as poorly timed (informational).
_TIMING_IOU_FLOOR = 0.3


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ClipMetrics:
    """Accuracy metrics for one pipeline output vs one ground-truth SRT."""

    n_gt: int
    n_hyp: int
    count_ratio: float          # n_hyp / n_gt. Catastrophic collapse: ~0.1
    cer: float                  # global character error rate (lower = better)
    precision: float            # matched / n_hyp
    recall: float               # matched / n_gt (capture rate)
    f1: float
    mean_iou: float             # timing quality over matched pairs
    gt_coverage: float          # GT speech seconds covered by hyp / GT seconds
    false_alarm_ratio: float    # hyp seconds outside GT speech / hyp seconds
    repetition_ratio: float     # longest run of identical texts / n_hyp

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _entries_to_dicts(entries: list[SubtitleEntry]) -> list[dict]:
    return [{"start": e.start, "end": e.end, "text": e.text} for e in entries]


def _overlap_seconds(
    intervals_a: list[tuple[float, float]],
    intervals_b: list[tuple[float, float]],
) -> float:
    """Total seconds of overlap between two interval sets (each pre-merged)."""
    total = 0.0
    i = j = 0
    while i < len(intervals_a) and j < len(intervals_b):
        a_start, a_end = intervals_a[i]
        b_start, b_end = intervals_b[j]
        total += max(0.0, min(a_end, b_end) - max(a_start, b_start))
        if a_end <= b_end:
            i += 1
        else:
            j += 1
    return total


def _merge_intervals(subs: list[dict]) -> list[tuple[float, float]]:
    """Union of subtitle time ranges, sorted and merged."""
    spans = sorted((s["start"], s["end"]) for s in subs if s["end"] > s["start"])
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _repetition_ratio(subs: list[dict]) -> float:
    """Longest run of consecutive identical (normalized) texts / total subs.

    The Whisper repetition pathology produces long runs of the same line;
    a healthy file stays near 1/n.
    """
    if not subs:
        return 0.0
    longest = current = 1
    prev = normalize_text(subs[0]["text"])
    for sub in subs[1:]:
        text = normalize_text(sub["text"])
        if text and text == prev:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
        prev = text
    if longest < 2:
        return 0.0  # no repeated run at all - not a loop, regardless of file size
    return longest / len(subs)


def score_clip(gt_subs: list[dict], hyp_subs: list[dict]) -> ClipMetrics:
    """Score one hypothesis subtitle list against ground truth."""
    n_gt = len(gt_subs)
    n_hyp = len(hyp_subs)

    if n_gt == 0:
        raise ValueError("Ground truth is empty - nothing to score against")

    result = match_subtitles(gt_subs, hyp_subs)
    matched = result["matched"]
    n_matched = len(matched)

    precision = n_matched / n_hyp if n_hyp else 0.0
    recall = n_matched / n_gt
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    ious = [
        compute_iou(gt["start"], gt["end"], hyp["start"], hyp["end"])
        for gt, hyp in matched
    ]
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    # Global CER in temporal order
    gt_sorted = sorted(gt_subs, key=lambda s: s["start"])
    hyp_sorted = sorted(hyp_subs, key=lambda s: s["start"])
    cer = compute_cer_from_segments(
        [s["text"] for s in hyp_sorted],
        [s["text"] for s in gt_sorted],
    )

    gt_intervals = _merge_intervals(gt_subs)
    hyp_intervals = _merge_intervals(hyp_subs)
    gt_seconds = sum(e - s for s, e in gt_intervals)
    hyp_seconds = sum(e - s for s, e in hyp_intervals)
    overlap = _overlap_seconds(gt_intervals, hyp_intervals)

    gt_coverage = overlap / gt_seconds if gt_seconds else 0.0
    false_alarm_ratio = (
        (hyp_seconds - overlap) / hyp_seconds if hyp_seconds else 0.0
    )

    return ClipMetrics(
        n_gt=n_gt,
        n_hyp=n_hyp,
        count_ratio=n_hyp / n_gt,
        cer=round(cer, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        mean_iou=round(mean_iou, 4),
        gt_coverage=round(gt_coverage, 4),
        false_alarm_ratio=round(false_alarm_ratio, 4),
        repetition_ratio=round(_repetition_ratio(hyp_subs), 4),
    )


def score_srt_files(gt_path: Path, hyp_path: Path) -> ClipMetrics:
    """Score a hypothesis SRT file against a ground-truth SRT file."""
    gt = _entries_to_dicts(load_ground_truth(Path(gt_path)))
    hyp_entries = _parse_srt_file(Path(hyp_path)) if Path(hyp_path).exists() else []
    return score_clip(gt, _entries_to_dicts(hyp_entries))


# ---------------------------------------------------------------------------
# Baseline + gate
# ---------------------------------------------------------------------------

# Direction of "better" per metric. +1: higher is better, -1: lower is better.
_METRIC_DIRECTION = {
    "cer": -1,
    "precision": +1,
    "recall": +1,
    "f1": +1,
    "mean_iou": +1,
    "gt_coverage": +1,
    "false_alarm_ratio": -1,
    "repetition_ratio": -1,
}

#: Maximum tolerated regression vs baseline (absolute deltas).
DEFAULT_THRESHOLDS: dict[str, float] = {
    "cer": 0.03,
    "precision": 0.03,
    "recall": 0.03,
    "f1": 0.03,
    "mean_iou": 0.05,
    "gt_coverage": 0.05,
    "false_alarm_ratio": 0.05,
    "repetition_ratio": 0.10,
}

#: Hard bounds that fail regardless of baseline (catastrophic-output guard).
DEFAULT_HARD_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    # F4/F6 signature: 6-10 subs against 68 GT entries => count_ratio ~0.1
    "count_ratio": (0.4, 3.0),
    "recall": (0.25, None),
    "repetition_ratio": (None, 0.5),
}


@dataclass
class Violation:
    clip_id: str
    metric: str
    kind: str        # "regression" | "hard_bound"
    baseline: float | None
    current: float
    limit: float

    def __str__(self) -> str:
        if self.kind == "regression":
            return (
                f"[{self.clip_id}] {self.metric}: {self.baseline} -> {self.current} "
                f"(allowed delta {self.limit})"
            )
        return (
            f"[{self.clip_id}] {self.metric}={self.current} outside hard bound "
            f"({self.limit})"
        )


def check_clip(
    clip_id: str,
    current: dict[str, float],
    baseline: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
    hard_bounds: dict[str, tuple[float | None, float | None]] | None = None,
) -> list[Violation]:
    """Compare one clip's metrics against hard bounds and (optionally) a baseline."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    hard_bounds = {**DEFAULT_HARD_BOUNDS, **(hard_bounds or {})}
    violations: list[Violation] = []

    for metric, (lo, hi) in hard_bounds.items():
        value = current.get(metric)
        if value is None:
            continue
        if lo is not None and value < lo:
            violations.append(Violation(clip_id, metric, "hard_bound", None, value, lo))
        if hi is not None and value > hi:
            violations.append(Violation(clip_id, metric, "hard_bound", None, value, hi))

    if baseline:
        for metric, allowed in thresholds.items():
            direction = _METRIC_DIRECTION.get(metric)
            base = baseline.get(metric)
            value = current.get(metric)
            if direction is None or base is None or value is None:
                continue
            regression = (base - value) * direction  # positive => got worse
            if regression > allowed + 1e-9:
                violations.append(
                    Violation(clip_id, metric, "regression", base, value, allowed)
                )

    return violations


def load_baseline(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "clips" not in data:
        raise ValueError(f"Baseline {path} missing 'clips' key")
    return data


def build_baseline(
    clip_metrics: dict[str, ClipMetrics],
    label: str = "",
    whisperjav_version: str = "",
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "whisperjav_version": whisperjav_version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clips": {cid: m.as_dict() for cid, m in clip_metrics.items()},
    }


def gate(
    clip_metrics: dict[str, ClipMetrics],
    baseline: dict | None = None,
    thresholds: dict[str, float] | None = None,
    hard_bounds: dict[str, tuple[float | None, float | None]] | None = None,
) -> list[Violation]:
    """Gate a full corpus run. Returns all violations (empty = pass)."""
    baseline_clips = (baseline or {}).get("clips", {})
    violations: list[Violation] = []
    for clip_id, metrics in clip_metrics.items():
        violations.extend(
            check_clip(
                clip_id,
                metrics.as_dict(),
                baseline_clips.get(clip_id),
                thresholds,
                hard_bounds,
            )
        )
    return violations
