"""
Quality-based segment routing using brouhaha VAD/SNR/C50.

This module computes per-segment quality metrics from brouhaha
inference outputs and classifies segments as 'direct' (ASR as-is)
or 'enhance' (run through enhancement chain before ASR).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class SegmentMetrics:
    start: float
    end: float
    avg_snr: float
    avg_c50: float
    coverage: float  # ratio of frames with vad >= vad_min
    score: float


@dataclass
class SegmentDecision:
    route: str  # 'direct' | 'enhance'
    label: str  # 'clean' | 'noisy' | 'reverberant' | 'music' | 'low_energy' | 'unknown'
    reasons: List[str]
    params: Dict[str, Any]


def aggregate_metrics(
    frames: List[Tuple[float, Tuple[float, float, float]]],
    vad_min: float,
    start: float,
    end: float,
    score_weights: Tuple[float, float] = (0.7, 0.3),
) -> SegmentMetrics:
    """
    Aggregate per-frame (time, (vad, snr, c50)) into segment metrics.

    frames: list of (time, (vad, snr, c50)) from brouhaha Inference.
    vad_min: threshold to consider a frame as speech.
    start, end: segment boundaries.
    score_weights: (w_snr, w_c50) for composite score.
    """
    if not frames:
        return SegmentMetrics(start, end, avg_snr=0.0, avg_c50=0.0, coverage=0.0, score=0.0)

    # Import numpy lazily to avoid hard dependency at import time
    import numpy as np  # type: ignore

    vad = np.array([v for _, (v, _, _) in frames], dtype=np.float32)
    snr = np.array([s for _, (_, s, _) in frames], dtype=np.float32)
    c50 = np.array([c for _, (_, _, c) in frames], dtype=np.float32)

    speech_mask = vad >= vad_min
    coverage = float(np.mean(speech_mask)) if vad.size else 0.0

    if speech_mask.any():
        avg_snr = float(np.mean(snr[speech_mask]))
        avg_c50 = float(np.mean(c50[speech_mask]))
    else:
        avg_snr = 0.0
        avg_c50 = 0.0

    w_snr, w_c50 = score_weights
    score = w_snr * avg_snr + w_c50 * avg_c50

    return SegmentMetrics(start, end, avg_snr=avg_snr, avg_c50=avg_c50, coverage=coverage, score=score)


def classify_segment(
    m: SegmentMetrics,
    cfg: Dict[str, Any],
) -> SegmentDecision:
    """
    Classify a segment as 'direct' or 'enhance' with an issue label.

    cfg expects keys:
      - VAD_MIN, SNR_CLEAN, C50_CLEAN, MIN_COVERAGE
      - SCORE: { w_snr, w_c50, thresh }
      - CHAINS: mapping from label -> list of step names
    """
    reasons: List[str] = []
    label = "clean"

    snr_clean = float(cfg.get("SNR_CLEAN", 7.5))
    c50_clean = float(cfg.get("C50_CLEAN", 0.5))
    min_cov = float(cfg.get("MIN_COVERAGE", 0.4))
    score_cfg = cfg.get("SCORE", {"w_snr": 0.7, "w_c50": 0.3, "thresh": 6.0})
    score_thresh = float(score_cfg.get("thresh", 6.0))

    if m.coverage < min_cov:
        label = "low_energy"
        reasons.append(f"coverage {m.coverage:.2f} < {min_cov}")

    if m.avg_snr < snr_clean:
        label = "noisy"
        reasons.append(f"avg_snr {m.avg_snr:.2f} < {snr_clean}")

    if m.avg_c50 < c50_clean:
        # prefer 'reverberant' if not already more critical
        if label == "clean":
            label = "reverberant"
        reasons.append(f"avg_c50 {m.avg_c50:.2f} < {c50_clean}")

    # Composite score gate
    if m.score < score_thresh and label == "clean":
        label = "unknown"
        reasons.append(f"score {m.score:.2f} < {score_thresh}")

    if label == "clean":
        return SegmentDecision(route="direct", label=label, reasons=["meets clean criteria"], params={})

    chain = (cfg.get("CHAINS", {}) or {}).get(label) or []
    return SegmentDecision(route="enhance", label=label, reasons=reasons, params={"chain": chain})


def select_route_for_segment(
    frames: List[Tuple[float, Tuple[float, float, float]]],
    start: float,
    end: float,
    cfg: Dict[str, Any],
) -> Tuple[SegmentMetrics, SegmentDecision]:
    """
    Convenience wrapper to aggregate metrics then classify.
    """
    vad_min = float(cfg.get("VAD_MIN", 0.5))
    score_cfg = cfg.get("SCORE", {"w_snr": 0.7, "w_c50": 0.3})
    weights = (float(score_cfg.get("w_snr", 0.7)), float(score_cfg.get("w_c50", 0.3)))
    metrics = aggregate_metrics(frames, vad_min=vad_min, start=start, end=end, score_weights=weights)
    decision = classify_segment(metrics, cfg)
    return metrics, decision
