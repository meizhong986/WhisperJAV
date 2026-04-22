"""Data models for VAD ground-truth analyser.

All reports/metrics are plain dataclasses for JSON-roundtrip simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Input segments
# ---------------------------------------------------------------------------

@dataclass
class GtSegment:
    """A single ground-truth speech segment, parsed from an SRT entry."""

    index: int              # 1-based SRT index
    start_sec: float
    end_sec: float
    text: str               # kept for hover tooltips; NOT used in metrics

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class VadSegment:
    """A single speech segment emitted by a backend."""

    start_sec: float
    end_sec: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class BackendMetrics:
    """Timing-accuracy metrics for one backend vs ground truth.

    Present only when ground truth is provided AND the backend ran successfully.
    All values are deterministic; no NaN — empty inputs map to 0 by convention.
    """

    # Frame-level (headline)
    frame_precision: float
    frame_recall: float
    frame_f1: float

    # Segment IoU matching
    iou_mean: float
    iou_median: float
    iou_per_gt: List[float]            # one IoU per GT segment (0.0 if unmatched)

    # Boundary drift (absolute errors, ms)
    onset_drift_mean_ms: float
    offset_drift_mean_ms: float
    onset_drift_median_ms: float
    offset_drift_median_ms: float

    # Coverage errors (frame-level, percentage)
    missed_speech_pct: float           # GT speech not covered by backend
    false_alarm_pct: float             # backend speech outside any GT segment

    # Match counts
    num_matched_segments: int
    num_unmatched_gt: int


@dataclass
class BackendReport:
    """Complete output for one backend run."""

    name: str                          # e.g. "whisperseg"
    display_name: str                  # e.g. "WhisperSeg (JA-ASMR)"
    available: bool
    success: bool
    error: Optional[str]

    # Run info
    processing_time_sec: float
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Results
    segments: List[VadSegment] = field(default_factory=list)
    num_segments: int = 0
    coverage_ratio: float = 0.0        # sum of segment durations / audio duration

    # GT-mode metrics; None in GT-less mode or when backend failed
    metrics: Optional[BackendMetrics] = None


# ---------------------------------------------------------------------------
# Agreement matrix (GT-less mode; also produced as cross-check in GT mode)
# ---------------------------------------------------------------------------

@dataclass
class AgreementMatrix:
    """Pairwise F1 between all successfully-run backends.

    - Symmetric: pair_f1[i][j] == pair_f1[j][i]
    - Diagonal = 1.0
    - pair_f1 values in [0, 1]
    """

    backend_order: List[str]           # row/col labels (only successful backends)
    pair_f1: List[List[float]]         # N x N matrix
    consensus_coverage_pct: float      # % frames where >= 2 backends agree on speech


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

@dataclass
class AnalysisReport:
    """Top-level serialisable report."""

    schema_version: str
    media_file: str
    audio_duration_sec: float
    sample_rate: int
    sensitivity: str
    frame_ms: int
    generated_at: str                  # ISO-8601 UTC

    # Ground-truth summary; None if not provided
    ground_truth: Optional[Dict[str, Any]] = None

    # Per-backend results (ordered map preserved via dict insertion order)
    backends: Dict[str, BackendReport] = field(default_factory=dict)

    # Inter-backend agreement (always produced when >= 2 backends succeed)
    agreement: Optional[AgreementMatrix] = None
