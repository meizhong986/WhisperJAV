"""
Shared data models for the pipeline analysis test suite.

All dataclasses used across runner, analyzer, presenter, and visualizer.
No WhisperJAV imports — pure Python data structures.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AudioInfo:
    """Information about the audio being analyzed."""

    path: Path
    sample_rate: int
    duration_sec: float
    num_samples: int
    source_media: Path  # Original media file (may be same as path)
    is_extracted: bool  # True if audio was extracted from video


@dataclass
class SegmentInfo:
    """Unified segment representation for both scene and speech results.

    Scene detection produces SceneInfo, speech segmentation produces SpeechSegment.
    Both are converted to SegmentInfo for uniform downstream processing.
    """

    start_sec: float
    end_sec: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class BackendRunResult:
    """Result from running a single backend (scene detection or speech segmentation)."""

    backend_name: str
    backend_type: str  # "scene_detection" or "speech_segmentation"
    display_name: str
    available: bool
    success: bool
    error: Optional[str]
    processing_time_sec: float
    segments: List[SegmentInfo]
    parameters: Dict[str, Any] = field(default_factory=dict)
    method: Optional[str] = None  # e.g. "auditok", "nemo-lite:frame_vad"


@dataclass
class AnalysisResult:
    """Metrics for a single backend run."""

    backend_name: str
    backend_type: str
    num_segments: int
    total_coverage_sec: float
    coverage_ratio: float
    mean_segment_duration: float
    median_segment_duration: float
    min_segment_duration: float
    max_segment_duration: float
    std_segment_duration: float
    processing_time_sec: float
    # Ground truth comparison (None if no ground truth):
    gt_recall: Optional[float] = None
    gt_precision: Optional[float] = None
    gt_f1: Optional[float] = None
    gt_matched_count: Optional[int] = None
    gt_total_count: Optional[int] = None


@dataclass
class ComparisonResult:
    """Pairwise comparison between two backend runs."""

    backend_a: str
    backend_b: str
    iou_score: float  # Intersection over Union of covered time
    coverage_diff: float  # Absolute difference in coverage ratio
    segment_count_diff: int  # Absolute difference in segment count
