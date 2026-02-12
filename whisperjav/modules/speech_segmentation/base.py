"""
Base classes and protocols for speech segmentation.

This module defines the core data structures and interface that all
speech segmentation backends must implement.
"""

from typing import Protocol, List, Dict, Any, Optional, Union, runtime_checkable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class SpeechSegment:
    """
    Unified speech segment representation.

    All backends MUST return segments using this structure for interoperability.

    Attributes:
        start_sec: Start time in seconds
        end_sec: End time in seconds
        start_sample: Start sample at source sample rate (for legacy compatibility)
        end_sample: End sample at source sample rate (for legacy compatibility)
        confidence: Detection confidence [0.0, 1.0], 1.0 if not available
        metadata: Backend-specific metadata (e.g., speaker_id for diarization)
    """
    start_sec: float
    end_sec: float
    start_sample: int = 0
    end_sample: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_sec - self.start_sec

    def to_dict(self) -> Dict[str, Any]:
        """Export segment for JSON serialization."""
        return {
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "confidence": round(self.confidence, 3),
        }

    def __repr__(self) -> str:
        return f"SpeechSegment({self.start_sec:.3f}s - {self.end_sec:.3f}s, conf={self.confidence:.2f})"


@dataclass
class SegmentationResult:
    """
    Complete result from speech segmentation.

    Contains segments, groups for ASR processing, and metadata for visualization.

    Attributes:
        segments: List of all detected speech segments
        groups: Grouped segments for chunk-based ASR processing
        method: Name of the backend used (e.g., "silero-v4.0")
        audio_duration_sec: Total audio duration in seconds
        parameters: Parameters used for segmentation
        processing_time_sec: Time taken to process (seconds)
    """
    segments: List[SpeechSegment]
    groups: List[List[SpeechSegment]]
    method: str
    audio_duration_sec: float
    parameters: Dict[str, Any]
    processing_time_sec: float = 0.0

    @property
    def speech_coverage_sec(self) -> float:
        """Total seconds of detected speech."""
        return sum(seg.duration_sec for seg in self.segments)

    @property
    def speech_coverage_ratio(self) -> float:
        """Ratio of speech to total audio duration."""
        if self.audio_duration_sec <= 0:
            return 0.0
        return self.speech_coverage_sec / self.audio_duration_sec

    @property
    def num_segments(self) -> int:
        """Total number of segments."""
        return len(self.segments)

    @property
    def num_groups(self) -> int:
        """Number of segment groups."""
        return len(self.groups)

    def to_legacy_format(self) -> List[List[Dict]]:
        """
        Convert to legacy grouped format for backward compatibility.

        Returns:
            List[List[Dict]] where each Dict has:
                - start: int (samples at 16kHz)
                - end: int (samples at 16kHz)
                - start_sec: float
                - end_sec: float
                - metadata: Dict (backend-specific, e.g. raw_start/raw_end for TEN)
        """
        legacy_groups = []
        for group in self.groups:
            legacy_group = []
            for seg in group:
                legacy_group.append({
                    "start": seg.start_sample,
                    "end": seg.end_sample,
                    "start_sec": seg.start_sec,
                    "end_sec": seg.end_sec,
                    "metadata": seg.metadata,
                })
            legacy_groups.append(legacy_group)
        return legacy_groups

    def to_flat_legacy_format(self) -> List[Dict]:
        """
        Convert to flat list of segments for metadata storage.

        Returns:
            List[Dict] with start_sec/end_sec for each segment
        """
        return [
            {"start_sec": round(seg.start_sec, 3), "end_sec": round(seg.end_sec, 3)}
            for seg in self.segments
        ]

    def __repr__(self) -> str:
        return (
            f"SegmentationResult(method={self.method}, "
            f"segments={self.num_segments}, groups={self.num_groups}, "
            f"coverage={self.speech_coverage_ratio:.1%})"
        )


@runtime_checkable
class SpeechSegmenter(Protocol):
    """
    Protocol defining the speech segmentation interface.

    All speech segmentation backends MUST implement this protocol.
    """

    @property
    def name(self) -> str:
        """
        Unique backend identifier.

        Examples: "silero-v4.0", "nemo", "ten", "none"
        """
        ...

    @property
    def display_name(self) -> str:
        """
        Human-readable name for GUI display.

        Examples: "Silero VAD v4.0", "NVIDIA NeMo VAD"
        """
        ...

    def segment(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
        **kwargs
    ) -> SegmentationResult:
        """
        Detect speech segments in audio.

        Args:
            audio: Audio data as numpy array, or path to audio file
            sample_rate: Sample rate of input audio (default 16kHz)
            **kwargs: Backend-specific parameters

        Returns:
            SegmentationResult with detected speech segments
        """
        ...

    def cleanup(self) -> None:
        """
        Release resources (GPU memory, model handles).

        Should be called when the segmenter is no longer needed.
        """
        ...

    def get_supported_sample_rates(self) -> List[int]:
        """
        Return list of supported sample rates.

        Most VAD models work best at 16kHz.
        """
        ...
