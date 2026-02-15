"""
Base classes and protocols for scene detection.

This module defines the core data structures and interface that all
scene detection backends must implement.

Architecture mirrors whisperjav/modules/speech_segmentation/base.py
for cognitive uniformity across the codebase.

Two-pass scene detection architecture:
  Pass 1 (Coarse): Find natural chapter boundaries via long silences
  Pass 2 (Fine):   Chunk each chapter to consumer's max_duration needs

Some backends (semantic) use a single-pass content-aware approach instead.

See docs/sprint3_scene_detection_refactor.md for the full design.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


class SceneDetectionError(Exception):
    """
    Raised when scene detection fails.

    Distinct from returning an empty SceneDetectionResult (valid: no speech found).
    This exception means detection itself failed (audio load error, backend crash, etc.).

    Callers MUST distinguish these:
      - SceneDetectionError  -> detection failed, cannot continue
      - Empty result.scenes  -> detection succeeded, no scenes found
    """


@dataclass
class SceneInfo:
    """
    Unified scene representation.

    All backends MUST return scenes using this structure for interoperability.

    Attributes:
        start_sec: Start time in seconds (relative to original audio)
        end_sec: End time in seconds (relative to original audio)
        scene_path: Path to extracted WAV file (set after extraction)
        detection_pass: Which pass detected this scene
                        (1=coarse/direct, 2=fine/split, 0=single-pass)
        metadata: Backend-specific metadata (e.g., context for semantic,
                  cluster_id, asr_prompt, split_method)
    """
    start_sec: float
    end_sec: float
    scene_path: Optional[Path] = None
    detection_pass: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        """Duration of the scene in seconds."""
        return self.end_sec - self.start_sec

    def to_dict(self) -> Dict[str, Any]:
        """Export scene for JSON serialization."""
        result = {
            "start_time_seconds": round(self.start_sec, 3),
            "end_time_seconds": round(self.end_sec, 3),
            "duration_seconds": round(self.duration_sec, 3),
            "detection_pass": self.detection_pass,
        }
        if self.scene_path is not None:
            result["filename"] = self.scene_path.name
            result["path"] = str(self.scene_path)
        return result

    def to_legacy_tuple(self) -> Tuple[Path, float, float, float]:
        """
        Convert to legacy 4-tuple format for backward compatibility.

        Returns:
            (scene_path, start_sec, end_sec, duration_sec)

        Raises:
            ValueError: If scene_path is not set (WAV not yet extracted)
        """
        if self.scene_path is None:
            raise ValueError("scene_path not set — WAV file not yet extracted")
        return (self.scene_path, self.start_sec, self.end_sec, self.duration_sec)

    def __repr__(self) -> str:
        name = self.scene_path.name if self.scene_path else "no-path"
        return (
            f"SceneInfo({self.start_sec:.3f}s - {self.end_sec:.3f}s, "
            f"pass={self.detection_pass}, {name})"
        )


@dataclass
class SceneDetectionResult:
    """
    Complete result from scene detection.

    Contains scenes, detection metadata, and methods for backward compatibility.

    Attributes:
        scenes: List of all detected scenes
        method: Name of the backend used (e.g., "auditok", "silero", "semantic")
        audio_duration_sec: Total audio duration in seconds
        parameters: Parameters used for detection
        processing_time_sec: Time taken to process (seconds)
        coarse_boundaries: Pass 1 chapter boundaries (if two-pass method)
        vad_segments: VAD segment data (if Silero method used for Pass 2)
    """
    scenes: List[SceneInfo]
    method: str
    audio_duration_sec: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    processing_time_sec: float = 0.0
    coarse_boundaries: Optional[List[Dict[str, Any]]] = None
    vad_segments: Optional[List[Dict[str, Any]]] = None

    @property
    def num_scenes(self) -> int:
        """Total number of detected scenes."""
        return len(self.scenes)

    @property
    def total_scene_duration_sec(self) -> float:
        """Sum of all scene durations."""
        return sum(s.duration_sec for s in self.scenes)

    @property
    def coverage_ratio(self) -> float:
        """Ratio of scene coverage to total audio duration."""
        if self.audio_duration_sec <= 0:
            return 0.0
        return self.total_scene_duration_sec / self.audio_duration_sec

    def to_legacy_tuples(self) -> List[Tuple[Path, float, float, float]]:
        """
        Convert to legacy list-of-tuples format for backward compatibility.

        Used during Phase 3 migration. Pipelines call this to get the same
        return type that DynamicSceneDetector.detect_scenes() produces.

        Returns:
            List of (scene_path, start_sec, end_sec, duration_sec) tuples

        Raises:
            ValueError: If any scene's scene_path is not set
        """
        return [scene.to_legacy_tuple() for scene in self.scenes]

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Export detection metadata matching DynamicSceneDetector.get_detection_metadata().

        Produces the same dict structure that existing pipeline metadata consumers
        expect. Compatible with balanced_pipeline, fast_pipeline, fidelity_pipeline.

        Returns:
            Dict with keys: 'scenes_detected', 'coarse_boundaries', 'vad_segments',
            'vad_method', 'vad_params'
        """
        return {
            "scenes_detected": [
                {"scene_index": idx, **scene.to_dict()}
                for idx, scene in enumerate(self.scenes)
            ],
            "coarse_boundaries": self.coarse_boundaries,
            "vad_segments": self.vad_segments,
            "vad_method": self.method,
            "vad_params": self.parameters if self.parameters else None,
        }

    def __repr__(self) -> str:
        return (
            f"SceneDetectionResult(method={self.method}, "
            f"scenes={self.num_scenes}, "
            f"coverage={self.coverage_ratio:.1%})"
        )


@runtime_checkable
class SceneDetector(Protocol):
    """
    Protocol defining the scene detection interface.

    All scene detection backends MUST implement this protocol.

    Scene detection identifies temporal boundaries in audio and extracts
    WAV files for each scene. The architecture supports two paradigms:

    1. Two-Pass (auditok, silero):
       Pass 1 (Coarse): Find natural chapter boundaries via long silences
       Pass 2 (Fine): Chunk each chapter to consumer's max_duration

    2. Single-Pass (semantic):
       Content-aware boundary detection via audio feature clustering
    """

    @property
    def name(self) -> str:
        """
        Unique backend identifier.

        Examples: "auditok", "silero", "semantic", "none"
        """
        ...

    @property
    def display_name(self) -> str:
        """
        Human-readable name for GUI display.

        Examples: "Auditok (Silence-Based)", "Silero VAD", "Semantic Clustering"
        """
        ...

    def detect_scenes(
        self,
        audio_path: Path,
        output_dir: Path,
        media_basename: str,
        **kwargs,
    ) -> SceneDetectionResult:
        """
        Detect scenes in audio and extract WAV files.

        Args:
            audio_path: Path to the input audio file
            output_dir: Directory to save scene WAV files
            media_basename: Base name for output files (e.g., "my_video")
            **kwargs: Backend-specific parameters

        Returns:
            SceneDetectionResult with detected scenes and metadata

        Raises:
            SceneDetectionError: If detection fails (audio load error, etc.)
        """
        ...

    def cleanup(self) -> None:
        """
        Release resources (loaded models, cached audio).

        Should be called when the detector is no longer needed.
        """
        ...
