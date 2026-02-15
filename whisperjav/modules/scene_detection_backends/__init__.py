"""
Scene Detection Backends.

This package provides a modular, pluggable scene detection system with
multiple backend support. Each backend implements the SceneDetector protocol.

Architecture mirrors whisperjav/modules/speech_segmentation/ for uniformity.

Available backends:
- auditok: Silence-based two-pass detection (default)
- silero: VAD-based two-pass detection (auditok Pass 1 + Silero Pass 2)
- semantic: Texture-based segmentation using MFCC features and clustering
- none: Full audio as single scene (skip detection)

Core types:
- SceneInfo: Unified scene representation (dataclass)
- SceneDetectionResult: Complete detection result with legacy compatibility
- SceneDetector: Protocol all backends must implement
- SceneDetectionError: Exception for detection failures
- SceneDetectorFactory: Factory for creating detector instances

Example usage:
    from whisperjav.modules.scene_detection_backends import SceneDetectorFactory

    # Create a detector
    detector = SceneDetectorFactory.create("auditok", max_duration=29.0)

    # Detect scenes
    result = detector.detect_scenes(audio_path, output_dir, "my_video")

    # Access scenes
    for scene in result.scenes:
        print(f"{scene.start_sec:.2f}s - {scene.end_sec:.2f}s")

    # Legacy compatibility
    scene_tuples = result.to_legacy_tuples()
    metadata = result.to_metadata_dict()
"""

from typing import TYPE_CHECKING

from .base import (
    SceneDetectionError,
    SceneDetectionResult,
    SceneDetector,
    SceneInfo,
)
from .factory import SceneDetectorFactory

if TYPE_CHECKING:
    from .semantic_adapter import SemanticClusteringAdapter

__all__ = [
    "SceneInfo",
    "SceneDetectionResult",
    "SceneDetector",
    "SceneDetectionError",
    "SceneDetectorFactory",
    "SemanticClusteringAdapter",
]
