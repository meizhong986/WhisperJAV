"""
Semantic Audio Clustering scene detector.

Wraps the existing SemanticClusteringAdapter to conform to the
SceneDetector Protocol. Single-pass content-aware boundary detection
via MFCC features and agglomerative clustering.

Natural operating range: 20s-420s scenes.
NOT a two-pass approach — finds content-aware boundaries directly.

Implements the SceneDetector Protocol.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import SceneDetectionError, SceneDetectionResult, SceneInfo

logger = logging.getLogger("whisperjav")


class SemanticSceneDetector:
    """
    Scene detector using Semantic Audio Clustering.

    Wraps SemanticClusteringAdapter and maps its output to
    SceneDetectionResult for Protocol conformance.

    The adapter is lazy-loaded to avoid importing heavy dependencies
    (scikit-learn, etc.) unless this backend is actually selected.
    """

    def __init__(self, **kwargs):
        """
        Initialize the semantic scene detector.

        Args:
            **kwargs: Passed to SemanticClusteringAdapter.
                      Known keys: min_duration, max_duration, snap_window,
                      clustering_threshold, sample_rate, preset, visualize,
                      preserve_original_sr.
                      Uses SemanticClusteringConfig defaults (20.0/420.0),
                      NOT auditok defaults (0.3/29.0).
        """
        self._adapter = None
        self._kwargs = kwargs
        self._init_adapter()

    def _init_adapter(self) -> None:
        """
        Initialize SemanticClusteringAdapter with proper defaults.

        Uses SemanticClusteringConfig defaults as fallbacks, NOT
        any inherited auditok values. This is the Phase 0.1 fix
        now baked directly into the backend (no namespace collision risk).
        """
        try:
            from .semantic_adapter import (
                SemanticClusteringAdapter,
                SemanticClusteringConfig,
            )

            # Build config from kwargs using semantic defaults
            _defaults = SemanticClusteringConfig()
            config = SemanticClusteringConfig(
                min_duration=float(self._kwargs.get(
                    "min_duration", _defaults.min_duration
                )),
                max_duration=float(self._kwargs.get(
                    "max_duration", _defaults.max_duration
                )),
                snap_window=float(self._kwargs.get(
                    "snap_window", _defaults.snap_window
                )),
                clustering_threshold=float(self._kwargs.get(
                    "clustering_threshold", _defaults.clustering_threshold
                )),
                sample_rate=int(self._kwargs.get(
                    "sample_rate", _defaults.sample_rate
                )),
                preserve_original_sr=bool(self._kwargs.get(
                    "preserve_original_sr", _defaults.preserve_original_sr
                )),
                visualize=bool(self._kwargs.get(
                    "visualize", _defaults.visualize
                )),
            )

            self._adapter = SemanticClusteringAdapter(
                config=config,
                preset=str(self._kwargs.get("preset", "default")),
            )

            logger.info("SemanticSceneDetector initialized")

        except ImportError as e:
            raise SceneDetectionError(
                "Semantic method selected but dependencies not available. "
                "Ensure 'semantic_audio_clustering' module is installed. "
                f"Fall back to --scene-detection-method auditok. Error: {e}"
            ) from e

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def display_name(self) -> str:
        return "Semantic Audio Clustering"

    def detect_scenes(
        self,
        audio_path: Path,
        output_dir: Path,
        media_basename: str,
        **kwargs,
    ) -> SceneDetectionResult:
        """
        Detect scenes using Semantic Audio Clustering.

        Delegates to SemanticClusteringAdapter.detect_scenes() and
        wraps the output in SceneDetectionResult.

        Args:
            audio_path: Path to the input audio file
            output_dir: Directory to save scene WAV files
            media_basename: Base name for output files

        Returns:
            SceneDetectionResult with detected scenes

        Raises:
            SceneDetectionError: If the adapter fails
        """
        import soundfile as sf

        start_time = time.time()

        if self._adapter is None:
            raise SceneDetectionError(
                "Semantic adapter not initialized."
            )

        # Get actual audio duration from file header (lightweight, no full load)
        try:
            audio_info = sf.info(str(audio_path))
            audio_duration = audio_info.duration
        except Exception:
            audio_duration = 0.0

        try:
            # Delegate to existing adapter
            scene_tuples = self._adapter.detect_scenes(
                audio_path=audio_path,
                output_dir=output_dir,
                media_basename=media_basename,
            )
        except Exception as e:
            raise SceneDetectionError(
                f"Semantic scene detection failed: {e}"
            ) from e

        # Map adapter output to SceneInfo objects
        scenes = []
        for idx, (scene_path, start_sec, end_sec, duration) in enumerate(scene_tuples):
            # Get context metadata from adapter if available
            context = self._adapter.get_scene_context(idx) or {}

            scenes.append(SceneInfo(
                start_sec=start_sec,
                end_sec=end_sec,
                scene_path=scene_path,
                detection_pass=0,  # Single-pass (semantic paradigm)
                metadata={
                    "context": context.get("context", {}),
                    "asr_prompt": context.get("asr_prompt", ""),
                },
            ))

        # Build parameters dict from adapter config
        parameters = {}
        if self._adapter.vad_params:
            parameters = dict(self._adapter.vad_params)

        processing_time = time.time() - start_time
        logger.info(
            f"Semantic scene detection complete: {len(scenes)} scenes "
            f"in {processing_time:.1f}s"
        )

        return SceneDetectionResult(
            scenes=scenes,
            method=self.name,
            audio_duration_sec=audio_duration,
            parameters=parameters,
            processing_time_sec=processing_time,
        )

    def cleanup(self) -> None:
        """Release adapter resources."""
        self._adapter = None

    def __repr__(self) -> str:
        return "SemanticSceneDetector()"
