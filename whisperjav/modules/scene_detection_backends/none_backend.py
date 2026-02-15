"""
Null/Passthrough scene detector.

Returns the entire audio as a single scene, effectively bypassing
scene detection. Used by Qwen pipeline when scene detection is disabled
(method="none"), eliminating ad-hoc if/else guards in pipeline code.

Implements the SceneDetector Protocol.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .base import SceneDetectionError, SceneDetectionResult, SceneInfo
from .utils import save_scene_wav

logger = logging.getLogger("whisperjav")


class NullSceneDetector:
    """
    Passthrough detector that returns entire audio as a single scene.

    Used when scene detection should be bypassed:
    - SceneDetectorFactory.create("none")
    - Qwen pipeline with method="none"

    The full audio is saved as a single WAV file and returned as one SceneInfo.
    """

    def __init__(self, **kwargs):
        """
        Initialize null detector.

        Args:
            **kwargs: Ignored (accepted for interface compatibility with
                      create_from_legacy_kwargs)
        """
        pass

    @property
    def name(self) -> str:
        return "none"

    @property
    def display_name(self) -> str:
        return "None (Skip Detection)"

    def detect_scenes(
        self,
        audio_path: Path,
        output_dir: Path,
        media_basename: str,
        **kwargs,
    ) -> SceneDetectionResult:
        """
        Return entire audio as a single scene.

        Args:
            audio_path: Path to the input audio file
            output_dir: Directory to save the scene WAV file
            media_basename: Base name for the output file

        Returns:
            SceneDetectionResult with a single scene spanning the full audio

        Raises:
            SceneDetectionError: If audio file cannot be loaded
        """
        start_time = time.time()

        # Load audio
        try:
            from whisperjav.modules.scene_detection import load_audio_unified
            audio_data, sample_rate = load_audio_unified(
                audio_path, target_sr=None, force_mono=True
            )
        except Exception as e:
            raise SceneDetectionError(
                f"Failed to load audio file {audio_path}: {e}"
            ) from e

        duration = len(audio_data) / sample_rate

        # Save as single scene
        output_dir.mkdir(parents=True, exist_ok=True)
        scene_path = save_scene_wav(
            audio_data, sample_rate, 0, output_dir, media_basename
        )

        scene = SceneInfo(
            start_sec=0.0,
            end_sec=duration,
            scene_path=scene_path,
            detection_pass=0,  # Single-pass (no detection)
            metadata={"bypass": True},
        )

        logger.info(
            f"NullSceneDetector: full audio as single scene "
            f"({duration:.1f}s, {sample_rate}Hz)"
        )

        return SceneDetectionResult(
            scenes=[scene],
            method=self.name,
            audio_duration_sec=duration,
            parameters={"mode": "passthrough"},
            processing_time_sec=time.time() - start_time,
        )

    def cleanup(self) -> None:
        """No resources to clean up."""
        pass

    def __repr__(self) -> str:
        return "NullSceneDetector()"
