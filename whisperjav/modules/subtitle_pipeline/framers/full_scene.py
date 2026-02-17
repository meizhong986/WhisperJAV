"""
Full-scene TemporalFramer — returns one frame spanning the entire scene.

This is what current ASSEMBLY and CONTEXT_AWARE modes do implicitly: they
process the full scene audio as a single unit.  The FullSceneFramer makes
this strategy explicit and composable.
"""

from typing import Any

import numpy as np

from whisperjav.modules.subtitle_pipeline.types import (
    FramingResult,
    TemporalFrame,
)


class FullSceneFramer:
    """
    Returns a single frame covering the entire scene duration.

    Simplest possible framer — no segmentation, no grouping.
    Suitable for scenes within the ASR model's context window
    (e.g., Qwen3-ASR: up to ~120s scenes, Whisper: up to ~30s).
    """

    def frame(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs: Any,
    ) -> FramingResult:
        """
        Frame the entire audio as one temporal window.

        Args:
            audio: Scene audio array (mono).
            sample_rate: Sample rate of the audio.

        Returns:
            FramingResult with a single frame [0, duration].
        """
        duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

        return FramingResult(
            frames=[
                TemporalFrame(
                    start=0.0,
                    end=duration,
                    source="full-scene",
                )
            ],
            metadata={
                "strategy": "full-scene",
                "frame_count": 1,
                "audio_duration_sec": duration,
            },
        )

    def cleanup(self) -> None:
        """No resources to release."""
