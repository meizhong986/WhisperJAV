"""
Manual TemporalFramer — accepts user-provided timestamps directly.

Useful for testing, debugging, and re-timing workflows where the user
already knows the dialogue boundaries they want to target.
"""

from typing import Any

import numpy as np

from whisperjav.modules.subtitle_pipeline.types import (
    FramingResult,
    TemporalFrame,
)


class ManualFramer:
    """
    Returns frames from user-provided (start, end) timestamp pairs.

    Timestamps are validated against the audio duration — frames beyond
    the audio boundary are clamped.
    """

    def __init__(
        self,
        timestamps: list[tuple[float, float]],
        texts: list[str] | None = None,
    ):
        """
        Initialize with explicit timestamp boundaries.

        Args:
            timestamps: List of (start_sec, end_sec) tuples.
            texts: Optional list of pre-existing texts for each frame.
                   Must match length of timestamps if provided.
        """
        self._timestamps = timestamps
        self._texts = texts

        if texts is not None and len(texts) != len(timestamps):
            raise ValueError(f"texts length ({len(texts)}) must match timestamps length ({len(timestamps)})")

    def frame(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs: Any,
    ) -> FramingResult:
        """
        Return the pre-defined temporal frames.

        Frames are clamped to [0, audio_duration].

        Args:
            audio: Scene audio array (mono).
            sample_rate: Sample rate of the audio.

        Returns:
            FramingResult with one frame per timestamp pair.
        """
        duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

        frames = []
        for i, (start, end) in enumerate(self._timestamps):
            # Clamp to audio boundary
            start = max(0.0, min(start, duration))
            end = max(start, min(end, duration))

            text = self._texts[i] if self._texts else None

            frames.append(
                TemporalFrame(
                    start=start,
                    end=end,
                    text=text,
                    source="manual",
                )
            )

        return FramingResult(
            frames=frames,
            metadata={
                "strategy": "manual",
                "frame_count": len(frames),
                "audio_duration_sec": duration,
            },
        )

    def cleanup(self) -> None:
        """No resources to release."""
