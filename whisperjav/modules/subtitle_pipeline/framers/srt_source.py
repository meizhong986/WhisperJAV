"""
SRT-source TemporalFramer — parses an existing SRT file as temporal frames.

Used for re-timing workflows (e.g., Issue #49: SRT-to-source) where an
existing subtitle file defines the dialogue boundaries.  Optionally
carries the SRT text in each frame for context or re-use.
"""

from pathlib import Path
from typing import Any, Union

import numpy as np

from whisperjav.modules.subtitle_pipeline.types import (
    FramingResult,
    TemporalFrame,
)
from whisperjav.utils.logger import logger


class SrtSourceFramer:
    """
    Parses an SRT file and uses its timestamps as temporal frames.

    Each SRT entry becomes a TemporalFrame.  When ``keep_text=True``,
    the frame's ``text`` field contains the SRT entry's text, which
    the orchestrator can use as-is (skipping generation) or as context.
    """

    def __init__(
        self,
        srt_path: Union[str, Path],
        keep_text: bool = False,
        min_frame_duration_s: float = 0.1,
    ):
        """
        Initialize with path to an SRT file.

        Args:
            srt_path: Path to the SRT file to parse.
            keep_text: Whether to carry SRT text in frames.
            min_frame_duration_s: Minimum frame duration; shorter entries
                are filtered out.
        """
        self._srt_path = Path(srt_path)
        self._keep_text = keep_text
        self._min_duration = min_frame_duration_s

        if not self._srt_path.exists():
            raise FileNotFoundError(f"SRT file not found: {self._srt_path}")

    def frame(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs: Any,
    ) -> FramingResult:
        """
        Parse the SRT file and return temporal frames.

        Frames are clamped to [0, audio_duration] and filtered by
        min_frame_duration_s.

        Args:
            audio: Scene audio array (mono).
            sample_rate: Sample rate of the audio.

        Returns:
            FramingResult with one frame per SRT entry.
        """
        try:
            import pysrt
        except ImportError as e:
            raise ImportError("pysrt is required for SrtSourceFramer. Install with: pip install pysrt") from e

        duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

        subs = pysrt.open(str(self._srt_path), encoding="utf-8")

        frames = []
        skipped = 0

        for entry in subs:
            # Convert SubRipTime to seconds
            start = (
                entry.start.hours * 3600
                + entry.start.minutes * 60
                + entry.start.seconds
                + entry.start.milliseconds / 1000.0
            )
            end = entry.end.hours * 3600 + entry.end.minutes * 60 + entry.end.seconds + entry.end.milliseconds / 1000.0

            # Clamp to audio boundary
            start = max(0.0, min(start, duration))
            end = max(start, min(end, duration))

            # Filter short entries
            if (end - start) < self._min_duration:
                skipped += 1
                continue

            text = entry.text.strip() if self._keep_text else None

            frames.append(
                TemporalFrame(
                    start=start,
                    end=end,
                    text=text,
                    source="srt-source",
                )
            )

        if skipped > 0:
            logger.debug(
                "[SrtSourceFramer] Filtered %d entries below %.2fs minimum",
                skipped,
                self._min_duration,
            )

        return FramingResult(
            frames=frames,
            metadata={
                "strategy": "srt-source",
                "srt_path": str(self._srt_path),
                "keep_text": self._keep_text,
                "frame_count": len(frames),
                "entries_skipped": skipped,
                "audio_duration_sec": duration,
            },
        )

    def cleanup(self) -> None:
        """No resources to release."""
