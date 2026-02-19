"""
VAD-grouped TemporalFramer — uses speech segmentation to produce frames.

Wraps the existing SpeechSegmenterFactory to detect speech regions, then
converts the segmenter's pre-grouped results into TemporalFrames.  This is
what current VAD_SLICING mode does: run VAD, group speech segments into
chunks within a duration limit, and process each group.

The raw speech regions are stored in FramingResult.metadata["speech_regions"]
so the orchestrator can pass them to the alignment sentinel for Strategy C
(VAD-guided) recovery.
"""

from typing import Any, Optional

import numpy as np

from whisperjav.modules.subtitle_pipeline.types import (
    FramingResult,
    TemporalFrame,
)
from whisperjav.utils.logger import logger


class VadGroupedFramer:
    """
    Produces temporal frames by running VAD and grouping speech regions.

    Each group of speech segments becomes a TemporalFrame spanning from
    the first segment's start to the last segment's end.

    Groups are produced by the SpeechSegmenter backend, which respects
    max_group_duration_s and chunk_threshold_s.
    """

    def __init__(
        self,
        segmenter_backend: str = "ten",
        max_group_duration_s: float = 29.0,
        chunk_threshold_s: float = 1.0,
        min_frame_duration_s: float = 0.1,
        segmenter_config: Optional[dict[str, Any]] = None,
    ):
        """
        Configure the VAD-grouped framer.

        Args:
            segmenter_backend: SpeechSegmenter backend name (e.g., "ten", "silero").
            max_group_duration_s: Maximum duration per group (Whisper 30s limit).
            chunk_threshold_s: Silence gap threshold for splitting groups.
            min_frame_duration_s: Minimum frame duration; shorter groups are filtered.
            segmenter_config: Additional kwargs passed to the segmenter factory.
        """
        self._backend = segmenter_backend
        self._max_group = max_group_duration_s
        self._chunk_threshold = chunk_threshold_s
        self._min_duration = min_frame_duration_s
        self._segmenter_config = segmenter_config or {}
        self._segmenter = None

    def _ensure_segmenter(self):
        """Lazily create the speech segmenter."""
        if self._segmenter is not None:
            return

        from whisperjav.modules.speech_segmentation.factory import (
            SpeechSegmenterFactory,
        )

        kwargs = {
            "max_group_duration_s": self._max_group,
            "chunk_threshold_s": self._chunk_threshold,
            **self._segmenter_config,
        }
        self._segmenter = SpeechSegmenterFactory.create(self._backend, **kwargs)

    def frame(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs: Any,
    ) -> FramingResult:
        """
        Run VAD on scene audio and convert groups to temporal frames.

        Each group from the segmenter becomes one TemporalFrame.
        Raw speech regions (per frame) are stored in metadata for
        the alignment sentinel's Strategy C (VAD-guided) recovery.

        Args:
            audio: Scene audio array (mono, typically 16kHz).
            sample_rate: Sample rate of the audio.

        Returns:
            FramingResult with one frame per VAD group.
        """
        self._ensure_segmenter()
        duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

        # Run speech segmentation
        seg_result = self._segmenter.segment(audio, sample_rate=sample_rate)

        frames = []
        speech_regions_per_frame: list[list[tuple[float, float]]] = []
        skipped = 0

        for group in seg_result.groups:
            if not group:
                continue

            # Group boundaries: first segment start → last segment end
            group_start = group[0].start_sec
            group_end = group[-1].end_sec

            # Filter short groups
            if (group_end - group_start) < self._min_duration:
                skipped += 1
                continue

            frames.append(
                TemporalFrame(
                    start=group_start,
                    end=group_end,
                    source="vad-grouped",
                )
            )

            # Store raw speech regions within this frame (group-relative)
            # These are used by the sentinel for VAD-guided redistribution
            regions = [(seg.start_sec, seg.end_sec) for seg in group]
            speech_regions_per_frame.append(regions)

        if skipped > 0:
            logger.debug(
                "[VadGroupedFramer] Filtered %d groups below %.2fs minimum",
                skipped,
                self._min_duration,
            )

        logger.info(
            "[VadGroupedFramer] %s: %d segments → %d groups → %d frames (%.1fs audio, %.1f%% speech)",
            seg_result.method,
            seg_result.num_segments,
            seg_result.num_groups,
            len(frames),
            duration,
            seg_result.speech_coverage_ratio * 100,
        )

        return FramingResult(
            frames=frames,
            metadata={
                "strategy": "vad-grouped",
                "segmenter_backend": seg_result.method,
                "frame_count": len(frames),
                "total_segments": seg_result.num_segments,
                "total_groups": seg_result.num_groups,
                "groups_skipped": skipped,
                "audio_duration_sec": duration,
                "speech_coverage_ratio": seg_result.speech_coverage_ratio,
                "speech_regions": speech_regions_per_frame,
                "segmenter_params": seg_result.parameters,
            },
        )

    def reframe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        max_group_duration_s: float,
        **kwargs: Any,
    ) -> FramingResult:
        """Re-frame with tighter grouping parameters (for step-down retry).

        Temporarily overrides ``max_group_duration_s`` and runs the same
        VAD + grouping pipeline as :meth:`frame`.  The original setting is
        restored on return.

        Args:
            audio: Scene audio array (mono, typically 16kHz).
            sample_rate: Sample rate of the audio.
            max_group_duration_s: Override for max group duration.
        """
        original = self._max_group
        self._max_group = max_group_duration_s

        # Force re-creation of the segmenter with the new max_group
        old_segmenter = self._segmenter
        self._segmenter = None

        try:
            result = self.frame(audio, sample_rate, **kwargs)
            return result
        finally:
            # Restore original settings and segmenter
            self._max_group = original
            if self._segmenter is not None:
                self._segmenter.cleanup()
            self._segmenter = old_segmenter

    def cleanup(self) -> None:
        """Release the speech segmenter."""
        if self._segmenter is not None:
            self._segmenter.cleanup()
            self._segmenter = None
