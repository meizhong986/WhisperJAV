"""
Speech Segmentation Module

Provides a modular, pluggable speech segmentation system with multiple backend support.
Users can select their preferred speech segmenter via CLI (--speech-segmenter) or GUI.

Available backends:
- silero (default): Silero VAD v4.0 or v3.1
- nemo: NVIDIA NeMo VAD (requires nemo_toolkit)
- ten: TEN Framework VAD (requires ten-vad)
- none: No segmentation (passthrough)

Example usage:
    from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory

    # Create a segmenter
    segmenter = SpeechSegmenterFactory.create("silero", threshold=0.4)

    # Segment audio
    result = segmenter.segment(audio_path, sample_rate=16000)

    # Access segments
    for segment in result.segments:
        print(f"{segment.start_sec:.2f}s - {segment.end_sec:.2f}s")
"""

from .base import (
    SpeechSegment,
    SegmentationResult,
    SpeechSegmenter,
)
from .factory import SpeechSegmenterFactory

__all__ = [
    "SpeechSegment",
    "SegmentationResult",
    "SpeechSegmenter",
    "SpeechSegmenterFactory",
]
