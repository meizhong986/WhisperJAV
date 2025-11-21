"""
Performance metrics schema for WhisperJAV.

Enables observability and performance tracking across transcription runs.
"""

from pydantic import Field

from .base import BaseConfig


class PerformanceMetrics(BaseConfig):
    """
    Performance metrics for transcription runs.

    Collected during processing for observability and optimization.
    """

    vad_processing_time: float = Field(
        default=0.0,
        description="Time spent in VAD processing (seconds)."
    )
    asr_processing_time: float = Field(
        default=0.0,
        description="Time spent in ASR processing (seconds)."
    )
    total_processing_time: float = Field(
        default=0.0,
        description="Total processing time (seconds)."
    )
    audio_duration: float = Field(
        default=0.0,
        description="Duration of input audio (seconds)."
    )
    real_time_factor: float = Field(
        default=0.0,
        description="Processing time / audio duration ratio."
    )
    segments_processed: int = Field(
        default=0,
        description="Number of audio segments processed."
    )
    words_transcribed: int = Field(
        default=0,
        description="Total words in transcription."
    )
