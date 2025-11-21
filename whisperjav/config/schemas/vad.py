"""
VAD (Voice Activity Detection) options schemas for WhisperJAV.

Defines parameters for different VAD backends.
"""

from typing import Optional

from pydantic import Field

from .base import BaseConfig


class SileroVADOptions(BaseConfig):
    """
    Silero VAD parameters for separate VAD handling.

    Maps to 'silero_vad_options' in asr_config.json.
    """

    threshold: float = Field(
        ge=0.0, le=1.0,
        description="Speech detection threshold."
    )
    min_speech_duration_ms: int = Field(
        ge=0,
        description="Minimum speech duration to keep (milliseconds)."
    )
    max_speech_duration_s: float = Field(
        gt=0,
        description="Maximum speech segment duration (seconds)."
    )
    min_silence_duration_ms: int = Field(
        ge=0,
        description="Minimum silence to split segments (milliseconds)."
    )
    neg_threshold: float = Field(
        ge=0.0, le=1.0,
        description="Negative threshold for speech end detection."
    )
    speech_pad_ms: int = Field(
        ge=0,
        description="Padding around speech segments (milliseconds)."
    )


class FasterWhisperVADOptions(BaseConfig):
    """
    Faster-Whisper integrated VAD parameters.

    Maps to 'faster_whisper_vad_options' in asr_config.json.
    These are merged into provider params when using faster-whisper.
    """

    vad_filter: bool = Field(
        description="Enable VAD filtering in faster-whisper."
    )
    vad_parameters: Optional[dict] = Field(
        default=None,
        description="Additional VAD parameters for faster-whisper."
    )


class StableTSVADOptions(BaseConfig):
    """
    Stable-TS VAD parameters.

    Maps to 'stable_ts_vad_options' in asr_config.json.
    These are merged into provider params when using stable-ts.
    """

    vad: bool = Field(
        description="Enable VAD in stable-ts."
    )
    vad_threshold: float = Field(
        ge=0.0, le=1.0,
        description="VAD threshold for stable-ts."
    )
