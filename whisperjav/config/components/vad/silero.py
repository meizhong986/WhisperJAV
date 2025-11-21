"""
Silero VAD Component.

High-quality Voice Activity Detection from Silero team.
"""

from pydantic import BaseModel, Field

from whisperjav.config.components.base import VADComponent, register_vad


class SileroVADOptions(BaseModel):
    """Silero VAD configuration options."""

    threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Speech probability threshold. Higher = less sensitive to speech."
    )
    min_speech_duration_ms: int = Field(
        250, ge=0, le=5000,
        description="Minimum speech segment duration in milliseconds."
    )
    max_speech_duration_s: float = Field(
        float('inf'), ge=0.0,
        description="Maximum speech segment duration in seconds."
    )
    min_silence_duration_ms: int = Field(
        100, ge=0, le=5000,
        description="Minimum silence duration to split segments."
    )
    speech_pad_ms: int = Field(
        400, ge=0, le=2000,
        description="Padding added around detected speech in milliseconds."
    )
    window_size_samples: int = Field(
        1536, ge=512, le=4096,
        description="Window size for VAD processing."
    )


@register_vad
class SileroVAD(VADComponent):
    """Silero Voice Activity Detection."""

    # === Metadata ===
    name = "silero"
    display_name = "Silero VAD"
    description = "High-quality neural VAD from Silero team. Good balance of accuracy and speed."
    version = "4.0.0"
    tags = ["vad", "neural", "silero"]

    # === VAD-specific ===
    compatible_asr = ["faster_whisper", "stable_ts", "openai_whisper"]

    # === Schema ===
    Options = SileroVADOptions

    # === Presets ===
    presets = {
        "conservative": SileroVADOptions(
            threshold=0.6,
            min_speech_duration_ms=300,
            min_silence_duration_ms=150,
            speech_pad_ms=500,
        ),
        "balanced": SileroVADOptions(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            speech_pad_ms=400,
        ),
        "aggressive": SileroVADOptions(
            threshold=0.35,
            min_speech_duration_ms=100,
            min_silence_duration_ms=50,
            speech_pad_ms=200,
        ),
    }
