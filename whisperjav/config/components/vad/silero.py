"""
Silero VAD Component.

High-quality Voice Activity Detection from Silero team.

.. deprecated:: 1.7.0
    This module is deprecated in favor of the v4 YAML configuration.
    Use ``whisperjav.config.v4.ecosystems.tools.silero-speech-segmentation.yaml``
    as the source of truth for Silero VAD presets.

    To load Silero VAD config in v4:

        from whisperjav.config.v4.registries.tool_registry import get_tool_registry
        registry = get_tool_registry()
        tool = registry.get("silero-speech-segmentation")
        config = tool.get_resolved_config("balanced")

    This file is kept for backward compatibility with existing code.
    It will be removed in a future version.

Parameter values match v1 asr_config.json exactly for backward compatibility.
"""

from pydantic import BaseModel, Field

from whisperjav.config.components.base import VADComponent, register_vad


class SileroVADOptions(BaseModel):
    """
    Silero VAD configuration options matching v1 silero_vad_options.
    """

    threshold: float = Field(
        0.25,
        ge=0.0, le=1.0,
        description="Speech probability threshold. Lower = more sensitive."
    )
    min_speech_duration_ms: int = Field(
        100,
        ge=0, le=5000,
        description="Minimum speech segment duration in milliseconds."
    )
    max_speech_duration_s: float = Field(
        6.0,
        ge=0.0, le=300.0,
        description="Maximum speech segment duration in seconds."
    )
    min_silence_duration_ms: int = Field(
        300,
        ge=0, le=5000,
        description="Minimum silence duration to split segments."
    )
    neg_threshold: float = Field(
        0.15,
        ge=0.0, le=1.0,
        description="Negative speech threshold for deactivation."
    )
    speech_pad_ms: int = Field(
        700,
        ge=0, le=2000,
        description="Padding added around detected speech in milliseconds."
    )
    chunk_threshold_s: float = Field(
        0.2,
        ge=0.0, le=30.0,
        description="Gap threshold for segment grouping (seconds). Segments with gaps larger than this are split into separate groups."
    )
    max_group_duration_s: float = Field(
        29.0,
        ge=1.0, le=60.0,
        description="Maximum duration for a segment group (seconds). Groups are split if adding a segment would exceed this limit. Default 29s to stay within Whisper's 30s context window."
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

    # === Presets - v1.7.1 exact values ===
    presets = {
        "conservative": SileroVADOptions(
            threshold=0.35,  # v1.7.1 value
            min_speech_duration_ms=150,
            max_speech_duration_s=9.0,  # v1.7.1 value
            min_silence_duration_ms=300,
            neg_threshold=0.3,
            speech_pad_ms=400,  # v1.7.1 value
            chunk_threshold_s=4.0,
            max_group_duration_s=29.0,  # Whisper 30s context window limit
        ),
        "balanced": SileroVADOptions(
            threshold=0.18,  # v1.7.1 value
            min_speech_duration_ms=100,
            max_speech_duration_s=11.0,  # v1.7.1 value
            min_silence_duration_ms=300,
            neg_threshold=0.15,
            speech_pad_ms=400,  # v1.7.1 value
            chunk_threshold_s=4.0,
            max_group_duration_s=29.0,  # Whisper 30s context window limit
        ),
        "aggressive": SileroVADOptions(
            threshold=0.05,  # v1.7.1 value
            min_speech_duration_ms=30,
            max_speech_duration_s=14.0,  # v1.7.1 value
            min_silence_duration_ms=300,
            neg_threshold=0.1,
            speech_pad_ms=600,  # v1.7.1 value
            chunk_threshold_s=4.0,
            max_group_duration_s=29.0,  # Whisper 30s context window limit
        ),
    }
