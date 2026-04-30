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
        0.28,
        ge=0.0, le=1.0,
        description="Speech probability threshold. Lower = more sensitive."
    )
    min_speech_duration_ms: int = Field(
        100,
        ge=0, le=5000,
        description="Minimum speech segment duration in milliseconds."
    )
    max_speech_duration_s: float = Field(
        5.0,                              # v1.8.12: 7.0→5.0, JA-sub-research tight default
        ge=0.0, le=300.0,
        description="Maximum speech segment duration in seconds."
    )
    min_silence_duration_ms: int = Field(
        300,
        ge=0, le=5000,
        description="Minimum silence duration to split segments."
    )
    speech_pad_ms: int = Field(
        700,
        ge=0, le=2000,
        description="Padding added around detected speech in milliseconds."
    )
    chunk_threshold_s: float = Field(
        2.5,
        ge=0.0, le=30.0,
        description="Gap threshold for segment grouping (seconds). Segments with gaps larger than this are split into separate groups."
    )
    max_group_duration_s: float = Field(
        6.0,                              # v1.8.12: 9.0→6.0, JA-sub-research tight default
        ge=1.0, le=60.0,
        description="Maximum duration for a segment group (seconds). Groups are split if adding a segment would exceed this limit."
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

    # === Presets - tuned for v1.8.9 ===
    # === Presets - v1.8.10-hf3: retuned for v3.1/v4.0 mega-group prevention ===
    # === Presets - v1.8.12: tight max_speech/max_group retune per JA-subtitle
    # === research (majority of JA subs <3s with ~800ms gaps). Sensitivity
    # === gradient inverted: aggressive=tightest caps, conservative=loosest.
    # === Mirrors v4 YAML config/v4/ecosystems/tools/silero-speech-segmentation.yaml.
    presets = {
        "conservative": SileroVADOptions(
            threshold=0.41,
            min_speech_duration_ms=150,
            max_speech_duration_s=6.0,         # v1.8.12: 6.0→6.0 (kept)
            min_silence_duration_ms=300,
            # neg_threshold: None — let VAD internal logic handle
            speech_pad_ms=500,
            chunk_threshold_s=2.5,
            max_group_duration_s=7.0,          # v1.8.12: 8.0→7.0
        ),
        "balanced": SileroVADOptions(
            threshold=0.28,
            min_speech_duration_ms=100,
            max_speech_duration_s=5.0,         # v1.8.12: 7.0→5.0
            min_silence_duration_ms=300,
            # neg_threshold: None — let VAD internal logic handle
            speech_pad_ms=400,
            chunk_threshold_s=2.5,
            max_group_duration_s=6.0,          # v1.8.12: 9.0→6.0
        ),
        "aggressive": SileroVADOptions(
            threshold=0.18,
            min_speech_duration_ms=30,
            max_speech_duration_s=4.0,         # v1.8.12: 8.0→4.0
            min_silence_duration_ms=300,
            # neg_threshold: None — let VAD internal logic handle
            speech_pad_ms=300,
            chunk_threshold_s=2.5,
            max_group_duration_s=5.0,          # v1.8.12: 10.0→5.0
        ),
    }
