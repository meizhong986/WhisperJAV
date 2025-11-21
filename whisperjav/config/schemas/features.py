"""
Feature configuration schemas for WhisperJAV.

Defines scene detection and post-processing configurations.
"""

from pydantic import Field

from .base import BaseConfig


class AuditokSceneDetectionConfig(BaseConfig):
    """
    Auditok-based scene detection configuration (25 parameters).

    Maps to 'feature_configs.scene_detection.auditok' in asr_config.json.
    """

    # General settings
    max_duration_s: float = Field(
        default=29.0,
        description="Maximum segment duration in seconds."
    )
    min_duration_s: float = Field(
        default=0.2,
        description="Minimum segment duration in seconds."
    )
    target_sr: int = Field(
        default=16000,
        description="Target sample rate for processing."
    )
    force_mono: bool = Field(
        default=True,
        description="Convert audio to mono."
    )
    preserve_original_sr: bool = Field(
        default=True,
        description="Preserve original sample rate for output."
    )
    assist_processing: bool = Field(
        default=False,
        description="Enable assisted processing mode."
    )
    verbose_summary: bool = Field(
        default=True,
        description="Show verbose summary output."
    )

    # Pass 1 (coarse detection)
    pass1_min_duration_s: float = Field(
        default=2.0,
        description="Pass 1 minimum duration."
    )
    pass1_max_duration_s: float = Field(
        default=2700.0,
        description="Pass 1 maximum duration."
    )
    pass1_max_silence_s: float = Field(
        default=2.5,
        description="Pass 1 maximum silence duration."
    )
    pass1_energy_threshold: int = Field(
        default=32,
        description="Pass 1 energy threshold."
    )

    # Pass 2 (fine detection)
    pass2_min_duration_s: float = Field(
        default=0.1,
        description="Pass 2 minimum duration."
    )
    pass2_max_duration_s: float = Field(
        default=1800.0,
        description="Pass 2 maximum duration."
    )
    pass2_max_silence_s: float = Field(
        default=1.8,
        description="Pass 2 maximum silence duration."
    )
    pass2_energy_threshold: int = Field(
        default=38,
        description="Pass 2 energy threshold."
    )

    # Audio processing
    bandpass_low_hz: int = Field(
        default=200,
        description="Bandpass filter low frequency."
    )
    bandpass_high_hz: int = Field(
        default=4000,
        description="Bandpass filter high frequency."
    )
    drc_threshold_db: float = Field(
        default=-24.0,
        description="Dynamic range compression threshold."
    )
    drc_ratio: float = Field(
        default=4.0,
        description="Dynamic range compression ratio."
    )
    drc_attack_ms: float = Field(
        default=5.0,
        description="DRC attack time in milliseconds."
    )
    drc_release_ms: float = Field(
        default=100.0,
        description="DRC release time in milliseconds."
    )
    skip_assist_on_loud_dbfs: float = Field(
        default=-5.0,
        description="Skip assist on loud audio threshold."
    )

    # Fallback
    brute_force_fallback: bool = Field(
        default=True,
        description="Enable brute force fallback."
    )
    brute_force_chunk_s: float = Field(
        default=29.0,
        description="Brute force chunk size."
    )
    pad_edges_s: float = Field(
        default=0.0,
        description="Padding at segment edges."
    )
    fade_ms: int = Field(
        default=0,
        description="Fade duration in milliseconds."
    )


class SileroSceneDetectionConfig(BaseConfig):
    """
    Silero-based scene detection configuration (30 parameters).

    Maps to 'feature_configs.scene_detection.silero' in asr_config.json.
    """

    method: str = Field(
        default="silero",
        description="Scene detection method."
    )

    # General settings
    max_duration_s: float = Field(
        default=29.0,
        description="Maximum segment duration in seconds."
    )
    min_duration_s: float = Field(
        default=0.2,
        description="Minimum segment duration in seconds."
    )
    target_sr: int = Field(
        default=16000,
        description="Target sample rate for processing."
    )
    force_mono: bool = Field(
        default=True,
        description="Convert audio to mono."
    )
    preserve_original_sr: bool = Field(
        default=True,
        description="Preserve original sample rate for output."
    )
    assist_processing: bool = Field(
        default=False,
        description="Enable assisted processing mode."
    )
    verbose_summary: bool = Field(
        default=True,
        description="Show verbose summary output."
    )

    # Pass 1 (coarse detection)
    pass1_min_duration_s: float = Field(
        default=2.0,
        description="Pass 1 minimum duration."
    )
    pass1_max_duration_s: float = Field(
        default=2700.0,
        description="Pass 1 maximum duration."
    )
    pass1_max_silence_s: float = Field(
        default=2.5,
        description="Pass 1 maximum silence duration."
    )
    pass1_energy_threshold: int = Field(
        default=32,
        description="Pass 1 energy threshold."
    )

    # Pass 2 (Silero-specific fine detection)
    pass2_min_duration_s: float = Field(
        default=0.1,
        description="Pass 2 minimum duration."
    )
    pass2_max_duration_s: float = Field(
        default=960.0,
        description="Pass 2 maximum duration."
    )
    silero_threshold: float = Field(
        default=0.02,
        description="Silero speech detection threshold."
    )
    silero_neg_threshold: float = Field(
        default=0.1,
        description="Silero negative threshold."
    )
    silero_min_silence_ms: int = Field(
        default=7800,
        description="Silero minimum silence in milliseconds."
    )
    silero_min_speech_ms: int = Field(
        default=100,
        description="Silero minimum speech in milliseconds."
    )
    silero_max_speech_s: int = Field(
        default=960,
        description="Silero maximum speech in seconds."
    )
    silero_min_silence_at_max: int = Field(
        default=750,
        description="Minimum silence at max speech."
    )
    silero_speech_pad_ms: int = Field(
        default=500,
        description="Silero speech padding in milliseconds."
    )

    # Audio processing
    bandpass_low_hz: int = Field(
        default=200,
        description="Bandpass filter low frequency."
    )
    bandpass_high_hz: int = Field(
        default=4000,
        description="Bandpass filter high frequency."
    )
    drc_threshold_db: float = Field(
        default=-24.0,
        description="Dynamic range compression threshold."
    )
    drc_ratio: float = Field(
        default=4.0,
        description="Dynamic range compression ratio."
    )
    drc_attack_ms: float = Field(
        default=5.0,
        description="DRC attack time in milliseconds."
    )
    drc_release_ms: float = Field(
        default=100.0,
        description="DRC release time in milliseconds."
    )
    skip_assist_on_loud_dbfs: float = Field(
        default=-5.0,
        description="Skip assist on loud audio threshold."
    )

    # Fallback
    brute_force_fallback: bool = Field(
        default=True,
        description="Enable brute force fallback."
    )
    brute_force_chunk_s: float = Field(
        default=29.0,
        description="Brute force chunk size."
    )
    pad_edges_s: float = Field(
        default=0.0,
        description="Padding at segment edges."
    )
    fade_ms: int = Field(
        default=0,
        description="Fade duration in milliseconds."
    )


class PostProcessingConfig(BaseConfig):
    """
    Post-processing configuration.

    Maps to 'feature_configs.post_processing' in asr_config.json.
    """

    enabled: bool = Field(
        default=True,
        description="Enable post-processing."
    )
    sanitize: bool = Field(
        default=True,
        description="Enable text sanitization."
    )
