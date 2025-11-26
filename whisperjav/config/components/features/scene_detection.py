"""
Scene Detection Feature Component.

Audio-based scene splitting using silence detection.

Parameter values match v1 asr_config.json exactly for backward compatibility.
"""

from pydantic import BaseModel, Field

from whisperjav.config.components.base import FeatureComponent, register_feature


class AuditokSceneDetectionOptions(BaseModel):
    """
    Complete Auditok scene detection options matching v1 asr_config.json.

    Uses a two-pass segmentation system with audio preprocessing.
    """

    # === Core Options ===
    max_duration_s: float = Field(
        29.0,
        ge=1.0, le=300.0,
        description="Maximum scene duration in seconds."
    )
    min_duration_s: float = Field(
        0.2,
        ge=0.1, le=10.0,
        description="Minimum scene duration in seconds."
    )
    target_sr: int = Field(
        16000,
        ge=8000, le=48000,
        description="Target sample rate for processing."
    )
    force_mono: bool = Field(
        True,
        description="Convert audio to mono before processing."
    )
    preserve_original_sr: bool = Field(
        True,
        description="Preserve original sample rate for output."
    )
    assist_processing: bool = Field(
        False,
        description="Enable assisted processing mode."
    )
    verbose_summary: bool = Field(
        True,
        description="Show verbose summary of scene detection."
    )

    # === Pass 1 Options (Coarse Segmentation) ===
    pass1_min_duration_s: float = Field(
        2.0,
        ge=0.1, le=60.0,
        description="Pass 1: Minimum segment duration in seconds."
    )
    pass1_max_duration_s: float = Field(
        2700.0,
        ge=60.0, le=7200.0,
        description="Pass 1: Maximum segment duration in seconds."
    )
    pass1_max_silence_s: float = Field(
        2.5,
        ge=0.1, le=30.0,
        description="Pass 1: Maximum silence to split on in seconds."
    )
    pass1_energy_threshold: int = Field(
        32,
        ge=1, le=100,
        description="Pass 1: Energy threshold for speech detection (dB)."
    )

    # === Pass 2 Options (Fine Segmentation) ===
    pass2_min_duration_s: float = Field(
        0.1,
        ge=0.01, le=10.0,
        description="Pass 2: Minimum segment duration in seconds."
    )
    pass2_max_duration_s: float = Field(
        1800.0,
        ge=30.0, le=3600.0,
        description="Pass 2: Maximum segment duration in seconds."
    )
    pass2_max_silence_s: float = Field(
        1.8,
        ge=0.1, le=10.0,
        description="Pass 2: Maximum silence to split on in seconds."
    )
    pass2_energy_threshold: int = Field(
        38,
        ge=1, le=100,
        description="Pass 2: Energy threshold for speech detection (dB)."
    )

    # === Audio Preprocessing Options ===
    bandpass_low_hz: int = Field(
        200,
        ge=20, le=1000,
        description="Bandpass filter low cutoff frequency in Hz."
    )
    bandpass_high_hz: int = Field(
        4000,
        ge=1000, le=20000,
        description="Bandpass filter high cutoff frequency in Hz."
    )
    drc_threshold_db: float = Field(
        -24.0,
        ge=-60.0, le=0.0,
        description="Dynamic range compression threshold in dB."
    )
    drc_ratio: float = Field(
        4.0,
        ge=1.0, le=20.0,
        description="Dynamic range compression ratio."
    )
    drc_attack_ms: float = Field(
        5.0,
        ge=0.1, le=100.0,
        description="DRC attack time in milliseconds."
    )
    drc_release_ms: float = Field(
        100.0,
        ge=10.0, le=1000.0,
        description="DRC release time in milliseconds."
    )
    skip_assist_on_loud_dbfs: float = Field(
        -5.0,
        ge=-30.0, le=0.0,
        description="Skip assist processing if audio is louder than this dBFS."
    )

    # === Fallback Options ===
    brute_force_fallback: bool = Field(
        True,
        description="Use brute force chunking as fallback."
    )
    brute_force_chunk_s: float = Field(
        29.0,
        ge=1.0, le=120.0,
        description="Chunk size for brute force fallback in seconds."
    )
    pad_edges_s: float = Field(
        0.0,
        ge=0.0, le=5.0,
        description="Padding to add at segment edges in seconds."
    )
    fade_ms: int = Field(
        0,
        ge=0, le=1000,
        description="Fade duration at segment edges in milliseconds."
    )


class SileroSceneDetectionOptions(BaseModel):
    """
    Complete Silero-based scene detection options matching v1 asr_config.json.

    Uses Silero VAD for more accurate speech detection combined with
    two-pass segmentation and audio preprocessing.
    """

    # === Core Options ===
    method: str = Field(
        "silero",
        description="Scene detection method identifier."
    )
    max_duration_s: float = Field(
        29.0,
        ge=1.0, le=300.0,
        description="Maximum scene duration in seconds."
    )
    min_duration_s: float = Field(
        0.2,
        ge=0.1, le=10.0,
        description="Minimum scene duration in seconds."
    )
    target_sr: int = Field(
        16000,
        ge=8000, le=48000,
        description="Target sample rate for processing."
    )
    force_mono: bool = Field(
        True,
        description="Convert audio to mono before processing."
    )
    preserve_original_sr: bool = Field(
        True,
        description="Preserve original sample rate for output."
    )
    assist_processing: bool = Field(
        False,
        description="Enable assisted processing mode."
    )
    verbose_summary: bool = Field(
        True,
        description="Show verbose summary of scene detection."
    )

    # === Pass 1 Options (Coarse Segmentation) ===
    pass1_min_duration_s: float = Field(
        2.0,
        ge=0.1, le=60.0,
        description="Pass 1: Minimum segment duration in seconds."
    )
    pass1_max_duration_s: float = Field(
        2700.0,
        ge=60.0, le=7200.0,
        description="Pass 1: Maximum segment duration in seconds."
    )
    pass1_max_silence_s: float = Field(
        2.5,
        ge=0.1, le=30.0,
        description="Pass 1: Maximum silence to split on in seconds."
    )
    pass1_energy_threshold: int = Field(
        32,
        ge=1, le=100,
        description="Pass 1: Energy threshold for speech detection (dB)."
    )

    # === Pass 2 Options (Fine Segmentation with Silero) ===
    pass2_min_duration_s: float = Field(
        0.1,
        ge=0.01, le=10.0,
        description="Pass 2: Minimum segment duration in seconds."
    )
    pass2_max_duration_s: float = Field(
        960.0,
        ge=30.0, le=3600.0,
        description="Pass 2: Maximum segment duration in seconds."
    )

    # === Silero VAD Options ===
    silero_threshold: float = Field(
        0.02,
        ge=0.0, le=1.0,
        description="Silero VAD speech probability threshold."
    )
    silero_neg_threshold: float = Field(
        0.1,
        ge=0.0, le=1.0,
        description="Silero VAD negative threshold for deactivation."
    )
    silero_min_silence_ms: int = Field(
        2000,
        ge=100, le=30000,
        description="Minimum silence duration to split in milliseconds."
    )
    silero_min_speech_ms: int = Field(
        100,
        ge=10, le=5000,
        description="Minimum speech duration in milliseconds."
    )
    silero_max_speech_s: float = Field(
        890.0,
        ge=10.0, le=3600.0,
        description="Maximum speech segment duration in seconds."
    )
    silero_min_silence_at_max: int = Field(
        750,
        ge=100, le=10000,
        description="Minimum silence at max speech duration in milliseconds."
    )
    silero_speech_pad_ms: int = Field(
        500,
        ge=0, le=2000,
        description="Padding around detected speech in milliseconds."
    )

    # === Audio Preprocessing Options ===
    bandpass_low_hz: int = Field(
        200,
        ge=20, le=1000,
        description="Bandpass filter low cutoff frequency in Hz."
    )
    bandpass_high_hz: int = Field(
        4000,
        ge=1000, le=20000,
        description="Bandpass filter high cutoff frequency in Hz."
    )
    drc_threshold_db: float = Field(
        -24.0,
        ge=-60.0, le=0.0,
        description="Dynamic range compression threshold in dB."
    )
    drc_ratio: float = Field(
        4.0,
        ge=1.0, le=20.0,
        description="Dynamic range compression ratio."
    )
    drc_attack_ms: float = Field(
        5.0,
        ge=0.1, le=100.0,
        description="DRC attack time in milliseconds."
    )
    drc_release_ms: float = Field(
        100.0,
        ge=10.0, le=1000.0,
        description="DRC release time in milliseconds."
    )
    skip_assist_on_loud_dbfs: float = Field(
        -5.0,
        ge=-30.0, le=0.0,
        description="Skip assist processing if audio is louder than this dBFS."
    )

    # === Fallback Options ===
    brute_force_fallback: bool = Field(
        True,
        description="Use brute force chunking as fallback."
    )
    brute_force_chunk_s: float = Field(
        29.0,
        ge=1.0, le=120.0,
        description="Chunk size for brute force fallback in seconds."
    )
    pad_edges_s: float = Field(
        0.0,
        ge=0.0, le=5.0,
        description="Padding to add at segment edges in seconds."
    )
    fade_ms: int = Field(
        0,
        ge=0, le=1000,
        description="Fade duration at segment edges in milliseconds."
    )


@register_feature
class AuditokSceneDetection(FeatureComponent):
    """Auditok-based audio scene detection."""

    # === Metadata ===
    name = "auditok_scene_detection"
    display_name = "Auditok Scene Detection"
    description = "Audio-based scene splitting using silence detection with two-pass segmentation."
    version = "1.0.0"
    tags = ["feature", "scene_detection", "auditok"]

    # === Feature-specific ===
    feature_type = "scene_detection"

    # === Schema ===
    Options = AuditokSceneDetectionOptions

    # === Presets - v1 has NO sensitivity presets, all identical to defaults ===
    presets = {
        "conservative": AuditokSceneDetectionOptions(),
        "balanced": AuditokSceneDetectionOptions(),
        "aggressive": AuditokSceneDetectionOptions(),
    }


@register_feature
class SileroSceneDetection(FeatureComponent):
    """Silero VAD-based audio scene detection."""

    # === Metadata ===
    name = "silero_scene_detection"
    display_name = "Silero Scene Detection"
    description = "Scene splitting using Silero VAD for accurate speech detection."
    version = "1.0.0"
    tags = ["feature", "scene_detection", "silero", "vad"]

    # === Feature-specific ===
    feature_type = "scene_detection"

    # === Schema ===
    Options = SileroSceneDetectionOptions

    # === Presets - v1 has NO sensitivity presets, all identical to defaults ===
    presets = {
        "conservative": SileroSceneDetectionOptions(),
        "balanced": SileroSceneDetectionOptions(),
        "aggressive": SileroSceneDetectionOptions(),
    }
