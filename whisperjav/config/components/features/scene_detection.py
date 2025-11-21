"""
Scene Detection Feature Component.

Audio-based scene splitting using silence detection.
"""

from pydantic import BaseModel, Field

from whisperjav.config.components.base import FeatureComponent, register_feature


class AuditokSceneDetectionOptions(BaseModel):
    """Auditok-based scene detection options."""

    min_duration_s: float = Field(
        0.5, ge=0.1, le=10.0,
        description="Minimum scene duration in seconds."
    )
    max_duration_s: float = Field(
        29.0, ge=5.0, le=120.0,
        description="Maximum scene duration in seconds."
    )
    max_silence_s: float = Field(
        0.5, ge=0.1, le=5.0,
        description="Maximum silence within a scene."
    )
    energy_threshold: int = Field(
        50, ge=10, le=100,
        description="Energy threshold for speech detection (dB)."
    )
    target_sr: int = Field(
        16000, ge=8000, le=48000,
        description="Target sample rate for processing."
    )
    force_mono: bool = Field(
        True,
        description="Convert audio to mono before processing."
    )
    drop_trailing_silence: bool = Field(
        True,
        description="Remove silence at end of segments."
    )


@register_feature
class AuditokSceneDetection(FeatureComponent):
    """Auditok-based audio scene detection."""

    # === Metadata ===
    name = "auditok_scene_detection"
    display_name = "Auditok Scene Detection"
    description = "Audio-based scene splitting using silence detection. Good for general content."
    version = "1.0.0"
    tags = ["feature", "scene_detection", "auditok"]

    # === Feature-specific ===
    feature_type = "scene_detection"

    # === Schema ===
    Options = AuditokSceneDetectionOptions

    # === Presets (scene detection typically doesn't vary by sensitivity) ===
    presets = {
        "conservative": AuditokSceneDetectionOptions(
            min_duration_s=1.0,
            max_duration_s=29.0,
            max_silence_s=0.7,
            energy_threshold=55,
        ),
        "balanced": AuditokSceneDetectionOptions(
            min_duration_s=0.5,
            max_duration_s=29.0,
            max_silence_s=0.5,
            energy_threshold=50,
        ),
        "aggressive": AuditokSceneDetectionOptions(
            min_duration_s=0.3,
            max_duration_s=29.0,
            max_silence_s=0.3,
            energy_threshold=45,
        ),
    }
