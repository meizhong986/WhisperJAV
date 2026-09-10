"""
Scene Detection Feature Component.

Audio-based scene splitting using silence detection.

These Pydantic presets are the SINGLE SOURCE OF TRUTH for scene detection
parameters across all pipeline modes. As of v1.8.9, asr_config.json is
no longer read for scene detection parameters.
"""

from typing import Optional

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
        ge=1.0, le=1200.0,
        description="Maximum scene duration in seconds."
    )
    min_duration_s: float = Field(
        0.2,
        ge=0.1, le=10.0,
        description=(
            "Minimum scene duration in seconds. NOTE: on the auditok and silero backends this is a "
            "DISCARD filter, not a merge floor -- a region shorter than this is dropped together "
            "with its audio. Keep it small; a scene-length floor is only enforceable on the "
            "semantic backend, whose min_duration merges short segments into their neighbours."
        )
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
    pass2_max_duration_s: Optional[float] = Field(
        None,
        ge=1.0, le=3600.0,
        description="Pass 2: Maximum segment duration in seconds. None = auto-derive from max_duration - 1.0."
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
        description="DEPRECATED: Not implemented by any backend. Accepted for backward compatibility only.",
        deprecated=True,
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
        ge=1.0, le=1200.0,
        description="Maximum scene duration in seconds."
    )
    min_duration_s: float = Field(
        0.2,
        ge=0.1, le=10.0,
        description=(
            "Minimum scene duration in seconds. NOTE: on the auditok and silero backends this is a "
            "DISCARD filter, not a merge floor -- a region shorter than this is dropped together "
            "with its audio. Keep it small; a scene-length floor is only enforceable on the "
            "semantic backend, whose min_duration merges short segments into their neighbours."
        )
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
        0.08,
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
        description="DEPRECATED: Not implemented by any backend. Accepted for backward compatibility only.",
        deprecated=True,
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

    # === Presets ===
    # Conservative: fewer, larger scenes — need longer silences to split.
    # Balanced: defaults.
    # Aggressive: more, smaller scenes — split on shorter silences.
    presets = {
        "conservative": AuditokSceneDetectionOptions(
            min_duration_s=1.0,
            pass1_max_silence_s=4.0,
            pass1_energy_threshold=40,
            pass2_max_silence_s=2.5,
            pass2_energy_threshold=45,
        ),
        "balanced": AuditokSceneDetectionOptions(),
        "aggressive": AuditokSceneDetectionOptions(
            min_duration_s=0.1,
            pass1_max_silence_s=1.5,
            pass1_energy_threshold=25,
            pass2_max_silence_s=1.0,
            pass2_energy_threshold=30,
        ),
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

    # === Presets ===
    # Conservative: fewer scenes — higher VAD threshold, longer required silences.
    # Balanced: defaults.
    # Aggressive: more scenes — lower VAD threshold, shorter silences trigger split.
    presets = {
        "conservative": SileroSceneDetectionOptions(
            min_duration_s=1.0,
            pass1_max_silence_s=4.0,
            pass1_energy_threshold=40,
            silero_threshold=0.12,
            silero_min_silence_ms=2500,
            silero_speech_pad_ms=300,
        ),
        "balanced": SileroSceneDetectionOptions(),
        "aggressive": SileroSceneDetectionOptions(
            min_duration_s=0.1,
            pass1_max_silence_s=1.5,
            pass1_energy_threshold=25,
            silero_threshold=0.05,
            silero_min_silence_ms=1200,
            silero_speech_pad_ms=150,
        ),
    }


class SemanticSceneDetectionOptions(BaseModel):
    """
    Semantic (texture-clustering) scene detection options.

    IMPORTANT -- FIELD NAMES: the semantic backend reads these keys WITHOUT the
    ``_s`` suffix (``whisperjav/modules/scene_detection_backends/semantic_backend.py``
    builds its config from ``min_duration`` / ``max_duration`` / ``snap_window`` /
    ``clustering_threshold``), and ``SceneDetectorFactory`` passes kwargs through
    untranslated. A key named ``min_duration_s`` is therefore silently ignored by this
    backend. That mismatch is why, before v1.9.2, every semantic run fell back to the
    engine's own hard-coded defaults regardless of the resolved configuration.

    Values below mirror ``config/v4/ecosystems/tools/semantic-scene-detection.yaml``
    and ``modules/scene_detection_backends/semantic_adapter.py``; nothing here is new.
    """

    method: str = Field(
        "semantic",
        description="Scene detection method identifier."
    )
    min_duration: float = Field(
        20.0,
        ge=1.0, le=1200.0,
        description=(
            "Minimum segment duration in seconds. Unlike the auditok/silero backends, this is a "
            "real merge floor: shorter segments are merged into a neighbour, so no audio is lost."
        )
    )
    max_duration: float = Field(
        420.0,
        ge=10.0, le=1200.0,
        description=(
            "Merge ceiling in seconds: a merge that would exceed it is declined. NOTE: this is "
            "NOT a splitter -- the engine logs a warning if clustering produces an overlong "
            "segment and keeps it (WJAV mod D)."
        )
    )
    snap_window: float = Field(
        6.0,
        ge=0.5, le=15.0,
        description=(
            "Window in seconds searched either side of a raw boundary when snapping it onto silence. "
            "v1.9.2 (owner O1): ONE value for every sensitivity -- this field default is the only "
            "place it is set. Do not re-introduce a per-sensitivity value."
        )
    )
    clustering_threshold: float = Field(
        22.0,
        ge=1.0, le=50.0,
        description=(
            "Agglomerative clustering distance separating one scene from the next. Lower = more scenes. "
            "v1.9.2 (owner O1): ONE value for every sensitivity -- this field default is the only "
            "place it is set. Do not re-introduce a per-sensitivity value."
        )
    )
    sample_rate: int = Field(
        16000,
        ge=8000, le=48000,
        description="Target sample rate for feature extraction."
    )
    preserve_original_sr: bool = Field(
        True,
        description="Write scene WAVs at the original sample rate."
    )
    visualize: bool = Field(
        False,
        description="Write a PNG plot of the scene boundaries and their classified types."
    )


@register_feature
class SemanticSceneDetection(FeatureComponent):
    """Semantic audio-clustering scene detection (v1.9.2 default)."""

    # === Metadata ===
    name = "semantic_scene_detection"
    display_name = "Semantic Audio Clustering"
    description = (
        "Texture-based scene splitting using MFCC features and agglomerative clustering, "
        "with boundaries snapped onto silence and anchored to the following sound onset."
    )
    version = "1.0.0"
    tags = ["feature", "scene_detection", "semantic", "clustering"]

    # === Feature-specific ===
    feature_type = "scene_detection"

    # === Schema ===
    Options = SemanticSceneDetectionOptions

    # === Presets ===
    # Identical to the conservative/balanced/aggressive presets already declared in
    # semantic-scene-detection.yaml, so the YAML-driven GUI panel and this component
    # cannot disagree. "balanced" is the spec default (an empty preset in the YAML).
    #
    # v1.9.2 (owner O1): snap_window and clustering_threshold are UNIFORM across every
    # sensitivity -- 6.0 s and 22.0. They are therefore deliberately ABSENT from the
    # preset constructors below, so the field defaults on SemanticSceneDetectionOptions
    # are the single place either value is written. Only the scene-length bounds still
    # vary by sensitivity, and on the balanced pipeline even those are replaced by
    # LEGACY_PIPELINES["balanced"]["scene_overrides"] (28 s / 1200 s).
    presets = {
        "conservative": SemanticSceneDetectionOptions(
            min_duration=30.0,
            max_duration=420.0,
        ),
        "balanced": SemanticSceneDetectionOptions(),
        "aggressive": SemanticSceneDetectionOptions(
            min_duration=10.0,
            max_duration=180.0,
        ),
    }
