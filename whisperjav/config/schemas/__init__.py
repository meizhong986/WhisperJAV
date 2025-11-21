"""
WhisperJAV Configuration Schemas v2.0.

Type-safe Pydantic models for all configuration parameters.
"""

from .base import BaseConfig, Backend, Sensitivity
from .decoder import DecoderOptions
from .engine import (
    FasterWhisperEngineOptions,
    OpenAIWhisperEngineOptions,
    StableTSEngineOptions,
)
from .features import (
    AuditokSceneDetectionConfig,
    PostProcessingConfig,
    SileroSceneDetectionConfig,
)
from .jav import JAVAudioConfig
from .metrics import PerformanceMetrics
from .model import MODELS, VAD_ENGINES, ModelConfig, VADEngineConfig
from .pipeline import ResolvedConfig, ResolvedParams, WorkflowConfig
from .transcriber import TranscriberOptions
from .ui import UIPreferences
from .vad import FasterWhisperVADOptions, SileroVADOptions, StableTSVADOptions
from .presets import (
    DECODER_PRESETS,
    FASTER_WHISPER_ENGINE_PRESETS,
    HALLUCINATION_THRESHOLDS,
    SILERO_VAD_PRESETS,
    STABLE_TS_ENGINE_OPTIONS,
    STABLE_TS_VAD_PRESETS,
    TRANSCRIBER_PRESETS,
    get_decoder_preset,
    get_faster_whisper_engine_preset,
    get_silero_vad_preset,
    get_stable_ts_vad_preset,
    get_transcriber_preset,
)

__all__ = [
    # Base classes
    "BaseConfig",
    "Backend",
    "Sensitivity",
    # Model schemas
    "ModelConfig",
    "VADEngineConfig",
    "MODELS",
    "VAD_ENGINES",
    # Parameter schemas
    "TranscriberOptions",
    "DecoderOptions",
    # VAD schemas
    "SileroVADOptions",
    "FasterWhisperVADOptions",
    "StableTSVADOptions",
    # Engine schemas
    "FasterWhisperEngineOptions",
    "OpenAIWhisperEngineOptions",
    "StableTSEngineOptions",
    # Feature schemas
    "AuditokSceneDetectionConfig",
    "SileroSceneDetectionConfig",
    "PostProcessingConfig",
    # Pipeline schemas
    "WorkflowConfig",
    "ResolvedParams",
    "ResolvedConfig",
    # UI
    "UIPreferences",
    # JAV-specific
    "JAVAudioConfig",
    # Metrics
    "PerformanceMetrics",
    # Presets
    "TRANSCRIBER_PRESETS",
    "DECODER_PRESETS",
    "SILERO_VAD_PRESETS",
    "STABLE_TS_VAD_PRESETS",
    "FASTER_WHISPER_ENGINE_PRESETS",
    "STABLE_TS_ENGINE_OPTIONS",
    "HALLUCINATION_THRESHOLDS",
    "get_transcriber_preset",
    "get_decoder_preset",
    "get_silero_vad_preset",
    "get_stable_ts_vad_preset",
    "get_faster_whisper_engine_preset",
]
