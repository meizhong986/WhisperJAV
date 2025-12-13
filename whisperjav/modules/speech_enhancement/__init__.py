"""
Speech Enhancement Module for WhisperJAV.

This module provides audio preprocessing capabilities to improve
transcription quality through noise reduction and vocal isolation.

Quick Start:
    from whisperjav.modules.speech_enhancement import (
        SpeechEnhancerFactory,
        EnhancementResult,
    )

    # Create enhancer (default: ClearVoice 48kHz)
    enhancer = SpeechEnhancerFactory.create("clearvoice")

    # Get preferred extraction rate
    extraction_sr = enhancer.get_preferred_sample_rate()  # 48000

    # Enhance audio
    result = enhancer.enhance(audio_array, sample_rate=extraction_sr)

    if result.success:
        enhanced_audio = result.audio
        # Resample to 16kHz for VAD/ASR
        from .base import resample_audio
        audio_16k = resample_audio(enhanced_audio, result.sample_rate, 16000)
    else:
        # Graceful degradation - use original
        audio_16k = original_audio

    # Always cleanup
    enhancer.cleanup()

Available Backends:
    - none: Passthrough (no enhancement)
    - clearvoice: ClearerVoice speech enhancement (denoising)
    - bs-roformer: BS-RoFormer vocal isolation

Design Principles:
    - Lazy loading: Backends loaded only when needed
    - Graceful degradation: Failures return original audio
    - Sample rate negotiation: Backends declare preferred rates
    - Memory efficiency: Sequential processing with cleanup
"""

from .base import (
    SpeechEnhancer,
    EnhancementResult,
    load_audio_to_array,
    resample_audio,
    create_failed_result,
)

from .factory import SpeechEnhancerFactory

from .pipeline_helper import (
    create_enhancer_from_config,
    create_enhancer_direct,
    get_extraction_sample_rate,
    enhance_scenes,
    enhance_single_audio,
)

__all__ = [
    # Protocol
    "SpeechEnhancer",
    # Data structures
    "EnhancementResult",
    # Factory
    "SpeechEnhancerFactory",
    # Utilities
    "load_audio_to_array",
    "resample_audio",
    "create_failed_result",
    # Pipeline integration
    "create_enhancer_from_config",
    "create_enhancer_direct",
    "get_extraction_sample_rate",
    "enhance_scenes",
    "enhance_single_audio",
]
