"""
Legacy Pipeline Mappings for WhisperJAV v3.0.

Maps old pipeline names to new component-based configurations
for backward compatibility.
"""

from typing import Any, Dict, List, Optional

from .resolver_v3 import resolve_config_v3


# Legacy pipeline definitions
LEGACY_PIPELINES = {
    "balanced": {
        "asr": "faster_whisper",
        "vad": "silero",
        "features": ["auditok_scene_detection"],
        "description": "Full feature set with scene detection and VAD. Best quality.",
    },
    "faster": {
        "asr": "stable_ts",
        "vad": "none",
        "features": [],
        "description": "Speed-optimized with Stable-TS. No VAD or scene detection.",
    },
    "fast": {
        "asr": "stable_ts",
        "vad": "none",
        "features": ["auditok_scene_detection"],
        "description": "Stable-TS with scene detection. Good speed/quality balance.",
    },
    "fidelity": {
        "asr": "faster_whisper",
        "vad": "none",
        "features": [],
        "description": "Faster-Whisper without preprocessing. Direct transcription.",
    },
}


def resolve_legacy_pipeline(
    pipeline_name: str,
    sensitivity: str = "balanced",
    task: str = "transcribe",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve configuration from legacy pipeline name.

    This provides backward compatibility with the old CLI interface.

    Args:
        pipeline_name: Legacy pipeline name ('balanced', 'faster', 'fast', 'fidelity')
        sensitivity: Sensitivity level
        task: Task type
        overrides: Parameter overrides

    Returns:
        Resolved configuration dictionary.

    Example:
        >>> config = resolve_legacy_pipeline('balanced', 'aggressive')
    """
    if pipeline_name not in LEGACY_PIPELINES:
        available = list(LEGACY_PIPELINES.keys())
        raise ValueError(f"Unknown pipeline: {pipeline_name}. Available: {available}")

    pipeline_def = LEGACY_PIPELINES[pipeline_name]

    # Resolve using new system
    config = resolve_config_v3(
        asr=pipeline_def["asr"],
        vad=pipeline_def["vad"],
        sensitivity=sensitivity,
        task=task,
        features=pipeline_def["features"],
        overrides=overrides,
    )

    # Add legacy compatibility fields
    config['pipeline_name'] = pipeline_name
    config['sensitivity_name'] = sensitivity

    # Map to old output structure for backward compatibility
    return _map_to_legacy_structure(config, pipeline_def)


def _map_to_legacy_structure(config: Dict[str, Any], pipeline_def: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map v3 config to legacy output structure.

    This ensures pipelines that expect the old structure continue to work.
    Maps ALL parameters from v3 flat structure to v1 nested decoder/provider structure.
    """
    # Build workflow structure (legacy)
    workflow = {
        'model': config['model']['model_name'],
        'vad': config['vad_name'] if config['vad_name'] != 'none' else 'none',
        'backend': _get_backend_name(config['asr_name']),
    }

    # Add features to workflow
    if config['features']:
        workflow['features'] = {
            feature_type: True for feature_type in config['features'].keys()
        }

    # Build params structure (legacy)
    # Map 'asr' to 'decoder' and 'provider' for backward compat
    asr_params = config['params']['asr']

    # Decoder params - from common_decoder_options
    decoder_params = {
        'task': asr_params.get('task', 'transcribe'),
        'language': asr_params.get('language', 'ja'),
        'beam_size': asr_params.get('beam_size', 2),
        'best_of': asr_params.get('best_of', 1),
        'patience': asr_params.get('patience', 2.0),
        'length_penalty': asr_params.get('length_penalty'),
        'prefix': asr_params.get('prefix'),
        'suppress_tokens': asr_params.get('suppress_tokens'),
        'suppress_blank': asr_params.get('suppress_blank', True),
        'without_timestamps': asr_params.get('without_timestamps', False),
        'max_initial_timestamp': asr_params.get('max_initial_timestamp'),
    }

    # Provider params - from common_transcriber_options + engine_options + exclusive
    provider_params = {
        # From common_transcriber_options
        'temperature': asr_params.get('temperature', [0.0, 0.1]),
        'compression_ratio_threshold': asr_params.get('compression_ratio_threshold', 2.4),
        'logprob_threshold': asr_params.get('logprob_threshold', -1.2),
        'logprob_margin': asr_params.get('logprob_margin', 0.2),
        'no_speech_threshold': asr_params.get('no_speech_threshold', 0.5),
        'drop_nonverbal_vocals': asr_params.get('drop_nonverbal_vocals', False),
        'condition_on_previous_text': asr_params.get('condition_on_previous_text', False),
        'initial_prompt': asr_params.get('initial_prompt'),
        'word_timestamps': asr_params.get('word_timestamps', True),
        'prepend_punctuations': asr_params.get('prepend_punctuations'),
        'append_punctuations': asr_params.get('append_punctuations'),
        'clip_timestamps': asr_params.get('clip_timestamps'),
        # From faster_whisper_engine_options
        'chunk_length': asr_params.get('chunk_length'),
        'repetition_penalty': asr_params.get('repetition_penalty', 1.5),
        'no_repeat_ngram_size': asr_params.get('no_repeat_ngram_size', 2),
        'prompt_reset_on_temperature': asr_params.get('prompt_reset_on_temperature'),
        'hotwords': asr_params.get('hotwords'),
        'multilingual': asr_params.get('multilingual', False),
        'max_new_tokens': asr_params.get('max_new_tokens'),
        'language_detection_threshold': asr_params.get('language_detection_threshold'),
        'language_detection_segments': asr_params.get('language_detection_segments'),
        'log_progress': asr_params.get('log_progress', False),
        # From exclusive_whisper_plus_faster_whisper
        'hallucination_silence_threshold': asr_params.get('hallucination_silence_threshold', 2.0),
    }

    return {
        'pipeline_name': config['pipeline_name'],
        'sensitivity_name': config['sensitivity_name'],
        'workflow': workflow,
        'model': config['model'],
        'params': {
            'decoder': decoder_params,
            'provider': provider_params,
            'vad': config['params']['vad'],
        },
        'features': config['features'],
        'task': config['task'],
        'language': config['language'],
    }


def _get_backend_name(asr_name: str) -> str:
    """Get backend name from ASR component name."""
    backend_map = {
        'faster_whisper': 'faster-whisper',
        'stable_ts': 'stable-ts',
        'openai_whisper': 'whisper',
    }
    return backend_map.get(asr_name, asr_name)


def list_legacy_pipelines() -> List[str]:
    """List available legacy pipeline names."""
    return list(LEGACY_PIPELINES.keys())


def get_legacy_pipeline_info(pipeline_name: str) -> Dict[str, Any]:
    """Get information about a legacy pipeline."""
    if pipeline_name not in LEGACY_PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    return {
        'name': pipeline_name,
        **LEGACY_PIPELINES[pipeline_name]
    }
