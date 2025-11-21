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

    decoder_params = {
        'task': asr_params.get('task', 'transcribe'),
        'language': asr_params.get('language', 'ja'),
        'beam_size': asr_params.get('beam_size', 5),
        'best_of': asr_params.get('best_of', 5),
        'patience': asr_params.get('patience', 1.2),
    }

    provider_params = {
        'temperature': asr_params.get('temperature', 0.0),
        'compression_ratio_threshold': asr_params.get('compression_ratio_threshold', 2.4),
        'logprob_threshold': asr_params.get('logprob_threshold', -1.0),
        'no_speech_threshold': asr_params.get('no_speech_threshold', 0.6),
    }

    # Add hallucination threshold if present
    if 'hallucination_silence_threshold' in asr_params:
        provider_params['hallucination_silence_threshold'] = asr_params['hallucination_silence_threshold']

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
