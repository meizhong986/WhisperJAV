"""
Legacy Pipeline Mappings for WhisperJAV v3.0.

.. deprecated:: 1.7.0
    This module is part of the LEGACY configuration system (v1-v3).
    For new development, use the v4 YAML-driven configuration system:

        from whisperjav.config.v4 import ConfigManager

    See: whisperjav/config/v4/README.md
    ADR: docs/adr/ADR-001-yaml-config-architecture.md

PURPOSE AND SCOPE
=================
This module provides configuration resolution for pipelines that use the
LEGACY CONFIG SYSTEM. It maps pipeline mode names (e.g., "balanced", "fast")
to component-based configurations with ASR, VAD, and feature settings.

WHAT BELONGS HERE
-----------------
Pipelines that:
- Use the `resolve_legacy_pipeline()` function for configuration
- Accept `resolved_config` parameter in their __init__
- Rely on the v3 config structure with model, params, and features sections
- Examples: balanced, fast, faster, fidelity, kotoba-faster-whisper

WHAT DOES NOT BELONG HERE
-------------------------
Pipelines that:
- Use dedicated CLI arguments instead of legacy config resolution
- Bypass `resolve_legacy_pipeline()` entirely in main.py
- Have their own independent configuration system
- Example: "transformers" mode uses --hf-* arguments directly

ADDING NEW PIPELINES
--------------------
Before adding a new pipeline to LEGACY_PIPELINES, ask:
1. Does it need the legacy config resolution system?
2. Will it accept `resolved_config` in its __init__?
3. Does it use the standard ASR/VAD/features component model?

If NO to any of these, the pipeline should NOT be added here.
Instead, handle its config resolution separately in main.py.

Maps old pipeline names to new component-based configurations
for backward compatibility.
"""

from typing import Any, Dict, List, Optional

from .resolver_v3 import resolve_config_v3


def _filter_none_values(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively remove None values from a dictionary.

    This is critical for backend compatibility. When passing parameters to
    libraries like faster-whisper, openai-whisper, or stable-ts:

    - Passing `func(arg=None)` explicitly passes None, NOT the library default
    - This causes errors like "TypeError: 'NoneType' object is not iterable"
    - By removing None keys, we allow library defaults to apply

    Preserves valid falsy values: 0, False, "", [], {}

    Args:
        params: Dictionary of parameters (may be nested)

    Returns:
        Filtered dictionary with None values removed
    """
    if not isinstance(params, dict):
        return params

    filtered = {}
    for key, value in params.items():
        if value is None:
            # Skip None values - let library defaults apply
            continue
        elif isinstance(value, dict):
            # Recursively filter nested dicts
            filtered_nested = _filter_none_values(value)
            # Only include if non-empty after filtering
            if filtered_nested:
                filtered[key] = filtered_nested
        else:
            # Preserve all other values including falsy ones (0, False, "", [])
            filtered[key] = value

    return filtered


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
        "asr": "openai_whisper",
        "vad": "silero",
        "features": ["auditok_scene_detection"],
        "description": "OpenAI Whisper with VAD and scene detection. Maximum fidelity.",
    },
    "kotoba-faster-whisper": {
        "asr": "kotoba_faster_whisper",
        "vad": "none",  # Uses internal VAD (faster-whisper built-in)
        "features": ["auditok_scene_detection"],  # Scene detection always enabled
        "description": "Japanese-optimized Kotoba Faster-Whisper with internal VAD.",
        "use_v3_structure": True,  # Return V3 config, not legacy mapped
    },
    # NOTE: "transformers" mode is NOT listed here.
    # It uses dedicated --hf-* CLI arguments and bypasses legacy config resolution entirely.
    # See whisperjav/pipelines/transformers_pipeline.py for its implementation.
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
        pipeline_name: Legacy pipeline name ('balanced', 'faster', 'fast', 'fidelity', 'kotoba-faster-whisper')
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

    # For pipelines that use V3 structure (like kotoba-faster-whisper),
    # return V3 config directly without legacy mapping
    if pipeline_def.get("use_v3_structure", False):
        return config

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
    asr_name = config['asr_name']

    # Decoder params - from common_decoder_options (same for all backends)
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

    # Provider params - BACKEND-SPECIFIC
    # Different backends accept different parameters
    # Common transcriber options (shared by all backends)
    provider_params = {
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
    }

    # Add backend-specific engine options
    if asr_name == 'faster_whisper':
        # faster_whisper_engine_options
        provider_params.update({
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
            # exclusive_whisper_plus_faster_whisper
            'hallucination_silence_threshold': asr_params.get('hallucination_silence_threshold', 2.0),
        })
    elif asr_name == 'openai_whisper':
        # openai_whisper_engine_options
        provider_params.update({
            'verbose': asr_params.get('verbose'),
            'carry_initial_prompt': asr_params.get('carry_initial_prompt'),
            'prompt': asr_params.get('prompt'),
            'fp16': asr_params.get('fp16', True),
            # exclusive_whisper_plus_faster_whisper
            'hallucination_silence_threshold': asr_params.get('hallucination_silence_threshold', 2.0),
        })
    elif asr_name == 'stable_ts':
        # stable_ts_engine_options
        provider_params.update({
            'stream': asr_params.get('stream'),
            'mel_first': asr_params.get('mel_first'),
            'split_callback': asr_params.get('split_callback'),
            'suppress_ts_tokens': asr_params.get('suppress_ts_tokens', False),
            'gap_padding': asr_params.get('gap_padding', ' ...'),
            'only_ffmpeg': asr_params.get('only_ffmpeg', False),
            'max_instant_words': asr_params.get('max_instant_words', 0.5),
            'avg_prob_threshold': asr_params.get('avg_prob_threshold'),
            'nonspeech_skip': asr_params.get('nonspeech_skip'),
            'progress_callback': asr_params.get('progress_callback'),
            'ignore_compatibility': asr_params.get('ignore_compatibility', True),
            'extra_models': asr_params.get('extra_models'),
            'dynamic_heads': asr_params.get('dynamic_heads'),
            'nonspeech_error': asr_params.get('nonspeech_error', 0.1),
            'only_voice_freq': asr_params.get('only_voice_freq', False),
            'min_word_dur': asr_params.get('min_word_dur'),
            'min_silence_dur': asr_params.get('min_silence_dur'),
            'regroup': asr_params.get('regroup', True),
            'ts_num': asr_params.get('ts_num', 0),
            'ts_noise': asr_params.get('ts_noise'),
            'suppress_silence': asr_params.get('suppress_silence', True),
            'suppress_word_ts': asr_params.get('suppress_word_ts', True),
            'suppress_attention': asr_params.get('suppress_attention', False),
            'use_word_position': asr_params.get('use_word_position', True),
            'q_levels': asr_params.get('q_levels', 20),
            'k_size': asr_params.get('k_size', 5),
            'time_scale': asr_params.get('time_scale'),
            'denoiser': asr_params.get('denoiser'),
            'denoiser_options': asr_params.get('denoiser_options'),
            'demucs': asr_params.get('demucs', False),
            'demucs_options': asr_params.get('demucs_options'),
            # VAD options for stable_ts
            'vad': asr_params.get('vad', True),
            'vad_threshold': asr_params.get('vad_threshold', 0.25),
        })

    # Filter None values from all param sections
    # This is critical: passing None explicitly to backends causes errors
    # By removing None, we let library defaults apply
    filtered_decoder = _filter_none_values(decoder_params)
    filtered_provider = _filter_none_values(provider_params)
    filtered_vad = _filter_none_values(config['params']['vad'])
    filtered_features = _filter_none_values(config['features'])

    return {
        'pipeline_name': config['pipeline_name'],
        'sensitivity_name': config['sensitivity_name'],
        'workflow': workflow,
        'model': config['model'],
        'params': {
            'decoder': filtered_decoder,
            'provider': filtered_provider,
            'vad': filtered_vad,
        },
        'features': filtered_features,
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


def resolve_ensemble_config(
    asr: str,
    vad: str = "none",
    task: str = "transcribe",
    features: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve configuration for ensemble mode (direct component specification).

    This is the entry point for the Ensemble Tab GUI, allowing users to
    specify components and parameters directly without using preset pipelines.

    Args:
        asr: ASR component name (e.g., 'faster_whisper', 'stable_ts', 'openai_whisper')
        vad: VAD component name or 'none'
        task: Task type ('transcribe' or 'translate')
        features: List of feature names (e.g., ['auditok_scene_detection'])
        overrides: Parameter overrides in flat dot notation
                   (e.g., {'asr.beam_size': 10, 'vad.threshold': 0.25})

    Returns:
        Resolved configuration in legacy structure for pipeline compatibility.
    """
    # Convert flat overrides to nested structure
    nested_overrides = None
    if overrides:
        nested_overrides = {}
        for key, value in overrides.items():
            parts = key.split('.')
            if len(parts) >= 2:
                comp_type = parts[0]
                param_path = '.'.join(parts[1:])

                if comp_type not in nested_overrides:
                    nested_overrides[comp_type] = {}

                # Handle nested params like features.scene_detection.max_duration_s
                if comp_type == 'features' and len(parts) >= 3:
                    feature_name = parts[1]
                    param_name = '.'.join(parts[2:])
                    if feature_name not in nested_overrides[comp_type]:
                        nested_overrides[comp_type][feature_name] = {}
                    nested_overrides[comp_type][feature_name][param_name] = value
                else:
                    nested_overrides[comp_type][param_path] = value

    # Resolve using v3 system
    config = resolve_config_v3(
        asr=asr,
        vad=vad,
        sensitivity='balanced',  # Ensemble doesn't use sensitivity presets
        task=task,
        features=features or [],
        overrides=nested_overrides,
    )

    # Add ensemble-specific metadata
    config['pipeline_name'] = 'ensemble'
    config['sensitivity_name'] = 'custom'

    # Build pipeline definition for mapping
    pipeline_def = {
        "asr": asr,
        "vad": vad,
        "features": features or [],
        "description": "Custom ensemble configuration",
    }

    # Map to legacy structure for pipeline compatibility
    return _map_to_legacy_structure(config, pipeline_def)


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
