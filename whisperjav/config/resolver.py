"""
Configuration Resolver for WhisperJAV Configuration System v2.0.

Main entry point: resolve_config() produces output identical to
TranscriptionTuner.resolve_params() for backward compatibility.
"""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from .errors import ConfigValidationError, UnknownComponentError

logger = logging.getLogger("whisperjav")
from .schemas import (
    DECODER_PRESETS,
    FASTER_WHISPER_ENGINE_PRESETS,
    HALLUCINATION_THRESHOLDS,
    MODELS,
    SILERO_VAD_PRESETS,
    STABLE_TS_ENGINE_OPTIONS,
    STABLE_TS_VAD_PRESETS,
    TRANSCRIBER_PRESETS,
    Sensitivity,
)


# Backend-specific type conversions
INTEGER_PARAMS_STABLE_TS = {
    'beam_size', 'best_of', 'batch_size', 'max_new_tokens',
    'language_detection_segments', 'ts_num', 'q_levels', 'k_size',
    'no_repeat_ngram_size',
}

FLOAT_PARAMS = {
    'patience', 'length_penalty', 'repetition_penalty',
    'compression_ratio_threshold', 'logprob_threshold', 'logprob_margin',
    'no_speech_threshold', 'vad_threshold', 'max_instant_words',
    'hallucination_silence_threshold', 'nonspeech_error',
}


def resolve_config(
    pipeline_name: str,
    sensitivity: str = "balanced",
    task: str = "transcribe",
    config_path: Path = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Resolve configuration for a pipeline run.

    This is the main entry point for the v2.0 configuration system.
    Output structure matches TranscriptionTuner.resolve_params() exactly.

    Args:
        pipeline_name: Pipeline name ('faster', 'fast', 'fidelity', 'balanced')
        sensitivity: Sensitivity level ('conservative', 'balanced', 'aggressive')
        task: Task to perform ('transcribe', 'translate')
        config_path: Optional path to asr_config.json
        **kwargs: Additional options (e.g., scene_detection_method)

    Returns:
        Resolved configuration dictionary matching pipeline expectations.
    """
    # Load JSON config for pipeline definitions and features
    config = _load_config(config_path)

    # Validate inputs
    _validate_inputs(pipeline_name, sensitivity, task, config)

    # Get sensitivity enum
    sens = Sensitivity(sensitivity)

    # Get pipeline configuration
    pipeline_cfg = config['pipelines'][pipeline_name]
    pipeline_map = config['pipeline_parameter_map'][pipeline_name]
    workflow = deepcopy(pipeline_cfg['workflow'])

    # Build parameters
    decoder_params = _build_decoder_params(sens, task)
    provider_params = _build_provider_params(
        pipeline_name, sens, pipeline_map, config
    )
    vad_params = _build_vad_params(sens, pipeline_map, workflow, config)

    # Handle model selection (with task-based override)
    model_cfg = _resolve_model(pipeline_cfg, workflow, task, config)

    # Extract features
    features = _extract_features(workflow, config, kwargs)

    # Get backend for type conversion
    backend = workflow.get('backend', 'whisper')

    # Remove None values and apply type conversions
    decoder_params = _remove_none_values(decoder_params)
    provider_params = _remove_none_values(provider_params)
    vad_params = _remove_none_values(vad_params)
    features = _remove_none_values(features)

    decoder_params = _convert_types(decoder_params, backend)
    provider_params = _convert_types(provider_params, backend)

    # Build return structure
    return {
        'pipeline_name': pipeline_name,
        'sensitivity_name': sensitivity,
        'workflow': workflow,
        'model': model_cfg,
        'params': {
            'decoder': decoder_params,
            'provider': provider_params,
            'vad': vad_params,
        },
        'features': features,
        'task': task,
        'language': decoder_params.get('language', 'ja')
    }


def _load_config(config_path: Path = None) -> Dict[str, Any]:
    """Load asr_config.json."""
    if not config_path:
        config_path = Path(__file__).parent / "asr_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"ASR configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    logger.debug(f"v2.0 resolver: Loaded config from {config_path}")
    return config


def _validate_inputs(
    pipeline_name: str,
    sensitivity: str,
    task: str,
    config: Dict[str, Any]
) -> None:
    """Validate input parameters."""
    # Validate pipeline
    valid_pipelines = list(config.get('pipeline_parameter_map', {}).keys())
    if pipeline_name not in valid_pipelines:
        raise UnknownComponentError("pipeline", pipeline_name, valid_pipelines)

    # Validate sensitivity
    try:
        Sensitivity(sensitivity)
    except ValueError:
        valid_sens = [s.value for s in Sensitivity]
        raise ConfigValidationError(
            f"Invalid sensitivity. Must be one of: {valid_sens}",
            field="sensitivity",
            value=sensitivity
        )

    # Validate task
    if task not in ("transcribe", "translate"):
        raise ConfigValidationError(
            "Task must be 'transcribe' or 'translate'",
            field="task",
            value=task
        )


def _build_decoder_params(sens: Sensitivity, task: str) -> Dict[str, Any]:
    """Build decoder parameters from presets."""
    preset = DECODER_PRESETS[sens]
    params = preset.model_dump()
    params['task'] = task  # Override task
    return params


def _build_provider_params(
    pipeline_name: str,
    sens: Sensitivity,
    pipeline_map: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Build provider parameters from presets and config."""
    provider_params = {}

    # Add transcriber options
    transcriber = TRANSCRIBER_PRESETS[sens]
    provider_params.update(transcriber.model_dump())

    # Add engine-specific options based on pipeline
    engine_sections = pipeline_map.get('engine', [])
    if isinstance(engine_sections, str):
        engine_sections = [engine_sections]

    for section_name in engine_sections:
        if 'faster_whisper_engine' in section_name:
            engine = FASTER_WHISPER_ENGINE_PRESETS[sens]
            provider_params.update(engine.model_dump())
        elif 'stable_ts_engine' in section_name:
            # Stable-TS options are same across sensitivities
            provider_params.update(STABLE_TS_ENGINE_OPTIONS.model_dump())
        elif 'openai_whisper_engine' in section_name:
            # OpenAI Whisper engine options from JSON
            if section_name in config and sens.value in config[section_name]:
                provider_params.update(config[section_name][sens.value])
        elif 'fast_pipeline_overrides' in section_name:
            # Fast pipeline overrides from JSON
            if section_name in config and sens.value in config[section_name]:
                provider_params.update(config[section_name][sens.value])

    # Add exclusive_whisper_plus_faster if in common sections
    common_sections = pipeline_map.get('common', [])
    if isinstance(common_sections, str):
        common_sections = [common_sections]

    for section_name in common_sections:
        if 'exclusive_whisper_plus_faster' in section_name:
            provider_params['hallucination_silence_threshold'] = HALLUCINATION_THRESHOLDS[sens]

    return provider_params


def _build_vad_params(
    sens: Sensitivity,
    pipeline_map: Dict[str, Any],
    workflow: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Build VAD parameters based on pipeline configuration."""
    vad_params = {}

    vad_section = pipeline_map.get('vad')
    vad_handling = pipeline_map.get('vad_handling', 'packed')

    if not vad_section:
        return vad_params

    # Get VAD options from presets
    if 'silero_vad' in vad_section:
        preset = SILERO_VAD_PRESETS[sens]
        vad_data = preset.model_dump()
    elif 'stable_ts_vad' in vad_section:
        preset = STABLE_TS_VAD_PRESETS[sens]
        vad_data = preset.model_dump()
    else:
        # Load from JSON for other VAD types
        if vad_section in config and sens.value in config[vad_section]:
            vad_data = deepcopy(config[vad_section][sens.value])
        else:
            vad_data = {}

    # Resolve VAD engine repo URL
    vad_engine_id = workflow.get('vad')
    if vad_engine_id and vad_engine_id != 'none':
        if 'vad_engines' in config and vad_engine_id in config['vad_engines']:
            vad_engine_config = config['vad_engines'][vad_engine_id]
            vad_data['vad_repo'] = vad_engine_config['repo']
            vad_data['vad_provider'] = vad_engine_config['provider']

    # Return based on handling type
    if vad_handling == 'packed':
        return {}  # VAD params were merged into provider
    else:
        return vad_data


def _resolve_model(
    pipeline_cfg: Dict[str, Any],
    workflow: Dict[str, Any],
    task: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve model configuration with task-based override."""
    model_overrides = pipeline_cfg.get('model_overrides', {})
    model_id = model_overrides.get(task, workflow['model'])

    if model_id in MODELS:
        # Use predefined model
        model = MODELS[model_id]
        return model.model_dump()
    elif model_id in config.get('models', {}):
        # Fall back to JSON config
        return deepcopy(config['models'][model_id])
    else:
        raise ConfigValidationError(
            f"Model not found: {model_id}",
            field="model",
            value=model_id
        )


def _extract_features(
    workflow: Dict[str, Any],
    config: Dict[str, Any],
    kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract feature configurations from workflow."""
    features = {}

    if 'features' not in workflow:
        return features

    for feature_name, feature_config in workflow['features'].items():
        if not feature_config or feature_config == 'none':
            continue

        if feature_name not in config.get('feature_configs', {}):
            continue

        if feature_name == 'scene_detection':
            # Handle scene detection method selection
            method = kwargs.get('scene_detection_method')
            if method is None:
                method = config['feature_configs']['scene_detection'].get(
                    'default_method', 'auditok'
                )

            if method in config['feature_configs']['scene_detection']:
                features[feature_name] = deepcopy(
                    config['feature_configs']['scene_detection'][method]
                )
                features[feature_name]['method'] = method
            else:
                # Fallback to auditok
                features[feature_name] = deepcopy(
                    config['feature_configs']['scene_detection']['auditok']
                )
                features[feature_name]['method'] = 'auditok'
        else:
            # Normal feature loading
            config_key = feature_config if isinstance(feature_config, str) else 'default'
            if config_key in config['feature_configs'][feature_name]:
                features[feature_name] = deepcopy(
                    config['feature_configs'][feature_name][config_key]
                )

    return features


def _remove_none_values(data: Any) -> Any:
    """Recursively remove None values from dictionary."""
    if not isinstance(data, dict):
        return data

    cleaned = {}
    for key, value in data.items():
        if value is None:
            continue
        elif isinstance(value, dict):
            cleaned_nested = _remove_none_values(value)
            if cleaned_nested:
                cleaned[key] = cleaned_nested
        elif isinstance(value, list):
            cleaned[key] = [v for v in value if v is not None]
        else:
            cleaned[key] = value

    return cleaned


def _convert_types(params: Dict[str, Any], backend: str) -> Dict[str, Any]:
    """Apply backend-specific type conversions."""
    if not isinstance(params, dict):
        return params

    result = params.copy()

    if backend == 'stable-ts':
        # Convert integer params
        for param in INTEGER_PARAMS_STABLE_TS:
            if param in result and result[param] is not None:
                result[param] = int(result[param])

        # Handle suppress_tokens specially
        if 'suppress_tokens' in result:
            val = result['suppress_tokens']
            if val == -1 or val == "-1":
                del result['suppress_tokens']
            elif isinstance(val, (int, str)) and val != -1:
                try:
                    result['suppress_tokens'] = [int(val)]
                except (ValueError, TypeError):
                    del result['suppress_tokens']

    # Convert float params
    for param in FLOAT_PARAMS:
        if param in result and result[param] is not None:
            result[param] = float(result[param])

    return result
