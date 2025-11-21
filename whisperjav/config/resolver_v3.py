"""
Configuration Resolver v3.0 - Component-Based.

Resolves configuration from component selections instead of JSON.
"""

import logging
from typing import Any, Dict, List, Optional

from .components import (
    get_asr_registry,
    get_vad_registry,
    get_feature_registry,
)

logger = logging.getLogger("whisperjav")


def resolve_config_v3(
    asr: str,
    vad: str = "none",
    sensitivity: str = "balanced",
    task: str = "transcribe",
    features: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve configuration from component selections.

    Args:
        asr: ASR component name (e.g., 'faster_whisper', 'stable_ts')
        vad: VAD component name (e.g., 'silero', 'none')
        sensitivity: Sensitivity level ('conservative', 'balanced', 'aggressive')
        task: Task type ('transcribe', 'translate')
        features: List of feature names (e.g., ['auditok_scene_detection'])
        overrides: Parameter overrides (e.g., {'asr.beam_size': 10})

    Returns:
        Resolved configuration dictionary.

    Example:
        >>> config = resolve_config_v3(
        ...     asr='faster_whisper',
        ...     vad='silero',
        ...     sensitivity='aggressive',
        ...     task='transcribe',
        ...     features=['auditok_scene_detection'],
        ...     overrides={'asr.beam_size': 10}
        ... )
    """
    if features is None:
        features = []
    if overrides is None:
        overrides = {}

    # Get ASR component
    asr_registry = get_asr_registry()
    if asr not in asr_registry:
        available = list(asr_registry.keys())
        raise ValueError(f"Unknown ASR component: {asr}. Available: {available}")

    asr_component = asr_registry[asr]
    asr_preset = asr_component.get_preset(sensitivity)
    if asr_preset is None:
        raise ValueError(f"ASR '{asr}' has no preset for sensitivity '{sensitivity}'")

    asr_params = asr_preset.model_dump()

    # Override task
    if 'task' in asr_params:
        asr_params['task'] = task

    # Get VAD component
    vad_params = {}
    if vad and vad != "none":
        vad_registry = get_vad_registry()
        if vad not in vad_registry:
            available = list(vad_registry.keys())
            raise ValueError(f"Unknown VAD component: {vad}. Available: {available}")

        vad_component = vad_registry[vad]
        vad_preset = vad_component.get_preset(sensitivity)
        if vad_preset is None:
            raise ValueError(f"VAD '{vad}' has no preset for sensitivity '{sensitivity}'")

        vad_params = vad_preset.model_dump()

        # Check compatibility
        if asr not in vad_component.compatible_asr:
            logger.warning(
                f"VAD '{vad}' may not be compatible with ASR '{asr}'. "
                f"Compatible ASR: {vad_component.compatible_asr}"
            )

    # Get feature configurations
    feature_configs = {}
    feature_registry = get_feature_registry()
    for feature_name in features:
        if feature_name not in feature_registry:
            available = list(feature_registry.keys())
            raise ValueError(f"Unknown feature: {feature_name}. Available: {available}")

        feature_component = feature_registry[feature_name]
        feature_preset = feature_component.get_preset(sensitivity)
        if feature_preset:
            feature_configs[feature_component.feature_type] = feature_preset.model_dump()
        else:
            # Use defaults if no preset
            feature_configs[feature_component.feature_type] = feature_component.get_defaults()

    # Apply overrides
    _apply_overrides(asr_params, overrides, 'asr')
    _apply_overrides(vad_params, overrides, 'vad')
    for feature_type, feature_config in feature_configs.items():
        _apply_overrides(feature_config, overrides, feature_type)

    # Build model configuration
    model_config = {
        'provider': asr_component.provider,
        'model_name': asr_component.model_id,
        'device': asr_component.default_device,
        'compute_type': asr_component.default_compute_type,
        'supported_tasks': asr_component.supported_tasks,
    }

    # Build result
    result = {
        'asr_name': asr,
        'vad_name': vad,
        'sensitivity_name': sensitivity,
        'task': task,
        'language': asr_params.get('language', 'ja'),
        'model': model_config,
        'params': {
            'asr': asr_params,
            'vad': vad_params,
        },
        'features': feature_configs,
    }

    logger.debug(f"v3.0 resolver: Resolved config for asr='{asr}', vad='{vad}', sensitivity='{sensitivity}'")

    return result


def _apply_overrides(params: Dict[str, Any], overrides: Dict[str, Any], prefix: str) -> None:
    """Apply parameter overrides."""
    for key, value in overrides.items():
        parts = key.split('.')
        if len(parts) == 2 and parts[0] == prefix:
            param_name = parts[1]
            if param_name in params:
                params[param_name] = value
            else:
                logger.warning(f"Override key '{key}' not found in {prefix} parameters")


def get_asr_options(asr: str) -> List[Dict[str, Any]]:
    """
    Get parameter schema for an ASR component.

    Useful for GUI to build parameter panel.
    """
    asr_registry = get_asr_registry()
    if asr not in asr_registry:
        raise ValueError(f"Unknown ASR component: {asr}")

    return asr_registry[asr].get_schema()['parameters']


def get_vad_options(vad: str) -> List[Dict[str, Any]]:
    """
    Get parameter schema for a VAD component.

    Useful for GUI to build parameter panel.
    """
    vad_registry = get_vad_registry()
    if vad not in vad_registry:
        raise ValueError(f"Unknown VAD component: {vad}")

    return vad_registry[vad].get_schema()['parameters']


def get_compatible_vads(asr: str) -> List[str]:
    """
    Get list of VAD components compatible with an ASR.

    Useful for GUI to filter VAD dropdown.
    """
    asr_registry = get_asr_registry()
    if asr not in asr_registry:
        raise ValueError(f"Unknown ASR component: {asr}")

    return asr_registry[asr].compatible_vad


def list_available_asr() -> List[Dict[str, Any]]:
    """List all available ASR components with metadata."""
    return [cls.to_dict() for cls in get_asr_registry().values()]


def list_available_vad() -> List[Dict[str, Any]]:
    """List all available VAD components with metadata."""
    return [cls.to_dict() for cls in get_vad_registry().values()]


def list_available_features() -> List[Dict[str, Any]]:
    """List all available feature components with metadata."""
    return [cls.to_dict() for cls in get_feature_registry().values()]
