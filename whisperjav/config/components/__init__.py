"""
WhisperJAV Configuration Components v3.0.

Auto-discovers and registers all ASR, VAD, and Feature components.
"""

from .base import (
    # Base classes
    ComponentBase,
    ASRComponent,
    VADComponent,
    FeatureComponent,
    # Registration decorators
    register_asr,
    register_vad,
    register_feature,
    # Registry access
    get_asr_registry,
    get_vad_registry,
    get_feature_registry,
)

# Import subpackages to trigger auto-registration
from . import asr
from . import vad
from . import features


def get_all_components():
    """
    Get all registered components organized by type.

    Returns:
        Dictionary with 'asr', 'vad', 'features' keys containing
        lists of component metadata dictionaries.
    """
    return {
        'asr': [cls.to_dict() for cls in get_asr_registry().values()],
        'vad': [cls.to_dict() for cls in get_vad_registry().values()],
        'features': [cls.to_dict() for cls in get_feature_registry().values()],
    }


def get_component(component_type: str, name: str):
    """
    Get a specific component class by type and name.

    Args:
        component_type: 'asr', 'vad', or 'features'
        name: Component name

    Returns:
        Component class

    Raises:
        KeyError: If component not found
    """
    if component_type == 'asr':
        return get_asr_registry()[name]
    elif component_type == 'vad':
        return get_vad_registry()[name]
    elif component_type == 'features':
        return get_feature_registry()[name]
    else:
        raise ValueError(f"Unknown component type: {component_type}")


__all__ = [
    # Base classes
    'ComponentBase',
    'ASRComponent',
    'VADComponent',
    'FeatureComponent',
    # Registration
    'register_asr',
    'register_vad',
    'register_feature',
    # Registry access
    'get_asr_registry',
    'get_vad_registry',
    'get_feature_registry',
    # Convenience functions
    'get_all_components',
    'get_component',
]
