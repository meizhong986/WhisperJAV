"""
Introspection API for WhisperJAV Configuration System v2.0.

Provides programmatic access to component parameters, constraints, and defaults.
Designed for future GUI integration and parameter discovery.
"""

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .registry import ComponentMeta, get_registry


def get_available_components() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all available components organized by type.

    Returns:
        Dictionary with keys 'vad', 'asr', 'scene_detection' containing
        lists of component information.

    Example:
        >>> components = get_available_components()
        >>> for vad in components['vad']:
        ...     print(f"{vad['name']}: {vad['display_name']}")
    """
    registry = get_registry()

    return {
        'vad': [_component_to_dict(m) for m in registry.list_vads()],
        'asr': [_component_to_dict(m) for m in registry.list_asr()],
        'scene_detection': [_component_to_dict(m) for m in registry.list_scene_detection()],
    }


def get_component_schema(component_type: str, name: str) -> Dict[str, Any]:
    """
    Get the full parameter schema for a component.

    Args:
        component_type: One of 'vad', 'asr', 'scene_detection'
        name: Component name (e.g., 'silero_vad')

    Returns:
        Dictionary with component metadata and parameter schema.

    Example:
        >>> schema = get_component_schema('vad', 'silero_vad')
        >>> for param in schema['parameters']:
        ...     print(f"{param['name']}: {param['type']} = {param['default']}")
    """
    registry = get_registry()

    # Get component metadata
    if component_type == 'vad':
        meta = registry.get_vad(name)
    elif component_type == 'asr':
        meta = registry.get_asr(name)
    elif component_type == 'scene_detection':
        meta = registry.get_scene_detection(name)
    else:
        raise ValueError(f"Unknown component type: {component_type}")

    return {
        **_component_to_dict(meta),
        'parameters': extract_parameter_schema(meta.config_class),
    }


def extract_parameter_schema(model_class: Type[BaseModel]) -> List[Dict[str, Any]]:
    """
    Extract parameter schema from a Pydantic model class.

    Returns a list of parameter dictionaries with:
    - name: Parameter name
    - type: Type annotation as string
    - default: Default value (if any)
    - description: Field description
    - constraints: Dict of constraints (ge, le, etc.)
    - required: Whether the parameter is required

    Args:
        model_class: Pydantic model class

    Returns:
        List of parameter schema dictionaries
    """
    parameters = []

    for field_name, field_info in model_class.model_fields.items():
        param_schema = _extract_field_schema(field_name, field_info)
        parameters.append(param_schema)

    return parameters


def _extract_field_schema(name: str, field_info: FieldInfo) -> Dict[str, Any]:
    """Extract schema from a single Pydantic field."""
    schema = {
        'name': name,
        'type': _get_type_string(field_info.annotation),
        'description': field_info.description or '',
        'required': field_info.is_required(),
    }

    # Default value
    if field_info.default is not None:
        schema['default'] = field_info.default
    elif field_info.default_factory is not None:
        schema['default'] = field_info.default_factory()

    # Extract constraints from metadata
    constraints = _extract_constraints(field_info)
    if constraints:
        schema['constraints'] = constraints

    return schema


def _get_type_string(annotation: Any) -> str:
    """Convert type annotation to readable string."""
    if annotation is None:
        return "Any"

    # Handle Optional types
    origin = getattr(annotation, '__origin__', None)
    if origin is not None:
        args = getattr(annotation, '__args__', ())

        # Union types (including Optional)
        if origin.__name__ == 'Union' if hasattr(origin, '__name__') else str(origin) == 'typing.Union':
            # Check for Optional (Union with None)
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1 and type(None) in args:
                return f"Optional[{_get_type_string(non_none_args[0])}]"
            return f"Union[{', '.join(_get_type_string(a) for a in args)}]"

        # List, Dict, etc.
        if args:
            args_str = ', '.join(_get_type_string(a) for a in args)
            return f"{origin.__name__}[{args_str}]"
        return origin.__name__

    # Simple types
    if hasattr(annotation, '__name__'):
        return annotation.__name__

    return str(annotation)


def _extract_constraints(field_info: FieldInfo) -> Dict[str, Any]:
    """Extract numeric constraints from field metadata."""
    constraints = {}

    # Check metadata for constraints
    for meta in field_info.metadata:
        if hasattr(meta, 'ge'):
            constraints['ge'] = meta.ge
        if hasattr(meta, 'le'):
            constraints['le'] = meta.le
        if hasattr(meta, 'gt'):
            constraints['gt'] = meta.gt
        if hasattr(meta, 'lt'):
            constraints['lt'] = meta.lt
        if hasattr(meta, 'min_length'):
            constraints['min_length'] = meta.min_length
        if hasattr(meta, 'max_length'):
            constraints['max_length'] = meta.max_length

    return constraints


def _component_to_dict(meta: ComponentMeta) -> Dict[str, Any]:
    """Convert ComponentMeta to dictionary."""
    result = {
        'name': meta.name,
        'display_name': meta.display_name,
        'description': meta.description,
        'tags': meta.tags,
        'version': meta.version,
    }

    if meta.compatible_with:
        result['compatible_with'] = meta.compatible_with

    if meta.deprecated:
        result['deprecated'] = True
        if meta.replacement:
            result['replacement'] = meta.replacement

    return result


def get_component_defaults(component_type: str, name: str) -> Dict[str, Any]:
    """
    Get default values for a component's parameters.

    Args:
        component_type: One of 'vad', 'asr', 'scene_detection'
        name: Component name

    Returns:
        Dictionary of parameter name to default value
    """
    registry = get_registry()

    # Get config class
    if component_type == 'vad':
        config_class = registry.get_vad_config_class(name)
    elif component_type == 'asr':
        config_class = registry.get_asr_config_class(name)
    elif component_type == 'scene_detection':
        config_class = registry.get_scene_detection_config_class(name)
    else:
        raise ValueError(f"Unknown component type: {component_type}")

    # Extract defaults
    defaults = {}
    for field_name, field_info in config_class.model_fields.items():
        if field_info.default is not None:
            defaults[field_name] = field_info.default
        elif field_info.default_factory is not None:
            defaults[field_name] = field_info.default_factory()

    return defaults


def list_pipelines() -> List[str]:
    """List available pipeline names."""
    return ["faster", "fast", "fidelity", "balanced"]


def list_sensitivities() -> List[str]:
    """List available sensitivity levels."""
    return ["conservative", "balanced", "aggressive"]
