"""
Base classes for WhisperJAV Configuration Components v3.0.

Components are self-contained modules that define ASR engines, VAD systems,
and features with their parameters, presets, and metadata.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


class ComponentBase(ABC):
    """Base class for all configuration components."""

    # === Required Metadata ===
    name: str = ""  # Unique identifier (e.g., "silero")
    display_name: str = ""  # Human-readable name (e.g., "Silero VAD")
    description: str = ""  # Brief description
    version: str = "1.0.0"

    # === Optional Metadata ===
    tags: List[str] = []  # For categorization/search
    deprecated: bool = False
    replacement: Optional[str] = None  # If deprecated, what to use instead

    # === Schema ===
    Options: Type[BaseModel] = None  # Pydantic model for parameters

    # === Presets ===
    presets: Dict[str, BaseModel] = {}  # sensitivity -> Options instance

    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """Get parameter schema for GUI/introspection."""
        if cls.Options is None:
            return {"parameters": []}

        parameters = []
        for field_name, field_info in cls.Options.model_fields.items():
            param = {
                "name": field_name,
                "type": _get_type_string(field_info.annotation),
                "description": field_info.description or "",
                "required": field_info.is_required(),
            }

            # Default value
            if field_info.default is not None:
                param["default"] = field_info.default

            # Constraints
            constraints = _extract_constraints(field_info)
            if constraints:
                param["constraints"] = constraints

            parameters.append(param)

        return {"parameters": parameters}

    @classmethod
    def get_preset(cls, sensitivity: str) -> Optional[BaseModel]:
        """Get preset for a sensitivity level."""
        return cls.presets.get(sensitivity)

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Get default values for all parameters."""
        if cls.Options is None:
            return {}

        defaults = {}
        for field_name, field_info in cls.Options.model_fields.items():
            if field_info.default is not None:
                defaults[field_name] = field_info.default
            elif field_info.default_factory is not None:
                defaults[field_name] = field_info.default_factory()

        return defaults

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert component metadata to dictionary."""
        return {
            "name": cls.name,
            "display_name": cls.display_name,
            "description": cls.description,
            "version": cls.version,
            "tags": cls.tags,
            "deprecated": cls.deprecated,
            "replacement": cls.replacement,
        }


class ASRComponent(ComponentBase):
    """Base class for ASR (Automatic Speech Recognition) components."""

    # === ASR-specific ===
    provider: str = ""  # e.g., "faster_whisper", "huggingface_transformers"
    model_id: str = ""  # Model identifier
    supported_tasks: List[str] = ["transcribe"]  # "transcribe", "translate"
    compatible_vad: List[str] = []  # List of compatible VAD component names

    # === Compute ===
    default_device: str = "cuda"
    default_compute_type: str = "float16"

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert ASR component to dictionary."""
        base = super().to_dict()
        base.update({
            "provider": cls.provider,
            "model_id": cls.model_id,
            "supported_tasks": cls.supported_tasks,
            "compatible_vad": cls.compatible_vad,
            "default_device": cls.default_device,
            "default_compute_type": cls.default_compute_type,
        })
        return base


class VADComponent(ComponentBase):
    """Base class for VAD (Voice Activity Detection) components."""

    # === VAD-specific ===
    compatible_asr: List[str] = []  # List of compatible ASR component names

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert VAD component to dictionary."""
        base = super().to_dict()
        base.update({
            "compatible_asr": cls.compatible_asr,
        })
        return base


class FeatureComponent(ComponentBase):
    """Base class for feature components (scene detection, post-processing, etc.)."""

    # === Feature-specific ===
    feature_type: str = ""  # e.g., "scene_detection", "post_processing"

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert feature component to dictionary."""
        base = super().to_dict()
        base.update({
            "feature_type": cls.feature_type,
        })
        return base


# === Component Registry ===

# Global registries
_asr_registry: Dict[str, Type[ASRComponent]] = {}
_vad_registry: Dict[str, Type[VADComponent]] = {}
_feature_registry: Dict[str, Type[FeatureComponent]] = {}


def register_asr(cls: Type[ASRComponent]) -> Type[ASRComponent]:
    """Decorator to register an ASR component."""
    if not cls.name:
        raise ValueError(f"ASR component {cls.__name__} must have a 'name' attribute")
    _asr_registry[cls.name] = cls
    return cls


def register_vad(cls: Type[VADComponent]) -> Type[VADComponent]:
    """Decorator to register a VAD component."""
    if not cls.name:
        raise ValueError(f"VAD component {cls.__name__} must have a 'name' attribute")
    _vad_registry[cls.name] = cls
    return cls


def register_feature(cls: Type[FeatureComponent]) -> Type[FeatureComponent]:
    """Decorator to register a feature component."""
    if not cls.name:
        raise ValueError(f"Feature component {cls.__name__} must have a 'name' attribute")
    _feature_registry[cls.name] = cls
    return cls


def get_asr_registry() -> Dict[str, Type[ASRComponent]]:
    """Get the ASR component registry."""
    return _asr_registry


def get_vad_registry() -> Dict[str, Type[VADComponent]]:
    """Get the VAD component registry."""
    return _vad_registry


def get_feature_registry() -> Dict[str, Type[FeatureComponent]]:
    """Get the feature component registry."""
    return _feature_registry


# === Helper Functions ===

def _get_type_string(annotation: Any) -> str:
    """Convert type annotation to readable string."""
    if annotation is None:
        return "Any"

    origin = getattr(annotation, '__origin__', None)
    if origin is not None:
        args = getattr(annotation, '__args__', ())

        # Handle Optional
        if hasattr(origin, '__name__'):
            origin_name = origin.__name__
        else:
            origin_name = str(origin)

        if 'Union' in origin_name and type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return f"Optional[{_get_type_string(non_none[0])}]"

        if args:
            args_str = ', '.join(_get_type_string(a) for a in args)
            return f"{origin_name}[{args_str}]"
        return origin_name

    if hasattr(annotation, '__name__'):
        return annotation.__name__

    return str(annotation)


def _extract_constraints(field_info) -> Dict[str, Any]:
    """Extract numeric constraints from field metadata."""
    constraints = {}

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
