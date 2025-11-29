"""
Configuration Registries for WhisperJAV v4.

Provides central registries for discovering and accessing:
- Ecosystems (Transformers, Kaldi, BERT)
- Models (kotoba-whisper-v2, etc.)
- Tools (scene detection, VAD)
- Presets (conservative, balanced, aggressive)

All registries are YAML-driven: adding a new model only requires
adding a YAML file to the appropriate directory.

Usage:
    from whisperjav.config.v4.registries import (
        ModelRegistry,
        ToolRegistry,
        get_model_registry,
    )

    registry = get_model_registry()
    model_config = registry.get("kotoba-whisper-v2")
"""

from .model_registry import ModelRegistry, get_model_registry
from .tool_registry import ToolRegistry, get_tool_registry
from .ecosystem_registry import EcosystemRegistry, get_ecosystem_registry
from .preset_registry import PresetRegistry, get_preset_registry

__all__ = [
    "ModelRegistry",
    "ToolRegistry",
    "EcosystemRegistry",
    "PresetRegistry",
    "get_model_registry",
    "get_tool_registry",
    "get_ecosystem_registry",
    "get_preset_registry",
]
