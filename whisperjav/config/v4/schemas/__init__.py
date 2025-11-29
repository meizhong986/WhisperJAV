"""
Pydantic Schema Definitions for WhisperJAV v4 Configuration.

These schemas define the structure of YAML configuration files.
All YAML files are validated against these schemas before use.

Schema Hierarchy:
    ConfigBase
    ├── ModelConfig      # For model YAML files
    ├── EcosystemConfig  # For ecosystem YAML files
    ├── ToolConfig       # For tool YAML files
    └── PresetConfig     # For sensitivity preset files
"""

from .base import (
    ConfigBase,
    MetadataBlock,
    GUIWidget,
    GUIHint,
)
from .model import ModelConfig, ModelSpec
from .ecosystem import EcosystemConfig
from .tool import ToolConfig, ToolContract
from .preset import PresetConfig, SensitivityLevel

__all__ = [
    # Base
    "ConfigBase",
    "MetadataBlock",
    "GUIWidget",
    "GUIHint",
    # Model
    "ModelConfig",
    "ModelSpec",
    # Ecosystem
    "EcosystemConfig",
    # Tool
    "ToolConfig",
    "ToolContract",
    # Preset
    "PresetConfig",
    "SensitivityLevel",
]
