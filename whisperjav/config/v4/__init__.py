"""
WhisperJAV Configuration System v4.0 - YAML-Driven Architecture.

This module provides a YAML-first configuration system designed for:
- Patchability: Update settings without code redistribution
- Extensibility: Add new models/ecosystems with YAML-only changes
- GUI Discovery: Auto-generate UI from config schemas
- Robustness: Strong validation with clear error messages

Architecture Overview:
    config/v4/
    ├── schemas/          # Pydantic models for YAML validation
    ├── loaders/          # YAML loading and merging logic
    ├── registries/       # Model, tool, and ecosystem registries
    ├── errors.py         # Custom exceptions
    └── ecosystems/       # YAML config files (source of truth)

This system is INDEPENDENT of the legacy config system (v1-v3).
No imports from legacy modules, no shared state.

Usage:
    from whisperjav.config.v4 import ConfigManager

    manager = ConfigManager()
    config = manager.get_model_config("kotoba-whisper-v2", sensitivity="balanced")
"""

__version__ = "4.0.0"
__schema_version__ = "v1"

from .manager import ConfigManager
from .errors import (
    V4ConfigError,
    SchemaValidationError,
    ModelNotFoundError,
    EcosystemNotFoundError,
    ToolNotFoundError,
    YAMLParseError,
    IncompatibleVersionError,
)

__all__ = [
    "ConfigManager",
    "V4ConfigError",
    "SchemaValidationError",
    "ModelNotFoundError",
    "EcosystemNotFoundError",
    "ToolNotFoundError",
    "YAMLParseError",
    "IncompatibleVersionError",
]
