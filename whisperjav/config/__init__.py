"""
WhisperJAV Configuration Module
Provides configuration management for the WhisperJAV transcription system.

The current production system uses TranscriptionTuner with v4.3 config structure,
featuring None value cleaning, backend-specific type validation, and parameter optimization.

v2.0 Configuration System:
- Pydantic-based schemas for type safety
- resolve_config() for backward-compatible resolution
- PipelineBuilder for fluent configuration building
- Persistence for saving/loading custom configs
- Introspection API for GUI integration
"""

# Current production tuner (supports v4.3 config structure)
from .transcription_tuner import TranscriptionTuner

# Configuration manager for general settings
from .manager import ConfigManager

# v2.0 Configuration System
from .resolver import resolve_config
from .builder import PipelineBuilder, quick_config
from .persistence import (
    save_config,
    load_config,
    list_configs,
    delete_config,
    config_exists,
    get_config_dir,
)
from .introspection import (
    get_available_components,
    get_component_schema,
    get_component_defaults,
    list_pipelines,
    list_sensitivities,
)
from .errors import (
    ConfigurationError,
    ConfigValidationError,
    UnknownComponentError,
    IncompatibleComponentError,
)

# Export primary classes
__all__ = [
    # Legacy (backward compatibility)
    'TranscriptionTuner',
    'ConfigManager',
    # v2.0 Core
    'resolve_config',
    'PipelineBuilder',
    'quick_config',
    # v2.0 Persistence
    'save_config',
    'load_config',
    'list_configs',
    'delete_config',
    'config_exists',
    'get_config_dir',
    # v2.0 Introspection
    'get_available_components',
    'get_component_schema',
    'get_component_defaults',
    'list_pipelines',
    'list_sensitivities',
    # Errors
    'ConfigurationError',
    'ConfigValidationError',
    'UnknownComponentError',
    'IncompatibleComponentError',
]

# For convenience, alias as default tuner
Tuner = TranscriptionTuner

# Version information for the configuration system  
CONFIG_VERSION = "4.3"

# Helper function for quick config updates
def quick_update_ui_preference(key: str, value, config_path=None):
    """
    Quick utility to update UI preferences without full config loading.
    
    Args:
        key: UI preference key to update
        value: New value to set
        config_path: Optional path to config file
    """
    from .manager import ConfigManager
    config_manager = ConfigManager(config_path)
    ui_prefs = config_manager.get_ui_preferences()
    ui_prefs[key] = value
    config_manager.update_ui_preferences(ui_prefs)