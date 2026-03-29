"""
WhisperJAV Configuration Module
Provides configuration management for the WhisperJAV transcription system.

Production config resolution uses resolve_legacy_pipeline() (config/legacy.py)
which calls resolve_config_v3() (config/resolver_v3.py) with Pydantic component
presets (config/components/). The v4 YAML system (config/v4/) serves Qwen/TF
pipelines and is the intended future for all pipelines.
"""

# Configuration manager for general settings
from .manager import ConfigManager

# Persistence for saving/loading custom configs
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
    'ConfigManager',
    # Persistence
    'save_config',
    'load_config',
    'list_configs',
    'delete_config',
    'config_exists',
    'get_config_dir',
    # Introspection
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