"""
WhisperJAV Configuration Module
Provides configuration management for the WhisperJAV transcription system.

The current production system uses TranscriptionTuner with v4.3 config structure,
featuring None value cleaning, backend-specific type validation, and parameter optimization.
"""

# Current production tuner (supports v4.3 config structure)
from .transcription_tuner import TranscriptionTuner

# Configuration manager for general settings
from .manager import ConfigManager

# Export primary classes
__all__ = ['TranscriptionTuner', 'ConfigManager']

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