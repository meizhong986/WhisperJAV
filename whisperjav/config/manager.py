#!/usr/bin/env python3
"""Configuration Manager for WhisperJAV V3.

This module provides a high-level API for safely reading and modifying
the asr_config.v3.json file, serving as the backend for future UI development.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import threading
from copy import deepcopy
import jsonschema
from dataclasses import dataclass

from whisperjav.utils.logger import logger


@dataclass
class ConfigValidationError(Exception):
    """Configuration validation error with details."""
    field: str
    message: str
    value: Any = None


class ConfigManager:
    """
    Manages WhisperJAV configuration with validation and safe updates.
    
    This class provides a stable API for configuration manipulation,
    abstracting away the JSON structure and ensuring data integrity.
    """
    
    # Configuration schema for validation
    CONFIG_SCHEMA = {
        "type": "object",
        "required": ["version", "models", "parameter_sets", "sensitivity_profiles", "pipelines"],
        "properties": {
            "version": {"type": "string", "pattern": "^\\d+\\.\\d+$"},
            "models": {"type": "object"},
            "parameter_sets": {
                "type": "object",
                "properties": {
                    "decoder_params": {"type": "object"},
                    "vad_params": {"type": "object"},
                    "provider_specific_params": {"type": "object"}
                }
            },
            "sensitivity_profiles": {"type": "object"},
            "pipelines": {"type": "object"},
            "feature_configs": {"type": "object"},
            "defaults": {"type": "object"},
            "ui_preferences": {"type": "object"}  # Added for UI settings
        }
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default to package configuration
            self.config_path = Path(__file__).parent / "asr_config.v3.json"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuration cache
        self._config: Dict[str, Any] = {}
        self._original_config: Dict[str, Any] = {}
        
        # Track modifications
        self._modified = False
        
        # Load initial configuration
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load or reload configuration from file.
        
        Returns:
            The complete configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
            ConfigValidationError: If config doesn't match schema
        """
        with self._lock:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    # Filter out comment lines
                    config_text = "".join(
                        line for line in f if not line.strip().startswith("//")
                    )
                    self._config = json.loads(config_text)
                    
                # Validate configuration
                self._validate_config(self._config)
                
                # Store original for comparison
                self._original_config = deepcopy(self._config)
                self._modified = False
                
                # Ensure UI preferences exist
                if 'ui_preferences' not in self._config:
                    self._config['ui_preferences'] = self._get_default_ui_preferences()
                
                logger.debug(f"Loaded configuration from {self.config_path}")
                return deepcopy(self._config)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in configuration file: {e}")
                raise
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                raise
    
    def save_config(self, backup: bool = True) -> None:
        """
        Save current configuration to file.
        
        Args:
            backup: If True, create backup before overwriting
            
        Raises:
            ConfigValidationError: If current config is invalid
            IOError: If file write fails
        """
        with self._lock:
            if not self._modified:
                logger.debug("No changes to save")
                return
            
            # Validate before saving
            self._validate_config(self._config)
            
            # Create backup if requested
            if backup and self.config_path.exists():
                backup_path = self.config_path.with_suffix(
                    f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                )
                try:
                    shutil.copy2(self.config_path, backup_path)
                    logger.info(f"Created backup: {backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to create backup: {e}")
            
            # Write configuration
            try:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
                
                self._original_config = deepcopy(self._config)
                self._modified = False
                logger.info(f"Saved configuration to {self.config_path}")
                
            except Exception as e:
                logger.error(f"Failed to save configuration: {e}")
                raise IOError(f"Failed to save configuration: {e}")
    
    def get_sensitivity_profiles(self) -> Dict[str, Dict[str, str]]:
        """
        Get all defined sensitivity profiles.
        
        Returns:
            Dictionary mapping profile names to their settings
        """
        with self._lock:
            return deepcopy(self._config.get('sensitivity_profiles', {}))
    
    def get_parameter_set(self, param_type: str, name: str) -> Dict[str, Any]:
        """
        Get a specific parameter set.
        
        Args:
            param_type: Type of parameters ('decoder_params', 'vad_params', etc.)
            name: Name of the parameter set ('conservative', 'balanced', etc.)
            
        Returns:
            The requested parameter set
            
        Raises:
            KeyError: If param_type or name not found
        """
        with self._lock:
            if param_type not in self._config.get('parameter_sets', {}):
                raise KeyError(f"Unknown parameter type: {param_type}")
            
            param_sets = self._config['parameter_sets'][param_type]
            if name not in param_sets:
                raise KeyError(f"Unknown {param_type} set: {name}")
            
            return deepcopy(param_sets[name])
    
    def update_parameter_set(self, param_type: str, name: str, new_settings: Dict[str, Any]) -> None:
        """
        Update or create a parameter set.
        
        Args:
            param_type: Type of parameters
            name: Name of the parameter set
            new_settings: New settings dictionary
            
        Raises:
            ConfigValidationError: If settings are invalid
        """
        with self._lock:
            # Ensure parameter_sets structure exists
            if 'parameter_sets' not in self._config:
                self._config['parameter_sets'] = {}
            if param_type not in self._config['parameter_sets']:
                self._config['parameter_sets'][param_type] = {}
            
            # Validate parameter values
            self._validate_parameters(param_type, new_settings)
            
            # Update settings
            self._config['parameter_sets'][param_type][name] = deepcopy(new_settings)
            self._modified = True
            
            logger.debug(f"Updated {param_type}.{name}")
    
    def add_sensitivity_profile(self, name: str, settings: Dict[str, str]) -> None:
        """
        Add a new sensitivity profile.
        
        Args:
            name: Profile name
            settings: Profile settings mapping parameter groups to set names
                     e.g., {'decoder': 'aggressive', 'vad': 'balanced', ...}
            
        Raises:
            ConfigValidationError: If profile already exists or settings invalid
        """
        with self._lock:
            if name in self._config.get('sensitivity_profiles', {}):
                raise ConfigValidationError('profile', f"Profile '{name}' already exists")
            
            # Validate settings reference existing parameter sets
            self._validate_profile_settings(settings)
            
            # Add profile
            if 'sensitivity_profiles' not in self._config:
                self._config['sensitivity_profiles'] = {}
            
            self._config['sensitivity_profiles'][name] = deepcopy(settings)
            self._modified = True
            
            logger.info(f"Added sensitivity profile: {name}")
    
    def remove_sensitivity_profile(self, name: str) -> None:
        """
        Remove a sensitivity profile.
        
        Args:
            name: Profile name to remove
            
        Raises:
            KeyError: If profile doesn't exist
            ValueError: If trying to remove a built-in profile
        """
        with self._lock:
            # Prevent removal of built-in profiles
            built_in_profiles = {'conservative', 'balanced', 'aggressive'}
            if name in built_in_profiles:
                raise ValueError(f"Cannot remove built-in profile: {name}")
            
            if name not in self._config.get('sensitivity_profiles', {}):
                raise KeyError(f"Profile not found: {name}")
            
            del self._config['sensitivity_profiles'][name]
            self._modified = True
            
            logger.info(f"Removed sensitivity profile: {name}")
    
    def get_ui_preferences(self) -> Dict[str, Any]:
        """
        Get UI-specific preferences.
        
        Returns:
            UI preferences dictionary
        """
        with self._lock:
            return deepcopy(self._config.get('ui_preferences', self._get_default_ui_preferences()))
    
    def update_ui_preference(self, key: str, value: Any) -> None:
        """
        Update a specific UI preference.
        
        Args:
            key: Preference key
            value: New value
        """
        with self._lock:
            if 'ui_preferences' not in self._config:
                self._config['ui_preferences'] = self._get_default_ui_preferences()
            
            self._config['ui_preferences'][key] = value
            self._modified = True
            
            logger.debug(f"Updated UI preference: {key} = {value}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available ASR models."""
        with self._lock:
            return list(self._config.get('models', {}).keys())
    
    def get_available_pipelines(self) -> List[str]:
        """Get list of available pipelines."""
        with self._lock:
            return list(self._config.get('pipelines', {}).keys())
    
    def export_profile(self, profile_name: str, output_path: Path) -> None:
        """
        Export a sensitivity profile to a separate file.
        
        Args:
            profile_name: Name of profile to export
            output_path: Path to save exported profile
        """
        with self._lock:
            if profile_name not in self._config.get('sensitivity_profiles', {}):
                raise KeyError(f"Profile not found: {profile_name}")
            
            profile_data = {
                'version': '1.0',
                'profile_name': profile_name,
                'settings': self._config['sensitivity_profiles'][profile_name],
                'parameter_sets': {}
            }
            
            # Include referenced parameter sets
            for param_type, set_name in profile_data['settings'].items():
                if param_type in ['decoder', 'vad', 'provider_settings']:
                    mapped_type = {
                        'decoder': 'decoder_params',
                        'vad': 'vad_params',
                        'provider_settings': 'provider_specific_params'
                    }.get(param_type, param_type)
                    
                    try:
                        params = self.get_parameter_set(mapped_type, set_name)
                        if mapped_type not in profile_data['parameter_sets']:
                            profile_data['parameter_sets'][mapped_type] = {}
                        profile_data['parameter_sets'][mapped_type][set_name] = params
                    except KeyError:
                        logger.warning(f"Parameter set not found: {mapped_type}.{set_name}")
            
            # Write export file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported profile '{profile_name}' to {output_path}")
    
    def import_profile(self, import_path: Path, profile_name: Optional[str] = None) -> str:
        """
        Import a sensitivity profile from file.
        
        Args:
            import_path: Path to profile file
            profile_name: Optional new name for profile
            
        Returns:
            Name of imported profile
        """
        with self._lock:
            with open(import_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
            
            # Validate format
            if 'settings' not in profile_data:
                raise ValueError("Invalid profile format: missing 'settings'")
            
            # Use provided name or original
            name = profile_name or profile_data.get('profile_name', import_path.stem)
            
            # Ensure unique name
            base_name = name
            counter = 1
            while name in self._config.get('sensitivity_profiles', {}):
                name = f"{base_name}_{counter}"
                counter += 1
            
            # Import parameter sets if included
            if 'parameter_sets' in profile_data:
                for param_type, sets in profile_data['parameter_sets'].items():
                    for set_name, params in sets.items():
                        try:
                            self.update_parameter_set(param_type, set_name, params)
                        except Exception as e:
                            logger.warning(f"Failed to import {param_type}.{set_name}: {e}")
            
            # Add profile
            self.add_sensitivity_profile(name, profile_data['settings'])
            
            logger.info(f"Imported profile as '{name}' from {import_path}")
            return name
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to original state."""
        with self._lock:
            self._config = deepcopy(self._original_config)
            self._modified = False
            logger.info("Reset configuration to defaults")
    
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        with self._lock:
            return self._modified
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration against schema."""
        try:
            jsonschema.validate(config, self.CONFIG_SCHEMA)
        except jsonschema.ValidationError as e:
            raise ConfigValidationError('config', f"Schema validation failed: {e.message}")
        
        # Additional semantic validation
        if config.get('version') != '3.0':
            raise ConfigValidationError('version', f"Unsupported version: {config.get('version')}")
    
    def _validate_parameters(self, param_type: str, params: Dict[str, Any]) -> None:
        """Validate parameter values based on type."""
        # Parameter-specific validation rules
        validation_rules = {
            'decoder_params': {
                'beam_size': (1, 10, int),
                'temperature': (0.0, 2.0, (float, list)),
                'length_penalty': (0.0, 2.0, float),
                'patience': (0.1, 10.0, float)
            },
            'vad_params': {
                'threshold': (0.0, 1.0, float),
                'min_speech_duration_ms': (0, 5000, int),
                'chunk_threshold': (0.1, 10.0, float)
            }
        }
        
        if param_type in validation_rules:
            rules = validation_rules[param_type]
            for param_name, value in params.items():
                if param_name in rules:
                    min_val, max_val, expected_type = rules[param_name]
                    
                    # Type check
                    if not isinstance(value, expected_type):
                        raise ConfigValidationError(
                            param_name,
                            f"Expected {expected_type.__name__}, got {type(value).__name__}",
                            value
                        )
                    
                    # Range check for numbers
                    if isinstance(value, (int, float)):
                        if value < min_val or value > max_val:
                            raise ConfigValidationError(
                                param_name,
                                f"Value {value} outside range [{min_val}, {max_val}]",
                                value
                            )
    
    def _validate_profile_settings(self, settings: Dict[str, str]) -> None:
        """Validate that profile settings reference existing parameter sets."""
        required_keys = {'decoder', 'vad', 'provider_settings'}
        
        # Check required keys
        missing = required_keys - set(settings.keys())
        if missing:
            raise ConfigValidationError(
                'settings',
                f"Missing required keys: {missing}"
            )
        
        # Check referenced sets exist
        mappings = {
            'decoder': 'decoder_params',
            'vad': 'vad_params',
            'provider_settings': 'provider_specific_params'
        }
        
        for key, set_name in settings.items():
            if key in mappings:
                param_type = mappings[key]
                available = self._config.get('parameter_sets', {}).get(param_type, {}).keys()
                if set_name not in available:
                    raise ConfigValidationError(
                        key,
                        f"Unknown {param_type} set: {set_name}. Available: {list(available)}"
                    )
    
    def _get_default_ui_preferences(self) -> Dict[str, Any]:
        """Get default UI preferences."""
        return {
            'console_verbosity': 'summary',
            'progress_batch_size': 10,
            'show_scene_details': False,
            'max_console_lines': 1000,
            'auto_scroll': True,
            'show_timestamps': False,
            'theme': 'default'
        }


# Convenience functions for common operations
def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """Get a configuration manager instance."""
    return ConfigManager(config_path)


def quick_update_ui_preference(key: str, value: Any, config_path: Optional[Path] = None) -> None:
    """Quickly update a UI preference."""
    manager = ConfigManager(config_path)
    manager.update_ui_preference(key, value)
    manager.save_config()