#!/usr/bin/env python3
"""Configuration Manager for WhisperJAV v4.3.
This module provides a high-level API for safely reading and modifying
the asr_config.json file (v4.3 structure).
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
    Manages WhisperJAV v4.3 configuration with validation and safe updates.
    
    This class provides a stable API for configuration manipulation,
    working with the current v4.3 config structure.
    """
    
    # Configuration schema for v4.3 validation
    CONFIG_SCHEMA = {
        "type": "object",
        "required": ["version", "pipeline_parameter_map", "models", "pipelines", "common_transcriber_options", "common_decoder_options"],
        "properties": {
            "version": {"type": "string", "pattern": "^4\\.\\d+$"},
            "pipeline_parameter_map": {"type": "object"},
            "models": {"type": "object"},
            "pipelines": {"type": "object"},
            "common_transcriber_options": {"type": "object"},
            "common_decoder_options": {"type": "object"},
            "feature_configs": {"type": "object"},
            "defaults": {"type": "object"},
            "ui_preferences": {"type": "object"}
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
            # FIX: Updated to use current v4.3 config file
            self.config_path = Path(__file__).parent / "asr_config.json"
        
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
            ConfigValidationError: If config doesn't match expected structure
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
                
                logger.debug(f"Loaded v4.3 configuration from {self.config_path}")
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
    
    def get_sensitivity_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all defined sensitivity profiles from v4.3 structure.
        
        Returns:
            Dictionary mapping profile names to their aggregated settings
        """
        with self._lock:
            profiles = {}
            # In v4.3, sensitivity profiles are keys in parameter sections
            profile_names = list(self._config.get('common_decoder_options', {}).keys())
            
            for profile_name in profile_names:
                profiles[profile_name] = {
                    'decoder': profile_name,
                    'transcriber': profile_name,
                    'vad': profile_name,
                    'engine_options': profile_name
                }
            
            return profiles
    
    def get_parameter_set(self, param_type: str, name: str) -> Dict[str, Any]:
        """
        Get a specific parameter set from v4.3 config structure.
        
        Args:
            param_type: Type of parameters (maps to v4.3 sections)
            name: Name of the parameter set ('conservative', 'balanced', 'aggressive')
            
        Returns:
            The requested parameter set
            
        Raises:
            KeyError: If param_type or name not found
        """
        with self._lock:
            # Map v3 param types to v4.3 sections
            param_type_mapping = {
                'decoder_params': 'common_decoder_options',
                'transcriber_params': 'common_transcriber_options', 
                'vad_params': 'silero_vad_options',
                'engine_params': 'openai_whisper_engine_options',
                'faster_whisper_params': 'faster_whisper_engine_options',
                'stable_ts_params': 'stable_ts_engine_options'
            }
            
            v43_section = param_type_mapping.get(param_type, param_type)
            
            if v43_section not in self._config:
                raise KeyError(f"Unknown parameter type: {param_type} (maps to {v43_section})")
            
            param_sets = self._config[v43_section]
            if name not in param_sets:
                raise KeyError(f"Unknown {param_type} set: {name}")
            
            return deepcopy(param_sets[name])
    
    def update_parameter_set(self, param_type: str, name: str, new_settings: Dict[str, Any]) -> None:
        """
        Update or create a parameter set in v4.3 structure.
        
        Args:
            param_type: Type of parameters
            name: Name of the parameter set
            new_settings: New settings dictionary
            
        Raises:
            ConfigValidationError: If settings are invalid
        """
        with self._lock:
            # Map to v4.3 section
            param_type_mapping = {
                'decoder_params': 'common_decoder_options',
                'transcriber_params': 'common_transcriber_options',
                'vad_params': 'silero_vad_options',
                'engine_params': 'openai_whisper_engine_options',
                'faster_whisper_params': 'faster_whisper_engine_options',
                'stable_ts_params': 'stable_ts_engine_options'
            }
            
            v43_section = param_type_mapping.get(param_type, param_type)
            
            # Ensure section exists
            if v43_section not in self._config:
                self._config[v43_section] = {}
            
            # Validate parameter values
            self._validate_parameters(param_type, new_settings)
            
            # Update settings
            self._config[v43_section][name] = deepcopy(new_settings)
            self._modified = True
            
            logger.debug(f"Updated {v43_section}.{name}")
    
    def add_sensitivity_profile(self, name: str, settings: Dict[str, str]) -> None:
        """
        Add a new sensitivity profile by creating parameter sets across sections.
        
        Args:
            name: Profile name
            settings: Profile settings mapping parameter groups to configurations
            
        Raises:
            ConfigValidationError: If profile already exists or settings invalid
        """
        with self._lock:
            existing_profiles = list(self._config.get('common_decoder_options', {}).keys())
            if name in existing_profiles:
                raise ConfigValidationError('profile', f"Profile '{name}' already exists")
            
            # Create parameter sets in relevant v4.3 sections
            sections_to_update = [
                'common_decoder_options',
                'common_transcriber_options', 
                'silero_vad_options',
                'openai_whisper_engine_options',
                'faster_whisper_engine_options',
                'stable_ts_engine_options'
            ]
            
            # Use 'balanced' as template for new profile
            for section in sections_to_update:
                if section in self._config and 'balanced' in self._config[section]:
                    self._config[section][name] = deepcopy(self._config[section]['balanced'])
            
            self._modified = True
            logger.info(f"Added sensitivity profile: {name}")
    
    def remove_sensitivity_profile(self, name: str) -> None:
        """
        Remove a sensitivity profile from all v4.3 sections.
        
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
            
            existing_profiles = list(self._config.get('common_decoder_options', {}).keys())
            if name not in existing_profiles:
                raise KeyError(f"Profile not found: {name}")
            
            # Remove from all sections
            sections_with_profiles = [
                'common_decoder_options',
                'common_transcriber_options',
                'silero_vad_options', 
                'openai_whisper_engine_options',
                'faster_whisper_engine_options',
                'stable_ts_engine_options'
            ]
            
            for section in sections_with_profiles:
                if section in self._config and name in self._config[section]:
                    del self._config[section][name]
            
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
    
    def update_ui_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Update UI preferences.
        
        Args:
            preferences: New preferences dictionary
        """
        with self._lock:
            if 'ui_preferences' not in self._config:
                self._config['ui_preferences'] = self._get_default_ui_preferences()
            
            self._config['ui_preferences'].update(preferences)
            self._modified = True
            
            logger.debug(f"Updated UI preferences")
    
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
            existing_profiles = list(self._config.get('common_decoder_options', {}).keys())
            if profile_name not in existing_profiles:
                raise KeyError(f"Profile not found: {profile_name}")
            
            profile_data = {
                'version': '4.3',
                'profile_name': profile_name,
                'parameter_sets': {}
            }
            
            # Export parameter sets from all relevant v4.3 sections
            sections_to_export = [
                'common_decoder_options',
                'common_transcriber_options',
                'silero_vad_options',
                'openai_whisper_engine_options', 
                'faster_whisper_engine_options',
                'stable_ts_engine_options'
            ]
            
            for section in sections_to_export:
                if section in self._config and profile_name in self._config[section]:
                    profile_data['parameter_sets'][section] = {
                        profile_name: self._config[section][profile_name]
                    }
            
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
            if 'parameter_sets' not in profile_data:
                raise ValueError("Invalid profile format: missing 'parameter_sets'")
            
            # Use provided name or original
            name = profile_name or profile_data.get('profile_name', import_path.stem)
            
            # Ensure unique name
            base_name = name
            counter = 1
            existing_profiles = list(self._config.get('common_decoder_options', {}).keys())
            while name in existing_profiles:
                name = f"{base_name}_{counter}"
                counter += 1
            
            # Import parameter sets
            for section, params in profile_data['parameter_sets'].items():
                if section in self._config:
                    for param_name, param_values in params.items():
                        self._config[section][name] = deepcopy(param_values)
            
            self._modified = True
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
        """Validate v4.3 configuration against schema."""
        try:
            jsonschema.validate(config, self.CONFIG_SCHEMA)
        except jsonschema.ValidationError as e:
            raise ConfigValidationError('config', f"Schema validation failed: {e.message}")
        
        # Additional semantic validation for v4.3
        version = config.get('version', '')
        if not version.startswith('4.'):
            raise ConfigValidationError('version', f"Expected v4.x, got: {version}")
    
    def _validate_parameters(self, param_type: str, params: Dict[str, Any]) -> None:
        """Validate parameter values based on type for v4.3 structure."""
        # Parameter-specific validation rules
        validation_rules = {
            'decoder_params': {
                'beam_size': (1, 10, int),
                'temperature': (0.0, 2.0, (float, list)),
                'length_penalty': (0.0, 2.0, float), 
                'patience': (0.1, 10.0, float),
                'best_of': (1, 20, int)
            },
            'transcriber_params': {
                'temperature': (0.0, 2.0, (float, list)),
                'compression_ratio_threshold': (1.0, 10.0, float),
                'logprob_threshold': (-10.0, 0.0, float),
                'no_speech_threshold': (0.0, 1.0, float)
            },
            'vad_params': {
                'threshold': (0.0, 1.0, float),
                'min_speech_duration_ms': (0, 5000, int),
                'max_speech_duration_s': (1, 30, int),
                'min_silence_duration_ms': (0, 1000, int)
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