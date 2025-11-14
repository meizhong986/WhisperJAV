"""
TranscriptionTuner - Configuration resolver for WhisperJAV.
Implements v4.3 config structure with explicit pipeline parameter mapping.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from copy import deepcopy
from whisperjav.utils.logger import logger


class TranscriptionTuner:
    """
    Configuration resolver for WhisperJAV using v4.3 config structure.
    Reads configuration and resolves parameters for each pipeline/sensitivity combination.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initializes the tuner by loading the configuration file.
        
        Args:
            config_path: Optional path to asr_config.json.
                        If None, loads default from config directory.
        """
        self.config = self._load_config(config_path)
        self._validate_config_structure()
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Loads and validates the configuration file."""
        if not config_path:
            config_path = Path(__file__).parent / "asr_config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"ASR configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate version
        version = config.get("version", "unknown")
        if not version.startswith("4."):
            logger.warning(f"Expected v4.x config, got version: {version}")
        
        logger.debug(f"Successfully loaded config version {version}")
        return config
    
    def _validate_config_structure(self):
        """Validates that all required config sections are present."""
        required_sections = [
            'pipeline_parameter_map',
            'models',
            'pipelines',
            'common_transcriber_options',
            'common_decoder_options'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate pipeline_parameter_map structure
        for pipeline_name, pipeline_map in self.config['pipeline_parameter_map'].items():
            if 'common' not in pipeline_map:
                raise ValueError(f"Pipeline {pipeline_name} missing 'common' in parameter map")
            if 'engine' not in pipeline_map:
                raise ValueError(f"Pipeline {pipeline_name} missing 'engine' in parameter map")
    
    @staticmethod
    def _validate_and_convert_parameter_types(params: Dict[str, Any], backend: str) -> Dict[str, Any]:
        """
        Validate and convert parameter types based on the backend being used.
        
        Different backends (stable-ts, whisper, faster-whisper) have different
        type requirements for the same parameters.
        
        Args:
            params: Parameters dictionary to validate
            backend: Backend name ('stable-ts', 'whisper', etc.)
            
        Returns:
            Parameters dictionary with correct types
        """
        if not isinstance(params, dict):
            return params
            
        validated_params = params.copy()
        
        # Backend-specific type requirements
        if backend == 'stable-ts':
            type_conversions = {
                # These should be integers for stable-ts/faster-whisper (ctranslate2)
                'beam_size': int,
                'best_of': int,
                'batch_size': int,
                'max_new_tokens': int,
                'language_detection_segments': int,
                'ts_num': int,
                'q_levels': int,
                'k_size': int,
                'no_repeat_ngram_size': int,  # FIX: ctranslate2 expects int, not float!
                
                # These should be floats for stable-ts
                'patience': float,
                'length_penalty': float,
                'repetition_penalty': float,
                'compression_ratio_threshold': float,
                'logprob_threshold': float,
                'no_speech_threshold': float,
                'vad_threshold': float,
                'max_instant_words': float,
                'avg_prob_threshold': float,
                'nonspeech_error': float,
                'min_word_dur': float,
                'min_silence_dur': float,
                'hallucination_silence_threshold': float,
                'time_scale': float,
                
                # These should be strings for stable-ts
                'task': str,
                'language': str,
                'gap_padding': str,
                'prefix': str,
                
                # These should be booleans for stable-ts
                'condition_on_previous_text': bool,
                'suppress_blank': bool,
                'without_timestamps': bool,
                'word_timestamps': bool,
                'fp16': bool,
                'vad': bool,
                'suppress_ts_tokens': bool,
                'only_ffmpeg': bool,
                'log_progress': bool,
                'multilingual': bool,
                'ignore_compatibility': bool,
                'regroup': bool,
                'suppress_silence': bool,
                'suppress_word_ts': bool,
                'suppress_attention': bool,
                'use_word_position': bool,
                'only_voice_freq': bool,
                'demucs': bool,
            }
        else:  # whisper and other backends are more flexible
            type_conversions = {
                'task': str,
                'language': str,
                'fp16': bool,
                'beam_size': int,
                'best_of': int,
                'patience': float,
            }
        
        # Special handling for suppress_tokens (common issue across backends)
        if 'suppress_tokens' in validated_params:
            suppress_val = validated_params['suppress_tokens']
            
            if backend == 'stable-ts':
                # Faster-whisper (used by stable-ts in turbo mode) expects suppress_tokens to be iterable
                if suppress_val == "-1" or suppress_val == -1:
                    # Convert -1 (disable suppression) to None for faster-whisper compatibility
                    del validated_params['suppress_tokens']  # Remove entirely - let library use defaults
                    logger.debug(f"Backend {backend}: Removed suppress_tokens=-1 for faster-whisper compatibility")
                elif isinstance(suppress_val, str):
                    try:
                        int_val = int(suppress_val)
                        if int_val == -1:
                            del validated_params['suppress_tokens']  # Remove -1
                            logger.debug(f"Backend {backend}: Removed suppress_tokens='{suppress_val}' for faster-whisper compatibility")
                        else:
                            validated_params['suppress_tokens'] = [int_val]  # Convert to list
                            logger.debug(f"Backend {backend}: Converted suppress_tokens to list: {[int_val]}")
                    except ValueError:
                        logger.warning(f"Invalid suppress_tokens value: {suppress_val}, removing parameter")
                        del validated_params['suppress_tokens']
                elif isinstance(suppress_val, int):
                    if suppress_val == -1:
                        del validated_params['suppress_tokens']  # Remove -1
                        logger.debug(f"Backend {backend}: Removed suppress_tokens=-1 for faster-whisper compatibility")
                    else:
                        validated_params['suppress_tokens'] = [suppress_val]  # Convert to list
                        logger.debug(f"Backend {backend}: Converted suppress_tokens to list: {[suppress_val]}")
                elif not isinstance(suppress_val, list):
                    logger.warning(f"Invalid suppress_tokens type for {backend}: {type(suppress_val)}, removing parameter")
                    del validated_params['suppress_tokens']
            else:
                # Standard whisper backend - original handling
                if suppress_val == "-1" or suppress_val == -1:
                    validated_params['suppress_tokens'] = -1  # Keep as -1 for standard whisper
                elif isinstance(suppress_val, str):
                    try:
                        validated_params['suppress_tokens'] = int(suppress_val)
                    except ValueError:
                        logger.warning(f"Invalid suppress_tokens value: {suppress_val}, removing parameter")
                        del validated_params['suppress_tokens']
                elif not isinstance(suppress_val, (int, list)):
                    logger.warning(f"Invalid suppress_tokens type: {type(suppress_val)}, removing parameter")
                    del validated_params['suppress_tokens']
        
        # Apply type conversions
        for param_name, target_type in type_conversions.items():
            if param_name in validated_params:
                current_value = validated_params[param_name]
                
                # Skip if already correct type or None
                if current_value is None or isinstance(current_value, target_type):
                    continue
                
                try:
                    # Convert to target type
                    if target_type == bool:
                        # Special handling for boolean conversion
                        if isinstance(current_value, str):
                            validated_params[param_name] = current_value.lower() in ('true', '1', 'yes', 'on')
                        else:
                            validated_params[param_name] = bool(current_value)
                    else:
                        validated_params[param_name] = target_type(current_value)
                        
                    logger.debug(f"Backend {backend}: Converted {param_name} from {type(current_value)} to {target_type}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Backend {backend}: Failed to convert {param_name}={current_value} to {target_type}: {e}")
                    logger.warning(f"Removing invalid parameter: {param_name}")
                    del validated_params[param_name]
        
        return validated_params

    @staticmethod
    def _remove_none_values(data: Dict[str, Any], preserve_none_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Recursively remove None values from a dictionary.
        This prevents None values from being passed to libraries that don't handle them properly.
        
        In JSON configs, 'null' becomes 'None' in Python. Most ML libraries expect parameters
        to either have valid values or be omitted entirely, not passed as None.
        
        Args:
            data: Dictionary that may contain None values
            preserve_none_keys: Optional list of keys where None values should be preserved
            
        Returns:
            Dictionary with None values removed (except for preserved keys)
        """
        if not isinstance(data, dict):
            return data
            
        preserve_none_keys = preserve_none_keys or []
        cleaned = {}
        
        for key, value in data.items():
            if value is None:
                # Only preserve None if explicitly specified
                if key in preserve_none_keys:
                    cleaned[key] = value
                    logger.debug(f"Preserved None value for parameter: {key}")
                else:
                    logger.debug(f"Removed None value for parameter: {key}")
                continue  # Skip None values (unless preserved)
            elif isinstance(value, dict):
                # Recursively clean nested dictionaries
                cleaned_nested = TranscriptionTuner._remove_none_values(value, preserve_none_keys)
                if cleaned_nested:  # Only add if not empty after cleaning
                    cleaned[key] = cleaned_nested
            elif isinstance(value, list):
                # Clean lists, removing None values (but preserve empty lists)
                cleaned_list = [v for v in value if v is not None]
                cleaned[key] = cleaned_list  # Keep even if empty - empty list != omitted parameter
            else:
                cleaned[key] = value
        
        return cleaned
        """
        Recursively remove None values from a dictionary.
        This prevents None values from being passed to libraries that don't handle them properly.
        
        In JSON configs, 'null' becomes 'None' in Python. Most ML libraries expect parameters
        to either have valid values or be omitted entirely, not passed as None.
        
        Args:
            data: Dictionary that may contain None values
            preserve_none_keys: Optional list of keys where None values should be preserved
            
        Returns:
            Dictionary with None values removed (except for preserved keys)
        """
        if not isinstance(data, dict):
            return data
            
        preserve_none_keys = preserve_none_keys or []
        cleaned = {}
        
        for key, value in data.items():
            if value is None:
                # Only preserve None if explicitly specified
                if key in preserve_none_keys:
                    cleaned[key] = value
                    logger.debug(f"Preserved None value for parameter: {key}")
                else:
                    logger.debug(f"Removed None value for parameter: {key}")
                continue  # Skip None values (unless preserved)
            elif isinstance(value, dict):
                # Recursively clean nested dictionaries
                cleaned_nested = TranscriptionTuner._remove_none_values(value, preserve_none_keys)
                if cleaned_nested:  # Only add if not empty after cleaning
                    cleaned[key] = cleaned_nested
            elif isinstance(value, list):
                # Clean lists, removing None values (but preserve empty lists)
                cleaned_list = [v for v in value if v is not None]
                cleaned[key] = cleaned_list  # Keep even if empty - empty list != omitted parameter
            else:
                cleaned[key] = value
        
        return cleaned
    
    def resolve_params(self, pipeline_name: str, sensitivity: str, task: str, **kwargs) -> Dict[str, Any]:
        """
        Resolves all necessary configurations for a given run.
        
        Args:
            pipeline_name: The name of the pipeline ('faster', 'fast', 'balanced')
            sensitivity: The sensitivity profile ('conservative', 'balanced', 'aggressive')
            task: The ASR task ('transcribe' or 'translate')
            
        Returns:
            A structured dictionary matching pipeline expectations.
        """
        # Validate inputs
        if pipeline_name not in self.config['pipeline_parameter_map']:
            raise ValueError(f"Unknown pipeline specified: '{pipeline_name}'")
        
        if pipeline_name not in self.config['pipelines']:
            raise ValueError(f"Pipeline definition not found: '{pipeline_name}'")
        
        # Get pipeline configuration
        pipeline_cfg = self.config['pipelines'][pipeline_name]
        pipeline_map = self.config['pipeline_parameter_map'][pipeline_name]
        workflow = deepcopy(pipeline_cfg['workflow'])
        
        # Initialize parameter containers
        decoder_params = {}
        provider_params = {}
        vad_params = {}
        stable_ts_params = {}
        
        # 1. Process common parameters
        common_sections = pipeline_map.get('common', [])
        if isinstance(common_sections, str):
            common_sections = [common_sections]
        
        for section_name in common_sections:
            if section_name not in self.config:
                logger.warning(f"Common section '{section_name}' not found in config")
                continue
            
            if sensitivity not in self.config[section_name]:
                raise ValueError(f"Sensitivity '{sensitivity}' not found in {section_name}")
            
            section_data = deepcopy(self.config[section_name][sensitivity])
            
            if section_name == 'common_decoder_options':
                # Decoder params go to their own section
                decoder_params = section_data
            else:
                # Everything else merges into provider params
                provider_params.update(section_data)
        
        # 2. Process engine-specific parameters
        engine_sections = pipeline_map.get('engine', [])
        if isinstance(engine_sections, str):
            engine_sections = [engine_sections]
        
        for section_name in engine_sections:
            if section_name not in self.config:
                logger.warning(f"Engine section '{section_name}' not found in config")
                continue
            
            if sensitivity not in self.config[section_name]:
                logger.warning(f"Sensitivity '{sensitivity}' not found in {section_name}")
                continue
            
            section_data = deepcopy(self.config[section_name][sensitivity])
            
            # Stable-ts engine options could be kept separate for clarity
            if 'stable_ts' in section_name:
                stable_ts_params.update(section_data)
            
            # All engine params go to provider
            provider_params.update(section_data)
        
        # 3. Process VAD parameters based on handling type
        if pipeline_map.get('vad'):
            vad_section = pipeline_map['vad']
            vad_handling = pipeline_map.get('vad_handling', 'packed')

            if vad_section in self.config and sensitivity in self.config[vad_section]:
                vad_data = deepcopy(self.config[vad_section][sensitivity])

                # NEW: Resolve VAD engine repo URL from workflow
                vad_engine_id = workflow.get('vad')  # e.g., "silero-v3.1"
                if vad_engine_id and vad_engine_id != 'none':
                    if 'vad_engines' in self.config and vad_engine_id in self.config['vad_engines']:
                        vad_engine_config = self.config['vad_engines'][vad_engine_id]
                        vad_data['vad_repo'] = vad_engine_config['repo']
                        vad_data['vad_provider'] = vad_engine_config['provider']
                        logger.debug(f"Resolved VAD engine '{vad_engine_id}' to repo: {vad_engine_config['repo']}")
                    else:
                        logger.warning(f"VAD engine '{vad_engine_id}' not found in vad_engines config")

                if vad_handling == 'packed':
                    # Pack VAD params into provider (for StableTSASR)
                    provider_params.update(vad_data)
                    logger.debug(f"Packed VAD params from {vad_section} into provider params")
                else:  # 'separate'
                    # Keep VAD params separate (for WhisperProASR)
                    vad_params = vad_data
                    logger.debug(f"Kept VAD params from {vad_section} separate")
        
        # 4. Handle model selection (with task-based override)
        model_overrides = pipeline_cfg.get('model_overrides', {})
        model_id = model_overrides.get(task, workflow['model'])
        
        if model_id not in self.config['models']:
            raise ValueError(f"Model not found: {model_id}")
        
        model_cfg = deepcopy(self.config['models'][model_id])
        
        # 5. Override task in decoder params
        if decoder_params:
            decoder_params['task'] = task
        
        # 6. Extract features from workflow
        features = {}
        if 'features' in workflow:
            for feature_name, feature_config in workflow['features'].items():
                if feature_config and feature_config != 'none':
                    # Look up feature configuration
                    if feature_name in self.config.get('feature_configs', {}):
                        # Special handling for scene_detection method selection
                        if feature_name == 'scene_detection':
                            # Check if method is specified in kwargs
                            method = kwargs.get('scene_detection_method', None)
                            if method is None:
                                method = self.config['feature_configs']['scene_detection'].get('default_method', 'auditok')

                            # Load method-specific config
                            if method in self.config['feature_configs']['scene_detection']:
                                features[feature_name] = deepcopy(
                                    self.config['feature_configs']['scene_detection'][method]
                                )
                                features[feature_name]['method'] = method  # Pass method to detector
                                logger.debug(f"Loaded scene_detection config for method: {method}")
                            else:
                                logger.warning(f"Unknown scene detection method: {method}, falling back to auditok")
                                features[feature_name] = deepcopy(
                                    self.config['feature_configs']['scene_detection']['auditok']
                                )
                                features[feature_name]['method'] = 'auditok'
                        else:
                            # Normal feature loading
                            config_key = feature_config if isinstance(feature_config, str) else 'default'
                            if config_key in self.config['feature_configs'][feature_name]:
                                features[feature_name] = deepcopy(
                                    self.config['feature_configs'][feature_name][config_key]
                                )
        
        # 7. CLEAN ALL PARAMETER DICTIONARIES - Remove None values before returning
        # This prevents None values from being passed to libraries that don't handle them properly
        decoder_params = self._remove_none_values(decoder_params)
        provider_params = self._remove_none_values(provider_params)
        vad_params = self._remove_none_values(vad_params)
        stable_ts_params = self._remove_none_values(stable_ts_params)
        features = self._remove_none_values(features)
        
        # 8. APPLY BACKEND-SPECIFIC TYPE VALIDATION
        # Determine backend from workflow
        backend = workflow.get('backend', 'whisper')  # default to 'whisper'
        
        # Apply type validation based on backend
        if backend == 'stable-ts':
            decoder_params = self._validate_and_convert_parameter_types(decoder_params, 'stable-ts')
            provider_params = self._validate_and_convert_parameter_types(provider_params, 'stable-ts')
            logger.debug(f"Applied stable-ts type validation for {pipeline_name}/{sensitivity}")
        else:
            decoder_params = self._validate_and_convert_parameter_types(decoder_params, 'whisper')
            provider_params = self._validate_and_convert_parameter_types(provider_params, 'whisper')
            logger.debug(f"Applied whisper type validation for {pipeline_name}/{sensitivity}")
        
        logger.debug(f"Cleaned parameters and applied type validation for {pipeline_name}/{sensitivity}")
        
        # 9. Build the return structure expected by pipelines
        return {
            'pipeline_name': pipeline_name,
            'sensitivity_name': sensitivity,
            'workflow': workflow,
            'model': model_cfg,
            'params': {
                'decoder': decoder_params,
                'provider': provider_params,
                'vad': vad_params,
                # Optionally include stable_ts separately if needed
                # 'stable_ts': stable_ts_params
            },
            'features': features,
            'task': task,
            'language': decoder_params.get('language', 'ja')
        }
    
    def get_ui_preferences(self) -> Dict[str, Any]:
        """Returns UI preferences from config."""
        return self.config.get('ui_preferences', {})
    
    def get_defaults(self) -> Dict[str, Any]:
        """Returns default settings from config."""
        return self.config.get('defaults', {})
    
    def list_pipelines(self) -> List[str]:
        """Returns list of available pipelines."""
        return list(self.config.get('pipeline_parameter_map', {}).keys())
    
    def list_sensitivity_profiles(self) -> List[str]:
        """Returns list of available sensitivity profiles."""
        # Get from any common section since all should have same profiles
        if 'common_decoder_options' in self.config:
            return list(self.config['common_decoder_options'].keys())
        return ['conservative', 'balanced', 'aggressive']
    
    def get_pipeline_info(self, pipeline_name: str) -> Dict[str, Any]:
        """Returns information about a specific pipeline."""
        if pipeline_name not in self.config.get('pipelines', {}):
            raise ValueError(f"Pipeline not found: {pipeline_name}")
        return deepcopy(self.config['pipelines'][pipeline_name])
    
    def validate_configuration(self) -> bool:
        """
        Validates the entire configuration for consistency.
        Returns True if valid, raises exceptions with details if not.
        """
        pipelines = self.list_pipelines()
        sensitivities = self.list_sensitivity_profiles()
        
        for pipeline in pipelines:
            for sensitivity in sensitivities:
                for task in ['transcribe', 'translate']:
                    try:
                        result = self.resolve_params(pipeline, sensitivity, task)
                        # Verify required structure
                        assert 'params' in result
                        assert 'decoder' in result['params']
                        assert 'provider' in result['params']
                        assert 'vad' in result['params']
                    except Exception as e:
                        raise ValueError(
                            f"Configuration validation failed for "
                            f"{pipeline}/{sensitivity}/{task}: {e}"
                        )
        
        logger.info("Configuration validation successful")
        return True