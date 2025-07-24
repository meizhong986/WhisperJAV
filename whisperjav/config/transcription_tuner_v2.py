"""
TranscriptionTunerV2 - Clean Break Implementation
Handles pipeline sensitivity control and provider-specific parameters
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from copy import deepcopy
from whisperjav.utils.logger import logger


class TranscriptionTunerV2:
    """
    Configuration resolver for WhisperJAV v2.
    Cleanly separates pipelines, models, sensitivity, and features.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self._validate_config()
        
    def resolve_params(self, 
                      pipeline: str, 
                      sensitivity: str,
                      task: str = "transcribe",
                      language: str = "ja",
                      user_overrides: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Resolve configuration for given pipeline, sensitivity, and task.
        
        Returns:
            {
                "model": { model configuration },
                "common_params": { sensitivity-affected parameters },
                "provider_params": { provider-specific parameters },
                "features": { enabled features and their configs },
                "workflow": { pipeline workflow definition },
                "effective_sensitivity": str (actual sensitivity used)
            }
        """
        # 1. Validate pipeline exists
        if pipeline not in self.config["pipelines"]:
            raise ValueError(f"Unknown pipeline: {pipeline}")
            
        pipeline_cfg = self.config["pipelines"][pipeline]
        
        # 2. Apply sensitivity control
        effective_sensitivity, control_message = self._apply_sensitivity_control(
            pipeline_cfg, sensitivity
        )
        
        if control_message:
            logger.info(control_message)
            
        # 3. Get model for task
        model_id = self._get_model_for_task(pipeline_cfg, task)
        model_cfg = self.config["models"][model_id]
        
        # 4. Validate task support
        if task not in model_cfg.get("supported_tasks", ["transcribe"]):
            raise ValueError(
                f"Model '{model_id}' doesn't support task '{task}'. "
                f"Supported tasks: {model_cfg.get('supported_tasks')}"
            )
            
        # 5. Build base parameters
        common_params = self._build_common_params(model_cfg, effective_sensitivity)
        
        # 6. Get provider-specific parameters
        provider = model_cfg["provider"]
        provider_params = model_cfg.get("provider_params", {}).get(provider, {}).copy()
        
        # Ensure provider params are not in common params
        # Remove any provider-specific params that might have leaked into common params
        if provider == "openai-whisper":
            whisper_specific = ["compression_ratio_threshold", "logprob_threshold", 
                              "condition_on_previous_text", "hallucination_silence_threshold",
                              "carry_initial_prompt", "word_timestamps"]
            for param in whisper_specific:
                if param in common_params and param in provider_params:
                    del common_params[param]
        
        # 7. Resolve features
        features = self._resolve_features(pipeline_cfg["workflow"]["features"], effective_sensitivity)
        
        # 8. Apply user overrides if provided
        if user_overrides:
            common_params = self._apply_overrides(common_params, 
                                                user_overrides.get("common_params", {}))
            provider_params = self._apply_overrides(provider_params,
                                                  user_overrides.get("provider_params", {}))
        
        # 9. Add language/task to appropriate params
        if provider == "openai-whisper":
            common_params["language"] = language
            common_params["task"] = task
        elif provider == "huggingface":
            provider_params["task"] = task
            provider_params["target_lang"] = language if task == "translate" else None
            
        return {
            "model": model_cfg,
            "common_params": common_params,
            "provider_params": provider_params,
            "features": features,
            "workflow": pipeline_cfg["workflow"],
            "effective_sensitivity": effective_sensitivity,
            "pipeline_name": pipeline
        }
    
    def _apply_sensitivity_control(self, 
                                  pipeline_cfg: Dict, 
                                  requested_sensitivity: str) -> Tuple[str, Optional[str]]:
        """Apply pipeline sensitivity control rules."""
        
        if "sensitivity_control" not in pipeline_cfg:
            return requested_sensitivity, None
            
        control = pipeline_cfg["sensitivity_control"]
        mode = control["mode"]
        
        if mode == "force":
            forced = control["sensitivity"]
            if requested_sensitivity != forced:
                message = control.get("user_message", 
                                    f"Pipeline forcing sensitivity to '{forced}'")
                return forced, message
            return forced, None
            
        elif mode == "restrict":
            allowed = control["allowed"]
            if requested_sensitivity not in allowed:
                default = control.get("default", allowed[0])
                message = control.get("user_message",
                                    f"Sensitivity '{requested_sensitivity}' not allowed. "
                                    f"Using '{default}'")
                return default, message
            return requested_sensitivity, None
            
        elif mode == "recommend":
            recommended = control["recommended"]
            if requested_sensitivity != recommended:
                message = control.get("warn_message",
                                    f"Note: Pipeline optimized for '{recommended}' sensitivity")
                return requested_sensitivity, message
            return requested_sensitivity, None
            
        return requested_sensitivity, None
    
    def _get_model_for_task(self, pipeline_cfg: Dict, task: str) -> str:
        """Get appropriate model for the task, considering overrides."""
        
        # Check for task-specific model override
        model_overrides = pipeline_cfg.get("model_overrides", {})
        if task in model_overrides:
            return model_overrides[task]
            
        # Use default model
        return pipeline_cfg["workflow"]["model"]
    
    def _build_common_params(self, model_cfg: Dict, sensitivity: str) -> Dict:
        """Build common parameters affected by sensitivity."""
        
        # Start with model's base parameters
        params = model_cfg.get("base_params", {}).copy()
        
        # Get sensitivity modifiers
        sensitivity_cfg = self.config["sensitivity_profiles"][sensitivity]
        modifiers = sensitivity_cfg["parameter_modifiers"]
        
        # Get provider interface to know which params are affected by sensitivity
        provider = model_cfg["provider"]
        interface = self.config.get("provider_interfaces", {}).get(provider, {})
        sensitivity_affected = set(interface.get("sensitivity_affected_params", []))
        
        # Apply modifiers
        for key, value in modifiers.items():
            if "." in key:
                # Provider-prefixed parameter
                prefix, param = key.split(".", 1)
                if prefix == provider and param in sensitivity_affected:
                    params[param] = value
            else:
                # Common parameter
                if not sensitivity_affected or key in sensitivity_affected:
                    params[key] = value
                    
        return params
    
    def _resolve_features(self, feature_flags: Dict, sensitivity: str = None) -> Dict:
        """Resolve feature configurations with sensitivity adjustments."""
        
        features = {}
        feature_configs = self.config.get("feature_configs", {})
        
        # Get sensitivity modifiers if provided
        sensitivity_modifiers = {}
        if sensitivity:
            sensitivity_cfg = self.config["sensitivity_profiles"].get(sensitivity, {})
            sensitivity_modifiers = sensitivity_cfg.get("parameter_modifiers", {})
        
        for feature, setting in feature_flags.items():
            if not setting or setting == "none":
                continue
                
            if feature == "vad" and setting != "none":
                # Special handling for VAD with sensitivity adjustments
                vad_engine = self.config["vad_engines"].get(setting)
                if vad_engine:
                    # Deep copy to avoid modifying the original
                    vad_config = deepcopy(vad_engine)
                    
                    # Apply VAD threshold multiplier if present
                    if "vad_threshold_multiplier" in sensitivity_modifiers:
                        default_threshold = vad_config.get("default_params", {}).get("threshold", 0.35)
                        vad_config["default_params"]["threshold"] = round(
                            default_threshold * sensitivity_modifiers["vad_threshold_multiplier"], 4
                        )
                    
                    # Apply min_speech_duration override if present
                    if "min_speech_duration_ms" in sensitivity_modifiers:
                        vad_config["default_params"]["min_speech_duration_ms"] = (
                            sensitivity_modifiers["min_speech_duration_ms"]
                        )
                    
                    features["vad"] = {
                        "engine": setting,
                        "config": vad_config
                    }
                    
            elif feature == "scene_detection" and setting:
                # Handle scene detection with sensitivity adjustments
                scene_config = feature_configs.get("scene_detection", {}).get("default", {}).copy()
                
                if "scene_min_duration_multiplier" in sensitivity_modifiers:
                    default_min_duration = scene_config.get("min_duration", 0.2)
                    scene_config["min_duration"] = (
                        default_min_duration * sensitivity_modifiers["scene_min_duration_multiplier"]
                    )
                
                features["scene_detection"] = scene_config
                    
            elif feature in feature_configs:
                # Standard feature
                config_name = setting if isinstance(setting, str) else "default"
                if config_name in feature_configs[feature]:
                    features[feature] = feature_configs[feature][config_name].copy()
                    
            else:
                # Simple boolean or value feature
                features[feature] = setting
                
        return features
    
    def _apply_overrides(self, base: Dict, overrides: Dict) -> Dict:
        """Apply user overrides to parameters."""
        result = base.copy()
        result.update(overrides)
        return result
    
    def _validate_config(self):
        """Validate configuration integrity."""
        
        errors = []
        
        # Check all pipeline models exist
        for name, pipeline in self.config.get("pipelines", {}).items():
            model_id = pipeline["workflow"]["model"]
            if model_id not in self.config.get("models", {}):
                errors.append(f"Pipeline '{name}' references unknown model '{model_id}'")
                
            # Check model overrides
            for task, override_model in pipeline.get("model_overrides", {}).items():
                if override_model not in self.config.get("models", {}):
                    errors.append(f"Pipeline '{name}' override references unknown model '{override_model}'")
                    
        # Check VAD engines in features
        for name, pipeline in self.config.get("pipelines", {}).items():
            vad = pipeline["workflow"]["features"].get("vad")
            if vad and vad != "none" and vad not in self.config.get("vad_engines", {}):
                errors.append(f"Pipeline '{name}' references unknown VAD engine '{vad}'")
                
        # Check sensitivity profiles
        for name, profile in self.config.get("sensitivity_profiles", {}).items():
            if "parameter_modifiers" not in profile:
                errors.append(f"Sensitivity profile '{name}' missing parameter_modifiers")
                
        if errors:
            for error in errors:
                logger.error(f"Config validation: {error}")
            raise ValueError(f"Configuration validation failed with {len(errors)} errors")
            
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        
        if not config_path:
            config_path = Path(__file__).parent / "config.v2.json"
            
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Validate version
        if config.get("version") != "2.0":
            raise ValueError(f"Unsupported config version: {config.get('version')}")
            
        return config
    
    def get_available_pipelines(self) -> List[str]:
        """Get list of available pipelines."""
        return list(self.config.get("pipelines", {}).keys())
    
    def get_available_sensitivities(self, pipeline: str) -> List[str]:
        """Get available sensitivities for a pipeline."""
        
        if pipeline not in self.config.get("pipelines", {}):
            return list(self.config.get("sensitivity_profiles", {}).keys())
            
        pipeline_cfg = self.config["pipelines"][pipeline]
        control = pipeline_cfg.get("sensitivity_control", {})
        
        if control.get("mode") == "force":
            return [control["sensitivity"]]
        elif control.get("mode") == "restrict":
            return control["allowed"]
        else:
            return list(self.config.get("sensitivity_profiles", {}).keys())
    
    def get_pipeline_info(self, pipeline: str) -> Dict[str, Any]:
        """Get detailed information about a pipeline."""
        
        if pipeline not in self.config.get("pipelines", {}):
            raise ValueError(f"Unknown pipeline: {pipeline}")
            
        pipeline_cfg = self.config["pipelines"][pipeline]
        model_cfg = self.config["models"][pipeline_cfg["workflow"]["model"]]
        
        return {
            "name": pipeline,
            "description": pipeline_cfg.get("description", ""),
            "model": model_cfg["model_name"],
            "provider": model_cfg["provider"],
            "supported_tasks": model_cfg.get("supported_tasks", ["transcribe"]),
            "features": pipeline_cfg["workflow"]["features"],
            "sensitivity_control": pipeline_cfg.get("sensitivity_control"),
            "requirements": pipeline_cfg.get("requirements", {})
        }


# Example usage:
if __name__ == "__main__":
    tuner = TranscriptionTunerV2()
    
    # Example 1: Standard usage
    config = tuner.resolve_params("faster", "aggressive")
    print(f"Model: {config['model']['model_name']}")
    print(f"Effective sensitivity: {config['effective_sensitivity']}")
    
    # Example 2: Pipeline with forced sensitivity
    config = tuner.resolve_params("ultra_quality", "aggressive")  # Will force to conservative
    print(f"Forced sensitivity: {config['effective_sensitivity']}")
    
    # Example 3: Get available sensitivities for streaming
    sensitivities = tuner.get_available_sensitivities("streaming")
    print(f"Streaming allows: {sensitivities}")  # Won't include 'conservative'