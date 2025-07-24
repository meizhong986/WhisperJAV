import json
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy
from whisperjav.utils.logger import logger

class TranscriptionTunerV3:
    """
    Configuration resolver for WhisperJAV v3.
    Reads the modular asr_config.v3.json to produce a structured config object.
    This class is the single source of truth for resolving run configurations.
    """
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initializes the tuner by loading the v3 configuration file.
        
        Args:
            config_path: Optional path to a custom asr_config.v3.json. 
                         If None, loads the default from the config directory.
        """
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Loads and validates the v3 configuration file."""
        if not config_path:
            config_path = Path(__file__).parent / "asr_config.v3.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"ASR configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            # Using a basic JSON load, ignoring comments for simplicity in the loader.
            # For production, a more robust loader that strips comments might be used.
            config_text = "".join(line for line in f if not line.strip().startswith("//"))
            config = json.loads(config_text)
            
        if config.get("version") != "3.0":
            raise ValueError(f"Unsupported config version: {config.get('version')}. Expected 3.0.")
            
        logger.debug("Successfully loaded asr_config.v3.json")
        return config

    def resolve_params(self, pipeline_name: str, sensitivity: str, task: str) -> Dict[str, Any]:
        """
        Resolves all necessary configurations for a given run into a single,
        structured dictionary. This is the primary entry point for the class.

        Args:
            pipeline_name: The name of the pipeline to use (e.g., 'balanced').
            sensitivity: The name of the sensitivity profile (e.g., 'aggressive').
            task: The ASR task to perform ('transcribe' or 'translate').

        Returns:
            A structured dictionary containing all resolved configurations.
        """
        if pipeline_name not in self.config["pipelines"]:
            raise ValueError(f"Unknown pipeline specified: '{pipeline_name}'")
        if sensitivity not in self.config["sensitivity_profiles"]:
             raise ValueError(f"Unknown sensitivity specified: '{sensitivity}'")
        
        pipeline_cfg = self.config["pipelines"][pipeline_name]
        sensitivity_profile = self.config["sensitivity_profiles"][sensitivity]

        # Determine the correct model, considering task-based overrides
        model_id = pipeline_cfg.get("model_overrides", {}).get(task, pipeline_cfg["workflow"]["model"])
        model_cfg = self.config["models"][model_id]
        provider = model_cfg["provider"]

        # Assemble the parameters by looking up the sets defined in the profile
        params = {
            "decoder": deepcopy(self.config["parameter_sets"]["decoder_params"][sensitivity_profile["decoder"]]),
            "vad": deepcopy(self.config["parameter_sets"]["vad_params"][sensitivity_profile["vad"]]),
            "provider": deepcopy(self.config["parameter_sets"]["provider_specific_params"][provider][sensitivity_profile["provider_settings"]])
        }

        # Assemble feature configurations from the pipeline definition
        features = {}
        for feature, setting in pipeline_cfg["workflow"].get("features", {}).items():
            if setting and setting != "none":
                config_name = setting if isinstance(setting, str) else "default"
                if feature in self.config["feature_configs"] and config_name in self.config["feature_configs"][feature]:
                    features[feature] = deepcopy(self.config["feature_configs"][feature][config_name])
        
        # Return the final, well-structured configuration object
        return {
            "pipeline_name": pipeline_name,
            "sensitivity_name": sensitivity,
            "workflow": pipeline_cfg["workflow"],
            "model": model_cfg,
            "params": params,
            "features": features,
            "task": task,
            "language": self.config.get("defaults", {}).get("language", "ja")
        }
