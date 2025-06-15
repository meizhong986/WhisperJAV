import json
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

from whisperjav.utils.logger import logger

def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    The `override` dictionary's values take precedence.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    return result

class TranscriptionTuner:
    """
    Resolves user intent (mode, sensitivity) into a final, fully-resolved
    set of technical parameters for the ASR pipelines. This class is the
    central engine for the configuration system.
    """
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initializes the tuner by loading the base configuration and merging
        any user-provided configuration file on top of it.
        
        Args:
            config_path (Optional[Path]): The path to the user's config.json file.
        """
        self.config = self._load_config(config_path)
        self.parameter_sets = self.config.get("parameter_sets", {})
        self.components = self.config.get("components", {})
        self.pipelines = self.config.get("pipelines", {})

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        base_config = {}

        # Corrected path to the template file inside whisperjav/config/
        template_path = Path(__file__).resolve().parent / "config.template.json"

        if template_path.exists():
            logger.debug(f"Found config template at: {template_path}")
            with open(template_path, 'r', encoding='utf-8') as f:
                base_config = json.load(f)
        else:
            logger.warning(f"Config template not found at: {template_path}. Using empty defaults.")

        user_config = {}
        if config_path and config_path.exists():
            logger.info(f"Loading user configuration from: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)

        return _merge_configs(base_config, user_config)


    def get_resolved_params(self, mode: str, sensitivity: str) -> Dict[str, Any]:
        """
        The main public method. Takes user choices and returns a dictionary
        of fully resolved parameter sets for the pipeline.
        
        Args:
            mode (str): The processing mode (e.g., 'balanced', 'fast').
            sensitivity (str): The sensitivity level (e.g., 'balanced', 'aggressive').
            
        Returns:
            Dict[str, Any]: A dictionary containing the final, resolved parameter sets.
        """
        # 1. Get the default component name from the pipeline's blueprint
        pipeline_blueprint = self.pipelines.get(mode)
        if not pipeline_blueprint:
            logger.error(f"No pipeline definition found for mode '{mode}' in config.")
            return {}
            
        default_component_name = pipeline_blueprint.get("transcription_component")
        if not default_component_name:
            logger.error(f"No default transcription component defined for mode '{mode}' in config.")
            return {}

        # 2. Apply the sensitivity variant to the component name
        # e.g., "ASR_COMPONENT_BALANCED" -> "ASR_COMPONENT_AGGRESSIVE"
        if sensitivity == 'balanced':
            final_component_name = default_component_name
        else:
            final_component_name = default_component_name.replace('BALANCED', f'{sensitivity.upper()}')
        
        logger.debug(f"Resolved component for mode '{mode}' with sensitivity '{sensitivity}': {final_component_name}")

        # 3. Get the component "recipe"
        component_recipe = self.components.get(final_component_name)
        if not component_recipe:
            logger.error(f"Component recipe '{final_component_name}' not found in config.")
            # Gracefully fall back to the balanced component if the variant doesn't exist
            if final_component_name != default_component_name:
                logger.warning(f"Falling back to balanced component: {default_component_name}")
                component_recipe = self.components.get(default_component_name, {})
            else:
                return {}

        # 4. Resolve the references in the recipe to build the final parameter dictionaries
        resolved_params = {}
        for key, ref_name in component_recipe.items():
            if key.endswith('_ref'):
                param_set = self.parameter_sets.get(ref_name)
                if not param_set:
                    logger.warning(f"Parameter set '{ref_name}' not found for component '{final_component_name}'.")
                    param_set = {}
                # Convert a key like "vad_options_ref" to "vad_options"
                resolved_key = key.replace('_ref', '')
                resolved_params[resolved_key] = param_set
        
        # Manually add other component types that don't have sensitivity variants
        for comp_type in ["post_processing_component", "scene_detection_component"]:
            comp_name = pipeline_blueprint.get(comp_type)
            if comp_name:
                resolved_key = comp_type.replace('_component', '_options')
                resolved_params[resolved_key] = self.parameter_sets.get(comp_name, {})

        return resolved_params

