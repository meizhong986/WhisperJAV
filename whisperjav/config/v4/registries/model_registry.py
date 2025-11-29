"""
Model Registry for WhisperJAV v4 Configuration.

Provides access to all model configurations across all ecosystems.
Models are discovered from YAML files in:
    config/v4/ecosystems/<ecosystem>/models/*.yaml

Usage:
    from whisperjav.config.v4.registries import get_model_registry

    registry = get_model_registry()

    # Get specific model
    model = registry.get("kotoba-whisper-v2")

    # List all models
    all_models = registry.list_all()

    # Find models by ecosystem
    transformers_models = registry.find_by_ecosystem("transformers")

    # Get resolved config with sensitivity
    config = registry.get_resolved_config("kotoba-whisper-v2", "aggressive")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import ModelNotFoundError
from ..schemas.model import ModelConfig
from .base_registry import BaseRegistry


class ModelRegistry(BaseRegistry[ModelConfig]):
    """
    Registry for model configurations.

    Discovers and manages all model configs across ecosystems.
    """

    _config_class = ModelConfig
    _config_dir_name = "models"
    _instance: Optional["ModelRegistry"] = None

    def _get_config_paths(self) -> List[Path]:
        """
        Get paths to all model YAML files.

        Searches: ecosystems/<ecosystem>/models/*.yaml
        """
        paths = []

        if not self._base_path.exists():
            return paths

        # Iterate through ecosystem directories
        for ecosystem_dir in self._base_path.iterdir():
            if not ecosystem_dir.is_dir():
                continue

            models_dir = ecosystem_dir / "models"
            if models_dir.exists():
                for yaml_file in models_dir.glob("*.yaml"):
                    paths.append(yaml_file)

        return paths

    def _raise_not_found(self, name: str) -> None:
        """Raise ModelNotFoundError with available models."""
        raise ModelNotFoundError(
            model_name=name,
            available_models=self.list_names(),
        )

    def find_by_ecosystem(self, ecosystem: str) -> List[ModelConfig]:
        """
        Find all models belonging to an ecosystem.

        Args:
            ecosystem: Ecosystem name (e.g., "transformers")

        Returns:
            List of model configs in that ecosystem
        """
        self._ensure_loaded()
        return [
            m
            for m in self._configs.values()
            if m.metadata.ecosystem == ecosystem
        ]

    def find_by_provider(self, provider: str) -> List[ModelConfig]:
        """
        Find all models using a specific provider.

        Args:
            provider: Provider name (e.g., "faster_whisper")

        Returns:
            List of model configs using that provider
        """
        self._ensure_loaded()
        return [
            m
            for m in self._configs.values()
            if m.provider == provider
        ]

    def get_resolved_config(
        self,
        model_name: str,
        sensitivity: str = "balanced",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get fully resolved model config with sensitivity and overrides.

        Args:
            model_name: Model name
            sensitivity: Sensitivity preset (conservative, balanced, aggressive)
            overrides: Additional user overrides

        Returns:
            Merged configuration dict ready for use
        """
        model = self.get(model_name)
        return model.get_resolved_config(sensitivity, overrides)

    def get_gui_schema(self, model_name: str) -> Dict[str, Any]:
        """
        Get GUI schema for a model.

        Args:
            model_name: Model name

        Returns:
            GUI schema dict for frontend rendering
        """
        model = self.get(model_name)
        return model.get_gui_schema()

    def list_by_ecosystem(self) -> Dict[str, List[str]]:
        """
        Get all models organized by ecosystem.

        Returns:
            Dict mapping ecosystem name to list of model names
        """
        self._ensure_loaded()
        result: Dict[str, List[str]] = {}

        for model in self._configs.values():
            ecosystem = model.metadata.ecosystem or "unknown"
            if ecosystem not in result:
                result[ecosystem] = []
            result[ecosystem].append(model.get_name())

        return result


# Singleton accessor
_registry_instance: Optional[ModelRegistry] = None


def get_model_registry(base_path: Optional[Path] = None) -> ModelRegistry:
    """
    Get the model registry singleton.

    Args:
        base_path: Optional base path (only used on first call)

    Returns:
        ModelRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry(base_path)
    return _registry_instance
