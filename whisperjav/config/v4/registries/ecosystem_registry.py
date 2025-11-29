"""
Ecosystem Registry for WhisperJAV v4 Configuration.

Provides access to all ecosystem configurations.
Ecosystems are discovered from YAML files in:
    config/v4/ecosystems/<ecosystem>/ecosystem.yaml

Usage:
    from whisperjav.config.v4.registries import get_ecosystem_registry

    registry = get_ecosystem_registry()

    # Get specific ecosystem
    ecosystem = registry.get("transformers")

    # List all ecosystems
    all_ecosystems = registry.list_all()
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import EcosystemNotFoundError
from ..schemas.ecosystem import EcosystemConfig
from .base_registry import BaseRegistry


class EcosystemRegistry(BaseRegistry[EcosystemConfig]):
    """
    Registry for ecosystem configurations.

    Discovers and manages all ecosystem configs.
    """

    _config_class = EcosystemConfig
    _config_dir_name = ""
    _instance: Optional["EcosystemRegistry"] = None

    def _get_config_paths(self) -> List[Path]:
        """
        Get paths to all ecosystem YAML files.

        Searches: ecosystems/<ecosystem>/ecosystem.yaml
        """
        paths = []

        if not self._base_path.exists():
            return paths

        for ecosystem_dir in self._base_path.iterdir():
            if not ecosystem_dir.is_dir():
                continue

            ecosystem_file = ecosystem_dir / "ecosystem.yaml"
            if ecosystem_file.exists():
                paths.append(ecosystem_file)

        return paths

    def _raise_not_found(self, name: str) -> None:
        """Raise EcosystemNotFoundError with available ecosystems."""
        raise EcosystemNotFoundError(
            ecosystem_name=name,
            available_ecosystems=self.list_names(),
        )

    def get_defaults(self, ecosystem_name: str) -> Dict[str, Any]:
        """
        Get default parameter values for an ecosystem.

        Args:
            ecosystem_name: Ecosystem name

        Returns:
            Default parameter dict
        """
        ecosystem = self.get(ecosystem_name)
        return ecosystem.get_model_base_config()

    def get_provider_class(self, ecosystem_name: str) -> type:
        """
        Get the ASR provider class for an ecosystem.

        Args:
            ecosystem_name: Ecosystem name

        Returns:
            The ASR class

        Raises:
            ImportError: If module cannot be imported
        """
        ecosystem = self.get(ecosystem_name)
        return ecosystem.get_provider_class()

    def check_dependencies(self, ecosystem_name: str) -> Dict[str, bool]:
        """
        Check if ecosystem dependencies are installed.

        Args:
            ecosystem_name: Ecosystem name

        Returns:
            Dict mapping package name to installation status
        """
        ecosystem = self.get(ecosystem_name)
        return ecosystem.check_dependencies()

    def get_compatible_tools(self, ecosystem_name: str) -> List[str]:
        """
        Get list of compatible tool names for an ecosystem.

        Args:
            ecosystem_name: Ecosystem name

        Returns:
            List of tool names
        """
        ecosystem = self.get(ecosystem_name)
        return ecosystem.compatible_tools


# Singleton accessor
_registry_instance: Optional[EcosystemRegistry] = None


def get_ecosystem_registry(base_path: Optional[Path] = None) -> EcosystemRegistry:
    """
    Get the ecosystem registry singleton.

    Args:
        base_path: Optional base path (only used on first call)

    Returns:
        EcosystemRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = EcosystemRegistry(base_path)
    return _registry_instance
