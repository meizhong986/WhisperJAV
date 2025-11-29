"""
Preset Registry for WhisperJAV v4 Configuration.

Provides access to global sensitivity preset configurations.
Presets are discovered from YAML files in:
    config/v4/ecosystems/presets/*.yaml

Usage:
    from whisperjav.config.v4.registries import get_preset_registry

    registry = get_preset_registry()

    # Get specific preset
    preset = registry.get("balanced")

    # Get overrides for a preset
    overrides = registry.get_overrides("aggressive")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import PresetNotFoundError
from ..schemas.preset import PresetConfig
from .base_registry import BaseRegistry


class PresetRegistry(BaseRegistry[PresetConfig]):
    """
    Registry for sensitivity preset configurations.

    Discovers and manages all preset configs.
    """

    _config_class = PresetConfig
    _config_dir_name = "presets"
    _instance: Optional["PresetRegistry"] = None

    def _get_config_paths(self) -> List[Path]:
        """
        Get paths to all preset YAML files.

        Searches: ecosystems/presets/*.yaml
        """
        paths = []
        presets_dir = self._base_path / "presets"

        if presets_dir.exists():
            for yaml_file in presets_dir.glob("*.yaml"):
                paths.append(yaml_file)

        return paths

    def _raise_not_found(self, name: str) -> None:
        """Raise PresetNotFoundError with available presets."""
        from ..errors import PresetNotFoundError

        raise PresetNotFoundError(
            preset_name=name,
            available_presets=self.list_names(),
        )

    def get_overrides(self, preset_name: str) -> Dict[str, Any]:
        """
        Get override dict for a preset.

        Args:
            preset_name: Preset name

        Returns:
            Override parameter dict
        """
        preset = self.get(preset_name)
        return preset.get_overrides()

    def apply_to_config(
        self, preset_name: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply preset overrides to a config dict.

        Args:
            preset_name: Preset name
            config: Base config dict

        Returns:
            Config with preset overrides applied
        """
        preset = self.get(preset_name)
        return preset.merge_with_config(config)

    def get_standard_presets(self) -> List[PresetConfig]:
        """
        Get the standard presets (conservative, balanced, aggressive).

        Returns:
            List of standard preset configs
        """
        standard_names = ["conservative", "balanced", "aggressive"]
        presets = []

        for name in standard_names:
            preset = self.get_or_none(name)
            if preset:
                presets.append(preset)

        return presets


# Singleton accessor
_registry_instance: Optional[PresetRegistry] = None


def get_preset_registry(base_path: Optional[Path] = None) -> PresetRegistry:
    """
    Get the preset registry singleton.

    Args:
        base_path: Optional base path (only used on first call)

    Returns:
        PresetRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = PresetRegistry(base_path)
    return _registry_instance
