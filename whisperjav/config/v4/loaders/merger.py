"""
Configuration Merger for WhisperJAV v4.

Provides deep merging capabilities for config composition:
- Strategic merge (like Kubernetes): lists are replaced, dicts are merged
- Override merge: later values completely replace earlier
- Additive merge: lists are concatenated

Usage:
    from whisperjav.config.v4.loaders import ConfigMerger, deep_merge

    # Simple deep merge
    result = deep_merge(base_dict, override_dict)

    # With merger object for more control
    merger = ConfigMerger(strategy="strategic")
    result = merger.merge(base, override)
"""

from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TypeVar

T = TypeVar("T")


class MergeStrategy(str, Enum):
    """Available merge strategies."""

    STRATEGIC = "strategic"  # K8s-style: dicts merged, lists replaced
    OVERRIDE = "override"  # Later values completely replace
    ADDITIVE = "additive"  # Lists are concatenated


class ConfigMerger:
    """
    Configuration merger with multiple strategies.

    Handles the complexities of merging nested configuration structures
    while respecting the intended merge semantics.
    """

    def __init__(
        self,
        strategy: MergeStrategy = MergeStrategy.STRATEGIC,
        list_merge_keys: Optional[Set[str]] = None,
    ):
        """
        Initialize the merger.

        Args:
            strategy: Merge strategy to use
            list_merge_keys: Keys where lists should be merged (additive)
                            even when using strategic merge
        """
        self.strategy = strategy
        self.list_merge_keys = list_merge_keys or set()

    def merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge override dict on top of base dict.

        Args:
            base: Base configuration
            override: Override configuration (takes precedence)

        Returns:
            Merged configuration
        """
        if self.strategy == MergeStrategy.OVERRIDE:
            return self._merge_override(base, override)
        elif self.strategy == MergeStrategy.ADDITIVE:
            return self._merge_additive(base, override)
        else:  # STRATEGIC
            return self._merge_strategic(base, override, path="")

    def _merge_override(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Override merge: later values completely replace earlier.

        Simple dict update with deep copy.
        """
        result = deepcopy(base)
        result.update(deepcopy(override))
        return result

    def _merge_additive(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Additive merge: lists are concatenated.
        """
        result = deepcopy(base)

        for key, override_value in override.items():
            if key not in result:
                result[key] = deepcopy(override_value)
            elif isinstance(result[key], dict) and isinstance(override_value, dict):
                result[key] = self._merge_additive(result[key], override_value)
            elif isinstance(result[key], list) and isinstance(override_value, list):
                # Concatenate lists
                result[key] = result[key] + deepcopy(override_value)
            else:
                result[key] = deepcopy(override_value)

        return result

    def _merge_strategic(
        self, base: Dict[str, Any], override: Dict[str, Any], path: str
    ) -> Dict[str, Any]:
        """
        Strategic merge: dicts are recursively merged, lists are replaced.

        This is similar to Kubernetes strategic merge patch.
        """
        result = deepcopy(base)

        for key, override_value in override.items():
            current_path = f"{path}.{key}" if path else key

            if key not in result:
                # New key, just add it
                result[key] = deepcopy(override_value)
            elif isinstance(result[key], dict) and isinstance(override_value, dict):
                # Both are dicts, merge recursively
                result[key] = self._merge_strategic(
                    result[key], override_value, current_path
                )
            elif isinstance(result[key], list) and isinstance(override_value, list):
                # Check if this key should be merged additively
                if key in self.list_merge_keys:
                    result[key] = result[key] + deepcopy(override_value)
                else:
                    # Replace list (strategic default)
                    result[key] = deepcopy(override_value)
            else:
                # Scalar or type mismatch: override wins
                result[key] = deepcopy(override_value)

        return result

    def merge_many(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple configs in order.

        Args:
            configs: List of configs, later ones take precedence

        Returns:
            Merged configuration
        """
        if not configs:
            return {}

        result = deepcopy(configs[0])
        for config in configs[1:]:
            result = self.merge(result, config)

        return result


def deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
    strategy: MergeStrategy = MergeStrategy.STRATEGIC,
) -> Dict[str, Any]:
    """
    Convenience function for deep merging two dicts.

    Args:
        base: Base configuration
        override: Override configuration (takes precedence)
        strategy: Merge strategy to use

    Returns:
        Merged configuration
    """
    merger = ConfigMerger(strategy=strategy)
    return merger.merge(base, override)


def apply_overrides(
    config: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply flat overrides to a config dict.

    Overrides use dot-notation keys (e.g., "model.device")
    which are expanded into nested structure.

    Args:
        config: Base configuration
        overrides: Flat override dict with dot-notation keys

    Returns:
        Config with overrides applied
    """
    result = deepcopy(config)

    for key, value in overrides.items():
        _set_nested_value(result, key, value)

    return result


def _set_nested_value(d: Dict[str, Any], key: str, value: Any) -> None:
    """
    Set a value in a nested dict using dot-notation key.

    Args:
        d: Dict to modify (in-place)
        key: Dot-notation key (e.g., "model.device")
        value: Value to set
    """
    parts = key.split(".")
    current = d

    # Navigate to parent
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        elif not isinstance(current[part], dict):
            # Can't navigate further, replace with dict
            current[part] = {}
        current = current[part]

    # Set final value
    current[parts[-1]] = value


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten a nested dict to dot-notation keys.

    Args:
        d: Nested dict to flatten
        prefix: Key prefix (for recursion)

    Returns:
        Flat dict with dot-notation keys
    """
    result = {}

    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            # Recurse into nested dict
            nested = flatten_dict(value, full_key)
            result.update(nested)
        else:
            result[full_key] = value

    return result


def unflatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unflatten a dot-notation dict to nested structure.

    Args:
        d: Flat dict with dot-notation keys

    Returns:
        Nested dict
    """
    result: Dict[str, Any] = {}

    for key, value in d.items():
        _set_nested_value(result, key, value)

    return result
