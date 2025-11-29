"""
Preset Configuration Schema for WhisperJAV v4.

Defines the structure for global sensitivity preset YAML files.
Presets provide cross-cutting configuration profiles (conservative, balanced, aggressive).

Example YAML:
    schemaVersion: v1
    kind: Preset

    metadata:
      name: balanced
      displayName: "Balanced"
      description: "Default settings balancing accuracy and speed"
      tags: [default, recommended]

    level: balanced

    description_long: |
      The balanced preset provides good accuracy for most content
      while maintaining reasonable processing speed.

    # Cross-cutting overrides applied to all models/tools
    overrides:
      decode.beam_size: 5
      quality.no_speech: 0.5
      vad.threshold: 0.3
"""

from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import Field

from .base import ConfigBase, ConfigKind


class SensitivityLevel(str, Enum):
    """Standard sensitivity levels."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class PresetConfig(ConfigBase):
    """
    Global sensitivity preset configuration schema.

    Presets provide cross-cutting configuration that can be applied
    to any model or tool. They represent different trade-offs:

    - conservative: Higher thresholds, fewer false positives, may miss quiet speech
    - balanced: Default settings, good for most content
    - aggressive: Lower thresholds, captures more detail, may have more artifacts
    """

    kind: Literal[ConfigKind.PRESET] = Field(
        default=ConfigKind.PRESET,
        description="Must be 'Preset' for preset configs",
    )

    level: SensitivityLevel = Field(
        ...,  # Required
        description="Sensitivity level (conservative, balanced, aggressive)",
    )

    description_long: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Detailed description of this preset's behavior",
    )

    # Cross-cutting overrides
    overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter overrides applied to all models/tools",
    )

    # Use case hints
    recommended_for: list[str] = Field(
        default_factory=list,
        description="Use cases this preset is recommended for",
    )
    not_recommended_for: list[str] = Field(
        default_factory=list,
        description="Use cases this preset should NOT be used for",
    )

    def get_overrides(self) -> Dict[str, Any]:
        """Get the override dict to apply to configs."""
        return dict(self.overrides)

    def get_override(self, key: str, default: Any = None) -> Any:
        """Get a specific override value."""
        return self.overrides.get(key, default)

    def applies_to_key(self, key: str) -> bool:
        """Check if this preset has an override for the given key."""
        return key in self.overrides

    def merge_with_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge preset overrides with a configuration dict.

        Preset values override config values.

        Args:
            config: Base configuration dict

        Returns:
            Merged configuration with preset overrides applied
        """
        result = dict(config)
        result.update(self.overrides)
        return result
