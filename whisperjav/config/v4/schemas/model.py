"""
Model Configuration Schema for WhisperJAV v4.

Defines the structure for model YAML configuration files.
A model represents a specific ASR model with its parameters and presets.

Example YAML:
    schemaVersion: v1
    kind: Model

    metadata:
      name: kotoba-whisper-v2
      ecosystem: transformers
      displayName: "Kotoba Whisper v2.0"
      description: "Japanese-optimized distil-whisper"
      tags: [japanese, distil, fast]

    spec:
      model.id: "kotoba-tech/kotoba-whisper-v2.0"
      model.device: auto
      chunk.length_s: 15
      decode.beam_size: 5

    presets:
      conservative:
        decode.beam_size: 3
      balanced: {}
      aggressive:
        decode.beam_size: 7

    gui:
      model.device:
        widget: dropdown
        options: [auto, cuda, cpu]
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .base import ConfigBase, ConfigKind, GUIHint, MetadataBlock


class ModelSpec(Dict[str, Any]):
    """
    Model specification as flat key-value pairs.

    Uses dot-prefix notation for grouping:
    - model.* : Model loading parameters
    - chunk.* : Chunking/segmentation parameters
    - decode.* : Decoding/generation parameters
    - quality.* : Quality threshold parameters
    - vad.* : Internal VAD parameters (if applicable)

    This flat structure is easier to patch and merge than nested dicts.
    """

    pass  # Type alias for Dict[str, Any]


class ModelConfig(ConfigBase):
    """
    Complete model configuration schema.

    Validates model YAML files and provides access to:
    - spec: Model parameters (flat dict with dot-prefixed keys)
    - presets: Sensitivity preset overrides
    - gui: GUI widget hints for each parameter
    """

    kind: Literal[ConfigKind.MODEL] = Field(
        default=ConfigKind.MODEL,
        description="Must be 'Model' for model configs",
    )

    spec: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model parameters as flat key-value pairs with dot-prefix grouping",
    )

    presets: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "conservative": {},
            "balanced": {},
            "aggressive": {},
        },
        description="Sensitivity preset overrides",
    )

    gui: Dict[str, GUIHint] = Field(
        default_factory=dict,
        description="GUI rendering hints for each parameter",
    )

    # Model-specific metadata (optional)
    provider: Optional[str] = Field(
        default=None,
        description="Backend provider (e.g., 'faster_whisper', 'transformers')",
    )
    compatible_tools: List[str] = Field(
        default_factory=list,
        description="List of compatible auxiliary tools",
    )

    @field_validator("spec")
    @classmethod
    def validate_spec_keys(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate spec keys use dot-prefix notation correctly."""
        for key in v.keys():
            if not isinstance(key, str):
                raise ValueError(f"Spec keys must be strings, got: {type(key)}")
            # Keys should be valid identifiers with optional dots
            parts = key.split(".")
            for part in parts:
                if not part:
                    raise ValueError(f"Invalid spec key (empty part): {key}")
                # Allow alphanumeric and underscores
                if not part.replace("_", "").isalnum():
                    raise ValueError(f"Invalid spec key format: {key}")
        return v

    @field_validator("presets")
    @classmethod
    def validate_presets(cls, v: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Ensure standard presets exist."""
        standard_presets = {"conservative", "balanced", "aggressive"}
        for preset in standard_presets:
            if preset not in v:
                v[preset] = {}  # Add empty preset if missing
        return v

    @model_validator(mode="after")
    def validate_gui_keys_exist_in_spec(self) -> "ModelConfig":
        """Ensure GUI hints reference existing spec keys."""
        for gui_key in self.gui.keys():
            if gui_key not in self.spec:
                # Warning: GUI hint for non-existent key
                # We allow this for extensibility but log a warning
                pass
        return self

    def get_resolved_config(
        self,
        sensitivity: str = "balanced",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get fully resolved configuration with preset and overrides applied.

        Resolution order (later wins):
        1. spec (base values)
        2. presets[sensitivity] (sensitivity overrides)
        3. overrides (user overrides)

        Args:
            sensitivity: Preset name (conservative, balanced, aggressive)
            overrides: Additional user overrides

        Returns:
            Merged configuration dict
        """
        result = dict(self.spec)  # Copy base spec

        # Apply preset
        preset = self.presets.get(sensitivity, {})
        result.update(preset)

        # Apply user overrides
        if overrides:
            result.update(overrides)

        return result

    def get_parameter_groups(self) -> Dict[str, List[str]]:
        """
        Get parameters organized by their dot-prefix group.

        Returns:
            Dict mapping group name to list of full parameter keys
            e.g., {"model": ["model.id", "model.device"], "chunk": [...]}
        """
        groups: Dict[str, List[str]] = {}
        for key in self.spec.keys():
            parts = key.split(".")
            if len(parts) >= 2:
                group = parts[0]
            else:
                group = "general"
            if group not in groups:
                groups[group] = []
            groups[group].append(key)
        return groups

    def get_gui_schema(self) -> Dict[str, Any]:
        """
        Get complete GUI schema for this model.

        Returns a structure suitable for frontend rendering:
        {
            "model_name": "...",
            "display_name": "...",
            "parameters": [
                {
                    "key": "model.device",
                    "value": "auto",
                    "gui": {...}
                },
                ...
            ],
            "groups": {...}
        }
        """
        parameters = []
        for key, value in self.spec.items():
            param = {
                "key": key,
                "value": value,
                "type": type(value).__name__,
            }
            if key in self.gui:
                param["gui"] = self.gui[key].model_dump()
            else:
                # Generate default GUI hint
                param["gui"] = self._infer_gui_hint(key, value).model_dump()
            parameters.append(param)

        return {
            "model_name": self.metadata.name,
            "display_name": self.metadata.displayName,
            "description": self.metadata.description,
            "ecosystem": self.metadata.ecosystem,
            "parameters": parameters,
            "groups": self.get_parameter_groups(),
            "presets": list(self.presets.keys()),
        }

    def _infer_gui_hint(self, key: str, value: Any) -> GUIHint:
        """Infer GUI hint from parameter key and value type."""
        from .base import GUIWidget

        # Infer widget from value type
        if isinstance(value, bool):
            widget = GUIWidget.CHECKBOX
        elif isinstance(value, int):
            widget = GUIWidget.SPINNER
        elif isinstance(value, float):
            widget = GUIWidget.NUMBER
        elif isinstance(value, list):
            widget = GUIWidget.TEXT  # JSON input for lists
        else:
            widget = GUIWidget.TEXT

        # Infer group from key prefix
        parts = key.split(".")
        group = parts[0] if len(parts) >= 2 else "general"

        # Convert key to label
        label = parts[-1].replace("_", " ").title()

        return GUIHint(
            widget=widget,
            label=label,
            group=group,
        )
