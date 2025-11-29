"""
Tool Configuration Schema for WhisperJAV v4.

Defines the structure for auxiliary tool YAML configuration files.
Tools are reusable components like scene detection, VAD, post-processing.

Example YAML:
    schemaVersion: v1
    kind: Tool

    metadata:
      name: auditok-scene-detection
      displayName: "Auditok Scene Detection"
      description: "Audio-based scene splitting using silence detection"
      tags: [scene-detection, auditok, audio]

    contract:
      input:
        audio_path: path
        sample_rate: int
      output:
        segments: list

    spec:
      max_duration_s: 29.0
      min_duration_s: 0.2
      pass1.energy_threshold: 32
      pass2.energy_threshold: 38

    presets:
      conservative:
        pass1.energy_threshold: 40
      balanced: {}
      aggressive:
        pass1.energy_threshold: 28

    gui:
      max_duration_s:
        widget: slider
        min: 5
        max: 60
        step: 1
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator

from .base import ConfigBase, ConfigKind, GUIHint


class ToolContract(Dict[str, Any]):
    """
    Tool input/output contract definition.

    Contracts ensure tools have well-defined interfaces that ecosystems
    can rely on. This enables tool sharing across ecosystems.

    Structure:
        input:
            param_name: type_name
            ...
        output:
            param_name: type_name
            ...
    """

    pass  # Type alias


class ToolConfig(ConfigBase):
    """
    Complete tool configuration schema.

    A tool defines:
    - contract: Input/output interface specification
    - spec: Tool parameters (flat dict with dot-prefix grouping)
    - presets: Sensitivity preset overrides
    - gui: GUI rendering hints
    """

    kind: Literal[ConfigKind.TOOL] = Field(
        default=ConfigKind.TOOL,
        description="Must be 'Tool' for tool configs",
    )

    # Tool type for categorization
    tool_type: str = Field(
        default="",
        description="Tool category (e.g., 'scene_detection', 'vad', 'post_processing')",
    )

    contract: Dict[str, Any] = Field(
        default_factory=lambda: {"input": {}, "output": {}},
        description="Input/output interface contract",
    )

    spec: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters as flat key-value pairs",
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

    # Implementation details
    provider: Dict[str, Any] = Field(
        default_factory=dict,
        description="Implementation provider (module, class)",
    )

    @field_validator("contract")
    @classmethod
    def validate_contract(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure contract has input and output sections."""
        if "input" not in v:
            v["input"] = {}
        if "output" not in v:
            v["output"] = {}
        return v

    @field_validator("presets")
    @classmethod
    def validate_presets(cls, v: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Ensure standard presets exist."""
        standard_presets = {"conservative", "balanced", "aggressive"}
        for preset in standard_presets:
            if preset not in v:
                v[preset] = {}
        return v

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
        """
        result = dict(self.spec)

        # Apply preset
        preset = self.presets.get(sensitivity, {})
        result.update(preset)

        # Apply user overrides
        if overrides:
            result.update(overrides)

        return result

    def get_input_contract(self) -> Dict[str, str]:
        """Get the input contract specification."""
        return self.contract.get("input", {})

    def get_output_contract(self) -> Dict[str, str]:
        """Get the output contract specification."""
        return self.contract.get("output", {})

    def validate_input(self, input_data: Dict[str, Any]) -> List[str]:
        """
        Validate input data against contract.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        input_contract = self.get_input_contract()

        for param, expected_type in input_contract.items():
            if param not in input_data:
                errors.append(f"Missing required input: {param}")
            # Type checking could be added here

        return errors

    def get_gui_schema(self) -> Dict[str, Any]:
        """
        Get complete GUI schema for this tool.

        Returns structure for frontend rendering.
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
                param["gui"] = self._infer_gui_hint(key, value).model_dump()
            parameters.append(param)

        return {
            "tool_name": self.metadata.name,
            "display_name": self.metadata.displayName,
            "description": self.metadata.description,
            "tool_type": self.tool_type,
            "parameters": parameters,
            "presets": list(self.presets.keys()),
            "contract": self.contract,
        }

    def _infer_gui_hint(self, key: str, value: Any) -> GUIHint:
        """Infer GUI hint from parameter key and value type."""
        from .base import GUIWidget

        if isinstance(value, bool):
            widget = GUIWidget.CHECKBOX
        elif isinstance(value, int):
            widget = GUIWidget.SPINNER
        elif isinstance(value, float):
            widget = GUIWidget.NUMBER
        else:
            widget = GUIWidget.TEXT

        parts = key.split(".")
        group = parts[0] if len(parts) >= 2 else "general"
        label = parts[-1].replace("_", " ").title()

        return GUIHint(widget=widget, label=label, group=group)
