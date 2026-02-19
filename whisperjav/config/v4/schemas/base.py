"""
Base Schema Definitions for WhisperJAV v4 Configuration.

Defines the foundational structures used by all config types:
- ConfigBase: Common fields for all config files
- MetadataBlock: Standard metadata structure
- GUIWidget/GUIHint: GUI auto-generation hints

Design Principles:
- All fields have explicit types and descriptions
- Required vs optional is clearly defined
- Validation constraints are specified at schema level
- GUI hints are separate from parameter values
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================


class GUIWidget(str, Enum):
    """Available GUI widget types for parameter display."""

    TEXT = "text"  # Single-line text input
    TEXTAREA = "textarea"  # Multi-line text input
    NUMBER = "number"  # Numeric input with optional step
    SLIDER = "slider"  # Range slider with min/max
    SPINNER = "spinner"  # Numeric spinner with arrows
    DROPDOWN = "dropdown"  # Dropdown select
    CHECKBOX = "checkbox"  # Boolean toggle
    TOGGLE = "toggle"  # Switch toggle
    COLOR = "color"  # Color picker
    FILE = "file"  # File path selector
    HIDDEN = "hidden"  # Not shown in GUI


class SchemaVersion(str, Enum):
    """Supported schema versions."""

    V1 = "v1"


class ConfigKind(str, Enum):
    """Types of configuration resources."""

    MODEL = "Model"
    ECOSYSTEM = "Ecosystem"
    TOOL = "Tool"
    PRESET = "Preset"
    PIPELINE = "Pipeline"


# =============================================================================
# GUI Hint Structures
# =============================================================================


class GUIHint(BaseModel):
    """
    GUI rendering hints for a single parameter.

    These hints enable automatic GUI generation from config schemas.
    The GUI reads these hints to determine how to display each parameter.
    """

    widget: GUIWidget = Field(
        default=GUIWidget.TEXT,
        description="Widget type to use for this parameter",
    )
    label: Optional[str] = Field(
        default=None,
        description="Display label (defaults to parameter name)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Help text shown on hover/tooltip",
    )
    placeholder: Optional[str] = Field(
        default=None,
        description="Placeholder text for input fields",
    )

    # Numeric constraints
    min: Optional[Union[int, float]] = Field(
        default=None,
        description="Minimum value for numeric inputs",
    )
    max: Optional[Union[int, float]] = Field(
        default=None,
        description="Maximum value for numeric inputs",
    )
    step: Optional[Union[int, float]] = Field(
        default=None,
        description="Step increment for numeric inputs",
    )

    # Dropdown options
    options: Optional[List[str]] = Field(
        default=None,
        description="Options for dropdown widget",
    )

    # Display options
    group: Optional[str] = Field(
        default=None,
        description="Group name for organizing related parameters",
    )
    order: Optional[int] = Field(
        default=None,
        description="Display order within group (lower = higher)",
    )
    advanced: bool = Field(
        default=False,
        description="If True, shown under 'Advanced' section",
    )
    readonly: bool = Field(
        default=False,
        description="If True, parameter cannot be edited",
    )
    visible: bool = Field(
        default=True,
        description="If False, parameter is hidden from GUI",
    )

    # Conditional display
    depends_on: Optional[str] = Field(
        default=None,
        description="Parameter name that controls visibility",
    )
    depends_value: Optional[Any] = Field(
        default=None,
        description="Value of depends_on that makes this visible",
    )


# =============================================================================
# Metadata Block
# =============================================================================


class MetadataBlock(BaseModel):
    """
    Standard metadata block for all config types.

    This block appears at the top of every YAML config file and
    provides identification and discovery information.
    """

    name: str = Field(
        ...,  # Required
        min_length=1,
        max_length=100,
        description="Unique identifier for this config (e.g., 'kotoba-whisper-v2')",
    )
    ecosystem: Optional[str] = Field(
        default=None,
        description="Parent ecosystem (for models)",
    )
    displayName: str = Field(
        ...,  # Required
        min_length=1,
        max_length=200,
        description="Human-readable display name",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Detailed description of the config",
    )
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Semantic version of this config",
    )
    tags: List[str] = Field(
        default_factory=list,
        max_length=20,
        description="Tags for search and categorization",
    )
    author: Optional[str] = Field(
        default=None,
        description="Author or maintainer",
    )
    deprecated: bool = Field(
        default=False,
        description="If True, this config is deprecated",
    )
    replacement: Optional[str] = Field(
        default=None,
        description="If deprecated, name of replacement config",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is a valid identifier."""
        # Allow alphanumeric, hyphens, and underscores
        import re

        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", v):
            raise ValueError(
                f"Name must start with letter and contain only "
                f"alphanumeric, hyphens, underscores. Got: {v}"
            )
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate tags are lowercase and valid."""
        import re

        validated = []
        for tag in v:
            tag_lower = tag.lower().strip()
            if not re.match(r"^[a-z0-9_-]+$", tag_lower):
                raise ValueError(f"Invalid tag format: {tag}")
            validated.append(tag_lower)
        return validated


# =============================================================================
# Config Base
# =============================================================================


class ConfigBase(BaseModel):
    """
    Base class for all v4 configuration schemas.

    Every YAML config file must have:
    - schemaVersion: Version of the schema format
    - kind: Type of configuration (Model, Ecosystem, Tool, Preset)
    - metadata: Standard metadata block

    The spec and gui blocks are defined in subclasses.
    """

    schemaVersion: SchemaVersion = Field(
        default=SchemaVersion.V1,
        description="Schema version for compatibility checking",
    )
    kind: ConfigKind = Field(
        ...,  # Required
        description="Type of configuration resource",
    )
    metadata: MetadataBlock = Field(
        ...,  # Required
        description="Identification and discovery metadata",
    )

    # Optional extends for inheritance
    extends: Optional[str] = Field(
        default=None,
        description="Name of config to extend/inherit from",
    )

    model_config = {
        "extra": "forbid",  # Reject unknown fields for safety
        "validate_assignment": True,  # Validate on attribute assignment
        "str_strip_whitespace": True,  # Strip whitespace from strings
    }

    def get_name(self) -> str:
        """Get the unique name of this config."""
        return self.metadata.name

    def get_display_name(self) -> str:
        """Get the human-readable display name."""
        return self.metadata.displayName

    def is_deprecated(self) -> bool:
        """Check if this config is deprecated."""
        return self.metadata.deprecated
