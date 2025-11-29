"""
Custom Exceptions for WhisperJAV Configuration System v4.0.

Design Principles:
- Every exception provides actionable guidance
- Error messages include context (what was expected, what was provided)
- Exceptions are hierarchical for flexible catching
- All exceptions are serializable for GUI/API error reporting
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class V4ConfigError(Exception):
    """
    Base exception for all v4 configuration errors.

    All v4 config exceptions inherit from this class, allowing
    callers to catch all config-related errors with a single except clause.

    Attributes:
        message: Human-readable error description
        context: Additional context as key-value pairs
        suggestion: Actionable suggestion to fix the error
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        self.message = message
        self.context = context or {}
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the complete error message."""
        parts = [self.message]

        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API/GUI serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "suggestion": self.suggestion,
        }


class SchemaValidationError(V4ConfigError):
    """
    Raised when YAML content fails schema validation.

    Provides detailed information about which field failed validation
    and what the expected type/constraints were.
    """

    def __init__(
        self,
        message: str,
        field_path: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Any = None,
        constraints: Optional[Dict[str, Any]] = None,
        source_file: Optional[Path] = None,
    ):
        self.field_path = field_path
        self.expected_type = expected_type
        self.actual_value = actual_value
        self.constraints = constraints or {}
        self.source_file = source_file

        context = {}
        if field_path:
            context["field"] = field_path
        if expected_type:
            context["expected_type"] = expected_type
        if actual_value is not None:
            context["actual_value"] = actual_value
        if constraints:
            context["constraints"] = constraints
        if source_file:
            context["file"] = str(source_file)

        suggestion = self._generate_suggestion()

        super().__init__(message, context=context, suggestion=suggestion)

    def _generate_suggestion(self) -> Optional[str]:
        """Generate actionable suggestion based on error type."""
        if self.expected_type and self.actual_value is not None:
            actual_type = type(self.actual_value).__name__
            return f"Change value from {actual_type} to {self.expected_type}"

        if self.constraints:
            if "ge" in self.constraints:
                return f"Value must be >= {self.constraints['ge']}"
            if "le" in self.constraints:
                return f"Value must be <= {self.constraints['le']}"

        return None


class ModelNotFoundError(V4ConfigError):
    """
    Raised when a requested model configuration is not found.

    Provides list of available models in the same ecosystem.
    """

    def __init__(
        self,
        model_name: str,
        ecosystem: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.ecosystem = ecosystem
        self.available_models = available_models or []

        context = {"model": model_name}
        if ecosystem:
            context["ecosystem"] = ecosystem
        if available_models:
            context["available"] = available_models

        suggestion = None
        if available_models:
            # Find closest match
            closest = self._find_closest(model_name, available_models)
            if closest:
                suggestion = f"Did you mean '{closest}'?"
            else:
                suggestion = f"Available models: {', '.join(available_models)}"

        super().__init__(
            f"Model '{model_name}' not found",
            context=context,
            suggestion=suggestion,
        )

    @staticmethod
    def _find_closest(name: str, options: List[str]) -> Optional[str]:
        """Find closest matching option using simple substring matching."""
        name_lower = name.lower()
        for opt in options:
            if name_lower in opt.lower() or opt.lower() in name_lower:
                return opt
        return None


class EcosystemNotFoundError(V4ConfigError):
    """
    Raised when a requested ecosystem is not found.

    Provides list of available ecosystems.
    """

    def __init__(
        self,
        ecosystem_name: str,
        available_ecosystems: Optional[List[str]] = None,
    ):
        self.ecosystem_name = ecosystem_name
        self.available_ecosystems = available_ecosystems or []

        context = {"ecosystem": ecosystem_name}
        if available_ecosystems:
            context["available"] = available_ecosystems

        suggestion = None
        if available_ecosystems:
            suggestion = f"Available ecosystems: {', '.join(available_ecosystems)}"

        super().__init__(
            f"Ecosystem '{ecosystem_name}' not found",
            context=context,
            suggestion=suggestion,
        )


class ToolNotFoundError(V4ConfigError):
    """
    Raised when a requested tool configuration is not found.

    Provides list of available tools of the same type.
    """

    def __init__(
        self,
        tool_name: str,
        tool_type: Optional[str] = None,
        available_tools: Optional[List[str]] = None,
    ):
        self.tool_name = tool_name
        self.tool_type = tool_type
        self.available_tools = available_tools or []

        context = {"tool": tool_name}
        if tool_type:
            context["type"] = tool_type
        if available_tools:
            context["available"] = available_tools

        suggestion = None
        if available_tools:
            suggestion = f"Available tools: {', '.join(available_tools)}"

        super().__init__(
            f"Tool '{tool_name}' not found",
            context=context,
            suggestion=suggestion,
        )


class YAMLParseError(V4ConfigError):
    """
    Raised when YAML parsing fails.

    Provides line number and column if available.
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[Path] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
        raw_error: Optional[str] = None,
    ):
        self.file_path = file_path
        self.line = line
        self.column = column
        self.raw_error = raw_error

        context = {}
        if file_path:
            context["file"] = str(file_path)
        if line is not None:
            context["line"] = line
        if column is not None:
            context["column"] = column

        suggestion = "Check YAML syntax: indentation, colons, quotes"
        if line:
            suggestion = f"Check line {line} for syntax errors"

        super().__init__(message, context=context, suggestion=suggestion)


class IncompatibleVersionError(V4ConfigError):
    """
    Raised when schema version in YAML is incompatible with loader.

    Provides migration guidance when possible.
    """

    def __init__(
        self,
        found_version: str,
        supported_versions: List[str],
        file_path: Optional[Path] = None,
    ):
        self.found_version = found_version
        self.supported_versions = supported_versions
        self.file_path = file_path

        context = {
            "found_version": found_version,
            "supported_versions": supported_versions,
        }
        if file_path:
            context["file"] = str(file_path)

        suggestion = f"Update schemaVersion to one of: {', '.join(supported_versions)}"

        super().__init__(
            f"Schema version '{found_version}' is not supported",
            context=context,
            suggestion=suggestion,
        )


class PresetNotFoundError(V4ConfigError):
    """
    Raised when a sensitivity preset is not found.

    Provides list of available presets.
    """

    def __init__(
        self,
        preset_name: str,
        model_name: Optional[str] = None,
        available_presets: Optional[List[str]] = None,
    ):
        self.preset_name = preset_name
        self.model_name = model_name
        self.available_presets = available_presets or []

        context = {"preset": preset_name}
        if model_name:
            context["model"] = model_name
        if available_presets:
            context["available"] = available_presets

        suggestion = None
        if available_presets:
            suggestion = f"Available presets: {', '.join(available_presets)}"

        super().__init__(
            f"Preset '{preset_name}' not found",
            context=context,
            suggestion=suggestion,
        )


class CircularDependencyError(V4ConfigError):
    """
    Raised when circular 'extends' references are detected.

    Provides the dependency chain that caused the cycle.
    """

    def __init__(
        self,
        chain: List[str],
        file_path: Optional[Path] = None,
    ):
        self.chain = chain
        self.file_path = file_path

        context = {"dependency_chain": " -> ".join(chain)}
        if file_path:
            context["file"] = str(file_path)

        suggestion = "Remove circular 'extends' reference"

        super().__init__(
            "Circular dependency detected in 'extends' chain",
            context=context,
            suggestion=suggestion,
        )


class MergeConflictError(V4ConfigError):
    """
    Raised when config merging produces conflicts.

    Provides details about which keys conflicted.
    """

    def __init__(
        self,
        message: str,
        conflicting_keys: Optional[List[str]] = None,
        base_file: Optional[Path] = None,
        override_file: Optional[Path] = None,
    ):
        self.conflicting_keys = conflicting_keys or []
        self.base_file = base_file
        self.override_file = override_file

        context = {}
        if conflicting_keys:
            context["conflicting_keys"] = conflicting_keys
        if base_file:
            context["base_file"] = str(base_file)
        if override_file:
            context["override_file"] = str(override_file)

        suggestion = "Use explicit override syntax or resolve conflicting keys"

        super().__init__(message, context=context, suggestion=suggestion)
