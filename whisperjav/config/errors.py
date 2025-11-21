"""
Custom exceptions for WhisperJAV Configuration System v2.0.

These exceptions provide clear, actionable error messages for configuration issues.
"""

from typing import Any, List


class ConfigurationError(Exception):
    """Base exception for all configuration errors."""
    pass


class ConfigValidationError(ConfigurationError):
    """
    Raised when configuration validation fails.

    Provides context about which field failed and what value was provided.
    """

    def __init__(self, message: str, field: str = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)

    def __str__(self):
        if self.field:
            return f"Validation error for '{self.field}': {self.args[0]} (got: {self.value!r})"
        return self.args[0]


class UnknownComponentError(ConfigurationError):
    """
    Raised when an unknown component (pipeline, model, VAD) is referenced.

    Provides list of available options for easy correction.
    """

    def __init__(self, component_type: str, name: str, available: List[str]):
        self.component_type = component_type
        self.name = name
        self.available = available
        super().__init__(
            f"Unknown {component_type}: '{name}'. "
            f"Available options: {', '.join(sorted(available))}"
        )


class IncompatibleComponentError(ConfigurationError):
    """
    Raised when components are incompatible with each other.

    For example, certain VAD engines only work with specific ASR backends.
    """

    def __init__(self, component1: str, component2: str, reason: str = None):
        self.component1 = component1
        self.component2 = component2
        msg = f"'{component1}' is not compatible with '{component2}'"
        if reason:
            msg += f". {reason}"
        super().__init__(msg)


class MigrationError(ConfigurationError):
    """Raised when configuration migration fails."""
    pass


class PersistenceError(ConfigurationError):
    """Raised when saving/loading configuration fails."""
    pass
