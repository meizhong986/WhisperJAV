"""
Tests for base configuration classes.
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from whisperjav.config.schemas import BaseConfig, Backend, Sensitivity
from whisperjav.config.errors import (
    ConfigurationError,
    ConfigValidationError,
    UnknownComponentError,
    IncompatibleComponentError,
)


class TestSensitivityEnum:
    """Test Sensitivity enum values."""

    def test_all_values_exist(self):
        """Test all three sensitivity levels exist."""
        assert Sensitivity.CONSERVATIVE.value == "conservative"
        assert Sensitivity.BALANCED.value == "balanced"
        assert Sensitivity.AGGRESSIVE.value == "aggressive"

    def test_string_conversion(self):
        """Test enum can be created from string."""
        assert Sensitivity("conservative") == Sensitivity.CONSERVATIVE
        assert Sensitivity("balanced") == Sensitivity.BALANCED
        assert Sensitivity("aggressive") == Sensitivity.AGGRESSIVE

    def test_invalid_value_raises(self):
        """Test invalid sensitivity raises ValueError."""
        with pytest.raises(ValueError):
            Sensitivity("invalid")


class TestBackendEnum:
    """Test Backend enum values."""

    def test_all_values_exist(self):
        """Test all backend types exist."""
        assert Backend.STABLE_TS.value == "stable-ts"
        assert Backend.WHISPER.value == "whisper"
        assert Backend.FASTER_WHISPER.value == "faster-whisper"

    def test_string_conversion(self):
        """Test enum can be created from string."""
        assert Backend("stable-ts") == Backend.STABLE_TS
        assert Backend("whisper") == Backend.WHISPER
        assert Backend("faster-whisper") == Backend.FASTER_WHISPER


class TestBaseConfig:
    """Test BaseConfig base model."""

    def test_extra_fields_forbidden(self):
        """Test that extra fields raise validation error (catches typos)."""
        class TestConfig(BaseConfig):
            field1: str = "value"

        # Valid
        config = TestConfig(field1="test")
        assert config.field1 == "test"

        # Extra field should raise
        with pytest.raises(PydanticValidationError) as exc_info:
            TestConfig(field1="test", typo_field="oops")
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_model_dump_without_none_simple(self):
        """Test None values are removed from export."""
        from typing import Optional

        class TestConfig(BaseConfig):
            required: str
            optional: Optional[str] = None

        config = TestConfig(required="value", optional=None)
        result = config.model_dump_without_none()

        assert "required" in result
        assert "optional" not in result
        assert result == {"required": "value"}

    def test_model_dump_without_none_nested(self):
        """Test None removal works recursively in nested dicts."""
        from typing import Any, Dict, Optional

        class TestConfig(BaseConfig):
            data: Dict[str, Any]

        config = TestConfig(data={
            "a": 1,
            "b": None,
            "c": {
                "d": "value",
                "e": None
            }
        })

        result = config.model_dump_without_none()
        assert result == {
            "data": {
                "a": 1,
                "c": {
                    "d": "value"
                }
            }
        }

    def test_model_dump_without_none_list(self):
        """Test None removal works in lists."""
        from typing import Any, List

        class TestConfig(BaseConfig):
            items: List[Any]

        # Note: Pydantic will include None in lists by default
        config = TestConfig(items=[1, None, 3])
        result = config.model_dump_without_none()

        # None values should be filtered out of lists
        assert result == {"items": [1, 3]}

    def test_validate_assignment(self):
        """Test that assignment re-validates."""
        from pydantic import Field

        class TestConfig(BaseConfig):
            value: int = Field(ge=0, le=10)

        config = TestConfig(value=5)

        # Valid change
        config.value = 8
        assert config.value == 8

        # Invalid change should raise
        with pytest.raises(PydanticValidationError):
            config.value = 15

    def test_enum_serialization(self):
        """Test enums are serialized as strings."""
        class TestConfig(BaseConfig):
            sensitivity: Sensitivity
            backend: Backend

        config = TestConfig(
            sensitivity=Sensitivity.AGGRESSIVE,
            backend=Backend.FASTER_WHISPER
        )

        result = config.model_dump()
        assert result["sensitivity"] == "aggressive"
        assert result["backend"] == "faster-whisper"


class TestConfigurationErrors:
    """Test custom exception classes."""

    def test_config_validation_error_with_field(self):
        """Test validation error message includes field and value."""
        error = ConfigValidationError(
            "must be less than 10",
            field="threshold",
            value=15
        )
        assert "threshold" in str(error)
        assert "15" in str(error)
        assert "must be less than 10" in str(error)

    def test_config_validation_error_without_field(self):
        """Test validation error works without field context."""
        error = ConfigValidationError("General validation failure")
        assert str(error) == "General validation failure"

    def test_unknown_component_error(self):
        """Test unknown component error shows available options."""
        error = UnknownComponentError(
            "pipeline",
            "unknwon",
            ["faster", "fast", "balanced", "fidelity"]
        )
        assert "unknwon" in str(error)
        assert "pipeline" in str(error)
        assert "balanced" in str(error)
        assert "faster" in str(error)

    def test_incompatible_component_error(self):
        """Test incompatible component error message."""
        error = IncompatibleComponentError(
            "stable_ts_vad",
            "whisper",
            "Use silero_vad instead"
        )
        assert "stable_ts_vad" in str(error)
        assert "whisper" in str(error)
        assert "silero_vad" in str(error)

    def test_exception_hierarchy(self):
        """Test all errors inherit from ConfigurationError."""
        assert issubclass(ConfigValidationError, ConfigurationError)
        assert issubclass(UnknownComponentError, ConfigurationError)
        assert issubclass(IncompatibleComponentError, ConfigurationError)
