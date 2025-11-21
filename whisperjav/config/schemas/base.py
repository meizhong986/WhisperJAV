"""
Base configuration classes for WhisperJAV Configuration System v2.0.

This module provides the foundation for type-safe configuration management:
- BaseConfig: Pydantic base model with None-value handling
- Sensitivity: Transcription sensitivity levels
- Backend: ASR backend types
"""

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict


class Sensitivity(str, Enum):
    """Transcription sensitivity levels."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class Backend(str, Enum):
    """ASR backend types."""
    STABLE_TS = "stable-ts"
    WHISPER = "whisper"
    FASTER_WHISPER = "faster-whisper"


class BaseConfig(BaseModel):
    """
    Base configuration with common settings for all config schemas.

    Features:
    - Strict validation (extra fields forbidden to catch typos)
    - Re-validation on attribute assignment
    - Enum values serialized as strings
    - None-value removal for library compatibility
    """

    model_config = ConfigDict(
        extra="forbid",           # Catch typos immediately
        validate_assignment=True, # Re-validate on attribute change
        use_enum_values=True,     # Serialize enums as strings
        populate_by_name=True,    # Allow field aliases
    )

    def model_dump_without_none(self, **kwargs) -> Dict[str, Any]:
        """
        Export dict without None values.

        This is critical for compatibility with ASR libraries (stable-ts,
        faster-whisper) that crash when receiving None for optional parameters.

        Replaces TranscriptionTuner._remove_none_values().
        """
        data = self.model_dump(**kwargs)
        return self._remove_none_recursive(data)

    @staticmethod
    def _remove_none_recursive(data: Any) -> Any:
        """Recursively remove None values from nested structures."""
        if isinstance(data, dict):
            return {
                k: BaseConfig._remove_none_recursive(v)
                for k, v in data.items()
                if v is not None
            }
        elif isinstance(data, list):
            return [
                BaseConfig._remove_none_recursive(v)
                for v in data
                if v is not None
            ]
        return data
