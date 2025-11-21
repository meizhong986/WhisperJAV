"""
Model configuration schemas for WhisperJAV.

Defines model and VAD engine configuration structures.
"""

from typing import List, Literal

from pydantic import Field

from .base import BaseConfig


class ModelConfig(BaseConfig):
    """
    Whisper model definition.

    Maps to the 'models' section in asr_config.json.
    """

    provider: Literal["openai_whisper", "faster_whisper"] = Field(
        description="ASR provider backend"
    )
    model_name: str = Field(
        description="Model identifier (e.g., 'large-v2', 'turbo')"
    )
    compute_type: Literal["float16", "float32", "int8_float16"] = Field(
        description="Computation precision type"
    )
    supported_tasks: List[Literal["transcribe", "translate"]] = Field(
        description="Tasks this model supports"
    )


class VADEngineConfig(BaseConfig):
    """
    VAD engine definition.

    Maps to the 'vad_engines' section in asr_config.json.
    """

    provider: Literal["silero"] = Field(
        description="VAD provider"
    )
    repo: str = Field(
        description="Repository path (e.g., 'snakers4/silero-vad:v4.0')"
    )


# Predefined models from asr_config.json
MODELS = {
    "whisper-turbo": ModelConfig(
        provider="openai_whisper",
        model_name="turbo",
        compute_type="float16",
        supported_tasks=["transcribe"]
    ),
    "whisper-large-v2": ModelConfig(
        provider="openai_whisper",
        model_name="large-v2",
        compute_type="float16",
        supported_tasks=["transcribe", "translate"]
    ),
    "faster-whisper-large-v2-int8": ModelConfig(
        provider="faster_whisper",
        model_name="large-v2",
        compute_type="int8_float16",
        supported_tasks=["transcribe", "translate"]
    ),
    "faster-whisper-large-v3": ModelConfig(
        provider="faster_whisper",
        model_name="large-v3",
        compute_type="float16",
        supported_tasks=["transcribe", "translate"]
    ),
}

# Predefined VAD engines from asr_config.json
VAD_ENGINES = {
    "silero-v4": VADEngineConfig(
        provider="silero",
        repo="snakers4/silero-vad:v4.0"
    ),
    "silero-v3.1": VADEngineConfig(
        provider="silero",
        repo="snakers4/silero-vad:v3.1"
    ),
    "silero-latest": VADEngineConfig(
        provider="silero",
        repo="snakers4/silero-vad"
    ),
}
