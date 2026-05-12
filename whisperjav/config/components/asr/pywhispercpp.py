"""
PyWhisperCpp ASR Component.

Whisper.cpp backend with Metal GPU acceleration on Apple Silicon.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from whisperjav.config.components.base import ASRComponent, register_asr


class PyWhisperCppOptions(BaseModel):
    """
    Whisper.cpp options for transcription.

    Simplified parameter set compared to faster-whisper, as whisper.cpp
    has fewer configurable options.
    """

    task: str = Field(
        "transcribe",
        description="Task: 'transcribe' or 'translate'"
    )
    language: str = Field(
        "ja",
        description="Language code for transcription"
    )
    beam_size: int = Field(
        5,
        ge=1, le=20,
        description="Beam size for decoding (whisper.cpp beam_search)"
    )
    temperature: float = Field(
        0.0,
        ge=0.0, le=1.0,
        description="Sampling temperature"
    )
    n_threads: int = Field(
        4,
        ge=1, le=16,
        description="CPU threads (ignored when Metal GPU active)"
    )
    max_len: int = Field(
        0,
        ge=0,
        description="Max segment length in characters (0 = no limit)"
    )
    max_tokens: int = Field(
        0,
        ge=0,
        description="Max tokens per segment (0 = no limit)"
    )
    suppress_blank: bool = Field(
        True,
        description="Suppress blank outputs at start"
    )
    suppress_non_speech_tokens: bool = Field(
        True,
        description="Suppress non-speech tokens"
    )


@register_asr
class PyWhisperCppASRComponent(ASRComponent):
    """Whisper.cpp ASR with Metal GPU acceleration."""

    # === Metadata ===
    name = "pywhispercpp"
    display_name = "WhisperCpp (Metal)"
    description = "Whisper.cpp with native Metal GPU on Apple Silicon. Fast on Mac."
    version = "1.0.0"
    tags = ["asr", "whisper", "whispercpp", "metal", "mac"]

    # === ASR-specific ===
    provider = "pywhispercpp"
    model_id = "large-v2"
    supported_tasks = ["transcribe", "translate"]
    compatible_vad = ["none", "auditok", "silero"]

    # === Compute ===
    default_device = "auto"  # Auto-detect Metal/CPU
    default_compute_type = "float16"

    # === Schema ===
    Options = PyWhisperCppOptions

    # === Presets ===
    presets = {
        "conservative": PyWhisperCppOptions(
            task="transcribe",
            language="ja",
            beam_size=3,
            temperature=0.0,
            n_threads=4,
            max_len=0,
            max_tokens=0,
            suppress_blank=True,
            suppress_non_speech_tokens=True,
        ),
        "balanced": PyWhisperCppOptions(
            task="transcribe",
            language="ja",
            beam_size=5,
            temperature=0.0,
            n_threads=4,
            max_len=0,
            max_tokens=0,
            suppress_blank=True,
            suppress_non_speech_tokens=True,
        ),
        "aggressive": PyWhisperCppOptions(
            task="transcribe",
            language="ja",
            beam_size=7,
            temperature=0.0,
            n_threads=4,
            max_len=0,
            max_tokens=0,
            suppress_blank=True,
            suppress_non_speech_tokens=True,
        ),
    }