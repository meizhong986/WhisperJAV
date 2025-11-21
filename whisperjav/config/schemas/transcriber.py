"""
Transcriber options schema for WhisperJAV.

Common transcriber parameters shared across backends.
"""

from typing import List, Optional, Union

from pydantic import Field

from .base import BaseConfig


class TranscriberOptions(BaseConfig):
    """
    Common transcriber parameters.

    Maps to 'common_transcriber_options' in asr_config.json.
    These parameters control the core transcription behavior.
    """

    temperature: Union[float, List[float]] = Field(
        description="Sampling temperature(s). Can be single value or fallback list."
    )
    compression_ratio_threshold: float = Field(
        ge=1.0, le=5.0,
        description="Skip segments exceeding this compression ratio."
    )
    logprob_threshold: float = Field(
        le=0.0,
        description="Log probability threshold for word acceptance."
    )
    logprob_margin: float = Field(
        ge=0.0,
        description="Margin for log probability comparison."
    )
    drop_nonverbal_vocals: bool = Field(
        description="Whether to drop non-verbal vocalizations."
    )
    no_speech_threshold: float = Field(
        ge=0.0, le=1.0,
        description="Threshold for no-speech probability."
    )
    condition_on_previous_text: bool = Field(
        description="Use previous output as context."
    )
    initial_prompt: Optional[str] = Field(
        default=None,
        description="Initial prompt text for context."
    )
    word_timestamps: bool = Field(
        description="Generate word-level timestamps."
    )
    prepend_punctuations: Optional[str] = Field(
        default=None,
        description="Punctuation to prepend to words."
    )
    append_punctuations: Optional[str] = Field(
        default=None,
        description="Punctuation to append to words."
    )
    clip_timestamps: Optional[str] = Field(
        default=None,
        description="Timestamp clipping mode."
    )
