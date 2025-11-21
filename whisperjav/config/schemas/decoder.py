"""
Decoder options schema for WhisperJAV.

Common decoder parameters for transcription output control.
"""

from typing import List, Literal, Optional, Union

from pydantic import Field

from .base import BaseConfig


class DecoderOptions(BaseConfig):
    """
    Common decoder parameters.

    Maps to 'common_decoder_options' in asr_config.json.
    These parameters control decoding behavior and output format.
    """

    task: Literal["transcribe", "translate"] = Field(
        default="transcribe",
        description="Task to perform."
    )
    language: str = Field(
        default="ja",
        description="Target language code."
    )
    best_of: int = Field(
        ge=1, le=10,
        description="Number of candidates for best-of sampling."
    )
    beam_size: int = Field(
        ge=1, le=10,
        description="Beam search width."
    )
    patience: float = Field(
        ge=0.0, le=5.0,
        description="Beam search patience factor."
    )
    length_penalty: Optional[float] = Field(
        default=None,
        description="Exponential length penalty."
    )
    prefix: Optional[str] = Field(
        default=None,
        description="Text prefix for decoder."
    )
    suppress_tokens: Optional[Union[int, List[int]]] = Field(
        default=None,
        description="Token IDs to suppress during decoding."
    )
    suppress_blank: bool = Field(
        default=True,
        description="Suppress blank tokens at start."
    )
    without_timestamps: bool = Field(
        default=False,
        description="Disable timestamp generation."
    )
    max_initial_timestamp: Optional[float] = Field(
        default=None,
        description="Maximum initial timestamp value."
    )
