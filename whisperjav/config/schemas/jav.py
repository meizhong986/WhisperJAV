"""
JAV-specific audio processing configuration for WhisperJAV.

These settings optimize processing for the unique characteristics
of Japanese Adult Video audio: overlapping dialogue, high background
noise, and quiet speech patterns.
"""

import logging
from typing import Literal

from pydantic import Field

from .base import BaseConfig

logger = logging.getLogger(__name__)


class JAVAudioConfig(BaseConfig):
    """
    JAV-specific audio processing settings.

    Optimizes VAD and processing for challenging audio characteristics
    commonly found in JAV content.
    """

    background_noise_profile: Literal["low", "medium", "high"] = Field(
        default="high",
        description="Expected background noise level."
    )
    overlapping_dialogue_strategy: Literal["split", "merge", "preserve"] = Field(
        default="split",
        description="How to handle overlapping speakers."
    )
    quiet_speech_boost: bool = Field(
        default=True,
        description="Boost sensitivity for whispered/quiet dialogue."
    )
    moaning_filter_enabled: bool = Field(
        default=True,
        description="Filter non-speech vocalizations."
    )
    scene_transition_detection: bool = Field(
        default=True,
        description="Detect scene transitions for better segmentation."
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.background_noise_profile == "high":
            logger.debug(
                "High background noise profile selected - "
                "VAD sensitivity adjustments recommended"
            )

    def get_vad_adjustments(self) -> dict:
        """
        Get VAD parameter adjustments based on JAV settings.

        Returns:
            Dictionary of parameter adjustments (deltas to apply)
        """
        adjustments = {}

        if self.background_noise_profile == "high":
            adjustments["threshold"] = -0.05  # Lower threshold
            adjustments["speech_pad_ms"] = 100  # More padding

        if self.quiet_speech_boost:
            adjustments["min_speech_duration_ms"] = -50  # Shorter min
            adjustments["neg_threshold"] = -0.05

        return adjustments
