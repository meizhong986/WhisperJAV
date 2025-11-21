"""
UI preferences schema for WhisperJAV.

User interface preferences for console and GUI output.
"""

from typing import Literal

from pydantic import Field

from .base import BaseConfig


class UIPreferences(BaseConfig):
    """
    User interface preferences (14 parameters).

    Maps to 'ui_preferences' in asr_config.json.
    """

    console_verbosity: Literal["summary", "verbose", "quiet"] = Field(
        default="summary",
        description="Console output verbosity level."
    )
    progress_batch_size: int = Field(
        default=10,
        ge=1,
        description="Number of items per progress update."
    )
    show_scene_details: bool = Field(
        default=False,
        description="Show detailed scene detection output."
    )
    max_console_lines: int = Field(
        default=1000,
        ge=1,
        description="Maximum lines in console output."
    )
    auto_scroll: bool = Field(
        default=True,
        description="Auto-scroll console output."
    )
    show_timestamps: bool = Field(
        default=False,
        description="Show timestamps in console output."
    )
    theme: str = Field(
        default="default",
        description="UI theme name."
    )
    last_mode: str = Field(
        default="fidelity",
        description="Last used pipeline mode."
    )
    last_sensitivity: str = Field(
        default="conservative",
        description="Last used sensitivity level."
    )
    last_language: str = Field(
        default="japanese",
        description="Last used language."
    )
    show_console: bool = Field(
        default=True,
        description="Show console panel in GUI."
    )
    adaptive_classification: bool = Field(
        default=True,
        description="Enable adaptive audio classification."
    )
    adaptive_audio_enhancement: bool = Field(
        default=True,
        description="Enable adaptive audio enhancement."
    )
    smart_postprocessing: bool = Field(
        default=True,
        description="Enable smart post-processing."
    )
