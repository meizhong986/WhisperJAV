"""
Decoupled Subtitle Pipeline — model-agnostic subtitle generation.

Separates temporal framing, text generation, text cleaning, and text alignment
into independent, swappable protocol domains. Enables aligner-free workflows
and cross-model experimentation.

See docs/architecture/ADR-006-decoupled-subtitle-pipeline.md for full design.

Public API:
    Types:      WordTimestamp, TemporalFrame, FramingResult,
                TranscriptionResult, AlignmentResult,
                TimestampMode, HardeningConfig, HardeningDiagnostics
    Protocols:  TemporalFramer, TextGenerator, TextCleaner, TextAligner
    Functions:  harden_scene_result, reconstruct_from_words
"""

from whisperjav.modules.subtitle_pipeline.hardening import harden_scene_result
from whisperjav.modules.subtitle_pipeline.protocols import (
    TemporalFramer,
    TextAligner,
    TextCleaner,
    TextGenerator,
)
from whisperjav.modules.subtitle_pipeline.reconstruction import reconstruct_from_words
from whisperjav.modules.subtitle_pipeline.types import (
    AlignmentResult,
    FramingResult,
    HardeningConfig,
    HardeningDiagnostics,
    TemporalFrame,
    TimestampMode,
    TranscriptionResult,
    WordTimestamp,
)

__all__ = [
    # Types
    "WordTimestamp",
    "TemporalFrame",
    "FramingResult",
    "TranscriptionResult",
    "AlignmentResult",
    "TimestampMode",
    "HardeningConfig",
    "HardeningDiagnostics",
    # Protocols
    "TemporalFramer",
    "TextGenerator",
    "TextCleaner",
    "TextAligner",
    # Functions
    "harden_scene_result",
    "reconstruct_from_words",
]
