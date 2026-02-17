"""
Protocol definitions for the Decoupled Subtitle Pipeline.

Each protocol defines a swappable component contract.  Implementations are
registered in per-domain factories (framers/, generators/, aligners/, cleaners/).

These are runtime-checkable protocols — ``isinstance(obj, TextGenerator)``
works.  No base class inheritance required.
"""

from pathlib import Path
from typing import Any, Optional, runtime_checkable

import numpy as np
from typing_extensions import Protocol

from whisperjav.modules.subtitle_pipeline.types import (
    AlignmentResult,
    FramingResult,
    TranscriptionResult,
)

# ---------------------------------------------------------------------------
# TemporalFramer — "WHEN does dialogue happen?"
# ---------------------------------------------------------------------------


@runtime_checkable
class TemporalFramer(Protocol):
    """
    Produces dialogue-level temporal frames from scene audio.

    Frames establish time windows within which downstream components
    (TextGenerator, TextAligner) operate.  Frames may carry pre-existing
    text if the framing source provides it (e.g., SRT, Whisper draft).

    Coordinate system: all frame timestamps are scene-relative seconds.
    """

    def frame(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs: Any,
    ) -> FramingResult:
        """
        Frame the scene audio into dialogue-level time windows.

        Args:
            audio: Scene audio as numpy array (mono, any sample rate).
            sample_rate: Sample rate of the audio array.

        Returns:
            FramingResult with list of TemporalFrames and metadata.
        """
        ...

    def cleanup(self) -> None:
        """Release any resources held by this framer."""
        ...


# ---------------------------------------------------------------------------
# TextGenerator — "WHAT was said?"
# ---------------------------------------------------------------------------


@runtime_checkable
class TextGenerator(Protocol):
    """
    Generates text transcription from audio.

    Manages its own model lifecycle via load()/unload().  The orchestrator
    calls load() before generation and unload() after to enable VRAM swaps.
    """

    def generate(
        self,
        audio_path: Path,
        language: str = "ja",
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe a single audio file."""
        ...

    def generate_batch(
        self,
        audio_paths: list[Path],
        language: str = "ja",
        contexts: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[TranscriptionResult]:
        """Transcribe a batch of audio files."""
        ...

    def load(self) -> None:
        """Load the model into memory (GPU/CPU)."""
        ...

    def unload(self) -> None:
        """Unload the model and release VRAM/memory."""
        ...

    def cleanup(self) -> None:
        """Final cleanup — called when the generator is no longer needed."""
        ...


# ---------------------------------------------------------------------------
# TextCleaner — mid-pipeline text cleaning
# ---------------------------------------------------------------------------


@runtime_checkable
class TextCleaner(Protocol):
    """
    Cleans raw transcription text before alignment.

    Lightweight — no model, no load/unload.  Each ASR model may need its
    own cleaner tuned to that model's artifact characteristics.
    """

    def clean(self, text: str, **kwargs: Any) -> str:
        """Clean a single transcription string."""
        ...

    def clean_batch(self, texts: list[str], **kwargs: Any) -> list[str]:
        """Clean a batch of transcription strings."""
        ...


# ---------------------------------------------------------------------------
# TextAligner — "WHEN exactly was each word said?" (optional)
# ---------------------------------------------------------------------------


@runtime_checkable
class TextAligner(Protocol):
    """
    Aligns text to audio, producing word-level timestamps.

    Optional — the TemporalFramer may provide sufficiently accurate
    timestamps, making the aligner unnecessary.

    Manages its own model lifecycle via load()/unload().
    """

    def align(
        self,
        audio_path: Path,
        text: str,
        language: str = "ja",
        **kwargs: Any,
    ) -> AlignmentResult:
        """Align text to a single audio file."""
        ...

    def align_batch(
        self,
        audio_paths: list[Path],
        texts: list[str],
        language: str = "ja",
        **kwargs: Any,
    ) -> list[AlignmentResult]:
        """Align text to a batch of audio files."""
        ...

    def load(self) -> None:
        """Load the alignment model into memory (GPU/CPU)."""
        ...

    def unload(self) -> None:
        """Unload the alignment model and release VRAM/memory."""
        ...

    def cleanup(self) -> None:
        """Final cleanup — called when the aligner is no longer needed."""
        ...
