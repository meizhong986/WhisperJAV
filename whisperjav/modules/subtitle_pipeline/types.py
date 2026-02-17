"""
Shared data types for the Decoupled Subtitle Pipeline.

All timestamps are in scene-relative seconds unless otherwise noted.
Group-relative coordinates are an internal concern of specific framers
(e.g., vad-grouped) and are never exposed in these types.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Word-level timestamp
# ---------------------------------------------------------------------------


@dataclass
class WordTimestamp:
    """A single word with its time boundaries."""

    word: str
    start: float  # seconds, scene-relative
    end: float  # seconds, scene-relative


# ---------------------------------------------------------------------------
# Temporal framing types
# ---------------------------------------------------------------------------


@dataclass
class TemporalFrame:
    """
    A dialogue-level time window produced by a TemporalFramer.

    Represents a segment of audio within which text generation and/or
    alignment will operate.  May carry pre-existing text from the framing
    source (e.g., an existing SRT or a Whisper draft).
    """

    start: float  # seconds, scene-relative
    end: float  # seconds, scene-relative
    text: Optional[str] = None  # pre-existing text (SRT, Whisper draft)
    confidence: Optional[float] = None  # source confidence, if available
    source: str = ""  # backend that produced this frame

    @property
    def duration(self) -> float:
        """Duration of this frame in seconds."""
        return max(0.0, self.end - self.start)


@dataclass
class FramingResult:
    """Output of a TemporalFramer.frame() call."""

    frames: list[TemporalFrame]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def total_duration(self) -> float:
        return sum(f.duration for f in self.frames)


# ---------------------------------------------------------------------------
# Text generation types
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """Output of a TextGenerator for a single audio segment."""

    text: str
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Alignment types
# ---------------------------------------------------------------------------


@dataclass
class AlignmentResult:
    """Output of a TextAligner for a single audio segment."""

    words: list[WordTimestamp]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def char_count(self) -> int:
        return sum(len(w.word) for w in self.words)

    @property
    def span_sec(self) -> float:
        """Time span from first word start to last word end."""
        if not self.words:
            return 0.0
        return max(0.0, self.words[-1].end - self.words[0].start)


# ---------------------------------------------------------------------------
# Timestamp resolution mode
# ---------------------------------------------------------------------------


class TimestampMode(str, Enum):
    """
    Timestamp resolution strategy applied during hardening.

    Controls how null/missing timestamps in the aligner output are resolved
    before the result is passed to SRT generation.
    """

    ALIGNER_WITH_INTERPOLATION = "aligner_interpolation"
    """Default. Keep valid aligner timestamps; mathematically interpolate
    null-timestamp segments based on character length between anchors."""

    ALIGNER_WITH_VAD_FALLBACK = "aligner_vad_fallback"
    """Keep valid aligner timestamps; distribute null-timestamp segments
    proportionally across the full duration (legacy VAD fallback)."""

    ALIGNER_ONLY = "aligner_only"
    """Keep aligner timestamps as-is.  Null timestamps remain as zeros.
    Useful for diagnosing aligner failures without masking."""

    VAD_ONLY = "vad_only"
    """Discard aligner timestamps entirely.  Distribute all segments
    proportionally by character count across the full duration."""


# ---------------------------------------------------------------------------
# Hardening configuration
# ---------------------------------------------------------------------------


@dataclass
class HardeningConfig:
    """Configuration for the post-reconstruction hardening stage."""

    timestamp_mode: TimestampMode = TimestampMode.ALIGNER_WITH_INTERPOLATION
    scene_duration_sec: float = 0.0
    speech_regions: Optional[list[tuple[float, float]]] = None


@dataclass
class HardeningDiagnostics:
    """Diagnostics emitted by harden_scene_result()."""

    segment_count: int = 0
    interpolated_count: int = 0
    fallback_count: int = 0
    clamped_count: int = 0
    sorted: bool = False
    timestamp_mode: str = ""
