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


class RegroupMode(str, Enum):
    """
    Subtitle regrouping strategy applied during reconstruction.

    Controls how raw word-level output from the aligner or VAD framing
    is split/merged into subtitle lines.
    """

    STANDARD = "standard"
    """Full REGROUP_JAV (Branch A) or REGROUP_VAD_ONLY (Branch B).
    Gap-split, fragment merge, punctuation-split, duration/char caps."""

    SENTENCE_ONLY = "sentence_only"
    """Punctuation-split + safety caps only. No gap heuristics (sg/mg removed).
    Preserves natural sentence boundaries without gap-based manipulation."""

    OFF = "off"
    """No regrouping. Each word becomes its own subtitle segment.
    For analysis, debugging, or external post-processing."""


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
class StepDownConfig:
    """Configuration for step-down retry on alignment collapse.

    Step-down is OPTIONAL.  When disabled (enabled=False), the orchestrator
    skips the retry pass and falls through to proportional recovery directly.
    Users control this via --no-step-down or --step-down-attempts 0.
    """

    enabled: bool = True
    fallback_max_group_s: float = 6.0
    max_retries: int = 1


@dataclass
class HardeningConfig:
    """Configuration for the post-reconstruction hardening stage."""

    timestamp_mode: TimestampMode = TimestampMode.ALIGNER_WITH_INTERPOLATION
    regroup_mode: RegroupMode = RegroupMode.STANDARD
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


@dataclass
class SceneDiagnostics:
    """Canonical per-scene diagnostics (schema 2.0.0).

    Emitted by the orchestrator for every processed scene.  Provides
    complete visibility into framing, alignment, sentinel recovery,
    timestamp resolution, hardening, and step-down decisions.
    """

    schema_version: str = "2.0.0"
    scene_index: int = 0
    scene_duration_sec: float = 0.0
    framer_backend: str = ""
    frame_count: int = 0
    word_count: int = 0
    segment_count: int = 0
    sentinel_status: str = "N/A"
    sentinel_triggers: list = field(default_factory=list)
    sentinel_recovery: Optional[dict] = None
    timing_aligner_native: int = 0
    timing_interpolated: int = 0
    timing_vad_fallback: int = 0
    timing_total_segments: int = 0
    hardening_clamped: int = 0
    hardening_sorted: bool = False
    stepdown: Optional[dict] = None  # {"attempted": bool, "enabled": bool, "improved": bool}
    vad_regions: Optional[list] = None
    group_details: Optional[list] = None
    error: Optional[str] = None
