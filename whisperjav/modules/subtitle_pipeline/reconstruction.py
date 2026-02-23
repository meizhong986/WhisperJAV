"""
WhisperResult reconstruction from word-level timestamps.

Converts raw word dicts (from alignment or temporal framing) into
stable_whisper.WhisperResult objects with sentence-level regrouping.

Regrouping uses a JAV-tuned algorithm instead of stable-ts's default ``'da'``.
The default ``'da'`` splits at any 0.5s gap, which fragments conversational
dialogue at natural thinking pauses.  The JAV-tuned algorithm:

    - Relaxes the gap-split threshold from 0.5s to 1.5s
    - Adds a merge-by-gap post-pass to rejoin punctuation-split fragments
    - Adds Japanese comma (、) to the comma-split set
    - Caps subtitle duration at 8 seconds

See QWEN-PIPELINE-REFERENCE.md §12 for the full rationale.

Audit H3 resolution:
    The ``suppress_silence`` parameter controls whether stable-ts adjusts
    timestamps based on audio silence detection.  For sentinel-recovered
    words (whose timestamps were carefully redistributed across speech
    regions), this should be False to avoid undoing the recovery.  For
    normal words, True is fine (improves timing accuracy).
"""

import re
from pathlib import Path
from typing import Any, Union

from whisperjav.utils.logger import logger

try:
    import stable_whisper
except ImportError:
    stable_whisper = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Regrouping algorithm
# ---------------------------------------------------------------------------
# stable-ts default 'da' expands to:
#   isp_cm_sp=.* /。/?/？_sg=.5_sp=,* /，++++50_sl=70_cm
#
# JAV-tuned changes (vs default 'da'):
#   sg:  0.5  → 1.5   Only split at 1.5s+ gaps (not every breath pause)
#   NEW  mg=1.5++80+1  Merge fragments if gap < 1.5s AND combined < 80 chars
#   sp2: add 、        Japanese comma for long-segment splitting
#   sl:  70   → 80    Match merge limit so sl doesn't undo mg
#   NEW  sd=8          Hard cap: no subtitle exceeds 8 seconds
#
# Full pipeline:
#   isp          ignore special periods (Mr., Dr.)
#   cm           initial safety clamp
#   sp=…         split at sentence-ending punctuation (。?？. )
#   sg=1.5       split at 1.5s+ gaps
#   mg=1.5++80+1 merge close fragments (gap < 1.5s, combined < 80 chars)
#   sp=…++++50   split at commas/、 if segment > 50 chars
#   sl=80        split if > 80 chars
#   sd=8         split if > 8 seconds
#   cm           final safety clamp

REGROUP_JAV = (
    "isp_cm"
    "_sp=.* /。/?/？"       # split at sentence-ending punctuation
    "_sg=1.5"               # split at 1.5s+ gaps (relaxed from 0.5s)
    "_mg=1.5++80+1"         # merge fragments: gap < 1.5s, combined < 80ch
    "_sp=,* /，/、++++50"   # split at commas if > 50 chars
    "_sl=80"                # split if > 80 chars
    "_sd=8"                 # max 8s per subtitle
    "_cm"                   # final clamp
)

# ---------------------------------------------------------------------------
# VAD-only regrouping (Branch B — no aligner, synthetic timestamps)
# ---------------------------------------------------------------------------
# Branch B timestamps come from split_frame_to_words() — proportional
# estimates, NOT real audio-derived.  REGROUP_JAV's gap-based heuristics
# (sg, mg) assume timestamps reflect actual audio gaps, which is false here.
#
# REGROUP_VAD_ONLY strips gap analysis and keeps only:
#   sp   sentence-ending punctuation split (redundant with split_frame_to_words
#        but harmless — catches any remaining multi-sentence pseudo-words)
#   sp2  comma split for long segments (>50 chars safety)
#   sl   char-length cap (80 chars)
#   sd   duration cap (8 seconds)
#   cm   boundary clamp
#
# Future: ADR-006 Tier 1 — replace with a composable Reconstructor protocol
# that lets the pipeline choose regrouping strategy per-mode.  See
# docs/architecture/A3-ROADMAP-SUBTITLE-PIPELINE.md §Tier 1 Option C.

REGROUP_VAD_ONLY = (
    "cm"
    "_sp=.* /。/?/？"       # split at sentence-ending punctuation
    "_sp=,* /，/、++++50"   # split at commas if > 50 chars
    "_sl=80"                # split if > 80 chars
    "_sd=8"                 # max 8s per subtitle
    "_cm"                   # final clamp
)


# ---------------------------------------------------------------------------
# Frame → pseudo-word splitting (Branch B granularity fix)
# ---------------------------------------------------------------------------
# Branch B (aligner-free / vad_only) creates one "word" per temporal frame.
# REGROUP_JAV expects word-level granularity (1-5 chars, 0.1-0.5s).  These
# helpers split frame text into sentence-level pseudo-words so REGROUP_JAV's
# sd=8 cap can actually enforce the 8s limit.
# ---------------------------------------------------------------------------

# Sentence-ending punctuation: split AFTER these, keeping punct with text
_SENTENCE_END_RE = re.compile(r"(?<=[。？?！!])")

# Comma punctuation: Japanese and Western commas
_COMMA_RE = re.compile(r"(?<=[、，,])")


def _split_at_sentences(text: str) -> list[str]:
    """Split text at sentence-ending punctuation, keeping punct with preceding text."""
    parts = _SENTENCE_END_RE.split(text)
    return [p for p in parts if p]


def _split_at_commas(text: str) -> list[str]:
    """Split text at Japanese/Western commas, keeping comma with preceding text."""
    parts = _COMMA_RE.split(text)
    return [p for p in parts if p]


def _split_into_chunks(text: str, max_chars: int = 10) -> list[str]:
    """Split text into roughly equal chunks of at most max_chars."""
    if len(text) <= max_chars:
        return [text]
    n_chunks = max(2, -(-len(text) // max_chars))  # ceil division
    chunk_size = -(-len(text) // n_chunks)  # ceil division for even split
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def split_frame_to_words(
    text: str,
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    """Split frame text into pseudo-words with proportional timestamps.

    Gives REGROUP_JAV fine-grained word boundaries for frame-level input.
    Splitting hierarchy: sentences → commas → character chunks.

    Args:
        text: Frame text (may contain multiple sentences).
        start: Frame start time in seconds.
        end: Frame end time in seconds.

    Returns:
        List of ``{'word': str, 'start': float, 'end': float}`` dicts.
        Empty list if text is empty/whitespace.
    """
    text = text.strip()
    if not text:
        return []

    # Try sentence-level splitting first
    chunks = _split_at_sentences(text)

    # If only 1 chunk and long, try comma splitting
    if len(chunks) == 1 and len(text) > 15:
        chunks = _split_at_commas(text)

    # If still 1 chunk and very long, fall back to character chunks
    if len(chunks) == 1 and len(text) > 20:
        chunks = _split_into_chunks(text, max_chars=10)

    # Single chunk (short text) — return as-is
    if len(chunks) == 1:
        return [{"word": chunks[0], "start": start, "end": end}]

    # Distribute timestamps proportionally by character count
    total_chars = sum(len(c) for c in chunks)
    duration = end - start
    words: list[dict[str, Any]] = []
    cursor = start

    for i, chunk in enumerate(chunks):
        if i == len(chunks) - 1:
            # Last chunk: snap to frame end to avoid float drift
            chunk_end = end
        else:
            chunk_end = cursor + duration * (len(chunk) / total_chars)
        words.append({"word": chunk, "start": cursor, "end": chunk_end})
        cursor = chunk_end

    return words


def reconstruct_from_words(
    words: list[dict[str, Any]],
    audio_path: Union[str, Path],
    suppress_silence: bool = True,
    regroup: Union[bool, str] = REGROUP_JAV,
) -> "stable_whisper.WhisperResult":
    """
    Reconstruct a WhisperResult from word dicts via stable-ts transcribe_any.

    Uses stable-ts's ``transcribe_any()`` with a pre-computed inference
    function to get proper sentence-level regrouping (Japanese-aware
    boundary detection, natural dialogue splitting).

    Args:
        words: List of ``{'word': str, 'start': float, 'end': float}`` dicts.
            Timestamps must be in scene-relative seconds.
        audio_path: Path to the audio file.  Required by transcribe_any for
            duration metadata — the audio is NOT re-transcribed.
        suppress_silence: Whether stable-ts should adjust timestamps based
            on silence detection.  Set False for sentinel-recovered words
            to preserve the recovery's timestamp distribution.
        regroup: Regrouping algorithm string for stable-ts. Defaults to
            ``REGROUP_JAV`` (tuned for conversational Japanese dialogue).
            Pass ``True`` for stable-ts default ``'da'``, or ``False``
            to skip regrouping entirely.

    Returns:
        Reconstructed WhisperResult with sentence-level regrouping.

    Raises:
        RuntimeError: If stable_whisper is not installed.
    """
    if stable_whisper is None:
        raise RuntimeError("stable_whisper is required for reconstruction but not installed")

    if not words:
        logger.debug("[RECONSTRUCT] Empty word list, creating minimal result")
        # Create a minimal result with no segments
        precomputed: list[dict[str, Any]] = []

        def empty_inference(audio, **kwargs):
            return [precomputed]

        return stable_whisper.transcribe_any(
            inference_func=empty_inference,
            audio=str(audio_path),
            audio_type="str",
            regroup=False,
            vad=False,
            demucs=False,
            suppress_silence=False,
            verbose=False,
        )

    # Normal path: reconstruct from pre-computed word timestamps
    precomputed_words = words

    def precomputed_inference(audio, **kwargs):
        return [precomputed_words]

    return stable_whisper.transcribe_any(
        inference_func=precomputed_inference,
        audio=str(audio_path),
        audio_type="str",
        regroup=regroup,
        vad=False,
        demucs=False,
        suppress_silence=suppress_silence,
        suppress_word_ts=suppress_silence,
        force_order=True,
        verbose=False,
    )
