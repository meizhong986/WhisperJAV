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
        verbose=False,
    )
