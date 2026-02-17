"""
WhisperResult reconstruction from word-level timestamps.

Converts raw word dicts (from alignment or temporal framing) into
stable_whisper.WhisperResult objects with sentence-level regrouping.

Extracted from QwenPipeline._reconstruct_from_words (qwen_pipeline.py:1846-1882).

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


def reconstruct_from_words(
    words: list[dict[str, Any]],
    audio_path: Union[str, Path],
    suppress_silence: bool = True,
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
        regroup=True,
        vad=False,
        demucs=False,
        suppress_silence=suppress_silence,
        suppress_word_ts=suppress_silence,
        verbose=False,
    )
