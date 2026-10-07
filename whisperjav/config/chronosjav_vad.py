#!/usr/bin/env python3
"""
Which ChronosJAV speech-segmenter defaults apply to a pass (v1.9.4).

ONE decision, used by every place that needs it, so they cannot disagree:

    - whisperjav/main.py                    (standalone --mode qwen)
    - whisperjav/ensemble/pass_worker.py    (ensemble passes, GUI Ensemble tab)
    - whisperjav/webview_gui/api.py         (Customize dialog: what it shows is what runs)
    - tests/test_chronosjav_segment_defaults_v194.py

The rule: the ChronosJAV values (config/qwen3_whisperseg_vad.py for Qwen3-ASR,
config/anime_whisper_vad.py for anime-whisper) are WhisperSeg-scale values and
apply ONLY when the segmenter is WhisperSeg. With any other segmenter (TEN,
Silero, FireRedVAD, ...) that segmenter's own sensitivity preset applies, so a
flat 0.25 or a 3 s cap never rides above, say, TEN's tuned gradient. Cohere is
not covered. Values the user set are kept (setdefault semantics).
"""

from typing import Any, Dict, Optional

from whisperjav.config.anime_whisper_vad import apply_anime_segmenter_defaults
from whisperjav.config.qwen3_whisperseg_vad import apply_qwen3_segmenter_defaults
from whisperjav.config.segmenter_presets import resolve_segmenter_sensitivity


def apply_chronosjav_segmenter_defaults(
    overrides: Dict[str, Any], generator: str, segmenter: str, sensitivity: str
) -> Dict[str, Any]:
    """Fill the ChronosJAV defaults for this generator into `overrides` when
    the segmenter is WhisperSeg; otherwise leave it unchanged. Returns it."""
    if segmenter != "whisperseg":
        return overrides
    if generator == "anime-whisper":
        apply_anime_segmenter_defaults(overrides, sensitivity)
    elif generator == "qwen3":
        apply_qwen3_segmenter_defaults(overrides)
    return overrides


def resolve_chronosjav_segmenter_config(
    generator: str, segmenter: str, sensitivity: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The segmenter config a pass resolves to: the segmenter's sensitivity
    preset, then the ChronosJAV defaults (WhisperSeg only), then `overrides`
    (the user's values win). Same functions the entry points call."""
    merged = dict(overrides or {})
    apply_chronosjav_segmenter_defaults(merged, generator, segmenter, sensitivity)
    return resolve_segmenter_sensitivity(segmenter, sensitivity, merged or None)
