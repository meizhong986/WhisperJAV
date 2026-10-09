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

One exception (owner, 2026-10-08): with TEN and FireRedVAD the longest segment
is 4 s at every sensitivity for Qwen3-ASR and anime-whisper, as with WhisperSeg
(their presets say 6/5/4 s and 7/6/5 s). Only that value; the rest of their
presets stays theirs. Other pipelines using TEN or FireRedVAD are not touched.

The end pad of Qwen3-ASR with WhisperSeg is 0 ms at every sensitivity (owner,
2026-10-09; it is a pipeline value, not part of the segmenter config):
chronosjav_end_pad_default().
"""

from typing import Any, Dict, Optional

from whisperjav.config.anime_whisper_vad import apply_anime_segmenter_defaults
from whisperjav.config.qwen3_whisperseg_vad import apply_qwen3_segmenter_defaults
from whisperjav.config.segmenter_presets import resolve_segmenter_sensitivity


CHRONOSJAV_GENERATORS = ("qwen3", "anime-whisper")

# Longest segment for TEN and FireRedVAD in ChronosJAV, every sensitivity (owner, 2026-10-08).
LONGEST_SEGMENT_S = {"ten": 4.0, "firered-vad": 4.0}

# End pad for Qwen3-ASR with WhisperSeg, every sensitivity (owner, 2026-10-09; the pipeline's
# own default, 100 ms, stays for the other segmenters). anime-whisper takes its pads from
# config/anime_whisper_vad.py.
QWEN3_WHISPERSEG_END_PAD_MS = 0


def chronosjav_end_pad_default(generator: str, segmenter: str) -> Optional[int]:
    """The end pad (ms) a Qwen3-ASR pass uses when the user set none, or None
    for the pipeline's own default. Read by main.py, the ensemble worker and
    the Customize dialog."""
    if generator == "qwen3" and (segmenter or "whisperseg").strip().lower() == "whisperseg":
        return QWEN3_WHISPERSEG_END_PAD_MS
    return None


def apply_chronosjav_segmenter_defaults(
    overrides: Dict[str, Any], generator: str, segmenter: str, sensitivity: str
) -> Dict[str, Any]:
    """Fill the ChronosJAV defaults for this generator into `overrides`: the
    full set for WhisperSeg, the longest segment only for TEN and FireRedVAD;
    otherwise leave it unchanged. Returns it."""
    if segmenter in LONGEST_SEGMENT_S and generator in CHRONOSJAV_GENERATORS:
        overrides.setdefault("max_speech_duration_s", LONGEST_SEGMENT_S[segmenter])
        return overrides
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
    preset, then the ChronosJAV defaults (WhisperSeg; longest segment for TEN
    and FireRedVAD), then `overrides`
    (the user's values win). Same functions the entry points call."""
    merged = dict(overrides or {})
    apply_chronosjav_segmenter_defaults(merged, generator, segmenter, sensitivity)
    return resolve_segmenter_sensitivity(segmenter, sensitivity, merged or None)
