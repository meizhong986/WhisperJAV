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

The group and pad values of an anime-whisper pass follow the same rule since 1.10.3
(owner W753 CL3, a bug fix): the anime-whisper table with WhisperSeg, otherwise the
segmenter's own preset: anime_whisper_group_defaults().
"""

from typing import Any, Dict, Optional

from whisperjav.config.anime_whisper_vad import anime_whisperseg_defaults, apply_anime_segmenter_defaults
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


# The pipeline values of a pass's VAD groups: the gap that starts a new group, the longest group, the pads.
GROUP_KEYS = ("chunk_threshold_s", "max_group_duration_s", "start_pad_ms", "end_pad_ms")


def anime_whisper_group_defaults(segmenter: str, sensitivity: str) -> Dict[str, Any]:
    """The gap, longest group and pads of an anime-whisper pass when the user set none.

    WhisperSeg: the anime-whisper table (config/anime_whisper_vad.py), which holds WhisperSeg-scale values. Any other
    segmenter: that segmenter's own preset at the sensitivity, as the fidelity pipeline takes it
    (resolve_segmenter_sensitivity); a value the preset does not have is left out, so the pipeline's own default
    applies. "none": nothing. Before 1.10.3 the table applied whatever the segmenter (owner W753 CL3: a bug).
    Read by main.py, the ensemble worker and the Customize dialog (webview_gui/api.py), so they cannot disagree."""
    seg = (segmenter or "whisperseg").strip().lower()
    if seg == "whisperseg":
        row = anime_whisperseg_defaults(sensitivity)
        return {k: row[k] for k in GROUP_KEYS}
    if seg == "none":
        return {}
    preset = resolve_segmenter_sensitivity(seg, sensitivity)
    return {k: preset[k] for k in GROUP_KEYS if preset.get(k) is not None}


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
