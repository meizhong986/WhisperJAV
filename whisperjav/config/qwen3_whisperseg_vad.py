#!/usr/bin/env python3
"""
Qwen3-ASR WhisperSeg VAD defaults (ChronosJAV / qwen pipeline).

SINGLE SOURCE OF TRUTH for the WhisperSeg speech-segmenter values that the
**qwen3** ASR backend gets on top of WhisperSeg's own sensitivity presets
(config/v4/ecosystems/tools/whisperseg-speech-segmentation.yaml). Read by every
entry point so the values stay consistent:

    - whisperjav/main.py                    (standalone single-pass qwen path)
    - whisperjav/ensemble/pass_worker.py    (ensemble two-pass path + GUI Ensemble tab)
    - whisperjav/webview_gui/api.py         (Customize-parameters dialog display)

The anime-whisper equivalent is config/anime_whisper_vad.py. Cohere is not
covered (it keeps the YAML presets).

The values are the SAME at every sensitivity (owner, 2026-10-05). The longest
segment is a subtitle line-length policy, not a detection-sensitivity setting:
the YAML's 6 / 5 / 4 s steps tied line length to sensitivity, and the cost of a
shorter window (worse text) depends on the ASR model, not on the sensitivity.

    threshold              0.25   unchanged since v1.9.0 (flat for qwen3)
    max_speech_duration_s  4.0    was 6 / 5 / 4 s (conservative / balanced /
                                  aggressive) from the YAML presets

Evidence for 4.0 s (1.9.4 timing work, 2026-10-05; 7 Netflix drama clips, 305
ground-truth lines, Qwen3-ASR balanced; docs/measurements/v1.9.4/MEASUREMENTS_v194_REQ1_REQ2.md
section 2.3): 5 s -> 4 s. On the 156 ground-truth lines both runs matched, the
median end error fell from 1.06 to 0.68 s and the median start error from 0.33 to
0.31 s (13 lines the 5 s run matched were not matched at 4 s). Counting each run on
its own, lines ending within 0.5 s of the ground truth rose from 50 to 79 (matched
lines 169 -> 186). Character error rate 0.390 -> 0.394. Limits of 3 s and below
gained more timing but raised it to about 0.43. Measured at the balanced
sensitivity only, on drama scenes, not JAV audio; conservative (was 6 s) and
aggressive (already 4 s) follow the uniform policy above, unmeasured. Only the
single-segment ceiling changed: the group cap (3.0 s) and group gap (0.3 s) are
pipeline constructor values and stay as they were.

Applied with setdefault semantics, so anything the user set (GUI Customize
dialog, --qwen-max-speech-duration, --qwen-vad-threshold, --passN-params) wins.
Applied only when the segmenter is WhisperSeg: these are WhisperSeg-scale
values and must not ride above TEN / Silero / FireRedVAD presets.
"""

from typing import Any, Dict

QWEN3_WHISPERSEG_DEFAULTS: Dict[str, Any] = {
    "threshold": 0.25,
    "max_speech_duration_s": 4.0,
}


def apply_qwen3_segmenter_defaults(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill the Qwen3-ASR WhisperSeg defaults into an overrides dict, WITHOUT
    clobbering values the user already set.

    Called by both entry points (main.py standalone qwen path and
    ensemble/pass_worker.py) so the two cannot drift.

    Args:
        overrides: The user segmenter-overrides dict to fill in (mutated).

    Returns:
        The same dict, for chaining.
    """
    for key, value in QWEN3_WHISPERSEG_DEFAULTS.items():
        overrides.setdefault(key, value)
    return overrides
