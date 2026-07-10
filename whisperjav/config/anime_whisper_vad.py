#!/usr/bin/env python3
"""
Anime-Whisper WhisperSeg VAD per-sensitivity defaults (ChronosJAV / qwen pipeline).

SINGLE SOURCE OF TRUTH for the WhisperSeg speech-segmenter grouping/threshold/
padding defaults used by the **anime-whisper** ASR backend inside the qwen
(ChronosJAV) pipeline. Read by every entry point so the values stay consistent:

    - whisperjav/main.py                    (standalone single-pass qwen path)
    - whisperjav/ensemble/pass_worker.py    (ensemble two-pass path + GUI Ensemble tab)
    - whisperjav/webview_gui/api.py         (Customize-parameters dialog display)

Scope note: this table is anime-whisper ONLY. qwen3 and cohere keep their own
defaults (qwen3: flat 0.3/3.0/0.25/100/100 via ctor + threshold injection;
cohere: 1.0/6.0/300). See the __init__ generator-backend branch in
qwen_pipeline.py and the anime/cohere branches in main.py / pass_worker.py.

Mapping to GUI labels / pipeline scalars:
    chunk_threshold_s     <-> "Frame Gap Threshold (ms)"  (GUI stores ms; /1000 -> s)
    max_group_duration_s  <-> "Max Group Duration (s)"
    threshold             <-> "VAD Threshold"             (rides in segmenter_config)
    start_pad_ms          <-> "VAD Start Pad (ms)"        (pipeline scalar)
    end_pad_ms            <-> "VAD End Pad (ms)"           (pipeline scalar)

Owner-specified values (v1.9.0, shipped in v1.8.15). If a sensitivity is unknown
or a design limitation prevents a per-sensitivity value from being honored, the
overriding default is the BALANCED row (see anime_whisperseg_defaults()).
"""

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Per-sensitivity defaults. Keys are the resolved runtime parameter names.
# ---------------------------------------------------------------------------
ANIME_WHISPER_WHISPERSEG_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "chunk_threshold_s": 0.3,
        "max_group_duration_s": 3.0,
        "threshold": 0.35,
        "start_pad_ms": 100,
        "end_pad_ms": 100,
    },
    "balanced": {
        "chunk_threshold_s": 0.25,
        "max_group_duration_s": 2.5,
        "threshold": 0.30,
        "start_pad_ms": 50,
        "end_pad_ms": 50,
    },
    "aggressive": {
        "chunk_threshold_s": 0.2,
        "max_group_duration_s": 2.0,
        "threshold": 0.25,
        "start_pad_ms": 0,
        "end_pad_ms": 0,
    },
}

# The overriding fallback per owner instruction: when the requested sensitivity
# is missing/unknown, use BALANCED.
_FALLBACK_SENSITIVITY = "balanced"


def anime_whisperseg_defaults(sensitivity: str) -> Dict[str, Any]:
    """
    Return the anime-whisper WhisperSeg VAD defaults for a sensitivity.

    Args:
        sensitivity: "conservative" | "balanced" | "aggressive" (case-insensitive).
            Any unknown/None value falls back to the BALANCED row (owner rule).

    Returns:
        A COPY of the defaults dict (safe to mutate) with keys:
        chunk_threshold_s, max_group_duration_s, threshold, start_pad_ms, end_pad_ms.
    """
    key = (sensitivity or "").strip().lower()
    row = ANIME_WHISPER_WHISPERSEG_DEFAULTS.get(
        key, ANIME_WHISPER_WHISPERSEG_DEFAULTS[_FALLBACK_SENSITIVITY]
    )
    return dict(row)
