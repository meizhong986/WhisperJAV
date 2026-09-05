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

Not GUI-displayed (segmenter_config only, injected by
apply_anime_segmenter_defaults): neg_threshold, speech_start_threshold,
force_split_mode, min_silence_duration_ms, max_speech_duration_s
(CLI: --qwen-max-speech-duration).

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
        # v1.9.0 tuning (2026-07-13): cross-clip VAD sweep over 4 Naked-Director
        # benchmarks showed lowering threshold 0.25->0.15 + a 30ms end-pad lifts
        # dialogue recall (time_recall 0.83->0.88 mean) at flat CER on every clip,
        # without lengthening subs. max_speech adjusted to 3.5 (typical JAV dialogue
        # ~2.7s; 4.0 was too long). See tools/vad_hypothesis_suite + reference_benchmarks/.
        #
        # neg_threshold decoupled from threshold (2026-07-28): the upstream formula
        # max(threshold-0.15, 0.01) collapses to 0.01 at threshold=0.15, disabling
        # natural segment endings entirely. An explicit neg_threshold restores
        # hysteresis while keeping the wide-net onset at 0.15.
        #
        # 2026-07-31 DEFAULT: "offline" two-level decoder (owner-approved).
        # Segments SEED at prob >= threshold (0.15), edges GROW while prob >=
        # grow_floor (0.05 — keeps quiet speech attached to confident speech),
        # gaps >= gap_merge_ms (350) become real dialog cuts, and overlong
        # segments split at smoothed-probability minima instead of a blind
        # chop. "3a" speech_start_threshold=0.30 still refines display starts
        # (T-sweep: median start error -0.489s -> -0.026s).
        # Tuning levers (GUI Customize / segmenter_config): grow_floor is the
        # capture-vs-cut dial (lower = keep more quiet speech, cut fewer
        # gaps); gap_merge_ms is the pause-length that splits dialogs.
        #
        # SECOND OPTION (segmentation_decoder="hysteresis"): the vendor-
        # lineage ChickenRice state machine, kept intact. The neg_threshold/
        # force_split_mode/min_silence values below reproduce the exact
        # pre-i2 capture regime (GT: char_recall 0.723, CER 0.539) when the
        # decoder is flipped back — they are INERT under "offline".
        # Doc: docs/architecture/WHISPERSEG_TRADEOFF_ANALYSIS_AND_RECOMMENDATION.md
        "chunk_threshold_s": 0.2,
        "max_group_duration_s": 2.0,
        "threshold": 0.15,
        "segmentation_decoder": "offline",
        "grow_floor": 0.05,
        "gap_merge_ms": 350,
        "split_smooth_ms": 120,
        "speech_start_threshold": 0.30,
        # hysteresis-decoder fallback values (exact pre-i2; inert under "offline"):
        "neg_threshold": 0.01,
        "force_split_mode": "chop",
        "min_silence_duration_ms": 80,
        "start_pad_ms": 0,
        "end_pad_ms": 30,
        "max_speech_duration_s": 4.0,
    },
}

# The overriding fallback per owner instruction: when the requested sensitivity
# is missing/unknown, use BALANCED.
_FALLBACK_SENSITIVITY = "balanced"

# Table keys that travel to the WhisperSeg backend via segmenter_config
# (resolve_qwen_sensitivity -> SEGMENTER_PARAMS filter -> factory). The pad,
# chunk and group keys are NOT listed here — they route via pipeline scalars
# (segmenter_start_pad_ms etc.) which the qwen pipeline injects at clobber
# time; putting them in segmenter_config would be overwritten.
#
# Every key listed here MUST also be present in
# whisperjav/ensemble/pass_worker.py::SEGMENTER_PARAMS, or the resolve filter
# silently strips it — that exact mismatch shipped neg_threshold as dead
# config in the first i2 attempt (2026-07-28).
SEGMENTER_CONFIG_KEYS = (
    "threshold",
    "neg_threshold",
    "speech_start_threshold",
    "force_split_mode",
    "segmentation_decoder",
    "grow_floor",
    "gap_merge_ms",
    "split_smooth_ms",
    "min_silence_duration_ms",
    "max_speech_duration_s",
)


def apply_anime_segmenter_defaults(
    overrides: Dict[str, Any], sensitivity: str
) -> Dict[str, Any]:
    """
    Inject the anime-whisper per-sensitivity segmenter-config defaults into an
    overrides dict, WITHOUT clobbering values the user already set (GUI
    Customize dialog / sliders / CLI flags all land in `overrides` first).

    Single source of the table->segmenter_config lift, called by BOTH entry
    points (main.py standalone qwen path and ensemble/pass_worker.py). Keeping
    the lift here prevents the two copies from drifting — the original i2 bug
    was one entry point lifting threshold+max_speech but not neg_threshold.

    Args:
        overrides: The user_segmenter_overrides dict to fill in (mutated).
        sensitivity: "conservative" | "balanced" | "aggressive".

    Returns:
        The same dict, for chaining.
    """
    row = anime_whisperseg_defaults(sensitivity)
    for key in SEGMENTER_CONFIG_KEYS:
        value = row.get(key)
        if value is not None:
            overrides.setdefault(key, value)
    return overrides


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
