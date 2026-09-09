"""
Speech-segmenter sensitivity presets and the Balanced-pipeline default segmenter.

Light module (no ML imports at module level) shared by the single-pass CLI path
(``whisperjav/main.py``), the ensemble pass worker (``whisperjav/ensemble/pass_worker.py``)
and the GUI backend (``whisperjav/webview_gui/api.py``), so the three entry points
resolve the same segmenter and the same per-sensitivity parameters.

History: ``SEGMENTER_PARAMS``, the backend→tool-name map and
``resolve_qwen_sensitivity`` lived in ``pass_worker.py`` (v1.8.12+), which imports
every pipeline class and is therefore too heavy for ``main.py``'s config-resolution
stage. v1.9.2 moved them here unchanged; ``pass_worker`` re-exports the old names.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from whisperjav.utils.logger import logger

# Segmenter params - routed to speech segmentation backends, not passed to Whisper ASR
# Covers all backends: Silero, TEN, Whisper VAD, FireRedVAD, and shared grouping params
SEGMENTER_PARAMS = {
    # Core VAD (Silero, shared)
    "threshold",
    "neg_threshold",           # v1.9.0: WhisperSeg decoupled offset threshold (anime table).
                               # MUST stay in sync with anime_whisper_vad.SEGMENTER_CONFIG_KEYS —
                               # keys missing here are silently stripped by resolve_segmenter_sensitivity.
    "speech_start_threshold",  # v1.9.0 "3a": refined display-start threshold (anime table).
    "force_split_mode",        # v1.9.0: WhisperSeg "dip"|"chop" force-split behavior (anime table).
    "segmentation_decoder",    # v1.9.0: WhisperSeg "hysteresis"|"offline" decoder (anime table + GUI).
    "grow_floor",              # v1.9.0 offline decoder: edge-growth floor (anime table + GUI).
    "gap_merge_ms",            # v1.9.0 offline decoder: dialog-cut gap length (anime table + GUI).
    "split_smooth_ms",         # v1.9.0 offline decoder: overlong-split smoothing (anime table).
    "min_speech_duration_ms",
    "max_speech_duration_s",   # Keep for CLI backward compat (not in GUI)
    "min_silence_duration_ms",
    "speech_pad_ms",
    # Grouping (shared across backends)
    "chunk_threshold_s",
    "max_group_duration_s",
    # TEN-specific
    "hop_size",
    "start_pad_ms",
    "end_pad_ms",
    # Whisper VAD-specific
    "cache_results",
    # FireRedVAD-specific (v1.9.0)
    "smooth_window_size",
    "use_gpu",
}

# Backend name → YAML tool name mapping for ConfigManager.get_tool_config()
SEGMENTER_TOOL_NAMES = {
    "silero-v6.2": "silero-v6-speech-segmentation",
    "silero": "silero-speech-segmentation",
    "silero-v4.0": "silero-speech-segmentation",
    "silero-v3.1": "silero-speech-segmentation",
    "ten": "ten-speech-segmentation",
    "nemo": "nemo-speech-segmentation",
    "nemo-lite": "nemo-speech-segmentation",
    "whisper-vad": "whisper-vad-speech-segmentation",
    "whisper-vad-tiny": "whisper-vad-speech-segmentation",
    "whisper-vad-base": "whisper-vad-speech-segmentation",
    "whisper-vad-small": "whisper-vad-speech-segmentation",
    "whisper-vad-medium": "whisper-vad-speech-segmentation",
    "whisperseg": "whisperseg-speech-segmentation",
    "firered-vad": "firered-vad-speech-segmentation",  # v1.9.0; installed by default since v1.9.2
}

# The Balanced pipeline's default speech segmenter is faster-whisper's built-in VAD
# (vad_filter=True, one recognizer call per scene), as in v1.9.0/v1.9.1. A v1.9.2
# development build defaulted to FireRedVAD with a fallback chain; the owner reversed
# that on 2026-09-05 (N3). A WhisperJAV segmenter is selected explicitly with
# --speech-segmenter (CLI), --passN-speech-segmenter (ensemble) or the GUI dropdown.
#
# v1.9.2 (owner S2/S9): on the BALANCED pipeline this is no longer a choice at all --
# balanced runs the built-in VAD and the user picks which Silero build it runs
# (--vad-version). The other pipelines are unaffected.
BALANCED_DEFAULT_SEGMENTER = "faster-whisper"

# The default SCENE detector for every scene-detecting legacy pipeline (v1.9.2).
#
# The owner approved this flip on 2026-04-20 for v1.8.12; the CLI half was never
# applied, while the GUI Ensemble tab has shipped semantic as its selected default
# since v1.8.11 — so the two entry points disagreed until now. Single-sourced here and
# read by main.py, ensemble/pass_worker.py and webview_gui/api.py so they cannot drift.
#
# Semantic is the only backend whose min_duration MERGES short segments, which is what
# makes a scene-length floor expressible at all; auditok and silero discard them.
DEFAULT_SCENE_DETECTOR = "semantic"

# REMOVED in v1.9.2: BALANCED_SINGLE_PASS_EXTERNAL.
#
# It exempted firered-vad and ten from main.py's routing-guard downgrade on
# `--mode balanced`, so an explicit choice there honoured --sensitivity (owner,
# 2026-09-06, "default only"). The owner's balanced requirements (S2/S9,
# 2026-09-09) supersede that: balanced runs faster-whisper's built-in VAD and
# accepts no --speech-segmenter at all, so the exemption became unreachable.
# Both backends remain fully available through --ensemble and --mode fidelity.


def resolve_segmenter_sensitivity(
    segmenter_backend: str,
    sensitivity: str,
    user_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve a segmenter backend's sensitivity preset into segmenter_config params.

    Layering: backend YAML spec < sensitivity preset < user overrides.
    Uses ConfigManager.get_tool_config() which already implements this
    exact layering.

    Args:
        segmenter_backend: Speech segmenter backend name (e.g., "silero-v6.2", "ten")
        sensitivity: Sensitivity level ("aggressive", "balanced", "conservative")
        user_overrides: Optional user custom params (win over preset)

    Returns:
        Dict of segmenter config params (filtered to SEGMENTER_PARAMS keys)
    """
    # "faster-whisper" = native VAD inside faster-whisper (vad_filter). Like
    # "none", it has no EXTERNAL segmenter config to resolve by sensitivity, so
    # return empty (the ASR enables vad_filter itself). Avoids a spurious
    # "Unknown segmenter backend" warning when balanced runs native VAD in a pass.
    if segmenter_backend in ("none", "faster-whisper") or not segmenter_backend:
        return {}

    tool_name = SEGMENTER_TOOL_NAMES.get(segmenter_backend)
    if not tool_name:
        logger.warning(
            "Unknown segmenter backend '%s' for sensitivity resolution; "
            "passing user overrides only",
            segmenter_backend,
        )
        return {k: v for k, v in (user_overrides or {}).items() if k in SEGMENTER_PARAMS}

    try:
        from whisperjav.config.v4 import ConfigManager

        cm = ConfigManager()
        resolved = cm.get_tool_config(tool_name, sensitivity, user_overrides)

        # Filter to SEGMENTER_PARAMS only — ConfigManager returns full tool config
        # including metadata keys we don't want to pass to the backend
        return {k: v for k, v in resolved.items() if k in SEGMENTER_PARAMS}
    except Exception as e:
        logger.warning(
            "ConfigManager failed for '%s' sensitivity '%s': %s. "
            "Falling back to user overrides only.",
            tool_name, sensitivity, e,
        )
        return {k: v for k, v in (user_overrides or {}).items() if k in SEGMENTER_PARAMS}

