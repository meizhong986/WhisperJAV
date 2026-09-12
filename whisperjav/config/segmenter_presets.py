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
    # FireRedVAD model directory (2026-09-12). Here so a model_dir set in the
    # segmenter config survives resolve_segmenter_sensitivity, which drops every
    # key not in this set.
    "model_dir",
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

# The Fidelity pipeline's default speech segmenter (owner, 2026-09-12).
#
# Before this, the two entry points disagreed: `--mode fidelity` (and the GUI
# Transcription tab, which sends no --speech-segmenter) fell through main.py's
# else-branch to silero-v3.1, while an --ensemble fidelity pass with no explicit
# segmenter left params["speech_segmenter"] unset and landed on WhisperProASR's
# own "whisperseg" fallback. Single-sourced here and read by main.py,
# ensemble/pass_worker.py and modules/whisper_pro_asr.py so they cannot drift;
# the GUI Ensemble tab carries its own copy in assets/app.js
# (applyPipelinePresets, the `pipeline === 'fidelity'` branch).
#
# fireredvad ships with the [cli] extra since v1.9.2, so this default needs no
# extra install; its model weights download from HuggingFace on first use.
FIDELITY_DEFAULT_SEGMENTER = "firered-vad"

# Non-Silero segmenters that main.py must NOT downgrade on a single-pass run.
#
# main.py's routing guard sends every non-Silero backend back to silero-v3.1 on a
# single-pass mode, because that path used not to resolve a segmenter's
# per-sensitivity preset -- whisperseg then ran with its own 29 s
# max_group_duration_s and triggered the Whisper repetition pathology on JAV
# audio. v1.9.2 added that resolution to the single-pass path
# (main.py, "resolve the backend's per-sensitivity YAML preset on the
# single-pass path too"), and firered-vad's tool YAML carries a complete set of
# grouping params (chunk_threshold_s, max_group_duration_s) in its spec AND in
# all three sensitivity presets, so it reaches the segmenter fully configured.
# Only backends verified to be in that state belong here, and only for the mode
# they were verified on: keyed by --mode, because `fast` and `faster` run
# stable_ts with no speech segmenter at all (config/legacy.py: "vad": "none"), so
# exempting them buys nothing and costs a run stopped over a model they never load.
SINGLE_PASS_EXTERNAL_OK = {"fidelity": frozenset({"firered-vad"})}

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


def effective_segmenter_for_pass(
    pipeline: Optional[str],
    speech_segmenter: Optional[str],
) -> Optional[str]:
    """
    The speech segmenter a pass will actually run, given its pipeline and whatever
    the user asked for (which is usually nothing).

    ONE rule, used by ensemble/pass_worker.py when it resolves the pass and by
    main.py's start-up check when it decides whether a model has to be fetched
    before the run. Those two answering differently is how a start-up check ends up
    guarding a segmenter the run does not use -- or missing the one it does.

    Balanced ignores the request entirely (v1.9.2, S2/S9: no external segmenter
    exists there). Fidelity falls back to FireRedVAD. Everything else takes what it
    was given, including None, which leaves the decision to the ASR module.
    """
    if pipeline == "balanced":
        return BALANCED_DEFAULT_SEGMENTER
    if pipeline == "fidelity" and not speech_segmenter:
        return FIDELITY_DEFAULT_SEGMENTER
    return speech_segmenter


def segmenter_accepts(segmenter_backend: Optional[str], param_name: str) -> bool:
    """
    True if ``param_name`` actually reaches ``segmenter_backend``.

    The segmenter factory strips any parameter that is not in the backend's own
    schema (``speech_segmentation/factory.py``, the foreign-key gate), and it does so
    at DEBUG level. A caller that has just told the user "setting applied" needs to
    know whether that is true. ``speech_pad_ms`` is the live case: every Silero
    backend takes it, firered-vad and ten do not -- they pad with start_pad_ms /
    end_pad_ms instead.

    Fails OPEN. An unknown backend, or one that is not an external segmenter at all
    ("faster-whisper" = the recogniser's built-in VAD, "none" = no segmentation),
    returns True, so a caller never warns about something it could not check.
    """
    if not segmenter_backend or segmenter_backend in ("none", "faster-whisper"):
        return True
    try:
        from whisperjav.modules.speech_segmentation.factory import _PARAM_SCHEMAS
    except Exception:  # pragma: no cover - defensive
        return True
    schema = _PARAM_SCHEMAS.get(segmenter_backend)
    if schema is None:
        return True
    return param_name in schema


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

