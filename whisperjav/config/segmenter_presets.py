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

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

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
    "firered-vad": "firered-vad-speech-segmentation",  # v1.9.0; Balanced default since v1.9.2
}

# v1.9.2 (owner CFF3 + D7): the Balanced pipeline defaults to a WhisperJAV external
# speech segmenter, FireRedVAD first. If its package is missing the next member is
# used — never faster-whisper's internal VAD (owner: "external" means a WhisperJAV
# segmenter). silero-v3.1 closes the chain because it is always installable.
BALANCED_DEFAULT_SEGMENTER_CHAIN: Tuple[str, ...] = ("firered-vad", "ten", "silero-v3.1")

# Backends the single-pass Balanced path may run without the routing-guard downgrade:
# their sensitivity presets are resolved by resolve_segmenter_sensitivity() below.
BALANCED_SINGLE_PASS_EXTERNAL = frozenset(BALANCED_DEFAULT_SEGMENTER_CHAIN)


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


def pick_balanced_default_segmenter(
    is_available: Optional[Callable[[str], bool]] = None,
    chain: Iterable[str] = BALANCED_DEFAULT_SEGMENTER_CHAIN,
) -> str:
    """
    Return the Balanced pipeline's default speech segmenter: the first member of
    ``chain`` whose package is installed. Logs a WARNING naming the pip command
    when FireRedVAD has to be skipped. The last member is returned unconditionally.

    Args:
        is_available: availability predicate (tests inject one). Default uses
            SpeechSegmenterFactory.is_backend_available, which checks importability
            without importing the backend.
        chain: ordered candidates; defaults to BALANCED_DEFAULT_SEGMENTER_CHAIN.
    """
    if is_available is None:
        from whisperjav.modules.speech_segmentation.factory import SpeechSegmenterFactory

        def is_available(name: str) -> bool:
            return bool(SpeechSegmenterFactory.is_backend_available(name)[0])

    chain = tuple(chain)
    for candidate in chain[:-1]:
        if is_available(candidate):
            return candidate
        logger.warning(
            "Balanced default speech segmenter '%s' is not installed "
            "(pip install %s); falling back to the next WhisperJAV segmenter.",
            candidate, "fireredvad" if candidate == "firered-vad" else candidate,
        )
    return chain[-1]
