"""
Canonical speech-segmenter parameter resolution (v1.9.0).

Single source of truth for routing segmenter parameters, shared by the legacy
resolver (`config/legacy.py`) and the ensemble path (`ensemble/pass_worker.py`).

Canonical location: ``params["speech_segmenter"]`` carries the runtime
``backend`` name, backend-specific detection params, and the backend-agnostic
grouping params (``chunk_threshold_s``, ``max_group_duration_s``). Silero-only
detection params live in ``params["vad"]`` and are merged into the segmenter
config by ASR constructors for Silero backends only.

See docs/architecture/SEGMENTER_ROUTING_UNIFICATION_v1.9.md.
"""

from typing import Any

from whisperjav.utils.logger import logger

# Backend name -> YAML tool name mapping for ConfigManager.get_tool_config().
# (Moved from ensemble/pass_worker.py in v1.9.0; keep the two in sync via import.)
SEGMENTER_TOOL_NAMES: dict[str, str] = {
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
}

# Segmenter params - routed to speech segmentation backends, not to Whisper ASR.
# Covers all backends: Silero, TEN, Whisper VAD, WhisperSeg, NeMo.
SEGMENTER_PARAM_KEYS = {
    # Core VAD (Silero, shared)
    "threshold",
    "min_speech_duration_ms",
    "max_speech_duration_s",
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
}

# Backend-agnostic grouping params (the SegmenterGroupingOptions split).
GROUPING_KEYS = {"chunk_threshold_s", "max_group_duration_s"}


def is_silero_backend(backend: str | None) -> bool:
    """True if the runtime segmenter backend is a Silero variant."""
    return bool(backend) and backend.startswith("silero")


def resolve_segmenter_config(
    segmenter_backend: str,
    sensitivity: str,
    user_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolve a segmenter backend's sensitivity preset into a config dict.

    Layering: backend YAML spec < sensitivity preset < user overrides.
    Uses ConfigManager.get_tool_config() which implements this exact layering.

    Args:
        segmenter_backend: Backend name (e.g., "whisperseg", "ten", "silero-v3.1")
        sensitivity: "aggressive", "balanced", or "conservative"
        user_overrides: Optional user custom params (win over preset)

    Returns:
        Dict of segmenter config params (filtered to SEGMENTER_PARAM_KEYS).
        Empty dict for "none"/empty backend.
    """
    if segmenter_backend == "none" or not segmenter_backend:
        return {}

    tool_name = SEGMENTER_TOOL_NAMES.get(segmenter_backend)
    if not tool_name:
        logger.warning(
            "Unknown segmenter backend '%s' for sensitivity resolution; "
            "passing user overrides only",
            segmenter_backend,
        )
        return {k: v for k, v in (user_overrides or {}).items() if k in SEGMENTER_PARAM_KEYS}

    try:
        from whisperjav.config.v4 import ConfigManager

        cm = ConfigManager()
        resolved = cm.get_tool_config(tool_name, sensitivity, user_overrides)

        # Filter to SEGMENTER_PARAM_KEYS only - ConfigManager returns full tool
        # config including metadata keys we don't want to pass to the backend.
        return {k: v for k, v in resolved.items() if k in SEGMENTER_PARAM_KEYS}
    except Exception as e:
        logger.warning(
            "ConfigManager failed for '%s' sensitivity '%s': %s. "
            "Falling back to user overrides only.",
            tool_name, sensitivity, e,
        )
        return {k: v for k, v in (user_overrides or {}).items() if k in SEGMENTER_PARAM_KEYS}


def apply_segmenter_routing(
    params: dict[str, Any],
    segmenter_backend: str | None,
    sensitivity: str,
) -> dict[str, Any]:
    """
    Route segmenter params to their canonical location at resolution time.

    Mutates and returns ``params`` (the resolved config's params dict with
    ``vad`` and optionally ``speech_segmenter`` sections):

    - Grouping keys are MOVED from params["vad"] to params["speech_segmenter"]
      (canonical location all consumers read from).
    - Silero backend (or unset backend): Silero detection params stay in
      params["vad"]; constructors merge vad + speech_segmenter for Silero.
    - Non-Silero backend: params["vad"] is emptied at the source (replaces the
      constructor firewall) and params["speech_segmenter"] is filled from the
      backend's own YAML sensitivity preset.
    - If ``segmenter_backend`` is provided, it is recorded in
      params["speech_segmenter"]["backend"].
    """
    vad_params = params.get("vad") or {}
    segmenter_config = params.get("speech_segmenter") or {}

    if segmenter_backend and not is_silero_backend(segmenter_backend) \
            and segmenter_backend != "none":
        # Non-Silero: Silero-resolved values (detection AND grouping) are
        # meaningless - discard params["vad"] entirely at the source (this
        # replaces the constructor firewall) and resolve this backend's own
        # sensitivity preset instead. Values already present in
        # segmenter_config (explicit user overrides) win over the preset.
        vad_params = {}
        preset = resolve_segmenter_config(segmenter_backend, sensitivity)
        for key, value in preset.items():
            segmenter_config.setdefault(key, value)
    else:
        # Silero backend (or unset): Silero detection params stay in
        # params["vad"]; move the backend-agnostic grouping keys to the
        # canonical location. Constructors merge vad + speech_segmenter
        # for Silero backends, so backends see the same effective values.
        for key in GROUPING_KEYS:
            if key in vad_params:
                segmenter_config.setdefault(key, vad_params.pop(key))

    if segmenter_backend:
        segmenter_config["backend"] = segmenter_backend

    params["vad"] = vad_params
    params["speech_segmenter"] = segmenter_config
    return params
