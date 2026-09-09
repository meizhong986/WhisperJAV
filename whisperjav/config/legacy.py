"""
Legacy Pipeline Mappings for WhisperJAV v3.0.

.. deprecated:: 1.7.0
    This module is part of the LEGACY configuration system (v1-v3).
    For new development, use the v4 YAML-driven configuration system:

        from whisperjav.config.v4 import ConfigManager

    See: whisperjav/config/v4/README.md
    ADR: docs/adr/ADR-001-yaml-config-architecture.md

PURPOSE AND SCOPE
=================
This module provides configuration resolution for pipelines that use the
LEGACY CONFIG SYSTEM. It maps pipeline mode names (e.g., "balanced", "fast")
to component-based configurations with ASR, VAD, and feature settings.

WHAT BELONGS HERE
-----------------
Pipelines that:
- Use the `resolve_legacy_pipeline()` function for configuration
- Accept `resolved_config` parameter in their __init__
- Rely on the v3 config structure with model, params, and features sections
- Examples: balanced, fast, faster, fidelity, kotoba-faster-whisper

WHAT DOES NOT BELONG HERE
-------------------------
Pipelines that:
- Use dedicated CLI arguments instead of legacy config resolution
- Bypass `resolve_legacy_pipeline()` entirely in main.py
- Have their own independent configuration system
- Example: "transformers" mode uses --hf-* arguments directly

ADDING NEW PIPELINES
--------------------
Before adding a new pipeline to LEGACY_PIPELINES, ask:
1. Does it need the legacy config resolution system?
2. Will it accept `resolved_config` in its __init__?
3. Does it use the standard ASR/VAD/features component model?

If NO to any of these, the pipeline should NOT be added here.
Instead, handle its config resolution separately in main.py.

Maps old pipeline names to new component-based configurations
for backward compatibility.
"""

from typing import Any, Dict, List, Optional

from .resolver_v3 import resolve_config_v3
from whisperjav.utils.logger import logger


def _filter_none_values(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively remove None values from a dictionary.

    This is critical for backend compatibility. When passing parameters to
    libraries like faster-whisper, openai-whisper, or stable-ts:

    - Passing `func(arg=None)` explicitly passes None, NOT the library default
    - This causes errors like "TypeError: 'NoneType' object is not iterable"
    - By removing None keys, we allow library defaults to apply

    Preserves valid falsy values: 0, False, "", [], {}

    Args:
        params: Dictionary of parameters (may be nested)

    Returns:
        Filtered dictionary with None values removed
    """
    if not isinstance(params, dict):
        return params

    filtered = {}
    for key, value in params.items():
        if value is None:
            # Skip None values - let library defaults apply
            continue
        elif isinstance(value, dict):
            # Recursively filter nested dicts
            filtered_nested = _filter_none_values(value)
            # Only include if non-empty after filtering
            if filtered_nested:
                filtered[key] = filtered_nested
        else:
            # Preserve all other values including falsy ones (0, False, "", [])
            filtered[key] = value

    return filtered


# Legacy pipeline definitions
LEGACY_PIPELINES = {
    "balanced": {
        "asr": "faster_whisper",
        # NOTE: this `vad` field names a VAD Pydantic component (registry has
        # only "silero" — defines the preset values that flow into
        # params["vad"]). It is NOT the runtime speech-segmenter backend
        # selector. The runtime segmenter default (v1.8.13: whisperseg) is
        # set in whisper_pro_asr.py / faster_whisper_pro_asr.py fallbacks
        # and gates params["vad"] via the firewall (clears silero presets
        # for non-silero runtime backends). Keep this as silero-v3.1 so the
        # resolver loads the v3.1 preset values; the firewall handles the
        # rest when whisperseg becomes the runtime default.
        "vad": "silero-v3.1",
        "features": ["auditok_scene_detection"],
        # v1.9.2 (owner decision): Balanced scenes are at least 28 s and at most 20
        # minutes. The keys differ by backend because the backends read different names,
        # and 28 is only SAFE on semantic, whose min_duration merges. On auditok
        # min_duration DISCARDS shorter regions, so only the ceiling is set there.
        # fast and fidelity deliberately declare nothing here and keep the backend's own
        # defaults (owner decision: balanced only).
        "scene_overrides": {
            "semantic": {
                "scene_detection.min_duration": 28.0,
                "scene_detection.max_duration": 1200.0,
            },
            "auditok": {
                "scene_detection.max_duration_s": 1200.0,
                "scene_detection.pass1_max_duration_s": 1200.0,
            },
        },
        "description": "Full feature set with scene detection and VAD. Best quality.",
    },
    "faster": {
        "asr": "stable_ts",
        "vad": "none",
        "features": [],
        "description": "Speed-optimized with Stable-TS. No VAD or scene detection.",
    },
    "fast": {
        "asr": "stable_ts",
        "vad": "none",
        "features": ["auditok_scene_detection"],
        "description": "Stable-TS with scene detection. Good speed/quality balance.",
    },
    "fidelity": {
        "asr": "openai_whisper",
        # See balanced note above — same architecture applies.
        "vad": "silero-v3.1",
        "features": ["auditok_scene_detection"],
        "description": "OpenAI Whisper with VAD and scene detection. Maximum fidelity.",
    },
    "kotoba-faster-whisper": {
        "asr": "kotoba_faster_whisper",
        "vad": "none",  # Uses internal VAD (faster-whisper built-in)
        "features": ["auditok_scene_detection"],  # Scene detection always enabled
        "description": "Japanese-optimized Kotoba Faster-Whisper with internal VAD.",
        "use_v3_structure": True,  # Return V3 config, not legacy mapped
    },
    # NOTE: "transformers" mode is NOT listed here.
    # It uses dedicated --hf-* CLI arguments and bypasses legacy config resolution entirely.
    # See whisperjav/pipelines/transformers_pipeline.py for its implementation.
}


# Scene-detection feature component per runtime backend (v1.9.2).
#
# Before v1.9.2 every scene-detecting pipeline declared "auditok_scene_detection"
# regardless of --scene-detection-method, so a semantic run was handed auditok's
# parameter names (max_duration_s, min_duration_s, pass1_*) and silently fell back to
# the engine's own hard-coded defaults -- the semantic backend reads min_duration /
# max_duration without the _s suffix and ignores the rest.
#
# Resolving the feature from the effective method also closes a trap that appears once
# a semantic component exists: the auditok backend falls back to the bare names when
# the _s ones are absent, and its min_duration DISCARDS shorter regions, so handing
# auditok a semantic min_duration of 28 would delete every region under 28 seconds.
SCENE_FEATURE_BY_METHOD = {
    "auditok": "auditok_scene_detection",
    "silero": "silero_scene_detection",
    "semantic": "semantic_scene_detection",
    "none": None,          # no scene detection feature at all
}

_SCENE_FEATURE_NAMES = frozenset(
    name for name in SCENE_FEATURE_BY_METHOD.values() if name
)

_METHOD_BY_SCENE_FEATURE = {
    feature: method
    for method, feature in SCENE_FEATURE_BY_METHOD.items()
    if feature
}


def _select_scene_feature(
    declared_features: List[str],
    scene_method: Optional[str],
) -> List[str]:
    """Swap the declared scene-detection feature for the one matching ``scene_method``.

    A pipeline that declares NO scene-detection feature (``faster``) never gains one,
    whatever the method says -- it has no scene detector to configure. An unknown
    method leaves the declaration untouched rather than guessing.

    CONTRACT for ``scene_method="none"``: the scene feature is removed, so the resolved
    config carries no ``features["scene_detection"]`` at all. The caller must then set
    ``{"method": "none"}`` itself, because ``SceneDetectorFactory`` falls back to auditok
    when no method is present. The ensemble worker does exactly that
    (``_apply_gui_overrides``); ``--scene-detection-method`` deliberately does not offer
    "none", so the single-pass path cannot reach this case.
    """
    if not scene_method:
        return list(declared_features)

    key = str(scene_method).strip().lower()
    if key not in SCENE_FEATURE_BY_METHOD:
        logger.warning(
            "Unknown scene detection method '%s'; keeping the pipeline's declared "
            "scene feature(s) %s", scene_method, list(declared_features),
        )
        return list(declared_features)

    declares_scene = any(f in _SCENE_FEATURE_NAMES for f in declared_features)
    if not declares_scene:
        return list(declared_features)

    kept = [f for f in declared_features if f not in _SCENE_FEATURE_NAMES]
    target = SCENE_FEATURE_BY_METHOD[key]
    if target:
        kept.append(target)
    return kept


def _normalise_scene_method(
    scene_method: Optional[str],
    declared_features: List[str],
) -> str:
    """The method whose parameter names this resolution will produce.

    Falls back to whatever the pipeline declares, so callers that pass no method keep
    exactly the pre-v1.9.2 behaviour.
    """
    if scene_method:
        key = str(scene_method).strip().lower()
        if key in SCENE_FEATURE_BY_METHOD:
            return key
    for feature in declared_features:
        if feature in _METHOD_BY_SCENE_FEATURE:
            return _METHOD_BY_SCENE_FEATURE[feature]
    return "none"


def resolve_legacy_pipeline(
    pipeline_name: str,
    sensitivity: str = "balanced",
    task: str = "transcribe",
    overrides: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    scene_method: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve configuration from legacy pipeline name.

    This provides backward compatibility with the old CLI interface.

    Args:
        pipeline_name: Legacy pipeline name ('balanced', 'faster', 'fast', 'fidelity', 'kotoba-faster-whisper')
        sensitivity: Sensitivity level
        task: Task type
        overrides: Parameter overrides
        device: Device override (None/'auto' = auto-detect, 'cuda'/'cpu' = explicit)
        compute_type: Compute type override (None/'auto' = provider-specific default)
        scene_method: Scene-detection backend that will actually run ('auditok',
            'silero', 'semantic', 'none'). v1.9.2: selects the matching scene feature
            component so the resolved parameter NAMES are the ones that backend reads,
            and selects the pipeline's per-backend scene overrides. None means "whatever
            the pipeline declares" (auditok for the legacy pipelines) -- the feature list
            is left alone, and that backend's scene overrides still apply, because they
            describe the backend that will actually run.

    Returns:
        Resolved configuration dictionary.

    Example:
        >>> config = resolve_legacy_pipeline('balanced', 'aggressive')
        >>> # With explicit hardware:
        >>> config = resolve_legacy_pipeline('balanced', device='cuda', compute_type='int8_float16')
    """
    if pipeline_name not in LEGACY_PIPELINES:
        available = list(LEGACY_PIPELINES.keys())
        raise ValueError(f"Unknown pipeline: {pipeline_name}. Available: {available}")

    pipeline_def = LEGACY_PIPELINES[pipeline_name]

    # v1.9.2: the scene-detection feature follows the backend that will actually run,
    # so the resolved parameter names match the backend that reads them.
    features = _select_scene_feature(pipeline_def["features"], scene_method)

    # v1.9.2: a pipeline may declare its own scene bounds, per backend, because the
    # parameter NAMES differ between backends. Merged under any caller overrides, which
    # keep precedence.
    effective_overrides = dict(pipeline_def.get("scene_overrides", {}).get(
        _normalise_scene_method(scene_method, pipeline_def["features"]), {}
    ))
    if overrides:
        effective_overrides.update(overrides)

    # Resolve using new system
    config = resolve_config_v3(
        asr=pipeline_def["asr"],
        vad=pipeline_def["vad"],
        sensitivity=sensitivity,
        task=task,
        features=features,
        overrides=effective_overrides or None,
        device=device,
        compute_type=compute_type,
    )

    # Add legacy compatibility fields
    config['pipeline_name'] = pipeline_name
    config['sensitivity_name'] = sensitivity

    # For pipelines that use V3 structure (like kotoba-faster-whisper),
    # return V3 config directly without legacy mapping
    if pipeline_def.get("use_v3_structure", False):
        return config

    # Map to old output structure for backward compatibility
    return _map_to_legacy_structure(config, pipeline_def)


def _map_to_legacy_structure(config: Dict[str, Any], pipeline_def: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map v3 config to legacy output structure.

    This ensures pipelines that expect the old structure continue to work.
    Maps ALL parameters from v3 flat structure to v1 nested decoder/provider structure.
    """
    # Build workflow structure (legacy)
    workflow = {
        'model': config['model']['model_name'],
        'vad': config['vad_name'] if config['vad_name'] != 'none' else 'none',
        'backend': _get_backend_name(config['asr_name']),
    }

    # Add features to workflow
    if config['features']:
        workflow['features'] = {
            feature_type: True for feature_type in config['features'].keys()
        }

    # Build params structure (legacy)
    # Map 'asr' to 'decoder' and 'provider' for backward compat
    asr_params = config['params']['asr']
    asr_name = config['asr_name']

    # Decoder params - from common_decoder_options (same for all backends)
    decoder_params = {
        'task': asr_params.get('task', 'transcribe'),
        'language': asr_params.get('language', 'ja'),
        'beam_size': asr_params.get('beam_size', 2),
        'best_of': asr_params.get('best_of', 2),              # v1.8.10-hf3: 1→2, match balanced Pydantic
        'patience': asr_params.get('patience', 1.6),            # v1.8.10-hf3: 2.0→1.6, match balanced Pydantic
        'length_penalty': asr_params.get('length_penalty'),
        'prefix': asr_params.get('prefix'),
        'suppress_tokens': asr_params.get('suppress_tokens'),
        'suppress_blank': asr_params.get('suppress_blank', True),
        'without_timestamps': asr_params.get('without_timestamps', False),
        'max_initial_timestamp': asr_params.get('max_initial_timestamp'),
    }

    # Provider params - BACKEND-SPECIFIC
    # Different backends accept different parameters
    # Common transcriber options (shared by all backends)
    provider_params = {
        'temperature': asr_params.get('temperature', [0.0]),                  # CL1b: match balanced Pydantic default
        'compression_ratio_threshold': asr_params.get('compression_ratio_threshold', 2.4),
        'logprob_threshold': asr_params.get('logprob_threshold', -1.00),        # v1.8.10-hf2: -0.75→-1.00, match balanced Pydantic
        'logprob_margin': asr_params.get('logprob_margin', 0.0),              # v1.8.10-hf2: 0.2→0.0, match balanced Pydantic
        'no_speech_threshold': asr_params.get('no_speech_threshold', 0.65),   # v1.8.10-hf3: 0.70→0.65, match balanced Pydantic
        'drop_nonverbal_vocals': asr_params.get('drop_nonverbal_vocals', False),
        'post_model_filter_enabled': asr_params.get('post_model_filter_enabled'),  # None = use ASR module's pipeline-specific default
        'condition_on_previous_text': asr_params.get('condition_on_previous_text', False),
        'initial_prompt': asr_params.get('initial_prompt'),
        'word_timestamps': asr_params.get('word_timestamps', True),
        'prepend_punctuations': asr_params.get('prepend_punctuations'),
        'append_punctuations': asr_params.get('append_punctuations'),
        'clip_timestamps': asr_params.get('clip_timestamps'),
    }

    # Add backend-specific engine options
    if asr_name == 'faster_whisper':
        # faster_whisper_engine_options
        provider_params.update({
            'chunk_length': asr_params.get('chunk_length'),
            'repetition_penalty': asr_params.get('repetition_penalty', 1.5),
            'no_repeat_ngram_size': asr_params.get('no_repeat_ngram_size', 3),     # H1: was 2, match Pydantic default
            'prompt_reset_on_temperature': asr_params.get('prompt_reset_on_temperature'),
            'hotwords': asr_params.get('hotwords'),
            'multilingual': asr_params.get('multilingual', False),
            'max_new_tokens': asr_params.get('max_new_tokens'),
            'language_detection_threshold': asr_params.get('language_detection_threshold'),
            'language_detection_segments': asr_params.get('language_detection_segments'),
            'log_progress': asr_params.get('log_progress', False),
            # exclusive_whisper_plus_faster_whisper
            'hallucination_silence_threshold': asr_params.get('hallucination_silence_threshold'),  # None = disabled, no fallback (C1 fix)
        })
    elif asr_name == 'openai_whisper':
        # openai_whisper_engine_options
        provider_params.update({
            'verbose': asr_params.get('verbose'),
            'carry_initial_prompt': asr_params.get('carry_initial_prompt'),
            'prompt': asr_params.get('prompt'),
            'fp16': asr_params.get('fp16', True),
            # exclusive_whisper_plus_faster_whisper
            'hallucination_silence_threshold': asr_params.get('hallucination_silence_threshold'),  # None = disabled, no fallback (C1 fix)
        })
    elif asr_name == 'stable_ts':
        # stable_ts_engine_options
        provider_params.update({
            'stream': asr_params.get('stream'),
            'mel_first': asr_params.get('mel_first'),
            'split_callback': asr_params.get('split_callback'),
            'suppress_ts_tokens': asr_params.get('suppress_ts_tokens', False),
            'gap_padding': asr_params.get('gap_padding', ' ...'),
            'only_ffmpeg': asr_params.get('only_ffmpeg', False),
            'max_instant_words': asr_params.get('max_instant_words', 0.5),
            'avg_prob_threshold': asr_params.get('avg_prob_threshold'),
            'nonspeech_skip': asr_params.get('nonspeech_skip'),
            'progress_callback': asr_params.get('progress_callback'),
            'ignore_compatibility': asr_params.get('ignore_compatibility', True),
            'extra_models': asr_params.get('extra_models'),
            'dynamic_heads': asr_params.get('dynamic_heads'),
            'nonspeech_error': asr_params.get('nonspeech_error', 0.1),
            'only_voice_freq': asr_params.get('only_voice_freq', False),
            'min_word_dur': asr_params.get('min_word_dur'),
            'min_silence_dur': asr_params.get('min_silence_dur'),
            'regroup': asr_params.get('regroup', True),
            'ts_num': asr_params.get('ts_num', 0),
            'ts_noise': asr_params.get('ts_noise'),
            'suppress_silence': asr_params.get('suppress_silence', True),
            'suppress_word_ts': asr_params.get('suppress_word_ts', True),
            'suppress_attention': asr_params.get('suppress_attention', False),
            'use_word_position': asr_params.get('use_word_position', True),
            'q_levels': asr_params.get('q_levels', 20),
            'k_size': asr_params.get('k_size', 5),
            'time_scale': asr_params.get('time_scale'),
            'denoiser': asr_params.get('denoiser'),
            'denoiser_options': asr_params.get('denoiser_options'),
            'demucs': asr_params.get('demucs', False),
            'demucs_options': asr_params.get('demucs_options'),
            # VAD options for stable_ts
            'vad': asr_params.get('vad', True),
            'vad_threshold': asr_params.get('vad_threshold', 0.25),
        })

    # Filter None values from all param sections
    # This is critical: passing None explicitly to backends causes errors
    # By removing None, we let library defaults apply
    filtered_decoder = _filter_none_values(decoder_params)
    filtered_provider = _filter_none_values(provider_params)
    filtered_vad = _filter_none_values(config['params']['vad'])
    filtered_features = _filter_none_values(config['features'])

    return {
        'pipeline_name': config['pipeline_name'],
        'sensitivity_name': config['sensitivity_name'],
        'workflow': workflow,
        'model': config['model'],
        'params': {
            'decoder': filtered_decoder,
            'provider': filtered_provider,
            'vad': filtered_vad,
        },
        'features': filtered_features,
        'task': config['task'],
        'language': config['language'],
    }


def _get_backend_name(asr_name: str) -> str:
    """Get backend name from ASR component name."""
    backend_map = {
        'faster_whisper': 'faster-whisper',
        'stable_ts': 'stable-ts',
        'openai_whisper': 'whisper',
    }
    return backend_map.get(asr_name, asr_name)


def resolve_ensemble_config(
    asr: str,
    vad: str = "none",
    task: str = "transcribe",
    features: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve configuration for ensemble mode (direct component specification).

    This is the entry point for the Ensemble Tab GUI, allowing users to
    specify components and parameters directly without using preset pipelines.

    Args:
        asr: ASR component name (e.g., 'faster_whisper', 'stable_ts', 'openai_whisper')
        vad: VAD component name or 'none'
        task: Task type ('transcribe' or 'translate')
        features: List of feature names (e.g., ['auditok_scene_detection'])
        overrides: Parameter overrides in flat dot notation
                   (e.g., {'asr.beam_size': 10, 'vad.threshold': 0.25})
        device: Device override (None/'auto' = auto-detect, 'cuda'/'cpu' = explicit)
        compute_type: Compute type override (None/'auto' = provider-specific default)

    Returns:
        Resolved configuration in legacy structure for pipeline compatibility.
    """
    # Convert flat overrides to nested structure
    nested_overrides = None
    if overrides:
        nested_overrides = {}
        for key, value in overrides.items():
            parts = key.split('.')
            if len(parts) >= 2:
                comp_type = parts[0]
                param_path = '.'.join(parts[1:])

                if comp_type not in nested_overrides:
                    nested_overrides[comp_type] = {}

                # Handle nested params like features.scene_detection.max_duration_s
                if comp_type == 'features' and len(parts) >= 3:
                    feature_name = parts[1]
                    param_name = '.'.join(parts[2:])
                    if feature_name not in nested_overrides[comp_type]:
                        nested_overrides[comp_type][feature_name] = {}
                    nested_overrides[comp_type][feature_name][param_name] = value
                else:
                    nested_overrides[comp_type][param_path] = value

    # Resolve using v3 system
    config = resolve_config_v3(
        asr=asr,
        vad=vad,
        sensitivity='balanced',  # Ensemble doesn't use sensitivity presets
        task=task,
        features=features or [],
        overrides=nested_overrides,
        device=device,
        compute_type=compute_type,
    )

    # Add ensemble-specific metadata
    config['pipeline_name'] = 'ensemble'
    config['sensitivity_name'] = 'custom'

    # Build pipeline definition for mapping
    pipeline_def = {
        "asr": asr,
        "vad": vad,
        "features": features or [],
        "description": "Custom ensemble configuration",
    }

    # Map to legacy structure for pipeline compatibility
    return _map_to_legacy_structure(config, pipeline_def)


def list_legacy_pipelines() -> List[str]:
    """List available legacy pipeline names."""
    return list(LEGACY_PIPELINES.keys())


def get_legacy_pipeline_info(pipeline_name: str) -> Dict[str, Any]:
    """Get information about a legacy pipeline."""
    if pipeline_name not in LEGACY_PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    return {
        'name': pipeline_name,
        **LEGACY_PIPELINES[pipeline_name]
    }


def apply_balanced_vad_defaults(
    resolved_config: Dict[str, Any],
    sensitivity: str = "balanced",
    is_balanced: bool = False,
) -> None:
    """
    Apply the v1.9.0 balanced-pipeline VAD defaults to a resolved config, IN PLACE.

    SHARED by the single-pass path (main.py) and the ensemble path
    (ensemble/pass_worker.py) so the two cannot drift. Behaviour:

    * speech_segmenter backend == "faster-whisper"  → overlay the tuned
      ``faster_whisper_vad`` VadOptions preset (threshold scaled for
      faster-whisper's BUNDLED Silero, e.g. balanced=0.40 — NOT the external
      silero-v3.1/v6.2 scale, e.g. 0.28, which over-triggers here).
    * ``is_balanced`` AND an EXTERNAL segmenter selected → "Test D" fine-grained
      grouping default (max_group_duration_s=9.0, chunk_threshold_s=0.1), the
      best-quality config from the owner A/B (2026-06-30).

    The effective backend is read from
    ``resolved_config["params"]["speech_segmenter"]["backend"]``. Call this AFTER
    the backend is set and BEFORE explicit user overrides (CLI flags / GUI
    Customize params), which must win over these defaults.

    Args:
        resolved_config: Resolved config dict (mutated in place).
        sensitivity: Sensitivity level for the native VAD preset lookup.
        is_balanced: True iff this is the balanced pipeline/mode (gates Test-D).
    """
    params = resolved_config.setdefault("params", {})
    backend = (params.get("speech_segmenter") or {}).get("backend")

    if backend == "faster-whisper":
        # Native VAD: use the scale-correct faster_whisper_vad preset.
        try:
            from whisperjav.config.components.base import get_vad_registry
            fw_vad = get_vad_registry().get("faster_whisper_vad")
            preset = fw_vad.get_preset(sensitivity) if fw_vad else None
            if preset is not None:
                params["vad"] = preset.model_dump()
                logger.debug(
                    "apply_balanced_vad_defaults: native faster_whisper_vad preset "
                    "(sensitivity=%s): %s", sensitivity, params["vad"],
                )
            else:
                logger.warning(
                    "faster_whisper_vad preset not found for sensitivity '%s'; "
                    "native VAD will use faster-whisper library defaults", sensitivity,
                )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not apply faster_whisper_vad preset: %s", e)

    elif is_balanced and backend not in (None, "", "none"):
        # Balanced + external segmenter → Test-D fine-grained grouping default.
        #
        # UNREACHABLE since v1.9.2 (owner S2/S9): both entry points normalise a balanced
        # pipeline to the built-in VAD before calling this, so `backend` is always
        # "faster-whisper" here and the first branch takes it. Kept, not deleted, because
        # this helper is the shared contract for both entry points and the branch would
        # be the correct behaviour again if balanced ever regained a segmenter.
        vad = params.setdefault("vad", {})
        ss = params.setdefault("speech_segmenter", {})
        vad["max_group_duration_s"] = 9.0
        vad["chunk_threshold_s"] = 0.1
        ss["max_group_duration_s"] = 9.0
        ss["chunk_threshold_s"] = 0.1
        logger.info(
            "Balanced + external segmenter '%s': Test-D grouping defaults applied "
            "(max_group=9.0s, chunk_threshold=0.1s)", backend,
        )
