# Implementation Plan: Ensemble Mode Parameter Handling Fixes

## Overview

This plan addresses parameter routing issues in the ensemble two-pass workflow, specifically:
1. Feature params (like `scene_detection_method`) incorrectly reaching ASR modules
2. Fragile parameter routing based on key existence
3. Lack of explicit validation and warnings for unknown parameters

## Scope

**Files to modify:**
- `whisperjav/ensemble/pass_worker.py` - Primary changes

**Files unchanged:**
- `whisperjav/ensemble/orchestrator.py` - No changes needed
- `whisperjav/ensemble/merge.py` - No changes needed
- `whisperjav/ensemble/utils.py` - No changes needed

---

## Implementation Details

### Step 1: Define Explicit Parameter Categories

**Location:** `pass_worker.py` (after line 47, after `DEFAULT_HF_PARAMS`)

**Add new constant dictionaries:**

```python
# Parameters that belong to different config categories
# These are used to explicitly route custom params to correct destinations

DECODER_PARAMS = {
    "task",
    "language",
    "beam_size",
    "best_of",
    "patience",
    "length_penalty",
    "prefix",
    "suppress_tokens",
    "suppress_blank",
    "without_timestamps",
    "max_initial_timestamp",
}

PROVIDER_PARAMS_COMMON = {
    "temperature",
    "compression_ratio_threshold",
    "logprob_threshold",
    "logprob_margin",
    "no_speech_threshold",
    "drop_nonverbal_vocals",
    "condition_on_previous_text",
    "initial_prompt",
    "word_timestamps",
    "prepend_punctuations",
    "append_punctuations",
    "clip_timestamps",
}

PROVIDER_PARAMS_FASTER_WHISPER = {
    "chunk_length",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "prompt_reset_on_temperature",
    "hotwords",
    "multilingual",
    "max_new_tokens",
    "language_detection_threshold",
    "language_detection_segments",
    "log_progress",
    "hallucination_silence_threshold",
}

PROVIDER_PARAMS_OPENAI_WHISPER = {
    "verbose",
    "carry_initial_prompt",
    "prompt",
    "fp16",
    "hallucination_silence_threshold",
}

PROVIDER_PARAMS_STABLE_TS = {
    "stream",
    "mel_first",
    "split_callback",
    "suppress_ts_tokens",
    "gap_padding",
    "only_ffmpeg",
    "max_instant_words",
    "avg_prob_threshold",
    "nonspeech_skip",
    "progress_callback",
    "ignore_compatibility",
    "extra_models",
    "dynamic_heads",
    "nonspeech_error",
    "only_voice_freq",
    "min_word_dur",
    "min_silence_dur",
    "regroup",
    "ts_num",
    "ts_noise",
    "suppress_silence",
    "suppress_word_ts",
    "suppress_attention",
    "use_word_position",
    "q_levels",
    "k_size",
    "time_scale",
    "denoiser",
    "denoiser_options",
    "demucs",
    "demucs_options",
    "vad",
    "vad_threshold",
}

VAD_PARAMS = {
    "threshold",
    "neg_threshold",
    "min_speech_duration_ms",
    "max_speech_duration_s",
    "min_silence_duration_ms",
    "speech_pad_ms",
}

# Feature params that should NOT be passed to ASR modules
# These are handled at pipeline level, not ASR level
FEATURE_PARAMS = {
    "scene_detection_method",
    "scene_detection",
    "post_processing",
}

# Model-level params (handled separately)
MODEL_PARAMS = {
    "model_name",
    "device",
}
```

---

### Step 2: Create Pipeline-to-Backend Mapping

**Location:** `pass_worker.py` (after the param category constants)

```python
# Map pipeline names to their ASR backends for param validation
PIPELINE_BACKENDS = {
    "balanced": "faster_whisper",
    "fast": "stable_ts",
    "faster": "stable_ts",
    "fidelity": "openai_whisper",
    "kotoba-faster-whisper": "kotoba_faster_whisper",
}

def get_valid_provider_params(pipeline_name: str) -> set:
    """Return the set of valid provider params for a given pipeline."""
    backend = PIPELINE_BACKENDS.get(pipeline_name, "faster_whisper")

    valid = PROVIDER_PARAMS_COMMON.copy()

    if backend == "faster_whisper" or backend == "kotoba_faster_whisper":
        valid.update(PROVIDER_PARAMS_FASTER_WHISPER)
    elif backend == "openai_whisper":
        valid.update(PROVIDER_PARAMS_OPENAI_WHISPER)
    elif backend == "stable_ts":
        valid.update(PROVIDER_PARAMS_STABLE_TS)

    return valid
```

---

### Step 3: Rewrite `apply_custom_params()` Function

**Location:** `pass_worker.py` - Replace the existing `apply_custom_params()` function (lines 284-343)

```python
def apply_custom_params(
    resolved_config: Dict[str, Any],
    custom_params: Dict[str, Any],
    pass_number: int,
    pipeline_name: str,
) -> List[str]:
    """
    Apply custom parameters to resolved config with explicit category routing.

    Args:
        resolved_config: The resolved configuration from resolve_legacy_pipeline()
        custom_params: User-provided custom parameters
        pass_number: Pass number (1 or 2) for logging
        pipeline_name: Pipeline name for backend-specific param validation

    Returns:
        List of unknown parameter names that were not applied
    """
    model_config = resolved_config["model"]
    params = resolved_config["params"]

    # Determine config structure (V3 vs legacy)
    is_v3_config = "asr" in params

    # Track results
    unknown_params: List[str] = []
    discarded_params: List[str] = []
    applied_params: Dict[str, str] = {}  # param -> category

    # Get valid provider params for this pipeline's backend
    valid_provider_params = get_valid_provider_params(pipeline_name)

    for key, value in custom_params.items():
        # 1. Model-level params (always handled first)
        if key in MODEL_PARAMS:
            model_config[key] = value
            applied_params[key] = "model"
            logger.debug("Pass %s: Set model.%s = %s", pass_number, key, value)
            continue

        # 2. Feature params - discard with warning (not applicable to ASR)
        if key in FEATURE_PARAMS:
            discarded_params.append(key)
            logger.debug(
                "Pass %s: Discarding feature param '%s' (not applicable to ASR)",
                pass_number, key
            )
            continue

        # 3. Route based on config structure
        if is_v3_config:
            # V3 structure: params["asr"] contains all ASR params
            asr_params = params["asr"]
            if key in VAD_PARAMS:
                # VAD params need special handling in V3
                if "vad" not in params:
                    params["vad"] = {}
                params["vad"][key] = value
                applied_params[key] = "vad"
            else:
                asr_params[key] = value
                applied_params[key] = "asr"
            logger.debug("Pass %s: Set %s.%s", pass_number, applied_params[key], key)
        else:
            # Legacy structure: params has decoder/provider/vad
            decoder_params = params.get("decoder", {})
            provider_params = params.get("provider", {})
            vad_params = params.get("vad", {})

            if key in DECODER_PARAMS:
                decoder_params[key] = value
                applied_params[key] = "decoder"
            elif key in VAD_PARAMS:
                vad_params[key] = value
                applied_params[key] = "vad"
            elif key in valid_provider_params:
                provider_params[key] = value
                applied_params[key] = "provider"
            else:
                # Unknown param - do NOT add to provider, just track it
                unknown_params.append(key)
                logger.warning(
                    "Pass %s: Unknown param '%s' not applied (not valid for %s)",
                    pass_number, key, pipeline_name
                )
                continue

            logger.debug("Pass %s: Set %s.%s", pass_number, applied_params[key], key)

    # Log summary
    if discarded_params:
        logger.info(
            "Pass %s: Discarded %d feature param(s): %s",
            pass_number, len(discarded_params), ", ".join(sorted(discarded_params))
        )

    if applied_params:
        logger.debug(
            "Pass %s: Applied %d custom param(s)",
            pass_number, len(applied_params)
        )

    return unknown_params
```

---

### Step 4: Update `_build_pipeline()` Call Site

**Location:** `pass_worker.py` - Update the call to `apply_custom_params()` (around line 264)

**Before:**
```python
if pass_config.get("params"):
    unknown_params = apply_custom_params(resolved_config, pass_config["params"], pass_number)
```

**After:**
```python
if pass_config.get("params"):
    unknown_params = apply_custom_params(
        resolved_config=resolved_config,
        custom_params=pass_config["params"],
        pass_number=pass_number,
        pipeline_name=pipeline_name,
    )
```

---

### Step 5: Add Validation for Transformers Feature Params

**Location:** `pass_worker.py` - Update `prepare_transformers_params()` to handle edge cases

The current implementation already correctly maps `scene` to `hf_scene`. However, if a user passes `scene_detection_method` (the legacy name) in hf_params, it should be handled:

**Add after line 99 in `prepare_transformers_params()`:**

```python
    # Handle legacy param name mapping
    legacy_mappings = {
        "scene_detection_method": "hf_scene",  # Legacy name -> HF name
    }

    for legacy_key, hf_key in legacy_mappings.items():
        if legacy_key in hf_params and legacy_key not in mapping:
            params[hf_key] = hf_params[legacy_key]
            logger.debug("Mapped legacy param '%s' to '%s'", legacy_key, hf_key)
```

---

## Testing Plan

### Unit Tests

Add tests to `tests/test_ensemble_params.py` (new file):

```python
"""Tests for ensemble parameter routing."""

import pytest
from whisperjav.ensemble.pass_worker import (
    apply_custom_params,
    get_valid_provider_params,
    DECODER_PARAMS,
    VAD_PARAMS,
    FEATURE_PARAMS,
)


def test_feature_params_discarded():
    """Feature params should be discarded, not routed to provider."""
    resolved_config = {
        "model": {"model_name": "large-v2", "device": "cuda"},
        "params": {
            "decoder": {"task": "transcribe", "language": "ja"},
            "provider": {"temperature": [0.0, 0.1]},
            "vad": {"threshold": 0.4},
        },
    }

    custom_params = {
        "beam_size": 5,
        "scene_detection_method": "auditok",  # Should be discarded
    }

    unknown = apply_custom_params(
        resolved_config, custom_params, pass_number=1, pipeline_name="fidelity"
    )

    # scene_detection_method should NOT be in provider
    assert "scene_detection_method" not in resolved_config["params"]["provider"]
    # beam_size should be in decoder
    assert resolved_config["params"]["decoder"]["beam_size"] == 5
    # No unknown params (scene_detection_method is discarded, not unknown)
    assert unknown == []


def test_unknown_params_not_added():
    """Unknown params should not be added to provider."""
    resolved_config = {
        "model": {"model_name": "large-v2", "device": "cuda"},
        "params": {
            "decoder": {"task": "transcribe"},
            "provider": {"temperature": [0.0]},
            "vad": {},
        },
    }

    custom_params = {
        "totally_fake_param": 123,
    }

    unknown = apply_custom_params(
        resolved_config, custom_params, pass_number=1, pipeline_name="fidelity"
    )

    assert "totally_fake_param" in unknown
    assert "totally_fake_param" not in resolved_config["params"]["provider"]


def test_decoder_params_routed_correctly():
    """Decoder params should go to decoder, not provider."""
    resolved_config = {
        "model": {},
        "params": {
            "decoder": {},
            "provider": {},
            "vad": {},
        },
    }

    custom_params = {
        "beam_size": 10,
        "best_of": 3,
        "patience": 1.5,
    }

    apply_custom_params(
        resolved_config, custom_params, pass_number=1, pipeline_name="balanced"
    )

    assert resolved_config["params"]["decoder"]["beam_size"] == 10
    assert resolved_config["params"]["decoder"]["best_of"] == 3
    assert resolved_config["params"]["decoder"]["patience"] == 1.5


def test_vad_params_routed_correctly():
    """VAD params should go to vad, not provider."""
    resolved_config = {
        "model": {},
        "params": {
            "decoder": {},
            "provider": {},
            "vad": {},
        },
    }

    custom_params = {
        "threshold": 0.3,
        "min_speech_duration_ms": 100,
        "speech_pad_ms": 500,
    }

    apply_custom_params(
        resolved_config, custom_params, pass_number=2, pipeline_name="fidelity"
    )

    assert resolved_config["params"]["vad"]["threshold"] == 0.3
    assert resolved_config["params"]["vad"]["min_speech_duration_ms"] == 100
    assert resolved_config["params"]["vad"]["speech_pad_ms"] == 500


def test_backend_specific_provider_params():
    """Provider params should be validated against backend type."""
    # Test that faster_whisper-specific params are valid for balanced
    valid_balanced = get_valid_provider_params("balanced")
    assert "repetition_penalty" in valid_balanced
    assert "no_repeat_ngram_size" in valid_balanced

    # Test that openai_whisper-specific params are valid for fidelity
    valid_fidelity = get_valid_provider_params("fidelity")
    assert "fp16" in valid_fidelity
    assert "verbose" in valid_fidelity

    # Test that stable_ts-specific params are valid for fast/faster
    valid_fast = get_valid_provider_params("fast")
    assert "regroup" in valid_fast
    assert "suppress_silence" in valid_fast
```

### Integration Tests

Run the user's exact test case:

```bash
whisperjav video.mp4 --ensemble \
  --pass1-pipeline transformers \
  --pass1-hf-params '{"scene": "auditok"}' \
  --pass2-pipeline fidelity \
  --pass2-sensitivity aggressive \
  --pass2-params '{"model_name": "large-v2", "scene_detection_method": "auditok", "beam_size": 2}' \
  --merge-strategy smart_merge
```

**Expected behavior:**
- Pass 1: `scene` correctly maps to `hf_scene`
- Pass 2: `scene_detection_method` is discarded with log message, other params applied correctly

---

## Summary of Changes

| File | Change | Purpose |
|------|--------|---------|
| `pass_worker.py` | Add param category constants | Explicit routing rules |
| `pass_worker.py` | Add `get_valid_provider_params()` | Backend-aware validation |
| `pass_worker.py` | Rewrite `apply_custom_params()` | Robust param routing |
| `pass_worker.py` | Update `_build_pipeline()` call | Pass pipeline_name |
| `pass_worker.py` | Update `prepare_transformers_params()` | Legacy name mapping |
| `tests/test_ensemble_params.py` | Add unit tests | Verify routing logic |

---

## Rollback Plan

If issues arise:
1. The changes are isolated to `pass_worker.py`
2. Revert to previous `apply_custom_params()` implementation
3. Add `scene_detection_method` to a blocklist as minimal fix

---

## Future Considerations

1. **Feature customization support**: If users need to customize features via params, add explicit routing to `resolved_config["features"]`

2. **Parameter documentation**: Generate documentation from the category constants to help users understand valid params per pipeline

3. **GUI validation**: Use the same category constants for frontend validation before submission
