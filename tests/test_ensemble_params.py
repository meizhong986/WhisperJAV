"""Tests for ensemble parameter routing in pass_worker.py.

These tests verify that custom parameters are correctly routed to the
appropriate config sections (model, decoder, provider, vad) and that
feature params are properly discarded.
"""

import pytest
from copy import deepcopy

from whisperjav.ensemble.pass_worker import (
    apply_custom_params,
    get_valid_provider_params,
    prepare_transformers_params,
    DECODER_PARAMS,
    SEGMENTER_PARAMS,
    FEATURE_PARAMS,
    MODEL_PARAMS,
    PROVIDER_PARAMS_COMMON,
    PROVIDER_PARAMS_FASTER_WHISPER,
    PROVIDER_PARAMS_OPENAI_WHISPER,
    PROVIDER_PARAMS_STABLE_TS,
    PIPELINE_BACKENDS,
)

# Backward-compat alias (renamed in Sprint 3 scene detection refactor)
VAD_PARAMS = SEGMENTER_PARAMS


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def legacy_resolved_config():
    """Sample legacy resolved config (decoder/provider/vad structure)."""
    return {
        "model": {"model_name": "large-v2", "device": "cuda"},
        "params": {
            "decoder": {
                "task": "transcribe",
                "language": "ja",
                "beam_size": 2,
                "best_of": 1,
                "patience": 2.0,
                "suppress_blank": True,
                "without_timestamps": False,
            },
            "provider": {
                "temperature": [0.0, 0.1],
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.2,
                "no_speech_threshold": 0.5,
                "word_timestamps": True,
            },
            "vad": {
                "threshold": 0.4,
                "min_speech_duration_ms": 150,
            },
        },
    }


@pytest.fixture
def v3_resolved_config():
    """Sample V3 resolved config (asr structure)."""
    return {
        "model": {"model_name": "large-v2", "device": "cuda"},
        "params": {
            "asr": {
                "task": "transcribe",
                "language": "ja",
                "beam_size": 2,
                "temperature": [0.0, 0.1],
            },
            "vad": {
                "threshold": 0.4,
            },
        },
    }


# =============================================================================
# Test: Feature params discarded
# =============================================================================

class TestFeatureParamsDiscarded:
    """Feature params should be discarded, not routed to provider."""

    def test_scene_detection_method_discarded(self, legacy_resolved_config):
        """scene_detection_method should NOT reach provider params."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            "beam_size": 5,
            "scene_detection_method": "auditok",  # Should be discarded
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="fidelity",
        )

        # scene_detection_method should NOT be in provider
        assert "scene_detection_method" not in config["params"]["provider"]
        # beam_size should be in decoder
        assert config["params"]["decoder"]["beam_size"] == 5
        # No unknown params (scene_detection_method is discarded, not unknown)
        assert unknown == []

    def test_multiple_feature_params_discarded(self, legacy_resolved_config):
        """Multiple feature params should all be discarded."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            "scene_detection_method": "auditok",
            "scene_detection": True,
            "post_processing": {"enabled": True},
            "beam_size": 10,
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=2,
            pipeline_name="balanced",
        )

        # None of the feature params should be in provider
        for key in FEATURE_PARAMS:
            assert key not in config["params"]["provider"]

        # beam_size should still be applied
        assert config["params"]["decoder"]["beam_size"] == 10
        assert unknown == []


# =============================================================================
# Test: Unknown params rejected
# =============================================================================

class TestUnknownParamsRejected:
    """Unknown params should not be added to provider."""

    def test_unknown_param_not_added(self, legacy_resolved_config):
        """Completely unknown params should be rejected, not added."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            "totally_fake_param": 123,
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="fidelity",
        )

        assert "totally_fake_param" in unknown
        assert "totally_fake_param" not in config["params"]["provider"]

    def test_multiple_unknown_params_tracked(self, legacy_resolved_config):
        """All unknown params should be tracked and rejected."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            "fake_param_1": "value1",
            "fake_param_2": "value2",
            "beam_size": 5,  # This is valid
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="balanced",
        )

        assert len(unknown) == 2
        assert "fake_param_1" in unknown
        assert "fake_param_2" in unknown
        # Valid param should still be applied
        assert config["params"]["decoder"]["beam_size"] == 5


# =============================================================================
# Test: Decoder params routing
# =============================================================================

class TestDecoderParamsRouting:
    """Decoder params should go to decoder, not provider."""

    def test_decoder_params_routed_correctly(self, legacy_resolved_config):
        """All decoder params should route to decoder section."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            "beam_size": 10,
            "best_of": 3,
            "patience": 1.5,
            "task": "translate",
            "language": "en",
        }

        apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="balanced",
        )

        assert config["params"]["decoder"]["beam_size"] == 10
        assert config["params"]["decoder"]["best_of"] == 3
        assert config["params"]["decoder"]["patience"] == 1.5
        assert config["params"]["decoder"]["task"] == "translate"
        assert config["params"]["decoder"]["language"] == "en"

    def test_suppress_tokens_routed_to_decoder(self, legacy_resolved_config):
        """suppress_tokens is a decoder param."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            "suppress_tokens": [-1],
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="fidelity",
        )

        assert config["params"]["decoder"]["suppress_tokens"] == [-1]
        assert unknown == []


# =============================================================================
# Test: VAD params routing
# =============================================================================

class TestVADParamsRouting:
    """VAD params should go to vad, not provider."""

    def test_vad_params_routed_correctly(self, legacy_resolved_config):
        """All VAD params should route to vad section."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            "threshold": 0.3,
            "min_speech_duration_ms": 100,
            "speech_pad_ms": 500,
            "neg_threshold": 0.2,
        }

        apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=2,
            pipeline_name="fidelity",
        )

        assert config["params"]["vad"]["threshold"] == 0.3
        assert config["params"]["vad"]["min_speech_duration_ms"] == 100
        assert config["params"]["vad"]["speech_pad_ms"] == 500
        assert config["params"]["vad"]["neg_threshold"] == 0.2


# =============================================================================
# Test: Provider params routing (backend-specific)
# =============================================================================

class TestProviderParamsRouting:
    """Provider params should be validated against backend type."""

    def test_faster_whisper_params_valid_for_balanced(self, legacy_resolved_config):
        """faster-whisper specific params valid for balanced pipeline."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            "repetition_penalty": 1.8,
            "no_repeat_ngram_size": 3,
            "hallucination_silence_threshold": 2.5,
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="balanced",
        )

        assert config["params"]["provider"]["repetition_penalty"] == 1.8
        assert config["params"]["provider"]["no_repeat_ngram_size"] == 3
        assert config["params"]["provider"]["hallucination_silence_threshold"] == 2.5
        assert unknown == []

    def test_openai_whisper_params_valid_for_fidelity(self, legacy_resolved_config):
        """openai-whisper specific params valid for fidelity pipeline."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            "fp16": False,
            "verbose": True,
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="fidelity",
        )

        assert config["params"]["provider"]["fp16"] is False
        assert config["params"]["provider"]["verbose"] is True
        assert unknown == []

    def test_stable_ts_params_valid_for_fast(self, legacy_resolved_config):
        """stable-ts specific params valid for fast pipeline."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            "regroup": False,
            "suppress_silence": True,
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="fast",
        )

        assert config["params"]["provider"]["regroup"] is False
        assert config["params"]["provider"]["suppress_silence"] is True
        assert unknown == []

    def test_wrong_backend_param_rejected(self, legacy_resolved_config):
        """Params for wrong backend should be rejected."""
        config = deepcopy(legacy_resolved_config)
        # repetition_penalty is faster-whisper only, not valid for fidelity (openai)
        custom_params = {
            "repetition_penalty": 1.5,  # Invalid for openai-whisper
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="fidelity",
        )

        assert "repetition_penalty" in unknown
        assert "repetition_penalty" not in config["params"]["provider"]


# =============================================================================
# Test: Model params routing
# =============================================================================

class TestModelParamsRouting:
    """Model params should go to model config."""

    def test_model_name_routed_to_model(self, legacy_resolved_config):
        """model_name should update model config."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            "model_name": "turbo",
        }

        apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="balanced",
        )

        assert config["model"]["model_name"] == "turbo"

    def test_device_routed_to_model(self, legacy_resolved_config):
        """device should update model config."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            "device": "cpu",
        }

        apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="balanced",
        )

        assert config["model"]["device"] == "cpu"


# =============================================================================
# Test: V3 config structure
# =============================================================================

class TestV3ConfigRouting:
    """Test routing for V3 config structure (asr in params)."""

    def test_v3_asr_params_routed(self, v3_resolved_config):
        """ASR params should go to params.asr in V3."""
        config = deepcopy(v3_resolved_config)
        custom_params = {
            "beam_size": 10,
            "temperature": 0.2,
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="balanced",
        )

        assert config["params"]["asr"]["beam_size"] == 10
        assert config["params"]["asr"]["temperature"] == 0.2
        assert unknown == []

    def test_v3_vad_params_routed(self, v3_resolved_config):
        """VAD params should go to params.vad in V3."""
        config = deepcopy(v3_resolved_config)
        custom_params = {
            "threshold": 0.5,
            "speech_pad_ms": 300,
        }

        apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="balanced",
        )

        assert config["params"]["vad"]["threshold"] == 0.5
        assert config["params"]["vad"]["speech_pad_ms"] == 300

    def test_v3_feature_params_discarded(self, v3_resolved_config):
        """Feature params should be discarded in V3 too."""
        config = deepcopy(v3_resolved_config)
        custom_params = {
            "scene_detection_method": "silero",
            "beam_size": 5,
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="balanced",
        )

        assert "scene_detection_method" not in config["params"]["asr"]
        assert config["params"]["asr"]["beam_size"] == 5
        assert unknown == []

    def test_v3_unknown_params_rejected(self, v3_resolved_config):
        """Unknown params should be rejected in V3 too."""
        config = deepcopy(v3_resolved_config)
        custom_params = {
            "completely_made_up_param": "value",
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="balanced",
        )

        assert "completely_made_up_param" in unknown
        assert "completely_made_up_param" not in config["params"]["asr"]


# =============================================================================
# Test: get_valid_provider_params
# =============================================================================

class TestGetValidProviderParams:
    """Test get_valid_provider_params returns correct sets."""

    def test_balanced_pipeline_params(self):
        """balanced pipeline should accept faster-whisper params."""
        valid = get_valid_provider_params("balanced")
        assert "repetition_penalty" in valid
        assert "no_repeat_ngram_size" in valid
        assert "hallucination_silence_threshold" in valid
        # Should NOT include openai-whisper specific
        assert "fp16" not in valid
        assert "verbose" not in valid

    def test_fidelity_pipeline_params(self):
        """fidelity pipeline should accept openai-whisper params."""
        valid = get_valid_provider_params("fidelity")
        assert "fp16" in valid
        assert "verbose" in valid
        assert "hallucination_silence_threshold" in valid
        # Should NOT include faster-whisper specific
        assert "repetition_penalty" not in valid
        assert "no_repeat_ngram_size" not in valid

    def test_fast_pipeline_params(self):
        """fast pipeline should accept stable-ts params."""
        valid = get_valid_provider_params("fast")
        assert "regroup" in valid
        assert "suppress_silence" in valid
        assert "gap_padding" in valid
        # Should NOT include faster-whisper specific
        assert "repetition_penalty" not in valid

    def test_common_params_in_all(self):
        """Common params should be in all pipelines."""
        for pipeline in ["balanced", "fast", "faster", "fidelity"]:
            valid = get_valid_provider_params(pipeline)
            assert "temperature" in valid
            assert "compression_ratio_threshold" in valid
            assert "word_timestamps" in valid


# =============================================================================
# Test: prepare_transformers_params
# =============================================================================

class TestPrepareTransformersParams:
    """Test transformers param preparation including legacy mappings."""

    def test_standard_hf_params_mapping(self):
        """Standard hf_params should be mapped correctly."""
        pass_config = {
            "hf_params": {
                "model_id": "openai/whisper-large-v3",
                "chunk_length_s": 20,
                "batch_size": 8,
            }
        }

        result = prepare_transformers_params(pass_config)

        assert result["hf_model_id"] == "openai/whisper-large-v3"
        assert result["hf_chunk_length"] == 20
        assert result["hf_batch_size"] == 8

    def test_scene_param_mapping(self):
        """scene param should map to hf_scene."""
        pass_config = {
            "hf_params": {
                "scene": "auditok",
            }
        }

        result = prepare_transformers_params(pass_config)

        assert result["hf_scene"] == "auditok"

    def test_legacy_scene_detection_method_mapping(self):
        """Legacy scene_detection_method should map to hf_scene."""
        pass_config = {
            "hf_params": {
                "scene_detection_method": "silero",
            }
        }

        result = prepare_transformers_params(pass_config)

        assert result["hf_scene"] == "silero"

    def test_scene_takes_precedence_over_legacy(self):
        """If both scene and scene_detection_method provided, scene wins."""
        pass_config = {
            "hf_params": {
                "scene": "auditok",
                "scene_detection_method": "silero",  # Should be ignored
            }
        }

        result = prepare_transformers_params(pass_config)

        # scene is in the primary mapping, so it wins
        assert result["hf_scene"] == "auditok"

    def test_prefixed_params_pass_through(self):
        """Params starting with hf_ should pass through."""
        pass_config = {
            "hf_params": {
                "hf_custom_param": "custom_value",
            }
        }

        result = prepare_transformers_params(pass_config)

        assert result["hf_custom_param"] == "custom_value"


# =============================================================================
# Test: Comprehensive integration test
# =============================================================================

class TestComprehensiveRouting:
    """Test routing with mixed param types."""

    def test_full_custom_params_mix(self, legacy_resolved_config):
        """Test with a realistic mix of all param types."""
        config = deepcopy(legacy_resolved_config)
        custom_params = {
            # Model params
            "model_name": "large-v3",
            "device": "cuda:1",
            # Decoder params
            "beam_size": 10,
            "task": "transcribe",
            "language": "ja",
            # Provider params (common)
            "temperature": [0.0, 0.2, 0.4],
            "no_speech_threshold": 0.3,
            "word_timestamps": False,
            # Provider params (faster-whisper specific for balanced)
            "hallucination_silence_threshold": 3.0,
            # VAD params
            "threshold": 0.25,
            "min_speech_duration_ms": 200,
            # Feature params (should be discarded)
            "scene_detection_method": "auditok",
            # Unknown params (should be rejected)
            "fake_param": "should_be_rejected",
        }

        unknown = apply_custom_params(
            resolved_config=config,
            custom_params=custom_params,
            pass_number=1,
            pipeline_name="balanced",
        )

        # Model params
        assert config["model"]["model_name"] == "large-v3"
        assert config["model"]["device"] == "cuda:1"

        # Decoder params
        assert config["params"]["decoder"]["beam_size"] == 10
        assert config["params"]["decoder"]["task"] == "transcribe"
        assert config["params"]["decoder"]["language"] == "ja"

        # Provider params
        assert config["params"]["provider"]["temperature"] == [0.0, 0.2, 0.4]
        assert config["params"]["provider"]["no_speech_threshold"] == 0.3
        assert config["params"]["provider"]["word_timestamps"] is False
        assert config["params"]["provider"]["hallucination_silence_threshold"] == 3.0

        # VAD params
        assert config["params"]["vad"]["threshold"] == 0.25
        assert config["params"]["vad"]["min_speech_duration_ms"] == 200

        # Feature params NOT in provider
        assert "scene_detection_method" not in config["params"]["provider"]

        # Unknown params tracked
        assert "fake_param" in unknown
        assert "fake_param" not in config["params"]["provider"]


# =============================================================================
# Test: Constants completeness
# =============================================================================

class TestConstantsCompleteness:
    """Verify that param category constants are complete and consistent."""

    def test_decoder_params_non_empty(self):
        """DECODER_PARAMS should contain expected params."""
        assert "task" in DECODER_PARAMS
        assert "language" in DECODER_PARAMS
        assert "beam_size" in DECODER_PARAMS

    def test_vad_params_non_empty(self):
        """VAD_PARAMS should contain expected params."""
        assert "threshold" in VAD_PARAMS
        assert "min_speech_duration_ms" in VAD_PARAMS
        assert "speech_pad_ms" in VAD_PARAMS

    def test_feature_params_contains_scene_detection(self):
        """FEATURE_PARAMS should contain scene_detection_method."""
        assert "scene_detection_method" in FEATURE_PARAMS

    def test_model_params_complete(self):
        """MODEL_PARAMS should contain model_name and device."""
        assert "model_name" in MODEL_PARAMS
        assert "device" in MODEL_PARAMS

    def test_pipeline_backends_complete(self):
        """All expected pipelines should be in PIPELINE_BACKENDS."""
        assert "balanced" in PIPELINE_BACKENDS
        assert "fast" in PIPELINE_BACKENDS
        assert "faster" in PIPELINE_BACKENDS
        assert "fidelity" in PIPELINE_BACKENDS
