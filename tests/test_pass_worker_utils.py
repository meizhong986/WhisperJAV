"""Unit tests for ensemble pass worker helpers."""

from copy import deepcopy

from whisperjav.ensemble.pass_worker import (
    DEFAULT_HF_PARAMS,
    apply_custom_params,
    prepare_transformers_params,
)


def test_prepare_transformers_params_applies_overrides_and_mapping():
    pass_config = {
        "hf_params": {
            "chunk_length_s": 30,
            "hf_stride": 2,
        },
        "overrides": {
            "language": "en",
        },
    }

    params = prepare_transformers_params(pass_config)

    assert params["hf_chunk_length"] == 30
    assert params["hf_stride"] == 2
    assert params["hf_language"] == "en"

    params["hf_model_id"] = "modified"
    assert DEFAULT_HF_PARAMS["hf_model_id"] != "modified"


def test_apply_custom_params_reports_unknown_for_legacy_config():
    resolved_config = {
        "model": {"model_name": "large-v2", "device": "cuda"},
        "params": {
            "decoder": {"temperature": 0.0},
            "provider": {},
            "vad": {},
        },
    }

    unknown = apply_custom_params(
        resolved_config,
        {"unknown_setting": 42, "temperature": 1.0},
        pass_number=1,
    )

    assert "unknown_setting" in unknown
    assert resolved_config["params"]["decoder"]["temperature"] == 1.0
    assert resolved_config["params"]["provider"]["unknown_setting"] == 42


def test_apply_custom_params_v3_keeps_unknown_list_empty():
    resolved_config = {
        "model": {"model_name": "large-v2", "device": "cuda"},
        "params": {"asr": {}},
    }

    unknown = apply_custom_params(
        deepcopy(resolved_config),
        {"custom_alpha": 0.7},
        pass_number=2,
    )

    assert unknown == []