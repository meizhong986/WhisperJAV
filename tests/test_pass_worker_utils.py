"""Unit tests for ensemble pass worker helpers."""

from copy import deepcopy
from pathlib import Path

from whisperjav.ensemble.pass_worker import (
    DEFAULT_HF_PARAMS,
    WorkerPayload,
    apply_custom_params,
    prepare_transformers_params,
    run_pass_worker,
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
    """Unknown params should be rejected and NOT added to provider."""
    resolved_config = {
        "model": {"model_name": "large-v2", "device": "cuda"},
        "params": {
            "decoder": {"temperature": 0.0},
            "provider": {},
            "vad": {},
        },
    }

    unknown = apply_custom_params(
        resolved_config=resolved_config,
        custom_params={"unknown_setting": 42, "temperature": 1.0},
        pass_number=1,
        pipeline_name="balanced",
    )

    # Unknown params should be tracked but NOT added to provider
    assert "unknown_setting" in unknown
    assert "unknown_setting" not in resolved_config["params"]["provider"]
    # temperature is a valid common provider param, should be applied
    assert resolved_config["params"]["provider"]["temperature"] == 1.0


def test_apply_custom_params_v3_rejects_unknown_params():
    """In V3 config, unknown params should be rejected (not added to asr)."""
    resolved_config = {
        "model": {"model_name": "large-v2", "device": "cuda"},
        "params": {"asr": {}},
    }

    unknown = apply_custom_params(
        resolved_config=deepcopy(resolved_config),
        custom_params={"custom_alpha": 0.7},
        pass_number=2,
        pipeline_name="balanced",
    )

    # Unknown params should be tracked and rejected
    assert "custom_alpha" in unknown


def _install_dummy_pipeline(monkeypatch, legacy_file):
    class DummyPipeline:
        def process(self, media_info):
            legacy_file.write_text("legacy", encoding="utf-8")
            return {
                "output_files": {"final_srt": str(legacy_file)},
                "summary": {
                    "final_subtitles_refined": 1,
                    "total_processing_time_seconds": 0.5,
                },
            }

        def cleanup(self):
            pass

    monkeypatch.setattr(
        "whisperjav.ensemble.pass_worker._build_pipeline",
        lambda *args, **kwargs: DummyPipeline(),
    )


def test_run_pass_worker_removes_legacy_file_when_not_keeping_temp(tmp_path, monkeypatch):
    media_path = tmp_path / "sample.wav"
    media_path.write_text("audio", encoding="utf-8")
    output_dir = tmp_path / "out"
    temp_dir = tmp_path / "tmp"
    output_dir.mkdir()
    temp_dir.mkdir()

    legacy_file = output_dir / "sample.ja.whisperjav.srt"
    _install_dummy_pipeline(monkeypatch, legacy_file)

    payload = WorkerPayload(
        pass_number=1,
        media_files=[{"basename": "sample", "path": str(media_path)}],
        pass_config={"pipeline": "balanced"},
        output_dir=str(output_dir),
        temp_dir=str(temp_dir),
        keep_temp_files=False,
        subs_language="native",
        extra_kwargs={},
        language_code="ja",
    )

    result = run_pass_worker(payload)

    pass_file = output_dir / "sample.ja.pass1.srt"
    assert pass_file.exists(), "Pass-specific copy should exist"
    assert not legacy_file.exists(), "Legacy pipeline output should be removed"
    assert result["results"][0]["srt_path"] == str(pass_file)


def test_run_pass_worker_retains_legacy_file_when_keeping_temp(tmp_path, monkeypatch):
    media_path = tmp_path / "sample.wav"
    media_path.write_text("audio", encoding="utf-8")
    output_dir = tmp_path / "out"
    temp_dir = tmp_path / "tmp"
    output_dir.mkdir()
    temp_dir.mkdir()

    legacy_file = output_dir / "sample.ja.whisperjav.srt"
    _install_dummy_pipeline(monkeypatch, legacy_file)

    payload = WorkerPayload(
        pass_number=1,
        media_files=[{"basename": "sample", "path": str(media_path)}],
        pass_config={"pipeline": "balanced"},
        output_dir=str(output_dir),
        temp_dir=str(temp_dir),
        keep_temp_files=True,
        subs_language="native",
        extra_kwargs={},
        language_code="ja",
    )

    run_pass_worker(payload)

    pass_file = output_dir / "sample.ja.pass1.srt"
    assert pass_file.exists()
    assert legacy_file.exists(), "Legacy file should remain when keep_temp_files is True"