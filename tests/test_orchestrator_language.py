"""Regression tests for EnsembleOrchestrator language handling."""

from pathlib import Path

from whisperjav.ensemble.orchestrator import EnsembleOrchestrator


def _build_orchestrator(tmp_path):
    output = tmp_path / "out"
    temp = tmp_path / "tmp"
    return EnsembleOrchestrator(
        output_dir=str(output),
        temp_dir=str(temp),
        keep_temp_files=True,
        subs_language="native",
    )


def test_process_batch_handles_none_overrides(tmp_path, monkeypatch):
    orchestrator = _build_orchestrator(tmp_path)

    media_info = {
        "basename": "sample",
        "path": str(tmp_path / "sample.wav"),
    }

    pass_result = {
        "status": "completed",
        "srt_path": str(tmp_path / "sample_pass1.srt"),
        "subtitles": 1,
        "processing_time": 1.0,
    }

    def fake_run_pass(*args, **kwargs):
        return {"sample": pass_result}

    monkeypatch.setattr(orchestrator, "_run_pass_in_subprocess", fake_run_pass)

    def fake_merge(*args, **kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")
        return {"merged_count": 1}

    monkeypatch.setattr(orchestrator.merge_engine, "merge", fake_merge)

    results = orchestrator.process_batch(
        media_files=[media_info],
        pass1_config={"pipeline": "balanced", "overrides": None},
        pass2_config={"pipeline": "balanced", "overrides": None},
        merge_strategy="pass1_primary",
    )

    assert results[0]["summary"]["final_output"].endswith("sample.ja.whisperjav.srt")