"""The Ensemble stop (owner W501): once pass 1 stops because its ASR model cannot be used (R-QFT, #494), pass 2 is
skipped for that file and no further file starts -- in the normal mode and with "Finish each file"
(--ensemble-serial). The files not started show as "skipped" with the same reason as a single-pass run
(``qwen_errors.NOT_STARTED``; decision 2, W501). Once pass 2 stops for that reason, every file keeps its pass-1
subtitles in both modes; with "Finish each file" the later files get pass 1 alone (owner W505 1).

No worker process starts: ``_run_pass_in_subprocess`` is replaced by a stand-in that answers as the pass worker does
(the stopping file and every later file of its list fail with the same message, pass_worker.py). Run alone:
python -m pytest tests/test_ensemble_stop_unusable_model.py
"""

import pytest

from whisperjav.modules.qwen_errors import NOT_STARTED, PREFIX

MESSAGE = PREFIX + "x/Qwen3-ASR-y cannot be used. Its model file has no output layer ..."
SRT = "1\n00:00:00,000 --> 00:00:01,000\nline\n"


def _orchestrator(tmp_path, serial, unusable_pass, calls):
    from whisperjav.ensemble.orchestrator import EnsembleOrchestrator

    orch = EnsembleOrchestrator(output_dir=str(tmp_path / "out"), temp_dir=str(tmp_path / "temp"),
                                serial_file_processing=serial)
    orch.temp_dir.mkdir(parents=True, exist_ok=True)

    def run_pass(pass_number, media_files, pass_config, language_code):
        calls.append((pass_number, [m["basename"] for m in media_files]))
        if pass_number == unusable_pass:
            return {m["basename"]: {"status": "failed", "error": MESSAGE} for m in media_files}
        out = {}
        for m in media_files:
            srt = tmp_path / "out" / f"{m['basename']}.ja.pass{pass_number}.srt"
            srt.parent.mkdir(parents=True, exist_ok=True)
            srt.write_text(SRT, encoding="utf-8")
            out[m["basename"]] = {"status": "completed", "srt_path": str(srt), "subtitles": 1,
                                  "processing_time": 1.0}
        return out

    orch._run_pass_in_subprocess = run_pass
    return orch


def _media(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir(exist_ok=True)
    return [{"path": str(folder / f"{n}.mp4"), "basename": n} for n in ("a", "b", "c")]


P1 = {"pipeline": "qwen", "sensitivity": "aggressive"}
P2 = {"pipeline": "qwen", "sensitivity": "aggressive"}


@pytest.mark.parametrize("serial", [False, True])
def test_pass_1_stops_pass_2_is_skipped_and_no_further_file_starts(tmp_path, serial):
    calls = []
    orch = _orchestrator(tmp_path, serial, 1, calls)
    results = orch.process_batch(_media(tmp_path), P1, P2, "pass1_primary")
    # no pass 2 at all; with "Finish each file" no later file's pass 1 either
    assert all(n == 1 for n, _ in calls)
    assert calls == ([(1, ["a"])] if serial else [(1, ["a", "b", "c"])])
    states = [(r["input"]["basename"], r["status"]) for r in results]
    assert states == [("a", "failed"), ("b", "skipped"), ("c", "skipped")]
    assert results[0]["pass1"]["error"] == MESSAGE
    assert results[0]["pass2"]["status"] == "skipped"
    for r in results[1:]:
        assert r["error"] == NOT_STARTED and r["pass1"]["status"] == "skipped"
        assert not r.get("kept_work")


def test_finish_each_file_pass_2_stops_the_later_files_get_pass_1_alone(tmp_path):
    """Owner W505 1: as with the box unticked -- every file keeps its pass-1 subtitles, marked with pass 2's reason;
    pass 2 is not tried again on the later files."""
    calls = []
    orch = _orchestrator(tmp_path, True, 2, calls)
    results = orch.process_batch(_media(tmp_path), P1, P2, "pass1_primary")
    assert calls == [(1, ["a"]), (2, ["a"]), (1, ["b"]), (1, ["c"])]
    assert [(r["input"]["basename"], r["status"]) for r in results] == [
        ("a", "degraded"), ("b", "degraded"), ("c", "degraded")]
    for r in results:
        assert r["summary"]["final_output"].endswith(f"{r['input']['basename']}.ja.pass1.srt")
        assert r["pass2"]["error"] == MESSAGE
        assert r["merge"]["status"] == "skipped"


def test_both_modes_end_alike_when_pass_2_stops(tmp_path):
    """The same states, reasons and outputs with and without "Finish each file" (owner W505 1)."""
    seen = {}
    for serial in (False, True):
        results = _orchestrator(tmp_path / str(serial), serial, 2, []).process_batch(
            _media(tmp_path / str(serial)), P1, P2, "pass1_primary")
        seen[serial] = [(r["input"]["basename"], r["status"], r["pass2"].get("error"),
                         r["summary"]["final_output"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]) for r in results]
    assert seen[True] == seen[False]


def test_the_run_summary_marks_a_pass_2_stop_suspect_and_exits_0(tmp_path):
    """main.py's line for a degraded Ensemble file names pass 2's reason; the file is "suspect", which fails a run
    only with --fail-on suspect (utils/run_outcome.py exit_status)."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "whisperjav" / "main.py").read_text(encoding="utf-8")
    assert 'f"pass 2 failed ({_p2_error}); output is pass 1 alone"' in src
    assert "_p2_error.startswith(QWEN_UNUSABLE_PREFIX)" in src
    from whisperjav.utils.run_outcome import FileOutcome, exit_status

    suspect = [FileOutcome(path=str(tmp_path / f"{n}.mp4"), state="suspect") for n in "abc"]
    assert exit_status(suspect) == 0
    assert exit_status(suspect, fail_on=("suspect",)) == 1


def test_normal_mode_pass_2_stops_every_file_keeps_its_pass_1(tmp_path):
    """In the normal mode pass 1 has finished every file before pass 2 starts: that work is kept (W462); pass 2's
    worker gives every file the same message without starting it."""
    calls = []
    orch = _orchestrator(tmp_path, False, 2, calls)
    results = orch.process_batch(_media(tmp_path), P1, P2, "pass1_primary")
    assert calls == [(1, ["a", "b", "c"]), (2, ["a", "b", "c"])]
    assert [r["status"] for r in results] == ["degraded"] * 3
    assert all(r["pass2"]["error"] == MESSAGE for r in results)


def test_one_file_alone(tmp_path):
    calls = []
    orch = _orchestrator(tmp_path, True, 1, calls)
    results = orch.process_batch(_media(tmp_path)[:1], P1, P2, "pass1_primary")
    assert calls == [(1, ["a"])]
    assert [r["status"] for r in results] == ["failed"]


def test_an_ordinary_failure_does_not_stop_the_run(tmp_path):
    calls = []
    orch = _orchestrator(tmp_path, True, None, calls)

    def run_pass(pass_number, media_files, pass_config, language_code, _orig=orch._run_pass_in_subprocess):
        if pass_number == 1 and media_files[0]["basename"] == "a":
            calls.append((1, ["a"]))
            return {"a": {"status": "failed", "error": "Traceback ... RuntimeError: boom"}}
        return _orig(pass_number, media_files, pass_config, language_code)

    orch._run_pass_in_subprocess = run_pass
    results = orch.process_batch(_media(tmp_path), P1, P2, "pass1_primary")
    assert [r["status"] for r in results] == ["failed", "completed", "completed"]
    assert (1, ["b"]) in calls and (1, ["c"]) in calls


def test_the_worker_never_started_files_keep_no_work(tmp_path):
    """_keep_failed_pass_work: only the stopping file is a file the pass did not finish; the later ones were not
    started by the worker and get no kept-work record."""
    from whisperjav.ensemble.orchestrator import EnsembleOrchestrator

    orch = EnsembleOrchestrator(output_dir=str(tmp_path / "out"), temp_dir=str(tmp_path / "temp"))
    orch.temp_dir.mkdir(parents=True, exist_ok=True)
    media = _media(tmp_path)
    results = {m["basename"]: {"status": "failed", "error": MESSAGE} for m in media}
    orch._keep_failed_pass_work(1, media, results, "ja", "qwen")
    assert "save_points" in results["a"]
    assert "save_points" not in results["b"] and "save_points" not in results["c"]


def test_the_run_summary_shows_skipped(tmp_path, monkeypatch):
    """main.py turns an Ensemble result with status "skipped" into the run summary's "skipped" with its reason."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "whisperjav" / "main.py").read_text(encoding="utf-8")
    body = src[src.index("results = orchestrator.process_batch("):]
    branch = body[body.index("if status == 'skipped':"):body.index("if result.get('error') or status == 'failed':")]
    assert "skipped_outcome(in_path, str(result.get('error') or QWEN_NOT_STARTED))" in branch
    assert "continue" in branch
    from whisperjav.utils.run_outcome import skipped_outcome

    outcome = skipped_outcome(str(tmp_path / "b.mp4"), NOT_STARTED)
    assert outcome.state == "skipped" and outcome.detail == NOT_STARTED
