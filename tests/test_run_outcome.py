"""Tests for the per-file outcome vocabulary and the exit-status contract.

Two layers:

1. The pure functions in ``whisperjav.utils.run_outcome`` -- the verdict, the
   ``--fail-on`` parser, the exit-status mapping, the summary and the manifest.
2. ``whisperjav.main.main()`` driven end to end through each execution path
   (sync, async, ensemble) with the pipeline layer stubbed out, asserting the
   real process exit status. These exist because the v1.9.2 pre-release
   regression -- every successful ensemble run exiting 1 -- was invisible to
   tests that only inspected source text.
"""

import datetime
import json
import os
import sys
from pathlib import Path

import pytest

from whisperjav.utils.run_outcome import (
    FAIL_ON_CHOICES,
    MANIFEST_NAME,
    STATES,
    FileOutcome,
    classify_output,
    count_states,
    default_manifest_path,
    exit_status,
    failed_outcome,
    format_summary,
    mark_translation,
    parse_fail_on,
    skipped_outcome,
    write_manifest,
)

PY = sys.executable


def _write_srt(path, cues):
    import srt as srt_lib

    subs = [
        srt_lib.Subtitle(index=i + 1, start=datetime.timedelta(seconds=s),
                         end=datetime.timedelta(seconds=e), content=t)
        for i, (s, e, t) in enumerate(cues)
    ]
    path.write_text(srt_lib.compose(subs), encoding="utf-8")
    return path


# ===========================================================================
# Layer 1: the vocabulary and the contract
# ===========================================================================

class TestVocabulary:
    def test_the_five_states_are_fixed(self):
        assert tuple(STATES) == ("done", "empty", "suspect", "failed", "skipped")

    def test_unknown_state_is_rejected(self):
        with pytest.raises(ValueError):
            FileOutcome(path="x", state="partial")

    def test_done(self, tmp_path):
        srt = _write_srt(tmp_path / "a.srt", [(1, 2, "a"), (3500, 3590, "b")])
        o = classify_output("a.mp4", srt, 3600.0)
        assert o.state == "done"
        assert o.coverage == "ok"
        assert o.subtitle_count == 2

    def test_empty_is_an_observation_not_a_failure(self, tmp_path):
        """f1: zero subtitles does not mean the process failed."""
        srt = tmp_path / "a.srt"
        srt.write_text("", encoding="utf-8")
        o = classify_output("a.mp4", srt, 3600.0)
        assert o.state == "empty"
        assert exit_status([o]) == 0

    def test_missing_output_is_empty(self):
        o = classify_output("a.mp4", None, 3600.0)
        assert o.state == "empty"

    def test_low_span_is_suspect(self, tmp_path):
        """@daoran9's 4.3% case."""
        srt = _write_srt(tmp_path / "gqn.srt", [(10, 12, "a"), (370, 376.9, "b")])
        o = classify_output("gqn.mp4", srt, 8766.0)
        assert o.state == "suspect"
        assert o.coverage == "low"
        assert o.coverage_ratio == pytest.approx(376.9 / 8766.0, rel=1e-3)

    def test_corroborated_full_length_output_is_suspect(self, tmp_path):
        """@13e5t's case: 60% span, which no threshold catches, but a probe failed."""
        srt = _write_srt(tmp_path / "a.srt", [(1, 2, "a"), (3500, 3590, "b")])
        o = classify_output("a.mp4", srt, 3600.0, speech_positive_empty_streak=99)
        assert o.state == "suspect"
        assert o.coverage == "ok"
        assert "consecutive empty results" in o.detail

    def test_degraded_ensemble_output_is_suspect(self, tmp_path):
        srt = _write_srt(tmp_path / "a.srt", [(1, 2, "a"), (3500, 3590, "b")])
        o = classify_output("a.mp4", srt, 3600.0, degraded=True)
        assert o.state == "suspect"
        assert "pass 1 alone" in o.detail

    def test_short_media_is_done_but_not_assessed(self, tmp_path):
        srt = _write_srt(tmp_path / "clip.srt", [(1, 2, "a")])
        o = classify_output("clip.mp4", srt, 30.0)
        assert o.state == "done"
        assert o.coverage == "not assessed"

    def test_unknown_duration_is_done_but_not_assessed(self, tmp_path):
        srt = _write_srt(tmp_path / "clip.srt", [(1, 2, "a")])
        o = classify_output("clip.mp4", srt, None)
        assert o.state == "done"
        assert o.coverage == "not assessed"

    def test_min_coverage_zero_disables_suspect_by_span(self, tmp_path):
        srt = _write_srt(tmp_path / "gqn.srt", [(10, 12, "a"), (370, 376.9, "b")])
        o = classify_output("gqn.mp4", srt, 8766.0, min_coverage=0)
        assert o.state == "done"

    def test_the_324_native_vad_run_is_suspect_not_failed(self, tmp_path):
        """#324: 7 cues ending at 30 s of a 1061 s music show under native VAD.
        No external detector, so no corroboration; low span only. Reported,
        never fatal by default."""
        srt = _write_srt(tmp_path / "kbs.srt", [(0.1, 5.2, "a"), (25.6, 30.5, "b")])
        o = classify_output("kbs.mp4", srt, 1061.4, speech_positive_empty_streak=0)
        assert o.state == "suspect"
        assert exit_status([o]) == 0
        assert exit_status([o], fail_on={"suspect"}) == 1

    def test_translation_failure_makes_the_file_failed(self, tmp_path):
        srt = _write_srt(tmp_path / "a.srt", [(1, 2, "a"), (3500, 3590, "b")])
        o = classify_output("a.mp4", srt, 3600.0)
        mark_translation(o, "failed", error="boom")
        assert o.state == "failed"
        assert o.translation == "failed"
        assert "boom" in o.detail

    def test_translation_skipped_on_purpose_is_not_a_failure(self, tmp_path):
        srt = _write_srt(tmp_path / "a.srt", [(1, 2, "a"), (3500, 3590, "b")])
        o = classify_output("a.mp4", srt, 3600.0)
        mark_translation(o, "skipped", error="too few subtitles (1)")
        assert o.state == "done"
        assert o.translation == "skipped"

    def test_translation_done_records_the_path(self, tmp_path):
        srt = _write_srt(tmp_path / "a.srt", [(1, 2, "a"), (3500, 3590, "b")])
        o = classify_output("a.mp4", srt, 3600.0)
        mark_translation(o, "done", translated_output=str(tmp_path / "a.en.srt"))
        assert o.state == "done"
        assert o.translated_output.endswith("a.en.srt")

    def test_reported_but_missing_file_is_failed(self, tmp_path):
        """A pipeline that says it wrote an SRT which is not there is an error,
        not a silent result (#263's shape)."""
        o = classify_output("a.mp4", tmp_path / "promised.srt", 3600.0)
        assert o.state == "failed"
        assert "was not created" in o.detail
        assert exit_status([o]) == 1

    def test_zero_cues_with_corroboration_is_suspect(self, tmp_path):
        """The #394 cascade must not print the same word as a silent clip."""
        srt = tmp_path / "a.srt"
        srt.write_text("", encoding="utf-8")
        o = classify_output("a.mp4", srt, 3600.0, speech_positive_empty_streak=99)
        assert o.state == "suspect"
        assert "consecutive empty results" in o.detail
        o2 = classify_output("a.mp4", None, 3600.0, speech_positive_empty_streak=99)
        assert o2.state == "suspect"

    def test_zero_cues_after_pass2_failure_is_suspect(self):
        o = classify_output("a.mp4", None, 3600.0, degraded=True)
        assert o.state == "suspect"
        assert "pass 1 alone" in o.detail


class TestExitStatusContract:
    def _one_of_each(self):
        return [
            FileOutcome(path="d", state="done"),
            FileOutcome(path="e", state="empty"),
            FileOutcome(path="s", state="suspect"),
            FileOutcome(path="k", state="skipped"),
        ]

    def test_only_failed_fails_by_default(self):
        assert exit_status(self._one_of_each()) == 0
        assert exit_status(self._one_of_each() + [failed_outcome("f", "x")]) == 1

    def test_empty_batch_is_success(self):
        assert exit_status([]) == 0

    def test_all_skipped_is_success(self):
        assert exit_status([skipped_outcome("a"), skipped_outcome("b")]) == 0

    def test_fail_on_empty(self):
        assert exit_status(self._one_of_each(), fail_on={"empty"}) == 1
        assert exit_status([FileOutcome(path="d", state="done")], fail_on={"empty"}) == 0

    def test_fail_on_suspect(self):
        assert exit_status(self._one_of_each(), fail_on={"suspect"}) == 1

    def test_fail_on_never_downgrades_failed(self):
        assert exit_status([failed_outcome("f", "x")], fail_on=set()) == 1

    def test_parse_fail_on_forms(self):
        assert parse_fail_on(None) == frozenset()
        assert parse_fail_on([]) == frozenset()
        assert parse_fail_on(["empty"]) == {"empty"}
        assert parse_fail_on(["empty,suspect"]) == {"empty", "suspect"}
        assert parse_fail_on(["empty", "suspect"]) == {"empty", "suspect"}
        assert parse_fail_on([" Empty , SUSPECT "]) == {"empty", "suspect"}

    def test_parse_fail_on_rejects_unknown(self):
        with pytest.raises(ValueError):
            parse_fail_on(["failed"])  # always fails; not a choice
        with pytest.raises(ValueError):
            parse_fail_on(["done"])
        with pytest.raises(ValueError):
            parse_fail_on(["partial"])

    def test_fail_on_choices_exclude_the_fixed_states(self):
        assert set(FAIL_ON_CHOICES) == {"empty", "suspect"}


class TestReporting:
    def test_summary_uses_the_vocabulary_and_states_the_rule(self):
        outs = [
            FileOutcome(path="/x/a.mp4", state="done", output="/x/a.ja.srt",
                        subtitle_count=12, coverage="ok", coverage_ratio=0.97),
            FileOutcome(path="/x/b.mp4", state="empty", detail="no subtitles"),
            FileOutcome(path="/x/c.mp4", state="suspect", output="/x/c.ja.srt",
                        subtitle_count=3, coverage="low", coverage_ratio=0.04,
                        detail="stops at 300s of 8766s"),
            failed_outcome("/x/d.mp4", "boom"),
            skipped_outcome("/x/e.mp4"),
        ]
        text = format_summary(outs, fail_on={"empty"}, status=1)
        assert "a run fails on: empty, failed" in text
        for word in STATES:
            assert word in text
        assert "done 1  empty 1  suspect 1  failed 1  skipped 1  total 5" in text
        assert "ok 97%" in text and "low 4%" in text and "not assessed" in text

    def test_count_states(self):
        c = count_states([skipped_outcome("a"), failed_outcome("b", "x"), skipped_outcome("c")])
        assert c == {"done": 0, "empty": 0, "suspect": 0, "failed": 1, "skipped": 2}

    def test_manifest_round_trips(self, tmp_path):
        outs = [FileOutcome(path="/x/a.mp4", state="done"), failed_outcome("/x/b.mp4", "boom")]
        p = write_manifest(outs, tmp_path / MANIFEST_NAME, mode="balanced",
                           fail_on={"empty"}, status=1, version="test")
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["exit_status"] == 1
        assert data["fails_on"] == ["empty", "failed"]
        assert data["counts"]["failed"] == 1
        assert [f["state"] for f in data["files"]] == ["done", "failed"]
        assert data["mode"] == "balanced" and data["whisperjav_version"] == "test"

    def test_manifest_failure_never_raises(self, tmp_path):
        bad = tmp_path / "file-not-dir"
        bad.write_text("x", encoding="utf-8")
        assert write_manifest([], bad / "sub" / MANIFEST_NAME, mode="m", fail_on=(), status=0) is None

    def test_default_manifest_path(self, tmp_path):
        movies = tmp_path / "movies"
        movies.mkdir()
        a = movies / "a.mp4"
        a.write_bytes(b"x")
        assert default_manifest_path(str(tmp_path / "out"), [str(a)]) == tmp_path / "out" / MANIFEST_NAME
        # source mode: beside a file input, inside a folder input
        assert default_manifest_path("source", [str(a)]) == movies / MANIFEST_NAME
        assert default_manifest_path("source", [str(movies)]) == movies / MANIFEST_NAME
        assert default_manifest_path("source", []) is None


# ===========================================================================
# Layer 2: main() executed through each path, real exit status
# ===========================================================================

def _fake_srt(dir_, name, cues):
    return _write_srt(Path(dir_) / name, cues)


@pytest.fixture
def media_dir(tmp_path):
    d = tmp_path / "media"
    d.mkdir()
    return d


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    return d


class _StubDiscovery:
    """Replaces MediaDiscovery: returns the files listed in _FILES."""
    files = []

    def discover(self, inputs):
        return list(self.files)


def _run_main(monkeypatch, argv, discovery_files, patches):
    """Drive whisperjav.main.main() with stubs and return the exit status.

    os._exit is redirected to SystemExit so the ctranslate2 nuclear exit can be
    observed instead of killing pytest.
    """
    import whisperjav.main as M

    _StubDiscovery.files = discovery_files
    monkeypatch.setattr(M, "MediaDiscovery", _StubDiscovery)
    for target, name, value in patches:
        monkeypatch.setattr(target, name, value)

    def _fake_os_exit(code=0):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", _fake_os_exit)
    monkeypatch.setattr(sys, "argv", ["whisperjav"] + argv)
    try:
        M.main()
    except SystemExit as e:
        return e.code or 0
    return 0


def _media(path, duration):
    return {"path": str(path), "basename": Path(path).stem, "type": "video", "duration": duration}


def _base_args(out_dir, tmp_path):
    return ["--output-dir", str(out_dir), "--temp-dir", str(tmp_path / "tmp"),
            "--accept-cpu-mode", "--no-signature", "--no-progress"]


# ----- ensemble ---------------------------------------------------------------

class _StubOrchestrator:
    """Replaces EnsembleOrchestrator. `plan` maps basename -> (status, srt or None)."""
    plan = {}
    last_pass1_config = None

    def __init__(self, **kw):
        pass

    def process_batch(self, media_files, pass1_config, pass2_config, merge_strategy):
        _StubOrchestrator.last_pass1_config = dict(pass1_config)
        results = []
        for m in media_files:
            status, srt, err = self.plan[m["basename"]]
            r = {"input": {"file": m["path"], "basename": m["basename"]},
                 "status": status,
                 "summary": {"final_output": str(srt) if srt else None,
                             "total_processing_time_seconds": 1.0}}
            if err:
                r["error"] = err
            results.append(r)
        return results


def _ensemble_argv(files, out_dir, tmp_path, extra=()):
    return [str(f) for f in files] + ["--ensemble", "--pass1-pipeline", "balanced"] + \
        _base_args(out_dir, tmp_path) + list(extra)


class TestEnsemblePath:
    """The path on which the v1.9.2 pre-release exited 1 for every success."""

    def test_successful_ensemble_run_exits_zero(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.ensemble.orchestrator as O

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        srt = _fake_srt(out_dir, "a.ja.merged.whisperjav.srt", [(1, 2, "x"), (3500, 3590, "y")])
        _StubOrchestrator.plan = {"a": ("completed", srt, None)}
        code = _run_main(monkeypatch, _ensemble_argv([a], out_dir, tmp_path),
                         [_media(a, 3600.0)], [(O, "EnsembleOrchestrator", _StubOrchestrator)])
        assert code == 0
        data = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert data["mode"] == "ensemble"
        assert [f["state"] for f in data["files"]] == ["done"]

    def test_failed_pass_exits_one(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.ensemble.orchestrator as O

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        _StubOrchestrator.plan = {"a": ("failed", None, "pass 1 crashed")}
        code = _run_main(monkeypatch, _ensemble_argv([a], out_dir, tmp_path),
                         [_media(a, 3600.0)], [(O, "EnsembleOrchestrator", _StubOrchestrator)])
        assert code == 1
        data = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert data["files"][0]["state"] == "failed"
        assert "pass 1 crashed" in data["files"][0]["error"]

    def test_degraded_is_suspect_and_exits_zero_unless_asked(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.ensemble.orchestrator as O

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        srt = _fake_srt(out_dir, "a.ja.pass1.srt", [(1, 2, "x"), (3500, 3590, "y")])
        _StubOrchestrator.plan = {"a": ("degraded", srt, None)}
        argv = _ensemble_argv([a], out_dir, tmp_path)
        code = _run_main(monkeypatch, argv, [_media(a, 3600.0)],
                         [(O, "EnsembleOrchestrator", _StubOrchestrator)])
        assert code == 0
        assert json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))["files"][0]["state"] == "suspect"

        code = _run_main(monkeypatch, argv + ["--fail-on", "suspect"], [_media(a, 3600.0)],
                         [(O, "EnsembleOrchestrator", _StubOrchestrator)])
        assert code == 1

    def test_empty_output_exits_zero_by_default_and_one_with_fail_on(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.ensemble.orchestrator as O

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        srt = out_dir / "a.ja.merged.whisperjav.srt"; srt.write_text("", encoding="utf-8")
        _StubOrchestrator.plan = {"a": ("completed", srt, None)}
        argv = _ensemble_argv([a], out_dir, tmp_path)
        assert _run_main(monkeypatch, argv, [_media(a, 3600.0)],
                         [(O, "EnsembleOrchestrator", _StubOrchestrator)]) == 0
        assert _run_main(monkeypatch, argv + ["--fail-on", "empty"], [_media(a, 3600.0)],
                         [(O, "EnsembleOrchestrator", _StubOrchestrator)]) == 1

    def test_batch_continues_and_one_failure_fails_the_run(self, monkeypatch, media_dir, out_dir, tmp_path):
        """f2: the rest of the batch is processed; the exit code reflects the worst file."""
        import whisperjav.ensemble.orchestrator as O

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        b = media_dir / "b.mp4"; b.write_bytes(b"x")
        srt = _fake_srt(out_dir, "b.ja.merged.whisperjav.srt", [(1, 2, "x"), (3500, 3590, "y")])
        _StubOrchestrator.plan = {"a": ("failed", None, "boom"), "b": ("completed", srt, None)}
        code = _run_main(monkeypatch, _ensemble_argv([a, b], out_dir, tmp_path),
                         [_media(a, 3600.0), _media(b, 3600.0)],
                         [(O, "EnsembleOrchestrator", _StubOrchestrator)])
        assert code == 1
        states = [f["state"] for f in json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))["files"]]
        assert states == ["failed", "done"]

    def test_pass_config_carries_the_telemetry_settings(self, monkeypatch, media_dir, out_dir, tmp_path):
        """main() hands --asr-telemetry / --no-asr-telemetry to the pass worker."""
        import whisperjav.ensemble.orchestrator as O

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        srt = _fake_srt(out_dir, "a.ja.merged.whisperjav.srt", [(1, 2, "x"), (3500, 3590, "y")])
        _StubOrchestrator.plan = {"a": ("completed", srt, None)}
        _run_main(monkeypatch, _ensemble_argv([a], out_dir, tmp_path),
                  [_media(a, 3600.0)], [(O, "EnsembleOrchestrator", _StubOrchestrator)])
        cfg = _StubOrchestrator.last_pass1_config
        assert cfg["asr_telemetry_enabled"] is True and cfg["asr_telemetry"] is None
        _run_main(monkeypatch, _ensemble_argv([a], out_dir, tmp_path, ["--no-asr-telemetry", "--asr-telemetry", str(tmp_path / "t")]),
                  [_media(a, 3600.0)], [(O, "EnsembleOrchestrator", _StubOrchestrator)])
        cfg = _StubOrchestrator.last_pass1_config
        assert cfg["asr_telemetry_enabled"] is False and cfg["asr_telemetry"] == str(tmp_path / "t")

    def test_failed_result_without_error_key_is_failed(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.ensemble.orchestrator as O

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        _StubOrchestrator.plan = {"a": ("failed", None, None)}
        code = _run_main(monkeypatch, _ensemble_argv([a], out_dir, tmp_path),
                         [_media(a, 3600.0)], [(O, "EnsembleOrchestrator", _StubOrchestrator)])
        assert code == 1
        f = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))["files"][0]
        assert f["state"] == "failed" and "ensemble pass failed" in f["detail"]

    def test_all_skipped_exits_zero_and_writes_manifest(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.ensemble.orchestrator as O

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        _fake_srt(out_dir, "a.ja.merged.whisperjav.srt", [(1, 2, "x")])
        _StubOrchestrator.plan = {}
        code = _run_main(monkeypatch, _ensemble_argv([a], out_dir, tmp_path, ["--skip-existing"]),
                         [_media(a, 3600.0)], [(O, "EnsembleOrchestrator", _StubOrchestrator)])
        assert code == 0
        data = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert [f["state"] for f in data["files"]] == ["skipped"]


# ----- sync -------------------------------------------------------------------

class _StubPipeline:
    """Replaces FasterPipeline in the sync path. `plan` maps basename -> callable."""
    plan = {}

    def __init__(self, **kw):
        self.output_dir = kw.get("output_dir")

    def process(self, media_info):
        return self.plan[media_info["basename"]]()

    def cleanup(self):
        pass


def _sync_argv(files, out_dir, tmp_path, extra=()):
    return [str(f) for f in files] + ["--mode", "faster"] + _base_args(out_dir, tmp_path) + list(extra)


def _metadata(srt, streak=0):
    return {"output_files": {"final_srt": str(srt) if srt else ""},
            "summary": {"final_subtitles_refined": 2, "total_processing_time_seconds": 1.0,
                        "speech_positive_empty_streak": streak}}


class TestSyncPath:
    def test_done_exits_zero_via_nuclear_exit(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        srt = _fake_srt(out_dir, "a.ja.whisperjav.srt", [(1, 2, "x"), (3500, 3590, "y")])
        _StubPipeline.plan = {"a": lambda: _metadata(srt)}
        code = _run_main(monkeypatch, _sync_argv([a], out_dir, tmp_path),
                         [_media(a, 3600.0)], [(M, "FasterPipeline", _StubPipeline)])
        assert code == 0
        data = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert data["mode"] == "faster"
        assert data["files"][0]["state"] == "done"

    def test_exception_in_pipeline_is_failed_and_exits_one(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")

        def _boom():
            raise RuntimeError("cuda died")

        _StubPipeline.plan = {"a": _boom}
        code = _run_main(monkeypatch, _sync_argv([a], out_dir, tmp_path),
                         [_media(a, 3600.0)], [(M, "FasterPipeline", _StubPipeline)])
        assert code == 1
        data = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert data["files"][0]["state"] == "failed"
        assert "cuda died" in data["files"][0]["error"]

    def test_empty_is_zero_by_default(self, monkeypatch, media_dir, out_dir, tmp_path):
        """#263's shape: a 0-byte SRT. Reported as empty; exit 0 unless --fail-on empty."""
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        srt = out_dir / "a.ja.whisperjav.srt"; srt.write_text("", encoding="utf-8")
        _StubPipeline.plan = {"a": lambda: _metadata(srt)}
        argv = _sync_argv([a], out_dir, tmp_path)
        assert _run_main(monkeypatch, argv, [_media(a, 3600.0)], [(M, "FasterPipeline", _StubPipeline)]) == 0
        assert _run_main(monkeypatch, argv + ["--fail-on", "empty,suspect"], [_media(a, 3600.0)],
                         [(M, "FasterPipeline", _StubPipeline)]) == 1

    def test_corroborated_low_span_is_suspect_not_failed(self, monkeypatch, media_dir, out_dir, tmp_path):
        """@daoran9's case with the streak signal: suspect, exit 0 by default."""
        import whisperjav.main as M

        a = media_dir / "gqn.mp4"; a.write_bytes(b"x")
        srt = _fake_srt(out_dir, "gqn.ja.whisperjav.srt", [(10, 12, "a"), (370, 376.9, "b")])
        _StubPipeline.plan = {"gqn": lambda: _metadata(srt, streak=8)}
        code = _run_main(monkeypatch, _sync_argv([a], out_dir, tmp_path),
                         [_media(a, 8766.0)], [(M, "FasterPipeline", _StubPipeline)])
        assert code == 0
        f = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))["files"][0]
        assert f["state"] == "suspect" and f["coverage"] == "low"
        assert "consecutive empty results" in f["detail"]

    def test_folder_input_with_source_output_puts_the_manifest_in_the_folder(self, monkeypatch, media_dir, out_dir, tmp_path):
        """The GUI's Add Folder + 'save next to source' case: the manifest goes
        inside the folder the user named, which is also where the GUI looks."""
        import whisperjav.main as M
        from whisperjav.utils.run_outcome import default_manifest_path

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        srt = _fake_srt(media_dir, "a.ja.whisperjav.srt", [(1, 2, "x"), (3500, 3590, "y")])
        _StubPipeline.plan = {"a": lambda: _metadata(srt)}
        argv = [str(media_dir), "--mode", "faster", "--output-dir", "source",
                "--temp-dir", str(tmp_path / "tmp"), "--accept-cpu-mode", "--no-signature", "--no-progress"]
        code = _run_main(monkeypatch, argv, [_media(a, 3600.0)], [(M, "FasterPipeline", _StubPipeline)])
        assert code == 0
        expected = default_manifest_path("source", [str(media_dir)])
        assert expected == media_dir / MANIFEST_NAME
        assert expected.exists()

    def test_skip_existing_is_skipped(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        _fake_srt(out_dir, "a.ja.whisperjav.srt", [(1, 2, "x")])
        _StubPipeline.plan = {}
        code = _run_main(monkeypatch, _sync_argv([a], out_dir, tmp_path, ["--skip-existing"]),
                         [_media(a, 3600.0)], [(M, "FasterPipeline", _StubPipeline)])
        assert code == 0
        assert json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))["files"][0]["state"] == "skipped"

    def test_interrupt_mid_batch_still_reports_finished_files(self, monkeypatch, media_dir, out_dir, tmp_path):
        """Ctrl-C after file 1: the table and manifest cover file 1; exit 1."""
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        b = media_dir / "b.mp4"; b.write_bytes(b"x")
        srt = _fake_srt(out_dir, "a.ja.whisperjav.srt", [(1, 2, "x"), (3500, 3590, "y")])

        def _interrupt():
            raise KeyboardInterrupt

        _StubPipeline.plan = {"a": lambda: _metadata(srt), "b": _interrupt}
        code = _run_main(monkeypatch, _sync_argv([a, b], out_dir, tmp_path),
                         [_media(a, 3600.0), _media(b, 3600.0)], [(M, "FasterPipeline", _StubPipeline)])
        assert code == 1
        data = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert [f["state"] for f in data["files"]] == ["done"]
        assert "Interrupted" in data["note"]

    def test_vtt_output_is_the_file_the_manifest_names(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        srt = _fake_srt(out_dir, "a.ja.whisperjav.srt", [(1, 2, "x"), (3500, 3590, "y")])
        _StubPipeline.plan = {"a": lambda: _metadata(srt)}
        code = _run_main(monkeypatch, _sync_argv([a], out_dir, tmp_path, ["--output-format", "vtt"]),
                         [_media(a, 3600.0)], [(M, "FasterPipeline", _StubPipeline)])
        assert code == 0
        f = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))["files"][0]
        assert f["state"] == "done"
        assert f["output"].endswith(".vtt") and Path(f["output"]).exists()
        assert not srt.exists()

    def test_error_after_classification_replaces_the_outcome(self, monkeypatch, media_dir, out_dir, tmp_path):
        """One outcome per file, even when a later step in the same iteration raises."""
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        srt = _fake_srt(out_dir, "a.ja.whisperjav.srt", [(1, 2, "x"), (3500, 3590, "y")])
        _StubPipeline.plan = {"a": lambda: _metadata(srt)}

        def _boom(*a_, **k_):
            raise RuntimeError("signature step exploded")

        # apply_vtt_conversion runs after classification, only for non-srt formats.
        code = _run_main(monkeypatch, _sync_argv([a], out_dir, tmp_path, ["--output-format", "both"]),
                         [_media(a, 3600.0)],
                         [(M, "FasterPipeline", _StubPipeline), (M, "apply_vtt_conversion", _boom)])
        assert code == 1
        data = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert len(data["files"]) == 1
        assert data["files"][0]["state"] == "failed"
        assert "signature step exploded" in data["files"][0]["error"]

    def test_bad_min_coverage_is_rejected_before_work_starts(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        called = []
        _StubPipeline.plan = {"a": lambda: called.append(1) or _metadata(None)}
        code = _run_main(monkeypatch, _sync_argv([a], out_dir, tmp_path, ["--min-coverage", "1.5"]),
                         [_media(a, 3600.0)], [(M, "FasterPipeline", _StubPipeline)])
        assert code == 2
        assert called == []

    def test_bad_fail_on_value_is_rejected_before_work_starts(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        called = []
        _StubPipeline.plan = {"a": lambda: called.append(1) or _metadata(None)}
        code = _run_main(monkeypatch, _sync_argv([a], out_dir, tmp_path, ["--fail-on", "partial"]),
                         [_media(a, 3600.0)], [(M, "FasterPipeline", _StubPipeline)])
        assert code == 2
        assert called == []


# ----- async ------------------------------------------------------------------

class _StubTask:
    def __init__(self, media_info, status, result=None, error=None):
        from whisperjav.utils.async_processor import ProcessingStatus

        self.task_id = media_info["basename"]
        self.media_info = media_info
        self.status = ProcessingStatus(status)
        self.result = result
        self.error = error
        self.start_time = 0.0
        self.end_time = 1.0


class _StubProcessor:
    def __init__(self, tasks):
        self._tasks = {t.task_id: t for t in tasks}

    def get_task_status(self, task_id):
        return self._tasks.get(task_id)

    def wait_for_task(self, task_id, timeout=None):
        return self._tasks.get(task_id)


class _StubAsyncManager:
    """Replaces AsyncPipelineManager. `plan` maps basename -> (status, result, error)."""
    plan = {}

    def __init__(self, ui_update_callback=None, verbosity=None):
        self.processor = None

    def process_files(self, media_files, mode, config):
        tasks = [_StubTask(m, *self.plan[m["basename"]]) for m in media_files]
        self.processor = _StubProcessor(tasks)
        return [t.task_id for t in tasks]

    def shutdown(self):
        pass


class TestAsyncPath:
    """Until v1.9.2 this path hard-coded a zero failure count."""

    def test_mixed_batch_exits_one_and_reports_each(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        b = media_dir / "b.mp4"; b.write_bytes(b"x")
        srt = _fake_srt(out_dir, "a.ja.whisperjav.srt", [(1, 2, "x"), (3500, 3590, "y")])
        _StubAsyncManager.plan = {
            "a": ("completed", _metadata(srt), None),
            "b": ("failed", None, RuntimeError("worker died")),
        }
        argv = _sync_argv([a, b], out_dir, tmp_path, ["--async-processing"])
        code = _run_main(monkeypatch, argv, [_media(a, 3600.0), _media(b, 3600.0)],
                         [(M, "AsyncPipelineManager", _StubAsyncManager)])
        assert code == 1
        states = [f["state"] for f in json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))["files"]]
        assert states == ["done", "failed"]

    def test_all_done_exits_zero(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        srt = _fake_srt(out_dir, "a.ja.whisperjav.srt", [(1, 2, "x"), (3500, 3590, "y")])
        _StubAsyncManager.plan = {"a": ("completed", _metadata(srt), None)}
        code = _run_main(monkeypatch, _sync_argv([a], out_dir, tmp_path, ["--async-processing"]),
                         [_media(a, 3600.0)], [(M, "AsyncPipelineManager", _StubAsyncManager)])
        assert code == 0

    def test_cancelled_task_is_failed(self, monkeypatch, media_dir, out_dir, tmp_path):
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        _StubAsyncManager.plan = {"a": ("cancelled", None, None)}
        code = _run_main(monkeypatch, _sync_argv([a], out_dir, tmp_path, ["--async-processing"]),
                         [_media(a, 3600.0)], [(M, "AsyncPipelineManager", _StubAsyncManager)])
        assert code == 1
        f = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))["files"][0]
        assert f["state"] == "failed" and f["detail"] == "cancelled"

    def test_task_that_never_settles_is_failed_with_its_state(self, monkeypatch, media_dir, out_dir, tmp_path):
        """If the completion callback never records a final status, say so
        rather than guess; the settle window is shortened for the test."""
        import whisperjav.main as M

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        _StubAsyncManager.plan = {"a": ("running", None, None)}
        monkeypatch.setattr(M, "_ASYNC_STATUS_SETTLE_S", 0.2)
        code = _run_main(monkeypatch, _sync_argv([a], out_dir, tmp_path, ["--async-processing"]),
                         [_media(a, 3600.0)], [(M, "AsyncPipelineManager", _StubAsyncManager)])
        assert code == 1
        f = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))["files"][0]
        assert f["state"] == "failed" and "state 'running'" in f["detail"]

    def test_async_source_mode_puts_each_files_output_beside_it(self, monkeypatch, media_dir, out_dir, tmp_path):
        """Pre-v1.9.2 every file's outputs landed beside the *first* input."""
        import whisperjav.pipelines.faster_pipeline as FP

        d1 = tmp_path / "src1"; d1.mkdir()
        d2 = tmp_path / "src2"; d2.mkdir()
        a = d1 / "a.mp4"; a.write_bytes(b"x")
        b = d2 / "b.mp4"; b.write_bytes(b"x")
        seen = {}

        class _RecordingPipeline:
            def __init__(self, **kw):
                self.output_dir = Path(kw.get("output_dir"))

            def process(self, media_info):
                name = media_info["basename"]
                seen[name] = self.output_dir
                srt = _fake_srt(self.output_dir, f"{name}.ja.whisperjav.srt", [(1, 2, "x"), (3500, 3590, "y")])
                return _metadata(srt)

            def cleanup(self):
                pass

        argv = [str(a), str(b), "--mode", "faster", "--async-processing", "--output-dir", "source",
                "--temp-dir", str(tmp_path / "tmp"), "--accept-cpu-mode", "--no-signature", "--no-progress"]
        code = _run_main(monkeypatch, argv, [_media(a, 3600.0), _media(b, 3600.0)],
                         [(FP, "FasterPipeline", _RecordingPipeline)])
        assert code == 0
        assert seen["a"] == d1 and seen["b"] == d2

    def test_real_async_manager_waits_for_its_tasks(self, monkeypatch, media_dir, out_dir, tmp_path):
        """The real AsyncPipelineManager submits with wait=False; the CLI path
        must wait, or it summarises tasks that have not run (the pre-v1.9.2
        defect). Only the pipeline is faked, and it takes real time."""
        import time as _time

        import whisperjav.pipelines.faster_pipeline as FP

        a = media_dir / "a.mp4"; a.write_bytes(b"x")
        srt = _fake_srt(out_dir, "a.ja.whisperjav.srt", [(1, 2, "x"), (3500, 3590, "y")])

        class _SlowPipeline:
            def __init__(self, **kw):
                self.output_dir = kw.get("output_dir")

            def process(self, media_info):
                _time.sleep(0.4)
                return _metadata(srt)

            def cleanup(self):
                pass

        code = _run_main(monkeypatch, _sync_argv([a], out_dir, tmp_path, ["--async-processing"]),
                         [_media(a, 3600.0)], [(FP, "FasterPipeline", _SlowPipeline)])
        assert code == 0
        f = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))["files"][0]
        assert f["state"] == "done" and f["coverage"] == "ok"


# ----- the flag is registered -------------------------------------------------

class TestCliRegistration:
    def test_fail_on_and_min_coverage_are_in_help(self):
        import subprocess

        r = subprocess.run([PY, "-m", "whisperjav.main", "--help"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=180)
        assert "--fail-on" in r.stdout and "--min-coverage" in r.stdout
        ok = subprocess.run([PY, "-m", "whisperjav.main", "--fail-on", "empty", "--help"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        assert ok.returncode == 0
