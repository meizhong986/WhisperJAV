"""The GUI speaks the run-outcome contract in the CLI's own words.

Guards three things:

1. Every GUI launch path forwards the user's "treat as failure" choices as the
   same ``--fail-on`` flag the CLI takes, so GUI and CLI runs obey one rule.
2. When the process exits, the API reads the manifest the CLI wrote for *this*
   run (not a previous run's), and reports the five-state tally.
3. The closing console line says the run finished and gives the tally; it
   never re-derives success from anything but the CLI's exit status.
"""

import json
import os
import time

import pytest

from whisperjav.utils.run_outcome import MANIFEST_NAME, STATES
from whisperjav.webview_gui.api import WhisperJAVAPI


@pytest.fixture
def api():
    return WhisperJAVAPI()


def _base_options(tmp_path, **extra):
    inp = tmp_path / "a.mp4"
    inp.write_bytes(b"x")
    opts = {
        "inputs": [str(inp)],
        "output_dir": str(tmp_path / "out"),
        "mode": "balanced",
        "sensitivity": "balanced",
        "source_language": "japanese",
        "subs_language": "native",
    }
    opts.update(extra)
    return opts


# ---------------------------------------------------------------------------
# 1. --fail-on on every launch path
# ---------------------------------------------------------------------------

class TestFailOnIsForwarded:
    def _flag(self, args):
        return args[args.index("--fail-on") + 1] if "--fail-on" in args else None

    def test_transcription_builder(self, api, tmp_path):
        assert self._flag(api.build_args(_base_options(tmp_path))) is None
        assert self._flag(api.build_args(_base_options(tmp_path, fail_on_empty=True))) == "empty"
        assert self._flag(api.build_args(_base_options(tmp_path, fail_on_suspect=True))) == "suspect"
        assert self._flag(api.build_args(
            _base_options(tmp_path, fail_on_empty=True, fail_on_suspect=True))) == "empty,suspect"

    def test_ensemble_builder(self, api, tmp_path):
        opts = _base_options(tmp_path, fail_on_empty=True, fail_on_suspect=True)
        assert self._flag(api._build_ensemble_args(opts)) == "empty,suspect"
        assert self._flag(api._build_ensemble_args(_base_options(tmp_path))) is None

    def test_twopass_builder(self, api, tmp_path):
        cfg = _base_options(tmp_path, fail_on_suspect=True)
        cfg.update({"pass1_pipeline": "balanced", "pass1_sensitivity": "balanced"})
        assert self._flag(api._build_twopass_args(cfg)) == "suspect"

    def test_settings_map_carries_both_keys(self):
        assert WhisperJAVAPI._GUI_SETTINGS_MAP["fail_on_empty"] == "failOnEmpty"
        assert WhisperJAVAPI._GUI_SETTINGS_MAP["fail_on_suspect"] == "failOnSuspect"
        assert WhisperJAVAPI._GUI_SETTINGS_MAP["asr_telemetry"] == "asrTelemetry"

    def test_telemetry_opt_out_reaches_every_builder(self, api, tmp_path):
        """On by default: nothing is emitted unless the checkbox is unticked."""
        assert "--no-asr-telemetry" not in api.build_args(_base_options(tmp_path))
        assert "--no-asr-telemetry" in api.build_args(_base_options(tmp_path, asr_telemetry=False))
        assert "--no-asr-telemetry" in api.build_args(_base_options(tmp_path, mode="crispasr", asr_telemetry=False))
        cfg = _base_options(tmp_path, asr_telemetry=False)
        cfg.update({"pass1_pipeline": "balanced", "pass1_sensitivity": "balanced"})
        assert "--no-asr-telemetry" in api._build_twopass_args(cfg)
        assert 'id="asrTelemetry" checked' in (
            __import__("pathlib").Path(__file__).resolve().parent.parent / "whisperjav" / "webview_gui" /
            "assets" / "index.html").read_text(encoding="utf-8")

    def test_every_mode_in_the_dropdown_forwards_the_flag(self, api, tmp_path):
        """build_args dispatches to per-mode builders (transformers, crispasr,
        ...); every value the mode dropdown offers must carry --fail-on."""
        import re
        from pathlib import Path

        html = (Path(__file__).resolve().parent.parent / "whisperjav" / "webview_gui" /
                "assets" / "index.html").read_text(encoding="utf-8")
        select = re.search(r'<select[^>]*id="mode"[^>]*>(.*?)</select>', html, re.S).group(1)
        modes = re.findall(r'<option[^>]*value="([^"]+)"', select)
        assert len(modes) >= 6
        for mode in modes:
            opts = _base_options(tmp_path, mode=mode, fail_on_empty=True, fail_on_suspect=True)
            try:
                args = api.build_args(opts)
            except ValueError:
                # a mode the builder rejects with these minimal options is not
                # a launch path; the ones that build must carry the flag
                continue
            assert self._flag(args) == "empty,suspect", f"mode {mode!r} dropped --fail-on"

    def test_checkbox_ids_exist_in_the_markup(self):
        from pathlib import Path

        html = (Path(__file__).resolve().parent.parent / "whisperjav" / "webview_gui" /
                "assets" / "index.html").read_text(encoding="utf-8")
        assert 'id="failOnEmpty"' in html and 'id="failOnSuspect"' in html

    def test_no_launch_path_prints_success_anymore(self):
        """Every closing line uses the contract's wording; 'SUCCESS' is gone
        from the API, including the separate translation runner."""
        import inspect

        import whisperjav.webview_gui.api as api_mod

        assert "[SUCCESS]" not in inspect.getsource(api_mod)


# ---------------------------------------------------------------------------
# 2. Reading the manifest for this run
# ---------------------------------------------------------------------------

def _write_manifest(out_dir, counts, files, exit_status=0, note=None, fails_on=("failed",),
                    started_at=None):
    import datetime as _dt

    out_dir.mkdir(parents=True, exist_ok=True)
    full = {s: 0 for s in STATES}
    full.update(counts)
    if started_at is None:
        started_at = time.time()
    payload = {
        "whisperjav_version": "test", "mode": "balanced",
        "started_at": _dt.datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
        "exit_status": exit_status, "fails_on": list(fails_on), "note": note,
        "counts": full, "files": files,
    }
    p = out_dir / MANIFEST_NAME
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


class TestReadRunSummary:
    def test_reads_this_runs_manifest(self, api, tmp_path):
        opts = _base_options(tmp_path)
        api._run_options = opts
        api._process_started_at = time.time() - 60
        _write_manifest(tmp_path / "out", {"done": 1, "empty": 1},
                        [{"path": str(tmp_path / "a.mp4"), "state": "done", "detail": "2 cue(s)"},
                         {"path": str(tmp_path / "b.mp4"), "state": "empty", "detail": "no subtitles"}])
        s = api._read_run_summary()
        assert s is not None
        assert s["counts"]["done"] == 1 and s["counts"]["empty"] == 1
        assert s["tally"] == "done 1 · empty 1 · suspect 0 · failed 0 · skipped 0"
        assert [f["state"] for f in s["files"]] == ["done", "empty"]
        assert s["files"][0]["name"] == "a.mp4"

    def test_ignores_a_previous_runs_manifest(self, api, tmp_path):
        """Decided by the manifest's own started_at, stamped by the CLI after
        this process began; a run that started one second before ours is a
        previous run, whatever the file's mtime says."""
        opts = _base_options(tmp_path)
        api._run_options = opts
        now = time.time()
        _write_manifest(tmp_path / "out", {"done": 1}, [], started_at=now - 1.0)
        api._process_started_at = now
        assert api._read_run_summary() is None
        _write_manifest(tmp_path / "out", {"done": 1}, [], started_at=now + 1.0)
        assert api._read_run_summary() is not None

    def test_manifest_without_started_at_falls_back_to_mtime(self, api, tmp_path):
        opts = _base_options(tmp_path)
        api._run_options = opts
        p = _write_manifest(tmp_path / "out", {"done": 1}, [])
        data = json.loads(p.read_text(encoding="utf-8"))
        del data["started_at"]
        p.write_text(json.dumps(data), encoding="utf-8")
        old = time.time() - 3600
        os.utime(p, (old, old))
        api._process_started_at = time.time()
        assert api._read_run_summary() is None

    def test_folder_input_reads_the_manifest_inside_the_folder(self, api, tmp_path):
        """Add Folder + save next to source: the CLI writes inside the folder
        (default_manifest_path on the raw input); the GUI must look there."""
        folder = tmp_path / "batch"
        folder.mkdir()
        (folder / "a.mp4").write_bytes(b"x")
        api._run_options = {"inputs": [str(folder)], "output_dir": "source"}
        api._process_started_at = time.time() - 5
        _write_manifest(folder, {"done": 1}, [{"path": str(folder / "a.mp4"), "state": "done", "detail": ""}])
        s = api._read_run_summary()
        assert s is not None and s["counts"]["done"] == 1

    def test_missing_manifest_is_none(self, api, tmp_path):
        api._run_options = _base_options(tmp_path)
        api._process_started_at = time.time()
        assert api._read_run_summary() is None

    def test_source_output_dir_looks_beside_the_first_input(self, api, tmp_path):
        opts = _base_options(tmp_path, output_dir="source")
        api._run_options = opts
        api._process_started_at = time.time() - 5
        _write_manifest(tmp_path, {"suspect": 1},
                        [{"path": str(tmp_path / "a.mp4"), "state": "suspect", "detail": "low span"}])
        s = api._read_run_summary()
        assert s is not None and s["counts"]["suspect"] == 1

    def test_corrupt_manifest_is_none(self, api, tmp_path):
        api._run_options = _base_options(tmp_path)
        api._process_started_at = time.time() - 5
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
        assert api._read_run_summary() is None


# ---------------------------------------------------------------------------
# 3. The closing line and the status payload
# ---------------------------------------------------------------------------

class _ExitedProcess:
    def __init__(self, code):
        self.returncode = code

    def poll(self):
        return self.returncode


def _drain(api):
    lines = []
    while not api.log_queue.empty():
        lines.append(api.log_queue.get())
    return "".join(lines)


class TestStatusOnExit:
    def test_finished_with_tally_on_exit_zero(self, api, tmp_path):
        api._run_options = _base_options(tmp_path)
        api._process_started_at = time.time() - 5
        _write_manifest(tmp_path / "out", {"done": 1, "empty": 1}, [])
        api.process = _ExitedProcess(0)
        api.status = "running"
        st = api.get_process_status()
        assert st["status"] == "completed" and st["exit_code"] == 0
        assert st["run_summary"]["counts"]["empty"] == 1
        line = _drain(api)
        assert "[FINISHED] done 1 · empty 1 · suspect 0 · failed 0 · skipped 0 (exit status 0)" in line
        assert "SUCCESS" not in line

    def test_finished_with_failures_on_exit_one(self, api, tmp_path):
        api._run_options = _base_options(tmp_path)
        api._process_started_at = time.time() - 5
        _write_manifest(tmp_path / "out", {"done": 1, "failed": 1}, [], exit_status=1)
        api.process = _ExitedProcess(1)
        api.status = "running"
        st = api.get_process_status()
        assert st["status"] == "error"
        assert "[FINISHED WITH FAILURES] done 1 · empty 0 · suspect 0 · failed 1 · skipped 0 (exit status 1" in _drain(api)

    def test_stopped_run_shows_the_note(self, api, tmp_path):
        api._run_options = _base_options(tmp_path)
        api._process_started_at = time.time() - 5
        _write_manifest(tmp_path / "out", {"done": 1}, [], exit_status=1,
                        note="Interrupted by user after 1 of 2 file(s); the table covers the files that finished.")
        api.process = _ExitedProcess(1)
        api.status = "running"
        st = api.get_process_status()
        assert st["status"] == "error"
        assert "[STOPPED] Interrupted by user" in _drain(api)

    def test_no_manifest_still_reports_the_exit_status(self, api, tmp_path):
        api._run_options = _base_options(tmp_path)
        api._process_started_at = time.time()
        api.process = _ExitedProcess(0)
        api.status = "running"
        st = api.get_process_status()
        assert st["status"] == "completed" and st["run_summary"] is None
        assert "[FINISHED] exit status 0 (no run summary was written)" in _drain(api)

    def test_the_gui_never_overrides_the_cli_exit_status(self, api, tmp_path):
        """An all-empty run exits 0 by the CLI's rule; the GUI reports it as
        finished with the tally, not as an error and not as a bare success."""
        api._run_options = _base_options(tmp_path)
        api._process_started_at = time.time() - 5
        _write_manifest(tmp_path / "out", {"empty": 2}, [])
        api.process = _ExitedProcess(0)
        api.status = "running"
        st = api.get_process_status()
        assert st["status"] == "completed"
        assert st["run_summary"]["tally"].startswith("done 0 · empty 2")
