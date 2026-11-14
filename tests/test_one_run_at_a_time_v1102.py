"""One run at a time (1.10.2, owner W647 1-3 and 5).

The owner's test of 2026-11-14: the GUI's start call took about 20 s on his PC, every click in that time
launched one more complete run of the same film, the runs shared one working folder and deleted each other's
files, Cancel ended one of them and closing the window ended none.

What is pinned here:
- a working folder is held by one run; a second run in it is refused, by name, and nothing else is;
- whisperjav.main stops at start, exit status 1, with that text, before it looks for the media;
- a start call that arrives while another is still launching launches nothing (every start method);
- closing the window ends the run, also one whose start call is still launching;
- the page greys Start at the click and never keeps two log readers.

Nothing here touches the real temporary folder: the lock files go to a folder of the test.
"""
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from whisperjav.utils import run_lock

REPO = Path(__file__).resolve().parents[1]
APP_JS = REPO / "whisperjav" / "webview_gui" / "assets" / "app.js"

HOLDER = (
    "import sys\n"
    "from whisperjav.utils.run_lock import claim_working_folder\n"
    "print(claim_working_folder(sys.argv[1])[1], flush=True)\n"
    "sys.stdin.readline()\n"
)


def _env(lock_home: Path) -> dict:
    env = os.environ.copy()
    for name in ("TMP", "TEMP", "TMPDIR"):
        env[name] = str(lock_home)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.fixture
def lock_home(tmp_path, monkeypatch):
    """The folder the lock files go to, for this process and for the processes the test starts."""
    home = tmp_path / "lock_home"
    home.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(home))
    monkeypatch.setattr(run_lock, "_lock_home", lambda: home)
    monkeypatch.setattr(run_lock, "_held", {})
    yield home
    for handle in run_lock._held.values():
        handle.close()


@pytest.fixture
def holder(lock_home, tmp_path):
    """Another process that holds a working folder until the test lets it go."""
    started = []

    def start(folder: Path) -> subprocess.Popen:
        proc = subprocess.Popen([sys.executable, "-c", HOLDER, str(folder)], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, text=True, env=_env(lock_home), cwd=str(tmp_path))
        assert proc.stdout.readline().strip() == "True"
        started.append(proc)
        return proc

    yield start
    for proc in started:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


class TestWorkingFolderClaim:
    def test_free_folder_is_claimed_and_claimed_again_by_the_same_process(self, lock_home, tmp_path):
        work = tmp_path / "work"
        lock_file, claimed = run_lock.claim_working_folder(str(work))
        assert claimed and lock_file.parent == lock_home
        assert lock_file.read_text().split()[0] == str(os.getpid())
        assert run_lock.claim_working_folder(str(work)) == (lock_file, True)

    def test_folder_held_by_another_process_is_refused_with_its_process_id(self, lock_home, tmp_path, holder):
        work = tmp_path / "work"
        proc = holder(work)
        with pytest.raises(run_lock.WorkingFolderInUse) as refused:
            run_lock.claim_working_folder(str(work))
        text = str(refused.value)
        assert refused.value.pid == proc.pid
        assert f"(process {proc.pid})" in text and str(work) in text and "--temp-dir" in text
        assert text.startswith("WhisperJAV has not started, because another WhisperJAV run")

    def test_the_same_folder_written_another_way_is_the_same_folder(self, lock_home, tmp_path, holder):
        work = tmp_path / "work"
        work.mkdir()
        holder(work)
        with pytest.raises(run_lock.WorkingFolderInUse):
            run_lock.claim_working_folder(str(work / ".." / "work"))

    def test_another_folder_is_not_refused(self, lock_home, tmp_path, holder):
        holder(tmp_path / "work")
        assert run_lock.claim_working_folder(str(tmp_path / "other"))[1] is True

    def test_claim_ends_with_the_process_that_held_it_even_when_killed(self, lock_home, tmp_path, holder):
        work = tmp_path / "work"
        proc = holder(work)
        proc.kill()
        proc.wait()
        assert run_lock.claim_working_folder(str(work))[1] is True

    def test_lock_file_is_removed_when_the_run_ends_normally(self, lock_home, tmp_path, holder):
        work = tmp_path / "work"
        proc = holder(work)
        lock_file = run_lock.lock_file_for(str(work))
        assert lock_file.exists()
        proc.communicate("\n", timeout=30)          # the holder ends by itself
        if sys.platform == "win32":
            assert not lock_file.exists()
        assert run_lock.claim_working_folder(str(work))[1] is True

    def test_lock_file_is_outside_the_working_folder(self, lock_home, tmp_path):
        work = tmp_path / "work"
        lock_file = run_lock.lock_file_for(str(work))
        assert work not in lock_file.parents and lock_file.name.startswith("whisperjav_run_")


class TestMainStopsAtStart:
    def test_second_run_in_a_held_working_folder_exits_1_before_looking_for_media(self, lock_home, tmp_path, holder):
        work = tmp_path / "work"
        proc = holder(work)
        run = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "whisperjav.main", str(tmp_path / "no_such_film.mp4"),
             "--mode", "faster", "--offline", "--temp-dir", str(work)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=_env(lock_home),
            cwd=str(tmp_path), timeout=300)
        out = run.stdout + run.stderr
        assert run.returncode == 1, out[-2000:]
        assert "is using the working folder" in out and f"(process {proc.pid})" in out, out[-2000:]
        assert "No valid media files" not in out, out[-2000:]

    def test_a_run_in_a_free_working_folder_is_not_refused(self, lock_home, tmp_path):
        run = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "whisperjav.main", str(tmp_path / "no_such_film.mp4"),
             "--mode", "faster", "--offline", "--temp-dir", str(tmp_path / "work")],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=_env(lock_home),
            cwd=str(tmp_path), timeout=300)
        out = run.stdout + run.stderr
        assert "is using the working folder" not in out, out[-2000:]
        assert "No valid media files" in out, out[-2000:]


# --- The GUI's start calls ---------------------------------------------------------------------------

class _SlowPopen:
    """subprocess.Popen as the owner's PC gave it: it returns only when the test lets it."""
    release = None
    entered = None
    calls = []

    def __init__(self, cmd, **kwargs):
        type(self).calls.append(cmd)
        type(self).entered.set()
        assert type(self).release.wait(20), "the test never released Popen"
        self.pid = 4242 + len(type(self).calls)
        self.stdout = iter(())
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def api(monkeypatch):
    from whisperjav.webview_gui import api as api_module
    _SlowPopen.release = threading.Event()
    _SlowPopen.entered = threading.Event()
    _SlowPopen.calls = []
    monkeypatch.setattr(api_module.subprocess, "Popen", _SlowPopen)
    obj = api_module.WhisperJAVAPI()
    for name in ("build_args", "_build_ensemble_args", "_build_twopass_args"):
        monkeypatch.setattr(obj, name, lambda options: ["film.mp4"])
    monkeypatch.setattr(obj, "_with_log_file", lambda args, options: args)
    monkeypatch.setattr(obj, "_compute_expected_outputs", lambda *a, **k: [])
    monkeypatch.setattr(obj, "validate_ensemble_config", lambda options: {"valid": True})
    ended = []
    monkeypatch.setattr(api_module, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(api_module, "terminate_process_tree",
                        lambda pid, **kw: ended.append(pid) or {"success": True, "errors": [], "terminated": [pid],
                                                                 "killed": []})
    obj._ended = ended
    yield obj
    _SlowPopen.release.set()


RUN_STARTS = ["start_process", "start_ensemble_process", "start_ensemble_twopass"]


class TestOneStartCallAtATime:
    @pytest.mark.parametrize("first", RUN_STARTS)
    @pytest.mark.parametrize("second", RUN_STARTS)
    def test_a_start_call_during_a_slow_launch_launches_nothing(self, api, first, second):
        results = {}
        worker = threading.Thread(target=lambda: results.update(first=getattr(api, first)({})))
        worker.start()
        assert _SlowPopen.entered.wait(10)
        # The first call is inside Popen: before 1.10.2 self.process was still None here and this launched a run.
        assert api.process is None
        refused = getattr(api, second)({})
        assert refused == {"success": False, "message": "Process already running"}
        assert len(_SlowPopen.calls) == 1
        _SlowPopen.release.set()
        worker.join(10)
        assert results["first"]["success"] is True
        assert getattr(api, second)({})["success"] is False      # running now: the old check answers
        assert len(_SlowPopen.calls) == 1

    def test_a_failed_start_does_not_keep_the_lock(self, api, monkeypatch):
        monkeypatch.setattr(api, "_build_twopass_args", lambda config: (_ for _ in ()).throw(ValueError("bad")))
        assert api.start_ensemble_twopass({}) == {"success": False, "message": "bad"}
        assert api._start_lock.acquire(blocking=False)
        api._start_lock.release()

    def test_the_page_sees_the_same_method_names_and_one_parameter(self, api):
        import inspect
        for name in RUN_STARTS + ["start_translation"]:
            method = getattr(api, name)
            assert method.__name__ == name
            assert len(inspect.getfullargspec(method).args) == 2      # self and the options

    def test_translation_start_during_a_slow_translation_launch_launches_nothing(self, api):
        options = {"inputs": [__file__], "provider": "ollama", "model": "m"}
        results = {}
        worker = threading.Thread(target=lambda: results.update(first=api.start_translation(options)))
        worker.start()
        if not _SlowPopen.entered.wait(10):
            worker.join(10)
            pytest.skip(f"start_translation did not reach Popen with these options: {results.get('first')}")
        assert api.start_translation(options) == {"success": False, "error": "Translation already in progress"}
        assert len(_SlowPopen.calls) == 1
        _SlowPopen.release.set()
        worker.join(10)


class TestClosingTheWindowEndsTheRun:
    def test_running_run_is_ended_and_no_start_is_taken_afterwards(self, api):
        _SlowPopen.release.set()
        assert api.start_ensemble_twopass({})["success"] is True
        pid = api.process.pid
        assert api._end_runs_on_close(wait_s=1) == [pid]
        assert api._ended == [pid]
        assert api.start_ensemble_twopass({}) == {"success": False, "message": "Process already running"}
        assert len(_SlowPopen.calls) == 1

    def test_a_run_whose_start_call_is_still_launching_is_ended_too(self, api):
        worker = threading.Thread(target=lambda: api.start_ensemble_twopass({}))
        worker.start()
        assert _SlowPopen.entered.wait(10)
        closed = {}
        closer = threading.Thread(target=lambda: closed.update(ended=api._end_runs_on_close(wait_s=15)))
        closer.start()
        time.sleep(0.3)
        assert closer.is_alive()          # waiting for the launch, not leaving its run behind
        _SlowPopen.release.set()
        worker.join(10)
        closer.join(20)
        assert closed["ended"] == [api.process.pid] and api._ended == closed["ended"]

    def test_a_run_that_already_ended_is_left_alone(self, api):
        _SlowPopen.release.set()
        api.start_ensemble_twopass({})
        api.process.returncode = 0
        assert api._end_runs_on_close(wait_s=1) == [] and api._ended == []

    def test_the_closing_call_is_not_callable_from_the_page(self):
        from whisperjav.webview_gui.api import WhisperJAVAPI
        assert not [n for n in dir(WhisperJAVAPI) if "end_runs" in n and not n.startswith("_")]

    def test_the_gui_calls_it_after_the_window_has_closed(self):
        source = (REPO / "whisperjav" / "webview_gui" / "main.py").read_text(encoding="utf-8")
        body = source[source.index("def main():"):]
        assert body.index("webview.start(") < body.index("_end_runs_after_close()")
        assert "_api._end_runs_on_close()" in source


# --- The page (app.js). A GUI change counts as verified only when the owner has clicked it; these keep the
# --- wiring from being taken out unnoticed.

def _function(source: str, header: str) -> str:
    """The text of one method of app.js, from its header to the next method at the same indent."""
    start = source.index(header)
    indent = re.match(r" *", header).group(0)
    end = re.search(r"\n%s\},?\n" % indent, source[start:])
    return source[start:start + end.end()]


class TestThePage:
    @pytest.fixture(scope="class")
    def js(self):
        return APP_JS.read_text(encoding="utf-8")

    @pytest.mark.parametrize("header,call", [
        ("    async start() {", "pywebview.api.start_process("),
        ("    async startProcessing() {", "pywebview.api.start_ensemble_twopass("),
        ("    async runOnCpu() {", "pywebview.api.rerun_on_cpu("),
    ])
    def test_start_is_greyed_before_the_start_call_and_released_after_it(self, js, header, call):
        body = _function(js, header)
        assert body.index("beginStart()") < body.index(call) < body.index("endStart(")
        assert "finally" in body

    def test_begin_start_refuses_a_second_start_and_greys_the_button(self, js):
        body = _function(js, "    beginStart() {")
        assert "if (AppState.isStarting || AppState.isRunning) return false;" in body
        assert body.index("AppState.isStarting = true") < body.index("updateButtonStates()")
        assert "setStatus('Starting...')" in body

    def test_both_places_that_set_the_start_button_know_a_run_is_starting(self, js):
        lines = [l for l in js.splitlines() if l.strip().startswith("startBtn.disabled =")]
        assert len(lines) == 2 and all("AppState.isStarting" in l and "AppState.isRunning" in l for l in lines)

    def test_cancel_stays_greyed_while_starting(self, js):
        assert "cancelBtn.disabled = !AppState.isRunning;" in js

    def test_never_two_log_readers_or_two_status_monitors(self, js):
        log = _function(js, "    startLogPolling() {")
        status = _function(js, "    startStatusMonitoring() {")
        assert log.index("this.stopLogPolling();") < log.index("setInterval(")
        assert status.index("this.stopStatusMonitoring();") < status.index("setInterval(")

    def test_translation_start_ignores_a_click_while_it_runs(self, js):
        body = _function(js, "    async startTranslation() {")
        assert body.index("if (this.state.isRunning) return;") < body.index("pywebview.api.start_translation(")


class TestTheGuiShowsTheRefusalAsARunThatDidNotStart:
    """Owner W648 2: the "The run did not start" dialog, with the GUI's control named instead of --temp-dir."""

    LINE = ("2026-11-14 21:00:00 - whisperjav - ERROR - " + run_lock.in_use_message(r"C:\Temp\whisperjav", 4321)
            + "\n")

    def test_the_text_names_the_gui_control_and_not_the_flag(self):
        text = run_lock.gui_refusal_text(["banner\n", self.LINE, "\n"])
        assert text == ("WhisperJAV has not started, because another WhisperJAV run (process 4321) is using the "
                        r"working folder C:\Temp\whisperjav. Two runs in one working folder delete each other's "
                        "files. Wait for that run to end, or give this run its own working folder in Temp dir "
                        "(Advanced Options tab).")
        assert "--temp-dir" not in text

    def test_no_text_for_a_run_that_printed_no_refusal(self):
        assert run_lock.gui_refusal_text(["No valid media files found\n"]) is None

    def test_the_console_text_is_the_approved_one(self):
        assert run_lock.in_use_message("F", 7) == (
            "WhisperJAV has not started, because another WhisperJAV run (process 7) is using the working folder F. "
            "Two runs in one working folder delete each other's files. Wait for that run to end, or give this run "
            "its own working folder with --temp-dir.")

    def test_the_status_the_page_reads_carries_it(self, api):
        import collections
        api.status = "error"
        api.exit_code = 1
        api._output_tail = collections.deque([self.LINE], maxlen=80)
        status = api.get_process_status()
        assert status["working_folder_in_use"].endswith("in Temp dir (Advanced Options tab).")
        api._output_tail = collections.deque(["Traceback\n"], maxlen=80)
        assert "working_folder_in_use" not in api.get_process_status()

    def test_the_page_shows_the_did_not_start_dialog_before_the_other_cases(self):
        js = APP_JS.read_text(encoding="utf-8")
        at = js.index("} else if (status.working_folder_in_use) {")
        branch = js[at:js.index("} else if (status.exit_code === 3) {")]
        assert "RunStopDialog.show('The run did not start'," in branch
        assert "${status.working_folder_in_use} Nothing was processed." in branch
        assert at < js.index("RunStopDialog.show('The run ended without a run summary'")

    def test_the_control_named_exists_in_the_window(self):
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        assert 'for="tempDir"' in html and ">Temp dir:</label>" in html and "Advanced Options" in html


class TestTheGroupsLineDoesNotCallItSpeech:
    def test_wording(self):
        source = (REPO / "whisperjav" / "pipelines" / "crispasr_pipeline.py").read_text(encoding="utf-8")
        assert 'VAD groups over {len(scenes)} scenes; "' in source
        assert 'the groups span {speech_s:.0f} s of {duration:.0f} s' in source
        assert "s of speech in" not in source


class TestSceneLinesLeaveTheConsole:
    def test_a_scene_without_aligner_is_logged_at_debug_like_an_ok_one(self):
        source = (REPO / "whisperjav" / "modules" / "subtitle_pipeline" / "orchestrator.py").read_text(
            encoding="utf-8")
        assert '(logger.debug if sentinel_status in ("OK", "N/A") else logger.info)(' in source
