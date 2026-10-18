"""
Starting CrispASR's program for one film and judging what it left behind
(whisperjav/crispasr/runner.py; master plan 4.1, 4.3, 4.4).

The program here is a stand-in: a small Python script, started by this test's own interpreter, that
does what a plan file in its working folder says -- write the JSON or not, print progress, warning or
download lines, exit with a code or a crash status, or sleep until it is ended. So every outcome
class is tested without CrispASR. Popen cannot start a .py file on Windows, so the request's argv()
puts the interpreter in front of the real command line; everything else -- the checks before the
start, the working folder, the pipes and their reader threads, the job object, the judging and
run.log -- is the runner's own.

Run with: python -m pytest tests/test_crispasr_runner.py -q
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from whisperjav.crispasr import runner
from whisperjav.crispasr.ojf import read_ojf
from whisperjav.crispasr.runner import (
    EXIT_MEANING,
    RUN_FOLDER,
    CrispOutcome,
    CrispRunCancelled,
    CrispRunRequest,
    default_threads,
    exit_code_meaning,
    film_run_dir,
    is_crash,
    run_crispasr_json,
)

MODEL = "C:\\m\\parakeet-tdt-0.6b-ja-q8_0.gguf"
STACK_BUFFER_OVERRUN = 0xC0000409  # run A5's crash status
# What os._exit() needs to end the stand-in with that status on Windows.
STACK_BUFFER_OVERRUN_EXIT = STACK_BUFFER_OVERRUN - (1 << 32)

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows exception statuses and job objects")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")

COMPLETE_JSON = json.dumps({
    "crispasr": {"backend": "parakeet", "model": MODEL, "language": "ja"},
    "transcription": [
        {"timestamps": {"from": "00:00:25,160", "to": "00:00:26,880"},
         "offsets": {"from": 25160, "to": 26880},
         "text": "テストです。", "chunk_id": 0,
         "words": [{"text": "テスト", "t0": 2516, "t1": 2548, "offsets": {"from": 25160, "to": 25480}}],
         "tokens": [{"text": "テスト", "p": 0.99, "t0": 2516, "t1": 2548,
                     "offsets": {"from": 25160, "to": 25480}}]},
    ],
}, ensure_ascii=False)

STAND_IN = '''\
"""Stand-in for crispasr.exe: does what standin.json in its working folder says."""
import json
import os
import sys
import threading
import time


def stdin_state():
    """Whether the input is a terminal, and whether reading it ends at once (closed) within 5 s."""
    if sys.stdin is None:
        return None, None
    got = []
    reader = threading.Thread(target=lambda: got.append(sys.stdin.buffer.read()), daemon=True)
    reader.start()
    reader.join(5)
    return sys.stdin.isatty(), (got[0] == b"" if got else False)


def main():
    with open("standin.json", encoding="utf-8") as fh:
        plan = json.load(fh)
    argv = sys.argv[1:]
    out_name = argv[argv.index("-of") + 1]
    cache = argv[argv.index("--cache-dir") + 1] if "--cache-dir" in argv else None
    isatty, eof = stdin_state()
    with open("seen.json", "w", encoding="utf-8") as fh:
        json.dump({"argv": argv, "cwd": os.getcwd(), "pid": os.getpid(),
                   "stale_json_present": os.path.exists(out_name + ".json"),
                   "stdin_isatty": isatty, "stdin_eof": eof,
                   "cache_dir_entries": sorted(os.listdir(cache)) if cache and os.path.isdir(cache) else None},
                  fh)
    for kind, value in plan["steps"]:
        if kind == "err":
            sys.stderr.buffer.write(value.encode("utf-8"))
            sys.stderr.buffer.flush()
        elif kind == "err_hex":
            sys.stderr.buffer.write(bytes.fromhex(value))
            sys.stderr.buffer.flush()
        elif kind == "out":
            sys.stdout.buffer.write(value.encode("utf-8"))
            sys.stdout.buffer.flush()
        elif kind == "json":
            with open(out_name + ".json", "w", encoding="utf-8") as fh:
                fh.write(value)
        elif kind == "sleep":
            time.sleep(value)
        elif kind == "wait_for":
            deadline = time.monotonic() + 5
            while not os.path.exists(value) and time.monotonic() < deadline:
                time.sleep(0.01)
            if not os.path.exists(value):
                with open("late.txt", "a", encoding="utf-8") as fh:
                    fh.write(value + "\\n")
        elif kind == "signal":
            os.kill(os.getpid(), value)
    os._exit(plan["exit"])


main()
'''


class StandInRequest(CrispRunRequest):
    """The real request; only its program is a Python script, started by this test's interpreter."""

    def argv(self) -> list[str]:
        return [sys.executable, "-I", *super().argv()]


@pytest.fixture
def stand_in(tmp_path):
    path = tmp_path / "bin" / "crispasr_standin.py"
    path.parent.mkdir()
    path.write_text(STAND_IN, encoding="utf-8")
    return path


@pytest.fixture
def run_dir(tmp_path):
    folder = tmp_path / "crispasr_out"
    folder.mkdir()
    (folder / "audio.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    (folder / "cuts.json").write_text('{"crispasr_vad": {"version": 1, "kind": "chunks"}}', encoding="utf-8")
    return folder


@pytest.fixture
def started_processes(monkeypatch):
    """Every process started during the test; any still running at the end is killed."""
    processes = []
    real_popen = subprocess.Popen

    class RecordingPopen(real_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            processes.append(self)

    monkeypatch.setattr(subprocess, "Popen", RecordingPopen)
    yield processes
    for proc in processes:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


@pytest.fixture
def make_request(stand_in, run_dir, started_processes):
    def make(steps=(), exit_code=0, **fields):
        plan = {"steps": [list(step) for step in steps], "exit": exit_code}
        (run_dir / "standin.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        fields.setdefault("threads", 2)
        return StandInRequest(exe=stand_in, run_dir=run_dir, backend="parakeet", model=MODEL, **fields)
    return make


def run(req, **kwargs):
    kwargs.setdefault("poll_s", 0.05)
    return run_crispasr_json(req, **kwargs)


def seen(run_dir):
    """What the stand-in recorded at its start: its arguments, working folder and the JSON it found."""
    return json.loads((run_dir / "seen.json").read_text(encoding="utf-8"))


def log_lines(req):
    return req.log_path.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------------------------------

class TestTheCommandLine:
    def test_the_arguments_in_the_planned_order(self):
        exe = Path("C:/p/crispasr.exe")
        req = CrispRunRequest(exe=exe, run_dir=Path("C:/t/crispasr_out"), backend="parakeet", model=MODEL,
                              threads=8)
        assert req.argv() == [str(exe), "-m", MODEL, "--backend", "parakeet", "-t", "8", "-f", "audio.wav",
                              "-l", "ja", "-ojf", "-pp", "-of", "out", "--vad-import", "cuts.json",
                              "--cache-dir", "no_cache"]

    def test_the_builds_and_the_models_switches_come_last(self):
        req = CrispRunRequest(exe=Path("crispasr"), run_dir=Path("r"), backend="fastconformer-ctc",
                              model="model.gguf", threads=4, extra_args=("--punc-model", "none", "--no-gpu"))
        assert req.argv() == ["crispasr", "-m", "model.gguf", "--backend", "fastconformer-ctc", "-t", "4",
                              "-f", "audio.wav", "-l", "ja", "-ojf", "-pp", "-of", "out",
                              "--vad-import", "cuts.json", "--cache-dir", "no_cache",
                              "--punc-model", "none", "--no-gpu"]

    def test_the_program_is_never_asked_to_download(self):
        # plan 3.8 item 6: WhisperJAV never passes --auto-download, and -m never asks for "auto"
        argv = CrispRunRequest(exe=Path("crispasr"), run_dir=Path("r"), backend="cohere", model="..\\m.gguf",
                               threads=1).argv()
        assert "--auto-download" not in argv and argv[argv.index("-m") + 1] != "auto"
        assert argv[argv.index("--cache-dir") + 1] == runner.NO_CACHE_FOLDER == "no_cache"

    def test_extra_arguments_are_passed_as_text(self):
        req = CrispRunRequest(exe=Path("crispasr"), run_dir=Path("r"), backend="parakeet", model="m",
                              threads=1, extra_args=(Path("x"), 3))
        assert req.argv()[-2:] == ["x", "3"]

    def test_other_names_and_language_are_used_where_they_belong(self):
        req = CrispRunRequest(exe=Path("crispasr"), run_dir=Path("r"), backend="cohere", model="m", threads=2,
                              audio_name="a.wav", cuts_name="c.json", out_name="res", language="en")
        argv = req.argv()
        assert argv[argv.index("-f") + 1] == "a.wav"
        assert argv[argv.index("--vad-import") + 1] == "c.json"
        assert argv[argv.index("-of") + 1] == "res"
        assert argv[argv.index("-l") + 1] == "en"

    def test_the_json_and_the_log_live_in_the_run_folder(self, tmp_path):
        req = CrispRunRequest(exe=Path("crispasr"), run_dir=tmp_path, backend="parakeet", model="m", threads=1)
        assert req.json_path == tmp_path / "out.json"
        assert req.log_path == tmp_path / "run.log"
        req.out_name = "res"
        assert req.json_path == tmp_path / "res.json"

    @pytest.mark.parametrize("cpus, expected", [(16, 8), (8, 8), (6, 6), (1, 1), (None, 4)])
    def test_threads_are_the_cpu_count_up_to_8(self, monkeypatch, cpus, expected):
        monkeypatch.setattr(os, "cpu_count", lambda: cpus)
        assert default_threads() == expected
        assert CrispRunRequest(exe=Path("x"), run_dir=Path("r"), backend="b", model="m").threads == expected

    def test_threads_on_this_machine(self):
        assert 1 <= default_threads() <= 8


class TestTheRunFolder:
    """One working folder per film, with a name the program can open on any PC (plan 1.7)."""

    def test_it_lies_under_crispasr_out_in_the_temp_folder(self, tmp_path):
        assert RUN_FOLDER == "crispasr_out"  # the name WhisperJAV's leftover handling knows
        folder = film_run_dir(tmp_path, "sample_film-01")
        assert folder.parent == tmp_path / RUN_FOLDER
        assert re.fullmatch(r"sample_film-01_[0-9a-f]{8}", folder.name)

    @pytest.mark.parametrize("basename", ["テスト動画 [1080p]", "film (1).mp4", "a" * 200, "用户 テスト"])
    def test_its_name_is_short_and_ascii(self, basename):
        name = film_run_dir(Path("temp"), basename).name
        assert re.fullmatch(r"[A-Za-z0-9_-]+", name), name
        assert len(name) <= 40 + 1 + 8

    def test_a_name_with_no_ascii_letters_still_gets_a_folder(self):
        assert re.fullmatch(r"film_[0-9a-f]{8}", film_run_dir(Path("temp"), "テスト").name)

    def test_two_films_never_share_a_folder(self):
        names = ["テスト.mp4", "サンプル.mp4", "test.mp4", "test!.mp4", "test?.mp4"]
        assert len({film_run_dir(Path("temp"), n) for n in names}) == len(names)

    def test_the_same_film_always_gets_the_same_folder(self):
        assert film_run_dir(Path("temp"), "テスト.mp4") == film_run_dir(Path("temp"), "テスト.mp4")


# ---------------------------------------------------------------------------------------------------
# Every outcome
# ---------------------------------------------------------------------------------------------------

class TestOutcomes:
    def test_exit_0_with_a_complete_json_is_ok(self, make_request, run_dir):
        req = make_request([("err", "crispasr: loading model\n"), ("json", COMPLETE_JSON),
                            ("out", "[00:00:25.160 --> 00:00:26.880]  テストです。\n")])
        result = run(req)
        assert result.outcome is CrispOutcome.OK
        assert result.exit_code == 0
        assert result.json_path == run_dir / "out.json"
        assert read_ojf(result.json_path)["crispasr"]["backend"] == "parakeet"
        assert result.seconds > 0
        # the program ran in the run folder, with exactly the request's command line
        started = seen(run_dir)
        assert os.path.samefile(started["cwd"], run_dir)
        assert started["argv"] == CrispRunRequest.argv(req)[1:]

    def test_exit_0_without_json_after_no_speech_detected_is_an_empty_result(self, make_request):
        result = run(make_request([("err", "crispasr: warning: no speech detected\n")]))
        assert result.outcome is CrispOutcome.NO_SPEECH
        assert result.json_path is None
        assert result.warnings == ["crispasr: warning: no speech detected"]

    def test_exit_0_with_nothing_written_is_no_output(self, make_request):
        result = run(make_request([("err", "crispasr: done\n")]))
        assert result.outcome is CrispOutcome.NO_OUTPUT
        assert result.json_path is None
        assert result.exit_code == 0

    def test_exit_0_with_a_damaged_json_is_no_output(self, make_request):
        result = run(make_request([("json", COMPLETE_JSON[:40])]))
        assert result.outcome is CrispOutcome.NO_OUTPUT
        assert result.json_path is None

    def test_a_documented_exit_code_fails_with_its_meaning_and_its_json_is_not_used(self, make_request, run_dir):
        result = run(make_request([("json", COMPLETE_JSON), ("err", "crispasr: error: no model\n")],
                                  exit_code=11))
        assert result.outcome is CrispOutcome.FAILED
        assert result.exit_code == 11
        assert result.meaning == "model could not be resolved"
        assert result.json_path is None
        assert (run_dir / "out.json").is_file()  # left where it is, but not used

    def test_an_undocumented_exit_code_fails_with_its_number(self, make_request):
        result = run(make_request(exit_code=77))
        assert result.outcome is CrispOutcome.FAILED
        assert result.meaning == "undocumented exit code 77"
        assert result.json_path is None

    @windows_only
    def test_a_crash_after_a_complete_json_keeps_the_json(self, make_request, run_dir):
        result = run(make_request([("json", COMPLETE_JSON)], exit_code=STACK_BUFFER_OVERRUN_EXIT))
        assert result.outcome is CrispOutcome.CRASHED
        assert result.exit_code == STACK_BUFFER_OVERRUN
        assert result.meaning == "crashed with status 0xC0000409 (stack buffer overrun (fail-fast))"
        assert result.json_path == run_dir / "out.json"

    @windows_only
    def test_a_crash_without_json_has_nothing_to_keep(self, make_request):
        result = run(make_request(exit_code=STACK_BUFFER_OVERRUN_EXIT))
        assert result.outcome is CrispOutcome.CRASHED
        assert result.json_path is None

    @windows_only
    def test_a_crash_while_writing_the_json_keeps_nothing(self, make_request):
        result = run(make_request([("json", COMPLETE_JSON[:60])], exit_code=STACK_BUFFER_OVERRUN_EXIT))
        assert result.outcome is CrispOutcome.CRASHED
        assert result.json_path is None

    @posix_only
    def test_a_signal_is_a_crash(self, make_request, run_dir):
        result = run(make_request([("json", COMPLETE_JSON), ("signal", 9)]))
        assert result.outcome is CrispOutcome.CRASHED
        assert result.exit_code == -9
        assert result.json_path == run_dir / "out.json"

    def test_a_stale_json_from_before_the_run_is_deleted_first(self, make_request, run_dir):
        (run_dir / "out.json").write_text(COMPLETE_JSON, encoding="utf-8")
        result = run(make_request())
        assert seen(run_dir)["stale_json_present"] is False
        assert result.outcome is CrispOutcome.NO_OUTPUT
        assert result.json_path is None
        assert not (run_dir / "out.json").exists()

    @pytest.mark.parametrize("missing", ["audio.wav", "cuts.json"])
    def test_a_missing_input_stops_before_anything_starts(self, make_request, run_dir, monkeypatch, missing):
        req = make_request()
        (run_dir / missing).unlink()
        started = []

        def no_start(*args, **kwargs):
            started.append(args)
            raise AssertionError("the program must not be started")

        monkeypatch.setattr(subprocess, "Popen", no_start)
        with pytest.raises(FileNotFoundError, match=missing):
            run(req)
        assert started == []
        assert not req.log_path.exists()

    def test_a_program_file_gone_since_it_was_checked_is_the_runs_own_error(self, make_request, stand_in,
                                                                            monkeypatch):
        # Deleted, or taken by an antivirus, after the start check: the file fails with that reason in
        # words (CrispASRRunError, which the pass worker keeps as the file's reason), never a traceback.
        req = make_request()
        stand_in.unlink()
        started = []

        def no_start(*args, **kwargs):
            started.append(args)
            raise AssertionError("the program must not be started")

        monkeypatch.setattr(subprocess, "Popen", no_start)
        with pytest.raises(runner.CrispASRRunError) as caught:
            run(req)
        assert str(caught.value) == f"the CrispASR program is missing: {stand_in}"
        assert not isinstance(caught.value, FileNotFoundError)
        assert started == []
        assert not req.log_path.exists()


class TestNoDownloadByTheProgram:
    """The program never downloads a model by itself (plan 3.8 item 6; U19): its input is a pipe closed at once,
    which no program takes for a terminal, and --cache-dir names an empty folder in the run folder."""

    def test_its_input_is_no_terminal_and_ends_at_once(self, make_request, run_dir):
        run(make_request([("json", COMPLETE_JSON)]))
        started = seen(run_dir)
        assert started["stdin_isatty"] is False
        assert started["stdin_eof"] is True  # reading it ends at once: nobody will ever answer

    def test_its_cache_folder_exists_and_is_empty_even_after_a_run_that_left_files_there(self, make_request,
                                                                                       run_dir):
        stale = run_dir / runner.NO_CACHE_FOLDER
        stale.mkdir()
        (stale / "parakeet-tdt-0.6b-ja-q8_0.gguf").write_bytes(b"left behind")
        run(make_request([("json", COMPLETE_JSON)]))
        assert seen(run_dir)["cache_dir_entries"] == []

    def test_a_model_not_found_locally_is_named_in_plain_words(self, make_request):
        # what the program prints for a missing -m file when it asks nothing (U19, probes B and D)
        lines = ("crispasr: model 'C:\\m\\parakeet-tdt-0.6b-ja-q8_0.gguf' not found locally.\n"
                 "  Available for download: parakeet-tdt-0.6b-ja-q8_0.gguf (~710 MB)\n"
                 "  License: CC-BY-4.0 (see the original NVIDIA model card)\n"
                 "  Use --auto-download or -m auto to download automatically.\n"
                 "crispasr: error: failed to initialise backend 'parakeet'\n")
        result = run(make_request([("err", lines)], exit_code=13))
        assert result.outcome is CrispOutcome.FAILED and result.exit_code == 13
        assert result.meaning == ("the model file could not be opened; the CrispASR program may not download "
                                  "one by itself")
        assert result.download_lines == []  # nothing was downloaded, so no download note follows

    def test_without_that_line_exit_13_keeps_its_documented_meaning(self, make_request):
        result = run(make_request([("err", "crispasr: error: failed to initialise backend 'parakeet'\n")],
                                  exit_code=13))
        assert result.meaning == EXIT_MEANING[13]


class TestTheProgramsOwnProcessorCheck:
    """The program checks the processor before any work and ends with exit code 1, which otherwise means
    bad arguments (cli.cpp:2579-2613). Its own line gives the plainer reason (CrispRunResult.meaning)."""

    LACKS = ("crispasr: error: this build was compiled for x86-64-v3, but this CPU lacks: AVX2, FMA, F16C "
             "(rebuild without GGML_NATIVE).\n")

    def test_a_processor_without_the_instructions_is_named_in_plain_words(self, make_request):
        result = run(make_request([("err", "crispasr: loading\n" + self.LACKS)], exit_code=1))
        assert result.outcome is CrispOutcome.FAILED and result.exit_code == 1
        assert result.detail == "this processor lacks instructions the CrispASR program needs (AVX2, FMA, F16C)"
        assert result.meaning == result.detail
        assert result.meaning != EXIT_MEANING[1]
        assert result.json_path is None

    def test_a_lacks_line_that_names_nothing_still_says_avx2(self, make_request):
        line = "error: this build was compiled for AVX2, but this CPU lacks:  (see the docs).\n"
        result = run(make_request([("err", line)], exit_code=1))
        assert result.meaning == "this processor lacks instructions the CrispASR program needs (AVX2)"

    @pytest.mark.parametrize("spelling", ["initialised", "initialized"])
    def test_no_cpu_backend_is_named_in_plain_words(self, make_request, spelling):
        result = run(make_request([("err", f"crispasr: error: no CPU ggml backend could be {spelling}\n")],
                                  exit_code=1))
        assert result.outcome is CrispOutcome.FAILED
        assert result.detail == "the CrispASR program found no processor support it can run on"
        assert result.meaning == result.detail

    def test_without_such_a_line_the_exit_code_gives_the_meaning(self, make_request):
        result = run(make_request([("err", "crispasr: error: bad argument -x\n")], exit_code=1))
        assert result.detail is None
        assert result.meaning == EXIT_MEANING[1]

    def test_a_run_that_ended_normally_is_never_given_such_a_reason(self, make_request):
        result = run(make_request([("err", self.LACKS + "crispasr: no CPU ggml backend could be initialised\n"),
                                   ("json", COMPLETE_JSON)]))
        assert result.outcome is CrispOutcome.OK
        assert result.detail is None
        assert result.meaning == "finished normally"


# ---------------------------------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------------------------------

def acknowledging(run_dir, calls):
    """A progress display that records each call and tells the stand-in it arrived (file ack_<done>)."""
    def on_progress(done, total):
        calls.append((done, total, threading.get_ident()))
        (run_dir / f"ack_{done}").write_text("", encoding="utf-8")
    return on_progress


def late_acks(run_dir):
    """Counts the stand-in waited for in vain (5 s): they reached the display only after it wrote more."""
    late = run_dir / "late.txt"
    return late.read_text(encoding="utf-8").split() if late.exists() else []


class TestProgress:
    @pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"], ids=["LF", "CRLF", "CR alone"])
    def test_a_progress_line_reaches_the_display_while_the_program_still_runs(self, make_request, run_dir,
                                                                              ending):
        # The stand-in prints one count and then waits, writing nothing more, until the display has it.
        req = make_request([("err", f"crispasr: progress =  50% (3/6 slices){ending}"), ("wait_for", "ack_3"),
                            ("json", COMPLETE_JSON)])
        calls = []
        result = run(req, on_progress=acknowledging(run_dir, calls))
        assert result.outcome is CrispOutcome.OK
        assert [(done, total) for done, total, _ in calls] == [(3, 6)]
        assert late_acks(run_dir) == [], "the line reached the display only after the program wrote more"

    def test_progress_reaches_the_display_in_the_calling_thread(self, make_request, run_dir):
        # Each step that prints a count waits until the display has it, so nothing depends on timing.
        steps = [
            ("err", "crispasr: loading model\n"),
            # one line arriving in two pieces
            ("err", "crispasr: progre"), ("sleep", 0.2), ("err", "ss =  17% (1/6 slices)\n"),
            ("wait_for", "ack_1"),
            ("err", "crispasr: progress =  50% (3/6 slices)\r\n"), ("wait_for", "ack_3"),
            # the same count again is not reported twice
            ("err", "crispasr: progress =  50% (3/6 slices)\n"), ("sleep", 0.3),
            ("err", "crispasr: progress =  83% (5/6 slices)\n"), ("wait_for", "ack_5"),
            ("json", COMPLETE_JSON),
            # the last line, printed just before the program ends, still arrives
            ("err", "crispasr: progress = 100% (6/6 slices)\n"),
        ]
        calls = []
        result = run(make_request(steps), on_progress=acknowledging(run_dir, calls))
        assert result.outcome is CrispOutcome.OK
        assert [(done, total) for done, total, _ in calls] == [(1, 6), (3, 6), (5, 6), (6, 6)]
        assert {ident for _, _, ident in calls} == {threading.get_ident()}
        assert late_acks(run_dir) == []
        assert result.warnings == []

    def test_a_failing_progress_display_does_not_break_the_run(self, make_request):
        req = make_request([("err", "crispasr: progress = 100% (1/1 slices)\n"), ("json", COMPLETE_JSON)])

        def broken(done, total):
            raise RuntimeError("the display is gone")

        assert run(req, on_progress=broken).outcome is CrispOutcome.OK


# ---------------------------------------------------------------------------------------------------
# What is collected from stderr, and run.log
# ---------------------------------------------------------------------------------------------------

class TestWhatIsCollected:
    def test_warning_and_download_lines_are_collected(self, make_request):
        lines = [
            "crispasr: loading model",
            "  crispasr: warning: language forced to ja  ",
            "parakeet: keeping ASR timestamps for slice 3",
            "vad-import: 2 slices out-of-range dropped",
            "crispasr: failed to load the punctuation model",
            "vad-import: slices exported at --chunk-seconds 20; this run uses 30",
            "crispasr: input file not found: extra.wav",
            "crispasr: progress =  50% (1/2 slices)",
            "crispasr: downloading punctuation model (58 MB) from https://huggingface.co/example",
            "crispasr: Download finished",
            # what a missing model prints when no download follows (U19, probes B and D): not collected
            "  Available for download: parakeet-tdt-0.6b-ja-q8_0.gguf (~710 MB)",
            "  Use --auto-download or -m auto to download automatically.",
            # the question a terminal input answers yes, and the library's own lines: collected
            "  Download now? [Y/n] ",
            "  Auto-downloading (--auto-download is set)...",
            "crispasr: downloading via WinHTTP...",
        ]
        result = run(make_request([("err", "\n".join(lines) + "\n"), ("json", COMPLETE_JSON)]))
        assert result.outcome is CrispOutcome.OK
        assert result.warnings == [
            "crispasr: warning: language forced to ja",
            "parakeet: keeping ASR timestamps for slice 3",
            "vad-import: 2 slices out-of-range dropped",
            "crispasr: failed to load the punctuation model",
            "vad-import: slices exported at --chunk-seconds 20; this run uses 30",
            "crispasr: input file not found: extra.wav",
        ]
        assert result.warning_count == 6
        assert result.download_lines == [
            "crispasr: downloading punctuation model (58 MB) from https://huggingface.co/example",
            "Download now? [Y/n]",
            "Auto-downloading (--auto-download is set)...",
            "crispasr: downloading via WinHTTP...",
        ]

    def test_at_most_500_warnings_are_kept_but_all_are_counted(self, make_request):
        text = "".join(f"crispasr: warning: slice {i}\n" for i in range(600))
        result = run(make_request([("err", text)]))
        assert result.warning_count == 600
        assert len(result.warnings) == 500
        assert (result.warnings[0], result.warnings[-1]) == ("crispasr: warning: slice 0",
                                                             "crispasr: warning: slice 499")

    def test_the_stderr_tail_is_its_last_2000_characters(self, make_request):
        lines = [f"crispasr: line {i:04d} " + "x" * 20 for i in range(300)]
        result = run(make_request([("err", "\n".join(lines) + "\n")]))
        assert result.stderr_tail == "\n".join(lines)[-2000:]
        assert result.stderr_tail.endswith("crispasr: line 0299 " + "x" * 20)

    def test_bytes_that_are_not_utf8_do_not_break_the_run(self, make_request):
        raw = b"crispasr: warning: bad bytes \xff\xfe here\n"
        result = run(make_request([("err_hex", raw.hex()), ("json", COMPLETE_JSON)]))
        assert result.outcome is CrispOutcome.OK
        assert result.warnings == ["crispasr: warning: bad bytes \ufffd\ufffd here"]

    def test_run_log_holds_the_command_folder_exit_code_outcome_and_both_streams(self, make_request, run_dir):
        req = make_request([("err", "crispasr: loading model\ncrispasr: warning: テスト\n"), ("json", COMPLETE_JSON),
                            ("out", "[00:00:25.160 --> 00:00:26.880]  テストです。\n")])
        result = run(req)
        assert result.log_path == run_dir / "run.log"
        lines = log_lines(req)
        assert lines[0] == "command: " + subprocess.list2cmdline(req.argv())
        assert lines[1] == f"working folder: {run_dir}"
        assert lines[2] == "exit code: 0"
        assert lines[3] == "outcome: ok"
        assert lines[4].startswith("seconds: ")
        assert lines[5] == "----- stderr"
        split = lines.index("----- stdout")
        assert lines[6:split] == ["crispasr: loading model", "crispasr: warning: テスト"]
        assert lines[split + 1:] == ["[00:00:25.160 --> 00:00:26.880]  テストです。"]

    def test_run_log_of_a_failed_run_names_the_code_and_its_meaning(self, make_request):
        req = make_request(exit_code=11)
        run(req)
        lines = log_lines(req)
        assert lines[2] == "exit code: 11 (0x0000000B): model could not be resolved"
        assert lines[3] == "outcome: failed"

    @windows_only
    def test_run_log_of_a_crash_names_the_status(self, make_request):
        req = make_request(exit_code=STACK_BUFFER_OVERRUN_EXIT)
        run(req)
        lines = log_lines(req)
        assert lines[2] == (f"exit code: {STACK_BUFFER_OVERRUN} (0xC0000409): "
                            "crashed with status 0xC0000409 (stack buffer overrun (fail-fast))")
        assert lines[3] == "outcome: crashed"


# ---------------------------------------------------------------------------------------------------
# Ending the program
# ---------------------------------------------------------------------------------------------------

LONG_RUN = [("err", "crispasr: progress =  10% (1/10 slices)\n"), ("sleep", 60)]


class TestEndingTheProgram:
    def test_a_stop_ends_the_program_promptly_and_run_log_says_so(self, make_request, started_processes):
        req = make_request(LONG_RUN)
        cancel = threading.Event()
        stop_times = []

        def stop_on_first_progress(done, total):
            stop_times.append(time.monotonic())
            cancel.set()

        with pytest.raises(CrispRunCancelled):
            run(req, on_progress=stop_on_first_progress, cancel_event=cancel)
        assert stop_times, "the stand-in never reported progress"
        assert time.monotonic() - stop_times[0] < 5
        (proc,) = started_processes
        assert proc.poll() is not None  # the stand-in has ended
        lines = log_lines(req)
        assert "outcome: ended by WhisperJAV" in lines
        assert "crispasr: progress =  10% (1/10 slices)" in lines

    def test_the_log_of_a_run_it_ended_does_not_blame_the_program(self, make_request):
        # run.log is how a run is understood later; a run WhisperJAV ended was not the program's error.
        req = make_request(LONG_RUN)
        cancel = threading.Event()
        with pytest.raises(CrispRunCancelled):
            run(req, on_progress=lambda done, total: cancel.set(), cancel_event=cancel)
        exit_line = next(line for line in log_lines(req) if line.startswith("exit code:"))
        assert not any(meaning in exit_line for meaning in EXIT_MEANING.values()), exit_line

    @pytest.mark.parametrize("stopped", [False, True], ids=["finished", "stopped"])
    def test_the_programs_pipes_are_closed_after_the_run(self, make_request, started_processes, stopped):
        # Each run opens two pipes; left to the garbage collector, a GUI session of many films keeps them.
        if stopped:
            cancel = threading.Event()
            with pytest.raises(CrispRunCancelled):
                run(make_request(LONG_RUN), on_progress=lambda done, total: cancel.set(), cancel_event=cancel)
        else:
            run(make_request([("json", COMPLETE_JSON)]))
        (proc,) = started_processes
        assert (proc.stdout.closed, proc.stderr.closed) == (True, True)

    def test_an_interrupt_while_waiting_ends_the_program_too(self, make_request, started_processes):
        req = make_request(LONG_RUN)

        def interrupt(done, total):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run(req, on_progress=interrupt)
        (proc,) = started_processes
        assert proc.poll() is not None
        assert "outcome: ended by WhisperJAV" in log_lines(req)

    @windows_only
    def test_each_run_is_tied_to_a_job_that_is_closed_after(self, make_request, monkeypatch):
        jobs = []

        class RecordingJob(runner._KillOnCloseJob):
            def __init__(self):
                super().__init__()
                self.assigned = None
                jobs.append(self)

            def assign(self, proc):
                self.assigned = super().assign(proc)
                return self.assigned

        monkeypatch.setattr(runner, "_KillOnCloseJob", RecordingJob)
        assert run(make_request([("json", COMPLETE_JSON)])).outcome is CrispOutcome.OK
        (job,) = jobs
        assert job.assigned is True
        assert job.handle is None

    @windows_only
    def test_closing_the_job_ends_a_program_still_running(self, started_processes):
        # What ties the program to WhisperJAV: if WhisperJAV's process ends, Windows closes the
        # handle and the program ends with it.
        job = runner._KillOnCloseJob()
        assert job.handle
        proc = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        assert job.assign(proc) is True
        job.close()
        proc.wait(timeout=10)
        assert job.handle is None

    def test_a_program_that_cannot_start_leaves_no_job_open(self, make_request, monkeypatch):
        jobs = []

        class FakeJob:
            def __init__(self):
                self.handle = "job"
                self.closed = False
                jobs.append(self)

            def assign(self, proc):
                return True

            def close(self):
                self.closed = True
                self.handle = None

        def cannot_start(*args, **kwargs):
            raise OSError(193, "%1 is not a valid Win32 application")

        req = make_request()
        monkeypatch.setattr(runner, "_KillOnCloseJob", FakeJob)
        monkeypatch.setattr(subprocess, "Popen", cannot_start)
        # The start failure reaches the pipeline as the run's own error, in plain words, with the
        # operating system's error kept as its cause.
        with pytest.raises(runner.CrispASRRunError, match="could not be started") as caught:
            run(req)
        assert isinstance(caught.value.__cause__, OSError)
        assert [job.closed for job in jobs] == [True], "the job object's handle stays open"
        assert "could not be started" in req.log_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# Exit codes in plain words
# ---------------------------------------------------------------------------------------------------

class TestExitCodeMeanings:
    @pytest.mark.parametrize("code", sorted(EXIT_MEANING))
    def test_documented_codes(self, code):
        assert exit_code_meaning(code) == EXIT_MEANING[code]
        assert not is_crash(code)

    def test_the_documented_list(self):
        assert sorted(EXIT_MEANING) == [1, 2, 10, 11, 12, 13, 14, 19, 20, 30, 31, 32]

    def test_code_4_belongs_to_the_legacy_whisper_route_only(self):
        assert exit_code_meaning(4) == "undocumented exit code 4"

    def test_a_signal(self):
        assert exit_code_meaning(-11) == "ended by signal 11"
        assert is_crash(-11)

    @pytest.mark.parametrize("status, text", [
        (0xC000001D, "crashed with status 0xC000001D (illegal instruction)"),
        (0xC0000135, "crashed with status 0xC0000135 (a DLL the program needs was not found)"),
        (0xC0000409, "crashed with status 0xC0000409 (stack buffer overrun (fail-fast))"),
        (0xC0000005, "crashed with status 0xC0000005 (access violation)"),
        (0xC00000FD, "crashed with status 0xC00000FD (stack overflow)"),
        (0xC0001234, "crashed with status 0xC0001234"),
    ])
    def test_crash_statuses(self, status, text):
        assert exit_code_meaning(status) == text
        assert is_crash(status)

    def test_an_undocumented_code(self):
        assert exit_code_meaning(77) == "undocumented exit code 77"
        assert not is_crash(77)

    def test_exit_code_0_is_not_called_undocumented(self):
        # 0 is the documented success code; CrispRunResult.meaning of an OK run returns this text.
        assert not is_crash(0)
        assert "undocumented" not in exit_code_meaning(0)
