#!/usr/bin/env python3
"""
Routine notes reach the run summary without changing anything, and --check reports CrispASR
(1.10.0; route B).

Notes (owner Q21, plan 2.6). The CrispASR pipelines put routine notes -- CrispASR ran on the
processor unexpectedly, VAD groups gave no text -- in summary["notes"], apart from "degradations", because any
degradation makes a file "suspect" and fails a run under --fail-on suspect. The route, and what
each class below pins:

  pipeline  -> metadata["summary"]["notes"]
  worker    -> FileResult.notes                    (survives the process)
  orchestra -> summary["notes"], both passes       (in order, each note once, used passes only)
  main.py   -> FileOutcome.notes                   -> a "note:" line under the file's row and the
                                                      manifest; the state and exit status unchanged

A CrispASR failure (plan 2.6): the worker keeps the run's or the setup's reason, prefixed, as
the file's error, and main.py shows it instead of "ensemble pass failed".

The start-up check (route B design 4.2): main() makes CrispASR, the device and each CrispASR pass's model
ready after the passes' segmenter checks and before media discovery; ensure_crispasr_ready stops
the run with status 1 and a box when they cannot be made ready.

--check (route B design 12, T3): one line for CrispASR -- its version, build and device -- and one per
model it offers; it never downloads, never loads CrispASR and is never fatal.

whisperjav.main and whisperjav.ensemble are never imported into this process (CLAUDE.md
"Testing"): the worker, orchestrator and main.py hops run in probe subprocesses with stand-ins
for the pipelines, the orchestrator and media discovery. Nothing here reads the real cache.

Run with: python -m pytest tests/test_crispasr_notes_and_check.py -q
"""

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from whisperjav.crispasr import install, manifest
from whisperjav.utils import preflight_check as pf
from whisperjav.utils.preflight_check import CheckStatus, PreflightChecker
from whisperjav.utils.run_outcome import (
    MANIFEST_NAME,
    SUMMARY_NOTE_LINE,
    FileOutcome,
    classify_output,
    exit_status,
    format_summary,
    write_manifest,
)

REPO = Path(__file__).resolve().parents[1]
CPU_NOTE = "CrispASR runs on the processor (CPU build), which is slower: no NVIDIA GPU was found."
WARN_NOTE = "Parakeet-CrispASR: the CrispASR program printed 2 warning(s); see run.log"


def _ts(seconds):
    return f"{int(seconds // 3600):02d}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d},000"


def _write_srt(path, spans):
    blocks = [f"{i}\n{_ts(start)} --> {_ts(end)}\nline {i}\n" for i, (start, end) in enumerate(spans, 1)]
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def _probe(code, *args, timeout=300):
    result = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=timeout,
                            cwd=str(REPO))
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    return result


def _printed(stdout, key):
    """The JSON value a probe printed on its line KEY=..."""
    lines = [line for line in stdout.splitlines() if line.startswith(key + "=")]
    assert lines, f"the probe printed no {key}= line:\n{stdout[-2000:]}"
    return json.loads(lines[-1][len(key) + 1:])


# ===========================================================================
# run_outcome: the field, the summary lines, the manifest
# ===========================================================================

class TestTheOutcomeCarriesNotes:
    def test_it_defaults_to_an_empty_list_of_its_own(self):
        first, second = FileOutcome(path="a.mp4", state="done"), FileOutcome(path="b.mp4", state="done")
        assert first.notes == [] and second.notes == []
        first.notes.append(CPU_NOTE)
        assert second.notes == []

    def test_notes_change_no_state_and_no_exit_status(self, tmp_path):
        srt = _write_srt(tmp_path / "a.ja.srt", [(n * 10, n * 10 + 9) for n in range(30)])
        outcome = classify_output("a.mp4", srt, 300.0)
        outcome.notes = [CPU_NOTE, WARN_NOTE]
        assert outcome.state == "done"
        assert exit_status([outcome], {"suspect", "empty"}) == 0

    def test_the_summary_prints_one_line_per_note_under_its_row(self, tmp_path):
        with_notes = FileOutcome(path="a.mp4", state="done", output=str(tmp_path / "a.srt"),
                                 subtitle_count=3, notes=[CPU_NOTE, WARN_NOTE])
        without = FileOutcome(path="b.mp4", state="done", output=str(tmp_path / "b.srt"), subtitle_count=2)
        lines = format_summary([with_notes, without], [], 0).splitlines()
        row_a = next(i for i, line in enumerate(lines) if "a.mp4" in line)
        assert lines[row_a + 1] == SUMMARY_NOTE_LINE.format(note=CPU_NOTE)
        assert lines[row_a + 2] == SUMMARY_NOTE_LINE.format(note=WARN_NOTE)
        assert "b.mp4" in lines[row_a + 3]
        assert sum("note:" in line for line in lines) == 2
        assert SUMMARY_NOTE_LINE.format(note="x") == "         note: x"   # the draft layout

    def test_the_counts_line_is_unchanged_by_notes(self):
        plain = FileOutcome(path="a.mp4", state="done")
        noted = FileOutcome(path="a.mp4", state="done", notes=[CPU_NOTE])
        count = [l for l in format_summary([plain], [], 0).splitlines() if l.startswith("done 1")]
        assert count and count == [l for l in format_summary([noted], [], 0).splitlines()
                                   if l.startswith("done 1")]

    def test_the_manifest_carries_them(self, tmp_path):
        path = write_manifest([FileOutcome(path="a.mp4", state="done", notes=[CPU_NOTE]),
                               FileOutcome(path="b.mp4", state="empty")],
                              tmp_path / MANIFEST_NAME, mode="ensemble", fail_on=[], status=0)
        files = json.loads(path.read_text(encoding="utf-8"))["files"]
        assert files[0]["notes"] == [CPU_NOTE]
        assert files[1]["notes"] == []


# ===========================================================================
# The worker and the orchestrator (probe subprocess)
# ===========================================================================

GATHER_PROBE = textwrap.dedent("""
    import json, shutil, sys
    from pathlib import Path
    from whisperjav.ensemble.orchestrator import EnsembleOrchestrator
    from whisperjav.ensemble.pass_worker import FileResult

    root = Path(sys.argv[1])
    out = {}

    a, b = FileResult(basename="a", status="completed"), FileResult(basename="b", status="completed")
    a.notes.append("x")
    out["default"] = [b.notes, FileResult(basename="c", status="completed", notes=["n"]).__dict__["notes"]]

    class Merge:
        def merge(self, srt1_path, srt2_path, output_path, strategy):
            shutil.copyfile(srt1_path, output_path)
            return {}

    orchestrator = EnsembleOrchestrator.__new__(EnsembleOrchestrator)
    orchestrator.temp_dir = root
    orchestrator.output_dir = root
    orchestrator.merge_engine = Merge()

    def done(name, notes=None, degradations=()):
        srt = root / name
        srt.write_text("1\\n00:00:00,000 --> 00:00:01,000\\nline\\n", encoding="utf-8")
        result = {"basename": "clip", "status": "completed", "srt_path": str(srt), "subtitles": 1,
                  "processing_time": 1.0, "degradations": list(degradations)}
        if notes is not None:
            result["notes"] = list(notes)
        return result

    def merge(pass1, pass2):
        md, _row = orchestrator._process_single_file_merge(
            media_info={"path": str(root / "clip.mp4"), "basename": "clip"},
            pass1_results={"clip": pass1}, pass2_results={"clip": pass2} if pass2 else {},
            pass1_config={"pipeline": "parakeet-crispasr"},
            pass2_config={"pipeline": "cohere-crispasr"} if pass2 else None,
            merge_strategy="pass1_primary", pass_languages={1: "ja", 2: "ja"})
        return {"status": md["status"], "summary": md.get("summary")}

    out["both"] = merge(done("p1.srt", ["A", "B"]), done("p2.srt", ["B", "C"], ["d2"]))
    out["failed_pass2"] = merge(done("p1.srt", ["A"]),
                                {"basename": "clip", "status": "failed", "error": "boom",
                                 "processing_time": 0.0, "notes": ["X"]})
    out["older_result"] = merge(done("p1.srt"), None)
    out["failed_pass1"] = merge({"basename": "clip", "status": "failed", "error": "boom",
                                 "notes": ["Y"]}, None)
    print("RESULT=" + json.dumps(out))
""")


@pytest.mark.slow
class TestTheEnsembleCarriesNotes:
    @pytest.fixture(scope="class")
    def gathered(self, tmp_path_factory):
        result = _probe(GATHER_PROBE, tmp_path_factory.mktemp("gather"))
        line = [l for l in result.stdout.splitlines() if l.startswith("RESULT=")][0]
        return json.loads(line[len("RESULT="):])

    def test_the_worker_result_has_its_own_list(self, gathered):
        assert gathered["default"] == [[], ["n"]]

    def test_both_passes_in_order_each_note_once(self, gathered):
        both = gathered["both"]
        assert both["status"] == "completed"
        assert both["summary"]["notes"] == ["A", "B", "C"]
        # Degradations keep their own route and their pass labels.
        assert both["summary"]["degradations"] == ["pass 2: d2"]

    def test_a_failed_pass_contributes_no_notes(self, gathered):
        assert gathered["failed_pass2"]["summary"]["notes"] == ["A"]

    def test_an_older_result_without_the_field_does_not_crash(self, gathered):
        assert gathered["older_result"]["summary"]["notes"] == []

    def test_a_failed_file_has_no_summary_to_carry_them(self, gathered):
        # main.py reads notes from the summary; a failed pass 1 leaves none, so none are shown.
        assert gathered["failed_pass1"]["status"] == "failed"
        assert gathered["failed_pass1"]["summary"] is None


MAIN_PROBE = textwrap.dedent("""
    import json, sys
    from pathlib import Path

    root, notes = Path(sys.argv[1]), json.loads(sys.argv[2])
    media = root / "media" / "a.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"x")
    out = root / "out"
    out.mkdir()
    srt = out / "a.ja.merged.whisperjav.srt"
    srt.write_text("\\n".join(
        f"{i + 1}\\n00:{i:02d}:00,000 --> 00:{i:02d}:09,000\\nline {i + 1}\\n" for i in range(60)),
        encoding="utf-8")
    sys.argv = ["whisperjav", str(media), "--ensemble", "--pass1-pipeline", "parakeet-crispasr",
                "--output-dir", str(out), "--temp-dir", str(root / "tmp"), "--accept-cpu-mode",
                "--no-signature", "--no-progress", "--fail-on", "suspect"]

    import whisperjav.main as M
    import whisperjav.ensemble.orchestrator as O

    class Discovery:
        def discover(self, inputs):
            return [{"path": str(media), "basename": "a", "type": "video", "duration": 3600.0}]

    class Orchestrator:
        def __init__(self, **kwargs):
            pass
        def process_batch(self, media_files, pass1_config, pass2_config, merge_strategy):
            return [{"input": {"file": m["path"], "basename": m["basename"]}, "status": "completed",
                     "summary": {"final_output": str(srt), "total_processing_time_seconds": 1.0,
                                 "degradations": [], "notes": notes}} for m in media_files]

    # The start-up checks: no segmenter package, no download, no CrispASR program (they are tested apart).
    ready = []
    M.ensure_segmenter_backend_available = lambda backend, **kwargs: True
    M.ensure_segmenter_model_available = lambda backend, **kwargs: True
    M.ensure_crispasr_ready = lambda passes, **kwargs: ready.append([list(p) for p in passes]) or True
    M.MediaDiscovery = Discovery
    O.EnsembleOrchestrator = Orchestrator
    try:
        M.main()
        code = 0
    except SystemExit as e:
        code = e.code or 0
    print("READY=" + json.dumps(ready))
    print("EXIT=" + str(code))
""")


@pytest.mark.slow
class TestMainPrintsThemAndChangesNothing:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("main")
        result = _probe(MAIN_PROBE, root, json.dumps([CPU_NOTE, WARN_NOTE]))
        return root, result.stdout

    def test_the_run_still_exits_0_under_fail_on_suspect(self, run):
        _root, stdout = run
        assert "EXIT=0" in stdout, stdout[-2000:]

    def test_the_start_up_check_made_the_crispasr_pass_ready_first(self, run):
        _root, stdout = run
        assert _printed(stdout, "READY") == [[[1, "parakeet-crispasr", None]]]

    def test_the_notes_are_printed_under_the_files_row(self, run):
        _root, stdout = run
        lines = stdout.splitlines()
        row = next(i for i, line in enumerate(lines)
                   if line.startswith("done") and "a.mp4 -> a.ja.merged.whisperjav.srt" in line)
        assert lines[row + 1] == SUMMARY_NOTE_LINE.format(note=CPU_NOTE)
        assert lines[row + 2] == SUMMARY_NOTE_LINE.format(note=WARN_NOTE)

    def test_the_manifest_has_them_and_the_file_is_done(self, run):
        root, _stdout = run
        data = json.loads((root / "out" / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert data["exit_status"] == 0
        assert data["files"][0]["state"] == "done"
        assert data["files"][0]["notes"] == [CPU_NOTE, WARN_NOTE]


# The worker turns a CrispASR failure into a plain, prefixed reason (not a traceback).
WORKER_FAIL_PROBE = textwrap.dedent("""
    import pickle, sys
    from pathlib import Path
    import whisperjav.ensemble.pass_worker as pw
    from whisperjav.crispasr.errors import CrispASRRunError, CrispASRSetupError

    root = Path(sys.argv[1])
    media = []
    for name in ("a", "b", "c"):
        p = root / (name + ".mp4")
        p.write_bytes(b"x")
        media.append({"path": str(p), "basename": name, "type": "video", "duration": 60.0})

    def process(self, media_info):
        name = media_info["basename"]
        if name == "a":
            raise CrispASRRunError("the CrispASR program failed: model could not be resolved")
        if name == "b":
            raise CrispASRSetupError("This processor lacks AVX2, which CrispASR's program needs.")
        raise RuntimeError("an ordinary bug")

    pw.ParakeetCrispASRPipeline.process = process
    payload = pw.WorkerPayload(pass_number=1, media_files=media,
                               pass_config={"pipeline": "parakeet-crispasr", "language": "ja"},
                               output_dir=str(root / "out"), temp_dir=str(root / "tmp"),
                               keep_temp_files=False, subs_language="native", extra_kwargs={},
                               language_code="ja")
    pw.run_pass_worker(payload, str(root / "dropbox.pkl"))
""")


@pytest.mark.slow
class TestTheWorkerKeepsTheCrispASRReason:
    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory):
        import pickle

        root = tmp_path_factory.mktemp("worker_fail")
        subprocess.run([sys.executable, "-c", WORKER_FAIL_PROBE, str(root)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300, cwd=str(REPO))
        with open(root / "dropbox.pkl", "rb") as fh:
            data = pickle.load(fh)
        return {r["basename"]: r for r in data["results"]}

    def test_a_program_failure_is_kept_as_its_plain_reason(self, results):
        assert results["a"]["status"] == "failed"
        assert results["a"]["error"] == "CrispASR: the CrispASR program failed: model could not be resolved"

    def test_a_setup_failure_is_kept_as_its_plain_reason(self, results):
        assert results["b"]["error"] == "CrispASR: This processor lacks AVX2, which CrispASR's program needs."

    def test_any_other_error_is_still_a_traceback(self, results):
        assert results["c"]["status"] == "failed"
        assert results["c"]["error"].startswith("Traceback")


# The worker's success path: the real run_pass_worker with the real ParakeetCrispASRPipeline, whose
# process() alone is replaced (it writes the final SRT and returns the summary a CrispASR run returns).
WORKER_OK_PROBE = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    import whisperjav.ensemble.pass_worker as pw

    root, summaries = Path(sys.argv[1]), json.loads(sys.argv[2])
    media = []
    for name in summaries:
        p = root / (name + ".mp4")
        p.write_bytes(b"x")
        media.append({"path": str(p), "basename": name, "type": "video", "duration": 60.0})

    def process(self, media_info):
        name = media_info["basename"]
        final = self.output_dir / (name + ".ja.whisperjav.srt")
        final.write_text("1\\n00:00:01,000 --> 00:00:02,000\\nline one\\n\\n"
                         "2\\n00:00:03,000 --> 00:00:04,000\\nline two\\n", encoding="utf-8")
        return {"output_files": {"final_srt": str(final)},
                "summary": dict(summaries[name], final_subtitles_refined=2, total_processing_time_seconds=1.5)}

    pw.ParakeetCrispASRPipeline.process = process
    payload = pw.WorkerPayload(pass_number=1, media_files=media,
                               pass_config={"pipeline": "parakeet-crispasr", "language": "ja"},
                               output_dir=str(root / "out"), temp_dir=str(root / "tmp"),
                               keep_temp_files=False, subs_language="native", extra_kwargs={},
                               language_code="ja")
    pw.run_pass_worker(payload, str(root / "dropbox.pkl"))
""")
OK_NOTE = "Parakeet-CrispASR: 1 of 4 slice(s) gave no text"
OK_SUMMARIES = {
    "a": {"degradations": [], "notes": [CPU_NOTE, OK_NOTE]},
    "b": {"degradations": ["Parakeet-CrispASR: 1 segment(s) of the CrispASR output lay outside every slice"],
          "notes": []},
}


@pytest.mark.slow
class TestTheWorkerCarriesACrispASRFilesNotes:
    @pytest.fixture(scope="class")
    def worker(self, tmp_path_factory):
        import pickle

        root = tmp_path_factory.mktemp("worker_ok")
        done = subprocess.run([sys.executable, "-c", WORKER_OK_PROBE, str(root), json.dumps(OK_SUMMARIES)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                              cwd=str(REPO))
        assert done.returncode == 0, (done.stdout + done.stderr)[-2000:]
        with open(root / "dropbox.pkl", "rb") as fh:
            data = pickle.load(fh)
        return root, data

    def test_every_file_completed_with_its_pass_srt(self, worker):
        root, data = worker
        assert data["worker_error"] is None
        results = {r["basename"]: r for r in data["results"]}
        assert sorted(results) == ["a", "b"]
        for name, r in results.items():
            assert (r["status"], r["error"], r["subtitles"], r["processing_time"]) == ("completed", None, 2, 1.5)
            assert r["srt_path"] == str(root / "out" / f"{name}.ja.pass1.srt")
            assert Path(r["srt_path"]).is_file()
            assert not (root / "out" / f"{name}.ja.whisperjav.srt").exists()  # moved, not copied

    def test_the_notes_survive_the_process_and_stay_apart_from_the_degradations(self, worker):
        _root, data = worker
        results = {r["basename"]: r for r in data["results"]}
        assert results["a"]["notes"] == [CPU_NOTE, OK_NOTE] and results["a"]["degradations"] == []
        assert results["b"]["notes"] == [] and results["b"]["degradations"] == OK_SUMMARIES["b"]["degradations"]


# A failed CrispASR program or setup reaches the summary in plain words (plan 2.6): the pass worker
# prefixes the reason, main.py shows it instead of "ensemble pass failed".
P1_REASON = "CrispASR: the CrispASR program failed: crashed with status 0xC0000409 (stack buffer overrun (fail-fast))"
P2_REASON = "CrispASR: The model Cohere ASR ja (q8_0) is not downloaded yet, and --offline allows no download."

MAIN_FAIL_PROBE = textwrap.dedent("""
    import json, sys
    from pathlib import Path

    root, p1_reason, p2_reason = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    folder = root / "media"
    folder.mkdir(parents=True)
    for name in ("a", "b", "c"):
        (folder / (name + ".mp4")).write_bytes(b"x")
    out = root / "out"
    out.mkdir()
    srt = out / "b.ja.pass1.srt"
    srt.write_text("\\n".join(
        f"{i + 1}\\n00:{i:02d}:00,000 --> 00:{i:02d}:09,000\\nline {i + 1}\\n" for i in range(60)),
        encoding="utf-8")
    sys.argv = ["whisperjav", str(folder), "--ensemble", "--pass1-pipeline", "parakeet-crispasr",
                "--pass2-pipeline", "cohere-crispasr", "--pass2-model", "cohere-asr-ja",
                "--output-dir", str(out), "--temp-dir", str(root / "tmp"), "--accept-cpu-mode",
                "--no-signature", "--no-progress"]

    import whisperjav.main as M
    import whisperjav.ensemble.orchestrator as O

    class Discovery:
        def discover(self, inputs):
            return [{"path": str(folder / (n + ".mp4")), "basename": n, "type": "video", "duration": 3600.0}
                    for n in ("a", "b", "c")]

    class Orchestrator:
        def __init__(self, **kwargs):
            pass
        def process_batch(self, media_files, pass1_config, pass2_config, merge_strategy):
            a, b, c = media_files
            return [
                {"input": {"file": a["path"], "basename": "a"}, "status": "failed",
                 "pass1": {"status": "failed", "error": p1_reason}, "summary": {}},
                {"input": {"file": b["path"], "basename": "b"}, "status": "degraded",
                 "pass1": {"status": "completed"}, "pass2": {"status": "failed", "error": p2_reason},
                 "summary": {"final_output": str(srt), "total_processing_time_seconds": 1.0,
                             "degradations": [], "notes": []}},
                {"input": {"file": c["path"], "basename": "c"}, "status": "failed",
                 "pass1": {"status": "failed", "error": "Traceback (most recent call last):\\n  boom"},
                 "summary": {}},
            ]

    # The start-up checks: no segmenter package, no download, no CrispASR program (they are tested apart).
    ready = []
    M.ensure_segmenter_backend_available = lambda backend, **kwargs: True
    M.ensure_segmenter_model_available = lambda backend, **kwargs: True
    M.ensure_crispasr_ready = lambda passes, **kwargs: ready.append([list(p) for p in passes]) or True
    M.MediaDiscovery = Discovery
    O.EnsembleOrchestrator = Orchestrator
    try:
        M.main()
        code = 0
    except SystemExit as e:
        code = e.code or 0
    print("READY=" + json.dumps(ready))
    print("EXIT=" + str(code))
""")


@pytest.mark.slow
class TestAFailedCrispASRPassNamesItsReason:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("fail")
        result = _probe(MAIN_FAIL_PROBE, root, P1_REASON, P2_REASON)
        data = json.loads((root / "out" / MANIFEST_NAME).read_text(encoding="utf-8"))
        return result.stdout, {Path(f["path"]).stem: f for f in data["files"]}

    def test_the_start_up_check_was_asked_for_both_passes_with_their_models(self, run):
        stdout, _files = run
        assert _printed(stdout, "READY") == [[[1, "parakeet-crispasr", None], [2, "cohere-crispasr", "cohere-asr-ja"]]]

    def test_a_failed_pass_1_shows_the_crispasr_reason(self, run):
        stdout, files = run
        assert files["a"]["state"] == "failed"
        assert P1_REASON in json.dumps(files["a"], ensure_ascii=False)
        assert P1_REASON in stdout

    def test_a_failed_pass_2_names_its_reason_beside_the_fallback(self, run):
        stdout, files = run
        assert f"pass 2 failed ({P2_REASON}); output is pass 1 alone" in json.dumps(files["b"], ensure_ascii=False)
        assert f"pass 2 failed ({P2_REASON})" in stdout

    def test_a_traceback_still_stays_out_of_the_summary(self, run):
        stdout, files = run
        assert files["c"]["state"] == "failed"
        assert "Traceback" not in json.dumps(files["c"])
        assert "ensemble pass failed" in json.dumps(files["c"])


# ===========================================================================
# The start-up check: CrispASR made ready before any audio is read (plan 1.5, 2.6)
# ===========================================================================

# main() with one CrispASR pass; the start-up check stands in, records what it was asked and stops the
# run as the real one does when a file cannot be made ready.
STARTUP_PROBE = textwrap.dedent("""
    import json, sys
    from pathlib import Path

    root = Path(sys.argv[1])
    media = root / "media" / "a.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"x")
    sys.argv = ["whisperjav", str(media), "--ensemble", "--pass1-pipeline", "parakeet-crispasr",
                "--output-dir", str(root / "out"), "--temp-dir", str(root / "tmp"), "--accept-cpu-mode",
                "--no-signature", "--no-progress"]

    import whisperjav.main as M

    order = []

    def segmenter_package(backend, **kwargs):
        order.append(["segmenter package", backend])
        return True

    def segmenter_model(backend, **kwargs):
        order.append(["segmenter model", backend])
        return True

    def crispasr_ready(passes, **kwargs):
        order.append(["crispasr", [list(p) for p in passes]])
        raise SystemExit(1)

    class Discovery:
        def discover(self, inputs):
            order.append(["media discovery"])
            return []

    M.ensure_segmenter_backend_available = segmenter_package
    M.ensure_segmenter_model_available = segmenter_model
    M.ensure_crispasr_ready = crispasr_ready
    M.MediaDiscovery = Discovery
    try:
        M.main()
        code = 0
    except SystemExit as e:
        code = e.code
    print("RESULT=" + json.dumps({"exit": code, "order": order}))
""")


@pytest.mark.slow
class TestMainMakesCrispASRReadyAtStartUp:
    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        return _printed(_probe(STARTUP_PROBE, tmp_path_factory.mktemp("startup")).stdout, "RESULT")

    def test_the_crispasr_pass_is_made_ready_after_its_segmenter_is_checked(self, result):
        assert result["order"][:3] == [["segmenter package", "whisperseg"], ["segmenter model", "whisperseg"],
                                       ["crispasr", [[1, "parakeet-crispasr", None]]]]

    def test_a_failure_there_stops_the_run_before_media_discovery(self, result):
        assert result["exit"] == 1
        assert ["media discovery"] not in result["order"]
        assert len(result["order"]) == 3


class TestEnsureCrispASRReady:
    """utils/preflight_check.ensure_crispasr_ready with install's own steps replaced (no download, no CrispASR)."""

    BOTH = [(1, "parakeet-crispasr", None), (2, "cohere-crispasr", "cohere-asr-ja")]
    REASON = "Downloading the model Cohere ASR ja (q8_0) failed: stand-in connection refused"

    @pytest.fixture
    def steps(self, monkeypatch):
        state = SimpleNamespace(asked=[], fail_on=None, offline=False, device=None, stage_roots=[])

        def forbidden(*args, **kwargs):
            raise AssertionError("the start-up check's test must not download or load anything")

        for name in ("_hf_download", "available_backends"):
            monkeypatch.setattr(install, name, forbidden)

        def prepare_runtime(device, model):
            state.asked.append(f"runtime {model.name}")
            state.device = device
            if state.fail_on == "runtime":
                raise install.CrispASRSetupError(self.REASON)
            return install.RuntimeChoice(True, "cuda", None, False, "0.8.44+cuda", "cuda")

        def ensure_model(model, offline=None):
            state.asked.append(model.name)
            if state.fail_on == model.name:
                raise install.CrispASRSetupError(self.REASON)
            return Path("model.gguf")

        def stage_model(path, temp_dir, sha256):
            state.asked.append("stage")
            state.stage_roots.append(Path(temp_dir))
            return Path(path)

        monkeypatch.setattr(install, "prepare_runtime", prepare_runtime)
        monkeypatch.setattr(install, "ensure_model", ensure_model)
        monkeypatch.setattr(install, "stage_model", stage_model)
        monkeypatch.setattr(install, "hub_offline", lambda: state.offline)
        return state

    @staticmethod
    def _box(text):
        """The printed box without its colour codes, one string per line."""
        return re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines()

    def test_each_pass_runtime_then_model_then_staging_and_nothing_printed(self, steps, capsys, tmp_path):
        assert pf.ensure_crispasr_ready(self.BOTH, device="cuda", temp_dir=str(tmp_path)) is True
        assert steps.asked == ["runtime parakeet-tdt-0.6b-ja", "parakeet-tdt-0.6b-ja", "stage",
                               "runtime cohere-asr-ja", "cohere-asr-ja", "stage"]
        assert steps.device == "cuda" and steps.stage_roots == [tmp_path, tmp_path]
        assert capsys.readouterr().out == ""

    def test_without_a_temp_folder_the_systems_whisperjav_folder_is_used(self, steps):
        import tempfile
        pf.ensure_crispasr_ready(self.BOTH[:1])
        assert steps.stage_roots == [Path(tempfile.gettempdir()) / "whisperjav"]

    @pytest.mark.parametrize("fail_on, asked", [("runtime", ["runtime parakeet-tdt-0.6b-ja"]),
                                                ("cohere-asr-ja", ["runtime parakeet-tdt-0.6b-ja",
                                                                   "parakeet-tdt-0.6b-ja", "stage",
                                                                   "runtime cohere-asr-ja", "cohere-asr-ja"])])
    def test_a_pass_that_cannot_be_made_ready_stops_the_run_with_status_1(self, steps, capsys, fail_on, asked):
        steps.fail_on = fail_on
        with pytest.raises(SystemExit) as stop:
            pf.ensure_crispasr_ready(self.BOTH)
        assert stop.value.code == 1
        assert steps.asked == asked
        text = " ".join(" ".join(line.strip("║ ") for line in self._box(capsys.readouterr().out)).split())
        assert "CrispASR is not ready for this run" in text
        assert ("This run uses parakeet-crispasr (pass 1), cohere-crispasr (pass 2). WhisperJAV checks CrispASR "
                "and downloads its model the first time it is needed. That did not work:") in text   # T4
        assert self.REASON in text
        assert "Nothing has been transcribed." in text
        assert "--offline" not in text

    def test_under_offline_the_box_says_downloads_are_switched_off(self, steps, capsys):
        steps.fail_on, steps.offline = "runtime", True
        with pytest.raises(SystemExit):
            pf.ensure_crispasr_ready(self.BOTH[:1])
        text = " ".join(" ".join(line.strip("║ ") for line in self._box(capsys.readouterr().out)).split())
        assert "Downloads are switched off for this run (--offline)" in text
        assert "Run once without --offline first." in text

    def test_without_exit_on_fail_it_answers_false_and_prints_nothing(self, steps, capsys):
        steps.fail_on = "runtime"
        assert pf.ensure_crispasr_ready(self.BOTH, exit_on_fail=False) is False
        assert capsys.readouterr().out == ""

    def test_any_other_error_is_not_turned_into_the_box(self, steps, monkeypatch):
        # Only a setup failure is the user's to act on; anything else stays a bug with its traceback.
        def broken(device, model):
            raise RuntimeError("stand-in bug")

        monkeypatch.setattr(install, "prepare_runtime", broken)
        with pytest.raises(RuntimeError, match="stand-in bug"):
            pf.ensure_crispasr_ready(self.BOTH)

    @pytest.mark.parametrize("passes", [BOTH[:1], BOTH], ids=["one pass", "two passes"])
    def test_the_box_keeps_its_frame(self, steps, capsys, passes):
        steps.fail_on, steps.offline = "runtime", True
        with pytest.raises(SystemExit):
            pf.ensure_crispasr_ready(passes)
        box = [line for line in self._box(capsys.readouterr().out) if line]
        assert box[0].startswith("╔") and box[-1].startswith("╚")
        too_wide = [line for line in box if len(line) != pf.BOX_WIDTH]
        assert not too_wide, "lines that break the box's right edge:\n" + "\n".join(too_wide)


# ===========================================================================
# --check (in process; install's checks replaced; nothing downloads and CrispASR is never loaded)
# ===========================================================================

MODEL_NAMES = list(dict.fromkeys(n for names in manifest.PIPELINE_MODELS.values() for n in names))


@pytest.fixture
def crispasr_pc(monkeypatch):
    """A Windows PC with the CUDA build and an NVIDIA GPU, as --check sees it, set up per test."""
    state = {"supported": True, "installed": True, "build": ("0.8.44+cuda", "cuda"), "avx2": True,
             "best": ("cuda", None), "models": set(), "model_error": None, "build_error": None}

    def forbidden(*args, **kwargs):
        raise AssertionError("--check must not download anything or load CrispASR")

    for name in ("ensure_model", "prepare_runtime", "available_backends", "_hf_download", "stage_model"):
        monkeypatch.setattr(install, name, forbidden)
    monkeypatch.setattr(install, "platform_supported", lambda *a: state["supported"])
    monkeypatch.setattr(install, "crispasr_installed", lambda: state["installed"])
    monkeypatch.setattr(install, "has_avx2", lambda: state["avx2"])

    def installed_build():
        if state["build_error"] is not None:
            raise state["build_error"]
        return state["build"]

    monkeypatch.setattr(install, "installed_build", installed_build)
    monkeypatch.setattr(install, "_best_device", lambda: state["best"])

    def model_status(model):
        if model.name == state["model_error"]:
            raise OSError("cache unreadable")
        return install.ModelStatus(model, model.name in state["models"])

    monkeypatch.setattr(install, "model_status", model_status)
    return state


def _crispasr_results():
    checker = PreflightChecker(verbose=False)
    checker._check_crispasr()
    return checker.results


def _crispasr(results):
    lines = [r for r in results if r.name == pf.CRISPASR_NAME]
    assert len(lines) == 1, results
    return lines[0]


class TestCheckReportsCrispASR:
    """T3 (route B design 12)."""

    # T46 (design 22.10).
    def test_the_cuda_build_on_the_nvidia_gpu_is_pass(self, crispasr_pc):
        line = _crispasr(_crispasr_results())
        assert line.status == CheckStatus.PASS and not line.fatal
        assert line.message == "0.8.44+cuda (CUDA build), GGUF pipelines: asks for the NVIDIA GPU (CUDA)"

    def test_the_processor_with_its_reason(self, crispasr_pc):
        crispasr_pc["best"] = ("cpu", None)
        line = _crispasr(_crispasr_results())
        assert line.status == CheckStatus.PASS
        assert line.message == ("0.8.44+cuda (CUDA build), GGUF pipelines: CPU (not accelerated): "
                                "no NVIDIA GPU was found")

    def test_the_cpu_build(self, crispasr_pc, monkeypatch):
        crispasr_pc["build"] = ("0.8.44", "cpu")
        monkeypatch.setattr(install.sys, "platform", "win32")
        line = _crispasr(_crispasr_results())
        assert line.message == ("0.8.44 (CPU build), GGUF pipelines: CPU (not accelerated): "
                                "this install has CrispASR's CPU build")

    def test_the_vulkan_build(self, crispasr_pc, monkeypatch):
        crispasr_pc["build"] = ("0.8.44+vulkan", "vulkan")
        monkeypatch.setattr(install.sys, "platform", "win32")
        line = _crispasr(_crispasr_results())
        assert line.message == ("0.8.44+vulkan (Vulkan build), GGUF pipelines: asks for the GPU through Vulkan "
                                "(experimental, not tested)")

    def test_not_installed_is_a_warning(self, crispasr_pc, monkeypatch):
        crispasr_pc["installed"] = False
        line = _crispasr(_crispasr_results())
        assert line.status == CheckStatus.WARN and not line.fatal
        assert line.message == ("not installed; the CrispASR pipelines cannot run here. Run the WhisperJAV "
                                "installer again.")

    def test_a_processor_without_avx2_is_a_warning(self, crispasr_pc):
        crispasr_pc["avx2"] = False
        line = _crispasr(_crispasr_results())
        assert line.status == CheckStatus.WARN and not line.fatal
        assert line.message == pf.CRISPASR_NO_AVX2_TEXT

    def test_an_unknown_avx2_answer_is_not_a_stop(self, crispasr_pc):
        crispasr_pc["avx2"] = None
        assert _crispasr(_crispasr_results()).status == CheckStatus.PASS

    def test_a_computer_with_no_build_is_a_warning(self, crispasr_pc):
        crispasr_pc["supported"] = False
        line = _crispasr(_crispasr_results())
        assert line.status == CheckStatus.WARN and not line.fatal
        assert line.message.startswith("no CrispASR build for this computer")

    def test_a_failing_check_is_a_warning_not_a_crash(self, crispasr_pc):
        crispasr_pc["build_error"] = OSError("metadata unreadable")
        results = _crispasr_results()
        line = _crispasr(results)
        assert line.status == CheckStatus.WARN and "metadata unreadable" in line.message
        assert len(results) == 1 + len(MODEL_NAMES)          # the models are still listed

    def test_one_line_per_model(self, crispasr_pc):
        crispasr_pc["models"] = {MODEL_NAMES[0]}
        models = [r for r in _crispasr_results() if r.name != pf.CRISPASR_NAME]
        assert [r.name for r in models] == [f"CrispASR model {n}" for n in MODEL_NAMES]
        assert models[0].status == CheckStatus.PASS and models[0].message.endswith(": downloaded")
        for line, name in zip(models[1:], MODEL_NAMES[1:]):
            model = manifest.MODELS[name]
            assert line.status == CheckStatus.INFO and not line.fatal
            assert line.message.endswith(f": not downloaded ({install.mb(model.size)})")
            assert model.label in line.message

    def test_the_ctc_model_has_no_line(self, crispasr_pc):
        names = [r.name for r in _crispasr_results()]
        assert pf.CRISPASR_MODEL_NAME.format(model="parakeet-ctc-1.1b-ja") not in names

    def test_a_failing_model_check_is_a_warning_on_that_line_only(self, crispasr_pc):
        crispasr_pc["model_error"] = MODEL_NAMES[1]
        models = [r for r in _crispasr_results() if r.name != pf.CRISPASR_NAME]
        assert [r.name for r in models] == [f"CrispASR model {n}" for n in MODEL_NAMES]
        assert models[1].status == CheckStatus.WARN and not models[1].fatal
        assert models[1].message == pf.CRISPASR_CHECK_FAILED.format(error="cache unreadable")
        assert all(r.status == CheckStatus.INFO for i, r in enumerate(models) if i != 1)

    def test_it_is_part_of_check_and_never_fatal(self, crispasr_pc, monkeypatch):
        for name in dir(PreflightChecker):
            if name.startswith("_check_") and name != "_check_crispasr":
                monkeypatch.setattr(PreflightChecker, name, lambda self: None)
        monkeypatch.setattr(PreflightChecker, "_display_results", lambda self: None)
        crispasr_pc["avx2"] = False                          # the worst case: a warning
        checker = PreflightChecker(verbose=False)
        assert checker.run_all_checks() is True
        assert any(r.name == pf.CRISPASR_NAME for r in checker.results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
