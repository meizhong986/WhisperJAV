"""Input files of the same name (1.10.3, owner W808 D2, W809-W811; docs: the D2 design note, version 2).

What is pinned here:
  - part 2, the intake check: a file is left out only when an earlier input would write its subtitle file into the
    same folder under the same name (name compared after NFC in lower case, every platform, W811 Q4); same-named
    files in different subfolders of one input folder both run; the warning and the run summary's reason (W811 Q2);
  - part 1: the Ensemble files each pass result under the file's path, so SeriesA\\01.mp4 and SeriesB\\01.mp4 each
    get their own merged subtitles, also when one of them fails; a failed ChronosJAV pass's save-point is named by
    the file's position in the pass; a pass pipeline writes into the pass's own folder and the worker moves each
    file's side files into the file's own folder with the pass in the name, never touching a subtitle file already
    in the user's folder (W811 Q1 (i)); the async single pass gives each file its own task;
  - the interrupted-run line counts only files that started.

No ASR runs and no model loads: pipelines and the pass worker's process are stand-ins. Run alone:
python -m pytest tests/test_same_named_inputs.py
"""

import pickle
import threading
import time
import unicodedata
from pathlib import Path

import pytest

from whisperjav.modules.media_discovery import (MediaDiscovery, left_out_lines, left_out_reason,
                                                same_subtitle_files)

SRT_A = "1\n00:00:00,000 --> 00:00:01,000\nseries A line\n"
SRT_B = "1\n00:00:00,000 --> 00:00:01,000\nseries B line\n"


def _tree(root: Path, names):
    for rel in names:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")


def _rel(root: Path, info):
    return Path(info["path"]).relative_to(root).as_posix()


# ---------------------------------------------------------------------------
# Part 2: the intake check
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("d2_tree")
    _tree(root, ["in/SeriesA/01.mp4", "in/SeriesB/01.mp4", "in/x.mkv", "in/x.mp4", "in/Ep01.mp4",
                 "in/sub/ep01.MP4", "other/01.mp4"])
    (root / "out").mkdir()
    return root


def _check(root, inputs, output_dir):
    media = MediaDiscovery().discover([str(root / i) for i in inputs])
    kept, left_out = same_subtitle_files(media, output_dir)
    return ([_rel(root, i) for i in kept],
            [(_rel(root, info), _rel(root, earlier)) for info, earlier in left_out])


def test_a_folder_input_same_name_in_two_subfolders_both_run(tree):
    kept, left_out = _check(tree, ["in"], str(tree / "out"))
    assert "in/SeriesA/01.mp4" in kept and "in/SeriesB/01.mp4" in kept
    assert "in/Ep01.mp4" in kept and "in/sub/ep01.MP4" in kept
    # x.mkv and x.mp4 write out/x.ja.whisperjav.srt: the first in the list is kept
    assert left_out == [("in/x.mp4", "in/x.mkv")]


def test_beside_each_video_the_same(tree):
    kept, left_out = _check(tree, ["in"], "source")
    assert len(kept) == 5 and left_out == [("in/x.mp4", "in/x.mkv")]


def test_files_given_one_by_one_into_one_chosen_folder(tree):
    kept, left_out = _check(tree, ["in/SeriesA/01.mp4", "in/SeriesB/01.mp4", "other/01.mp4"], str(tree / "out"))
    assert kept == ["in/SeriesA/01.mp4"]
    assert left_out == [("in/SeriesB/01.mp4", "in/SeriesA/01.mp4"), ("other/01.mp4", "in/SeriesA/01.mp4")]


def test_two_folders_added_separately(tree):
    """W811 Q3, as designed: each folder input is its own top, so into one chosen folder the second is left out;
    beside each video both run."""
    kept, left_out = _check(tree, ["in/SeriesA", "in/SeriesB"], str(tree / "out"))
    assert kept == ["in/SeriesA/01.mp4"] and left_out == [("in/SeriesB/01.mp4", "in/SeriesA/01.mp4")]
    kept, left_out = _check(tree, ["in/SeriesA", "in/SeriesB"], "source")
    assert kept == ["in/SeriesA/01.mp4", "in/SeriesB/01.mp4"] and left_out == []


def test_the_same_file_twice_is_still_removed_by_discovery(tree):
    kept, left_out = _check(tree, ["in/SeriesA", "in/SeriesA/01.mp4"], str(tree / "out"))
    assert kept == ["in/SeriesA/01.mp4"] and left_out == []


def _info(folder, name, ext=".mp4"):
    return {"path": str(Path(folder) / (name + ext)), "basename": name}


def test_names_are_compared_in_lower_case_after_nfc_on_every_platform(tmp_path):
    out = str(tmp_path / "out")
    nfd = unicodedata.normalize("NFD", "パパ")
    media = [_info(tmp_path / "a", "Ep01"), _info(tmp_path / "b", "ep01", ".mkv"),
             _info(tmp_path / "a", "パパ"), _info(tmp_path / "b", nfd, ".mkv"),
             _info(tmp_path / "a", "Straße"), _info(tmp_path / "b", "Strasse")]
    kept, left_out = same_subtitle_files(media, out)
    assert [i["basename"] for i in kept] == ["Ep01", "パパ", "Straße", "Strasse"]
    assert [(i["basename"], e["basename"]) for i, e in left_out] == [("ep01", "Ep01"), (nfd, "パパ")]


def test_the_words_the_user_sees():
    """W811 Q2: the approved warning and the run summary's reason."""
    a = {"path": r"videos\in\x.mkv", "basename": "x"}
    b = {"path": r"videos\in\x.mp4", "basename": "x"}
    c = {"path": r"videos\in\SeriesA\01.mp4", "basename": "01"}
    d = {"path": r"videos\other\01.mp4", "basename": "01"}
    assert left_out_reason(a) == r"left out: same name as videos\in\x.mkv, which writes the same subtitle file"
    assert left_out_lines([(b, a), (d, c)]) == [
        "2 input files are left out: each has the same name as an earlier input file and would write its subtitles "
        "to the same file in the same folder.",
        r"  left out: videos\in\x.mp4 (same name as videos\in\x.mkv)",
        r"  left out: videos\other\01.mp4 (same name as videos\in\SeriesA\01.mp4)",
        "To process a file that was left out, rename it, or run it on its own with another output folder. Files in "
        "different subfolders of one input folder do not clash."]
    assert left_out_lines([(b, a)])[0] == (
        "1 input file is left out: it has the same name as an earlier input file and would write its subtitles to "
        "the same file in the same folder.")


def test_main_calls_it_once_after_discovery_and_lists_the_left_out_in_the_summary():
    src = (Path(__file__).resolve().parents[1] / "whisperjav" / "main.py").read_text(encoding="utf-8")
    body = src[src.index("media_files = discovery.discover(args.input)"):]
    check = body.index("media_files, _left_out = same_subtitle_files(media_files, args.output_dir)")
    assert check < body.index('logger.info(f"Found {len(media_files)} "')
    assert "skipped_outcome(str(_info['path']), left_out_reason(_earlier))" in body
    assert src.count("same_subtitle_files(") == 1


def test_a_left_out_file_does_not_fail_the_run():
    """W811 Q5: state "skipped", which never fails a run (as --skip-existing)."""
    from whisperjav.utils.run_outcome import FileOutcome, exit_status, skipped_outcome

    rows = [skipped_outcome(r"videos\in\x.mp4", left_out_reason({"path": r"videos\in\x.mkv"})),
            FileOutcome(path=r"videos\in\x.mkv", state="done")]
    assert rows[0].state == "skipped" and exit_status(rows) == 0


def test_the_interrupted_run_line_counts_only_files_that_started():
    from whisperjav.main import _interrupted_counts
    from whisperjav.utils.run_outcome import FileOutcome, skipped_outcome

    to_run = [{"path": f"videos/in/{n}.mp4"} for n in ("a", "b", "c")]       # after the intake check
    rows = [skipped_outcome("videos/in/x.mp4", left_out_reason({"path": "videos/in/x.mkv"})),   # left out at intake
            skipped_outcome("videos/in/a.mp4"),                                              # --skip-existing
            FileOutcome(path="videos/in/b.mp4", state="done")]
    assert _interrupted_counts(rows, to_run) == (1, 2)


# ---------------------------------------------------------------------------
# Part 1: the Ensemble files each result under the file's path
# ---------------------------------------------------------------------------

def _two_series(tmp_path):
    media = []
    for series in ("SeriesA", "SeriesB"):
        folder = tmp_path / "in" / series
        folder.mkdir(parents=True, exist_ok=True)
        media.append({"path": str(folder / "01.mp4"), "basename": "01", "input_folder": str(tmp_path / "in")})
    return media


def _orchestrator(tmp_path, serial, failing=()):
    """The real EnsembleOrchestrator with the pass worker replaced: each pass writes the file's own text into its
    own pass file (in the file's own output folder, as the worker does) and answers filed by path."""
    from whisperjav.ensemble.orchestrator import EnsembleOrchestrator

    orch = EnsembleOrchestrator(output_dir=str(tmp_path / "out"), temp_dir=str(tmp_path / "temp"),
                                serial_file_processing=serial)
    orch.temp_dir.mkdir(parents=True, exist_ok=True)

    def run_pass(pass_number, media_files, pass_config, language_code, **_progress):
        out = {}
        for m in media_files:
            if Path(m["path"]).parent.name in failing:
                out[m["path"]] = {"basename": m["basename"], "path": m["path"], "status": "failed", "error": "boom"}
                continue
            srt = Path(m["output_dir"]) / f"{m['basename']}.ja.pass{pass_number}.srt"
            srt.write_text(SRT_A if "SeriesA" in m["path"] else SRT_B, encoding="utf-8")
            out[m["path"]] = {"basename": m["basename"], "path": m["path"], "status": "completed",
                              "srt_path": str(srt), "subtitles": 1, "processing_time": 1.0}
        return out

    class CopyPass1:
        def merge(self, srt1_path, srt2_path, output_path, strategy):
            Path(output_path).write_text(Path(srt1_path).read_text(encoding="utf-8"), encoding="utf-8")
            return {}

    orch._run_pass_in_subprocess = run_pass
    orch.merge_engine = CopyPass1()
    return orch


P = {"pipeline": "balanced", "sensitivity": "balanced"}


@pytest.mark.parametrize("serial", [False, True])
def test_each_file_gets_its_own_merged_subtitles(tmp_path, serial):
    """The example of W809. Before 1.10.3's fix, unticked, SeriesA's merged file held SeriesB's subtitles."""
    results = _orchestrator(tmp_path, serial).process_batch(_two_series(tmp_path), P, P, "pass1_primary")
    assert [r["status"] for r in results] == ["completed", "completed"]
    a = tmp_path / "out" / "SeriesA" / "01.ja.merged.whisperjav.srt"
    b = tmp_path / "out" / "SeriesB" / "01.ja.merged.whisperjav.srt"
    assert a.read_text(encoding="utf-8") == SRT_A and b.read_text(encoding="utf-8") == SRT_B
    assert [r["summary"]["final_output"] for r in results] == [str(a), str(b)]


@pytest.mark.parametrize("failing", ["SeriesA", "SeriesB"])
def test_a_failure_is_reported_for_the_file_that_failed(tmp_path, failing):
    results = _orchestrator(tmp_path, False, failing=(failing,)).process_batch(
        _two_series(tmp_path), P, P, "pass1_primary")
    states = {Path(r["input"]["file"]).parent.name: r["status"] for r in results}
    other = "SeriesB" if failing == "SeriesA" else "SeriesA"
    assert states == {failing: "failed", other: "completed"}
    merged = tmp_path / "out" / other / "01.ja.merged.whisperjav.srt"
    assert merged.read_text(encoding="utf-8") == (SRT_A if other == "SeriesA" else SRT_B)
    assert not (tmp_path / "out" / failing / "01.ja.merged.whisperjav.srt").exists()


def test_two_failed_chronosjav_files_of_one_name_keep_their_own_save_points(tmp_path):
    """D2 1b: the save-point of the k-th file of the pass's list is <name>.<k>.savepoint.srt."""
    from whisperjav.ensemble.orchestrator import EnsembleOrchestrator
    from whisperjav.ensemble.pass_worker import save_point_name

    orch = EnsembleOrchestrator.__new__(EnsembleOrchestrator)
    orch.temp_dir, orch.output_dir, orch.source_output_mode = tmp_path / "temp", tmp_path / "out", True
    work = orch.temp_dir / "pass1_worker"
    work.mkdir(parents=True)
    media = _two_series(tmp_path)
    for position, text in ((1, SRT_A), (2, SRT_B)):
        (work / f"{save_point_name('01', position)}.savepoint.srt").write_text(text, encoding="utf-8")
    results = {m["path"]: {"status": "failed", "error": "x"} for m in media}

    orch._keep_failed_pass_work(1, media, results, "ja", "qwen")

    a = Path(media[0]["path"]).parent / "01.ja.pass1.unfinished.srt"
    b = Path(media[1]["path"]).parent / "01.ja.pass1.unfinished.srt"
    assert a.read_text(encoding="utf-8") == SRT_A and b.read_text(encoding="utf-8") == SRT_B
    assert results[media[0]["path"]]["savepoint_srt"] == str(a)


def test_the_qwen_pipeline_writes_and_clears_its_save_point_under_the_given_name():
    src = (Path(__file__).resolve().parents[1] / "whisperjav" / "pipelines" / "qwen_pipeline.py").read_text(
        encoding="utf-8")
    assert src.count("self._save_point_stem(media_basename)") == 3
    worker = (Path(__file__).resolve().parents[1] / "whisperjav" / "ensemble" / "pass_worker.py").read_text(
        encoding="utf-8")
    assert "pipeline.save_point_name = save_point_name(basename, idx)" in worker


# ---------------------------------------------------------------------------
# Part 1: the pass works in its own folder; each file's side files go to its own folder (W811 Q1 (i))
# ---------------------------------------------------------------------------

def _run_worker(tmp_path, monkeypatch, media, failing=()):
    from whisperjav.ensemble import pass_worker as pw

    seen = {}

    class StandIn:
        """Writes what a balanced pass and a ChronosJAV pass write into their own output folder."""

        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def process(self, media_info):
            name = media_info["basename"]
            seen.setdefault("output_dirs", []).append(self.output_dir)
            text = SRT_A if "SeriesA" in str(media_info["path"]) else SRT_B
            raw = self.output_dir / "raw_subs"
            raw.mkdir(exist_ok=True)
            (raw / f"{name}_stitched.original.srt").write_text(text, encoding="utf-8")
            (raw / f"{name}.parakeet-crispasr.groups.jsonl").write_text("{}", encoding="utf-8")
            (self.output_dir / f"{name}.ja.whisperjav.analytics.json").write_text(text, encoding="utf-8")
            if Path(media_info["path"]).parent.name in failing:
                raise RuntimeError("boom")
            final = self.output_dir / f"{name}.ja.whisperjav.srt"
            final.write_text(text, encoding="utf-8")
            return {"output_files": {"final_srt": str(final)},
                    "summary": {"final_subtitles_refined": 1, "total_processing_time_seconds": 0.5}}

        def cleanup(self):
            pass

    monkeypatch.setattr(pw, "_build_pipeline", lambda *a, **kw: StandIn(kw["output_dir"]))
    captured = {}
    monkeypatch.setattr(pw, "_write_dropbox_and_exit", lambda rf, result, tracer, code: captured.update(result))
    payload = pw.WorkerPayload(pass_number=1, media_files=media, pass_config={"pipeline": "balanced"},
                               output_dir=str(tmp_path / "out"), temp_dir=str(tmp_path / "temp"),
                               keep_temp_files=True, subs_language="native", extra_kwargs={}, language_code="ja")
    pw.run_pass_worker(payload, str(tmp_path / "result.pkl"))
    return captured["results"], seen


def _with_output_dirs(tmp_path, media):
    for m in media:
        m["output_dir"] = str(tmp_path / "out" / Path(m["path"]).parent.name)
        Path(m["output_dir"]).mkdir(parents=True, exist_ok=True)
    return media


def test_side_files_go_to_each_files_own_folder_with_the_pass_and_a_users_file_is_untouched(tmp_path, monkeypatch):
    top = tmp_path / "out"
    top.mkdir()
    earlier = top / "01.ja.whisperjav.srt"          # e.g. an earlier single-pass run into the same folder
    earlier.write_text("the user's earlier subtitles", encoding="utf-8")
    media = _with_output_dirs(tmp_path, _two_series(tmp_path))

    results, seen = _run_worker(tmp_path, monkeypatch, media)

    assert [r["status"] for r in results] == ["completed", "completed"]
    assert [r["path"] for r in results] == [m["path"] for m in media]
    assert all(d == tmp_path / "temp" / "pass1_worker" / "output" for d in seen["output_dirs"])
    assert earlier.read_text(encoding="utf-8") == "the user's earlier subtitles"
    assert not (top / "raw_subs").exists() and not (top / "01.ja.whisperjav.analytics.json").exists()
    for series, text in (("SeriesA", SRT_A), ("SeriesB", SRT_B)):
        folder = top / series
        assert (folder / "01.ja.pass1.srt").read_text(encoding="utf-8") == text
        assert (folder / "raw_subs" / "01.pass1.balanced_stitched.original.srt").read_text(encoding="utf-8") == text
        assert (folder / "raw_subs" / "01.pass1.parakeet-crispasr.groups.jsonl").exists()
        assert (folder / "01.ja.pass1.analytics.json").read_text(encoding="utf-8") == text
        assert not (folder / "01.ja.whisperjav.srt").exists()


def test_a_failed_files_side_files_are_set_aside_not_taken_by_the_next(tmp_path, monkeypatch):
    media = _with_output_dirs(tmp_path, _two_series(tmp_path))
    results, _seen = _run_worker(tmp_path, monkeypatch, media, failing=("SeriesA",))
    assert [r["status"] for r in results] == ["failed", "completed"]
    aside = tmp_path / "temp" / "pass1_worker" / "output_file1"
    assert (aside / "raw_subs" / "01_stitched.original.srt").read_text(encoding="utf-8") == SRT_A
    b = tmp_path / "out" / "SeriesB"
    assert (b / "raw_subs" / "01.pass1.balanced_stitched.original.srt").read_text(encoding="utf-8") == SRT_B
    assert not (tmp_path / "out" / "SeriesA" / "raw_subs").exists()


def test_the_side_file_names():
    from whisperjav.ensemble.pass_worker import _side_file_target as t

    assert t("01_stitched.original.srt", "01", "ja", 2, "fidelity", True) == "01.pass2.fidelity_stitched.original.srt"
    assert t("01_raw.original.srt", "01", "ja", 1, "faster", True) == "01.pass1.faster_raw.original.srt"
    assert t("01.cohere-crispasr.groups.jsonl", "01", "ja", 1, "cohere-crispasr", True) == \
        "01.pass1.cohere-crispasr.groups.jsonl"
    assert t("01.pass1.qwen_stitched.original.srt", "01", "ja", 1, "qwen", True) == \
        "01.pass1.qwen_stitched.original.srt"
    assert t("01.ja.whisperjav.analytics.json", "01", "ja", 2, "qwen", False) == "01.ja.pass2.analytics.json"
    assert t("01.ja.whisperjav.srt", "01", "ja", 2, "qwen", False) is None
    assert t("01 2_stitched.original.srt", "01", "ja", 1, "balanced", True) is None


def test_the_raw_copy_step_finds_a_name_in_square_brackets(tmp_path):
    from whisperjav.pipelines.base_pipeline import own_files

    for name in ("[ABC-123] Film_stitched.original.srt", "[ABC-123] Film.artifacts.json",
                 "[ABC-123] Film 2_stitched.original.srt", "A_stitched.original.srt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert sorted(p.name for p in own_files(tmp_path, "[ABC-123] Film")) == [
        "[ABC-123] Film.artifacts.json", "[ABC-123] Film_stitched.original.srt"]


def test_the_five_whisper_pipelines_use_the_escaped_rule():
    root = Path(__file__).resolve().parents[1] / "whisperjav" / "pipelines"
    for name in ("balanced", "fast", "faster", "fidelity", "transformers"):
        src = (root / f"{name}_pipeline.py").read_text(encoding="utf-8")
        assert 'glob(f"{media_basename}*")' not in src, name
        assert "own_files(" in src, name


# ---------------------------------------------------------------------------
# Part 1: the async single pass gives each file its own task (D2 1c)
# ---------------------------------------------------------------------------

class _Quick:
    def __init__(self, **kwargs):
        pass

    def process(self, media_info):
        return {"summary": {"final_subtitles_refined": 1}, "output_files": {"final_srt": None}}

    def cleanup(self):
        pass


def test_two_files_of_one_name_queued_at_once_get_two_tasks():
    from whisperjav.utils.async_processor import AsyncPipelineManager
    from whisperjav.utils.progress_aggregator import VerbosityLevel

    manager = AsyncPipelineManager(ui_update_callback=lambda m: None, verbosity=VerbosityLevel.SUMMARY)
    processor = manager.processor
    try:
        files = [{"path": rf"videos\in\{s}\01.mp4", "basename": "01", "file_number": n, "total_files": 2}
                 for n, s in ((1, "SeriesA"), (2, "SeriesB"))]
        ids = [processor.submit_task(info, _Quick, {}) for info in files]
        assert len(set(ids)) == 2 and all(i.startswith("01_") for i in ids)
        assert [processor.get_task_status(i).media_info["path"] for i in ids] == [f["path"] for f in files]
        for i in ids:
            processor.wait_for_task(i, timeout=10)
    finally:
        manager.shutdown()
