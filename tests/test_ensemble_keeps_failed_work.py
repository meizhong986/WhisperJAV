"""A pass that does not finish keeps what it made, and the user is told (owner W462, W463 step 1, step 3, C1).

What is pinned here:
  - the pass worker writes each file's result as soon as it is known (``<result file>.partial``);
  - for a file a pass did not finish, the Ensemble keeps the pass's working folder under ``<temp>/kept/<time>_pass<N>``
    without its audio (owner W485), then copies the pipeline's last save-point (``<basename>.savepoint.srt``, its
    stage in ``.savepoint.json``) beside the video as ``<basename>.<lang>.pass<N>.unfinished.srt`` (W485), and names
    both in the file's record;
  - the words for the run summary (W485, the plainer wording) are true for every pipeline: the unfinished subtitles
    with their count and how far they got; "stopped before any of its subtitles were ready" only for a pipeline that
    writes save-points (ChronosJAV); the kept folder;
  - a pass that finished every file leaves nothing behind;
  - a failed pass 1 still has no summary (test_crispasr_notes_and_check); its words are in ``kept_work``, with pass 2's.

No pipeline runs: the worker is not started; its outputs are written by hand. Run alone:
python -m pytest tests/test_ensemble_keeps_failed_work.py
"""

import json
import pickle
from pathlib import Path

from whisperjav.ensemble.pass_worker import FileResult, _write_partial_results, partial_result_file

SRT = "1\n00:00:00,000 --> 00:00:01,000\nline\n"


def _orchestrator(tmp_path):
    from whisperjav.ensemble.orchestrator import EnsembleOrchestrator

    orch = EnsembleOrchestrator.__new__(EnsembleOrchestrator)
    orch.temp_dir = tmp_path / "temp"
    orch.output_dir = tmp_path / "out"
    orch.source_output_mode = True
    orch.temp_dir.mkdir()
    orch.output_dir.mkdir()
    return orch


def _media(tmp_path, name):
    folder = tmp_path / "videos"
    folder.mkdir(exist_ok=True)
    return {"path": str(folder / f"{name}.mp4"), "basename": name}


def test_the_worker_writes_each_files_result_as_it_finishes(tmp_path):
    result_file = str(tmp_path / "worker_result_pass2.pkl")
    _write_partial_results(result_file, [FileResult(basename="a", status="completed", srt_path="a.srt")])
    with open(partial_result_file(result_file), "rb") as f:
        carried = pickle.load(f)
    assert [r["basename"] for r in carried["results"]] == ["a"]
    assert not Path(partial_result_file(result_file) + ".tmp").exists()


def test_a_failed_chronosjav_file_keeps_its_save_point_and_the_working_files(tmp_path):
    from whisperjav.ensemble.orchestrator import _kept_work_sentences

    orch = _orchestrator(tmp_path)
    work = orch.temp_dir / "pass2_worker"
    work.mkdir()
    (work / "clip.savepoint.srt").write_text(SRT, encoding="utf-8")
    (work / "clip.savepoint.json").write_text(
        json.dumps({"stage": "after the aligner, before the timing model", "subtitles": 1}), encoding="utf-8")
    media = [_media(tmp_path, "clip")]
    results = {"clip": {"status": "failed", "error": "Worker crash: exit code 3221226505, no result file"}}

    orch._keep_failed_pass_work(2, media, results, "ja", "qwen")

    saved = Path(media[0]["path"]).parent / "clip.ja.pass2.unfinished.srt"
    assert saved.read_text(encoding="utf-8") == SRT
    entry = results["clip"]
    assert entry["savepoint_srt"] == str(saved) and entry["savepoint_subtitles"] == 1
    kept = Path(entry["kept_dir"])
    assert kept.parent == orch.temp_dir / "kept" and kept.name.endswith("_pass2")
    assert (kept / "clip.savepoint.srt").exists() and not work.exists()
    assert _kept_work_sentences(2, entry) == [
        f"pass 2's unfinished subtitles are saved as {saved} (1 subtitle, timed by the aligner; the timing model did "
        f"not run)",
        f"pass 2's working files are kept in {kept}"]


def test_a_chronosjav_file_without_a_save_point_is_said_so(tmp_path):
    from whisperjav.ensemble.orchestrator import _kept_work_sentences

    orch = _orchestrator(tmp_path)
    (orch.temp_dir / "pass2_worker").mkdir()
    results = {"clip": {"status": "failed", "error": "x"}}
    orch._keep_failed_pass_work(2, [_media(tmp_path, "clip")], results, "ja", "qwen")
    sentences = _kept_work_sentences(2, results["clip"])
    assert sentences[0] == "pass 2 stopped before any of its subtitles were ready, so none could be saved"
    assert sentences[1].startswith("pass 2's working files are kept in ")


def test_a_pipeline_without_save_points_gets_no_word_about_them(tmp_path):
    from whisperjav.ensemble.orchestrator import _kept_work_sentences

    orch = _orchestrator(tmp_path)
    (orch.temp_dir / "pass1_worker").mkdir()
    results = {"clip": {"status": "failed", "error": "x"}}
    orch._keep_failed_pass_work(1, [_media(tmp_path, "clip")], results, "ja", "moss-crispasr")
    sentences = _kept_work_sentences(1, results["clip"])
    assert len(sentences) == 1 and sentences[0].startswith("pass 1's working files are kept in ")


def test_a_pass_that_finished_every_file_leaves_nothing_behind(tmp_path):
    orch = _orchestrator(tmp_path)
    work = orch.temp_dir / "pass1_worker"
    work.mkdir()
    results = {"clip": {"status": "completed", "srt_path": "x.srt"}}
    orch._keep_failed_pass_work(1, [_media(tmp_path, "clip")], results, "ja", "qwen")
    assert work.exists() and not (orch.temp_dir / "kept").exists()
    assert "kept_dir" not in results["clip"]


def _merge(orch, tmp_path, pass1, pass2):
    return orch._process_single_file_merge(
        media_info=_media(tmp_path, "clip"), pass1_results={"clip": pass1}, pass2_results={"clip": pass2},
        pass1_config={"pipeline": "moss-crispasr"}, pass2_config={"pipeline": "qwen"},
        merge_strategy="pass1_primary", pass_languages={1: "ja", 2: "ja"})[0]


def test_the_merge_record_carries_them_for_a_failed_pass_2(tmp_path):
    orch = _orchestrator(tmp_path)
    p1 = tmp_path / "p1.srt"
    p1.write_text(SRT, encoding="utf-8")
    pass1 = {"basename": "clip", "status": "completed", "srt_path": str(p1), "subtitles": 1, "processing_time": 1.0}
    pass2 = {"basename": "clip", "status": "failed", "error": "Worker crash", "savepoint_srt": "S.srt",
             "savepoint_stage": "after the ASR", "kept_dir": "K", "save_points": True}
    metadata = _merge(orch, tmp_path, pass1, pass2)
    assert metadata["status"] == "degraded"
    assert metadata["pass2"]["savepoint_srt"] == "S.srt" and metadata["pass2"]["kept_dir"] == "K"
    assert metadata["kept_work"] == ["pass 2's unfinished subtitles are saved as S.srt (text only, timed by the VAD "
                                     "groups; the aligner did not run)",
                                     "pass 2's working files are kept in K"]


def test_a_failed_pass_1_names_both_passes_and_keeps_no_summary(tmp_path):
    orch = _orchestrator(tmp_path)
    pass1 = {"basename": "clip", "status": "failed", "error": "x", "kept_dir": "K1"}
    pass2 = {"basename": "clip", "status": "failed", "error": "y", "kept_dir": "K2"}
    metadata = _merge(orch, tmp_path, pass1, pass2)
    assert metadata["status"] == "failed" and metadata.get("summary") is None
    assert metadata["kept_work"] == ["pass 1's working files are kept in K1", "pass 2's working files are kept in K2"]


def test_main_prints_them_in_the_files_line():
    main_src = (Path(__file__).resolve().parent.parent / "whisperjav" / "main.py").read_text(encoding="utf-8")
    assert "_reasons.extend(result.get('kept_work') or [])" in main_src
    assert "_why = \"; \".join([str(_why or 'ensemble pass failed')] + list(result['kept_work']))" in main_src


def test_after_a_crash_the_finished_files_keep_their_results():
    src = (Path(__file__).resolve().parent.parent / "whisperjav" / "ensemble" / "orchestrator.py").read_text(
        encoding="utf-8")
    assert "done.get(info['basename']) or {" in src and "partial_file.unlink(missing_ok=True)" in src


def test_how_far_the_unfinished_subtitles_got_in_the_users_words():
    """W485: every stage the pipelines record has its words; the count is singular for one subtitle."""
    from whisperjav.ensemble.orchestrator import _kept_work_sentences, _stage_words

    assert _stage_words("after the ASR") == "text only, timed by the VAD groups; the aligner did not run"
    assert _stage_words("after the ASR, before the timing model") == ("timed by the VAD groups; the timing model did "
                                                                      "not run")
    assert _stage_words("after the aligner, before the timing model") == ("timed by the aligner; the timing model did "
                                                                          "not run")
    assert _stage_words("before the clean-up") == _stage_words("before the clean-up, after the timing model") \
        == "timed, but not cleaned up"
    assert _stage_words("") == _stage_words(None) == _stage_words("something new") == ""
    many = {"savepoint_srt": "S.srt", "savepoint_stage": "before the clean-up", "savepoint_subtitles": 1647}
    assert _kept_work_sentences(2, many) == ["pass 2's unfinished subtitles are saved as S.srt (1647 subtitles, timed, "
                                             "but not cleaned up)"]
    bare = {"savepoint_srt": "S.srt"}
    assert _kept_work_sentences(2, bare) == ["pass 2's unfinished subtitles are saved as S.srt"]


def test_the_chronosjav_stage_without_an_aligner_is_named_by_what_ran():
    """W485: with no aligner, the save-point before the timing model does not claim the aligner ran."""
    src = (Path(__file__).resolve().parent.parent / "whisperjav" / "modules" / "subtitle_pipeline"
           / "orchestrator.py").read_text(encoding="utf-8")
    flat = " ".join(src.split())
    assert ('stage=("after the aligner, before the timing model" if self.aligner is not None '
            'else "after the ASR, before the timing model")') in flat


def test_a_kept_folder_keeps_subtitles_and_records_but_not_the_audio(tmp_path):
    """W485: the extracted audio and the scene files are left out; subtitles, records and logs stay."""
    orch = _orchestrator(tmp_path)
    work = orch.temp_dir / "pass2_worker"
    (work / "scenes").mkdir(parents=True)
    (work / "clip_extracted.wav").write_bytes(b"RIFF" + b"\0" * 64)
    (work / "scenes" / "clip_scene_0001.wav").write_bytes(b"RIFF" + b"\0" * 64)
    (work / "scenes" / "clip_scene_0001.srt").write_text(SRT, encoding="utf-8")
    (work / "clip.savepoint.srt").write_text(SRT, encoding="utf-8")
    (work / "clip.savepoint.json").write_text(json.dumps({"stage": "after the ASR", "subtitles": 1}), encoding="utf-8")
    (work / "run.log").write_text("log", encoding="utf-8")
    media = [_media(tmp_path, "clip")]
    results = {"clip": {"status": "failed", "error": "x"}}

    orch._keep_failed_pass_work(2, media, results, "ja", "qwen")

    kept = Path(results["clip"]["kept_dir"])
    assert sorted(p.name for p in kept.rglob("*.wav")) == []
    assert (kept / "scenes" / "clip_scene_0001.srt").exists() and (kept / "clip.savepoint.srt").exists()
    assert (kept / "clip.savepoint.json").exists() and (kept / "run.log").exists()
    assert (Path(media[0]["path"]).parent / "clip.ja.pass2.unfinished.srt").read_text(encoding="utf-8") == SRT
