"""ChronosJAV keeps the joined subtitles before its clean-up (requirement R-RAW, owner W334 D1, W514 6), as the GGUF
pipelines do: raw_subs/<name>.qwen_stitched.original.srt, in an Ensemble pass <name>.pass<N>.qwen_stitched.original.srt
beside the pass's file."""

from pathlib import Path
from types import SimpleNamespace

from whisperjav.ensemble import pass_worker
from whisperjav.pipelines.qwen_pipeline import QwenPipeline

REPO = Path(__file__).resolve().parents[1]
SRT = "1\n00:00:01,000 --> 00:00:02,000\nあっ\n\n2\n00:00:03,000 --> 00:00:04,000\nこんにちは\n"


def _keep(tmp_path, raw_subs_dir=None, pass_tag=None):
    stitched = tmp_path / "temp" / "film_stitched.srt"
    stitched.parent.mkdir()
    stitched.write_text(SRT, encoding="utf-8")
    fake = SimpleNamespace(raw_subs_dir=raw_subs_dir, pass_tag=pass_tag, output_dir=tmp_path / "out")
    return QwenPipeline._keep_raw_copy(fake, stitched, "film")


def test_a_single_pass_run_keeps_it_in_the_output_folders_raw_subs(tmp_path):
    copy = _keep(tmp_path)
    assert copy == tmp_path / "out" / "raw_subs" / "film.qwen_stitched.original.srt"
    assert copy.read_text(encoding="utf-8") == SRT


def test_an_ensemble_pass_names_the_pass_and_uses_its_folder(tmp_path):
    copy = _keep(tmp_path, raw_subs_dir=str(tmp_path / "pass_out" / "raw_subs"), pass_tag="pass2")
    assert copy == tmp_path / "pass_out" / "raw_subs" / "film.pass2.qwen_stitched.original.srt"


def test_the_pass_worker_points_chronosjav_at_its_folder(tmp_path):
    pipeline = SimpleNamespace(raw_subs_dir=None, pass_tag=None)
    pass_worker._configure_trace(pipeline, 1, tmp_path)
    assert pipeline.raw_subs_dir == str(tmp_path / "raw_subs") and pipeline.pass_tag == "pass1"
    source = (REPO / "whisperjav" / "pipelines" / "qwen_pipeline.py").read_text(encoding="utf-8")
    assert "self.raw_subs_dir: Optional[str] = None" in source and "self.pass_tag: Optional[str] = None" in source


def test_the_copy_is_made_before_phase_8(tmp_path):
    source = (REPO / "whisperjav" / "pipelines" / "qwen_pipeline.py").read_text(encoding="utf-8")
    copy_at = source.index("self._keep_raw_copy(stitched_srt_path, media_basename)")
    assert source.index("Phase 7: Stitched %d subtitles") < copy_at < source.index("# PHASE 8: SANITISATION")


def test_a_copy_that_cannot_be_written_does_not_stop_the_run(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    assert _keep(tmp_path, raw_subs_dir=str(blocker / "raw_subs")) is None
