"""
The run summary's timing sources (whisperjav/modules/pipeline_analytics.py, _compute_timing_metrics).

The decoupled ChronosJAV pipeline saves each scene's record with flat timing_* fields (SceneDiagnostics through
dataclasses.asdict, orchestrator); the summary read only a nested "timing_sources" record, so every ChronosJAV run
said "Aligner native: 0.0%" (found in tuning, owner W296 4).

Run with: python -m pytest tests/test_pipeline_analytics_timing.py -q
"""
from dataclasses import asdict

from whisperjav.modules.pipeline_analytics import (
    _compute_alignment_metrics,
    _compute_timing_metrics,
    scene_timing_sources,
)
from whisperjav.modules.subtitle_pipeline.types import SceneDiagnostics


def test_the_saved_scene_record_counts():
    record = asdict(SceneDiagnostics(timing_aligner_native=8, timing_vad_fallback=2, timing_total_segments=10))
    metrics = _compute_timing_metrics([record, record])
    assert metrics["aligner_native"] == 16 and metrics["total_segments"] == 20
    assert metrics["aligner_native_pct"] == 80.0 and metrics["vad_fallback_pct"] == 20.0


def test_the_nested_record_still_counts():
    scene = {"timing_sources": {"aligner_native": 3, "vad_fallback": 1, "interpolated": 0, "total_segments": 4}}
    assert scene_timing_sources(scene) == scene["timing_sources"]
    assert _compute_timing_metrics([scene])["aligner_native_pct"] == 75.0


def test_no_scenes_no_share():
    assert _compute_timing_metrics([])["aligner_native_pct"] == 0.0


def test_each_file_keeps_its_own_scene_records():
    """W300 2: the run summary reads every scene record in the folder; a batch shares temp_dir, so each file
    writes into its own folder (since 1.9.5 the second and later files counted the earlier files' scenes)."""
    from pathlib import Path

    src = (Path(__file__).parent.parent / "whisperjav" / "pipelines" / "qwen_pipeline.py").read_text(encoding="utf-8")
    assert 'raw_subs_dir = self.temp_dir / "raw_subs" / media_basename' in src
    assert 'for stale in raw_subs_dir.glob("scene_*_diagnostics.json"):' in src


def test_the_collapse_rate_counts_the_collapsed_scenes():
    """W300 1: the scene records carry no step-down tiers, so the rate was always 0.0%; it is now the scenes the
    collapse check flagged, out of the scenes it checked ("N/A": an aligner-free scene, not checked)."""
    scenes = [asdict(SceneDiagnostics(sentinel_status=s)) for s in ("OK", "OK", "OK", "COLLAPSED", "N/A")]
    metrics = _compute_alignment_metrics(scenes)
    assert metrics["collapse_rate"] == 0.25 and metrics["total_collapsed"] == 1
    assert _compute_alignment_metrics([asdict(SceneDiagnostics())])["collapse_rate"] == 0.0


# --- owner W501 2: no red aligner mark when the run loads no aligner (ChronosJAV's VAD Only default) -----------

def _health(timestamp_mode):
    from whisperjav.modules.pipeline_analytics import _compute_health_scorecard

    alignment = {"collapse_rate": 0.0}
    timing = _compute_timing_metrics([{"timing_sources": {"aligner_native": 0, "vad_fallback": 10,
                                                          "interpolated": 0, "total_segments": 10}}])
    subtitle = {"max_gap_sec": 3.0, "large_gaps": [], "short_subs_pct": 5.0}
    scene = {"speech_ratio": 0.5}
    return _compute_health_scorecard(alignment, timing, subtitle, scene, timestamp_mode)


def test_vad_only_says_the_aligner_is_not_used():
    lines = {i["name"]: i for i in _health("vad_only")}
    assert "Aligner native" not in lines
    assert lines["Aligner"]["value"] == "not used (VAD Only)" and lines["Aligner"]["level"] == "OFF"


def test_an_aligner_mode_still_flags_a_low_share():
    lines = {i["name"]: i for i in _health("aligner_vad_fallback")}
    assert lines["Aligner native"]["value"] == "0.0%" and lines["Aligner native"]["level"] == "RED"
    assert {i["name"]: i for i in _health(None)}["Aligner native"]["level"] == "RED"   # not known: as before


def test_the_console_line_under_vad_only(caplog):
    import logging
    from whisperjav.modules import pipeline_analytics as pa

    analytics = {"scene": {}, "alignment": {}, "subtitle": {}, "timing": {},
                 "health": [{"name": "Aligner", "value": pa.ALIGNER_NOT_USED, "level": "OFF"}]}
    with caplog.at_level(logging.INFO, logger=pa.logger.name):
        pa.print_summary(analytics, title="t")
    assert " [-] Aligner: not used (VAD Only)" in caplog.text
    assert "[!] Aligner" not in caplog.text
