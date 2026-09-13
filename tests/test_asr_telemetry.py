"""Tests for per-scene ASR telemetry (#394 diagnostics).

The point of this module is to make the *approach* to a failure visible, not
just its aftermath. @daoran9 measured the median ASR call going from 1.05s to
31.65s before the empty scenes began; nothing in WhisperJAV recorded that.

These tests use synthetic records because the behaviour under test is the
aggregation and reporting, not the recognizer.
"""

import json

import pytest

from whisperjav.utils.asr_telemetry import AsrTelemetry, summarise_segments


def _segment(temperature=0.0, avg_logprob=-0.4, compression_ratio=1.6,
             no_speech_prob=0.1):
    return {
        "temperature": temperature,
        "avg_logprob": avg_logprob,
        "compression_ratio": compression_ratio,
        "no_speech_prob": no_speech_prob,
    }


class TestSegmentSummary:
    def test_detects_temperature_fallback(self):
        """A temperature above the first configured value means a retry fired."""
        segs = [_segment(temperature=0.0), _segment(temperature=0.17),
                _segment(temperature=0.17)]
        out = summarise_segments(segs)
        assert out["max_temperature"] == 0.17
        assert out["fallback_segments"] == 2

    def test_clean_decode_reports_no_fallback(self):
        out = summarise_segments([_segment(), _segment()])
        assert out["fallback_segments"] == 0
        assert out["max_temperature"] == 0.0

    def test_reports_the_gates_that_trigger_retries(self):
        segs = [_segment(avg_logprob=-0.2, compression_ratio=1.4, no_speech_prob=0.05),
                _segment(avg_logprob=-1.9, compression_ratio=2.9, no_speech_prob=0.88)]
        out = summarise_segments(segs)
        assert out["min_avg_logprob"] == -1.9
        assert out["max_compression_ratio"] == 2.9
        assert out["max_no_speech_prob"] == 0.88

    def test_empty_scene_is_representable(self):
        out = summarise_segments([])
        assert out["n_segments"] == 0
        assert out["max_temperature"] is None
        assert out["fallback_segments"] == 0

    def test_tolerates_missing_or_odd_fields(self):
        """Must keep working if the upstream dataclass changes."""
        out = summarise_segments([{"start": 0.0}, {"temperature": None}, {}])
        assert out["n_segments"] == 3
        assert out["max_temperature"] is None


class TestTelemetryFile:
    def test_writes_one_json_object_per_scene(self, tmp_path):
        t = AsrTelemetry(tmp_path / "t.jsonl", "CLIP-001")
        for i in range(3):
            t.record_scene(index=i, audio_duration_s=28.0, wall_s=5.0,
                           segments=[_segment()],
                           produced_output=True)
        path = t.write()
        assert path and path.exists()

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 3
        assert rows[0]["media"] == "CLIP-001"
        assert rows[0]["rtf"] == pytest.approx(5.0 / 28.0, abs=0.001)

    def test_nothing_recorded_writes_nothing(self, tmp_path):
        assert AsrTelemetry(tmp_path / "t.jsonl", "x").write() is None

    def test_recording_never_raises(self, tmp_path):
        """Telemetry failure must stay a telemetry failure."""
        t = AsrTelemetry(tmp_path / "t.jsonl", "x")
        t.record_scene(index=0, audio_duration_s=None, wall_s=1.0,
                       segments=[{"bad": object()}])
        t.record_scene(index=1, audio_duration_s="nonsense", wall_s=1.0)
        # Whatever happened, the run continues.
        assert isinstance(t.records, list)

    def test_unwritable_destination_degrades(self, tmp_path):
        t = AsrTelemetry(tmp_path / "nope" / "\0bad" / "t.jsonl", "x")
        t.record_scene(index=0, audio_duration_s=10.0, wall_s=1.0)
        assert t.write() is None


class TestTrendSummary:
    def test_surfaces_the_394_shape(self, tmp_path):
        """Healthy early, degrading late — the pattern reporters described."""
        t = AsrTelemetry(tmp_path / "t.jsonl", "x")  # records go to disk as they happen
        for i in range(10):                      # healthy
            t.record_scene(index=i, audio_duration_s=28.0, wall_s=1.05,
                           segments=[_segment()],
                           produced_output=True)
        for i in range(10, 20):                  # degraded
            t.record_scene(index=i, audio_duration_s=28.0, wall_s=31.65,
                           segments=[_segment(temperature=0.17),
                                     _segment(temperature=0.17)],
                           produced_output=False)

        trend = t.trend_summary(window=10)
        assert trend is not None
        assert "RTF" in trend
        assert "fallback segs/scene 0.00 -> 2.00" in trend

    def test_too_few_scenes_gives_no_trend(self, tmp_path):
        """Don't imply a trend from a handful of samples."""
        t = AsrTelemetry(tmp_path / "t.jsonl", "x")
        for i in range(4):
            t.record_scene(index=i, audio_duration_s=28.0, wall_s=1.0)
        assert t.trend_summary(window=10) is None
