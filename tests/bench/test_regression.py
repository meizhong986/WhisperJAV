"""
Tests for the accuracy regression gate (whisperjav/bench/regression.py).

Uses synthetic SRT data modeled on the failure modes the gate exists to
catch: the F4/F6 catastrophic subtitle collapse, Whisper repetition loops,
and gradual preset-retune drift.
"""

import json
from pathlib import Path

import pytest

from whisperjav.bench.regression import (
    build_baseline,
    check_clip,
    gate,
    load_baseline,
    score_clip,
    score_srt_files,
)


def _mk(start, end, text):
    return {"start": float(start), "end": float(end), "text": text}


GT = [
    _mk(0.0, 2.0, "こんにちは"),
    _mk(3.0, 5.0, "元気ですか"),
    _mk(6.0, 8.5, "今日はいい天気ですね"),
    _mk(10.0, 12.0, "そうですね"),
    _mk(14.0, 16.0, "また明日"),
]


class TestScoreClip:
    def test_perfect_output(self):
        m = score_clip(GT, [dict(s) for s in GT])
        assert m.cer == 0.0
        assert m.repetition_ratio == 0.0
        assert m.recall == 1.0 and m.precision == 1.0 and m.f1 == 1.0
        assert m.mean_iou == 1.0
        assert m.gt_coverage == 1.0
        assert m.false_alarm_ratio == 0.0
        assert m.count_ratio == 1.0

    def test_missed_subtitle_lowers_recall(self):
        hyp = [dict(s) for s in GT[:-1]]  # drop the last line
        m = score_clip(GT, hyp)
        assert m.recall == pytest.approx(0.8)
        assert m.precision == 1.0

    def test_text_errors_raise_cer(self):
        hyp = [dict(s) for s in GT]
        hyp[0]["text"] = "こんばんは"  # 2-char error
        m = score_clip(GT, hyp)
        assert m.cer > 0.0

    def test_timing_drift_lowers_iou(self):
        hyp = [_mk(s["start"] + 0.8, s["end"] + 0.8, s["text"]) for s in GT]
        m = score_clip(GT, hyp)
        assert m.mean_iou < 0.8
        assert m.recall == 1.0  # still matched, just shifted

    def test_catastrophic_collapse_signature(self):
        """The F4/F6 pattern: a handful of subs against a full GT."""
        m = score_clip(GT, [dict(GT[0])])
        assert m.count_ratio == pytest.approx(0.2)
        assert m.recall == pytest.approx(0.2)

    def test_repetition_loop_detected(self):
        hyp = [_mk(i * 2.0, i * 2.0 + 1.5, "ああああ") for i in range(10)]
        m = score_clip(GT, hyp)
        assert m.repetition_ratio == 1.0

    def test_false_alarms_outside_gt(self):
        hyp = [dict(s) for s in GT] + [_mk(20.0, 30.0, "幻覚テキスト")]
        m = score_clip(GT, hyp)
        assert m.false_alarm_ratio > 0.4  # 10s of 21.5s total hyp time

    def test_empty_hypothesis(self):
        m = score_clip(GT, [])
        assert m.recall == 0.0 and m.n_hyp == 0 and m.count_ratio == 0.0

    def test_empty_ground_truth_raises(self):
        with pytest.raises(ValueError):
            score_clip([], [dict(GT[0])])


class TestGate:
    def test_hard_bound_catches_collapse_without_baseline(self):
        m = score_clip(GT, [dict(GT[0])])
        violations = check_clip("clip", m.as_dict())
        kinds = {(v.metric, v.kind) for v in violations}
        assert ("count_ratio", "hard_bound") in kinds
        assert ("recall", "hard_bound") in kinds

    def test_hard_bound_catches_repetition_loop(self):
        hyp = [_mk(i * 2.0, i * 2.0 + 1.5, "ああああ") for i in range(10)]
        m = score_clip(GT, hyp)
        violations = check_clip("clip", m.as_dict())
        assert any(v.metric == "repetition_ratio" and v.kind == "hard_bound"
                   for v in violations)

    def test_regression_vs_baseline(self):
        good = score_clip(GT, [dict(s) for s in GT])
        drifted = [_mk(s["start"] + 1.2, s["end"] + 1.2, s["text"]) for s in GT]
        bad = score_clip(GT, drifted)
        baseline = build_baseline({"clip": good})
        violations = gate({"clip": bad}, baseline)
        assert any(v.metric == "mean_iou" and v.kind == "regression"
                   for v in violations)

    def test_small_drift_within_threshold_passes(self):
        good = score_clip(GT, [dict(s) for s in GT])
        nearly = [dict(s) for s in GT]
        nearly[0] = _mk(0.05, 2.0, nearly[0]["text"])  # 50ms shift on one sub
        m = score_clip(GT, nearly)
        baseline = build_baseline({"clip": good})
        assert gate({"clip": m}, baseline) == []

    def test_identical_run_passes(self):
        m = score_clip(GT, [dict(s) for s in GT])
        baseline = build_baseline({"clip": m})
        assert gate({"clip": m}, baseline) == []

    def test_improvement_never_fails(self):
        drifted = [_mk(s["start"] + 1.2, s["end"] + 1.2, s["text"]) for s in GT]
        worse = score_clip(GT, drifted)
        better = score_clip(GT, [dict(s) for s in GT])
        baseline = build_baseline({"clip": worse})
        assert gate({"clip": better}, baseline) == []

    def test_unknown_clip_in_run_only_hard_bounded(self):
        m = score_clip(GT, [dict(s) for s in GT])
        baseline = build_baseline({"other_clip": m})
        assert gate({"new_clip": m}, baseline) == []


class TestSerialization:
    def test_baseline_roundtrip(self, tmp_path: Path):
        m = score_clip(GT, [dict(s) for s in GT])
        baseline = build_baseline({"clip": m}, label="test",
                                  whisperjav_version="1.8.14")
        p = tmp_path / "baseline.json"
        p.write_text(json.dumps(baseline), encoding="utf-8")
        loaded = load_baseline(p)
        assert loaded["clips"]["clip"]["f1"] == 1.0
        assert loaded["label"] == "test"

    def test_score_srt_files(self, tmp_path: Path):
        def ts(sec):
            ms = int(round(sec * 1000))
            return f"{ms//3600000:02d}:{ms%3600000//60000:02d}:{ms%60000//1000:02d},{ms%1000:03d}"
        srt = "\n".join(
            f"{i+1}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n"
            for i, s in enumerate(GT)
        )
        gt_path = tmp_path / "gt.srt"
        hyp_path = tmp_path / "hyp.srt"
        gt_path.write_text(srt, encoding="utf-8")
        hyp_path.write_text(srt, encoding="utf-8")
        m = score_srt_files(gt_path, hyp_path)
        assert m.f1 == 1.0 and m.cer == 0.0

    def test_missing_hyp_file_scores_as_empty(self, tmp_path: Path):
        def ts(sec):
            ms = int(round(sec * 1000))
            return f"{ms//3600000:02d}:{ms%3600000//60000:02d}:{ms%60000//1000:02d},{ms%1000:03d}"
        srt = "\n".join(
            f"{i+1}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n"
            for i, s in enumerate(GT)
        )
        gt_path = tmp_path / "gt.srt"
        gt_path.write_text(srt, encoding="utf-8")
        m = score_srt_files(gt_path, tmp_path / "missing.srt")
        assert m.n_hyp == 0 and m.recall == 0.0
