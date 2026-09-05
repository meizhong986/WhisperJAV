"""Tests for the v7.2.0 WhisperJAV modifications to the semantic scene engine.

Covers the four modifications (see vendor module docstring):
  A. per-chunk anchored time axis (drift fix)
  B. silence-aware onset-anchored snapping
  C. silence-clamped per-boundary padding
  D. overlong-scene warning (no splitter) — via the E2E metadata run

Synthetic-signal tests exercise the module-level helpers and the snapper
directly; the E2E test runs the real process_movie_v7 + adapter on generated
audio and asserts the structural invariants the owner requires (full
coverage, strict timestamps tiling, pads clamped).
"""

import json
import numpy as np
import pytest
import soundfile as sf

from whisperjav.vendor.semantic_audio_clustering import (
    FeatureRegistry,
    SegmentationConfig,
    SemanticSegmenter,
    StreamFeatureExtractor,
    _silence_runs,
    compute_adaptive_pads,
    compute_silence_floor,
    process_movie_v7,
)

FRAME_DT = 512 / 16000  # 32ms


def _features_from_rms(rms):
    """Build a (36, N) feature matrix with the given RMS row."""
    n = len(rms)
    feats = np.zeros((FeatureRegistry.TOTAL_DIM, n), dtype=np.float32)
    feats[FeatureRegistry.RMS, :] = rms
    return feats


def _times(n):
    return np.arange(n) * FRAME_DT


class TestSilenceRuns:
    def test_empty_and_full(self):
        assert _silence_runs(np.array([], dtype=bool)) == []
        assert _silence_runs(np.array([True] * 4)) == [(0, 3)]
        assert _silence_runs(np.array([False] * 4)) == []

    def test_multiple_runs(self):
        mask = np.array([False, True, True, False, True, False, True])
        assert _silence_runs(mask) == [(1, 2), (4, 4), (6, 6)]


class TestOnsetAnchoredSnap:
    """Snapping behavior on a controlled RMS landscape."""

    def _snap(self, rms, boundary_sec, cfg=None):
        cfg = cfg or SegmentationConfig(snap_window=5.0)
        feats = _features_from_rms(np.asarray(rms, dtype=np.float32))
        t = _times(len(rms))
        seg = SemanticSegmenter(cfg)
        floor = 0.01  # explicit floor for determinism
        out = seg._snap_to_silence(
            [0.0, boundary_sec, t[-1]], feats, t, silence_floor=floor
        )
        # out = [0.0, snapped, end]
        assert len(out) == 3, out
        return out[1]

    def test_cut_lands_at_end_of_silence_run(self):
        # Speech 0-3s (rms .1), silence 3-4s (rms .005), speech 4-8s (.1).
        n = int(8 / FRAME_DT)
        rms = np.full(n, 0.1)
        sil_lo, sil_hi = int(3 / FRAME_DT), int(4 / FRAME_DT)
        rms[sil_lo:sil_hi] = 0.005
        # Raw boundary at 3.2s (inside the silence)
        cut = self._snap(rms, 3.2)
        run_end_time = (sil_hi - 1) * FRAME_DT
        # Onset-anchored: at the run end minus the 2-frame back-off.
        assert cut == pytest.approx(run_end_time - 2 * FRAME_DT, abs=1e-9)
        # The scene AFTER the cut starts within ~3 frames of the sound onset.
        assert (4.0 - cut) < 4 * FRAME_DT

    def test_nearest_wide_run_preferred(self):
        # Two silences: 1.0-1.5s and 3.0-4.0s; boundary at 2.8s -> nearer run
        # is the 3-4s one.
        n = int(8 / FRAME_DT)
        rms = np.full(n, 0.1)
        rms[int(1 / FRAME_DT):int(1.5 / FRAME_DT)] = 0.005
        rms[int(3 / FRAME_DT):int(4 / FRAME_DT)] = 0.005
        cut = self._snap(rms, 2.8)
        assert 3.5 < cut < 4.0  # inside the nearer run, at its end

    def test_argmin_fallback_when_no_silence(self):
        # No frame below floor; a wide dip (0.02, above the 0.01 floor) at
        # 5.0s must be chosen via the historical argmin fallback. The dip is
        # 7 frames wide so it survives the 5-frame median smoothing (a
        # 1-frame dip would be erased by the filter, in v7.1 exactly as now).
        n = int(8 / FRAME_DT)
        rms = np.full(n, 0.1)
        dip = int(5 / FRAME_DT)
        rms[dip - 3:dip + 4] = 0.02
        cut = self._snap(rms, 4.0)
        # argmin returns the first minimal frame (start of the flat dip)
        assert cut == pytest.approx((dip - 3) * FRAME_DT, abs=1e-9)

    def test_short_silence_still_preferred_over_argmin(self):
        # A short (~130ms, 4-frame) true silence at 4.0s and a wide non-silent
        # dip (0.02) at 6.0s: the silence must win over the deeper... no —
        # the silence IS deeper; the point is that a short-but-real silence is
        # chosen and onset-anchored rather than argmin-centered.
        n = int(8 / FRAME_DT)
        rms = np.full(n, 0.1)
        lo = int(4 / FRAME_DT)
        rms[lo:lo + 6] = 0.005  # ≥5 frames so the median filter keeps its core
        cut = self._snap(rms, 3.9)
        assert lo * FRAME_DT <= cut <= (lo + 6) * FRAME_DT


class TestAdaptivePads:
    def test_pads_clamped_to_silence_extent(self):
        # Silence 2.0-3.0s; segment boundary (start) at 2.9s: backward extent
        # ~0.9s -> clamped to 0.35 cap.
        n = int(6 / FRAME_DT)
        rms = np.full(n, 0.1)
        rms[int(2 / FRAME_DT):int(3 / FRAME_DT)] = 0.005
        t = _times(n)
        start_pad, end_pad = compute_adaptive_pads(2.9, 5.0, rms, t, 0.01)
        assert start_pad == pytest.approx(0.35)
        # End at 5.0s is mid-speech: no silence forward -> floor 0.05.
        assert end_pad == pytest.approx(0.05)

    def test_partial_silence_extent_used(self):
        # Silence only ~2.8-3.0s immediately before a start at 3.0s -> pad
        # ≈ 0.2 (between floor and cap). The +1 keeps the frame just before
        # the searchsorted start index inside the silent slice.
        n = int(6 / FRAME_DT)
        rms = np.full(n, 0.1)
        rms[int(2.8 / FRAME_DT):int(3.0 / FRAME_DT) + 1] = 0.005
        t = _times(n)
        start_pad, _ = compute_adaptive_pads(3.0, 5.0, rms, t, 0.01)
        assert 0.1 <= start_pad <= 0.3

    def test_first_segment_has_zero_start_pad(self):
        n = int(4 / FRAME_DT)
        rms = np.full(n, 0.005)  # all silence
        t = _times(n)
        start_pad, end_pad = compute_adaptive_pads(0.0, 2.0, rms, t, 0.01, is_first=True)
        assert start_pad == 0.0
        assert end_pad == pytest.approx(0.35)  # forward silence -> cap


class TestCoverageInvariant:
    def test_tiling_with_gaps_and_overlaps(self):
        seg = SemanticSegmenter(SegmentationConfig())
        segs = [
            {"start": 5.0, "end": 30.0, "dur": 25.0, "vec": np.zeros(36)},
            {"start": 40.0, "end": 70.0, "dur": 30.0, "vec": np.zeros(36)},
            {"start": 65.0, "end": 90.0, "dur": 25.0, "vec": np.zeros(36)},
        ]
        out = seg._ensure_timeline_coverage(segs, 100.0)
        assert out[0]["start"] == 0.0
        assert out[-1]["end"] == 100.0
        for a, b in zip(out, out[1:]):
            assert b["start"] == pytest.approx(a["end"])  # no gaps, no overlap


@pytest.fixture(scope="module")
def synthetic_wav(tmp_path_factory):
    """~150s audio alternating speech-like noise bursts and silences.

    Long enough for 3 feature chunks (60s each) so the drift fix is
    exercised, with hard silences so snapping has real targets.
    """
    sr = 16000
    rng = np.random.default_rng(42)
    total = 150.0
    y = np.zeros(int(total * sr), dtype=np.float32)
    # Speech-like bursts: [start, end) seconds
    bursts = [
        (0.5, 28.0), (30.0, 55.0), (58.0, 88.0), (91.0, 118.0), (121.0, 149.0)
    ]
    for b0, b1 in bursts:
        seg = rng.normal(0, 0.25, int((b1 - b0) * sr)).astype(np.float32)
        # amplitude modulation to look speech-ish
        mod = 0.5 + 0.5 * np.abs(np.sin(np.linspace(0, 40 * np.pi, len(seg))))
        y[int(b0 * sr):int(b0 * sr) + len(seg)] = seg * mod
    path = tmp_path_factory.mktemp("semantic") / "synthetic.wav"
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path, total


class TestDriftFix:
    def test_time_axis_anchored_per_chunk(self, synthetic_wav):
        path, total = synthetic_wav
        cfg = SegmentationConfig()
        ext = StreamFeatureExtractor(cfg)
        feats, times, duration = ext.extract(str(path))
        assert duration == pytest.approx(total, abs=0.01)
        # Old bug: times[-1] exceeded duration by ~32ms * n_chunks.
        # Anchored axis: the final frame time stays within one frame of the
        # true duration.
        assert times[-1] <= duration + FRAME_DT + 1e-6
        assert times[-1] >= duration - 2 * FRAME_DT
        # Chunk-seam sanity: monotonic non-decreasing everywhere.
        assert np.all(np.diff(times) >= -1e-9)
        # Frame count matches features.
        assert len(times) == feats.shape[1]


class TestEndToEnd:
    def test_process_movie_v7_invariants(self, synthetic_wav, tmp_path):
        path, total = synthetic_wav
        out_json = tmp_path / "meta.json"
        cfg = SegmentationConfig(min_duration=12, max_duration=48)
        process_movie_v7(str(path), str(out_json), config=cfg)

        data = json.loads(out_json.read_text(encoding="utf-8"))
        segs = data["segments"]
        assert segs, "engine produced no segments"

        # 1. FULL COVERAGE: strict timestamps tile 0 -> duration exactly.
        assert segs[0]["timestamps"]["start"] == pytest.approx(0.0, abs=1e-3)
        assert segs[-1]["timestamps"]["end"] == pytest.approx(
            data["meta"]["total_duration_seconds"], abs=1e-3
        )
        for a, b in zip(segs, segs[1:]):
            assert b["timestamps"]["start"] == pytest.approx(
                a["timestamps"]["end"], abs=1e-3
            ), "strict timestamps must be continuous (no gap, no overlap)"

        # 2. Pads are per-boundary, clamped to [0, 0.35]; first start pad 0.
        for i, s in enumerate(segs):
            pads = s["asr_processing"]["padding_applied"]
            assert 0.0 <= pads["start"] <= 0.35
            assert 0.0 <= pads["end"] <= 0.35
            if i == 0:
                assert pads["start"] == 0.0
            # asr window contains the strict window
            assert s["asr_processing"]["start"] <= s["timestamps"]["start"]
            assert s["asr_processing"]["end"] >= s["timestamps"]["end"]

        # 3. Overlap between adjacent ASR windows is bounded by the pad sum
        #    (<= 0.7s, and typically far less at silence-anchored cuts).
        for a, b in zip(segs, segs[1:]):
            overlap = a["asr_processing"]["end"] - b["asr_processing"]["start"]
            assert overlap <= 0.70 + 1e-6

    def test_adapter_transform(self, synthetic_wav, tmp_path):
        from whisperjav.modules.scene_detection_backends.semantic_adapter import (
            SemanticClusteringAdapter, SemanticClusteringConfig,
        )
        path, total = synthetic_wav
        cfg = SemanticClusteringConfig(min_duration=12, max_duration=48)
        adapter = SemanticClusteringAdapter(config=cfg)
        scenes = adapter.detect_scenes(path, tmp_path, "synthetic", temp_dir=tmp_path)

        assert scenes, "adapter returned no scenes"
        # Every scene extracted, WAV exists, WAV length matches the padded
        # window clamped to the audio's end (the last scene's end pad
        # over-reads past EOF by design; extraction clamps it).
        for scene_path, start, end, dur in scenes:
            assert scene_path.exists()
            info = sf.info(str(scene_path))
            expected = min(end, total) - start
            assert info.duration == pytest.approx(expected, abs=0.01)
        # Coverage through the padded windows: union covers [0, total] —
        # each scene must start no later than the previous scene's end
        # (padded windows overlap or touch; no audible gap).
        for (_sp, s0, e0, _d0), (_sp2, s1, e1, _d1) in zip(scenes, scenes[1:]):
            assert s1 <= e0 + 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
