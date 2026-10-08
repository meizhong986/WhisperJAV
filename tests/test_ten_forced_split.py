"""TEN longest-segment split (max_speech_duration_s).

A speech region longer than the limit must be cut into pieces that are each no longer than the limit, at the
deepest point of TEN's speech probability, with the cut placed at the right time even when the region contains a
bridged pause (min_silence_duration_ms) or a start pad.

The model is replaced by a scripted one (fixed probability per 16 ms frame), so no ten_vad package is needed.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_module_direct(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_root = Path(__file__).parent.parent / "whisperjav" / "modules" / "speech_segmentation"
_load_module_direct("whisperjav.modules.speech_segmentation.base", _root / "base.py")
_ten = _load_module_direct("whisperjav.modules.speech_segmentation.backends.ten", _root / "backends" / "ten.py")
TenSpeechSegmenter = _ten.TenSpeechSegmenter

SR = 16000
HOP = 256
FD = HOP / SR  # 16 ms per frame


class _ScriptedModel:
    """Stands in for ten_vad.TenVad: returns one scripted probability per frame."""

    def __init__(self, probs, threshold):
        self._probs = iter(probs)
        self._threshold = threshold
        self.out_flags = SimpleNamespace(value=0)
        self.out_probability = SimpleNamespace(value=0.0)

    def process(self, frame):
        p = next(self._probs)
        self.out_probability.value = p
        self.out_flags.value = int(p >= self._threshold)


def _probs(total_s, spans, floor=0.02):
    """Per-frame probabilities: `floor` everywhere, then each (start_s, end_s, value) span; a value may be a
    (from, to) pair for a straight ramp."""
    n = int(round(total_s / FD))
    p = np.full(n, floor, dtype=np.float64)
    for a, b, v in spans:
        i, j = int(round(a / FD)), int(round(b / FD))
        p[i:j] = np.linspace(v[0], v[1], j - i) if isinstance(v, tuple) else v
    return p.tolist()


def _run(probs, **kw):
    params = dict(threshold=0.5, hop_size=HOP, min_speech_duration_ms=100, min_silence_duration_ms=200,
                  max_speech_duration_s=4.0, start_pad_ms=0, end_pad_ms=0)
    params.update(kw)
    seg = TenSpeechSegmenter(**params)
    seg._model = _ScriptedModel(probs, params["threshold"])
    audio = np.zeros(len(probs) * HOP, dtype=np.float32)
    return seg.segment(audio, sample_rate=SR).segments


def _bounds(segments):
    return [(round(s.start_sec, 3), round(s.end_sec, 3)) for s in segments]


def test_cut_at_deepest_dip_not_first_dip_after_80_percent():
    # 1.0-8.0 s speech at 0.9 (two pieces needed, so the cut must lie between 4 and 5 s); a deep dip (0.52) at
    # 4.3-4.4 s and a shallow one (0.7) at 4.7-4.8 s.
    probs = _probs(10.0, [(1.0, 8.0, 0.9), (4.3, 4.4, 0.52), (4.7, 4.8, 0.7)])
    segs = _run(probs)
    assert len(segs) == 2, _bounds(segs)
    assert 4.3 <= segs[0].end_sec <= 4.4, _bounds(segs)


@pytest.mark.parametrize("length", [4.26, 6.0, 8.0, 10.0, 12.5, 20.0])
def test_fewest_pieces(length):
    # Speech-like probability (0.9 +/- 0.08, random): the region is cut into ceil(length / 4 s) pieces, no more.
    rng = np.random.default_rng(int(length * 100))
    n = int(round((length + 2.0) / FD))
    p = np.full(n, 0.02)
    i, j = int(round(1.0 / FD)), int(round((1.0 + length) / FD))
    p[i:j] = np.clip(0.9 + rng.uniform(-0.08, 0.08, j - i), 0.0, 1.0)
    segs = _run(p.tolist())
    real_length = segs[-1].end_sec - segs[0].start_sec
    assert len(segs) == int(np.ceil(real_length / 4.0 - 1e-9)), _bounds(segs)
    assert all(1.0 - 1e-9 <= s.end_sec - s.start_sec <= 4.0 + 1e-9 for s in segs), _bounds(segs)


def test_no_piece_longer_than_the_limit():
    # Probability rises steadily from 1 to 7 s, drops, rises again to 12 s: the only dip is 6 s after the start.
    probs = _probs(13.0, [(1.0, 7.0, (0.55, 0.95)), (7.0, 12.0, (0.55, 0.95))])
    segs = _run(probs)
    lengths = [s.end_sec - s.start_sec for s in segs]
    assert max(lengths) <= 4.0 + 1e-9, _bounds(segs)
    assert segs[0].start_sec == pytest.approx(1.0, abs=FD) and segs[-1].end_sec == pytest.approx(12.0, abs=FD)


def test_cut_falls_in_bridged_pause_with_start_pad():
    # Speech 1.0-4.5 s, a 150 ms pause (bridged by min silence 200 ms), speech 4.65-8.0 s; start pad 300 ms.
    probs = _probs(9.0, [(1.0, 4.5, 0.9), (4.65, 8.0, 0.9)], floor=0.05)
    segs = _run(probs, start_pad_ms=300)
    assert segs[0].start_sec == pytest.approx(0.7, abs=FD), _bounds(segs)
    assert 4.5 <= segs[0].end_sec <= 4.65, _bounds(segs)


def test_pieces_touch_and_cover_the_region():
    probs = _probs(23.0, [(1.0, 22.0, 0.9)])
    segs = _run(probs)
    assert all(a.end_sec == b.start_sec for a, b in zip(segs, segs[1:])), _bounds(segs)
    assert segs[0].start_sec == pytest.approx(1.0, abs=FD) and segs[-1].end_sec == pytest.approx(22.0, abs=FD)
    assert all(1.0 - 1e-9 <= s.end_sec - s.start_sec <= 4.0 + 1e-9 for s in segs), _bounds(segs)


def test_detection_keeps_one_region_split_happens_once():
    # With min silence 0 the merge step does not hide a detection-stage cut; there must be none.
    seg = TenSpeechSegmenter(threshold=0.5, hop_size=HOP, min_speech_duration_ms=100,
                             min_silence_duration_ms=0, max_speech_duration_s=4.0)
    n = int(round(10.0 / FD))
    raw = seg._flags_to_segments([1] * n, [0.9] * n, SR, audio_duration=10.0)
    assert len(raw) == 1


def test_short_regions_unchanged():
    probs = _probs(10.0, [(1.0, 3.0, 0.9), (5.0, 8.5, 0.9)])
    segs = _run(probs, start_pad_ms=100, end_pad_ms=100)
    assert len(segs) == 2, _bounds(segs)
    for s, (a, b) in zip(segs, [(0.9, 3.1), (4.9, 8.6)]):
        assert s.start_sec == pytest.approx(a, abs=FD) and s.end_sec == pytest.approx(b, abs=FD)


def test_no_limit_means_no_split():
    probs = _probs(12.0, [(1.0, 11.0, 0.9)])
    segs = _run(probs, max_speech_duration_s=0)
    assert len(segs) == 1, _bounds(segs)
    assert segs[0].start_sec == pytest.approx(1.0, abs=FD) and segs[0].end_sec == pytest.approx(11.0, abs=FD)
