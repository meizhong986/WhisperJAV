"""The shortest audio WhisperJAV hands to a model: 0.2 s (owner W461; whisperjav/modules/model_audio.py).

What is pinned here:
  - the floor is 0.2 s, below Netflix's Japanese minimum subtitle duration (0.5 s) with a margin, and above the
    band that stops CrispASR's DTW word timing (fewer than 2600 samples at 16 kHz);
  - widen_short_spans widens a short span into the room beside it, both sides alike, the other side taking what one
    side cannot, never past the scene's edges nor into a neighbour; long spans are untouched;
  - the VAD-grouped framer cuts groups under 0.2 s (owner W463 CL2: usually VAD false positives); the other
    strategy, widening them, is kept ready and tested; widening never cuts or inverts a span; a span is
    never cut or inverted by its neighbours;
  - the timing model, the aligner and the CrispASR engine refuse audio under 0.2 s without calling the model;
  - ChronosJAV's orchestrator gives a scene's one frame the whole scene only when the frame covers the scene.

No model is loaded. Run alone: python -m pytest tests/test_model_audio_minimum.py
"""

import types

import numpy as np
import pytest

from whisperjav.modules import model_audio
from whisperjav.modules.model_audio import MIN_MODEL_AUDIO_S, min_samples, too_short, widen_short_spans

SR = 16000


def test_the_floor():
    assert MIN_MODEL_AUDIO_S == 0.2
    assert MIN_MODEL_AUDIO_S < 0.5 / 2          # Netflix Japanese minimum subtitle duration 500 ms, with margin
    assert min_samples(SR) == 3200 > 2600       # above the DTW band (n_frames <= 15 below 2600 samples)
    assert too_short(3199, SR) and not too_short(3200, SR) and too_short(0, SR)


def test_long_spans_are_untouched():
    spans = [(1.0, 2.0), (3.0, 3.25)]
    assert widen_short_spans(spans, 0.0, 10.0) == spans


def test_a_short_span_grows_on_both_sides():
    (start, end), = widen_short_spans([(5.0, 5.15)], 0.0, 10.0)
    assert end - start >= MIN_MODEL_AUDIO_S
    assert start < 5.0 and end > 5.15 and abs((5.0 - start) - (end - 5.15)) < 1e-9


def test_the_edge_and_the_neighbours_are_respected():
    # at the scene's start: the right side takes what the left cannot
    (start, end), = widen_short_spans([(0.01, 0.13)], 0.0, 10.0)
    assert start == 0.0 and end - start >= MIN_MODEL_AUDIO_S
    # between two neighbours with little room: never into them
    spans = widen_short_spans([(1.0, 2.0), (2.02, 2.14), (2.16, 3.0)], 0.0, 10.0)
    assert spans[0] == (1.0, 2.0) and spans[2] == (2.16, 3.0)
    assert spans[1][0] >= 2.0 and spans[1][1] <= 2.16
    # no room at all: stays as it is (the model entry points refuse it)
    assert widen_short_spans([(0.0, 0.12)], 0.0, 0.12) == [(0.0, 0.12)]


def test_overlapping_or_unordered_neighbours_never_shrink_or_invert_a_span():
    # review finding 1: an overlapping or out-of-order neighbour leaves no room on that side; nothing is cut
    cases = [[(0.0, 1.0), (0.9, 1.05)], [(0.9, 1.05), (1.0, 3.0)], [(5.0, 6.0), (2.0, 2.1), (3.0, 4.0)]]
    for spans in cases:
        for (s0, e0), (s1, e1) in zip(spans, widen_short_spans(spans, 0.0, 10.0)):
            assert s1 <= s0 and e1 >= e0 and e1 > s1, (spans, (s1, e1))


def test_a_widened_span_survives_cutting_at_whole_samples():
    (start, end), = widen_short_spans([(1.23456, 1.35)], 0.0, 10.0)
    assert not too_short(int(end * SR) - int(start * SR), SR)


def _framer_with(groups, duration_s):
    from whisperjav.modules.subtitle_pipeline.framers.vad_grouped import VadGroupedFramer

    result = types.SimpleNamespace(
        groups=[[types.SimpleNamespace(start_sec=s, end_sec=e, metadata={})] for s, e in groups],
        method="stub", num_segments=len(groups), num_groups=len(groups), speech_coverage_ratio=0.1,
        parameters={}, processing_time_sec=0.0, speech_regions=[])
    framer = VadGroupedFramer(min_frame_duration_s=0.1)
    framer._segmenter = types.SimpleNamespace(segment=lambda audio, sample_rate: result)
    return framer.frame(np.zeros(int(duration_s * SR), dtype=np.float32), SR)


def test_the_framer_cuts_short_groups_by_default():
    # owner W463 CL2: VAD groups under 200 ms are usually false positives
    assert model_audio.SHORT_VAD_GROUPS == "cut"
    out = _framer_with([(1.0, 1.05), (2.0, 2.15), (4.0, 6.0)], 10.0)
    assert [(f.start, f.end) for f in out.frames] == [(4.0, 6.0)]
    assert out.metadata["groups_skipped"] == 1 and out.metadata["groups_cut_short"] == 1
    assert out.metadata["groups_widened"] == 0


def test_the_widen_strategy_is_ready(monkeypatch):
    monkeypatch.setattr(model_audio, "SHORT_VAD_GROUPS", "widen")
    out = _framer_with([(1.0, 1.05), (2.0, 2.15), (4.0, 6.0)], 10.0)
    spans = [(f.start, f.end) for f in out.frames]
    assert len(spans) == 2                                   # the 0.05 s group is VAD noise, dropped as before
    assert spans[0][1] - spans[0][0] >= MIN_MODEL_AUDIO_S    # the 0.15 s group is widened
    assert spans[1] == (4.0, 6.0)
    assert out.metadata["groups_widened"] == 1 and out.metadata["groups_cut_short"] == 0


def test_the_timing_model_never_sees_a_short_piece():
    from whisperjav.crispasr.timer import CrispTimer

    calls = []
    timer = object.__new__(CrispTimer)
    timer._model = types.SimpleNamespace(transcribe_pcm=lambda *a, **k: calls.append(1) or [])
    timer.label, timer.model = "test", types.SimpleNamespace(label="Whisper small")
    assert timer.time(np.zeros(min_samples(SR) - 1, dtype=np.float32)) == []
    assert calls == []
    timer.time(np.zeros(min_samples(SR), dtype=np.float32))
    assert calls == [1]


def test_the_aligner_never_sees_a_short_piece():
    from whisperjav.crispasr.aligner import CrispAligner

    calls = []
    aligner = object.__new__(CrispAligner)
    aligner._crispasr = types.SimpleNamespace(align_words=lambda *a, **k: calls.append(1) or [])
    aligner._path, aligner.n_threads, aligner.label = "x", 1, "test"
    assert aligner.align(np.zeros(min_samples(SR) - 1, dtype=np.float32), "あ") == []
    assert calls == []


def test_the_crispasr_engine_never_sees_a_short_piece():
    from whisperjav.crispasr.engine import BindingEngine

    calls = []
    engine = object.__new__(BindingEngine)
    engine._session = types.SimpleNamespace(transcribe=lambda *a, **k: calls.append(1) or [])
    assert engine.transcribe(np.zeros(min_samples(SR) - 1, dtype=np.float32)) == []
    assert calls == []


@pytest.mark.parametrize("frames, scene_s, whole", [
    ([(0.0, 30.0)], 30.0, True),       # the full-scene framer
    ([(0.0, 0.205)], 30.0, False),     # a short first VAD group widened back to 0.0: cut, not the whole scene
    ([(0.0, 5.0), (6.0, 7.0)], 30.0, False),
])
def test_only_a_frame_that_covers_the_scene_is_the_whole_scene(frames, scene_s, whole):
    from whisperjav.modules.subtitle_pipeline.orchestrator import _is_full_scene

    objs = [types.SimpleNamespace(start=s, end=e) for s, e in frames]
    assert _is_full_scene(objs, scene_s) is whole


def test_one_module_holds_the_rule():
    assert model_audio.__doc__ and "500 ms" in model_audio.__doc__
