"""
The timing model in ChronosJAV's orchestrator (DecoupledSubtitlePipeline step 8; item 12 C3, design note
docs/plans/design/2026-10-28-chronosjav-timing-c3-c1.md section 5.3), and the kept text of a frame whose text got
no word times (rule T3).

Stand-in timer and aligner results built from the probe figures of 2026-10-27 (group A of S02E02 Scene 5); the
scene audio is a short sine WAV. No model is loaded.

Run with: python -m pytest tests/test_subtitle_pipeline_timer.py -q
"""
import numpy as np
import pytest

pytest.importorskip("soundfile")
pytest.importorskip("stable_whisper")

from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
from whisperjav.modules.subtitle_pipeline.timing_rules import TimedSegment
from whisperjav.modules.subtitle_pipeline.types import HardeningConfig, TemporalFrame, TimestampMode

TEXT_A = "近くのカラオケでそのまま酔っぱらっちゃった。それで？"
GROUP_A = [("近", 0.26, 0.28), ("く", 0.38, 0.40), ("の", 0.48, 0.50), ("カ", 0.60, 0.62), ("ラ", 0.72, 0.74),
           ("オ", 0.80, 0.82), ("ケ", 0.90, 0.92), ("で", 1.00, 1.02), ("そ", 1.12, 1.14), ("の", 1.26, 1.28),
           ("ま", 1.52, 1.54), ("ま", 1.74, 1.76), ("酔", 1.78, 1.80), ("っ", 1.86, 1.88), ("ぱ", 1.90, 1.92),
           ("ら", 2.08, 2.10), ("っ", 2.32, 2.34), ("ち", 2.36, 2.38), ("ゃ", 2.40, 2.42), ("っ", 2.60, 2.64),
           ("た。", 2.66, 2.68), ("そ", 3.10, 3.12), ("れ", 3.34, 3.36), ("で？", 3.76, 3.79)]
WHISPER_A = [TimedSegment(0.32, 0.0, True), TimedSegment(2.00, 0.0, True), TimedSegment(3.30, 0.0, True)]
FRAME_A = TemporalFrame(1.0, 4.79)  # group A, placed 1.0 s into a 6 s scene


class _Timer:
    accepts_array = True

    def __init__(self, result=WHISPER_A, load_error=None):
        self.result, self.load_error = result, load_error
        self.loaded = self.unloaded = 0
        self.timed = []

    def load(self):
        self.loaded += 1
        if self.load_error is not None:
            raise self.load_error

    def time(self, audio):
        self.timed.append(audio)
        return list(self.result)

    def unload(self):
        self.unloaded += 1

    def cleanup(self):
        self.unload()


def _orchestrator(timer=None, mode=TimestampMode.ALIGNER_WITH_VAD_FALLBACK, aligner=object()):
    return DecoupledSubtitlePipeline(
        framer=object(), generator=object(), cleaner=object(), aligner=aligner,
        hardening_config=HardeningConfig(timestamp_mode=mode), timer=timer,
    )


@pytest.fixture
def scene_wav(tmp_path):
    import soundfile as sf

    wav = tmp_path / "scene_0000.wav"
    sf.write(str(wav), (0.1 * np.sin(np.linspace(0, 6000, 96000))).astype(np.float32), 16000)
    return wav


def _aligned():
    return [[[{"word": w, "start": s, "end": e} for w, s, e in GROUP_A]]]


def _subs(results):
    return [(seg.text.replace(" ", ""), round(seg.start, 2), round(seg.end, 2)) for seg in results[0][0].segments]


def test_with_timings_group_a_becomes_two_subtitles(scene_wav):
    orch = _orchestrator(_Timer())
    timings = orch._step8_time([[FRAME_A]], [[np.zeros(60640, dtype=np.float32)]], [[TEXT_A]])
    results = orch._step9_reconstruct_and_harden([[FRAME_A]], [[TEXT_A]], _aligned(), [scene_wav], [6.0],
                                                 [None], None, scene_timings=timings)
    assert _subs(results) == [("近くのカラオケでそのまま酔っぱらっちゃった。", 1.32, 4.30), ("それで？", 4.30, 4.79)]
    assert results[0][1]["timing_model"] == {"frames_timed": 1, "starts_moved": 1, "splits": 1,
                                             "split_times": [4.3]}
    assert orch.timer.loaded == 1 and orch.timer.unloaded == 1


def _collapsed():
    # Every character piled onto the group's first 20 ms: the collapse check's case.
    return [[[{"word": w, "start": 0.0, "end": 0.02} for w, _s, _e in GROUP_A]]]


def test_a_collapsed_scene_takes_the_vad_group_times(scene_wav):
    """Owner W266 1: under aligner_vad_fallback with regrouping off, a collapsed scene takes the aligner-free
    times: the VAD group's edges, its start moved by the timing model, no split; its text is kept whole."""
    orch = _orchestrator(_Timer())
    timings = orch._step8_time([[FRAME_A]], [[np.zeros(60640, dtype=np.float32)]], [[TEXT_A]])
    results = orch._step9_reconstruct_and_harden([[FRAME_A]], [[TEXT_A]], _collapsed(), [scene_wav], [6.0],
                                                 [None], None, scene_timings=timings)
    assert _subs(results) == [(TEXT_A, 1.32, 4.79)]
    diag = results[0][1]
    assert diag["sentinel_status"] == "COLLAPSED"
    assert diag["timing_model"]["splits"] == 0
    assert orch.sentinel_stats["recovery_strategies"]["vad_fallback"] == 1


def test_a_collapsed_scene_under_another_mode_keeps_the_old_path(scene_wav):
    orch = _orchestrator(_Timer(), mode=TimestampMode.ALIGNER_WITH_INTERPOLATION)
    timings = orch._step8_time([[FRAME_A]], [[np.zeros(60640, dtype=np.float32)]], [[TEXT_A]])
    results = orch._step9_reconstruct_and_harden([[FRAME_A]], [[TEXT_A]], _collapsed(), [scene_wav], [6.0],
                                                 [None], None, scene_timings=timings)
    assert orch.sentinel_stats["recovery_strategies"]["vad_fallback"] == 0
    assert "".join(t for t, _s, _e in _subs(results)) == TEXT_A


def test_without_a_timer_the_result_is_todays(scene_wav):
    orch = _orchestrator()
    assert orch._step8_time([[FRAME_A]], [[np.zeros(10, dtype=np.float32)]], [[TEXT_A]]) is None
    results = orch._step9_reconstruct_and_harden([[FRAME_A]], [[TEXT_A]], _aligned(), [scene_wav], [6.0],
                                                 [None], None)
    assert _subs(results) == [(TEXT_A, 1.26, 4.79)]
    assert results[0][1]["timing_model"] is None


def test_a_failed_timer_load_keeps_the_subtitles_and_says_why(scene_wav):
    orch = _orchestrator(_Timer(load_error=OSError("Connection broken\nsecond line")))
    timings = orch._step8_time([[FRAME_A]], [[np.zeros(10, dtype=np.float32)]], [[TEXT_A]])
    assert timings is None and orch.timer.unloaded == 1
    assert orch.timing_unavailable.startswith(
        "ChronosJAV's timing model (timing model) could not be loaded (OSError: Connection broken)")
    assert orch.timing_unavailable.endswith("the text is kept.")
    assert "second line" not in orch.timing_unavailable
    results = orch._step9_reconstruct_and_harden([[FRAME_A]], [[TEXT_A]], _aligned(), [scene_wav], [6.0],
                                                 [None], None, scene_timings=timings)
    assert _subs(results) == [(TEXT_A, 1.26, 4.79)]


def test_a_frame_with_text_but_no_word_times_keeps_its_text(scene_wav):
    frames = [[TemporalFrame(0.1, 0.9), FRAME_A]]
    texts = [["出身福島なんですね", TEXT_A]]
    aligned = [[[], _aligned()[0][0]]]
    results = _orchestrator()._step9_reconstruct_and_harden(frames, texts, aligned, [scene_wav], [6.0],
                                                            [None], None)
    subs = _subs(results)
    assert subs[0] == ("出身福島なんですね", 0.1, 0.9) and subs[1][0] == TEXT_A


def test_a_frame_without_word_times_is_not_split_but_its_start_moves(scene_wav):
    orch = _orchestrator(_Timer())
    timings = orch._step8_time([[FRAME_A]], [[np.zeros(10, dtype=np.float32)]], [[TEXT_A]])
    results = orch._step9_reconstruct_and_harden([[FRAME_A]], [[TEXT_A]], [[[]]], [scene_wav], [6.0],
                                                 [None], None, scene_timings=timings)
    assert _subs(results) == [(TEXT_A, 1.32, 4.79)]


def test_aligner_free_the_start_moves_and_nothing_is_split(scene_wav):
    orch = _orchestrator(_Timer(), mode=TimestampMode.VAD_ONLY, aligner=None)
    timings = orch._step8_time([[FRAME_A]], [[np.zeros(10, dtype=np.float32)]], [[TEXT_A]])
    results = orch._step9_reconstruct_and_harden([[FRAME_A]], [[TEXT_A]], None, [scene_wav], [6.0],
                                                 [None], None, scene_timings=timings)
    assert _subs(results) == [(TEXT_A, 1.32, 4.79)]
    assert results[0][1]["timing_model"]["starts_moved"] == 1


def test_frames_without_text_are_not_timed():
    timer = _Timer()
    timings = _orchestrator(timer)._step8_time([[FRAME_A, TemporalFrame(5.0, 5.5)]],
                                               [[np.zeros(10, dtype=np.float32)] * 2], [[TEXT_A, "  "]])
    assert len(timer.timed) == 1 and timings[0][1] is None


def test_pathless_mode_stays_with_an_array_capable_timer():
    import inspect

    src = inspect.getsource(DecoupledSubtitlePipeline._step1_frame_and_slice)
    assert 'self.timer is None or getattr(self.timer, "accepts_array", False)' in src


def test_the_note_is_cleared_for_the_next_file():
    import inspect

    assert "self.timing_unavailable = None" in inspect.getsource(DecoupledSubtitlePipeline.process_scenes)
