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

from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline, line_filter_test
from whisperjav.modules.subtitle_pipeline.timing_rules import TimedSegment
from whisperjav.modules.subtitle_pipeline.types import HardeningConfig, RegroupMode, TemporalFrame, TimestampMode

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


def _orchestrator(timer=None, mode=TimestampMode.ALIGNER_WITH_VAD_FALLBACK, aligner=object(),
                  regroup=RegroupMode.OFF):
    return DecoupledSubtitlePipeline(
        framer=object(), generator=object(), cleaner=object(), aligner=aligner,
        hardening_config=HardeningConfig(timestamp_mode=mode, regroup_mode=regroup), timer=timer,
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
    # T2/T6 since tuning (owner W296 1): each part starts at its first character's aligner time.
    assert _subs(results) == [("近くのカラオケでそのまま酔っぱらっちゃった。", 1.26, 4.10), ("それで？", 4.10, 4.79)]
    assert results[0][1]["timing_model"] == {"frames_timed": 1, "starts_moved": 0, "splits": 1,
                                             "split_times": [4.1]}
    assert orch.timer.loaded == 1 and orch.timer.unloaded == 1


class TestLineFilterTest:
    """T9 (owner W296 2, W300 3): the parts a split must not leave are the ones the Phase 8 line filters of the
    run remove (qwen_pipeline.py Phase 8), following the run's settings."""

    def test_the_non_linguistic_filter_always(self):
        for test in (line_filter_test(), line_filter_test(ellipsis_filter=True, nonverbal_filter=False)):
            assert test("はぁはぁ…") and test("  ")
            assert not test("他にはどんな体位をしたんです?") and not test("それで？")

    def test_the_ellipsis_filter_only_for_anime_whisper(self):
        assert line_filter_test(ellipsis_filter=True, nonverbal_filter=False)("…」")
        assert not line_filter_test(ellipsis_filter=False, nonverbal_filter=False)("…」")

    def test_the_non_verbal_filter_and_its_punctuation_rule_only_when_on(self):
        on, off = line_filter_test(nonverbal_filter=True), line_filter_test(nonverbal_filter=False)
        for text in ("はい。", "うん。", "。", "、", "！？", "♪"):
            assert on(text), text
            assert not off(text), text


def test_the_pipeline_passes_the_runs_filters():
    from pathlib import Path

    src = (Path(__file__).parent.parent / "whisperjav" / "pipelines" / "qwen_pipeline.py").read_text(encoding="utf-8")
    assert ('line_filter=line_filter_test(\n                ellipsis_filter=self.generator_backend == "anime-whisper",\n'
            '                nonverbal_filter=self.drop_nonverbal_lines,\n            ),') in src.replace("\r\n", "\n")


MOAN_WORDS = [("は", 0.10, 0.12), ("ぁ", 0.20, 0.22), ("は", 0.30, 0.32), ("ぁ…", 0.40, 0.42), ("他", 1.00, 1.02),
              ("に", 1.10, 1.12), ("は", 1.20, 1.22), ("ど", 1.30, 1.32), ("ん", 1.40, 1.42), ("な", 1.50, 1.52),
              ("体", 1.60, 1.62), ("位", 1.70, 1.72), ("を", 1.80, 1.82), ("し", 1.90, 1.92), ("た", 2.00, 2.02),
              ("ん", 2.10, 2.12), ("で", 2.20, 2.22), ("す", 2.30, 2.32), ("?", 2.40, 2.80)]
TWO_ONSETS = [TimedSegment(0.10, 0.0, True), TimedSegment(1.00, 0.0, True)]


def _texts(groups):
    return ["".join(w["word"] for w in g) for g in groups]


def test_a_split_never_leaves_a_moan_alone():
    """The case found in tuning (anime-whisper, The.Naked.Director S02E02 Scene 5): the split after "…" put
    "はぁはぁ…" on its own subtitle and the non-linguistic filter deleted it. Now it stays with what follows."""
    words = [[{"word": w, "start": s, "end": e} for w, s, e in MOAN_WORDS]]
    frame = TemporalFrame(1.0, 3.8)
    groups, info = DecoupledSubtitlePipeline._timed_frame_groups(
        [frame], words, None, [TWO_ONSETS], removable=line_filter_test(ellipsis_filter=True))
    assert _texts(groups) == ["はぁはぁ…他にはどんな体位をしたんです?"]
    assert info["splits"] == 0
    groups, _info = DecoupledSubtitlePipeline._timed_frame_groups([frame], words, None, [TWO_ONSETS])
    assert _texts(groups) == ["はぁはぁ…", "他にはどんな体位をしたんです?"]  # without the test


def test_the_orchestrator_applies_its_line_filter_test(scene_wav):
    text = "".join(w for w, _s, _e in MOAN_WORDS)
    frame = TemporalFrame(1.0, 3.8)
    aligned = [[[{"word": w, "start": s, "end": e} for w, s, e in MOAN_WORDS]]]
    for orch, want in ((_orchestrator(_Timer(TWO_ONSETS)), [text]),
                       (DecoupledSubtitlePipeline(framer=object(), generator=object(), cleaner=object(),
                                                  aligner=object(), hardening_config=HardeningConfig(),
                                                  timer=_Timer(TWO_ONSETS), line_filter=lambda t: False),
                        ["はぁはぁ…", "他にはどんな体位をしたんです?"])):
        timings = orch._step8_time([[frame]], [[np.zeros(44800, dtype=np.float32)]], [[text]])
        results = orch._step9_reconstruct_and_harden([[frame]], [[text]], aligned, [scene_wav], [6.0],
                                                     [None], None, scene_timings=timings)
        assert [t for t, _s, _e in _subs(results)] == want


def test_a_lone_hai_is_split_off_only_when_no_filter_removes_it():
    """W300 3: with the non-verbal filter on, a lone はい。 stays in its neighbour's subtitle, as in 1.9.5."""
    words = [[{"word": w, "start": s, "end": e}
              for w, s, e in [("は", 0.10, 0.12), ("い。", 0.30, 0.32)] + MOAN_WORDS[4:]]]
    frame = TemporalFrame(1.0, 3.8)
    kept, _ = DecoupledSubtitlePipeline._timed_frame_groups(
        [frame], words, None, [TWO_ONSETS], removable=line_filter_test(nonverbal_filter=True))
    split, _ = DecoupledSubtitlePipeline._timed_frame_groups(
        [frame], words, None, [TWO_ONSETS], removable=line_filter_test(nonverbal_filter=False))
    assert _texts(kept) == ["はい。他にはどんな体位をしたんです?"]
    assert _texts(split) == ["はい。", "他にはどんな体位をしたんです?"]


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
    subs = _subs(results)
    assert "".join(t for t, _s, _e in subs) == TEXT_A
    # The piled-up aligner times are not used for the start: it moves to Whisper's onset (T2 for a collapsed
    # scene), 1.0 + 0.32 s, not the aligner's 1.0 s.
    assert subs[0][1] == 1.32
    assert results[0][1]["timing_aligner_native"] == 0


def test_with_regrouping_the_start_is_the_aligners(scene_wav):
    """W296 1, W300 4: with regrouping on, the regrouping makes the splits and the start is the aligner's;
    the timing model does not move it to Whisper's onset (1.32 s)."""
    orch = _orchestrator(_Timer(), regroup=RegroupMode.STANDARD)
    timings = orch._step8_time([[FRAME_A]], [[np.zeros(60640, dtype=np.float32)]], [[TEXT_A]])
    results = orch._step9_reconstruct_and_harden([[FRAME_A]], [[TEXT_A]], _aligned(), [scene_wav], [6.0],
                                                 [None], None, scene_timings=timings)
    subs = _subs(results)
    assert "".join(t for t, _s, _e in subs) == TEXT_A
    assert subs[0][1] == 1.26
    assert results[0][1]["timing_model"]["starts_moved"] == 0


def test_every_run_records_the_vad_group_edges(scene_wav):
    """F7, owner W307 1: the scene record keeps each VAD group's start and end (the audio sent to the ASR) in every
    run, not only in vad_only runs; the refined display start only where it is used."""
    orch = _orchestrator(_Timer())
    timings = orch._step8_time([[FRAME_A]], [[np.zeros(60640, dtype=np.float32)]], [[TEXT_A]])
    results = orch._step9_reconstruct_and_harden([[FRAME_A]], [[TEXT_A]], _aligned(), [scene_wav], [6.0],
                                                 [None], None, scene_timings=timings)
    assert results[0][1]["frame_timing"] == [{"vad_start": 1.0, "vad_end": 4.79, "speech_start": None,
                                              "internal_breaks": []}]


class TestRunSummaryCounts:
    """W300 1: the scene record counts the subtitles whose times came from the aligner."""

    def test_aligned_subtitles_count_and_a_text_only_frame_does_not(self, scene_wav):
        frames = [[TemporalFrame(0.1, 0.9), FRAME_A]]
        texts = [["出身福島なんですね", TEXT_A]]
        orch = _orchestrator(_Timer())
        timings = orch._step8_time(frames, [[np.zeros(12800, dtype=np.float32), np.zeros(60640, dtype=np.float32)]],
                                   texts)
        results = orch._step9_reconstruct_and_harden(frames, texts, [[[], _aligned()[0][0]]], [scene_wav], [6.0],
                                                     [None], None, scene_timings=timings)
        diag = results[0][1]
        assert diag["timing_total_segments"] == 3  # the text-only frame, then group A in two
        assert diag["timing_aligner_native"] == 2 and diag["timing_vad_fallback"] == 1

    def test_a_collapsed_scene_counts_nothing_for_the_aligner(self, scene_wav):
        orch = _orchestrator(_Timer())
        timings = orch._step8_time([[FRAME_A]], [[np.zeros(60640, dtype=np.float32)]], [[TEXT_A]])
        results = orch._step9_reconstruct_and_harden([[FRAME_A]], [[TEXT_A]], _collapsed(), [scene_wav], [6.0],
                                                     [None], None, scene_timings=timings)
        diag = results[0][1]
        assert diag["timing_aligner_native"] == 0 and diag["timing_vad_fallback"] == diag["timing_total_segments"] == 1

    def test_vad_only_counts_nothing_for_the_aligner(self, scene_wav):
        orch = _orchestrator(mode=TimestampMode.VAD_ONLY, aligner=None)
        results = orch._step9_reconstruct_and_harden([[FRAME_A]], [[TEXT_A]], None, [scene_wav], [6.0],
                                                     [None], None)
        diag = results[0][1]
        assert diag["timing_aligner_native"] == 0 and diag["timing_vad_fallback"] == diag["timing_total_segments"] == 1


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
