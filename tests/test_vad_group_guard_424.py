"""#424: a sub-0.1 s speech group (a scene-boundary pad artifact, e.g. 160 samples at 27.99-28.00 s) must not reach
CTranslate2, which divides by zero below one feature frame and is killed by Windows (0xC0000094). The reporter's fix:
clamp the slice to the audio and skip groups under 0.1 s. No model is loaded: the guard returns before any ASR call.
"""
import numpy as np

from whisperjav.modules.faster_whisper_pro_asr import FasterWhisperProASR


def _asr():
    return FasterWhisperProASR.__new__(FasterWhisperProASR)


def test_ten_ms_group_at_the_scene_edge_is_skipped():
    audio = np.zeros(28 * 16000, dtype=np.float32)
    assert _asr()._transcribe_vad_group(audio, 16000, [{"start_sec": 27.99, "end_sec": 28.00}]) == []


def test_group_whose_end_runs_past_the_audio_is_clamped_and_skipped_when_too_short():
    audio = np.zeros(28 * 16000, dtype=np.float32)
    assert _asr()._transcribe_vad_group(audio, 16000, [{"start_sec": 27.95, "end_sec": 28.30}]) == []


# Owner, 2026-10-02: "Both: source + guard"; under 0.1 s "joins the piece next to it".
from whisperjav.modules.faster_whisper_pro_asr import join_short_groups


def _g(*spans):
    return [{"start_sec": s, "end_sec": e} for s, e in spans]


def test_the_ten_ms_group_at_the_scene_end_joins_the_group_before_it():
    groups = [_g((20.0, 24.0)), _g((27.99, 28.00))]
    assert join_short_groups(groups) == [_g((20.0, 24.0), (27.99, 28.00))]


def test_a_short_group_joins_the_nearer_neighbour():
    groups = [_g((0.0, 2.0)), _g((5.00, 5.05)), _g((5.3, 8.0))]
    assert join_short_groups(groups) == [_g((0.0, 2.0)), _g((5.00, 5.05), (5.3, 8.0))]


def test_a_short_first_group_joins_the_next_one_and_long_groups_are_untouched():
    groups = [_g((0.0, 0.05)), _g((1.0, 3.0)), _g((4.0, 6.0))]
    assert join_short_groups(groups) == [_g((0.0, 0.05), (1.0, 3.0)), _g((4.0, 6.0))]
    long_only = [_g((0.0, 1.0)), _g((2.0, 3.0))]
    assert join_short_groups(long_only) == long_only


def test_a_lone_short_group_is_left_for_the_guard():
    assert join_short_groups([_g((27.99, 28.00))]) == [_g((27.99, 28.00))]


def test_the_transcribe_path_joins_before_transcribing():
    import inspect
    src = inspect.getsource(FasterWhisperProASR)
    assert "join_short_groups(self._run_speech_segmentation(audio_data, sample_rate))" in src
