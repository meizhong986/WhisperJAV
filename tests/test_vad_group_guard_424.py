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
