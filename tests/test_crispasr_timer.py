"""
The CrispASR timing model (whisperjav/crispasr/timer.py) and ChronosJAV's adapter for it
(whisperjav/modules/subtitle_pipeline/timers/crispasr.py); item 12 C3, design note
docs/plans/design/2026-10-28-chronosjav-timing-c3-c1.md sections 5.1 and 14.3.

CrispASR's binding is replaced by a stand-in ``crispasr`` module whose Whisper-only ``CrispASR`` class answers as
the real one did in probe P2c (2026-10-27): special tokens without DTW time, text tokens with one. Download,
staging and the device check are replaced too: no library is loaded, nothing outside tmp_path is touched.

Run with: python -m pytest tests/test_crispasr_timer.py -q
"""
import re
import sys
import types

import numpy as np
import pytest

from whisperjav.crispasr import install, manifest
from whisperjav.crispasr import timer as timer_module
from whisperjav.crispasr.errors import CrispASRSetupError
from whisperjav.crispasr.timer import DECODE_OPTIONS, CrispTimer, WhisperSegment, segments_from_binding


def tok(text, dtw):
    return types.SimpleNamespace(text=text, dtw_time=dtw)


# Probe P2c, Whisper small on group A (shortened).
P2C = [types.SimpleNamespace(text="近くのカラオケでそのまま", no_speech_prob=0.0,
                             tokens=[tok("[_BEG_]", None), tok("近", 0.32), tok("く", 0.42), tok("[_TT_100]", None)]),
       types.SimpleNamespace(text="それで?", no_speech_prob=0.0,
                             tokens=[tok("それで", 3.30), tok("?", 3.76), tok("[_TT_200]", None)])]


class TestSegmentsFromBinding:
    def test_the_onset_is_the_first_text_tokens_dtw_time(self):
        assert segments_from_binding(P2C) == [WhisperSegment(0.32, 0.0, True), WhisperSegment(3.30, 0.0, True)]

    def test_special_and_blank_tokens_are_skipped(self):
        seg = types.SimpleNamespace(text=" あ", no_speech_prob=0.2,
                                    tokens=[tok("[_BEG_]", None), tok(" ", 0.1), tok("あ", 0.5)])
        assert segments_from_binding([seg]) == [WhisperSegment(0.5, 0.2, True)]

    def test_a_segment_without_text_or_times(self):
        seg = types.SimpleNamespace(text="", no_speech_prob=0.9, tokens=[tok("[_TT_0]", None)])
        assert segments_from_binding([seg]) == [WhisperSegment(None, 0.9, False)]


@pytest.fixture
def binding(monkeypatch, tmp_path):
    state = types.SimpleNamespace(opened=[], calls=[], closed=0, result=P2C, error=None, warnings=[], infos=[],
                                  staged=[], open_error=None)

    class CrispASR:
        def __init__(self, model_path, lib_path=None, helpers_lib_path=None, *, dtw=False, dtw_aheads_preset=1,
                     dtw_n_top=1):
            state.opened.append({"path": model_path, "dtw": dtw, "preset": dtw_aheads_preset})
            if state.open_error is not None:
                raise state.open_error

        def transcribe_pcm(self, pcm, sample_rate=16000, language="auto", decode_options=None, **kwargs):
            state.calls.append({"len": len(pcm), "dtype": pcm.dtype, "language": language,
                                "options": decode_options})
            if state.error is not None:
                raise state.error
            return state.result

        def close(self):
            state.closed += 1

    monkeypatch.setitem(sys.modules, "crispasr", types.SimpleNamespace(CrispASR=CrispASR))

    def runtime_status(device=None):
        if (device or "auto") == "cpu":
            return install.RuntimeChoice(False, "cpu", install.REASON_DEVICE_CPU, False, "0.8.44+cuda", "cuda")
        return install.RuntimeChoice(True, "cuda", None, False, "0.8.44+cuda", "cuda")

    def ensure_model(model, offline=None):
        path = tmp_path / "hub" / model.filename
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"x")
        return path

    def stage_model(path, temp_dir, sha256, keep_name=False):
        state.staged.append({"path": path, "keep_name": keep_name})
        return path

    monkeypatch.setattr(install, "runtime_status", runtime_status)
    monkeypatch.setattr(install, "ensure_model", ensure_model)
    monkeypatch.setattr(install, "stage_model", stage_model)
    monkeypatch.setattr(timer_module, "logger", types.SimpleNamespace(
        warning=state.warnings.append, info=state.infos.append, debug=lambda *a, **k: None))
    return state


def audio(seconds=3.79):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


class TestCrispTimer:
    def test_load_opens_whisper_small_with_dtw_preset_8_under_its_own_name(self, binding, tmp_path):
        CrispTimer(temp_dir=tmp_path).load()
        assert binding.staged[0]["keep_name"] is True
        assert binding.opened == [{"path": str(tmp_path / "hub" / "ggml-small.bin"), "dtw": True, "preset": 8}]

    def test_time_decodes_japanese_fast_and_maps_the_segments(self, binding, tmp_path):
        t = CrispTimer(temp_dir=tmp_path)
        t.load()
        assert t.time(audio()) == [WhisperSegment(0.32, 0.0, True), WhisperSegment(3.30, 0.0, True)]
        call = binding.calls[0]
        assert call["language"] == "ja" and call["dtype"] == np.float32
        assert call["options"] == {"token_timestamps": 1, "no_context": 1, "temperature": 0.0, "max_tokens": 96}
        assert call["options"] is not DECODE_OPTIONS  # the module's options are never handed out to be changed

    def test_empty_audio_is_not_sent_and_a_failure_gives_nothing(self, binding, tmp_path):
        t = CrispTimer(temp_dir=tmp_path)
        t.load()
        assert t.time(audio(0)) == [] and binding.calls == []
        binding.error = RuntimeError("Transcription failed (error code 3)")
        assert t.time(audio()) == []

    def test_time_before_load_is_an_error(self, binding):
        with pytest.raises(RuntimeError, match="not loaded"):
            CrispTimer().time(audio())

    def test_unload_closes_once(self, binding, tmp_path):
        t = CrispTimer(temp_dir=tmp_path)
        t.load()
        t.unload()
        t.unload()
        assert binding.closed == 1 and not t.is_loaded

    def test_device_cpu_on_a_gpu_build_says_it_still_uses_the_gpu(self, binding, tmp_path):
        CrispTimer(device="cpu", temp_dir=tmp_path).load()
        assert any("timing model still asks for the GPU with --device cpu" in w for w in binding.warnings)

    def test_device_cpu_on_a_cuda_build_without_a_usable_gpu_says_nothing_of_the_gpu(self, binding, monkeypatch,
                                                                                     tmp_path):
        """Item 12 review: a CUDA build whose NVIDIA GPU cannot be used runs on the CPU anyway."""
        monkeypatch.setattr(install, "runtime_status", lambda device=None: install.RuntimeChoice(
            False, "cpu", "no NVIDIA GPU was found", False, "0.8.44+cuda", "cuda"))
        CrispTimer(device="cpu", temp_dir=tmp_path).load()
        assert binding.warnings == []

    def test_a_close_that_fails_does_not_end_the_file(self, binding, monkeypatch, tmp_path):
        t = CrispTimer(temp_dir=tmp_path)
        t.load()
        monkeypatch.setattr(t._model, "close", lambda: (_ for _ in ()).throw(RuntimeError("free failed")))
        t.unload()
        assert not t.is_loaded

    def test_a_model_that_cannot_be_opened_is_a_setup_error(self, binding, tmp_path):
        binding.open_error = RuntimeError("Failed to load model")
        with pytest.raises(CrispASRSetupError, match="could not be opened"):
            CrispTimer(temp_dir=tmp_path).load()

    def test_a_crispasr_without_its_whisper_class_is_a_setup_error(self, binding, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "crispasr", types.SimpleNamespace())
        with pytest.raises(CrispASRSetupError, match="no Whisper engine"):
            CrispTimer(temp_dir=tmp_path).load()

    def test_an_unknown_model_is_refused_with_the_choices(self):
        with pytest.raises(ValueError, match="choose one of: whisper-small"):
            CrispTimer("kotoba-whisper-v2.0")


class TestModelRow:
    def test_whisper_small_only_pinned(self):
        assert set(manifest.TIMING_MODELS) == {"whisper-small"}  # W255: small only
        m = manifest.TIMING_MODELS["whisper-small"]
        assert re.fullmatch(r"[0-9a-f]{40}", m.revision) and re.fullmatch(r"[0-9a-f]{64}", m.sha256)
        assert (m.repo_id, m.filename, m.size, m.backend) == ("ggerganov/whisper.cpp", "ggml-small.bin",
                                                              487_601_967, "whisper")
        assert manifest.DEFAULT_TIMING_MODEL == "whisper-small"
        assert manifest.timing_model_for(None) is m
        assert set(timer_module.DTW_PRESET) == set(manifest.TIMING_MODELS)


class TestChronosJAVAdapter:
    def test_it_takes_arrays_and_returns_timed_segments(self, binding, tmp_path):
        from whisperjav.modules.subtitle_pipeline.timers.crispasr import CrispASRTimer
        from whisperjav.modules.subtitle_pipeline.timing_rules import TimedSegment

        t = CrispASRTimer(temp_dir=tmp_path)
        assert t.accepts_array
        t.load()
        assert t.time(audio()) == [TimedSegment(0.32, 0.0, True), TimedSegment(3.30, 0.0, True)]
        t.cleanup()
        assert not t.is_loaded and binding.closed == 1

    def test_before_load_time_is_an_error(self, binding, tmp_path):
        from whisperjav.modules.subtitle_pipeline.timers.crispasr import CrispASRTimer

        with pytest.raises(RuntimeError, match="before load"):
            CrispASRTimer(temp_dir=tmp_path).time(audio())
