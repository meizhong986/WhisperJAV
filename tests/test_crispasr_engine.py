"""
The ASR call behind one contract (whisperjav/crispasr/engine.py; route B design 5).

CrispASR's binding is replaced by a stand-in ``crispasr`` module whose Session records what it is asked
and answers as the real one did in the binding check of 2026-10-24. No library is loaded.

Run with: python -m pytest tests/test_crispasr_engine.py -q
"""
import sys
import types

import numpy as np
import pytest

from whisperjav.crispasr import engine
from whisperjav.crispasr.engine import BindingEngine, CrispSegment, CrispToken, CrispWord
from whisperjav.crispasr.errors import CrispASRRunError, CrispASRSetupError


def binding_segment(text="監督。", start=1.92, end=2.88):
    word = types.SimpleNamespace(text="督。", start=2.24, end=None, confidence=1.0)
    tokens = [types.SimpleNamespace(text="督", start=2.24, end=2.56, confidence=0.98),
              types.SimpleNamespace(text="。", start=None, end=None, confidence=-1.0)]
    return types.SimpleNamespace(text=text, start=start, end=end, words=[word], tokens=tokens)


@pytest.fixture
def binding(monkeypatch):
    """A stand-in crispasr module."""
    state = types.SimpleNamespace(opened=[], closed=0, transcribed=[], backend="parakeet", open_error=None,
                                  transcribe_error=None, result=[binding_segment()])

    class Session:
        def __init__(self, model_path, lib_path=None, n_threads=4, backend=None, *, use_gpu=None, verbosity=0):
            state.opened.append({"path": model_path, "n_threads": n_threads, "backend": backend, "use_gpu": use_gpu})
            if state.open_error is not None:
                raise state.open_error
            self.backend = state.backend

        def transcribe(self, pcm, sample_rate=16000, *, language=None):
            state.transcribed.append({"len": len(pcm), "dtype": pcm.dtype, "rate": sample_rate,
                                      "language": language})
            if state.transcribe_error is not None:
                raise state.transcribe_error
            return state.result

        def close(self):
            state.closed += 1

    monkeypatch.setitem(sys.modules, "crispasr", types.SimpleNamespace(Session=Session))
    return state


def audio(seconds=1.0):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


class TestOpen:
    def test_open_passes_the_path_threads_and_use_gpu_but_not_the_back_end(self, binding, tmp_path):
        e = BindingEngine(tmp_path / "m.gguf", "parakeet", use_gpu=True, n_threads=6)
        e.open()
        assert binding.opened == [{"path": str(tmp_path / "m.gguf"), "n_threads": 6, "backend": None,
                                   "use_gpu": True}]
        assert e.is_open

    def test_use_gpu_false_is_passed_as_false(self, binding, tmp_path):
        BindingEngine(tmp_path / "m.gguf", "parakeet", use_gpu=False).open()
        assert binding.opened[0]["use_gpu"] is False

    def test_threads_default_to_min_8_and_the_cpu_count(self, binding, tmp_path, monkeypatch):
        monkeypatch.setattr(engine.os, "cpu_count", lambda: 32)
        assert BindingEngine(tmp_path / "m.gguf", "parakeet", True).n_threads == 8
        monkeypatch.setattr(engine.os, "cpu_count", lambda: 2)
        assert BindingEngine(tmp_path / "m.gguf", "parakeet", True).n_threads == 2

    def test_a_session_of_another_back_end_is_closed_and_refused(self, binding, tmp_path):
        binding.backend = "whisper"
        with pytest.raises(CrispASRSetupError, match="'whisper', not 'parakeet'"):
            BindingEngine(tmp_path / "m.gguf", "parakeet", True, label="Parakeet").open()
        assert binding.closed == 1

    def test_a_model_the_library_cannot_open_is_a_setup_error_with_its_message(self, binding, tmp_path):
        binding.open_error = RuntimeError("Failed to open 'x' — backend not supported")
        with pytest.raises(CrispASRSetupError) as e:
            BindingEngine(tmp_path / "m.gguf", "parakeet", True, label="Parakeet TDT").open()
        assert "Parakeet TDT could not be opened" in str(e.value) and "backend not supported" in str(e.value)

    def test_libraries_that_will_not_load_are_a_setup_error(self, binding, tmp_path):
        binding.open_error = OSError("[WinError 126]")
        with pytest.raises(CrispASRSetupError, match="could not be loaded"):
            BindingEngine(tmp_path / "m.gguf", "parakeet", True).open()

    def test_opening_twice_opens_once(self, binding, tmp_path):
        e = BindingEngine(tmp_path / "m.gguf", "parakeet", True)
        e.open()
        e.open()
        assert len(binding.opened) == 1


class TestTranscribe:
    def test_one_call_at_16_khz_in_japanese(self, binding, tmp_path):
        e = BindingEngine(tmp_path / "m.gguf", "parakeet", True)
        e.open()
        e.transcribe(audio(0.5))
        assert binding.transcribed == [{"len": 8000, "dtype": np.float32, "rate": 16000, "language": "ja"}]

    def test_the_result_becomes_whisperjavs_own_types_with_none_kept(self, binding, tmp_path):
        e = BindingEngine(tmp_path / "m.gguf", "parakeet", True)
        e.open()
        (seg,) = e.transcribe(audio())
        assert seg == CrispSegment("監督。", 1.92, 2.88, (CrispWord("督。", 2.24, None, 1.0),),
                                   (CrispToken("督", 2.24, 2.56, 0.98), CrispToken("。", None, None, None)))

    @pytest.mark.parametrize("error", [RuntimeError("crispasr_session_transcribe failed"),
                                       ValueError("pcm must be one-dimensional")])
    def test_a_failed_call_is_a_run_error(self, binding, tmp_path, error):
        binding.transcribe_error = error
        e = BindingEngine(tmp_path / "m.gguf", "parakeet", True)
        e.open()
        with pytest.raises(CrispASRRunError, match=str(error)):
            e.transcribe(audio())

    def test_a_closed_engine_refuses_before_the_library_is_called(self, binding, tmp_path):
        e = BindingEngine(tmp_path / "m.gguf", "parakeet", True)
        with pytest.raises(CrispASRRunError):
            e.transcribe(audio())
        e.open()
        e.close()
        with pytest.raises(CrispASRRunError):
            e.transcribe(audio())
        assert binding.transcribed == []


class TestClose:
    def test_close_twice_is_safe_and_closes_once(self, binding, tmp_path):
        e = BindingEngine(tmp_path / "m.gguf", "parakeet", True)
        e.open()
        e.close()
        e.close()
        assert binding.closed == 1 and not e.is_open

    def test_cleanup_closes(self, binding, tmp_path):
        e = BindingEngine(tmp_path / "m.gguf", "parakeet", True)
        e.open()
        e.cleanup()
        assert binding.closed == 1
