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
                                  transcribe_error=None, result=[binding_segment()], calls=[])

    class Session:
        def __init__(self, model_path, lib_path=None, n_threads=4, backend=None, *, use_gpu=None, verbosity=0):
            state.opened.append({"path": model_path, "n_threads": n_threads, "backend": backend, "use_gpu": use_gpu})
            if state.open_error is not None:
                raise state.open_error
            self.backend = state.backend

        def transcribe(self, pcm, sample_rate=16000, *, language=None):
            state.transcribed.append({"len": len(pcm), "dtype": pcm.dtype, "rate": sample_rate,
                                      "language": language})
            state.calls.append(("transcribe", len(pcm)))
            if state.transcribe_error is not None:
                raise state.transcribe_error
            return state.result

        def set_max_new_tokens(self, n):
            state.calls.append(("set_max_new_tokens", n))

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


class TestSplitCharacters:
    """A byte-level BPE token (Whisper) can split one character over two tokens; the binding's text then shows
    U+FFFD and raw_text holds the bytes (found by the review of W412: "監督" written as four U+FFFD)."""

    @staticmethod
    def tok(raw, start, end, conf=0.9):
        return types.SimpleNamespace(text=raw.decode("utf-8", "replace"), raw_text=raw, start=start, end=end,
                                     confidence=conf)

    def test_a_character_split_over_two_tokens_is_whole_again(self):
        kan, toku = "監".encode(), "督".encode()
        toks = [self.tok(kan[:2], 0.10, 0.20), self.tok(kan[2:] + toku[:1], 0.20, 0.30), self.tok(toku[1:], 0.30, 0.42)]
        out = engine.tokens_from_binding(toks)
        assert "".join(t.text for t in out) == "監督"
        assert all("�" not in t.text for t in out)
        assert [(t.text, t.start, t.end) for t in out] == [("監", 0.20, 0.30), ("督", 0.30, 0.42)]

    def test_whole_tokens_are_kept_as_they_are(self):
        toks = [self.tok("本番".encode(), 0.0, 0.3), self.tok("を".encode(), 0.3, 0.4, -1.0)]
        out = engine.tokens_from_binding(toks)
        assert out == (CrispToken("本番", 0.0, 0.3, 0.9), CrispToken("を", 0.3, 0.4, None))

    def test_a_token_without_bytes_keeps_its_text(self):
        out = engine.tokens_from_binding([types.SimpleNamespace(text="監督", start=0.1, end=0.5, confidence=0.8)])
        assert out == (CrispToken("監督", 0.1, 0.5, 0.8),)


@pytest.fixture(autouse=True)
def _crispasr_cutting_as_shipped(monkeypatch):
    """CrispASR's own switches for its 30 s cutting are not set (the token limit follows them)."""
    for name in ("CRISPASR_SESSION_AUTOCHUNK", "CRISPASR_SESSION_CHUNK_SECONDS"):
        monkeypatch.delenv(name, raising=False)


class TestTokenLimit:
    """Owner W406 i1: MOSS-CrispASR limits the tokens of each call to its VAD group's length; W505 2 a: counting at
    most the 30 s piece CrispASR decodes at a time, and at least the row's floor (MOSS 128)."""

    def test_no_rate_leaves_the_models_own_limit(self, binding, tmp_path):
        e = BindingEngine(tmp_path / "m.gguf", "parakeet", True)
        e.open()
        e.transcribe(audio(3.0))
        assert binding.calls == [("transcribe", 48000)]

    def test_the_limit_is_set_before_every_call_and_follows_the_audio_length(self, binding, tmp_path):
        binding.backend = "moss-diarize"
        e = BindingEngine(tmp_path / "m.gguf", "moss-diarize", True, tokens_per_second=65536 / 3000)
        e.open()
        e.transcribe(audio(3.0))
        e.transcribe(audio(4.0))
        assert binding.calls == [("set_max_new_tokens", 66), ("transcribe", 48000),
                                 ("set_max_new_tokens", 88), ("transcribe", 64000)]

    def test_a_short_group_gets_at_least_the_minimum(self, binding, tmp_path):
        e = BindingEngine(tmp_path / "m.gguf", "parakeet", True, tokens_per_second=65536 / 3000)
        e.open()
        e.transcribe(audio(0.5))
        assert binding.calls[0] == ("set_max_new_tokens", engine.MIN_NEW_TOKENS)

    def test_only_moss_crispasr_sets_a_rate(self):
        # The pipeline modules directly, not the pass worker (CLAUDE.md: importing the pass worker can hang).
        import importlib
        from whisperjav.crispasr.manifest import CRISPASR_PIPELINES
        from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline
        rates = {}
        for name in CRISPASR_PIPELINES:
            module = importlib.import_module("whisperjav.pipelines." + name.replace("-", "_") + "_pipeline")
            (cls,) = [c for c in vars(module).values() if isinstance(c, type) and issubclass(c, CrispASRPipeline)
                      and c is not CrispASRPipeline]
            rates[cls.PIPELINE_NAME] = cls.ENGINE.tokens_per_second
        assert set(rates) == set(CRISPASR_PIPELINES)
        assert rates.pop("moss-crispasr") == 20.0
        assert set(rates.values()) == {None}

    def test_the_limit_counts_at_most_crispasrs_piece_and_has_a_floor(self, tmp_path):
        e = BindingEngine(tmp_path / "m.gguf", "moss-diarize", True, tokens_per_second=20.0, token_floor=128)
        limits = [e.token_limit(int(s * 16000)) for s in (0.5, 3.5, 6.4, 6.5, 29.9, 30.0, 137.0, 240.0)]
        assert limits == [128, 128, 128, 130, 598, 600, 600, 600]

    def test_the_limit_follows_crispasrs_own_switches(self, tmp_path, monkeypatch):
        e = BindingEngine(tmp_path / "m.gguf", "moss-diarize", True, tokens_per_second=20.0, token_floor=128)
        monkeypatch.setenv("CRISPASR_SESSION_CHUNK_SECONDS", "60")
        assert e.token_limit(16000 * 137) == 1200
        monkeypatch.setenv("CRISPASR_SESSION_CHUNK_SECONDS", "2")        # CrispASR: at least 5 s
        assert engine.session_piece_seconds() == 5.0
        monkeypatch.setenv("CRISPASR_SESSION_AUTOCHUNK", "0")            # no cutting: the whole VAD group
        assert e.token_limit(16000 * 137) == 2740
        assert engine.session_piece_seconds() is None
        monkeypatch.setenv("CRISPASR_SESSION_AUTOCHUNK", "1")
        assert engine.session_piece_seconds() == 5.0

    def test_the_backends_that_cut_their_audio_themselves(self):
        assert engine.session_piece_seconds() == 30.0
        for backend in ("parakeet", "reazonspeech", "index-echo"):
            assert engine.session_piece_seconds(backend) is None
        assert engine.session_piece_seconds("moss-diarize") == 30.0


class TestOlderCrispASR:
    """Owner W505 3: a MOSS pass that asks for sampling or beam search on a CrispASR older than 0.8.50 says once
    that they are not applied (0.8.44 accepts the calls and ignores them)."""

    PRESET = {"do_sample": True, "temperature": 0.7, "top_k": 50, "seed": 42, "beam_size": 1}
    LINE = "CrispASR 0.8.44 is older than 0.8.50: sampling and beam size are not applied."

    def test_the_line(self):
        line = engine.older_decoding_line
        assert line(self.PRESET, (0, 8, 44)) == self.LINE
        assert line({**self.PRESET, "do_sample": False, "beam_size": 3}, (0, 8, 44)) == self.LINE
        assert line({**self.PRESET, "do_sample": False}, (0, 8, 44)) is None     # greedy: what 0.8.44 does
        for found in ((0, 8, 50), (0, 8, 51), (0, 9, 0), (1, 0, 0), None):
            assert line(self.PRESET, found) is None, found
        assert line(None, (0, 8, 44)) is None

    def test_the_installed_version_is_read_from_the_package(self, monkeypatch):
        import importlib.metadata as md
        monkeypatch.setattr(md, "version", lambda name: "0.8.44+cuda")
        assert engine.installed_version() == (0, 8, 44)
        monkeypatch.setattr(md, "version", lambda name: (_ for _ in ()).throw(md.PackageNotFoundError(name)))
        assert engine.installed_version() is None

    def test_open_says_it_once(self, binding, tmp_path, monkeypatch):
        import logging

        class Session(sys.modules["crispasr"].Session):
            def set_beam_size(self, n): pass
            def set_temperature(self, t, seed=0): pass
            def set_top_k(self, k): pass
            def set_do_sample(self, on): pass

        monkeypatch.setitem(sys.modules, "crispasr", types.SimpleNamespace(Session=Session))
        monkeypatch.setattr(engine, "installed_version", lambda: (0, 8, 44))
        monkeypatch.setattr(engine, "_LINES_SHOWN", set())
        seen = []
        handler = logging.Handler()
        handler.emit = lambda record: seen.append(record.getMessage())
        engine.logger.addHandler(handler)
        try:
            binding.backend = "moss-diarize"
            e = BindingEngine(tmp_path / "m.gguf", "moss-diarize", True, decoding=self.PRESET)
            e.open()
            e.open()                                   # already open: nothing again
            e.transcribe(audio(1.0))
            e.close()
            e.open()                                   # reopened for the next file (a speech enhancer): not again
            BindingEngine(tmp_path / "m.gguf", "moss-diarize", True, decoding=self.PRESET).open()   # nor a new engine
        finally:
            engine.logger.removeHandler(handler)
        assert seen.count(self.LINE) == 1


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
