"""
The CrispASR aligner (whisperjav/crispasr/aligner.py) and ChronosJAV's adapter for it
(whisperjav/modules/subtitle_pipeline/aligners/crispasr.py); item 12 C1, design note
docs/plans/design/2026-10-28-chronosjav-timing-c3-c1.md section 4.

CrispASR's binding is replaced by a stand-in ``crispasr`` module that answers as the real one did in probe
P1 (2026-10-27): one unit per Japanese character with the mark after it attached ("た。"), spike times for
wav2vec2. Model download, staging and the device check are replaced too, so no library is loaded, no file
is fetched and nothing outside tmp_path is touched.

Run with: python -m pytest tests/test_crispasr_aligner.py -q
"""
import sys
import types

import numpy as np
import pytest

from whisperjav.crispasr import aligner as aligner_module
from whisperjav.crispasr import install
from whisperjav.crispasr.aligner import CrispAligner, TimedUnit, units_from_binding
from whisperjav.crispasr.errors import CrispASRSetupError
from whisperjav.crispasr.manifest import ALIGNER_MODELS


def unit(text, start, end):
    return types.SimpleNamespace(text=text, start=start, end=end, characters=[])


# Probe P1, wav2vec2-ja on S02E02 Scene 5, 6.40-10.19 s (shortened): spike times, marks attached.
P1_TEXT = "酔っぱらっちゃった。それで？"
P1_UNITS = [unit("酔", 1.78, 1.80), unit("っ", 1.86, 1.88), unit("ぱ", 1.90, 1.92), unit("ら", 2.08, 2.10),
            unit("っ", 2.32, 2.34), unit("ち", 2.36, 2.38), unit("ゃ", 2.40, 2.42), unit("っ", 2.60, 2.64),
            unit("た。", 2.66, 2.68), unit("そ", 3.10, 3.12), unit("れ", 3.34, 3.36), unit("で？", 3.76, 3.78)]


class TestUnitsFromBinding:
    def test_the_probe_shape_keeps_every_character_and_the_marks_with_the_character_before(self):
        units = units_from_binding(P1_TEXT, P1_UNITS, audio_s=3.79)
        assert "".join(u.text for u in units) == P1_TEXT
        assert units[8] == TimedUnit("た。", 2.66, 2.68)
        assert units[9].start == 3.10  # the 0.42 s pause at the sentence change stays visible

    def test_the_last_unit_ends_at_the_end_of_the_audio(self):
        units = units_from_binding(P1_TEXT, P1_UNITS, audio_s=3.79)
        assert units[-1] == TimedUnit("で？", 3.76, 3.79)
        assert all(u.end == b.end for u, b in zip(units[:-1], P1_UNITS[:-1]))  # inner units as given

    def test_a_mark_the_aligner_left_out_goes_with_the_unit_before_it(self):
        units = units_from_binding("はい、そう。", [unit("は", 0.1, 0.12), unit("い", 0.2, 0.22),
                                                   unit("そ", 0.6, 0.62), unit("う", 0.7, 0.72)], audio_s=1.0)
        assert [u.text for u in units] == ["は", "い、", "そ", "う。"]

    def test_a_space_goes_with_the_unit_before_it(self):
        units = units_from_binding("あ い", [unit("あ", 0.1, 0.12), unit("い", 0.5, 0.52)], audio_s=1.0)
        assert [u.text for u in units] == ["あ ", "い"]

    def test_an_opening_bracket_goes_with_the_unit_after_it(self):
        units = units_from_binding("あ。「い」", [unit("あ。「", 0.1, 0.12), unit("い」", 0.5, 0.52)], audio_s=1.0)
        assert [u.text for u in units] == ["あ。", "「い」"]
        units = units_from_binding("「あ」", [unit("あ", 0.1, 0.12)], audio_s=1.0)
        assert [u.text for u in units] == ["「あ」"]

    def test_units_that_do_not_spell_the_text_give_no_alignment(self):
        assert units_from_binding("あいう", [unit("a", 0.1, 0.2), unit("i", 0.3, 0.4)], audio_s=1.0) == []
        assert units_from_binding("あいう", [unit("あ", 0.1, 0.2)], audio_s=1.0) == []  # a character dropped
        assert units_from_binding("あい", [unit("あ", 0.1, 0.2), unit("い", 0.3, 0.4), unit("う", 0.5, 0.6)],
                                  audio_s=1.0) == []

    def test_empty_input_gives_no_alignment(self):
        assert units_from_binding("", P1_UNITS, 1.0) == []
        assert units_from_binding("  ", P1_UNITS, 1.0) == []
        assert units_from_binding(P1_TEXT, [], 1.0) == []

    def test_an_end_before_its_start_is_raised_to_the_start(self):
        units = units_from_binding("あい", [unit("あ", 0.5, 0.4), unit("い", 0.6, 0.62)], audio_s=None)
        assert units[0] == TimedUnit("あ", 0.5, 0.5)
        assert units[1].end == 0.62  # no audio length given: the last end is the aligner's


@pytest.fixture
def binding(monkeypatch, tmp_path):
    """A stand-in crispasr module and stand-ins for the download, staging and device check."""
    state = types.SimpleNamespace(calls=[], freed=0, result=P1_UNITS, error=None, warnings=[], infos=[],
                                  runtime=install.RuntimeChoice(True, "cuda", None, False, "0.8.44+cuda", "cuda"),
                                  auto_runtime=None, staged=[])

    def align_words(model, text, pcm, *, t_offset=0.0, n_threads=4, lib_path=None):
        state.calls.append({"model": model, "text": text, "len": len(pcm), "dtype": pcm.dtype,
                            "n_threads": n_threads})
        if state.error is not None:
            raise state.error
        return state.result

    def free_aligner_cache(*, lib_path=None):
        state.freed += 1

    monkeypatch.setitem(sys.modules, "crispasr",
                        types.SimpleNamespace(align_words=align_words, free_aligner_cache=free_aligner_cache))

    def runtime_status(device=None):
        if (device or "auto") == "cpu":
            return install.RuntimeChoice(False, "cpu", install.REASON_DEVICE_CPU, False, "0.8.44+cuda", "cuda")
        return state.runtime

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
    monkeypatch.setattr(aligner_module, "logger", types.SimpleNamespace(
        warning=state.warnings.append, info=state.infos.append, debug=lambda *a, **k: None))
    return state


def audio(seconds=3.79):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


class TestCrispAligner:
    def test_load_stages_the_file_under_its_own_name_and_align_passes_it(self, binding, tmp_path):
        a = CrispAligner(temp_dir=tmp_path)
        a.load()
        assert binding.staged[0]["keep_name"] is True
        assert binding.staged[0]["path"].name == ALIGNER_MODELS["wav2vec2-ja"].filename
        units = a.align(audio(), P1_TEXT)
        assert binding.calls[0]["model"].endswith("wav2vec2-large-xlsr-53-japanese-q4_k.gguf")
        assert binding.calls[0]["text"] == P1_TEXT and binding.calls[0]["dtype"] == np.float32
        assert units[-1].end == pytest.approx(3.79)

    def test_align_before_load_is_an_error(self, binding):
        with pytest.raises(RuntimeError, match="not loaded"):
            CrispAligner().align(audio(), P1_TEXT)

    def test_empty_text_or_audio_is_not_sent(self, binding, tmp_path):
        a = CrispAligner(temp_dir=tmp_path)
        a.load()
        assert a.align(audio(), " ") == [] and a.align(audio(0), P1_TEXT) == []
        assert binding.calls == []

    def test_a_failed_call_gives_no_alignment(self, binding, tmp_path):
        a = CrispAligner(temp_dir=tmp_path)
        a.load()
        binding.error = RuntimeError("crispasr_align_words_abi failed")
        assert a.align(audio(), P1_TEXT) == []

    def test_unload_frees_the_cache_once_and_is_safe_twice(self, binding, tmp_path):
        a = CrispAligner(temp_dir=tmp_path)
        a.load()
        a.unload()
        a.unload()
        assert binding.freed == 1 and not a.is_loaded

    def test_the_parakeet_aligner_is_loaded_with_its_warning(self, binding, tmp_path):
        CrispAligner("parakeet-ctc-ja", temp_dir=tmp_path).load()
        assert any("evenly spread" in w for w in binding.warnings)

    def test_device_cpu_on_a_gpu_build_says_the_aligner_still_uses_the_gpu(self, binding, tmp_path):
        CrispAligner(device="cpu", temp_dir=tmp_path).load()
        assert any("still asks for the GPU with --device cpu" in w for w in binding.warnings)

    def test_device_cpu_on_a_cuda_build_without_a_usable_gpu_says_nothing_of_the_gpu(self, binding, tmp_path):
        """Item 12 review: a CUDA build whose NVIDIA GPU cannot be used runs on the CPU anyway."""
        binding.runtime = install.RuntimeChoice(False, "cpu", "no NVIDIA GPU was found", False,
                                                "0.8.44+cuda", "cuda")
        CrispAligner(device="cpu", temp_dir=tmp_path).load()
        assert binding.warnings == []

    def test_on_the_gpu_the_device_line_is_logged(self, binding, tmp_path):
        CrispAligner(temp_dir=tmp_path).load()
        assert binding.warnings == [] and any("NVIDIA GPU" in i for i in binding.infos)

    def test_an_unknown_model_name_is_refused_with_the_choices(self):
        with pytest.raises(ValueError, match="wav2vec2-ja, parakeet-ctc-ja, qwen3-fa-gguf"):
            CrispAligner("nope")

    def test_a_crispasr_without_align_words_is_a_setup_error(self, binding, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "crispasr", types.SimpleNamespace())
        with pytest.raises(CrispASRSetupError, match="no forced alignment"):
            CrispAligner(temp_dir=tmp_path).load()


class TestChronosJAVAdapter:
    def make(self, tmp_path, **kwargs):
        from whisperjav.modules.subtitle_pipeline.aligners.factory import TextAlignerFactory

        return TextAlignerFactory.create("crispasr", temp_dir=tmp_path, **kwargs)

    def test_the_factory_makes_it_and_it_takes_arrays(self, binding, tmp_path):
        from whisperjav.modules.subtitle_pipeline.aligners.crispasr import CrispASRAligner

        a = self.make(tmp_path)
        assert isinstance(a, CrispASRAligner) and a.accepts_array

    def test_align_batch_gives_one_word_per_character_with_the_marks(self, binding, tmp_path):
        a = self.make(tmp_path)
        a.load()
        results = a.align_batch([audio(), audio()], [P1_TEXT, ""])
        words = results[0].words
        assert [w.word for w in words][8:10] == ["た。", "そ"]
        assert words[-1].end == pytest.approx(3.79)
        assert results[0].metadata["aligner"] == "crispasr:wav2vec2-ja"
        assert results[1].words == [] and results[1].metadata["skipped"] is True
        assert len(binding.calls) == 1  # the empty text is not sent

    def test_a_wav_path_is_read_and_resampled_to_16_khz(self, binding, tmp_path):
        import soundfile as sf

        wav = tmp_path / "frame.wav"
        sf.write(str(wav), np.zeros(48000, dtype=np.float32), 48000)  # 1 s at 48 kHz
        a = self.make(tmp_path)
        a.load()
        a.align(wav, P1_TEXT)
        assert binding.calls[0]["len"] == 16000

    def test_before_load_align_batch_is_an_error(self, binding, tmp_path):
        with pytest.raises(RuntimeError, match="before load"):
            self.make(tmp_path).align_batch([audio()], [P1_TEXT])

    def test_a_failed_load_raises_so_the_orchestrator_keeps_the_text(self, binding, monkeypatch, tmp_path):
        def no_runtime(device=None):
            raise CrispASRSetupError("CrispASR is not installed in this WhisperJAV.")

        monkeypatch.setattr(install, "runtime_status", no_runtime)
        a = self.make(tmp_path)
        with pytest.raises(CrispASRSetupError):
            a.load()
        assert not a.is_loaded

    def test_cleanup_unloads(self, binding, tmp_path):
        a = self.make(tmp_path)
        a.load()
        a.cleanup()
        assert not a.is_loaded and binding.freed == 1

    def test_an_unknown_model_is_refused_when_made(self, tmp_path):
        with pytest.raises(ValueError, match="not a CrispASR aligner"):
            self.make(tmp_path, model="nope")
