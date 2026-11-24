"""When models are released (1.10.3, owner W764 "A model shall not be loaded into VRAM or RAM before it is needed",
W776; the model-memory inspection of 2026-11-24, F1, F2, F5, F7, and its section 3 for the speech enhancer).

- F1, F5: ChronosJAV's and decoupled's release points give PyTorch's cached GPU memory back also in an Ensemble pass
  process, where the shared gpu_utils.safe_cuda_cleanup() skips it (issue #82); that shared function is unchanged.
- F2, F7: the VAD-grouped framer's speech segmenter is released once every scene is framed, so it never stays beside
  the ASR model, and a step-down re-framing never holds two.
- The timing model's failed calls are named once per file (text approved, W776).
- The speech enhancer is released also when enhancing fails, without hiding the run's own error.

No model is loaded; torch and the speech segmenter are stood in for.
"""
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("soundfile")

from whisperjav.modules.subtitle_pipeline import orchestrator as orch_module
from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
from whisperjav.modules.subtitle_pipeline.types import FramingResult, HardeningConfig, TemporalFrame

REPO = Path(__file__).resolve().parents[1]


class _FakeCuda:
    def __init__(self):
        self.emptied = 0

    def is_available(self):
        return True

    def empty_cache(self):
        self.emptied += 1


@pytest.fixture
def fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = _FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


# --- F1, F5 -----------------------------------------------------------------------------------------------------------

def test_the_subtitle_pipeline_gives_pytorchs_memory_back_in_a_pass_process(monkeypatch, fake_torch):
    monkeypatch.setenv("WHISPERJAV_SUBPROCESS_WORKER", "1")       # an Ensemble pass process
    DecoupledSubtitlePipeline._safe_cuda_cleanup()
    assert fake_torch.cuda.emptied == 1


def test_the_shared_cleanup_still_skips_in_a_pass_process(monkeypatch, fake_torch):
    """balanced, fast and faster go through this shared function: unchanged (issue #82)."""
    from whisperjav.utils.gpu_utils import safe_cuda_cleanup
    monkeypatch.setenv("WHISPERJAV_SUBPROCESS_WORKER", "1")
    safe_cuda_cleanup()
    assert fake_torch.cuda.emptied == 0


def test_without_torch_loaded_the_release_imports_nothing(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    DecoupledSubtitlePipeline._safe_cuda_cleanup()
    assert "torch" not in sys.modules


def test_each_release_point_of_the_subtitle_pipeline_uses_it():
    import inspect
    gen = inspect.getsource(DecoupledSubtitlePipeline._step2_4_generate_and_clean)
    align = inspect.getsource(DecoupledSubtitlePipeline._step5_7_align)
    assert "self.generator.unload()\n                self._safe_cuda_cleanup()" in gen.replace("\r\n", "\n")
    assert align.count("self._safe_cuda_cleanup()") == 2      # after a failed load and after the alignment
    own = inspect.getsource(DecoupledSubtitlePipeline._safe_cuda_cleanup)
    assert "torch.cuda.empty_cache()" in own and "from whisperjav.utils.gpu_utils import" not in own


# --- F2, F7 -----------------------------------------------------------------------------------------------------------

class _Segmenters:
    """Stands in for SpeechSegmenterFactory.create: counts the segmenters alive at once."""

    def __init__(self):
        self.alive = 0
        self.most = 0
        self.made = 0

    def create(self, backend, **kwargs):
        owner = self

        class Segmenter:
            def segment(self, audio, sample_rate=16000):
                return types.SimpleNamespace(groups=[], method="stand-in", num_segments=0, num_groups=0,
                                             speech_coverage_ratio=0.0, parameters={}, error=None)

            def cleanup(self):
                owner.alive -= 1

        self.made += 1
        self.alive += 1
        self.most = max(self.most, self.alive)
        return Segmenter()


@pytest.fixture
def segmenters(monkeypatch):
    from whisperjav.modules.speech_segmentation import factory
    made = _Segmenters()
    monkeypatch.setattr(factory.SpeechSegmenterFactory, "create", staticmethod(made.create))
    return made


@pytest.fixture
def scene_wav(tmp_path):
    import soundfile as sf
    wav = tmp_path / "scene_0000.wav"
    sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000)
    return wav


class _Generator:
    accepts_array = True

    def load(self):
        pass

    def unload(self):
        pass

    def cleanup(self):
        pass


def _orchestrator(framer, tmp_path):
    return DecoupledSubtitlePipeline(framer=framer, generator=_Generator(), cleaner=object(), aligner=None,
                                     hardening_config=HardeningConfig(), artifacts_dir=tmp_path)


def test_step_1_releases_the_speech_segmenter_before_the_asr_model(segmenters, scene_wav, tmp_path):
    from whisperjav.modules.subtitle_pipeline.framers.vad_grouped import VadGroupedFramer
    framer = VadGroupedFramer(segmenter_backend="stand-in")
    orch = _orchestrator(framer, tmp_path)
    orch._step1_frame_and_slice([scene_wav, scene_wav], [1.0, 1.0])
    assert segmenters.made == 1 and segmenters.alive == 0 and framer._segmenter is None


def test_a_step_down_reframing_never_holds_two_segmenters(segmenters, scene_wav, tmp_path):
    from whisperjav.modules.subtitle_pipeline.framers.vad_grouped import VadGroupedFramer
    orch = _orchestrator(VadGroupedFramer(segmenter_backend="stand-in"), tmp_path)
    orch._step1_frame_and_slice([scene_wav], [1.0])
    orch._step1_frame_and_slice([scene_wav], [1.0], framer_override_max_group=3.0)   # the step-down pass
    assert segmenters.most == 1 and segmenters.alive == 0


def test_the_next_file_builds_the_segmenter_again(segmenters, scene_wav, tmp_path):
    from whisperjav.modules.subtitle_pipeline.framers.vad_grouped import VadGroupedFramer
    orch = _orchestrator(VadGroupedFramer(segmenter_backend="stand-in"), tmp_path)
    for _file in range(2):
        orch._step1_frame_and_slice([scene_wav], [1.0])
    assert segmenters.made == 2 and segmenters.most == 1 and segmenters.alive == 0


def test_a_framer_without_a_segmenter_is_released_harmlessly(scene_wav, tmp_path):
    class Framer:
        cleaned = 0

        def frame(self, audio, sr, **kwargs):
            return FramingResult(frames=[TemporalFrame(0.0, 1.0)], metadata={})

        def cleanup(self):
            Framer.cleaned += 1

    _orchestrator(Framer(), tmp_path)._step1_frame_and_slice([scene_wav], [1.0])
    assert Framer.cleaned == 1


# --- The timing model's failed calls (text approved, W776) --------------------------------------------------------------

APPROVED_NOTE = ("ChronosJAV's timing model (Whisper small) failed on 2 of 3 VAD group(s): those subtitles are not "
                 "split, and their start times come from the aligner or the VAD group; the text is kept.")


class _Timer:
    accepts_array = True
    display_name = "Whisper small"

    def __init__(self, failed):
        self.failed_calls = 0
        self._failed = failed

    def load(self):
        self.failed_calls = 0

    def time(self, audio):
        if self._failed:
            self._failed -= 1
            self.failed_calls += 1
        return []

    def unload(self):
        self.failed_calls = 0            # as the real one: the count goes with the model


def test_the_note_is_the_approved_text():
    assert orch_module.TIMING_FAILED_TEXT.format(name="Whisper small", n=2, m=3) == APPROVED_NOTE


def test_failed_calls_are_counted_before_the_model_closes_and_named_once(caplog):
    orch = DecoupledSubtitlePipeline(framer=object(), generator=object(), cleaner=object(), aligner=None,
                                     hardening_config=HardeningConfig(), timer=_Timer(failed=2))
    frames = [[TemporalFrame(0.0, 1.0), TemporalFrame(1.0, 2.0), TemporalFrame(2.0, 3.0), TemporalFrame(3.0, 4.0)]]
    audio = [[np.zeros(16000, dtype=np.float32)] * 4]
    with caplog.at_level(logging.WARNING):
        orch._step8_time(frames, audio, [["あ", "い", "  ", "う"]])
    assert orch._last_timing_counts == (3, 2)
    lines = [r.getMessage() for r in caplog.records if "timing model failed" in r.getMessage()]
    assert lines == ["The timing model failed on 2 of 3 VAD group(s)"]


def test_no_failure_says_nothing(caplog):
    orch = DecoupledSubtitlePipeline(framer=object(), generator=object(), cleaner=object(), aligner=None,
                                     hardening_config=HardeningConfig(), timer=_Timer(failed=0))
    with caplog.at_level(logging.WARNING):
        orch._step8_time([[TemporalFrame(0.0, 1.0)]], [[np.zeros(16000, dtype=np.float32)]], [["あ"]])
    assert orch._last_timing_counts == (1, 0)
    assert not [r for r in caplog.records if "timing model failed" in r.getMessage()]


def test_the_note_is_made_from_the_first_pass_reset_per_file_and_put_in_the_run_summary():
    import inspect
    src = inspect.getsource(DecoupledSubtitlePipeline.process_scenes)
    assert "self.timing_partial = None" in src and "self._last_timing_counts = (0, 0)" in src
    assert "self.timing_partial = TIMING_FAILED_TEXT.format(" in src
    qwen = (REPO / "whisperjav" / "pipelines" / "qwen_pipeline.py").read_text(encoding="utf-8")
    assert 'if getattr(self._subtitle_pipeline, "timing_partial", None):\n' \
           '            self.notes.append(self._subtitle_pipeline.timing_partial)' in qwen.replace("\r\n", "\n")


def test_the_crispasr_timing_model_counts_its_failed_calls():
    from whisperjav.crispasr.timer import CrispTimer
    from whisperjav.modules.subtitle_pipeline.timers.crispasr import CrispASRTimer

    class Broken:
        def transcribe_pcm(self, pcm, language=None, decode_options=None):
            raise RuntimeError("stand-in failure")

        def close(self):
            pass

    timer = CrispTimer(model_name="whisper-small")
    assert timer.failed_calls == 0
    timer._model = Broken()
    assert timer.time(np.zeros(16000, dtype=np.float32)) == []
    assert timer.time(np.zeros(800, dtype=np.float32)) == []      # under 0.2 s: not sent, not a failure
    assert timer.failed_calls == 1
    adapter = CrispASRTimer(model="whisper-small")
    assert adapter.failed_calls == 0
    adapter._timer = timer
    assert adapter.failed_calls == 1
    adapter.unload()
    assert adapter.failed_calls == 0


# --- The speech enhancer ----------------------------------------------------------------------------------------------

FAILURE_RELEASE = (
    "            except BaseException:\n"
    "                # 1.10.3 (owner W776; model-memory inspection, section 3): the enhancer is released also when\n"
    "                # enhancing fails. A failure of the release itself is logged only, so the run's own error goes on.\n"
    "                try:\n"
    "                    enhancer.cleanup()\n"
    "                except Exception:  # noqa: BLE001\n"
    "                    logger.debug(\"Speech enhancer release after a failed enhancement failed\", exc_info=True)\n"
    "                raise\n"
    "\n"
    "            # VRAM Block 1 cleanup — release enhancer before loading ASR\n"
    "            enhancer.cleanup()\n"
)


@pytest.mark.parametrize("pipeline", ["qwen_pipeline.py", "decoupled_pipeline.py"])
def test_the_enhancer_is_released_once_on_each_path_and_the_error_goes_on(pipeline):
    src = (REPO / "whisperjav" / "pipelines" / pipeline).read_text(encoding="utf-8").replace("\r\n", "\n")
    # Both enhancement blocks (dual track, single track): the failure path releases and re-raises; the success path
    # releases once, as before.
    assert src.count(FAILURE_RELEASE) == 2


def test_the_pattern_releases_once_and_keeps_the_original_error():
    """The pattern above, run: a failing enhancement releases the enhancer once and its own error comes out, also
    when the release fails too; a working one releases it once."""
    calls = []

    class Enhancer:
        def __init__(self, release_fails=False):
            self.release_fails = release_fails

        def cleanup(self):
            calls.append("cleanup")
            if self.release_fails:
                raise OSError("the release failed too")

    def phase3(enhancer, enhance):
        try:
            enhance()
        except BaseException:
            try:
                enhancer.cleanup()
            except Exception:  # noqa: BLE001
                pass
            raise
        enhancer.cleanup()

    def fails():
        raise ValueError("every scene failed")

    for release_fails in (False, True):
        calls.clear()
        with pytest.raises(ValueError, match="every scene failed"):
            phase3(Enhancer(release_fails), fails)
        assert calls == ["cleanup"]
    calls.clear()
    phase3(Enhancer(), lambda: None)
    assert calls == ["cleanup"]
