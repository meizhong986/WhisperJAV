"""#436: "an interrupted aligner model download fails the whole job (rc=1) and the transcribed work is lost".

The ForcedAligner loads only after all text is generated. A failed load now keeps the text: the orchestrator takes
the aligner-free path (times from the speech segments) and records why, and the pipelines put that in the run summary.
No model is loaded: the aligner is a stand-in whose load() raises.
"""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("soundfile")

from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
from whisperjav.modules.subtitle_pipeline.types import HardeningConfig, TemporalFrame, TimestampMode


class _AlignerWhoseDownloadFails:
    def __init__(self):
        self.unloaded = False

    def load(self):
        raise OSError("Connection broken: IncompleteRead(1048576 bytes read)\nsecond line of the hub error")

    def unload(self):
        self.unloaded = True

    def align_batch(self, *a, **k):
        raise AssertionError("must not align after a failed load")


def _orchestrator(aligner):
    return DecoupledSubtitlePipeline(
        framer=object(), generator=object(), cleaner=object(), aligner=aligner,
        hardening_config=HardeningConfig(timestamp_mode=TimestampMode.ALIGNER_WITH_VAD_FALLBACK),
    )


def test_failed_aligner_load_keeps_the_text_and_says_why(tmp_path):
    import soundfile as sf

    wav = tmp_path / "scene_0000.wav"
    sf.write(str(wav), (0.1 * np.sin(np.linspace(0, 2000, 32000))).astype(np.float32), 16000)
    frames = [[TemporalFrame(0.0, 0.9), TemporalFrame(1.0, 1.9)]]
    texts = [["出身福島なんですね", "どうして?"]]
    aligner = _AlignerWhoseDownloadFails()
    orch = _orchestrator(aligner)

    alignments = orch._step5_7_align(frames, [[wav, wav]], texts, [2.0])

    assert alignments is None and aligner.unloaded
    note = orch.alignment_unavailable
    assert note.startswith("ForcedAligner could not be loaded (OSError: Connection broken")
    assert "second line" not in note and note.endswith("the text is kept")

    results = orch._step9_reconstruct_and_harden(frames, texts, alignments, [wav], [2.0], [None], None)
    kept = "".join(seg.text for seg in results[0][0].segments).replace(" ", "")
    assert "福島" in kept and "どうして" in kept


def test_note_is_cleared_for_the_next_file():
    import inspect
    src = inspect.getsource(DecoupledSubtitlePipeline.process_scenes)
    assert "self.alignment_unavailable = None" in src


def test_both_pipelines_report_it():
    root = Path(__file__).resolve().parents[1] / "whisperjav" / "pipelines"
    for name in ("qwen_pipeline.py", "decoupled_pipeline.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert "self.degradations.append(self._subtitle_pipeline.alignment_unavailable)" in src, name


def test_aligner_failing_only_for_the_step_down_retry_keeps_the_first_pass(tmp_path):
    """Review finding: pass 1 aligned (scene collapsed), the retry could not reload the aligner. The retry's
    aligner-free result must not replace pass 1 as an 'improvement'."""
    from whisperjav.modules.subtitle_pipeline.types import StepDownConfig

    class _Framer:
        def reframe(self, *a, **k):
            pass

    orch = DecoupledSubtitlePipeline(
        framer=_Framer(), generator=object(), cleaner=object(), aligner=object(),
        hardening_config=HardeningConfig(timestamp_mode=TimestampMode.ALIGNER_WITH_VAD_FALLBACK),
        stepdown_config=StepDownConfig(enabled=True),
    )
    pass1 = [("pass1-result", {"sentinel_status": "COLLAPSED"})]

    def _retry(*a, **k):
        orch.alignment_unavailable = ("ForcedAligner could not be loaded (RuntimeError: CUDA out of memory); "
                                      "subtitle times come from the speech segments instead; the text is kept")
        return [("retry-result", {"sentinel_status": "N/A"})]

    orch._run_pass = lambda *a, **k: pass1
    orch._run_stepdown_pass = _retry
    results = orch.process_scenes([tmp_path / "s.wav"], [10.0])
    assert results[0][0] == "pass1-result"
    assert orch.alignment_unavailable == ("ForcedAligner could not be reloaded for the step-down retry "
                                          "(RuntimeError: CUDA out of memory); 1 collapsed scene(s) keep their "
                                          "first-pass timing")
