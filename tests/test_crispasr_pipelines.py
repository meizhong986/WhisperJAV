"""
The CrispASR pipelines: one shared chain, two pipelines (whisperjav/pipelines/crispasr_pipeline.py and the
two small pipeline files; route B design 3, 4, 6; owner W175, W176, W199 DIR1, W202-W204).

What runs for real: the pipeline classes, the speech segmenter settings (the same functions fidelity and
the qwen pipeline use), the line rules, the scene SRTs, the stitcher and the three filters. What is a
stand-in: CrispASR's readiness and model file, audio extraction, scene detection (the scene WAVs are
written by the test), ChronosJAV's framer (its module is replaced, so stable-ts and torch are never
loaded) and the engine. No ASR module is imported and nothing is downloaded.

Run with: python -m pytest tests/test_crispasr_pipelines.py -q
"""
import ast
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from whisperjav.crispasr import install, lines
from whisperjav.crispasr.aligner import TimedUnit
from whisperjav.crispasr.engine import CrispSegment, CrispToken
from whisperjav.crispasr.errors import CrispASRRunError, CrispASRSetupError
from whisperjav.crispasr.pass_settings import (QWEN_CHUNK_THRESHOLD_S, QWEN_END_PAD_MS, QWEN_MAX_GROUP_S,
                                               QWEN_START_PAD_MS, segmenter_settings, speech_pad_ignored)
from whisperjav.pipelines import crispasr_pipeline as chain
from whisperjav.pipelines.cohere_crispasr_pipeline import CohereCrispASRPipeline
from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline
from whisperjav.pipelines.parakeet_crispasr_pipeline import ParakeetCrispASRPipeline

REPO = Path(__file__).resolve().parents[1]
RATE = 16000
FRAMER_MODULE = "whisperjav.modules.subtitle_pipeline.framers.vad_grouped"


# ---------------------------------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------------------------------

def toks(*items):
    return tuple(CrispToken(text, start, end, 1.0) for text, start, end in items)


def parakeet_segment(*items):
    tokens = toks(*items)
    return CrispSegment("".join(t.text for t in tokens).strip(), tokens[0].start, tokens[-1].end, (), tokens)


class Stage:
    """Everything outside the chain, replaced; what was asked of it is recorded."""

    def __init__(self, monkeypatch, tmp_path):
        self.tmp = tmp_path
        self.calls = {"prepare": 0, "ensure": 0, "stage": 0, "extract": 0, "scenes": 0, "analytics": 0}
        self.runtime = install.RuntimeChoice(True, "cuda", None, False, "0.8.44+cuda", "cuda")
        self.setup_error = None
        # Per scene: list of (start, end) VAD groups in scene time; or an Exception; or {"error": "..."}.
        self.frames = {0: [(1.0, 2.0), (3.0, 4.5)], 1: [(0.5, 1.5)]}
        self.framer_made = []
        self.framers_cleaned = 0
        # The engine's answers, one per call, in order; an Exception is raised.
        self.answers = []
        self.engines = []
        self.scene_starts = [0.0, 10.0]
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF")
        self.model_file = model_file

        def prepare_runtime(device, model):
            self.calls["prepare"] += 1
            self.device_asked = device
            if self.setup_error is not None:
                raise self.setup_error
            return self.runtime

        def ensure_model(model, offline=None):
            self.calls["ensure"] += 1
            return model_file

        def stage_model(path, temp_dir, sha256):
            self.calls["stage"] += 1
            return Path(path)

        monkeypatch.setattr(install, "prepare_runtime", prepare_runtime)
        monkeypatch.setattr(install, "ensure_model", ensure_model)
        monkeypatch.setattr(install, "stage_model", stage_model)

        stage = self

        def extract(pipe, input_file, basename):
            stage.calls["extract"] += 1
            wav = pipe.temp_dir / f"{basename}_extracted.wav"
            sf.write(str(wav), np.zeros(20 * RATE, dtype=np.float32), RATE)
            return wav, 20.0

        def detect(pipe, wav, basename, md):
            stage.calls["scenes"] += 1
            folder = pipe.temp_dir / "scenes"
            folder.mkdir(parents=True, exist_ok=True)
            scenes = []
            for i, start in enumerate(stage.scene_starts):
                path = folder / f"{basename}_scene_{i:04d}.wav"
                sf.write(str(path), np.full(10 * RATE, 0.01, dtype=np.float32), RATE)
                scenes.append(chain._Scene(i, path, start, start + 10.0))
            if pipe.ENGINE.audio_analytics:
                stage.calls["analytics"] += 1
            return scenes

        monkeypatch.setattr(CrispASRPipeline, "_extract_audio", extract)
        monkeypatch.setattr(CrispASRPipeline, "_detect_scenes", detect)

        class Framer:
            def __init__(self, segmenter_backend, max_group_duration_s, chunk_threshold_s, min_frame_duration_s,
                         segmenter_config):
                stage.framer_made.append({"backend": segmenter_backend, "max_group": max_group_duration_s,
                                          "chunk": chunk_threshold_s, "min": min_frame_duration_s,
                                          "config": dict(segmenter_config)})
                self.scene = 0

            def frame(self, audio, sample_rate):
                assert sample_rate == RATE and audio.dtype == np.float32
                answer = stage.frames.get(self.scene, [])
                self.scene += 1
                if isinstance(answer, Exception):
                    raise answer
                if isinstance(answer, dict):
                    return types.SimpleNamespace(frames=[], metadata={"segmenter_error": answer["error"]})
                return types.SimpleNamespace(
                    frames=[types.SimpleNamespace(start=s, end=e) for s, e in answer], metadata={})

            def cleanup(self):
                stage.framers_cleaned += 1

        monkeypatch.setitem(sys.modules, FRAMER_MODULE, types.SimpleNamespace(VadGroupedFramer=Framer))

        class Engine:
            def __init__(self, model_path, backend, use_gpu, n_threads=None, label=""):
                self.model_path, self.backend, self.use_gpu = model_path, backend, use_gpu
                self.n_threads = 8
                self.opened = 0
                self.closed = 0
                self.audio = []
                self.open_error = getattr(stage, "open_error", None)
                stage.engines.append(self)

            @property
            def is_open(self):
                return self.opened > self.closed

            def open(self):
                if self.open_error is not None:
                    raise self.open_error
                self.opened += 1

            def transcribe(self, audio):
                assert self.is_open
                self.audio.append(len(audio))
                answer = stage.answers.pop(0) if stage.answers else []
                if isinstance(answer, Exception):
                    raise answer
                return answer

            def close(self):
                if self.is_open:
                    self.closed += 1

            def cleanup(self):
                self.close()

        monkeypatch.setattr(chain, "BindingEngine", Engine)

        # The CrispASR aligner (Cohere-CrispASR, W248 3): its answers, one per call, in order; [] = nothing aligned.
        self.alignments = []
        self.aligners = []
        self.aligner_load_error = None

        class Aligner:
            def __init__(self, model_name=None, device=None, temp_dir=None, n_threads=None, label=""):
                self.model_name, self.device, self.label = model_name, device, label
                self.loads = self.unloads = 0
                self.texts = []
                self._loaded = False
                stage.aligners.append(self)

            @property
            def is_loaded(self):
                return self._loaded

            def load(self):
                self.loads += 1
                if stage.aligner_load_error is not None:
                    raise stage.aligner_load_error
                self._loaded = True

            def align(self, audio, text):
                assert self._loaded and audio.dtype == np.float32
                self.texts.append((len(audio), text))
                return stage.alignments.pop(0) if stage.alignments else []

            def unload(self):
                if self._loaded:
                    self.unloads += 1
                    self._loaded = False

        monkeypatch.setattr(chain, "CrispAligner", Aligner)

    def media(self, name="film"):
        path = self.tmp / f"{name}.mp4"
        path.write_bytes(b"")
        return {"path": str(path), "basename": name}


@pytest.fixture
def stage(monkeypatch, tmp_path):
    return Stage(monkeypatch, tmp_path)


def make(cls, tmp_path, **kwargs):
    return cls(output_dir=str(tmp_path / "out"), temp_dir=str(tmp_path / "temp"), **kwargs)


def srt_entries(path):
    import srt
    return [(s.content, round(s.start.total_seconds(), 3), round(s.end.total_seconds(), 3))
            for s in srt.parse(Path(path).read_text(encoding="utf-8"))]


# ---------------------------------------------------------------------------------------------------
# Building the pipelines
# ---------------------------------------------------------------------------------------------------

class TestTheTwoPipelines:
    def test_names_labels_and_rows(self):
        assert (ParakeetCrispASRPipeline.PIPELINE_NAME, ParakeetCrispASRPipeline.LABEL) == \
            ("parakeet-crispasr", "Parakeet-CrispASR")
        assert (CohereCrispASRPipeline.PIPELINE_NAME, CohereCrispASRPipeline.LABEL) == \
            ("cohere-crispasr", "Cohere-CrispASR")
        assert ParakeetCrispASRPipeline.ENGINE == chain.EngineRow("tokens", audio_analytics=True)
        # Cohere-CrispASR's text is aligned by the CrispASR aligner (W248 3, W249); "group" for a group it cannot.
        assert CohereCrispASRPipeline.ENGINE == chain.EngineRow("group", audio_analytics=True, aligner="wav2vec2-ja")
        assert ParakeetCrispASRPipeline.ENGINE.aligner is None

    def test_both_share_one_chain(self):
        for cls in (ParakeetCrispASRPipeline, CohereCrispASRPipeline):
            assert cls.__mro__[1] is CrispASRPipeline
            assert set(vars(cls)) <= {"PIPELINE_NAME", "LABEL", "ENGINE", "__module__", "__doc__", "__qualname__",
                                      "__firstlineno__", "__static_attributes__",
                                      "_abc_impl", "__abstractmethods__"}

    @pytest.mark.parametrize("cls, model", [(ParakeetCrispASRPipeline, "parakeet-tdt-0.6b-ja"),
                                            (CohereCrispASRPipeline, "cohere-asr-ja-v0.1")])
    def test_defaults(self, tmp_path, cls, model):
        pipe = make(cls, tmp_path)
        assert pipe.model.name == model
        assert pipe.sensitivity == "balanced"  # W135 2, W202 2
        assert pipe.device is None

    def test_parakeet_takes_fidelitys_settings_with_6_s(self, tmp_path):
        pipe = make(ParakeetCrispASRPipeline, tmp_path, sensitivity="aggressive")
        assert pipe.speech_segmenter == "whisperseg"
        assert pipe.segmenter == segmenter_settings("parakeet-crispasr", "whisperseg", "aggressive")
        assert (pipe.segmenter.max_group_duration_s, pipe.segmenter.config["max_speech_duration_s"]) == (6.0, 6.0)
        assert pipe.segmenter.config["threshold"] == 0.25

    def test_cohere_takes_the_qwen_pipelines_settings(self, tmp_path):
        pipe = make(CohereCrispASRPipeline, tmp_path, sensitivity="conservative")
        assert pipe.speech_segmenter == "firered-vad"
        s = pipe.segmenter
        assert (s.max_group_duration_s, s.chunk_threshold_s) == (3.0, 0.3)
        assert (s.config["max_speech_duration_s"], s.config["start_pad_ms"], s.config["end_pad_ms"]) == (4.0, 100, 100)
        assert s.config["threshold"] == 0.5  # FireRedVAD's conservative preset: sensitivity moves it (W202 4)

    BACKENDS = ["whisperseg", "firered-vad", "ten", "silero-v3.1", "silero-v4.0", "silero-v6.2", "whisper-vad", "nemo"]

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("sensitivity", ["conservative", "balanced", "aggressive"])
    def test_parakeet_equals_what_a_fidelity_pass_resolves_but_for_the_6_s(self, backend, sensitivity):
        # A fidelity Ensemble pass: pass_worker._apply_gui_overrides -> resolve_segmenter_sensitivity(b, s, None)
        from whisperjav.config.segmenter_presets import resolve_segmenter_sensitivity

        fidelity = dict(resolve_segmenter_sensitivity(backend, sensitivity, None))
        ours = segmenter_settings("parakeet-crispasr", backend, sensitivity)
        merged = {**ours.config, "max_group_duration_s": ours.max_group_duration_s}
        if ours.chunk_threshold_s is not None:
            merged["chunk_threshold_s"] = ours.chunk_threshold_s
        for key in ("max_group_duration_s", "max_speech_duration_s"):  # W202 1: 6 s at every sensitivity
            if key in merged:
                assert merged.pop(key) == 6.0
            fidelity.pop(key, None)
        assert merged == fidelity

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("sensitivity", ["conservative", "balanced", "aggressive"])
    def test_cohere_equals_what_the_qwen_pipeline_resolves(self, backend, sensitivity):
        # The qwen pipeline with Qwen3-ASR: resolve_chronosjav_segmenter_config, then QwenPipeline's
        # groups and pads on its framer (qwen_pipeline.py, framer kwargs).
        from whisperjav.config.chronosjav_vad import chronosjav_end_pad_default, resolve_chronosjav_segmenter_config

        qwen = dict(resolve_chronosjav_segmenter_config("qwen3", backend, sensitivity))
        qwen.pop("max_group_duration_s", None)
        qwen.pop("chunk_threshold_s", None)
        qwen["start_pad_ms"] = QWEN_START_PAD_MS
        end_pad = chronosjav_end_pad_default("qwen3", backend)
        qwen["end_pad_ms"] = QWEN_END_PAD_MS if end_pad is None else end_pad
        ours = segmenter_settings("cohere-crispasr", backend, sensitivity)
        assert ours.config == qwen
        assert (ours.max_group_duration_s, ours.chunk_threshold_s) == (QWEN_MAX_GROUP_S, QWEN_CHUNK_THRESHOLD_S)

    def test_the_qwen_values_are_the_qwen_pipelines_own_defaults(self):
        tree = ast.parse((REPO / "whisperjav" / "pipelines" / "qwen_pipeline.py").read_text(encoding="utf-8"))
        init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        args = init.args.args[-len(init.args.defaults):]
        defaults = {a.arg: ast.literal_eval(d) for a, d in zip(args, init.args.defaults)
                    if isinstance(d, ast.Constant)}
        assert defaults["segmenter_max_group_duration"] == QWEN_MAX_GROUP_S
        assert defaults["segmenter_chunk_threshold"] == QWEN_CHUNK_THRESHOLD_S
        assert defaults["segmenter_start_pad_ms"] == QWEN_START_PAD_MS
        assert defaults["segmenter_end_pad_ms"] == QWEN_END_PAD_MS

    # W208 "make the flags work": --passN-vad-threshold and --passN-speech-pad-ms, as fidelity and qwen use them.
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_parakeet_takes_the_two_flags_as_a_fidelity_pass_does(self, backend):
        # pass_worker, fidelity: threshold clamped to 0-1 into the segmenter's threshold; speech_pad_ms only
        # where the backend has that setting (segmenter_accepts), else a warning and nothing applied.
        from whisperjav.config.segmenter_presets import resolve_segmenter_sensitivity, segmenter_accepts

        fidelity = dict(resolve_segmenter_sensitivity(backend, "balanced", None))
        fidelity["threshold"] = 0.61
        if segmenter_accepts(backend, "speech_pad_ms"):
            fidelity["speech_pad_ms"] = 120
        ours = segmenter_settings("parakeet-crispasr", backend, "balanced", vad_threshold=0.61, speech_pad_ms=120)
        merged = {**ours.config, "max_group_duration_s": ours.max_group_duration_s}
        if ours.chunk_threshold_s is not None:
            merged["chunk_threshold_s"] = ours.chunk_threshold_s
        for key in ("max_group_duration_s", "max_speech_duration_s"):
            if key in merged:
                assert merged.pop(key) == 6.0
            fidelity.pop(key, None)
        assert merged == fidelity
        assert speech_pad_ignored("parakeet-crispasr", backend, 120) == (not segmenter_accepts(backend, "speech_pad_ms"))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cohere_takes_the_two_flags_as_the_qwen_pipeline_does(self, backend):
        # pass_worker, qwen: the threshold, clamped, is a user override of the segmenter config; the speech pad
        # sets both the start and the end pad of the pipeline.
        from whisperjav.config.chronosjav_vad import resolve_chronosjav_segmenter_config

        qwen = dict(resolve_chronosjav_segmenter_config("qwen3", backend, "balanced", {"threshold": 0.61}))
        qwen.pop("max_group_duration_s", None)
        qwen.pop("chunk_threshold_s", None)
        qwen["start_pad_ms"] = qwen["end_pad_ms"] = 50
        ours = segmenter_settings("cohere-crispasr", backend, "balanced", vad_threshold=0.61, speech_pad_ms=50)
        assert ours.config == qwen
        assert not speech_pad_ignored("cohere-crispasr", backend, 50)

    @pytest.mark.parametrize("given,used", [(1.7, 1.0), (-0.2, 0.0)])
    def test_the_threshold_is_clamped_as_fidelity_and_qwen_clamp_it(self, given, used):
        for pipeline in ("parakeet-crispasr", "cohere-crispasr"):
            assert segmenter_settings(pipeline, None, None, vad_threshold=given).config["threshold"] == used

    def test_the_pipeline_passes_the_two_flags_on(self, tmp_path):
        pipe = make(CohereCrispASRPipeline, tmp_path, vad_threshold=0.2, speech_pad_ms=30)
        assert pipe.segmenter == segmenter_settings("cohere-crispasr", None, None, vad_threshold=0.2, speech_pad_ms=30)
        assert (pipe.segmenter.config["threshold"], pipe.segmenter.config["start_pad_ms"]) == (0.2, 30)

    def test_the_pause_values_default_and_can_be_set(self, tmp_path):
        assert make(ParakeetCrispASRPipeline, tmp_path).cues == lines.CueSettings(0.6, 0.4)
        cues = lines.CueSettings(pause_cut_s=0.8, weak_mark_pause_s=0.3)
        assert make(ParakeetCrispASRPipeline, tmp_path, cues=cues).cues is cues

    @pytest.mark.parametrize("language", ["ja", "JA", "japanese", "jpn", None])
    def test_japanese_is_accepted(self, tmp_path, language):
        make(ParakeetCrispASRPipeline, tmp_path, language=language)

    def test_another_language_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="Japanese only"):
            make(ParakeetCrispASRPipeline, tmp_path, language="en")

    def test_the_segmenter_none_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="needs a speech segmenter"):
            make(CohereCrispASRPipeline, tmp_path, speech_segmenter="none")

    def test_a_model_of_another_pipeline_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="is not a model of parakeet-crispasr"):
            make(ParakeetCrispASRPipeline, tmp_path, model_name="cohere-asr-ja")

    def test_settings_it_does_not_use_are_accepted_and_ignored(self, tmp_path):
        make(ParakeetCrispASRPipeline, tmp_path, scene_detector="auditok", speech_enhancer="ffmpeg-dsp")

    def test_nothing_is_touched_when_it_is_built(self, stage, tmp_path):
        make(ParakeetCrispASRPipeline, tmp_path)
        assert stage.calls == {"prepare": 0, "ensure": 0, "stage": 0, "extract": 0, "scenes": 0, "analytics": 0}
        assert stage.engines == []


# ---------------------------------------------------------------------------------------------------
# The whole chain
# ---------------------------------------------------------------------------------------------------

class TestParakeetEndToEnd:
    def test_tokens_become_subtitles_in_media_file_time(self, stage, tmp_path):
        stage.answers = [
            [parakeet_segment((" ", 0.0, 0.1), ("本番", 0.1, 0.4), ("。", 0.4, 0.48), ("行こう", 0.48, 0.9))],
            [parakeet_segment(("待って", 0.0, 0.5))],
            [parakeet_segment(("監", 0.1, 0.3), ("督", 0.3, 0.5), ("。", 0.5, 0.7))],
        ]
        pipe = make(ParakeetCrispASRPipeline, tmp_path, save_metadata_json=True)
        md = pipe.process(stage.media())
        final = Path(md["output_files"]["final_srt"])
        assert final.name == "film.ja.whisperjav.srt"
        # scene 0 starts at 0 s: groups at 1.0 and 3.0; scene 1 starts at 10 s: group at 10.5
        assert srt_entries(final) == [("本番。", 1.1, 1.4), ("行こう", 1.48, 1.9), ("待って", 3.0, 3.5),
                                      ("監督。", 10.6, 11.0)]
        s = md["summary"]
        assert (s["final_subtitles_refined"], s["final_subtitles_raw"]) == (4, 4)
        assert s["degradations"] == [] and s["notes"] == []
        assert md["stages"]["groups"] == {"count": 3, "seconds": 3.5, "failed_scenes": 0}
        assert md["stages"]["asr"]["groups"] == 3 and md["stages"]["asr"]["failed"] == 0
        assert stage.calls["analytics"] == 1  # as fidelity and Cohere-CrispASR (W211 C2)

    def test_each_group_is_one_call_with_its_own_audio(self, stage, tmp_path):
        make(ParakeetCrispASRPipeline, tmp_path).process(stage.media())
        assert stage.engines[0].audio == [16000, 24000, 16000]

    def test_the_framer_runs_with_the_settings_and_is_released_before_the_model_runs(self, stage, tmp_path):
        pipe = make(ParakeetCrispASRPipeline, tmp_path)
        pipe.process(stage.media())
        made = stage.framer_made[0]
        assert (made["backend"], made["max_group"], made["chunk"], made["min"]) == ("whisperseg", 6.0, 1.0, 0.1)
        assert made["config"] == pipe.segmenter.config
        assert stage.framers_cleaned == 1

    def test_the_engine_gets_the_staged_model_and_the_device_choice(self, stage, tmp_path):
        stage.runtime = install.RuntimeChoice(False, "cpu", "--device cpu was given", False, "0.8.44+cuda", "cuda")
        make(ParakeetCrispASRPipeline, tmp_path, device="cpu").process(stage.media())
        engine = stage.engines[0]
        assert (engine.model_path, engine.backend, engine.use_gpu) == (stage.model_file, "parakeet", False)
        assert stage.device_asked == "cpu"


class TestCohereEndToEnd:
    def test_one_subtitle_per_group_with_the_groups_edges(self, stage, tmp_path):
        stage.answers = [[CrispSegment("本番。", 0.0, 1.0)], [CrispSegment("いいよ", 0.0, 1.5)],
                         [CrispSegment("監督", 0.0, 1.0)]]
        md = make(CohereCrispASRPipeline, tmp_path).process(stage.media())
        assert srt_entries(md["output_files"]["final_srt"]) == [("本番。", 1.0, 2.0), ("いいよ", 3.0, 4.5),
                                                                ("監督", 10.5, 11.5)]
        assert stage.calls["analytics"] == 1  # as the qwen pipeline (W202 5b)
        made = stage.framer_made[0]
        assert (made["backend"], made["max_group"], made["chunk"]) == ("firered-vad", 3.0, 0.3)


class TestCohereAligned:
    """Cohere-CrispASR's text aligned by the CrispASR aligner (design note 2026-10-28, section 6; W248 3, W249)."""

    TEXT = "近くのカラオケでそのまま酔っぱらっちゃった。それで？"
    # Probe P1, wav2vec2-ja on group A of S02E02 Scene 5 (3.79 s): one unit per character, marks attached; the
    # last unit ends at the end of the group's audio (whisperjav/crispasr/aligner.py).
    UNITS = [TimedUnit(t, s, e) for t, s, e in (
        ("近", 0.26, 0.28), ("く", 0.38, 0.40), ("の", 0.48, 0.50), ("カ", 0.60, 0.62), ("ラ", 0.72, 0.74),
        ("オ", 0.80, 0.82), ("ケ", 0.90, 0.92), ("で", 1.00, 1.02), ("そ", 1.12, 1.14), ("の", 1.26, 1.28),
        ("ま", 1.52, 1.54), ("ま", 1.74, 1.76), ("酔", 1.78, 1.80), ("っ", 1.86, 1.88), ("ぱ", 1.90, 1.92),
        ("ら", 2.08, 2.10), ("っ", 2.32, 2.34), ("ち", 2.36, 2.38), ("ゃ", 2.40, 2.42), ("っ", 2.60, 2.64),
        ("た。", 2.66, 2.68), ("そ", 3.10, 3.12), ("れ", 3.34, 3.36), ("で？", 3.76, 3.79))]

    def test_an_aligned_group_is_cut_at_the_sentence_end_and_the_subtitles_are_chained(self, stage, tmp_path):
        stage.frames = {0: [(1.0, 4.79)], 1: []}
        stage.answers = [[CrispSegment(self.TEXT, 0.0, 3.79)]]
        stage.alignments = [self.UNITS]
        md = make(CohereCrispASRPipeline, tmp_path).process(stage.media())
        assert srt_entries(md["output_files"]["final_srt"]) == [
            ("近くのカラオケでそのまま酔っぱらっちゃった。", 1.26, 4.1), ("それで？", 4.1, 4.79)]
        aligner = stage.aligners[0]
        assert (aligner.model_name, aligner.label) == ("wav2vec2-ja", "Cohere-CrispASR")
        assert aligner.texts == [(int(3.79 * RATE), self.TEXT)]
        assert md["stages"]["alignment"]["aligned"] == 1 and md["summary"]["notes"] == []
        assert md["config"]["crispasr"]["aligner"] == "wav2vec2-ja"

    def test_a_group_the_aligner_gives_nothing_for_keeps_its_edges_and_is_counted(self, stage, tmp_path):
        stage.answers = [[CrispSegment("本番。", 0.0, 1.0)], [CrispSegment("いいよ", 0.0, 1.5)],
                         [CrispSegment("監督", 0.0, 1.0)]]
        stage.alignments = [[TimedUnit("本", 0.2, 0.22), TimedUnit("番。", 0.4, 1.0)], [], []]
        md = make(CohereCrispASRPipeline, tmp_path).process(stage.media())
        assert srt_entries(md["output_files"]["final_srt"]) == [("本番。", 1.2, 2.0), ("いいよ", 3.0, 4.5),
                                                                ("監督", 10.5, 11.5)]
        assert md["summary"]["notes"] == [
            "Cohere-CrispASR: 2 of 3 VAD group(s) could not be aligned; they keep the VAD group's edges"]

    def test_groups_without_text_are_not_sent_to_the_aligner(self, stage, tmp_path):
        stage.answers = [[CrispSegment("", 0.0, 1.0)], [], [CrispSegment("監督", 0.0, 1.0)]]
        make(CohereCrispASRPipeline, tmp_path).process(stage.media())
        assert [t for _n, t in stage.aligners[0].texts] == ["監督"]

    def test_an_aligner_that_cannot_load_keeps_every_groups_edges_says_so_and_is_tried_once(self, stage, tmp_path):
        stage.aligner_load_error = CrispASRSetupError("Downloading the model wav2vec2 failed: offline\nmore")
        pipe = make(CohereCrispASRPipeline, tmp_path)
        for name in ("a", "b"):
            stage.answers = [[CrispSegment("本番。", 0.0, 1.0)], [CrispSegment("いいよ", 0.0, 1.5)],
                             [CrispSegment("監督", 0.0, 1.0)]]
            md = pipe.process(stage.media(name))
            assert srt_entries(md["output_files"]["final_srt"])[0] == ("本番。", 1.0, 2.0)
            assert md["summary"]["notes"] == [
                "Cohere-CrispASR's aligner (wav2vec2-ja) could not be loaded (Downloading the model wav2vec2 "
                "failed: offline): each VAD group is one subtitle at the group's edges, as before 1.10.0."]
        assert len(stage.aligners) == 1 and stage.aligners[0].loads == 1

    def test_the_aligner_loads_once_per_pass_and_unloads_at_cleanup(self, stage, tmp_path):
        pipe = make(CohereCrispASRPipeline, tmp_path)
        pipe.process(stage.media("a"))
        pipe.process(stage.media("b"))
        assert len(stage.aligners) == 1 and stage.aligners[0].loads == 1 and stage.aligners[0].unloads == 0
        pipe.cleanup()
        assert stage.aligners[0].unloads == 1

    def test_with_a_speech_enhancer_the_aligner_unloads_after_each_files_asr_step(self, stage, tmp_path, monkeypatch):
        monkeypatch.setattr(CrispASRPipeline, "_enhance", lambda self, scenes, md: (scenes, scenes))
        pipe = make(CohereCrispASRPipeline, tmp_path, speech_enhancer="ffmpeg-dsp")
        pipe.process(stage.media("a"))
        pipe.process(stage.media("b"))
        assert stage.aligners[0].loads == 2 and stage.aligners[0].unloads == 2

    def test_the_debug_record_says_which_groups_were_aligned(self, stage, tmp_path):
        stage.frames = {0: [(1.0, 4.79)], 1: [(0.5, 1.5)]}
        stage.answers = [[CrispSegment(self.TEXT, 0.0, 3.79)], [CrispSegment("監督", 0.0, 1.0)]]
        stage.alignments = [self.UNITS, []]
        make(CohereCrispASRPipeline, tmp_path, save_metadata_json=True).process(stage.media())
        record = tmp_path / "out" / "raw_subs" / "film.cohere-crispasr.groups.jsonl"
        rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
        assert [r["aligned"] for r in rows] == [True, False]
        assert rows[0]["segments"][0]["tokens"][20] == ["た。", 2.66, 2.68, None]

    def test_parakeet_makes_no_aligner(self, stage, tmp_path):
        stage.answers = [[parakeet_segment(("はい", 0.1, 0.4))]]
        make(ParakeetCrispASRPipeline, tmp_path).process(stage.media())
        assert stage.aligners == []


class TestOncePerPass:
    def test_readiness_and_the_model_open_once_for_every_file_and_close_at_the_end(self, stage, tmp_path):
        pipe = make(ParakeetCrispASRPipeline, tmp_path)
        pipe.process(stage.media("a"))
        pipe.process(stage.media("b"))
        assert (stage.calls["prepare"], stage.calls["ensure"], stage.calls["stage"]) == (1, 1, 1)
        assert len(stage.engines) == 1 and stage.engines[0].opened == 1 and stage.engines[0].closed == 0
        pipe.cleanup()
        assert stage.engines[0].closed == 1

    def test_a_model_file_gone_since_it_was_prepared_is_prepared_again(self, stage, tmp_path):
        pipe = make(ParakeetCrispASRPipeline, tmp_path)
        pipe.process(stage.media("a"))
        stage.model_file.unlink()
        stage.model_file.write_bytes(b"GGUF")  # fetched again by ensure_model
        pipe._model_path = tmp_path / "gone.gguf"
        pipe.process(stage.media("b"))
        assert stage.calls["ensure"] == 2 and stage.calls["prepare"] == 1

    def test_the_device_line_is_shown_once_per_pass(self, stage, tmp_path, caplog):
        pipe = make(ParakeetCrispASRPipeline, tmp_path)
        with caplog.at_level("INFO", logger="whisperjav"):
            pipe.process(stage.media("a"))
            pipe.process(stage.media("b"))
        lines = [r.getMessage() for r in caplog.records if "asks CrispASR to use" in r.getMessage()]
        assert lines == ["Parakeet-CrispASR asks CrispASR to use the NVIDIA GPU (CUDA) (CrispASR 0.8.44+cuda)."]

    def test_an_unexpected_processor_run_is_also_a_note_in_every_files_summary(self, stage, tmp_path):
        stage.runtime = install.RuntimeChoice(False, "cpu", "the card is too old", True, "0.8.44+cuda", "cuda")
        pipe = make(ParakeetCrispASRPipeline, tmp_path)
        for name in ("a", "b"):
            stage.answers = [[parakeet_segment(("監督", 0.0, 0.3))]] * 3
            md = pipe.process(stage.media(name))
            assert md["summary"]["notes"] == [
                "Parakeet-CrispASR runs on the CPU (not accelerated), several times slower than on a GPU: the card is too old."]


class TestFailures:
    def test_a_setup_failure_is_kept_and_raised_for_the_passs_other_files(self, stage, tmp_path):
        stage.setup_error = CrispASRSetupError("CrispASR is not installed in this WhisperJAV.")
        pipe = make(ParakeetCrispASRPipeline, tmp_path)
        for name in ("a", "b"):
            with pytest.raises(CrispASRSetupError, match="not installed"):
                pipe.process(stage.media(name))
        assert stage.calls["prepare"] == 1 and stage.calls["extract"] == 0

    def test_a_model_that_cannot_be_opened_is_kept_as_a_setup_failure(self, stage, tmp_path):
        stage.open_error = CrispASRSetupError("The model could not be opened")
        pipe = make(ParakeetCrispASRPipeline, tmp_path)
        with pytest.raises(CrispASRSetupError):
            pipe.process(stage.media("a"))
        with pytest.raises(CrispASRSetupError):
            pipe.process(stage.media("b"))
        assert len(stage.engines) == 1 and stage.calls["extract"] == 1

    def test_no_speech_opens_no_model_and_writes_an_empty_srt(self, stage, tmp_path):
        stage.frames = {}
        md = make(ParakeetCrispASRPipeline, tmp_path).process(stage.media())
        assert stage.engines == []
        assert Path(md["output_files"]["final_srt"]).read_text(encoding="utf-8") == ""
        assert md["summary"]["final_subtitles_refined"] == 0

    def test_one_failed_group_is_a_degradation_and_the_file_goes_on(self, stage, tmp_path):
        stage.answers = [CrispASRRunError("crispasr_session_transcribe failed"),
                         [parakeet_segment(("待って", 0.0, 0.5))], [parakeet_segment(("監督", 0.0, 0.3))]]
        md = make(ParakeetCrispASRPipeline, tmp_path).process(stage.media())
        assert md["summary"]["degradations"] == [
            "Parakeet-CrispASR: 1 of 3 VAD group(s) could not be transcribed (crispasr_session_transcribe failed)"]
        assert md["summary"]["final_subtitles_refined"] == 2

    def test_every_group_failing_fails_the_file(self, stage, tmp_path):
        stage.answers = [CrispASRRunError("boom")] * 3
        with pytest.raises(CrispASRRunError, match="failed for every VAD group of this file: boom"):
            make(ParakeetCrispASRPipeline, tmp_path).process(stage.media())

    def test_groups_without_text_are_a_note(self, stage, tmp_path):
        stage.answers = [[], [CrispSegment("", 0.0, 1.0)], [parakeet_segment(("監督", 0.0, 0.3))]]
        md = make(ParakeetCrispASRPipeline, tmp_path).process(stage.media())
        assert md["summary"]["notes"] == ["Parakeet-CrispASR: 2 of 3 VAD group(s) gave no text"]

    @pytest.mark.parametrize("failure", [RuntimeError("segmenter broke"), {"error": "model missing"}])
    def test_a_scene_the_segmenter_failed_on_is_a_note(self, stage, tmp_path, failure):
        stage.frames = {0: failure, 1: [(0.5, 1.5)]}
        stage.answers = [[parakeet_segment(("監督", 0.0, 0.3))]]
        md = make(ParakeetCrispASRPipeline, tmp_path).process(stage.media())
        assert md["summary"]["notes"] == ["Parakeet-CrispASR: the speech segmenter failed on 1 scene(s); their "
                                          "speech was not transcribed."]
        assert md["stages"]["groups"]["failed_scenes"] == 1

    def test_groups_are_clamped_to_their_scenes_audio_and_tiny_ones_dropped(self, stage, tmp_path):
        stage.frames = {0: [(9.5, 12.0), (5.0, 5.05)], 1: []}
        make(ParakeetCrispASRPipeline, tmp_path).process(stage.media())
        assert stage.engines[0].audio == [8000]  # 9.5-10.0 s: the scene is 10 s long


class TestTheDebugRecord:
    def test_with_debug_one_line_per_group_in_the_output_folder(self, stage, tmp_path):
        stage.answers = [[parakeet_segment(("はい", 0.1, 0.4))]]
        make(ParakeetCrispASRPipeline, tmp_path, save_metadata_json=True).process(stage.media())
        record = tmp_path / "out" / "raw_subs" / "film.parakeet-crispasr.groups.jsonl"
        rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
        assert [(r["scene"], r["start"], r["end"]) for r in rows] == [(0, 1.0, 2.0), (0, 3.0, 4.5), (1, 10.5, 11.5)]
        assert rows[0]["segments"][0]["tokens"] == [["はい", 0.1, 0.4, 1.0]]

    def test_two_passes_of_one_film_keep_their_own_record(self, stage, tmp_path):
        make(ParakeetCrispASRPipeline, tmp_path, save_metadata_json=True).process(stage.media())
        stage.frames = {0: [(1.0, 2.0)], 1: []}
        make(CohereCrispASRPipeline, tmp_path, save_metadata_json=True).process(stage.media())
        folder = tmp_path / "out" / "raw_subs"
        assert len((folder / "film.parakeet-crispasr.groups.jsonl").read_text(encoding="utf-8").splitlines()) == 3
        assert len((folder / "film.cohere-crispasr.groups.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    def test_without_debug_nothing_is_written(self, stage, tmp_path):
        make(ParakeetCrispASRPipeline, tmp_path).process(stage.media())
        assert not (tmp_path / "out" / "raw_subs").exists()
