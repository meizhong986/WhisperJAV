"""Parakeet-CrispASR and Cohere-CrispASR, the two CrispASR Ensemble pipelines of 1.10.0 (master plan 2.6, 3.1-3.6, 4.2-4.3).

Covered for both pipeline classes:
- construction: the defaults, the refusals, and pass settings the pipelines accept but do not use;
- the slice rules (speech groups; fixed windows over the scenes), checked against the cut-point file's contract:
  sorted, not overlapping, inside the audio, at most the engine's limit and at least the minimum slice;
- each pipeline's parser (Parakeet TDT words, Parakeet CTC tokens, Cohere tokens) and its fallback;
- line building with the real stable-ts reconstruction (no model is loaded), and the stitching of scene SRTs;
- the stage methods with stand-in segmenter, scene detector and audio extractor, and the failure field
  (SegmentationResult.error) of the three speech segmenters they use, with each one's model replaced;
- process() end to end, with every heavy collaborator replaced by a stand-in;
- post-processing with content: each step does its job, and only when it is chosen.

Nothing here starts the CrispASR program, downloads a model, reaches the network or touches ~/.cache: an autouse
fixture replaces every such call with one that fails the test. The Japanese lines are neutral test sentences, not
film text. Importing the pipelines imports torch through stable_whisper, as tests/test_hardening.py does; no model
is loaded.
"""

import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import soundfile as sf
import srt

from whisperjav.crispasr import install, manifest
from whisperjav.crispasr.install import CrispASRSetupError, ProgramChoice, StartCheck
from whisperjav.crispasr.line_length import OFF, LineLimits
from whisperjav.crispasr.ojf import CrispSegment, CrispToken, CrispWord, CutPoint, write_vad_import
from whisperjav.crispasr.runner import RUN_FOLDER, CrispASRRunError, CrispOutcome, CrispRunResult, film_run_dir
from whisperjav.modules.subtitle_pipeline import reconstruction
from whisperjav.modules.subtitle_pipeline.reconstruction import REGROUP_JAV, split_frame_to_words
from whisperjav.pipelines import cohere_crispasr_pipeline as cohere_mod
from whisperjav.pipelines import parakeet_crispasr_pipeline as parakeet_mod
from whisperjav.pipelines.cohere_crispasr_pipeline import CohereCrispASRPipeline
from whisperjav.pipelines.parakeet_crispasr_pipeline import ParakeetCrispASRPipeline
from whisperjav.utils.progress_display import DummyProgress

RATE = 16000


# ---------------------------------------------------------------------------------------------------
# The two pipelines, and helpers
# ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Kind:
    """One pipeline class and what these tests need to know about it."""
    cls: type
    module: object
    model: str     # a model this pipeline offers
    max_s: float   # the engine's slice limit (plan 3.4)


PARAKEET = Kind(ParakeetCrispASRPipeline, parakeet_mod, "parakeet-tdt-0.6b-ja", 12.0)
COHERE = Kind(CohereCrispASRPipeline, cohere_mod, "cohere-asr-ja-v0.1", 30.0)
KINDS = [pytest.param(PARAKEET, id="parakeet"), pytest.param(COHERE, id="cohere")]

# Every model each pipeline can run, for the end-to-end runs: those it offers (manifest PIPELINE_MODELS) and
# the Parakeet CTC 1.1B, which only step 2b's measurement reaches through the constructor (MEASUREMENT_MODELS).
MODEL_CASES = [
    pytest.param(PARAKEET, "parakeet-tdt-0.6b-ja", id="parakeet-tdt"),
    pytest.param(PARAKEET, "parakeet-ctc-1.1b-ja", id="parakeet-ctc"),
    pytest.param(COHERE, "cohere-asr-ja-v0.1", id="cohere-v0.1"),
    pytest.param(COHERE, "cohere-asr-ja", id="cohere-ja"),
]


def make(kind: Kind, tmp_path: Path, **kwargs):
    kwargs.setdefault("model_name", kind.model)
    return kind.cls(output_dir=str(tmp_path / "out"), temp_dir=str(tmp_path / "temp"), **kwargs)


def flat(items) -> List[float]:
    """[start, end, start, end, ...] of CutPoints or (start, end) pairs."""
    out: List[float] = []
    for it in items:
        out += [it.start, it.end] if isinstance(it, CutPoint) else [it[0], it[1]]
    return out


def write_wav(path: Path, seconds: float, bursts: Sequence[Tuple[float, float]] = ()) -> Path:
    """A 16 kHz mono WAV: silence, with loud noise in each (start, end) burst."""
    audio = np.zeros(max(1, int(round(seconds * RATE))), dtype=np.float32)
    noise = np.random.default_rng(1234)
    for a, b in bursts:
        i, j = int(a * RATE), min(len(audio), int(b * RATE))
        audio[i:j] = noise.uniform(-0.6, 0.6, j - i).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, RATE, subtype="PCM_16")
    return path


def srt_lines(path: Path) -> List[Tuple[str, float, float]]:
    return [(s.content, s.start.total_seconds(), s.end.total_seconds())
            for s in srt.parse(Path(path).read_text(encoding="utf-8"))]


def assert_lines(actual, expected, tol: float = 0.0015) -> None:
    assert [t for t, _, _ in actual] == [t for t, _, _ in expected]
    assert [v for _, a, b in actual for v in (a, b)] == pytest.approx(
        [v for _, a, b in expected for v in (a, b)], abs=tol)


def assert_words(actual, expected) -> None:
    """Word dicts carry exactly the keys stable-ts accepts here, in the expected order and times."""
    assert all(set(w) == {"word", "start", "end"} for w in actual), actual
    assert [w["word"] for w in actual] == [t for t, _, _ in expected]
    assert [v for w in actual for v in (w["start"], w["end"])] == pytest.approx(
        [v for _, a, b in expected for v in (a, b)], abs=1e-9)


def seg(start, end, text, words=(), tokens=(), chunk_id=0) -> CrispSegment:
    return CrispSegment(start=start, end=end, text=text, chunk_id=chunk_id, words=list(words), tokens=list(tokens))


def crisp_segment(backend: str, parts, cut: CutPoint, chunk_id: int = 0) -> CrispSegment:
    """One segment shaped as the backend's -ojf JSON is (second study, runs S2-S4), from (text, start, end) pieces.

    Parakeet TDT: native words, plus the blank " " token that starts long before the segment (runs S9, M1).
    Cohere: one "word" holding the whole segment, the segment ending at the slice end, and tokens with a
    zero-length first token (24 of 54 in run S1).
    """
    text = "".join(p for p, _, _ in parts)
    first, a, b = parts[0]
    if backend == "parakeet":
        return seg(a, parts[-1][2], text, words=[CrispWord(p, x, y) for p, x, y in parts],
                   tokens=[CrispToken(" ", a - 10.9, a)] + [CrispToken(p, x, y) for p, x, y in parts],
                   chunk_id=chunk_id)
    if backend == "cohere":
        tokens = ([CrispToken(first[:1], a, a), CrispToken(first[1:], a, b)]
                  + [CrispToken(p, x, y) for p, x, y in parts[1:]])
        return seg(a, cut.end, text, words=[CrispWord(text, a, cut.end)], tokens=tokens, chunk_id=chunk_id)
    raise ValueError(backend)


# ---------------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_program_model_or_network(monkeypatch):
    """No test may start the program, download anything or read ~/.cache: each test sets its own stand-ins."""
    def refuse(name):
        def call(*args, **kwargs):
            raise AssertionError(f"{name} was called: the test must replace it with a stand-in")
        return call

    for name in ("prepare_program", "ensure_program", "start_check", "ensure_model", "_download", "_hf_download"):
        monkeypatch.setattr(install, name, refuse(f"install.{name}"))
    for module in (parakeet_mod, cohere_mod):
        monkeypatch.setattr(module, "run_crispasr_json", refuse("run_crispasr_json"))


@pytest.fixture
def console():
    """The whisperjav logger's lines. It does not propagate to the root logger, so caplog cannot see them."""
    lines: List[str] = []

    class Keep(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = Keep(level=logging.DEBUG)
    log = logging.getLogger("whisperjav")
    log.addHandler(handler)
    try:
        yield lines
    finally:
        log.removeHandler(handler)


@pytest.fixture
def reconstruct_calls(monkeypatch):
    """Records what reaches the real reconstruct_from_words, then calls it."""
    real = reconstruction.reconstruct_from_words
    calls: List[dict] = []

    def spy(words, audio_path, suppress_silence=True, regroup=REGROUP_JAV):
        calls.append({"words": [dict(w) for w in words], "audio_path": audio_path,
                      "suppress_silence": suppress_silence, "regroup": regroup})
        return real(words, audio_path, suppress_silence=suppress_silence, regroup=regroup)

    monkeypatch.setattr(reconstruction, "reconstruct_from_words", spy)
    return calls


# ---------------------------------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------------------------------

def test_parakeet_defaults(tmp_path):
    pipe = ParakeetCrispASRPipeline(output_dir=str(tmp_path / "out"), temp_dir=str(tmp_path / "temp"))
    assert pipe.get_mode_name() == pipe.PIPELINE_NAME == "parakeet-crispasr"
    assert pipe.model is manifest.MODELS[manifest.PIPELINE_DEFAULT_MODEL["parakeet-crispasr"]]
    assert (pipe.model.name, pipe.model.backend) == ("parakeet-tdt-0.6b-ja", "parakeet")
    assert pipe.speech_segmenter == "whisperseg"
    # Parakeet's own segments, one line each (W129 4); the line-length rule off (W138 3)
    assert (pipe.slice_form, pipe.line_mode, pipe.lang_code) == ("speech", "segments", "ja")
    assert pipe.line_limits == OFF and not pipe.line_limits.active
    # balanced sets WhisperSeg's detection threshold to its built-in 0.35, the value step 2b ran with
    assert (pipe.sensitivity, pipe.segmenter_detection) == ("balanced", {"threshold": 0.35})
    assert pipe.postprocess == ("nonverbal", "nonlinguistic", "overlap")
    assert (pipe.MAX_SLICE_S, pipe.MIN_SLICE_S) == (12.0, 0.1)
    assert pipe.keep_temp_files is False and pipe.save_metadata_json is False
    assert pipe.degradations == [] and pipe.notes == []
    assert isinstance(pipe.progress, DummyProgress)
    assert (pipe.output_dir, pipe.temp_dir) == (tmp_path / "out", tmp_path / "temp")


def test_cohere_defaults(tmp_path):
    pipe = CohereCrispASRPipeline(output_dir=str(tmp_path / "out"), temp_dir=str(tmp_path / "temp"))
    assert pipe.get_mode_name() == pipe.PIPELINE_NAME == "cohere-crispasr"
    # v0.1 by default, on the step 2b scores (W129 1)
    assert pipe.model is manifest.MODELS[manifest.PIPELINE_DEFAULT_MODEL["cohere-crispasr"]]
    assert pipe.model is manifest.MODELS["cohere-asr-ja-v0.1"] and pipe.model.backend == "cohere"
    assert pipe.speech_segmenter == "whisperseg"
    assert (pipe.slice_form, pipe.line_mode, pipe.lang_code) == ("speech", "regroup", "ja")
    assert pipe.line_limits == OFF
    assert (pipe.sensitivity, pipe.segmenter_detection) == ("balanced", {"threshold": 0.35})
    assert pipe.postprocess == ("nonverbal", "nonlinguistic", "overlap")
    assert (pipe.MAX_SLICE_S, pipe.MIN_SLICE_S) == (30.0, 0.1)
    assert pipe.LINE_MODES == ("regroup",)
    assert pipe.degradations == [] and pipe.notes == []


@pytest.mark.parametrize("kind", KINDS)
def test_construction_prepares_nothing(kind, tmp_path):
    """The program and model are made ready by process(), never by the constructor (the guard fixture fails it)."""
    pipe = make(kind, tmp_path)
    assert pipe._program is None and pipe._model_path is None and pipe._setup_error is None


@pytest.mark.parametrize("kind", KINDS)
def test_every_model_listed_for_the_pipeline_constructs(kind, tmp_path):
    name = kind.cls.PIPELINE_NAME
    assert name in manifest.CRISPASR_PIPELINES
    for model in manifest.PIPELINE_MODELS[name] + manifest.MEASUREMENT_MODELS[name]:
        assert make(kind, tmp_path, model_name=model).model is manifest.MODELS[model]


def test_the_parakeet_ctc_model_is_not_offered_but_the_constructor_accepts_it(tmp_path):
    # Plan 3.3: offered only if step 2b supports it; the measurement script builds the pipeline directly.
    assert "parakeet-ctc-1.1b-ja" not in manifest.PIPELINE_MODELS["parakeet-crispasr"]
    pipe = make(PARAKEET, tmp_path, model_name="parakeet-ctc-1.1b-ja")
    assert pipe.model is manifest.MODELS["parakeet-ctc-1.1b-ja"]
    assert pipe.model.backend == "fastconformer-ctc"
    assert pipe.model.extra_args == ("--punc-model", "none")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("language", ["en", "ko", "zh", "English"])
def test_a_non_japanese_source_language_is_refused(kind, language, tmp_path):
    with pytest.raises(ValueError, match="Japanese only"):
        make(kind, tmp_path, language=language)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("language", ["ja", "JA", " jp ", "jpn", "Japanese", None])
def test_japanese_is_accepted_in_its_spellings(kind, language, tmp_path):
    assert make(kind, tmp_path, language=language).lang_code == "ja"


@pytest.mark.parametrize("kind", KINDS)
def test_direct_to_english_still_writes_japanese_with_one_console_line(kind, tmp_path, console):
    pipe = make(kind, tmp_path, subs_language="direct-to-english")
    assert pipe.lang_code == "ja" and pipe.subs_language == "direct-to-english"
    assert len([line for line in console if "cannot translate" in line]) == 1


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", ["none", "None", " NONE "])
def test_no_speech_segmenter_is_refused_for_speech_slices(kind, value, tmp_path):
    with pytest.raises(ValueError, match="needs a speech segmenter"):
        make(kind, tmp_path, speech_segmenter=value)


@pytest.mark.parametrize("kind", KINDS)
def test_no_speech_segmenter_is_allowed_with_window_slices(kind, tmp_path):
    pipe = make(kind, tmp_path, speech_segmenter="none", slice_form="windows")
    assert (pipe.speech_segmenter, pipe.slice_form) == ("none", "windows")


@pytest.mark.parametrize("kind", KINDS)
def test_speech_segmenter_names_are_normalised(kind, tmp_path):
    assert make(kind, tmp_path, speech_segmenter=None).speech_segmenter == "whisperseg"
    assert make(kind, tmp_path, speech_segmenter=" TEN ").speech_segmenter == "ten"


@pytest.mark.parametrize("kind", KINDS)
def test_an_unknown_model_is_refused_with_the_choices(kind, tmp_path):
    with pytest.raises(ValueError, match="is not a model of") as info:
        make(kind, tmp_path, model_name="whisper-large-v3")
    for choice in manifest.PIPELINE_MODELS[kind.cls.PIPELINE_NAME]:
        assert choice in str(info.value)


@pytest.mark.parametrize("name", [None, ""])
def test_cohere_without_a_model_runs_its_default(name, tmp_path):
    pipe = CohereCrispASRPipeline(output_dir=str(tmp_path / "out"), temp_dir=str(tmp_path / "temp"), model_name=name)
    assert pipe.model is manifest.MODELS["cohere-asr-ja-v0.1"]
    # the other fine-tune stays a choice (Q13)
    assert make(COHERE, tmp_path, model_name="cohere-asr-ja").model is manifest.MODELS["cohere-asr-ja"]


@pytest.mark.parametrize("model", manifest.PIPELINE_MODELS["parakeet-crispasr"]
                         + manifest.MEASUREMENT_MODELS["parakeet-crispasr"])
def test_a_parakeet_model_on_the_cohere_pipeline_is_refused(model, tmp_path):
    with pytest.raises(ValueError, match="is not a model of cohere-crispasr"):
        make(COHERE, tmp_path, model_name=model)


@pytest.mark.parametrize("model", manifest.PIPELINE_MODELS["cohere-crispasr"])
def test_a_cohere_model_on_the_parakeet_pipeline_is_refused(model, tmp_path):
    with pytest.raises(ValueError, match="is not a model of parakeet-crispasr"):
        make(PARAKEET, tmp_path, model_name=model)


@pytest.mark.parametrize("kind", KINDS)
def test_unknown_slice_form_line_mode_or_post_processing_step_is_refused(kind, tmp_path):
    with pytest.raises(ValueError, match="unknown slice form"):
        make(kind, tmp_path, slice_form="scenes")
    with pytest.raises(ValueError, match="unknown line mode"):
        make(kind, tmp_path, line_mode="sentences")
    with pytest.raises(ValueError, match="unknown post-processing step"):
        make(kind, tmp_path, postprocess=("nonverbal", "sanitizer"))


def test_cohere_has_no_segments_line_mode(tmp_path):
    with pytest.raises(ValueError, match="unknown line mode"):
        make(COHERE, tmp_path, line_mode="segments")


def test_parakeet_accepts_the_segments_line_mode(tmp_path):
    assert make(PARAKEET, tmp_path, line_mode="segments").line_mode == "segments"


@pytest.mark.parametrize("kind", KINDS)
def test_post_processing_steps_can_be_chosen(kind, tmp_path):
    assert make(kind, tmp_path, postprocess=()).postprocess == ()
    assert make(kind, tmp_path, postprocess=["overlap"]).postprocess == ("overlap",)


@pytest.mark.parametrize("kind", KINDS)
def test_pass_settings_the_pipeline_does_not_use_are_accepted(kind, tmp_path):
    pipe = make(kind, tmp_path, scene_detector="auditok", speech_enhancer="none", task="transcribe")
    assert pipe.get_mode_name() == kind.cls.PIPELINE_NAME


# Each segmenter's detection setting(s) at aggressive / balanced / conservative, from its own preset YAML
# (plan 3.8 item 4; W134 2, W136, W138 2). Written out, so a change to a preset file shows here.
DETECTION = {
    "whisperseg": [{"threshold": 0.25}, {"threshold": 0.35}, {"threshold": 0.45}],
    "ten": [{"threshold": 0.22}, {"threshold": 0.32}, {"threshold": 0.42}],
    "silero-v6.2": [{"threshold": 0.12}, {"threshold": 0.32}, {"threshold": 0.42}],
    "silero-v4.0": [{"threshold": 0.18}, {"threshold": 0.28}, {"threshold": 0.41}],
    "silero-v3.1": [{"threshold": 0.18}, {"threshold": 0.28}, {"threshold": 0.41}],
    "firered-vad": [{"threshold": 0.3}, {"threshold": 0.4}, {"threshold": 0.5}],
    "whisper-vad": [{"no_speech_threshold": 0.5, "logprob_threshold": -1.5},
                    {"no_speech_threshold": 0.6, "logprob_threshold": -1.0},
                    {"no_speech_threshold": 0.7, "logprob_threshold": -0.8}],
    "whisper-vad-tiny": [{"no_speech_threshold": 0.5, "logprob_threshold": -1.5},
                         {"no_speech_threshold": 0.6, "logprob_threshold": -1.0},
                         {"no_speech_threshold": 0.7, "logprob_threshold": -0.8}],
    "nemo-lite": [{"onset": 0.7, "offset": 0.5}, {"onset": 0.8, "offset": 0.6}, {"onset": 0.85, "offset": 0.7}],
    "nemo-diarization": [{"onset": 0.7, "offset": 0.5}, {"onset": 0.8, "offset": 0.6},
                         {"onset": 0.85, "offset": 0.7}],
}
SENSITIVITIES = ("aggressive", "balanced", "conservative")


@pytest.mark.parametrize("segmenter", sorted(DETECTION))
@pytest.mark.parametrize("kind", KINDS)
def test_sensitivity_sets_only_the_segmenters_detection_setting(kind, segmenter, tmp_path):
    for sensitivity, expected in zip(SENSITIVITIES, DETECTION[segmenter]):
        pipe = make(kind, tmp_path, speech_segmenter=segmenter, sensitivity=sensitivity)
        assert (pipe.sensitivity, pipe.segmenter_detection) == (sensitivity, expected)


@pytest.mark.parametrize("segmenter", sorted(DETECTION))
def test_the_detection_settings_pass_the_segmenter_factorys_filter_unchanged(segmenter):
    """The factory drops arguments a back end does not take (_sanitize_params); these must all arrive."""
    from whisperjav.modules.speech_segmentation.factory import SpeechSegmenterFactory

    for expected in DETECTION[segmenter]:
        sent = dict(max_group_duration_s=12.0, chunk_threshold_s=1.0, start_pad_ms=100, end_pad_ms=0,
                    max_speech_duration_s=11.9, **expected)
        kept = SpeechSegmenterFactory._sanitize_params(segmenter, dict(sent))
        assert {k: kept.get(k) for k in expected} == expected


@pytest.mark.parametrize("kind", KINDS)
def test_sensitivity_is_balanced_when_none_is_given(kind, tmp_path):
    for value in (None, ""):
        pipe = make(kind, tmp_path, sensitivity=value)
        assert (pipe.sensitivity, pipe.segmenter_detection) == ("balanced", {"threshold": 0.35})


@pytest.mark.parametrize("kind", KINDS)
def test_window_slices_use_no_segmenter_so_sensitivity_sets_nothing(kind, tmp_path):
    pipe = make(kind, tmp_path, slice_form="windows", sensitivity="aggressive")
    assert pipe.segmenter_detection == {}
    assert make(kind, tmp_path, slice_form="windows", speech_segmenter="none").segmenter_detection == {}


@pytest.mark.parametrize("kind", KINDS)
def test_a_segmenter_whose_detection_setting_is_unknown_runs_its_own_values_with_a_warning(kind, tmp_path,
                                                                                           console):
    pipe = make(kind, tmp_path, speech_segmenter="stand-in-vad", sensitivity="aggressive")
    assert pipe.segmenter_detection == {}
    assert any("Sensitivity does not reach the speech segmenter 'stand-in-vad'" in line for line in console)


@pytest.mark.parametrize("kind", KINDS)
def test_the_ready_line_and_the_metadata_name_the_sensitivity_and_its_setting(kind, tmp_path, console):
    pipe = make(kind, tmp_path, sensitivity="conservative")
    assert any(f"speech segmenter whisperseg (conservative: threshold 0.45)" in line for line in console)
    md = pipe._new_metadata(tmp_path / "film.mp4", "film")
    config = md["config"]["crispasr"]
    assert (config["sensitivity"], config["segmenter_detection"]) == ("conservative", {"threshold": 0.45})
    assert config["line_limits"] == {"max_s": None, "max_chars": None}
    assert config["line_mode"] == pipe.line_mode


# ---------------------------------------------------------------------------------------------------
# Slice rules (plan 3.4): hand-made cases
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_groups_overlapping_at_a_scene_edge_become_one_slice_of_the_earlier_scene(kind, tmp_path):
    pipe = make(kind, tmp_path)
    G = kind.module._Group
    cuts, owner = pipe._speech_slices([G(1, 14.6, 18.0), G(0, 10.0, 15.0)], 100.0)  # given out of order
    assert flat(cuts) == pytest.approx([10.0, 18.0])
    assert owner == [0]


@pytest.mark.parametrize("kind", KINDS)
def test_a_merged_slice_over_the_limit_is_split_and_every_piece_stays_with_the_earlier_scene(kind, tmp_path):
    pipe = make(kind, tmp_path)
    G, m = kind.module._Group, kind.max_s
    groups = [G(0, 0.0, 0.6 * m, [0.4 * m]), G(1, 0.6 * m - 0.4, 1.3 * m, [1.0 * m])]
    cuts, owner = pipe._speech_slices(groups, 10 * m)
    # merged 0..1.3m; the pause nearest its middle (0.65m) is 0.4m
    assert flat(cuts) == pytest.approx([0.0, 0.4 * m, 0.4 * m, 1.3 * m])
    assert owner == [0, 0]


@pytest.mark.parametrize("kind", KINDS)
def test_a_long_group_is_split_at_the_pause_nearest_its_middle(kind, tmp_path):
    pipe = make(kind, tmp_path)
    G, m = kind.module._Group, kind.max_s
    s, e = 10.0, 10.0 + 1.5 * m
    cuts, owner = pipe._speech_slices([G(2, s, e, [s + 0.2 * m, s + 0.8 * m, s + 1.4 * m])], 10 * m)
    assert flat(cuts) == pytest.approx([s, s + 0.8 * m, s + 0.8 * m, e])
    assert owner == [2, 2]


@pytest.mark.parametrize("kind", KINDS)
def test_a_long_group_without_a_pause_inside_the_margins_is_split_at_the_limit(kind, tmp_path):
    pipe = make(kind, tmp_path)
    G, m = kind.module._Group, kind.max_s
    s, e = 10.0, 10.0 + 1.5 * m
    # pauses within 0.5 s of an end (two of them exactly at the margin) or outside the group are not used
    pauses = [s + 0.3, s + 0.5, e - 0.5, e - 0.2, s - 2.0, e + 1.0]
    cuts, owner = pipe._speech_slices([G(0, s, e, pauses)], 10 * m)
    assert flat(cuts) == pytest.approx([s, s + m, s + m, e])
    assert owner == [0, 0]


@pytest.mark.parametrize("kind", KINDS)
def test_a_group_several_limits_long_without_pauses_is_cut_at_each_limit(kind, tmp_path):
    pipe = make(kind, tmp_path)
    G, m = kind.module._Group, kind.max_s
    s = 3.0
    cuts, owner = pipe._speech_slices([G(1, s, s + 2.5 * m)], 10 * m)
    assert flat(cuts) == pytest.approx([s, s + m, s + m, s + 2 * m, s + 2 * m, s + 2.5 * m])
    assert owner == [1, 1, 1]


@pytest.mark.parametrize("kind", KINDS)
def test_a_pauseless_group_just_over_the_limit_leaves_no_sliver_slice(kind, tmp_path):
    """A group a little over the limit with no usable pause must not leave a slice shorter than MIN_SLICE_S.

    The slice rule drops groups shorter than MIN_SLICE_S (0.1 s), so no slice it writes may be shorter; a
    remainder of 0.05 s, or of one sample, is a slice the program is not meant to get (plan 3.4).
    """
    pipe = make(kind, tmp_path)
    G, m = kind.module._Group, kind.max_s
    problems = []
    for k in (1, 2):
        for over in (0.05, 1.0 / RATE):
            s, e = 7.0, 7.0 + k * m + over
            cuts, _ = pipe._speech_slices([G(0, s, e)], 1000.0)
            # still a contiguous cover of the group, within the limit
            assert cuts[0].start == s and cuts[-1].end == pytest.approx(e)
            assert all(b.start == a.end for a, b in zip(cuts, cuts[1:]))
            assert all(c.duration <= m + 1e-9 for c in cuts)
            short = [round(c.duration, 6) for c in cuts if c.duration < pipe.MIN_SLICE_S]
            if short:
                problems.append(f"{k} x {m:.0f} s + {over:.6f} s -> slices of {short} s")
            path = tmp_path / "cuts.json"
            write_vad_import(path, cuts, audio_seconds=1000.0)
            for sl in json.loads(path.read_text(encoding="utf-8"))["crispasr_vad"]["slices"]:
                if sl["end"] - sl["start"] < pipe.MIN_SLICE_S * RATE:
                    problems.append(f"{k} x {m:.0f} s + {over:.6f} s -> a slice of {sl['end'] - sl['start']} "
                                    f"sample(s) in the cut-point file")
    assert not problems, "slices shorter than MIN_SLICE_S:\n" + "\n".join(problems)


@pytest.mark.parametrize("kind", KINDS)
def test_groups_shorter_than_the_minimum_are_dropped(kind, tmp_path):
    pipe = make(kind, tmp_path)
    G = kind.module._Group
    assert pipe._speech_slices([G(0, 5.0, 5.05), G(0, 8.0, 8.0)], 100.0) == ([], [])
    cuts, owner = pipe._speech_slices([G(0, 5.0, 5.15)], 100.0)
    assert flat(cuts) == pytest.approx([5.0, 5.15]) and owner == [0]


@pytest.mark.parametrize("kind", KINDS)
def test_groups_outside_the_audio_are_clamped_or_dropped(kind, tmp_path):
    pipe = make(kind, tmp_path)
    G = kind.module._Group
    audio = 30.0
    assert flat(pipe._speech_slices([G(0, -0.3, 2.0)], audio)[0]) == pytest.approx([0.0, 2.0])
    assert flat(pipe._speech_slices([G(0, 29.5, 31.0)], audio)[0]) == pytest.approx([29.5, 30.0])
    for start, end in ((29.95, 31.0), (30.0, 31.0), (31.0, 33.0), (-3.0, -1.0)):
        assert pipe._speech_slices([G(0, start, end)], audio) == ([], []), (start, end)


@pytest.mark.parametrize("kind", KINDS)
def test_no_groups_give_no_slices(kind, tmp_path):
    assert make(kind, tmp_path)._speech_slices([], 100.0) == ([], [])


@pytest.mark.parametrize("kind", KINDS)
def test_windows_cut_overlapping_scenes_at_the_middle_and_never_merge(kind, tmp_path):
    pipe = make(kind, tmp_path)
    S, m = kind.module._Scene, kind.max_s
    scenes = [S(1, Path("b.wav"), 19.8, 45.0), S(0, Path("a.wav"), 0.0, 20.4)]  # given out of order
    cuts, owner = pipe._window_slices(scenes, 45.0)
    middle = (19.8 + 20.4) / 2.0
    a = [c for c, o in zip(cuts, owner) if o == 0]
    b = [c for c, o in zip(cuts, owner) if o == 1]
    assert owner == [0] * len(a) + [1] * len(b)
    assert (a[0].start, a[-1].end) == pytest.approx((0.0, middle))
    assert (b[0].start, b[-1].end) == pytest.approx((middle, 45.0))
    assert len(a) == math.ceil(middle / m) and len(b) == math.ceil((45.0 - middle) / m)
    for windows in (a, b):
        assert all(w.duration <= m + 1e-9 for w in windows)
        assert [w.duration for w in windows] == pytest.approx([windows[0].duration] * len(windows))


@pytest.mark.parametrize("kind", KINDS)
def test_windows_are_the_fewest_equal_pieces_within_the_limit(kind, tmp_path):
    pipe = make(kind, tmp_path)
    S, m = kind.module._Scene, kind.max_s
    for length, count in ((2.5 * m, 3), (2.0 * m, 2), (0.5 * m, 1), (m, 1)):
        cuts, owner = pipe._window_slices([S(4, Path("s.wav"), 1.0, 1.0 + length)], 10 * m)
        assert len(cuts) == count and owner == [4] * count, length
        assert flat(cuts) == pytest.approx(
            [v for i in range(count) for v in (1.0 + i * length / count, 1.0 + (i + 1) * length / count)])


@pytest.mark.parametrize("kind", KINDS)
def test_windows_stay_inside_the_audio(kind, tmp_path):
    pipe = make(kind, tmp_path)
    S = kind.module._Scene
    scenes = [S(0, Path("a.wav"), -0.2, 20.0), S(1, Path("b.wav"), 19.5, 40.7), S(2, Path("c.wav"), 40.2, 50.0)]
    cuts, owner = pipe._window_slices(scenes, 40.0)
    assert cuts[0].start == 0.0 and cuts[-1].end == 40.0
    assert 2 not in owner  # the scene that starts after the audio ends gives no window
    a_end = max(c.end for c, o in zip(cuts, owner) if o == 0)
    b_start = min(c.start for c, o in zip(cuts, owner) if o == 1)
    assert a_end == b_start == pytest.approx((19.5 + 20.0) / 2.0)


# ---------------------------------------------------------------------------------------------------
# Slice rules: deterministic synthetic inputs, checked against the cut-point file's contract
# ---------------------------------------------------------------------------------------------------

SEEDS = range(120)


def random_groups(kind: Kind, rng: random.Random):
    """Speech groups as the segmenter and the scene edges can give them, plus the awkward cases around the rules."""
    G, m = kind.module._Group, kind.max_s
    audio_s = rng.uniform(5.0, 600.0)
    n_scenes = rng.randint(1, 10)
    groups = []
    for _ in range(rng.randint(0, 40)):
        start = rng.uniform(-2.0, audio_s + 2.0)
        r = rng.random()
        if r < 0.15:
            length = rng.uniform(0.0, 0.25)                      # around the 0.1 s minimum
        elif r < 0.55:
            length = rng.uniform(0.25, m)                        # an ordinary group
        elif r < 0.85:
            length = rng.uniform(m, 4.0 * m)                     # too long for one slice
        else:
            length = rng.randint(1, 3) * m + rng.uniform(-0.05, 0.3)  # around a multiple of the limit
        end = start + length
        pauses = sorted(rng.uniform(start - 1.0, end + 1.0) for _ in range(rng.choice((0, 0, 1, 2, 4, 9))))
        groups.append(G(rng.randrange(n_scenes), start, end, pauses))
    # Neighbouring semantic scenes share up to 0.7 s, so a group can start before another one ends.
    for g in list(groups):
        if rng.random() < 0.25:
            s2 = g.end - rng.uniform(0.0, 0.7)
            groups.append(G(rng.randrange(n_scenes), s2, s2 + rng.uniform(0.1, m)))
    rng.shuffle(groups)
    return groups, audio_s


def random_scenes(kind: Kind, rng: random.Random):
    """Scenes in film time as the semantic detector gives them: neighbours overlap by up to 0.7 s, or leave a gap."""
    S = kind.module._Scene
    scenes = []
    t = rng.choice((0.0, rng.uniform(-0.3, 0.35)))
    for i in range(rng.randint(1, 12)):
        length = rng.uniform(1.5, 70.0)
        scenes.append(S(i, Path(f"scene_{i:04d}.wav"), t, t + length))
        t = t + length - rng.choice((0.0, rng.uniform(0.05, 0.7))) + rng.choice((0.0, 0.0, rng.uniform(0.0, 5.0)))
    audio_s = scenes[-1].end - rng.uniform(0.0, 0.5)  # the last scene's padded end may pass the audio's end
    rng.shuffle(scenes)
    return scenes, audio_s


def union(intervals) -> List[List[float]]:
    """Merged intervals; touching ones join too."""
    out: List[List[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def contract_problems(cuts, owner, audio_s: float, max_s: float, path: Path) -> List[str]:
    """Sorted, not overlapping, not empty, inside the audio and within the limit; write_vad_import accepts them."""
    problems = []
    if len(cuts) != len(owner):
        problems.append(f"{len(cuts)} slices but {len(owner)} owners")
    previous_end = 0.0
    for i, c in enumerate(cuts):
        if c.start < previous_end:
            problems.append(f"slice {i} {c} starts before the previous one ends ({previous_end})")
        if c.start < 0.0 or c.end > audio_s:
            problems.append(f"slice {i} {c} lies outside the audio (0-{audio_s})")
        if not c.end > c.start:
            problems.append(f"slice {i} {c} is empty")
        if c.duration > max_s + 1e-9:
            problems.append(f"slice {i} {c} is longer than {max_s} s")
        previous_end = c.end
    if cuts:
        try:
            write_vad_import(path, cuts, audio_seconds=audio_s)
        except ValueError as e:
            problems.append(f"write_vad_import refused the slices: {e}")
        else:
            doc = json.loads(path.read_text(encoding="utf-8"))["crispasr_vad"]
            if doc["kind"] != "chunks" or doc["num_slices"] != len(cuts):
                problems.append(f"cut-point file header {doc['kind']}/{doc['num_slices']}")
            previous = 0
            for i, sl in enumerate(doc["slices"]):
                if sl["start"] < previous:
                    problems.append(f"slice {i} overlaps the previous one in samples")
                previous = sl["end"]
    return problems


@pytest.mark.parametrize("kind", KINDS)
def test_speech_slices_keep_the_cut_point_contract(kind, tmp_path):
    """For many synthetic group lists: the contract, no speech lost or added, and each slice's scene."""
    pipe = make(kind, tmp_path)
    failures = []
    for seed in SEEDS:
        groups, audio_s = random_groups(kind, random.Random(seed))
        cuts, owner = pipe._speech_slices(groups, audio_s)
        problems = contract_problems(cuts, owner, audio_s, kind.max_s, tmp_path / "cuts.json")

        kept = sorted(((max(0.0, g.start), min(audio_s, g.end), g.scene) for g in groups
                       if not (min(audio_s, g.end) - max(0.0, g.start) < pipe.MIN_SLICE_S)),
                      key=lambda t: (t[0], t[1]))
        expected_cover = union((s, e) for s, e, _ in kept)
        actual_cover = union((c.start, c.end) for c in cuts)
        if len(expected_cover) != len(actual_cover) or flat(actual_cover) != pytest.approx(flat(expected_cover)):
            problems.append(f"slices cover {actual_cover}, the kept groups {expected_cover}")

        # Overlapping groups merge; the merged slice belongs to the scene of its earliest group.
        clusters: List[list] = []
        for s, e, scene in kept:
            if clusters and s < clusters[-1][1]:
                clusters[-1][1] = max(clusters[-1][1], e)
            else:
                clusters.append([s, e, scene])
        for c, scene in zip(cuts, owner):
            home = [k for k in clusters if k[0] - 1e-9 <= c.start and c.end <= k[1] + 1e-9]
            if len(home) != 1 or home[0][2] != scene:
                problems.append(f"slice {c} owned by scene {scene}, expected {[k[2] for k in home]}")
        if problems:
            failures.append(f"seed {seed}: " + "; ".join(problems[:3]))
    assert not failures, "\n".join(failures[:10])


@pytest.mark.parametrize("kind", KINDS)
def test_speech_slices_are_never_shorter_than_the_minimum(kind, tmp_path):
    """MIN_SLICE_S holds for every slice written, not only for the groups read (see the sliver test above)."""
    pipe = make(kind, tmp_path)
    failures = []
    for seed in SEEDS:
        groups, audio_s = random_groups(kind, random.Random(seed))
        cuts, _ = pipe._speech_slices(groups, audio_s)
        short = [f"{c.start:.3f}-{c.end:.3f}" for c in cuts if c.duration < pipe.MIN_SLICE_S]
        if short:
            failures.append(f"seed {seed}: {short}")
    assert not failures, f"{len(failures)} of {len(SEEDS)} inputs gave slices shorter than " \
                         f"{pipe.MIN_SLICE_S} s:\n" + "\n".join(failures[:10])


@pytest.mark.parametrize("kind", KINDS)
def test_window_slices_keep_the_cut_point_contract(kind, tmp_path):
    """For many synthetic scene lists: the contract, equal windows per scene, the middle cut, no merging."""
    pipe = make(kind, tmp_path)
    m = kind.max_s
    failures = []
    for seed in SEEDS:
        scenes, audio_s = random_scenes(kind, random.Random(seed))
        cuts, owner = pipe._window_slices(scenes, audio_s)
        problems = contract_problems(cuts, owner, audio_s, m, tmp_path / "cuts.json")
        problems += [f"window {c} shorter than the minimum" for c in cuts if c.duration < pipe.MIN_SLICE_S]

        span = {sc.index: (max(0.0, sc.start), min(audio_s, sc.end)) for sc in scenes}
        windows = {}
        for c, o in zip(cuts, owner):
            windows.setdefault(o, []).append(c)
            if not (span[o][0] - 1e-9 <= c.start and c.end <= span[o][1] + 1e-9):
                problems.append(f"window {c} of scene {o} lies outside that scene {span[o]}")
        for o, ws in windows.items():
            length = ws[-1].end - ws[0].start
            if len(ws) != max(1, math.ceil(length / m - 1e-9)):
                problems.append(f"scene {o}: {len(ws)} windows over {length:.3f} s")
            if any(abs(w.duration - ws[0].duration) > 1e-9 * max(1.0, length) for w in ws):
                problems.append(f"scene {o}: unequal windows {[round(w.duration, 6) for w in ws]}")
            if any(b.start != a.end for a, b in zip(ws, ws[1:])):
                problems.append(f"scene {o}: windows do not join")

        ordered = sorted(scenes, key=lambda sc: sc.start)
        for a, b in zip(ordered, ordered[1:]):
            a_end, b_start = span[a.index][1], span[b.index][0]
            if b_start < a_end:  # an overlap: cut at its middle, never merged
                middle = (b_start + a_end) / 2.0
                if (windows[a.index][-1].end != pytest.approx(middle)
                        or windows[b.index][0].start != pytest.approx(middle)):
                    problems.append(f"scenes {a.index}/{b.index} overlap; not cut at its middle {middle:.3f}")

        if flat(union((c.start, c.end) for c in cuts)) != pytest.approx(flat(union(span.values()))):
            problems.append("the windows do not cover the scenes")
        if problems:
            failures.append(f"seed {seed}: " + "; ".join(problems[:3]))
    assert not failures, "\n".join(failures[:10])


# ---------------------------------------------------------------------------------------------------
# Parsers (plan 3.5)
# ---------------------------------------------------------------------------------------------------

def test_parakeet_words_are_stripped_and_blank_words_dropped(tmp_path):
    pipe = make(PARAKEET, tmp_path)
    s = seg(10.2, 11.1, " テストです。", words=[
        CrispWord(" テスト", 10.2, 10.6), CrispWord(" ", 10.6, 10.7), CrispWord("", 10.7, 10.7),
        CrispWord("です ", 10.7, 11.0), CrispWord("。", 11.0, 11.1)])
    words, clamped = pipe._word_dicts(s, CutPoint(10.0, 14.0))
    assert_words(words, [("テスト", 10.2, 10.6), ("です", 10.7, 11.0), ("。", 11.0, 11.1)])
    assert clamped == 0


def test_parakeet_word_times_are_clamped_to_the_slice_and_counted(tmp_path):
    pipe = make(PARAKEET, tmp_path)
    s = seg(9.0, 15.0, "テストです。", words=[
        CrispWord("テスト", 9.0, 10.3), CrispWord("です", 13.9, 14.03), CrispWord("。", 14.5, 15.0)])
    words, clamped = pipe._word_dicts(s, CutPoint(10.0, 14.0))
    assert_words(words, [("テスト", 10.0, 10.3), ("です", 13.9, 14.0), ("。", 14.0, 14.0)])
    # moved more than 0.05 s: the first start (1.0 s), the last word's start and end; 0.03 s is not counted
    assert clamped == 3


def test_parakeet_never_reads_its_blank_tokens(tmp_path):
    pipe = make(PARAKEET, tmp_path)
    cut = CutPoint(20.0, 26.0)
    words = [CrispWord("テスト", 21.0, 21.4), CrispWord("です", 21.4, 21.7)]
    tokens = [CrispToken(" ", 21.0 - 10.9, 21.0, 0.99), CrispToken("テ", 21.0, 21.1), CrispToken("スト", 21.1, 21.4),
              CrispToken(" ", 5.0, 6.0), CrispToken("です", 21.4, 21.7)]
    with_tokens = pipe._word_dicts(seg(21.0, 21.7, "テストです", words, tokens), cut)
    without = pipe._word_dicts(seg(21.0, 21.7, "テストです", words, []), cut)
    assert with_tokens == without
    assert with_tokens[1] == 0
    # nor are the tokens turned into words when there are none: the segment's own text is shared out instead
    only_tokens, clamped = pipe._word_dicts(seg(21.0, 21.7, "テストです", [], tokens), cut)
    assert only_tokens == split_frame_to_words("テストです", 21.0, 21.7) and clamped == 0


def test_parakeet_ctc_tokens_end_where_the_next_starts_and_the_last_at_the_slice_end(tmp_path):
    pipe = make(PARAKEET, tmp_path, model_name="parakeet-ctc-1.1b-ja")
    tokens = [CrispToken("テ", 20.40, 20.48), CrispToken(" ", 20.48, 20.56), CrispToken(" ス", 20.56, 20.64),
              CrispToken("ト", 20.80, 20.88), CrispToken("", 21.00, 21.08), CrispToken("で", 21.20, 21.28),
              CrispToken("す ", 21.36, 21.44)]
    words, clamped = pipe._word_dicts(seg(20.40, 21.44, "テストです", [], tokens), CutPoint(20.0, 26.0))
    assert_words(words, [("テ", 20.40, 20.56), ("ス", 20.56, 20.80), ("ト", 20.80, 21.20), ("で", 21.20, 21.36),
                         ("す", 21.36, 26.0)])
    assert clamped == 0


def test_parakeet_ctc_token_times_are_clamped_to_the_slice_and_counted(tmp_path):
    pipe = make(PARAKEET, tmp_path, model_name="parakeet-ctc-1.1b-ja")
    tokens = [CrispToken("テ", 19.5, 19.58), CrispToken("ス", 20.5, 20.58), CrispToken("ト", 26.5, 26.58)]
    words, clamped = pipe._word_dicts(seg(19.5, 26.58, "テスト", [], tokens), CutPoint(20.0, 26.0))
    assert_words(words, [("テ", 20.0, 20.5), ("ス", 20.5, 26.0), ("ト", 26.0, 26.0)])
    assert clamped == 3  # the first start, the end taken from the last start, and the last start


def test_cohere_drops_blank_tokens(tmp_path):
    pipe = make(COHERE, tmp_path)
    tokens = [CrispToken("テスト", 31.0, 31.4), CrispToken(" ", 31.4, 31.5), CrispToken("", 31.5, 31.5),
              CrispToken("です", 31.5, 31.8)]
    words, clamped = pipe._word_dicts(seg(31.0, 40.0, "テストです", [], tokens), CutPoint(30.0, 40.0))
    assert_words(words, [("テスト", 31.0, 31.4), ("です", 31.5, 31.8)])
    assert clamped == 0


def test_cohere_folds_a_zero_length_token_into_the_next(tmp_path):
    pipe = make(COHERE, tmp_path)
    tokens = [CrispToken("テ", 31.0, 31.0), CrispToken("スト", 31.0, 31.4), CrispToken("です", 31.4, 31.7)]
    words, _ = pipe._word_dicts(seg(31.0, 40.0, "テストです", [], tokens), CutPoint(30.0, 40.0))
    assert_words(words, [("テスト", 31.0, 31.4), ("です", 31.4, 31.7)])


def test_cohere_folds_a_run_of_zero_length_tokens_into_the_next(tmp_path):
    pipe = make(COHERE, tmp_path)
    tokens = [CrispToken("今", 31.0, 31.0), CrispToken("日", 31.0, 31.0), CrispToken("は", 31.0, 31.0),
              CrispToken("晴れ", 31.0, 31.5), CrispToken("。", 31.5, 31.6)]
    words, _ = pipe._word_dicts(seg(31.0, 40.0, "今日は晴れ。", [], tokens), CutPoint(30.0, 40.0))
    assert_words(words, [("今日は晴れ", 31.0, 31.5), ("。", 31.5, 31.6)])


def test_cohere_folds_across_a_dropped_blank_token(tmp_path):
    pipe = make(COHERE, tmp_path)
    tokens = [CrispToken("テ", 31.0, 31.0), CrispToken(" ", 31.0, 31.1), CrispToken("スト", 31.1, 31.4)]
    words, _ = pipe._word_dicts(seg(31.0, 40.0, "テスト", [], tokens), CutPoint(30.0, 40.0))
    assert_words(words, [("テスト", 31.1, 31.4)])


def test_cohere_keeps_a_zero_length_last_token_as_its_own_word(tmp_path):
    pipe = make(COHERE, tmp_path)
    cut = CutPoint(30.0, 40.0)
    last = [CrispToken("テスト", 31.0, 31.4), CrispToken("です", 31.4, 31.4)]
    words, _ = pipe._word_dicts(seg(31.0, 40.0, "テストです", [], last), cut)
    assert_words(words, [("テスト", 31.0, 31.4), ("です", 31.4, 31.4)])
    # a run of zero-length tokens at the end becomes that last word
    run = [CrispToken("テスト", 31.0, 31.4), CrispToken("で", 31.4, 31.4), CrispToken("す", 31.4, 31.4)]
    words, _ = pipe._word_dicts(seg(31.0, 40.0, "テストです", [], run), cut)
    assert_words(words, [("テスト", 31.0, 31.4), ("です", 31.4, 31.4)])
    # so is a zero-length token followed only by blank tokens
    trailing_blank = [CrispToken("テスト", 31.0, 31.4), CrispToken("です", 31.4, 31.4), CrispToken(" ", 31.4, 31.5)]
    words, _ = pipe._word_dicts(seg(31.0, 40.0, "テストです", [], trailing_blank), cut)
    assert_words(words, [("テスト", 31.0, 31.4), ("です", 31.4, 31.4)])


def test_cohere_token_times_are_clamped_to_the_slice_and_counted(tmp_path):
    pipe = make(COHERE, tmp_path)
    tokens = [CrispToken("テスト", 29.0, 30.2), CrispToken("です", 39.9, 40.3), CrispToken("。", 41.0, 41.2)]
    words, clamped = pipe._word_dicts(seg(29.0, 41.2, "テストです。", [], tokens), CutPoint(30.0, 40.0))
    assert_words(words, [("テスト", 30.0, 30.2), ("です", 39.9, 40.0), ("。", 40.0, 40.0)])
    assert clamped == 4


def test_cohere_reads_its_tokens_not_its_one_whole_segment_word(tmp_path):
    pipe = make(COHERE, tmp_path)
    cut = CutPoint(30.0, 40.0)
    tokens = [CrispToken("テ", 31.0, 31.0), CrispToken("スト", 31.0, 31.4), CrispToken("です", 31.4, 31.7)]
    whole = [CrispWord("テストです", 31.0, 40.0)]
    assert (pipe._word_dicts(seg(31.0, 40.0, "テストです", whole, tokens), cut)
            == pipe._word_dicts(seg(31.0, 40.0, "テストです", [], tokens), cut))


def test_cohere_zero_length_punctuation_joins_the_word_before_it_and_keeps_its_times(tmp_path):
    # W129 5 (ii): "。" no longer opens the next word
    pipe = make(COHERE, tmp_path)
    tokens = [CrispToken("テスト", 31.0, 31.4), CrispToken("。", 31.4, 31.4), CrispToken("です", 31.5, 31.8),
              CrispToken("、", 31.8, 31.8), CrispToken("!?", 31.8, 31.8), CrispToken("もう", 31.9, 32.1)]
    words, clamped = pipe._word_dicts(seg(31.0, 40.0, "テスト。です、!?もう", [], tokens), CutPoint(30.0, 40.0))
    assert_words(words, [("テスト。", 31.0, 31.4), ("です、!?", 31.5, 31.8), ("もう", 31.9, 32.1)])
    assert clamped == 0


def test_cohere_zero_length_punctuation_after_a_waiting_token_joins_that_token(tmp_path):
    # S01E04, 256.25 s in step 2b: "っ", then "、", both zero-length: the text keeps its order
    pipe = make(COHERE, tmp_path)
    tokens = [CrispToken("はい", 30.5, 30.9), CrispToken("っ", 31.0, 31.0), CrispToken("、", 31.0, 31.0),
              CrispToken("て", 31.0, 31.3)]
    words, _ = pipe._word_dicts(seg(30.5, 40.0, "はいっ、て", [], tokens), CutPoint(30.0, 40.0))
    assert_words(words, [("はい", 30.5, 30.9), ("っ、て", 31.0, 31.3)])


def test_cohere_zero_length_punctuation_at_a_segment_start_waits_for_the_next_word(tmp_path):
    pipe = make(COHERE, tmp_path)
    tokens = [CrispToken("、", 31.0, 31.0), CrispToken("テスト", 31.0, 31.4)]
    words, _ = pipe._word_dicts(seg(31.0, 40.0, "、テスト", [], tokens), CutPoint(30.0, 40.0))
    assert_words(words, [("、テスト", 31.0, 31.4)])


def test_cohere_a_zero_length_opening_bracket_still_opens_the_next_word(tmp_path):
    pipe = make(COHERE, tmp_path)
    tokens = [CrispToken("テスト", 31.0, 31.4), CrispToken("「", 31.4, 31.4), CrispToken("です", 31.5, 31.8),
              CrispToken("」", 31.8, 31.8)]
    words, _ = pipe._word_dicts(seg(31.0, 40.0, "テスト「です」", [], tokens), CutPoint(30.0, 40.0))
    assert_words(words, [("テスト", 31.0, 31.4), ("「です」", 31.5, 31.8)])


def test_cohere_a_segment_whose_tokens_all_have_zero_length_is_one_word_set_apart(tmp_path):
    # W129 5 (iii): Cohere v0.1, S01E04, 204.28-223.08 s in step 2b, 24 tokens all at 204.28 s
    pipe = make(COHERE, tmp_path)
    text = "いきます!手の回し!はっ!"
    tokens = [CrispToken(ch, 204.28, 204.28) for ch in text]
    words, clamped = pipe._word_dicts(seg(204.28, 223.08, text, [CrispWord(text, 204.28, 223.08)], tokens),
                                      CutPoint(204.0, 230.0))
    assert words == [{"word": text, "start": 204.28, "end": 223.08, "alone": True}] and clamped == 0
    # its times are clamped to the slice, as every time is
    words, clamped = pipe._word_dicts(seg(204.28, 223.08, text, [], tokens), CutPoint(204.0, 220.0))
    assert (words[0]["start"], words[0]["end"], clamped) == (204.28, 220.0, 1)
    # one token with a time is enough for the usual reading
    timed_one = tokens[:-1] + [CrispToken("!", 204.28, 204.5)]
    words, _ = pipe._word_dicts(seg(204.28, 223.08, text, [], timed_one), CutPoint(204.0, 230.0))
    assert [w["word"] for w in words] == [text] and "alone" not in words[0]


@pytest.mark.parametrize("kind, model", MODEL_CASES)
def test_a_segment_with_text_but_nothing_usable_is_shared_out_over_its_clamped_times(kind, model, tmp_path):
    pipe = make(kind, tmp_path, model_name=model)
    cut = CutPoint(10.0, 14.0)
    blank_words = [CrispWord(" ", 9.0, 9.5)]
    blank_tokens = [CrispToken(" ", 9.0, 9.5), CrispToken("", 9.5, 9.5)]
    expected = split_frame_to_words("テストです。もう一度。", 10.0, 12.0)
    assert [w["word"] for w in expected] == ["テストです。", "もう一度。"]
    for words_in, tokens_in in (((), ()), (blank_words, blank_tokens)):
        words, clamped = pipe._word_dicts(seg(9.0, 12.0, " テストです。もう一度。 ", words_in, tokens_in), cut)
        assert words == expected
        assert words[0]["start"] == 10.0 and words[-1]["end"] == 12.0
        assert clamped == 1  # the segment's start, 1 s before its slice
    assert pipe._word_dicts(seg(9.0, 12.0, "  "), cut) == ([], 0)


# ---------------------------------------------------------------------------------------------------
# Line building (plan 3.2 step 9, 3.3) with the real stable-ts reconstruction
# ---------------------------------------------------------------------------------------------------

# Speech-like noise only in these stretches of the scene WAV. Word starts lie in the silence before them, so
# stable-ts's silence adjustment, if it ran, would move them (1.2 -> 1.42 s, 4.0 -> 4.12 s): the exact times
# below show it did not (W116).
BURSTS = ((1.45, 1.85), (4.15, 4.55))
FIRST = [("テスト", 101.2, 101.6), ("です", 101.6, 101.9), ("。", 101.9, 102.0)]
SECOND = [("もう", 104.0, 104.3), ("一度", 104.3, 104.6), ("。", 104.6, 104.7)]


@pytest.mark.parametrize("kind", KINDS)
def test_regrouped_lines_keep_the_word_order_and_the_engine_times(kind, tmp_path, reconstruct_calls):
    pipe = make(kind, tmp_path)
    scene = kind.module._Scene(3, write_wav(tmp_path / "scene.wav", 30.0, BURSTS), 100.0, 130.0)
    cut = CutPoint(101.0, 110.0)
    backend = pipe.model.backend
    items = [(crisp_segment(backend, SECOND, cut), cut), (crisp_segment(backend, FIRST, cut), cut)]  # out of order
    out = tmp_path / "scene.srt"
    written, clamped = pipe._write_regrouped_lines(items, scene, out)
    assert written is True and clamped == 0
    assert_lines(srt_lines(out), [("テストです。", 1.2, 2.0), ("もう一度。", 4.0, 4.7)])
    assert len(reconstruct_calls) == 1
    call = reconstruct_calls[0]
    assert call["suppress_silence"] is False and call["regroup"] == REGROUP_JAV
    assert Path(call["audio_path"]) == scene.path
    assert_words(call["words"], [(t, a - 100.0, b - 100.0) for t, a, b in FIRST + SECOND])


@pytest.mark.filterwarnings("ignore:Multiple consecutive timestamps are out of order")
def test_cohere_token_times_that_step_backwards_keep_the_text_order(tmp_path, reconstruct_calls):
    pipe = make(COHERE, tmp_path)
    scene = cohere_mod._Scene(0, write_wav(tmp_path / "scene.wav", 30.0), 50.0, 80.0)
    cut = CutPoint(51.0, 60.0)
    tokens = [CrispToken("テスト", 52.0, 52.5), CrispToken("です", 51.8, 52.2), CrispToken("。", 52.6, 52.7)]
    s = seg(51.8, 60.0, "テストです。", words=[CrispWord("テストです。", 51.8, 60.0)], tokens=tokens)
    out = tmp_path / "scene.srt"
    written, clamped = pipe._write_regrouped_lines([(s, cut)], scene, out)
    assert written is True and clamped == 0
    assert_lines(srt_lines(out), [("テストです。", 2.0, 2.7)])
    sent = reconstruct_calls[0]["words"]
    assert [w["word"] for w in sent] == ["テスト", "です", "。"]
    assert all(b["start"] >= a["start"] for a, b in zip(sent, sent[1:]))  # only the backward start is held back
    assert all(w["end"] >= w["start"] for w in sent)


@pytest.mark.parametrize("kind", KINDS)
def test_line_building_retries_once_without_regrouping(kind, tmp_path, monkeypatch, console):
    pipe = make(kind, tmp_path)
    real = reconstruction.reconstruct_from_words
    seen = []

    def regrouping_fails(words, audio_path, suppress_silence=True, regroup=REGROUP_JAV):
        seen.append((regroup, suppress_silence))
        if regroup:
            raise RuntimeError("stand-in regrouping failure")
        return real(words, audio_path, suppress_silence=suppress_silence, regroup=regroup)

    monkeypatch.setattr(reconstruction, "reconstruct_from_words", regrouping_fails)
    scene = kind.module._Scene(0, write_wav(tmp_path / "scene.wav", 30.0), 100.0, 130.0)
    cut = CutPoint(101.0, 110.0)
    items = [(crisp_segment(pipe.model.backend, part, cut), cut) for part in (FIRST, SECOND)]
    out = tmp_path / "scene.srt"
    written, _ = pipe._write_regrouped_lines(items, scene, out)
    assert written is True
    assert seen == [(REGROUP_JAV, False), (False, False)]
    assert "".join(t for t, _, _ in srt_lines(out)) == "テストです。もう一度。"
    assert any("retrying without regrouping" in line for line in console)


@pytest.mark.parametrize("kind", KINDS)
def test_a_scene_whose_segments_hold_no_text_writes_no_srt(kind, tmp_path, reconstruct_calls):
    pipe = make(kind, tmp_path)
    scene = kind.module._Scene(0, write_wav(tmp_path / "scene.wav", 1.0), 100.0, 130.0)
    out = tmp_path / "scene.srt"
    assert pipe._write_regrouped_lines([(seg(101.0, 102.0, "  "), CutPoint(101.0, 105.0))], scene, out) == (False, 0)
    assert not out.exists() and reconstruct_calls == []


def test_parakeet_segment_lines_are_one_line_per_segment_with_clamped_native_times(tmp_path):
    pipe = make(PARAKEET, tmp_path, line_mode="segments")
    scene = parakeet_mod._Scene(2, write_wav(tmp_path / "scene.wav", 1.0), 100.0, 130.0)
    cut = CutPoint(101.0, 106.0)
    first = seg(100.5, 102.0, " テストです。", words=[CrispWord("テストです。", 100.5, 102.0)])  # starts 0.5 s early
    second = seg(103.0, 107.0, "もう一度。")                                               # ends 1 s late
    blank = seg(104.0, 105.0, "  ")
    out = tmp_path / "scene.srt"
    written, clamped = pipe._write_segment_lines([(second, cut), (first, cut), (blank, cut)], scene, out)
    assert written is True and clamped == 2
    assert_lines(srt_lines(out), [("テストです。", 1.0, 2.0), ("もう一度。", 3.0, 6.0)])


def test_parakeet_segments_line_mode_is_used_by_build_lines(tmp_path, reconstruct_calls):
    pipe = make(PARAKEET, tmp_path, line_mode="segments")
    scene = parakeet_mod._Scene(0, write_wav(tmp_path / "scene.wav", 1.0), 0.0, 30.0)
    cut = CutPoint(1.0, 9.0)
    md = {"stages": {}}
    stitched = pipe._build_lines([scene], [cut], [0], [(0, seg(1.2, 3.0, "テストです。もう一度。"))], "film", md)
    assert_lines(srt_lines(stitched), [("テストです。もう一度。", 1.2, 3.0)])
    assert reconstruct_calls == [] and md["stages"]["lines"]["mode"] == "segments"


ALONE_TEXT = "いきます!手の回し!はっ!くりぃすけた!くるぅううう"


def alone_case(tmp_path: Path, **kwargs):
    """Cohere: a sentence; a segment the program gave no times inside (18.8 s), alone in its slice; and the next
    slice's sentence 0.26 s after it -- the shape of S01E04 at 204.28-223.08 s in step 2b."""
    pipe = make(COHERE, tmp_path, **kwargs)
    scene = cohere_mod._Scene(0, write_wav(tmp_path / "scene.wav", 1.0), 200.0, 230.0)
    first, middle, last = CutPoint(200.5, 202.0), CutPoint(202.05, 221.09), CutPoint(221.19, 229.5)
    before = seg(201.0, first.end, "テストです", tokens=[CrispToken("テスト", 201.0, 201.4), CrispToken("です", 201.4, 201.8)])
    alone = seg(202.28, 221.08, ALONE_TEXT, tokens=[CrispToken(ch, 202.28, 202.28) for ch in ALONE_TEXT])
    after = seg(221.34, last.end, "すごい。おいしい。", tokens=[CrispToken("すごい。", 221.34, 221.8),
                                                         CrispToken("おいしい。", 221.8, 222.5)])
    return pipe, scene, [(after, last), (alone, middle), (before, first)]


def test_cohere_a_segment_without_times_inside_is_a_line_of_its_own_kept_whole(tmp_path):
    # W135 3: "Keep the 18.8s line unified initially": one line, with its segment's own times. (Here REGROUP_JAV
    # keeps it apart by itself; the two tests below cover the shapes where it would not.)
    pipe, scene, items = alone_case(tmp_path)
    out = tmp_path / "scene.srt"
    written, clamped = pipe._write_regrouped_lines(items, scene, out)
    assert written is True and clamped == 0
    assert_lines(srt_lines(out), [("テストです", 1.0, 1.8), (ALONE_TEXT, 2.28, 21.08), ("すごい。おいしい。", 21.34, 22.5)])


def test_cohere_regroupings_clamp_does_not_shorten_a_segment_without_times_inside(tmp_path):
    # First in its scene and followed closely by a sentence, as on S01E04, REGROUP_JAV's clamp ("cm") moves its
    # start from 2.28 s to 19.33 s and the sentence joins it; the segment's own times come back
    pipe, scene, items = alone_case(tmp_path)
    items = [item for item in items if item[0].text != "テストです"]
    out = tmp_path / "scene.srt"
    pipe._write_regrouped_lines(items, scene, out)
    assert_lines(srt_lines(out), [(ALONE_TEXT, 2.28, 21.08), ("すごい。おいしい。", 21.34, 22.5)])


def test_cohere_a_short_segment_without_times_inside_is_not_merged_with_its_neighbours(tmp_path):
    # REGROUP_JAV alone would join all three (gaps under 1.5 s, under 80 characters, under 8 s)
    pipe = make(COHERE, tmp_path)
    scene = cohere_mod._Scene(0, write_wav(tmp_path / "scene.wav", 1.0), 200.0, 230.0)
    cut = CutPoint(200.5, 229.5)
    before = seg(201.0, cut.end, "テストです", tokens=[CrispToken("テスト", 201.0, 201.4), CrispToken("です", 201.4, 201.8)])
    alone = seg(202.0, 202.6, "はっ", tokens=[CrispToken("は", 202.0, 202.0), CrispToken("っ", 202.0, 202.0)])
    after = seg(202.8, cut.end, "もう一度", tokens=[CrispToken("もう", 202.8, 203.1), CrispToken("一度", 203.1, 203.5)])
    out = tmp_path / "scene.srt"
    pipe._write_regrouped_lines([(before, cut), (alone, cut), (after, cut)], scene, out)
    assert_lines(srt_lines(out), [("テストです", 1.0, 1.8), ("はっ", 2.0, 2.6), ("もう一度", 2.8, 3.5)])


def test_cohere_the_line_length_rule_splits_that_line_at_its_marks_with_times_shared_by_characters(tmp_path):
    # W135 3: "allow the length rule to split it at `!` punctuation marks using estimated proportional timing"
    pipe, scene, items = alone_case(tmp_path, line_limits=LineLimits(max_s=5.0))
    out = tmp_path / "scene.srt"
    pipe._write_regrouped_lines(items, scene, out)
    lines = srt_lines(out)
    assert [t for t, _, _ in lines] == ["テストです", "いきます!", "手の回し!", "はっ!", "くりぃすけた!", "くるぅううう",
                                        "すごい。おいしい。"]
    # 18.8 s over 26 characters: each part's share of the time is its share of the characters
    per_char = (21.08 - 2.28) / len("".join(ALONE_TEXT.split()))
    starts = [2.28]
    for part in ["いきます!", "手の回し!", "はっ!", "くりぃすけた!"]:
        starts.append(starts[-1] + per_char * len(part))
    assert [a for _, a, _ in lines[1:6]] == pytest.approx(starts, abs=0.0015)
    assert lines[5][2] == pytest.approx(21.08, abs=0.0015)


@pytest.mark.parametrize("kind", KINDS)
def test_the_line_length_rule_splits_a_regrouped_line_only_after_a_mark(kind, tmp_path):
    pipe = make(kind, tmp_path, line_limits=LineLimits(max_chars=6))
    scene = kind.module._Scene(0, write_wav(tmp_path / "scene.wav", 1.0), 100.0, 130.0)
    cut = CutPoint(101.0, 110.0)
    parts = [("今日", 101.2, 101.5), ("は", 101.5, 101.6), ("晴れ", 101.6, 102.0), ("、", 102.0, 102.1),
             ("明日", 102.2, 102.6), ("も", 102.6, 102.7), ("晴れ", 102.7, 103.1), ("。", 103.1, 103.2),
             ("それ", 103.3, 103.6), ("から", 103.6, 104.0)]  # no mark inside the last part
    items = [(crisp_segment(pipe.model.backend, parts, cut), cut)]
    out = tmp_path / "scene.srt"
    pipe._write_regrouped_lines(items, scene, out)
    assert_lines(srt_lines(out), [("今日は晴れ、", 1.2, 2.1), ("明日も晴れ。", 2.2, 3.2), ("それから", 3.3, 4.0)])
    # switched off, the same words make one line
    pipe.line_limits = OFF
    pipe._write_regrouped_lines(items, scene, out)
    assert [t for t, _, _ in srt_lines(out)] == ["今日は晴れ、明日も晴れ。それから"]


def test_the_line_length_rule_splits_a_parakeet_segment_line_at_the_mark_nearest_its_middle(tmp_path):
    pipe = make(PARAKEET, tmp_path, line_limits=LineLimits(max_s=3.0))
    scene = parakeet_mod._Scene(0, write_wav(tmp_path / "scene.wav", 1.0), 100.0, 130.0)
    cut = CutPoint(100.5, 112.0)
    words = [CrispWord("あの", 101.0, 101.4), CrispWord("、", 101.4, 101.5), CrispWord("今日", 101.6, 102.0),
             CrispWord("は", 102.0, 102.2), CrispWord("、", 102.2, 102.3), CrispWord("晴れ", 102.4, 103.0),
             CrispWord("です", 103.0, 104.6)]
    long_one = seg(101.0, 104.6, "あの、今日は、晴れです", words=words)
    no_mark = seg(105.0, 110.0, "テストです", words=[CrispWord("テスト", 105.0, 107.0), CrispWord("です", 107.0, 110.0)])
    out = tmp_path / "scene.srt"
    written, clamped = pipe._write_segment_lines([(long_one, cut), (no_mark, cut)], scene, out)
    assert written is True and clamped == 0
    # 3.6 s over 3 s: split after the "、" nearest 102.8 s (102.3 s); each part within 3 s; the 5 s line has no mark
    assert_lines(srt_lines(out), [("あの、今日は、", 1.0, 2.3), ("晴れです", 2.4, 4.6), ("テストです", 5.0, 10.0)])


def build_lines_case(kind: Kind, tmp_path: Path, first_word_start: float = 1.2):
    """Two scenes overlapping by 0.6 s; a slice merged across their edge belongs to the earlier scene."""
    pipe = make(kind, tmp_path)
    S = kind.module._Scene
    scenes = [S(0, write_wav(tmp_path / "s0.wav", 1.0), 0.0, 20.4), S(1, write_wav(tmp_path / "s1.wav", 1.0), 19.8, 40.0)]
    cuts = [CutPoint(1.0, 5.0), CutPoint(18.0, 22.0), CutPoint(25.0, 30.0)]
    owner = [0, 0, 1]
    backend = pipe.model.backend
    parts = [
        [("テスト", first_word_start, 1.6), ("です", 1.6, 1.9), ("。", 1.9, 2.0)],
        [("もう", 20.5, 20.8), ("一度", 20.8, 21.1), ("。", 21.1, 21.2)],  # past scene 0's end (20.4)
        [("今日", 26.0, 26.3), ("は", 26.3, 26.6), ("晴れ", 26.6, 26.9), ("。", 26.9, 27.0)],
    ]
    segments = [(i, crisp_segment(backend, p, cuts[i], chunk_id=i)) for i, p in enumerate(parts)]
    return pipe, scenes, cuts, owner, segments


@pytest.mark.parametrize("kind", KINDS)
def test_build_lines_puts_each_slice_in_its_scene_and_stitches_with_the_scene_offsets(kind, tmp_path):
    pipe, scenes, cuts, owner, segments = build_lines_case(kind, tmp_path)
    md = {"stages": {}}
    stitched = pipe._build_lines(scenes, cuts, owner, segments, "film", md)
    assert stitched == pipe.temp_dir / "film_stitched.srt"
    scene_srts = pipe.temp_dir / "scene_srts"
    assert_lines(srt_lines(scene_srts / "film_scene_0000.srt"), [("テストです。", 1.2, 2.0), ("もう一度。", 20.5, 21.2)])
    assert_lines(srt_lines(scene_srts / "film_scene_0001.srt"), [("今日は晴れ。", 26.0 - 19.8, 27.0 - 19.8)])
    assert_lines(srt_lines(stitched), [("テストです。", 1.2, 2.0), ("もう一度。", 20.5, 21.2), ("今日は晴れ。", 26.0, 27.0)])
    # each pipeline's default: Parakeet's own segments, Cohere's regrouped tokens
    expected_mode = {"parakeet-crispasr": "segments", "cohere-crispasr": "regroup"}[kind.cls.PIPELINE_NAME]
    assert md["stages"]["lines"] == {"mode": expected_mode, "scenes_with_lines": 2, "clamped_times": 0}
    assert pipe.notes == []


@pytest.mark.parametrize("kind", KINDS)
def test_word_times_outside_their_slice_become_a_note(kind, tmp_path):
    pipe, scenes, cuts, owner, segments = build_lines_case(kind, tmp_path, first_word_start=0.5)  # slice starts 1.0
    md = {"stages": {}}
    stitched = pipe._build_lines(scenes, cuts, owner, segments, "film", md)
    assert srt_lines(stitched)[0][1] == pytest.approx(1.0, abs=0.0015)
    assert md["stages"]["lines"]["clamped_times"] == 1
    assert len(pipe.notes) == 1 and "1 word time" in pipe.notes[0]
    assert pipe.degradations == []


@pytest.mark.parametrize("kind", KINDS)
def test_build_lines_without_segments_writes_an_empty_stitched_srt(kind, tmp_path):
    pipe, scenes, cuts, owner, _ = build_lines_case(kind, tmp_path)
    md = {"stages": {}}
    stitched = pipe._build_lines(scenes, cuts, owner, [], "film", md)
    assert stitched.read_text(encoding="utf-8") == ""
    assert md["stages"]["lines"]["scenes_with_lines"] == 0


# ---------------------------------------------------------------------------------------------------
# Stage methods with stand-ins: speech groups, scenes, audio, progress lines
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_speech_groups_are_put_in_film_time_with_their_pauses_and_the_segmenter_is_released(kind, tmp_path,
                                                                                           monkeypatch):
    from whisperjav.modules.speech_segmentation import SegmentationResult, SpeechSegment, SpeechSegmenterFactory

    def result(groups):
        return SegmentationResult(segments=[s for g in groups for s in g], groups=groups, method="stand-in",
                                  audio_duration_sec=30.0, parameters={})

    answers = {
        "s0.wav": result([[SpeechSegment(1.0, 2.0), SpeechSegment(2.5, 3.0)], [],
                          [SpeechSegment(5.0, 6.0), SpeechSegment(6.0, 7.0)]]),
        "s1.wav": RuntimeError("stand-in segmenter failure"),
        "s2.wav": result([[SpeechSegment(0.25, 1.25)]]),
        "s3.wav": SegmentationResult(segments=[], groups=None, method="stand-in", audio_duration_sec=30.0,
                                     parameters={}),
    }

    class Segmenter:
        def __init__(self):
            self.calls, self.released = [], 0

        def segment(self, audio, sample_rate=16000, **kwargs):
            self.calls.append((Path(audio).name, sample_rate))
            answer = answers[Path(audio).name]
            if isinstance(answer, Exception):
                raise answer
            return answer

        def cleanup(self):
            self.released += 1

    made = []

    def create(name, config=None, **kwargs):
        made.append((name, config, kwargs))
        return segmenter

    segmenter = Segmenter()
    monkeypatch.setattr(SpeechSegmenterFactory, "create", staticmethod(create))
    pipe = make(kind, tmp_path)
    S, m = kind.module._Scene, kind.max_s
    scenes = [S(i, tmp_path / f"s{i}.wav", start, start + 30.0) for i, start in enumerate((100.0, 129.5, 159.5, 189.5))]
    groups, failed = pipe._speech_groups(scenes)

    assert len(made) == 1 and made[0][0] == "whisperseg" and made[0][1] is None
    kwargs = made[0][2]
    assert kwargs.pop("max_speech_duration_s") == pytest.approx(m - 0.1)  # the limit less the 100 ms start pad
    # balanced's detection threshold, and nothing else of WhisperSeg's preset (W134 2)
    assert kwargs == {"max_group_duration_s": m, "chunk_threshold_s": 1.0, "start_pad_ms": 100, "end_pad_ms": 0,
                      "threshold": 0.35}
    assert segmenter.calls == [(f"s{i}.wav", 16000) for i in range(4)]
    assert failed == 1 and segmenter.released == 1
    assert [g.scene for g in groups] == [0, 0, 2]
    assert flat([(g.start, g.end) for g in groups]) == pytest.approx([101.0, 103.0, 105.0, 107.0, 159.75, 160.75])
    assert [g.pauses for g in groups] == [pytest.approx([102.25]), [], []]  # touching segments leave no pause


@pytest.mark.parametrize("kind", KINDS)
def test_a_segmenter_that_fails_to_release_does_not_fail_the_file(kind, tmp_path, monkeypatch):
    from whisperjav.modules.speech_segmentation import SegmentationResult, SpeechSegment, SpeechSegmenterFactory

    class Segmenter:
        def segment(self, audio, sample_rate=16000, **kwargs):
            return SegmentationResult(segments=[], groups=[[SpeechSegment(1.0, 2.0)]], method="stand-in",
                                      audio_duration_sec=30.0, parameters={})

        def cleanup(self):
            raise RuntimeError("stand-in release failure")

    monkeypatch.setattr(SpeechSegmenterFactory, "create", staticmethod(lambda name, config=None, **kw: Segmenter()))
    pipe = make(kind, tmp_path)
    groups, failed = pipe._speech_groups([kind.module._Scene(0, tmp_path / "s0.wav", 10.0, 40.0)])
    assert failed == 0 and flat([(g.start, g.end) for g in groups]) == pytest.approx([11.0, 12.0])


@pytest.mark.parametrize("kind", KINDS)
def test_a_segmenter_that_caught_its_own_failure_counts_as_a_failed_scene_not_as_silence(kind, tmp_path,
                                                                                        monkeypatch):
    """WhisperSeg, TEN and FireRedVAD catch their own inference failure and return an empty result with
    SegmentationResult.error set: that scene failed, exactly as when the segmenter raises."""
    from whisperjav.modules.speech_segmentation import SegmentationResult, SpeechSegment, SpeechSegmenterFactory

    def result(groups, error=None):
        return SegmentationResult(segments=[s for g in groups for s in g], groups=groups, method="stand-in",
                                  audio_duration_sec=30.0, parameters={}, error=error)

    answers = {
        "s0.wav": result([], error="stand-in inference failure"),  # caught by the backend itself
        "s1.wav": RuntimeError("stand-in segmenter failure"),      # raised
        "s2.wav": result([]),                                      # silence: no failure
        "s3.wav": result([[SpeechSegment(1.0, 2.0)]]),
    }
    assert SegmentationResult(segments=[], groups=[], method="x", audio_duration_sec=1.0,
                              parameters={}).error is None  # the default: no failure

    class Segmenter:
        def segment(self, audio, sample_rate=16000, **kwargs):
            answer = answers[Path(audio).name]
            if isinstance(answer, Exception):
                raise answer
            return answer

        def cleanup(self):
            pass

    monkeypatch.setattr(SpeechSegmenterFactory, "create", staticmethod(lambda name, config=None, **kw: Segmenter()))
    pipe = make(kind, tmp_path)
    scenes = [kind.module._Scene(i, tmp_path / f"s{i}.wav", 30.0 * i, 30.0 * i + 30.0) for i in range(4)]
    groups, failed = pipe._speech_groups(scenes)
    assert failed == 2
    assert [g.scene for g in groups] == [3]
    assert flat([(g.start, g.end) for g in groups]) == pytest.approx([91.0, 92.0])


def _segmenter_backend(name: str, monkeypatch, tmp_path, error: Optional[Exception]):
    """WhisperSeg, TEN or FireRedVAD with its model replaced: nothing is loaded or downloaded.

    The stand-in model raises ``error`` during inference, or finds no speech when it is None.
    """
    if name == "whisperseg":
        from whisperjav.modules.speech_segmentation.backends.whisperseg import WhisperSegSpeechSegmenter

        backend = WhisperSegSpeechSegmenter()
        backend._session = object()  # the model counts as loaded

        def forward(audio):
            if error is not None:
                raise error
            return np.zeros(50, dtype=np.float32)

        monkeypatch.setattr(backend, "_audio_forward", forward)
    elif name == "ten":
        from whisperjav.modules.speech_segmentation.backends.ten import TenSpeechSegmenter

        class TenModel:
            out_flags = SimpleNamespace(value=0)
            out_probability = SimpleNamespace(value=0.0)

            def process(self, frame):
                if error is not None:
                    raise error

        backend = TenSpeechSegmenter()
        backend._model = TenModel()
    else:
        from whisperjav.modules.speech_segmentation.backends import firered_vad

        class FireRedModel:
            def detect(self, wav_path):
                if error is not None:
                    raise error
                return {"timestamps": []}, None

        # Its constructor resolves (and would fetch) the model folder: answered here instead.
        monkeypatch.setattr(firered_vad, "ensure_model_downloaded", lambda model_dir=None, **kwargs: str(tmp_path))
        backend = firered_vad.FireRedVadSpeechSegmenter()
        backend._model = FireRedModel()
    return backend


@pytest.mark.parametrize("name", ["whisperseg", "ten", "firered-vad"])
@pytest.mark.parametrize("error, words", [(RuntimeError("stand-in inference failure"), "stand-in inference failure"),
                                          (RuntimeError(), "RuntimeError")], ids=["with-message", "no-message"])
def test_a_backend_that_catches_its_own_inference_failure_says_so_in_its_result(name, error, words, monkeypatch,
                                                                                tmp_path):
    result = _segmenter_backend(name, monkeypatch, tmp_path, error).segment(np.zeros(16000, dtype=np.float32),
                                                                             sample_rate=16000)
    assert (result.segments, result.groups) == ([], [])
    assert result.error == words  # never empty: an empty error would read as silence


@pytest.mark.parametrize("name", ["whisperseg", "ten", "firered-vad"])
def test_a_backend_that_finds_no_speech_is_silence_not_a_failure(name, monkeypatch, tmp_path):
    result = _segmenter_backend(name, monkeypatch, tmp_path, None).segment(np.zeros(16000, dtype=np.float32),
                                                                            sample_rate=16000)
    assert result.segments == [] and result.error is None


@pytest.mark.parametrize("kind", KINDS)
def test_scenes_come_from_the_semantic_detector_with_chronosjav_bounds(kind, tmp_path, monkeypatch):
    from whisperjav.modules.scene_detection_backends import SceneDetectorFactory

    seen = {"released": 0}

    class Result:
        def to_legacy_tuples(self):
            return [(str(tmp_path / "a.wav"), 0.0, 30.35, 30.35), (str(tmp_path / "b.wav"), 29.65, 61.0, 31.35)]

    class Detector:
        def detect_scenes(self, audio_path, output_dir, media_basename, **kwargs):
            seen["args"] = (audio_path, output_dir, media_basename)
            if seen.get("fail"):
                raise RuntimeError("stand-in detector failure")
            return Result()

        def cleanup(self):
            seen["released"] += 1

    def create(**kwargs):
        seen["kwargs"] = kwargs
        return Detector()

    monkeypatch.setattr(SceneDetectorFactory, "safe_create_from_legacy_kwargs", staticmethod(create))
    pipe = make(kind, tmp_path)
    S = kind.module._Scene
    scenes = pipe._detect_scenes(str(tmp_path / "film.wav"), "film")
    assert seen["kwargs"] == {"method": "semantic", "min_duration": 12, "max_duration": 48}
    assert seen["args"] == (tmp_path / "film.wav", pipe.temp_dir / "scenes", "film")
    assert isinstance(seen["args"][0], Path) and (pipe.temp_dir / "scenes").is_dir()
    assert scenes == [S(0, tmp_path / "a.wav", 0.0, 30.35), S(1, tmp_path / "b.wav", 29.65, 61.0)]
    assert seen["released"] == 1
    seen["fail"] = True
    with pytest.raises(RuntimeError, match="stand-in detector failure"):
        pipe._detect_scenes(tmp_path / "film.wav", "film")
    assert seen["released"] == 2  # released even when detection fails


@pytest.mark.parametrize("kind", KINDS)
def test_audio_is_extracted_at_16_khz_by_one_extractor_and_its_integrity_report_applied(kind, tmp_path, monkeypatch):
    from whisperjav.modules import audio_extraction, audio_integrity

    made, applied = [], []

    class Extractor:
        def __init__(self, sample_rate=16000, **kwargs):
            self.sample_rate = sample_rate
            self.last_integrity = object()
            made.append(self)

        def extract(self, input_file, output_path):
            Path(output_path).write_bytes(b"RIFF stand-in")
            return str(output_path), 12

    monkeypatch.setattr(audio_extraction, "AudioExtractor", Extractor)
    monkeypatch.setattr(audio_integrity, "apply_to_run",
                        lambda report, degradations, in_summary=True: applied.append((report, degradations, in_summary)))
    pipe = make(kind, tmp_path)
    wav, duration = pipe._extract_audio(tmp_path / "film.mp4", "film")
    assert wav == pipe.temp_dir / "film_extracted.wav" and duration == 12.0 and isinstance(duration, float)
    pipe._extract_audio(tmp_path / "other.mp4", "other")
    assert len(made) == 1 and made[0].sample_rate == 16000
    assert len(applied) == 2
    report, degradations, in_summary = applied[0]
    assert report is made[0].last_integrity and degradations is pipe.degradations and in_summary is True


@pytest.mark.parametrize("kind", KINDS)
def test_progress_lines_come_at_every_tenth_of_the_slices(kind, tmp_path, console):
    report = make(kind, tmp_path)._progress_logger()
    for done in range(1, 41):
        report(done, 40)
    lines = [line for line in console if "slices transcribed" in line]
    assert len(lines) == 10 and lines[-1].endswith("40 of 40 slices transcribed (100%)")


# ---------------------------------------------------------------------------------------------------
# process() end to end, every heavy collaborator replaced
# ---------------------------------------------------------------------------------------------------

# The CPU build's console line (plan 2.6): a PC without an NVIDIA GPU is the normal set-up for it, so the
# line is not repeated in each file's summary; a GPU build that failed its start check was not expected, so
# its line is (install.ProgramChoice.note_in_summary).
NOTE_CPU = "CrispASR runs on the processor (CPU build), which is slower: no NVIDIA GPU was found."
NOTE_GPU_FAILED = ("CrispASR runs on the processor (CPU build), which is slower: the GPU build's start check "
                   "failed: it found no CUDA device.")
WIN_CPU = next(b for b in manifest.PROGRAM_BUILDS if b.platform == "win32" and b.gpu is None)
APPLE_CPU = next(b for b in manifest.PROGRAM_BUILDS if b.extra_args)
EXIT_CRASH = 0xC0000409

SENTENCES = (("テスト", "です", "。"), ("もう", "一度", "。"), ("今日", "は", "晴れ", "。"), ("明日", "も", "雨", "。"),
             ("本当", "です", "。"), ("大丈夫", "です", "。"))
LEAD_S, PIECE_S, STOP_S = 0.2, 0.3, 0.1  # each sentence starts LEAD_S into its slice; "。" lasts STOP_S

# The film: 60 s, two semantic scenes sharing 0.7 s, four speech groups; two of them overlap at the scene edge.
DURATION = 60.0
SCENES = ((0.0, 30.35), (29.65, 60.0))
GROUPS = ((0, 2.0, 6.0, ()), (0, 27.0, 30.3, ()), (1, 29.7, 33.0, ()), (1, 40.0, 55.0, (47.0,)))
EXPECTED_CUTS = {"parakeet-crispasr": [(2.0, 6.0), (27.0, 33.0), (40.0, 47.0), (47.0, 55.0)],
                 "cohere-crispasr": [(2.0, 6.0), (27.0, 33.0), (40.0, 55.0)]}


def timed(pieces, t0: float):
    out, t = [], t0
    for p in pieces:
        d = STOP_S if p == "。" else PIECE_S
        out.append((p, t, t + d))
        t += d
    return out


def expected_lines(cuts):
    lines = []
    for i, (a, _) in enumerate(cuts):
        parts = timed(SENTENCES[i % len(SENTENCES)], a + LEAD_S)
        lines.append(("".join(p for p, _, _ in parts), parts[0][1], parts[-1][2]))
    return lines


def _ms(t: float) -> int:
    return int(round(t * 1000))


def _item(text: str, a: float, b: float, p: Optional[float] = None) -> dict:
    item = {"text": text, "t0": int(round(a * 100)), "t1": int(round(b * 100)), "offsets": {"from": _ms(a), "to": _ms(b)}}
    if p is not None:
        item["p"] = p
    return item


def json_segment(backend: str, index: int, sl, pieces, n_slices: int, lead: float = LEAD_S) -> dict:
    """One -ojf segment in the shape the program writes for this back end (second study S2-S4, P1)."""
    parts = timed(pieces, sl[0] + lead)
    text = "".join(pieces)
    if backend == "parakeet":
        start, end = parts[0][1], parts[-1][2]
        words = [_item(p, a, b) for p, a, b in parts]
        tokens = [_item(" ", max(0.0, start - 10.9), start, 0.99)] + [_item(p, a, b, 0.98) for p, a, b in parts]
    elif backend == "fastconformer-ctc":  # tokens only, 80 ms each (run P1)
        start, end = sl
        words = []
        tokens = [_item(ch, parts[0][1] + 0.12 * k, parts[0][1] + 0.12 * k + 0.08, 0.9) for k, ch in enumerate(text)]
    elif backend == "cohere":  # one word for the whole segment; ends at the slice end; zero-length tokens
        (first, a, b), rest = parts[0], parts[1:]
        start, end = a, sl[1]
        words = [_item(text, start, end)]
        tokens = ([_item(first[:1], a, a, 0.5), _item(first[1:], a, b, 0.99)]
                  + [_item(p, x, y, 0.99) for p, x, y in rest])
    else:
        raise ValueError(backend)
    segment = {"offsets": {"from": _ms(start), "to": _ms(end)}, "text": text, "chunk_id": index,
               "words": words, "tokens": tokens}
    if n_slices == 1:
        del segment["chunk_id"]  # the program omits it when the run had one slice
    return segment


def one_sentence_per_slice(slices, request) -> List[dict]:
    return [json_segment(request.backend, i, sl, SENTENCES[i % len(SENTENCES)], len(slices))
            for i, sl in enumerate(slices)]


@dataclass
class Program:
    """Stands in for run_crispasr_json: records the request and the run folder, reads cuts.json, writes the JSON."""
    respond: Callable = one_sentence_per_slice
    outcome: CrispOutcome = CrispOutcome.OK
    exit_code: int = 0
    write_json: bool = True
    json_backend: Optional[str] = None  # None: echo the request, as the program does
    json_model: Optional[str] = None
    json_language: Optional[str] = None
    json_text: Optional[str] = None     # written instead of a JSON document
    warning_count: int = 0
    download_lines: Tuple[str, ...] = ()
    detail: Optional[str] = None        # the program's own plainer reason (CrispRunResult.detail)
    calls: list = field(default_factory=list)

    def __call__(self, request, on_progress=None, cancel_event=None, poll_s=0.5):
        run_dir = Path(request.run_dir)
        cuts = json.loads((run_dir / request.cuts_name).read_text(encoding="utf-8"))["crispasr_vad"]
        rate = cuts["sample_rate"]
        slices = [(s["start"] / rate, s["end"] / rate) for s in cuts["slices"]]
        # -m as the program opens it: a relative name from its working folder, the run folder (plan 1.7).
        model = Path(request.model)
        staged = None if model.is_absolute() else run_dir / model
        self.calls.append(SimpleNamespace(
            request=request, run_dir=run_dir, files=sorted(p.name for p in run_dir.iterdir()), cuts=cuts,
            slices=slices, argv=request.argv(),
            staged_model=staged.read_bytes() if staged is not None and staged.is_file() else None))
        if on_progress is not None:
            for done in range(1, len(slices) + 1):
                on_progress(done, len(slices))
        if self.write_json:
            if self.json_text is not None:
                text = self.json_text
            else:
                doc = {"crispasr": {"backend": self.json_backend or request.backend,
                                    "model": self.json_model or request.model,
                                    "language": self.json_language or request.language},
                       "transcription": self.respond(slices, request)}
                text = json.dumps(doc, ensure_ascii=False)
            request.json_path.write_text(text, encoding="utf-8")
        request.log_path.write_text("command: stand-in\n", encoding="utf-8")
        usable = self.write_json and self.outcome in (CrispOutcome.OK, CrispOutcome.CRASHED)
        return CrispRunResult(outcome=self.outcome, exit_code=self.exit_code,
                              json_path=request.json_path if usable else None,
                              warnings=["warning: stand-in"] * min(self.warning_count, 500),
                              warning_count=self.warning_count, download_lines=list(self.download_lines),
                              stderr_tail="crispasr: stand-in stderr", seconds=1.5, log_path=request.log_path,
                              detail=self.detail)


@dataclass
class Stage:
    pipe: object
    program: Program
    choice: ProgramChoice
    model_path: Path
    calls: dict
    media: dict

    @property
    def final(self) -> Path:
        return self.pipe.output_dir / f"{self.media['basename']}.ja.whisperjav.srt"

    @property
    def run_dir(self) -> Path:
        return film_run_dir(self.pipe.temp_dir, self.media["basename"])


def stage(monkeypatch, tmp_path: Path, kind: Kind, *, model: Optional[str] = None, build=WIN_CPU,
          note: Optional[str] = NOTE_CPU, note_in_summary: bool = False, program: Optional[Program] = None,
          scenes=SCENES, groups=GROUPS, unsegmented: int = 0, real_speech_groups: bool = False,
          basename: str = "film", model_dir: str = "models", **pipe_kwargs) -> Stage:
    """A pipeline whose program, model, audio, scenes, segmenter and program run are stand-ins.

    ``real_speech_groups`` keeps the pipeline's own _speech_groups; the test then replaces the segmenter.
    """
    pipe = make(kind, tmp_path, model_name=model or kind.model, **pipe_kwargs)
    exe = tmp_path / "program" / build.exe_name
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"stand-in program")
    choice = ProgramChoice(build=build, exe=exe, check=StartCheck(True, None), note=note,
                           note_in_summary=note_in_summary)
    model_path = tmp_path / model_dir / pipe.model.filename
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"GGUF stand-in")
    calls = {"prepare_program": 0, "ensure_model": 0, "extract": 0, "scenes": 0, "groups": 0}

    def prepare_program(offline=None):
        calls["prepare_program"] += 1
        return choice

    def ensure_model(model_file, offline=None):
        calls["ensure_model"] += 1
        assert model_file is pipe.model
        return model_path

    def extract(input_file, name):
        calls["extract"] += 1
        return write_wav(tmp_path / "media" / f"{name}_audio.wav", 0.1), DURATION

    def detect(wav, name):
        calls["scenes"] += 1
        return [kind.module._Scene(i, write_wav(pipe.temp_dir / "scenes" / f"{name}_scene_{i:04d}.wav", 0.1), a, b)
                for i, (a, b) in enumerate(scenes)]

    def speech_groups(scene_list):
        calls["groups"] += 1
        return [kind.module._Group(sc, a, b, list(p)) for sc, a, b, p in groups], unsegmented

    monkeypatch.setattr(install, "prepare_program", prepare_program)
    monkeypatch.setattr(install, "ensure_model", ensure_model)
    monkeypatch.setattr(pipe, "_extract_audio", extract)
    monkeypatch.setattr(pipe, "_detect_scenes", detect)
    if not real_speech_groups:
        monkeypatch.setattr(pipe, "_speech_groups", speech_groups)
    program = program or Program()
    monkeypatch.setattr(kind.module, "run_crispasr_json", program)
    media = {"path": str(tmp_path / "media" / f"{basename}.mp4"), "basename": basename}
    return Stage(pipe, program, choice, model_path, calls, media)


def all_text(path: Path) -> str:
    return "".join(t for t, _, _ in srt_lines(path))


@pytest.mark.parametrize("kind, model", MODEL_CASES)
def test_process_runs_the_program_once_and_writes_the_final_srt(kind, model, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind, model=model)
    md = st.pipe.process(st.media)
    expected_cuts = EXPECTED_CUTS[kind.cls.PIPELINE_NAME]

    # one program run, with the command line of plan 4.1
    assert len(st.program.calls) == 1
    call = st.program.calls[0]
    req = call.request
    assert req.run_dir == st.run_dir and st.run_dir.parent.name == RUN_FOLDER
    assert req.exe == st.choice.exe and req.backend == st.pipe.model.backend
    # an all-ASCII model path is passed as it is; any other is staged beside the run folder (plan 1.7)
    assert req.model == (str(st.model_path) if str(st.model_path).isascii()
                         else os.path.join("..", st.pipe.model.filename))
    assert tuple(req.extra_args) == tuple(WIN_CPU.extra_args) + tuple(st.pipe.model.extra_args)
    assert ("--punc-model" in call.argv) == (model == "parakeet-ctc-1.1b-ja")
    for switch, value in (("-m", req.model), ("--backend", req.backend), ("-f", "audio.wav"), ("-l", "ja"),
                          ("-of", "out"), ("--vad-import", "cuts.json")):
        assert call.argv[call.argv.index(switch) + 1] == value
    assert "-ojf" in call.argv and "-pp" in call.argv
    assert {"audio.wav", "cuts.json"} <= set(call.files)
    # the cut points are this pipeline's slice rule, written as a "chunks" file
    assert call.cuts["kind"] == "chunks" and call.cuts["num_slices"] == len(expected_cuts)
    assert flat(call.slices) == pytest.approx(flat(expected_cuts))

    # the final SRT
    assert md["output_files"]["final_srt"] == str(st.final) and st.final.is_file()
    lines = srt_lines(st.final)
    if model == "parakeet-ctc-1.1b-ja":
        # A CTC segment spans its whole slice, and Parakeet's default lines are its segments: the text and
        # its order are the expected ones, and the first line starts at the first slice
        assert all_text(st.final) == "".join(t for t, _, _ in expected_lines(expected_cuts))
        assert lines[0][1] == pytest.approx(expected_cuts[0][0], abs=0.0015)
    else:
        assert_lines(lines, expected_lines(expected_cuts))

    summary = md["summary"]
    assert summary["final_subtitles_refined"] == summary["final_subtitles_raw"] == len(lines)
    assert summary["degradations"] == []
    assert summary["notes"] == []  # the expected CPU build (no NVIDIA GPU) is a console line only (plan 2.6)
    assert isinstance(summary["total_processing_time_seconds"], float)
    assert summary["total_processing_time_seconds"] >= 0
    assert set(summary["quality_metrics"]) == {"hallucinations_removed", "repetitions_removed", "duration_adjustments",
                                               "empty_removed", "cps_filtered", "logprob_filtered", "nonverbal_filtered"}
    assert md["pipeline"] == kind.cls.PIPELINE_NAME and md["srt_path"] == str(st.final)
    config = md["config"]["crispasr"]
    assert (config["model"], config["backend"], config["build"]) == (model, st.pipe.model.backend, WIN_CPU.key)
    assert (config["program"], config["model_path"]) == (str(st.choice.exe), str(st.model_path))
    stages = md["stages"]
    assert stages["slices"]["count"] == len(expected_cuts)
    assert stages["crispasr"]["outcome"] == "ok" and stages["crispasr"]["segments"] == len(expected_cuts)
    assert (stages["crispasr"]["retied"], stages["crispasr"]["dropped"]) == (0, 0)

    # made ready once; the run folder and the temporary SRTs are removed after success
    assert (st.calls["prepare_program"], st.calls["ensure_model"], st.calls["extract"]) == (1, 1, 1)
    assert not st.run_dir.exists()
    assert not (st.pipe.temp_dir / "film_stitched.srt").exists()
    assert not list((st.pipe.temp_dir / "scene_srts").glob("film_scene_*.srt"))


@pytest.mark.parametrize("kind", KINDS)
def test_keep_temp_keeps_the_run_folder_with_its_inputs_output_and_log(kind, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind, keep_temp_files=True)
    st.pipe.process(st.media)
    assert {"audio.wav", "cuts.json", "out.json", "run.log"} <= {p.name for p in st.run_dir.iterdir()}
    cuts = json.loads((st.run_dir / "cuts.json").read_text(encoding="utf-8"))["crispasr_vad"]
    assert cuts["num_slices"] == len(EXPECTED_CUTS[kind.cls.PIPELINE_NAME])
    # audio.wav is the extracted film audio, linked or copied into the run folder (plan 1.7)
    assert (st.run_dir / "audio.wav").read_bytes() == (tmp_path / "media" / "film_audio.wav").read_bytes()
    assert (st.pipe.temp_dir / "film_stitched.srt").is_file()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("keep", [False, True], ids=["no-keep-temp", "keep-temp"])
def test_program_warnings_and_downloads_become_notes_that_say_where_the_log_is(kind, keep, tmp_path, monkeypatch):
    program = Program(warning_count=3, download_lines=("downloading stand-in punctuation model",))
    st = stage(monkeypatch, tmp_path, kind, program=program, keep_temp_files=keep, note=NOTE_GPU_FAILED,
               note_in_summary=True)
    md = st.pipe.process(st.media)
    notes = md["summary"]["notes"]
    assert notes[0] == NOTE_GPU_FAILED  # the build's note comes first, the run's own notes after it
    warned = [n for n in notes if "3 warning" in n]
    downloaded = [n for n in notes if "downloading stand-in punctuation model" in n]
    assert len(warned) == 1 and len(downloaded) == 1 and len(notes) == 3
    for note in warned + downloaded:
        assert (str(st.run_dir / "run.log") in note) if keep else ("--keep-temp" in note)
    assert md["summary"]["degradations"] == []  # notes change no state (Q21)
    assert md["summary"]["final_subtitles_refined"] == len(EXPECTED_CUTS[kind.cls.PIPELINE_NAME])


@pytest.mark.parametrize("kind", KINDS)
def test_a_segment_in_no_slice_is_a_degradation_and_a_misnamed_one_a_note(kind, tmp_path, monkeypatch):
    def respond(slices, request):
        segments = one_sentence_per_slice(slices, request)
        # names slice 0 but starts inside the last slice: tied by its time instead (plan 4.2)
        retied = json_segment(request.backend, 0, slices[-1], SENTENCES[4], len(slices), lead=1.5)
        # starts in the gap between the first two slices (2-6 s, 27-33 s): no slice holds it
        lost = json_segment(request.backend, 0, (15.0, 20.0), SENTENCES[5], len(slices))
        return segments + [retied, lost]

    st = stage(monkeypatch, tmp_path, kind, program=Program(respond=respond))
    md = st.pipe.process(st.media)
    degradations = md["summary"]["degradations"]
    assert len(degradations) == 1 and "1 segment" in degradations[0]
    retied_notes = [n for n in md["summary"]["notes"] if "wrong slice" in n]
    assert len(retied_notes) == 1 and "1 segment" in retied_notes[0]
    assert (md["stages"]["crispasr"]["retied"], md["stages"]["crispasr"]["dropped"]) == (1, 1)
    text = all_text(st.final)
    assert "本当です。" in text and "大丈夫です。" not in text


@pytest.mark.parametrize("kind", KINDS)
def test_no_speech_from_the_program_gives_an_empty_srt_and_no_error(kind, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind, program=Program(outcome=CrispOutcome.NO_SPEECH, write_json=False))
    md = st.pipe.process(st.media)
    assert st.final.is_file() and st.final.read_text(encoding="utf-8") == ""
    assert md["summary"]["final_subtitles_refined"] == 0 and md["summary"]["degradations"] == []
    assert md["stages"]["crispasr"]["outcome"] == "no_speech"
    assert not st.run_dir.exists()


LACKS_AVX2 = "this processor lacks instructions the CrispASR program needs (AVX2)"
FAILURES = [
    pytest.param(CrispOutcome.FAILED, 30, True, None, "required VAD model failed", id="failed-exit-30-partial-json"),
    pytest.param(CrispOutcome.FAILED, 1, False, LACKS_AVX2, f"the CrispASR program failed: {LACKS_AVX2}",
                 id="failed-the-programs-own-processor-check"),
    pytest.param(CrispOutcome.NO_OUTPUT, 0, False, None, "wrote no output", id="no-output"),
    pytest.param(CrispOutcome.CRASHED, EXIT_CRASH, False, None, "0xC0000409", id="crashed-without-json"),
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("outcome, code, json_written, detail, words", FAILURES)
def test_a_program_run_that_left_nothing_usable_fails_the_file(kind, outcome, code, json_written, detail, words,
                                                               tmp_path, monkeypatch, console):
    program = Program(outcome=outcome, exit_code=code, write_json=json_written, detail=detail)
    st = stage(monkeypatch, tmp_path, kind, program=program)
    with pytest.raises(CrispASRRunError, match=re.escape(words)):
        st.pipe.process(st.media)
    assert not st.final.exists()
    assert any("crispasr: stand-in stderr" in line for line in console)  # the program's last output (plan 4.3)
    # Plan 1.4 item 4: a failed run is never run again, on the CPU build or any other; the file fails.
    assert len(st.program.calls) == 1
    assert (st.calls["prepare_program"], st.calls["ensure_model"]) == (1, 1)


@pytest.mark.parametrize("kind", KINDS)
def test_a_crash_after_a_complete_json_uses_that_json_with_a_note(kind, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind, program=Program(outcome=CrispOutcome.CRASHED, exit_code=EXIT_CRASH))
    md = st.pipe.process(st.media)
    assert_lines(srt_lines(st.final), expected_lines(EXPECTED_CUTS[kind.cls.PIPELINE_NAME]))
    crash = [n for n in md["summary"]["notes"] if "0xC0000409" in n]
    assert len(crash) == 1 and "used" in crash[0]
    assert md["summary"]["degradations"] == [] and md["stages"]["crispasr"]["outcome"] == "crashed"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("override, words", [({"json_model": "another-model-q8_0.gguf"}, "names model"),
                                             ({"json_backend": "whisper"}, "names back end"),
                                             ({"json_language": "en"}, "names language 'en', not 'ja'")],
                         ids=["another-model", "another-back-end", "another-language"])
def test_a_json_naming_another_model_back_end_or_language_is_not_used(kind, override, words, tmp_path,
                                                                      monkeypatch):
    st = stage(monkeypatch, tmp_path, kind, program=Program(**override))
    with pytest.raises(CrispASRRunError, match=words):
        st.pipe.process(st.media)
    assert not st.final.exists()
    assert st.program.calls[0].request.language == "ja"  # what -l asked for


@pytest.mark.parametrize("kind", KINDS)
def test_an_unreadable_json_is_not_used(kind, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind, program=Program(json_text='{"crispasr": {"backend": "par'))
    with pytest.raises(CrispASRRunError, match="could not be read"):
        st.pipe.process(st.media)


@pytest.mark.parametrize("kind", KINDS)
def test_a_program_setup_failure_stops_before_the_audio_and_is_not_retried_for_the_next_file(kind, tmp_path,
                                                                                           monkeypatch):
    st = stage(monkeypatch, tmp_path, kind)
    error = CrispASRSetupError("CrispASR's CPU build cannot start on this computer: stand-in.")
    attempts = []

    def failing(offline=None):
        attempts.append(1)
        raise error

    monkeypatch.setattr(install, "prepare_program", failing)
    for _ in range(2):
        with pytest.raises(CrispASRSetupError) as info:
            st.pipe.process(st.media)
        assert info.value is error
    assert len(attempts) == 1
    assert (st.calls["ensure_model"], st.calls["extract"]) == (0, 0) and st.program.calls == []


@pytest.mark.parametrize("kind", KINDS)
def test_a_model_setup_failure_is_not_retried_for_the_next_file(kind, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind)
    error = CrispASRSetupError("Downloading the model failed: stand-in.")
    attempts = []

    def failing(model_file, offline=None):
        attempts.append(1)
        raise error

    monkeypatch.setattr(install, "ensure_model", failing)
    for _ in range(2):
        with pytest.raises(CrispASRSetupError):
            st.pipe.process(st.media)
    assert len(attempts) == 1 and st.calls["prepare_program"] == 1 and st.calls["extract"] == 0


@pytest.mark.parametrize("kind", KINDS)
def test_one_pipeline_object_serves_every_file_of_the_pass(kind, tmp_path, monkeypatch):
    def first_file(slices, request):
        return one_sentence_per_slice(slices, request) + [
            json_segment(request.backend, 0, (15.0, 20.0), SENTENCES[5], len(slices))]  # dropped: a degradation

    program = Program(respond=first_file, warning_count=2)
    st = stage(monkeypatch, tmp_path, kind, program=program, note=NOTE_GPU_FAILED, note_in_summary=True)
    md1 = st.pipe.process(st.media)
    program.respond, program.warning_count = one_sentence_per_slice, 0
    md2 = st.pipe.process({"path": str(tmp_path / "media" / "second.mp4"), "basename": "second"})

    assert (st.calls["prepare_program"], st.calls["ensure_model"]) == (1, 1)
    assert len(md1["summary"]["degradations"]) == 1 and len(md1["summary"]["notes"]) == 2
    # The first file's degradation and warnings do not carry over; the build's note is in every file's summary.
    assert md2["summary"]["degradations"] == [] and md2["summary"]["notes"] == [NOTE_GPU_FAILED]
    assert (st.pipe.output_dir / "second.ja.whisperjav.srt").is_file() and st.final.is_file()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("note, in_summary", [(NOTE_CPU, False), (NOTE_GPU_FAILED, True)],
                         ids=["expected-cpu-build", "unexpected-cpu-build"])
def test_the_cpu_build_note_is_one_console_line_per_pass_and_in_the_summary_only_when_unexpected(
        kind, note, in_summary, tmp_path, monkeypatch, console):
    st = stage(monkeypatch, tmp_path, kind, note=note, note_in_summary=in_summary)
    md1 = st.pipe.process(st.media)
    md2 = st.pipe.process({"path": str(tmp_path / "media" / "second.mp4"), "basename": "second"})
    assert console.count(note) == 1  # once for the pass, not once per film
    for md in (md1, md2):
        assert md["summary"]["notes"] == ([note] if in_summary else [])
        assert md["summary"]["degradations"] == []  # a note never makes a file suspect (Q21)
    assert st.calls["prepare_program"] == 1


@pytest.mark.parametrize("kind", KINDS)
def test_a_program_or_model_file_gone_since_it_was_prepared_is_prepared_again(kind, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind)
    st.pipe.process(st.media)
    assert (st.calls["prepare_program"], st.calls["ensure_model"]) == (1, 1)

    prepare, ensure = install.prepare_program, install.ensure_model  # the stage's counting stand-ins

    def prepare_again(offline=None):
        choice = prepare(offline)
        choice.exe.write_bytes(b"stand-in program, fetched again")
        return choice

    def ensure_again(model_file, offline=None):
        path = ensure(model_file, offline)
        path.write_bytes(b"GGUF stand-in")
        return path

    monkeypatch.setattr(install, "prepare_program", prepare_again)
    monkeypatch.setattr(install, "ensure_model", ensure_again)

    st.choice.exe.unlink()  # deleted, or taken by an antivirus, between two films
    st.pipe.process({"path": str(tmp_path / "media" / "second.mp4"), "basename": "second"})
    assert (st.calls["prepare_program"], st.calls["ensure_model"]) == (2, 1)

    st.model_path.unlink()
    st.pipe.process({"path": str(tmp_path / "media" / "third.mp4"), "basename": "third"})
    assert (st.calls["prepare_program"], st.calls["ensure_model"]) == (2, 2)

    st.pipe.process({"path": str(tmp_path / "media" / "fourth.mp4"), "basename": "fourth"})
    assert (st.calls["prepare_program"], st.calls["ensure_model"]) == (2, 2)  # both in place: nothing again
    assert len(st.program.calls) == 4


@pytest.mark.parametrize("kind", KINDS)
def test_no_speech_groups_means_no_program_run_and_an_empty_srt(kind, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind, groups=())
    md = st.pipe.process(st.media)
    assert st.program.calls == []
    assert st.final.is_file() and st.final.read_text(encoding="utf-8") == ""
    assert md["summary"]["final_subtitles_refined"] == 0 and md["summary"]["degradations"] == []
    assert md["stages"]["slices"]["count"] == 0 and "crispasr" not in md["stages"]


@pytest.mark.parametrize("kind", KINDS)
def test_scenes_the_segmenter_failed_on_become_a_note(kind, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind, unsegmented=2)
    md = st.pipe.process(st.media)
    failed = [n for n in md["summary"]["notes"] if "2 scene" in n]
    assert len(failed) == 1 and md["summary"]["degradations"] == []


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("how", ["error-in-result", "exception"])
def test_a_scene_whose_segmentation_failed_either_way_becomes_that_note(kind, how, tmp_path, monkeypatch):
    from whisperjav.modules.speech_segmentation import SegmentationResult, SpeechSegment, SpeechSegmenterFactory

    class Segmenter:
        def segment(self, audio, sample_rate=16000, **kwargs):
            if Path(audio).name == "film_scene_0000.wav":
                if how == "exception":
                    raise RuntimeError("stand-in segmenter failure")
                return SegmentationResult(segments=[], groups=[], method="stand-in", audio_duration_sec=30.0,
                                          parameters={}, error="stand-in inference failure")
            speech = SpeechSegment(10.0, 13.0)
            return SegmentationResult(segments=[speech], groups=[[speech]], method="stand-in",
                                      audio_duration_sec=30.0, parameters={})

        def cleanup(self):
            pass

    monkeypatch.setattr(SpeechSegmenterFactory, "create", staticmethod(lambda name, config=None, **kw: Segmenter()))
    st = stage(monkeypatch, tmp_path, kind, real_speech_groups=True)
    md = st.pipe.process(st.media)
    failed = [n for n in md["summary"]["notes"] if "segmenter failed on 1 scene" in n]
    assert len(failed) == 1 and md["summary"]["degradations"] == []
    # only scene 1 (from 29.65 s) gave speech: 10-13 s into it
    assert flat(st.program.calls[0].slices) == pytest.approx([39.65, 42.65], abs=1.0 / RATE)


@pytest.mark.parametrize("kind", KINDS)
def test_window_slices_run_without_the_speech_segmenter(kind, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind, speech_segmenter="none", slice_form="windows")
    md = st.pipe.process(st.media)
    assert st.calls["groups"] == 0
    S = kind.module._Scene
    expected, _ = st.pipe._window_slices([S(i, Path("x.wav"), a, b) for i, (a, b) in enumerate(SCENES)], DURATION)
    assert flat(st.program.calls[0].slices) == pytest.approx(flat(expected), abs=1.0 / RATE)
    assert md["stages"]["slices"]["form"] == "windows"
    assert_lines(srt_lines(st.final), expected_lines([(c.start, c.end) for c in expected]))


@pytest.mark.parametrize("kind", KINDS)
def test_a_model_path_outside_ascii_is_staged_once_for_the_pass_beside_the_run_folders(kind, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind, model_dir="モデル")
    made = []
    real = install.link_or_copy
    monkeypatch.setattr(install, "link_or_copy",
                        lambda src, dst, what=None: made.append((Path(dst), what)) or real(src, dst, what=what))
    md1 = st.pipe.process(st.media)
    md2 = st.pipe.process({"path": str(tmp_path / "media" / "second.mp4"), "basename": "second"})
    first, second = st.program.calls
    name = st.pipe.model.filename
    # -m is a short relative ASCII name from the film's run folder, the program's working folder (plan 1.7)
    assert first.request.model == second.request.model == os.path.join("..", name)
    assert first.staged_model == second.staged_model == b"GGUF stand-in"
    assert first.run_dir != second.run_dir and first.run_dir.parent == second.run_dir.parent
    # one staged model for the pass; the audio goes into each film's own run folder
    assert [dst for dst, what in made if what == "a copy of the model"] == [first.run_dir.parent / name]
    assert [dst for dst, what in made if what == "a copy of the audio"] == [first.run_dir / "audio.wav",
                                                                           second.run_dir / "audio.wav"]
    # the program echoes "-m ..\<file>", which the header check accepts
    for md in (md1, md2):
        assert md["summary"]["final_subtitles_refined"] == len(EXPECTED_CUTS[kind.cls.PIPELINE_NAME])


def test_a_build_with_its_own_switches_passes_them_before_the_model_switches(tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, PARAKEET, model="parakeet-ctc-1.1b-ja", build=APPLE_CPU, note=None)
    md = st.pipe.process(st.media)
    call = st.program.calls[0]
    assert tuple(call.request.extra_args) == ("--no-gpu", "--punc-model", "none")
    assert call.argv[-3:] == ["--no-gpu", "--punc-model", "none"]
    assert md["summary"]["notes"] == [] and md["config"]["crispasr"]["build"] == APPLE_CPU.key


@pytest.mark.parametrize("kind", KINDS)
def test_a_japanese_file_name_keeps_the_run_folder_name_ascii(kind, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind, basename="テスト動画")
    st.pipe.process(st.media)
    run_dir = st.program.calls[0].run_dir
    assert run_dir == st.run_dir and run_dir.name.isascii()
    assert st.final.name == "テスト動画.ja.whisperjav.srt"
    assert_lines(srt_lines(st.final), expected_lines(EXPECTED_CUTS[kind.cls.PIPELINE_NAME]))


@pytest.mark.parametrize("kind", KINDS)
def test_the_metadata_file_is_written_when_asked(kind, tmp_path, monkeypatch):
    st = stage(monkeypatch, tmp_path, kind, save_metadata_json=True, note=NOTE_GPU_FAILED, note_in_summary=True)
    md = st.pipe.process(st.media)
    saved = json.loads((st.pipe.temp_dir / "film_master.json").read_text(encoding="utf-8"))
    assert saved["summary"]["notes"] == md["summary"]["notes"] == [NOTE_GPU_FAILED]
    assert saved["config"]["crispasr"]["model"] == st.pipe.model.name


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("steps", [None, ()], ids=["default-steps", "no-steps"])
def test_a_lone_nonverbal_line_is_dropped_only_when_that_step_is_chosen(kind, steps, tmp_path, monkeypatch):
    def respond(slices, request):
        segments = one_sentence_per_slice(slices, request)
        segments[1] = json_segment(request.backend, 1, slices[1], ("あ", "。"), len(slices))
        return segments

    st = stage(monkeypatch, tmp_path, kind, program=Program(respond=respond), postprocess=steps)
    md = st.pipe.process(st.media)
    summary = md["summary"]
    if steps is None:
        assert "あ。" not in [t for t, _, _ in srt_lines(st.final)]
        assert summary["quality_metrics"]["nonverbal_filtered"] == 1
        assert summary["final_subtitles_raw"] == summary["final_subtitles_refined"] + 1
    else:
        assert "あ。" in [t for t, _, _ in srt_lines(st.final)]
        assert summary["quality_metrics"]["nonverbal_filtered"] == 0
        assert summary["final_subtitles_raw"] == summary["final_subtitles_refined"]


@pytest.mark.parametrize("kind", KINDS)
def test_slices_that_gave_no_text_are_counted_in_a_note(kind, tmp_path, monkeypatch):
    """Plan 2.6 (Q10, Q21): "slices with no text" are counted in the end-of-run summary as a note.

    Slice 1 comes back as one empty segment (the program's answer to silence, runs M1/M2); slice 2 gives no
    segment at all. The wording is the owner's (step 3); this checks only that the count reaches the summary.
    """
    def respond(slices, request):
        segments = one_sentence_per_slice(slices, request)
        silent = dict(segments[1], text="", words=[], tokens=[])
        return [segments[0], silent] + segments[3:]

    st = stage(monkeypatch, tmp_path, kind, program=Program(respond=respond))
    md = st.pipe.process(st.media)
    assert md["summary"]["degradations"] == []
    counted = [n for n in md["summary"]["notes"] if n != NOTE_CPU and re.search(r"\b2\b", n) and "slice" in n]
    assert counted, f"no note counts the 2 slices that gave no text; notes: {md['summary']['notes']}"


@pytest.mark.parametrize("kind", KINDS)
def test_the_last_run_is_kept_for_the_measurement_script_before_the_lines_are_built(kind, tmp_path, monkeypatch):
    """Step 2b rebuilds the lines with other settings from the same program run (plan 3.6), so the run's
    scenes, slices and placed segments are kept even when building the lines then fails."""
    st = stage(monkeypatch, tmp_path, kind)

    def line_building_fails(*args, **kwargs):
        raise RuntimeError("stand-in line-building failure")

    monkeypatch.setattr(st.pipe, "_build_lines", line_building_fails)
    with pytest.raises(RuntimeError, match="stand-in line-building failure"):
        st.pipe.process(st.media)
    run = st.pipe.last_run
    assert set(run) == {"basename", "scenes", "cuts", "cut_scene", "segments"}
    expected = EXPECTED_CUTS[kind.cls.PIPELINE_NAME]
    assert run["basename"] == "film" and [sc.index for sc in run["scenes"]] == [0, 1]
    assert flat(run["cuts"]) == pytest.approx(flat(expected), abs=1.0 / RATE)
    assert run["cut_scene"] == ([0, 0, 1, 1] if len(expected) == 4 else [0, 0, 1])
    assert [i for i, _ in run["segments"]] == list(range(len(expected)))
    assert [seg.text for _, seg in run["segments"]] == ["".join(s) for s in SENTENCES[:len(expected)]]


# ---------------------------------------------------------------------------------------------------
# Post-processing with content (plan 3.6): each step does its own job, and only when it is chosen
# ---------------------------------------------------------------------------------------------------

SOUND_ONLY = "はぁはぁ…んっ"  # sound kana only: the non-linguistic filter's case
POSTPROCESS_INPUT = [
    ("テストです。", 1.0, 2.0),
    (SOUND_ONLY, 3.0, 4.0),
    ("もう一度言ってください。", 10.0, 14.0),
    ("もう一度", 11.0, 12.0),           # nested inside the line before: a truncated duplicate
    ("今日は晴れです。", 20.0, 22.5),
    ("明日も雨です。", 22.0, 24.0),     # starts 0.5 s before the line before it ends
]
SHIFTED = ("明日も雨です。", 22.501, 24.0)  # its start pushed to just after the earlier line's end


def write_srt(path: Path, lines) -> Path:
    path.write_text(srt.compose([srt.Subtitle(index=i, start=timedelta(seconds=a), end=timedelta(seconds=b),
                                              content=text) for i, (text, a, b) in enumerate(lines, 1)]),
                    encoding="utf-8")
    return path


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("steps, dropped_sound, resolved, stats", [
    (None, True, True, {"nonverbal_dropped": 0, "nonlinguistic_dropped": 1, "empty_removed": 0,
                        "overlap_nested_dropped": 1, "overlap_shifted": 1}),
    (("nonverbal", "overlap"), False, True, {"nonverbal_dropped": 0, "empty_removed": 0,
                                             "overlap_nested_dropped": 1, "overlap_shifted": 1}),
    (("nonverbal", "nonlinguistic"), True, False, {"nonverbal_dropped": 0, "nonlinguistic_dropped": 1,
                                                   "empty_removed": 0}),
    ((), False, False, {}),
], ids=["default-steps", "no-nonlinguistic", "no-overlap", "no-steps"])
def test_post_processing_does_each_steps_job_only_when_chosen(kind, steps, dropped_sound, resolved, stats, tmp_path):
    pipe = make(kind, tmp_path, postprocess=steps)
    stitched = write_srt(tmp_path / "film_stitched.srt", POSTPROCESS_INPUT)
    assert pipe._postprocess(stitched) == stats
    expected = [line for line in POSTPROCESS_INPUT
                if not (dropped_sound and line[0] == SOUND_ONLY) and not (resolved and line[0] == "もう一度")]
    if resolved:
        expected = expected[:-1] + [SHIFTED]
    assert_lines(srt_lines(stitched), expected)
