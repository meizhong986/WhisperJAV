"""The step 2b measurement script, scripts/measure/crispasr_step2b.py (CrispASR master plan 3.6, 3.7, 4.5).

The script is imported from its path. Its runs are checked with every heavy collaborator replaced as
tests/test_crispasr_pipelines.py replaces it: install.prepare_program and install.ensure_model; the pipeline's
_extract_audio, _detect_scenes and _speech_groups; run_crispasr_json in the pipeline module's namespace. The
script's git probe is replaced too. Nothing here starts the CrispASR program, downloads anything, reaches the
network, runs git or reads ~/.cache: an autouse fixture makes every such call fail the test. Line building uses the
real stable-ts reconstruction (no model is loaded), as the pipeline tests do. The Japanese lines are neutral test
sentences, not film text.

Covered:
- arguments: the six configurations of plan 4.5 and their folder names, the measurement-only model, the refusals;
- the script's copies of the pipelines' slice forms, line modes and post-processing steps are the classes' own;
- one process() call -- one program run -- writes the pipeline's own SRT and every variant under the expected
  names, with one SRT per film in each folder (subeval's pairing rule);
- the measured pipeline keeps every default but the model, the slice form and the kept temporary files;
- variants differ where their settings differ: Parakeet's own segments against REGROUP_JAV, the non-verbal
  line filter on and off, the scene-overlap resolver on and off; the pipeline's own SRT is the variant with
  its own settings;
- run.json: the configuration, the pinned model, the program build, the program run, the slices, the variants;
- a failed program run: run.json says why, the program's log is copied, no variant is written;
- an earlier measurement is kept unless --overwrite is given;
- --dry-run starts, imports and writes nothing (in this process and in a fresh interpreter); --help exits 0.

Run with: python -m pytest tests/test_crispasr_measure_script.py -q -p no:cacheprovider
"""

import importlib.util
import inspect
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import soundfile as sf
import srt

from whisperjav.crispasr import install, manifest
from whisperjav.crispasr.install import ProgramChoice, StartCheck
from whisperjav.crispasr.runner import CrispOutcome, CrispRunResult, film_run_dir
from whisperjav.pipelines import cohere_crispasr_pipeline as cohere_mod
from whisperjav.pipelines import parakeet_crispasr_pipeline as parakeet_mod
from whisperjav.pipelines.cohere_crispasr_pipeline import CohereCrispASRPipeline
from whisperjav.pipelines.parakeet_crispasr_pipeline import ParakeetCrispASRPipeline

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "measure" / "crispasr_step2b.py"
RATE = 16000

# Plan 4.5: the six configurations of the Q12 runs, written out here rather than read from the script.
SIX = (
    ("parakeet-crispasr", "parakeet-tdt-0.6b-ja", "speech"),
    ("parakeet-crispasr", "parakeet-tdt-0.6b-ja", "windows"),
    ("parakeet-crispasr", "parakeet-ctc-1.1b-ja", "speech"),
    ("cohere-crispasr", "cohere-asr-ja-v0.1", "speech"),
    ("cohere-crispasr", "cohere-asr-ja-v0.1", "windows"),
    ("cohere-crispasr", "cohere-asr-ja", "speech"),
)
SIX_CASES = [pytest.param(p, m, s, id=f"{m}_{s}") for p, m, s in SIX]

EXPECTED_VARIANTS = {
    "parakeet-crispasr": ["lines-regroup_post-none", "lines-regroup_post-text", "lines-regroup_post-text+overlap",
                          "lines-segments_post-none", "lines-segments_post-text", "lines-segments_post-text+overlap"],
    "cohere-crispasr": ["lines-regroup_post-none", "lines-regroup_post-text", "lines-regroup_post-text+overlap"],
}
CLASSES = {"parakeet-crispasr": ParakeetCrispASRPipeline, "cohere-crispasr": CohereCrispASRPipeline}

GIT = {"commit": "0" * 40, "branch": "stand-in", "checkout": "stand-in", "uncommitted_in_whisperjav": []}
NOTE_CPU = "CrispASR runs on the processor (CPU build), which is slower: no NVIDIA GPU was found."
WIN_CPU = next(b for b in manifest.PROGRAM_BUILDS if b.platform == "win32" and b.gpu is None)


def load_script(name: str = "crispasr_step2b"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script()


@pytest.fixture(autouse=True)
def nothing_real(monkeypatch, script):
    """No test may start the program, download anything, read ~/.cache or run git: each sets its own stand-ins."""
    def refuse(name):
        def call(*args, **kwargs):
            raise AssertionError(f"{name} was called: the test must replace it with a stand-in")
        return call

    for name in ("prepare_program", "ensure_program", "start_check", "ensure_model", "_download", "_hf_download"):
        monkeypatch.setattr(install, name, refuse(f"install.{name}"))
    for module in (parakeet_mod, cohere_mod):
        monkeypatch.setattr(module, "run_crispasr_json", refuse("run_crispasr_json"))
    monkeypatch.setattr(script, "git_state", lambda repo, run=None: dict(GIT))


# ---------------------------------------------------------------------------------------------------
# The film, the program's JSON, and the stand-ins
# ---------------------------------------------------------------------------------------------------

# 60 s, two semantic scenes sharing 0.7 s, four speech groups; two of them overlap at the scene edge.
DURATION = 60.0
SCENES = ((0.0, 30.35), (29.65, 60.0))
GROUPS = ((0, 2.0, 6.0, ()), (0, 27.0, 30.3, ()), (1, 29.7, 33.0, ()), (1, 40.0, 55.0, (47.0,)))

SENTENCES = (("テスト", "です", "。"), ("もう", "一度", "。"), ("今日", "は", "晴れ", "。"), ("明日", "も", "雨", "。"),
             ("本当", "です", "。"), ("大丈夫", "です", "。"))
NONVERBAL = ("あ", "。")  # a lone token the non-verbal line filter drops
LEAD_S, PIECE_S, STOP_S = 0.2, 0.3, 0.1


def write_wav(path: Path, seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(max(1, int(round(seconds * RATE))), dtype=np.float32), RATE, subtype="PCM_16")
    return path


def srt_lines(path: Path) -> List[Tuple[str, float, float]]:
    return [(s.content, s.start.total_seconds(), s.end.total_seconds())
            for s in srt.parse(Path(path).read_text(encoding="utf-8"))]


def timed(pieces: Sequence[str], t0: float):
    out, t = [], t0
    for p in pieces:
        d = STOP_S if p == "。" else PIECE_S
        out.append((p, t, t + d))
        t += d
    return out


def _ms(t: float) -> int:
    return int(round(t * 1000))


def _item(text: str, a: float, b: float, p: Optional[float] = None) -> dict:
    item = {"text": text, "t0": int(round(a * 100)), "t1": int(round(b * 100)), "offsets": {"from": _ms(a), "to": _ms(b)}}
    if p is not None:
        item["p"] = p
    return item


def segment_json(backend: str, chunk_id: int, parts, sl, n_slices: int) -> dict:
    """One -ojf segment in the shape the program writes for this back end (second study S2-S4, P1)."""
    text = "".join(p for p, _, _ in parts)
    if backend == "parakeet":  # native words; the blank token that starts long before the segment
        start, end = parts[0][1], parts[-1][2]
        words = [_item(p, a, b) for p, a, b in parts]
        tokens = [_item(" ", max(0.0, start - 10.9), start, 0.99)] + [_item(p, a, b, 0.98) for p, a, b in parts]
    elif backend == "fastconformer-ctc":  # tokens only, 80 ms each
        start, end = sl
        words = []
        tokens = [_item(ch, parts[0][1] + 0.12 * k, parts[0][1] + 0.12 * k + 0.08, 0.9) for k, ch in enumerate(text)]
    elif backend == "cohere":  # one word for the whole segment; it ends at the slice end; a zero-length token
        (first, a, b), rest = parts[0], parts[1:]
        start, end = a, sl[1]
        words = [_item(text, start, end)]
        tokens = ([_item(first[:1], a, a, 0.5), _item(first[1:], a, b, 0.99)]
                  + [_item(p, x, y, 0.99) for p, x, y in rest])
    else:
        raise ValueError(backend)
    segment = {"offsets": {"from": _ms(start), "to": _ms(end)}, "text": text, "chunk_id": chunk_id,
               "words": words, "tokens": tokens}
    if n_slices == 1:
        del segment["chunk_id"]  # the program omits it when the run had one slice
    return segment


def one_sentence_per_slice(slices, request) -> List[dict]:
    return [segment_json(request.backend, i, timed(SENTENCES[i % len(SENTENCES)], sl[0] + LEAD_S), sl, len(slices))
            for i, sl in enumerate(slices)]


def mixed(slices, request) -> List[dict]:
    """For the speech slices of this film: lines whose variants must differ.

    Slice 0: two sentences 2.1 s apart in ONE segment. Slice 1: the lone non-verbal token. Slice 2: two segments
    whose times overlap by 0.2 s. Any further slice: one sentence.
    """
    b, n = request.backend, len(slices)
    s0, s1, s2 = slices[0], slices[1], slices[2]
    out = [segment_json(b, 0, timed(SENTENCES[0], s0[0] + 0.2) + timed(SENTENCES[1], s0[0] + 3.0), s0, n),
           segment_json(b, 1, timed(NONVERBAL, s1[0] + 0.2), s1, n),
           segment_json(b, 2, timed(SENTENCES[2], s2[0] + 0.2), s2, n),
           segment_json(b, 2, timed(SENTENCES[3], s2[0] + 1.0), s2, n)]
    out += [segment_json(b, i, timed(SENTENCES[4], sl[0] + LEAD_S), sl, n) for i, sl in enumerate(slices[3:], start=3)]
    return out


@dataclass
class Program:
    """Stands in for run_crispasr_json: reads cuts.json, writes the JSON and a run.log, as the program does."""
    respond: Callable = one_sentence_per_slice
    outcome: CrispOutcome = CrispOutcome.OK
    exit_code: int = 0
    write_json: bool = True
    warning_count: int = 0
    calls: list = field(default_factory=list)

    def __call__(self, request, on_progress=None, cancel_event=None, poll_s=0.5):
        run_dir = Path(request.run_dir)
        cuts = json.loads((run_dir / request.cuts_name).read_text(encoding="utf-8"))["crispasr_vad"]
        rate = cuts["sample_rate"]
        slices = [(s["start"] / rate, s["end"] / rate) for s in cuts["slices"]]
        self.calls.append(SimpleNamespace(request=request, slices=slices))
        if self.write_json:
            doc = {"crispasr": {"backend": request.backend, "model": request.model, "language": request.language},
                   "transcription": self.respond(slices, request)}
            request.json_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        request.log_path.write_text(
            f"command: stand-in\nworking folder: {run_dir}\nexit code: {self.exit_code}\n"
            f"outcome: {self.outcome.value}\nseconds: 1.5\n----- stderr\ncrispasr: stand-in stderr\n", encoding="utf-8")
        usable = self.write_json and self.outcome in (CrispOutcome.OK, CrispOutcome.CRASHED)
        return CrispRunResult(outcome=self.outcome, exit_code=self.exit_code,
                              json_path=request.json_path if usable else None,
                              warnings=["warning: stand-in"] * self.warning_count, warning_count=self.warning_count,
                              download_lines=[], stderr_tail="crispasr: stand-in stderr", seconds=1.5,
                              log_path=request.log_path)


@dataclass
class Bench:
    script: object
    program: Program
    choice: ProgramChoice
    film: Path
    out: Path
    models: Path
    built: list   # the pipelines the script's build_pipeline made
    calls: dict

    def argv(self, pipeline: str, model: Optional[str], slice_form: str, *extra: str) -> List[str]:
        args = [str(self.film), "--pipeline", pipeline, "--slice-form", slice_form, "--out", str(self.out), *extra]
        return args + (["--model", model] if model else [])

    def folder(self, model: str, slice_form: str) -> Path:
        return self.out / f"{model}_{slice_form}"

    def record(self, model: str, slice_form: str) -> dict:
        return json.loads((self.folder(model, slice_form) / "film.run.json").read_text(encoding="utf-8"))


def make_film(tmp_path: Path) -> Path:
    film = tmp_path / "media" / "film.mkv"
    film.parent.mkdir(parents=True, exist_ok=True)
    film.write_bytes(b"stand-in film")
    return film


def make_bench(monkeypatch, tmp_path: Path, script, program: Optional[Program] = None) -> Bench:
    """The script's build_pipeline, wrapped so the pipeline it builds gets the pipeline tests' stand-ins."""
    exe = tmp_path / "program" / WIN_CPU.exe_name
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"stand-in program")
    choice = ProgramChoice(build=WIN_CPU, exe=exe, note=NOTE_CPU,
                           check=StartCheck(True, None, version=manifest.CRISPASR_VERSION, git=manifest.CRISPASR_GIT))
    models = tmp_path / "models"
    calls = {"prepare_program": 0, "ensure_model": 0, "extract": 0, "scenes": 0, "groups": 0, "process": 0}

    def prepare_program(offline=None):
        calls["prepare_program"] += 1
        return choice

    def ensure_model(model_file, offline=None):
        calls["ensure_model"] += 1
        path = models / model_file.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF stand-in")
        return path

    monkeypatch.setattr(install, "prepare_program", prepare_program)
    monkeypatch.setattr(install, "ensure_model", ensure_model)
    program = program or Program()
    for module in (parakeet_mod, cohere_mod):
        monkeypatch.setattr(module, "run_crispasr_json", program)

    real_build = script.build_pipeline
    built = []

    def build_pipeline(cfg):
        pipe = real_build(cfg)
        module = sys.modules[type(pipe).__module__]
        real_process = pipe.process

        def extract(input_file, name):
            calls["extract"] += 1
            return write_wav(tmp_path / "media" / f"{name}_audio.wav", 0.1), DURATION

        def detect(wav, name):
            calls["scenes"] += 1
            return [module._Scene(i, write_wav(pipe.temp_dir / "scenes" / f"{name}_scene_{i:04d}.wav", 0.1), a, b)
                    for i, (a, b) in enumerate(SCENES)]

        def speech_groups(scene_list):
            calls["groups"] += 1
            return [module._Group(sc, a, b, list(p)) for sc, a, b, p in GROUPS], 0

        def process(media_info):
            calls["process"] += 1
            return real_process(media_info)

        monkeypatch.setattr(pipe, "_extract_audio", extract)
        monkeypatch.setattr(pipe, "_detect_scenes", detect)
        monkeypatch.setattr(pipe, "_speech_groups", speech_groups)
        monkeypatch.setattr(pipe, "process", process)
        built.append(pipe)
        return pipe

    monkeypatch.setattr(script, "build_pipeline", build_pipeline)
    return Bench(script, program, choice, make_film(tmp_path), tmp_path / "batch", models, built, calls)


@pytest.fixture
def bench(monkeypatch, tmp_path, script):
    return make_bench(monkeypatch, tmp_path, script)


# ---------------------------------------------------------------------------------------------------
# Arguments and names
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pipeline, model, slice_form", SIX_CASES)
def test_each_configuration_of_plan_4_5_parses_and_names_its_folder(script, tmp_path, pipeline, model, slice_form):
    film = make_film(tmp_path)
    cfg = script.parse_args([str(film), "--pipeline", pipeline, "--model", model, "--slice-form", slice_form,
                             "--out", str(tmp_path / "batch")])
    assert (cfg.pipeline, cfg.slice_form) == (pipeline, slice_form)
    assert cfg.model is manifest.MODELS[model]
    assert cfg.label == f"{model}_{slice_form}"
    assert cfg.folder == tmp_path / "batch" / cfg.label and cfg.temp_dir == cfg.folder / "temp"
    assert cfg.film == film and cfg.basename == "film"
    assert (cfg.overwrite, cfg.dry_run) == (False, False)


def test_the_script_lists_the_six_configurations_with_six_folder_names(script):
    assert script.STEP_2B_CONFIGURATIONS == SIX
    assert len({script.config_label(m, s) for _, m, s in SIX}) == 6


def test_the_parakeet_default_model_comes_from_the_manifest(script, tmp_path):
    cfg = script.parse_args([str(make_film(tmp_path)), "--pipeline", "parakeet-crispasr", "--slice-form", "speech",
                             "--out", str(tmp_path / "batch")])
    assert cfg.model.name == manifest.PIPELINE_DEFAULT_MODEL["parakeet-crispasr"] == "parakeet-tdt-0.6b-ja"
    assert cfg.label == "parakeet-tdt-0.6b-ja_speech"


def test_the_measurement_only_model_is_accepted_but_not_offered(script, tmp_path):
    assert "parakeet-ctc-1.1b-ja" not in manifest.PIPELINE_MODELS["parakeet-crispasr"]
    cfg = script.parse_args([str(make_film(tmp_path)), "--pipeline", "parakeet-crispasr", "--model",
                             "parakeet-ctc-1.1b-ja", "--slice-form", "speech", "--out", str(tmp_path / "batch")])
    assert cfg.model is manifest.MODELS["parakeet-ctc-1.1b-ja"] and cfg.label == "parakeet-ctc-1.1b-ja_speech"


def test_another_longest_slice_names_its_own_folder(script, tmp_path):
    cfg = script.parse_args([str(make_film(tmp_path)), "--pipeline", "cohere-crispasr", "--model", "cohere-asr-ja-v0.1",
                             "--slice-form", "speech", "--max-slice-s", "12", "--out", str(tmp_path / "batch")])
    assert cfg.max_slice_s == 12.0 and cfg.label == "cohere-asr-ja-v0.1_speech_max12s"
    assert cfg.folder == tmp_path / "batch" / "cohere-asr-ja-v0.1_speech_max12s"


REFUSED = [
    pytest.param(["--pipeline", "cohere-crispasr", "--slice-form", "speech"],
                 "has no default model; name one with --model: cohere-asr-ja-v0.1, cohere-asr-ja", id="cohere-no-model"),
    pytest.param(["--pipeline", "cohere-crispasr", "--model", "parakeet-ctc-1.1b-ja", "--slice-form", "speech"],
                 "is not a model of cohere-crispasr", id="parakeet-model-on-cohere"),
    pytest.param(["--pipeline", "parakeet-crispasr", "--model", "cohere-asr-ja", "--slice-form", "speech"],
                 "choose one of: parakeet-tdt-0.6b-ja, parakeet-ctc-1.1b-ja", id="cohere-model-on-parakeet"),
    pytest.param(["--pipeline", "parakeet-crispasr", "--slice-form", "scenes"], "invalid choice", id="slice-form"),
    pytest.param(["--pipeline", "whisper", "--slice-form", "speech"], "invalid choice", id="pipeline"),
    pytest.param(["--pipeline", "parakeet-crispasr"], "required: --slice-form", id="no-slice-form"),
    pytest.param(["--pipeline", "parakeet-crispasr", "--slice-form", "speech", "--max-slice-s", "0"],
                 "--max-slice-s must be more than 0 seconds", id="max-slice-zero"),
]


@pytest.mark.parametrize("args, words", REFUSED)
def test_bad_arguments_stop_with_exit_status_2(script, tmp_path, capsys, args, words):
    with pytest.raises(SystemExit) as info:
        script.parse_args([str(make_film(tmp_path)), *args, "--out", str(tmp_path / "batch")])
    assert info.value.code == 2
    assert words in capsys.readouterr().err


def test_a_missing_film_stops_with_exit_status_2(script, tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        script.parse_args([str(tmp_path / "no such film.mkv"), "--pipeline", "parakeet-crispasr", "--slice-form",
                           "speech", "--out", str(tmp_path / "batch")])
    assert info.value.code == 2 and "the film was not found" in capsys.readouterr().err
    assert not (tmp_path / "batch").exists()


@pytest.mark.parametrize("cls", [ParakeetCrispASRPipeline, CohereCrispASRPipeline])
def test_the_script_s_names_are_the_pipelines_own(script, cls):
    name = cls.PIPELINE_NAME
    assert tuple(script.PIPELINE_CLASSES) == manifest.CRISPASR_PIPELINES
    assert script.PIPELINE_CLASSES[name] == (cls.__module__, cls.__name__)
    assert script.pipeline_class(name) is cls
    assert script.SLICE_FORMS == tuple(cls.SLICE_FORMS)
    assert script.LINE_MODES[name] == tuple(cls.LINE_MODES)
    assert {s for steps in script.POSTPROCESS_SETS.values() for s in steps} <= set(cls.POSTPROCESS_STEPS)
    script.check_against(cls, name)  # no refusal


def test_the_post_processing_candidates_are_those_of_plan_3_6(script):
    assert script.POSTPROCESS_SETS == {"none": (), "text": ("nonverbal", "nonlinguistic"),
                                       "text+overlap": ("nonverbal", "nonlinguistic", "overlap")}


def test_a_pipeline_that_no_longer_matches_the_script_is_refused(script):
    class MoreLineModes(ParakeetCrispASRPipeline):
        LINE_MODES = ("regroup", "segments", "sentences")

    class FewerSteps(CohereCrispASRPipeline):
        POSTPROCESS_STEPS = ("nonverbal", "overlap")

    with pytest.raises(script.Refused, match="line modes"):
        script.check_against(MoreLineModes, "parakeet-crispasr")
    with pytest.raises(script.Refused, match="nonlinguistic"):
        script.check_against(FewerSteps, "cohere-crispasr")


# ---------------------------------------------------------------------------------------------------
# One run, every variant
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pipeline, model, slice_form", SIX_CASES)
def test_one_program_run_writes_the_pipeline_srt_and_every_variant(bench, pipeline, model, slice_form):
    assert bench.script.main(bench.argv(pipeline, model, slice_form)) == 0

    # one process() call, one program run; the variants prepared, extracted and ran nothing
    assert bench.calls["process"] == 1 and len(bench.program.calls) == 1 and len(bench.built) == 1
    assert (bench.calls["prepare_program"], bench.calls["ensure_model"], bench.calls["extract"]) == (1, 1, 1)
    assert bench.calls["groups"] == (1 if slice_form == "speech" else 0)

    folder = bench.folder(model, slice_form)
    names = EXPECTED_VARIANTS[pipeline]
    assert {p.name for p in folder.iterdir() if p.is_dir()} == set(names) | {"temp"}
    assert {p.name for p in folder.iterdir() if p.is_file()} == {"film.ja.whisperjav.srt", "film.run.json",
                                                                 "film.run.log", "film.cuts.json"}
    # subeval pairs a reference with the one test SRT whose name starts with the film's: one per folder
    assert [p.name for p in folder.glob("*.srt")] == ["film.ja.whisperjav.srt"]
    for name in names:
        assert [p.name for p in (folder / name).glob("*.srt")] == [f"film.{name}.srt"]
        assert srt_lines(folder / name / f"film.{name}.srt"), name

    # the copies are the program's own files from this run
    run_dir = film_run_dir(folder / "temp", "film")
    assert (folder / "film.cuts.json").read_bytes() == (run_dir / "cuts.json").read_bytes()
    assert (folder / "film.run.log").read_bytes() == (run_dir / "run.log").read_bytes()
    assert (run_dir / "out.json").is_file()  # temporary files kept

    record = bench.record(model, slice_form)
    assert record["error"] is None
    assert [v["name"] for v in record["variants"]] == names
    own = record["pipeline_srt"]
    assert own["file"] == str(folder / "film.ja.whisperjav.srt")
    assert own["settings_variant"] is not None and own["settings_variant"] in own["same_lines_as"]


@pytest.mark.parametrize("pipeline, model, slice_form", [SIX_CASES[0], SIX_CASES[4]])
def test_the_measured_pipeline_keeps_every_default_but_model_slices_and_kept_temp_files(bench, pipeline, model,
                                                                                         slice_form):
    assert bench.script.main(bench.argv(pipeline, model, slice_form)) == 0
    pipe, cls = bench.built[0], CLASSES[pipeline]
    defaults = inspect.signature(cls.__init__).parameters
    folder = bench.folder(model, slice_form)
    assert type(pipe) is cls
    assert (pipe.model.name, pipe.slice_form, pipe.keep_temp_files) == (model, slice_form, True)
    assert (pipe.output_dir, pipe.temp_dir) == (folder, folder / "temp")
    assert pipe.line_mode == defaults["line_mode"].default
    assert pipe.postprocess == cls.DEFAULT_POSTPROCESS
    assert pipe.speech_segmenter == defaults["speech_segmenter"].default
    assert pipe.save_metadata_json is defaults["save_metadata_json"].default
    assert pipe.lang_code == "ja" and pipe.subs_language == defaults["subs_language"].default


def test_another_longest_slice_reaches_the_slices_and_leaves_the_pipeline_class_alone(bench):
    assert bench.script.main(bench.argv("cohere-crispasr", "cohere-asr-ja-v0.1", "speech", "--max-slice-s", "12")) == 0
    pipe = bench.built[0]
    assert isinstance(pipe, CohereCrispASRPipeline) and pipe.MAX_SLICE_S == 12.0
    assert CohereCrispASRPipeline.MAX_SLICE_S == 30.0
    assert max(b - a for a, b in bench.program.calls[0].slices) <= 12.0 + 1e-6  # the 15 s speech group is split
    record = json.loads((bench.out / "cohere-asr-ja-v0.1_speech_max12s" / "film.run.json").read_text(encoding="utf-8"))
    c = record["configuration"]
    assert (c["label"], c["max_slice_s"], c["max_slice_s_option"]) == ("cohere-asr-ja-v0.1_speech_max12s", 12.0, 12.0)


def test_without_the_option_the_pipeline_s_own_longest_slice_is_used_and_recorded(bench):
    assert bench.script.main(bench.argv("cohere-crispasr", "cohere-asr-ja-v0.1", "speech")) == 0
    assert type(bench.built[0]) is CohereCrispASRPipeline
    assert max(b - a for a, b in bench.program.calls[0].slices) > 12.0  # the 15 s speech group stays whole
    c = bench.record("cohere-asr-ja-v0.1", "speech")["configuration"]
    assert (c["max_slice_s"], c["max_slice_s_option"]) == (30.0, None)


def lines_of(folder: Path, names) -> dict:
    return {name: srt_lines(folder / name / f"film.{name}.srt") for name in names}


def test_parakeet_variants_differ_where_their_settings_differ(script, monkeypatch, tmp_path):
    b = make_bench(monkeypatch, tmp_path, script, program=Program(respond=mixed))
    assert b.script.main(b.argv("parakeet-crispasr", "parakeet-tdt-0.6b-ja", "speech")) == 0
    folder = b.folder("parakeet-tdt-0.6b-ja", "speech")
    v = lines_of(folder, EXPECTED_VARIANTS["parakeet-crispasr"])
    texts = {name: [t for t, _, _ in lines] for name, lines in v.items()}

    # line building: Parakeet's own segment keeps both sentences on one line; REGROUP_JAV splits them at the gap
    for post in ("none", "text", "text+overlap"):
        assert "テストです。もう一度。" in texts[f"lines-segments_post-{post}"]
        assert "テストです。もう一度。" not in texts[f"lines-regroup_post-{post}"]
        assert {"テストです。", "もう一度。"} <= set(texts[f"lines-regroup_post-{post}"])

    # the non-verbal line filter: the lone token stays only where no post-processing runs
    for lines in ("regroup", "segments"):
        assert "あ。" in texts[f"lines-{lines}_post-none"]
        assert "あ。" not in texts[f"lines-{lines}_post-text"]
        assert "あ。" not in texts[f"lines-{lines}_post-text+overlap"]

    # the scene-overlap resolver: two of Parakeet's segments overlap by 0.2 s; it moves the later line's start
    before = {t: (a, e) for t, a, e in v["lines-segments_post-text"]}
    after = {t: (a, e) for t, a, e in v["lines-segments_post-text+overlap"]}
    assert before["今日は晴れ。"] == pytest.approx((40.2, 41.2), abs=0.0015)
    assert before["明日も雨。"][0] == pytest.approx(41.0, abs=0.0015)
    assert after["明日も雨。"][0] == pytest.approx(41.201, abs=0.0015)
    assert after["今日は晴れ。"] == before["今日は晴れ。"]

    # the pipeline's own SRT is the variant with its own settings, from the same single program run
    own = srt_lines(folder / "film.ja.whisperjav.srt")
    assert own == v["lines-regroup_post-text+overlap"] and own != v["lines-regroup_post-none"]
    assert len(b.program.calls) == 1

    record = b.record("parakeet-tdt-0.6b-ja", "speech")
    by_name = {r["name"]: r for r in record["variants"]}
    assert by_name["lines-regroup_post-text+overlap"]["same_lines_as_pipeline_srt"] is True
    assert by_name["lines-regroup_post-none"]["same_lines_as_pipeline_srt"] is False
    assert by_name["lines-segments_post-none"]["postprocess_stats"] == {}
    assert by_name["lines-segments_post-text"]["postprocess_stats"]["nonverbal_dropped"] == 1
    assert by_name["lines-segments_post-text+overlap"]["postprocess_stats"]["overlap_shifted"] == 1
    assert {k: by_name[k]["subtitles"] for k in by_name} == {k: len(lines) for k, lines in v.items()}
    assert set(record["line_building"]) == {"regroup", "segments"}
    assert record["line_building"]["segments"]["mode"] == "segments"


def test_cohere_variants_differ_where_their_settings_differ(script, monkeypatch, tmp_path):
    b = make_bench(monkeypatch, tmp_path, script, program=Program(respond=mixed))
    assert b.script.main(b.argv("cohere-crispasr", "cohere-asr-ja-v0.1", "speech")) == 0
    folder = b.folder("cohere-asr-ja-v0.1", "speech")
    v = lines_of(folder, EXPECTED_VARIANTS["cohere-crispasr"])
    texts = {name: [t for t, _, _ in lines] for name, lines in v.items()}
    assert "あ。" in texts["lines-regroup_post-none"]
    assert "あ。" not in texts["lines-regroup_post-text"] and "あ。" not in texts["lines-regroup_post-text+overlap"]
    assert {"テストです。", "もう一度。"} <= set(texts["lines-regroup_post-none"])
    assert srt_lines(folder / "film.ja.whisperjav.srt") == v["lines-regroup_post-text+overlap"]
    assert not (folder / "lines-segments_post-none").exists()  # Cohere has no segments line mode
    assert len(b.program.calls) == 1


def test_run_json_records_the_configuration_model_program_run_slices_and_variants(bench):
    bench.program.warning_count = 2
    argv = bench.argv("parakeet-crispasr", "parakeet-ctc-1.1b-ja", "speech")
    assert bench.script.main(argv) == 0
    folder = bench.folder("parakeet-ctc-1.1b-ja", "speech")
    record = bench.record("parakeet-ctc-1.1b-ja", "speech")

    assert record["plan"] == "CrispASR master plan 4.5, step 2b" and record["command"] == argv
    c = record["configuration"]
    assert (c["label"], c["pipeline"], c["pipeline_class"], c["model"], c["slice_form"]) == (
        "parakeet-ctc-1.1b-ja_speech", "parakeet-crispasr", "ParakeetCrispASRPipeline", "parakeet-ctc-1.1b-ja",
        "speech")
    assert (c["film"], c["basename"], c["folder"]) == (str(bench.film), "film", str(folder))
    assert c["constructor_arguments"] == {"output_dir": str(folder), "temp_dir": str(folder / "temp"),
                                          "keep_temp_files": True, "model_name": "parakeet-ctc-1.1b-ja",
                                          "slice_form": "speech"}
    assert (c["line_mode"], c["postprocess"]) == ("regroup", list(ParakeetCrispASRPipeline.DEFAULT_POSTPROCESS))
    assert (c["speech_segmenter"], c["speech_segmenter_used"]) == ("whisperseg", True)

    pin = manifest.MODELS["parakeet-ctc-1.1b-ja"]
    m = record["model"]
    assert (m["repo"], m["file"], m["revision"], m["sha256"], m["size"]) == (
        pin.repo_id, pin.filename, pin.revision, pin.sha256, pin.size)
    assert (m["backend"], m["extra_args"], m["offered_to_users"]) == ("fastconformer-ctc", ["--punc-model", "none"],
                                                                      False)
    assert m["local_path"] == str(bench.models / pin.filename)

    p = record["program"]
    assert (p["version"], p["git"]) == (manifest.CRISPASR_VERSION, manifest.CRISPASR_GIT)
    assert (p["build"], p["build_label"], p["asset"], p["sha256"]) == ("cpu", WIN_CPU.label, WIN_CPU.asset,
                                                                       WIN_CPU.sha256)
    assert p["exe"] == str(bench.choice.exe) and p["cpu_build_reason"] == NOTE_CPU
    assert p["start_check"]["version"] == manifest.CRISPASR_VERSION

    run = record["program_run"]
    assert (run["outcome"], run["exit_code"], run["seconds"], run["warnings"]) == ("ok", 0, 1.5, 2)
    assert run["segments"] == len(bench.program.calls[0].slices)
    assert any("2 warning" in note for note in record["notes"]) and record["degradations"] == []

    slices = bench.program.calls[0].slices
    assert (record["slices"]["form"], record["slices"]["count"]) == ("speech", len(slices)) == ("speech", 4)
    assert record["slices"]["seconds"] == pytest.approx(sum(e - s for s, e in slices), abs=0.01)
    assert (record["scenes"], record["audio_seconds"]) == (len(SCENES), DURATION)

    assert [v["name"] for v in record["variants"]] == EXPECTED_VARIANTS["parakeet-crispasr"]
    for v in record["variants"]:
        assert Path(v["file"]) == folder / v["name"] / f"film.{v['name']}.srt"
        assert v["subtitles"] == len(srt_lines(Path(v["file"])))
        assert v["steps"] == list(bench.script.POSTPROCESS_SETS[v["postprocess"]])
        assert v["line_mode"] == v["name"].split("_")[0][len("lines-"):]

    assert record["evidence"]["run_log"] == str(folder / "film.run.log")
    assert record["evidence"]["cuts"] == str(folder / "film.cuts.json")
    assert record["evidence"]["program_output"] == str(film_run_dir(folder / "temp", "film") / "out.json")
    assert record["whisperjav"]["git"] == GIT and record["whisperjav"]["version"]
    assert record["pipeline_metadata"]["pipeline"] == "parakeet-crispasr"
    assert record["error"] is None


def test_a_failed_program_run_is_recorded_with_its_log_and_writes_no_variant(script, monkeypatch, tmp_path):
    b = make_bench(monkeypatch, tmp_path, script,
                   program=Program(outcome=CrispOutcome.FAILED, exit_code=30, write_json=False))
    assert b.script.main(b.argv("cohere-crispasr", "cohere-asr-ja", "speech")) == 1
    folder = b.folder("cohere-asr-ja", "speech")
    record = b.record("cohere-asr-ja", "speech")
    error = record["error"]
    assert (error["stage"], error["type"]) == ("the pipeline run", "CrispASRRunError")
    assert "required VAD model failed" in error["message"] and "Traceback" in error["traceback"]
    assert record["variants"] == [] and record["pipeline_srt"] is None
    assert not (folder / "film.ja.whisperjav.srt").exists()
    assert not any((folder / name).exists() for name in EXPECTED_VARIANTS["cohere-crispasr"])
    assert (folder / "film.run.log").read_text(encoding="utf-8").startswith("command: stand-in")
    assert (folder / "film.cuts.json").is_file()
    assert record["program_run"] == {"read_from_run_log": {
        "command": "stand-in", "working folder": str(film_run_dir(folder / "temp", "film")), "exit code": "30",
        "outcome": "failed", "seconds": "1.5"}}
    assert record["program"]["build"] == "cpu"  # the build is known although the run failed
    assert len(b.program.calls) == 1


def test_an_earlier_measurement_is_kept_unless_overwrite_is_given(bench, capsys):
    argv = bench.argv("parakeet-crispasr", "parakeet-tdt-0.6b-ja", "windows")
    folder = bench.folder("parakeet-tdt-0.6b-ja", "windows")
    assert bench.script.main(argv) == 0
    first = (folder / "film.run.json").read_text(encoding="utf-8")
    capsys.readouterr()

    assert bench.script.main(argv) == 2
    assert "already measured" in capsys.readouterr().err
    assert len(bench.built) == 1 and len(bench.program.calls) == 1  # nothing was built or run again
    assert (folder / "film.run.json").read_text(encoding="utf-8") == first

    # --overwrite measures again; a failure then leaves none of the earlier SRTs to pass as this run's
    bench.program.outcome, bench.program.exit_code, bench.program.write_json = CrispOutcome.FAILED, 30, False
    assert bench.script.main(argv + ["--overwrite"]) == 1
    assert len(bench.program.calls) == 2
    assert not (folder / "film.ja.whisperjav.srt").exists()
    assert not any((folder / name / f"film.{name}.srt").exists() for name in EXPECTED_VARIANTS["parakeet-crispasr"])
    assert bench.record("parakeet-tdt-0.6b-ja", "windows")["error"]["type"] == "CrispASRRunError"


# ---------------------------------------------------------------------------------------------------
# --dry-run and --help
# ---------------------------------------------------------------------------------------------------

def test_dry_run_starts_and_writes_nothing(bench, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("the dry run built or ran something")

    monkeypatch.setattr(bench.script, "build_pipeline", refuse)
    monkeypatch.setattr(bench.script, "measure", refuse)
    assert bench.script.main(bench.argv("parakeet-crispasr", None, "speech", "--dry-run")) == 0
    out = capsys.readouterr().out
    assert not bench.out.exists()
    assert bench.program.calls == [] and set(bench.calls.values()) == {0}
    pin = manifest.MODELS["parakeet-tdt-0.6b-ja"]
    assert "Dry run: nothing is started, downloaded or written." in out
    for words in ("parakeet-tdt-0.6b-ja_speech", "ParakeetCrispASRPipeline", pin.repo_id, pin.revision,
                  "keep_temp_files=True", "process() once", "film.ja.whisperjav.srt", "film.run.json"):
        assert words in out, words
    for name in EXPECTED_VARIANTS["parakeet-crispasr"]:
        assert f"film.{name}.srt" in out


def test_dry_run_refuses_what_the_run_would_refuse_and_deletes_nothing(bench, capsys):
    argv = bench.argv("cohere-crispasr", "cohere-asr-ja-v0.1", "windows")
    assert bench.script.main(argv) == 0
    folder = bench.folder("cohere-asr-ja-v0.1", "windows")
    before = sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file())
    capsys.readouterr()

    assert bench.script.main(argv + ["--dry-run"]) == 2
    assert "already measured" in capsys.readouterr().err
    assert bench.script.main(argv + ["--dry-run", "--overwrite"]) == 0
    assert "would delete" in capsys.readouterr().out
    assert sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file()) == before
    assert len(bench.program.calls) == 1


DRY_RUN_PROBE = textwrap.dedent("""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("crispasr_step2b", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rc = module.main(sys.argv[2:])
    loaded = [n for n in ("torch", "stable_whisper", "whisperjav.pipelines", "whisperjav.crispasr.install",
                          "whisperjav.crispasr.runner", "huggingface_hub", "requests") if n in sys.modules]
    print("RESULT", rc, loaded)
""")


def test_dry_run_in_a_fresh_interpreter_imports_no_pipeline_and_writes_nothing(tmp_path):
    film, out = make_film(tmp_path), tmp_path / "batch"
    result = subprocess.run([sys.executable, "-c", DRY_RUN_PROBE, str(SCRIPT), str(film), "--pipeline",
                             "cohere-crispasr", "--model", "cohere-asr-ja", "--slice-form", "windows", "--out",
                             str(out), "--dry-run"], capture_output=True, text=True, encoding="utf-8",
                            errors="replace", cwd=str(REPO), timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    assert "RESULT 0 []" in result.stdout, result.stdout[-2000:]
    assert "cohere-asr-ja_windows" in result.stdout
    assert not out.exists()


def test_help_exits_0_and_names_every_option_and_configuration():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", cwd=str(REPO), timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    for option in ("--pipeline", "--model", "--slice-form", "--max-slice-s", "--out", "--overwrite", "--dry-run"):
        assert option in result.stdout
    for pipeline, model, slice_form in SIX:
        assert f"--pipeline {pipeline}" in result.stdout and model in result.stdout


# ---------------------------------------------------------------------------------------------------
# The git record
# ---------------------------------------------------------------------------------------------------

def test_the_git_record_is_read_only_and_names_uncommitted_files(tmp_path):
    fresh = load_script("crispasr_step2b_git")  # its git_state is the real one
    answers = {("rev-parse", "HEAD"): "abc123\n", ("rev-parse", "--abbrev-ref", "HEAD"): "v1.10.0\n",
               ("status", "--porcelain", "--", "whisperjav"): " M whisperjav/main.py\n?? whisperjav/crispasr/\n"}
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        assert argv[:4] == ["git", "--no-optional-locks", "-C", str(tmp_path)]
        return SimpleNamespace(returncode=0, stdout=answers[tuple(argv[4:])])

    assert fresh.git_state(tmp_path, run=run) == {
        "commit": "abc123", "branch": "v1.10.0", "checkout": str(tmp_path),
        "uncommitted_in_whisperjav": [" M whisperjav/main.py", "?? whisperjav/crispasr/"]}
    assert {argv[4] for argv in seen} == {"rev-parse", "status"}  # reading commands only

    def no_git(argv, **kwargs):
        raise FileNotFoundError("git")

    def not_a_checkout(argv, **kwargs):
        return SimpleNamespace(returncode=128, stdout="")

    assert fresh.git_state(tmp_path, run=no_git)["commit"] is None
    assert fresh.git_state(tmp_path, run=not_a_checkout)["commit"] is None
