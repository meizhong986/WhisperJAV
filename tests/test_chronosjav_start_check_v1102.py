"""ChronosJAV's aligner and timing model are downloaded at the start-of-run check (1.10.2, owner W624 C, W626).

Before, each was fetched when the pass first needed it, after its first file's ASR, and a failed download let the
pass go on without it (subtitles at the VAD groups' edges, or no timing step), with a note. Now ONE place decides
which files a ChronosJAV run or pass uses (config/chronosjav_timing.py): the pass worker's settings merge, QwenPipeline
and main's start check and --dump-params all read it, and the start check downloads the files before any audio is
read; a failure stops the run with the box agreed by the owner (W625 3). The dormant Cohere-in-ChronosJAV stays as
it is and outside the check (W626 2).

ROWS pins what each GUI row and each --mode qwen case used BEFORE this change: measured on 2026-11-13 by running the
code of b496712..833f2bc on each row (pass worker or main, then QwenPipeline's built parts; no model loaded).

No download and no model load: the downloads are stand-ins; the main() probe runs in a child process with -X utf8.
"""
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from whisperjav.config import chronosjav_timing as ct
from whisperjav.crispasr import install
from whisperjav.utils import preflight_check as pf

REPO = Path(__file__).resolve().parents[1]
PT = "Qwen/Qwen3-ForcedAligner-0.6B"
BASE = {"aligner_backend": "crispasr", "aligner_id": "wav2vec2-ja", "timing_model": "none",
        "timestamp_mode": "vad_only", "stepdown": False, "assembly_cleaner": "qwen3", "framer": "vad-grouped"}

# What the GUI sends as --pass1-qwen-params for each row (webview_gui/api.py _build_twopass_args), and what the
# pass used before 1.10.2's change: (timestamp mode, aligner backend, aligner, timing model).
GUI_ROWS = {
    "G1 Qwen3-ASR, not customized": ({"framer": "vad-grouped"}, ("vad_only", None, None, None)),
    "G2 anime-whisper, not customized": ({"framer": "vad-grouped", "generator_backend": "anime-whisper",
                                          "assembly_cleaner": "passthrough"}, ("vad_only", None, None, None)),
    "G3 customized, Alignment at its defaults": (dict(BASE), ("vad_only", None, None, None)),
    "G4 aligner mode, wav2vec2-ja": ({**BASE, "timestamp_mode": "aligner_vad_fallback"},
                                     ("aligner_vad_fallback", "crispasr", "wav2vec2-ja", None)),
    "G5 aligner mode, qwen3-fa-gguf": ({**BASE, "timestamp_mode": "aligner_vad_fallback",
                                       "aligner_id": "qwen3-fa-gguf"},
                                      ("aligner_vad_fallback", "crispasr", "qwen3-fa-gguf", None)),
    "G6 aligner mode, PyTorch aligner": ({**BASE, "timestamp_mode": "aligner_vad_fallback", "aligner_backend": "qwen3",
                                         "aligner_id": PT}, ("aligner_vad_fallback", "qwen3", PT, None)),
    "G7 VAD only + timing model": ({**BASE, "timing_model": "whisper-small"},
                                   ("vad_only", None, None, "whisper-small")),
    "G8 aligner mode + timing model": ({**BASE, "timestamp_mode": "aligner_vad_fallback",
                                       "timing_model": "whisper-small"},
                                      ("aligner_vad_fallback", "crispasr", "wav2vec2-ja", "whisper-small")),
    "G9 aligner None + timing model": ({**BASE, "aligner_backend": "none", "timing_model": "whisper-small"},
                                       ("vad_only", None, None, "whisper-small")),
    "G10 anime-whisper, aligner mode + timing model": ({**BASE, "timestamp_mode": "aligner_vad_fallback",
                                                       "timing_model": "whisper-small",
                                                       "generator_backend": "anime-whisper"},
                                                      ("aligner_vad_fallback", "crispasr", "wav2vec2-ja",
                                                       "whisper-small")),
    "G11 anime-whisper, PyTorch backend with a CrispASR name": ({**BASE, "timestamp_mode": "aligner_only",
                                                                "aligner_backend": "qwen3",
                                                                "generator_backend": "anime-whisper"},
                                                               ("aligner_only", "qwen3", PT, None)),
}
COHERE_ROW = {"framer": "vad-grouped", "generator_backend": "cohere", "timestamp_mode": "aligner_vad_fallback",
              "assembly_cleaner": "passthrough"}   # the dormant Cohere-in-ChronosJAV, as the GUI would send it
SINGLE_ROWS = {
    "S1 --mode qwen": ([], ("vad_only", None, None, None)),
    "S2 anime-whisper": (["--qwen-generator", "anime-whisper"], ("vad_only", None, None, None)),
    "S3 --qwen-aligner qwen3-fa-gguf": (["--qwen-aligner", "qwen3-fa-gguf"],
                                        ("aligner_vad_fallback", "crispasr", "qwen3-fa-gguf", None)),
    "S4 aligner named, vad_only": (["--qwen-aligner", "wav2vec2-ja", "--qwen-timestamp-mode", "vad_only"],
                                   ("vad_only", None, None, None)),
    "S5 aligner mode only": (["--qwen-timestamp-mode", "aligner_vad_fallback"],
                             ("aligner_vad_fallback", "crispasr", "wav2vec2-ja", None)),
    "S6 timing model only": (["--qwen-timing-model", "whisper-small"], ("vad_only", None, None, "whisper-small")),
    "S7 PyTorch aligner by ID": (["--qwen-aligner", PT], ("aligner_vad_fallback", "qwen3", PT, None)),
    "S8 aligner + timing model": (["--qwen-aligner", "wav2vec2-ja", "--qwen-timing-model", "whisper-small"],
                                  ("aligner_vad_fallback", "crispasr", "wav2vec2-ja", "whisper-small")),
    "S9 aligner mode, --qwen-timestamps none": (["--qwen-timestamp-mode", "aligner_only", "--qwen-timestamps", "none"],
                                                ("aligner_only", None, None, None)),
    "S10 --qwen-aligner none": (["--qwen-aligner", "none"], ("vad_only", None, None, None)),
}


def _files_tuple(files):
    return (files.timestamp_mode, files.aligner_backend, files.aligner, files.timing_model)


def _built_tuple(pipeline, aligner_id):
    """What a built QwenPipeline uses, in the same four places (no model is loaded)."""
    from whisperjav.modules.subtitle_pipeline.aligners.crispasr import CrispASRAligner
    sub = pipeline._subtitle_pipeline
    if sub.aligner is None:
        backend, aligner = None, None
    elif isinstance(sub.aligner, CrispASRAligner):
        backend, aligner = "crispasr", sub.aligner._config["model"]
    else:
        backend, aligner = "qwen3", aligner_id
    timer = None if sub.timer is None else sub.timer._config["model"]
    return (pipeline.timestamp_mode.value, backend, aligner, timer)


# --- the GUI sends what the rows say ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(GUI_ROWS))
def test_the_rows_are_what_the_gui_sends(tmp_path, name):
    from whisperjav.webview_gui.api import WhisperJAVAPI
    params, _expected = GUI_ROWS[name]
    pipe = "anime-whisper" if params.get("generator_backend") == "anime-whisper" else "qwen"
    customized = "aligner_backend" in params
    sent_params = {k: v for k, v in params.items() if k not in ("framer", "generator_backend")}
    if pipe == "anime-whisper" and not customized:
        sent_params.pop("assembly_cleaner", None)
    inp = tmp_path / "a.mp4"
    inp.write_bytes(b"x")
    args = WhisperJAVAPI()._build_twopass_args({
        "inputs": [str(inp)],
        "pass1": {"pipeline": pipe, "isQwen": True, "isAnimeWhisper": pipe == "anime-whisper", "isCohere": False,
                  "customized": customized, "params": sent_params, "framer": "vad-grouped", "sensitivity": "balanced"},
        "pass2": {"enabled": False},
    })
    assert json.loads(args[args.index("--pass1-qwen-params") + 1]) == params


# --- the one place gives the values each row used before ------------------------------------------------------

@pytest.mark.parametrize("name", list(GUI_ROWS))
def test_an_ensemble_pass_uses_what_it_used_before(name):
    params, expected = GUI_ROWS[name]
    generator = ct.generator_for_pass(params)
    timing = ct.pass_timing(ct.pass_start_values(params), params, generator)
    assert _files_tuple(ct.chronosjav_files(**timing)) == expected


class _Built(Exception):
    pass


@pytest.mark.parametrize("name", list(GUI_ROWS) + ["cohere"])
def test_the_pass_worker_and_the_pipeline_agree_with_the_one_place(monkeypatch, tmp_path, name):
    """The pass worker's settings for the row, built into a real QwenPipeline: the parts it makes are the files the
    one place names (and, but for the dormant Cohere, the values of before)."""
    import whisperjav.ensemble.pass_worker as pw
    from whisperjav.pipelines.qwen_pipeline import QwenPipeline

    params, expected = GUI_ROWS[name] if name in GUI_ROWS else (COHERE_ROW, None)
    seen = {}

    def recorder(**kwargs):
        seen.update(kwargs)
        raise _Built()

    monkeypatch.setattr(pw, "QwenPipeline", recorder)
    monkeypatch.setitem(pw.PIPELINE_CLASSES, "qwen", recorder)
    config = {"pipeline": "qwen", "sensitivity": "balanced", "scene_detector": None, "speech_segmenter": None,
              "speech_enhancer": None, "model": None, "params": None, "qwen_params": params, "language": "ja",
              "device": None}
    with pytest.raises(_Built):
        pw._build_pipeline(config, 1, str(tmp_path / "out"), False, "native", {}, tmp_path / "pass")
    built = QwenPipeline(**{**seen, "output_dir": str(tmp_path / "o"), "temp_dir": str(tmp_path / "t")})
    files = ct.chronosjav_files(**{k: seen[k] for k in ("timestamps", "aligner_id", "aligner_backend",
                                                        "timing_model", "timestamp_mode")})
    assert _built_tuple(built, seen["aligner_id"]) == _files_tuple(files)
    if expected is not None:
        assert _files_tuple(files) == expected


@pytest.mark.parametrize("qwen_params", [p for p, _e in GUI_ROWS.values()] + [
    COHERE_ROW, {}, {"qwen_aligner": "qwen3-fa-gguf", "aligner_id": "wav2vec2-ja"},
    {"aligner_id": "wav2vec2-ja", "qwen_aligner": "qwen3-fa-gguf"}, {"use_aligner": False},
    {"use_aligner": False, "aligner_backend": "qwen3"}, {"aligner_backend": "qwen3"},
    {"aligner_backend": "crispasr", "aligner_id": PT}, {"qwen_timing_model": "whisper-small"},
    {"qwen_timestamps": "none", "timestamp_mode": "aligner_only"}, {"qwen_aligner_backend": "bogus"},
])
def test_main_reads_the_pass_workers_values(qwen_params):
    """main.py has no pass worker to ask (it loads the ASR modules); pass_start_values gives the same values."""
    from whisperjav.ensemble.pass_worker import prepare_qwen_params
    worker = prepare_qwen_params({"qwen_params": qwen_params})
    mine = ct.pass_start_values(qwen_params)
    for key in ct.TIMING_KEYS.values():
        assert mine.get(key) == worker.get(key), key
    if "qwen_stepdown" in mine:   # set only when the aligner is turned off (the triple flip)
        assert mine["qwen_stepdown"] == worker["qwen_stepdown"]
    assert ct.generator_for_pass(qwen_params) == worker["qwen_generator_backend"]


def _args(monkeypatch, tmp_path, argv):
    from whisperjav import main as wj_main
    media = tmp_path / "a.mp4"
    media.write_bytes(b"")
    monkeypatch.setattr("sys.argv", ["whisperjav", str(media), "--output-dir", str(tmp_path / "out"), *argv])
    return wj_main.parse_arguments()


@pytest.mark.parametrize("name", list(SINGLE_ROWS))
def test_mode_qwen_uses_what_it_used_before(monkeypatch, tmp_path, name):
    from whisperjav import main as wj_main
    from whisperjav.pipelines.qwen_pipeline import QwenPipeline
    extra, expected = SINGLE_ROWS[name]
    args = _args(monkeypatch, tmp_path, ["--mode", "qwen", *extra])
    [(n, label, files)] = wj_main.chronosjav_runs(args)
    assert (n, label) == (None, "ChronosJAV")
    assert _files_tuple(files) == expected
    timing = wj_main._chronosjav_single_pass_timing(args)
    built = QwenPipeline(output_dir=str(tmp_path / "o"), temp_dir=str(tmp_path / "t"), **timing)
    assert _built_tuple(built, timing["aligner_id"]) == expected


def test_main_names_each_chronosjav_pass_and_leaves_out_cohere(monkeypatch, tmp_path):
    from whisperjav import main as wj_main
    args = _args(monkeypatch, tmp_path, [
        "--ensemble", "--pass1-pipeline", "qwen", "--pass1-qwen-params", json.dumps(GUI_ROWS["G6 aligner mode, "
                                                                                             "PyTorch aligner"][0]),
        "--pass2-pipeline", "qwen", "--pass2-qwen-params", json.dumps(COHERE_ROW)])
    runs = wj_main.chronosjav_runs(args)
    assert [(n, label, _files_tuple(f)) for n, label, f in runs] == [
        (1, "ChronosJAV (pass 1)", ("aligner_vad_fallback", "qwen3", PT, None))]
    assert wj_main.chronosjav_runs(_args(monkeypatch, tmp_path, ["--mode", "balanced"])) == []


def test_decoupled_runs_instead_of_mode_qwen_so_no_chronosjav_files(monkeypatch, tmp_path):
    """--pipeline decoupled runs instead of --mode's pipeline (process_files_sync): nothing of ChronosJAV's to
    download or show (review, 2026-11-13; the decoupled pipeline's own aligner is not covered, his word)."""
    from whisperjav import main as wj_main
    args = _args(monkeypatch, tmp_path, ["--mode", "qwen", "--qwen-aligner", "wav2vec2-ja", "--pipeline", "decoupled"])
    assert wj_main.chronosjav_runs(args) == []


@pytest.mark.parametrize("backend, aligner", [("crispasr", "wav2vec2-ja"), ("qwen3", PT)])
def test_a_backend_named_with_no_model_takes_its_default_everywhere(monkeypatch, tmp_path, backend, aligner):
    """Review 2026-11-13: {"qwen_aligner_backend": ..., "qwen_aligner": null} named no file at the start check while
    the pass loaded the CrispASR default during the run (or, for the PyTorch backend, failed to load with no ID)."""
    import whisperjav.ensemble.pass_worker as pw
    from whisperjav import main as wj_main
    from whisperjav.pipelines.qwen_pipeline import QwenPipeline

    params = {"qwen_aligner_backend": backend, "qwen_aligner": None, "timestamp_mode": "aligner_vad_fallback"}
    args = _args(monkeypatch, tmp_path, ["--ensemble", "--pass1-pipeline", "qwen", "--pass1-qwen-params",
                                         json.dumps(params)])
    [(_n, _label, files)] = wj_main.chronosjav_runs(args)
    assert _files_tuple(files) == ("aligner_vad_fallback", backend, aligner, None)
    seen = {}

    def recorder(**kwargs):
        seen.update(kwargs)
        raise _Built()

    monkeypatch.setattr(pw, "QwenPipeline", recorder)
    monkeypatch.setitem(pw.PIPELINE_CLASSES, "qwen", recorder)
    config = {"pipeline": "qwen", "sensitivity": "balanced", "scene_detector": None, "speech_segmenter": None,
              "speech_enhancer": None, "model": None, "params": None, "qwen_params": params, "language": "ja",
              "device": None}
    with pytest.raises(_Built):
        pw._build_pipeline(config, 1, str(tmp_path / "out"), False, "native", {}, tmp_path / "pass")
    built = QwenPipeline(**{**seen, "output_dir": str(tmp_path / "o"), "temp_dir": str(tmp_path / "t")})
    sub = built._subtitle_pipeline
    model = sub.aligner._config.get("model") or sub.aligner._config.get("aligner_id")
    assert model == aligner


def test_timestamp_modes_are_the_pipelines():
    from whisperjav.modules.subtitle_pipeline.types import TimestampMode
    assert set(ct.TIMESTAMP_MODES) == {m.value for m in TimestampMode}
    assert ct.chronosjav_files(timestamps="word", aligner_id="wav2vec2-ja", aligner_backend=None,
                               timing_model="none", timestamp_mode="nonsense").timestamp_mode == "aligner_vad_fallback"


# --- the start-of-run check downloads them ---------------------------------------------------------------------

@pytest.fixture
def downloads(monkeypatch):
    state = type("S", (), {})()
    state.asked, state.fail_on, state.offline = [], None, False
    state.runtime, state.runtime_fail = [], None   # the CrispASR checks asked (part, device); a reason makes them fail

    from whisperjav.crispasr.aligner import CrispAligner
    from whisperjav.crispasr.timer import CrispTimer

    def check_runtime(part):
        def check(device=None):
            state.runtime.append((part, device))
            if state.runtime_fail:
                raise install.CrispASRSetupError(state.runtime_fail)
            return None, None
        return staticmethod(check)
    monkeypatch.setattr(CrispAligner, "check_runtime", check_runtime("aligner"))
    monkeypatch.setattr(CrispTimer, "check_runtime", check_runtime("timer"))

    def ensure_model(model, offline=None):
        state.asked.append(model.name)
        if state.fail_on == model.name:
            raise install.CrispASRSetupError(f"Downloading the model {model.label} failed: stand-in refused")
        return Path("model.gguf")

    def snapshot_download(repo_id, allow_patterns=None, **kwargs):
        state.asked.append(f"download {repo_id}")
        state.patterns = allow_patterns
        if state.fail_on == repo_id:
            raise OSError("stand-in connection refused\nsecond line")
        return "folder"

    state.cache = {}   # model ID -> path of its cached config.json (the stand-in Hugging Face cache)

    def try_to_load_from_cache(repo_id, filename, **kwargs):
        assert filename == "config.json"
        return state.cache.get(repo_id)

    import huggingface_hub
    monkeypatch.setattr(install, "ensure_model", ensure_model)
    monkeypatch.setattr(install, "hub_offline", lambda: state.offline)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", try_to_load_from_cache)
    return state


def _files(mode="aligner_vad_fallback", backend="crispasr", aligner="wav2vec2-ja", timing="whisper-small"):
    return ct.ChronosFiles(mode, backend, aligner, timing)


def _box_text(out):
    box = [line for line in re.sub(r"\x1b\[[0-9;]*m", "", out).splitlines() if line.startswith(("║", "╔", "╚"))]
    assert {len(line) for line in box} == {pf.BOX_WIDTH}   # the frame holds
    return " ".join(" ".join(line.strip("║ ") for line in box).split())


def test_the_crispasr_aligner_and_the_timing_model_are_downloaded(downloads, capsys):
    assert pf.ensure_chronosjav_ready([("ChronosJAV (pass 2)", _files())]) is True
    assert downloads.asked == ["wav2vec2-ja", "whisper-small"]
    assert capsys.readouterr().out == ""


def test_the_pytorch_aligner_is_fetched_into_the_hugging_face_cache(downloads):
    pf.ensure_chronosjav_ready([("ChronosJAV", _files(backend="qwen3", aligner=PT, timing=None))])
    assert downloads.asked == [f"download {PT}"]
    assert downloads.patterns == pf.PYTORCH_ALIGNER_FILES   # the files its load reads, not README.md and the like


def _cached(downloads, tmp_path, *names):
    folder = tmp_path / "snapshot"
    folder.mkdir()
    for name in ("config.json", *names):
        (folder / name).write_bytes(b"x")
    downloads.cache[PT] = str(folder / "config.json")


@pytest.mark.parametrize("offline", [False, True])
def test_a_pytorch_aligner_in_the_cache_is_used_without_asking_the_hub(downloads, tmp_path, offline):
    """Review 2026-11-13, shown on the developer's PC: with --offline, snapshot_download refused a complete model
    because its stored file list named README.md and .gitattributes, which the load never reads."""
    downloads.offline = offline
    _cached(downloads, tmp_path, "model.safetensors")
    assert pf.ensure_chronosjav_ready([("ChronosJAV", _files(backend="qwen3", aligner=PT, timing=None))]) is True
    assert downloads.asked == []


def test_a_cached_config_without_its_weights_is_not_enough(downloads, tmp_path):
    _cached(downloads, tmp_path)
    pf.ensure_chronosjav_ready([("ChronosJAV", _files(backend="qwen3", aligner=PT, timing=None))])
    assert downloads.asked == [f"download {PT}"]


def test_a_local_folder_aligner_is_taken_as_it_is(downloads, tmp_path):
    pf.ensure_chronosjav_ready([("ChronosJAV", _files(backend="qwen3", aligner=str(tmp_path), timing=None))])
    assert downloads.asked == []


def test_a_name_not_offered_is_left_to_the_pipeline(downloads):
    assert pf.ensure_chronosjav_ready([("ChronosJAV", _files(aligner="nope", timing="nope"))]) is True
    assert downloads.asked == []


def test_a_failed_download_stops_the_run_with_the_agreed_box(downloads, capsys):
    downloads.fail_on = "whisper-small"
    with pytest.raises(SystemExit) as stop:
        pf.ensure_chronosjav_ready([("ChronosJAV (pass 1)", _files(mode="vad_only", backend=None, aligner=None)),
                                    ("ChronosJAV (pass 2)", _files())])
    assert stop.value.code == 1
    text = _box_text(capsys.readouterr().out)
    assert "A model this run needs could not be downloaded" in text
    # pass 1 (its timing model) is the one that failed: named alone, "this model"
    assert ("This run uses ChronosJAV (pass 1) with the timing model Whisper small. WhisperJAV downloads this model "
            "the first time it is needed. That did not work:") in text
    assert "Downloading the model Whisper small failed: stand-in refused" in text
    assert "Nothing has been transcribed." in text and "--offline" not in text
    assert downloads.asked == ["whisper-small"]   # pass 2 is not reached


def test_both_files_are_named_and_the_offline_lines_follow(downloads, capsys):
    downloads.fail_on, downloads.offline = PT, True
    with pytest.raises(SystemExit):
        pf.ensure_chronosjav_ready([("ChronosJAV (pass 2)", _files(backend="qwen3", aligner=PT))])
    text = _box_text(capsys.readouterr().out)
    assert (f"This run uses ChronosJAV (pass 2) with the aligner {PT} and the timing model Whisper small. "
            "WhisperJAV downloads these models the first time they are needed. That did not work:") in text
    assert f"The model {PT} is not downloaded yet, and --offline allows no download." in text
    assert ("Downloads are switched off for this run (--offline), so nothing can be fetched now. Run once without "
            "--offline first.") in text


def test_without_exit_on_fail_it_returns_false(downloads):
    downloads.fail_on = "wav2vec2-ja"
    assert pf.ensure_chronosjav_ready([("ChronosJAV", _files())], exit_on_fail=False) is False


# --- CrispASR must be able to run them (owner W632 2) ------------------------------------------------------------

NO_AVX2 = "this processor lacks AVX2, which CrispASR needs; the CrispASR pipelines cannot run here"


def test_crispasr_is_checked_with_the_pass_device_before_any_download(downloads):
    assert pf.ensure_chronosjav_ready([("ChronosJAV (pass 2)", _files())],
                                      devices={"ChronosJAV (pass 2)": "cpu"}) is True
    assert downloads.runtime == [("aligner", "cpu"), ("timer", "cpu")]
    assert downloads.asked == ["wav2vec2-ja", "whisper-small"]


def test_crispasr_that_cannot_run_stops_the_run_with_its_box_and_downloads_nothing(downloads, capsys):
    downloads.runtime_fail = NO_AVX2
    with pytest.raises(SystemExit) as stop:
        pf.ensure_chronosjav_ready([("ChronosJAV (pass 2)", _files())])
    assert stop.value.code == 1
    assert downloads.asked == []
    text = _box_text(capsys.readouterr().out)
    assert ("CrispASR is not ready for this run This run uses ChronosJAV (pass 2) with the aligner "
            "wav2vec2-ja and the timing model Whisper small, run by CrispASR. WhisperJAV checks CrispASR "
            "before any audio is read. That did not work: " + NO_AVX2 + " Nothing has been transcribed.") in text
    assert "--offline" not in text   # the download box's offline lines do not belong here


def test_the_crispasr_box_names_only_what_crispasr_runs(downloads, capsys):
    # The PyTorch aligner is not CrispASR's: only the timing model is named, and the PyTorch aligner is not fetched.
    downloads.runtime_fail = NO_AVX2
    with pytest.raises(SystemExit):
        pf.ensure_chronosjav_ready([("ChronosJAV", _files(backend="qwen3", aligner=PT))])
    assert downloads.runtime == [("timer", None)]
    assert downloads.asked == []
    text = _box_text(capsys.readouterr().out)
    assert "This run uses ChronosJAV with the timing model Whisper small, run by CrispASR." in text
    assert PT not in text


def test_a_pytorch_aligner_alone_asks_nothing_of_crispasr(downloads):
    downloads.runtime_fail = NO_AVX2   # would stop the run if CrispASR were asked
    assert pf.ensure_chronosjav_ready([("ChronosJAV", _files(backend="qwen3", aligner=PT, timing=None))]) is True
    assert downloads.runtime == []


def test_the_loads_run_the_same_check(monkeypatch):
    """CrispAligner.load and CrispTimer.load run check_runtime first: the start check and the run check alike."""
    from whisperjav.crispasr.aligner import CrispAligner
    from whisperjav.crispasr.timer import CrispTimer

    def refuse(device=None):
        raise install.CrispASRSetupError(f"refused for {device}")
    for cls in (CrispAligner, CrispTimer):
        monkeypatch.setattr(cls, "check_runtime", staticmethod(refuse))
        with pytest.raises(install.CrispASRSetupError, match="refused for cuda"):
            cls(device="cuda").load()


@pytest.mark.parametrize("argv, n, device", [
    (["--mode", "qwen"], None, "auto"),
    (["--mode", "qwen", "--device", "cpu"], None, "cpu"),
    (["--ensemble", "--pass1-pipeline", "qwen"], 1, "auto"),
    (["--ensemble", "--pass1-pipeline", "qwen", "--pass1-qwen-params", '{"device": "cuda"}'], 1, "cuda"),
    (["--ensemble", "--pass1-pipeline", "qwen", "--device", "cpu", "--pass1-qwen-params", '{"device": "cuda"}'],
     1, "cpu"),                                                             # --device cpu is absolute
    (["--ensemble", "--pass1-pipeline", "qwen", "--pass1-qwen-params", '{"device": "cpu", "qwen_device": "cuda"}'],
     1, "cuda"),                                                            # the later key wins, as the pass worker
], ids=["mode qwen", "mode qwen --device cpu", "pass", "pass device cuda", "--device cpu wins", "later key"])
def test_each_pass_hands_the_check_the_device_the_pipeline_gets(monkeypatch, tmp_path, argv, n, device):
    from whisperjav import main as wj_main
    from whisperjav.ensemble.pass_worker import prepare_qwen_params
    from whisperjav.utils.device_detector import pass_device
    args = _args(monkeypatch, tmp_path, argv)
    assert wj_main.chronosjav_device(args, n) == device
    if n is not None:   # what the pass worker hands QwenPipeline (pass_worker._build_pipeline)
        worker = prepare_qwen_params({"qwen_params": json.loads(getattr(args, f"pass{n}_qwen_params") or "{}")})
        assert pass_device(args.device, worker["qwen_device"]) == device


# --- main() hands the files to the check before any audio, and --dump-params shows them -------------------------

STARTUP_PROBE = textwrap.dedent("""
    import json, sys
    from pathlib import Path

    root = Path(sys.argv[1])
    extra = json.loads(sys.argv[2])
    media = root / "media" / "a.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"x")
    sys.argv = ["whisperjav", str(media), "--output-dir", str(root / "out"), "--temp-dir", str(root / "tmp"),
                "--accept-cpu-mode", "--no-signature", "--no-progress"] + extra

    import whisperjav.main as M

    asked = {}

    def chronosjav_ready(runs, **kwargs):
        asked["runs"] = [[label, files.as_dict()] for label, files in runs]
        asked["devices"] = kwargs.get("devices")
        raise SystemExit(1)

    M.ensure_segmenter_backend_available = lambda backend, **kwargs: True
    M.ensure_segmenter_model_available = lambda backend, **kwargs: True
    M.ensure_chronosjav_ready = chronosjav_ready
    M.process_files_sync = lambda *a, **k: asked.setdefault("audio_read", True)
    try:
        M.main()
        code = 0
    except SystemExit as e:
        code = e.code
    print("RESULT=" + json.dumps({"exit": code, "asked": asked}))
""")


def _probe(tmp_path, extra, script=STARTUP_PROBE):
    done = subprocess.run([sys.executable, "-X", "utf8", "-c", script, str(tmp_path), json.dumps(extra)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                          cwd=str(REPO))
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT=")]
    assert lines, (done.stdout + done.stderr)[-2000:]
    return json.loads(lines[-1][len("RESULT="):])


def _as_dict(t):
    return dict(zip(("timestamp_mode", "aligner_backend", "aligner", "timing_model"), t))


@pytest.mark.slow
@pytest.mark.parametrize("extra, runs", [
    (["--mode", "qwen"], []),                                                     # the default: no files
    (["--mode", "qwen", "--qwen-aligner", "wav2vec2-ja", "--qwen-timing-model", "whisper-small"],
     [["ChronosJAV", _as_dict(SINGLE_ROWS["S8 aligner + timing model"][1])]]),
    (["--ensemble", "--pass1-pipeline", "qwen", "--pass1-qwen-params",
      json.dumps(GUI_ROWS["G6 aligner mode, PyTorch aligner"][0]), "--pass2-pipeline", "qwen",
      "--pass2-qwen-params", json.dumps(GUI_ROWS["G7 VAD only + timing model"][0])],
     [["ChronosJAV (pass 1)", _as_dict(GUI_ROWS["G6 aligner mode, PyTorch aligner"][1])],
      ["ChronosJAV (pass 2)", _as_dict(GUI_ROWS["G7 VAD only + timing model"][1])]]),
    (["--ensemble", "--pass1-pipeline", "qwen", "--pass1-qwen-params", json.dumps(COHERE_ROW)], []),
], ids=["mode qwen default", "mode qwen aligner and timing model", "two ChronosJAV passes", "cohere left out"])
def test_main_hands_the_files_to_the_start_check_before_any_audio(tmp_path, extra, runs):
    result = _probe(tmp_path, extra)
    assert result["asked"]["runs"] == runs
    assert result["exit"] == 1 and "audio_read" not in result["asked"]   # the stand-in stopped the run first
    # each run's device, for the CrispASR checks (owner W632 2): --device not given, so auto
    assert result["asked"]["devices"] == {label: "auto" for label, _files in runs}


@pytest.mark.slow
def test_dump_params_shows_the_files(tmp_path):
    media = tmp_path / "a.mp4"
    media.write_bytes(b"x")

    def dump(extra):
        out = tmp_path / f"dump{len(list(tmp_path.glob('dump*')))}.json"
        done = subprocess.run([sys.executable, "-X", "utf8", "-m", "whisperjav.main", str(media), *extra,
                               "--dump-params", str(out)], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=300, cwd=str(REPO))
        assert done.returncode == 0, (done.stdout + done.stderr)[-2000:]
        return json.loads(out.read_text(encoding="utf-8"))

    single = dump(["--mode", "qwen", "--qwen-aligner", PT])
    assert single["chronosjav_effective"] == _as_dict(SINGLE_ROWS["S7 PyTorch aligner by ID"][1])
    assert dump(["--mode", "qwen"])["chronosjav_effective"] == _as_dict(SINGLE_ROWS["S1 --mode qwen"][1])
    ens = dump(["--ensemble", "--pass1-pipeline", "qwen", "--pass1-qwen-params",
                json.dumps(GUI_ROWS["G8 aligner mode + timing model"][0]), "--pass2-pipeline", "qwen",
                "--pass2-qwen-params", json.dumps(COHERE_ROW)])["ensemble_config"]
    assert ens["pass1"]["chronosjav_effective"] == _as_dict(GUI_ROWS["G8 aligner mode + timing model"][1])
    assert "chronosjav_effective" not in ens["pass2"]                      # the dormant Cohere (W626 2)


# --- the GUI's status panel reads the one place (owner W632 4) -------------------------------------------------

# What app.js gives the panel for a ChronosJAV pass: QwenManager.defaults, over which a customized pass's own values.
JS_DEFAULTS = {"aligner_backend": "crispasr", "aligner_id": "wav2vec2-ja", "timing_model": "none",
               "timestamp_mode": "vad_only"}
PANEL_KEYS = ("aligner_backend", "aligner_id", "timing_model", "timestamp_mode")


@pytest.fixture
def panel(monkeypatch):
    """The panel with its two outside reads stood in: each part line names the model; no CrispASR runtime asked."""
    from whisperjav.webview_gui.api import WhisperJAVAPI
    monkeypatch.setattr(WhisperJAVAPI, "_crispasr_part_line",
                        classmethod(lambda cls, _install, part, model: f"{part}: {model.name}"))
    monkeypatch.setattr(WhisperJAVAPI, "_with_crispasr_runtime",
                        lambda self, lines: {"success": True, "lines": lines, "can_run": True})
    api = WhisperJAVAPI()

    def parts(values):
        result = api.get_chronosjav_crispasr_status(*(values.get(k) for k in PANEL_KEYS))
        assert result["success"], result
        return [line for line in result["lines"] if line != api.CRISPASR_CHRONOSJAV_HEADING]
    return api, parts


def _expected_parts(api, files_tuple):
    _mode, backend, aligner, timing = files_tuple
    expected = []
    if backend == "crispasr":
        expected.append(f"{api.CRISPASR_PART_ALIGNER}: {aligner}")
    if timing:
        expected.append(f"{api.CRISPASR_PART_TIMING}: {timing}")
    return expected


@pytest.mark.parametrize("name", list(GUI_ROWS))
def test_the_gui_status_panel_names_what_the_run_uses(panel, name):
    """For each GUI row, the panel (given what app.js gives it) names the CrispASR files the run uses."""
    api, parts = panel
    params, expected = GUI_ROWS[name]
    customized = "aligner_backend" in params        # as in test_the_rows_are_what_the_gui_sends
    values = {**JS_DEFAULTS, **({k: params[k] for k in PANEL_KEYS if k in params} if customized else {})}
    assert parts(values) == _expected_parts(api, expected)


def test_the_panel_names_the_aligner_a_named_aligner_brings(panel):
    """An aligner named with no timestamp mode (an older preset, a hand-written pass): the run takes
    aligner_vad_fallback (owner W501 1) and loads the aligner. The panel's own copy took no mode as vad_only."""
    api, parts = panel
    values = {"aligner_backend": "crispasr", "aligner_id": "qwen3-fa-gguf"}
    assert ct.pass_files(values).aligner == "qwen3-fa-gguf"
    assert parts(values) == [f"{api.CRISPASR_PART_ALIGNER}: qwen3-fa-gguf"]


def test_the_panel_names_no_crispasr_aligner_for_a_pytorch_model_id(panel):
    """No backend named and a PyTorch model ID: the run runs the PyTorch aligner, not CrispASR. The panel's own copy
    took no backend as crispasr and named wav2vec2-ja."""
    _api, parts = panel
    values = {"aligner_id": PT, "timestamp_mode": "aligner_vad_fallback"}
    assert ct.pass_files(values).aligner_backend == "qwen3"
    assert parts(values) == []


def test_main_and_the_panel_read_the_same_function():
    """One rule: main's start check (chronosjav_runs) and the panel both call chronosjav_timing.pass_files."""
    import inspect
    from whisperjav import main as wj_main
    from whisperjav.webview_gui.api import WhisperJAVAPI
    assert "chronosjav_pass_files(qwen_params)" in inspect.getsource(wj_main.chronosjav_runs)
    assert "pass_files(named)" in inspect.getsource(WhisperJAVAPI.get_chronosjav_crispasr_status)
