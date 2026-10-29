#!/usr/bin/env python3
"""
The CrispASR pipelines on the command line (1.10.0; route B).

What is pinned here:
  - the removed 1.9.x "bring your own CrispASR" mode stops the run with exit status 2 and a
    message naming what was typed (owner W118 3, plan 2.5), found in the raw command line;
  - the new Ensemble pass values parakeet-crispasr and cohere-crispasr are in --help, and a pass
    that cannot run (unknown model, no model where there is no default, a language other than
    Japanese, the speech segmenter "none") is a usage error with exit status 2 (plan 2.4, Q14, Q19);
  - a CrispASR pass with no speech segmenter named runs its pipeline's default: WhisperSeg for
    Parakeet-CrispASR, FireRedVAD for Cohere-CrispASR (W202 2-3);
  - the pass worker builds the two pipelines from the pass settings they use (--device included,
    W194 1), and names in one log line the settings they do not use.

whisperjav.main and the pass worker are never imported into this process (CLAUDE.md "Testing"):
the command line runs as `python -m whisperjav.main`, the worker in a probe subprocess. No run
here reaches audio, a download or CrispASR: every CLI run stops while its arguments
are checked, or is --help or --dump-params, and the worker probe replaces the two pipeline classes
with recorders before building them.

Run with: python -m pytest tests/test_crispasr_cli.py -q
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from whisperjav.config.segmenter_presets import CRISPASR_DEFAULT_SEGMENTERS, effective_segmenter_for_pass
from whisperjav.crispasr import manifest
from whisperjav.crispasr.legacy import legacy_crispasr_message, legacy_crispasr_use
from whisperjav.crispasr.pass_settings import (
    crispasr_pass_problem,
    ignored_hardware_message,
    ignored_hardware_settings,
    models_help,
    segmenter_settings,
    unused_pass_settings,
    unused_settings_message,
    validate_crispasr_passes,
)

REPO = Path(__file__).resolve().parents[1]
BANNER = "Japanese Adult Video Subtitle Generator"
# The Parakeet CTC model is not offered (W129 2): never on the command line.
CTC = "parakeet-ctc-1.1b-ja"


def _cli(*args, timeout=300):
    # A wide terminal: argparse wraps help text at hyphens too ("parakeet-" / "crispasr").
    return subprocess.run([sys.executable, "-m", "whisperjav.main", *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, cwd=str(REPO), env={**os.environ, "COLUMNS": "1000"})


def _first_line(text):
    return next((line for line in text.splitlines() if line.strip()), "")


# ===========================================================================
# The old mode, found in the raw command line
# ===========================================================================

class TestTheOldModeIsFound:
    @pytest.mark.parametrize("argv,used", [
        (["--mode", "crispasr", "x.mp4"], "--mode crispasr"),
        (["x.mp4", "--mode=crispasr"], "--mode=crispasr"),
        (["--mode", "CrispASR", "x.mp4"], "--mode CrispASR"),
        (["--ensemble", "--pass1-pipeline", "crispasr", "x.mp4"], "--pass1-pipeline crispasr"),
        (["--ensemble", "--pass1-pipeline=CRISPASR"], "--pass1-pipeline=CRISPASR"),
        (["--ensemble", "--pass2-pipeline", "crispasr"], "--pass2-pipeline crispasr"),
        (["--pass2-pipeline=crispasr", "x.mp4"], "--pass2-pipeline=crispasr"),
        (["--crispasr-exe", r"C:\x.exe", "x.mp4"], "--crispasr-exe"),
        ([r"--crispasr-exe=C:\Program Files\crispasr\crispasr.exe", "x.mp4"], "--crispasr-exe"),
        (["--crispasr-backend", "parakeet"], "--crispasr-backend"),
        (["--crispasr-backend=cohere"], "--crispasr-backend"),
        (["--crispasr-args", "--vad", "x.mp4"], "--crispasr-args"),
        (["--crispasr-args=--vad"], "--crispasr-args"),
        (["--crispasr-e", "x.exe"], "--crispasr-e"),            # an abbreviation argparse would take
        (["--crispasr"], "--crispasr"),
        (["--CrispASR-Exe", "x.exe"], "--CrispASR-Exe"),
    ])
    def test_every_old_form(self, argv, used):
        assert legacy_crispasr_use(argv) == used

    def test_the_first_one_typed_is_named(self):
        assert legacy_crispasr_use(["--crispasr-exe", "x", "--mode", "crispasr"]) == "--crispasr-exe"
        assert legacy_crispasr_use(["--mode", "crispasr", "--crispasr-exe", "x"]) == "--mode crispasr"

    @pytest.mark.parametrize("argv", [
        [],
        ["x.mp4"],
        ["--ensemble", "--pass1-pipeline", "parakeet-crispasr", "x.mp4"],
        ["--ensemble", "--pass2-pipeline", "cohere-crispasr", "--pass2-model", "cohere-asr-ja"],
        ["--pass1-pipeline=parakeet-crispasr"],
        ["crispasr.mp4"],
        ["crispasr"],                                    # a file named crispasr
        ["--mode", "balanced", "crispasr"],              # ... after another flag's value
        ["--pass1-model", "crispasr"],                   # not a flag that chose the old mode
        ["--mode"],                                      # no value at all
        ["x.mp4", "--", "--crispasr-exe", "--mode", "crispasr"],   # after a bare "--": files
        ["--qwen-context", "--crispasr is the old name"],          # a value with a space, not a flag
    ])
    def test_what_is_not_the_old_mode(self, argv):
        assert legacy_crispasr_use(argv) is None

    def test_the_message_names_what_was_typed_and_where_crispasr_went(self):
        message = legacy_crispasr_message("--crispasr-exe")
        assert message.startswith("--crispasr-exe is no longer available")
        assert "Ensemble Mode" in message
        assert "--ensemble --pass1-pipeline parakeet-crispasr" in message
        assert "Parakeet-CrispASR" in message and "Cohere-CrispASR" in message


# ===========================================================================
# Usage checks for a CrispASR pass (light module, no main.py)
# ===========================================================================

class TestAPassThatCannotRunIsRefused:
    def test_parakeet_runs_on_its_default_model(self):
        assert crispasr_pass_problem(1, "parakeet-crispasr", model=None, speech_segmenter=None,
                                     language="japanese") is None

    @pytest.mark.parametrize("pipeline", sorted(manifest.PIPELINE_MODELS))
    def test_every_listed_model_is_accepted(self, pipeline):
        for model in manifest.PIPELINE_MODELS[pipeline]:
            assert crispasr_pass_problem(2, pipeline, model=model, speech_segmenter=None,
                                         language="japanese") is None

    def test_an_unknown_model_is_refused_with_the_choices(self):
        problem = crispasr_pass_problem(1, "parakeet-crispasr", model="nope", speech_segmenter=None,
                                        language="japanese")
        assert problem.startswith("--pass1-model: 'nope' is not a model of parakeet-crispasr")
        for model in manifest.PIPELINE_MODELS["parakeet-crispasr"]:
            assert model in problem

    def test_another_pipelines_model_is_refused(self):
        other = manifest.PIPELINE_MODELS["parakeet-crispasr"][0]
        problem = crispasr_pass_problem(2, "cohere-crispasr", model=other, speech_segmenter=None,
                                        language="japanese")
        assert problem and problem.startswith("--pass2-model:")

    def test_the_ctc_model_is_refused_like_an_unknown_one(self):
        problem = crispasr_pass_problem(1, "parakeet-crispasr", model=CTC, speech_segmenter=None,
                                        language="japanese")
        assert problem == (f"--pass1-model: '{CTC}' is not a model of parakeet-crispasr; "
                           f"choose one of: parakeet-tdt-0.6b-ja")

    @pytest.mark.parametrize("pipeline", manifest.CRISPASR_PIPELINES)
    def test_no_model_named_runs_the_default(self, pipeline):
        # every pipeline has one since step 2b (Cohere: v0.1, W129 1)
        assert crispasr_pass_problem(1, pipeline, model=None, speech_segmenter=None, language="japanese") is None

    @pytest.mark.parametrize("language", ["english", "korean", "chinese"])
    def test_a_language_other_than_japanese_is_refused(self, language):
        problem = crispasr_pass_problem(1, "parakeet-crispasr", model=None, speech_segmenter=None,
                                        language=language)
        assert "transcribes Japanese only" in problem and language in problem

    @pytest.mark.parametrize("segmenter", ["none", "None", " NONE "])
    def test_the_segmenter_none_is_refused(self, segmenter):
        problem = crispasr_pass_problem(2, "parakeet-crispasr", model=None, speech_segmenter=segmenter,
                                        language="japanese")
        assert problem.startswith("--pass2-speech-segmenter none is not available")
        assert CRISPASR_DEFAULT_SEGMENTERS["parakeet-crispasr"] in problem
        assert "it transcribes the VAD groups the speech segmenter finds" in problem

    def test_another_segmenter_is_accepted(self):
        assert crispasr_pass_problem(1, "parakeet-crispasr", model=None, speech_segmenter="ten",
                                     language="japanese") is None

    def test_validate_reads_both_passes_and_ignores_the_others(self):
        def args(**kw):
            base = dict(pass1_pipeline="balanced", pass2_pipeline=None, pass1_model=None, pass2_model=None,
                        pass1_speech_segmenter=None, pass2_speech_segmenter=None, language="japanese")
            return SimpleNamespace(**{**base, **kw})

        validate_crispasr_passes(args())                                      # no CrispASR pass
        validate_crispasr_passes(args(pass1_pipeline="qwen", language="english",
                                      pass1_speech_segmenter="none"))         # not ours to judge
        validate_crispasr_passes(args(pass1_pipeline="parakeet-crispasr"))
        with pytest.raises(ValueError, match="--pass2-model"):
            validate_crispasr_passes(args(pass2_pipeline="parakeet-crispasr", pass2_model="nope"))
        with pytest.raises(ValueError, match=f"--pass1-model: '{CTC}' is not a model of parakeet-crispasr"):
            validate_crispasr_passes(args(pass1_pipeline="parakeet-crispasr", pass1_model=CTC))
        with pytest.raises(ValueError, match="Japanese only"):
            validate_crispasr_passes(args(pass1_pipeline="qwen", pass2_pipeline="parakeet-crispasr",
                                          language="korean"))


class TestTheSegmenterDefault:
    @pytest.mark.parametrize("pipeline, default", [("parakeet-crispasr", "ten"),
                                                   ("cohere-crispasr", "firered-vad")])
    def test_a_crispasr_pass_with_none_named_runs_its_pipelines_default(self, pipeline, default):
        # W344 ("Go, CLI follows too": Parakeet TEN VAD, was W202 2 whisperseg), W202 3 ("fireredvad")
        assert CRISPASR_DEFAULT_SEGMENTERS[pipeline] == default
        assert effective_segmenter_for_pass(pipeline, None) == default
        assert effective_segmenter_for_pass(pipeline, "") == default
        assert effective_segmenter_for_pass(pipeline, "ten") == "ten"


class TestTheSettingsAPassDoesNotUse:
    def test_nothing_is_named_when_nothing_was_set(self):
        assert unused_pass_settings({"pipeline": "parakeet-crispasr", "sensitivity": "balanced"}, 1) == []

    def test_what_the_pipeline_does_anyway_is_not_named(self):
        # The GUI sends these values for every pass.
        config = {"scene_detector": "semantic", "speech_enhancer": "none", "enhance_for_vad": False,
                  "vad_threshold": None, "speech_pad_ms": None, "params": None, "hf_params": None,
                  "qwen_params": None}
        assert unused_pass_settings(config, 1) == []

    def test_every_unused_setting_is_named_in_the_order_of_the_flags(self):
        config = {"scene_detector": "auditok", "speech_enhancer": "ffmpeg-dsp:loudnorm",
                  "enhance_for_vad": True, "vad_threshold": 0.4, "speech_pad_ms": 0,
                  "params": {"x": 1}, "hf_params": {"y": 2}, "qwen_params": {"z": 3}}
        # --pass2-vad-threshold and --pass2-speech-pad-ms are used (W208), and since design 22.9.1
        # (owner W232 8) the scene detector, the speech enhancer, enhance-for-vad and --pass2-params.
        assert unused_pass_settings(config, 2) == ["--pass2-hf-params", "--pass2-qwen-params"]

    def test_the_line(self):
        assert unused_settings_message("Parakeet-CrispASR", ["--pass1-scene-detector", "--pass1-speech-enhancer"]) \
            == "Parakeet-CrispASR does not use: --pass1-scene-detector, --pass1-speech-enhancer"

    @pytest.mark.parametrize("sensitivity", ["aggressive", "balanced", "conservative"])
    def test_sensitivity_is_used_so_never_named(self, sensitivity):
        assert unused_pass_settings({"pipeline": "cohere-crispasr", "sensitivity": sensitivity}, 1) == []


class TestComputeType:
    """--compute-type is not used, with a logged warning; --device is used (W194 1). Left out or auto,
    nothing is said."""

    @pytest.mark.parametrize("config", [{}, {"device": None, "compute_type": None},
                                        {"device": "auto", "compute_type": "AUTO"}, {"device": "cuda"}])
    def test_left_out_or_auto_is_not_named_and_device_never_is(self, config):
        assert ignored_hardware_settings(config) == []

    def test_a_compute_type_given_is_named_with_its_flag(self):
        assert ignored_hardware_settings({"compute_type": "int8"}) == ["--compute-type int8"]
        assert ignored_hardware_settings({"device": "cpu", "compute_type": "float16"}) == ["--compute-type float16"]

    def test_the_warning(self):
        assert ignored_hardware_message("Cohere-CrispASR", ["--compute-type int8"]) == (
            "Cohere-CrispASR ignores --compute-type int8: CrispASR's model files come in one number "
            "format (8-bit).")

    def test_the_8_bit_in_the_warning_is_true_of_every_model_file(self):
        names = {name for models in manifest.PIPELINE_MODELS.values() for name in models}
        assert names and all(manifest.MODELS[name].filename.endswith("-q8_0.gguf") for name in names)


class TestTheModelHelp:
    def test_it_lists_every_model_and_marks_the_defaults(self):
        text = models_help()
        for pipeline, models in manifest.PIPELINE_MODELS.items():
            assert pipeline in text
            default = manifest.PIPELINE_DEFAULT_MODEL[pipeline]
            for model in models:
                assert model + (" (default)" if model == default else "") in text

    def test_it_does_not_list_the_ctc_model(self):
        assert CTC not in models_help()


# ===========================================================================
# The command line itself (subprocess; about 5 s each)
# ===========================================================================

@pytest.mark.slow
class TestTheCommandLine:
    def test_help_lists_the_new_pass_values_and_not_the_old_flags(self):
        result = _cli("--help")
        assert result.returncode == 0, result.stderr[-600:]
        text = " ".join(result.stdout.split())
        assert "parakeet-crispasr" in text and "cohere-crispasr" in text
        for models in manifest.PIPELINE_MODELS.values():
            for model in models:
                assert model in text
        assert CTC not in text   # not offered (W129 2)
        # T1, T2, T24 (route B design 12)
        assert "run through CrispASR, installed with WhisperJAV" in text
        assert ("Experimental in 1.10.0 (early release). "   # owner W212 D4 1
                "The GGUF (CrispASR) pipelines need folder names in plain Western letters (ASCII). When your user "
                "folder has Chinese, Japanese or Korean characters, WhisperJAV links or copies the model to a "
                "plain-named folder (the original stays where it is).") in text
        # T2 (design 22.10, owner W237 1): --device for every pipeline; the GGUF pipelines' builds.
        assert "Where this run's ASR and other models run, for every pipeline." in text
        assert "the GPU through Vulkan on Windows (Vulkan build, experimental)" in text
        assert "ten for parakeet-crispasr and firered-vad for cohere-crispasr" in text   # W344
        assert "CrispASR's own program" not in text
        assert "--crispasr-exe" not in text and "--crispasr-backend" not in text
        assert "--crispasr-args" not in text
        assert "{fidelity,balanced,fast,faster,transformers,qwen}" in text   # --mode has no crispasr

    @pytest.mark.parametrize("argv,used", [
        (["--mode", "crispasr", "x.mp4"], "--mode crispasr"),
        (["--crispasr-exe", r"C:\x.exe", "x.mp4"], "--crispasr-exe"),
        (["--pass2-pipeline=crispasr", "x.mp4"], "--pass2-pipeline=crispasr"),
        (["--crispasr-args", "--vad", "x.mp4"], "--crispasr-args"),
    ])
    def test_the_old_mode_stops_with_exit_2(self, argv, used):
        result = _cli(*argv)
        assert result.returncode == 2, (result.stdout + result.stderr)[-800:]
        assert _first_line(result.stderr) == legacy_crispasr_message(used).splitlines()[0]
        assert legacy_crispasr_message(used) in result.stderr
        assert BANNER not in result.stdout          # stopped before anything else ran

    @pytest.mark.parametrize("extra,expected", [
        (["--pass1-pipeline", "parakeet-crispasr", "--pass1-model", "nope"],
         "error: --pass1-model: 'nope' is not a model of parakeet-crispasr; choose one of: "
         + ", ".join(manifest.PIPELINE_MODELS["parakeet-crispasr"])),
        # not offered (W129 2): not on the command line
        (["--pass1-pipeline", "parakeet-crispasr", "--pass1-model", CTC],
         f"error: --pass1-model: '{CTC}' is not a model of parakeet-crispasr; choose one of: parakeet-tdt-0.6b-ja"),
        (["--pass1-pipeline", "parakeet-crispasr", "--language", "english"],
         "error: --pass1-pipeline parakeet-crispasr transcribes Japanese only"),
        (["--pass1-pipeline", "parakeet-crispasr", "--pass1-speech-segmenter", "none"],
         "error: --pass1-speech-segmenter none is not available"),
    ])
    def test_a_pass_that_cannot_run_is_a_usage_error(self, extra, expected, tmp_path):
        # The input does not exist: had the check not stopped the run, media discovery would
        # have, with exit status 1 and no "error:" line.
        result = _cli(tmp_path / "x.mp4", "--ensemble", "--accept-cpu-mode", *extra)
        assert result.returncode == 2, (result.stdout + result.stderr)[-800:]
        assert expected in result.stderr, result.stderr[-800:]
        assert BANNER not in result.stdout

    @pytest.fixture(scope="class")
    def dumped(self, tmp_path_factory):
        """One --dump-params run with a CrispASR pipeline in each pass."""
        root = tmp_path_factory.mktemp("dump")
        dump = root / "params.json"
        result = _cli(root / "x.mp4", "--ensemble",
                      "--pass1-pipeline", "parakeet-crispasr",
                      "--pass2-pipeline", "cohere-crispasr", "--pass2-model", "cohere-asr-ja",
                      "--pass2-speech-segmenter", "ten", "--pass2-scene-detector", "auditok",
                      "--pass2-sensitivity", "conservative", "--device", "cpu", "--dump-params", dump)
        assert result.returncode == 0, (result.stdout + result.stderr)[-800:]
        return json.loads(dump.read_text(encoding="utf-8"))["ensemble_config"]

    def test_a_valid_crispasr_ensemble_parses_and_dumps(self, dumped):
        ensemble = dumped
        assert ensemble["pass1"]["pipeline"] == "parakeet-crispasr"
        assert ensemble["pass2"]["pipeline"] == "cohere-crispasr"
        assert ensemble["pass2"]["model"] == "cohere-asr-ja"
        assert ensemble["pass1"]["model"] is None                   # the values as typed keep their keys
        assert ensemble["pass2"]["speech_segmenter"] == "ten"
        for p in ("pass1", "pass2"):
            # the 1.9.x keys are gone; the one CrispASR key is what the run will use (below)
            assert [k for k in ensemble[p] if k.startswith("crispasr")] == ["crispasr_effective"], ensemble[p]
            assert not {"crispasr_exe", "crispasr_backend", "crispasr_args"} & set(ensemble[p])
        # The dump imports whisperjav.ensemble (safety caps), which imports the two pipelines.
        assert "_safety_cap_error" not in ensemble, ensemble.get("_safety_cap_error")

    def test_the_dump_shows_what_each_crispasr_pass_will_use(self, dumped):
        # T17: the resolved speech segmenter settings, as the fidelity or the qwen pipeline resolves them
        not_used = ["hf_params", "qwen_params", "compute_type"]   # design 22.9.1
        p1 = dumped["pass1"]["crispasr_effective"]
        assert p1 == {
            "model": manifest.PIPELINE_DEFAULT_MODEL["parakeet-crispasr"],  # none named: the default
            "speech_segmenter": "ten",                                        # none named: Parakeet's default (W344)
            "sensitivity": "aggressive",                                      # none named: Parakeet's (W344)
            "speech_segmenter_settings": segmenter_settings("parakeet-crispasr", "ten", "aggressive").as_dict(),
            "device": "cpu",                                                  # --device as given
            "scene_detector": "semantic",                                     # none named: semantic (W214 2)
            "speech_enhancer": "none",
            "enhance_for_vad": False,
            "not_used": not_used,
        }
        s1 = p1["speech_segmenter_settings"]
        assert (s1["threshold"], s1["max_group_duration_s"], s1["max_speech_duration_s"], s1["chunk_threshold_s"]) \
            == (0.22, 6.0, 6.0, 1.0)                                          # TEN's aggressive preset; 6 s (W202 1)
        p2 = dumped["pass2"]["crispasr_effective"]
        assert (p2["model"], p2["speech_segmenter"], p2["sensitivity"], p2["scene_detector"]) == \
            ("cohere-asr-ja", "ten", "conservative", "auditok")               # --pass2-scene-detector is used (22.9.1)
        s2 = p2["speech_segmenter_settings"]
        assert (s2["max_group_duration_s"], s2["chunk_threshold_s"], s2["max_speech_duration_s"],
                s2["start_pad_ms"], s2["end_pad_ms"]) == (3.0, 0.3, 4.0, 100, 100)  # the qwen pipeline's
        assert s2["threshold"] == 0.42                                        # TEN's conservative preset (W202 4)

    def test_the_dump_shows_the_vad_threshold_and_speech_pad_flags_at_work(self, tmp_path):
        # W208 "make the flags work": Parakeet as fidelity, Cohere as the qwen pipeline.
        dump = tmp_path / "params.json"
        result = _cli(tmp_path / "x.mp4", "--ensemble",
                      "--pass1-pipeline", "parakeet-crispasr", "--pass1-speech-segmenter", "whisperseg",
                      "--pass1-vad-threshold", "0.5", "--pass1-speech-pad-ms", "120",
                      "--pass2-pipeline", "cohere-crispasr",
                      "--pass2-vad-threshold", "1.7", "--pass2-speech-pad-ms", "50",
                      "--device", "cpu", "--dump-params", dump)
        assert result.returncode == 0, (result.stdout + result.stderr)[-800:]
        ensemble = json.loads(dump.read_text(encoding="utf-8"))["ensemble_config"]
        s1 = ensemble["pass1"]["crispasr_effective"]["speech_segmenter_settings"]
        assert (s1["speech_segmenter"], s1["threshold"], s1["speech_pad_ms"], s1["max_group_duration_s"]) \
            == ("whisperseg", 0.5, 120, 6.0)                                  # the 6 s cap still holds
        s2 = ensemble["pass2"]["crispasr_effective"]["speech_segmenter_settings"]
        assert (s2["speech_segmenter"], s2["threshold"], s2["start_pad_ms"], s2["end_pad_ms"]) \
            == ("firered-vad", 1.0, 50, 50)                                   # clamped to 1.0; both pads
        for p in ("pass1", "pass2"):
            assert not {"vad_threshold", "speech_pad_ms"} & set(ensemble[p]["crispasr_effective"]["not_used"])


# ===========================================================================
# The pass worker (probe subprocess; the pipeline classes are replaced by recorders)
# ===========================================================================

WORKER_PROBE = textwrap.dedent("""
    import json, logging, sys
    from pathlib import Path
    import whisperjav.ensemble.pass_worker as pw
    from whisperjav.utils.logger import logger

    root = Path(sys.argv[1])
    out = {"real": {n: pw.PIPELINE_CLASSES[n].__name__ for n in ("parakeet-crispasr", "cohere-crispasr")},
           "names": sorted(pw.PIPELINE_CLASSES),
           "ignored": sorted(pw.ENHANCE_FOR_VAD_IGNORED_BY),
           "builds": []}
    seen, lines = {}, []

    def recorder(label):
        class Recorder:
            LABEL = label
            def __init__(self, **kwargs):
                seen.update(kwargs)
        return Recorder

    for name in ("parakeet-crispasr", "cohere-crispasr"):
        pw.PIPELINE_CLASSES[name] = recorder(pw.PIPELINE_CLASSES[name].LABEL)

    class Collect(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    logger.addHandler(Collect())
    for i, config in enumerate(json.loads(sys.argv[2])):
        seen.clear(); lines.clear()
        try:
            pw._build_pipeline(pass_config=config, pass_number=1, output_dir=str(root / "out"),
                               keep_temp_files=False, subs_language="native",
                               extra_kwargs={"save_metadata_json": True, "model_refresh_audio_minutes": 20.0},
                               pass_temp_dir=root / f"pass1_worker_{i}")
            error = None
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        out["builds"].append({"kwargs": dict(seen), "log": list(lines), "error": error})
    print("RESULT=" + json.dumps(out, default=str))
""")


@pytest.mark.slow
class TestThePassWorkerBuildsThem:
    @pytest.fixture(scope="class")
    def probe(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("worker")
        configs = [
            # A pass with every setting these pipelines do not use. WhisperSeg named, so the speech pad is used
            # (Parakeet's default since W344, TEN VAD, has no speech-pad setting; TestTheSegmenterDefault).
            {"pipeline": "parakeet-crispasr", "sensitivity": "balanced", "model": None,
             "speech_segmenter": "whisperseg", "language": "ja", "scene_detector": "auditok",
             "speech_enhancer": "ffmpeg-dsp", "enhance_for_vad": True, "vad_threshold": 0.4,
             "speech_pad_ms": 100, "params": {"beam_size": 5}, "hf_params": {"x": 1},
             "qwen_params": {"y": 2}},
            # A pass as the GUI would send it, with its own model and segmenter.
            {"pipeline": "cohere-crispasr", "sensitivity": "balanced", "model": "cohere-asr-ja",
             "speech_segmenter": "ten", "language": "ja", "scene_detector": "semantic",
             "speech_enhancer": "none", "enhance_for_vad": False},
            # A hand-written configuration naming the removed pipeline.
            {"pipeline": "crispasr", "language": "ja"},
            # --device and --compute-type given on the command line, which main.py puts in every pass.
            {"pipeline": "parakeet-crispasr", "sensitivity": "aggressive", "model": None,
             "speech_segmenter": "nemo-lite", "language": "ja", "device": "cuda", "compute_type": "int8"},
            # A speech pad for a backend with no speech-pad setting (fidelity ignores it, with a warning).
            {"pipeline": "parakeet-crispasr", "sensitivity": "balanced", "model": None,
             "speech_segmenter": "firered-vad", "language": "ja", "speech_pad_ms": 80},
        ]
        result = subprocess.run([sys.executable, "-c", WORKER_PROBE, str(root), json.dumps(configs)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=300, cwd=str(REPO))
        assert result.returncode == 0, result.stderr[-1500:]
        line = [l for l in result.stdout.splitlines() if l.startswith("RESULT=")][0]
        return root, json.loads(line[len("RESULT="):])

    def test_the_registry_has_the_new_names_and_not_the_old_one(self, probe):
        _root, out = probe
        assert out["real"] == {"parakeet-crispasr": "ParakeetCrispASRPipeline",
                               "cohere-crispasr": "CohereCrispASRPipeline"}
        assert "crispasr" not in out["names"]
        assert not {"parakeet-crispasr", "cohere-crispasr"} & set(out["ignored"])   # they read it (22.9.1)
        assert "crispasr" not in out["ignored"]

    def test_the_pipeline_gets_the_settings_it_uses(self, probe):
        root, out = probe
        built = out["builds"][0]
        assert built["error"] is None
        assert built["kwargs"] == {
            "output_dir": str(root / "out"),
            "temp_dir": str(root / "pass1_worker_0"),
            "keep_temp_files": False,
            "save_metadata_json": True,          # --debug, carried in the worker's extra kwargs
            "progress_display": None,
            "model_name": None,                  # the pipeline's default model
            "speech_segmenter": "whisperseg",    # as named
            "language": "ja",
            "subs_language": "native",
            "sensitivity": "balanced",           # picks the speech segmenter's preset
            "device": None,                      # --device left out (the GUI sends none)
            "vad_threshold": 0.4,                # used since W208
            "speech_pad_ms": 100,                # used since W208 (WhisperSeg has the setting)
            # design 22.9.1 (owner W212 D3, W232 8): scene detector, enhancer, enhance-for-vad, Customize
            "scene_detector": "auditok",
            "enhance_for_vad": True,
            "speech_enhancer": "ffmpeg-dsp",
            "speech_enhancer_model": None,
            "segmenter_overrides": {},           # beam_size is neither a segmenter nor a scene setting
            "scene_params": {},
        }
        assert not [l for l in built["log"] if "ignores --" in l], built["log"]
        assert not [l for l in built["log"] if "speech pad of" in l], built["log"]

    def test_a_speech_pad_the_backend_cannot_take_is_a_warning(self, probe):
        _root, out = probe
        built = out["builds"][4]
        assert built["error"] is None
        assert built["kwargs"]["speech_pad_ms"] == 80
        assert [l for l in built["log"] if "the speech pad of 80ms is ignored -- the 'firered-vad' speech "
                "segmenter has no speech-pad setting" in l], built["log"]

    def test_device_reaches_the_pipeline_and_compute_type_is_a_warning(self, probe):
        _root, out = probe
        built = out["builds"][3]
        assert built["error"] is None
        assert (built["kwargs"]["sensitivity"], built["kwargs"]["speech_segmenter"]) == ("aggressive", "nemo-lite")
        assert built["kwargs"]["device"] == "cuda" and "compute_type" not in built["kwargs"]
        assert ignored_hardware_message("Parakeet-CrispASR", ["--compute-type int8"]) in built["log"]
        assert not [l for l in built["log"] if "--device" in l], built["log"]

    def test_one_line_names_what_it_does_not_use(self, probe):
        _root, out = probe
        log = out["builds"][0]["log"]
        assert "Parakeet-CrispASR does not use: --pass1-hf-params, --pass1-qwen-params" in log
        # The generic enhance-for-vad line would give a wrong reason for these pipelines.
        assert not [l for l in log if "has no effect on the" in l], log

    def test_a_pass_as_the_gui_sends_it(self, probe):
        _root, out = probe
        built = out["builds"][1]
        assert built["error"] is None
        assert built["kwargs"]["model_name"] == "cohere-asr-ja"
        assert built["kwargs"]["speech_segmenter"] == "ten"
        assert built["kwargs"]["sensitivity"] == "balanced"
        assert not [l for l in built["log"] if "does not use" in l or "ignores --" in l], built["log"]

    def test_a_configuration_naming_the_removed_pipeline_fails_to_build(self, probe):
        _root, out = probe
        assert out["builds"][2]["error"] == "ValueError: Unknown pipeline: crispasr"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
