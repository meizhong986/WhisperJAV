"""Item 12, step 5: ChronosJAV gets the CrispASR aligner and the Whisper timing model by default.

Design note docs/plans/design/2026-10-28-chronosjav-timing-c3-c1.md sections 7-8 (owner W254 Q6, W255):
- every ChronosJAV model (Qwen3-ASR, anime-whisper, Cohere) takes the aligner wav2vec2-ja with
  aligner_vad_fallback and step-down, and the timing model whisper-small;
- the PyTorch Qwen3-ForcedAligner stays a choice (its model ID), "none" stays, and choosing "none" in the GUI
  flips the timestamp mode to vad_only and step-down off for all three models;
- the timing model runs without an aligner too (rule T5: it moves the start only).

No model is loaded: QwenPipeline builds its parts without loading them, the pass worker's settings are pure,
and the GUI's argument builder runs nothing.
"""
import json
from pathlib import Path

import pytest

from whisperjav.modules.subtitle_pipeline.aligners.crispasr import CrispASRAligner
from whisperjav.modules.subtitle_pipeline.aligners.qwen3 import Qwen3ForcedAlignerAdapter
from whisperjav.modules.subtitle_pipeline.timers.crispasr import CrispASRTimer
from whisperjav.modules.subtitle_pipeline.types import TimestampMode


def _pipeline(tmp_path, **kwargs):
    from whisperjav.pipelines.qwen_pipeline import QwenPipeline

    return QwenPipeline(output_dir=str(tmp_path / "out"), temp_dir=str(tmp_path / "tmp"), **kwargs)


# --- QwenPipeline ---------------------------------------------------------------------------------

@pytest.mark.parametrize("generator", ["qwen3", "anime-whisper", "cohere"])
def test_every_chronosjav_model_gets_the_aligner_and_the_timing_model_without_step_down(tmp_path, generator):
    p = _pipeline(tmp_path, generator_backend=generator)
    sub = p._subtitle_pipeline
    assert p.timestamp_mode == TimestampMode.ALIGNER_WITH_VAD_FALLBACK
    assert isinstance(sub.aligner, CrispASRAligner) and sub.aligner.display_name == "wav2vec2-ja"
    assert isinstance(sub.timer, CrispASRTimer) and sub.timer.display_name == "Whisper small"
    assert sub.stepdown_config is None          # owner W266 1: off by default
    assert sub.aligner._config["temp_dir"] == p.temp_dir


def test_step_down_can_still_be_turned_on(tmp_path):
    sub = _pipeline(tmp_path, stepdown_enabled=True)._subtitle_pipeline
    assert sub.stepdown_config is not None and sub.stepdown_config.enabled


def test_pass_worker_and_cli_default_step_down_off():
    from whisperjav.ensemble.pass_worker import prepare_qwen_params

    assert prepare_qwen_params({})["qwen_stepdown"] is False


def test_the_pytorch_aligner_is_still_a_choice(tmp_path):
    sub = _pipeline(tmp_path, aligner_id="Qwen/Qwen3-ForcedAligner-0.6B")._subtitle_pipeline
    assert isinstance(sub.aligner, Qwen3ForcedAlignerAdapter)
    assert sub.aligner.display_name == "Qwen/Qwen3-ForcedAligner-0.6B, PyTorch"


@pytest.mark.parametrize("name", ["wav2vec2-ja", "parakeet-ctc-ja", "qwen3-fa-gguf"])
def test_each_crispasr_aligner_by_its_short_name(tmp_path, name):
    sub = _pipeline(tmp_path, aligner_id=name)._subtitle_pipeline
    assert isinstance(sub.aligner, CrispASRAligner) and sub.aligner._config["model"] == name


@pytest.mark.parametrize("kwargs", [{"timestamp_mode": "vad_only"}, {"aligner_backend": "none"},
                                    {"timestamps": "none"}])
def test_no_aligner_but_the_timing_model_stays(tmp_path, kwargs):
    sub = _pipeline(tmp_path, **kwargs)._subtitle_pipeline
    assert sub.aligner is None and sub.stepdown_config is None
    assert isinstance(sub.timer, CrispASRTimer)


def test_the_timing_model_can_be_turned_off(tmp_path):
    sub = _pipeline(tmp_path, timing_model="none")._subtitle_pipeline
    assert sub.timer is None and isinstance(sub.aligner, CrispASRAligner)


def test_cohere_honours_vad_only_from_the_gui_none_choice(tmp_path):
    p = _pipeline(tmp_path, generator_backend="cohere", timestamp_mode="vad_only", aligner_backend="none")
    assert p.timestamp_mode == TimestampMode.VAD_ONLY and p._subtitle_pipeline.aligner is None


def test_an_unknown_crispasr_aligner_name_fails_with_the_choices(tmp_path):
    with pytest.raises(ValueError, match="wav2vec2-ja"):
        _pipeline(tmp_path, aligner_id="nope", aligner_backend="crispasr")


# --- the pass worker --------------------------------------------------------------------------------

def test_pass_worker_defaults():
    from whisperjav.ensemble.pass_worker import prepare_qwen_params

    p = prepare_qwen_params({})
    assert (p["qwen_aligner"], p["qwen_timestamp_mode"], p["qwen_timing_model"]) == \
        ("wav2vec2-ja", "aligner_vad_fallback", "whisper-small")


@pytest.mark.parametrize("qwen_params,aligner,timestamps", [
    ({"aligner_backend": "crispasr", "aligner_id": "qwen3-fa-gguf"}, "qwen3-fa-gguf", "word"),
    ({"aligner_backend": "crispasr", "aligner_id": "Qwen/Qwen3-ForcedAligner-0.6B"}, "wav2vec2-ja", "word"),
    ({"aligner_backend": "qwen3"}, "Qwen/Qwen3-ForcedAligner-0.6B", "word"),
    ({"aligner_backend": "qwen3", "aligner_id": "Qwen/Qwen3-ForcedAligner-0.6B"},
     "Qwen/Qwen3-ForcedAligner-0.6B", "word"),
    ({"aligner_backend": "none"}, None, "none"),
])
def test_pass_worker_aligner_backend(qwen_params, aligner, timestamps):
    from whisperjav.ensemble.pass_worker import prepare_qwen_params

    p = prepare_qwen_params({"qwen_params": qwen_params})
    assert (p["qwen_aligner"], p["qwen_timestamps"]) == (aligner, timestamps)


@pytest.mark.parametrize("qwen_params", [{"aligner_backend": "none"},
                                         {"use_aligner": False, "aligner_backend": "crispasr"},
                                         {"use_aligner": False, "aligner_backend": "qwen3"}])
def test_pass_worker_no_aligner_flips_like_the_gui(qwen_params):
    """Item 12 review: hand-written qwen_params get the GUI's triple flip, and use_aligner false is not undone."""
    from whisperjav.ensemble.pass_worker import prepare_qwen_params

    p = prepare_qwen_params({"qwen_params": qwen_params})
    assert (p["qwen_aligner"], p["qwen_timestamps"], p["qwen_timestamp_mode"], p["qwen_stepdown"]) == \
        (None, "none", "vad_only", False)


def test_pass_worker_no_aligner_keeps_a_mode_the_pass_names():
    from whisperjav.ensemble.pass_worker import prepare_qwen_params

    p = prepare_qwen_params({"qwen_params": {"aligner_backend": "none", "timestamp_mode": "aligner_vad_fallback"}})
    assert p["qwen_timestamp_mode"] == "aligner_vad_fallback" and p["qwen_stepdown"] is False


def test_pass_worker_timing_model_key():
    from whisperjav.ensemble.pass_worker import prepare_qwen_params

    assert prepare_qwen_params({"qwen_params": {"timing_model": "none"}})["qwen_timing_model"] == "none"


def test_pass_worker_no_longer_forces_anime_whisper_to_vad_only():
    import whisperjav.ensemble.pass_worker as pw

    src = Path(pw.__file__).read_text(encoding="utf-8")
    assert 'qwen_pipeline_params["timestamp_mode"] = "vad_only"' not in src
    assert 'qwen_pipeline_params["stepdown_enabled"] = False' not in src
    assert '"timing_model": qwen_defaults.get("qwen_timing_model", "whisper-small")' in src


# --- the GUI ------------------------------------------------------------------------------------------

def _qwen_params(api, tmp_path, pipeline, params):
    inp = tmp_path / "a.mp4"
    inp.write_bytes(b"x")
    args = api._build_twopass_args({
        "inputs": [str(inp)],
        "pass1": {"pipeline": pipeline, "isQwen": True, "isAnimeWhisper": pipeline == "anime-whisper",
                  "isCohere": pipeline == "cohere", "customized": True, "params": dict(params)},
        "pass2": {"enabled": False},
    })
    return json.loads(args[args.index("--pass1-qwen-params") + 1])


@pytest.fixture
def api():
    from whisperjav.webview_gui.api import WhisperJAVAPI

    return WhisperJAVAPI()


@pytest.mark.parametrize("pipeline", ["qwen", "anime-whisper", "cohere"])
def test_gui_none_aligner_flips_all_three_models(api, tmp_path, pipeline):
    sent = _qwen_params(api, tmp_path, pipeline, {"aligner_backend": "none", "timestamp_mode": "aligner_vad_fallback",
                                                  "stepdown": True, "timing_model": "whisper-small"})
    assert (sent["timestamp_mode"], sent["stepdown"], sent["timing_model"]) == ("vad_only", False, "whisper-small")


def test_gui_anime_whisper_is_no_longer_sent_as_vad_only(api, tmp_path):
    sent = _qwen_params(api, tmp_path, "anime-whisper", {"timing_model": "whisper-small"})
    assert "timestamp_mode" not in sent and "stepdown" not in sent
    assert sent["generator_backend"] == "anime-whisper"


def test_gui_alignment_schema(api):
    result = api.get_qwen_schema()
    assert result["success"]
    alignment = result["schema"]["alignment"]
    assert alignment["aligner_backend"]["default"] == "crispasr"
    assert alignment["aligner_id"]["default"] == "wav2vec2-ja"
    assert {o["backend"] for o in alignment["aligner_id"]["options"]} == {"crispasr", "qwen3"}
    assert alignment["timing_model"]["default"] == "whisper-small"
    assert [o["value"] for o in alignment["timing_model"]["options"]] == ["whisper-small", "none"]
    assert alignment["timestamp_mode"]["default"] == "aligner_vad_fallback"


def test_gui_alignment_tab_values_reach_the_run():
    """Before 1.10.0 a ChronosJAV window's Alignment tab (the 'enhancer' panel) was set aside with the
    speech enhancer's settings and never sent; the timing model dropdown lives there."""
    js = (Path(__file__).resolve().parents[1] / "whisperjav" / "webview_gui" / "assets" / "app.js").read_text(
        encoding="utf-8")
    assert "const target = (tab === 'enhancer' && !passState.isQwen) ? enhancerParams : fullParams;" in js
    # ... and a saved preset keeps them too
    assert "if (tab === 'enhancer' && !passState.isQwen) return;" in js
    assert "if (tab === 'enhancer') return;" not in js
    assert "timing_model: 'whisper-small'," in js and "aligner_backend: 'crispasr'," in js


# --- the command line -----------------------------------------------------------------------------------

def test_cli_defaults_and_help():
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-m", "whisperjav.main", "--help"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
    assert done.returncode == 0
    out = " ".join(done.stdout.split())
    assert "--qwen-timing-model {whisper-small,none}" in out
    assert "wav2vec2-ja (default" in out and "parakeet-ctc-ja" in out and "qwen3-fa-gguf" in out
    assert "inert under the vad_only default" not in out
