"""1.10.1 (owner W509, "TEN VAD for both"): an Ensemble run on the command line starts Qwen3-ASR as pass 2 on
TEN VAD at balanced, as the GUI Ensemble tab does (assets/app.js applyPipelinePresets, the qwen branch).

Checked through `--dump-params` (no ASR model is loaded): the speech segmenter and sensitivity main.py fills in
for pass 2 (apply_ensemble_pass_defaults). An empty segmenter there means the pass worker's own choice applies
(the JSON's qwen_segmenter, else WhisperSeg); the dump does not show the worker's resolved values.
"""

import json
import subprocess
import sys

import pytest


def _pass2(tmp_path, *extra):
    out = tmp_path / "dump.json"
    cmd = [sys.executable, "-m", "whisperjav.main", "--dump-params", str(out), "--ensemble",
           "--pass1-pipeline", "qwen", "--pass2-pipeline", "qwen", *extra]
    run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    p2 = json.loads(out.read_text(encoding="utf-8"))["ensemble_config"]["pass2"]
    return p2["speech_segmenter"], p2["sensitivity"]


def test_qwen3_as_pass2_gets_ten_at_balanced(tmp_path):
    assert _pass2(tmp_path) == ("ten", "balanced")


def test_anime_whisper_as_pass2_unchanged(tmp_path):
    assert _pass2(tmp_path, "--pass2-qwen-params", json.dumps({"generator_backend": "anime-whisper"})) == (
        "ten", "aggressive")


@pytest.mark.parametrize("extra, expected", [
    (("--pass2-speech-segmenter", "whisperseg"), ("whisperseg", "balanced")),
    (("--pass2-sensitivity", "aggressive"), ("ten", "aggressive")),
])
def test_the_users_choice_wins(tmp_path, extra, expected):
    assert _pass2(tmp_path, *extra) == expected


@pytest.mark.parametrize("key", ["generator_backend", "qwen_generator_backend"])
def test_the_dormant_cohere_pass_is_not_reached(tmp_path, key):
    seg, sens = _pass2(tmp_path, "--pass2-qwen-params", json.dumps({key: "cohere"}))
    assert (seg, sens) == (None, "balanced")   # the pass worker's WhisperSeg, as before


def test_anime_whisper_in_the_other_key_form(tmp_path):
    assert _pass2(tmp_path, "--pass2-qwen-params", json.dumps({"qwen_generator_backend": "anime-whisper"})) == (
        "ten", "aggressive")


def test_a_segmenter_inside_the_pass_json_is_not_overridden(tmp_path):
    seg, _ = _pass2(tmp_path, "--pass2-qwen-params", json.dumps({"qwen_segmenter": "silero-v6.2"}))
    assert seg is None   # the pass worker then applies the JSON's qwen_segmenter
