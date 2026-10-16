"""
v1.10.0 item 3: ClearVoice is no longer included (owner, W73).

A value that still names it -- a saved GUI setting, an Ensemble preset, a script's
command line or pass parameters -- stops the run with text M1 (owner, W75; text
approved W79; exit code 2, W78 C5), before any audio is read, rather than falling
back quietly to no clean-up. The GUI marks the pass and refuses Start with text M2.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from whisperjav.main import REMOVED_SPEECH_ENHANCERS, SPEECH_ENHANCER_CHOICES, speech_enhancer_spec
from whisperjav.modules.speech_enhancement import pipeline_helper
from whisperjav.modules.speech_enhancement.base import SpeechEnhancerUnavailable
from whisperjav.modules.speech_enhancement.factory import (
    REMOVED_BACKENDS,
    SpeechEnhancerFactory,
    removed_backend_in_pass,
    removed_backend_message,
)

REPO = Path(__file__).resolve().parents[1]
M1_TAIL = ("(none, zipenhancer, bs-roformer, ffmpeg-dsp or htdemucs) and run again.")


def m1(where):
    return ("ClearVoice is no longer included in WhisperJAV 1.10. Choose another speech "
            f"enhancer for {where} {M1_TAIL}")


class TestTheText:
    @pytest.mark.parametrize("value", ["clearvoice", "clearvoice:FRCRN_SE_16K", " ClearVoice:MossFormer2_SE_48K"])
    def test_m1_names_the_place(self, value):
        assert removed_backend_message(value, "--pass1-speech-enhancer") == m1("--pass1-speech-enhancer")

    @pytest.mark.parametrize("value", ["none", "zipenhancer", "bs-roformer:vocals", "", None, 3])
    def test_other_values_are_not_touched(self, value):
        assert removed_backend_message(value, "x") is None

    def test_m1_offers_exactly_the_backends_that_exist(self):
        assert sorted(M1_TAIL[1:M1_TAIL.index(")")].replace(" or ", ", ").split(", ")) == \
            sorted(SpeechEnhancerFactory.list_backends())


class TestNoDrift:
    def test_cli_removed_list_matches_the_factory(self):
        assert sorted(REMOVED_SPEECH_ENHANCERS) == sorted(REMOVED_BACKENDS)

    def test_removed_names_are_not_offered(self):
        for name in REMOVED_BACKENDS:
            assert name not in SPEECH_ENHANCER_CHOICES
            assert name not in SpeechEnhancerFactory.list_backends()

    def test_argument_parsing_lets_it_through_to_the_named_check(self):
        """argparse's own error cannot name the flag; the check after parsing does."""
        assert speech_enhancer_spec("clearvoice:FRCRN_SE_16K") == "clearvoice:FRCRN_SE_16K"


class TestNoQuietFallback:
    """Without the guard an unknown name would become "none" with only a warning."""

    def test_direct(self):
        with pytest.raises(SpeechEnhancerUnavailable) as excinfo:
            pipeline_helper.create_enhancer_direct(backend="clearvoice", model="FRCRN_SE_16K")
        assert str(excinfo.value) == m1("this run")

    def test_from_config(self):
        with pytest.raises(SpeechEnhancerUnavailable):
            pipeline_helper.create_enhancer_from_config(
                {"params": {"speech_enhancer": {"backend": "clearvoice"}}})

    def test_factory(self):
        with pytest.raises(SpeechEnhancerUnavailable):
            SpeechEnhancerFactory.create("clearvoice")


class TestThePass:
    """What the pass worker checks before a pass reads audio."""

    def test_speech_enhancer(self):
        assert removed_backend_in_pass({"speech_enhancer": "clearvoice:FRCRN_SE_16K"}, 2) == \
            m1("--pass2-speech-enhancer")

    def test_qwen_parameters(self):
        cfg = {"speech_enhancer": None, "qwen_params": {"qwen_enhancer": "clearvoice"}}
        assert removed_backend_in_pass(cfg, 1) == m1("qwen_enhancer in --pass1-qwen-params")

    def test_transformers_parameters(self):
        cfg = {"hf_params": {"hf_speech_enhancer": "clearvoice"}}
        assert removed_backend_in_pass(cfg, 2) == m1("hf_speech_enhancer in --pass2-hf-params")

    def test_a_good_pass_is_left_alone(self):
        cfg = {"speech_enhancer": "zipenhancer", "qwen_params": {"qwen_enhancer": "none"}}
        assert removed_backend_in_pass(cfg, 1) is None

    def test_the_worker_calls_it_before_building_the_pipeline(self):
        source = (REPO / "whisperjav" / "ensemble" / "pass_worker.py").read_text(encoding="utf-8")
        check = source.index("removed = _removed_enhancer_in_pass(pass_config, pass_number)")
        build = source.index("pipeline = _build_pipeline(")
        assert check < build


@pytest.mark.parametrize("args,where", [
    (["--pass1-speech-enhancer", "clearvoice"], "--pass1-speech-enhancer"),
    (["--pass2-speech-enhancer", "clearvoice:FRCRN_SE_16K"], "--pass2-speech-enhancer"),
    (["--qwen-enhancer", "clearvoice"], "--qwen-enhancer"),
    (["--pass1-qwen-params", json.dumps({"qwen_enhancer": "clearvoice"})], "qwen_enhancer in --pass1-qwen-params"),
])
def test_the_command_line_stops_with_m1_and_exit_2(args, where):
    """G2. The input file does not exist: had the run gone on, it would have stopped
    later with a different message and status."""
    result = subprocess.run(
        [sys.executable, "-m", "whisperjav.main", "dummy.mp4", "--ensemble",
         "--pass1-pipeline", "qwen", "--pass2-pipeline", "balanced", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert result.returncode == 2, (result.stdout + result.stderr)[-800:]
    assert m1(where) in result.stderr


class TestTheGui:
    def test_the_lists_no_longer_offer_it(self):
        html = (REPO / "whisperjav" / "webview_gui" / "assets" / "index.html").read_text(encoding="utf-8")
        assert "clearvoice" not in html.lower()

    def test_start_is_refused_with_m2(self):
        js = (REPO / "whisperjav" / "webview_gui" / "assets" / "app.js").read_text(encoding="utf-8")
        assert "ErrorHandler.show('ClearVoice is no longer included'," in js
        assert ("ClearVoice is no longer included in WhisperJAV 1.10. ` +\n"
                "                `Choose another speech enhancer for ${which}, then press Start again.`") in js
        # Checked before the configuration is built and sent.
        assert js.index("const marked = this.passesWithRemovedEnhancer();") < \
            js.index("const config = this.collectConfig();\n\n            // Persist BYOP")

    def test_load_keeps_the_saved_value_and_choosing_clears_the_mark(self):
        js = (REPO / "whisperjav" / "webview_gui" / "assets" / "app.js").read_text(encoding="utf-8")
        assert "if (saved) out[this.FIELDS[`${pass}-enhancer`].key] = saved;" in js
        assert js.count("this.clearRemovedEnhancer('pass1');") == 1
        assert js.count("this.clearRemovedEnhancer('pass2');") == 1
