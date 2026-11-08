"""
1.10.2 (owner W603): a speech enhancer name WhisperJAV does not offer.

ClearVoice (gone in 1.10.0) and BS-RoFormer (gone in 1.10.2) no longer have stops of their own. Two general rules
take their place, with texts the owner approved (W603):

(i)  a speech enhancer name WhisperJAV does not offer stops the run before any audio is read. The three speech
     enhancer flags reject it while the arguments are parsed (argparse's "invalid choice", #306); a name inside the
     pass parameters, a preset or a pipeline config gets the message below.
(ii) the GUI marks a pass whose saved speech enhancer the Speech Enhancer menu no longer offers, and refuses Start.

Replaces tests/test_clearvoice_removed_v110.py (texts M1 and M2 of W75/W79 retired by W603).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from whisperjav.modules.speech_enhancement import pipeline_helper
from whisperjav.modules.speech_enhancement.base import SpeechEnhancerUnavailable
from whisperjav.modules.speech_enhancement.factory import (
    SpeechEnhancerFactory,
    unknown_backend_in_pass,
    unknown_backend_message,
)

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "whisperjav" / "webview_gui" / "assets"
OFFERED = "none, ffmpeg-dsp, zipenhancer, htdemucs, mel-band-roformer or mini-bs-roformer"


def message(name, where):
    return f"'{name}' is not a speech enhancer in WhisperJAV. Choose one of {OFFERED} for {where} and run again."


class TestTheText:
    @pytest.mark.parametrize("value,name", [("clearvoice", "clearvoice"), ("clearvoice:FRCRN_SE_16K", "clearvoice"),
                                            (" bs-roformer:vocals", "bs-roformer"), ("zipenhance", "zipenhance"),
                                            ("ZipEnhancer", "ZipEnhancer")])
    def test_it_names_the_value_and_the_place(self, value, name):
        assert unknown_backend_message(value, "--pass1-speech-enhancer") == message(name, "--pass1-speech-enhancer")

    @pytest.mark.parametrize("value", ["none", "ffmpeg-dsp:loudnorm", "zipenhancer", "htdemucs", "mel-band-roformer",
                                       "mini-bs-roformer:vocals", "", "  ", None, 3])
    def test_a_name_it_offers_or_no_name_is_left_alone(self, value):
        assert unknown_backend_message(value, "x") is None

    def test_it_offers_exactly_the_backends_that_exist(self):
        listed = OFFERED.replace(" or ", ", ").split(", ")
        assert listed == SpeechEnhancerFactory.list_backends()

    def test_no_name_is_special_any_more(self):
        """W603: the named stops (REMOVED_BACKENDS, REPLACED_BY, M1, M2) are gone."""
        from whisperjav.modules.speech_enhancement import factory
        import whisperjav.main as main
        for gone in ("REMOVED_BACKENDS", "REPLACED_BY", "removed_backend_message", "removed_backend_in_pass"):
            assert not hasattr(factory, gone), gone
        assert not hasattr(main, "REMOVED_SPEECH_ENHANCERS")


class TestNoQuietFallback:
    """Without the rule an unknown name would become "none" with only a warning."""

    def test_direct(self):
        with pytest.raises(SpeechEnhancerUnavailable) as excinfo:
            pipeline_helper.create_enhancer_direct(backend="clearvoice", model="FRCRN_SE_16K")
        assert str(excinfo.value) == message("clearvoice", "this run")

    def test_from_config(self):
        with pytest.raises(SpeechEnhancerUnavailable):
            pipeline_helper.create_enhancer_from_config({"params": {"speech_enhancer": {"backend": "bs-roformer"}}})

    def test_factory(self):
        with pytest.raises(SpeechEnhancerUnavailable):
            SpeechEnhancerFactory.create("bs-roformer")

    def test_none_and_empty_still_mean_no_clean_up(self):
        assert pipeline_helper.create_enhancer_direct(backend="none").name == "none"
        assert pipeline_helper.create_enhancer_direct(backend="").name == "none"


class TestThePass:
    """What the pass worker checks before a pass reads audio."""

    def test_speech_enhancer(self):
        assert unknown_backend_in_pass({"speech_enhancer": "clearvoice:FRCRN_SE_16K"}, 2) == \
            message("clearvoice", "--pass2-speech-enhancer")

    def test_qwen_parameters(self):
        cfg = {"speech_enhancer": None, "qwen_params": {"qwen_enhancer": "bs-roformer"}}
        assert unknown_backend_in_pass(cfg, 1) == message("bs-roformer", "qwen_enhancer in --pass1-qwen-params")

    def test_transformers_parameters(self):
        cfg = {"hf_params": {"hf_speech_enhancer": "clearvoice"}}
        assert unknown_backend_in_pass(cfg, 2) == message("clearvoice", "hf_speech_enhancer in --pass2-hf-params")

    def test_a_good_pass_is_left_alone(self):
        cfg = {"speech_enhancer": "mel-band-roformer", "qwen_params": {"qwen_enhancer": "none"}}
        assert unknown_backend_in_pass(cfg, 1) is None

    def test_the_worker_calls_it_before_building_the_pipeline(self):
        source = (REPO / "whisperjav" / "ensemble" / "pass_worker.py").read_text(encoding="utf-8")
        check = source.index("unknown = _unknown_enhancer_in_pass(pass_config, pass_number)")
        build = source.index("pipeline = _build_pipeline(")
        assert check < build


@pytest.mark.parametrize("args,flag", [
    (["--pass1-speech-enhancer", "clearvoice"], "--pass1-speech-enhancer"),
    (["--pass2-speech-enhancer", "bs-roformer:vocals"], "--pass2-speech-enhancer"),
    (["--qwen-enhancer", "clearvoice"], "--qwen-enhancer"),
])
def test_a_flag_stops_with_argparse_and_exit_2(args, flag):
    """The flags reject an unknown name while the arguments are parsed (#306), naming the flag and the choices."""
    result = subprocess.run(
        [sys.executable, "-m", "whisperjav.main", "dummy.mp4", "--ensemble",
         "--pass1-pipeline", "qwen", "--pass2-pipeline", "fidelity", *args],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert result.returncode == 2, (result.stdout + result.stderr)[-800:]
    assert f"argument {flag}: invalid choice" in result.stderr
    assert "mini-bs-roformer" in result.stderr


@pytest.mark.parametrize("flag,key,name", [
    ("--pass1-qwen-params", "qwen_enhancer", "clearvoice"),
    ("--pass1-hf-params", "hf_speech_enhancer", "bs-roformer"),
])
def test_a_pass_parameter_stops_with_the_message_and_exit_2(flag, key, name):
    """Rule (i) for what the flags do not see. The input file does not exist: had the run gone on, it would have
    stopped later with a different message and status."""
    result = subprocess.run(
        [sys.executable, "-m", "whisperjav.main", "dummy.mp4", "--ensemble",
         "--pass1-pipeline", "qwen", flag, json.dumps({key: name})],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert result.returncode == 2, (result.stdout + result.stderr)[-800:]
    assert message(name, f"{key} in {flag}") in result.stderr


class TestTheGui:
    """Rule (ii)."""

    def test_the_menus_do_not_offer_them(self):
        html = (ASSETS / "index.html").read_text(encoding="utf-8")
        assert "clearvoice" not in html.lower()
        assert 'value="bs-roformer' not in html

    def test_a_value_the_menu_does_not_offer_is_marked(self):
        js = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert "REMOVED_ENHANCERS" not in js and "ClearVoice" not in js
        body = js[js.index("    isRemovedEnhancer(value) {"):][:500]
        assert "return !Array.from(select.options).some(o => o.value === wanted && !o.dataset.removedEnhancer);" in body
        assert "marker.textContent = `${name} (no longer offered)`;" in js

    def test_start_is_refused_with_the_approved_text(self):
        js = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert "ErrorHandler.show('Speech enhancer no longer offered', marked.length === 2" in js
        assert ("`${which} uses a speech enhancer that WhisperJAV no longer offers (\"${shown}\") in your saved ` +\n"
                "                  `settings. Choose another speech enhancer for ${which}, then press Start again.`") in js
        # Checked before the configuration is built and sent.
        assert js.index("const marked = this.passesWithRemovedEnhancer();") < \
            js.index("const config = this.collectConfig();\n\n            // Persist BYOP")

    def test_load_keeps_the_saved_value_and_choosing_clears_the_mark(self):
        js = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert "if (saved) out[this.FIELDS[`${pass}-enhancer`].key] = saved;" in js
        assert js.count("this.clearRemovedEnhancer('pass1');") == 1
        assert js.count("this.clearRemovedEnhancer('pass2');") == 1

    def test_the_texts_have_chinese(self):
        zh = (ASSETS / "i18n_zh.js").read_text(encoding="utf-8")
        assert "ClearVoice" not in zh
        for english in ('"{raw_name} (no longer offered)": ', '"Speech enhancer no longer offered": ',
                        '"{pass} uses a speech enhancer that WhisperJAV no longer offers (\\"{raw_name}\\") in your saved '
                        'settings. Choose another speech enhancer for {pass}, then press Start again.": ',
                        '"Pass 1 and Pass 2 use a speech enhancer that WhisperJAV no longer offers (\\"{raw_name}\\") in '
                        'your saved settings. Choose another speech enhancer for Pass 1 and Pass 2, then press Start '
                        'again.": '):
            assert english in zh, english
