#!/usr/bin/env python3
"""
Tests for --passN-speech-enhancer accepting "backend:detail" again.

Everything downstream always understood that form: pass_worker splits it, and
FFmpegDSPBackend turns a comma-separated detail into its effects list. The #306
fix in v1.9.2 put a plain choices= list on these two flags, which rejected every
value containing a colon -- and the GUI builds exactly that value whenever FFmpeg
DSP is chosen for a pass, so such a run stopped with a usage error before reading
any audio.

These tests pin both halves: the form is accepted and normalised, an unknown
backend or effect is still refused at the boundary in #306's own words, and the
detail really does reach the FFmpeg filter chain.

Run with: pytest tests/test_speech_enhancer_spec.py -v
"""

import argparse
import subprocess
import sys

import pytest

from whisperjav.main import (
    FFMPEG_DSP_EFFECTS,
    SPEECH_ENHANCER_CHOICES,
    speech_enhancer_spec,
)


class TestAccepted:
    @pytest.mark.parametrize("value,expected", [
        ("none", "none"),
        ("ffmpeg-dsp", "ffmpeg-dsp"),
        ("clearvoice", "clearvoice"),
        ("ffmpeg-dsp:loudnorm", "ffmpeg-dsp:loudnorm"),
        ("ffmpeg-dsp:loudnorm,denoise,highpass", "ffmpeg-dsp:loudnorm,denoise,highpass"),
        ("clearvoice:MossFormer2_SE_48K", "clearvoice:MossFormer2_SE_48K"),
        ("bs-roformer:vocals", "bs-roformer:vocals"),
        # Whitespace is tidied so downstream sees a clean value.
        ("  ffmpeg-dsp : loudnorm , denoise ", "ffmpeg-dsp:loudnorm,denoise"),
    ])
    def test_value_is_accepted_and_normalised(self, value, expected):
        assert speech_enhancer_spec(value) == expected

    def test_every_backend_is_accepted_on_its_own(self):
        for backend in SPEECH_ENHANCER_CHOICES:
            assert speech_enhancer_spec(backend) == backend

    def test_every_known_effect_is_accepted(self):
        for effect in FFMPEG_DSP_EFFECTS:
            assert speech_enhancer_spec(f"ffmpeg-dsp:{effect}") == f"ffmpeg-dsp:{effect}"


class TestRejected:
    """#306: an unknown name is a user error and belongs at the boundary."""

    @pytest.mark.parametrize("value,must_mention", [
        ("zipenhance", "zipenhancer"),           # the typo #306 was about
        ("ffmpeg-dsp:loudnrm", "loudnorm"),      # a mistyped effect
        ("ffmpeg-dsp:loudnorm,denois", "denoise"),
        ("none:loudnorm", "none"),               # 'none' takes no detail
        ("ffmpeg-dsp:", "colon"),                # nothing after the colon
        ("ffmpeg-dsp:loudnorm,", "unknown"),     # a trailing comma is an empty effect
        ("", "choose from"),
    ])
    def test_value_is_refused(self, value, must_mention):
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            speech_enhancer_spec(value)
        assert must_mention in str(excinfo.value)

    def test_the_wording_306_asked_for_is_kept(self):
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            speech_enhancer_spec("zipenhance")
        message = str(excinfo.value)
        assert "invalid choice" in message
        for backend in SPEECH_ENHANCER_CHOICES:
            assert backend in message


class TestNoDrift:
    def test_the_effect_list_matches_the_backend(self):
        """The CLI list is hardcoded to keep --help cheap; this catches drift."""
        from whisperjav.modules.speech_enhancement.backends.ffmpeg_dsp import (
            AVAILABLE_EFFECTS,
        )
        assert sorted(FFMPEG_DSP_EFFECTS) == sorted(AVAILABLE_EFFECTS)


class TestReachesTheFilterChain:
    """The detail must not just be accepted -- it must change what FFmpeg runs."""

    def test_the_chosen_effects_end_up_in_the_ffmpeg_filters(self):
        # Run in a subprocess: importing the pass worker into the test process is
        # avoided by house rule, and a timeout here fails the test rather than
        # hanging the run.
        probe = (
            "from whisperjav.ensemble.pass_worker import _parse_speech_enhancer;"
            "from whisperjav.modules.speech_enhancement.backends.ffmpeg_dsp import FFmpegDSPBackend;"
            "backend, detail = _parse_speech_enhancer('ffmpeg-dsp:loudnorm,denoise');"
            "b = FFmpegDSPBackend(model=detail);"
            "print('EFFECTS=' + ','.join(b.effects));"
            "print('CHAIN=' + b._build_filter_chain())"
        )
        result = subprocess.run([sys.executable, "-c", probe],
                                capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=300)
        assert result.returncode == 0, result.stderr
        assert "EFFECTS=loudnorm,denoise" in result.stdout, result.stdout
        chain = [l for l in result.stdout.splitlines() if l.startswith("CHAIN=")][0]
        assert "loudnorm" in chain and "afftdn" in chain, chain


class TestTheFlagsThemselves:
    """What a user actually types, through argparse."""

    @pytest.mark.parametrize("flag", ["--pass1-speech-enhancer", "--pass2-speech-enhancer"])
    @pytest.mark.parametrize("value,expected_exit", [
        ("ffmpeg-dsp:loudnorm,denoise", 0),   # what the GUI builds
        ("ffmpeg-dsp", 0),
        ("zipenhance", 2),
        ("ffmpeg-dsp:loudnrm", 2),
    ])
    def test_the_flag_accepts_or_refuses(self, flag, value, expected_exit, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "dummy.mp4",
             "--dump-params", str(tmp_path / "params.json"),
             "--ensemble", "--pass1-pipeline", "balanced", "--pass2-pipeline", "balanced",
             flag, value],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        assert result.returncode == expected_exit, (result.stdout + result.stderr)[-600:]


class TestEnhanceForVadIsIgnoredWhereWeSayItIs:
    """
    The owner accepted on 2026-09-17 that fast, faster, transformers and crispasr
    ignore --passN-enhance-for-vad, on condition it is documented. The list that
    documents it must not drift from the pipelines themselves, so this reads their
    source rather than their docstrings.
    """

    def test_the_list_matches_the_pipeline_sources(self):
        probe = (
            "import inspect, json;"
            "from whisperjav.ensemble.pass_worker import PIPELINE_CLASSES, ENHANCE_FOR_VAD_IGNORED_BY;"
            "ignores = sorted(n for n, c in PIPELINE_CLASSES.items()"
            " if 'enhance_for_vad' not in inspect.getsource(c));"
            "print('IGNORES=' + json.dumps(ignores));"
            "print('DECLARED=' + json.dumps(sorted(ENHANCE_FOR_VAD_IGNORED_BY)))"
        )
        result = subprocess.run([sys.executable, "-c", probe],
                                capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=300)
        assert result.returncode == 0, result.stderr
        lines = dict(l.split("=", 1) for l in result.stdout.splitlines() if "=" in l)
        import json
        assert json.loads(lines["IGNORES"]) == json.loads(lines["DECLARED"]), (
            "ENHANCE_FOR_VAD_IGNORED_BY in pass_worker.py has drifted from what the "
            "pipeline classes actually read")

    def test_the_help_names_the_pipelines_that_ignore_it(self):
        result = subprocess.run([sys.executable, "-m", "whisperjav.main", "--help"],
                                capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=300)
        assert result.returncode == 0
        help_text = " ".join(result.stdout.split())
        assert "fast, faster, transformers and crispasr ignore it" in help_text
