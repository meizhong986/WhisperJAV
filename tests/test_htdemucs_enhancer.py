#!/usr/bin/env python3
"""
Tests for the htdemucs (Demucs v4) vocal-isolation backend.

Two things are being pinned here.

First, that it is offered everywhere it should be: registered in the factory,
in the CLI's list of enhancers, in both of the GUI's per-pass dropdowns, and in
the notebook -- the owner asked on 2026-09-17 for htdemucs to be "available for
all pipelines".

Second, and more important, that it does NOT fail quietly. Every other enhancer
here degrades to "no enhancement" with a warning when it cannot run. htdemucs is
chosen because the audio needs it, so falling back would hand the user subtitles
made from the untouched audio, from a run that exited 0. The owner's decision was
"it should fail with good user communication", so the run must stop instead.

The demucs package is deliberately not a WhisperJAV dependency, so on a machine
without it these tests exercise the not-installed path for real; on a machine with
it, the not-installed path is simulated.

Run with: pytest tests/test_htdemucs_enhancer.py -v
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from whisperjav.main import SPEECH_ENHANCER_CHOICES
from whisperjav.modules.speech_enhancement.base import SpeechEnhancerUnavailable
from whisperjav.modules.speech_enhancement.factory import (
    FATAL_WHEN_UNAVAILABLE,
    SpeechEnhancerFactory,
)
from whisperjav.modules.speech_enhancement import pipeline_helper

REPO = Path(__file__).resolve().parents[1]
DEMUCS_INSTALLED = importlib.util.find_spec("demucs") is not None


class TestItIsOffered:
    def test_the_factory_knows_it(self):
        assert "htdemucs" in SpeechEnhancerFactory.list_backends()

    def test_the_cli_offers_it(self):
        assert "htdemucs" in SPEECH_ENHANCER_CHOICES

    def test_the_cli_list_matches_the_factory(self):
        assert sorted(SPEECH_ENHANCER_CHOICES) == sorted(SpeechEnhancerFactory.list_backends())

    def test_both_gui_passes_offer_it(self):
        html = (REPO / "whisperjav" / "webview_gui" / "assets" / "index.html").read_text(
            encoding="utf-8")
        enabled = re.findall(r'<option value="htdemucs"(?![^>]*\bdisabled\b)', html)
        assert len(enabled) == 2, f"expected an enabled htdemucs option per pass, found {len(enabled)}"

    def test_the_notebook_offers_it_and_installs_it_when_chosen(self):
        nb = json.loads((REPO / "notebook" / "WhisperJAV_colab_edition_expert.ipynb").read_text(
            encoding="utf-8"))
        source = "".join("".join(c["source"]) for c in nb["cells"])
        assert source.count('"htdemucs"') >= 2, "both passes should offer htdemucs"
        assert "extra_packages.add('demucs')" in source

    def test_it_is_not_installed_with_whisperjav(self):
        """Owner, 2026-09-17: it arrives only when someone picks it."""
        pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        dependency_lines = [
            line for line in pyproject.splitlines()
            if re.match(r'\s*"demucs[",>=<]', line)
        ]
        assert dependency_lines == [], f"demucs should not be a declared dependency: {dependency_lines}"


class TestItDoesNotFailQuietly:
    def test_it_is_marked_fatal(self):
        assert "htdemucs" in FATAL_WHEN_UNAVAILABLE

    def test_the_other_enhancers_are_not(self):
        """The change must not turn every backend into a hard stop."""
        for backend in ("ffmpeg-dsp", "zipenhancer", "clearvoice", "bs-roformer", "none"):
            assert backend not in FATAL_WHEN_UNAVAILABLE

    @pytest.mark.skipif(DEMUCS_INSTALLED, reason="demucs is installed on this machine")
    def test_creating_it_without_the_package_raises(self):
        with pytest.raises(SpeechEnhancerUnavailable) as excinfo:
            pipeline_helper.create_enhancer_direct(backend="htdemucs")
        message = str(excinfo.value)
        assert "htdemucs" in message
        assert "pip install demucs" in message

    @pytest.mark.skipif(DEMUCS_INSTALLED, reason="demucs is installed on this machine")
    def test_the_config_path_raises_too(self):
        with pytest.raises(SpeechEnhancerUnavailable):
            pipeline_helper.create_enhancer_from_config(
                {"params": {"speech_enhancer": {"backend": "htdemucs"}}})

    def test_an_ordinary_missing_enhancer_still_falls_back(self, monkeypatch):
        """bs-roformer keeps the old behaviour: warn, carry on without it."""
        real = SpeechEnhancerFactory.is_backend_available
        monkeypatch.setattr(
            SpeechEnhancerFactory, "is_backend_available",
            staticmethod(lambda name: (False, "pip install something")
                         if name == "bs-roformer" else real(name)))
        enhancer = pipeline_helper.create_enhancer_direct(backend="bs-roformer")
        assert enhancer.name == "none"


class TestTheBackendItself:
    def test_an_unknown_model_is_refused_with_the_real_names(self):
        from whisperjav.modules.speech_enhancement.backends.htdemucs import (
            HtDemucsSpeechEnhancer,
            SUPPORTED_MODELS,
        )
        with pytest.raises(SpeechEnhancerUnavailable) as excinfo:
            HtDemucsSpeechEnhancer(model="htdemux")
        for name in SUPPORTED_MODELS:
            assert name in str(excinfo.value)

    def test_it_reports_itself_consistently(self):
        from whisperjav.modules.speech_enhancement.backends.htdemucs import (
            DEFAULT_SAMPLE_RATE,
            HtDemucsSpeechEnhancer,
        )
        enhancer = HtDemucsSpeechEnhancer()
        assert enhancer.name == "htdemucs"
        assert "Demucs" in enhancer.display_name
        assert enhancer.get_preferred_sample_rate() == DEFAULT_SAMPLE_RATE
        assert enhancer.get_output_sample_rate() == DEFAULT_SAMPLE_RATE
        assert enhancer.is_lightweight() is False
        enhancer.cleanup()  # must be safe before anything was loaded

    @pytest.mark.skipif(DEMUCS_INSTALLED, reason="demucs is installed on this machine")
    def test_enhancing_without_the_package_raises_rather_than_returning_audio(self):
        import numpy as np
        from whisperjav.modules.speech_enhancement.backends.htdemucs import (
            HtDemucsSpeechEnhancer,
        )
        enhancer = HtDemucsSpeechEnhancer()
        with pytest.raises(SpeechEnhancerUnavailable):
            enhancer.enhance(np.zeros(16000, dtype=np.float32), 16000)


class TestTheRunStops:
    """What a user actually sees, through the real command line."""

    @pytest.fixture(scope="class")
    def media(self, tmp_path_factory):
        import shutil
        if shutil.which("ffmpeg") is None:
            pytest.skip("FFmpeg is not installed")
        path = tmp_path_factory.mktemp("media") / "tone.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=5:duration=2",
            "-c:a", "aac", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-shortest", str(path)
        ], check=True, capture_output=True)
        return path

    @pytest.mark.skipif(DEMUCS_INSTALLED, reason="demucs is installed on this machine")
    @pytest.mark.parametrize("args", [
        ["--ensemble", "--pass1-pipeline", "fidelity",
         "--pass1-speech-enhancer", "htdemucs", "--pass2-pipeline", "balanced"],
        ["--mode", "qwen", "--qwen-enhancer", "htdemucs"],
    ])
    def test_it_stops_before_transcribing_and_says_why(self, media, args):
        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", str(media), "--accept-cpu-mode"] + args,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        output = result.stdout + result.stderr
        assert result.returncode == 1, output[-800:]
        assert "not installed" in output
        assert "pip install demucs" in output
        assert "Nothing has been transcribed" in output
        # And it really did stop before doing any work.
        assert "Extracting the audio" not in output

    def test_the_flag_is_accepted_by_the_parser(self, tmp_path):
        """Whether or not demucs is here, the name itself must parse."""
        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "dummy.mp4",
             "--dump-params", str(tmp_path / "params.json"),
             "--ensemble", "--pass1-pipeline", "balanced", "--pass2-pipeline", "balanced",
             "--pass1-speech-enhancer", "htdemucs"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        assert result.returncode == 0, (result.stdout + result.stderr)[-600:]
