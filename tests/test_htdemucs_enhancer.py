#!/usr/bin/env python3
"""
Tests for the htdemucs (Demucs v4) vocal-isolation backend.

Two things are being pinned here.

First, that it is offered everywhere it should be: registered in the factory,
in the CLI's list of enhancers, in both of the GUI's per-pass dropdowns, and in
the notebook -- the owner asked on 2026-09-17 for htdemucs to be "available for
all pipelines".

Second, that it follows the agreed error-handling rules (2026-09-17) and has no
rules of its own: a clean-up that cannot install or cannot start stops the run,
while a failure on one piece of audio is reported as a failed result and the
caller decides -- some pieces failing is a shortfall the user is told about,
every piece failing fails the file.

Third, that the model actually works. demucs is an ordinary WhisperJAV
dependency, so where it is installed the last class below runs the real model on
real audio; the not-installed path is then simulated instead.

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

    def test_it_is_installed_with_whisperjav_like_the_others(self):
        """
        Owner, 2026-09-17, replacing his earlier decision: htdemucs is an
        ordinary dependency, installed with WhisperJAV rather than by hand.

        It belongs in the same extra as the other model-based clean-ups, so that
        nothing about it is special: whatever installs clearvoice and
        bs-roformer installs demucs.
        """
        import tomllib

        with open(REPO / "pyproject.toml", "rb") as handle:
            project = tomllib.load(handle)["project"]

        enhance = project["optional-dependencies"]["enhance"]
        assert "demucs" in enhance

        # The peers it must travel with.
        assert "bs-roformer-infer" in enhance
        assert any(spec.startswith("clearvoice") for spec in enhance)

    def test_the_windows_installer_includes_that_extra(self):
        build = (REPO / "installer" / "build_release.py").read_text(encoding="utf-8")
        assert '"enhance"' in build, "the installer must build requirements from the enhance extra"


class TestItDoesNotFailQuietly:
    def test_it_is_marked_fatal(self):
        assert "htdemucs" in FATAL_WHEN_UNAVAILABLE

    def test_every_installable_enhancer_is_fatal(self):
        """
        Owner, 2026-09-17: "if user selected any but they cannot run then it is
        a failure and the process shall stop with helpful communication." That
        is every clean-up the user has to install.
        """
        for backend in ("zipenhancer", "clearvoice", "bs-roformer", "htdemucs"):
            assert backend in FATAL_WHEN_UNAVAILABLE, backend

    def test_the_two_that_cannot_be_missing_are_not(self):
        """"none" does nothing and "ffmpeg-dsp" uses the FFmpeg we already need."""
        for backend in ("none", "ffmpeg-dsp"):
            assert backend not in FATAL_WHEN_UNAVAILABLE

    def test_the_fatal_set_matches_what_has_to_be_installed(self):
        from whisperjav.modules.speech_enhancement.factory import _BACKEND_DEPENDENCIES
        must_install = {name for name, info in _BACKEND_DEPENDENCIES.items()
                        if not info["always_available"]}
        assert set(FATAL_WHEN_UNAVAILABLE) == must_install

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

    @pytest.mark.parametrize("backend", ["zipenhancer", "clearvoice", "bs-roformer"])
    def test_an_installable_enhancer_that_is_missing_now_stops_the_run(
            self, backend, monkeypatch):
        """
        These used to warn and carry on with the untouched audio. Since
        2026-09-17 they stop, like htdemucs.
        """
        real = SpeechEnhancerFactory.is_backend_available
        monkeypatch.setattr(
            SpeechEnhancerFactory, "is_backend_available",
            staticmethod(lambda name: (False, "pip install something")
                         if name == backend else real(name)))
        with pytest.raises(SpeechEnhancerUnavailable) as excinfo:
            pipeline_helper.create_enhancer_direct(backend=backend)
        assert backend in str(excinfo.value)
        assert "pip install something" in str(excinfo.value)

    def test_the_two_that_cannot_be_missing_still_never_stop(self):
        """A clean-up that is always there must not gain a way to fail."""
        assert pipeline_helper.create_enhancer_direct(backend="none").name == "none"
        # The FFmpeg backend labels itself "ffmpeg" rather than "ffmpeg-dsp";
        # what matters here is that it was built and not downgraded to "none".
        ffmpeg = pipeline_helper.create_enhancer_direct(backend="ffmpeg-dsp")
        assert ffmpeg.name != "none"
        assert "FFmpeg" in ffmpeg.display_name


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


class TestTheModelActuallyWorks:
    """
    Runs the real model. Skipped where demucs is not installed.

    Until 2026-09-17 nothing here had ever been executed: every behavioural test
    was skipped on a machine WITH demucs, so the separation itself, the tensor
    shapes and the stem index were covered nowhere at all.
    """

    pytestmark = pytest.mark.skipif(
        not DEMUCS_INSTALLED, reason="demucs is not installed on this machine")

    @staticmethod
    def _enhancer():
        from whisperjav.modules.speech_enhancement.factory import SpeechEnhancerFactory
        return SpeechEnhancerFactory.create("htdemucs", config={})

    @staticmethod
    def _audio(seconds=1.0, rate=44100):
        import numpy as np
        t = np.arange(int(rate * seconds)) / rate
        tone = 0.3 * np.sin(2 * np.pi * 220 * t)
        return (tone + 0.02 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)

    def test_it_separates_and_returns_usable_audio(self):
        import numpy as np

        enhancer = self._enhancer()
        try:
            audio = self._audio()
            result = enhancer.enhance(audio, 44100)

            assert result.success is True
            assert result.error_message is None
            # Same length in and out: the scenes are paired by position with the
            # untouched track, so a change in length would misplace every line.
            assert len(result.audio) == len(audio)
            assert result.audio.dtype == np.float32
            assert result.audio.ndim == 1, "the rest of WhisperJAV works in mono"
            assert bool(np.isfinite(result.audio).all()), "NaN or inf would poison the ASR"
            assert result.sample_rate == 44100
            assert result.parameters["stem"] == "vocals"
        finally:
            enhancer.cleanup()

    def test_input_at_another_rate_is_resampled_not_refused(self):
        # Scenes arrive at 48kHz from the extractor, not at the model's 44.1kHz.
        enhancer = self._enhancer()
        try:
            result = enhancer.enhance(self._audio(rate=48000), 48000)
            assert result.success is True
            assert result.parameters["input_sr"] == 48000
            assert result.sample_rate == 44100
        finally:
            enhancer.cleanup()

    def test_an_unknown_model_is_refused_before_anything_runs(self):
        from whisperjav.modules.speech_enhancement.base import SpeechEnhancerUnavailable
        from whisperjav.modules.speech_enhancement.factory import SpeechEnhancerFactory

        with pytest.raises(SpeechEnhancerUnavailable) as caught:
            SpeechEnhancerFactory.create("htdemucs", config={"model": "not-a-model"})
        assert "not-a-model" in str(caught.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
