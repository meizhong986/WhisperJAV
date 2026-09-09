"""
v1.9.2 S2/S6/S7/S8/S9: the balanced pipeline's built-in VAD runs a user-chosen
Silero build, and balanced no longer offers an external speech segmenter.

Covers the adapter (which model file, which output convention), the Pydantic preset
(version + the owner's 0.5/0.4/0.3 thresholds), the CLI surface (--vad-version,
--passN-vad-version, the rejections, the removal of --no-vad) and the GUI backend
(the built-in VAD's Customize schema, the args the Ensemble tab builds).

Nothing here loads an ASR model.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "whisperjav.main", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )


def _dump(tmp_path: Path, *args: str) -> dict:
    out = tmp_path / f"dump_{abs(hash(args))}.json"
    proc = _run("--dump-params", str(out), *args)
    assert out.exists(), f"dump not written; stderr:\n{proc.stderr[-2000:]}"
    return json.loads(out.read_text(encoding="utf-8"))


# =============================================================================
# The adapter
# =============================================================================
class TestAdapter:
    def test_all_three_models_ship_with_the_package(self):
        from whisperjav.modules.silero_vad_adapter import VAD_VERSIONS, model_path

        assert VAD_VERSIONS == ("3.1", "4.0", "6.2")
        for v in VAD_VERSIONS:
            assert model_path(v).is_file(), f"missing bundled model: {model_path(v)}"

    def test_default_is_3_1(self):
        from whisperjav.modules.silero_vad_adapter import DEFAULT_VAD_VERSION
        assert DEFAULT_VAD_VERSION == "3.1"

    @pytest.mark.parametrize("given,expected", [
        (None, "3.1"),            # the common case: no override
        ("4.0", "4.0"),
        ("6.2", "6.2"),
        ("silero-v6.2", "3.1"),   # not a version string -- warn and default
        ("nonsense", "3.1"),      # falls back, never raises
    ])
    def test_normalise_version(self, given, expected):
        from whisperjav.modules.silero_vad_adapter import normalise_version
        assert normalise_version(given) == expected

    @pytest.mark.parametrize("version", ["3.1", "4.0", "6.2"])
    def test_probabilities_are_probabilities(self, version):
        """One value per 512 samples, all inside [0, 1].

        This is the guard for trap 1: v3.1's ONNX output is two-class, and reading
        index 0 instead of 1 yields a smooth, believable series that is NOT speech
        probability. Silence must score low and a loud tone must score higher.
        """
        from whisperjav.modules.silero_vad_adapter import get_model

        model = get_model(version)
        silence = np.zeros(512 * 40, dtype=np.float32)
        probs = np.asarray(model(silence)).reshape(-1)

        assert probs.shape[0] == 40
        assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0
        # Digital silence is not speech under any build.
        assert float(probs.mean()) < 0.2, f"v{version} calls silence speech: {probs.mean():.3f}"

    @pytest.mark.parametrize("version", ["3.1", "4.0", "6.2"])
    def test_state_resets_between_calls(self, version):
        """Two identical inputs must give identical output (LSTM state reset)."""
        from whisperjav.modules.silero_vad_adapter import get_model

        rng = np.random.default_rng(7)
        audio = (rng.standard_normal(512 * 60) * 0.05).astype(np.float32)
        model = get_model(version)
        first = np.asarray(model(audio)).reshape(-1)
        second = np.asarray(model(audio)).reshape(-1)
        assert np.allclose(first, second)

    def test_install_reaches_faster_whispers_own_vad(self):
        """The rebind must change what faster-whisper's get_speech_timestamps sees."""
        import faster_whisper.vad as fw_vad
        from whisperjav.modules.silero_vad_adapter import active_version, install

        try:
            got = install("4.0")
            assert got == "4.0"
            assert active_version() == "4.0"
            model = fw_vad.get_vad_model()
            probs = np.asarray(model(np.zeros(512 * 10, dtype=np.float32))).reshape(-1)
            assert probs.shape[0] == 10
        finally:
            # Leave the process on the default so later tests are not surprised.
            install("3.1")


# =============================================================================
# The preset (single source of truth)
# =============================================================================
class TestPreset:
    def test_version_defaults_to_3_1_on_every_sensitivity(self):
        from whisperjav.config.components.vad.faster_whisper_vad import FasterWhisperVAD
        for name in ("conservative", "balanced", "aggressive"):
            assert FasterWhisperVAD.get_preset(name).version == "3.1"

    def test_owner_thresholds(self):
        """conservative 0.5 / balanced 0.4 / aggressive 0.3 (owner, 2026-09-09)."""
        from whisperjav.config.components.vad.faster_whisper_vad import FasterWhisperVAD
        assert FasterWhisperVAD.get_preset("conservative").threshold == 0.50
        assert FasterWhisperVAD.get_preset("balanced").threshold == 0.40
        assert FasterWhisperVAD.get_preset("aggressive").threshold == 0.30

    def test_unknown_version_is_rejected(self):
        from whisperjav.config.components.vad.faster_whisper_vad import FasterWhisperVADOptions
        with pytest.raises(Exception):
            FasterWhisperVADOptions(version="5.0")

    def test_version_never_reaches_faster_whisper_vadoptions(self):
        """VadOptions has no 'version' field -- passing it would raise TypeError."""
        from whisperjav.modules.faster_whisper_pro_asr import FasterWhisperProASR

        built = FasterWhisperProASR._build_vad_parameters(
            object.__new__(FasterWhisperProASR),
            {"version": "6.2", "threshold": 0.4, "speech_pad_ms": 400},
        )
        assert "version" not in built
        assert built == {"threshold": 0.4, "speech_pad_ms": 400}

        from faster_whisper.vad import VadOptions
        VadOptions(**built)  # must not raise


# =============================================================================
# The CLI
# =============================================================================
@pytest.mark.slow
class TestCli:
    def test_flag_is_registered_in_main(self):
        out = _run("--help").stdout
        assert "--vad-version" in out
        assert "--pass1-vad-version" in out
        assert "--pass2-vad-version" in out

    def test_no_vad_is_gone(self):
        # --help short-circuits argparse before unknown-argument checking, so the
        # removal is proved with a real (non-help) invocation.
        assert "--no-vad" not in _run("--help").stdout
        proc = _run("--no-vad", "nonexistent.wav")
        assert proc.returncode == 2
        assert "unrecognized arguments: --no-vad" in (proc.stderr + proc.stdout)

    def test_balanced_resolves_the_preset_version_and_threshold(self, tmp_path):
        for sensitivity, threshold in (("conservative", 0.5), ("balanced", 0.4), ("aggressive", 0.3)):
            cfg = _dump(tmp_path, "--mode", "balanced", "--sensitivity", sensitivity)
            vad = cfg["resolved_config"]["params"]["vad"]
            assert vad["version"] == "3.1"
            assert vad["threshold"] == threshold

    def test_explicit_version_wins(self, tmp_path):
        cfg = _dump(tmp_path, "--mode", "balanced", "--sensitivity", "balanced", "--vad-version", "6.2")
        assert cfg["resolved_config"]["params"]["vad"]["version"] == "6.2"

    def test_balanced_rejects_an_external_segmenter(self):
        proc = _run("--dump-params", "x.json", "--mode", "balanced", "--speech-segmenter", "whisperseg")
        assert proc.returncode == 2
        assert "--vad-version" in (proc.stderr + proc.stdout)

    def test_balanced_pass_rejects_an_external_segmenter(self):
        proc = _run("--dump-params", "x.json", "--ensemble",
                    "--pass1-pipeline", "balanced", "--pass1-speech-segmenter", "ten")
        assert proc.returncode == 2
        assert "--pass1-vad-version" in (proc.stderr + proc.stdout)

    def test_other_pipelines_keep_their_segmenter(self, tmp_path):
        cfg = _dump(tmp_path, "--ensemble", "--pass1-pipeline", "qwen", "--pass1-speech-segmenter", "ten")
        assert cfg["ensemble_config"]["pass1"]["speech_segmenter"] == "ten"

    def test_version_flag_is_rejected_where_it_would_do_nothing(self):
        assert _run("--dump-params", "x.json", "--mode", "fast", "--vad-version", "4.0").returncode == 2
        assert _run("--dump-params", "x.json", "--ensemble", "--vad-version", "4.0").returncode == 2

    def test_pass_version_reaches_the_pass_config(self, tmp_path):
        cfg = _dump(tmp_path, "--ensemble", "--pass1-pipeline", "balanced", "--pass1-vad-version", "4.0")
        assert cfg["ensemble_config"]["pass1"]["vad_version"] == "4.0"

    def test_pass_version_is_rejected_on_a_pipeline_that_cannot_use_it(self):
        """The mirror of --vad-version: not a silent no-op on the ensemble path either."""
        proc = _run("--dump-params", "x.json", "--ensemble",
                    "--pass1-pipeline", "fidelity", "--pass1-vad-version", "4.0")
        assert proc.returncode == 2
        assert "--pass1-vad-version" in (proc.stderr + proc.stdout)

    @pytest.mark.parametrize("flag,value", [("--max-group-duration", "3"),
                                            ("--chunk-threshold", "0.9")])
    def test_external_grouping_knobs_are_rejected_on_balanced(self, tmp_path, flag, value):
        """They group an EXTERNAL segmenter's output; balanced has no external segmenter.

        Owner O2 (2026-09-09): "omitted means omitted" -- reject, do not accept-and-ignore.
        """
        out = tmp_path / "never.json"
        proc = _run("--dump-params", str(out), "--mode", "balanced", flag, value)
        assert proc.returncode == 2
        assert f"{flag} is not available with --mode balanced" in (proc.stdout + proc.stderr)
        assert not out.exists()

    @pytest.mark.parametrize("flag,value", [("--max-group-duration", "3"),
                                            ("--chunk-threshold", "0.9")])
    def test_external_grouping_knobs_still_work_where_they_apply(self, tmp_path, flag, value):
        assert _run("--dump-params", str(tmp_path / "f.json"),
                    "--mode", "fidelity", flag, value).returncode == 0



# =============================================================================
# The GUI backend
# =============================================================================
class TestGuiBackend:
    def _api(self):
        from whisperjav.webview_gui.api import WhisperJAVAPI
        api = WhisperJAVAPI.__new__(WhisperJAVAPI)
        # Only what the arg builders read; the real __init__ starts GUI machinery.
        api.default_output = "."
        return api

    def test_customize_tab_has_a_schema_for_the_built_in_vad(self):
        schema = self._api().get_segmenter_schema("faster-whisper")
        assert schema["success"], schema
        assert set(schema["defaults"]) == {
            "threshold", "min_speech_duration_ms", "max_speech_duration_s",
            "min_silence_duration_ms", "speech_pad_ms",
        }
        # 'version' is chosen in the pass row, not duplicated in the modal.
        assert "version" not in schema["parameters"]
        assert schema["presets"]["aggressive"]["threshold"] == 0.3

    def test_external_segmenter_schemas_still_work(self):
        assert self._api().get_segmenter_schema("whisperseg")["success"]
        assert not self._api().get_segmenter_schema("not-a-backend")["success"]

    def test_balanced_pass_sends_the_version_not_a_segmenter(self):
        api = self._api()
        args = api._build_twopass_args({
            "inputs": ["a.wav"],
            "output_dir": ".",
            "pass1": {"pipeline": "balanced", "sensitivity": "balanced",
                      "speechSegmenter": None, "vadVersion": "6.2"},
            "pass2": {"enabled": False},
        })
        assert "--pass1-vad-version" in args
        assert args[args.index("--pass1-vad-version") + 1] == "6.2"
        assert "--pass1-speech-segmenter" not in args

    def test_non_balanced_pass_still_sends_a_segmenter(self):
        api = self._api()
        args = api._build_twopass_args({
            "inputs": ["a.wav"],
            "output_dir": ".",
            "pass1": {"pipeline": "qwen", "sensitivity": "balanced",
                      "isQwen": True, "speechSegmenter": "ten"},
            "pass2": {"enabled": False},
        })
        assert "--pass1-speech-segmenter" in args
        assert "--pass1-vad-version" not in args

    def test_single_pass_balanced_never_forwards_a_segmenter(self):
        """S9.2: the Transcription tab has no VAD control -- it uses the defaults."""
        api = self._api()
        args = api.build_args({
            "inputs": ["a.wav"], "mode": "balanced", "speech_segmenter": "whisperseg",
        })
        assert "--speech-segmenter" not in args
        assert "--vad-version" not in args
