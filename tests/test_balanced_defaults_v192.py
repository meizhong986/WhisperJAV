"""
v1.9.2: Balanced pipeline speech-segmenter default and single-pass preset resolution.

Owner decisions: CFF3 (2026-09-05) made FireRedVAD the Balanced default with a
fallback chain; the owner reversed the default on 2026-09-05 (N3) — Balanced runs
faster-whisper's built-in VAD again, as in v1.9.0/v1.9.1. His balanced requirements
(S2/S9, 2026-09-09) then went further: Balanced accepts **no** external speech
segmenter at all, so `--speech-segmenter` with `--mode balanced` is now a usage error
and the firered-vad / ten routing-guard exemption that CFF3 added has been removed as
unreachable. Both backends remain available through --ensemble; the per-sensitivity
YAML resolution and the Test-D grouping overlay are unchanged there.

2026-09-12: the owner made FireRedVAD the FIDELITY default, on `--mode fidelity` and
on an --ensemble fidelity pass. A single-pass exemption from the routing guard is back
for that one backend (`SINGLE_PASS_EXTERNAL_OK`), scoped to fidelity rather than to
balanced, and justified by the per-sensitivity resolution v1.9.2 added to the
single-pass path. whisperseg / NeMo / whisper-vad / TEN still downgrade there.

Two layers are covered without importing any ML stack:
1. `whisperjav.config.segmenter_presets` — the shared constants and resolver the
   entry points (main.py, pass_worker, GUI api) use.
2. The real CLI through `--dump-params`, which exercises main.py's resolution
   order backend → YAML preset → Test-D → CLI override, and the routing guard.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from whisperjav.config.segmenter_presets import (
    BALANCED_DEFAULT_SEGMENTER,
    SEGMENTER_PARAMS,
    resolve_segmenter_sensitivity,
)
from whisperjav.utils.output_coverage import PASSTHROUGH_SEGMENTERS

# Repository root, for the tests that read installer templates as text.
REPO = Path(__file__).resolve().parents[1]


# ─── helper layer ────────────────────────────────────────────────────────────

class TestSceneBoundsOverrides:
    """v1.9.2 (owner, 2026-09-11): scene ceiling 240 s on Balanced and Fidelity (semantic).

    These read the override tables directly so the values are pinned even when the
    slow --dump-params tests are skipped. The CLI-level guard is
    tests/test_scene_clustering_threshold_v192.py::TestSceneCeiling.
    """

    def test_balanced_semantic_bounds(self):
        from whisperjav.config.legacy import LEGACY_PIPELINES
        sem = LEGACY_PIPELINES["balanced"]["scene_overrides"]["semantic"]
        assert sem == {
            "scene_detection.min_duration": 28.0,
            "scene_detection.max_duration": 240.0,
        }

    def test_balanced_auditok_untouched(self):
        # Owner decision 2026-09-11: semantic first; auditok stays at 20 minutes and
        # keeps its own (_s) parameter names.
        from whisperjav.config.legacy import LEGACY_PIPELINES
        aud = LEGACY_PIPELINES["balanced"]["scene_overrides"]["auditok"]
        assert aud == {
            "scene_detection.max_duration_s": 1200.0,
            "scene_detection.pass1_max_duration_s": 1200.0,
        }
        assert "silero" not in LEGACY_PIPELINES["balanced"]["scene_overrides"]

    def test_fidelity_semantic_floor_and_ceiling(self):
        # Owner decision 2026-09-11 (night): Fidelity gets the same 28 s floor as
        # Balanced, at every sensitivity, on the semantic detector only. auditok is
        # deliberately absent -- its min_duration DISCARDS shorter regions instead of
        # merging them, so a 28 s floor there would drop speech.
        from whisperjav.config.legacy import LEGACY_PIPELINES
        assert LEGACY_PIPELINES["fidelity"]["scene_overrides"] == {
            "semantic": {
                "scene_detection.min_duration": 28.0,
                "scene_detection.max_duration": 240.0,
            },
        }

    def test_fast_declares_no_override(self):
        from whisperjav.config.legacy import LEGACY_PIPELINES
        assert "scene_overrides" not in LEGACY_PIPELINES["fast"]


class TestBalancedDefault:
    def test_default_is_the_built_in_vad(self):
        # Owner N3 (2026-09-05): reversal of CFF3.
        assert BALANCED_DEFAULT_SEGMENTER == "faster-whisper"

    def test_built_in_vad_has_no_external_preset(self):
        # Consequence for the run-outcome contract: under the built-in VAD the
        # recogniser reports its segmenter as "none" -- v1.9.2 builds no external
        # segmenter there at all -- so a zero-cue Balanced file is `empty`, never
        # `suspect`. Nothing corroborates it, because there is no external speech
        # signal to corroborate against.
        assert resolve_segmenter_sensitivity(BALANCED_DEFAULT_SEGMENTER, "balanced") == {}
        assert "none" in PASSTHROUGH_SEGMENTERS

    def test_the_balanced_exemption_constant_is_gone(self):
        """v1.9.2 S2/S9: nothing exempts an external segmenter on Balanced any more."""
        import whisperjav.config.segmenter_presets as sp
        assert not hasattr(sp, "BALANCED_SINGLE_PASS_EXTERNAL")

    def test_firered_and_ten_are_still_real_segmenters_with_presets(self):
        """They keep their presets — they are simply reached through --ensemble now."""
        for name in ("firered-vad", "ten"):
            assert name.lower() not in PASSTHROUGH_SEGMENTERS
            assert resolve_segmenter_sensitivity(name, "balanced")  # non-empty preset

    def test_effective_segmenter_for_pass_is_the_one_rule(self):
        """pass_worker and main.py's start-up model check must agree on which
        segmenter a pass will run, or the check guards the wrong thing."""
        from whisperjav.config.segmenter_presets import (
            BALANCED_DEFAULT_SEGMENTER,
            FIDELITY_DEFAULT_SEGMENTER,
            effective_segmenter_for_pass,
        )
        # Balanced ignores the request entirely (S2/S9).
        assert effective_segmenter_for_pass("balanced", None) == BALANCED_DEFAULT_SEGMENTER
        assert effective_segmenter_for_pass("balanced", "whisperseg") == BALANCED_DEFAULT_SEGMENTER
        # Fidelity falls back, but an explicit choice wins.
        assert effective_segmenter_for_pass("fidelity", None) == FIDELITY_DEFAULT_SEGMENTER
        assert effective_segmenter_for_pass("fidelity", "") == FIDELITY_DEFAULT_SEGMENTER
        assert effective_segmenter_for_pass("fidelity", "whisperseg") == "whisperseg"
        # Everything else takes what it was given, None included.
        for pipe in ("fast", "faster", "qwen", "transformers", "crispasr", None):
            assert effective_segmenter_for_pass(pipe, None) is None
            assert effective_segmenter_for_pass(pipe, "ten") == "ten"

    def test_local_model_dir_accepts_both_shapes_a_user_can_have(self, tmp_path):
        """Upstream's two download commands both write <root>/VAD/. A user may point
        at either the root or the VAD folder; an empty folder is not a model."""
        from whisperjav.modules.speech_segmentation.backends.firered_vad import (
            _VAD_FILES,
            local_model_dir,
        )
        root = tmp_path / "FireRedVAD"
        vad = root / "VAD"
        vad.mkdir(parents=True)
        for name in _VAD_FILES:
            (vad / name).write_bytes(b"x")

        assert local_model_dir(str(root)) == str(vad)
        assert local_model_dir(str(vad)) == str(vad)
        assert local_model_dir(str(tmp_path / "does-not-exist")) is None
        assert local_model_dir(None) is None
        assert local_model_dir("") is None

        # A directory missing one of the two files is not a model directory.
        half = tmp_path / "half" / "VAD"
        half.mkdir(parents=True)
        (half / _VAD_FILES[0]).write_bytes(b"x")
        assert local_model_dir(str(half)) is None

    def test_a_wrong_model_dir_raises_instead_of_downloading(self, tmp_path, monkeypatch):
        """A setting that is silently ignored is worse than one that fails."""
        from whisperjav.modules.speech_segmentation.backends import firered_vad

        def _must_not_run(*a, **k):
            raise AssertionError("downloaded despite an explicit model_dir")

        monkeypatch.setattr(firered_vad, "_snapshot", _must_not_run)
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError) as exc:
            firered_vad.ensure_model_downloaded(str(empty))
        # The message has to tell the user how to get a real one.
        assert "huggingface-cli" in str(exc.value)
        assert "modelscope" in str(exc.value)

    def test_env_var_is_read_when_no_argument_is_given(self, tmp_path, monkeypatch):
        from whisperjav.modules.speech_segmentation.backends import firered_vad
        vad = tmp_path / "FireRedVAD" / "VAD"
        vad.mkdir(parents=True)
        for name in firered_vad._VAD_FILES:
            (vad / name).write_bytes(b"x")
        monkeypatch.setattr(firered_vad, "_snapshot", lambda *a, **k: "SHOULD-NOT-BE-USED")
        monkeypatch.setenv(firered_vad._ENV_MODEL_DIR, str(tmp_path / "FireRedVAD"))
        assert firered_vad.ensure_model_downloaded() == str(vad)

    def test_the_cache_is_consulted_before_the_network(self, monkeypatch):
        """local_files_only=True first, and only a miss reaches the network. A plain
        snapshot_download still contacts the hub even with a full cache; behind a
        firewall that is the #415 stall in miniature."""
        from whisperjav.modules.speech_segmentation.backends import firered_vad
        calls = []

        def _fake(local_files_only):
            calls.append(local_files_only)
            if local_files_only:
                raise OSError("not cached")
            return "downloaded"

        monkeypatch.delenv(firered_vad._ENV_MODEL_DIR, raising=False)
        monkeypatch.setattr(firered_vad, "_snapshot", _fake)
        assert firered_vad.ensure_model_downloaded() == "downloaded"
        assert calls == [True, False], calls

        # And a cache hit must not reach the network at all.
        calls.clear()
        monkeypatch.setattr(firered_vad, "_snapshot",
                            lambda local_files_only: calls.append(local_files_only) or "cached")
        assert firered_vad.ensure_model_downloaded() == "cached"
        assert calls == [True], calls

    def test_model_dir_survives_the_factory_gate(self):
        """The foreign-key gate deletes anything not in the backend's schema."""
        from whisperjav.modules.speech_segmentation.factory import SpeechSegmenterFactory
        out = SpeechSegmenterFactory._sanitize_params(
            "firered-vad",
            {"backend": "firered-vad", "model_dir": "/some/where", "speech_pad_ms": 250},
        )
        assert out["model_dir"] == "/some/where"
        assert "speech_pad_ms" not in out  # still stripped — firered has no such knob

    def test_the_installer_fetches_the_model_during_installation(self):
        """Owner, 2026-09-12: the download belongs at install time, so a user behind a
        firewall needs their proxy once, while installing, not when transcribing."""
        template = (REPO / "installer" / "templates" / "post_install.py.template").read_text(
            encoding="utf-8", errors="replace")
        # The CALL site, at its own indentation — not the comment above it and not
        # the definition. A guard test that matches a comment cannot fail; two did
        # on 2026-09-11.
        # The CALL site at its own indentation, matched as a whole line — not the
        # comment above it and not the definition. A guard test that matches a
        # comment cannot fail; two did on 2026-09-11.
        assert "    fetch_firered_vad_model()" in template.splitlines()
        assert "def fetch_firered_vad_model()" in template
        # It must run in the INSTALLED interpreter and go through WhisperJAV's own
        # resolver, so there is one place that knows the repository.
        assert "ensure_model_downloaded" in template
        assert "patch_hf_hub_downloads" in template  # so the China mirror applies

    def test_the_exemption_is_scoped_to_the_mode_it_was_verified_on(self):
        """It exempted firered-vad on EVERY single-pass mode for a few hours on
        2026-09-12, which stopped `--mode fast --speech-segmenter firered-vad` over a
        model fast never loads. fast/faster run stable_ts with no segmenter at all."""
        from whisperjav.config.segmenter_presets import SINGLE_PASS_EXTERNAL_OK
        assert SINGLE_PASS_EXTERNAL_OK == {"fidelity": frozenset({"firered-vad"})}

    def test_check_reports_without_downloading(self, monkeypatch):
        """A diagnostic must not change the machine it is diagnosing."""
        from whisperjav.modules.speech_segmentation.backends import firered_vad
        calls = []

        def _fake(local_files_only):
            calls.append(local_files_only)
            raise OSError("not cached")

        monkeypatch.delenv(firered_vad._ENV_MODEL_DIR, raising=False)
        monkeypatch.setattr(firered_vad, "_snapshot", _fake)
        with pytest.raises(OSError):
            firered_vad.ensure_model_downloaded(download=False)
        assert calls == [True], "download=False must never reach the network arm"

    def test_building_the_segmenter_without_a_model_raises(self, tmp_path, monkeypatch):
        """THE guarantee. main.py's start-up check has to PREDICT which segmenter a run
        will build and cannot see every path (--mode qwen, the decoupled pipeline, and
        whisper_pro_asr substituting this backend for 'faster-whisper' on a fidelity
        pass after the check has run). The constructor sees what is actually built.

        It has to raise from __init__ specifically: the fidelity pipeline builds the ASR
        outside its per-scene try/except and calls segment() inside it, so a failure on
        the first scene was swallowed per scene and the run produced an empty subtitle
        file at exit 0.
        """
        from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory
        from whisperjav.modules.speech_segmentation.backends import firered_vad

        monkeypatch.delenv(firered_vad._ENV_MODEL_DIR, raising=False)
        monkeypatch.setattr(
            firered_vad, "_snapshot",
            lambda local_files_only: (_ for _ in ()).throw(OSError("no model")),
        )
        # OSError: huggingface_hub's LocalEntryNotFoundError is a FileNotFoundError.
        with pytest.raises(OSError):
            SpeechSegmenterFactory.create("firered-vad", config={"threshold": 0.4})

    def test_only_firered_needs_a_model_fetched_at_start_up(self):
        """The start-up check must be a no-op for every segmenter that ships its
        model, or it would stop runs that have nothing to download."""
        from whisperjav.utils.preflight_check import ensure_segmenter_model_available
        for backend in ("silero-v3.1", "silero-v6.2", "whisperseg", "ten",
                        "faster-whisper", "none", "", None, "made-up"):
            assert ensure_segmenter_model_available(backend, exit_on_fail=False) is True

    def test_segmenter_accepts_knows_who_takes_a_speech_pad(self):
        """The factory drops a parameter a backend's schema does not list, at DEBUG.
        Callers that announce a setting need to know before they announce it."""
        from whisperjav.config.segmenter_presets import segmenter_accepts
        assert segmenter_accepts("silero-v3.1", "speech_pad_ms")
        assert not segmenter_accepts("firered-vad", "speech_pad_ms")
        assert not segmenter_accepts("ten", "speech_pad_ms")
        assert segmenter_accepts("firered-vad", "start_pad_ms")
        # Fails open: not an external segmenter, or not a backend we know.
        for unknown in ("faster-whisper", "none", "", None, "made-up-backend"):
            assert segmenter_accepts(unknown, "speech_pad_ms")


class TestFireRedPresetResolution:
    @pytest.mark.parametrize("sensitivity,threshold,max_speech,end_pad", [
        ("conservative", 0.5, 7, 250),
        ("balanced", 0.4, 6, 150),
        ("aggressive", 0.3, 5, 100),
    ])
    def test_firered_yaml_rows(self, sensitivity, threshold, max_speech, end_pad):
        preset = resolve_segmenter_sensitivity("firered-vad", sensitivity)
        assert preset["threshold"] == threshold
        assert preset["max_speech_duration_s"] == max_speech
        assert preset["end_pad_ms"] == end_pad
        assert set(preset) <= SEGMENTER_PARAMS

    def test_native_and_none_resolve_to_nothing(self):
        assert resolve_segmenter_sensitivity("faster-whisper", "aggressive") == {}
        assert resolve_segmenter_sensitivity("none", "aggressive") == {}


# ─── CLI layer (--dump-params, no model load) ───────────────────────────────

def _dump(tmp_path: Path, *args: str) -> tuple[dict, str]:
    out = tmp_path / f"dump_{abs(hash(args))}.json"
    proc = subprocess.run(
        [sys.executable, "-m", "whisperjav.main", "--dump-params", str(out), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    assert out.exists(), f"dump not written; stderr:\n{proc.stderr[-2000:]}"
    return json.loads(out.read_text(encoding="utf-8")), proc.stdout + proc.stderr


def _segmenter(dump: dict) -> dict:
    return dump["resolved_config"]["params"].get("speech_segmenter") or {}


@pytest.mark.slow
class TestBalancedCliDefaults:
    @pytest.mark.parametrize("sensitivity", ["conservative", "balanced", "aggressive"])
    def test_default_is_built_in_vad_with_native_preset(self, tmp_path, sensitivity):
        dump, _ = _dump(tmp_path, "--mode", "balanced", "--sensitivity", sensitivity)
        assert _segmenter(dump)["backend"] == "faster-whisper"
        # The native faster_whisper_vad preset is kept (scale-correct for the
        # bundled Silero; the firewall mirror only clears it for external backends).
        vad = dump["resolved_config"]["params"]["vad"]
        assert vad["threshold"] > 0
        assert "_dump_note" not in dump["resolved_config"]["params"]

    @pytest.mark.parametrize("segmenter", ["firered-vad", "ten", "whisperseg", "silero-v3.1", "none"])
    def test_balanced_rejects_every_external_segmenter(self, tmp_path, segmenter):
        """S2/S9: the whole option is gone on Balanced, not just the awkward ones.

        Before v1.9.2 these either resolved (firered-vad, ten), silently downgraded to
        silero-v3.1 (whisperseg), or quietly changed the pipeline. Now the run stops.
        """
        out = tmp_path / "never_written.json"
        proc = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "--dump-params", str(out),
             "--mode", "balanced", "--speech-segmenter", segmenter],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        assert proc.returncode == 2
        assert "--vad-version" in (proc.stdout + proc.stderr)
        assert not out.exists()

    def test_the_native_vad_is_not_a_segmenter_choice_any_more(self):
        """'faster-whisper' was only ever meaningful on Balanced, which now rejects it."""
        proc = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "--speech-segmenter", "faster-whisper", "x.wav"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        assert proc.returncode == 2
        assert "invalid choice" in (proc.stdout + proc.stderr)

    def test_ensemble_still_resolves_firered_preset_then_test_d(self, tmp_path):
        """The CFF3 behaviour lives on where it was always fully wired: --ensemble."""
        dump, log = _dump(tmp_path, "--ensemble", "--pass1-pipeline", "fidelity",
                          "--pass1-speech-segmenter", "firered-vad",
                          "--pass1-sensitivity", "aggressive")
        p1 = dump["ensemble_config"]["pass1"]
        assert p1["speech_segmenter"] == "firered-vad"
        assert p1["sensitivity"] == "aggressive"

    def test_fidelity_still_downgrades_an_unwired_segmenter(self, tmp_path):
        dump, log = _dump(tmp_path, "--mode", "fidelity", "--speech-segmenter", "whisperseg")
        assert _segmenter(dump)["backend"] == "silero-v3.1"
        assert "Falling back to silero-v3.1" in log

    def test_cli_overrides_still_win_on_a_mode_that_takes_a_segmenter(self, tmp_path):
        dump, _ = _dump(tmp_path, "--mode", "fidelity", "--speech-segmenter", "silero-v3.1",
                        "--vad-threshold", "0.6")
        assert _segmenter(dump)["threshold"] == 0.6

    def test_fidelity_defaults_to_firered_vad(self, tmp_path):
        """Owner, 2026-09-12. It was silero-v3.1 before that, and an --ensemble
        fidelity pass with no segmenter landed on whisperseg -- the two entry
        points now resolve the same backend."""
        dump, log = _dump(tmp_path, "--mode", "fidelity")
        assert _segmenter(dump)["backend"] == "firered-vad"
        # It must NOT be sent back to silero by the routing guard...
        assert "Falling back to silero-v3.1" not in log
        # ...and it must arrive with its grouping params, which is the whole
        # reason the guard exists. A backend that reached the segmenter without
        # these ran on its own defaults and produced the repetition pathology.
        # --sensitivity defaults to aggressive, so these are the aggressive
        # values from firered-vad-speech-segmentation.yaml.
        seg = _segmenter(dump)
        assert seg["max_group_duration_s"] == 5
        assert seg["chunk_threshold_s"] == 1.0

    @pytest.mark.parametrize("sensitivity,threshold,max_group", [
        ("conservative", 0.5, 7), ("balanced", 0.4, 6), ("aggressive", 0.3, 5),
    ])
    def test_fidelity_firered_preset_follows_sensitivity(
            self, tmp_path, sensitivity, threshold, max_group):
        """The default must track --sensitivity, not sit on one fixed preset."""
        dump, _ = _dump(tmp_path, "--mode", "fidelity", "--sensitivity", sensitivity)
        seg = _segmenter(dump)
        assert seg["backend"] == "firered-vad"
        assert seg["threshold"] == threshold
        assert seg["max_group_duration_s"] == max_group

    def _run_with_empty_model_cache(self, tmp_path, *args):
        """Run the CLI with a Hugging Face cache that has no FireRedVAD model and no
        way to fetch one."""
        env = dict(os.environ)
        env["HF_HOME"] = str(tmp_path / "empty-hf")
        env["HF_HUB_OFFLINE"] = "1"
        env.pop("WHISPERJAV_FIREREDVAD_MODEL_DIR", None)
        return subprocess.run(
            [sys.executable, "-m", "whisperjav.main", *args, "nonexistent.wav"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600, env=env,
        )

    def test_a_fidelity_run_stops_at_start_up_when_the_model_is_missing(self, tmp_path):
        """The wiring itself: main.py must CALL the check, before it does any work."""
        proc = self._run_with_empty_model_cache(tmp_path, "--mode", "fidelity")
        out = proc.stdout + proc.stderr
        assert proc.returncode == 1
        assert "speech-detection model is not on this machine" in out
        assert "silero-v3.1" in out            # names a segmenter that needs no download
        assert "WHISPERJAV_FIREREDVAD_MODEL_DIR" in out  # and the local-directory route
        # It has to stop BEFORE any work: media discovery never runs.
        assert "No valid media files" not in out

    def test_an_ensemble_fidelity_pass_stops_the_same_way(self, tmp_path):
        proc = self._run_with_empty_model_cache(
            tmp_path, "--ensemble", "--pass1-pipeline", "fidelity")
        assert proc.returncode == 1
        assert "speech-detection model is not on this machine" in (proc.stdout + proc.stderr)

    @pytest.mark.parametrize("mode", ["balanced", "fast", "faster"])
    def test_modes_that_do_not_use_the_model_are_not_stopped(self, tmp_path, mode):
        """The false-block guard. These modes never load FireRedVAD."""
        proc = self._run_with_empty_model_cache(tmp_path, "--mode", mode)
        out = proc.stdout + proc.stderr
        assert "speech-detection model is not on this machine" not in out
        assert "No valid media files" in out  # it got all the way to media discovery

    def test_speech_pad_ms_is_not_confirmed_when_it_cannot_be_used(self, tmp_path):
        """Regression guard for the 2026-09-12 default flip: --speech-pad-ms used to
        reach silero-v3.1 on Fidelity and does not reach firered-vad, so the run must
        say it is ignored instead of confirming it was applied."""
        _, log = _dump(tmp_path, "--mode", "fidelity", "--speech-pad-ms", "250")
        assert "is ignored" in log
        assert "Speech pad set via CLI" not in log

    @pytest.mark.parametrize("extra", [
        ("--mode", "fidelity", "--speech-segmenter", "silero-v3.1"),
        ("--mode", "balanced"),
        ("--mode", "fast"),
    ])
    def test_speech_pad_ms_is_still_confirmed_where_it_works(self, tmp_path, extra):
        _, log = _dump(tmp_path, *extra, "--speech-pad-ms", "250")
        assert "Speech pad set via CLI: 250ms" in log
        assert "is ignored" not in log
