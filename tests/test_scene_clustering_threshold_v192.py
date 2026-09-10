"""
v1.9.2 CFF2: the semantic scene detector's clustering threshold is exposed on the
CLI (`--scene-clustering-threshold`, `--qwen-scene-clustering-threshold`) and in
the Ensemble-tab Customize modal.

The knob itself (`clustering_threshold`, Ward distance; lower = more scenes) was
already a constructor kwarg all the way down to the vendored engine; nothing above
the factory set it. These tests cover the new plumbing without loading any model.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _dump(tmp_path: Path, *args: str) -> tuple[dict, str]:
    out = tmp_path / f"dump_{abs(hash(args))}.json"
    proc = subprocess.run(
        [sys.executable, "-m", "whisperjav.main", "--dump-params", str(out), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    assert out.exists(), f"dump not written; stderr:\n{proc.stderr[-2000:]}"
    return json.loads(out.read_text(encoding="utf-8")), proc.stdout + proc.stderr


class TestBackendAcceptsTheKnob:
    def test_semantic_backend_reads_clustering_threshold(self):
        from whisperjav.modules.scene_detection_backends.semantic_backend import SemanticSceneDetector
        det = SemanticSceneDetector(clustering_threshold=10)
        assert det._adapter.config.clustering_threshold == 10.0

    def test_engine_fallback_matches_the_shipped_value(self):
        # v1.9.2 (owner O1): 22.0 / 6.0 everywhere, including the engine's own
        # last-resort fallback, so a caller that supplies nothing cannot run on a
        # value the product no longer ships.
        from whisperjav.modules.scene_detection_backends.semantic_adapter import SemanticClusteringConfig
        assert SemanticClusteringConfig().clustering_threshold == 22.0
        assert SemanticClusteringConfig().snap_window == 6.0

    def test_gui_schema_exposes_the_slider(self):
        # The legacy Customize modal renders from the YAML gui hints.
        from whisperjav.webview_gui.api import WhisperJAVAPI
        api = WhisperJAVAPI.__new__(WhisperJAVAPI)
        schema = WhisperJAVAPI.get_scene_detector_schema(api, "semantic")
        assert schema["success"], schema
        assert "clustering_threshold" in schema["parameters"]
        assert schema["defaults"]["clustering_threshold"] == 22.0
        assert schema["defaults"]["snap_window"] == 6.0

    def test_no_sensitivity_preset_overrides_the_uniform_values(self):
        # v1.9.2 (owner O1): the two values are uniform, so no sensitivity preset --
        # in the Pydantic component or in the tool YAML -- may carry either key.
        from whisperjav.config.components.features.scene_detection import SemanticSceneDetection
        from whisperjav.webview_gui.api import WhisperJAVAPI

        for name, preset in SemanticSceneDetection.presets.items():
            assert preset.clustering_threshold == 22.0, name
            assert preset.snap_window == 6.0, name

        api = WhisperJAVAPI.__new__(WhisperJAVAPI)
        yaml_presets = WhisperJAVAPI.get_scene_detector_schema(api, "semantic")["presets"]
        for name in ("conservative", "balanced", "aggressive"):
            assert "clustering_threshold" not in yaml_presets[name], name
            assert "snap_window" not in yaml_presets[name], name

    def test_qwen_schema_exposes_the_slider(self):
        # v1.9.2 (owner O1): 22.0 is the one product-wide value, so the Qwen panel must
        # show it too. Qwen leaves the knob unset by default (qwen_pipeline.py:673-674),
        # which falls through to the engine default -- so a panel still saying 18 would
        # have been displaying a number no run would use.
        from whisperjav.webview_gui.api import WhisperJAVAPI
        api = WhisperJAVAPI.__new__(WhisperJAVAPI)
        base = WhisperJAVAPI._get_qwen_schema_base(api)
        audio = (base.get("schema") or base)["audio"]
        assert audio["scene_clustering_threshold"]["default"] == 22


@pytest.mark.slow
class TestCli:
    def test_flags_parse(self):
        for flag in ("--scene-clustering-threshold", "--qwen-scene-clustering-threshold"):
            proc = subprocess.run(
                [sys.executable, "-m", "whisperjav.main", flag, "10", "--help"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
            )
            assert proc.returncode == 0, proc.stderr[-500:]
            assert flag in proc.stdout

    def test_legacy_threshold_reaches_features(self, tmp_path):
        dump, log = _dump(tmp_path, "--mode", "balanced", "--scene-detection-method", "semantic",
                          "--scene-clustering-threshold", "10")
        sd = dump["resolved_config"]["features"]["scene_detection"]
        assert sd["method"] == "semantic"
        assert sd["clustering_threshold"] == 10.0
        assert dump["cli_args"]["scene_clustering_threshold"] == 10.0
        assert "only affects the semantic" not in log

    def test_threshold_without_semantic_warns(self, tmp_path):
        # v1.9.2: semantic is the DEFAULT scene detector, so the warning is now keyed to
        # an explicitly non-semantic choice rather than to "no --scene-detection-method".
        dump, log = _dump(tmp_path, "--mode", "balanced",
                          "--scene-detection-method", "auditok",
                          "--scene-clustering-threshold", "10")
        assert "only affects the semantic scene detector" in log
        # Still injected (harmless: auditok/silero ignore it), so switching the
        # method later in the same config picks it up.
        assert dump["resolved_config"]["features"]["scene_detection"]["clustering_threshold"] == 10.0

    def test_threshold_with_the_default_detector_does_not_warn(self, tmp_path):
        # The v1.9.2 default is semantic, so the knob applies and must not warn.
        dump, log = _dump(tmp_path, "--mode", "balanced", "--scene-clustering-threshold", "10")
        assert "only affects the semantic" not in log
        sd = dump["resolved_config"]["features"]["scene_detection"]
        assert sd["method"] == "semantic"
        assert sd["clustering_threshold"] == 10.0

    def test_absent_flag_leaves_the_uniform_value_on_every_sensitivity(self, tmp_path):
        # Before v1.9.2 the resolved feature was auditok's, which has no such field, so
        # "absent" meant "key missing". The semantic component declares the parameter, so
        # absence now means "still the shipped value, not an injected one".
        #
        # v1.9.2 (owner O1): that shipped value is the SAME on every sensitivity --
        # clustering_threshold 22.0 and snap_window 6.0. Only the scene-length bounds
        # still vary. This is the end-to-end guard for the uniformity requirement.
        for sensitivity in ("conservative", "balanced", "aggressive"):
            dump, _ = _dump(tmp_path, "--mode", "balanced", "--sensitivity", sensitivity,
                            "--scene-detection-method", "semantic")
            sd = dump["resolved_config"]["features"]["scene_detection"]
            assert sd["clustering_threshold"] == 22.0, sensitivity
            assert sd["snap_window"] == 6.0, sensitivity


class TestSceneCeiling:
    """v1.9.2 (owner, 2026-09-11): the semantic scene ceiling is 240 s on Balanced and
    Fidelity, at every sensitivity. Balanced keeps its 28 s floor; Fidelity keeps the
    semantic presets' floors. auditok on Balanced is deliberately untouched (semantic
    first; other backends after user feedback), and its parameter names must stay
    separate -- a bare ``min_duration`` reaching auditok would DISCARD every region
    under 28 s.
    """

    @pytest.mark.parametrize("sensitivity", ["conservative", "balanced", "aggressive"])
    def test_balanced_resolves_28_and_240(self, tmp_path, sensitivity):
        dump, _ = _dump(tmp_path, "--mode", "balanced", "--sensitivity", sensitivity)
        sd = dump["resolved_config"]["features"]["scene_detection"]
        assert sd["method"] == "semantic"
        assert sd["min_duration"] == 28.0
        assert sd["max_duration"] == 240.0

    @pytest.mark.parametrize("sensitivity,floor", [
        ("conservative", 30.0), ("balanced", 20.0), ("aggressive", 10.0),
    ])
    def test_fidelity_resolves_240_and_keeps_its_floor(self, tmp_path, sensitivity, floor):
        dump, _ = _dump(tmp_path, "--mode", "fidelity", "--sensitivity", sensitivity)
        sd = dump["resolved_config"]["features"]["scene_detection"]
        assert sd["method"] == "semantic"
        assert sd["max_duration"] == 240.0
        assert sd["min_duration"] == floor

    def test_fast_is_unchanged(self, tmp_path):
        dump, _ = _dump(tmp_path, "--mode", "fast", "--sensitivity", "aggressive")
        sd = dump["resolved_config"]["features"]["scene_detection"]
        assert sd["method"] == "semantic"
        assert sd["max_duration"] == 180.0
        assert sd["min_duration"] == 10.0

    def test_balanced_auditok_keeps_its_own_ceiling_and_names(self, tmp_path):
        dump, _ = _dump(tmp_path, "--mode", "balanced", "--scene-detection-method", "auditok")
        sd = dump["resolved_config"]["features"]["scene_detection"]
        assert sd["method"] == "auditok"
        assert sd["max_duration_s"] == 1200.0
        assert sd["pass1_max_duration_s"] == 1200.0
        assert "max_duration" not in sd
        assert "min_duration" not in sd
