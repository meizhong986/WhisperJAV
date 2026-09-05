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

    def test_yaml_default_is_18(self):
        from whisperjav.modules.scene_detection_backends.semantic_adapter import SemanticClusteringConfig
        assert SemanticClusteringConfig().clustering_threshold == 18.0

    def test_gui_schema_exposes_the_slider(self):
        # The legacy Customize modal renders from the YAML gui hints.
        from whisperjav.webview_gui.api import WhisperJAVAPI
        api = WhisperJAVAPI.__new__(WhisperJAVAPI)
        schema = WhisperJAVAPI.get_scene_detector_schema(api, "semantic")
        assert schema["success"], schema
        assert "clustering_threshold" in schema["parameters"]
        assert schema["defaults"]["clustering_threshold"] == 18.0

    def test_qwen_schema_exposes_the_slider(self):
        from whisperjav.webview_gui.api import WhisperJAVAPI
        api = WhisperJAVAPI.__new__(WhisperJAVAPI)
        base = WhisperJAVAPI._get_qwen_schema_base(api)
        audio = (base.get("schema") or base)["audio"]
        assert audio["scene_clustering_threshold"]["default"] == 18


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
        dump, log = _dump(tmp_path, "--mode", "balanced", "--scene-clustering-threshold", "10")
        assert "only affects the semantic scene detector" in log
        # Still injected (harmless: auditok/silero ignore it), so switching the
        # method later in the same config picks it up.
        assert dump["resolved_config"]["features"]["scene_detection"]["clustering_threshold"] == 10.0

    def test_absent_flag_leaves_features_untouched(self, tmp_path):
        dump, _ = _dump(tmp_path, "--mode", "balanced", "--scene-detection-method", "semantic")
        assert "clustering_threshold" not in dump["resolved_config"]["features"]["scene_detection"]
