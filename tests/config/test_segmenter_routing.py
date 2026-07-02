"""
Tests for v1.9.0 unified segmenter param routing.

Covers whisperjav/config/segmenter_resolution.py and its integration into
resolve_legacy_pipeline(). These lock in the fix for issue #323 (WhisperSeg /
TEN unusable outside --ensemble) and the elimination of the constructor
firewall pattern.

See docs/architecture/SEGMENTER_ROUTING_UNIFICATION_v1.9.md.
"""

import pytest

from whisperjav.config.legacy import resolve_legacy_pipeline
from whisperjav.config.segmenter_resolution import (
    GROUPING_KEYS,
    SEGMENTER_PARAM_KEYS,
    SEGMENTER_TOOL_NAMES,
    apply_segmenter_routing,
    is_silero_backend,
    resolve_segmenter_config,
)


class TestGroupingKeySplit:
    """Grouping keys are moved out of params['vad'] into the canonical location."""

    def test_silero_grouping_moved_to_canonical(self):
        config = resolve_legacy_pipeline("balanced", "balanced", speech_segmenter="silero-v3.1")
        params = config["params"]
        for key in GROUPING_KEYS:
            assert key not in params["vad"], f"{key} must not remain in params['vad']"
            assert key in params["speech_segmenter"], f"{key} missing from canonical location"

    def test_silero_detection_params_stay_in_vad(self):
        config = resolve_legacy_pipeline("balanced", "balanced", speech_segmenter="silero-v3.1")
        vad = config["params"]["vad"]
        for key in ("threshold", "min_speech_duration_ms", "max_speech_duration_s",
                    "min_silence_duration_ms", "speech_pad_ms"):
            assert key in vad, f"Silero detection param {key} missing from params['vad']"

    def test_no_backend_still_moves_grouping(self):
        """Backward compat: resolver without a segmenter arg still canonicalizes."""
        config = resolve_legacy_pipeline("balanced", "balanced")
        params = config["params"]
        for key in GROUPING_KEYS:
            assert key not in params["vad"]
            assert key in params["speech_segmenter"]
        assert "backend" not in params["speech_segmenter"]


class TestNonSileroRouting:
    """Non-Silero backends get their own YAML preset; params['vad'] is emptied."""

    def test_whisperseg_in_balanced_mode(self):
        """The #323 fix: whisperseg works in simple balanced mode."""
        config = resolve_legacy_pipeline("balanced", "aggressive", speech_segmenter="whisperseg")
        params = config["params"]
        assert params["vad"] == {}, "params['vad'] must be empty for non-Silero (no firewall needed)"
        seg = params["speech_segmenter"]
        assert seg["backend"] == "whisperseg"
        # Values from whisperseg-speech-segmentation.yaml aggressive preset -
        # NOT the Silero preset, and NOT the pathological 29s fallback.
        assert seg["max_group_duration_s"] == 5
        assert seg["chunk_threshold_s"] == 1.0
        assert seg["threshold"] == 0.25

    def test_whisperseg_balanced_sensitivity(self):
        config = resolve_legacy_pipeline("balanced", "balanced", speech_segmenter="whisperseg")
        seg = config["params"]["speech_segmenter"]
        # whisperseg YAML spec defaults (balanced preset = spec)
        assert seg["max_group_duration_s"] == 6
        assert seg["chunk_threshold_s"] == 1.0

    def test_ten_in_fidelity_mode(self):
        config = resolve_legacy_pipeline("fidelity", "balanced", speech_segmenter="ten")
        params = config["params"]
        assert params["vad"] == {}
        seg = params["speech_segmenter"]
        assert seg["backend"] == "ten"
        assert "max_group_duration_s" in seg
        assert "hop_size" in seg  # TEN-specific param from its own YAML

    def test_none_backend(self):
        config = resolve_legacy_pipeline("balanced", "balanced", speech_segmenter="none")
        assert config["params"]["speech_segmenter"]["backend"] == "none"


class TestConstructorMergeGuard:
    """The defense-in-depth merge guard semantics (mirrors ASR constructors)."""

    @staticmethod
    def _merge(params):
        vad_params = params["vad"]
        ssc = params.get("speech_segmenter", {})
        backend = ssc.get("backend", "whisperseg")
        if backend.startswith("silero"):
            return {**vad_params, **ssc}
        return dict(ssc)

    def test_silero_backend_receives_detection_and_grouping(self):
        config = resolve_legacy_pipeline("balanced", "balanced", speech_segmenter="silero-v3.1")
        merged = self._merge(config["params"])
        assert merged["threshold"] == config["params"]["vad"]["threshold"]
        assert merged["chunk_threshold_s"] == 2.5

    def test_stale_silero_params_cannot_contaminate_non_silero(self):
        """Direct instantiation with legacy params: guard still protects."""
        merged = self._merge({
            "vad": {"threshold": 0.068, "chunk_threshold_s": 2.5},
            "speech_segmenter": {"backend": "ten", "max_group_duration_s": 6},
        })
        assert "threshold" not in merged
        assert merged["max_group_duration_s"] == 6


class TestSharedDefinitions:
    """pass_worker must share the canonical definitions (no drift)."""

    def test_pass_worker_imports_shared_constants(self):
        # pass_worker transitively imports ASR deps; skip in minimal envs
        pytest.importorskip("stable_whisper")
        from whisperjav.ensemble import pass_worker
        assert pass_worker.SEGMENTER_PARAMS is SEGMENTER_PARAM_KEYS
        assert pass_worker._SEGMENTER_TOOL_NAMES is SEGMENTER_TOOL_NAMES

    def test_grouping_keys_subset_of_segmenter_params(self):
        assert GROUPING_KEYS <= SEGMENTER_PARAM_KEYS

    def test_all_backends_have_tool_mapping(self):
        for backend in ("whisperseg", "ten", "nemo", "silero-v3.1", "whisper-vad"):
            assert backend in SEGMENTER_TOOL_NAMES


class TestHelpers:
    def test_is_silero_backend(self):
        assert is_silero_backend("silero-v3.1")
        assert is_silero_backend("silero")
        assert not is_silero_backend("whisperseg")
        assert not is_silero_backend("none")
        assert not is_silero_backend(None)
        assert not is_silero_backend("")

    def test_resolve_segmenter_config_none(self):
        assert resolve_segmenter_config("none", "balanced") == {}
        assert resolve_segmenter_config("", "balanced") == {}

    def test_resolve_segmenter_config_overrides_win(self):
        cfg = resolve_segmenter_config("whisperseg", "balanced",
                                       {"max_group_duration_s": 9.5})
        assert cfg["max_group_duration_s"] == 9.5

    def test_apply_segmenter_routing_user_override_survives(self):
        """Explicit user values in speech_segmenter beat the backend preset."""
        params = {
            "vad": {"threshold": 0.2, "chunk_threshold_s": 2.5},
            "speech_segmenter": {"max_group_duration_s": 12.0},
        }
        apply_segmenter_routing(params, "whisperseg", "balanced")
        assert params["vad"] == {}
        assert params["speech_segmenter"]["max_group_duration_s"] == 12.0
        assert params["speech_segmenter"]["backend"] == "whisperseg"
