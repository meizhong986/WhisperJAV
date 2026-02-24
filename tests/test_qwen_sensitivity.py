"""
Tests for Qwen pipeline sensitivity preset resolution (E1-E5)
and O1/O2/O3 features (regroup toggle, chunk threshold, inverted padding).

Tests resolve_qwen_sensitivity() and updated DEFAULT_QWEN_PARAMS.
"""

import pytest

from whisperjav.ensemble.pass_worker import (
    DEFAULT_QWEN_PARAMS,
    SEGMENTER_PARAMS,
    resolve_qwen_sensitivity,
    prepare_qwen_params,
)
from whisperjav.modules.subtitle_pipeline.reconstruction import (
    REGROUP_JAV,
    REGROUP_SENTENCE_ONLY,
    REGROUP_VAD_ONLY,
    resolve_regroup,
)
from whisperjav.modules.subtitle_pipeline.types import RegroupMode


# ─── resolve_qwen_sensitivity() ─────────────────────────────────────────


class TestResolveQwenSensitivity:
    """Tests for resolve_qwen_sensitivity()."""

    def test_balanced_silero_v6(self):
        """Balanced sensitivity for silero-v6.2 returns spec defaults."""
        result = resolve_qwen_sensitivity("silero-v6.2", "balanced")
        # Balanced preset is empty {} → uses spec defaults
        assert result["threshold"] == 0.35
        assert result["speech_pad_ms"] == 250
        assert result["min_speech_duration_ms"] == 100

    def test_aggressive_silero_v6(self):
        """Aggressive sensitivity for silero-v6.2 returns lower thresholds, lower padding."""
        result = resolve_qwen_sensitivity("silero-v6.2", "aggressive")
        assert result["threshold"] == 0.2
        assert result["neg_threshold"] == 0.08
        assert result["min_speech_duration_ms"] == 50
        # O3: Inverted padding — aggressive detects early, needs LESS padding
        assert result["speech_pad_ms"] == 150

    def test_conservative_silero_v6(self):
        """Conservative sensitivity for silero-v6.2 returns higher thresholds, higher padding."""
        result = resolve_qwen_sensitivity("silero-v6.2", "conservative")
        assert result["threshold"] == 0.5
        assert result["min_speech_duration_ms"] == 150
        # O3: Inverted padding — conservative detects late, needs MORE padding
        assert result["speech_pad_ms"] == 350

    def test_user_overrides_win(self):
        """User overrides take precedence over sensitivity preset values."""
        result = resolve_qwen_sensitivity(
            "silero-v6.2", "aggressive", {"threshold": 0.3}
        )
        # User override (0.3) should win over aggressive's 0.2
        assert result["threshold"] == 0.3
        # Other aggressive values should still be present
        assert result["neg_threshold"] == 0.08

    def test_none_backend(self):
        """Backend 'none' returns empty dict."""
        result = resolve_qwen_sensitivity("none", "aggressive")
        assert result == {}

    def test_empty_backend(self):
        """Empty backend string returns empty dict."""
        result = resolve_qwen_sensitivity("", "balanced")
        assert result == {}

    def test_unknown_backend_passthrough(self):
        """Unknown backend passes through user overrides only (filtered to SEGMENTER_PARAMS)."""
        result = resolve_qwen_sensitivity(
            "unknown-backend", "balanced", {"threshold": 0.4, "not_a_param": True}
        )
        assert result == {"threshold": 0.4}

    def test_ten_backend(self):
        """TEN backend resolves against TEN YAML presets."""
        result = resolve_qwen_sensitivity("ten", "balanced")
        # TEN has its own spec — just verify we get some valid config back
        assert isinstance(result, dict)
        # All keys should be in SEGMENTER_PARAMS
        for key in result:
            assert key in SEGMENTER_PARAMS, f"Unexpected key: {key}"

    def test_silero_v4_backend(self):
        """Silero v4.0 resolves against silero YAML presets."""
        result = resolve_qwen_sensitivity("silero-v4.0", "balanced")
        assert isinstance(result, dict)
        for key in result:
            assert key in SEGMENTER_PARAMS

    def test_all_keys_in_segmenter_params(self):
        """All returned keys are valid SEGMENTER_PARAMS."""
        for sensitivity in ("aggressive", "balanced", "conservative"):
            result = resolve_qwen_sensitivity("silero-v6.2", sensitivity)
            for key in result:
                assert key in SEGMENTER_PARAMS, (
                    f"Key '{key}' not in SEGMENTER_PARAMS for sensitivity '{sensitivity}'"
                )


# ─── Updated defaults ───────────────────────────────────────────────────


class TestUpdatedDefaults:
    """Tests for updated DEFAULT_QWEN_PARAMS."""

    def test_default_segmenter(self):
        """Default segmenter is silero-v6.2."""
        assert DEFAULT_QWEN_PARAMS["qwen_segmenter"] == "silero-v6.2"

    def test_default_sensitivity(self):
        """Default sensitivity is balanced."""
        assert DEFAULT_QWEN_PARAMS["qwen_sensitivity"] == "balanced"

    def test_default_framer(self):
        """Default framer is vad-grouped."""
        assert DEFAULT_QWEN_PARAMS["qwen_framer"] == "vad-grouped"


class TestPrepareQwenParamsSensitivity:
    """Tests for sensitivity mapping in prepare_qwen_params."""

    def test_sensitivity_mapping_exists(self):
        """prepare_qwen_params maps 'sensitivity' key to 'qwen_sensitivity'."""
        pass_config = {
            "qwen_params": {"sensitivity": "aggressive"},
        }
        result = prepare_qwen_params(pass_config)
        assert result["qwen_sensitivity"] == "aggressive"

    def test_sensitivity_default(self):
        """Default sensitivity is 'balanced' when not overridden."""
        result = prepare_qwen_params({})
        assert result["qwen_sensitivity"] == "balanced"


# ─── O1: RegroupMode and resolve_regroup() ────────────────────────────


class TestRegroupMode:
    """Tests for RegroupMode enum and resolve_regroup()."""

    def test_standard_branch_a(self):
        """Standard + Branch A → REGROUP_JAV."""
        result = resolve_regroup(RegroupMode.STANDARD, is_branch_b=False)
        assert result == REGROUP_JAV

    def test_standard_branch_b(self):
        """Standard + Branch B → REGROUP_VAD_ONLY (no gap heuristics)."""
        result = resolve_regroup(RegroupMode.STANDARD, is_branch_b=True)
        assert result == REGROUP_VAD_ONLY

    def test_sentence_only_branch_a(self):
        """Sentence Only + Branch A → REGROUP_SENTENCE_ONLY (= REGROUP_VAD_ONLY)."""
        result = resolve_regroup(RegroupMode.SENTENCE_ONLY, is_branch_b=False)
        assert result == REGROUP_SENTENCE_ONLY

    def test_sentence_only_branch_b(self):
        """Sentence Only + Branch B → REGROUP_SENTENCE_ONLY (same as standard B)."""
        result = resolve_regroup(RegroupMode.SENTENCE_ONLY, is_branch_b=True)
        assert result == REGROUP_SENTENCE_ONLY

    def test_off_branch_a(self):
        """Off + Branch A → False (no regrouping)."""
        result = resolve_regroup(RegroupMode.OFF, is_branch_b=False)
        assert result is False

    def test_off_branch_b(self):
        """Off + Branch B → False (no regrouping)."""
        result = resolve_regroup(RegroupMode.OFF, is_branch_b=True)
        assert result is False

    def test_sentence_only_is_vad_only_alias(self):
        """REGROUP_SENTENCE_ONLY is an alias for REGROUP_VAD_ONLY."""
        assert REGROUP_SENTENCE_ONLY is REGROUP_VAD_ONLY

    def test_enum_values(self):
        """RegroupMode enum has expected string values."""
        assert RegroupMode.STANDARD.value == "standard"
        assert RegroupMode.SENTENCE_ONLY.value == "sentence_only"
        assert RegroupMode.OFF.value == "off"

    def test_enum_from_string(self):
        """RegroupMode can be constructed from string values."""
        assert RegroupMode("standard") == RegroupMode.STANDARD
        assert RegroupMode("sentence_only") == RegroupMode.SENTENCE_ONLY
        assert RegroupMode("off") == RegroupMode.OFF


# ─── O1: Regroup mode in defaults/mapping ─────────────────────────────


class TestRegroupModeDefaults:
    """Tests for regroup_mode in pass_worker defaults and mapping."""

    def test_default_regroup_mode(self):
        """Default regroup mode is 'standard'."""
        assert DEFAULT_QWEN_PARAMS["qwen_regroup_mode"] == "standard"

    def test_regroup_mode_mapping(self):
        """prepare_qwen_params maps 'regroup_mode' to 'qwen_regroup_mode'."""
        pass_config = {"qwen_params": {"regroup_mode": "sentence_only"}}
        result = prepare_qwen_params(pass_config)
        assert result["qwen_regroup_mode"] == "sentence_only"

    def test_chunk_threshold_mapping(self):
        """prepare_qwen_params maps 'chunk_threshold' to 'qwen_chunk_threshold'."""
        pass_config = {"qwen_params": {"chunk_threshold": 0.5}}
        result = prepare_qwen_params(pass_config)
        assert result["qwen_chunk_threshold"] == 0.5


# ─── O3: Inverted padding relationship ────────────────────────────────


class TestInvertedPadding:
    """Verify padding-sensitivity inversion is correct across all backends."""

    def test_silero_v6_aggressive_less_padding_than_conservative(self):
        """Silero v6.2: aggressive has LESS padding than conservative."""
        aggressive = resolve_qwen_sensitivity("silero-v6.2", "aggressive")
        conservative = resolve_qwen_sensitivity("silero-v6.2", "conservative")
        assert aggressive["speech_pad_ms"] < conservative["speech_pad_ms"]

    def test_silero_v6_balanced_between(self):
        """Silero v6.2: balanced padding is between aggressive and conservative."""
        aggressive = resolve_qwen_sensitivity("silero-v6.2", "aggressive")
        balanced = resolve_qwen_sensitivity("silero-v6.2", "balanced")
        conservative = resolve_qwen_sensitivity("silero-v6.2", "conservative")
        assert aggressive["speech_pad_ms"] < balanced["speech_pad_ms"] < conservative["speech_pad_ms"]

    def test_ten_aggressive_less_padding_than_conservative(self):
        """TEN: aggressive has LESS end_pad_ms than conservative."""
        aggressive = resolve_qwen_sensitivity("ten", "aggressive")
        conservative = resolve_qwen_sensitivity("ten", "conservative")
        if not aggressive or not conservative:
            pytest.skip("TEN YAML not available in ConfigManager")
        assert aggressive["end_pad_ms"] < conservative["end_pad_ms"]

    def test_silero_v4_aggressive_less_padding_than_conservative(self):
        """Silero v4: aggressive has LESS padding than conservative."""
        aggressive = resolve_qwen_sensitivity("silero-v4.0", "aggressive")
        conservative = resolve_qwen_sensitivity("silero-v4.0", "conservative")
        assert aggressive["speech_pad_ms"] < conservative["speech_pad_ms"]
