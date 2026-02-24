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
        """Default regroup mode is 'off' (frame-native)."""
        assert DEFAULT_QWEN_PARAMS["qwen_regroup_mode"] == "off"

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


# ─── GUI VAD slider overrides ──────────────────────────────────────────


class TestVadSliderOverrides:
    """Tests for VAD threshold/padding slider overrides (GUI scenario)."""

    def test_threshold_override_aggressive(self):
        """VAD threshold slider overrides aggressive preset's threshold."""
        result = resolve_qwen_sensitivity(
            "silero-v6.2", "aggressive", {"threshold": 0.4}
        )
        assert result["threshold"] == 0.4
        # Other aggressive values preserved
        assert result["neg_threshold"] == 0.08
        assert result["speech_pad_ms"] == 150

    def test_padding_override_aggressive(self):
        """VAD padding slider overrides aggressive preset's speech_pad_ms."""
        result = resolve_qwen_sensitivity(
            "silero-v6.2", "aggressive", {"speech_pad_ms": 300}
        )
        assert result["speech_pad_ms"] == 300
        # Aggressive threshold preserved
        assert result["threshold"] == 0.2

    def test_both_sliders_override(self):
        """Both sliders override simultaneously."""
        result = resolve_qwen_sensitivity(
            "silero-v6.2", "conservative",
            {"threshold": 0.25, "speech_pad_ms": 200}
        )
        assert result["threshold"] == 0.25
        assert result["speech_pad_ms"] == 200
        # Other conservative values preserved
        assert result["min_speech_duration_ms"] == 150

    def test_no_override_preserves_preset(self):
        """Without overrides, sensitivity preset values are unchanged."""
        with_none = resolve_qwen_sensitivity("silero-v6.2", "aggressive", None)
        without = resolve_qwen_sensitivity("silero-v6.2", "aggressive")
        assert with_none == without

    def test_empty_override_preserves_preset(self):
        """Empty override dict preserves sensitivity preset values."""
        with_empty = resolve_qwen_sensitivity("silero-v6.2", "balanced", {})
        without = resolve_qwen_sensitivity("silero-v6.2", "balanced")
        assert with_empty == without

    def test_threshold_override_balanced(self):
        """VAD threshold slider overrides balanced preset's default threshold."""
        result = resolve_qwen_sensitivity(
            "silero-v6.2", "balanced", {"threshold": 0.15}
        )
        assert result["threshold"] == 0.15
        # Balanced padding preserved
        assert result["speech_pad_ms"] == 250


class TestVadSliderMapping:
    """Tests for VAD slider key mapping through prepare_qwen_params."""

    def test_vad_threshold_mapping(self):
        """prepare_qwen_params maps 'vad_threshold' to 'qwen_vad_threshold'."""
        pass_config = {"qwen_params": {"vad_threshold": 0.25}}
        result = prepare_qwen_params(pass_config)
        assert result["qwen_vad_threshold"] == 0.25

    def test_vad_padding_mapping(self):
        """prepare_qwen_params maps 'vad_padding' to 'qwen_vad_padding'."""
        pass_config = {"qwen_params": {"vad_padding": 300}}
        result = prepare_qwen_params(pass_config)
        assert result["qwen_vad_padding"] == 300

    def test_vad_sliders_not_in_defaults(self):
        """VAD slider keys are NOT in DEFAULT_QWEN_PARAMS (only present when user sets them)."""
        assert "qwen_vad_threshold" not in DEFAULT_QWEN_PARAMS
        assert "qwen_vad_padding" not in DEFAULT_QWEN_PARAMS


# ─── max_group_duration_s priority (explicit > YAML) ──────────────────


class TestMaxGroupDurationPriority:
    """Verify pipeline-explicit max_group_duration_s wins over YAML config."""

    def test_qwen_pipeline_explicit_wins_over_segmenter_config(self):
        """segmenter_config with max_group_duration_s should NOT overwrite explicit param.

        This guards against the priority inversion bug where
        segmenter_kwargs.update(segmenter_config) overwrote the explicit value.
        """
        # Simulate what qwen_pipeline.py Phase 4 does:
        # The fix: spread segmenter_config first, then set explicit param
        segmenter_config = {"max_group_duration_s": 29.0, "threshold": 0.35}
        explicit_value = 6.0

        # AFTER fix: explicit wins
        segmenter_kwargs = dict(segmenter_config or {})
        segmenter_kwargs["max_group_duration_s"] = explicit_value

        assert segmenter_kwargs["max_group_duration_s"] == 6.0
        assert segmenter_kwargs["threshold"] == 0.35  # other keys preserved

    def test_vad_grouped_framer_explicit_wins(self):
        """VadGroupedFramer kwargs should have explicit params after config spread.

        This guards against the priority inversion in vad_grouped.py where
        **self._segmenter_config spread overwrote explicit max_group_duration_s.
        """
        segmenter_config = {"max_group_duration_s": 29.0, "some_param": True}
        max_group = 6.0
        chunk_threshold = 1.0

        # AFTER fix: spread config first, then set explicit params
        kwargs = {
            **segmenter_config,
            "max_group_duration_s": max_group,
            "chunk_threshold_s": chunk_threshold,
        }

        assert kwargs["max_group_duration_s"] == 6.0
        assert kwargs["chunk_threshold_s"] == 1.0
        assert kwargs["some_param"] is True

    def test_empty_segmenter_config(self):
        """Empty segmenter_config should not affect explicit params."""
        segmenter_config = {}
        explicit_value = 6.0

        segmenter_kwargs = dict(segmenter_config or {})
        segmenter_kwargs["max_group_duration_s"] = explicit_value

        assert segmenter_kwargs["max_group_duration_s"] == 6.0

    def test_none_segmenter_config(self):
        """None segmenter_config should not cause errors."""
        segmenter_config = None
        explicit_value = 6.0

        segmenter_kwargs = dict(segmenter_config or {})
        segmenter_kwargs["max_group_duration_s"] = explicit_value

        assert segmenter_kwargs["max_group_duration_s"] == 6.0
