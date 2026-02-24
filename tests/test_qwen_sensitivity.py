"""
Tests for Qwen pipeline sensitivity preset resolution (E1-E5).

Tests resolve_qwen_sensitivity() and updated DEFAULT_QWEN_PARAMS.
"""

import pytest

from whisperjav.ensemble.pass_worker import (
    DEFAULT_QWEN_PARAMS,
    SEGMENTER_PARAMS,
    resolve_qwen_sensitivity,
    prepare_qwen_params,
)


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
        """Aggressive sensitivity for silero-v6.2 returns lower thresholds."""
        result = resolve_qwen_sensitivity("silero-v6.2", "aggressive")
        assert result["threshold"] == 0.2
        assert result["neg_threshold"] == 0.08
        assert result["min_speech_duration_ms"] == 50
        assert result["speech_pad_ms"] == 350

    def test_conservative_silero_v6(self):
        """Conservative sensitivity for silero-v6.2 returns higher thresholds."""
        result = resolve_qwen_sensitivity("silero-v6.2", "conservative")
        assert result["threshold"] == 0.5
        assert result["min_speech_duration_ms"] == 150
        assert result["speech_pad_ms"] == 150

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
