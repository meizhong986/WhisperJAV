"""
Tests for presets - verify values match asr_config.json.
"""

import json
from pathlib import Path

import pytest

from whisperjav.config.schemas import (
    DECODER_PRESETS,
    FASTER_WHISPER_ENGINE_PRESETS,
    HALLUCINATION_THRESHOLDS,
    SILERO_VAD_PRESETS,
    STABLE_TS_ENGINE_OPTIONS,
    STABLE_TS_VAD_PRESETS,
    TRANSCRIBER_PRESETS,
    Sensitivity,
    get_decoder_preset,
    get_faster_whisper_engine_preset,
    get_silero_vad_preset,
    get_stable_ts_vad_preset,
    get_transcriber_preset,
)


@pytest.fixture
def asr_config():
    """Load asr_config.json for comparison."""
    config_path = Path(__file__).parent.parent.parent / "whisperjav" / "config" / "asr_config.json"
    with open(config_path) as f:
        return json.load(f)


class TestTranscriberPresets:
    """Test TranscriberOptions presets match asr_config.json."""

    def test_balanced_values(self, asr_config):
        """Test balanced preset matches JSON."""
        preset = TRANSCRIBER_PRESETS[Sensitivity.BALANCED]
        json_values = asr_config["common_transcriber_options"]["balanced"]

        assert preset.temperature == json_values["temperature"]
        assert preset.compression_ratio_threshold == json_values["compression_ratio_threshold"]
        assert preset.logprob_threshold == json_values["logprob_threshold"]
        assert preset.no_speech_threshold == json_values["no_speech_threshold"]
        assert preset.word_timestamps == json_values["word_timestamps"]

    def test_conservative_values(self, asr_config):
        """Test conservative preset matches JSON."""
        preset = TRANSCRIBER_PRESETS[Sensitivity.CONSERVATIVE]
        json_values = asr_config["common_transcriber_options"]["conservative"]

        assert preset.temperature == json_values["temperature"]
        assert preset.no_speech_threshold == json_values["no_speech_threshold"]
        assert preset.logprob_threshold == json_values["logprob_threshold"]

    def test_aggressive_values(self, asr_config):
        """Test aggressive preset matches JSON."""
        preset = TRANSCRIBER_PRESETS[Sensitivity.AGGRESSIVE]
        json_values = asr_config["common_transcriber_options"]["aggressive"]

        assert preset.temperature == json_values["temperature"]
        assert preset.compression_ratio_threshold == json_values["compression_ratio_threshold"]
        assert preset.no_speech_threshold == json_values["no_speech_threshold"]

    def test_getter_function(self):
        """Test get_transcriber_preset returns correct preset."""
        preset = get_transcriber_preset(Sensitivity.AGGRESSIVE)
        assert preset == TRANSCRIBER_PRESETS[Sensitivity.AGGRESSIVE]


class TestDecoderPresets:
    """Test DecoderOptions presets match asr_config.json."""

    def test_balanced_values(self, asr_config):
        """Test balanced preset matches JSON."""
        preset = DECODER_PRESETS[Sensitivity.BALANCED]
        json_values = asr_config["common_decoder_options"]["balanced"]

        assert preset.beam_size == json_values["beam_size"]
        assert preset.patience == json_values["patience"]
        assert preset.suppress_blank == json_values["suppress_blank"]

    def test_conservative_values(self, asr_config):
        """Test conservative preset matches JSON."""
        preset = DECODER_PRESETS[Sensitivity.CONSERVATIVE]
        json_values = asr_config["common_decoder_options"]["conservative"]

        assert preset.beam_size == json_values["beam_size"]
        assert preset.patience == json_values["patience"]

    def test_aggressive_values(self, asr_config):
        """Test aggressive preset matches JSON."""
        preset = DECODER_PRESETS[Sensitivity.AGGRESSIVE]
        json_values = asr_config["common_decoder_options"]["aggressive"]

        assert preset.beam_size == json_values["beam_size"]
        assert preset.patience == json_values["patience"]
        assert preset.suppress_blank == json_values["suppress_blank"]
        assert preset.suppress_tokens == json_values["suppress_tokens"]

    def test_getter_function(self):
        """Test get_decoder_preset returns correct preset."""
        preset = get_decoder_preset(Sensitivity.BALANCED)
        assert preset == DECODER_PRESETS[Sensitivity.BALANCED]


class TestSileroVADPresets:
    """Test SileroVADOptions presets from v4 YAML configuration."""

    @pytest.fixture
    def silero_tool(self):
        """Load Silero speech segmentation tool from v4 YAML."""
        from whisperjav.config.v4.registries.tool_registry import get_tool_registry
        registry = get_tool_registry()
        return registry.get("silero-speech-segmentation")

    def test_balanced_values(self, silero_tool):
        """Test balanced preset values from v4 YAML."""
        # Balanced uses spec defaults
        spec = silero_tool.spec

        assert spec["vad.threshold"] == 0.18
        assert spec["vad.min_speech_duration_ms"] == 100
        assert spec["vad.max_speech_duration_s"] == 11.0
        assert spec["vad.speech_pad_ms"] == 400

    def test_conservative_values(self, silero_tool):
        """Test conservative preset values from v4 YAML."""
        config = silero_tool.get_resolved_config("conservative")

        assert config["vad.threshold"] == 0.35
        assert config["vad.neg_threshold"] == 0.3
        assert config["vad.min_speech_duration_ms"] == 150
        assert config["vad.max_speech_duration_s"] == 9.0

    def test_aggressive_values(self, silero_tool):
        """Test aggressive preset values from v4 YAML."""
        config = silero_tool.get_resolved_config("aggressive")

        assert config["vad.threshold"] == 0.05
        assert config["vad.neg_threshold"] == 0.1
        assert config["vad.min_speech_duration_ms"] == 30
        assert config["vad.speech_pad_ms"] == 600

    def test_legacy_getter_function(self):
        """Test legacy get_silero_vad_preset still works."""
        # Legacy function still available for backward compatibility
        preset = get_silero_vad_preset(Sensitivity.CONSERVATIVE)
        assert preset == SILERO_VAD_PRESETS[Sensitivity.CONSERVATIVE]

    def test_v4_presets_available(self, silero_tool):
        """Test all expected presets exist in v4 YAML."""
        assert "conservative" in silero_tool.presets
        assert "balanced" in silero_tool.presets
        assert "aggressive" in silero_tool.presets


class TestStableTSVADPresets:
    """Test StableTSVADOptions presets match asr_config.json."""

    def test_balanced_values(self, asr_config):
        """Test balanced preset matches JSON."""
        preset = STABLE_TS_VAD_PRESETS[Sensitivity.BALANCED]
        json_values = asr_config["stable_ts_vad_options"]["balanced"]

        assert preset.vad == json_values["vad"]
        assert preset.vad_threshold == json_values["vad_threshold"]

    def test_conservative_values(self, asr_config):
        """Test conservative preset matches JSON."""
        preset = STABLE_TS_VAD_PRESETS[Sensitivity.CONSERVATIVE]
        json_values = asr_config["stable_ts_vad_options"]["conservative"]

        assert preset.vad_threshold == json_values["vad_threshold"]

    def test_aggressive_values(self, asr_config):
        """Test aggressive preset matches JSON."""
        preset = STABLE_TS_VAD_PRESETS[Sensitivity.AGGRESSIVE]
        json_values = asr_config["stable_ts_vad_options"]["aggressive"]

        assert preset.vad_threshold == json_values["vad_threshold"]

    def test_getter_function(self):
        """Test get_stable_ts_vad_preset returns correct preset."""
        preset = get_stable_ts_vad_preset(Sensitivity.AGGRESSIVE)
        assert preset == STABLE_TS_VAD_PRESETS[Sensitivity.AGGRESSIVE]


class TestFasterWhisperEnginePresets:
    """Test FasterWhisperEngineOptions presets match asr_config.json."""

    def test_balanced_values(self, asr_config):
        """Test balanced preset matches JSON."""
        preset = FASTER_WHISPER_ENGINE_PRESETS[Sensitivity.BALANCED]
        json_values = asr_config["faster_whisper_engine_options"]["balanced"]

        assert preset.repetition_penalty == json_values["repetition_penalty"]
        # JSON has 2.0 but we use int 2
        assert preset.no_repeat_ngram_size == int(json_values["no_repeat_ngram_size"])

    def test_conservative_values(self, asr_config):
        """Test conservative preset matches JSON."""
        preset = FASTER_WHISPER_ENGINE_PRESETS[Sensitivity.CONSERVATIVE]
        json_values = asr_config["faster_whisper_engine_options"]["conservative"]

        assert preset.repetition_penalty == json_values["repetition_penalty"]

    def test_aggressive_values(self, asr_config):
        """Test aggressive preset matches JSON."""
        preset = FASTER_WHISPER_ENGINE_PRESETS[Sensitivity.AGGRESSIVE]
        json_values = asr_config["faster_whisper_engine_options"]["aggressive"]

        assert preset.chunk_length == json_values["chunk_length"]
        assert preset.repetition_penalty == json_values["repetition_penalty"]

    def test_getter_function(self):
        """Test get_faster_whisper_engine_preset returns correct preset."""
        preset = get_faster_whisper_engine_preset(Sensitivity.BALANCED)
        assert preset == FASTER_WHISPER_ENGINE_PRESETS[Sensitivity.BALANCED]


class TestStableTSEngineOptions:
    """Test StableTSEngineOptions values match asr_config.json."""

    def test_default_values(self, asr_config):
        """Test stable-ts engine options match JSON (same across sensitivities)."""
        json_values = asr_config["stable_ts_engine_options"]["balanced"]

        assert STABLE_TS_ENGINE_OPTIONS.gap_padding == json_values["gap_padding"]
        assert STABLE_TS_ENGINE_OPTIONS.regroup == json_values["regroup"]
        assert STABLE_TS_ENGINE_OPTIONS.suppress_silence == json_values["suppress_silence"]
        assert STABLE_TS_ENGINE_OPTIONS.q_levels == json_values["q_levels"]
        assert STABLE_TS_ENGINE_OPTIONS.k_size == json_values["k_size"]


class TestHallucinationThresholds:
    """Test hallucination threshold values match asr_config.json."""

    def test_values(self, asr_config):
        """Test hallucination thresholds match JSON."""
        json_values = asr_config["exclusive_whisper_plus_faster_whisper"]

        assert HALLUCINATION_THRESHOLDS[Sensitivity.BALANCED] == json_values["balanced"]["hallucination_silence_threshold"]
        assert HALLUCINATION_THRESHOLDS[Sensitivity.CONSERVATIVE] == json_values["conservative"]["hallucination_silence_threshold"]
        assert HALLUCINATION_THRESHOLDS[Sensitivity.AGGRESSIVE] == json_values["aggressive"]["hallucination_silence_threshold"]


class TestPresetExport:
    """Test presets can be exported without None values."""

    def test_transcriber_export(self):
        """Test transcriber preset exports correctly."""
        preset = TRANSCRIBER_PRESETS[Sensitivity.BALANCED]
        result = preset.model_dump_without_none()

        # Should not contain None values
        assert "initial_prompt" not in result
        assert "prepend_punctuations" not in result

        # Should contain actual values
        assert result["temperature"] == [0.0, 0.1]
        assert result["word_timestamps"] is True

    def test_decoder_export(self):
        """Test decoder preset exports correctly."""
        preset = DECODER_PRESETS[Sensitivity.AGGRESSIVE]
        result = preset.model_dump_without_none()

        # Should contain empty list, not None
        assert "suppress_tokens" in result
        assert result["suppress_tokens"] == []

        # Should not contain None values
        assert "length_penalty" not in result

    def test_vad_export(self):
        """Test VAD preset exports all values."""
        preset = SILERO_VAD_PRESETS[Sensitivity.AGGRESSIVE]
        result = preset.model_dump_without_none()

        # All VAD params are required, so all should be present
        assert result["threshold"] == 0.05
        assert result["min_speech_duration_ms"] == 30
        assert result["speech_pad_ms"] == 600
