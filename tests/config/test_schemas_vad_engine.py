"""
Tests for VAD, Engine, and JAV configuration schemas.
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from whisperjav.config.schemas import (
    FasterWhisperEngineOptions,
    FasterWhisperVADOptions,
    JAVAudioConfig,
    OpenAIWhisperEngineOptions,
    SileroVADOptions,
    StableTSEngineOptions,
    StableTSVADOptions,
)


class TestSileroVADOptions:
    """Test SileroVADOptions schema."""

    def test_valid_balanced_preset(self):
        """Test balanced preset values are valid."""
        config = SileroVADOptions(
            threshold=0.18,
            min_speech_duration_ms=100,
            max_speech_duration_s=11,
            min_silence_duration_ms=300,
            neg_threshold=0.15,
            speech_pad_ms=400
        )
        assert config.threshold == 0.18

    def test_threshold_bounds(self):
        """Test threshold must be 0.0-1.0."""
        with pytest.raises(PydanticValidationError):
            SileroVADOptions(
                threshold=1.5,
                min_speech_duration_ms=100,
                max_speech_duration_s=11,
                min_silence_duration_ms=300,
                neg_threshold=0.15,
                speech_pad_ms=400
            )

        with pytest.raises(PydanticValidationError):
            SileroVADOptions(
                threshold=-0.1,
                min_speech_duration_ms=100,
                max_speech_duration_s=11,
                min_silence_duration_ms=300,
                neg_threshold=0.15,
                speech_pad_ms=400
            )

    def test_negative_duration_rejected(self):
        """Test negative durations are rejected."""
        with pytest.raises(PydanticValidationError):
            SileroVADOptions(
                threshold=0.18,
                min_speech_duration_ms=-100,
                max_speech_duration_s=11,
                min_silence_duration_ms=300,
                neg_threshold=0.15,
                speech_pad_ms=400
            )

    def test_max_speech_duration_must_be_positive(self):
        """Test max_speech_duration_s must be > 0."""
        with pytest.raises(PydanticValidationError):
            SileroVADOptions(
                threshold=0.18,
                min_speech_duration_ms=100,
                max_speech_duration_s=0,
                min_silence_duration_ms=300,
                neg_threshold=0.15,
                speech_pad_ms=400
            )


class TestFasterWhisperVADOptions:
    """Test FasterWhisperVADOptions schema."""

    def test_valid_config(self):
        """Test valid config with vad_filter."""
        config = FasterWhisperVADOptions(vad_filter=True)
        assert config.vad_filter is True

    def test_with_parameters(self):
        """Test with additional vad_parameters."""
        config = FasterWhisperVADOptions(
            vad_filter=True,
            vad_parameters={"threshold": 0.5}
        )
        assert config.vad_parameters["threshold"] == 0.5


class TestStableTSVADOptions:
    """Test StableTSVADOptions schema."""

    def test_valid_config(self):
        """Test valid stable-ts VAD config."""
        config = StableTSVADOptions(
            vad=True,
            vad_threshold=0.35
        )
        assert config.vad is True
        assert config.vad_threshold == 0.35

    def test_threshold_bounds(self):
        """Test vad_threshold must be 0.0-1.0."""
        with pytest.raises(PydanticValidationError):
            StableTSVADOptions(vad=True, vad_threshold=1.5)


class TestFasterWhisperEngineOptions:
    """Test FasterWhisperEngineOptions schema."""

    def test_valid_balanced_preset(self):
        """Test balanced preset values."""
        config = FasterWhisperEngineOptions(
            repetition_penalty=1.5,
            no_repeat_ngram_size=2
        )
        assert config.repetition_penalty == 1.5
        assert config.no_repeat_ngram_size == 2

    def test_repetition_penalty_bounds(self):
        """Test repetition_penalty must be 1.0-2.0."""
        with pytest.raises(PydanticValidationError):
            FasterWhisperEngineOptions(
                repetition_penalty=0.5,
                no_repeat_ngram_size=2
            )

        with pytest.raises(PydanticValidationError):
            FasterWhisperEngineOptions(
                repetition_penalty=2.5,
                no_repeat_ngram_size=2
            )

    def test_no_repeat_ngram_size_non_negative(self):
        """Test no_repeat_ngram_size must be >= 0."""
        with pytest.raises(PydanticValidationError):
            FasterWhisperEngineOptions(
                repetition_penalty=1.5,
                no_repeat_ngram_size=-1
            )

    def test_optional_fields_default_to_none(self):
        """Test optional fields default to None."""
        config = FasterWhisperEngineOptions(
            repetition_penalty=1.5,
            no_repeat_ngram_size=2
        )
        assert config.chunk_length is None
        assert config.hotwords is None
        assert config.max_new_tokens is None


class TestOpenAIWhisperEngineOptions:
    """Test OpenAIWhisperEngineOptions schema."""

    def test_default_values(self):
        """Test default values."""
        config = OpenAIWhisperEngineOptions()
        assert config.fp16 is True
        assert config.hallucination_silence_threshold is None

    def test_custom_values(self):
        """Test custom values."""
        config = OpenAIWhisperEngineOptions(
            fp16=False,
            hallucination_silence_threshold=0.5
        )
        assert config.fp16 is False
        assert config.hallucination_silence_threshold == 0.5


class TestStableTSEngineOptions:
    """Test StableTSEngineOptions schema (29 parameters)."""

    def test_default_values(self):
        """Test default values are correct."""
        config = StableTSEngineOptions()
        assert config.suppress_ts_tokens is False
        assert config.gap_padding == " ..."
        assert config.only_ffmpeg is False
        assert config.max_instant_words == 0.5
        assert config.nonspeech_error == 0.1
        assert config.ignore_compatibility is True
        assert config.only_voice_freq is False
        assert config.regroup is True
        assert config.ts_num == 0
        assert config.suppress_silence is True
        assert config.suppress_word_ts is True
        assert config.suppress_attention is False
        assert config.use_word_position is True
        assert config.q_levels == 20
        assert config.k_size == 5
        assert config.demucs is False

    def test_all_optional_fields_can_be_set(self):
        """Test all optional fields can be set."""
        config = StableTSEngineOptions(
            stream=True,
            mel_first=True,
            avg_prob_threshold=0.5,
            nonspeech_skip=0.1,
            min_word_dur=0.05,
            min_silence_dur=0.5,
            ts_noise=0.1,
            time_scale=1.5,
            denoiser="spectral",
            denoiser_options={"strength": 0.5},
            demucs=True,
            demucs_options={"model": "htdemucs"}
        )
        assert config.stream is True
        assert config.demucs is True
        assert config.denoiser_options["strength"] == 0.5

    def test_model_dump_without_none_removes_optionals(self):
        """Test None values are removed from dump."""
        config = StableTSEngineOptions()
        result = config.model_dump_without_none()

        # Optional fields with None should not be present
        assert "stream" not in result
        assert "mel_first" not in result
        assert "avg_prob_threshold" not in result

        # Default values should be present
        assert result["regroup"] is True
        assert result["q_levels"] == 20


class TestJAVAudioConfig:
    """Test JAVAudioConfig schema."""

    def test_default_values(self):
        """Test default values are JAV-optimized."""
        config = JAVAudioConfig()
        assert config.background_noise_profile == "high"
        assert config.overlapping_dialogue_strategy == "split"
        assert config.quiet_speech_boost is True
        assert config.moaning_filter_enabled is True
        assert config.scene_transition_detection is True

    def test_get_vad_adjustments_high_noise(self):
        """Test VAD adjustments for high noise profile."""
        config = JAVAudioConfig(background_noise_profile="high")
        adjustments = config.get_vad_adjustments()

        assert "threshold" in adjustments
        assert adjustments["threshold"] < 0  # Lower threshold
        assert "speech_pad_ms" in adjustments

    def test_get_vad_adjustments_low_noise(self):
        """Test VAD adjustments for low noise profile."""
        config = JAVAudioConfig(
            background_noise_profile="low",
            quiet_speech_boost=False
        )
        adjustments = config.get_vad_adjustments()

        # No adjustments needed for low noise without boost
        assert len(adjustments) == 0

    def test_get_vad_adjustments_quiet_boost(self):
        """Test VAD adjustments with quiet speech boost."""
        config = JAVAudioConfig(
            background_noise_profile="low",
            quiet_speech_boost=True
        )
        adjustments = config.get_vad_adjustments()

        assert "min_speech_duration_ms" in adjustments
        assert "neg_threshold" in adjustments

    def test_invalid_noise_profile(self):
        """Test invalid noise profile is rejected."""
        with pytest.raises(PydanticValidationError):
            JAVAudioConfig(background_noise_profile="invalid")

    def test_invalid_dialogue_strategy(self):
        """Test invalid dialogue strategy is rejected."""
        with pytest.raises(PydanticValidationError):
            JAVAudioConfig(overlapping_dialogue_strategy="invalid")
