"""
Tests for core configuration schemas: Model, Transcriber, Decoder.
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from whisperjav.config.schemas import (
    DecoderOptions,
    ModelConfig,
    MODELS,
    PerformanceMetrics,
    TranscriberOptions,
    VADEngineConfig,
    VAD_ENGINES,
)


class TestModelConfig:
    """Test ModelConfig schema."""

    def test_valid_model(self):
        """Test creating a valid model config."""
        config = ModelConfig(
            provider="faster_whisper",
            model_name="large-v2",
            compute_type="int8_float16",
            supported_tasks=["transcribe", "translate"]
        )
        assert config.provider == "faster_whisper"
        assert config.model_name == "large-v2"

    def test_invalid_provider(self):
        """Test invalid provider is rejected."""
        with pytest.raises(PydanticValidationError):
            ModelConfig(
                provider="invalid_provider",
                model_name="test",
                compute_type="float16",
                supported_tasks=["transcribe"]
            )

    def test_invalid_compute_type(self):
        """Test invalid compute type is rejected."""
        with pytest.raises(PydanticValidationError):
            ModelConfig(
                provider="faster_whisper",
                model_name="test",
                compute_type="invalid",
                supported_tasks=["transcribe"]
            )

    def test_invalid_task(self):
        """Test invalid task in supported_tasks is rejected."""
        with pytest.raises(PydanticValidationError):
            ModelConfig(
                provider="faster_whisper",
                model_name="test",
                compute_type="float16",
                supported_tasks=["invalid_task"]
            )

    def test_predefined_models_exist(self):
        """Test all predefined models are valid."""
        assert len(MODELS) == 4
        assert "whisper-turbo" in MODELS
        assert "whisper-large-v2" in MODELS
        assert "faster-whisper-large-v2-int8" in MODELS
        assert "faster-whisper-large-v3" in MODELS

    def test_predefined_model_values(self):
        """Test predefined model values are correct."""
        model = MODELS["faster-whisper-large-v2-int8"]
        assert model.provider == "faster_whisper"
        assert model.model_name == "large-v2"
        assert model.compute_type == "int8_float16"
        assert "transcribe" in model.supported_tasks
        assert "translate" in model.supported_tasks


class TestVADEngineConfig:
    """Test VADEngineConfig schema."""

    def test_valid_vad_engine(self):
        """Test creating a valid VAD engine config."""
        config = VADEngineConfig(
            provider="silero",
            repo="snakers4/silero-vad:v4.0"
        )
        assert config.provider == "silero"
        assert "v4.0" in config.repo

    def test_predefined_vad_engines(self):
        """Test all predefined VAD engines are valid."""
        assert len(VAD_ENGINES) == 3
        assert "silero-v4" in VAD_ENGINES
        assert "silero-v3.1" in VAD_ENGINES
        assert "silero-latest" in VAD_ENGINES


class TestTranscriberOptions:
    """Test TranscriberOptions schema."""

    def test_valid_single_temperature(self):
        """Test single temperature value."""
        config = TranscriberOptions(
            temperature=0.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.2,
            logprob_margin=0.2,
            drop_nonverbal_vocals=False,
            no_speech_threshold=0.5,
            condition_on_previous_text=False,
            word_timestamps=True
        )
        assert config.temperature == 0.0

    def test_valid_temperature_list(self):
        """Test temperature as fallback list."""
        config = TranscriberOptions(
            temperature=[0.0, 0.1, 0.2],
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.2,
            logprob_margin=0.2,
            drop_nonverbal_vocals=False,
            no_speech_threshold=0.5,
            condition_on_previous_text=False,
            word_timestamps=True
        )
        assert config.temperature == [0.0, 0.1, 0.2]

    def test_compression_ratio_bounds(self):
        """Test compression_ratio_threshold bounds (1.0-5.0)."""
        # Too low
        with pytest.raises(PydanticValidationError):
            TranscriberOptions(
                temperature=0.0,
                compression_ratio_threshold=0.5,  # < 1.0
                logprob_threshold=-1.0,
                logprob_margin=0.1,
                drop_nonverbal_vocals=False,
                no_speech_threshold=0.5,
                condition_on_previous_text=False,
                word_timestamps=True
            )

        # Too high
        with pytest.raises(PydanticValidationError):
            TranscriberOptions(
                temperature=0.0,
                compression_ratio_threshold=6.0,  # > 5.0
                logprob_threshold=-1.0,
                logprob_margin=0.1,
                drop_nonverbal_vocals=False,
                no_speech_threshold=0.5,
                condition_on_previous_text=False,
                word_timestamps=True
            )

    def test_no_speech_threshold_bounds(self):
        """Test no_speech_threshold bounds (0.0-1.0)."""
        with pytest.raises(PydanticValidationError):
            TranscriberOptions(
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                logprob_margin=0.1,
                drop_nonverbal_vocals=False,
                no_speech_threshold=1.5,  # > 1.0
                condition_on_previous_text=False,
                word_timestamps=True
            )

    def test_logprob_threshold_must_be_negative(self):
        """Test logprob_threshold must be <= 0."""
        with pytest.raises(PydanticValidationError):
            TranscriberOptions(
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=1.0,  # > 0
                logprob_margin=0.1,
                drop_nonverbal_vocals=False,
                no_speech_threshold=0.5,
                condition_on_previous_text=False,
                word_timestamps=True
            )

    def test_optional_fields_default_to_none(self):
        """Test optional fields default to None."""
        config = TranscriberOptions(
            temperature=0.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            logprob_margin=0.1,
            drop_nonverbal_vocals=False,
            no_speech_threshold=0.5,
            condition_on_previous_text=False,
            word_timestamps=True
        )
        assert config.initial_prompt is None
        assert config.prepend_punctuations is None
        assert config.append_punctuations is None
        assert config.clip_timestamps is None

    def test_model_dump_without_none(self):
        """Test None values are excluded from dump."""
        config = TranscriberOptions(
            temperature=0.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            logprob_margin=0.1,
            drop_nonverbal_vocals=False,
            no_speech_threshold=0.5,
            condition_on_previous_text=False,
            word_timestamps=True
        )
        result = config.model_dump_without_none()
        assert "initial_prompt" not in result
        assert "prepend_punctuations" not in result


class TestDecoderOptions:
    """Test DecoderOptions schema."""

    def test_valid_decoder_options(self):
        """Test creating valid decoder options."""
        config = DecoderOptions(
            task="transcribe",
            language="ja",
            best_of=1,
            beam_size=2,
            patience=2.0
        )
        assert config.task == "transcribe"
        assert config.language == "ja"

    def test_beam_size_bounds(self):
        """Test beam_size bounds (1-10)."""
        with pytest.raises(PydanticValidationError):
            DecoderOptions(
                best_of=1,
                beam_size=0,  # < 1
                patience=2.0
            )

        with pytest.raises(PydanticValidationError):
            DecoderOptions(
                best_of=1,
                beam_size=15,  # > 10
                patience=2.0
            )

    def test_best_of_bounds(self):
        """Test best_of bounds (1-10)."""
        with pytest.raises(PydanticValidationError):
            DecoderOptions(
                best_of=0,  # < 1
                beam_size=1,
                patience=2.0
            )

    def test_patience_bounds(self):
        """Test patience bounds (0.0-5.0)."""
        with pytest.raises(PydanticValidationError):
            DecoderOptions(
                best_of=1,
                beam_size=1,
                patience=6.0  # > 5.0
            )

    def test_task_literal(self):
        """Test task must be 'transcribe' or 'translate'."""
        with pytest.raises(PydanticValidationError):
            DecoderOptions(
                task="invalid",
                best_of=1,
                beam_size=1,
                patience=2.0
            )

        # Valid tasks
        config1 = DecoderOptions(task="transcribe", best_of=1, beam_size=1, patience=2.0)
        config2 = DecoderOptions(task="translate", best_of=1, beam_size=1, patience=2.0)
        assert config1.task == "transcribe"
        assert config2.task == "translate"

    def test_suppress_tokens_types(self):
        """Test suppress_tokens accepts int, list, or None."""
        # Integer
        config = DecoderOptions(suppress_tokens=-1, best_of=1, beam_size=1, patience=2.0)
        assert config.suppress_tokens == -1

        # List
        config = DecoderOptions(suppress_tokens=[1, 2, 3], best_of=1, beam_size=1, patience=2.0)
        assert config.suppress_tokens == [1, 2, 3]

        # None (default)
        config = DecoderOptions(best_of=1, beam_size=1, patience=2.0)
        assert config.suppress_tokens is None

    def test_default_values(self):
        """Test default values are applied."""
        config = DecoderOptions(best_of=1, beam_size=1, patience=2.0)
        assert config.task == "transcribe"
        assert config.language == "ja"
        assert config.suppress_blank is True
        assert config.without_timestamps is False


class TestPerformanceMetrics:
    """Test PerformanceMetrics schema."""

    def test_default_values(self):
        """Test all defaults are zero."""
        metrics = PerformanceMetrics()
        assert metrics.vad_processing_time == 0.0
        assert metrics.asr_processing_time == 0.0
        assert metrics.total_processing_time == 0.0
        assert metrics.audio_duration == 0.0
        assert metrics.real_time_factor == 0.0
        assert metrics.segments_processed == 0
        assert metrics.words_transcribed == 0

    def test_can_set_values(self):
        """Test setting metric values."""
        metrics = PerformanceMetrics(
            vad_processing_time=1.5,
            asr_processing_time=10.2,
            total_processing_time=11.7,
            audio_duration=60.0,
            real_time_factor=0.195,
            segments_processed=15,
            words_transcribed=450
        )
        assert metrics.total_processing_time == 11.7
        assert metrics.segments_processed == 15
