#!/usr/bin/env python3
"""
Unit tests for QwenASR module.

Tests the Qwen3-ASR wrapper with stable-ts regrouping integration.
Uses mocks for qwen_asr package and stable_whisper.transcribe_any()
to enable testing without the actual packages loaded.
"""

import gc
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch, call

import pytest
import torch
import stable_whisper


# ============================================================================
# Mock Classes for qwen_asr package
# ============================================================================


class MockTimestamp:
    """Mock timestamp object returned by ForcedAligner."""

    def __init__(self, text: str, start_time: float, end_time: float):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


class MockResult:
    """Mock result object returned by Qwen3ASRModel.transcribe()."""

    def __init__(
        self,
        text: str,
        language: str = "Japanese",
        time_stamps: List[MockTimestamp] = None
    ):
        self.text = text
        self.language = language
        self.time_stamps = time_stamps


class MockQwen3ASRModel:
    """Mock Qwen3ASRModel for testing without actual model."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self._transcribe_return = None

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "MockQwen3ASRModel":
        """Mock from_pretrained that returns a MockQwen3ASRModel instance."""
        instance = cls(**kwargs)
        instance.model_id = model_id
        return instance

    def transcribe(
        self,
        audio: str,
        context: str = "",
        language: str = None,
        return_time_stamps: bool = True,
    ) -> List[MockResult]:
        """Mock transcribe that returns predefined results."""
        if self._transcribe_return is not None:
            return self._transcribe_return

        # Default mock response with timestamps
        if return_time_stamps:
            return [
                MockResult(
                    text="こんにちは世界",
                    language="Japanese",
                    time_stamps=[
                        MockTimestamp("こんにちは", 0.0, 0.8),
                        MockTimestamp("世界", 0.9, 1.5),
                    ]
                )
            ]
        else:
            return [
                MockResult(
                    text="こんにちは世界",
                    language="Japanese",
                    time_stamps=None
                )
            ]

    def set_transcribe_return(self, results: List[MockResult]):
        """Set custom return value for transcribe()."""
        self._transcribe_return = results


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_qwen_asr_module():
    """Create a mock qwen_asr module and patch it into sys.modules."""
    mock_module = MagicMock()
    mock_module.Qwen3ASRModel = MockQwen3ASRModel

    with patch.dict(sys.modules, {'qwen_asr': mock_module}):
        yield mock_module


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA as available."""
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.is_bf16_supported', return_value=True):
            yield


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA as unavailable."""
    with patch('torch.cuda.is_available', return_value=False):
        yield


@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)  # Minimal WAV header
    return audio_file


@pytest.fixture
def mock_whisper_result():
    """Create a mock WhisperResult for testing."""
    result_dict = {
        'language': 'ja',
        'segments': [
            {'start': 0.0, 'end': 2.5, 'text': 'Test segment one'},
            {'start': 2.5, 'end': 5.0, 'text': 'Test segment two'},
        ]
    }
    return stable_whisper.WhisperResult(result_dict)


# ============================================================================
# Test Classes
# ============================================================================


class TestQwenASRInitialization:
    """Tests for QwenASR initialization."""

    def test_default_initialization(self, mock_qwen_asr_module):
        """Test default initialization with all default values."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()

        assert asr.model_id == "Qwen/Qwen3-ASR-1.7B"
        assert asr.device_request == "auto"
        assert asr.dtype_request == "auto"
        assert asr.batch_size == 1  # Default batch_size=1 for accuracy
        assert asr.max_new_tokens == 4096  # Supports ~10 min audio
        assert asr.language is None
        assert asr.task == "transcribe"
        assert asr.timestamps == "word"
        assert asr.use_aligner is True
        assert asr.aligner_id == "Qwen/Qwen3-ForcedAligner-0.6B"
        assert asr.context == ""  # Default empty context
        assert asr.attn_implementation == "auto"  # Default auto attention
        assert asr.model is None  # Lazy loading

    def test_custom_initialization(self, mock_qwen_asr_module):
        """Test initialization with custom values."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(
            model_id="Qwen/Qwen3-ASR-0.6B",
            device="cuda:1",
            dtype="float16",
            batch_size=16,
            max_new_tokens=128,
            language="ja",
            timestamps="none",
            use_aligner=False,
            context="Adult video transcription",
            attn_implementation="flash_attention_2",
        )

        assert asr.model_id == "Qwen/Qwen3-ASR-0.6B"
        assert asr.device_request == "cuda:1"
        assert asr.dtype_request == "float16"
        assert asr.batch_size == 16
        assert asr.max_new_tokens == 128
        assert asr.language == "ja"
        assert asr.use_aligner is False  # Disabled because timestamps="none"
        assert asr.context == "Adult video transcription"
        assert asr.attn_implementation == "flash_attention_2"

    def test_translation_task_warning(self, mock_qwen_asr_module):
        """Test that translation task triggers warning and falls back to transcribe."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(task="translate")

        assert asr.task == "transcribe"  # Falls back to transcribe


class TestQwenASRDeviceDetection:
    """Tests for device detection."""

    def test_auto_device_with_cuda(self, mock_qwen_asr_module, mock_cuda_available):
        """Test auto device selects CUDA when available."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(device="auto")
        device = asr._detect_device()

        assert device == "cuda:0"

    def test_auto_device_without_cuda(self, mock_qwen_asr_module, mock_cuda_unavailable):
        """Test auto device falls back to CPU when CUDA unavailable."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(device="auto")
        device = asr._detect_device()

        assert device == "cpu"

    def test_explicit_cuda_fallback(self, mock_qwen_asr_module, mock_cuda_unavailable):
        """Test explicit CUDA request falls back to CPU with warning."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(device="cuda")
        device = asr._detect_device()

        assert device == "cpu"


class TestQwenASRDtypeDetection:
    """Tests for dtype detection."""

    def test_auto_dtype_cuda_bf16(self, mock_qwen_asr_module, mock_cuda_available):
        """Test auto dtype selects bfloat16 on CUDA when supported."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(dtype="auto")
        dtype = asr._detect_dtype("cuda:0")

        assert dtype == torch.bfloat16

    def test_auto_dtype_cpu(self, mock_qwen_asr_module):
        """Test auto dtype selects float32 on CPU."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(dtype="auto")
        dtype = asr._detect_dtype("cpu")

        assert dtype == torch.float32

    def test_explicit_dtype(self, mock_qwen_asr_module):
        """Test explicit dtype selection."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(dtype="float16")
        dtype = asr._detect_dtype("cuda:0")

        assert dtype == torch.float16


class TestQwenASRTranscription:
    """Tests for transcription functionality."""

    def test_transcribe_returns_whisper_result(
        self,
        mock_qwen_asr_module,
        mock_cuda_available,
        temp_audio_file,
        mock_whisper_result
    ):
        """Test that transcribe returns a WhisperResult."""
        from whisperjav.modules.qwen_asr import QwenASR

        with patch('stable_whisper.transcribe_any', return_value=mock_whisper_result):
            asr = QwenASR()
            result = asr.transcribe(temp_audio_file)

            assert isinstance(result, stable_whisper.WhisperResult)
            assert len(result.segments) == 2

    def test_transcribe_without_aligner_returns_whisper_result(
        self,
        mock_qwen_asr_module,
        mock_cuda_available,
        temp_audio_file
    ):
        """Test transcription without aligner returns WhisperResult."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(use_aligner=False, timestamps="none")
        result = asr.transcribe(temp_audio_file)

        assert isinstance(result, stable_whisper.WhisperResult)

    def test_transcribe_any_called_with_correct_params(
        self,
        mock_qwen_asr_module,
        mock_cuda_available,
        temp_audio_file,
        mock_whisper_result
    ):
        """Test that transcribe_any is called with correct parameters."""
        from whisperjav.modules.qwen_asr import QwenASR

        with patch('stable_whisper.transcribe_any', return_value=mock_whisper_result) as mock_transcribe_any:
            asr = QwenASR()
            asr.transcribe(temp_audio_file)

            # Verify transcribe_any was called
            mock_transcribe_any.assert_called_once()

            # Check key parameters
            call_kwargs = mock_transcribe_any.call_args[1]
            assert call_kwargs['audio'] == str(temp_audio_file)
            assert call_kwargs['audio_type'] == 'str'
            assert call_kwargs['regroup'] is True
            assert call_kwargs['vad'] is False


class TestQwenASRLanguageMapping:
    """Tests for language code mapping."""

    def test_language_mapping(self, mock_qwen_asr_module):
        """Test language name to code mapping."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()

        assert asr._map_language_code("Japanese") == "ja"
        assert asr._map_language_code("English") == "en"
        assert asr._map_language_code("Chinese") == "zh"
        assert asr._map_language_code("JAPANESE") == "ja"
        assert asr._map_language_code("unknown_language") == "un"
        assert asr._map_language_code(None) == "ja"
        assert asr._map_language_code("") == "ja"


class TestQwenASRCleanup:
    """Tests for model cleanup."""

    def test_unload_model(self, mock_qwen_asr_module, mock_cuda_available):
        """Test model unloading."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()
        asr.load_model()

        assert asr.model is not None

        asr.unload_model()

        assert asr.model is None

    def test_cleanup_alias(self, mock_qwen_asr_module, mock_cuda_available):
        """Test cleanup is alias for unload_model."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()
        asr.load_model()
        asr.cleanup()

        assert asr.model is None


class TestQwenASRRepr:
    """Tests for string representation."""

    def test_repr_before_load(self, mock_qwen_asr_module):
        """Test repr before model is loaded."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()
        repr_str = repr(asr)

        assert "QwenASR" in repr_str
        assert "Qwen/Qwen3-ASR-1.7B" in repr_str
        assert "use_aligner=True" in repr_str


class TestQwenASRIntegrationWithPipeline:
    """Tests for integration with TransformersPipeline."""

    def test_pipeline_handles_whisper_result(
        self,
        mock_qwen_asr_module,
        mock_whisper_result
    ):
        """Test that pipeline can convert WhisperResult to segments."""
        from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

        # Create a minimal mock pipeline
        class MockPipeline:
            pass

        mp = MockPipeline()
        mp._convert_asr_result_to_segments = TransformersPipeline._convert_asr_result_to_segments.__get__(mp, MockPipeline)

        segments = mp._convert_asr_result_to_segments(mock_whisper_result)

        assert len(segments) == 2
        assert segments[0]['text'] == 'Test segment one'
        assert segments[0]['start'] == 0.0
        assert segments[0]['end'] == 2.5
        assert segments[1]['text'] == 'Test segment two'

    def test_pipeline_handles_list_dict(self, mock_qwen_asr_module):
        """Test that pipeline passes through List[Dict] unchanged."""
        from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

        class MockPipeline:
            pass

        mp = MockPipeline()
        mp._convert_asr_result_to_segments = TransformersPipeline._convert_asr_result_to_segments.__get__(mp, MockPipeline)

        input_segments = [
            {'text': 'Hello', 'start': 0.0, 'end': 1.0},
            {'text': 'World', 'start': 1.0, 'end': 2.0},
        ]

        segments = mp._convert_asr_result_to_segments(input_segments)

        assert segments is input_segments  # Same object, not copied


class TestQwenASRConstants:
    """Tests for QwenASR class constants."""

    def test_default_max_new_tokens(self, mock_qwen_asr_module):
        """Test DEFAULT_MAX_NEW_TOKENS is set for 10 min audio support."""
        from whisperjav.modules.qwen_asr import QwenASR

        # Default should be 4096 (supports ~10 min audio)
        assert QwenASR.DEFAULT_MAX_NEW_TOKENS == 4096

    def test_force_align_limit_constant(self, mock_qwen_asr_module):
        """Test MAX_FORCE_ALIGN_SECONDS matches qwen-asr limit."""
        from whisperjav.modules.qwen_asr import QwenASR

        # ForcedAligner limit is 180 seconds (3 minutes)
        assert QwenASR.MAX_FORCE_ALIGN_SECONDS == 180

    def test_asr_limit_constant(self, mock_qwen_asr_module):
        """Test MAX_ASR_SECONDS matches qwen-asr limit."""
        from whisperjav.modules.qwen_asr import QwenASR

        # ASR without aligner limit is 1200 seconds (20 minutes)
        assert QwenASR.MAX_ASR_SECONDS == 1200


class TestQwenASRTokenEstimation:
    """Tests for token estimation functionality."""

    def test_estimate_tokens_zero_duration(self, mock_qwen_asr_module):
        """Test token estimation with zero duration."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()
        tokens = asr._estimate_tokens(0)
        assert tokens == 0

    def test_estimate_tokens_negative_duration(self, mock_qwen_asr_module):
        """Test token estimation with negative duration."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()
        tokens = asr._estimate_tokens(-10)
        assert tokens == 0

    def test_estimate_tokens_one_minute(self, mock_qwen_asr_module):
        """Test token estimation for 1 minute audio."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()
        # 1 min × 400 chars/min × 2 tokens/char = 800 tokens
        tokens = asr._estimate_tokens(60)
        assert tokens == 800

    def test_estimate_tokens_ten_minutes(self, mock_qwen_asr_module):
        """Test token estimation for 10 minute audio."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()
        # 10 min × 400 chars/min × 2 tokens/char = 8000 tokens
        tokens = asr._estimate_tokens(600)
        assert tokens == 8000

    def test_estimate_tokens_formula(self, mock_qwen_asr_module):
        """Test token estimation formula: duration_min × 800."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()

        # Test several durations
        test_cases = [
            (30, 400),     # 0.5 min × 800 = 400
            (120, 1600),   # 2 min × 800 = 1600
            (300, 4000),   # 5 min × 800 = 4000
        ]

        for duration_sec, expected_tokens in test_cases:
            tokens = asr._estimate_tokens(duration_sec)
            assert tokens == expected_tokens, f"Duration {duration_sec}s: expected {expected_tokens}, got {tokens}"


class TestQwenASRAudioDuration:
    """Tests for audio duration detection."""

    def test_get_audio_duration_nonexistent_file(self, mock_qwen_asr_module):
        """Test duration detection with non-existent file."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()
        duration = asr._get_audio_duration("/nonexistent/path/audio.wav")
        assert duration == 0.0

    def test_get_audio_duration_returns_float(self, mock_qwen_asr_module, tmp_path):
        """Test that duration detection returns float type."""
        from whisperjav.modules.qwen_asr import QwenASR

        # Create a minimal test file
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"RIFF" + b"\x00" * 100)

        asr = QwenASR()
        duration = asr._get_audio_duration(test_file)

        # Should return float (may be 0.0 for invalid audio)
        assert isinstance(duration, float)


class TestQwenASRAudioLimits:
    """Tests for audio limit checking."""

    def test_check_audio_limits_short_audio(self, mock_qwen_asr_module, tmp_path, caplog):
        """Test that short audio (< 3 min) generates no warnings."""
        from whisperjav.modules.qwen_asr import QwenASR
        from unittest.mock import patch

        asr = QwenASR()

        # Mock duration to return 120 seconds (2 min)
        with patch.object(asr, '_get_audio_duration', return_value=120.0):
            with caplog.at_level('WARNING'):
                asr._check_audio_limits(tmp_path / "short.wav")

        # No warnings expected for 2 min audio
        warning_messages = [r.message for r in caplog.records if r.levelname == 'WARNING']
        assert len(warning_messages) == 0

    def test_check_audio_limits_long_audio_with_aligner(self, mock_qwen_asr_module, tmp_path, caplog):
        """Test that audio > 3 min with ForcedAligner generates warning."""
        from whisperjav.modules.qwen_asr import QwenASR
        from unittest.mock import patch
        import logging

        asr = QwenASR(use_aligner=True)

        # Mock duration to return 300 seconds (5 min)
        with patch.object(asr, '_get_audio_duration', return_value=300.0):
            with caplog.at_level(logging.WARNING):
                asr._check_audio_limits(tmp_path / "long.wav")

        # Should have warning about ForcedAligner limit
        warning_messages = [r.message for r in caplog.records if r.levelname == 'WARNING']
        assert any("ForcedAligner limit" in msg for msg in warning_messages)

    def test_check_audio_limits_exceeds_tokens(self, mock_qwen_asr_module, tmp_path, caplog):
        """Test that audio exceeding token limit generates warning."""
        from whisperjav.modules.qwen_asr import QwenASR
        from unittest.mock import patch
        import logging

        # Create ASR with small token limit
        asr = QwenASR(max_new_tokens=256, use_aligner=False)

        # Mock duration to return 120 seconds (2 min = 1600 tokens, exceeds 256)
        with patch.object(asr, '_get_audio_duration', return_value=120.0):
            with caplog.at_level(logging.WARNING):
                asr._check_audio_limits(tmp_path / "long.wav")

        # Should have warning about token limit
        warning_messages = [r.message for r in caplog.records if r.levelname == 'WARNING']
        assert any("truncated" in msg.lower() or "max_new_tokens" in msg for msg in warning_messages)


class TestQwenASRContextParameter:
    """Tests for context parameter functionality."""

    def test_context_empty_default(self, mock_qwen_asr_module):
        """Test that context defaults to empty string."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()
        assert asr.context == ""

    def test_context_custom_value(self, mock_qwen_asr_module):
        """Test that custom context is stored correctly."""
        from whisperjav.modules.qwen_asr import QwenASR

        test_context = "Adult video with actress Minami"
        asr = QwenASR(context=test_context)
        assert asr.context == test_context

    def test_context_passed_to_transcribe(
        self,
        mock_qwen_asr_module,
        mock_cuda_available,
        temp_audio_file,
        mock_whisper_result
    ):
        """Test that context is used in transcription calls."""
        from whisperjav.modules.qwen_asr import QwenASR

        test_context = "JAV transcription context"

        with patch('stable_whisper.transcribe_any', return_value=mock_whisper_result):
            asr = QwenASR(context=test_context)
            asr.transcribe(temp_audio_file)

            # Context should be stored for use in inference function
            assert asr.context == test_context


class TestQwenASRAttentionImplementation:
    """Tests for attention implementation auto-detection."""

    def test_attn_auto_default(self, mock_qwen_asr_module):
        """Test that attn_implementation defaults to 'auto'."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR()
        assert asr.attn_implementation == "auto"

    def test_attn_explicit_sdpa(self, mock_qwen_asr_module):
        """Test explicit SDPA attention selection."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(attn_implementation="sdpa")
        assert asr.attn_implementation == "sdpa"
        assert asr._detect_attn_implementation() == "sdpa"

    def test_attn_explicit_flash_attention_2(self, mock_qwen_asr_module):
        """Test explicit flash_attention_2 selection."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(attn_implementation="flash_attention_2")
        assert asr.attn_implementation == "flash_attention_2"
        assert asr._detect_attn_implementation() == "flash_attention_2"

    def test_attn_explicit_eager(self, mock_qwen_asr_module):
        """Test explicit eager attention selection."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(attn_implementation="eager")
        assert asr.attn_implementation == "eager"
        assert asr._detect_attn_implementation() == "eager"

    def test_attn_auto_detects_sdpa_on_cpu(self, mock_qwen_asr_module, mock_cuda_unavailable):
        """Test that auto detection returns SDPA on CPU."""
        from whisperjav.modules.qwen_asr import QwenASR

        asr = QwenASR(attn_implementation="auto", device="cpu")
        result = asr._detect_attn_implementation()

        assert result == "sdpa"

    def test_attn_auto_detects_sdpa_when_flash_attn_unavailable(
        self,
        mock_qwen_asr_module,
        mock_cuda_available
    ):
        """Test auto detection returns SDPA when flash-attn not installed."""
        from whisperjav.modules.qwen_asr import QwenASR

        with patch.dict(sys.modules, {'flash_attn': None}):
            # Force ImportError for flash_attn
            with patch('builtins.__import__', side_effect=lambda name, *args: exec('raise ImportError()') if name == 'flash_attn' else __import__(name, *args)):
                asr = QwenASR(attn_implementation="auto")
                asr._device = "cuda:0"  # Ensure CUDA device set
                result = asr._detect_attn_implementation()

                assert result == "sdpa"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
