#!/usr/bin/env python3
"""
Integration tests for QwenASR with TransformersPipeline.

Verifies that the pipeline correctly initializes with qwen backend
and uses QwenASR instead of TransformersASR.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestPipelineBackendSelection:
    """Tests for ASR backend selection in TransformersPipeline."""

    def test_default_backend_is_hf(self, tmp_path):
        """Test that default backend is 'hf' (HuggingFace/Transformers)."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                pipeline = TransformersPipeline(
                    output_dir=str(tmp_path / "output"),
                    temp_dir=str(tmp_path / "temp"),
                )

                assert pipeline.asr_backend == "hf"
                assert "model_id" in pipeline._asr_config
                assert pipeline._asr_config["model_id"] == "kotoba-tech/kotoba-whisper-bilingual-v1.0"

    def test_qwen_backend_selection(self, tmp_path):
        """Test that qwen backend can be selected."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                pipeline = TransformersPipeline(
                    output_dir=str(tmp_path / "output"),
                    temp_dir=str(tmp_path / "temp"),
                    asr_backend="qwen",
                    qwen_model_id="Qwen/Qwen3-ASR-1.7B",
                )

                assert pipeline.asr_backend == "qwen"
                assert pipeline._asr_config["model_id"] == "Qwen/Qwen3-ASR-1.7B"

    def test_qwen_mode_name(self, tmp_path):
        """Test that mode name is 'qwen' when using qwen backend."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                pipeline = TransformersPipeline(
                    output_dir=str(tmp_path / "output"),
                    temp_dir=str(tmp_path / "temp"),
                    asr_backend="qwen",
                )

                assert pipeline.get_mode_name() == "qwen"

    def test_hf_mode_name(self, tmp_path):
        """Test that mode name is 'transformers' when using hf backend."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                pipeline = TransformersPipeline(
                    output_dir=str(tmp_path / "output"),
                    temp_dir=str(tmp_path / "temp"),
                    asr_backend="hf",
                )

                assert pipeline.get_mode_name() == "transformers"

    def test_qwen_config_stored(self, tmp_path):
        """Test that qwen config is stored correctly."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                pipeline = TransformersPipeline(
                    output_dir=str(tmp_path / "output"),
                    temp_dir=str(tmp_path / "temp"),
                    asr_backend="qwen",
                    qwen_model_id="Qwen/Qwen3-ASR-1.7B",
                    qwen_device="cuda",
                    qwen_dtype="bfloat16",
                    qwen_batch_size=16,
                    qwen_max_tokens=128,
                    qwen_language="ja",
                    qwen_timestamps="word",
                    qwen_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                )

                assert pipeline.qwen_config["model_id"] == "Qwen/Qwen3-ASR-1.7B"
                assert pipeline.qwen_config["device"] == "cuda"
                assert pipeline.qwen_config["dtype"] == "bfloat16"
                assert pipeline.qwen_config["batch_size"] == 16
                assert pipeline.qwen_config["max_new_tokens"] == 128
                assert pipeline.qwen_config["language"] == "ja"
                assert pipeline.qwen_config["timestamps"] == "word"
                assert pipeline.qwen_config["aligner_id"] == "Qwen/Qwen3-ForcedAligner-0.6B"
                assert pipeline.qwen_config["use_aligner"] is True

    def test_qwen_asr_config_format(self, tmp_path):
        """Test that ASR config is in QwenASR-compatible format when using qwen backend."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                pipeline = TransformersPipeline(
                    output_dir=str(tmp_path / "output"),
                    temp_dir=str(tmp_path / "temp"),
                    asr_backend="qwen",
                )

                # QwenASR uses these parameter names
                assert "model_id" in pipeline._asr_config
                assert "device" in pipeline._asr_config
                assert "dtype" in pipeline._asr_config
                assert "batch_size" in pipeline._asr_config
                assert "max_new_tokens" in pipeline._asr_config
                assert "language" in pipeline._asr_config
                assert "timestamps" in pipeline._asr_config
                assert "use_aligner" in pipeline._asr_config
                assert "aligner_id" in pipeline._asr_config

                # Should NOT have TransformersASR-specific params
                assert "chunk_length_s" not in pipeline._asr_config
                assert "stride_length_s" not in pipeline._asr_config
                assert "beam_size" not in pipeline._asr_config

    def test_invalid_backend_raises(self, tmp_path):
        """Test that invalid backend raises ValueError."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                with pytest.raises(ValueError, match="Invalid ASR backend"):
                    TransformersPipeline(
                        output_dir=str(tmp_path / "output"),
                        temp_dir=str(tmp_path / "temp"),
                        asr_backend="invalid",
                    )

    def test_qwen_scene_detection_used(self, tmp_path):
        """Test that qwen_scene is used when asr_backend='qwen'."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                # When using qwen backend with scene="none", qwen_scene should be used
                # (Using "none" to avoid torchaudio import issues)
                pipeline = TransformersPipeline(
                    output_dir=str(tmp_path / "output"),
                    temp_dir=str(tmp_path / "temp"),
                    asr_backend="qwen",
                    hf_scene="auditok",  # This should be ignored
                    qwen_scene="none",  # This should be used
                )

                assert pipeline.scene_method == "none"

    def test_qwen_scene_detection_auditok(self, tmp_path):
        """Test that qwen backend can use auditok scene detection."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                # Patch the scene_detection module's DynamicSceneDetector
                with patch('whisperjav.modules.scene_detection.DynamicSceneDetector'):
                    from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                    pipeline = TransformersPipeline(
                        output_dir=str(tmp_path / "output"),
                        temp_dir=str(tmp_path / "temp"),
                        asr_backend="qwen",
                        qwen_scene="auditok",
                    )

                    assert pipeline.scene_method == "auditok"

    def test_hf_scene_detection_used(self, tmp_path):
        """Test that hf_scene is used when asr_backend='hf'."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                # When using hf backend, should use hf_scene
                pipeline = TransformersPipeline(
                    output_dir=str(tmp_path / "output"),
                    temp_dir=str(tmp_path / "temp"),
                    asr_backend="hf",
                    hf_scene="auditok",  # This should be used
                    qwen_scene="silero",  # This should be ignored
                )

                assert pipeline.scene_method == "auditok"


class TestQwenLanguageCode:
    """Tests for language code handling with Qwen backend."""

    def test_qwen_language_code_default(self, tmp_path):
        """Test default language code when qwen_language is None."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                pipeline = TransformersPipeline(
                    output_dir=str(tmp_path / "output"),
                    temp_dir=str(tmp_path / "temp"),
                    asr_backend="qwen",
                    qwen_language=None,  # Auto-detect
                )

                assert pipeline.lang_code == "ja"  # Default fallback

    def test_qwen_language_code_explicit(self, tmp_path):
        """Test explicit language code setting."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                pipeline = TransformersPipeline(
                    output_dir=str(tmp_path / "output"),
                    temp_dir=str(tmp_path / "temp"),
                    asr_backend="qwen",
                    qwen_language="en",
                )

                assert pipeline.lang_code == "en"

    def test_direct_to_english_overrides(self, tmp_path):
        """Test that direct-to-english overrides language code."""
        with patch('whisperjav.pipelines.transformers_pipeline.AudioExtractor'):
            with patch('whisperjav.pipelines.transformers_pipeline.SCENE_EXTRACTION_SR', 48000):
                from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

                pipeline = TransformersPipeline(
                    output_dir=str(tmp_path / "output"),
                    temp_dir=str(tmp_path / "temp"),
                    asr_backend="qwen",
                    qwen_language="ja",
                    subs_language="direct-to-english",
                )

                assert pipeline.lang_code == "en"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
