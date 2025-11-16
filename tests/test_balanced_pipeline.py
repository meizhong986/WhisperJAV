#!/usr/bin/env python3
"""Test balanced pipeline functionality."""

import pytest
from pathlib import Path
import sys

# Add parent directory to path to import whisperjav modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
from whisperjav.modules.faster_whisper_pro_asr import FasterWhisperProASR
from whisperjav.config.transcription_tuner import TranscriptionTuner


def test_balanced_pipeline_instantiation():
    """Test that balanced pipeline can be instantiated with valid config."""
    tuner = TranscriptionTuner()
    resolved_config = tuner.resolve_params(
        pipeline_name="balanced",
        sensitivity="balanced",
        task="transcribe"
    )

    pipeline = BalancedPipeline(
        output_dir="./test_output",
        temp_dir="./test_temp",
        keep_temp_files=False,
        subs_language="native",
        resolved_config=resolved_config
    )

    assert pipeline is not None
    assert pipeline.get_mode_name() == "balanced"
    assert isinstance(pipeline.asr, FasterWhisperProASR)


def test_balanced_vs_fidelity_asr_difference():
    """Verify balanced uses FasterWhisperProASR, fidelity uses WhisperProASR."""
    from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline
    from whisperjav.modules.whisper_pro_asr import WhisperProASR

    tuner = TranscriptionTuner()

    # Balanced config
    balanced_config = tuner.resolve_params("balanced", "balanced", "transcribe")
    balanced = BalancedPipeline(
        output_dir="./test_output",
        temp_dir="./test_temp",
        keep_temp_files=False,
        subs_language="native",
        resolved_config=balanced_config
    )

    # Fidelity config
    fidelity_config = tuner.resolve_params("fidelity", "balanced", "transcribe")
    fidelity = FidelityPipeline(
        output_dir="./test_output",
        temp_dir="./test_temp",
        keep_temp_files=False,
        subs_language="native",
        resolved_config=fidelity_config
    )

    # Verify ASR module types
    assert isinstance(balanced.asr, FasterWhisperProASR)
    assert isinstance(fidelity.asr, WhisperProASR)
    assert type(balanced.asr) != type(fidelity.asr)

    # Verify mode names
    assert balanced.get_mode_name() == "balanced"
    assert fidelity.get_mode_name() == "fidelity"


def test_faster_whisper_pro_asr_initialization():
    """Test that FasterWhisperProASR can initialize models."""
    model_config = {
        "model_name": "turbo",
        "device": "cpu",  # Use CPU for testing
        "compute_type": "int8"
    }

    params = {
        "decoder": {
            "task": "transcribe",
            "language": "ja",
            "beam_size": 5
        },
        "vad": {
            "threshold": 0.4,
            "min_speech_duration_ms": 150,
            "chunk_threshold": 4.0,
            "vad_repo": "snakers4/silero-vad:v3.1"
        },
        "provider": {
            "temperature": [0.0],
            "fp16": False
        }
    }

    asr = FasterWhisperProASR(
        model_config=model_config,
        params=params,
        task="transcribe"
    )

    assert asr is not None
    assert asr.model_name == "turbo"
    assert asr.device == "cpu"
    assert asr.task == "transcribe"


def test_balanced_pipeline_config_resolution():
    """Test that balanced pipeline config is properly resolved by tuner."""
    tuner = TranscriptionTuner()

    for sensitivity in ["conservative", "balanced", "aggressive"]:
        resolved_config = tuner.resolve_params(
            pipeline_name="balanced",
            sensitivity=sensitivity,
            task="transcribe"
        )

        # Verify config structure
        assert "model" in resolved_config
        assert "params" in resolved_config
        assert "features" in resolved_config
        assert "task" in resolved_config

        # Verify model config
        assert "model_name" in resolved_config["model"]
        # Note: device is auto-detected, not in config
        assert "compute_type" in resolved_config["model"]

        # Verify params structure
        assert "decoder" in resolved_config["params"]
        assert "vad" in resolved_config["params"]
        assert "provider" in resolved_config["params"]

        # Verify features
        assert "scene_detection" in resolved_config["features"]
        assert "post_processing" in resolved_config["features"]


def test_balanced_pipeline_mode_names():
    """Test that all pipelines return correct mode names."""
    from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline
    from whisperjav.pipelines.fast_pipeline import FastPipeline
    from whisperjav.pipelines.faster_pipeline import FasterPipeline

    tuner = TranscriptionTuner()

    # Create all pipelines
    pipelines = {
        "balanced": BalancedPipeline,
        "fidelity": FidelityPipeline,
        "fast": FastPipeline,
        "faster": FasterPipeline
    }

    for mode_name, pipeline_class in pipelines.items():
        config = tuner.resolve_params(mode_name, "balanced", "transcribe")
        pipeline = pipeline_class(
            output_dir="./test_output",
            temp_dir="./test_temp",
            keep_temp_files=False,
            subs_language="native",
            resolved_config=config
        )

        assert pipeline.get_mode_name() == mode_name


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
