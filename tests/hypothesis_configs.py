#!/usr/bin/env python3
"""
Configuration definitions for ASR regression hypothesis testing.

This module defines parameter variations to test specific hypotheses about
why v1.7.3 produces ~20% fewer subtitles than v1.7.1.

Each hypothesis modifies only the parameter(s) under investigation, keeping
all others at v1.7.3 defaults for isolated testing.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class TestConfig:
    """A single test configuration variant."""
    name: str
    description: str
    hypothesis: str
    params_override: Dict[str, Any]

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class V173Defaults:
    """
    v1.7.3 default parameters for balanced/aggressive mode.
    These are the baseline values extracted from:
    - whisperjav/config/components/asr/faster_whisper.py
    - whisperjav/config/components/vad/silero.py
    """

    # ASR parameters (aggressive sensitivity)
    ASR = {
        "task": "transcribe",
        "language": "ja",
        "beam_size": 3,
        "best_of": 1,
        "patience": 1.6,
        "temperature": [0.0, 0.3],
        "compression_ratio_threshold": 3.0,
        "logprob_threshold": -2.5,
        "logprob_margin": 0.0,
        "no_speech_threshold": 0.22,
        "word_timestamps": True,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 2,
        "hallucination_silence_threshold": 2.1,
        "chunk_length": 30,
        "suppress_blank": False,
        "suppress_tokens": [],
    }

    # VAD/Speech Segmentation parameters (aggressive sensitivity)
    VAD = {
        "threshold": 0.05,
        "min_speech_duration_ms": 30,
        "max_speech_duration_s": 6.0,
        "min_silence_duration_ms": 300,
        "neg_threshold": 0.1,
        "speech_pad_ms": 500,  # NOTE: Changed from 700 in v1.7.3 config
        "chunk_threshold_s": 4.0,
    }


class HypothesisConfigs:
    """Factory for generating test configurations for each hypothesis."""

    @staticmethod
    def get_baseline() -> TestConfig:
        """V1.7.3 default configuration (aggressive sensitivity)."""
        return TestConfig(
            name="v1.7.3_baseline",
            description="v1.7.3 default configuration (aggressive)",
            hypothesis="baseline",
            params_override={}  # No overrides - use defaults
        )

    @staticmethod
    def get_vad_parameter_tests() -> List[TestConfig]:
        """
        Hypothesis 1: VAD Parameter Wiring

        Test if speech_pad_ms and min_silence_duration_ms are being
        applied correctly to Silero VAD.
        """
        configs = []

        # Test 1a: Zero speech padding
        configs.append(TestConfig(
            name="vad_zero_speech_pad",
            description="speech_pad_ms = 0 (no padding around speech)",
            hypothesis="vad_params",
            params_override={
                "vad": {"speech_pad_ms": 0}
            }
        ))

        # Test 1b: Zero minimum silence duration
        configs.append(TestConfig(
            name="vad_zero_min_silence",
            description="min_silence_duration_ms = 0 (no silence filtering)",
            hypothesis="vad_params",
            params_override={
                "vad": {"min_silence_duration_ms": 0}
            }
        ))

        # Test 1c: Both zeros (maximum sensitivity)
        configs.append(TestConfig(
            name="vad_both_zeros",
            description="speech_pad_ms = 0, min_silence_duration_ms = 0",
            hypothesis="vad_params",
            params_override={
                "vad": {
                    "speech_pad_ms": 0,
                    "min_silence_duration_ms": 0
                }
            }
        ))

        # Test 1d: V1.7.1 values (700ms padding was in old config)
        configs.append(TestConfig(
            name="vad_v171_padding",
            description="speech_pad_ms = 700 (potential v1.7.1 default)",
            hypothesis="vad_params",
            params_override={
                "vad": {"speech_pad_ms": 700}
            }
        ))

        return configs

    @staticmethod
    def get_asr_duration_filter_tests() -> List[TestConfig]:
        """
        Hypothesis 2: ASR Duration Filtering

        Test if no_speech_threshold and logprob_threshold are filtering
        out short segments.
        """
        configs = []

        # Test 2a: Lower no_speech_threshold (more permissive)
        configs.append(TestConfig(
            name="asr_no_speech_default",
            description="no_speech_threshold = 0.5 (balanced default)",
            hypothesis="asr_duration_filter",
            params_override={
                "asr": {"no_speech_threshold": 0.5}
            }
        ))

        # Test 2b: Conservative no_speech_threshold
        configs.append(TestConfig(
            name="asr_no_speech_conservative",
            description="no_speech_threshold = 0.6 (conservative)",
            hypothesis="asr_duration_filter",
            params_override={
                "asr": {"no_speech_threshold": 0.6}
            }
        ))

        # Test 2c: Balanced logprob_threshold
        configs.append(TestConfig(
            name="asr_logprob_balanced",
            description="logprob_threshold = -1.2 (balanced default)",
            hypothesis="asr_duration_filter",
            params_override={
                "asr": {"logprob_threshold": -1.2}
            }
        ))

        # Test 2d: Conservative logprob_threshold
        configs.append(TestConfig(
            name="asr_logprob_conservative",
            description="logprob_threshold = -1.0 (conservative)",
            hypothesis="asr_duration_filter",
            params_override={
                "asr": {"logprob_threshold": -1.0}
            }
        ))

        # Test 2e: Combined balanced thresholds
        configs.append(TestConfig(
            name="asr_both_balanced",
            description="no_speech = 0.5, logprob = -1.2 (both balanced)",
            hypothesis="asr_duration_filter",
            params_override={
                "asr": {
                    "no_speech_threshold": 0.5,
                    "logprob_threshold": -1.2
                }
            }
        ))

        return configs

    @staticmethod
    def get_temperature_fallback_tests() -> List[TestConfig]:
        """
        Hypothesis 3: Temperature Fallback

        Test if temperature fallback configuration affects segment detection.
        """
        configs = []

        # Test 3a: No fallback (single temperature)
        configs.append(TestConfig(
            name="temp_no_fallback",
            description="temperature = [0.0] (no fallback)",
            hypothesis="temperature_fallback",
            params_override={
                "asr": {"temperature": [0.0]}
            }
        ))

        # Test 3b: Balanced fallback
        configs.append(TestConfig(
            name="temp_balanced_fallback",
            description="temperature = [0.0, 0.1] (balanced fallback)",
            hypothesis="temperature_fallback",
            params_override={
                "asr": {"temperature": [0.0, 0.1]}
            }
        ))

        # Test 3c: More aggressive fallback
        configs.append(TestConfig(
            name="temp_multi_fallback",
            description="temperature = [0.0, 0.2, 0.4, 0.6] (multi-stage)",
            hypothesis="temperature_fallback",
            params_override={
                "asr": {"temperature": [0.0, 0.2, 0.4, 0.6]}
            }
        ))

        return configs

    @staticmethod
    def get_patience_beam_tests() -> List[TestConfig]:
        """
        Hypothesis 4: Patience/Beam Interaction

        Test beam search parameters that affect decoding quality.
        """
        configs = []

        # Test 4a: Lower patience (faster, potentially less thorough)
        configs.append(TestConfig(
            name="beam_patience_low",
            description="patience = 1.6 (aggressive default)",
            hypothesis="patience_beam",
            params_override={
                "asr": {"patience": 1.6}
            }
        ))

        # Test 4b: Balanced patience
        configs.append(TestConfig(
            name="beam_patience_balanced",
            description="patience = 2.0 (balanced default)",
            hypothesis="patience_beam",
            params_override={
                "asr": {"patience": 2.0}
            }
        ))

        # Test 4c: Higher patience (more thorough)
        configs.append(TestConfig(
            name="beam_patience_high",
            description="patience = 2.9 (conservative)",
            hypothesis="patience_beam",
            params_override={
                "asr": {"patience": 2.9}
            }
        ))

        # Test 4d: Smaller beam size
        configs.append(TestConfig(
            name="beam_size_small",
            description="beam_size = 2 (balanced default)",
            hypothesis="patience_beam",
            params_override={
                "asr": {"beam_size": 2}
            }
        ))

        # Test 4e: Balanced combination
        configs.append(TestConfig(
            name="beam_balanced_combo",
            description="beam_size = 2, patience = 2.0 (balanced)",
            hypothesis="patience_beam",
            params_override={
                "asr": {
                    "beam_size": 2,
                    "patience": 2.0
                }
            }
        ))

        return configs

    @staticmethod
    def get_compute_type_tests() -> List[TestConfig]:
        """
        Hypothesis 5: Compute Type Effect

        Test if compute_type affects transcription quality/quantity.
        CTranslate2 supports: auto, int8, int8_float16, float16, float32
        """
        configs = []

        # Test 5a: Auto (let CTranslate2 decide internally)
        configs.append(TestConfig(
            name="compute_auto",
            description="compute_type = auto (CTranslate2 decides)",
            hypothesis="compute_type",
            params_override={
                "model": {"compute_type": "auto"}
            }
        ))

        # Test 5b: int8 (fully quantized)
        configs.append(TestConfig(
            name="compute_int8",
            description="compute_type = int8 (fully quantized)",
            hypothesis="compute_type",
            params_override={
                "model": {"compute_type": "int8"}
            }
        ))

        # Test 5c: int8_float16 (quantized weights, float16 compute)
        configs.append(TestConfig(
            name="compute_int8_float16",
            description="compute_type = int8_float16 (quantized weights, fp16 compute)",
            hypothesis="compute_type",
            params_override={
                "model": {"compute_type": "int8_float16"}
            }
        ))

        # Test 5d: float16 (full precision fp16)
        configs.append(TestConfig(
            name="compute_float16",
            description="compute_type = float16 (full precision fp16)",
            hypothesis="compute_type",
            params_override={
                "model": {"compute_type": "float16"}
            }
        ))

        return configs

    @staticmethod
    def get_all_hypotheses() -> Dict[str, List[TestConfig]]:
        """Get all hypothesis test configurations organized by hypothesis."""
        return {
            "baseline": [HypothesisConfigs.get_baseline()],
            "vad_params": HypothesisConfigs.get_vad_parameter_tests(),
            "asr_duration_filter": HypothesisConfigs.get_asr_duration_filter_tests(),
            "temperature_fallback": HypothesisConfigs.get_temperature_fallback_tests(),
            "patience_beam": HypothesisConfigs.get_patience_beam_tests(),
            "compute_type": HypothesisConfigs.get_compute_type_tests(),
        }

    @staticmethod
    def get_quick_suite() -> List[TestConfig]:
        """
        Get a quick test suite with 1-2 key tests per hypothesis.

        Useful for rapid iteration during investigation.
        """
        configs = [
            HypothesisConfigs.get_baseline(),
        ]

        # Key VAD tests
        configs.append(TestConfig(
            name="vad_both_zeros",
            description="speech_pad_ms = 0, min_silence_duration_ms = 0",
            hypothesis="vad_params",
            params_override={
                "vad": {
                    "speech_pad_ms": 0,
                    "min_silence_duration_ms": 0
                }
            }
        ))

        # Key ASR filter test
        configs.append(TestConfig(
            name="asr_both_balanced",
            description="no_speech = 0.5, logprob = -1.2 (both balanced)",
            hypothesis="asr_duration_filter",
            params_override={
                "asr": {
                    "no_speech_threshold": 0.5,
                    "logprob_threshold": -1.2
                }
            }
        ))

        # Key temperature test
        configs.append(TestConfig(
            name="temp_balanced_fallback",
            description="temperature = [0.0, 0.1] (balanced fallback)",
            hypothesis="temperature_fallback",
            params_override={
                "asr": {"temperature": [0.0, 0.1]}
            }
        ))

        # Key beam search test
        configs.append(TestConfig(
            name="beam_balanced_combo",
            description="beam_size = 2, patience = 2.0 (balanced)",
            hypothesis="patience_beam",
            params_override={
                "asr": {
                    "beam_size": 2,
                    "patience": 2.0
                }
            }
        ))

        return configs


# Convenience function for external use
def list_all_configs() -> List[TestConfig]:
    """Get a flat list of all test configurations."""
    all_configs = []
    for hypothesis_configs in HypothesisConfigs.get_all_hypotheses().values():
        all_configs.extend(hypothesis_configs)
    return all_configs


if __name__ == "__main__":
    # Print all configurations for inspection
    print("=== Hypothesis Test Configurations ===\n")

    all_hypotheses = HypothesisConfigs.get_all_hypotheses()

    for hypothesis_name, configs in all_hypotheses.items():
        print(f"\n{hypothesis_name.upper()}:")
        print("-" * 60)
        for config in configs:
            print(f"  {config.name}")
            print(f"    {config.description}")
            if config.params_override:
                print(f"    Overrides: {config.params_override}")
        print()

    print(f"\nTotal configurations: {len(list_all_configs())}")
    print(f"Quick suite configurations: {len(HypothesisConfigs.get_quick_suite())}")
