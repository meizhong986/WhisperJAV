#!/usr/bin/env python3
"""
Validate Hypothesis Test Suite Setup

This script validates that the hypothesis test suite is properly configured
and can access all necessary WhisperJAV modules.

Usage:
    python validate_hypothesis_suite.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check that all required imports are available."""
    print("Checking required imports...")

    errors = []

    # Test hypothesis config imports
    try:
        from hypothesis_configs import (
            HypothesisConfigs,
            TestConfig,
            V173Defaults,
            list_all_configs
        )
        print("  [OK] hypothesis_configs imports")
    except Exception as e:
        errors.append(f"hypothesis_configs import failed: {e}")
        print(f"  [FAIL] hypothesis_configs imports: {e}")

    # Test WhisperJAV imports
    try:
        from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
        print("  [OK] BalancedPipeline import")
    except Exception as e:
        errors.append(f"BalancedPipeline import failed: {e}")
        print(f"  [FAIL] BalancedPipeline import: {e}")

    try:
        from whisperjav.config.transcription_tuner import TranscriptionTuner
        print("  [OK] TranscriptionTuner import")
    except Exception as e:
        errors.append(f"TranscriptionTuner import failed: {e}")
        print(f"  [FAIL] TranscriptionTuner import: {e}")

    try:
        import srt as srt_module
        print("  [OK] srt module import")
    except Exception as e:
        errors.append(f"srt module import failed: {e}")
        print(f"  [FAIL] srt module import: {e}")

    return errors


def check_config_generation():
    """Check that configurations can be generated."""
    print("\nChecking configuration generation...")

    errors = []

    try:
        from hypothesis_configs import HypothesisConfigs, list_all_configs

        # Test baseline
        baseline = HypothesisConfigs.get_baseline()
        if baseline.name != "v1.7.3_baseline":
            errors.append(f"Baseline name mismatch: {baseline.name}")
        else:
            print("  [OK] Baseline config")

        # Test all hypotheses
        all_hypotheses = HypothesisConfigs.get_all_hypotheses()
        expected_hypotheses = ["baseline", "vad_params", "asr_duration_filter",
                               "temperature_fallback", "patience_beam"]

        for hyp in expected_hypotheses:
            if hyp not in all_hypotheses:
                errors.append(f"Missing hypothesis: {hyp}")
            else:
                print(f"  [OK] Hypothesis '{hyp}' ({len(all_hypotheses[hyp])} configs)")

        # Test quick suite
        quick = HypothesisConfigs.get_quick_suite()
        if len(quick) < 3:
            errors.append(f"Quick suite too small: {len(quick)} configs")
        else:
            print(f"  [OK] Quick suite ({len(quick)} configs)")

        # Test full list
        all_configs = list_all_configs()
        if len(all_configs) < 10:
            errors.append(f"Total configs too small: {len(all_configs)}")
        else:
            print(f"  [OK] All configs ({len(all_configs)} configs)")

    except Exception as e:
        errors.append(f"Config generation failed: {e}")
        print(f"  [FAIL] Config generation: {e}")

    return errors


def check_tuner_integration():
    """Check that TranscriptionTuner can generate configs."""
    print("\nChecking TranscriptionTuner integration...")

    errors = []

    try:
        from whisperjav.config.transcription_tuner import TranscriptionTuner

        tuner = TranscriptionTuner()

        # Test balanced/aggressive config
        config = tuner.resolve_params(
            pipeline_name="balanced",
            sensitivity="aggressive",
            task="transcribe"
        )

        # Check config structure
        required_keys = ["model", "params", "features", "task"]
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing key in resolved config: {key}")
            else:
                print(f"  [OK] Config has '{key}'")

        # Check params structure
        param_sections = ["decoder", "vad", "provider"]
        for section in param_sections:
            if section not in config["params"]:
                errors.append(f"Missing params section: {section}")
            else:
                print(f"  [OK] Config params has '{section}'")

    except Exception as e:
        errors.append(f"TranscriptionTuner check failed: {e}")
        print(f"  [FAIL] TranscriptionTuner: {e}")

    return errors


def check_defaults():
    """Check that V173Defaults are properly defined."""
    print("\nChecking v1.7.3 defaults...")

    errors = []

    try:
        from hypothesis_configs import V173Defaults

        # Check ASR defaults
        asr_params = ["task", "language", "beam_size", "patience", "temperature",
                      "no_speech_threshold", "logprob_threshold"]
        for param in asr_params:
            if param not in V173Defaults.ASR:
                errors.append(f"Missing ASR default: {param}")
            else:
                value = V173Defaults.ASR[param]
                print(f"  [OK] ASR.{param} = {value}")

        # Check VAD defaults
        vad_params = ["threshold", "min_speech_duration_ms", "min_silence_duration_ms",
                      "speech_pad_ms"]
        for param in vad_params:
            if param not in V173Defaults.VAD:
                errors.append(f"Missing VAD default: {param}")
            else:
                value = V173Defaults.VAD[param]
                print(f"  [OK] VAD.{param} = {value}")

    except Exception as e:
        errors.append(f"Defaults check failed: {e}")
        print(f"  [FAIL] Defaults: {e}")

    return errors


def main():
    """Run all validation checks."""
    print("="*70)
    print("HYPOTHESIS TEST SUITE VALIDATION")
    print("="*70)

    all_errors = []

    # Run checks
    all_errors.extend(check_imports())
    all_errors.extend(check_config_generation())
    all_errors.extend(check_tuner_integration())
    all_errors.extend(check_defaults())

    # Summary
    print("\n" + "="*70)
    if all_errors:
        print("VALIDATION FAILED")
        print("="*70)
        print(f"\nFound {len(all_errors)} error(s):")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        print()
        sys.exit(1)
    else:
        print("VALIDATION PASSED")
        print("="*70)
        print("\n[SUCCESS] All checks passed! The hypothesis test suite is ready to use.")
        print("\nNext steps:")
        print("  1. Prepare test audio: ffmpeg -i video.mp4 -ss 01:52:00 -t 00:25:00 -vn -acodec pcm_s16le subset.wav")
        print("  2. Run quick test: python hypothesis_test_suite.py --audio subset.wav --quick")
        print("  3. Run full suite: python hypothesis_test_suite.py --audio subset.wav")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
