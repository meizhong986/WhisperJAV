#!/usr/bin/env python3
"""
Test script to validate the TranscriptionTuner v4.3 implementation.
Run this after setting up the new configuration.
"""

import sys
import json
from pathlib import Path
from pprint import pprint

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisperjav.config.transcription_tuner import TranscriptionTuner
from whisperjav.utils.logger import setup_logger

def test_parameter_resolution():
    """Test all pipeline/sensitivity/task combinations."""
    
    logger = setup_logger("test", "DEBUG")
    
    print("=" * 60)
    print("Testing TranscriptionTuner v4.3 Parameter Resolution")
    print("=" * 60)
    
    # Initialize tuner
    try:
        tuner = TranscriptionTuner()
        print("✓ Tuner initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize tuner: {e}")
        return False
    
    # Test matrix
    pipelines = ['faster', 'fast', 'balanced']
    sensitivities = ['conservative', 'balanced', 'aggressive']
    tasks = ['transcribe', 'translate']
    
    success_count = 0
    failure_count = 0
    
    for pipeline in pipelines:
        print(f"\n{'-'*40}")
        print(f"Pipeline: {pipeline.upper()}")
        print(f"{'-'*40}")
        
        for sensitivity in sensitivities:
            for task in tasks:
                test_name = f"{pipeline}/{sensitivity}/{task}"
                try:
                    # Resolve parameters
                    result = tuner.resolve_params(pipeline, sensitivity, task)
                    
                    # Validate structure
                    assert 'params' in result, "Missing 'params' in result"
                    assert 'decoder' in result['params'], "Missing 'decoder' in params"
                    assert 'provider' in result['params'], "Missing 'provider' in params"
                    assert 'vad' in result['params'], "Missing 'vad' in params"
                    
                    # Validate decoder task override
                    assert result['params']['decoder'].get('task') == task, \
                        f"Task override failed: expected {task}, got {result['params']['decoder'].get('task')}"
                    
                    # Check VAD handling
                    if pipeline == 'balanced':
                        # Should have separate VAD params
                        if not result['params']['vad']:
                            print(f"  ⚠ {test_name}: No VAD params (expected for balanced)")
                    else:
                        # VAD should be packed into provider for faster/fast
                        if 'vad' in result['params']['provider'] or 'vad_threshold' in result['params']['provider']:
                            print(f"  ✓ {test_name}: VAD packed in provider")
                        else:
                            print(f"  ✓ {test_name}: No VAD (none specified)")
                    
                    print(f"  ✓ {test_name}: SUCCESS")
                    success_count += 1
                    
                except Exception as e:
                    print(f"  ✗ {test_name}: FAILED - {e}")
                    failure_count += 1
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {success_count} passed, {failure_count} failed")
    print(f"{'='*60}")
    
    return failure_count == 0


def test_specific_configuration(pipeline='faster', sensitivity='balanced', task='transcribe'):
    """Test and display a specific configuration in detail."""
    
    print(f"\n{'='*60}")
    print(f"Detailed Test: {pipeline}/{sensitivity}/{task}")
    print(f"{'='*60}\n")
    
    tuner = TranscriptionTuner()
    result = tuner.resolve_params(pipeline, sensitivity, task)
    
    print("Structure returned:")
    print("-" * 40)
    
    # Show high-level structure
    for key in result:
        if key == 'params':
            print(f"params:")
            for param_key in result['params']:
                param_count = len(result['params'][param_key]) if result['params'][param_key] else 0
                print(f"  {param_key}: {param_count} parameters")
        elif isinstance(result[key], dict):
            print(f"{key}: {len(result[key])} items")
        else:
            print(f"{key}: {result[key]}")
    
    print("\n" + "-" * 40)
    print("Key parameters:")
    print("-" * 40)
    
    # Show some key parameters
    decoder = result['params']['decoder']
    provider = result['params']['provider']
    
    print(f"Decoder:")
    print(f"  task: {decoder.get('task')}")
    print(f"  language: {decoder.get('language')}")
    print(f"  beam_size: {decoder.get('beam_size')}")
    print(f"  best_of: {decoder.get('best_of')}")
    
    print(f"\nProvider:")
    print(f"  temperature: {provider.get('temperature')}")
    print(f"  no_speech_threshold: {provider.get('no_speech_threshold')}")
    
    # Check for engine-specific params
    if 'batch_size' in provider:
        print(f"  batch_size: {provider.get('batch_size')} (faster-whisper specific)")
    if 'repetition_penalty' in provider:
        print(f"  repetition_penalty: {provider.get('repetition_penalty')} (faster-whisper specific)")
    if 'regroup' in provider:
        print(f"  regroup: {provider.get('regroup')} (stable-ts specific)")
    if 'vad' in provider:
        print(f"  vad: {provider.get('vad')} (VAD packed in provider)")
    
    if result['params']['vad']:
        print(f"\nVAD (separate):")
        print(f"  threshold: {result['params']['vad'].get('threshold')}")
        print(f"  min_speech_duration_ms: {result['params']['vad'].get('min_speech_duration_ms')}")
    
    return result


def validate_config_consistency():
    """Validate that the configuration is internally consistent."""
    
    print(f"\n{'='*60}")
    print("Configuration Consistency Validation")
    print(f"{'='*60}\n")
    
    tuner = TranscriptionTuner()
    
    try:
        result = tuner.validate_configuration()
        print("✓ Configuration validation PASSED")
        return True
    except Exception as e:
        print(f"✗ Configuration validation FAILED: {e}")
        return False


def main():
    """Run all tests."""
    
    # Test parameter resolution for all combinations
    if not test_parameter_resolution():
        sys.exit(1)
    
    # Test specific configuration in detail
    print("\n" + "="*60)
    print("DETAILED CONFIGURATION EXAMPLES")
    print("="*60)
    
    test_specific_configuration('faster', 'conservative', 'transcribe')
    test_specific_configuration('balanced', 'aggressive', 'translate')
    
    # Validate configuration consistency
    if not validate_config_consistency():
        sys.exit(1)
    
    print("\n✓ All tests passed successfully!")


if __name__ == "__main__":
    main()