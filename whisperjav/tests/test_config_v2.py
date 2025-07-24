#!/usr/bin/env python3
"""
Comprehensive test suite for WhisperJAV v2.0 configuration system.
Run this after implementing the new configuration to verify everything works.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add WhisperJAV to path
# Since tests are now in whisperjav/tests, we need to go up two levels
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from whisperjav.config.transcription_tuner_v2 import TranscriptionTunerV2
    print("✓ Successfully imported TranscriptionTunerV2")
except ImportError as e:
    print(f"✗ Failed to import TranscriptionTunerV2: {e}")
    print("  Make sure you've created transcription_tuner_v2.py in whisperjav/config/")
    print("  Current working directory:", Path.cwd())
    print("  Looking for file at:", Path(__file__).parent.parent / "config" / "transcription_tuner_v2.py")
    sys.exit(1)


class ConfigV2Tester:
    """Test suite for the new configuration system."""
    
    def __init__(self):
        self.tuner = None
        self.passed = 0
        self.failed = 0
        self.tests_run = 0
        
    def run_all_tests(self):
        """Run all test suites."""
        print("\n" + "="*60)
        print("WhisperJAV v2.0 Configuration Test Suite")
        print("="*60)
        
        # Initialize tuner
        if not self._test_initialization():
            print("\n✗ FATAL: Could not initialize TranscriptionTunerV2")
            return False
            
        # Run test suites
        self._test_basic_resolution()
        self._test_sensitivity_profiles()
        self._test_model_selection()
        self._test_vad_configuration()
        self._test_pipeline_sensitivity_control()
        self._test_provider_specific_params()
        self._test_feature_resolution()
        self._test_error_handling()
        
        # Summary
        print("\n" + "="*60)
        print(f"Test Summary: {self.passed} passed, {self.failed} failed out of {self.tests_run}")
        print("="*60)
        
        return self.failed == 0
    
    def _test_initialization(self):
        """Test configuration loading."""
        print("\n--- Testing Initialization ---")
        
        try:
            self.tuner = TranscriptionTunerV2()
            self._pass("Configuration loaded successfully")
            
            # Check version
            if self.tuner.config.get("version") == "2.0":
                self._pass("Configuration version is 2.0")
            else:
                self._fail(f"Wrong version: {self.tuner.config.get('version')}")
                
            return True
            
        except Exception as e:
            self._fail(f"Failed to initialize: {e}")
            return False
    
    def _test_basic_resolution(self):
        """Test basic parameter resolution."""
        print("\n--- Testing Basic Resolution ---")
        
        # Test all pipeline/sensitivity combinations
        pipelines = ["faster", "fast", "balanced"]
        sensitivities = ["conservative", "balanced", "aggressive"]
        
        for pipeline in pipelines:
            for sensitivity in sensitivities:
                try:
                    config = self.tuner.resolve_params(pipeline, sensitivity)
                    
                    # Check required keys
                    required_keys = ["model", "common_params", "provider_params", 
                                   "features", "workflow", "effective_sensitivity"]
                    missing = [k for k in required_keys if k not in config]
                    
                    if missing:
                        self._fail(f"{pipeline}/{sensitivity}: Missing keys: {missing}")
                    else:
                        self._pass(f"{pipeline}/{sensitivity}: Resolution successful")
                        
                except Exception as e:
                    self._fail(f"{pipeline}/{sensitivity}: {e}")
    
    def _test_sensitivity_profiles(self):
        """Test sensitivity parameter values."""
        print("\n--- Testing Sensitivity Profiles ---")
        
        # Expected values from reviewer's table
        expected = {
            "conservative": {
                "beam_size": 1,
                "temperature": [0.0],
                "no_speech_threshold": 0.8,
                "vad_threshold": 0.5
            },
            "balanced": {
                "beam_size": 2,
                "temperature": [0.0, 0.2],
                "no_speech_threshold": 0.6,
                "vad_threshold": 0.35
            },
            "aggressive": {
                "beam_size": 5,
                "temperature": [0.0, 0.4],
                "no_speech_threshold": 0.4,
                "vad_threshold": 0.2
            }
        }
        
        for sensitivity, expected_values in expected.items():
            config = self.tuner.resolve_params("balanced", sensitivity)
            params = config["common_params"]
            
            # Check beam size
            if params.get("beam_size") == expected_values["beam_size"]:
                self._pass(f"{sensitivity}: Correct beam_size")
            else:
                self._fail(f"{sensitivity}: Wrong beam_size: {params.get('beam_size')}")
                
            # Check temperature
            if params.get("temperature") == expected_values["temperature"]:
                self._pass(f"{sensitivity}: Correct temperature")
            else:
                self._fail(f"{sensitivity}: Wrong temperature: {params.get('temperature')}")
                
            # Check no_speech_threshold
            if params.get("no_speech_threshold") == expected_values["no_speech_threshold"]:
                self._pass(f"{sensitivity}: Correct no_speech_threshold")
            else:
                self._fail(f"{sensitivity}: Wrong no_speech_threshold: {params.get('no_speech_threshold')}")
    
    def _test_model_selection(self):
        """Test model selection for different modes."""
        print("\n--- Testing Model Selection ---")
        
        # Faster mode should use turbo
        config = self.tuner.resolve_params("faster", "balanced")
        if config["model"]["model_name"] == "turbo":
            self._pass("Faster mode uses turbo model")
        else:
            self._fail(f"Faster mode wrong model: {config['model']['model_name']}")
            
        # Fast/Balanced should use large-v2
        for mode in ["fast", "balanced"]:
            config = self.tuner.resolve_params(mode, "balanced")
            if config["model"]["model_name"] == "large-v2":
                self._pass(f"{mode} mode uses large-v2 model")
            else:
                self._fail(f"{mode} mode wrong model: {config['model']['model_name']}")
                
        # Translation override
        config = self.tuner.resolve_params("faster", "balanced", task="translate")
        if config["model"]["model_name"] == "large-v2":
            self._pass("Faster mode + translate uses large-v2")
        else:
            self._fail(f"Translation override failed: {config['model']['model_name']}")
    
    def _test_vad_configuration(self):
        """Test VAD configuration and sensitivity adjustments."""
        print("\n--- Testing VAD Configuration ---")
        
        # Test VAD threshold multipliers
        config = self.tuner.resolve_params("balanced", "conservative")
        vad_config = config["features"].get("vad", {}).get("config", {})
        vad_threshold = vad_config.get("default_params", {}).get("threshold")
        
        # Conservative should have ~0.5 (0.35 * 1.43)
        expected_conservative = 0.35 * 1.43
        
        # Debug output
        print(f"  DEBUG: VAD threshold = {vad_threshold}")
        print(f"  DEBUG: Expected = {expected_conservative}")
        print(f"  DEBUG: Full VAD config = {vad_config}")
        
        if abs(vad_threshold - expected_conservative) < 0.01:
            self._pass("Conservative VAD threshold correct")
        else:
            self._fail(f"Conservative VAD threshold: {vad_threshold}, expected ~{expected_conservative}")
            
        # Test min_speech_duration override
        min_duration = vad_config.get("default_params", {}).get("min_speech_duration_ms")
        if min_duration == 300:
            self._pass("Conservative min_speech_duration correct")
        else:
            self._fail(f"Conservative min_speech_duration: {min_duration}, expected 300")
    
    def _test_pipeline_sensitivity_control(self):
        """Test pipeline-specific sensitivity control."""
        print("\n--- Testing Pipeline Sensitivity Control ---")
        
        # Add test pipelines with control
        test_pipelines = {
            "test_force": {
                "description": "Test forced sensitivity",
                "workflow": {
                    "model": "whisper-large-v2",
                    "backend": "whisper",
                    "features": {"vad": "none"}
                },
                "sensitivity_control": {
                    "mode": "force",
                    "sensitivity": "conservative",
                    "user_message": "Test force message"
                }
            },
            "test_restrict": {
                "description": "Test restricted sensitivity",
                "workflow": {
                    "model": "whisper-turbo",
                    "backend": "stable-ts",
                    "features": {"vad": "none"}
                },
                "sensitivity_control": {
                    "mode": "restrict",
                    "allowed": ["balanced", "aggressive"],
                    "default": "aggressive"
                }
            }
        }
        
        # Temporarily add test pipelines
        original_pipelines = self.tuner.config["pipelines"].copy()
        self.tuner.config["pipelines"].update(test_pipelines)
        
        try:
            # Test forced sensitivity
            config = self.tuner.resolve_params("test_force", "aggressive")
            if config["effective_sensitivity"] == "conservative":
                self._pass("Force mode works correctly")
            else:
                self._fail(f"Force mode failed: {config['effective_sensitivity']}")
                
            # Test restricted sensitivity
            config = self.tuner.resolve_params("test_restrict", "conservative")
            if config["effective_sensitivity"] == "aggressive":
                self._pass("Restrict mode works correctly")
            else:
                self._fail(f"Restrict mode failed: {config['effective_sensitivity']}")
                
            # Test available sensitivities
            available = self.tuner.get_available_sensitivities("test_restrict")
            if available == ["balanced", "aggressive"]:
                self._pass("get_available_sensitivities works correctly")
            else:
                self._fail(f"Wrong available sensitivities: {available}")
                
        finally:
            # Restore original pipelines
            self.tuner.config["pipelines"] = original_pipelines
    
    def _test_provider_specific_params(self):
        """Test provider-specific parameter handling."""
        print("\n--- Testing Provider-Specific Parameters ---")
        
        config = self.tuner.resolve_params("faster", "balanced")
        
        # Check common params
        common = config["common_params"]
        if "temperature" in common and "beam_size" in common:
            self._pass("Common parameters present")
        else:
            self._fail("Missing common parameters")
            
        # Debug output
        print(f"  DEBUG: provider_params = {config.get('provider_params', {})}")
        
        # Check provider params - should be wrapped in provider name
        provider = config["provider_params"]
        if provider and isinstance(provider, dict):
            # Check if any provider-specific params exist
            if "whisper" in provider and provider["whisper"]:
                self._pass("Provider-specific parameters present")
            else:
                # Maybe it's a different structure, check what we got
                print(f"  DEBUG: Provider params structure: {list(provider.keys()) if provider else 'empty'}")
                if any(provider.values()):  # If any provider has params
                    self._pass("Provider-specific parameters present")
                else:
                    self._fail("Missing provider-specific parameters")
        else:
            self._fail("Missing provider-specific parameters structure")
            
        # Check separation - provider params should not be in common
        whisper_specific = ["compression_ratio_threshold", "logprob_threshold", "word_timestamps"]
        overlapping = [p for p in whisper_specific if p in common]
        if not overlapping:
            self._pass("Provider params correctly separated from common")
        else:
            self._fail(f"Provider params mixed with common params: {overlapping}")
    
    def _test_feature_resolution(self):
        """Test feature configuration resolution."""
        print("\n--- Testing Feature Resolution ---")
        
        # Test scene detection
        config = self.tuner.resolve_params("fast", "balanced")
        if "scene_detection" in config["features"]:
            self._pass("Scene detection enabled for fast mode")
        else:
            self._fail("Scene detection missing for fast mode")
            
        # Test no scene detection
        config = self.tuner.resolve_params("faster", "balanced")
        if "scene_detection" not in config["features"]:
            self._pass("Scene detection disabled for faster mode")
        else:
            self._fail("Scene detection wrongly enabled for faster mode")
            
        # Test VAD
        config = self.tuner.resolve_params("balanced", "balanced")
        if "vad" in config["features"]:
            self._pass("VAD enabled for balanced mode")
            vad = config["features"]["vad"]
            if vad.get("engine") == "silero-v4":
                self._pass("Correct VAD engine selected")
            else:
                self._fail(f"Wrong VAD engine: {vad.get('engine')}")
        else:
            self._fail("VAD missing for balanced mode")
    
    def _test_error_handling(self):
        """Test error handling."""
        print("\n--- Testing Error Handling ---")
        
        # Test unknown pipeline
        try:
            config = self.tuner.resolve_params("unknown_pipeline", "balanced")
            self._fail("Should have raised error for unknown pipeline")
        except ValueError as e:
            if "Unknown pipeline" in str(e):
                self._pass("Correctly handles unknown pipeline")
            else:
                self._fail(f"Wrong error message: {e}")
        except Exception as e:
            self._fail(f"Wrong exception type: {type(e)}")
            
        # Test unsupported task
        try:
            config = self.tuner.resolve_params("faster", "balanced", task="translate")
            # Should succeed due to model override
            self._pass("Model override for unsupported task works")
        except Exception as e:
            self._fail(f"Model override failed: {e}")
    
    def _pass(self, message: str):
        """Record a passing test."""
        self.passed += 1
        self.tests_run += 1
        print(f"  ✓ {message}")
        
    def _fail(self, message: str):
        """Record a failing test."""
        self.failed += 1
        self.tests_run += 1
        print(f"  ✗ {message}")


def print_example_usage():
    """Print example usage of the new system."""
    print("\n" + "="*60)
    print("Example Usage")
    print("="*60)
    
    try:
        tuner = TranscriptionTunerV2()
        
        # Example 1: Basic usage
        print("\nExample 1: Basic Usage")
        config = tuner.resolve_params("faster", "aggressive")
        print(f"  Model: {config['model']['model_name']}")
        print(f"  Beam Size: {config['common_params']['beam_size']}")
        print(f"  Temperature: {config['common_params']['temperature']}")
        
        # Example 2: Translation
        print("\nExample 2: Translation Override")
        config = tuner.resolve_params("faster", "balanced", task="translate")
        print(f"  Model: {config['model']['model_name']} (switched for translation)")
        
        # Example 3: Pipeline info
        print("\nExample 3: Pipeline Information")
        info = tuner.get_pipeline_info("balanced")
        print(f"  Description: {info['description']}")
        print(f"  Features: {json.dumps(info['features'], indent=4)}")
        
    except Exception as e:
        print(f"Error in examples: {e}")


if __name__ == "__main__":
    # Run tests
    tester = ConfigV2Tester()
    success = tester.run_all_tests()
    
    # Show examples if tests passed
    if success:
        print_example_usage()
        
    # Exit with appropriate code
    sys.exit(0 if success else 1)