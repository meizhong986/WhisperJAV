#!/usr/bin/env python3
"""
Comprehensive test suite for the WhisperJAV v3 configuration system.
This suite verifies that TranscriptionTunerV3 correctly interprets the modular
asr_config.v3.json schema.
"""

import unittest
import json
from pathlib import Path
import sys
import os
import tempfile

# Add WhisperJAV project root to the Python path to allow imports
# This assumes the test script is run from the project root or a similar context.
# Adjust the path as necessary for your project structure.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from whisperjav.config.transcription_tuner_v3 import TranscriptionTunerV3
except ImportError as e:
    print("="*80)
    print("FATAL: Could not import TranscriptionTunerV3.")
    print(f"Error: {e}")
    print("\nPlease ensure that:")
    print("1. You have created 'whisperjav/config/transcription_tuner_v3.py'.")
    print("2. You are running this test from a location where 'whisperjav' is a recognizable package.")
    print(f"   Current Working Directory: {Path.cwd()}")
    print("="*80)
    sys.exit(1)


class TestTranscriptionTunerV3(unittest.TestCase):
    """Test suite for the TranscriptionTunerV3 and the v3 configuration schema."""

    @classmethod
    def setUpClass(cls):
        """Set up a temporary configuration file for all tests."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.config_path = Path(cls.temp_dir.name) / "asr_config.v3.json"

        # This is the comprehensive v3 config structure the tests will run against.
        cls.test_config_data = {
            "version": "3.0",
            "models": {
                "whisper-turbo": {"provider": "openai_whisper", "model_name": "turbo", "supported_tasks": ["transcribe"]},
                "whisper-large-v2": {"provider": "openai_whisper", "model_name": "large-v2", "supported_tasks": ["transcribe", "translate"]}
            },
            "vad_engines": {"silero-v4": {"provider": "silero", "repo": "snakers4/silero-vad:v4.0"}},
            "parameter_sets": {
                "decoder_params": {
                    "conservative": {"beam_size": 1, "temperature": [0.0]},
                    "balanced": {"beam_size": 2, "temperature": [0.0, 0.2]},
                    "aggressive": {"beam_size": 5, "temperature": [0.0, 0.4]}
                },
                "vad_params": {
                    "conservative": {"threshold": 0.5},
                    "balanced": {"threshold": 0.35},
                    "aggressive": {"threshold": 0.2}
                },
                "provider_specific_params": {
                    "openai_whisper": {
                        "conservative": {"no_speech_threshold": 0.8},
                        "balanced": {"no_speech_threshold": 0.6},
                        "aggressive": {"no_speech_threshold": 0.4}
                    }
                }
            },
            "sensitivity_profiles": {
                "conservative": {"decoder": "conservative", "vad": "conservative", "provider_settings": "conservative"},
                "balanced": {"decoder": "balanced", "vad": "balanced", "provider_settings": "balanced"},
                "aggressive": {"decoder": "aggressive", "vad": "aggressive", "provider_settings": "aggressive"}
            },
            "feature_configs": {
                "scene_detection": {"default": {"max_duration": 30.0}},
                "post_processing": {"standard": {"remove_hallucinations": True}}
            },
            "pipelines": {
                "faster": {
                    "workflow": {"model": "whisper-turbo", "vad": "none", "features": {"post_processing": "standard"}},
                    "model_overrides": {"translate": "whisper-large-v2"}
                },
                "fast": {
                    "workflow": {"model": "whisper-large-v2", "vad": "none", "features": {"scene_detection": "default"}}
                },
                "balanced": {
                    "workflow": {"model": "whisper-large-v2", "vad": "silero-v4", "features": {"scene_detection": "default", "post_processing": "standard"}}
                }
            },
            "defaults": {"language": "ja"}
        }

        with open(cls.config_path, 'w', encoding='utf-8') as f:
            json.dump(cls.test_config_data, f, indent=2)

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory after all tests."""
        cls.temp_dir.cleanup()

    def setUp(self):
        """Create a new tuner instance for each test."""
        self.tuner = TranscriptionTunerV3(config_path=self.config_path)

    def test_01_initialization(self):
        """Test that the tuner initializes correctly."""
        self.assertIsNotNone(self.tuner)
        self.assertEqual(self.tuner.config["version"], "3.0")

    def test_02_initialization_errors(self):
        """Test that initialization fails for missing or incorrect files."""
        with self.assertRaises(FileNotFoundError):
            TranscriptionTunerV3(config_path=Path("non_existent_file.json"))

        # Create a config with the wrong version
        bad_config_path = Path(self.temp_dir.name) / "bad_version.json"
        bad_config_data = {"version": "2.0"}
        with open(bad_config_path, 'w') as f:
            json.dump(bad_config_data, f)
        
        with self.assertRaises(ValueError):
            TranscriptionTunerV3(config_path=bad_config_path)

    def test_03_basic_resolution_structure(self):
        """Test that a resolved config has the correct top-level structure."""
        config = self.tuner.resolve_params("balanced", "balanced", "transcribe")
        
        self.assertIn("workflow", config)
        self.assertIn("model", config)
        self.assertIn("params", config)
        self.assertIn("features", config)
        self.assertIn("task", config)
        self.assertIn("language", config)
        
        # Check nested structure of params
        self.assertIn("decoder", config["params"])
        self.assertIn("vad", config["params"])
        self.assertIn("provider", config["params"])

    def test_04_parameter_correctness_conservative(self):
        """Verify that the 'conservative' profile resolves to the correct values."""
        config = self.tuner.resolve_params("balanced", "conservative", "transcribe")
        params = config["params"]

        # Check decoder params
        self.assertEqual(params["decoder"]["beam_size"], 1)
        self.assertEqual(params["decoder"]["temperature"], [0.0])

        # Check VAD params
        self.assertEqual(params["vad"]["threshold"], 0.5)
        
        # Check provider-specific params
        self.assertEqual(params["provider"]["no_speech_threshold"], 0.8)

    def test_05_parameter_correctness_aggressive(self):
        """Verify that the 'aggressive' profile resolves to the correct values."""
        config = self.tuner.resolve_params("balanced", "aggressive", "transcribe")
        params = config["params"]

        # Check decoder params
        self.assertEqual(params["decoder"]["beam_size"], 5)
        self.assertEqual(params["decoder"]["temperature"], [0.0, 0.4])

        # Check VAD params
        self.assertEqual(params["vad"]["threshold"], 0.2)
        
        # Check provider-specific params
        self.assertEqual(params["provider"]["no_speech_threshold"], 0.4)

    def test_06_model_selection_and_override(self):
        """Test that the correct model is selected, including task-based overrides."""
        # Test default model for 'faster' pipeline
        config_transcribe = self.tuner.resolve_params("faster", "balanced", "transcribe")
        self.assertEqual(config_transcribe["model"]["model_name"], "turbo")

        # Test model override for 'translate' task
        config_translate = self.tuner.resolve_params("faster", "balanced", "translate")
        self.assertEqual(config_translate["model"]["model_name"], "large-v2")
        
        # Test a pipeline without an override
        config_no_override = self.tuner.resolve_params("balanced", "balanced", "translate")
        self.assertEqual(config_no_override["model"]["model_name"], "large-v2")

    def test_07_feature_resolution(self):
        """Test that feature configurations are correctly resolved."""
        # 'fast' pipeline should have scene_detection
        config_fast = self.tuner.resolve_params("fast", "balanced", "transcribe")
        self.assertIn("scene_detection", config_fast["features"])
        self.assertEqual(config_fast["features"]["scene_detection"]["max_duration"], 30.0)
        self.assertNotIn("post_processing", config_fast["features"])

        # 'balanced' pipeline should have both
        config_balanced = self.tuner.resolve_params("balanced", "balanced", "transcribe")
        self.assertIn("scene_detection", config_balanced["features"])
        self.assertIn("post_processing", config_balanced["features"])
        self.assertTrue(config_balanced["features"]["post_processing"]["remove_hallucinations"])

        # 'faster' pipeline should only have post_processing
        config_faster = self.tuner.resolve_params("faster", "balanced", "transcribe")
        self.assertNotIn("scene_detection", config_faster["features"])
        self.assertIn("post_processing", config_faster["features"])

    def test_08_error_handling_for_unknown_inputs(self):
        """Test that the tuner raises appropriate errors for invalid inputs."""
        with self.assertRaises(ValueError):
            self.tuner.resolve_params("unknown_pipeline", "balanced", "transcribe")

        with self.assertRaises(KeyError):
            self.tuner.resolve_params("balanced", "unknown_sensitivity", "transcribe")

    def test_09_deep_copy_of_parameters(self):
        """Test that resolved parameters are deep copies, not references."""
        config1 = self.tuner.resolve_params("balanced", "balanced", "transcribe")
        
        # Modify a value in the resolved config
        config1["params"]["decoder"]["beam_size"] = 99
        
        # Resolve the same config again
        config2 = self.tuner.resolve_params("balanced", "balanced", "transcribe")

        # The second resolved config should NOT have the modified value
        self.assertNotEqual(config2["params"]["decoder"]["beam_size"], 99)
        self.assertEqual(config2["params"]["decoder"]["beam_size"], 2)


if __name__ == '__main__':
    print("="*70)
    print(" Running WhisperJAV v3 Configuration Test Suite")
    print("="*70)
    unittest.main(verbosity=2)
