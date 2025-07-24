#!/usr/bin/env python3
"""
Comprehensive test suite for the WhisperJAV v3 configuration system.

This suite verifies that TranscriptionTunerV3 correctly interprets the modular
asr_config.v3.json schema and produces a valid, structured configuration object.
"""

import unittest
import json
from pathlib import Path
import sys
import tempfile
from copy import deepcopy

# Add WhisperJAV project root to the Python path to allow imports.
# This assumes the test script is run from a directory where 'whisperjav' is a sub-folder,
# or that the project is installed as a package.
try:
    from whisperjav.config.transcription_tuner_v3 import TranscriptionTunerV3
except ImportError:
    # If running from the repo root, this adjustment might be needed.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from whisperjav.config.transcription_tuner_v3 import TranscriptionTunerV3


class TestTranscriptionTunerV3(unittest.TestCase):
    """Test suite for the TranscriptionTunerV3 and the v3 configuration schema."""

    @classmethod
    def setUpClass(cls):
        """Set up a temporary configuration file for all tests."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.config_path = Path(cls.temp_dir.name) / "asr_config.v3.json"

        # This is the comprehensive v3 config structure the tests will run against.
        # It mirrors the final design we implemented.
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

    def test_initialization_and_validation(self):
        """Test that the tuner initializes correctly and handles errors."""
        self.assertIsNotNone(self.tuner)
        self.assertEqual(self.tuner.config["version"], "3.0")

        # Test failure on missing file
        with self.assertRaises(FileNotFoundError):
            TranscriptionTunerV3(config_path=Path("non_existent_file.json"))

        # Test failure on wrong version
        bad_config_data = deepcopy(self.test_config_data)
        bad_config_data["version"] = "2.0"
        bad_config_path = Path(self.temp_dir.name) / "bad_version.json"
        with open(bad_config_path, 'w') as f:
            json.dump(bad_config_data, f)
        
        with self.assertRaises(ValueError):
            TranscriptionTunerV3(config_path=bad_config_path)

    def test_resolved_config_structure(self):
        """Test that a resolved config has the correct V3 top-level structure."""
        config = self.tuner.resolve_params("balanced", "balanced", "transcribe")
        
        expected_keys = ["pipeline_name", "sensitivity_name", "workflow", "model", "params", "features", "task", "language"]
        for key in expected_keys:
            self.assertIn(key, config)
        
        # Check nested structure of params
        self.assertIn("decoder", config["params"])
        self.assertIn("vad", config["params"])
        self.assertIn("provider", config["params"])

    def test_parameter_values_for_conservative_profile(self):
        """Verify that the 'conservative' profile resolves to the correct parameter values."""
        config = self.tuner.resolve_params("balanced", "conservative", "transcribe")
        params = config["params"]

        self.assertEqual(config["sensitivity_name"], "conservative")
        self.assertEqual(params["decoder"]["beam_size"], 1)
        self.assertEqual(params["vad"]["threshold"], 0.5)
        self.assertEqual(params["provider"]["no_speech_threshold"], 0.8)

    def test_parameter_values_for_aggressive_profile(self):
        """Verify that the 'aggressive' profile resolves to the correct parameter values."""
        config = self.tuner.resolve_params("balanced", "aggressive", "transcribe")
        params = config["params"]

        self.assertEqual(config["sensitivity_name"], "aggressive")
        self.assertEqual(params["decoder"]["beam_size"], 5)
        self.assertEqual(params["vad"]["threshold"], 0.2)
        self.assertEqual(params["provider"]["no_speech_threshold"], 0.4)

    def test_model_selection_with_override(self):
        """Test that the correct model is selected, including task-based overrides."""
        # 'faster' pipeline defaults to 'whisper-turbo' for transcribe
        config_transcribe = self.tuner.resolve_params("faster", "balanced", "transcribe")
        self.assertEqual(config_transcribe["model"]["model_name"], "turbo")

        # 'faster' pipeline should switch to 'whisper-large-v2' for translate
        config_translate = self.tuner.resolve_params("faster", "balanced", "translate")
        self.assertEqual(config_translate["model"]["model_name"], "large-v2")
        
        # 'balanced' pipeline has no override, should use its default model for both tasks
        config_balanced_transcribe = self.tuner.resolve_params("balanced", "balanced", "transcribe")
        self.assertEqual(config_balanced_transcribe["model"]["model_name"], "large-v2")
        config_balanced_translate = self.tuner.resolve_params("balanced", "balanced", "translate")
        self.assertEqual(config_balanced_translate["model"]["model_name"], "large-v2")

    def test_feature_resolution_per_pipeline(self):
        """Test that feature configurations are correctly resolved for each pipeline."""
        # 'faster' pipeline: only post_processing
        config_faster = self.tuner.resolve_params("faster", "balanced", "transcribe")
        self.assertIn("post_processing", config_faster["features"])
        self.assertNotIn("scene_detection", config_faster["features"])
        self.assertTrue(config_faster["features"]["post_processing"]["remove_hallucinations"])

        # 'fast' pipeline: only scene_detection
        config_fast = self.tuner.resolve_params("fast", "balanced", "transcribe")
        self.assertIn("scene_detection", config_fast["features"])
        self.assertNotIn("post_processing", config_fast["features"])
        self.assertEqual(config_fast["features"]["scene_detection"]["max_duration"], 30.0)

        # 'balanced' pipeline: both scene_detection and post_processing
        config_balanced = self.tuner.resolve_params("balanced", "balanced", "transcribe")
        self.assertIn("scene_detection", config_balanced["features"])
        self.assertIn("post_processing", config_balanced["features"])

    def test_error_handling_for_invalid_names(self):
        """Test that the tuner raises appropriate errors for invalid pipeline or sensitivity names."""
        with self.assertRaisesRegex(ValueError, "Unknown pipeline specified"):
            self.tuner.resolve_params("unknown_pipeline", "balanced", "transcribe")

        with self.assertRaisesRegex(ValueError, "Unknown sensitivity specified"):
            self.tuner.resolve_params("balanced", "unknown_sensitivity", "transcribe")

    def test_deep_copy_of_resolved_parameters(self):
        """Test that resolved parameters are deep copies, preventing mutation of the base config."""
        config1 = self.tuner.resolve_params("balanced", "balanced", "transcribe")
        
        # Mutate a value in the resolved config
        config1["params"]["decoder"]["beam_size"] = 999
        
        # Resolve the same config again
        config2 = self.tuner.resolve_params("balanced", "balanced", "transcribe")

        # The second resolved config should NOT have the mutated value
        self.assertNotEqual(config2["params"]["decoder"]["beam_size"], 999)
        self.assertEqual(config2["params"]["decoder"]["beam_size"], 2) # Should be the original value

if __name__ == '__main__':
    print("="*70)
    print(" Running WhisperJAV v3 Configuration Test Suite")
    print("="*70)
    unittest.main(verbosity=2)
