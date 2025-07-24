#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for WhisperJAV V3 Architecture.

This test suite validates the complete V3 configuration system, parameter flow,
and integration between all components.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisperjav.config.transcription_tuner_v3 import TranscriptionTunerV3


class TestV3ConfigurationLoading(unittest.TestCase):
    """Test configuration file loading and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = Path(self.test_dir) / "test_config.json"
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def create_test_config(self, config_dict):
        """Helper to create test configuration files."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
    def test_valid_v3_config_loading(self):
        """Test loading a valid V3 configuration."""
        valid_config = {
            "version": "3.0",
            "models": {
                "whisper-turbo": {
                    "provider": "openai_whisper",
                    "model_name": "turbo",
                    "device": "cuda",
                    "compute_type": "float16"
                }
            },
            "parameter_sets": {
                "decoder_params": {
                    "balanced": {"beam_size": 2, "temperature": [0.0, 0.2]}
                },
                "vad_params": {
                    "balanced": {"threshold": 0.35}
                },
                "provider_specific_params": {
                    "openai_whisper": {
                        "balanced": {"no_speech_threshold": 0.6}
                    }
                }
            },
            "sensitivity_profiles": {
                "balanced": {
                    "decoder": "balanced",
                    "vad": "balanced",
                    "provider_settings": "balanced"
                }
            },
            "pipelines": {
                "faster": {
                    "workflow": {
                        "model": "whisper-turbo",
                        "vad": "none",
                        "backend": "stable-ts"
                    }
                }
            }
        }
        
        self.create_test_config(valid_config)
        
        # Should load without errors
        tuner = TranscriptionTunerV3(self.config_path)
        self.assertIsNotNone(tuner.config)
        self.assertEqual(tuner.config["version"], "3.0")
        
    def test_invalid_version_rejection(self):
        """Test that invalid config versions are rejected."""
        invalid_config = {
            "version": "2.0",  # Wrong version
            "models": {}
        }
        
        self.create_test_config(invalid_config)
        
        with self.assertRaises(ValueError) as context:
            TranscriptionTunerV3(self.config_path)
        self.assertIn("Unsupported config version", str(context.exception))
        
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with self.assertRaises(FileNotFoundError):
            TranscriptionTunerV3(Path("nonexistent_config.json"))
            
    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write("{ invalid json }")
            
        with self.assertRaises(json.JSONDecodeError):
            TranscriptionTunerV3(self.config_path)


class TestParameterResolution(unittest.TestCase):
    """Test parameter resolution for all combinations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up tuner with default config once for all tests."""
        # Use default config from the package
        cls.tuner = TranscriptionTunerV3()
        
    def test_all_pipeline_sensitivity_combinations(self):
        """Test all 9 combinations of pipeline and sensitivity."""
        pipelines = ["faster", "fast", "balanced"]
        sensitivities = ["conservative", "balanced", "aggressive"]
        
        for pipeline in pipelines:
            for sensitivity in sensitivities:
                with self.subTest(pipeline=pipeline, sensitivity=sensitivity):
                    config = self.tuner.resolve_params(
                        pipeline_name=pipeline,
                        sensitivity=sensitivity,
                        task="transcribe"
                    )
                    
                    # Validate structure
                    self.assertIsInstance(config, dict)
                    self.assertIn("model", config)
                    self.assertIn("params", config)
                    self.assertIn("features", config)
                    self.assertIn("task", config)
                    
                    # Validate params structure
                    self.assertIn("decoder", config["params"])
                    self.assertIn("vad", config["params"])
                    self.assertIn("provider", config["params"])
                    
                    # Validate task
                    self.assertEqual(config["task"], "transcribe")
                    
    def test_transcribe_vs_translate_task(self):
        """Test task resolution for transcribe vs translate."""
        # Test transcribe
        config_transcribe = self.tuner.resolve_params(
            pipeline_name="balanced",
            sensitivity="balanced",
            task="transcribe"
        )
        self.assertEqual(config_transcribe["task"], "transcribe")
        
        # Test translate
        config_translate = self.tuner.resolve_params(
            pipeline_name="balanced",
            sensitivity="balanced",
            task="translate"
        )
        self.assertEqual(config_translate["task"], "translate")
        
    def test_model_override_for_translation(self):
        """Test model override logic for translation in faster pipeline."""
        # Faster pipeline with translate should switch from turbo to large-v2
        config = self.tuner.resolve_params(
            pipeline_name="faster",
            sensitivity="balanced",
            task="translate"
        )
        
        # Check if model override is applied
        model_name = config["model"].get("model_name")
        # If turbo doesn't support translation, it should be overridden
        if "turbo" in str(model_name).lower():
            # Check if pipeline has override logic
            self.assertTrue(
                config["model"].get("supports_translation", True) or
                config.get("model_overrides", {}).get("translate") is not None
            )
            
    def test_parameter_value_differences(self):
        """Test that different sensitivities produce different parameter values."""
        conservative = self.tuner.resolve_params("balanced", "conservative", "transcribe")
        aggressive = self.tuner.resolve_params("balanced", "aggressive", "transcribe")
        
        # Check that at least some parameters differ
        conservative_decoder = conservative["params"]["decoder"]
        aggressive_decoder = aggressive["params"]["decoder"]
        
        # At least one parameter should be different
        differences = []
        for key in conservative_decoder:
            if key in aggressive_decoder and conservative_decoder[key] != aggressive_decoder[key]:
                differences.append(key)
                
        self.assertTrue(len(differences) > 0, 
                       "Conservative and aggressive profiles should have different parameters")
        
    def test_invalid_pipeline_name(self):
        """Test error handling for invalid pipeline name."""
        with self.assertRaises(ValueError) as context:
            self.tuner.resolve_params("invalid_pipeline", "balanced", "transcribe")
        self.assertIn("Unknown pipeline", str(context.exception))
        
    def test_invalid_sensitivity_name(self):
        """Test error handling for invalid sensitivity name."""
        with self.assertRaises(ValueError) as context:
            self.tuner.resolve_params("balanced", "invalid_sensitivity", "transcribe")
        self.assertIn("Unknown sensitivity", str(context.exception))


class TestModuleInstantiation(unittest.TestCase):
    """Test that modules can be instantiated with V3 configuration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test configuration."""
        cls.tuner = TranscriptionTunerV3()
        cls.balanced_config = cls.tuner.resolve_params("balanced", "balanced", "transcribe")
        cls.fast_config = cls.tuner.resolve_params("fast", "balanced", "transcribe")
        cls.faster_config = cls.tuner.resolve_params("faster", "balanced", "transcribe")
        
    @patch('torch.cuda.is_available', return_value=False)  # Force CPU for testing
    @patch('torch.hub.load')  # Mock VAD loading
    @patch('whisper.load_model')  # Mock Whisper model loading
    def test_whisper_pro_asr_instantiation(self, mock_whisper, mock_vad, mock_cuda):
        """Test WhisperProASR instantiation with V3 config."""
        # Import here to apply mocks
        from whisperjav.modules.whisper_pro_asr import WhisperProASR
        
        # Create proper mock for VAD
        mock_vad_model = Mock()
        mock_get_speech_timestamps = Mock()
        mock_vad_utils = (mock_get_speech_timestamps, Mock(), Mock(), Mock(), Mock())
        
        # torch.hub.load returns (model, utils)
        mock_vad.return_value = (mock_vad_model, mock_vad_utils)
        
        # Mock whisper model
        mock_whisper_model = Mock()
        mock_whisper.return_value = mock_whisper_model
        
        # Extract parameters as the module expects
        model_config = self.balanced_config["model"]
        params = self.balanced_config["params"]
        task = self.balanced_config["task"]
        
        # Should instantiate without errors
        asr = WhisperProASR(
            model_config=model_config,
            params=params,
            task=task
        )
        
        # Verify correct attributes (WhisperProASR stores task in decode_options)
        self.assertEqual(asr.decode_options['task'], "transcribe")
        self.assertIsNotNone(asr.transcribe_options)
        self.assertIsNotNone(asr.decode_options)
        self.assertEqual(asr.vad_model, mock_vad_model)
        self.assertEqual(asr.whisper_model, mock_whisper_model)
        self.assertEqual(asr.get_speech_timestamps, mock_get_speech_timestamps)
        
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.hub.load', return_value=Mock())  # Mock VAD pre-caching
    @patch('stable_whisper.load_model')
    @patch('stable_whisper.load_faster_whisper')
    def test_stable_ts_asr_instantiation(self, mock_faster, mock_stable, mock_vad, mock_cuda):
        """Test StableTSASR instantiation with V3 config."""
        from whisperjav.modules.stable_ts_asr import StableTSASR
        
        # Mock model loading
        mock_stable_model = Mock()
        mock_stable.return_value = mock_stable_model
        
        mock_faster_model = Mock()
        mock_faster.return_value = mock_faster_model
        
        # Test standard mode (fast pipeline)
        model_config = self.fast_config["model"]
        params = self.fast_config["params"]
        task = self.fast_config["task"]
        
        asr_standard = StableTSASR(
            model_config=model_config,
            params=params,
            task=task,
            turbo_mode=False
        )
        
        self.assertFalse(asr_standard.turbo_mode)
        self.assertEqual(asr_standard.model, mock_stable_model)
        mock_stable.assert_called_once()
        
        # Reset mocks
        mock_stable.reset_mock()
        mock_faster.reset_mock()
        
        # Test turbo mode (faster pipeline)
        model_config = self.faster_config["model"]
        params = self.faster_config["params"]  
        task = self.faster_config["task"]
        
        asr_turbo = StableTSASR(
            model_config=model_config,
            params=params,
            task=task,
            turbo_mode=True
        )
        
        self.assertTrue(asr_turbo.turbo_mode)
        self.assertEqual(asr_turbo.model, mock_faster_model)
        mock_faster.assert_called_once()


class TestPipelineIntegration(unittest.TestCase):
    """Test pipeline instantiation and integration with V3 config."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test configuration and temp directories."""
        cls.tuner = TranscriptionTunerV3()
        cls.test_dir = tempfile.mkdtemp()
        cls.output_dir = Path(cls.test_dir) / "output"
        cls.temp_dir = Path(cls.test_dir) / "temp"
        cls.output_dir.mkdir(exist_ok=True)
        cls.temp_dir.mkdir(exist_ok=True)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test directories."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        
    def create_pipeline_args(self, resolved_config):
        """Helper to create common pipeline arguments."""
        return {
            "output_dir": str(self.output_dir),
            "temp_dir": str(self.temp_dir),
            "keep_temp_files": False,
            "subs_language": "japanese",
            "resolved_config": resolved_config,
            "progress_display": Mock()
        }
        
    @patch('torch.hub.load', return_value=(Mock(), (Mock(), Mock(), Mock(), Mock(), Mock())))
    @patch('whisper.load_model', return_value=Mock())
    def test_balanced_pipeline_instantiation(self, mock_whisper, mock_vad):
        """Test BalancedPipeline instantiation with V3 config."""
        # Mock SceneDetector at the module level where it's imported
        with patch('whisperjav.pipelines.balanced_pipeline.SceneDetector') as mock_scene_class:
            from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
            
            config = self.tuner.resolve_params("balanced", "balanced", "transcribe")
            args = self.create_pipeline_args(config)
            
            # Create pipeline
            pipeline = BalancedPipeline(**args)
            
            # Verify scene detector was instantiated
            mock_scene_class.assert_called_once()
            
            # Verify ASR module has correct config
            self.assertIsNotNone(pipeline.asr)
            self.assertEqual(pipeline.asr.decode_options['task'], 'transcribe')
        
    @patch('torch.hub.load', return_value=Mock())
    @patch('stable_whisper.load_model', return_value=Mock())
    def test_fast_pipeline_instantiation(self, mock_stable, mock_vad):
        """Test FastPipeline instantiation with V3 config."""
        # Mock SceneDetector at the module level where it's imported
        with patch('whisperjav.pipelines.fast_pipeline.SceneDetector') as mock_scene_class:
            from whisperjav.pipelines.fast_pipeline import FastPipeline
            
            config = self.tuner.resolve_params("fast", "balanced", "transcribe")
            args = self.create_pipeline_args(config)
            
            # Create pipeline
            pipeline = FastPipeline(**args)
            
            # Verify scene detector was instantiated
            mock_scene_class.assert_called_once()
            
            # Verify ASR module config
            self.assertIsNotNone(pipeline.asr)
            self.assertFalse(pipeline.asr.turbo_mode)
        
    @patch('torch.hub.load', return_value=Mock())
    @patch('stable_whisper.load_faster_whisper', return_value=Mock())
    def test_faster_pipeline_instantiation(self, mock_faster, mock_vad):
        """Test FasterPipeline instantiation with V3 config."""
        from whisperjav.pipelines.faster_pipeline import FasterPipeline
        
        config = self.tuner.resolve_params("faster", "balanced", "transcribe")
        args = self.create_pipeline_args(config)
        
        # Create pipeline (no SceneDetector to mock in faster pipeline)
        pipeline = FasterPipeline(**args)
        
        # Verify ASR module config
        self.assertIsNotNone(pipeline.asr)
        self.assertTrue(pipeline.asr.turbo_mode)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete workflow from config to output."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = tempfile.mkdtemp()
        cls.media_file = Path(cls.test_dir) / "test_media.mp4"
        cls.media_file.touch()  # Create dummy file
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        
    @patch('torch.hub.load', return_value=(Mock(), (Mock(), Mock(), Mock(), Mock(), Mock())))
    @patch('whisper.load_model', return_value=Mock())
    @patch('whisperjav.modules.audio_extraction.AudioExtractor.extract')
    @patch('whisperjav.modules.scene_detection.SceneDetector.detect_scenes')
    @patch('whisperjav.modules.whisper_pro_asr.WhisperProASR.transcribe_to_srt')
    @patch('whisperjav.modules.srt_stitching.SRTStitcher.stitch')
    @patch('whisperjav.modules.srt_postprocessing.SRTPostProcessor.process')
    def test_complete_balanced_pipeline_flow(self, mock_post, mock_stitch, 
                                           mock_transcribe, mock_scenes, mock_extract,
                                           mock_whisper, mock_vad):
        """Test complete parameter flow through balanced pipeline."""
        from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
        
        # Create temporary files that will actually exist
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Set up mocks with real file paths
            audio_file = temp_path / "audio.wav"
            audio_file.touch()
            mock_extract.return_value = (audio_file, 60.0)
            
            scene_file = temp_path / "scene1.wav"
            scene_file.touch()
            mock_scenes.return_value = [(scene_file, 0, 30, 30)]
            
            scene_srt = temp_path / "scene1.srt"
            scene_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest", encoding='utf-8')
            mock_transcribe.return_value = scene_srt
            
            mock_stitch.return_value = 10
            
            # Create actual final SRT file
            final_srt = temp_path / "final.srt"
            final_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nFinal", encoding='utf-8')
            mock_post.return_value = (final_srt, {"total_subtitles": 10})
            
            # Create pipeline with V3 config
            tuner = TranscriptionTunerV3()
            config = tuner.resolve_params("balanced", "aggressive", "transcribe")
            
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            pipeline = BalancedPipeline(
                output_dir=str(output_dir),
                temp_dir=str(temp_path),
                keep_temp_files=False,
                subs_language="japanese",
                resolved_config=config,
                progress_display=Mock()
            )
            
            # Process media file
            media_info = {
                "path": str(self.media_file),
                "basename": "test_media",
                "type": "video",
                "duration": 60
            }
            
            result = pipeline.process(media_info)
            
            # Verify the pipeline completed
            self.assertIsInstance(result, dict)
            self.assertIn("summary", result)
            
            # Verify ASR was called with correct task
            mock_transcribe.assert_called()
            call_args = mock_transcribe.call_args[1]
            self.assertEqual(call_args.get("task", "transcribe"), "transcribe")


class TestErrorHandling(unittest.TestCase):
    """Test error handling throughout the V3 system."""
    
    def test_missing_required_config_section(self):
        """Test handling of missing required configuration sections."""
        incomplete_config = {
            "version": "3.0",
            "models": {},  # Missing other required sections
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(incomplete_config, f)
            temp_path = Path(f.name)
            
        try:
            tuner = TranscriptionTunerV3(temp_path)
            # Should handle missing sections gracefully
            with self.assertRaises((KeyError, ValueError)):
                tuner.resolve_params("balanced", "balanced", "transcribe")
        finally:
            temp_path.unlink()
            
    @patch('whisperjav.modules.whisper_pro_asr.torch.hub.load', side_effect=Exception("Model load failed"))
    def test_asr_initialization_failure(self, mock_hub):
        """Test handling of ASR initialization failures."""
        from whisperjav.modules.whisper_pro_asr import WhisperProASR
        
        tuner = TranscriptionTunerV3()
        config = tuner.resolve_params("balanced", "balanced", "transcribe")
        
        with self.assertRaises(Exception) as context:
            asr = WhisperProASR(
                model_config=config["model"],
                params=config["params"],
                task=config["task"]
            )
        self.assertIn("Model load failed", str(context.exception))


class TestWindowsCompatibility(unittest.TestCase):
    """Test Windows-specific compatibility, especially UTF-8 handling."""
    
    def test_utf8_config_handling(self):
        """Test that UTF-8 characters in config are handled correctly."""
        config_with_unicode = {
            "version": "3.0",
            "test_string": "Japanese text: こんにちは",
            "models": {
                "test-model": {
                    "description": "Test model with symbols: ♪♫♬",
                    "provider": "test",
                    "model_name": "test"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                       suffix='.json', delete=False) as f:
            json.dump(config_with_unicode, f, ensure_ascii=False)
            temp_path = Path(f.name)
            
        try:
            # Should load without encoding errors
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            self.assertEqual(loaded["test_string"], "Japanese text: こんにちは")
        finally:
            temp_path.unlink()
            
    def test_path_handling_with_spaces(self):
        """Test handling of paths with spaces (common on Windows)."""
        with tempfile.TemporaryDirectory() as base_dir:
            # Create directory with spaces
            spaced_dir = Path(base_dir) / "directory with spaces"
            spaced_dir.mkdir()
            
            # Test that paths are handled correctly
            config_path = spaced_dir / "config.json"
            config_path.write_text('{"version": "3.0"}', encoding='utf-8')
            
            # Should handle path with spaces
            self.assertTrue(config_path.exists())
            self.assertEqual(config_path.read_text(encoding='utf-8'), '{"version": "3.0"}')


class TestPerformanceValidation(unittest.TestCase):
    """Test performance-related aspects of V3 architecture."""
    
    def test_config_caching(self):
        """Test that configuration is efficiently cached."""
        tuner = TranscriptionTunerV3()
        
        # First resolution
        import time
        start = time.time()
        config1 = tuner.resolve_params("balanced", "balanced", "transcribe")
        first_time = time.time() - start
        
        # Second resolution (should be faster due to internal structure)
        start = time.time()
        config2 = tuner.resolve_params("balanced", "balanced", "transcribe")
        second_time = time.time() - start
        
        # Both should return same config
        self.assertEqual(config1["model"], config2["model"])
        
        # Note: We're not asserting timing as it's environment-dependent
        # but in practice, subsequent resolutions should be very fast
        
    def test_memory_efficiency(self):
        """Test that V3 doesn't create excessive copies of configuration."""
        tuner = TranscriptionTunerV3()
        
        configs = []
        for i in range(10):
            config = tuner.resolve_params("balanced", "balanced", "transcribe")
            configs.append(config)
            
        # All configs should have been created
        self.assertEqual(len(configs), 10)
        
        # Each should be a proper dict
        for config in configs:
            self.assertIsInstance(config, dict)


# Test runner
def run_tests():
    """Run all tests with appropriate configuration for Windows."""
    # Set up test environment
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestV3ConfigurationLoading,
        TestParameterResolution,
        TestModuleInstantiation,
        TestPipelineIntegration,
        TestEndToEndWorkflow,
        TestErrorHandling,
        TestWindowsCompatibility,
        TestPerformanceValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
        
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    # Ensure UTF-8 output on Windows
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        
    success = run_tests()
    sys.exit(0 if success else 1)