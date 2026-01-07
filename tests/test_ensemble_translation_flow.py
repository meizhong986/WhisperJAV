#!/usr/bin/env python3
"""
E2E tests for WhisperJAV GUI Ensemble Mode + Translation Flow.

Tests the complete flow from transcription to translation triggering:
1. _compute_expected_outputs() returns correct filenames for ensemble mode
2. start_ensemble_twopass() stores expected output files
3. get_process_status() returns output_files when completed
4. Translation API receives correct file paths

Run with: pytest tests/test_ensemble_translation_flow.py -v
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_api(tmp_path, monkeypatch):
    """
    Create a WhisperJAVAPI instance with mocked dependencies.
    """
    # Patch webview to avoid GUI initialization
    with patch('webview.create_window'), \
         patch('webview.windows', []):
        from whisperjav.webview_gui.api import WhisperJAVAPI

        api = WhisperJAVAPI()
        api.default_output = str(tmp_path / "output")
        Path(api.default_output).mkdir(parents=True, exist_ok=True)

        return api


# =============================================================================
# Tests for _compute_expected_outputs
# =============================================================================

class TestComputeExpectedOutputs:
    """Tests for the _compute_expected_outputs helper method."""

    def test_ensemble_mode_explicit_flag(self, mock_api, tmp_path):
        """Test ensemble mode with is_ensemble=True flag."""
        output_dir = str(tmp_path / "output")
        config = {
            'inputs': ['/path/to/video1.mp4', '/path/to/video2.wav'],
            'output_dir': output_dir,
            'source_language': 'japanese'
        }

        result = mock_api._compute_expected_outputs(config, is_ensemble=True)

        assert len(result) == 2
        assert result[0] == str(Path(output_dir) / "video1.ja.merged.whisperjav.srt")
        assert result[1] == str(Path(output_dir) / "video2.ja.merged.whisperjav.srt")

    def test_ensemble_mode_via_mode_key(self, mock_api, tmp_path):
        """Test ensemble mode detected via mode='ensemble' key."""
        output_dir = str(tmp_path / "output")
        options = {
            'inputs': ['/path/to/test.mp4'],
            'output_dir': output_dir,
            'source_language': 'japanese',
            'mode': 'ensemble'
        }

        result = mock_api._compute_expected_outputs(options)

        assert len(result) == 1
        assert result[0] == str(Path(output_dir) / "test.ja.merged.whisperjav.srt")

    def test_non_ensemble_mode(self, mock_api, tmp_path):
        """Test non-ensemble mode produces simple .srt filename."""
        output_dir = str(tmp_path / "output")
        options = {
            'inputs': ['/path/to/video.mp4'],
            'output_dir': output_dir,
            'source_language': 'japanese',
            'mode': 'balanced'
        }

        result = mock_api._compute_expected_outputs(options)

        assert len(result) == 1
        assert result[0] == str(Path(output_dir) / "video.srt")

    def test_english_source_language(self, mock_api, tmp_path):
        """Test with English source language."""
        output_dir = str(tmp_path / "output")
        config = {
            'inputs': ['/path/to/audio.wav'],
            'output_dir': output_dir,
            'source_language': 'english'
        }

        result = mock_api._compute_expected_outputs(config, is_ensemble=True)

        assert len(result) == 1
        assert "audio.en.merged.whisperjav.srt" in result[0]

    def test_default_output_dir(self, mock_api):
        """Test using default output directory when not specified."""
        config = {
            'inputs': ['/path/to/video.mp4'],
            'source_language': 'japanese'
        }

        result = mock_api._compute_expected_outputs(config, is_ensemble=True)

        assert len(result) == 1
        assert mock_api.default_output in result[0]
        assert "video.ja.merged.whisperjav.srt" in result[0]

    def test_windows_paths(self, mock_api, tmp_path):
        """Test with Windows-style paths."""
        output_dir = str(tmp_path / "output")
        config = {
            'inputs': ['C:\\Users\\Test\\Videos\\movie.mp4'],
            'output_dir': output_dir,
            'source_language': 'japanese'
        }

        result = mock_api._compute_expected_outputs(config, is_ensemble=True)

        assert len(result) == 1
        assert "movie.ja.merged.whisperjav.srt" in result[0]

    def test_complex_filename(self, mock_api, tmp_path):
        """Test with complex filename containing dots and special chars."""
        output_dir = str(tmp_path / "output")
        config = {
            'inputs': ['/path/to/MIAA-432.20sec_piano.wav'],
            'output_dir': output_dir,
            'source_language': 'japanese'
        }

        result = mock_api._compute_expected_outputs(config, is_ensemble=True)

        assert len(result) == 1
        # Should preserve the full stem
        assert "MIAA-432.20sec_piano.ja.merged.whisperjav.srt" in result[0]


# =============================================================================
# Tests for start_ensemble_twopass output file tracking
# =============================================================================

class TestEnsembleTwopassOutputTracking:
    """Tests that start_ensemble_twopass correctly stores expected output files."""

    def test_output_files_stored_on_start(self, mock_api, tmp_path):
        """Test that _expected_output_files is populated when process starts."""
        output_dir = str(tmp_path / "output")

        config = {
            'inputs': ['/path/to/video.mp4'],
            'output_dir': output_dir,
            'source_language': 'japanese',
            'pass1': {'pipeline': 'balanced', 'sensitivity': 'balanced'},
            'pass2': {'enabled': False}
        }

        # Mock subprocess.Popen to avoid actually running the process
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            result = mock_api.start_ensemble_twopass(config)

        assert result['success'] == True
        assert len(mock_api._expected_output_files) == 1
        assert "video.ja.merged.whisperjav.srt" in mock_api._expected_output_files[0]

    def test_multiple_files_tracked(self, mock_api, tmp_path):
        """Test multiple input files are all tracked."""
        output_dir = str(tmp_path / "output")

        config = {
            'inputs': [
                '/path/to/video1.mp4',
                '/path/to/video2.wav',
                '/path/to/video3.mkv'
            ],
            'output_dir': output_dir,
            'source_language': 'japanese',
            'pass1': {'pipeline': 'balanced', 'sensitivity': 'balanced'},
            'pass2': {'enabled': False}
        }

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            mock_api.start_ensemble_twopass(config)

        assert len(mock_api._expected_output_files) == 3
        expected_names = [
            "video1.ja.merged.whisperjav.srt",
            "video2.ja.merged.whisperjav.srt",
            "video3.ja.merged.whisperjav.srt"
        ]
        for name in expected_names:
            assert any(name in f for f in mock_api._expected_output_files)


# =============================================================================
# Tests for get_process_status returning output_files
# =============================================================================

class TestProcessStatusOutputFiles:
    """Tests that get_process_status returns output_files when completed."""

    def test_output_files_returned_on_completion(self, mock_api, tmp_path):
        """Test output_files is included in status when completed."""
        output_dir = str(tmp_path / "output")

        # Simulate starting a process
        config = {
            'inputs': ['/path/to/video.mp4'],
            'output_dir': output_dir,
            'source_language': 'japanese',
            'pass1': {'pipeline': 'balanced', 'sensitivity': 'balanced'},
            'pass2': {'enabled': False}
        }

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_process.returncode = None
            mock_popen.return_value = mock_process

            mock_api.start_ensemble_twopass(config)

            # Simulate process completing successfully
            mock_process.poll.return_value = 0
            mock_process.returncode = 0

            status = mock_api.get_process_status()

        assert status['status'] == 'completed'
        assert 'output_files' in status
        assert len(status['output_files']) == 1
        assert "video.ja.merged.whisperjav.srt" in status['output_files'][0]

    def test_output_files_not_returned_when_running(self, mock_api, tmp_path):
        """Test output_files is not returned while process is running."""
        output_dir = str(tmp_path / "output")

        config = {
            'inputs': ['/path/to/video.mp4'],
            'output_dir': output_dir,
            'source_language': 'japanese',
            'pass1': {'pipeline': 'balanced', 'sensitivity': 'balanced'},
            'pass2': {'enabled': False}
        }

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Still running
            mock_popen.return_value = mock_process

            mock_api.start_ensemble_twopass(config)
            status = mock_api.get_process_status()

        assert status['status'] == 'running'
        assert 'output_files' not in status

    def test_output_files_not_returned_on_error(self, mock_api, tmp_path):
        """Test output_files is not returned when process errors."""
        output_dir = str(tmp_path / "output")

        config = {
            'inputs': ['/path/to/video.mp4'],
            'output_dir': output_dir,
            'source_language': 'japanese',
            'pass1': {'pipeline': 'balanced', 'sensitivity': 'balanced'},
            'pass2': {'enabled': False}
        }

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = 1  # Error exit code
            mock_process.returncode = 1
            mock_popen.return_value = mock_process

            mock_api.start_ensemble_twopass(config)
            status = mock_api.get_process_status()

        assert status['status'] == 'error'
        assert 'output_files' not in status


# =============================================================================
# Tests for Translation API contract
# =============================================================================

class TestTranslationApiContract:
    """Tests that translation API receives correct parameters."""

    def test_start_translation_accepts_inputs_key(self, mock_api):
        """Test that start_translation accepts 'inputs' key (not 'input_files')."""
        options = {
            'inputs': ['/path/to/video.ja.merged.whisperjav.srt'],
            'provider': 'deepseek',
            'target': 'english'
        }

        # This should not raise KeyError or return "No input files specified"
        # We mock the subprocess to avoid actual translation
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            result = mock_api.start_translation(options)

        assert result.get('error') != "No input files specified"

    def test_start_translation_rejects_input_files_key(self, mock_api):
        """Test that start_translation fails with old 'input_files' key."""
        options = {
            'input_files': ['/path/to/video.srt'],  # Wrong key!
            'provider': 'deepseek',
            'target': 'english'
        }

        result = mock_api.start_translation(options)

        # Should fail because 'inputs' is empty
        assert result['success'] == False
        assert "No input files specified" in result.get('error', '')

    def test_start_translation_accepts_target_key(self, mock_api):
        """Test that start_translation accepts 'target' key (not 'target_language')."""
        options = {
            'inputs': ['/path/to/video.srt'],
            'provider': 'deepseek',
            'target': 'english'  # Correct key
        }

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            result = mock_api.start_translation(options)

        # Should succeed (or at least not fail on target parsing)
        assert result['success'] == True or 'target' not in result.get('error', '')


# =============================================================================
# Integration test: Full ensemble + translation flow
# =============================================================================

class TestFullEnsembleTranslationFlow:
    """Integration test for the complete flow."""

    def test_full_flow_simulation(self, mock_api, tmp_path):
        """
        Simulate the full flow:
        1. Start ensemble transcription
        2. Process completes successfully
        3. get_process_status returns output_files
        4. start_translation receives correct files
        """
        output_dir = str(tmp_path / "output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Step 1: Start ensemble transcription
        ensemble_config = {
            'inputs': [
                str(tmp_path / 'MIAA-432.20sec_piano.wav'),
                str(tmp_path / 'SONE-966-15_sec_test.wav')
            ],
            'output_dir': output_dir,
            'source_language': 'japanese',
            'pass1': {'pipeline': 'balanced', 'sensitivity': 'aggressive'},
            'pass2': {'enabled': True, 'pipeline': 'balanced', 'sensitivity': 'conservative'}
        }

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            # Start transcription
            result = mock_api.start_ensemble_twopass(ensemble_config)
            assert result['success'] == True

            # Verify output files are tracked
            assert len(mock_api._expected_output_files) == 2

            # Step 2: Simulate process completion
            mock_process.poll.return_value = 0
            mock_process.returncode = 0

            # Step 3: Get status with output_files
            status = mock_api.get_process_status()
            assert status['status'] == 'completed'
            assert 'output_files' in status
            output_files = status['output_files']
            assert len(output_files) == 2

            # Verify filename patterns
            assert any('MIAA-432.20sec_piano.ja.merged.whisperjav.srt' in f for f in output_files)
            assert any('SONE-966-15_sec_test.ja.merged.whisperjav.srt' in f for f in output_files)

        # Step 4: Start translation with the output files
        # Reset mock for translation subprocess
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            translation_options = {
                'inputs': output_files,  # Correct key
                'provider': 'deepseek',
                'target': 'english',     # Correct key
                'model': 'deepseek-chat'
            }

            # Initialize translation state
            mock_api._init_translation_state()

            result = mock_api.start_translation(translation_options)

            # Should succeed
            assert result['success'] == True, f"Translation failed: {result.get('error')}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
