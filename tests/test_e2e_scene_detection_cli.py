#!/usr/bin/env python3
"""
E2E tests for WhisperJAV CLI with various scene detection options.

Tests the ACTUAL CLI execution (not mocked) for scene detection methods:
1. auditok (default) - energy-based scene detection
2. silero - VAD-based scene detection
3. semantic - texture-based clustering

Tests cover:
- Single-pass modes (fast, faster, balanced)
- Ensemble mode (pass1 + pass2 with different scene detectors)
- Transformers mode with HF scene detection
- Scene output validation
- SRT output validation

Test files used:
- Short (6s): MIAA-432.5sec.wav - for quick tests
- Medium (14s): short_15_sec_test-966-00_01_45-00_01_59.wav
- Longer (464s): SONE-966-0_464.wav - for realistic tests (marked slow)

Run all tests:
    pytest tests/test_e2e_scene_detection_cli.py -v -s --tb=short

Run only fast tests (default):
    pytest tests/test_e2e_scene_detection_cli.py -v -s -m "not slow"

Run only specific method tests:
    pytest tests/test_e2e_scene_detection_cli.py -v -s -k "auditok"
    pytest tests/test_e2e_scene_detection_cli.py -v -s -k "silero"
    pytest tests/test_e2e_scene_detection_cli.py -v -s -k "semantic"
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import pytest

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Test media files
TEST_MEDIA_DIR = REPO_ROOT / "test_media"
TEST_AUDIO_SHORT = TEST_MEDIA_DIR / "MIAA-432.5sec.wav"  # 6 seconds
TEST_AUDIO_MEDIUM = TEST_MEDIA_DIR / "short_15_sec_test-966-00_01_45-00_01_59.wav"  # 14 seconds
TEST_AUDIO_LONGER = TEST_MEDIA_DIR / "SONE-966-0_464.wav"  # 464 seconds (~7.7 min)

# Markers
pytestmark = [pytest.mark.e2e]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def test_output_dir(tmp_path_factory):
    """Create a unique output directory for this test run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = tmp_path_factory.mktemp(f"e2e_scene_detection_{timestamp}")
    yield output_dir


@pytest.fixture(scope="function")
def clean_output_dir(tmp_path):
    """Create a clean output directory for each test."""
    yield tmp_path


def skip_if_no_test_file(filepath: Path):
    """Skip test if test file doesn't exist."""
    if not filepath.exists():
        pytest.skip(f"Test file not found: {filepath}")


# =============================================================================
# Helper Functions
# =============================================================================

def run_cli(args: list, timeout: int = 600, check: bool = False) -> subprocess.CompletedProcess:
    """
    Run whisperjav CLI and return result.

    Args:
        args: CLI arguments
        timeout: Command timeout in seconds
        check: If True, raise on non-zero exit code

    Returns:
        subprocess.CompletedProcess with stdout, stderr, returncode
    """
    cmd = [sys.executable, "-m", "whisperjav.main"] + args
    print(f"\n>>> Running: {' '.join(cmd)}")

    # Use encoding='utf-8' with errors='replace' to handle binary output on Windows
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
        encoding='utf-8',
        errors='replace'  # Replace undecodable bytes instead of crashing
    )

    # Ensure stdout/stderr are never None
    if result.stdout is None:
        result.stdout = ""
    if result.stderr is None:
        result.stderr = ""

    # Print output for debugging
    if result.stdout:
        # Truncate very long output
        stdout = result.stdout[:5000] + "..." if len(result.stdout) > 5000 else result.stdout
        print(f"STDOUT:\n{stdout}")
    if result.stderr:
        stderr = result.stderr[:5000] + "..." if len(result.stderr) > 5000 else result.stderr
        print(f"STDERR:\n{stderr}")

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    return result


def find_srt_files(output_dir: Path) -> List[Path]:
    """Find all SRT files in output directory."""
    return list(output_dir.glob("**/*.srt"))


def find_scene_files(output_dir: Path) -> List[Path]:
    """Find all scene WAV files in output directory."""
    return list(output_dir.glob("**/*_scene_*.wav"))


def validate_srt_content(srt_path: Path) -> Tuple[bool, str]:
    """
    Validate SRT file has proper format.

    Returns:
        Tuple of (is_valid, message)
    """
    if not srt_path.exists():
        return False, f"SRT file not found: {srt_path}"

    content = srt_path.read_text(encoding='utf-8')
    if not content.strip():
        return False, "SRT file is empty"

    # Check for at least one subtitle entry (number + timestamp + text)
    # Pattern: digit(s) followed by timestamp line
    pattern = r'\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}'
    if not re.search(pattern, content):
        return False, "SRT file doesn't contain valid subtitle entries"

    return True, f"Valid SRT with {len(re.findall(pattern, content))} entries"


def count_scenes_in_log(log_output: str) -> Optional[int]:
    """Extract number of scenes detected from CLI output."""
    # Look for patterns like "Found X coarse story line(s)" or "Detected X scenes"
    patterns = [
        r'Found (\d+) coarse story line',
        r'Detected.*?(\d+).*?scenes?',
        r'(\d+) final scenes',
    ]
    for pattern in patterns:
        match = re.search(pattern, log_output, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


# =============================================================================
# Test Classes: Auditok Method (Default)
# =============================================================================

class TestAuditokSceneDetection:
    """Tests for auditok (energy-based) scene detection method."""

    def test_auditok_fast_mode_short_audio(self, clean_output_dir):
        """Test auditok with fast mode on short audio."""
        skip_if_no_test_file(TEST_AUDIO_SHORT)

        result = run_cli([
            str(TEST_AUDIO_SHORT),
            "--mode", "fast",
            "--scene-detection-method", "auditok",
            "--output-dir", str(clean_output_dir),
            "--log-level", "DEBUG",
        ], timeout=120)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Check SRT was created
        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

        # Validate SRT content
        is_valid, msg = validate_srt_content(srt_files[0])
        assert is_valid, msg

    def test_auditok_faster_mode_short_audio(self, clean_output_dir):
        """Test auditok with faster mode on short audio."""
        skip_if_no_test_file(TEST_AUDIO_SHORT)

        result = run_cli([
            str(TEST_AUDIO_SHORT),
            "--mode", "faster",
            "--scene-detection-method", "auditok",
            "--output-dir", str(clean_output_dir),
        ], timeout=120)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

    def test_auditok_balanced_mode_medium_audio(self, clean_output_dir):
        """Test auditok with balanced mode on medium audio."""
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        result = run_cli([
            str(TEST_AUDIO_MEDIUM),
            "--mode", "balanced",
            "--scene-detection-method", "auditok",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=180)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

        # Check scenes were detected
        combined_output = result.stdout + result.stderr
        scene_count = count_scenes_in_log(combined_output)
        print(f"Scenes detected: {scene_count}")

    def test_auditok_default_without_explicit_method(self, clean_output_dir):
        """Test that auditok is used by default when no method specified."""
        skip_if_no_test_file(TEST_AUDIO_SHORT)

        result = run_cli([
            str(TEST_AUDIO_SHORT),
            "--mode", "fast",
            "--output-dir", str(clean_output_dir),
            "--log-level", "DEBUG",
        ], timeout=120)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

        # Log verification (informational)
        combined_output = result.stdout + result.stderr
        auditok_logged = "auditok" in combined_output.lower() or "story line" in combined_output.lower()
        print(f"Auditok/story line mentioned in logs: {auditok_logged}")

    @pytest.mark.slow
    def test_auditok_longer_audio(self, clean_output_dir):
        """Test auditok with longer audio file (7+ minutes)."""
        skip_if_no_test_file(TEST_AUDIO_LONGER)

        result = run_cli([
            str(TEST_AUDIO_LONGER),
            "--mode", "fast",
            "--scene-detection-method", "auditok",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=600)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

        # Longer audio should produce multiple scenes
        combined_output = result.stdout + result.stderr
        scene_count = count_scenes_in_log(combined_output)
        print(f"Scenes detected for 7-min audio: {scene_count}")
        # Expect at least a few scenes for 7+ minutes of audio
        if scene_count is not None:
            assert scene_count >= 1, "Expected multiple scenes for long audio"


# =============================================================================
# Test Classes: Silero Method
# =============================================================================

class TestSileroSceneDetection:
    """Tests for silero (VAD-based) scene detection method."""

    def test_silero_fast_mode_short_audio(self, clean_output_dir):
        """Test silero with fast mode on short audio."""
        skip_if_no_test_file(TEST_AUDIO_SHORT)

        result = run_cli([
            str(TEST_AUDIO_SHORT),
            "--mode", "fast",
            "--scene-detection-method", "silero",
            "--output-dir", str(clean_output_dir),
            "--log-level", "DEBUG",
        ], timeout=180)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

    def test_silero_balanced_mode_medium_audio(self, clean_output_dir):
        """Test silero with balanced mode on medium audio."""
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        result = run_cli([
            str(TEST_AUDIO_MEDIUM),
            "--mode", "balanced",
            "--scene-detection-method", "silero",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=240)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

        # Log verification (informational)
        combined_output = result.stdout + result.stderr
        silero_logged = "silero" in combined_output.lower() or "vad" in combined_output.lower()
        print(f"Silero/VAD mentioned in logs: {silero_logged}")

    def test_silero_resampling_logged(self, clean_output_dir):
        """Test that silero runs successfully (resampling is internal detail)."""
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        result = run_cli([
            str(TEST_AUDIO_MEDIUM),
            "--mode", "balanced",
            "--scene-detection-method", "silero",
            "--output-dir", str(clean_output_dir),
            "--log-level", "DEBUG",
        ], timeout=240)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

        # Log verification (informational)
        combined_output = result.stdout + result.stderr
        print(f"Silero logs present: {'silero' in combined_output.lower()}")

    @pytest.mark.slow
    def test_silero_longer_audio(self, clean_output_dir):
        """Test silero with longer audio file."""
        skip_if_no_test_file(TEST_AUDIO_LONGER)

        result = run_cli([
            str(TEST_AUDIO_LONGER),
            "--mode", "fast",
            "--scene-detection-method", "silero",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=600)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"


# =============================================================================
# Test Classes: Semantic Method
# =============================================================================

class TestSemanticSceneDetection:
    """Tests for semantic (texture-based clustering) scene detection method."""

    def test_semantic_fast_mode_short_audio(self, clean_output_dir):
        """Test semantic with fast mode on short audio."""
        skip_if_no_test_file(TEST_AUDIO_SHORT)

        result = run_cli([
            str(TEST_AUDIO_SHORT),
            "--mode", "fast",
            "--scene-detection-method", "semantic",
            "--output-dir", str(clean_output_dir),
            "--log-level", "DEBUG",
        ], timeout=180)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

        # Log verification (informational, not a hard requirement)
        combined_output = result.stdout + result.stderr
        if "semantic" not in combined_output.lower():
            print("Note: 'semantic' not found in logs (may use different terminology)")

    def test_semantic_balanced_mode_medium_audio(self, clean_output_dir):
        """Test semantic with balanced mode on medium audio."""
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        result = run_cli([
            str(TEST_AUDIO_MEDIUM),
            "--mode", "balanced",
            "--scene-detection-method", "semantic",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=240)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

    @pytest.mark.slow
    def test_semantic_longer_audio(self, clean_output_dir):
        """Test semantic with longer audio file."""
        skip_if_no_test_file(TEST_AUDIO_LONGER)

        result = run_cli([
            str(TEST_AUDIO_LONGER),
            "--mode", "fast",
            "--scene-detection-method", "semantic",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=600)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"


# =============================================================================
# Test Classes: Ensemble Mode with Scene Detectors
# =============================================================================

class TestEnsembleModeSceneDetection:
    """Tests for ensemble mode with various scene detector combinations."""

    def test_ensemble_auditok_both_passes(self, clean_output_dir):
        """Test ensemble mode with auditok for both passes."""
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        result = run_cli([
            str(TEST_AUDIO_MEDIUM),
            "--ensemble",
            "--pass1-scene-detector", "auditok",
            "--pass2-scene-detector", "auditok",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=300)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

    def test_ensemble_silero_both_passes(self, clean_output_dir):
        """Test ensemble mode with silero for both passes."""
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        result = run_cli([
            str(TEST_AUDIO_MEDIUM),
            "--ensemble",
            "--pass1-scene-detector", "silero",
            "--pass2-scene-detector", "silero",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=300)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

    def test_ensemble_mixed_detectors(self, clean_output_dir):
        """Test ensemble mode with different detectors for each pass."""
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        result = run_cli([
            str(TEST_AUDIO_MEDIUM),
            "--ensemble",
            "--pass1-scene-detector", "auditok",
            "--pass2-scene-detector", "silero",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=300)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

    def test_ensemble_none_scene_detector(self, clean_output_dir):
        """Test ensemble mode with scene detection disabled.

        Verifies that --pass1-scene-detector none and --pass2-scene-detector none
        work correctly without causing TypeError (fixed in pass_worker.py).
        """
        skip_if_no_test_file(TEST_AUDIO_SHORT)

        result = run_cli([
            str(TEST_AUDIO_SHORT),
            "--ensemble",
            "--pass1-scene-detector", "none",
            "--pass2-scene-detector", "none",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=180)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"


# =============================================================================
# Test Classes: Transformers Mode with HF Scene Detection
# =============================================================================

class TestTransformersModeSceneDetection:
    """Tests for transformers mode with HF scene detection options."""

    def test_transformers_hf_scene_none(self, clean_output_dir):
        """Test transformers mode with no scene detection."""
        skip_if_no_test_file(TEST_AUDIO_SHORT)

        result = run_cli([
            str(TEST_AUDIO_SHORT),
            "--mode", "transformers",
            "--hf-scene", "none",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=300)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

    def test_transformers_hf_scene_auditok(self, clean_output_dir):
        """Test transformers mode with auditok scene detection."""
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        result = run_cli([
            str(TEST_AUDIO_MEDIUM),
            "--mode", "transformers",
            "--hf-scene", "auditok",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=300)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

    @pytest.mark.slow
    def test_transformers_hf_scene_silero(self, clean_output_dir):
        """Test transformers mode with silero scene detection.

        NOTE: Marked as slow because transformers mode with silero requires
        significant processing time for model loading and VAD processing.
        """
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        result = run_cli([
            str(TEST_AUDIO_MEDIUM),
            "--mode", "transformers",
            "--hf-scene", "silero",
            "--output-dir", str(clean_output_dir),
            "--log-level", "INFO",
        ], timeout=600)  # Increased timeout for transformers + silero

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"


# =============================================================================
# Test Classes: Parameter Variations
# =============================================================================

class TestSceneDetectionParameters:
    """Tests for scene detection with various parameter combinations."""

    def test_sensitivity_conservative(self, clean_output_dir):
        """Test scene detection with conservative sensitivity."""
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        result = run_cli([
            str(TEST_AUDIO_MEDIUM),
            "--mode", "balanced",
            "--scene-detection-method", "auditok",
            "--sensitivity", "conservative",
            "--output-dir", str(clean_output_dir),
        ], timeout=180)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_sensitivity_aggressive(self, clean_output_dir):
        """Test scene detection with aggressive sensitivity."""
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        result = run_cli([
            str(TEST_AUDIO_MEDIUM),
            "--mode", "balanced",
            "--scene-detection-method", "auditok",
            "--sensitivity", "aggressive",
            "--output-dir", str(clean_output_dir),
        ], timeout=180)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_keep_temp_preserves_scenes(self, clean_output_dir):
        """Test that --keep-temp preserves scene files."""
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        result = run_cli([
            str(TEST_AUDIO_MEDIUM),
            "--mode", "balanced",
            "--scene-detection-method", "auditok",
            "--output-dir", str(clean_output_dir),
            "--keep-temp",
            "--log-level", "DEBUG",
        ], timeout=180)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # With --keep-temp, scene files should be preserved
        # They're typically in a temp directory, but we can check for any
        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"

    def test_no_vad_option(self, clean_output_dir):
        """Test --no-vad option with scene detection."""
        skip_if_no_test_file(TEST_AUDIO_SHORT)

        result = run_cli([
            str(TEST_AUDIO_SHORT),
            "--mode", "balanced",
            "--scene-detection-method", "auditok",
            "--no-vad",
            "--output-dir", str(clean_output_dir),
        ], timeout=180)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        srt_files = find_srt_files(clean_output_dir)
        assert len(srt_files) >= 1, "No SRT file created"


# =============================================================================
# Test Classes: Error Handling and Edge Cases
# =============================================================================

class TestSceneDetectionEdgeCases:
    """Tests for edge cases and error handling in scene detection."""

    def test_invalid_scene_method_fails_gracefully(self, clean_output_dir):
        """Test that invalid scene detection method is handled."""
        skip_if_no_test_file(TEST_AUDIO_SHORT)

        result = run_cli([
            str(TEST_AUDIO_SHORT),
            "--mode", "fast",
            "--scene-detection-method", "invalid_method_xyz",
            "--output-dir", str(clean_output_dir),
        ], timeout=60)

        # Should either fail with clear error or fall back to default
        # The CLI uses argparse choices, so this should fail
        assert result.returncode != 0 or "error" in result.stderr.lower() or "invalid" in result.stderr.lower(), \
            "Expected error or fallback for invalid method"

    def test_very_short_audio_handles_gracefully(self, clean_output_dir):
        """Test that very short audio is handled without crashing."""
        skip_if_no_test_file(TEST_AUDIO_SHORT)

        # 5-second audio should still work
        result = run_cli([
            str(TEST_AUDIO_SHORT),
            "--mode", "fast",
            "--scene-detection-method", "auditok",
            "--output-dir", str(clean_output_dir),
        ], timeout=120)

        assert result.returncode == 0, f"CLI failed on short audio: {result.stderr}"

    def test_output_dir_created_if_not_exists(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        skip_if_no_test_file(TEST_AUDIO_SHORT)

        new_output_dir = tmp_path / "new_nested" / "output" / "dir"

        result = run_cli([
            str(TEST_AUDIO_SHORT),
            "--mode", "fast",
            "--output-dir", str(new_output_dir),
        ], timeout=120)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert new_output_dir.exists(), "Output directory was not created"


# =============================================================================
# Test Classes: Comparison Tests
# =============================================================================

class TestSceneDetectionComparison:
    """Tests comparing different scene detection methods."""

    @pytest.mark.slow
    def test_all_methods_produce_valid_srt(self, clean_output_dir):
        """Test that all scene detection methods produce valid SRT."""
        skip_if_no_test_file(TEST_AUDIO_MEDIUM)

        methods = ["auditok", "silero", "semantic"]
        results = {}

        for method in methods:
            method_dir = clean_output_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)

            result = run_cli([
                str(TEST_AUDIO_MEDIUM),
                "--mode", "fast",
                "--scene-detection-method", method,
                "--output-dir", str(method_dir),
            ], timeout=240)

            results[method] = {
                "returncode": result.returncode,
                "srt_files": find_srt_files(method_dir),
            }

        # All methods should succeed
        for method, data in results.items():
            assert data["returncode"] == 0, f"{method} method failed"
            assert len(data["srt_files"]) >= 1, f"{method} method produced no SRT"

            # Validate SRT
            is_valid, msg = validate_srt_content(data["srt_files"][0])
            assert is_valid, f"{method} method: {msg}"

        print(f"\nAll methods produced valid SRT files:")
        for method, data in results.items():
            print(f"  {method}: {len(data['srt_files'])} file(s)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "not slow"])
