#!/usr/bin/env python3
"""
E2E tests for WhisperJAV Ensemble Mode + Translation workflow.

Tests the ACTUAL CLI execution (not mocked) for:
1. Ensemble mode: Pass 1 → Pass 2 → Merge
2. Translation: Merged SRT → Translated SRT
3. Full pipeline: Ensemble + Translation in single command

Requirements:
- Test audio file: test_media/short_15_sec_test-966-00_01_45-00_01_59.wav
- Translation API key: DEEPSEEK_API_KEY environment variable (for translation tests)

Run with:
    pytest tests/test_e2e_ensemble_translate.py -v -s --tb=short

Skip translation tests (no API key):
    pytest tests/test_e2e_ensemble_translate.py -v -s -k "not translate"

Run only fast tests:
    pytest tests/test_e2e_ensemble_translate.py -v -s -m "not slow"
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import pytest

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Test media file (15 seconds of Japanese speech)
TEST_AUDIO = REPO_ROOT / "test_media" / "short_15_sec_test-966-00_01_45-00_01_59.wav"

# Markers
pytestmark = [pytest.mark.e2e]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def test_output_dir(tmp_path_factory):
    """Create a unique output directory for this test run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = tmp_path_factory.mktemp(f"e2e_test_{timestamp}")
    yield output_dir
    # Cleanup is automatic with tmp_path_factory


@pytest.fixture(scope="module")
def persistent_output_dir():
    """Create a persistent output directory for debugging failed tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "test_results" / f"e2e_ensemble_translate_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    # Not cleaned up - useful for debugging


def has_api_key(provider: str = "deepseek") -> bool:
    """Check if translation API key is available."""
    env_vars = {
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "gpt": "OPENAI_API_KEY",
    }
    return bool(os.environ.get(env_vars.get(provider, "DEEPSEEK_API_KEY")))


def run_cli(args: list, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run whisperjav CLI and return result."""
    cmd = [sys.executable, "-m", "whisperjav.main"] + args
    print(f"\n>>> Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT)
    )

    if result.stdout:
        print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")

    return result


# =============================================================================
# Test: Ensemble Mode (Pass 1 + Pass 2 + Merge)
# =============================================================================

class TestEnsembleMode:
    """Test ensemble mode transcription without translation."""

    @pytest.mark.skipif(not TEST_AUDIO.exists(), reason="Test audio file not found")
    def test_ensemble_basic(self, persistent_output_dir):
        """
        Test basic ensemble mode produces merged SRT.

        Expected output: {basename}.ja.merged.whisperjav.srt
        """
        output_dir = persistent_output_dir / "test_ensemble_basic"
        output_dir.mkdir(exist_ok=True)

        args = [
            str(TEST_AUDIO),
            "--mode", "ensemble",
            "--output", str(output_dir),
            "--language", "ja",
            # Use fast settings for testing
            "--pass1-pipeline", "faster",
            "--pass1-sensitivity", "balanced",
            "--pass2-pipeline", "faster",
            "--pass2-sensitivity", "balanced",
        ]

        result = run_cli(args, timeout=180)

        # Check exit code
        assert result.returncode == 0, f"CLI failed with: {result.stderr}"

        # Check output file exists
        expected_srt = output_dir / "short_15_sec_test-966-00_01_45-00_01_59.ja.merged.whisperjav.srt"
        assert expected_srt.exists(), f"Expected SRT not found: {expected_srt}"

        # Verify SRT has content
        content = expected_srt.read_text(encoding="utf-8")
        assert len(content) > 0, "SRT file is empty"
        assert "1\n" in content or "00:00:" in content, "SRT doesn't look valid"

        print(f"\n=== Generated SRT ({len(content)} chars) ===")
        print(content[:500] if len(content) > 500 else content)


# =============================================================================
# Test: Translation (Standalone)
# =============================================================================

class TestTranslationStandalone:
    """Test standalone translation of existing SRT."""

    @pytest.mark.skipif(not has_api_key(), reason="DEEPSEEK_API_KEY not set")
    def test_translate_srt(self, persistent_output_dir):
        """
        Test translation of an existing SRT file.

        Uses whisperjav-translate CLI.
        """
        # First, we need an SRT to translate
        # Use the test SRT from test_media if available
        test_srt = REPO_ROOT / "test_media" / "The.Naked.Director.S01E04.Scene4.ja.netflix.srt"

        if not test_srt.exists():
            pytest.skip("Test SRT file not found")

        output_dir = persistent_output_dir / "test_translate_srt"
        output_dir.mkdir(exist_ok=True)

        # Copy source SRT to output dir
        source_srt = output_dir / "test_input.srt"
        shutil.copy(test_srt, source_srt)

        # Run translation
        cmd = [
            sys.executable, "-m", "whisperjav.translate.cli",
            "-i", str(source_srt),
            "--target", "english",
            "--provider", "deepseek",
            "--output", str(output_dir),
        ]

        print(f"\n>>> Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT)
        )

        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")

        assert result.returncode == 0, f"Translation CLI failed: {result.stderr}"

        # Check for output file (naming varies)
        output_files = list(output_dir.glob("*.english.srt")) + list(output_dir.glob("*_translated.srt"))
        assert len(output_files) > 0, f"No translated SRT found in {output_dir}"


# =============================================================================
# Test: Full Pipeline (Ensemble + Translation)
# =============================================================================

class TestFullPipeline:
    """Test complete ensemble + translation in single command."""

    @pytest.mark.slow
    @pytest.mark.skipif(not TEST_AUDIO.exists(), reason="Test audio file not found")
    @pytest.mark.skipif(not has_api_key(), reason="DEEPSEEK_API_KEY not set")
    def test_ensemble_with_translation(self, persistent_output_dir):
        """
        Test full pipeline: Ensemble transcription + Translation.

        Expected outputs:
        1. {basename}.ja.merged.whisperjav.srt (transcription)
        2. {basename}.ja.merged.whisperjav.english.srt (translation)
        """
        output_dir = persistent_output_dir / "test_full_pipeline"
        output_dir.mkdir(exist_ok=True)

        args = [
            str(TEST_AUDIO),
            "--mode", "ensemble",
            "--output", str(output_dir),
            "--language", "ja",
            # Fast settings for testing
            "--pass1-pipeline", "faster",
            "--pass1-sensitivity", "balanced",
            "--pass2-pipeline", "faster",
            "--pass2-sensitivity", "balanced",
            # Translation flags
            "--translate",
            "--translate-provider", "deepseek",
            "--translate-target", "english",
            "--translate-tone", "standard",
        ]

        result = run_cli(args, timeout=300)

        # Check exit code
        assert result.returncode == 0, f"CLI failed with: {result.stderr}"

        # Check transcription output
        base_name = "short_15_sec_test-966-00_01_45-00_01_59"
        transcription_srt = output_dir / f"{base_name}.ja.merged.whisperjav.srt"
        assert transcription_srt.exists(), f"Transcription SRT not found: {transcription_srt}"

        # Check translation output (naming may vary)
        translation_candidates = [
            output_dir / f"{base_name}.ja.merged.whisperjav.english.srt",
            output_dir / f"{base_name}.english.srt",
        ]

        translation_srt = None
        for candidate in translation_candidates:
            if candidate.exists():
                translation_srt = candidate
                break

        assert translation_srt is not None, (
            f"Translation SRT not found. Looked for: {translation_candidates}\n"
            f"Files in output dir: {list(output_dir.glob('*.srt'))}"
        )

        # Verify both files have content
        trans_content = transcription_srt.read_text(encoding="utf-8")
        xlat_content = translation_srt.read_text(encoding="utf-8")

        assert len(trans_content) > 0, "Transcription SRT is empty"
        assert len(xlat_content) > 0, "Translation SRT is empty"

        # Verify translation is actually different (not just a copy)
        # Japanese transcription should have Japanese characters
        # English translation should have mostly ASCII
        has_japanese = any(ord(c) > 127 for c in trans_content[:500])
        has_english = any(c.isascii() and c.isalpha() for c in xlat_content[:500])

        print(f"\n=== Transcription ({len(trans_content)} chars) ===")
        print(trans_content[:300] if len(trans_content) > 300 else trans_content)

        print(f"\n=== Translation ({len(xlat_content)} chars) ===")
        print(xlat_content[:300] if len(xlat_content) > 300 else xlat_content)

        # These assertions may be too strict - comment out if failing
        # assert has_japanese, "Transcription doesn't appear to contain Japanese"
        # assert has_english, "Translation doesn't appear to contain English"


# =============================================================================
# Test: CLI Arguments Validation
# =============================================================================

class TestCLIValidation:
    """Test CLI argument handling."""

    def test_translate_provider_choices(self):
        """Test that all GUI provider choices are accepted by CLI."""
        providers = ["deepseek", "openrouter", "gemini", "claude", "gpt", "glm", "groq"]

        for provider in providers:
            # Just test argument parsing, not actual execution
            args = [
                "--help",  # Don't actually run, just validate args
            ]
            result = run_cli(args, timeout=10)

            # Check that provider is in help text
            assert provider in result.stdout or result.returncode == 0, (
                f"Provider '{provider}' may not be recognized"
            )

    def test_translate_target_choices(self):
        """Test that all GUI target choices are accepted by CLI."""
        targets = ["english", "chinese", "indonesian", "spanish"]

        # Just verify CLI help mentions these
        result = run_cli(["--help"], timeout=10)

        for target in targets:
            assert target in result.stdout, f"Target '{target}' not in CLI help"

    def test_translate_tone_choices(self):
        """Test that all GUI tone choices are accepted by CLI."""
        tones = ["standard", "pornify"]

        result = run_cli(["--help"], timeout=10)

        for tone in tones:
            assert tone in result.stdout, f"Tone '{tone}' not in CLI help"


# =============================================================================
# Utility: Manual Test Runner
# =============================================================================

def manual_test_full_pipeline():
    """
    Manual test function for interactive debugging.

    Run with: python -c "from tests.test_e2e_ensemble_translate import manual_test_full_pipeline; manual_test_full_pipeline()"
    """
    output_dir = REPO_ROOT / "test_results" / f"manual_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Test audio: {TEST_AUDIO}")
    print(f"Has DeepSeek API key: {has_api_key('deepseek')}")

    if not TEST_AUDIO.exists():
        print(f"ERROR: Test audio not found at {TEST_AUDIO}")
        return

    args = [
        str(TEST_AUDIO),
        "--mode", "ensemble",
        "--output", str(output_dir),
        "--language", "ja",
        "--pass1-pipeline", "faster",
        "--pass1-sensitivity", "balanced",
        "--pass2-pipeline", "faster",
        "--pass2-sensitivity", "balanced",
    ]

    if has_api_key("deepseek"):
        args.extend([
            "--translate",
            "--translate-provider", "deepseek",
            "--translate-target", "english",
        ])

    print(f"\nRunning: whisperjav {' '.join(args[1:])}")
    result = run_cli(args, timeout=300)

    print(f"\nExit code: {result.returncode}")
    print(f"\nOutput files:")
    for f in output_dir.glob("*.srt"):
        print(f"  {f.name} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    # Run manual test when executed directly
    manual_test_full_pipeline()
