#!/usr/bin/env python3
"""
Real transcription test for QwenASR module.

This script tests the QwenASR module with actual audio files,
bypassing the main.py import chain that has torchaudio dependency issues.

NOTE: QwenASR.transcribe() returns stable_whisper.WhisperResult, not List[Dict].
The result has .segments containing segment objects with .text, .start, .end attributes.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_qwen_real_transcription():
    """Test QwenASR with a real audio file."""
    import stable_whisper
    from whisperjav.modules.qwen_asr import QwenASR
    from whisperjav.utils.logger import setup_logger

    # Setup logging
    logger = setup_logger("test", "INFO")

    # Test audio file
    test_media_dir = Path(__file__).parent.parent / "test_media"
    audio_file = test_media_dir / "MIAA-432.5sec.wav"

    if not audio_file.exists():
        print(f"Test file not found: {audio_file}")
        return False

    print(f"\n{'='*60}")
    print("QwenASR Real Transcription Test")
    print(f"{'='*60}")
    print(f"Audio file: {audio_file}")
    print(f"File size: {audio_file.stat().st_size / 1024:.1f} KB")

    # Initialize QwenASR
    print("\nInitializing QwenASR...")
    asr = QwenASR(
        model_id="Qwen/Qwen3-ASR-1.7B",
        device="auto",
        dtype="auto",
        timestamps="word",  # Use ForcedAligner
    )

    print(f"  Model: {asr.model_id}")
    print(f"  Device: {asr.device_request}")
    print(f"  Timestamps: {asr.timestamps}")
    print(f"  Use Aligner: {asr.use_aligner}")

    # Transcribe
    print("\nTranscribing...")
    try:
        result = asr.transcribe(audio_file)

        # Verify return type is WhisperResult
        assert isinstance(result, stable_whisper.WhisperResult), \
            f"Expected WhisperResult, got {type(result).__name__}"
        print(f"\n[OK] Return type: {type(result).__name__}")

        # Access segments from WhisperResult
        segments = result.segments
        print(f"Results: {len(segments)} segments")
        print("-" * 40)

        for i, seg in enumerate(segments, 1):
            text = seg.text.strip() if hasattr(seg, 'text') else ''
            start = float(seg.start) if hasattr(seg, 'start') else 0.0
            end = float(seg.end) if hasattr(seg, 'end') else 0.0
            print(f"{i}. [{start:.2f}s - {end:.2f}s] {text}")

        print("-" * 40)

        # Verify output format
        if segments:
            assert all(hasattr(s, 'text') for s in segments), "Missing 'text' attribute"
            assert all(hasattr(s, 'start') for s in segments), "Missing 'start' attribute"
            assert all(hasattr(s, 'end') for s in segments), "Missing 'end' attribute"
            print("\n[OK] Output format verification passed")

        # Verify we got sentence-level segments, not word-level
        # (Word-level would have many short segments, typically < 0.5s each)
        if len(segments) > 0:
            avg_duration = sum(float(s.end) - float(s.start) for s in segments) / len(segments)
            print(f"[INFO] Average segment duration: {avg_duration:.2f}s")
            if avg_duration < 0.3 and len(segments) > 10:
                print("[WARN] Segments appear to be word-level, not sentence-level!")
            else:
                print("[OK] Segments appear to be sentence-level (regrouped)")

        return True

    except Exception as e:
        print(f"\nError during transcription: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print("\nCleaning up...")
        asr.cleanup()
        print("[OK] Cleanup complete")


def test_qwen_longer_audio():
    """Test QwenASR with a longer audio file (15 seconds)."""
    import stable_whisper
    from whisperjav.modules.qwen_asr import QwenASR

    test_media_dir = Path(__file__).parent.parent / "test_media"
    audio_file = test_media_dir / "short_15_sec_test-966-00_01_45-00_01_59.wav"

    if not audio_file.exists():
        print(f"Test file not found: {audio_file}")
        return False

    print(f"\n{'='*60}")
    print("QwenASR 15-Second Audio Test")
    print(f"{'='*60}")
    print(f"Audio file: {audio_file.name}")

    asr = QwenASR(
        model_id="Qwen/Qwen3-ASR-1.7B",
        device="auto",
        timestamps="word",
    )

    try:
        result = asr.transcribe(audio_file)

        # Verify return type
        assert isinstance(result, stable_whisper.WhisperResult), \
            f"Expected WhisperResult, got {type(result).__name__}"

        segments = result.segments
        print(f"\nResults: {len(segments)} segments")

        # Show first 10 segments
        for i, seg in enumerate(segments[:10], 1):
            text = seg.text.strip() if hasattr(seg, 'text') else ''
            start = float(seg.start) if hasattr(seg, 'start') else 0.0
            end = float(seg.end) if hasattr(seg, 'end') else 0.0
            print(f"{i}. [{start:.2f}s - {end:.2f}s] {text}")

        if len(segments) > 10:
            print(f"... and {len(segments) - 10} more segments")

        # Verify sentence-level output
        if len(segments) > 0:
            avg_duration = sum(float(s.end) - float(s.start) for s in segments) / len(segments)
            print(f"\n[INFO] Average segment duration: {avg_duration:.2f}s")

        return True

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        asr.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test QwenASR with real audio")
    parser.add_argument("--long", action="store_true", help="Also test 15-second audio")
    args = parser.parse_args()

    # Run tests
    success = test_qwen_real_transcription()

    if args.long:
        success = test_qwen_longer_audio() and success

    print(f"\n{'='*60}")
    if success:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print(f"{'='*60}")

    sys.exit(0 if success else 1)
