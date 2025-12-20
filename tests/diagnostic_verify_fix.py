#!/usr/bin/env python3
"""
Verify the chunk_length fix prevents the crash.
"""

import os
import sys
import subprocess

# Change to project directory
os.chdir(r"C:\BIN\git\whisperJav_V1_Minami_Edition")
sys.path.insert(0, r"C:\BIN\git\whisperJav_V1_Minami_Edition")

TEST_AUDIO = r"C:\BIN\git\whisperJav_V1_Minami_Edition\test_media\173_acceptance_test\SONE-966-15_sec_test-966-00_01_45-00_01_59.wav"


def test_direct_transcription():
    """Test directly with faster-whisper to verify the fix works."""
    import soundfile as sf
    import numpy as np
    from faster_whisper import WhisperModel

    print("=" * 60)
    print("VERIFICATION TEST: chunk_length=30 fix")
    print("=" * 60)

    # Load audio
    audio, sr = sf.read(TEST_AUDIO, dtype='float32')
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    print(f"Audio: {len(audio)} samples, {len(audio)/sr:.3f}s @ {sr}Hz")

    # Exact crash params with fix applied
    params = {
        "task": "transcribe",
        "language": "ja",
        "beam_size": 3,
        "best_of": 1,
        "patience": 1.6,
        "suppress_tokens": [],
        "suppress_blank": False,
        "without_timestamps": False,
        "temperature": 0.0,
        "compression_ratio_threshold": 3.0,
        "no_speech_threshold": 0.22,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "chunk_length": 30,  # THE FIX: changed from 14 to 30
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 2,
        "multilingual": False,
        "log_progress": False,
        "log_prob_threshold": -2.5,
        "vad_filter": False,
    }

    print(f"chunk_length: {params['chunk_length']} (fix applied)")

    # Load model
    print("Loading model...")
    model = WhisperModel("large-v2", device="cuda", compute_type="int8_float16")
    print("Model loaded")

    # Transcribe
    print("Transcribing...")
    segments_gen, info = model.transcribe(audio, **params)
    print(f"Generator created, language={info.language}")

    # Iterate all segments
    segments = []
    for i, seg in enumerate(segments_gen):
        print(f"  Segment {i}: {seg.start:.2f}s - {seg.end:.2f}s")
        segments.append(seg)

    print(f"\nSUCCESS! {len(segments)} segments transcribed without crash")
    return True


def test_via_whisperjav():
    """Test via WhisperJAV pipeline."""
    print("\n" + "=" * 60)
    print("VERIFICATION TEST: WhisperJAV pipeline")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable, "-m", "whisperjav.main",
            TEST_AUDIO,
            "--mode", "balanced",
            "--sensitivity", "aggressive",
            "--crash-trace",
            "--debug"
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=r"C:\BIN\git\whisperJav_V1_Minami_Edition",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}
    )

    print("STDOUT:")
    for line in result.stdout.split('\n')[-30:]:  # Last 30 lines
        if line.strip():
            print(f"  {line}")

    if result.returncode != 0:
        print(f"\nEXIT CODE: {result.returncode}")
        if result.returncode == -1073741676:
            print(">>> CRASH: STATUS_INTEGER_DIVIDE_BY_ZERO <<<")
        if result.stderr:
            print("STDERR (last 500 chars):")
            print(result.stderr[-500:])
        return False

    print("\nSUCCESS! WhisperJAV completed without crash")
    return True


if __name__ == "__main__":
    # Test 1: Direct transcription
    try:
        test_direct_transcription()
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Via WhisperJAV (optional, takes longer)
    # test_via_whisperjav()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
