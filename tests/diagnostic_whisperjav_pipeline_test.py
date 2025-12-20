"""
Diagnostic Test: WhisperJAV Pipeline vs Direct faster-whisper

Tests the SAME audio file through:
1. Direct faster-whisper (control)
2. WhisperJAV's FasterWhisperProASR
3. Full balanced pipeline simulation

This identifies WHERE in WhisperJAV's processing the crash originates.
"""

import sys
import time
import gc
from pathlib import Path
import numpy as np

TEST_FILE = Path(r"C:\BIN\git\whisperJav_V1_Minami_Edition\test_media\173_acceptance_test\SONE-966-15_sec_test-966-00_01_45-00_01_59.wav")

def test_1_direct_faster_whisper():
    """Control test - direct faster-whisper without WhisperJAV."""
    print("\n" + "="*60)
    print("TEST 1: Direct faster-whisper (control)")
    print("="*60)

    from faster_whisper import WhisperModel
    import soundfile as sf

    # Load audio exactly as WhisperJAV does
    audio, sr = sf.read(str(TEST_FILE), dtype='float32')
    print(f"Audio: shape={audio.shape}, dtype={audio.dtype}, sr={sr}")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        print(f"Converted to mono: {audio.shape}")

    # Load model
    model = WhisperModel("large-v2", device="cuda", compute_type="float16")

    # Use WhisperJAV's typical parameters
    params = {
        'task': 'transcribe',
        'language': 'ja',
        'beam_size': 3,
        'best_of': 1,
        'patience': 1.6,
        'suppress_tokens': [],
        'suppress_blank': False,
        'without_timestamps': False,
        'temperature': 0.0,
        'compression_ratio_threshold': 3.0,
        'log_prob_threshold': -2.5,
        'no_speech_threshold': 0.22,
        'condition_on_previous_text': False,
        'word_timestamps': True,
        'chunk_length': 14,
        'repetition_penalty': 1.1,
        'no_repeat_ngram_size': 2,
        'multilingual': False,
        'log_progress': False,
        'vad_filter': False,
    }

    print(f"Transcribing with {len(params)} parameters...")
    segments_gen, info = model.transcribe(audio, **params)

    print("Iterating segments...")
    count = 0
    for seg in segments_gen:
        count += 1
        print(f"  [{count}] {seg.start:.2f}s-{seg.end:.2f}s")

    print(f"\nTEST 1 PASSED: {count} segments")

    del model
    import torch
    torch.cuda.empty_cache()
    gc.collect()

    return True


def test_2_whisperjav_asr():
    """Test WhisperJAV's FasterWhisperProASR directly."""
    print("\n" + "="*60)
    print("TEST 2: WhisperJAV FasterWhisperProASR")
    print("="*60)

    try:
        from whisperjav.modules.faster_whisper_pro_asr import FasterWhisperProASR
        from whisperjav.modules.speech_segmentation import create_speech_segmenter
    except ImportError as e:
        print(f"SKIPPED: Cannot import WhisperJAV: {e}")
        return None

    # Create segmenter (required by ASR)
    segmenter = create_speech_segmenter("none")

    # Create ASR with typical balanced+aggressive settings
    asr = FasterWhisperProASR(
        model_size="large-v2",
        device="cuda",
        compute_type="float16",
        task='transcribe',
        language='ja',
        external_segmenter=segmenter,
    )

    print("ASR created, transcribing...")
    result = asr.transcribe(TEST_FILE)

    segments = result.get('segments', [])
    print(f"  Got {len(segments)} segments")
    for i, seg in enumerate(segments[:3]):
        print(f"  [{i+1}] {seg['start']:.2f}s-{seg['end']:.2f}s")

    print(f"\nTEST 2 PASSED: {len(segments)} segments")
    return True


def test_3_multiple_calls():
    """Test multiple transcription calls (immortal object pattern simulation)."""
    print("\n" + "="*60)
    print("TEST 3: Multiple calls (immortal pattern)")
    print("="*60)

    from faster_whisper import WhisperModel
    import soundfile as sf

    audio, sr = sf.read(str(TEST_FILE), dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Create model once, reuse multiple times
    model = WhisperModel("large-v2", device="cuda", compute_type="float16")

    params = {
        'task': 'transcribe',
        'language': 'ja',
        'beam_size': 3,
        'word_timestamps': True,
        'no_speech_threshold': 0.22,
    }

    for i in range(5):
        print(f"\nCall {i+1}/5...")
        segments_gen, info = model.transcribe(audio, **params)
        count = sum(1 for _ in segments_gen)
        print(f"  {count} segments")

    print(f"\nTEST 3 PASSED: 5 calls completed")

    # DON'T delete model - simulate immortal pattern
    # del model

    return True


def test_4_audio_slicing():
    """Test if audio slicing (like scene detection does) causes issues."""
    print("\n" + "="*60)
    print("TEST 4: Audio slicing (scene detection simulation)")
    print("="*60)

    from faster_whisper import WhisperModel
    import soundfile as sf

    audio, sr = sf.read(str(TEST_FILE), dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Simulate scene detection - slice audio into chunks
    duration = len(audio) / sr
    print(f"Full audio: {duration:.2f}s")

    model = WhisperModel("large-v2", device="cuda", compute_type="float16")

    params = {
        'task': 'transcribe',
        'language': 'ja',
        'beam_size': 3,
        'word_timestamps': True,
    }

    # Process in chunks like scene detection does
    chunk_size = 5.0  # seconds
    chunks = []
    for start_sec in np.arange(0, duration, chunk_size):
        end_sec = min(start_sec + chunk_size, duration)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        chunk = audio[start_sample:end_sample]
        chunks.append((start_sec, end_sec, chunk))

    print(f"Split into {len(chunks)} chunks")

    for i, (start, end, chunk) in enumerate(chunks):
        print(f"\nChunk {i+1}: {start:.2f}s-{end:.2f}s ({len(chunk)} samples)")
        segments_gen, info = model.transcribe(chunk, **params)
        count = sum(1 for _ in segments_gen)
        print(f"  {count} segments")

    print(f"\nTEST 4 PASSED: All chunks processed")
    return True


def main():
    print("="*60)
    print("WhisperJAV Pipeline Diagnostic Test")
    print("="*60)
    print(f"Test file: {TEST_FILE}")
    print(f"File exists: {TEST_FILE.exists()}")

    if not TEST_FILE.exists():
        print("ERROR: Test file not found!")
        sys.exit(1)

    results = {}

    # Run tests
    try:
        results['test_1'] = test_1_direct_faster_whisper()
    except Exception as e:
        print(f"\nTEST 1 FAILED: {e}")
        results['test_1'] = False

    try:
        results['test_2'] = test_2_whisperjav_asr()
    except Exception as e:
        print(f"\nTEST 2 FAILED: {e}")
        results['test_2'] = False

    try:
        results['test_3'] = test_3_multiple_calls()
    except Exception as e:
        print(f"\nTEST 3 FAILED: {e}")
        results['test_3'] = False

    try:
        results['test_4'] = test_4_audio_slicing()
    except Exception as e:
        print(f"\nTEST 4 FAILED: {e}")
        results['test_4'] = False

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "PASSED" if passed else ("SKIPPED" if passed is None else "FAILED")
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
