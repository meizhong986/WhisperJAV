#!/usr/bin/env python3
"""
Minimal reproduction test for ctranslate2 STATUS_INTEGER_DIVIDE_BY_ZERO crash.

This test uses raw faster-whisper WITHOUT WhisperJAV to determine if the crash
is in faster-whisper/ctranslate2 itself or in WhisperJAV's audio handling.

The crash occurs during the 5th next() call on the segment generator when:
- Audio is 14.015 seconds
- 4 segments cover 1.2s-13.98s
- Remaining 0.035s (35ms) triggers divide-by-zero in beam search

HYPOTHESIS: The crash is triggered by very short audio tail (< 100ms) at the end.
"""

import os
import sys
import json
import time
from pathlib import Path

# Test audio file
TEST_AUDIO = Path(r"C:\BIN\git\whisperJav_V1_Minami_Edition\test_media\173_acceptance_test\SONE-966-15_sec_test-966-00_01_45-00_01_59.wav")

# Exact parameters from crash trace
CRASH_PARAMS = {
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
    "chunk_length": 14,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 2,
    "multilingual": False,
    "log_progress": False,
    "log_prob_threshold": -2.5,
    "vad_filter": False,
    "vad_parameters": {
        "threshold": 0.187,
        "neg_threshold": 0.1,
        "min_speech_duration_ms": 30,
        "max_speech_duration_s": 6.0,
        "min_silence_duration_ms": 300,
        "speech_pad_ms": 500
    }
}


def trace(msg: str):
    """Write trace with immediate flush."""
    timestamp = time.time()
    line = f"[{timestamp:.3f}] {msg}"
    print(line, flush=True)
    # Also write to file for crash analysis
    with open("raw_fw_trace.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_audio_whisperjav_style():
    """Load audio the same way WhisperJAV does."""
    import soundfile as sf
    import numpy as np

    trace(f"Loading audio: {TEST_AUDIO}")
    audio, sr = sf.read(str(TEST_AUDIO), dtype='float32')
    trace(f"Loaded: shape={audio.shape}, dtype={audio.dtype}, sr={sr}")

    # Mono conversion (if stereo)
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        trace("Converting stereo to mono")
        audio = np.mean(audio, axis=1)

    # Ensure float32
    audio = audio.astype(np.float32)
    trace(f"Final: shape={audio.shape}, duration={len(audio)/sr:.3f}s")

    return audio, sr


def load_audio_raw_librosa():
    """Load audio with librosa (how faster-whisper typically loads)."""
    try:
        import librosa
        trace(f"Loading audio with librosa: {TEST_AUDIO}")
        audio, sr = librosa.load(str(TEST_AUDIO), sr=16000, mono=True)
        trace(f"Loaded: shape={audio.shape}, duration={len(audio)/sr:.3f}s")
        return audio, sr
    except ImportError:
        trace("librosa not available, skipping this method")
        return None, None


def test_raw_faster_whisper(audio, sr, params, test_name):
    """Test with raw faster-whisper."""
    from faster_whisper import WhisperModel

    trace(f"\n=== TEST: {test_name} ===")
    trace(f"Audio: {len(audio)} samples, {len(audio)/sr:.3f}s @ {sr}Hz")
    trace(f"Params: {json.dumps({k: v for k, v in params.items() if k != 'vad_parameters'}, default=str)}")

    trace("Loading model...")
    model = WhisperModel("large-v2", device="cuda", compute_type="int8_float16")
    trace("Model loaded")

    trace("Calling transcribe()...")
    segments_generator, info = model.transcribe(audio, **params)
    trace(f"Generator created, language={info.language}, prob={info.language_probability:.3f}")

    trace("Iterating segments...")
    segments = []
    seg_idx = 0

    while True:
        trace(f"  next() for segment {seg_idx}...")
        try:
            segment = next(segments_generator)
            trace(f"  Segment {seg_idx}: {segment.start:.2f}s-{segment.end:.2f}s, text_len={len(segment.text)}")
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text[:50] if segment.text else ""
            })
            seg_idx += 1
        except StopIteration:
            trace(f"  StopIteration - total {len(segments)} segments")
            break

    trace(f"SUCCESS: {test_name} completed with {len(segments)} segments")
    return True


def test_with_audio_padding(audio, sr, params):
    """Test with audio padded to avoid short tail."""
    import numpy as np

    # Pad to next full second
    current_duration = len(audio) / sr
    target_duration = int(current_duration) + 1  # Round up to next second
    padding_samples = int((target_duration - current_duration) * sr)

    trace(f"\n=== TEST: Audio padding ===")
    trace(f"Original: {current_duration:.3f}s, padding {padding_samples} samples to {target_duration}s")

    padded_audio = np.pad(audio, (0, padding_samples), mode='constant', constant_values=0)
    return test_raw_faster_whisper(padded_audio, sr, params, "padded_audio")


def test_without_word_timestamps(audio, sr, params):
    """Test with word_timestamps=False."""
    test_params = params.copy()
    test_params["word_timestamps"] = False
    return test_raw_faster_whisper(audio, sr, test_params, "word_timestamps=False")


def test_with_truncation(audio, sr, params):
    """Test with audio truncated to 13.5 seconds."""
    truncated = audio[:int(13.5 * sr)]
    return test_raw_faster_whisper(truncated, sr, params, "truncated_to_13.5s")


def main():
    # Clear trace log
    with open("raw_fw_trace.log", "w") as f:
        f.write("")

    trace("=" * 60)
    trace("RAW FASTER-WHISPER CRASH REPRODUCTION TEST")
    trace("=" * 60)

    # Load audio both ways
    audio_wj, sr_wj = load_audio_whisperjav_style()
    audio_librosa, sr_librosa = load_audio_raw_librosa()

    # Use WhisperJAV-style loading (to match the crash conditions)
    audio = audio_wj
    sr = sr_wj

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        trace(f"Resampling from {sr}Hz to 16000Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    results = {}

    # Test 1: Exact crash params (should crash)
    try:
        test_raw_faster_whisper(audio, sr, CRASH_PARAMS, "exact_crash_params")
        results["exact_crash_params"] = "PASSED"
    except Exception as e:
        trace(f"EXCEPTION: {type(e).__name__}: {e}")
        results["exact_crash_params"] = f"EXCEPTION: {e}"

    # Test 2: Without word_timestamps
    try:
        test_without_word_timestamps(audio, sr, CRASH_PARAMS)
        results["without_word_timestamps"] = "PASSED"
    except Exception as e:
        trace(f"EXCEPTION: {type(e).__name__}: {e}")
        results["without_word_timestamps"] = f"EXCEPTION: {e}"

    # Test 3: With audio padding
    try:
        test_with_audio_padding(audio, sr, CRASH_PARAMS)
        results["audio_padding"] = "PASSED"
    except Exception as e:
        trace(f"EXCEPTION: {type(e).__name__}: {e}")
        results["audio_padding"] = f"EXCEPTION: {e}"

    # Test 4: Truncated audio
    try:
        test_with_truncation(audio, sr, CRASH_PARAMS)
        results["truncated"] = "PASSED"
    except Exception as e:
        trace(f"EXCEPTION: {type(e).__name__}: {e}")
        results["truncated"] = f"EXCEPTION: {e}"

    trace("\n" + "=" * 60)
    trace("RESULTS:")
    for name, result in results.items():
        trace(f"  {name}: {result}")
    trace("=" * 60)

    # Write results to JSON
    with open("raw_fw_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
