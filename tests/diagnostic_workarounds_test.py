#!/usr/bin/env python3
"""
Test individual workarounds for the ctranslate2 divide-by-zero crash.
Runs each test in isolation via subprocess to prevent crash cascade.
"""

import os
import sys
import subprocess
from pathlib import Path

TEST_AUDIO = r"C:\BIN\git\whisperJav_V1_Minami_Edition\test_media\173_acceptance_test\SONE-966-15_sec_test-966-00_01_45-00_01_59.wav"

# Use repr() for Python literal syntax in subprocess
CRASH_PARAMS_STR = """{
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
}"""


def run_test(test_name, params_str, audio_transform=""):
    """Run test in subprocess."""
    code = f'''
import os
import sys
os.chdir(r"C:\\BIN\\git\\whisperJav_V1_Minami_Edition")

import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

TEST_AUDIO = r"{TEST_AUDIO}"
params = {params_str}

# Load audio
audio, sr = sf.read(TEST_AUDIO, dtype="float32")
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)
audio = audio.astype(np.float32)

# Audio transform
{audio_transform if audio_transform else "pass"}

print(f"Audio: {{len(audio)}} samples, {{len(audio)/sr:.3f}}s @ {{sr}}Hz", flush=True)

# Load model
model = WhisperModel("large-v2", device="cuda", compute_type="int8_float16")
print("Model loaded", flush=True)

# Transcribe
segments_gen, info = model.transcribe(audio, **params)
print(f"Generator created, lang={{info.language}}", flush=True)

segments = []
for i, seg in enumerate(segments_gen):
    text_preview = seg.text[:20].replace("\\n", " ") if seg.text else ""
    print(f"Segment {{i}}: {{seg.start:.2f}}s-{{seg.end:.2f}}s", flush=True)
    segments.append(seg)

print(f"SUCCESS: {{len(segments)}} segments", flush=True)
'''

    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return False

    # Print stdout
    for line in result.stdout.split('\n'):
        if line.strip():
            print(f"  {line}")

    if result.returncode != 0:
        print(f"  EXIT CODE: {result.returncode}")
        if result.returncode == -1073741676:
            print("  >>> STATUS_INTEGER_DIVIDE_BY_ZERO <<<")
        return False

    return True


def main():
    results = {}

    # Test 1: Baseline (should crash)
    results["1_baseline"] = run_test("baseline_crash_params", CRASH_PARAMS_STR)

    # Test 2: word_timestamps=False
    params = CRASH_PARAMS_STR.replace('"word_timestamps": True', '"word_timestamps": False')
    results["2_word_timestamps_false"] = run_test("word_timestamps=False", params)

    # Test 3: Pad audio to 15 seconds
    results["3_audio_pad_15s"] = run_test(
        "audio_pad_15s",
        CRASH_PARAMS_STR,
        "audio = np.pad(audio, (0, int(15 * sr) - len(audio)), mode='constant')"
    )

    # Test 4: Truncate audio to 13.5 seconds
    results["4_truncate_13.5s"] = run_test(
        "audio_truncate_13.5s",
        CRASH_PARAMS_STR,
        "audio = audio[:int(13.5 * sr)]"
    )

    # Test 5: chunk_length=30
    params = CRASH_PARAMS_STR.replace('"chunk_length": 14', '"chunk_length": 30')
    results["5_chunk_length_30"] = run_test("chunk_length=30", params)

    # Test 6: beam_size=1
    params = CRASH_PARAMS_STR.replace('"beam_size": 3', '"beam_size": 1')
    results["6_beam_size_1"] = run_test("beam_size=1", params)

    # Test 7: vad_filter=True (use built-in VAD)
    params = CRASH_PARAMS_STR.replace('"vad_filter": False', '"vad_filter": True')
    results["7_vad_filter_true"] = run_test("vad_filter=True", params)

    # Test 8: Pad audio to exactly 14.0 seconds (remove the 0.016s tail)
    results["8_truncate_14.0s"] = run_test(
        "truncate_14.0s",
        CRASH_PARAMS_STR,
        "audio = audio[:int(14.0 * sr)]"
    )

    # Test 9: Pad audio to exactly 14.5 seconds
    results["9_pad_14.5s"] = run_test(
        "pad_14.5s",
        CRASH_PARAMS_STR,
        "target = int(14.5 * sr); audio = np.pad(audio, (0, target - len(audio)), mode='constant')"
    )

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in sorted(results.items()):
        status = "PASSED" if passed else "CRASHED"
        print(f"  {name}: {status}")

    passed_tests = [k for k, v in results.items() if v]
    print(f"\nWorking workarounds: {len(passed_tests)}/{len(results)}")
    if passed_tests:
        print("  " + ", ".join(passed_tests))


if __name__ == "__main__":
    main()
