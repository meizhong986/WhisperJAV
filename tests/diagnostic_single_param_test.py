"""
Diagnostic Test: Single Parameter Variation (No Subprocess)

Runs ONE parameter variation at a time in the main process.
If it crashes, that variation doesn't help.
If it passes, that parameter change PREVENTS the crash.

Usage:
    python diagnostic_single_param_test.py <test_name>
    python diagnostic_single_param_test.py --list

Tests:
    baseline     - Exact crash params (WILL CRASH)
    beam1        - beam_size=1 (greedy, no beam search)
    beam5        - beam_size=5
    no_word_ts   - word_timestamps=False
    no_speech_06 - no_speech_threshold=0.6
    no_suppress  - Remove suppress_tokens
    no_patience  - Remove patience parameter
    no_chunk     - Remove chunk_length
    no_rep_pen   - Remove repetition_penalty
    minimal      - Minimal params (only required)
"""

import sys
import time
from pathlib import Path

TEST_FILE = Path(r"C:\BIN\git\whisperJav_V1_Minami_Edition\test_media\173_acceptance_test\SONE-966-15_sec_test-966-00_01_45-00_01_59.wav")
MODEL_SIZE = "large-v2"

# Baseline params that CAUSE the crash
BASELINE = {
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

def get_test_params(test_name: str) -> dict:
    """Get params for a specific test."""
    params = BASELINE.copy()

    if test_name == 'baseline':
        pass  # Use baseline as-is
    elif test_name == 'beam1':
        params['beam_size'] = 1
    elif test_name == 'beam5':
        params['beam_size'] = 5
    elif test_name == 'no_word_ts':
        params['word_timestamps'] = False
    elif test_name == 'no_speech_06':
        params['no_speech_threshold'] = 0.6
    elif test_name == 'no_suppress':
        del params['suppress_tokens']
    elif test_name == 'no_patience':
        del params['patience']
    elif test_name == 'no_chunk':
        del params['chunk_length']
    elif test_name == 'no_rep_pen':
        del params['repetition_penalty']
    elif test_name == 'no_ngram':
        del params['no_repeat_ngram_size']
    elif test_name == 'minimal':
        # Only essential params
        params = {
            'task': 'transcribe',
            'language': 'ja',
            'beam_size': 5,
            'word_timestamps': False,
        }
    else:
        print(f"Unknown test: {test_name}")
        sys.exit(1)

    return params


def run_test(test_name: str):
    """Run a single test."""
    from faster_whisper import WhisperModel
    import soundfile as sf

    print("="*60)
    print(f"TEST: {test_name}")
    print("="*60)

    params = get_test_params(test_name)

    # Show what's different from baseline
    print("\nParameter changes from baseline:")
    if test_name == 'baseline':
        print("  (none - using exact baseline)")
    elif test_name == 'minimal':
        print("  Using minimal params only")
    else:
        for k, v in BASELINE.items():
            if k not in params:
                print(f"  {k}: REMOVED")
            elif params[k] != v:
                print(f"  {k}: {v} -> {params[k]}")

    print(f"\nKey params: beam_size={params.get('beam_size', 'N/A')}, "
          f"word_timestamps={params.get('word_timestamps', 'N/A')}, "
          f"no_speech_threshold={params.get('no_speech_threshold', 'N/A')}")

    # Load audio
    print(f"\nLoading audio: {TEST_FILE.name}")
    audio, sr = sf.read(str(TEST_FILE))
    print(f"Audio: {len(audio)/sr:.2f}s @ {sr}Hz")

    # Load model
    print("\nLoading model...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    print("Model loaded")

    # Transcribe
    print("\nTranscribing...")
    start = time.time()

    segments_gen, info = model.transcribe(audio, **params)
    print(f"Generator created (lang={info.language}, prob={info.language_probability:.3f})")

    print("Iterating segments...")
    segments = []
    for i, seg in enumerate(segments_gen):
        segments.append(seg)
        # Safe print (no Japanese to avoid encoding issues)
        print(f"  [{i+1}] {seg.start:.2f}s - {seg.end:.2f}s ({len(seg.text)} chars)")

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"PASSED: {test_name}")
    print(f"{'='*60}")
    print(f"Segments: {len(segments)}")
    print(f"Duration: {elapsed:.2f}s")

    # Cleanup
    del model
    import torch
    torch.cuda.empty_cache()

    return True


def list_tests():
    """List available tests."""
    print("Available tests:")
    print("  baseline     - Exact crash params (WILL CRASH)")
    print("  beam1        - beam_size=1 (greedy decoding)")
    print("  beam5        - beam_size=5")
    print("  no_word_ts   - word_timestamps=False")
    print("  no_speech_06 - no_speech_threshold=0.6")
    print("  no_suppress  - Remove suppress_tokens=[]")
    print("  no_patience  - Remove patience=1.6")
    print("  no_chunk     - Remove chunk_length=14")
    print("  no_rep_pen   - Remove repetition_penalty=1.1")
    print("  no_ngram     - Remove no_repeat_ngram_size=2")
    print("  minimal      - Minimal params only")
    print()
    print("Recommended order (safest first):")
    print("  1. minimal")
    print("  2. beam1")
    print("  3. no_word_ts")
    print("  4. beam5")
    print("  5. no_speech_06")
    print("  6. baseline (will crash)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnostic_single_param_test.py <test_name>")
        print("       python diagnostic_single_param_test.py --list")
        sys.exit(1)

    if sys.argv[1] == '--list':
        list_tests()
        sys.exit(0)

    test_name = sys.argv[1]

    if not TEST_FILE.exists():
        print(f"ERROR: Test file not found: {TEST_FILE}")
        sys.exit(1)

    try:
        run_test(test_name)
        sys.exit(0)
    except Exception as e:
        print(f"\nFAILED with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
