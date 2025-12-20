"""
Diagnostic Test: Task Override Mutation (Shared State Corruption)

Tests whether mutating whisper_params AFTER model initialization causes
crashes due to stale references in ctranslate2 C++ layer.

Hypothesis:
- WhisperJAV keeps ASR alive via _IMMORTAL_ASR_REFERENCE
- whisper_params dict is passed to ctranslate2 at model init
- Task override mutates whisper_params['task'] AFTER init
- ctranslate2 may hold reference to original dict contents
- Mutation causes memory corruption -> STATUS_INTEGER_DIVIDE_BY_ZERO

This test simulates the immortal object pattern with task overrides.

Created: 2025-12-19
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Test configuration
TEST_FILE = Path(r"C:\BIN\git\whisperJav_V1_Minami_Edition\test_media\173_acceptance_test\SONE-966-15_sec_test-966-00_01_45-00_01_59.wav")
MODEL_SIZE = "large-v2"

# Output files
SCRIPT_DIR = Path(__file__).parent
OUTPUT_LOG = SCRIPT_DIR / "diagnostic_task_override_results.log"
OUTPUT_JSON = SCRIPT_DIR / "diagnostic_task_override_results.json"


class TeeOutput:
    """Write to both stdout and file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def test_mutation_pattern_direct():
    """
    Test 1: Direct faster-whisper with dict mutation pattern.

    Simulates the problematic pattern:
    1. Create model with params dict
    2. Run transcription
    3. MUTATE the params dict
    4. Run transcription again
    5. Check for crash
    """
    from faster_whisper import WhisperModel
    import soundfile as sf

    print("\n" + "="*70)
    print("TEST 1: Direct faster-whisper dict mutation")
    print("="*70)

    # Load audio once
    print(f"Loading audio: {TEST_FILE.name}")
    audio, sr = sf.read(str(TEST_FILE))
    print(f"Audio: {len(audio)} samples, {len(audio)/sr:.2f}s")

    # Create params dict (simulating whisper_params)
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

    print("\nStep 1: Loading model...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    print(f"Model loaded: {MODEL_SIZE} on CUDA")
    print(f"Initial params['task'] = '{params['task']}'")

    # First transcription with task='transcribe'
    print("\nStep 2: First transcription (task='transcribe')...")
    try:
        segments_gen, info = model.transcribe(audio, **params)
        segments = list(segments_gen)
        print(f"  SUCCESS: {len(segments)} segments")
    except Exception as e:
        print(f"  FAILED: {e}")
        return {'test': 'direct_mutation', 'status': 'FAILED', 'step': 2, 'error': str(e)}

    # MUTATE the params dict (simulating task override)
    print("\nStep 3: MUTATING params dict (task='transcribe' -> 'translate')...")
    params['task'] = 'translate'  # <-- THE DANGEROUS MUTATION
    print(f"  params['task'] is now '{params['task']}'")
    print("  (This simulates WhisperJAV's task override mutation)")

    # Second transcription with mutated dict
    print("\nStep 4: Second transcription with MUTATED params...")
    try:
        segments_gen, info = model.transcribe(audio, **params)
        segments = list(segments_gen)
        print(f"  SUCCESS: {len(segments)} segments")
    except Exception as e:
        print(f"  FAILED: {e}")
        return {'test': 'direct_mutation', 'status': 'FAILED', 'step': 4, 'error': str(e)}

    # Mutate back
    print("\nStep 5: MUTATING back (task='translate' -> 'transcribe')...")
    params['task'] = 'transcribe'
    print(f"  params['task'] is now '{params['task']}'")

    # Third transcription
    print("\nStep 6: Third transcription after mutation cycle...")
    try:
        segments_gen, info = model.transcribe(audio, **params)
        segments = list(segments_gen)
        print(f"  SUCCESS: {len(segments)} segments")
    except Exception as e:
        print(f"  FAILED: {e}")
        return {'test': 'direct_mutation', 'status': 'FAILED', 'step': 6, 'error': str(e)}

    # Cleanup
    del model
    import torch
    torch.cuda.empty_cache()

    print("\n  TEST 1 PASSED: Dict mutation did not cause crash in direct faster-whisper")
    return {'test': 'direct_mutation', 'status': 'PASSED', 'steps_completed': 6}


def test_mutation_pattern_whisperjav():
    """
    Test 2: Through WhisperJAV's FasterWhisperProASR with task overrides.

    Tests the actual code path that has the mutation.
    """
    print("\n" + "="*70)
    print("TEST 2: WhisperJAV FasterWhisperProASR task override")
    print("="*70)

    try:
        from whisperjav.modules.faster_whisper_pro_asr import FasterWhisperProASR
    except ImportError as e:
        print(f"  SKIPPED: Cannot import FasterWhisperProASR: {e}")
        return {'test': 'whisperjav_mutation', 'status': 'SKIPPED', 'error': str(e)}

    print(f"\nUsing audio: {TEST_FILE.name}")

    # Create ASR with task='transcribe' (initial state)
    print("\nStep 1: Creating FasterWhisperProASR with task='transcribe'...")
    try:
        asr = FasterWhisperProASR(
            model_size=MODEL_SIZE,
            device="cuda",
            compute_type="float16",
            task='transcribe',
            language='ja',
        )
        print("  ASR created successfully")
        print(f"  asr.task = '{asr.task}'")
        print(f"  asr.whisper_params['task'] = '{asr.whisper_params.get('task', 'NOT SET')}'")
    except Exception as e:
        print(f"  FAILED: {e}")
        return {'test': 'whisperjav_mutation', 'status': 'FAILED', 'step': 1, 'error': str(e)}

    # Store reference to simulate immortal pattern
    _immortal_ref = asr
    print("  (Stored reference to simulate _IMMORTAL_ASR_REFERENCE)")

    # First transcription
    print("\nStep 2: First transcription (no task override)...")
    try:
        result = asr.transcribe(TEST_FILE)
        segments = result.get('segments', [])
        print(f"  SUCCESS: {len(segments)} segments")
    except Exception as e:
        print(f"  FAILED: {e}")
        return {'test': 'whisperjav_mutation', 'status': 'FAILED', 'step': 2, 'error': str(e)}

    # Second transcription WITH task override (triggers mutation)
    print("\nStep 3: Second transcription WITH task='translate' override...")
    print("  (This triggers the mutation in transcribe() method)")
    try:
        result = asr.transcribe(TEST_FILE, task='translate')
        segments = result.get('segments', [])
        print(f"  SUCCESS: {len(segments)} segments")
        print(f"  asr.task is now '{asr.task}' (MUTATED)")
        print(f"  asr.whisper_params['task'] is now '{asr.whisper_params.get('task', 'NOT SET')}' (MUTATED)")
    except Exception as e:
        print(f"  FAILED: {e}")
        return {'test': 'whisperjav_mutation', 'status': 'FAILED', 'step': 3, 'error': str(e)}

    # Third transcription - back to transcribe
    print("\nStep 4: Third transcription WITH task='transcribe' override...")
    print("  (Mutating back - potential for stale state)")
    try:
        result = asr.transcribe(TEST_FILE, task='transcribe')
        segments = result.get('segments', [])
        print(f"  SUCCESS: {len(segments)} segments")
    except Exception as e:
        print(f"  FAILED: {e}")
        return {'test': 'whisperjav_mutation', 'status': 'FAILED', 'step': 4, 'error': str(e)}

    # Multiple rapid mutations
    print("\nStep 5: Rapid task switching (stress test)...")
    for i in range(5):
        task = 'translate' if i % 2 == 0 else 'transcribe'
        try:
            result = asr.transcribe(TEST_FILE, task=task)
            segments = result.get('segments', [])
            print(f"  Iteration {i+1} (task='{task}'): {len(segments)} segments")
        except Exception as e:
            print(f"  FAILED at iteration {i+1}: {e}")
            return {'test': 'whisperjav_mutation', 'status': 'FAILED', 'step': 5, 'iteration': i+1, 'error': str(e)}

    # Keep reference alive (don't delete)
    print("\n  (Keeping _immortal_ref alive to simulate WhisperJAV pattern)")

    print("\n  TEST 2 PASSED: Task override mutation did not cause crash")
    return {'test': 'whisperjav_mutation', 'status': 'PASSED', 'steps_completed': 5}


def test_copy_vs_mutate():
    """
    Test 3: Compare copy-on-override vs direct mutation.

    Tests the proposed fix: creating a copy instead of mutating.
    """
    from faster_whisper import WhisperModel
    import soundfile as sf
    import copy

    print("\n" + "="*70)
    print("TEST 3: Copy-on-override vs Direct mutation comparison")
    print("="*70)

    # Load audio
    audio, sr = sf.read(str(TEST_FILE))
    print(f"Audio loaded: {len(audio)/sr:.2f}s")

    # Base params
    base_params = {
        'task': 'transcribe',
        'language': 'ja',
        'beam_size': 3,
        'word_timestamps': True,
        'no_speech_threshold': 0.22,
    }

    print("\nStep 1: Loading model...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    # Test A: Direct mutation (current WhisperJAV behavior)
    print("\nTest 3A: Direct mutation pattern...")
    params_a = base_params.copy()
    try:
        # First call
        segments_gen, _ = model.transcribe(audio, **params_a)
        list(segments_gen)
        # Mutate
        params_a['task'] = 'translate'
        # Second call with mutated dict
        segments_gen, _ = model.transcribe(audio, **params_a)
        list(segments_gen)
        print("  3A PASSED: Direct mutation works")
        result_a = 'PASSED'
    except Exception as e:
        print(f"  3A FAILED: {e}")
        result_a = f'FAILED: {e}'

    # Test B: Copy-on-override (proposed fix)
    print("\nTest 3B: Copy-on-override pattern...")
    params_b = base_params.copy()
    try:
        # First call
        segments_gen, _ = model.transcribe(audio, **params_b)
        list(segments_gen)
        # Create NEW dict for override (proposed fix)
        params_b_override = params_b.copy()
        params_b_override['task'] = 'translate'
        # Second call with NEW dict
        segments_gen, _ = model.transcribe(audio, **params_b_override)
        list(segments_gen)
        print("  3B PASSED: Copy-on-override works")
        result_b = 'PASSED'
    except Exception as e:
        print(f"  3B FAILED: {e}")
        result_b = f'FAILED: {e}'

    del model
    import torch
    torch.cuda.empty_cache()

    print(f"\n  TEST 3 COMPLETE: Mutation={result_a}, Copy={result_b}")
    return {'test': 'copy_vs_mutate', 'mutation': result_a, 'copy': result_b}


def main():
    # Set up logging
    tee = TeeOutput(OUTPUT_LOG)
    sys.stdout = tee

    timestamp = datetime.now().isoformat()

    print("="*70)
    print("DIAGNOSTIC: Task Override Mutation Test")
    print("="*70)
    print(f"Timestamp: {timestamp}")
    print(f"Test file: {TEST_FILE}")
    print(f"Model: {MODEL_SIZE}")
    print()
    print("This test checks if mutating whisper_params after model init")
    print("causes crashes due to stale references in ctranslate2.")
    print()

    if not TEST_FILE.exists():
        print(f"ERROR: Test file not found: {TEST_FILE}")
        sys.exit(1)

    results = []

    # Run tests
    try:
        results.append(test_mutation_pattern_direct())
    except Exception as e:
        print(f"TEST 1 EXCEPTION: {e}")
        results.append({'test': 'direct_mutation', 'status': 'EXCEPTION', 'error': str(e)})

    try:
        results.append(test_mutation_pattern_whisperjav())
    except Exception as e:
        print(f"TEST 2 EXCEPTION: {e}")
        results.append({'test': 'whisperjav_mutation', 'status': 'EXCEPTION', 'error': str(e)})

    try:
        results.append(test_copy_vs_mutate())
    except Exception as e:
        print(f"TEST 3 EXCEPTION: {e}")
        results.append({'test': 'copy_vs_mutate', 'status': 'EXCEPTION', 'error': str(e)})

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        test_name = r.get('test', 'unknown')
        status = r.get('status', 'unknown')
        print(f"  {test_name}: {status}")

    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)

    all_passed = all(r.get('status') == 'PASSED' for r in results if r.get('status') not in ['SKIPPED'])
    if all_passed:
        print("  All tests PASSED - Task override mutation is NOT the cause of the crash")
        print("  The crash must be triggered by something else in the pipeline")
    else:
        failed = [r for r in results if r.get('status') not in ['PASSED', 'SKIPPED']]
        print(f"  {len(failed)} test(s) FAILED - Task override mutation MAY be the cause")
        for r in failed:
            print(f"    - {r.get('test')}: {r.get('error', 'Unknown')}")

    print("\n" + "="*70)

    # Save JSON
    json_output = {
        'timestamp': timestamp,
        'test_file': str(TEST_FILE),
        'model': MODEL_SIZE,
        'results': results,
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  Log: {OUTPUT_LOG}")
    print(f"  JSON: {OUTPUT_JSON}")

    tee.close()
    sys.stdout = tee.terminal


if __name__ == "__main__":
    main()
