"""
Diagnostic Test: Parameter Variation Tests for Crash Investigation

Tests faster-whisper directly with varying parameters to identify which
parameter causes the STATUS_INTEGER_DIVIDE_BY_ZERO crash.

Based on crash analysis from isolation test (2025-12-19):
- Exit code: -1073741676 (0xC0000094 = STATUS_INTEGER_DIVIDE_BY_ZERO)
- Crash occurs during generator iteration in ctranslate2 beam search
- VRAM fragmentation ruled out (crash occurs in isolation)

Hypothesis testing:
- H2: no_speech_threshold=0.22 causes edge case
- H3: beam_size=3 with word_timestamps creates vulnerable code path
- H4: vad_filter=False with vad_parameters dict present causes issue
- H5: Empty suppress_tokens=[] causes edge case in ctranslate2

Updated 2025-12-19: Added exact parameters from crash log including vad_parameters
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Test configuration
TEST_FILE = Path(r"C:\BIN\git\whisperJav_V1_Minami_Edition\test_media\173_acceptance_test\SONE-966-15_sec_test-966-00_01_45-00_01_59.wav")
MODEL_SIZE = "large-v2"

# Output files - in tests directory for easy access
SCRIPT_DIR = Path(__file__).parent
OUTPUT_LOG = SCRIPT_DIR / "diagnostic_param_variation_results.log"
OUTPUT_JSON = SCRIPT_DIR / "diagnostic_param_variation_results.json"


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


def run_single_test_subprocess(test_name: str) -> dict:
    """Run a single test in an isolated subprocess to handle crashes gracefully."""
    print(f"\n{'='*70}")
    print(f"LAUNCHING SUBPROCESS: {test_name}")
    print(f"{'='*70}")

    # Run this script with --single flag
    cmd = [sys.executable, __file__, '--single', test_name]

    # Set UTF-8 encoding for subprocess to avoid Windows charmap errors
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout per test
            encoding='utf-8',
            errors='replace',
            env=env,
        )

        # Print subprocess output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        # Check exit code
        if result.returncode == 0:
            # Try to parse result from stdout
            # Look for JSON marker in output
            for line in result.stdout.split('\n'):
                if line.startswith('JSON_RESULT:'):
                    return json.loads(line[12:])
            # Fallback if no JSON marker
            return {
                'test_name': test_name,
                'status': 'PASSED',
                'segments': 0,
                'duration': 0,
                'error': None,
                'exit_code': 0,
            }
        else:
            return {
                'test_name': test_name,
                'status': 'CRASHED',
                'segments': 0,
                'duration': 0,
                'error': f'Exit code: {result.returncode}',
                'exit_code': result.returncode,
                'stdout': result.stdout[-500:] if result.stdout else '',
                'stderr': result.stderr[-500:] if result.stderr else '',
            }

    except subprocess.TimeoutExpired:
        return {
            'test_name': test_name,
            'status': 'TIMEOUT',
            'error': 'Test exceeded 120 second timeout',
        }
    except Exception as e:
        return {
            'test_name': test_name,
            'status': 'ERROR',
            'error': str(e),
        }

# Baseline parameters - EXACT match from crash log
# Note: vad_filter=False but vad_parameters dict is still present (potential issue)
BASELINE_PARAMS = {
    'task': 'transcribe',
    'language': 'ja',
    'beam_size': 3,
    'best_of': 1,
    'patience': 1.6,
    'suppress_tokens': [],  # Empty list - potential issue (H5)
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
    'vad_parameters': {  # Present even though vad_filter=False (H4)
        'threshold': 0.187,
        'neg_threshold': 0.1,
        'min_speech_duration_ms': 30,
        'max_speech_duration_s': 6.0,
        'min_silence_duration_ms': 300,
        'speech_pad_ms': 500
    },
}

# Special marker for parameter removal (not just override)
REMOVE_PARAM = object()

# Test variations - ordered by likelihood of being the cause
TEST_VARIATIONS = {
    # H4: VAD parameters with vad_filter=False
    'A_no_vad_params': {'vad_parameters': REMOVE_PARAM},  # Remove vad_parameters entirely

    # H5: Empty suppress_tokens
    'B_no_suppress_tokens': {'suppress_tokens': REMOVE_PARAM},  # Remove empty list
    'C_suppress_tokens_none': {'suppress_tokens': None},  # Try None instead

    # H3: beam_size variations
    'D_beam_size_1': {'beam_size': 1},  # Greedy decoding (no beam search)
    'E_beam_size_5': {'beam_size': 5},  # Different beam size

    # H2: no_speech_threshold
    'F_no_speech_0.6': {'no_speech_threshold': 0.6},  # Default value

    # Word timestamps
    'G_word_timestamps_off': {'word_timestamps': False},

    # Combined: Remove both suspicious params
    'H_clean_params': {
        'vad_parameters': REMOVE_PARAM,
        'suppress_tokens': REMOVE_PARAM,
    },

    # Baseline - should crash (run last)
    'Z_baseline_exact': {},  # Exact params from crash - expected to crash
}


def run_test(test_name: str, param_overrides: dict) -> dict:
    """Run a single test with parameter overrides."""
    from faster_whisper import WhisperModel
    import soundfile as sf
    import copy

    # Deep copy baseline (for nested dicts like vad_parameters)
    params = copy.deepcopy(BASELINE_PARAMS)

    # Apply overrides, handling REMOVE_PARAM marker
    for key, value in param_overrides.items():
        if value is REMOVE_PARAM:
            params.pop(key, None)  # Remove the parameter entirely
        else:
            params[key] = value

    result = {
        'test_name': test_name,
        'overrides': param_overrides,
        'status': 'unknown',
        'segments': 0,
        'text': '',
        'error': None,
        'duration': 0,
    }

    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")

    # Display changes clearly
    if param_overrides:
        changes = []
        for k, v in param_overrides.items():
            if v is REMOVE_PARAM:
                changes.append(f"{k}=REMOVED")
            else:
                changes.append(f"{k}={v}")
        print(f"Changes: {', '.join(changes)}")
    else:
        print("Changes: NONE (exact baseline)")

    # Show key params in final form
    print(f"Final params: beam_size={params.get('beam_size', 'N/A')}, "
          f"word_timestamps={params.get('word_timestamps', 'N/A')}, "
          f"no_speech_threshold={params.get('no_speech_threshold', 'N/A')}, "
          f"suppress_tokens={'ABSENT' if 'suppress_tokens' not in params else params['suppress_tokens']}, "
          f"vad_parameters={'ABSENT' if 'vad_parameters' not in params else 'PRESENT'}")
    print()

    try:
        # Load model
        print("Loading model...")
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        print(f"Model loaded: {MODEL_SIZE} on CUDA")

        # Load audio
        print(f"Loading audio: {TEST_FILE.name}")
        audio, sr = sf.read(str(TEST_FILE))
        if sr != 16000:
            print(f"Warning: Sample rate is {sr}, expected 16000")
        print(f"Audio: {len(audio)} samples, {len(audio)/sr:.2f}s")

        # Transcribe
        print("Starting transcription...")
        start_time = time.time()

        segments_gen, info = model.transcribe(audio, **params)
        print(f"Generator created, info: language={info.language}, prob={info.language_probability:.3f}")

        # Iterate through segments (this is where crash occurs)
        print("Iterating segments...")
        segments = []
        for seg in segments_gen:
            segments.append(seg)
            # Safe print - avoid Windows console encoding errors with Japanese text
            try:
                text_preview = seg.text[:50].encode('ascii', 'replace').decode('ascii')
            except:
                text_preview = "[Japanese text]"
            print(f"  Segment: [{seg.start:.2f}-{seg.end:.2f}] {text_preview}...")

        elapsed = time.time() - start_time

        result['status'] = 'PASSED'
        result['segments'] = len(segments)
        result['text'] = ' '.join(s.text for s in segments)
        result['duration'] = elapsed

        print(f"\nRESULT: PASSED")
        print(f"Segments: {len(segments)}")
        print(f"Duration: {elapsed:.2f}s")

        # Cleanup
        del model
        import torch
        torch.cuda.empty_cache()

    except Exception as e:
        result['status'] = 'FAILED'
        result['error'] = str(e)
        print(f"\nRESULT: FAILED")
        print(f"Error: {e}")

    # Output JSON marker for subprocess parsing (only include serializable data)
    json_result = {
        'test_name': result['test_name'],
        'status': result['status'],
        'segments': result['segments'],
        'duration': result['duration'],
        'error': result['error'],
    }
    print(f"JSON_RESULT:{json.dumps(json_result)}")

    return result


def main():
    # Set up tee output to write to both console and file
    tee = TeeOutput(OUTPUT_LOG)
    sys.stdout = tee

    run_timestamp = datetime.now().isoformat()

    print("="*70)
    print("DIAGNOSTIC: Parameter Variation Tests for Crash Investigation")
    print("="*70)
    print(f"Timestamp: {run_timestamp}")
    print(f"Test file: {TEST_FILE}")
    print(f"Model: {MODEL_SIZE}")
    print(f"Output log: {OUTPUT_LOG}")
    print(f"Output JSON: {OUTPUT_JSON}")
    print()
    print("Hypotheses under test:")
    print("  H2: no_speech_threshold=0.22 causes edge case")
    print("  H3: beam_size=3 with word_timestamps creates vulnerable code path")
    print("  H4: vad_filter=False with vad_parameters dict present causes issue")
    print("  H5: Empty suppress_tokens=[] causes edge case in ctranslate2")
    print()

    # Verify test file exists
    if not TEST_FILE.exists():
        print(f"ERROR: Test file not found: {TEST_FILE}")
        sys.exit(1)

    # Parse command line arguments
    include_baseline = '--baseline' in sys.argv or '-b' in sys.argv
    test_filter = None
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            test_filter = arg
            break

    if test_filter:
        tests_to_run = {k: v for k, v in TEST_VARIATIONS.items() if test_filter.lower() in k.lower()}
        if not tests_to_run:
            print(f"No tests match filter: {test_filter}")
            print(f"Available: {list(TEST_VARIATIONS.keys())}")
            sys.exit(1)
    else:
        # Run all tests except baseline (unless --baseline flag)
        if include_baseline:
            tests_to_run = TEST_VARIATIONS
        else:
            tests_to_run = {k: v for k, v in TEST_VARIATIONS.items() if not k.startswith('Z_')}

    print(f"Tests to run ({len(tests_to_run)}): {list(tests_to_run.keys())}")
    if not include_baseline:
        print("(Use --baseline or -b to include baseline crash test)")

    # Check for subprocess isolation mode
    use_subprocess = '--no-subprocess' not in sys.argv
    if use_subprocess:
        print("\nRunning tests in ISOLATED SUBPROCESSES (crash-safe mode)")
        print("Use --no-subprocess to run in single process (faster but may crash)")

    # Run tests
    results = []
    for test_name, overrides in tests_to_run.items():
        if use_subprocess:
            result = run_single_test_subprocess(test_name)
            results.append(result)
        else:
            try:
                result = run_test(test_name, overrides)
                results.append(result)
            except SystemExit:
                # Process killed - likely the crash
                results.append({
                    'test_name': test_name,
                    'status': 'CRASHED',
                    'error': 'Process terminated unexpectedly'
                })
                break

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        status = r['status']
        name = r['test_name']
        if status == 'PASSED':
            print(f"  [PASS] {name} - {r['segments']} segments in {r['duration']:.1f}s")
        else:
            print(f"  [FAIL] {name} - {r.get('error', 'Unknown error')}")

    # Interpretation
    passed = [r for r in results if r['status'] == 'PASSED']
    failed = [r for r in results if r['status'] != 'PASSED']

    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)

    if passed:
        print(f"\nTests that PREVENT crash ({len(passed)}):")
        for r in passed:
            print(f"  - {r['test_name']}")
        print("\n  -> These parameter changes AVOID the divide-by-zero bug.")

    if failed:
        print(f"\nTests that still CRASH ({len(failed)}):")
        for r in failed:
            print(f"  - {r['test_name']}")

    # Specific hypothesis interpretation
    print("\n" + "-"*70)
    print("HYPOTHESIS RESULTS:")
    print("-"*70)

    passed_names = [r['test_name'] for r in passed]
    failed_names = [r['test_name'] for r in failed]

    # H4: vad_parameters
    if 'A_no_vad_params' in passed_names:
        print("  H4 (vad_parameters with vad_filter=False): CONFIRMED as cause")
        print("      -> FIX: Remove vad_parameters when vad_filter=False")
    elif 'A_no_vad_params' in failed_names:
        print("  H4 (vad_parameters): RULED OUT")

    # H5: suppress_tokens
    if 'B_no_suppress_tokens' in passed_names or 'C_suppress_tokens_none' in passed_names:
        print("  H5 (empty suppress_tokens=[]): CONFIRMED as cause")
        print("      -> FIX: Remove suppress_tokens if empty, or don't pass it")
    elif 'B_no_suppress_tokens' in failed_names and 'C_suppress_tokens_none' in failed_names:
        print("  H5 (suppress_tokens): RULED OUT")

    # H3: beam_size
    if 'D_beam_size_1' in passed_names:
        print("  H3 (beam_size=3): CONFIRMED - greedy decoding works")
        if 'E_beam_size_5' in passed_names:
            print("      -> beam_size=3 specifically triggers the bug")
        elif 'E_beam_size_5' in failed_names:
            print("      -> Any beam search (not just size=3) triggers the bug")
    elif 'D_beam_size_1' in failed_names:
        print("  H3 (beam_size): RULED OUT - crash happens even with greedy")

    # H2: no_speech_threshold
    if 'F_no_speech_0.6' in passed_names:
        print("  H2 (no_speech_threshold=0.22): CONFIRMED as cause")
        print("      -> FIX: Use higher threshold (0.6 default works)")
    elif 'F_no_speech_0.6' in failed_names:
        print("  H2 (no_speech_threshold): RULED OUT")

    # Word timestamps
    if 'G_word_timestamps_off' in passed_names:
        print("  Word timestamps: CONFIRMED as cause")
        print("      -> FIX: Disable word_timestamps or investigate alignment code")
    elif 'G_word_timestamps_off' in failed_names:
        print("  Word timestamps: RULED OUT")

    # Clean params
    if 'H_clean_params' in passed_names:
        print("\n  Combined fix (remove vad_params + suppress_tokens): WORKS")

    print("\n" + "="*70)

    # Save JSON results
    json_results = {
        'timestamp': run_timestamp,
        'test_file': str(TEST_FILE),
        'model': MODEL_SIZE,
        'baseline_params': {k: (str(v) if v is REMOVE_PARAM else v) for k, v in BASELINE_PARAMS.items()},
        'tests_run': list(tests_to_run.keys()),
        'results': [],
        'summary': {
            'passed': passed_names,
            'failed': failed_names,
        }
    }

    for r in results:
        # Convert for JSON serialization
        json_r = {
            'test_name': r['test_name'],
            'status': r['status'],
            'segments': r.get('segments', 0),
            'duration': r.get('duration', 0),
            'error': r.get('error'),
        }
        # Handle REMOVE_PARAM in overrides
        if 'overrides' in r:
            json_r['overrides'] = {
                k: 'REMOVED' if v is REMOVE_PARAM else v
                for k, v in r['overrides'].items()
            }
        json_results['results'].append(json_r)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to:")
    print(f"  Log: {OUTPUT_LOG}")
    print(f"  JSON: {OUTPUT_JSON}")

    # Close tee output
    tee.close()
    sys.stdout = tee.terminal


def run_single_test_mode(test_name: str):
    """Entry point for running a single test (called from subprocess)."""
    if test_name not in TEST_VARIATIONS:
        print(f"ERROR: Unknown test: {test_name}")
        print(f"Available: {list(TEST_VARIATIONS.keys())}")
        sys.exit(1)

    overrides = TEST_VARIATIONS[test_name]
    result = run_test(test_name, overrides)

    # Exit with appropriate code
    sys.exit(0 if result['status'] == 'PASSED' else 1)


if __name__ == "__main__":
    # Check for --single mode (subprocess entry point)
    if '--single' in sys.argv:
        idx = sys.argv.index('--single')
        if idx + 1 < len(sys.argv):
            run_single_test_mode(sys.argv[idx + 1])
        else:
            print("ERROR: --single requires a test name")
            sys.exit(1)
    else:
        main()
