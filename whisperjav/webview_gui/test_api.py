"""
Standalone API Test Script

Tests the WhisperJAV API class without GUI.
Verifies argument building, process lifecycle, and log streaming.

Usage:
    python -m whisperjav.webview_gui.test_api
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from whisperjav.webview_gui.api import WhisperJAVAPI, DEFAULT_OUTPUT


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_initialization():
    """Test API initialization."""
    print_section("Test 1: API Initialization")

    api = WhisperJAVAPI()

    assert api.status == "idle", "Initial status should be idle"
    assert api.process is None, "Process should be None initially"
    assert api.exit_code is None, "Exit code should be None initially"
    assert api.default_output == str(DEFAULT_OUTPUT), "Default output should be set"

    print("[PASS] API initialized successfully")
    print(f"  - Status: {api.status}")
    print(f"  - Default output: {api.default_output}")

    return api


def test_build_args():
    """Test CLI argument construction."""
    print_section("Test 2: Build Arguments")

    api = WhisperJAVAPI()

    # Test basic options
    print("\n[Test 2a] Basic options:")
    options = {
        'inputs': ['test.mp4'],
        'mode': 'balanced',
        'sensitivity': 'aggressive',
        'language': 'japanese',
        'output_dir': '/tmp/output',
    }
    args = api.build_args(options)
    print(f"  Input options: {options}")
    print(f"  Generated args: {args}")

    # Verify required arguments
    assert 'test.mp4' in args, "Input file should be in args"
    assert '--mode' in args, "--mode flag should be present"
    assert 'balanced' in args, "Mode value should be present"
    assert '--sensitivity' in args, "--sensitivity flag should be present"
    assert 'aggressive' in args, "Sensitivity value should be present"
    assert '--subs-language' in args, "--subs-language flag should be present"
    assert 'japanese' in args, "Language value should be present"
    assert '--output-dir' in args, "--output-dir flag should be present"
    assert '/tmp/output' in args, "Output directory should be present"

    print("  [PASS] Basic arguments built correctly")

    # Test advanced options
    print("\n[Test 2b] Advanced options:")
    options = {
        'inputs': ['video1.mp4', 'video2.mkv'],
        'mode': 'faster',
        'sensitivity': 'conservative',
        'language': 'japanese',
        'output_dir': '/tmp/output',
        'temp_dir': '/tmp/whisperjav',
        'keep_temp': True,
        'verbosity': 'verbose',
        'async_processing': True,
        'no_vad': True,
        'model_override': 'large-v3',
        'credit': 'Produced by Test Studio',
    }
    args = api.build_args(options)
    print(f"  Generated args: {args}")

    assert 'video1.mp4' in args, "First input should be in args"
    assert 'video2.mkv' in args, "Second input should be in args"
    assert '--temp-dir' in args, "--temp-dir flag should be present"
    assert '/tmp/whisperjav' in args, "Temp dir should be present"
    assert '--keep-temp' in args, "--keep-temp flag should be present"
    assert '--verbosity' in args, "--verbosity flag should be present"
    assert 'verbose' in args, "Verbosity value should be present"
    assert '--async-processing' in args, "--async-processing flag should be present"
    assert '--no-vad' in args, "--no-vad flag should be present"
    assert '--model' in args, "--model flag should be present"
    assert 'large-v3' in args, "Model value should be present"
    assert '--credit' in args, "--credit flag should be present"
    assert 'Produced by Test Studio' in args, "Credit text should be present"

    print("  [PASS] Advanced arguments built correctly")

    # Test check_only mode
    print("\n[Test 2c] Check-only mode:")
    options = {
        'check_only': True
    }
    args = api.build_args(options)
    print(f"  Generated args: {args}")

    assert '--check' in args, "--check flag should be present"
    assert '--check-verbose' in args, "--check-verbose flag should be present"

    print("  [PASS] Check-only arguments built correctly")

    # Test missing inputs error
    print("\n[Test 2d] Missing inputs error:")
    options = {
        'mode': 'balanced'
    }
    try:
        args = api.build_args(options)
        print("  [FAIL] Should have raised ValueError for missing inputs")
        sys.exit(1)
    except ValueError as e:
        print(f"  [PASS] Correctly raised ValueError: {e}")


def test_process_status():
    """Test process status checking."""
    print_section("Test 3: Process Status")

    api = WhisperJAVAPI()

    status = api.get_process_status()
    print(f"  Initial status: {status}")

    assert status['status'] == 'idle', "Status should be idle"
    assert status['exit_code'] is None, "Exit code should be None"
    assert status['has_logs'] is False, "Should have no logs"

    print("  [PASS] Process status check works")


def test_log_queue():
    """Test log queue functionality."""
    print_section("Test 4: Log Queue")

    api = WhisperJAVAPI()

    # Add some test logs
    api.log_queue.put("Log line 1\n")
    api.log_queue.put("Log line 2\n")
    api.log_queue.put("Log line 3\n")

    # Retrieve logs
    logs = api.get_logs()
    print(f"  Retrieved {len(logs)} log lines:")
    for i, log in enumerate(logs, 1):
        print(f"    {i}. {log.strip()}")

    assert len(logs) == 3, "Should retrieve 3 log lines"
    assert logs[0] == "Log line 1\n", "First log should match"
    assert logs[1] == "Log line 2\n", "Second log should match"
    assert logs[2] == "Log line 3\n", "Third log should match"

    # Queue should be empty now
    logs = api.get_logs()
    assert len(logs) == 0, "Queue should be empty after retrieval"

    print("  [PASS] Log queue works correctly")


def test_file_dialog_simulation():
    """Test file dialog methods (without actual dialogs)."""
    print_section("Test 5: File Dialog Methods")

    api = WhisperJAVAPI()

    # Note: These will fail without an active window, but we can verify
    # they don't crash and return expected error structure

    print("\n[Test 5a] select_files (no window):")
    result = api.select_files()
    print(f"  Result: {result}")
    assert 'success' in result, "Result should have success key"
    assert result['success'] is False, "Should fail without window"
    assert 'message' in result, "Should have error message"
    print("  [PASS] Returns proper error structure")

    print("\n[Test 5b] select_folder (no window):")
    result = api.select_folder()
    print(f"  Result: {result}")
    assert 'success' in result, "Result should have success key"
    assert result['success'] is False, "Should fail without window"
    print("  [PASS] Returns proper error structure")

    print("\n[Test 5c] select_output_directory (no window):")
    result = api.select_output_directory()
    print(f"  Result: {result}")
    assert 'success' in result, "Result should have success key"
    assert result['success'] is False, "Should fail without window"
    print("  [PASS] Returns proper error structure")


def test_default_output():
    """Test default output directory."""
    print_section("Test 6: Default Output Directory")

    api = WhisperJAVAPI()

    default_dir = api.get_default_output_dir()
    print(f"  Default output directory: {default_dir}")

    assert isinstance(default_dir, str), "Should return string"
    assert len(default_dir) > 0, "Should not be empty"
    assert "WhisperJAV" in default_dir, "Should contain WhisperJAV"

    print("  [PASS] Default output directory set correctly")


def test_hello_world():
    """Test backward compatibility with Phase 1."""
    print_section("Test 7: Phase 1 Compatibility (hello_world)")

    api = WhisperJAVAPI()

    result = api.hello_world()
    print(f"  Result: {result}")

    assert result['success'] is True, "Should succeed"
    assert 'message' in result, "Should have message"
    assert 'timestamp' in result, "Should have timestamp"
    assert 'api_version' in result, "Should have API version"
    assert "Phase 2" in result['api_version'], "Should indicate Phase 2"

    print("  [PASS] Phase 1 methods still work")


def test_cancel_without_process():
    """Test cancel behavior when no process is running."""
    print_section("Test 8: Cancel Without Process")

    api = WhisperJAVAPI()

    result = api.cancel_process()
    print(f"  Result: {result}")

    assert result['success'] is False, "Should fail when no process"
    assert 'message' in result, "Should have error message"

    print("  [PASS] Cancel handles no-process case correctly")


def test_start_without_inputs():
    """Test start behavior with invalid options."""
    print_section("Test 9: Start Without Inputs")

    api = WhisperJAVAPI()

    # Try to start without inputs
    result = api.start_process({
        'mode': 'balanced'
    })
    print(f"  Result: {result}")

    assert result['success'] is False, "Should fail without inputs"
    assert 'message' in result, "Should have error message"
    assert "at least one file" in result['message'].lower(), "Error should mention missing files"

    print("  [PASS] Start validates inputs correctly")


def run_all_tests():
    """Run all test functions."""
    print("\n")
    print("=" * 70)
    print(" " * 15 + "WhisperJAV API Standalone Tests")
    print("=" * 70)

    tests = [
        test_initialization,
        test_build_args,
        test_process_status,
        test_log_queue,
        test_file_dialog_simulation,
        test_default_output,
        test_hello_world,
        test_cancel_without_process,
        test_start_without_inputs,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[ERROR] {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n")
    print("=" * 70)
    print(" " * 25 + "Test Summary")
    print("=" * 70)
    print(f"  Total tests:  {len(tests)}")
    print(f"  Passed:       {passed}")
    print(f"  Failed:       {failed}")
    print("=" * 70)

    if failed == 0:
        print("\n[SUCCESS] All tests passed! API is ready for Phase 3 (Frontend Integration).")
        return 0
    else:
        print(f"\n[FAILURE] {failed} test(s) failed. Please fix issues before proceeding.")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
