@echo off
REM ============================================================================
REM DIAGNOSTIC: Parameter Variation Tests for Crash Investigation
REM ============================================================================
REM
REM Tests faster-whisper directly with varying parameters to identify which
REM parameter causes the STATUS_INTEGER_DIVIDE_BY_ZERO crash.
REM
REM Hypotheses under test:
REM   H2: no_speech_threshold=0.22 causes edge case
REM   H3: beam_size=3 with word_timestamps creates vulnerable code path
REM   H4: vad_filter=False with vad_parameters dict present causes issue
REM   H5: Empty suppress_tokens=[] causes edge case in ctranslate2
REM
REM Usage:
REM   diagnostic_param_variation_test.bat              - Run all safe tests (subprocess mode)
REM   diagnostic_param_variation_test.bat -b           - Include baseline crash test
REM   diagnostic_param_variation_test.bat A_no_vad     - Run only matching test(s)
REM   diagnostic_param_variation_test.bat beam         - Run beam_size tests
REM   diagnostic_param_variation_test.bat --no-subprocess - Run in single process (faster)
REM
REM Test cases:
REM   A_no_vad_params      - Remove vad_parameters dict (H4)
REM   B_no_suppress_tokens - Remove empty suppress_tokens (H5)
REM   C_suppress_tokens_none - Set suppress_tokens to None (H5)
REM   D_beam_size_1        - Greedy decoding (H3)
REM   E_beam_size_5        - Different beam size (H3)
REM   F_no_speech_0.6      - Higher threshold (H2)
REM   G_word_timestamps_off - Disable word timestamps
REM   H_clean_params       - Remove both vad_params and suppress_tokens
REM   Z_baseline_exact     - Exact crash params (use -b flag)
REM
REM ============================================================================

echo.
echo ============================================================================
echo  DIAGNOSTIC: Parameter Variation Tests for Crash Investigation
echo ============================================================================
echo.
echo  Tests faster-whisper directly with varying parameters.
echo  Each test that PASSES identifies a parameter change that prevents the crash.
echo.

cd /d %~dp0\..

REM Run the Python test script with all arguments passed through
python tests\diagnostic_param_variation_test.py %*

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================================================
echo  Tests complete. Exit code: %EXIT_CODE%
echo ============================================================================

if %EXIT_CODE% neq 0 (
    echo.
    echo  NOTE: Non-zero exit code may indicate a crash occurred.
    echo  This is EXPECTED if testing with --baseline flag.
)

echo.
pause
