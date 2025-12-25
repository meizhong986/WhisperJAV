@echo off
REM Hypothesis Testing Quick Start Script (Windows)
REM
REM This script guides you through running hypothesis tests with minimal setup.
REM
REM Usage: run_hypothesis_quickstart.bat

setlocal enabledelayedexpansion

echo ========================================================================
echo HYPOTHESIS TESTING SUITE - QUICK START
echo ========================================================================
echo.

REM Check if running in correct directory
if not exist "hypothesis_test_suite.py" (
    echo ERROR: This script must be run from the tests\ directory
    echo Please run: cd tests ^&^& run_hypothesis_quickstart.bat
    exit /b 1
)

REM Step 1: Validate setup
echo Step 1: Validating setup...
echo ------------------------------------------------------------------------
python validate_hypothesis_suite.py
if errorlevel 1 (
    echo.
    echo ERROR: Validation failed. Please fix issues above before proceeding.
    exit /b 1
)

REM Step 2: Check for test audio
echo.
echo Step 2: Checking for test audio...
echo ------------------------------------------------------------------------

set AUDIO_FILE=
if exist "subset.wav" (
    echo Found: subset.wav
    set AUDIO_FILE=subset.wav
) else if exist "..\subset.wav" (
    echo Found: ..\subset.wav
    set AUDIO_FILE=..\subset.wav
) else (
    echo No test audio found.
    echo.
    echo Please prepare test audio using ffmpeg:
    echo.
    echo   ffmpeg -i video.mp4 -ss 01:52:00 -t 00:25:00 -vn -acodec pcm_s16le -ar 16000 -ac 1 subset.wav
    echo.
    echo Then run this script again.
    exit /b 1
)

REM Step 3: Check for reference SRT (optional)
echo.
echo Step 3: Checking for reference SRT (optional)...
echo ------------------------------------------------------------------------

set REFERENCE_ARG=
if exist "v1.7.1_subset.srt" (
    echo Found: v1.7.1_subset.srt
    set REFERENCE_ARG=--reference v1.7.1_subset.srt
) else if exist "reference.srt" (
    echo Found: reference.srt
    set REFERENCE_ARG=--reference reference.srt
) else (
    echo No reference SRT found (this is optional^).
)

REM Step 4: Choose test mode
echo.
echo Step 4: Choose test mode...
echo ------------------------------------------------------------------------
echo 1^) Quick mode (5 tests, ~15 minutes^)
echo 2^) Specific hypothesis
echo 3^) Full suite (18 tests, ~54 minutes^)
echo.
set /p choice="Enter choice (1-3): "

set MODE_ARG=
if "%choice%"=="1" (
    echo Selected: Quick mode
    set MODE_ARG=--quick
) else if "%choice%"=="2" (
    echo.
    echo Available hypotheses:
    echo   1^) vad_params
    echo   2^) asr_duration_filter
    echo   3^) temperature_fallback
    echo   4^) patience_beam
    echo.
    set /p hyp_choice="Enter hypothesis number (1-4): "
    if "!hyp_choice!"=="1" set MODE_ARG=--hypothesis vad_params
    if "!hyp_choice!"=="2" set MODE_ARG=--hypothesis asr_duration_filter
    if "!hyp_choice!"=="3" set MODE_ARG=--hypothesis temperature_fallback
    if "!hyp_choice!"=="4" set MODE_ARG=--hypothesis patience_beam
    if "!MODE_ARG!"=="" (
        echo Invalid choice. Using full suite.
        set MODE_ARG=
    )
) else if "%choice%"=="3" (
    echo Selected: Full suite
    set MODE_ARG=
) else (
    echo Invalid choice. Using quick mode.
    set MODE_ARG=--quick
)

REM Step 5: Run tests
echo.
echo Step 5: Running tests...
echo ------------------------------------------------------------------------
echo Command: python hypothesis_test_suite.py --audio %AUDIO_FILE% %REFERENCE_ARG% %MODE_ARG%
echo.

python hypothesis_test_suite.py --audio "%AUDIO_FILE%" %REFERENCE_ARG% %MODE_ARG%

REM Step 6: Show results
echo.
echo ========================================================================
echo TESTING COMPLETE
echo ========================================================================
echo.
echo Results saved to:
echo   - hypothesis_results.json (detailed results^)
echo   - hypothesis_outputs\*.srt (generated subtitles^)
echo.
echo Next steps:
echo   1. Review the summary table above
echo   2. Check hypothesis_results.json for detailed metrics
echo   3. Compare winning configs manually
echo   4. Run full video with best config
echo.

pause
