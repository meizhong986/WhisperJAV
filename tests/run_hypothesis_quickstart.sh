#!/bin/bash
# Hypothesis Testing Quick Start Script
#
# This script guides you through running hypothesis tests with minimal setup.
#
# Usage:
#   bash run_hypothesis_quickstart.sh

set -e  # Exit on error

echo "========================================================================"
echo "HYPOTHESIS TESTING SUITE - QUICK START"
echo "========================================================================"
echo ""

# Check if running in correct directory
if [ ! -f "hypothesis_test_suite.py" ]; then
    echo "ERROR: This script must be run from the tests/ directory"
    echo "Please run: cd tests && bash run_hypothesis_quickstart.sh"
    exit 1
fi

# Step 1: Validate setup
echo "Step 1: Validating setup..."
echo "------------------------------------------------------------------------"
python validate_hypothesis_suite.py
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Validation failed. Please fix issues above before proceeding."
    exit 1
fi

# Step 2: Check for test audio
echo ""
echo "Step 2: Checking for test audio..."
echo "------------------------------------------------------------------------"

if [ -f "subset.wav" ]; then
    echo "Found: subset.wav"
    AUDIO_FILE="subset.wav"
elif [ -f "../subset.wav" ]; then
    echo "Found: ../subset.wav"
    AUDIO_FILE="../subset.wav"
else
    echo "No test audio found."
    echo ""
    echo "Please prepare test audio using ffmpeg:"
    echo ""
    echo "  ffmpeg -i video.mp4 -ss 01:52:00 -t 00:25:00 -vn -acodec pcm_s16le -ar 16000 -ac 1 subset.wav"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Step 3: Check for reference SRT (optional)
echo ""
echo "Step 3: Checking for reference SRT (optional)..."
echo "------------------------------------------------------------------------"

REFERENCE_ARG=""
if [ -f "v1.7.1_subset.srt" ]; then
    echo "Found: v1.7.1_subset.srt"
    REFERENCE_ARG="--reference v1.7.1_subset.srt"
elif [ -f "reference.srt" ]; then
    echo "Found: reference.srt"
    REFERENCE_ARG="--reference reference.srt"
else
    echo "No reference SRT found (this is optional)."
fi

# Step 4: Choose test mode
echo ""
echo "Step 4: Choose test mode..."
echo "------------------------------------------------------------------------"
echo "1) Quick mode (5 tests, ~15 minutes)"
echo "2) Specific hypothesis"
echo "3) Full suite (18 tests, ~54 minutes)"
echo ""
read -p "Enter choice (1-3): " choice

MODE_ARG=""
case $choice in
    1)
        echo "Selected: Quick mode"
        MODE_ARG="--quick"
        ;;
    2)
        echo ""
        echo "Available hypotheses:"
        echo "  1) vad_params"
        echo "  2) asr_duration_filter"
        echo "  3) temperature_fallback"
        echo "  4) patience_beam"
        echo ""
        read -p "Enter hypothesis number (1-4): " hyp_choice
        case $hyp_choice in
            1) MODE_ARG="--hypothesis vad_params" ;;
            2) MODE_ARG="--hypothesis asr_duration_filter" ;;
            3) MODE_ARG="--hypothesis temperature_fallback" ;;
            4) MODE_ARG="--hypothesis patience_beam" ;;
            *) echo "Invalid choice. Using full suite."; MODE_ARG="" ;;
        esac
        ;;
    3)
        echo "Selected: Full suite"
        MODE_ARG=""
        ;;
    *)
        echo "Invalid choice. Using quick mode."
        MODE_ARG="--quick"
        ;;
esac

# Step 5: Run tests
echo ""
echo "Step 5: Running tests..."
echo "------------------------------------------------------------------------"
echo "Command: python hypothesis_test_suite.py --audio $AUDIO_FILE $REFERENCE_ARG $MODE_ARG"
echo ""

python hypothesis_test_suite.py --audio "$AUDIO_FILE" $REFERENCE_ARG $MODE_ARG

# Step 6: Show results
echo ""
echo "========================================================================"
echo "TESTING COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - hypothesis_results.json (detailed results)"
echo "  - hypothesis_outputs/*.srt (generated subtitles)"
echo ""
echo "Next steps:"
echo "  1. Review the summary table above"
echo "  2. Check hypothesis_results.json for detailed metrics"
echo "  3. Compare winning configs manually"
echo "  4. Run full video with best config"
echo ""
