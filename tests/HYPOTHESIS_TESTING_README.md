# Hypothesis Testing Suite for ASR Regression Investigation

## Overview

This automated testing suite investigates why v1.7.3 produces ~20% fewer subtitles than v1.7.1. It tests specific parameter hypotheses in isolation against a test audio subset.

## Background

**Forensic Analysis Findings:**
- 128 missing subtitles (657 → 529) in v1.7.3
- No large continuous gaps (rules out VAD over-merging)
- 95 small scattered gaps throughout
- Short segments (<1s) have 2.6x higher loss rate
- Opening segment has 95% loss rate (separate issue)

## Files

- **`hypothesis_test_suite.py`**: Main test orchestrator
- **`hypothesis_configs.py`**: Parameter configuration definitions
- **`HYPOTHESIS_TESTING_README.md`**: This documentation

## Setup

### 1. Prepare Test Audio

Extract a 25-minute subset from your test video:

```bash
ffmpeg -i video.mp4 -ss 01:52:00 -t 00:25:00 -vn -acodec pcm_s16le -ar 16000 -ac 1 subset.wav
```

This extracts audio from 1h52m to 2h17m (the time range with most missing subtitles).

### 2. Prepare Reference SRT (Optional but Recommended)

If you have a v1.7.1 SRT for the same time range:

```bash
# Extract subtitles for the same time range
# Use subtitle editor or script to extract entries from 1:52:00 to 2:17:00
```

## Usage

### Run All Hypotheses

```bash
python tests/hypothesis_test_suite.py \
    --audio subset.wav \
    --reference v1.7.1_subset.srt
```

### Run Specific Hypothesis

```bash
python tests/hypothesis_test_suite.py \
    --audio subset.wav \
    --hypothesis vad_params
```

Available hypotheses:
- `vad_params`: VAD parameter wiring tests
- `asr_duration_filter`: ASR duration filtering tests
- `temperature_fallback`: Temperature fallback configuration tests
- `patience_beam`: Beam search parameter tests
- `all`: Run all hypotheses (default)

### Quick Mode

Run a smaller subset of key tests:

```bash
python tests/hypothesis_test_suite.py \
    --audio subset.wav \
    --quick
```

### Skip Baseline

If you've already run the baseline test and want to continue:

```bash
python tests/hypothesis_test_suite.py \
    --audio subset.wav \
    --skip-baseline \
    --hypothesis temperature_fallback
```

### Custom Output Locations

```bash
python tests/hypothesis_test_suite.py \
    --audio subset.wav \
    --output-dir ./my_outputs \
    --temp-dir ./my_temp \
    --results-file ./my_results.json
```

## Hypotheses Tested

### Hypothesis 1: VAD Parameter Wiring

**Theory**: `speech_pad_ms` and `min_silence_duration_ms` may not be properly applied to Silero VAD.

**Tests**:
- `vad_zero_speech_pad`: speech_pad_ms = 0
- `vad_zero_min_silence`: min_silence_duration_ms = 0
- `vad_both_zeros`: Both set to 0 (maximum sensitivity)
- `vad_v171_padding`: speech_pad_ms = 700 (potential v1.7.1 default)

**Expected Outcome**: If wiring is broken, changing these parameters will have no effect.

### Hypothesis 2: ASR Duration Filtering

**Theory**: `no_speech_threshold` and `logprob_threshold` are filtering out short segments.

**Tests**:
- `asr_no_speech_default`: no_speech_threshold = 0.5 (balanced)
- `asr_no_speech_conservative`: no_speech_threshold = 0.6
- `asr_logprob_balanced`: logprob_threshold = -1.2 (balanced)
- `asr_logprob_conservative`: logprob_threshold = -1.0
- `asr_both_balanced`: Both set to balanced defaults

**Expected Outcome**: More permissive thresholds should recover missing short segments.

### Hypothesis 3: Temperature Fallback

**Theory**: Temperature fallback configuration affects segment detection.

**Tests**:
- `temp_no_fallback`: temperature = [0.0] (no fallback)
- `temp_balanced_fallback`: temperature = [0.0, 0.1]
- `temp_multi_fallback`: temperature = [0.0, 0.2, 0.4, 0.6]

**Expected Outcome**: Different fallback strategies may affect edge case handling.

### Hypothesis 4: Patience/Beam Interaction

**Theory**: Beam search parameters affect decoding quality.

**Tests**:
- `beam_patience_low`: patience = 1.6 (aggressive)
- `beam_patience_balanced`: patience = 2.0 (balanced)
- `beam_patience_high`: patience = 2.9 (conservative)
- `beam_size_small`: beam_size = 2 (balanced)
- `beam_balanced_combo`: beam_size = 2, patience = 2.0

**Expected Outcome**: More thorough beam search may recover missed segments.

## Output Files

### SRT Files

Each test produces an SRT file in the output directory:

```
hypothesis_outputs/
├── v1.7.3_baseline.srt
├── vad_zero_speech_pad.srt
├── vad_zero_min_silence.srt
├── asr_no_speech_default.srt
└── ...
```

### JSON Results

The suite generates a comprehensive JSON results file:

```json
{
  "audio_file": "subset.wav",
  "reference_srt": "v1.7.1_subset.srt",
  "total_tests": 15,
  "successful_tests": 15,
  "failed_tests": 0,
  "baseline": {
    "config_name": "v1.7.3_baseline",
    "total_subtitles": 55,
    "total_speech_duration_sec": 287.3,
    "avg_subtitle_duration_sec": 5.22
  },
  "reference_metrics": {
    "total_count": 68,
    "total_duration_sec": 356.1
  },
  "results": [
    {
      "config_name": "vad_both_zeros",
      "total_subtitles": 64,
      "comparison": {
        "subtitle_delta_vs_baseline": 9,
        "improvement_vs_baseline_pct": 16.4
      }
    }
  ],
  "recommendations": [
    "Found 5 configurations that improved over baseline:",
    "  1. vad_both_zeros: +9 subtitles (+16.4%) - vad_params"
  ]
}
```

### Console Output

The suite prints a summary table:

```
======================================================================================================
HYPOTHESIS TEST SUMMARY
======================================================================================================
Config                         Hypothesis            Subs   Duration       Avg   <1s   >5s
------------------------------------------------------------------------------------------------------
v1.7.3_baseline                baseline                55      287.3s    5.22s    12     8
------------------------------------------------------------------------------------------------------
vad_zero_speech_pad            vad_params              58 (+3)  295.1s    5.09s    14     7
vad_both_zeros                 vad_params              64 (+9)  324.5s    5.07s    16     8
asr_both_balanced              asr_duration_filter     61 (+6)  310.2s    5.08s    15     7
...
======================================================================================================

Reference (v1.7.1): 68 subtitles, 356.1s total speech

RECOMMENDATIONS:
------------------------------------------------------------------------------------------------------
Found 3 configurations that improved over baseline:
  1. vad_both_zeros: +9 subtitles (+16.4%) - vad_params
  2. asr_both_balanced: +6 subtitles (+10.9%) - asr_duration_filter
  3. vad_zero_speech_pad: +3 subtitles (+5.5%) - vad_params

Most impactful hypothesis: vad_params (max improvement: +9 subtitles)
```

## Interpreting Results

### Metrics Explained

- **Total Subtitles**: Number of subtitle entries generated
- **Duration**: Total speech time detected (sum of all subtitle durations)
- **Avg**: Average subtitle duration
- **<1s**: Count of very short subtitles (may indicate over-segmentation)
- **>5s**: Count of long subtitles (may indicate under-segmentation)

### What to Look For

1. **Large improvements** (+10+ subtitles): Strong evidence for the hypothesis
2. **Consistent improvements** within a hypothesis: Parameter is likely impactful
3. **No improvement**: Parameter may not be the root cause
4. **Negative impact**: Parameter makes things worse

### Next Steps

1. **Identify best configuration**: Look at top recommendations
2. **Validate with full video**: Re-run winning config on full video
3. **Compare SRT files**: Manual review of recovered subtitles
4. **Code investigation**: If hypothesis confirmed, investigate why parameter has this effect

## Modular Design

### Adding New Hypotheses

Edit `hypothesis_configs.py`:

```python
@staticmethod
def get_my_new_hypothesis_tests() -> List[TestConfig]:
    """Test my new hypothesis."""
    configs = []

    configs.append(TestConfig(
        name="my_test_variant",
        description="Description of what this tests",
        hypothesis="my_hypothesis",
        params_override={
            "vad": {"some_param": 123},
            "asr": {"other_param": "value"}
        }
    ))

    return configs
```

Then add to `get_all_hypotheses()`:

```python
return {
    "baseline": [HypothesisConfigs.get_baseline()],
    "vad_params": HypothesisConfigs.get_vad_parameter_tests(),
    "my_hypothesis": HypothesisConfigs.get_my_new_hypothesis_tests(),  # Add this
}
```

## Performance Notes

- Each test processes the full 25-minute audio
- Estimated time: 2-5 minutes per test (depending on GPU)
- Quick mode: ~5-6 tests (~15-30 minutes total)
- Full mode: ~20 tests (~1-2 hours total)

## Troubleshooting

### "Audio file not found"

Ensure the audio file path is correct:

```bash
ls -lh subset.wav
```

### "Test failed: out of memory"

Reduce audio length or use CPU mode:

```bash
# Extract shorter subset (10 minutes)
ffmpeg -i video.mp4 -ss 01:52:00 -t 00:10:00 -vn -acodec pcm_s16le subset_short.wav
```

### "No results to display"

Check that tests completed successfully. Review console output for errors.

## Example Workflow

```bash
# 1. Extract test audio
ffmpeg -i JAV-123.mp4 -ss 01:52:00 -t 00:25:00 -vn -acodec pcm_s16le -ar 16000 -ac 1 subset.wav

# 2. Run quick test first to validate setup
python tests/hypothesis_test_suite.py --audio subset.wav --quick

# 3. If quick test works, run full suite
python tests/hypothesis_test_suite.py --audio subset.wav --reference v1.7.1_subset.srt

# 4. Review results
cat hypothesis_results.json

# 5. Test winning hypothesis on full video
whisperjav JAV-123.mp4 --mode balanced --sensitivity aggressive \
    --custom-params '{"vad": {"speech_pad_ms": 0, "min_silence_duration_ms": 0}}'
```

## Credits

Created by Claude Code for automated ASR regression investigation.
Based on forensic analysis by investigative agents.

## License

Same as WhisperJAV main project.
