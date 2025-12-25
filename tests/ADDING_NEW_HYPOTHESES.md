# Adding New Hypotheses to the Test Suite

This guide shows how to add new parameter hypotheses to the test suite.

## Example: Adding a "Repetition Penalty" Hypothesis

Let's say you want to test if the `repetition_penalty` parameter affects subtitle count.

### Step 1: Define Test Configurations

Edit `hypothesis_configs.py` and add a new method:

```python
@staticmethod
def get_repetition_penalty_tests() -> List[TestConfig]:
    """
    Hypothesis 5: Repetition Penalty

    Test if repetition penalty affects segment detection.
    """
    configs = []

    # Test 5a: No repetition penalty
    configs.append(TestConfig(
        name="rep_penalty_none",
        description="repetition_penalty = 1.0 (no penalty)",
        hypothesis="repetition_penalty",
        params_override={
            "asr": {"repetition_penalty": 1.0}
        }
    ))

    # Test 5b: Balanced penalty (v1.7.1 default)
    configs.append(TestConfig(
        name="rep_penalty_balanced",
        description="repetition_penalty = 1.5 (balanced)",
        hypothesis="repetition_penalty",
        params_override={
            "asr": {"repetition_penalty": 1.5}
        }
    ))

    # Test 5c: High penalty
    configs.append(TestConfig(
        name="rep_penalty_high",
        description="repetition_penalty = 2.0 (high)",
        hypothesis="repetition_penalty",
        params_override={
            "asr": {"repetition_penalty": 2.0}
        }
    ))

    return configs
```

### Step 2: Register in get_all_hypotheses()

Still in `hypothesis_configs.py`, update the `get_all_hypotheses()` method:

```python
@staticmethod
def get_all_hypotheses() -> Dict[str, List[TestConfig]]:
    """Get all hypothesis test configurations organized by hypothesis."""
    return {
        "baseline": [HypothesisConfigs.get_baseline()],
        "vad_params": HypothesisConfigs.get_vad_parameter_tests(),
        "asr_duration_filter": HypothesisConfigs.get_asr_duration_filter_tests(),
        "temperature_fallback": HypothesisConfigs.get_temperature_fallback_tests(),
        "patience_beam": HypothesisConfigs.get_patience_beam_tests(),
        "repetition_penalty": HypothesisConfigs.get_repetition_penalty_tests(),  # ADD THIS
    }
```

### Step 3: Update CLI Choices (Optional)

If you want the new hypothesis to be selectable via `--hypothesis`, edit `hypothesis_test_suite.py`:

```python
parser.add_argument(
    "--hypothesis",
    choices=["vad_params", "asr_duration_filter", "temperature_fallback",
             "patience_beam", "repetition_penalty", "all"],  # ADD YOUR HYPOTHESIS
    default="all",
    help="Specific hypothesis to test (default: all)"
)
```

### Step 4: Test Your New Hypothesis

```bash
# Validate setup
python validate_hypothesis_suite.py

# Inspect new configurations
python inspect_hypothesis_configs.py --hypothesis repetition_penalty

# Run the test
python hypothesis_test_suite.py --audio subset.wav --hypothesis repetition_penalty
```

## Parameter Mapping Reference

When defining `params_override`, you need to know which section parameters belong to:

### ASR Parameters (`params_override: {"asr": {...}}`)

These go into either `params.decoder` or `params.provider`:

**Decoder parameters** (beam search, sampling):
- `beam_size`
- `patience`
- `best_of`
- `length_penalty`
- `suppress_tokens`
- `suppress_blank`
- `max_initial_timestamp`

**Provider parameters** (transcription control):
- `temperature`
- `compression_ratio_threshold`
- `logprob_threshold`
- `no_speech_threshold`
- `condition_on_previous_text`
- `initial_prompt`
- `word_timestamps`
- `repetition_penalty`
- `no_repeat_ngram_size`
- `hallucination_silence_threshold`
- `chunk_length`

### VAD Parameters (`params_override: {"vad": {...}}`)

These go into `params.vad`:

- `threshold`: Speech detection threshold
- `min_speech_duration_ms`: Minimum speech duration
- `max_speech_duration_s`: Maximum speech duration
- `min_silence_duration_ms`: Minimum silence duration
- `neg_threshold`: Negative speech threshold
- `speech_pad_ms`: Padding around speech
- `chunk_threshold_s`: Gap threshold for grouping

### Backend Selection

To test different backends:

```python
params_override={
    "speech_segmenter": {"backend": "silero-v3.1"}  # or "silero-v4.0", "nemo-lite", etc.
}
```

## Example: Testing Multiple Parameters

You can override multiple parameters in a single test:

```python
TestConfig(
    name="combined_relaxed",
    description="Multiple relaxed parameters",
    hypothesis="combined_relaxed",
    params_override={
        "asr": {
            "no_speech_threshold": 0.3,
            "logprob_threshold": -2.0,
            "patience": 2.5
        },
        "vad": {
            "threshold": 0.15,
            "speech_pad_ms": 1000
        }
    }
)
```

## Quick Test Template

Copy and customize this template for quick hypothesis testing:

```python
@staticmethod
def get_my_hypothesis_tests() -> List[TestConfig]:
    """
    Hypothesis X: [Your hypothesis description]

    Test if [parameter] affects [expected outcome].
    """
    configs = []

    configs.append(TestConfig(
        name="my_test_variant1",
        description="[What this variant tests]",
        hypothesis="my_hypothesis",
        params_override={
            "asr": {"parameter_name": value1},
            # or
            "vad": {"parameter_name": value1}
        }
    ))

    configs.append(TestConfig(
        name="my_test_variant2",
        description="[What this variant tests]",
        hypothesis="my_hypothesis",
        params_override={
            "asr": {"parameter_name": value2}
        }
    ))

    return configs
```

## Validation Checklist

Before running your new hypothesis:

- [ ] Configuration compiles without syntax errors: `python -m py_compile hypothesis_configs.py`
- [ ] Validation passes: `python validate_hypothesis_suite.py`
- [ ] Can inspect configs: `python inspect_hypothesis_configs.py --hypothesis my_hypothesis`
- [ ] Parameter overrides are in correct section (asr/vad/speech_segmenter)
- [ ] Test names are unique across all hypotheses
- [ ] Description clearly explains what's being tested

## Debugging Tips

### "Unknown config section" warning

Your parameter override section is not recognized. Check that you're using one of:
- `"asr"` (for ASR/decoder/provider parameters)
- `"vad"` (for VAD/speech segmentation parameters)
- `"speech_segmenter"` (for backend selection)

### Parameter has no effect

The parameter might not be wired correctly in the pipeline. Check:
1. `_build_resolved_config()` in `hypothesis_test_suite.py`
2. Parameter mapping (decoder vs provider)
3. Parameter name spelling

### Test crashes with "KeyError"

The parameter doesn't exist in the v1.7.3 defaults. Either:
1. Add it to `V173Defaults.ASR` or `V173Defaults.VAD` in `hypothesis_configs.py`
2. Or handle it specially in `_build_resolved_config()`

## Advanced: Parameter Ranges

To test a range of values systematically:

```python
@staticmethod
def get_threshold_sweep_tests() -> List[TestConfig]:
    """Test VAD threshold across a range."""
    configs = []

    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        configs.append(TestConfig(
            name=f"vad_threshold_{int(threshold*100)}",
            description=f"VAD threshold = {threshold}",
            hypothesis="vad_threshold_sweep",
            params_override={
                "vad": {"threshold": threshold}
            }
        ))

    return configs
```

## Contributing

If you develop a useful hypothesis that helps identify the regression, please:
1. Document your findings in the test results
2. Add comments explaining why the parameter matters
3. Consider submitting a PR with the hypothesis included

## Questions?

- Check existing hypotheses in `hypothesis_configs.py` for patterns
- Run `python inspect_hypothesis_configs.py --detailed` to see full configs
- Compare configs with `python inspect_hypothesis_configs.py --compare config1 config2`
