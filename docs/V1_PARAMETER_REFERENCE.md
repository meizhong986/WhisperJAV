# v1 Configuration Parameter Reference

> **LEGACY REFERENCE DOCUMENT**
>
> This document describes the v1/v3 configuration system which is now **superseded by v4**.
> For new development, see: `whisperjav/config/v4/README.md`
>
> The v4 system uses YAML-driven configuration and does not require Python changes
> to add new models. See [ADR-001](adr/ADR-001-yaml-config-architecture.md) for details.

---

This document contains the complete parameter values from v1 (asr_config.json) for reference during v3 implementation.

**Source:** `whisperjav/config/asr_config.json` version 4.4

---

## FasterWhisper ASR Parameters

### Conservative Sensitivity

```python
# From common_decoder_options.conservative
task = "transcribe"
language = "ja"
best_of = 1
beam_size = 1
patience = 1.5
length_penalty = None
prefix = None
suppress_tokens = None  # Note: null in JSON
suppress_blank = True
without_timestamps = False
max_initial_timestamp = None

# From common_transcriber_options.conservative
temperature = [0.0]  # Single-element list
compression_ratio_threshold = 2.4
logprob_threshold = -1.0
no_speech_threshold = 0.74
condition_on_previous_text = False
initial_prompt = None
word_timestamps = True
prepend_punctuations = None
append_punctuations = None
clip_timestamps = None

# From faster_whisper_engine_options.conservative
chunk_length = None
repetition_penalty = 1.8
no_repeat_ngram_size = 2.0
prompt_reset_on_temperature = None
hotwords = None
multilingual = False
max_new_tokens = None
language_detection_threshold = None
language_detection_segments = None
log_progress = False

# From exclusive_whisper_plus_faster_whisper.conservative
hallucination_silence_threshold = 1.5
```

### Balanced Sensitivity

```python
# From common_decoder_options.balanced
task = "transcribe"
language = "ja"
best_of = 1
beam_size = 2
patience = 2.0  # Note: integer 2 in JSON, use 2.0 for float
length_penalty = None
prefix = None
suppress_tokens = None
suppress_blank = True
without_timestamps = False
max_initial_timestamp = None

# From common_transcriber_options.balanced
temperature = [0.0, 0.1]  # Two-element list for fallback
compression_ratio_threshold = 2.4
logprob_threshold = -1.2
no_speech_threshold = 0.5
condition_on_previous_text = False
initial_prompt = None
word_timestamps = True
prepend_punctuations = None
append_punctuations = None
clip_timestamps = None

# From faster_whisper_engine_options.balanced
chunk_length = None
repetition_penalty = 1.5
no_repeat_ngram_size = 2.0
prompt_reset_on_temperature = None
hotwords = None
multilingual = False
max_new_tokens = None
language_detection_threshold = None
language_detection_segments = None
log_progress = False

# From exclusive_whisper_plus_faster_whisper.balanced
hallucination_silence_threshold = 2.0
```

### Aggressive Sensitivity

```python
# From common_decoder_options.aggressive
task = "transcribe"
language = "ja"
best_of = 1
beam_size = 2
patience = 2.9
length_penalty = None
prefix = None
suppress_blank = False  # Different from others!
suppress_tokens = []    # Empty list, not None!
without_timestamps = False
max_initial_timestamp = None

# From common_transcriber_options.aggressive
temperature = [0.0, 0.3]  # Higher fallback temperature
compression_ratio_threshold = 3.0  # Higher threshold
logprob_threshold = -2.5  # More permissive
no_speech_threshold = 0.22  # Much lower - more sensitive
condition_on_previous_text = False
initial_prompt = None
word_timestamps = True
prepend_punctuations = None
append_punctuations = None
clip_timestamps = None

# From faster_whisper_engine_options.aggressive
chunk_length = 14  # Only aggressive has this set!
repetition_penalty = 1.1  # Lower penalty
no_repeat_ngram_size = 2.0
prompt_reset_on_temperature = None
hotwords = None
multilingual = False
max_new_tokens = None
language_detection_threshold = None
language_detection_segments = None
log_progress = False

# From exclusive_whisper_plus_faster_whisper.aggressive
hallucination_silence_threshold = 2.5  # Higher threshold
```

---

## Silero VAD Parameters

### Conservative Sensitivity

```python
# From silero_vad_options.conservative
threshold = 0.35
min_speech_duration_ms = 150
max_speech_duration_s = 9
min_silence_duration_ms = 300
neg_threshold = 0.3
speech_pad_ms = 400
```

### Balanced Sensitivity

```python
# From silero_vad_options.balanced
threshold = 0.18
min_speech_duration_ms = 100
max_speech_duration_s = 11
min_silence_duration_ms = 300
neg_threshold = 0.15
speech_pad_ms = 400
```

### Aggressive Sensitivity

```python
# From silero_vad_options.aggressive
threshold = 0.05
min_speech_duration_ms = 30
max_speech_duration_s = 14
min_silence_duration_ms = 300
neg_threshold = 0.1
speech_pad_ms = 600
```

---

## Key Observations

### Temperature Handling
- v1 uses lists for temperature fallback: `[0.0]`, `[0.0, 0.1]`, `[0.0, 0.3]`
- faster-whisper accepts either float or tuple/list
- Must preserve this behavior in v3

### Null vs Empty
- `suppress_tokens = None` means use default
- `suppress_tokens = []` means suppress nothing extra
- Aggressive specifically uses `[]`

### Sensitivity Philosophy
- **Conservative**: Strict thresholds, single temperature, higher no_speech_threshold (0.74)
- **Balanced**: Middle ground, small temperature fallback
- **Aggressive**: Permissive thresholds, larger temperature fallback, low no_speech_threshold (0.22)

### Parameters That Differ Significantly by Sensitivity

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| beam_size | 1 | 2 | 2 |
| patience | 1.5 | 2.0 | 2.9 |
| temperature | [0.0] | [0.0, 0.1] | [0.0, 0.3] |
| no_speech_threshold | 0.74 | 0.5 | 0.22 |
| logprob_threshold | -1.0 | -1.2 | -2.5 |
| compression_ratio_threshold | 2.4 | 2.4 | 3.0 |
| repetition_penalty | 1.8 | 1.5 | 1.1 |
| hallucination_silence_threshold | 1.5 | 2.0 | 2.5 |
| suppress_blank | True | True | False |
| chunk_length | None | None | 14 |

### VAD Threshold Pattern
- Conservative: 0.35 (high = less sensitive)
- Balanced: 0.18
- Aggressive: 0.05 (low = very sensitive)

---

## Validation Checklist

When implementing v3 presets, verify:

- [ ] All 34+ ASR parameters are present in Options class
- [ ] Temperature is List[float] type
- [ ] All three presets set ALL parameters explicitly
- [ ] Values match this reference exactly
- [ ] VAD has neg_threshold field
- [ ] VAD presets match reference exactly
- [ ] legacy.py passes through all parameters
