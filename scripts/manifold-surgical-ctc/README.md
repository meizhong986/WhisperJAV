# Manifold Surgical CTC

A reference implementation for gap-filling in ASR workflows using CTC-based models.

## Problem Statement

Whisper excels at transcribing clean speech but systematically misses challenging audio segments in JAV content (noisy backgrounds, overlapping sounds, etc.). Speech enhancement can make these segments audible to humans, but **enhancement changes the acoustic distribution** that Whisper expects, leading to mistranscription.

**Key insight**: CTC-based models (MMS, Wav2Vec2, HuBERT) are trained on augmented audio and tolerate enhancement artifacts much better than Whisper. They operate on a different "acoustic manifold."

## Design Philosophy

This is **NOT** an ASR replacement. It's a "speech recall probe":

- **ONLY** operates on gaps where Whisper failed
- **ONLY** uses enhanced audio (post speech-enhancement)
- **NEVER** overrides Whisper output
- **ONLY** adds fragments where silence currently exists
- **Precision > Recall** (aggressive filtering to avoid garbage)

## Architecture

```
RAW AUDIO ─────────┬──────────────────────────────────────────────────┐
                   │                                                   │
                   ▼                                                   │
            Whisper (Pass 1)                                          │
                   │                                                   │
                   ▼                                                   │
         whisper_transcript.srt ──────┐                               │
                                      │                               │
                                      ▼                               │
                               Gap Detection                          │
                                      │                               │
                                      ▼                               │
                              triage_gaps.srt                         │
                                      │                               │
          ┌───────────────────────────┴───────────────────────┐       │
          ▼                                                   ▼       │
   Extract Segment                                     Extract Segment│
          │                                                   │       │
          ▼                                                   ▼       │
   Speech Enhancement                              Speech Enhancement │
          │                                                   │       │
          ▼                                                   ▼       │
   CTC Transcription                               CTC Transcription  │
          │                                                   │       │
          ▼                                                   ▼       │
      Filtering                                           Filtering   │
          │                                                   │       │
          └─────────────────────┬─────────────────────────────┘       │
                                │                                     │
                                ▼                                     │
                    ctc_filtered_transcripts.json                     │
                                │                                     │
                                ▼                                     │
                     Conservative Merge ◄─────────────────────────────┘
                                │
                                ▼
                      Final Subtitles
```

## Supported CTC Backends

| Backend | Model | Japanese Quality | Enhancement Tolerance |
|---------|-------|------------------|----------------------|
| **MMS** | `facebook/mms-1b-all` | Good (multilingual) | Very High |
| **ReazonSpeech** | `reazon-research/reazonspeech-wav2vec2-large-rs35kh` | Excellent (native) | High |

**Note**: `facebook/mms-300m` is the base pretrained model WITHOUT language adapters.
Always use `facebook/mms-1b-all` for actual ASR (the script will auto-redirect).

### Why MMS for Gap-Filling?

- **Weak language model** = doesn't hallucinate structure
- **High phonetic recall** = catches fragments Whisper misses
- **Trained on augmented audio** = tolerates enhancement artifacts
- Will output kana-like approximations and short interjections that Whisper ignores

### Why ReazonSpeech Alternative?

- **35,000 hours of Japanese training** = better phonetic accuracy
- **CER 11%** on Japanese benchmarks
- May offer higher accuracy at the cost of slightly less enhancement tolerance

## Installation

```bash
# From WhisperJAV root directory
cd scripts/manifold-surgical-ctc

# Install additional dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage (MMS with auto gap detection)

```bash
python surgical_ctc_reference_implementation.py \
    --source-audio INPUT/source_audio.wav \
    --whisper-srt INPUT/whisper_transcript.srt \
    --enhancer clearvoice \
    --ctc-model facebook/mms-1b-all \
    --output-dir OUTPUT/results
```

### With Manual Gap Specification

```bash
python surgical_ctc_reference_implementation.py \
    --source-audio INPUT/source_audio.wav \
    --manual-gaps INPUT/triage_gaps.srt \
    --enhancer bs-roformer \
    --ctc-model facebook/mms-1b-all \
    --output-dir OUTPUT/results
```

### Testing ReazonSpeech Backend

```bash
python surgical_ctc_reference_implementation.py \
    --source-audio INPUT/source_audio.wav \
    --whisper-srt INPUT/whisper_transcript.srt \
    --ctc-backend reazonspeech \
    --ctc-model reazon-research/reazonspeech-wav2vec2-large-rs35kh \
    --enhancer clearvoice \
    --output-dir OUTPUT/results
```

### Without Enhancement (Baseline Test)

```bash
python surgical_ctc_reference_implementation.py \
    --source-audio INPUT/source_audio.wav \
    --whisper-srt INPUT/whisper_transcript.srt \
    --enhancer none \
    --ctc-model facebook/mms-1b-all \
    --output-dir OUTPUT/results_no_enhance
```

## CLI Arguments

### Input/Output

| Argument | Description | Required |
|----------|-------------|----------|
| `--source-audio` | Source audio file (WAV, MP4, etc.) | Yes |
| `--whisper-srt` | Whisper's SRT output for auto gap detection | One of these |
| `--manual-gaps` | Manually specified gaps file | One of these |
| `--output-dir` | Output directory (default: `./results`) | No |

### CTC Backend

| Argument | Description | Default |
|----------|-------------|---------|
| `--ctc-backend` | Backend: `mms` or `reazonspeech` | `mms` |
| `--ctc-model` | Model identifier | `facebook/mms-1b-all` |
| `--device` | Device: `auto`, `cuda`, `cpu` | `auto` |

### Enhancement

| Argument | Description | Default |
|----------|-------------|---------|
| `--enhancer` | Backend: `none`, `clearvoice`, `bs-roformer`, `zipenhancer` | `none` |
| `--enhancer-model` | Model variant (optional) | Backend default |

### Gap Detection

| Argument | Description | Default |
|----------|-------------|---------|
| `--min-gap-duration` | Minimum gap to consider (seconds) | `0.3` |
| `--max-gap-duration` | Maximum chunk duration (seconds) | `3.0` |
| `--gap-padding` | Padding around gaps (seconds) | `0.15` |
| `--split-long-gaps` | Split gaps longer than max into chunks | `True` |
| `--no-split-long-gaps` | Ignore gaps longer than max instead | - |

**Gap Detection Behavior:**
- Gaps shorter than `--min-gap-duration` are ignored
- By default, gaps longer than `--max-gap-duration` are split into chunks
- Use `--no-split-long-gaps` to ignore long gaps entirely (original behavior)
- If Whisper SRT has no segments, entire audio is chunked and processed

### Filtering

| Argument | Description | Default |
|----------|-------------|---------|
| `--min-output-chars` | Minimum characters in output | `2` |
| `--max-output-chars` | Maximum characters in output | `100` |
| `--min-confidence` | Minimum confidence threshold | `0.3` |
| `--min-japanese-ratio` | Minimum ratio of Japanese characters (0.0-1.0) | `0.5` |
| `--no-require-japanese` | Allow outputs without Japanese characters (not recommended) | - |

**Japanese Character Filtering (v1.1):**

MMS can produce garbage output (Latin ligatures like `fi`, `ffi`, or random letters) when processing noise/non-speech audio. The Japanese filtering ensures:

1. **Unicode ligature detection** - Rejects outputs containing Unicode ligatures (U+FB00-U+FB4F)
2. **Japanese character requirement** - By default, outputs must contain Japanese characters
3. **Japanese ratio threshold** - At least 50% of characters must be Japanese (hiragana, katakana, or kanji)

This dramatically improves precision by rejecting MMS garbage while keeping actual Japanese speech.

### Misc

| Argument | Description |
|----------|-------------|
| `--keep-temp` | Keep temporary audio files for inspection |
| `--verbose` | Enable debug logging |

## Input Formats

### Whisper SRT (`--whisper-srt`)

Standard SRT format from Whisper:

```srt
1
00:00:01,000 --> 00:00:03,500
こんにちは

2
00:00:05,000 --> 00:00:08,000
今日は天気がいいですね
```

### Manual Gaps (`--manual-gaps`)

SRT-like format:

```srt
1
00:00:03,500 --> 00:00:05,000

2
00:00:08,000 --> 00:00:10,500
```

Or simple format:

```
# Gaps in seconds (start - end)
3.5 - 5.0
8.0 - 10.5
```

## Output Files

All output files are prefixed with the source audio basename.

For input `25min-HODV-22019.wav`:

| File | Description |
|------|-------------|
| `25min-HODV-22019.detected_gaps.json` | All gaps identified with timestamps and metadata |
| `25min-HODV-22019.ctc_raw_transcripts.json` | Raw CTC output before filtering |
| `25min-HODV-22019.ctc_filtered_transcripts.json` | Post-filtering output with acceptance status |
| `25min-HODV-22019.surgical_ctc.srt` | **SRT subtitle file with all accepted outputs** |
| `25min-HODV-22019.surgical_evaluation.txt` | Human-readable summary report |

### Example `ctc_filtered_transcripts.json`

```json
[
  {
    "gap_index": 0,
    "start": 3.5,
    "end": 5.0,
    "text": "あっ",
    "confidence": 0.72,
    "was_filtered": false,
    "filter_reason": null
  },
  {
    "gap_index": 1,
    "start": 8.0,
    "end": 10.5,
    "text": null,
    "confidence": 0.21,
    "was_filtered": true,
    "filter_reason": "low_confidence (0.210 < 0.3)"
  }
]
```

### Example `surgical_ctc.srt`

The SRT file contains only accepted (non-filtered) results:

```srt
1
00:00:03,500 --> 00:00:05,000
あっ

2
00:00:12,300 --> 00:00:14,800
うん、そう
```

This SRT can be:
- Merged with Whisper's original SRT to create a complete transcript
- Used standalone for analysis
- Loaded into subtitle editors for manual review

## Evaluation Workflow

### Step 1: Create Benchmark

1. Select a 2-3 minute clip where Whisper consistently misses utterances
2. Run Whisper to get `whisper_transcript.srt`
3. Manually transcribe the gaps to create ground truth

### Step 2: Run PoC

```bash
python surgical_ctc_reference_implementation.py \
    --source-audio benchmark.wav \
    --whisper-srt whisper_transcript.srt \
    --enhancer clearvoice \
    --ctc-model facebook/mms-1b-all \
    --output-dir benchmark_results
```

### Step 3: Evaluate

Compare `ctc_filtered_transcripts.json` against your ground truth:

- **True Positives**: CTC correctly recovered missed speech
- **False Positives**: CTC output garbage (filter tuning needed)
- **False Negatives**: CTC also missed the speech (enhancement or model change needed)

### Step 4: Iterate

- If too many false positives: Increase `--min-output-chars`, `--min-confidence`
- If too many false negatives: Try different enhancer
- If Japanese accuracy is poor: Try `--ctc-backend reazonspeech`

## Known Limitations

### MMS Limitations

- May output weird kana approximations
- Can drop long vowels
- No punctuation output

### Enhancement Limitations

- Aggressive enhancement can remove speech information irreversibly
- Different enhancers work better for different noise types
- BS-RoFormer is for vocal isolation (music/effects), ClearVoice for denoising

### General Limitations

- Long gaps are split into chunks (default 3s max per chunk)
- Cannot infer semantic meaning from noise
- Fragmentary success is expected and acceptable

### Confidence Scoring

Both MMS and ReazonSpeech backends now compute confidence scores from CTC logits:
- Mean of max softmax probabilities across significant frames
- `--min-confidence` threshold applies consistently to both backends
- Low confidence often indicates noise/silence rather than speech

## Extending to Other CTC Backends

The implementation uses a `CTCBackend` protocol. To add a new backend:

```python
class MyNewBackend:
    def transcribe(self, audio_path: Path, language: str = "ja") -> Tuple[str, Optional[float]]:
        # Your implementation
        return text, confidence

    def cleanup(self) -> None:
        # Release resources
        pass
```

Then add to `create_ctc_backend()` factory function.

## License

MIT License - Part of the WhisperJAV project.
