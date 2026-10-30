# CLI Reference

WhisperJAV provides several command-line tools.

---

## whisperjav (Main CLI)

Transcribe video/audio files to SRT subtitles.

```bash
whisperjav [OPTIONS] INPUT [INPUT...]
```

### Basic Options

| Option | Default | Description |
|--------|---------|-------------|
| `INPUT` | (required) | One or more video/audio file paths |
| `--mode` | `balanced` | Pipeline: `balanced`, `fast`, `faster`, `fidelity`, `transformers`, `qwen` |
| `--sensitivity` | `aggressive` | Detection sensitivity: `aggressive`, `balanced`, `conservative` |
| `--language` | `japanese` | Source audio language: `japanese`, `korean`, `chinese`, `english` |
| `--output-dir` | Same as source | Directory for output SRT files |
| `--output-format` | `srt` | Output format: `srt`, `vtt`, `both` |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (auto) | Whisper model: `large-v2`, `large-v3`, `turbo` |
| `--translate` | Off | Translate to English during transcription |

### Ensemble Options

| Option | Default | Description |
|--------|---------|-------------|
| `--ensemble` | Off | Enable ensemble mode |
| `--ensemble-serial` | Off | Complete each file fully before starting the next |
| `--pass1-pipeline` | `balanced` | Pipeline for first pass; also `parakeet-crispasr`, `cohere-crispasr` (GGUF, see [Parakeet-CrispASR and Cohere-CrispASR](../pipelines/crispasr.md)) and `xxl` (your own Faster Whisper XXL, see [BYOP: XXL Faster Whisper](../pipelines/ensemble.md#byop-xxl-faster-whisper-v189-pass-1-since-1100)) |
| `--pass2-pipeline` | (none) | Pipeline for second pass, the same choices; giving it turns pass 2 on |
| `--xxl-exe` | (none) | Path to `faster-whisper-xxl.exe`; required when either pass is `xxl`. `--pass1-model` / `--pass2-model` pick the model XXL runs (default `large-v2`) |
| `--merge-strategy` | `pass1_primary` | Merge strategy: `pass1_primary`, `pass2_primary`, `smart_merge`, `full_merge`, `pass1_overlap`, `pass2_overlap`, `longest` |

### ChronosJAV Timing Options (`--mode qwen`)

| Option | Default | Description |
|--------|---------|-------------|
| `--qwen-aligner` | `wav2vec2-ja` | The aligner that times the ASR model's text: `wav2vec2-ja` (232 MB), `qwen3-fa-gguf` (529 MB), `parakeet-ctc-ja` (not recommended: its times came back evenly spread in our tests), all run by CrispASR; or a PyTorch model ID such as `Qwen/Qwen3-ForcedAligner-0.6B` |
| `--qwen-timestamp-mode` | `aligner_vad_fallback` | Default: the aligner times the text; a piece of speech it cannot time keeps the speech segmenter's times. `vad_only` loads no aligner: each subtitle takes its piece of speech's times (the timing model can still move the start) |
| `--qwen-timing-model` | `whisper-small` | Whisper small (488 MB; its text is not used) moves each subtitle's start later to where speech starts, and splits a piece of speech at a punctuation mark or space near where a new sentence starts. `none` turns it off |
| `--qwen-stepdown` | Off | Retry a scene whose alignment collapses, with pieces of speech up to 6 s. When off (the default), such a scene takes the times of its pieces of speech |

The aligner and the timing model are new in 1.10.0, are downloaded on first use, and run where the install's CrispASR runs; `--device cpu` does not hold them on the CPU. See [Subtitle timing](../pipelines/chronosjav.md#subtitle-timing-1100). In an Ensemble pass, give them in `--pass1-qwen-params`, for example `'{"aligner_id": "qwen3-fa-gguf", "timing_model": "none"}'`.

### Processing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--device` | `auto` | Where this run's ASR and other models run, for every pipeline. `auto` = as the install is made for: its GPU when it can be used, else the CPU (not accelerated); `cpu` = this run on the CPU; `cuda` = an NVIDIA GPU is required. See [Accelerated or not](../getting-started/index.md#accelerated-or-not). |
| `--async-processing` | Off | Enable async processing |
| `--temp-dir` | System temp | Custom directory for temporary files |
| `--keep-temp` | Off | Keep intermediate files |
| `--debug` | Off | Enable debug logging |

### Examples

```bash
# Basic transcription
whisperjav video.mp4

# Fast mode with conservative sensitivity
whisperjav video.mp4 --mode faster --sensitivity conservative

# Ensemble with serial mode
whisperjav *.mp4 --ensemble --ensemble-serial --merge-strategy smart_merge

# VTT output
whisperjav video.mp4 --output-format vtt

# Custom output directory
whisperjav video.mp4 --output-dir ./subtitles/

# This run on the CPU (not accelerated)
whisperjav video.mp4 --mode faster --device cpu

# ChronosJAV without the timing model
whisperjav video.mp4 --mode qwen --qwen-timing-model none

# Parakeet-CrispASR as one Ensemble pass
whisperjav video.mp4 --ensemble --pass1-pipeline parakeet-crispasr

# Check the install: accelerator, where each part runs, models
whisperjav --check
```

### When a run does not start

| Exit status | Meaning |
|---|---|
| 0 | the run finished |
| 1 | the run finished, and at least one file failed (or is in a state named by `--fail-on`) |
| 2 | the run did not start: a setting it cannot use (the message says which) |
| 3 | the run did not start: the install is made for the NVIDIA GPU, and the GPU cannot be used now. The box names the reason and the fix; `--device cpu` runs the job on the CPU this time |
| 4 | the run did not start: PyTorch is missing or damaged (on an Own PyTorch install: not yet added). The message gives the command that fixes it |

When the install's GPU can be used (an NVIDIA GPU or Apple GPU install), nothing is shown at the start of a run. On every other install, a box says where each part runs on this computer; the run then goes on. `--hf-device` and `--qwen-device` from earlier releases still work as other names for `--device`.

---

## whisperjav-gui

Launch the GUI application.

```bash
whisperjav-gui
```

No additional arguments. All configuration is done through the GUI interface.

---

## whisperjav-translate

Standalone subtitle translation tool.

```bash
whisperjav-translate [OPTIONS] -i INPUT.srt
```

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input` | (required) | Input SRT file(s) |
| `--provider` | `deepseek` | AI provider |
| `--model` | (auto) | Model name |
| `--api-key` | (env var) | API key (or set via environment variable) |
| `--target-language` | `English` | Target language |
| `--tone` | `standard` | Translation tone: `standard`, `adult` |

---

## whisperjav-upgrade

Upgrade WhisperJAV to the latest version.

```bash
whisperjav-upgrade [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--check` | Check for updates without upgrading |
| `--wheel-only` | Code-only upgrade (skip dependency reinstall) |
| `--list-snapshots` | Show available rollback points |
| `--rollback` | Rollback to previous version |
| `--extras` | Upgrade specific extras only (e.g., `cli,gui`) |

See [Upgrade Guide](../UPGRADE.md) for details.
