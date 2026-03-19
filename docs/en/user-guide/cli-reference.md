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
| `--mode` | `balanced` | Pipeline: `balanced`, `fast`, `faster`, `fidelity`, `transformers` |
| `--sensitivity` | `aggressive` | Detection sensitivity: `aggressive`, `balanced`, `conservative` |
| `--language` | `ja` | Source language code (`ja`, `ko`, `zh`, `en`) |
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
| `--pass1-pipeline` | `balanced` | Pipeline for first pass |
| `--pass2-pipeline` | `qwen` | Pipeline for second pass |
| `--merge-strategy` | `smart` | Merge strategy: `pass1_primary`, `smart`, `full`, `pass2_primary`, `longest` |

### Processing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cpu-only` | Off | Force CPU mode (no GPU) |
| `--async` | Off | Enable async processing |
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
whisperjav *.mp4 --ensemble --ensemble-serial --merge-strategy smart

# VTT output
whisperjav video.mp4 --output-format vtt

# Custom output directory
whisperjav video.mp4 --output-dir ./subtitles/

# CPU-only mode
whisperjav video.mp4 --mode faster --cpu-only
```

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
