# WhisperJAV v1.8.9

## What's New

- **BYOP Faster Whisper XXL** — Bring Your Own Provider integration. Use PurfView's Faster Whisper XXL as Pass 2 in ensemble mode via GUI or CLI (`--pass2-pipeline xxl --xxl-exe /path/to/xxl`). XXL runs as an external subprocess — WhisperJAV reads the SRT output and merges it with Pass 1. Real-time console streaming of XXL progress
- **Transcription quality overhaul** — Default model upgraded from `large-v2` to `large-v3` (10-20% better Japanese accuracy). Compute type changed from `int8_float16` to `float16` (lossless precision). This is the largest quality improvement since v1.7
- **Hardened pipeline defaults** — All sensitivity presets (conservative/balanced/aggressive) retuned for accuracy over coverage. Tighter temperature fallback ranges, tuned VAD thresholds, stronger repetition penalty. CPS filter threshold raised from 20 to 30 for Japanese (prevents valid fast speech from being discarded)
- **OllamaManager smart integration** — Automatic model detection, VRAM-aware model selection, server auto-start, curated configs for popular models (Gemma 2 9B, Qwen 2.5 14B, Llama 3.1 8B). Builds on v1.8.8's `--provider ollama` preview
- **Config system cleanup** — `asr_config.json` stripped to UI preferences only. All pipeline parameters now come from Pydantic presets (single source of truth). Legacy config sections backed up to `asr_config.pre_v189_cleanup.json`
- **VAD failover** — When Silero VAD returns empty speech segments for a scene, WhisperJAV now falls back to full-audio transcription instead of silently producing empty results

## Installation

### Windows — Standalone Installer (.exe)

The easiest way. No Python knowledge needed.

1. Download **WhisperJAV-1.8.9-Windows-x86_64.exe** from the Assets below
2. Run the installer (no admin rights required)
3. Wait 10-20 minutes for setup to complete
4. Launch from the Desktop shortcut

Installs to `%LOCALAPPDATA%\WhisperJAV`. A desktop shortcut is created automatically. Your GPU is detected automatically.

### Windows — Source Install

Requires [Git](https://git-scm.com/downloads) and [Python 3.10-3.12](https://www.python.org/downloads/). Open a terminal and run:

```
cd %USERPROFILE%
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.9
installer\install_windows.bat
```

After installation, double-click **WhisperJAV.bat** to launch the GUI.

### macOS

Requires [Git](https://git-scm.com/downloads). The install script checks for everything else (Xcode CLI Tools, Python, FFmpeg, PortAudio) and tells you exactly what to install if anything is missing. Open Terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.9
installer/install_mac.sh
```

After installation, open the `whisperjav` folder in Finder and double-click **WhisperJAV.command** to launch the GUI.

### Linux

Requires Git and Python 3.10-3.12. The install script handles PEP 668 (externally-managed) environments on Debian 12+ / Ubuntu 24.04+. Open a terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.9
installer/install_linux.sh
```

After installation, launch the GUI with `./WhisperJAV.sh`.

### Google Colab / Kaggle

Use the notebooks in the `notebook/` folder, or install directly:

```python
!pip install "whisperjav[colab] @ git+https://github.com/meizhong986/whisperjav.git@v1.8.9"
```

### Upgrade from v1.8.8

If you already have WhisperJAV installed:

```bash
whisperjav-upgrade
```

Or manually:

```bash
cd whisperjav       # your existing clone
git pull
git checkout v1.8.9
installer\install_windows.bat        # Windows
installer/install_mac.sh             # macOS
installer/install_linux.sh           # Linux
```

## What Changed (Technical Details)

### New Features
- `--pass2-pipeline xxl`: BYOP ensemble mode with Faster Whisper XXL as external Pass 2
- `--xxl-exe PATH`: Path to faster-whisper-xxl executable
- `--xxl-args ARGS`: Extra arguments passed directly to XXL
- GUI: BYOP panel with Browse button, extra args field, and pipeline dropdown including XXL option
- `OllamaManager`: Smart Ollama lifecycle management with model metadata queries, VRAM detection, auto-start/stop, curated model configs

### Quality Improvements
- **Model**: `large-v2` → `large-v3` (10-20% better Japanese accuracy)
- **Compute type**: `int8_float16` → `float16` (lossless precision, requires 8GB+ VRAM)
- **Sensitivity presets retuned**: beam_size, best_of, patience, temperature, no_speech_threshold, repetition_penalty, logprob_threshold all adjusted per sensitivity level
- **VAD presets retuned**: threshold, neg_threshold, min_speech_duration, min_silence_duration, speech_pad_ms adjusted per sensitivity level
- **CPS threshold**: 20.0 → 30.0 (Japanese natural speech reaches 15-25 CPS)
- **VAD failover**: Empty VAD results now trigger full-audio transcription fallback
- **Scene detection**: `pass2_max_duration_s` auto-derives from `max_duration - 1.0` (28.0s) instead of hardcoded 1900.0

### Config Cleanup
- `asr_config.json` reduced from ~600 lines to 19 lines (UI preferences only)
- Pydantic presets in `config/components/` are now the single source of truth for all pipeline parameters
- Legacy sections backed up to `asr_config.pre_v189_cleanup.json`

## Compatibility

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 - 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |
| NumPy | 2.0+ |

### VRAM Requirements

The compute type change from int8 to float16 increases VRAM usage:

| Model | int8_float16 (v1.8.8) | float16 (v1.8.9) |
|-------|----------------------|------------------|
| large-v3 | ~4-5 GB | ~6-8 GB |

Users with less than 8 GB VRAM may experience slower performance due to memory swapping. The faster/fast pipelines (speed-focused) still use auto compute type.

## Known Issues

- **Apple Silicon MPS + whisper-large-v3-turbo**: Produces garbage output on MPS for this specific model. Use `--hf-device cpu` or stick with the default kotoba model which works fine on MPS. Investigating whether this affects M3/M4 chips or only M1.
- **LLM translation quality**: `--provider ollama` or `--provider deepseek` are recommended. Full Ollama migration planned for v1.9.0.

## Troubleshooting

- **Installation takes too long**: First install downloads ~2 GB of packages. Subsequent installs are much faster (cached).
- **GPU not detected**: Check that `nvidia-smi` works in your terminal. On Optimus laptops, try `python install.py --force-cuda cu124`.
- **Out of VRAM**: v1.8.9 uses float16 which needs more VRAM than v1.8.8. If you have less than 8 GB VRAM, try `--mode faster` which uses auto compute type.
- **Something went wrong**: Check `install_log.txt` in the project folder for details. Include it when reporting issues.

## What's Next (v1.9.0)

- Full Ollama migration with `LLMBackend` protocol abstraction
- Deprecate llama-cpp-python (move to `[legacy]` extra)
- Standalone subtitle merge CLI tool (`whisperjav-merge`)
- Gemma 3 model configs for OllamaManager
