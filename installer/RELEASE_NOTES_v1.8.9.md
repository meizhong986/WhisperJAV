# WhisperJAV v1.8.9

## What's New

- **XXL Faster Whisper (BYOP)** — Use PurfView's Faster Whisper XXL as Pass 2 in ensemble mode. Pick your XXL executable in the GUI, add any extra flags you want, and hit Start. WhisperJAV runs XXL as a subprocess, reads the SRT, and merges it with Pass 1. Real-time console output so you can see what XXL is doing. (#223, #224)

- **Better transcription out of the box** — Default model upgraded from large-v2 to large-v3 (10-20% better Japanese accuracy). Compute type changed from int8 to float16 (full precision, no rounding). All sensitivity presets retuned — tighter temperature ranges, better VAD thresholds, stronger repetition penalty. Fast Japanese speech is no longer incorrectly discarded. This is the largest accuracy improvement since v1.7. (#223, #224)

- **Smarter Ollama integration** — OllamaManager detects your GPU, picks the right model for your VRAM, starts the Ollama server automatically, and pulls models with your consent. Curated configs for Qwen 2.5, Gemma 2, and Llama 3.1. Dynamic context window from model metadata instead of hardcoded 8192. New CLI flags: `--ollama-url`, `--list-ollama-models`, `--yes`. (#132, #212, #214, #128)

- **VAD failover** — When voice activity detection finds no speech in a scene, WhisperJAV now transcribes the full audio instead of silently producing empty subtitles.

- **Honest ensemble reporting** — When Pass 2 fails and falls back to Pass 1, the output is now clearly marked as "Partial (fallback)" instead of "Successful". Exit code is non-zero. Translation is skipped for fallback files to avoid wasting API cost on inferior output.

- **Settings cleanup** — Config file reduced from ~600 lines to 19. Fewer config conflicts, faster startup. Your old settings are backed up automatically.

## Bug Fixes

### Ensemble / BYOP

- **False success reporting** — Ensemble mode always exited with code 0 even when passes failed, causing the GUI to show "[SUCCESS]" incorrectly. Now reflects reality. (#223)
- **XXL crash-on-exit recovery** — XXL's underlying engine (ctranslate2) sometimes crashes during shutdown after the transcription is already complete and the SRT is written. Previously this threw away the valid output. Now WhisperJAV checks for the SRT file first and keeps it. (#223)
- **XXL `--verbose` crash** — XXL's `--verbose` flag requires a value (e.g. `--verbose True`), not a bare flag. The runner no longer assumes what flags XXL wants — only the 4 required integration args are sent. Everything else is your choice via the GUI extra args field. (#223)
- **XXL extra args parsing** — Quoted arguments like `"--verbose True"` were broken by simple string splitting. Now uses proper shell-aware parsing.
- **BYOP preferences never saved** — The GUI's save function called a method that doesn't exist on the config manager. The error was silently caught. Your XXL exe path and extra args were never actually persisted between sessions. Fixed.
- **XXL exe path lost on ensemble start** — Path was only saved when you clicked Browse, not when you started processing. Now saved in both places.
- **Translation ran on fallback files** — When Pass 2 failed, the pipeline still tried to translate the inferior Pass 1 output, wasting time and API cost. Now skipped.
- **Pre-translation SRT validation** — Previously loaded the full LLM model (5+ GB) before checking if the SRT had enough content. Now checks first and skips with a clear message if fewer than 2 subtitle blocks exist. (#212, #214)

### GUI

- **Segmenter schema error** — "Unknown segmenter backend: silero-v6.2" when opening Customize Parameters. Missing mapping entry.
- **Fidelity mode showed wrong default model** — Displayed "Turbo" instead of "Large V2" due to array ordering.
- **Pass 2 defaults out of sync** — Switching pipelines used stale v1.7 defaults instead of the current v1.8.9 values.

### Pipeline / Engine

- **Analytics Phase 9 crash** — NoneType error when diagnostics JSON contained null values. Fixed in 4 locations.
- **VAD threshold fallback** — Incorrect fallback value (0.4 instead of 0.18) when no preset matched.
- **torch_dtype deprecation** — Replaced deprecated parameter with `dtype` in HuggingFace model loading. Removed 5 warning suppression filters that were masking the issue.
- **Misleading config log in ensemble mode** — Debug log showed stale default values instead of actual ensemble configuration.

### Cleanup

- Removed dead CLI flags: `--adaptive-classification`, `--adaptive-audio-enhancement`, `--smart-postprocessing`
- Removed `--xxl-args` CLI flag (extra args are persisted via the GUI config file)

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

## Compatibility

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 - 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |
| NumPy | 2.0+ |

### VRAM Requirements

The move from int8 to float16 uses more GPU memory:

| Model | v1.8.8 (int8) | v1.8.9 (float16) |
|-------|--------------|------------------|
| large-v3 | ~4-5 GB | ~6-8 GB |

If you have less than 8 GB VRAM, try `--mode faster` which uses automatic compute type and lower memory.

## Known Issues

- **Apple Silicon MPS + whisper-large-v3-turbo** — Produces garbage output on MPS for this specific model. Use `--hf-device cpu` or stick with the default kotoba model which works fine on MPS. Investigating whether this affects M3/M4 chips or only M1. (#198, #227)
- **LLM translation quality** — `--provider ollama` or `--provider deepseek` are recommended for best results. Full Ollama migration planned for v1.9.0.

## Troubleshooting

- **Installation takes too long**: First install downloads ~2 GB of packages. Subsequent installs are much faster (cached).
- **GPU not detected**: Check that `nvidia-smi` works in your terminal. On Optimus laptops, try `python install.py --force-cuda cu124`. (#200)
- **Out of VRAM**: v1.8.9 uses float16 which needs more VRAM than v1.8.8. If you have less than 8 GB VRAM, try `--mode faster` which uses auto compute type.
- **Something went wrong**: Check `install_log.txt` in the project folder for details. Include it when reporting issues.

## What's Next (v1.9.0)

- Full Ollama migration — deprecate llama-cpp-python
- Standalone subtitle merge CLI tool (`whisperjav-merge`) (#230)
- Gemma 3 model configs (#128)
- Chinese GUI (partial i18n) (#175, #180)

## Technical Details

<details>
<summary>Click to expand — for power users and contributors</summary>

### New CLI Flags
- `--pass2-pipeline xxl` — BYOP ensemble mode with XXL Faster Whisper as external Pass 2
- `--xxl-exe PATH` — Path to faster-whisper-xxl executable
- `--ollama-url URL` — Custom Ollama server URL
- `--list-ollama-models` — Show locally available Ollama models
- `--yes` / `-y` — Auto-confirm prompts (model pulls, etc.)

### Quality Parameter Changes
- **Model**: `large-v2` → `large-v3`
- **Compute type**: `int8_float16` → `float16` (CUDA), `auto` (CPU/MPS)
- **Conservative**: beam 3→1, best_of 3→1, patience 1.5→1.2, temperature deterministic (0.0), compression_ratio 2.4→2.2
- **Balanced**: beam 5→2, best_of 3→2, temperature 6-step→(0.0, 0.2), no_speech 0.5→0.4
- **Aggressive**: beam 5→3, best_of 5→3, patience 2.9→2.5, temperature 6-step→(0.0, 0.2, 0.4), rep_penalty 1.1→1.3
- **VAD Conservative**: threshold 0.35→0.28, neg 0.3→0.4, min_speech 150→120ms, pad 400→300ms
- **VAD Aggressive**: threshold 0.05→0.10, neg 0.1→0.08, min_silence 300→200ms, pad 600→500ms
- **CPS threshold**: 20→30 (Japanese natural speech reaches 15-25 CPS)
- **Scene detection**: Pass 2 max duration auto-derives from max_duration instead of hardcoded 1900s

### Config Architecture
- `asr_config.json` reduced from ~600 lines to 19 lines (UI preferences only)
- Pydantic presets in `config/components/` are the single source of truth
- Legacy config backed up to `asr_config.pre_v189_cleanup.json`

### OllamaManager
- Server lifecycle: detect, auto-start with atexit cleanup, Windows process group support
- Model management: list, check, pull with streaming progress
- VRAM-aware recommendation: qwen2.5:3b (CPU), qwen2.5:7b (8GB), gemma3:12b (12GB), qwen2.5:14b (16GB+)
- Dynamic context window from `/api/show` metadata
- Zero new pip dependencies (uses urllib.request)

</details>
