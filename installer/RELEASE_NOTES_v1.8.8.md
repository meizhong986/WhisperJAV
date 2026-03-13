# WhisperJAV v1.8.8

## What's New

- **Ollama translation support** — New `--provider ollama` option for local LLM translation. Auto-detects a running Ollama server, sets proper context window (8192 tokens), and streams responses. A simpler, more reliable alternative to the built-in llama-cpp-python backend
- **Improved LLM translation diagnostics** — Token usage tracking, batch statistics, and detailed logging when translations fail. Helps diagnose "No matches found" and server error issues
- **Stale settings detection** — `.subtrans` resume files from previous WhisperJAV versions are now automatically detected and cleared, preventing old settings from silently overriding new CLI arguments
- **NVIDIA Optimus laptop support** — Added `--force-cuda` flag to `install.py` for laptops where GPU auto-detection fails. Improved error messages with Optimus-specific guidance
- **Better repetition cleaning** — New pattern catches repeated phrase clusters (e.g., `はい、はい、はい・・・。` repeated 14 times)
- **ClearVoice speech separation fix** — MossFormer2_SS_16K model now works correctly. Previously crashed with "too many dimensions" error
- **cu118 wheel compatibility** — Fixed `uv` rejecting llama-cpp-python CUDA 11.8 wheels due to version metadata mismatch

### Also includes everything from v1.8.8b1

- GUI launcher for source installs (double-clickable `.bat`/`.command`/`.sh`)
- Smart environment detection (installs into active conda/venv)
- numpy 2.x support
- Apple Silicon MPS beam search fix (#198)
- Startup warning suppression

## Installation

### Windows — Standalone Installer (.exe)

The easiest way. No Python knowledge needed.

1. Download **WhisperJAV-1.8.8-Windows-x86_64.exe** from the Assets below
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
git checkout v1.8.8
installer\install_windows.bat
```

After installation, double-click **WhisperJAV.bat** to launch the GUI.

### macOS

Requires [Git](https://git-scm.com/downloads). The install script checks for everything else (Xcode CLI Tools, Python, FFmpeg, PortAudio) and tells you exactly what to install if anything is missing. Open Terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.8
installer/install_mac.sh
```

After installation, open the `whisperjav` folder in Finder and double-click **WhisperJAV.command** to launch the GUI.

### Linux

Requires Git and Python 3.10-3.12. The install script handles PEP 668 (externally-managed) environments on Debian 12+ / Ubuntu 24.04+. Open a terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.8
installer/install_linux.sh
```

After installation, launch the GUI with `./WhisperJAV.sh`.

### Google Colab / Kaggle

Use the notebooks in the `notebook/` folder, or install directly:

```python
!pip install "whisperjav[colab] @ git+https://github.com/meizhong986/whisperjav.git@v1.8.8"
```

### Upgrade from v1.8.7

If you already have WhisperJAV installed:

```bash
whisperjav-upgrade
```

Or manually:

```bash
cd whisperjav       # your existing clone
git pull
git checkout v1.8.8
installer\install_windows.bat        # Windows
installer/install_mac.sh             # macOS
installer/install_linux.sh           # Linux
```

## What Changed (Technical Details)

### New Features
- `--provider ollama`: Auto-detects Ollama at `localhost:11434`, sets `num_ctx=8192`, auto-caps batch size, always streams. Friendly error with install link if Ollama not found
- `--force-cuda VERSION`: Manual CUDA version override for `install.py` when GPU auto-detection fails (e.g., Optimus laptops)
- Diagnostic token logging: batch stats tracker, "No matches" warning handler, server error handler with per-translation summary
- `.subtrans` version stamping: stale project files from previous versions are auto-deleted on version change

### Bug Fixes
- **fix(#219)**: ClearVoice MossFormer2_SS_16K speech separation model outputs 3D tensor — now handled by taking first separated source
- **fix(#218)**: Set `UV_SKIP_WHEEL_FILENAME_CHECK=1` for uv to accept cu118 llama-cpp-python wheels with mismatched internal version metadata
- **fix(#209)**: New repetition cleaner pattern #8 (`sentence_phrase_repetition`) catches clustered phrase repetitions
- **fix(#200)**: Improved GPU detection error messages with Optimus-specific guidance
- **fix**: Reduced `max_tokens` multiplier from 2x to 1x — prevents models from filling output with verbose/garbled text
- **fix**: `pornify.txt` instruction file rewritten to sectioned format with proper `#N / Original> / Translation>` example

### Translation Improvements
- `pornify.txt` converted from legacy format to sectioned format (`### prompt` / `### instructions` / `### retry_instructions`), matching `standard.txt` structure
- Ollama provider uses Custom Server PySubtrans backend (OpenAI-compatible API)

## Compatibility

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 - 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |
| NumPy | 2.0+ |

## Known Issues

- **LLM translation quality** (#196/#212/#214/#132): Tactical fixes included (diagnostics, max_tokens, .subtrans fix). Full resolution planned for v1.9.0 with Ollama migration. In the meantime, `--provider ollama` or `--provider deepseek` are recommended alternatives.
- **stable_whisper SyntaxWarning on startup**: Harmless warning from upstream code. Does not affect functionality.

## Troubleshooting

- **Installation takes too long**: First install downloads ~2 GB of packages. Subsequent installs are much faster (cached).
- **GPU not detected**: Check that `nvidia-smi` works in your terminal. On Optimus laptops, try `python install.py --force-cuda cu124`.
- **Something went wrong**: Check `install_log.txt` in the project folder for details. Include it when reporting issues.

## What's Next (v1.9.0)

- Full Ollama backend with `LLMBackend` protocol abstraction
- Deprecate llama-cpp-python (move to `[legacy]` extra)
- Remove ~3,100 lines of fragile LLM infrastructure code
