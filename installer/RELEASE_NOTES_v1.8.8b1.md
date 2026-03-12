# WhisperJAV v1.8.8b1 (Pre-release)

**This is a beta release for testing.** It will not trigger upgrade notifications for regular users. If you encounter issues, please report them at [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues).

## What's New

### Install into your existing environment

The installer now detects your active conda or venv environment and installs directly into it — no more surprise `.venv` folders. If you've been using conda, this release respects your workflow.

```bash
conda activate my_env
python install.py          # installs into my_env, not .venv
```

Use `--local` if you specifically want a project-local `.venv`:

```bash
python install.py --local  # creates .venv even if conda is active
```

When installing into an existing environment, packages you already have (pandas, scikit-learn, etc.) are preserved automatically.

### numpy 2.x support

WhisperJAV now requires numpy >= 2.0.0. The numpy < 2.0 pin has been removed. All internal code has been updated to use numpy 2.x compatible APIs.

### Apple Silicon fix

Fixed a crash when using beam search on MPS (Apple Silicon). WhisperJAV now automatically uses greedy decoding on MPS devices. (#198)

### Cleaner startup

Suppressed the `RequestsDependencyWarning` that appeared on every launch due to a version check mismatch between `requests` and `chardet`. The warning was cosmetic — everything worked fine, but it looked concerning.

## Changes

### Installation
- **Environment detection**: `install.py` detects `VIRTUAL_ENV` and `CONDA_PREFIX` and installs there instead of always creating `.venv`
- **`--local` flag**: Forces project-local `.venv` even when a conda/venv is active
- **`--inexact` automatic**: When targeting an existing environment, uv won't remove unrelated packages
- **Python version safety**: If your active environment has an unsupported Python (outside 3.10-3.12), the installer warns and falls back to `.venv`
- **uv sync migration**: Replaced the old 6-stage pip install pipeline with a single `uv sync` command. Faster, more reliable, lockfile-pinned
- **Upgrade path**: `upgrade.py` now also respects active environments for source installs
- **Removed `.python-version`**: This file was causing uv to download its own Python 3.11 instead of using yours

### Bug Fixes
- **fix(#198)**: Force greedy decoding on MPS to prevent beam search crash (`transformers_asr.py`)
- **fix**: Remove `np.int_` usage from `metadata_manager.py` (crashes on numpy 2.x)
- **fix**: Correct numpy 2.x lower bounds across all dependency sources
- **fix**: Align `silero-vad>=6.2` across pyproject.toml, requirements.txt, and registry
- **fix**: GPU protection for upgrades — pip upgrade paths now use `--constraint gpu_constraints.txt`
- **fix**: Graceful error handling when pip commands fail during installation
- **fix**: Suppress `RequestsDependencyWarning` from requests/chardet version mismatch

### Documentation & Testing
- Dependency cross-match table: 81 packages validated across pyproject.toml, requirements.txt, and registry
- 65-test validation suite for dependency consistency
- Installation system proposal and state review documents

## Installation (Source Install)

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- FFmpeg in your PATH
- An NVIDIA GPU with CUDA 11.8+ drivers (optional, CPU works too)

### Fresh install

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.8b1
python install.py
```

The installer will:
1. Auto-install `uv` if you don't have it
2. Detect your GPU and select the right PyTorch build
3. Install all dependencies from the lockfile (fast, reproducible)
4. Verify the installation works

### Install into a conda environment

```bash
conda create -n whisperjav python=3.12 -y
conda activate whisperjav
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.8b1
python install.py
```

After installation, `whisperjav` and `whisperjav-gui` commands are available directly — no need to activate a separate `.venv`.

### Install into an existing venv

```bash
python -m venv ~/my-wj-env
source ~/my-wj-env/bin/activate    # Windows: my-wj-env\Scripts\activate
cd /path/to/whisperjav
git checkout v1.8.8b1
python install.py
```

### Upgrade from v1.8.7

```bash
cd /path/to/whisperjav
git pull
python install.py
```

Or if you use the built-in upgrade command:

```bash
whisperjav-upgrade
```

### Common options

| Flag | Effect |
|------|--------|
| `--local` | Force project-local `.venv` (ignore active conda/venv) |
| `--cpu-only` | Skip GPU detection, install CPU-only PyTorch |
| `--cuda cu118` | Use a specific CUDA version (cu118, cu124, cu128) |
| `--minimal` | Transcription only (no GUI, translation, or enhancement) |
| `--dev` | Include development tools (pytest, ruff, pre-commit) |
| `--no-local-llm` | Skip the local LLM translation prompt |
| `--skip-preflight` | Skip network/disk/version checks |

### Verify your installation

```bash
whisperjav --help
whisperjav-gui          # should open the GUI window
python -c "import torch; print(torch.cuda.is_available())"
```

### Troubleshooting

- **"uv not found"**: The installer tries to auto-install uv via pip. If that fails, install manually: `pip install uv`
- **Wrong Python version warning**: Make sure your active environment uses Python 3.10-3.12. The installer will warn and fall back to `.venv` if not.
- **GPU not detected**: Check that `nvidia-smi` works in your terminal. Use `--cpu-only` as a fallback.
- **Installation log**: Check `install_log.txt` in the project directory for detailed output.

## Compatibility

```
Python:       3.10, 3.11, 3.12
PyTorch:      2.4.0 - 2.10.x
CUDA:         11.8+ (12.4+ recommended)
NumPy:        2.0+
```

## Windows installer (.exe)

No new `.exe` installer for this pre-release. The conda-constructor installer will be updated in a future release (Phase 1b). This pre-release is for source-install testing only.

## Known Issues

- **LLM translation "No matches found"** (#196/#212/#132): Root cause identified (model output quality, not context overflow). Fix planned for v1.9.0 with Ollama backend migration.
- **stable_whisper SyntaxWarning**: A harmless warning from the upstream stable-ts fork about an invalid escape sequence in a docstring. Does not affect functionality.
