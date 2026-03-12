# WhisperJAV v1.8.8b1 (Pre-release)

**This is a beta release for testing.** It will not trigger upgrade notifications for regular users. If you encounter issues, please report them at [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues).

## What's New

- **Smarter installation** — The installer now detects your active conda or venv environment and installs directly into it, instead of always creating a separate `.venv` folder
- **numpy 2.x support** — Lifted the numpy < 2.0 restriction. All internal code updated for numpy 2.x
- **Apple Silicon fix** — Fixed a crash when using beam search on MPS devices (#198)
- **Cleaner startup** — Removed a warning message about `requests`/`chardet` version mismatch that appeared on every launch

## Installation

### Windows — Standalone Installer (.exe)

The easiest way. No Python knowledge needed.

1. Download **WhisperJAV-1.8.8b1-Windows-x86_64.exe** from the Assets below
2. Run the installer (no admin rights required)
3. Wait 10-20 minutes for setup to complete
4. Launch from the Desktop shortcut

Your GPU is detected automatically. The installer picks the right PyTorch version for your NVIDIA driver.

### Windows — Source Install

If you manage your own Python environment.

```
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.8b1
installer\install_windows.bat
```

### macOS

```bash
# Install prerequisites (if not already installed)
xcode-select --install
brew install python@3.12 ffmpeg portaudio git

# Install WhisperJAV
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.8b1
installer/install_mac.sh
```

### Linux

```bash
# Install WhisperJAV
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.8b1
installer/install_linux.sh
```

### Google Colab / Kaggle

Use the notebooks in the `notebook/` folder, or install directly:

```python
!pip install "whisperjav[colab] @ git+https://github.com/meizhong986/whisperjav.git@v1.8.8b1"
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
git checkout v1.8.8b1
installer\install_windows.bat        # Windows
installer/install_mac.sh             # macOS
installer/install_linux.sh           # Linux
```

### After Installation

Launch the GUI:
```
whisperjav-gui
```

Or use the command line:
```
whisperjav video.mp4 --mode balanced
```

## What Changed (Technical Details)

### Installation Improvements
- Environment detection: installs into your active conda/venv instead of always creating `.venv`
- New `--local` flag to force a project-local `.venv` when you want one
- Existing packages in your environment (pandas, scikit-learn, etc.) are preserved automatically
- Python version safety check: warns and falls back to `.venv` if your environment's Python is outside 3.10-3.12
- Replaced the old 6-stage pip pipeline with `uv sync` — faster and lockfile-pinned
- `upgrade.py` also respects active environments for source installs
- Removed `.python-version` file that was causing uv to download its own Python

### Bug Fixes
- **fix(#198)**: Force greedy decoding on MPS to prevent beam search crash
- **fix**: Remove `np.int_` usage from `metadata_manager.py` (crashes on numpy 2.x)
- **fix**: Correct numpy 2.x lower bounds across all dependency sources
- **fix**: Align `silero-vad>=6.2` across pyproject.toml, requirements.txt, and registry
- **fix**: GPU protection for upgrades — pip upgrade paths now use `--constraint gpu_constraints.txt`
- **fix**: Graceful error handling when pip commands fail during installation
- **fix**: Suppress `RequestsDependencyWarning` from requests/chardet version mismatch

### Documentation & Testing
- Dependency cross-match table: 81 packages validated across pyproject.toml, requirements.txt, and registry
- 65-test validation suite for dependency consistency

## Compatibility

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 - 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |
| NumPy | 2.0+ |

## Known Issues

- **LLM translation "No matches found"** (#196/#212/#132): Root cause identified. Fix planned for v1.9.0 with Ollama backend migration.
- **stable_whisper SyntaxWarning on startup**: Harmless warning from upstream code. Does not affect functionality.

## Troubleshooting

- **Installation takes too long**: First install downloads ~2 GB of packages. Subsequent installs are much faster (cached).
- **GPU not detected**: Check that `nvidia-smi` works in your terminal. The installer will fall back to CPU mode automatically.
- **Something went wrong**: Check `install_log.txt` in the project folder for details. Include it when reporting issues.
