# WhisperJAV v1.8.0 Release Notes

**Release Date:** January 2026

---

## Upgrading from v1.7.x

v1.8.0 includes significant dependency changes. The standalone installer now detects previous installations and offers to replace them.

### Windows (.exe Installer)

1. **Run the v1.8.0 installer** on your existing installation folder
2. **If previous version detected**, you'll see:
   ```
   A previous WhisperJAV installation was found.
   This is a major installation that will completely replace
   your previous WhisperJAV version.
   ```
3. **Click Yes** to replace, or **No** to choose a different folder

### macOS / Linux

```bash
# Remove old virtual environment
rm -rf whisperjav-env  # or your venv name

# Fresh install
./installer/install_linux.sh
```

### What's Preserved

Your data is stored **outside** the installation directory:
- **AI Models:** `~/.cache/huggingface/` (no re-download needed)
- **Transcription outputs:** Wherever you saved them
- **Custom instruction files:** Your documents folder

---

## New Feature: Local LLM Translation

Translate subtitles on your GPU without cloud APIs or API keys.

```bash
whisperjav-translate -i subtitles.srt --provider local
```

### How It Works

On first use, WhisperJAV downloads `llama-cpp-python` (~700MB). This is a one-time setup.

```
============================================================
  FIRST-TIME SETUP: Installing llama-cpp-python
============================================================

llama-cpp-python is required for local LLM translation.
This is a one-time download (~700MB).

Detected CUDA: cu124
Downloading llama-cpp-python from HuggingFace...
  Successfully installed!
```

### Available Models

| Model | VRAM Required | Notes |
|-------|---------------|-------|
| `llama-8b` | 6GB+ | Default - Llama 3.1 8B |
| `gemma-9b` | 8GB+ | Gemma 2 9B |
| `llama-3b` | 3GB+ | Llama 3.2 3B (for low VRAM) |
| `auto` | varies | Selects based on available VRAM |

### CLI Examples

```bash
# Auto-select model based on VRAM
whisperjav-translate -i subtitles.srt --provider local

# Use specific model
whisperjav-translate -i subtitles.srt --provider local --model gemma-9b

# Control GPU layer offloading
whisperjav-translate -i subtitles.srt --provider local --translate-gpu-layers 32
```

---

## New Translation Providers

| Provider | Model | Notes |
|----------|-------|-------|
| **GLM** | glm-4-flash | Chinese AI provider (BigModel) |
| **Groq** | llama-3.3-70b-versatile | Fast inference API |

```bash
whisperjav-translate -i subtitles.srt --provider glm
whisperjav-translate -i subtitles.srt --provider groq
```

---

## Installer Improvements

### Upgrade Detection

The installer now detects previous WhisperJAV installations and offers to replace them instead of showing a generic "folder not empty" error.

### Optional Local LLM During Installation

During installation, you'll be prompted:

```
Install local LLM translation? (y/N):
```

- **Default: No** - Faster installation, can install later
- **Yes** - Downloads prebuilt wheel (~700MB) if CUDA 12.4+ detected
- **Silent install flag:** `/InstallLocalLLM=1`

If you skip this step, the local LLM is automatically downloaded on first use of `--provider local`.

### Faster Package Installation (uv)

The installer now uses `uv` instead of `pip` for package installation:
- 10-30x faster package downloads
- Better timeout handling for slow connections
- Automatic fallback to pip if uv unavailable

### Simplified CUDA Support

PyTorch installation now targets two CUDA versions:

| Driver Version | CUDA Build |
|----------------|------------|
| 570+ | CUDA 12.8 |
| 450+ | CUDA 11.8 |

This simplifies the matrix while maintaining broad compatibility. Users with drivers 530-569 use CUDA 11.8, which works well. To get CUDA 12.8, update your driver to 570+.

### Manual Install Script Improvements

**Windows (`install_windows.bat`):**
- Added VC++ Redistributable detection with download link if missing

**Linux (`install_linux.sh`):**
- Added PEP 668 / externally-managed environment detection
- Added optional `.desktop` file creation for application menu integration

---

## Bug Fixes

| Issue | Description |
|-------|-------------|
| - | Fixed uv package manager `--progress-bar` compatibility |
| - | Fixed installer error propagation to NSIS (removed blocking `pause` commands) |
| - | Fixed `--pass1-scene-detector none` TypeError |
| - | Fixed SciPy/NumPy 2.0 version conflicts |
| - | Fixed scene detection memory usage for large files |
| - | Fixed hardcoded English references in translation instruction files |
| - | Improved error handling in local LLM server lifecycle |

---

## Technical Changes

### Python Version Support

- **Supported:** Python 3.10, 3.11, 3.12
- **Dropped:** Python 3.9 (due to pysubtrans dependency)
- **Not yet supported:** Python 3.13+ (openai-whisper incompatibility)

### Dependency Updates

- NumPy 2.0 compatibility
- SciPy >= 1.14.0
- Added `fsspec>=2025.3.0` constraint

### Scene Detection Optimizations

- Reduced memory usage for files with 100+ scenes
- Added diagnostic logging for empty auditok results
- Improved exception handling in audio loading
- Graceful degradation when Silero VAD fails

### New/Modified Files

| File | Change |
|------|--------|
| `whisperjav/translate/local_backend.py` | Local LLM server + lazy download |
| `whisperjav/translate/providers.py` | Added GLM, Groq providers |
| `whisperjav/translate/cli.py` | Added `--translate-gpu-layers` flag |
| `installer/templates/post_install.py.template` | uv support, optional LLM prompt |
| `installer/templates/custom_template.nsi.tmpl.template` | Upgrade detection, `/InstallLocalLLM` |
| `installer/install_windows.bat` | VC++ check |
| `installer/install_linux.sh` | PEP 668 check, desktop entry |

---

## Installation

### Windows (Recommended)

**[Download WhisperJAV-1.8.0-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/tag/v1.8.0)**

1. Run the installer
2. If upgrading, confirm replacement when prompted
3. Wait for post-installation (8-15 minutes)
4. Optional: Accept local LLM installation when prompted
5. Launch from desktop shortcut

### macOS (Apple Silicon & Intel)

```bash
# Install prerequisites
xcode-select --install
brew install python@3.11 ffmpeg git

# Clone and install
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

### Linux

```bash
# Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y python3-dev python3-pip build-essential ffmpeg libsndfile1 git

# Clone and install
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

### Advanced / Developer

<details>
<summary><b>Manual pip install</b></summary>

```bash
# Create environment
python -m venv whisperjav-env
source whisperjav-env/bin/activate

# Install PyTorch first
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git@main
```

</details>

<details>
<summary><b>Editable / Dev install</b></summary>

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# Windows
installer\install_windows.bat --dev

# Mac/Linux
./installer/install_linux.sh --dev
```

</details>

---

## Known Issues

- Local LLM requires CUDA 11.8+ for GPU acceleration; CPU-only is slow but functional
- First local LLM translation takes longer due to model loading (~30 seconds)
- Self-upgrade on Windows may require Administrator privileges if installed in Program Files

---

## Contributors

- **@hyiip** - Local LLM translation implementation (PR #128)
- **MeiZhong** - Development, testing, and installer improvements
- **Claude (Anthropic)** - Code assistance, documentation, and installer hardening

---

## Acknowledgments

This release incorporates feedback from users who reported installation issues, particularly around:
- Chinese network environments (GFW timeout handling)
- Upgrade scenarios from v1.7.x
- CUDA version compatibility

---

**Questions or Issues?** [Open a GitHub issue](https://github.com/meizhong986/WhisperJAV/issues)
