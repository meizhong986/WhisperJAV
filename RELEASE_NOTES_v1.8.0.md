# WhisperJAV v1.8.0 Release Notes

**Release Date:** January 2026

---

## Upgrading from v1.7.x

**v1.8.0 requires a fresh installation.** This is not an in-place upgrade due to significant dependency changes (NumPy 2.0, new PyTorch versions, new speech enhancement backends).

### Windows (.exe Installer)

1. **Uninstall v1.7.x first:**
   - Open **Settings** → **Apps** → Search "WhisperJAV" → **Uninstall**
   - Or run: `%LOCALAPPDATA%\WhisperJAV\Uninstall-WhisperJAV.exe`
2. **Install v1.8.0:** Run the new installer to the same location.

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

## Headline Feature: Local LLM Translation

Translate subtitles entirely on your GPU - no cloud API, no API key required.

```bash
whisperjav-translate -i subtitles.srt --provider local
```

### Zero-Config Setup

On first use, WhisperJAV automatically downloads and installs `llama-cpp-python` (~700MB). No manual installation needed.

```
============================================================
  FIRST-TIME SETUP: Installing llama-cpp-python
============================================================

llama-cpp-python is required for local LLM translation.
This is a one-time download (~700MB).

Detected CUDA: cu124
Downloading llama-cpp-python from HuggingFace...
  Successfully installed!

Future runs will start immediately.
```

### Available Models

| Model | VRAM | Notes |
|-------|------|-------|
| `llama-8b` | 6GB+ | **Default** - Llama 3.1 8B |
| `gemma-9b` | 8GB+ | Gemma 2 9B (alternative) |
| `llama-3b` | 3GB+ | Llama 3.2 3B (low VRAM only) |
| `auto` | varies | Auto-selects based on available VRAM |

### CLI Examples

```bash
# Auto-select model based on VRAM
whisperjav-translate -i subtitles.srt --provider local

# Use specific model
whisperjav-translate -i subtitles.srt --provider local --model gemma-9b

# Control GPU offloading
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

### Optional Local LLM During Installation

The GUI installer now prompts for local LLM installation:

```
Install local LLM translation? (y/N):
```

- **Default: No** - Skip for faster installation
- **Yes** - Downloads prebuilt wheel (~700MB) if CUDA 12.4+ detected
- Can also be set via silent install: `/InstallLocalLLM=1`

### Lazy Download Alternative

Even if you skip during installation, local LLM is automatically downloaded on first use of `--provider local`. No need to reinstall.

---

## Bug Fixes

| Issue | Description |
|-------|-------------|
| - | Fixed hardcoded English references in translation instruction files |
| - | Improved error handling in local LLM server lifecycle |

---

## Technical Details

### New/Modified Files

| File | Change |
|------|--------|
| `whisperjav/translate/local_backend.py` | Local LLM server + lazy download |
| `whisperjav/translate/providers.py` | Added GLM, Groq providers |
| `whisperjav/translate/cli.py` | `--translate-gpu-layers` flag |
| `installer/templates/post_install.py.template` | Optional local LLM prompt |
| `installer/templates/custom_template.nsi.tmpl.template` | `/InstallLocalLLM` argument |

### Download Sources (Priority Order)

1. HuggingFace (`mei986/whisperjav-wheels`) - Primary
2. JamePeng GitHub releases - Fallback
3. Manual install instructions - Last resort

---

## Installation

### Upgrading from v1.7.x?

Simply run the new installer. Your AI models, settings, and cached downloads will be preserved.

### Windows (Recommended)

**[Download WhisperJAV-1.8.0-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/tag/v1.8.0)**

### macOS / Linux

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
./installer/install_linux.sh
```

---

## Contributors

- @hyiip - Local LLM translation implementation (PR #128)
- MeiZhong - Development and testing
- Claude (Anthropic) - Code assistance and documentation

---

**Questions or Issues?** [Open a GitHub issue](https://github.com/meizhong986/WhisperJAV/issues)
