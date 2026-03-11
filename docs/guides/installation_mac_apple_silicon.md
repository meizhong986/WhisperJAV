# WhisperJAV Installation Guide for Mac Apple Silicon

**Version:** 1.8.3
**Platforms:** macOS 13 (Ventura) or later on Apple M1/M2/M3/M4/M5
**Last Updated:** 2026-02-10

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [MPS GPU Acceleration](#4-mps-gpu-acceleration)
5. [Running WhisperJAV](#5-running-whisperjav)
6. [Pipeline Mode Selection](#6-pipeline-mode-selection)
7. [Qwen Pipeline on Mac](#7-qwen-pipeline-on-mac)
8. [GUI Application](#8-gui-application)
9. [Local LLM Translation](#9-local-llm-translation)
10. [Performance Expectations](#10-performance-expectations)
11. [Troubleshooting](#11-troubleshooting)
12. [Known Limitations](#12-known-limitations)

---

## 1. System Requirements

### Hardware

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Chip | Apple M1 | Apple M2 Pro or later |
| Memory (RAM) | 16 GB | 32 GB or more |
| Storage | 15 GB free | 30 GB free |
| macOS | 13.0 (Ventura) | 14.0 (Sonoma) or later |

**Why memory matters:** Apple Silicon uses unified memory architecture -- the GPU and CPU share the same RAM. Whisper models require 2-6 GB depending on model size. With 8 GB total RAM, you will only be able to run the `small` or `base` models. With 16 GB, `large-v2` is feasible but tight. 32 GB is comfortable for all model sizes.

### Software

- Python 3.10, 3.11, or 3.12 (3.13+ is not compatible with openai-whisper)
- FFmpeg (for audio/video processing)
- Git (for installing packages from source)
- Xcode Command Line Tools (for compiling C extensions)

---

## 2. Prerequisites

### Step 1: Install Xcode Command Line Tools

This is required for compiling native Python packages (numpy, scipy, etc.).

```bash
xcode-select --install
```

A dialog will appear. Click "Install" and wait for it to complete (may take 5-10 minutes).

**Verify:**
```bash
xcode-select -p
# Should output: /Library/Developer/CommandLineTools
```

### Step 2: Install Homebrew

If you do not already have Homebrew installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installation, follow the on-screen instructions to add Homebrew to your PATH. For Apple Silicon Macs, Homebrew installs to `/opt/homebrew/`. You need to add it to your shell profile.

For **zsh** (default on macOS):
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
```

For **bash**:
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.bash_profile
source ~/.bash_profile
```

**Verify:**
```bash
brew --version
```

### Step 3: Install Python

macOS does not ship with a suitable Python version. Install Python 3.12 via Homebrew:

```bash
brew install python@3.12
```

**Verify:**
```bash
python3 --version
# Should output: Python 3.12.x
```

If you prefer version management, you can use `pyenv` instead:

```bash
brew install pyenv
pyenv install 3.12.8
pyenv global 3.12.8
```

When using pyenv, add this to your `~/.zshrc`:
```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

**Important:** Do NOT use the system Python that comes with macOS. It is too old and is managed by the OS.

### Step 4: Install FFmpeg and PortAudio

```bash
brew install ffmpeg portaudio
```

- **FFmpeg** is used for audio/video extraction and format conversion.
- **PortAudio** is a system library required by `pyaudio`, which is used by `auditok` for scene detection.

**Verify:**
```bash
ffmpeg -version
```

### Step 5: Install Git

Git is usually pre-installed with Xcode Command Line Tools. Verify:

```bash
git --version
```

If not found:
```bash
brew install git
```

---

## 3. Installation

### Step 1: Create a Virtual Environment

Always use a virtual environment. This prevents conflicts with system packages and other projects.

```bash
# Create virtual environment
python3 -m venv ~/venvs/whisperjav

# Activate it
source ~/venvs/whisperjav/bin/activate

# Verify you are in the venv
which python
# Should output: /Users/<you>/venvs/whisperjav/bin/python
```

You must activate this environment every time you open a new terminal to use WhisperJAV:
```bash
source ~/venvs/whisperjav/bin/activate
```

### Step 2: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 3: Clone the Repository

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
```

### Step 4: Install PyTorch with MPS Support

On Apple Silicon, PyTorch from the default PyPI index includes MPS (Metal Performance Shaders) support. Do NOT use the `--index-url` flag that is used for CUDA installations.

```bash
pip install torch torchaudio
```

**Verify MPS support:**
```bash
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    x = torch.ones(1, device='mps')
    print(f'MPS tensor test: {x} (SUCCESS)')
"
```

Expected output:
```
PyTorch version: 2.x.x
MPS available: True
MPS built: True
MPS tensor test: tensor([1.], device='mps:0') (SUCCESS)
```

If MPS is not available, see [Troubleshooting](#11-troubleshooting).

### Step 5: Install Core Dependencies

```bash
# Scientific stack (order matters: numpy before numba)
pip install "numpy>=1.26.0" "scipy>=1.12.0" "numba>=0.59.0"

# Audio processing
pip install soundfile pydub "librosa>=0.10.2" pyloudnorm

# Subtitle processing
pip install pysrt srt

# Utilities
pip install tqdm colorama requests aiofiles regex

# Configuration
pip install "pydantic>=2.0,<3.0" "PyYAML>=6.0" jsonschema

# VAD (Voice Activity Detection)
pip install pyaudio auditok "silero-vad>=6.2" ten-vad

# Performance
pip install "psutil>=5.9.0" "scikit-learn>=1.3.0"
```

### Step 6: Install Whisper Packages

```bash
# OpenAI Whisper (main branch for latest fixes)
pip install git+https://github.com/openai/whisper@main

# Stable-ts (custom fork)
pip install git+https://github.com/meizhong986/stable-ts-fix-setup.git@main

# FFmpeg Python bindings (git version, PyPI has build issues)
pip install git+https://github.com/kkroening/ffmpeg-python.git

# Faster-Whisper (CTranslate2-based -- runs on CPU only on Mac, see limitations)
pip install "faster-whisper>=1.1.0"
```

### Step 7: Install HuggingFace and Qwen Support

```bash
# HuggingFace Transformers ecosystem
pip install "huggingface-hub>=0.25.0" "transformers>=4.40.0" "accelerate>=0.26.0" hf_xet

# Qwen3-ASR (v1.8.3+)
pip install "qwen-asr>=0.0.6"
```

### Step 8: Install Optional Features

**Translation module:**
```bash
pip install "pysubtrans>=1.5.0" "openai>=1.35.0" "google-genai>=1.39.0"
```

**GUI:**
```bash
pip install "pywebview>=5.0.0"
```
Note: On macOS, pywebview uses the native WebKit engine. No additional dependencies are required.

**Compatibility layer:**
```bash
pip install "av>=13.0.0" "imageio>=2.31.0" "imageio-ffmpeg>=0.4.9" "httpx>=0.27.0" "websockets>=13.0" "soxr>=0.3.0"
```

### Step 9: Install WhisperJAV

```bash
# Standard installation (do NOT let pip re-resolve torch)
pip install --no-deps .

# OR for development mode:
pip install --no-deps -e .
```

The `--no-deps` flag is critical. It prevents pip from re-resolving dependencies and potentially replacing your MPS-enabled PyTorch with an incompatible version.

### Step 10: Verify Installation

```bash
python3 -c "import whisperjav; print(f'WhisperJAV {whisperjav.__version__} installed successfully')"
```

Run the preflight check:
```bash
whisperjav --check
```

---

## Alternative: Using the Install Script

You can also use the provided install script, but be aware of its limitations on Mac:

```bash
cd whisperjav

# Create and activate venv first
python3 -m venv ~/venvs/whisperjav
source ~/venvs/whisperjav/bin/activate

# Run the install script with CPU flag
# (On Mac, --cpu-only installs the correct PyTorch with MPS support)
python install.py --cpu-only
```

The `--cpu-only` flag tells the installer to skip NVIDIA GPU detection and use the default PyTorch index, which on Apple Silicon includes MPS support. This is the correct behavior despite the misleading flag name.

**Important:** The script will report "No NVIDIA GPU detected - using CPU". This is normal on Mac. MPS acceleration is still available at runtime.

---

## 4. MPS GPU Acceleration

### What is MPS?

MPS (Metal Performance Shaders) is Apple's GPU compute framework for Apple Silicon. It provides GPU acceleration for PyTorch operations without CUDA. WhisperJAV automatically detects and uses MPS when available.

### Verify MPS is Working

```bash
python3 -c "
from whisperjav.utils.device_detector import get_best_device, get_device_info
device = get_best_device()
print(f'Best device: {device}')
info = get_device_info()
print(f'MPS available: {info[\"mps\"][\"available\"]}')
print(f'MPS name: {info[\"mps\"][\"name\"]}')
"
```

Expected output:
```
Best device: mps
MPS available: True
MPS name: Apple Silicon (arm64)
```

### MPS Compatibility by Pipeline

| Pipeline Mode | MPS Support | Actual Device Used |
|--------------|-------------|-------------------|
| `balanced` | No | CPU (faster-whisper uses CTranslate2, no MPS support) |
| `fast` | No | CPU (uses faster-whisper backend) |
| `faster` | No | CPU (uses faster-whisper backend) |
| `transformers` | **Yes** | MPS (uses HuggingFace Transformers) |
| `qwen` | Partial | CPU (see [Section 7](#7-qwen-pipeline-on-mac)) |

---

## 5. Running WhisperJAV

### Basic CLI Usage

```bash
# Activate your virtual environment first
source ~/venvs/whisperjav/bin/activate

# Basic transcription (uses default balanced mode -- CPU on Mac)
whisperjav video.mp4

# Use transformers mode for MPS GPU acceleration (RECOMMENDED on Mac)
whisperjav video.mp4 --mode transformers

# With sensitivity control
whisperjav video.mp4 --mode transformers --sensitivity aggressive

# Specify model size (adjust based on available RAM)
whisperjav video.mp4 --mode transformers --model large-v2

# With translation
whisperjav video.mp4 --mode transformers --translate
```

### Recommended Settings for Mac

For Apple Silicon Macs, the recommended configuration is:

```bash
# 16 GB RAM Mac
whisperjav video.mp4 --mode transformers --model medium

# 32 GB+ RAM Mac
whisperjav video.mp4 --mode transformers --model large-v2

# Maximum quality (32 GB+ RAM)
whisperjav video.mp4 --mode transformers --model large-v2 --sensitivity aggressive
```

---

## 6. Pipeline Mode Selection

This section explains which pipeline mode to use on Mac.

### For GPU Acceleration: `--mode transformers`

The `transformers` pipeline uses HuggingFace Transformers, which fully supports MPS. This is the recommended mode for Mac users who want GPU acceleration.

```bash
whisperjav video.mp4 --mode transformers
```

### For Fastest CPU Processing: `--mode faster`

If you want the fastest CPU-only processing, the `faster` mode uses CTranslate2's optimized CPU inference with Apple Accelerate framework.

```bash
whisperjav video.mp4 --mode faster
```

### For Maximum Accuracy: `--mode balanced` or `--mode transformers`

The `balanced` mode provides the full preprocessing pipeline (scene detection + VAD) but runs ASR on CPU via faster-whisper. The `transformers` mode runs ASR on MPS GPU. For maximum accuracy with GPU acceleration:

```bash
whisperjav video.mp4 --mode transformers --sensitivity aggressive
```

### For Qwen Pipeline: `--mode qwen`

See [Section 7](#7-qwen-pipeline-on-mac) for details.

---

## 7. Qwen Pipeline on Mac

### Current Status (v1.8.3)

The Qwen3-ASR pipeline (`--mode qwen`) is a new feature in v1.8.3 that provides high-quality multilingual ASR. However, there is a known limitation on Mac Apple Silicon:

**The Qwen ASR module does not currently detect MPS.** When running on Mac, it falls back to CPU mode. This is a code limitation in `qwen_asr.py` where the device detection only checks for CUDA, skipping MPS.

### Running Qwen on Mac

Despite the CPU limitation, Qwen still works on Mac:

```bash
# Qwen pipeline (will run on CPU)
whisperjav video.mp4 --mode qwen

# With assembly input mode (recommended for long content)
whisperjav video.mp4 --mode qwen --input-mode assembly
```

### Memory Considerations

Qwen models are large. On Mac:

| Mac Configuration | Feasibility |
|-------------------|------------|
| 8 GB RAM | Not recommended -- model may not fit in memory |
| 16 GB RAM | Possible but tight -- close other applications |
| 32 GB RAM | Comfortable |
| 64 GB+ RAM | Optimal |

The Qwen pipeline uses decoupled text generation and alignment in assembly mode. This means models are loaded and unloaded sequentially, reducing peak memory usage compared to having both in memory simultaneously.

### Future MPS Support

MPS support for the Qwen pipeline is a known gap. The fix involves updating `_detect_device()` in `qwen_asr.py` to check `torch.backends.mps.is_available()`. This may be addressed in a future release. Check the release notes for updates.

---

## 8. GUI Application

### Running the GUI

```bash
source ~/venvs/whisperjav/bin/activate
whisperjav-gui
```

On macOS, the GUI uses the native WebKit engine (WKWebView) via pywebview. This provides a native-looking application window without any additional runtime dependencies.

### Notes

- The window icon may not display correctly because WhisperJAV ships with `.ico` format icons (Windows). macOS prefers `.icns` format. This is cosmetic only.
- The GUI launches a local web server and displays the interface in a native window. It is not a web browser.
- If you encounter rendering issues, you can set `WHISPERJAV_DEBUG=1` for debug mode with developer tools.

---

## 9. Local LLM Translation

WhisperJAV supports local LLM-based subtitle translation using llama-cpp-python with Metal backend on Apple Silicon. This allows offline translation without API keys.

### Installing Local LLM Support

During the main installation, you can add local LLM support:

```bash
python install.py --local-llm
```

Or install separately:

```bash
# The install script detects Apple Silicon and builds with Metal support
# This may take 10-15 minutes as it compiles from source
CMAKE_ARGS="-DGGML_METAL=on" pip install "llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

Alternatively, check if a prebuilt Metal wheel is available:
```bash
python3 -c "
from whisperjav.translate.llama_build_utils import get_prebuilt_wheel_url
url, desc = get_prebuilt_wheel_url(verbose=True)
if url:
    print(f'Found: {desc}')
    print(f'URL: {url}')
else:
    print('No prebuilt wheel found -- will need to build from source')
"
```

### Using Local LLM Translation

```bash
# Transcribe and translate using local LLM
whisperjav video.mp4 --mode transformers --translate --translate-provider local
```

---

## 10. Performance Expectations

### Compared to NVIDIA GPU (RTX 4090)

| Metric | RTX 4090 (CUDA) | M2 Pro (MPS) | M2 Pro (CPU) |
|--------|-----------------|--------------|--------------|
| Whisper large-v2 (transformers) | ~10x realtime | ~2-3x realtime | ~0.5x realtime |
| Whisper large-v2 (faster-whisper) | ~15x realtime | N/A (no MPS) | ~2x realtime |
| Model loading | 3-5 sec | 5-10 sec | 5-10 sec |
| Memory usage | GPU VRAM | Unified RAM | Unified RAM |

"Realtime" means the ratio of audio duration to processing time. 3x realtime means a 60-minute video processes in ~20 minutes.

### Model Size vs Memory

| Model | Size | Min RAM (Mac) | Recommended RAM |
|-------|------|---------------|-----------------|
| tiny | 39 MB | 8 GB | 8 GB |
| base | 74 MB | 8 GB | 8 GB |
| small | 244 MB | 8 GB | 16 GB |
| medium | 769 MB | 16 GB | 16 GB |
| large-v2 | 1.55 GB | 16 GB | 32 GB |
| large-v3 | 1.55 GB | 16 GB | 32 GB |

These figures include overhead for audio processing, VAD, and scene detection running simultaneously.

### Tips for Better Performance

1. **Close other applications** to free memory for the GPU.
2. **Use `--mode transformers`** for MPS acceleration.
3. **Use a smaller model** if processing is slow or you run out of memory.
4. **Process shorter videos** or use `--mode faster` for long batch jobs where speed matters more than GPU acceleration.
5. **Enable MPS memory fallback** if you encounter out-of-memory errors:
   ```bash
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   whisperjav video.mp4 --mode transformers
   ```
   This allows MPS to use system memory as fallback, preventing crashes at the cost of speed.

---

## 11. Troubleshooting

### MPS Not Available

**Symptom:** `torch.backends.mps.is_available()` returns `False`.

**Solutions:**
1. Ensure you are running macOS 13 (Ventura) or later.
2. Ensure you installed PyTorch from the default index (not a CUDA index):
   ```bash
   pip uninstall torch torchaudio
   pip install torch torchaudio
   ```
3. Verify you are on Apple Silicon:
   ```bash
   uname -m
   # Should output: arm64
   ```
4. If running under Rosetta 2 (x86 emulation), MPS is not available. Ensure you are using a native ARM64 Python:
   ```bash
   python3 -c "import platform; print(platform.machine())"
   # Should output: arm64
   ```

### "No module named 'whisperjav'" After Installation

**Symptom:** Import error when running `whisperjav` command.

**Solutions:**
1. Ensure your virtual environment is activated:
   ```bash
   source ~/venvs/whisperjav/bin/activate
   ```
2. Verify the package is installed:
   ```bash
   pip show whisperjav
   ```
3. If you installed in development mode, make sure you are in the repository directory:
   ```bash
   cd ~/path/to/whisperjav
   pip install --no-deps -e .
   ```

### FFmpeg Not Found

**Symptom:** `FileNotFoundError: ffmpeg not found` or `FFmpeg not found in PATH`.

**Solutions:**
```bash
brew install ffmpeg
# Verify
which ffmpeg
ffmpeg -version
```

If `which ffmpeg` returns nothing after installing, your shell may need reloading:
```bash
source ~/.zshrc
```

### NumPy Build Errors

**Symptom:** Compilation errors when installing numpy or scipy.

**Solutions:**
1. Ensure Xcode Command Line Tools are installed:
   ```bash
   xcode-select --install
   ```
2. Try installing pre-built wheels:
   ```bash
   pip install --only-binary=:all: "numpy>=1.26.0"
   ```

### pyaudio / PortAudio Errors

**Symptom:** `ERROR: Could not build wheels for pyaudio` or `portaudio.h: No such file or directory`

**Solution:** Install the PortAudio system library before installing pyaudio:
```bash
brew install portaudio
pip install pyaudio
```

PortAudio is required by `auditok`, which is used for scene detection.

### soundfile / libsndfile Errors

**Symptom:** `OSError: cannot load library 'libsndfile.dylib'`

**Solutions:**
```bash
brew install libsndfile
```

Then reinstall soundfile:
```bash
pip uninstall soundfile
pip install soundfile
```

### MPS Out of Memory

**Symptom:** `RuntimeError: MPS backend out of memory`

**Solutions:**
1. Use a smaller model:
   ```bash
   whisperjav video.mp4 --mode transformers --model medium
   ```
2. Enable MPS memory fallback:
   ```bash
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   ```
3. Close other applications to free memory.
4. If the problem persists, fall back to CPU mode:
   ```bash
   whisperjav video.mp4 --mode faster
   ```

### slower-than-expected Performance

**Symptom:** Processing is very slow even with MPS.

**Check:**
1. Verify you are using `--mode transformers` (not `balanced` or `faster`, which use CPU on Mac):
   ```bash
   whisperjav video.mp4 --mode transformers
   ```
2. Verify MPS is being used (look for "MPS device detected" in output or run with `--verbose`).
3. Check Activity Monitor for memory pressure. If the system is swapping, processing will be very slow.

### Homebrew Python Not Found

**Symptom:** `python3: command not found` after installing Python via Homebrew.

**Solutions:**

On Apple Silicon, Homebrew installs to `/opt/homebrew/`. Ensure it is in your PATH:

```bash
# Add to ~/.zshrc
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Then reload:
```bash
source ~/.zshrc
```

### PEP 668 Error ("externally-managed-environment")

**Symptom:** `error: externally-managed-environment`

This occurs when trying to install packages globally with Homebrew Python. Always use a virtual environment:

```bash
python3 -m venv ~/venvs/whisperjav
source ~/venvs/whisperjav/bin/activate
# Then install within the venv
```

### Git Timeout During Installation

**Symptom:** `Failed to connect to github.com port 443 after 21 ms`

**Solutions:**
1. Check your internet connection.
2. If behind a VPN or firewall, configure Git timeouts:
   ```bash
   git config --global http.connectTimeout 120
   git config --global http.timeout 300
   ```
3. Retry the installation.

---

## 12. Known Limitations

### CTranslate2 / faster-whisper Does Not Support MPS

The faster-whisper backend (used by `balanced`, `fast`, and `faster` modes) relies on CTranslate2, which only supports CUDA and CPU. On Mac, these modes automatically fall back to CPU. Use `--mode transformers` for MPS acceleration.

Reference: [CTranslate2 Issue #1562](https://github.com/OpenNMT/CTranslate2/issues/1562)

### Qwen Pipeline Runs on CPU

As of v1.8.3, the Qwen ASR module (`qwen_asr.py`) does not detect MPS and falls back to CPU on Apple Silicon. This is a code limitation, not a framework limitation -- the underlying `transformers` library supports MPS. A fix is expected in a future release.

### Speech Enhancement Backend Compatibility

The speech enhancement backends (ClearVoice, BS-RoFormer) may have limited testing on macOS. If you encounter issues:

```bash
# Skip speech enhancement during installation
python install.py --no-speech-enhancement

# Or skip at runtime
whisperjav video.mp4 --mode transformers --no-enhance
```

### No Standalone Installer for Mac

The standalone installer (conda-constructor `.exe`) is only available for Windows. Mac users must install from source using this guide.

### ONNX Runtime on Apple Silicon

The `onnxruntime` package (used by the enhance extra) may not have optimized Apple Silicon builds. If it fails to install, you can skip it -- it is only needed for speech enhancement:

```bash
pip install onnxruntime  # If this fails, speech enhancement won't work but transcription is unaffected
```

### openai-whisper MPS Stability

While OpenAI Whisper supports MPS, some operations may fall back to CPU for numerical stability. This is handled automatically by PyTorch and may result in slightly lower GPU utilization compared to CUDA. You may see warnings like:
```
UserWarning: MPS: fallback to CPU for op 'aten::...'
```
These are informational and do not indicate errors.

---

## Quick Reference Card

```bash
# === Setup (one time) ===
xcode-select --install
brew install python@3.12 ffmpeg portaudio git
python3 -m venv ~/venvs/whisperjav
source ~/venvs/whisperjav/bin/activate
pip install --upgrade pip
pip install torch torchaudio

# === Install WhisperJAV ===
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
python install.py --cpu-only

# === Daily usage ===
source ~/venvs/whisperjav/bin/activate

# Recommended for Mac (MPS GPU acceleration)
whisperjav video.mp4 --mode transformers

# GUI
whisperjav-gui

# Check environment
whisperjav --check
```
