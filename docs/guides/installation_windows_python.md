# WhisperJAV v1.8.3 -- Windows Installation Guide (Python Source)

This guide is for experienced Python developers who want to install WhisperJAV from source on Windows. If you are looking for the standalone installer (no Python required), see the [Releases page](https://github.com/meizhong986/whisperjav/releases).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Installation (Automated)](#3-installation-automated)
4. [Installation (Manual)](#4-installation-manual)
5. [Installing Specific Extras Only](#5-installing-specific-extras-only)
6. [GPU Setup (CUDA)](#6-gpu-setup-cuda)
7. [Running WhisperJAV](#7-running-whisperjav)
8. [Updating to New Versions](#8-updating-to-new-versions)
9. [Environment Variables](#9-environment-variables)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### Required Software

| Software | Version | Purpose | Download |
|----------|---------|---------|----------|
| **Python** | 3.10, 3.11, or 3.12 | Runtime | [python.org](https://www.python.org/downloads/) |
| **Git** | Any recent version | Clone repos, install git-based packages | [git-scm.com](https://git-scm.com/download/win) |
| **FFmpeg** | 6.x or 7.x recommended | Audio/video processing | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) |

### Required for GPU Acceleration

| Software | Version | Purpose | Download |
|----------|---------|---------|----------|
| **NVIDIA GPU Driver** | 450+ (for CUDA 11.8) or 570+ (for CUDA 12.8) | GPU compute | [nvidia.com](https://www.nvidia.com/Download/index.aspx) |
| **Visual C++ Redistributable** | 2015-2022 (x64) | Native library support | [microsoft.com](https://aka.ms/vs/17/release/vc_redist.x64.exe) |

### Required for GUI

| Software | Version | Purpose | Download |
|----------|---------|---------|----------|
| **Microsoft Edge WebView2** | Any | GUI rendering engine | [microsoft.com](https://go.microsoft.com/fwlink/p/?LinkId=2124703) |

### Python Version Compatibility

- **Python 3.10-3.12:** Fully supported.
- **Python 3.9:** Not supported (dropped due to `pysubtrans` dependency).
- **Python 3.13+:** Not supported (`openai-whisper` does not compile on 3.13+).

### Verifying Prerequisites

Open Command Prompt or PowerShell and run:

```cmd
python --version
git --version
ffmpeg -version
nvidia-smi
```

All four commands should produce output without errors. The `nvidia-smi` command will fail if you do not have an NVIDIA GPU, which is fine -- WhisperJAV supports CPU-only operation.

### Installing FFmpeg

FFmpeg is not bundled with Python and must be installed separately:

1. Download the **essentials** build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
2. Extract the archive (e.g., to `C:\ffmpeg`).
3. Add the `bin` directory to your system PATH:
   - Open **System Properties** > **Environment Variables**
   - Under **System variables**, find **Path**, click **Edit**
   - Click **New**, add `C:\ffmpeg\bin`
   - Click **OK** to close all dialogs
4. Open a new Command Prompt and verify: `ffmpeg -version`

Alternatively, if you use a package manager:

```cmd
REM Using Chocolatey:
choco install ffmpeg

REM Using Scoop:
scoop install ffmpeg

REM Using winget:
winget install --id=Gyan.FFmpeg -e
```

---

## 2. Environment Setup

**Important:** Always install WhisperJAV in a virtual environment or conda environment. Never install into your global Python.

### Option A: Python venv (Recommended for Most Users)

```cmd
REM Create the virtual environment
python -m venv whisperjav-env

REM Activate it (Command Prompt)
whisperjav-env\Scripts\activate

REM Activate it (PowerShell)
whisperjav-env\Scripts\Activate.ps1

REM Verify you are in the venv (should show the venv path)
where python
```

### Option B: Conda / Miniconda

```cmd
REM Create conda environment with Python 3.11
conda create -n whisperjav python=3.11 -y

REM Activate it
conda activate whisperjav

REM Verify
python --version
```

### Option C: Using an Existing Environment

If you already have a virtual environment for ML work (with PyTorch installed), you can install WhisperJAV into it. Ensure PyTorch is the CUDA version, not CPU-only:

```cmd
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

If this prints `CUDA: True`, you can skip the PyTorch installation step in Section 4.

---

## 3. Installation (Automated)

The automated installer handles GPU detection, CUDA selection, and staged package installation.

### Step 1: Clone the Repository

```cmd
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
```

### Step 2: Activate Your Environment

```cmd
REM venv:
whisperjav-env\Scripts\activate

REM conda:
conda activate whisperjav
```

### Step 3: Run the Installer

```cmd
REM Standard install (auto-detects GPU)
python install.py

REM Or use the batch wrapper:
installer\install_windows.bat
```

Both commands do the same thing. The `.bat` wrapper simply locates and runs `install.py`.

### Installer Options

```
--cpu-only              Install CPU-only PyTorch (no CUDA)
--cuda118               Install PyTorch for CUDA 11.8 (driver 450+)
--cuda128               Install PyTorch for CUDA 12.8 (driver 570+, default)
--no-speech-enhancement Skip speech enhancement packages (faster install)
--minimal               Minimal install (transcription only, no GUI/translation/enhancement)
--dev                   Install in development/editable mode (pip install -e)
--local-llm             Install local LLM translation (prebuilt wheel)
--local-llm-build       Install local LLM translation (build from source)
--no-local-llm          Skip local LLM installation
--skip-preflight        Skip disk space and network checks
--help                  Show all options
```

### Common Invocations

```cmd
REM Standard install (recommended)
python install.py

REM Force CUDA 11.8 (older GPU driver)
python install.py --cuda118

REM CPU-only (no NVIDIA GPU)
python install.py --cpu-only

REM Minimal install for quick testing
python install.py --minimal

REM Developer install (editable mode)
python install.py --dev

REM Everything including local LLM
python install.py --local-llm

REM Fast install (skip slow optional packages)
python install.py --no-speech-enhancement --no-local-llm
```

### What the Installer Does

The installer performs these steps in order:

1. **Preflight checks** -- Verifies disk space (8GB free), network connectivity, WebView2, VC++ Redistributable
2. **Prerequisites** -- Validates Python version, FFmpeg, Git
3. **GPU detection** -- Identifies NVIDIA GPU and driver version, selects CUDA version
4. **pip upgrade** -- Upgrades pip to latest
5. **PyTorch** -- Installs `torch` and `torchaudio` with the correct CUDA index URL
6. **Core dependencies** -- numpy, scipy, numba, librosa, audio/subtitle packages
7. **Whisper packages** -- openai-whisper (from GitHub), stable-ts (custom fork), faster-whisper
8. **Optional packages** -- HuggingFace Transformers, Qwen3-ASR, translation (pysubtrans, OpenAI, Gemini), VAD (Silero, TEN), speech enhancement (ClearVoice, BS-RoFormer, ModelScope)
9. **GUI packages** -- PyWebView, pythonnet, pywin32
10. **WhisperJAV** -- Installs the application itself (with `--no-deps` to preserve staged environment)
11. **Verification** -- Imports whisperjav and checks torch CUDA status

### Installation Time

| Configuration | Approximate Time | Notes |
|--------------|-----------------|-------|
| Full (with GPU) | 10-20 minutes | Depends on network speed |
| Minimal | 5-10 minutes | Transcription only |
| CPU-only | 10-15 minutes | Slightly faster (no CUDA wheels) |

A log file is saved to `install_log.txt` in the repository root.

---

## 4. Installation (Manual)

If you prefer to install packages yourself, follow these steps in order.

### Step 1: Upgrade pip

```cmd
python -m pip install --upgrade pip
```

### Step 2: Install PyTorch with CUDA

This step is critical. You must install PyTorch from the correct index URL to get GPU support.

```cmd
REM For CUDA 12.8 (driver 570+, recommended for modern GPUs)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

REM For CUDA 11.8 (driver 450+, universal fallback)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

REM For CPU only (no NVIDIA GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify the installation:

```cmd
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

If `CUDA available: True` appears, GPU acceleration is working.

### Step 3: Install Core Dependencies

```cmd
REM Scientific stack (numpy MUST be before numba)
pip install "numpy>=1.26.0" "scipy>=1.12.0" "numba>=0.59.0"
pip install "librosa>=0.10.2" soundfile pydub pyloudnorm

REM Subtitle processing
pip install pysrt srt

REM Utilities
pip install tqdm colorama requests aiofiles regex jsonschema
pip install "pydantic>=2.0,<3.0" "PyYAML>=6.0"

REM VAD (Voice Activity Detection)
pip install "silero-vad>=6.2" auditok ten-vad

REM Performance
pip install "psutil>=5.9.0" "scikit-learn>=1.3.0"
```

### Step 4: Install Whisper Packages

These must be installed AFTER PyTorch. They depend on `torch`, and since torch is already installed with CUDA, pip will not re-download a CPU version.

```cmd
REM OpenAI Whisper (main branch for latest fixes)
pip install git+https://github.com/openai/whisper@main

REM Stable-ts (custom fork for Japanese)
pip install git+https://github.com/meizhong986/stable-ts-fix-setup.git@main

REM ffmpeg-python (must use git, PyPI tarball has build issues)
pip install git+https://github.com/kkroening/ffmpeg-python.git

REM Faster-Whisper (CTranslate2 backend)
pip install "faster-whisper>=1.1.0"
```

### Step 5: Install Optional Packages

Install only the extras you need:

```cmd
REM HuggingFace (required for Qwen3-ASR and kotoba-whisper models)
pip install "huggingface-hub>=0.25.0" "transformers>=4.40.0" "accelerate>=0.26.0" hf_xet

REM Qwen3-ASR (new in v1.8.3, requires HuggingFace packages above)
pip install "qwen-asr>=0.0.6"

REM Translation
pip install "pysubtrans>=1.5.0" "openai>=1.35.0" "google-genai>=1.39.0"

REM GUI (Windows)
pip install "pywebview>=5.0.0" "pythonnet>=3.0" "pywin32>=305"

REM Speech Enhancement
pip install "modelscope>=1.20" oss2 addict "datasets>=2.14.0,<4.0" simplejson sortedcontainers packaging
pip install git+https://github.com/meizhong986/ClearerVoice-Studio.git#subdirectory=clearvoice
pip install bs-roformer-infer "onnxruntime>=1.16.0"

REM Compatibility (pyvideotrans interop)
pip install "av>=13.0.0" "imageio>=2.31.0" "imageio-ffmpeg>=0.4.9" "httpx>=0.27.0" "websockets>=13.0" "soxr>=0.3.0"

REM Analysis/Visualization
pip install matplotlib Pillow
```

### Step 6: Install WhisperJAV

```cmd
REM Standard install (from local source, preserves staged deps)
pip install --no-deps .

REM Or development/editable mode
pip install --no-deps -e .
```

The `--no-deps` flag is essential. Without it, pip would re-resolve all dependencies and potentially replace your CUDA PyTorch with a CPU version.

### Step 7: Verify

```cmd
python -c "import whisperjav; print(f'WhisperJAV {whisperjav.__version__}')"
whisperjav --help
```

---

## 5. Installing Specific Extras Only

If you only need certain features, you can install just those extras. However, because of the GPU lock-in requirement, you should always install PyTorch manually first (Step 2 above), then use `--no-deps`:

```cmd
REM WRONG: This pulls CPU PyTorch from PyPI
pip install "whisperjav[cli]"

REM RIGHT: Install PyTorch first, then WhisperJAV with no-deps
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps -e "."
pip install "whisperjav[cli]" --no-deps
```

For a clean approach using the automated installer, you can combine flags:

```cmd
REM Minimal (transcription only, no GUI/translation/enhancement)
python install.py --minimal

REM No speech enhancement (faster install)
python install.py --no-speech-enhancement
```

### Available Extras

| Extra | Contents | Use Case |
|-------|----------|----------|
| `cli` | numpy, scipy, librosa, VAD, scikit-learn | CLI audio processing |
| `gui` | pywebview, pythonnet, pywin32 | GUI application |
| `translate` | pysubtrans, openai, google-genai | AI subtitle translation |
| `llm` | uvicorn, fastapi | Local LLM server |
| `enhance` | modelscope, clearvoice, bs-roformer | Speech enhancement |
| `huggingface` | transformers, accelerate, hf_xet | HuggingFace model support |
| `qwen` | qwen-asr (+ huggingface deps) | Qwen3-ASR pipeline (v1.8.3+) |
| `analysis` | matplotlib, Pillow | Visualization tools |
| `compatibility` | av, imageio, httpx, websockets, soxr | pyvideotrans interop |
| `dev` | pytest, ruff, pre-commit | Development tools |
| `all` | Everything above | Full installation |
| `colab` | cli + translate + huggingface | Google Colab |
| `windows` | Same as `all` | Windows full experience |

---

## 6. GPU Setup (CUDA)

### Determining Your CUDA Version

The CUDA version for PyTorch depends on your NVIDIA driver version, NOT the CUDA Toolkit version installed on your system.

```cmd
REM Check your driver version
nvidia-smi
```

Look for the "Driver Version" in the output header:

```
+---------------------------+
| NVIDIA-SMI 570.xx.xx      |   <-- This is your driver version
| Driver Version: 570.xx.xx |
| CUDA Version: 12.8        |   <-- Maximum CUDA version supported
+---------------------------+
```

### Driver to CUDA Mapping

| Driver Version | Recommended `--index-url` | Flag |
|---------------|--------------------------|------|
| 570+ | `https://download.pytorch.org/whl/cu128` | `--cuda128` (default) |
| 450-569 | `https://download.pytorch.org/whl/cu118` | `--cuda118` |
| Below 450 | `https://download.pytorch.org/whl/cpu` | `--cpu-only` |

### Verifying CUDA Works

After installation, verify CUDA is operational:

```cmd
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"
```

### Switching Between CUDA Versions

If you installed the wrong CUDA version, uninstall and reinstall:

```cmd
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### CUDA Toolkit (Not Usually Needed)

You do NOT need to install the CUDA Toolkit separately for WhisperJAV. PyTorch bundles its own CUDA runtime. The CUDA Toolkit is only needed if you are building packages from source (e.g., `--local-llm-build`).

If needed: [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)

---

## 7. Running WhisperJAV

### CLI Usage

```cmd
REM Basic transcription
whisperjav video.mp4

REM With mode selection
whisperjav video.mp4 --mode balanced    # Full pipeline (recommended)
whisperjav video.mp4 --mode fast        # Scene detection + standard Whisper
whisperjav video.mp4 --mode faster      # Direct Faster-Whisper (fastest)

REM With sensitivity
whisperjav video.mp4 --mode balanced --sensitivity aggressive

REM With translation
whisperjav video.mp4 --translate

REM With Qwen3-ASR pipeline (new in v1.8.3)
whisperjav video.mp4 --mode qwen --input-mode assembly

REM Process a directory
whisperjav /path/to/videos/ --mode balanced

REM See all options
whisperjav --help
```

### GUI Usage

```cmd
REM Launch the GUI
whisperjav-gui
```

Requirements for GUI:
- Microsoft Edge WebView2 Runtime
- The `[gui]` extra installed (included in default installation)

### Translation CLI

```cmd
REM Translate existing subtitles
whisperjav-translate -i subtitles.srt

REM See translation options
whisperjav-translate --help
```

### Running from Source (Without Installing)

If you are developing and have not run `pip install`:

```cmd
python -m whisperjav.main video.mp4 --mode balanced
python -m whisperjav.webview_gui.main
python -m whisperjav.translate.cli -i subtitles.srt
```

---

## 8. Updating to New Versions

### Method 1: Git Pull + Reinstall (Development Mode)

If you installed in editable mode (`--dev`):

```cmd
cd whisperjav
git pull
pip install --no-deps -e .
```

This updates to the latest code without re-downloading dependencies. If the new version adds new dependencies, you may need to install them separately or re-run `python install.py`.

### Method 2: Full Reinstall

```cmd
cd whisperjav
git pull
python install.py
```

This re-runs the full installer, which will upgrade packages as needed.

### Method 3: Upgrade Command (for pip-installed)

```cmd
REM Upgrade WhisperJAV only (no dependency changes)
pip install -U --no-deps git+https://github.com/meizhong986/whisperjav.git

REM Upgrade with all dependencies (may change PyTorch -- use with caution)
pip install -U "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git"
```

### Method 4: Built-in Upgrade Tool

```cmd
REM Check for updates
whisperjav-upgrade --check

REM Interactive upgrade
whisperjav-upgrade

REM Upgrade package only, skip dependencies
whisperjav-upgrade --wheel-only
```

---

## 9. Environment Variables

WhisperJAV respects the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPERJAV_DEBUG` | `0` | Set to `1` to enable GUI debug mode (DevTools) |
| `WHISPERJAV_NO_ICON` | `0` | Set to `1` to skip icon loading (debug rendering issues) |
| `WHISPERJAV_CACHE_DIR` | `.whisperjav_cache` | Cache directory for metadata |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace model cache location |
| `TORCH_HOME` | `~/.cache/torch` | PyTorch model cache location |
| `CUDA_VISIBLE_DEVICES` | All GPUs | Restrict to specific GPU (e.g., `0`) |

### Setting Environment Variables (Windows)

```cmd
REM Temporary (current session only)
set WHISPERJAV_DEBUG=1

REM Permanent (PowerShell, user scope)
[Environment]::SetEnvironmentVariable("WHISPERJAV_DEBUG", "1", "User")
```

---

## 10. Troubleshooting

### PyTorch / CUDA Issues

**Problem: `torch.cuda.is_available()` returns `False`**

Causes and fixes:

1. **Wrong PyTorch version installed (CPU instead of CUDA):**
   ```cmd
   pip uninstall torch torchaudio -y
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

2. **Driver too old for selected CUDA version:**
   ```cmd
   REM Check driver version
   nvidia-smi
   REM If driver < 570, use CUDA 11.8
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **NVIDIA driver not installed:**
   Download from [nvidia.com](https://www.nvidia.com/Download/index.aspx).

**Problem: `RuntimeError: CUDA out of memory`**

- WhisperJAV models require ~3-4 GB of GPU VRAM for large-v2 model.
- Close other GPU-intensive applications.
- Try a smaller model: `whisperjav video.mp4 --model medium`
- Use `CUDA_VISIBLE_DEVICES=0` if you have multiple GPUs.

### pip / Package Installation Issues

**Problem: `pip install` fails with "Could not build wheels"**

```cmd
REM Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

REM Install Visual C++ Build Tools if needed
REM Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

**Problem: Dependency conflicts**

```cmd
REM Start fresh in a new venv
deactivate
rmdir /s /q whisperjav-env
python -m venv whisperjav-env
whisperjav-env\Scripts\activate
python install.py
```

**Problem: `pip install git+https://...` fails with timeout**

This is common behind firewalls or VPN. The automated installer handles this automatically, but for manual installation:

```cmd
REM Configure Git with extended timeouts
git config --global http.connectTimeout 120
git config --global http.timeout 300
git config --global http.postBuffer 524288000

REM Retry the installation
pip install --timeout 120 git+https://github.com/openai/whisper@main
```

**Problem: `numpy` / `numba` import errors**

This happens when numba is installed before numpy:

```cmd
pip uninstall numpy numba -y
pip install "numpy>=1.26.0"
pip install "numba>=0.59.0"
```

### FFmpeg Issues

**Problem: `FFmpeg is not installed or not in PATH`**

```cmd
REM Verify FFmpeg is accessible
ffmpeg -version

REM If not found, add to PATH (example)
set PATH=C:\ffmpeg\bin;%PATH%

REM Or install via package manager
choco install ffmpeg
```

**Problem: `ffmpeg-python` import error**

The PyPI version of `ffmpeg-python` has build issues. Install from Git:

```cmd
pip install git+https://github.com/kkroening/ffmpeg-python.git
```

### GUI Issues

**Problem: GUI window is blank or does not open**

1. Ensure WebView2 is installed:
   Download from [microsoft.com](https://go.microsoft.com/fwlink/p/?LinkId=2124703)

2. Check pythonnet is installed:
   ```cmd
   pip install "pythonnet>=3.0"
   ```

3. Try running with debug mode:
   ```cmd
   set WHISPERJAV_DEBUG=1
   whisperjav-gui
   ```

**Problem: `ImportError: No module named 'webview'`**

```cmd
pip install "pywebview>=5.0.0"
```

### Speech Enhancement Issues

**Problem: ModelScope download fails**

ModelScope downloads models from China CDN. If you are outside China, downloads may be slow. The `oss2` package is required:

```cmd
pip install oss2 "modelscope>=1.20"
```

**Problem: `datasets` version conflict**

```cmd
pip install "datasets>=2.14.0,<4.0"
```

`datasets>=4.0` is incompatible with ModelScope.

### Qwen3-ASR Issues (v1.8.3)

**Problem: `ImportError: No module named 'qwen_asr'`**

```cmd
pip install "qwen-asr>=0.0.6"
```

Note: The pip package name is `qwen-asr` (hyphen) but the Python import is `qwen_asr` (underscore).

**Problem: Qwen model download is slow**

Models are downloaded from HuggingFace. Set a mirror if needed:

```cmd
set HF_ENDPOINT=https://hf-mirror.com
```

### General Tips

1. **Always check the log file:** After running `python install.py`, check `install_log.txt` in the repository root for detailed error messages.

2. **Clear pip cache:** If packages seem corrupted:
   ```cmd
   pip cache purge
   ```

3. **Check which Python is being used:**
   ```cmd
   where python
   python -c "import sys; print(sys.executable)"
   ```
   Ensure this points to your venv/conda Python, not the system Python.

4. **Verify your environment:**
   ```cmd
   python -c "
   import sys
   print(f'Python: {sys.version}')
   print(f'Executable: {sys.executable}')
   print(f'Prefix: {sys.prefix}')
   print(f'In venv: {sys.prefix != sys.base_prefix}')
   "
   ```

5. **First transcription takes extra time:** WhisperJAV downloads AI models (~1-3 GB) on first use. This is a one-time download cached in `~/.cache/huggingface/`.

---

## Appendix: Disk Space Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Python packages | ~4-6 GB | PyTorch is the largest (~2 GB) |
| Whisper model (large-v2) | ~3 GB | Downloaded on first use |
| Qwen3-ASR model | ~2-3 GB | Downloaded on first use (if using Qwen mode) |
| Speech enhancement models | ~1 GB | Downloaded on first use |
| **Total (recommended free space)** | **~15 GB** | Includes headroom for temp files |

## Appendix: Complete Package List

For the complete list of packages and their versions, see:

- `pyproject.toml` -- Extras and dependency specifications
- `whisperjav/installer/core/registry.py` -- Single source of truth for all packages
- Run `python -m whisperjav.installer.validation` to check your installation against the registry
