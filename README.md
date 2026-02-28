# WhisperJAV

<p align="center">
  <a href="https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_edition_expert.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://kaggle.com/kernels/welcome?src=https://github.com/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_kaggle_parallel_edition.ipynb">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"/>
  </a>
  <br>
  <img src="https://img.shields.io/badge/version-1.8.5-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10--3.12-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License">
</p>

A subtitle generator for Japanese Adult Videos.

---





### What is the idea: 

Transformer-based ASR architectures like Whisper suffer significant performance degradation when applied to the **spontaneous and noisy domain of JAV**. This degradation is driven by specific acoustic and temporal characteristics that defy the statistical distributions of standard training data.

#### 1. The Acoustic Profile
JAV audio is defined by "acoustic hell" and a low Signal-to-Noise Ratio (SNR), characterized by:

*   **Non-Verbal Vocalisations (NVVs):** A high density of physiological sounds (heavy breathing, gasps, sighs) and "obscene sounds" that lack clear harmonic structure.
*   **Spectral Mimicry:** These vocalizations often possess "curve-like spectrum features" that mimic the formants of fricative consonants or Japanese syllables (e.g., *fu*), acting as accidental adversarial examples that trick the model into recognizing words where none exist.
*   **Extreme Dynamics:** Volatile shifts in audio intensity, ranging from faint whispers (*sasayaki*) to high-decibel screams, which confuse standard gain control and attention mechanisms.
*   **Linguistic Variance:** The prevalence of theatrical onomatopoeia and *Role Language* (*Yakuwarigo*) containing exaggerated intonations and slang absent from standard corpora.

#### 2. Temporal Drift and Hallucination
While standard ASR models are typically trained on short, curated clips, JAV content comprises long-form media often exceeding 120 minutes. Research indicates that processing such extended inputs causes **contextual drift** and error accumulation. Specifically, extended periods of "ambiguous audio" (silence or rhythmic breathing) cause the Transformer's attention mechanism to collapse, triggering repetitive **hallucination loops** where the model generates unrelated text to fill the acoustic void.

#### 3. The Pre-processing Paradox & Fine-Tuning Risks
Standard audio engineering intuition—such as aggressive denoising or vocal separation—often fails in this domain. Because Whisper relies on specific **log-Mel spectrogram** features, generic normalization tools can inadvertently strip high-frequency transients essential for distinguishing consonants, resulting in "domain shift" and erroneous transcriptions. Consequently, audio processing requires a "surgical," multi-stage approach (like VAD clamping) rather than blanket filtering.

Furthermore, while fine-tuning models on domain-specific data can be effective, it presents a high risk of **overfitting**. Due to the scarcity of high-quality, ethically sourced JAV datasets, fine-tuned models often become brittle, losing their generalization capabilities and leading to inconsistent "hit or miss" quality outputs.




**WhisperJAV** is an attempt to address above failure points. The inference pipelines do:

1.  **Acoustic Filtering:** Deploys **scene-based segmentation** and VAD clamping under the hypothesis that distinct scenes possess uniform acoustic characteristics, ensuring the model processes coherent audio environments rather than mixed streams [1-3].
2.  **Linguistic Adaptation:** Normalizes **domain-specific terminology** and preserves onomatopoeia, specifically correcting dialect-induced tokenization errors (e.g., in *Kansai-ben*) that standard BPE tokenizers fail to parse [4, 5].
3.  **Defensive Decoding:** Tunes **log-probability thresholding** and `no_speech_threshold` to systematically discard low-confidence outputs (hallucinations), while utilizing regex filters to clean non-lexical markers (e.g., `(moans)`) from the final subtitle track [6, 7].



---

## Quick Start

### GUI (Recommended for most users)

```bash
whisperjav-gui
```

A window opens. Add your files, pick a mode, click Start.

### Command Line

```bash
# Basic usage
whisperjav video.mp4

# Specify mode and sensitivity
whisperjav audio.mp3 --mode balanced --sensitivity aggressive

# Process a folder
whisperjav /path/to/media_folder --output-dir ./subtitles
```

---

## Features

### Qwen3-ASR Pipeline (New in v1.8.3 — Preview)

A new ASR engine based on [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), available as an alternative to Whisper. It uses a Qwen language model for text generation and a separate forced aligner for word-level timestamps.

The Qwen pipeline supports three ways to process your audio:

| Mode | How It Works | Best For |
|------|-------------|----------|
| **Assembly** | Generates text first, then aligns timestamps in a separate pass. Batches scenes up to 120s. | Most content. Decoupled design means each step can be optimized independently. |
| **Context-Aware** | Runs ASR and alignment together on full scenes (30-90s). | When you need the model to "see" more context around each utterance. |
| **VAD Slicing** (default) | Coupled ASR+alignment with a step-wise fallback system. If the aligner collapses, it falls back to VAD-based grouping (up to 29s). | More detail, less context. |

Access it via the CLI (`--mode qwen`) or the Ensemble tab in the GUI.

**Known limitations (work in progress):**
- **Timestamps** — The forced aligner sometimes drifts, especially on long scenes with background music. Assembly mode mitigates this with tighter scene boundaries (120s max), but it's not as precise as we want.
- **Hallucination** — The Qwen model can hallucinate during piano music and moans. Japanese post-processing catches some of these, but it's not fully solved. Other languages don't have that post-processing yet.
- **vLLM backend** — Current implementation uses transformers. The Assembly mode is architecturally ready for vLLM (the text generation step can be swapped), but the integration isn't built yet.
- **MPS (Apple Silicon)** — The Qwen pipeline currently runs on CPU on Macs. The underlying `transformers` library supports MPS, but the forced aligner doesn't detect it yet.

### Processing Modes (Whisper)

| Mode | Backend | Scene Detection | VAD | Best For |
|------|---------|-----------------|-----|----------|
| **faster** | stable-ts (turbo) | No | No | Speed priority, clean audio |
| **fast** | stable-ts | Yes | No | General use, mixed quality |
| **balanced** | faster-whisper | Yes | Yes | Default. Noisy audio, dialogue-heavy |
| **fidelity** | OpenAI Whisper | Yes | Yes (Silero) | Maximum accuracy, slower |
| **transformers** | HuggingFace | Optional | Internal | Japanese-optimized model, customizable |

### Sensitivity Settings

- **Conservative**: Higher thresholds, fewer hallucinations. Good for noisy content.
- **Balanced**: Default. Works for most content.
- **Aggressive**: Lower thresholds, catches more dialogue. Good for whisper/ASMR content.

### Transformers Mode (New in v1.7)

Uses HuggingFace's `kotoba-tech/kotoba-whisper-v2.2` model, which is optimized for Japanese conversational speech:

```bash
whisperjav video.mp4 --mode transformers

# Customize parameters
whisperjav video.mp4 --mode transformers --hf-beam-size 5 --hf-chunk-length 20
```

**Transformers-specific options:**
- `--hf-model-id`: Model (default: `kotoba-tech/kotoba-whisper-v2.2`)
- `--hf-chunk-length`: Seconds per chunk (default: 15)
- `--hf-beam-size`: Beam search width (default: 5)
- `--hf-temperature`: Sampling temperature (default: 0.0)
- `--hf-scene`: Scene detection method (`none`, `auditok`, `silero`, `semantic`)

### Two-Pass Ensemble Mode (New in v1.7)

Runs your video through two different pipelines and merges results. Different models catch different things.

```bash
# Pass 1 with transformers, Pass 2 with balanced
whisperjav video.mp4 --ensemble --pass1-pipeline transformers --pass2-pipeline balanced

# Custom sensitivity per pass
whisperjav video.mp4 --ensemble --pass1-pipeline balanced --pass1-sensitivity aggressive --pass2-pipeline fidelity
```

**Merge strategies:**
- `smart_merge` (default): Intelligent overlap detection
- `pass1_primary` / `pass2_primary`: Prioritize one pass, fill gaps from other
- `full_merge`: Combine everything from both passes

### Speech Enhancement tools (New in v1.7.3)

Pre-process audio scenes. When selected runs per-scene after scene detection.
Note: Only use for surgical reasons. In general any audio processing that may alter mel-spectogram has the potential to introduce more artefacts and hallucination.

```bash
# ClearVoice denoising (48kHz, best quality)
whisperjav video.mp4 --mode balanced --pass1-speech-enhancer clearvoice

# ClearVoice with specific 16kHz model
whisperjav video.mp4 --mode balanced --pass1-speech-enhancer clearvoice:FRCRN_SE_16K

# FFmpeg DSP filters (lightweight, always available)
whisperjav video.mp4 --mode balanced --pass1-speech-enhancer ffmpeg-dsp:loudnorm,denoise

# ZipEnhancer (lightweight SOTA)
whisperjav video.mp4 --mode balanced --pass1-speech-enhancer zipenhancer

# BS-RoFormer vocal isolation
whisperjav video.mp4 --mode balanced --pass1-speech-enhancer bs-roformer

# Ensemble with different enhancers per pass
whisperjav video.mp4 --ensemble \
    --pass1-pipeline balanced --pass1-speech-enhancer clearvoice \
    --pass2-pipeline transformers --pass2-speech-enhancer none
```

**Available backends:**

| Backend | Description | Models/Options |
|---------|-------------|----------------|
| `none` | No enhancement (default) | - |
| `ffmpeg-dsp` | FFmpeg audio filters | `loudnorm`, `denoise`, `compress`, `highpass`, `lowpass`, `deess` |
| `clearvoice` | ClearerVoice denoising | `MossFormer2_SE_48K` (default), `FRCRN_SE_16K` |
| `zipenhancer` | ZipEnhancer 16kHz | `torch` (GPU), `onnx` (CPU) |
| `bs-roformer` | Vocal isolation | `vocals`, `other` |

**Syntax:** `--pass1-speech-enhancer <backend>` or `--pass1-speech-enhancer <backend>:<model>`

### GUI Parameter Customization

The GUI has three tabs:

1. **Transcription Mode**: Select pipeline, sensitivity, language
2. **Advanced Options**: Model override, scene detection method, debug settings
3. **Two-Pass Ensemble**: Configure both passes with full parameter customization via JSON editor

The Ensemble tab lets you customize beam size, temperature, VAD thresholds, and other ASR parameters without editing config files.

### AI Translation

Generate subtitles and translate them in one step:

```bash
# Generate and translate
whisperjav video.mp4 --translate

# Or translate existing subtitles
whisperjav-translate -i subtitles.srt --provider deepseek
```

Supports DeepSeek (cheap), Gemini (free tier), Claude, GPT-4, OpenRouter, GLM, Groq, and local LLMs.

#### Local LLM Translation (New in v1.8)

Run translation entirely on your GPU - no cloud API, no API key required:

```bash
whisperjav-translate -i subtitles.srt --provider local
```

**Zero-Config Setup**: On first use, WhisperJAV automatically downloads and installs `llama-cpp-python` (~700MB). No manual installation needed.

Available models:
| Model | VRAM | Notes |
|-------|------|-------|
| `llama-8b` | 6GB+ | **Default** - Llama 3.1 8B |
| `gemma-9b` | 8GB+ | Gemma 2 9B (alternative) |
| `llama-3b` | 3GB+ | Llama 3.2 3B (low VRAM only) |
| `auto` | varies | Auto-selects based on available VRAM |

```bash
# Use specific model
whisperjav-translate -i subtitles.srt --provider local --model gemma-9b

# Control GPU offloading
whisperjav-translate -i subtitles.srt --provider local --translate-gpu-layers 32
```

**Resume Support**: If translation is interrupted, just run the same command again. It automatically resumes from where it left off using the `.subtrans` project file.

---

## What Makes It Work for JAV

### Scene Detection
Splits audio at natural breaks instead of forcing fixed-length chunks. This prevents cutting off sentences mid-word.

Three methods are available:
- **Auditok** (default): Energy-based detection, fast and reliable
- **Silero**: Neural VAD-based detection, better for noisy audio
- **Semantic** (new in v1.7.4): Texture-based clustering using MFCC features, groups acoustically similar segments together

### Voice Activity Detection (VAD)
Identifies when someone is actually speaking vs. background noise or music. Reduces false transcriptions during quiet moments.

### Japanese Post-Processing
- Handles sentence-ending particles (ね, よ, わ, の)
- Preserves aizuchi (うん, はい, ええ)
- Recognizes dialect patterns (Kansai-ben, feminine/masculine speech)
- Filters out common Whisper hallucinations

### Hallucination Removal
Whisper sometimes generates repeated text or phrases that weren't spoken. WhisperJAV detects and removes these patterns.

---

## Content-Specific Recommendations

| Content Type | Mode | Sensitivity | Notes |
|--------------|------|-------------|-------|
| Drama / Dialogue Heavy | balanced | aggressive | Or try transformers mode |
| Group Scenes | faster | conservative | Speed matters, less precision needed |
| Amateur / Homemade | fast | conservative | Variable audio quality |
| ASMR / VR / Whisper | fidelity | aggressive | Maximum accuracy for quiet speech |
| Heavy Background Music | balanced | conservative | VAD helps filter music |
| Maximum Accuracy | ensemble | varies | Two-pass with different pipelines |

---

## Installation

> **If upgrading from v1.7.x or earlier**, a `--wheel-only` upgrade won't be enough — you'll be missing packages that the Qwen pipeline needs. We recommend a full install. It takes longer, but it avoids half-working states where Whisper pipelines work but Qwen silently fails because a dependency is missing.
>
> Recommended: Uninstall the old version first (Settings > Apps > WhisperJAV on Windows), then install fresh. Your models and output files are stored separately and won't be lost.

---

### Which Installation Path Should I Follow?

| Your Situation | Go To |
|----------------|-------|
| **New user on Windows** wanting GUI | [Windows Standalone Installer](#windows-standalone-installer) |
| **Developer on Windows** | [Windows Source Install](#windows-source-install) |
| **macOS user** | [macOS (Apple Silicon)](#macos-apple-silicon) |
| **Linux user** | [Linux](#linux-ubuntu-debian-fedora-arch) |
| **Colab or Kaggle** | [Cloud Notebooks](#google-colab--kaggle) |
| **Developer or expert** wanting pip | [Expert Installation](#expert-installation) |

---

### Windows Standalone Installer

The easiest way. No Python knowledge needed.

**Download:** [**WhisperJAV-1.8.5-Windows-x86_64.exe**](https://github.com/meizhong986/WhisperJAV/releases/latest)

1. **Download** the `.exe` from the link above
2. **Run the installer.** No admin rights required. Installs to `%LOCALAPPDATA%\WhisperJAV`.
3. **Wait 10-20 minutes.** It downloads and configures Python, PyTorch, FFmpeg, and all dependencies.
4. **Launch** from the Desktop shortcut.
5. **First run** downloads models (~3 GB, another several minutes).

**GPU auto-detection:** The installer checks your NVIDIA driver version and picks the right PyTorch:
- Driver 570+ gets CUDA 12.8 (optimal for RTX 20/30/40/50-series)
- Driver 450-569 gets CUDA 11.8 (broad compatibility)
- No NVIDIA GPU gets CPU-only mode

---

### Windows Source Install

For people who manage their own Python environments.

**Prerequisites:** Python 3.10-3.12, Git, FFmpeg in PATH.

```batch
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

:: Full automated install (auto-detects GPU)
installer\install_windows.bat

:: Or with options:
installer\install_windows.bat --cpu-only        :: Force CPU
installer\install_windows.bat --cuda118         :: Force CUDA 11.8
installer\install_windows.bat --cuda128         :: Force CUDA 12.8
installer\install_windows.bat --local-llm       :: Include local LLM translation
```

The installer runs in 5 phases: PyTorch first (with GPU detection), then scientific stack, Whisper packages, audio/CLI tools, and optional extras. This order matters — PyTorch must be installed before anything that depends on it, or you end up with CPU-only wheels.

For the full walkthrough, see [docs/guides/installation_windows_python.md](docs/guides/installation_windows_python.md).

---

### macOS (Apple Silicon)

**Prerequisites:**
```bash
xcode-select --install                    # Xcode Command Line Tools
brew install python@3.12 ffmpeg git       # Or python@3.11
```

**Install:**
```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# Create a virtual environment (required for Homebrew Python)
python3 -m venv ~/venvs/whisperjav
source ~/venvs/whisperjav/bin/activate

# Run the macOS installer
chmod +x installer/install_mac.sh
./installer/install_mac.sh
```

**GPU acceleration:** Apple Silicon (M1/M2/M3/M4/M5) gets MPS acceleration automatically for Whisper pipelines. Use `--mode transformers` for best performance. The `balanced`, `fast`, and `faster` modes use CTranslate2 which doesn't support MPS, so those fall back to CPU.

**Qwen pipeline on Mac:** Currently runs on CPU only. The forced aligner doesn't detect MPS yet. This is a known limitation we plan to fix.

**Intel Macs:** CPU-only. No GPU acceleration available.

For the full walkthrough, see [docs/guides/installation_mac_apple_silicon.md](docs/guides/installation_mac_apple_silicon.md).

---

### Linux (Ubuntu, Debian, Fedora, Arch)

**1. Install system packages first** — these can't come from pip:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv python3-dev \
    build-essential ffmpeg git libsndfile1 libsndfile1-dev

# Fedora / RHEL
sudo dnf install -y python3 python3-pip python3-devel gcc gcc-c++ \
    ffmpeg git libsndfile libsndfile-devel

# Arch
sudo pacman -S --noconfirm python python-pip base-devel ffmpeg git libsndfile
```

For the GUI, you'll also need WebKit2GTK (`libwebkit2gtk-4.0-dev` on Ubuntu, `webkit2gtk4.0-devel` on Fedora).

**2. Install WhisperJAV:**

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# Recommended: use the install script
chmod +x installer/install_linux.sh
./installer/install_linux.sh

# With options:
./installer/install_linux.sh --cpu-only
./installer/install_linux.sh --local-llm
```

**NVIDIA GPU:** You need the NVIDIA driver (450+ or 570+) but NOT the CUDA Toolkit — PyTorch bundles its own CUDA runtime.

**PEP 668 note:** If your distro's Python is "externally managed" (Ubuntu 24.04+, Fedora 38+), you'll need a virtual environment. The install script detects this and tells you what to do.

For the full walkthrough including Colab/Kaggle setup, headless servers, and systemd services, see [docs/guides/installation_linux.md](docs/guides/installation_linux.md).

---

### Google Colab / Kaggle

Two notebooks are maintained:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_edition_expert.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_kaggle_parallel_edition.ipynb)

If you run into issues, please open a [GitHub issue](https://github.com/meizhong986/WhisperJAV/issues) with your system info, the console log, and the error output.

---

### Expert Installation

For users comfortable with Python package management. Choose the components you need.

#### Modular Installation

WhisperJAV supports modular extras. Install only what you need:

```bash
# Core only (minimal)
pip install git+https://github.com/meizhong986/whisperjav.git

# CLI with audio processing
pip install "whisperjav[cli] @ git+https://github.com/meizhong986/whisperjav.git"

# GUI support
pip install "whisperjav[gui] @ git+https://github.com/meizhong986/whisperjav.git"

# Translation support
pip install "whisperjav[translate] @ git+https://github.com/meizhong986/whisperjav.git"

# Everything
pip install "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git"
```

**Available extras:**

| Extra | Description |
|-------|-------------|
| `cli` | Audio processing, VAD, scene detection |
| `gui` | PyWebView GUI (Windows: WebView2, Linux/Mac: WebKit) |
| `translate` | AI translation (PySubtrans, OpenAI, Gemini) |
| `llm` | Local LLM server (FastAPI, llama-cpp) |
| `enhance` | Speech enhancement (ClearVoice, BS-RoFormer) |
| `huggingface` | HuggingFace Transformers pipeline |
| `analysis` | Scientific analysis, visualization |
| `all` | All of the above |
| `colab` | Optimized for Colab/Kaggle (cli + translate + huggingface) |

#### PyTorch Installation (Required First)

PyTorch must be installed before WhisperJAV. Choose your platform:

```bash
# NVIDIA GPU (CUDA 12.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# NVIDIA GPU (CUDA 11.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (MPS)
pip install torch torchaudio

# CPU only
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Development Installation

For contributing or modifying the code:

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/
```

#### Windows Source Installation

```batch
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

:: Standard install
installer\install_windows.bat

:: Options
installer\install_windows.bat --cpu-only
installer\install_windows.bat --cuda118
installer\install_windows.bat --local-llm
installer\install_windows.bat --dev
```

**Prerequisites for Windows source install:**
- Python 3.10-3.12 from [python.org](https://www.python.org/downloads/)
- FFmpeg in PATH from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
- Git from [git-scm.com](https://git-scm.com/download/win)

---

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Windows 10, macOS 11, Ubuntu 20.04 | Windows 11, macOS 14, Ubuntu 22.04 |
| **Python** | 3.10 | 3.11 |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 8 GB | 15 GB (with models) |
| **GPU** | None (CPU works) | NVIDIA RTX 2060+ or Apple Silicon |

**GPU Support:**
- NVIDIA: CUDA 11.8 or 12.8 (Windows, Linux)
- Apple Silicon: MPS acceleration for Whisper (M1/M2/M3/M4/M5). Qwen pipeline is CPU-only on Mac for now.
- AMD ROCm: Experimental (Linux only)
- CPU fallback: Works on all platforms, 5-10x slower

**Processing Time Estimates (per hour of video):**

| Hardware | Time |
|----------|------|
| NVIDIA RTX GPU | 5-10 minutes |
| Apple Silicon | 8-15 minutes |
| CPU | 30-60 minutes |

---

## CLI Reference

```bash
# Basic usage
whisperjav video.mp4
whisperjav video.mp4 --mode balanced --sensitivity aggressive

# All modes: faster, fast, balanced, fidelity, transformers
whisperjav video.mp4 --mode fidelity

# Transformers mode with custom parameters
whisperjav video.mp4 --mode transformers --hf-beam-size 5 --hf-chunk-length 20

# Two-pass ensemble
whisperjav video.mp4 --ensemble --pass1-pipeline transformers --pass2-pipeline balanced
whisperjav video.mp4 --ensemble --pass1-pipeline balanced --pass2-pipeline fidelity --merge-strategy smart_merge

# Output options
whisperjav video.mp4 --output-dir ./subtitles
whisperjav video.mp4 --subs-language english-direct

# Batch processing
whisperjav /path/to/folder --output-dir ./subtitles
whisperjav /path/to/folder --skip-existing    # Resume interrupted batch (skip already processed)

# Debugging
whisperjav video.mp4 --debug --keep-temp

# Translation
whisperjav video.mp4 --translate --translate-provider deepseek
whisperjav-translate -i subtitles.srt --provider gemini
```

Run `whisperjav --help` for all options.

---

## Troubleshooting

**FFmpeg not found**: Install FFmpeg and add it to your PATH.

**Slow processing / GPU warning**: Your PyTorch might be CPU-only. Reinstall with GPU support:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**model.bin error in faster mode**: Enable Windows Developer Mode or run as Administrator, then delete the cached model folder:
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--Systran--faster-whisper-large-v2"
```

---

## Contributing

Contributions welcome. See `CONTRIBUTING.md` for guidelines.

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
pip install -e ".[dev]"
python -m pytest tests/
python -m ruff check whisperjav/
```

---

## License

MIT License. See [LICENSE](LICENSE) file.

---

## Citation and credits

- Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio." (2025). arXiv:2501.11378.
- Calm-Whisper: Reduce Whisper Hallucination On Non-Speech By Calming Crazy Heads Down." (2025). arXiv:2505.12969.
- PromptASR for Contextualized ASR with Controllable Style." (2024). arXiv:2309.07414.
- In-Context Learning Boosts Speech Recognition." (2025). arXiv:2505.1
- Koenecke, A., et al. (2024). "Careless Whisper: Speech-to-Text Hallucination Harms." ACM FAccT 2024.
- Bain, M., et al. (2023). "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio." arXiv:2303.00747.


## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - The underlying ASR model
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) - Qwen-based ASR with forced alignment
- [stable-ts](https://github.com/jianfch/stable-ts) - Timestamp refinement
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized CTranslate2 inference
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Transformers pipeline backend
- [Kotoba-Whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) - Japanese-optimized Whisper model
- [PySubtrans](https://github.com/machinewrapped/llm-subtrans) - AI-powered subtitle translation engine
- The testing community for feedback and bug reports

---

## Disclaimer

This tool generates accessibility subtitles. Users are responsible for compliance with applicable laws regarding the content they process.
