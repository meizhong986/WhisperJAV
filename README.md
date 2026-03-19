# WhisperJAV

<p align="center">
  <a href="https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_edition_expert.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://kaggle.com/kernels/welcome?src=https://github.com/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_kaggle_parallel_edition.ipynb">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"/>
  </a>
  <a href="https://buymeacoffee.com/meizhong">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee">
  </a>
</p>

A subtitle generator for Japanese Adult Videos.

**Documentation:** [English](https://meizhong986.github.io/WhisperJAV/) | [简体中文](https://meizhong986.github.io/WhisperJAV/zh/)

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

### Processing Modes

Seven pipelines, each with different tradeoffs. Scene detection, speech enhancement, and speech segmenter are configurable for all pipelines that support them — the table shows defaults.

| Pipeline | Backend | Scene Detection | Speech Enhancer | Speech Segmenter | Best For |
|----------|---------|-----------------|-----------------|------------------|----------|
| **faster** | Faster-Whisper (turbo) | — | — | — | Speed, clean audio |
| **fast** | OpenAI Whisper | Auditok | — | — | General use, mixed quality |
| **balanced** | Faster-Whisper | Auditok | Configurable | Silero | Default. Noisy, dialogue-heavy |
| **fidelity** | OpenAI Whisper | Auditok | Configurable | Silero | Maximum accuracy, slower |
| **transformers** | HuggingFace Kotoba | Optional | Configurable | Optional | Kotoba Japanese models |
| **qwen** | Qwen3-ASR | Semantic | Configurable | Silero | Qwen ASR with forced alignment |
| **anime** | anime-whisper | Semantic | Configurable | TEN | Anime/JAV-tuned dialogue |

### Sensitivity Settings

- **Conservative**: Higher thresholds, fewer hallucinations. Good for noisy content.
- **Balanced**: Default. Works for most content.
- **Aggressive**: Lower thresholds, catches more dialogue. Good for whisper/ASMR content.

### ChronosJAV

ChronosJAV is a dedicated pipeline for transcribing with models that do not produce their own timestamps — LLMs, Qwen ASR, anime-whisper, Kotoba, and similar. It handles text generation and timestamp alignment as separate stages, so any model that can produce text from audio can be plugged in.

#### Qwen3-ASR

Uses [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) models (1.7B, 0.6B) for text generation with a local forced aligner for word-level timestamps. Three processing modes:

| Mode | How It Works | Best For |
|------|-------------|----------|
| **Assembly** | Text first, then align timestamps. Batches scenes up to 120s. | Most content |
| **Context-Aware** | ASR and alignment together on full scenes (30-90s). | More context per utterance |
| **VAD Slicing** (default) | Coupled ASR+alignment with step-wise fallback. | More detail, less context |

#### Anime-Whisper

Uses [`litagin/anime-whisper`](https://huggingface.co/litagin/anime-whisper), a Whisper large-v3 model fine-tuned on anime and JAV dialogue. Greedy decoding with TEN VAD segmentation for tight subtitle timing. Also supports Kotoba v2.0 and v2.1 (lighter models; v2.1 adds punctuation).

#### Future: LLM-based transcription

The decoupled architecture means any model that generates text from audio can be wired in — including future LLM-based ASR models. New models can be deployed via YAML configuration without pipeline code changes.

### Two-Pass Ensemble Mode

Runs your video through two different pipelines and merges results. Different models catch different things.

```bash
# Pass 1 with transformers, Pass 2 with balanced
whisperjav video.mp4 --ensemble --pass1-pipeline transformers --pass2-pipeline balanced

# Serial mode: finish each file before starting the next
whisperjav video.mp4 --ensemble --ensemble-serial --pass1-pipeline balanced --pass2-pipeline fidelity
```

**Merge strategies:**
- `pass1_primary` (default) / `pass2_primary`: Prioritize one pass, fill gaps from other
- `smart_merge`: Intelligent overlap detection
- `full_merge`: Combine everything from both passes
- `pass1_overlap` / `pass2_overlap`: Overlap-aware priority merge
- `longest`: Keep whichever pass produced the longer subtitle for each segment

**Ensemble presets**: Save, load, and delete named ensemble configurations from the GUI. Reuse your tuned settings across sessions and across different pipeline combinations.

**Serial mode** (`--ensemble-serial`): Completes each file fully (Pass 1 → Pass 2 → Merge) before starting the next. See results as they finish instead of waiting for the entire batch.

**BYOP: Faster Whisper XXL** (v1.8.9+): Use [PurfView's Faster Whisper XXL](https://github.com/Purfview/whisper-standalone-win) as Pass 2 in ensemble mode. Select "XXL Faster Whisper" as the Pass 2 pipeline, point to your `faster-whisper-xxl.exe`, and add any extra args. CLI: `--pass2-pipeline xxl --xxl-exe /path/to/faster-whisper-xxl.exe`

### Speech Enhancement

Pre-process audio per-scene after scene detection. Use surgically — audio processing that alters the mel-spectrogram can introduce artefacts.

```bash
# ClearVoice denoising (48kHz, best quality)
whisperjav video.mp4 --mode balanced --pass1-speech-enhancer clearvoice

# FFmpeg DSP filters (lightweight, always available)
whisperjav video.mp4 --mode balanced --pass1-speech-enhancer ffmpeg-dsp:loudnorm,denoise

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

### Output Formats

SRT (default) and WebVTT for HTML5 video players:

```bash
whisperjav video.mp4 --output-format vtt
whisperjav video.mp4 --output-format both    # generates .srt and .vtt
```

Also available as a dropdown in the GUI Advanced Options tab.

### GUI

The GUI has four tabs:

1. **Transcription Mode**: Pipeline, sensitivity, model, language
2. **Advanced Options**: Output format, scene detection method, debug settings
3. **Ensemble Mode**: Two-pass configuration with presets, serial mode, and per-pass parameter customization
4. **AI SRT Translate**: Translate existing subtitle files

Settings persist across application restarts.

### AI Translation

Generate subtitles and translate them in one step:

```bash
# Generate and translate
whisperjav video.mp4 --translate

# Or translate existing subtitles
whisperjav-translate -i subtitles.srt --provider deepseek
```

Supports Ollama (local, recommended), DeepSeek (cheap), Gemini (free tier), Claude, GPT-4, OpenRouter, GLM, Groq, and local LLMs.

#### Ollama Translation (Recommended for Local)

Run translation locally using [Ollama](https://ollama.com/) — no cloud API, no API key required:

```bash
whisperjav-translate -i subtitles.srt --provider ollama
```

OllamaManager auto-starts the server, detects your GPU, and picks the best model for your VRAM:

| VRAM | Recommended Model |
|------|-------------------|
| CPU only | qwen2.5:3b |
| 8 GB | qwen2.5:7b |
| 12 GB | gemma3:12b |
| 16 GB+ | qwen2.5:14b |

#### Local LLM Translation (Legacy)

Run translation entirely on your GPU — no cloud API, no API key required:

```bash
whisperjav-translate -i subtitles.srt --provider local
```

**Zero-Config Setup**: On first use, WhisperJAV automatically downloads and installs `llama-cpp-python` (~700MB). No manual installation needed. Batch size auto-adjusts to your model's context window.

Available models:
| Model | VRAM | Notes |
|-------|------|-------|
| `llama-8b` | 6GB+ | **Default** — Llama 3.1 8B |
| `gemma-9b` | 8GB+ | Gemma 2 9B (alternative) |
| `llama-3b` | 3GB+ | Llama 3.2 3B (low VRAM only) |
| `auto` | varies | Auto-selects based on available VRAM |

**Resume Support**: If translation is interrupted, just run the same command again. It automatically resumes from where it left off using the `.subtrans` project file.

### Supported Input Formats

Any format FFmpeg can read: MP4, MKV, AVI, MOV, WMV, FLV, WAV, MP3, FLAC, M4A, M4B (audiobooks), and many more.

---

## What Makes It Work for JAV

### Scene Detection
Splits audio at natural breaks instead of forcing fixed-length chunks. This prevents cutting off sentences mid-word.

Four methods are available:
- **Semantic** (default): Texture-based clustering using MFCC features, groups acoustically similar segments together
- **Auditok**: Energy-based detection, fast and reliable
- **Silero**: Neural VAD-based detection, better for noisy audio
- **TEN**: Used by ChronosJAV pipeline for tight subtitle timing

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

| Content Type | Pipeline | Sensitivity | Notes |
|--------------|----------|-------------|-------|
| Drama / Dialogue Heavy | balanced | aggressive | Full pipeline with Silero VAD |
| Anime / JAV Dialogue | anime | aggressive | anime-whisper model with TEN VAD |
| Group Scenes | faster | conservative | Speed matters, less precision needed |
| Amateur / Homemade | fast | conservative | Variable audio quality |
| ASMR / VR / Whisper | fidelity | aggressive | Maximum accuracy for quiet speech |
| Heavy Background Music | balanced | conservative | VAD helps filter music |
| Maximum Accuracy | ensemble | varies | anime + balanced, or two different pipelines |

---

## Installation

> **Upgrading?** Run `whisperjav-upgrade` (works on Windows, Linux, and macOS). For code-only updates: `whisperjav-upgrade --wheel-only`. See the [Upgrade Guide](docs/en/UPGRADE.md) for details.

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

**Download:** [**Latest Windows Installer**](https://github.com/meizhong986/WhisperJAV/releases/latest)

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

For the full walkthrough, see [docs/en/guides/installation_windows_python.md](docs/en/guides/installation_windows_python.md).

---

### macOS (Apple Silicon)

**Prerequisites:**
```bash
xcode-select --install                    # Xcode Command Line Tools
brew install python@3.12 ffmpeg portaudio git  # Or python@3.11
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

For the full walkthrough, see [docs/en/guides/installation_mac_apple_silicon.md](docs/en/guides/installation_mac_apple_silicon.md).

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

For the full walkthrough including Colab/Kaggle setup, headless servers, and systemd services, see [docs/en/guides/installation_linux.md](docs/en/guides/installation_linux.md).

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

# All modes: faster, fast, balanced, fidelity, transformers, qwen
whisperjav video.mp4 --mode fidelity

# Output format (SRT, VTT, or both)
whisperjav video.mp4 --output-format vtt
whisperjav video.mp4 --output-format both --output-dir ./subtitles

# Two-pass ensemble
whisperjav video.mp4 --ensemble --pass1-pipeline transformers --pass2-pipeline balanced
whisperjav video.mp4 --ensemble --ensemble-serial --merge-strategy longest

# Batch processing
whisperjav /path/to/folder --output-dir ./subtitles
whisperjav /path/to/folder --skip-existing    # Resume interrupted batch

# Translation
whisperjav video.mp4 --translate --translate-provider deepseek
whisperjav-translate -i subtitles.srt --provider local

# Debugging
whisperjav video.mp4 --debug --keep-temp
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

- Chen, Y., et al. (2025). "ChronusOmni: Improving Time Awareness of Omni Large Language Models." arXiv:2512.09841. *(Inspiration for the ChronosJAV pipeline)*
- Roll, N., et al. (2025). "In-Context Learning Boosts Speech Recognition via Human-like Adaptation to Speakers and Language Varieties." arXiv:2505.14887.
- Wang, Y., et al. (2025). "Calm-Whisper: Reduce Whisper Hallucination On Non-Speech By Calming Crazy Heads Down." Interspeech 2025. arXiv:2505.12969.
- Barański, M., et al. (2025). "Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio." arXiv:2501.11378.
- Koenecke, A., et al. (2024). "Careless Whisper: Speech-to-Text Hallucination Harms." ACM FAccT 2024.
- Yang, X., et al. (2024). "PromptASR for Contextualized ASR with Controllable Style." ICASSP 2024. arXiv:2309.07414.
- Bain, M., et al. (2023). "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio." arXiv:2303.00747.


## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - The underlying ASR model
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) - Qwen-based ASR with forced alignment
- [stable-ts](https://github.com/jianfch/stable-ts) - Timestamp refinement
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized CTranslate2 inference
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Transformers pipeline backend
- [Anime-Whisper](https://huggingface.co/litagin/anime-whisper) - Anime/JAV-tuned speech model (ChronosJAV pipeline)
- [Kotoba-Whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) - Japanese-optimized Whisper model
- [PySubtrans](https://github.com/machinewrapped/llm-subtrans) - AI-powered subtitle translation engine
- The testing community for feedback and bug reports

---

## Disclaimer

This tool generates accessibility subtitles. Users are responsible for compliance with applicable laws regarding the content they process.
