# WhisperJAV

<p align="center">
  <a href="https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_parallel_expert.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://kaggle.com/kernels/welcome?src=https://github.com/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_parallel_expert.ipynb">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"/>
  </a>
  <br>
  <img src="https://img.shields.io/badge/version-1.8.2-blue.svg" alt="Version">
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

### Processing Modes

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

Find your situation below and follow the guide.

---

### Which Installation Path Should I Follow?

| Your Situation | Go To |
|----------------|-------|
| **New user on Windows** wanting GUI | [Windows Standalone Installer](#windows-standalone-installer) |
| **Already have v1.8.x** and want to update | [Upgrading from v1.8.x](#upgrading-from-v18x) |
| **Have v1.7.x or earlier** | [Upgrading from v1.7.x or Earlier](#upgrading-from-v17x-or-earlier) |
| **Linux or macOS user** | [Linux](#linux-ubuntudebian) or [macOS](#macos-apple-silicon--intel) |
| **Colab or Kaggle** | [Cloud Notebooks](#google-colab--kaggle) |
| **Developer or expert** wanting pip | [Expert Installation](#expert-installation) |

---

### Windows Standalone Installer

The standalone installer provides a complete, self-contained WhisperJAV environment with GUI. No Python knowledge required.

**Download:** [**WhisperJAV-1.8.2-Windows-x86_64.exe**](https://github.com/meizhong986/WhisperJAV/releases/latest)

**Steps:**
1. Download the `.exe` file from the link above
2. Double-click to run (no admin rights needed)
3. Follow the on-screen prompts
4. Launch from the Desktop shortcut when complete

**What gets installed:**
- Python 3.10 (bundled, does not affect your system Python)
- FFmpeg for audio/video processing
- PyTorch with CUDA support (auto-detected)
- All WhisperJAV dependencies

**Timing:**
- Installation: 10-20 minutes (depends on internet speed)
- First transcription: Additional 5-10 minutes for AI model download (~3GB)

**Location:** `%LOCALAPPDATA%\WhisperJAV` (typically `C:\Users\YourName\AppData\Local\WhisperJAV`)

---

### Upgrading from v1.8.x

If you have v1.8.0, v1.8.1, or any 1.8.x version, you can upgrade in place.

#### Windows GUI Users

Open Command Prompt and run:
```batch
"%LOCALAPPDATA%\WhisperJAV\python.exe" -m whisperjav.upgrade
```

Or from the WhisperJAV environment:
```batch
whisperjav-upgrade
```

The upgrade tool will:
1. Create a rollback snapshot (in case you need to revert)
2. Download and install the new version
3. Update your desktop shortcut

**Rollback if needed:**
```batch
whisperjav-upgrade --rollback
```

#### Linux / macOS / Expert Users

From your WhisperJAV virtual environment:
```bash
source whisperjav-env/bin/activate
whisperjav-upgrade
```

Or upgrade specific components only:
```bash
whisperjav-upgrade --extras cli,translate
```

**Available extras:** `cli`, `gui`, `translate`, `llm`, `enhance`, `huggingface`, `analysis`, `all`

---

### Upgrading from v1.7.x or Earlier

Version 1.8.x has breaking dependency changes. A clean installation is required.

#### Windows GUI Users

1. **Uninstall v1.7.x:**
   - Open **Settings** → **Apps** → Search "WhisperJAV" → **Uninstall**
   - Or run: `%LOCALAPPDATA%\WhisperJAV\Uninstall-WhisperJAV.exe`

2. **Install v1.8.2:** Download and run the [new installer](#windows-standalone-installer)

#### Linux / macOS Users

1. **Remove the old environment:**
   ```bash
   rm -rf whisperjav-env
   ```

2. **Re-run the installer:**
   ```bash
   cd whisperjav
   git pull
   ./installer/install_linux.sh
   ```

**What is preserved across upgrades:**
- AI models: `~/.cache/huggingface/` (Linux/Mac) or `%USERPROFILE%\.cache\huggingface\` (Windows)
- Your transcription output files
- Any files outside the installation directory

---

### macOS (Apple Silicon & Intel)

#### Prerequisites

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required tools
brew install python@3.11 ffmpeg git
```

#### Installation

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

**With local LLM support (builds llama-cpp with Metal):**
```bash
./installer/install_linux.sh --local-llm-build
```

**Notes:**
- Apple Silicon (M1/M2/M3/M4): Uses MPS acceleration automatically
- Intel Macs: CPU-only mode, significantly slower than Apple Silicon

---

### Linux (Ubuntu/Debian)

#### Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip python3-venv build-essential ffmpeg libsndfile1 git
```

For Fedora/RHEL:
```bash
sudo dnf install python3-devel gcc ffmpeg libsndfile git
```

#### Installation

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

**Options:**
```bash
./installer/install_linux.sh --cpu-only        # Force CPU mode (no CUDA)
./installer/install_linux.sh --local-llm       # Include local LLM support
```

**NVIDIA GPU users:** The installer auto-detects CUDA. Ensure you have recent NVIDIA drivers installed (version 525+).

---

### Google Colab / Kaggle

No local installation required. Use these notebooks directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_parallel_expert.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_parallel_expert.ipynb)

The notebooks handle all installation automatically. Free GPU runtime is available on both platforms.

---

### Expert Installation

For users comfortable with Python package management. Choose the components you need.

#### Modular Installation (New in v1.8.2)

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
- NVIDIA: CUDA 11.8 or 12.x (Windows, Linux)
- Apple Silicon: MPS acceleration (M1/M2/M3/M4)
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
- [stable-ts](https://github.com/jianfch/stable-ts) - Timestamp refinement
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized CTranslate2 inference
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Transformers pipeline backend
- [Kotoba-Whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) - Japanese-optimized Whisper model
- The testing community for feedback and bug reports

---

## Disclaimer

This tool generates accessibility subtitles. Users are responsible for compliance with applicable laws regarding the content they process.
