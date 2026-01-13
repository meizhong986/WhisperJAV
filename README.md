# WhisperJAV

<p align="center">
  <a href="https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_parallel_expert.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://kaggle.com/kernels/welcome?src=https://github.com/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_parallel_expert.ipynb">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"/>
  </a>
  <br>
  <img src="https://img.shields.io/badge/version-1.7.4-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.9--3.12-green.svg" alt="Python">
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

Supports DeepSeek (cheap), Gemini (free tier), Claude, GPT-4, OpenRouter, and local LLMs.

#### Local LLM Translation (No API Key Required)

Run translation entirely on your GPU - no cloud API needed:

```bash
whisperjav-translate -i subtitles.srt --provider local
```

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

### Windows (Recommended)

*Best for: Most users, beginners, and those who want a GUI.*

1. **Download the Installer:**
   **[Download WhisperJAV-1.7.5-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/latest)**
2. **Run the File:** Double-click the downloaded `.exe`.
3. **Follow the Prompts:** The installer handles all dependencies (Python, FFmpeg, Git) automatically.
4. **Launch:** Open "WhisperJAV" from your Desktop shortcut.

> **Note:** The first launch may take a few minutes as it initializes the engine. GPU is auto-detected; CPU-only mode is used if no compatible GPU is found.

**Upgrading?** Just run the new installer. Your AI models (~3GB), settings, and cached downloads will be preserved.

---

### macOS (Apple Silicon & Intel)

*Best for: M1/M2/M3/M4 users and Intel Mac users.*

The install script auto-detects your Mac architecture and handles PyTorch dependencies automatically.

**1. Install Prerequisites**

```bash
# Install Xcode Command Line Tools (required for GUI)
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system tools
brew install python@3.11 ffmpeg git
```

> **GUI Requirement:** The Xcode Command Line Tools are required to compile `pyobjc`, which enables the GUI. Without it, only CLI mode will work.

**2. Install WhisperJAV**

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh

# Run the installer (auto-detects Mac architecture)
./installer/install_linux.sh
```

> **Intel Macs:** The script automatically uses CPU-only mode. Expect slower processing (5-10x) compared to Apple Silicon with MPS acceleration.

---

### Linux (Ubuntu/Debian/Fedora)

*Best for: Servers, desktops with NVIDIA GPUs.*

The install script auto-detects NVIDIA GPUs and installs the matching CUDA version.

**1. Install System Dependencies**

```bash
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y python3-dev python3-pip build-essential ffmpeg libsndfile1 git

# Fedora / RHEL
sudo dnf install python3-devel gcc ffmpeg libsndfile git
```

**2. Install WhisperJAV**

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
chmod +x installer/install_linux.sh

# Standard Install (auto-detects GPU)
./installer/install_linux.sh

# Or force CPU-only (for servers without GPU)
./installer/install_linux.sh --cpu-only
```

> **Performance:** A 2-hour video takes ~5-10 minutes on GPU vs ~30-60 minutes on CPU.

---

### Advanced / Developer

*Best for: Contributors and Python experts.*

<details>
<summary><b>Manual pip install</b></summary>

> **Warning:** Manual `pip install` is risky due to dependency conflicts (NumPy 2.x vs SciPy). We strongly recommend using the scripts above.

**1. Create Environment**

```bash
python -m venv whisperjav-env
source whisperjav-env/bin/activate   # Linux/Mac
# whisperjav-env\Scripts\activate    # Windows
```

**2. Install PyTorch First (Critical)**

You must install PyTorch *before* the main package to ensure hardware acceleration works.

- **NVIDIA GPU:** `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`
- **Apple Silicon:** `pip install torch torchaudio`
- **CPU only:** `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`

**3. Install WhisperJAV**

```bash
pip install git+https://github.com/meizhong986/whisperjav.git@main
```

</details>

<details>
<summary><b>Editable / Dev install</b></summary>

Use this if you plan to modify the code.

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# Windows
installer\install_windows.bat --dev

# Mac/Linux
./installer/install_linux.sh --dev

# Or manual
pip install -e ".[dev]"
```

</details>

<details>
<summary><b>Windows source install</b></summary>

```batch
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
installer\install_windows.bat              # Auto-detects GPU
installer\install_windows.bat --cpu-only   # Force CPU only
installer\install_windows.bat --cuda118    # Force CUDA 11.8
installer\install_windows.bat --cuda124    # Force CUDA 12.4
```

</details>

### Prerequisites

- **Python 3.10-3.12** (3.9 not supported due to pysubtrans dependency, 3.13+ incompatible with openai-whisper)
- **FFmpeg** in your system PATH
- **GPU recommended**: NVIDIA CUDA, Apple MPS, or AMD ROCm
- **8GB+ disk space** for installation

<details>
<summary>Detailed Windows Prerequisites</summary>

#### NVIDIA GPU Setup
1. Install latest [NVIDIA drivers](https://www.nvidia.com/drivers)
2. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) matching your driver version
3. Install [cuDNN](https://developer.nvidia.com/cudnn) matching your CUDA version

#### FFmpeg
1. Download from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your PATH

#### Python
Download from [python.org](https://www.python.org/downloads/windows/). Check "Add Python to PATH" during installation.

</details>

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
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**model.bin error in faster mode**: Enable Windows Developer Mode or run as Administrator, then delete the cached model folder:
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--Systran--faster-whisper-large-v2"
```

---

## Performance

Rough estimates for processing time per hour of video:

| Platform | Time |
|----------|------|
| NVIDIA GPU (CUDA) | 5-10 minutes |
| Apple Silicon (MPS) | 8-15 minutes |
| AMD GPU (ROCm) | 10-20 minutes |
| CPU only | 30-60 minutes |

---

## Contributing

Contributions welcome. See `CONTRIBUTING.md` for guidelines.

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
pip install -e .[dev]
python -m pytest tests/
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
