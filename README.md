# WhisperJAV

<p align="center">
  <img src="https://img.shields.io/badge/version-1.7.1-blue.svg" alt="Version">
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

| Mode | Speed | When to Use |
|------|-------|-------------|
| **Faster** | Fast | Clean audio, just need quick results |
| **Fast** | Medium | Mixed quality, some background noise |
| **Balanced** | Slower | Best accuracy, noisy audio, dialogue-heavy content |

### Sensitivity Settings

- **Conservative**: Fewer false positives, may miss quiet speech. Good for noisy content.
- **Balanced**: Default. Works for most content.
- **Aggressive**: Catches more dialogue, but may include more errors. Good for ASMR/whisper content.

### Two-Pass / Ensemble Mode (New in v1.7)

Different AI models catch different things. Ensemble mode runs your video through two pipelines and merges the results for better coverage.

```bash
whisperjav video.mp4 --ensemble --pass1-pipeline kotoba-faster-whisper --pass2-pipeline balanced
```

The GUI's Ensemble tab lets you customize parameters like beam size, temperature, and VAD thresholds without editing config files.

### Kotoba Model Support (New in v1.7)

The Kotoba model is specifically trained on Japanese speech and handles conversational patterns better than standard Whisper:

```bash
whisperjav video.mp4 --mode kotoba-faster-whisper
```

### AI Translation

Generate subtitles and translate them in one step:

```bash
# Generate and translate
whisperjav video.mp4 --translate

# Or translate existing subtitles
whisperjav-translate -i subtitles.srt --provider deepseek
```

Supports DeepSeek (cheap), Gemini (free tier), Claude, GPT-4, and OpenRouter.

---

## What Makes It Work for JAV

### Scene Detection
Splits audio at natural breaks instead of forcing fixed-length chunks. This prevents cutting off sentences mid-word.

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

| Content Type | Mode | Sensitivity |
|--------------|------|-------------|
| Drama / Dialogue Heavy | balanced | aggressive |
| Group Scenes | faster | conservative |
| Amateur / Homemade | fast | conservative |
| ASMR / VR | balanced | aggressive |
| Heavy Background Music | balanced | conservative |

---

## Installation

### Windows Installer (Easiest)

Download and run: **WhisperJAV-1.7.1-Windows-x86_64.exe**

This installs everything you need including Python and dependencies.

### Upgrading from Previous Installer Versions

If you installed v1.5.x or v1.6.x via the Windows installer:

1. Download [upgrade_whisperjav.bat](https://github.com/meizhong986/whisperjav/raw/main/installer/upgrade_whisperjav.bat)
2. Double-click to run
3. Wait 1-2 minutes

This updates WhisperJAV without re-downloading PyTorch (~2.5GB) or your AI models (~3GB).

### Install from Source

Requires Python 3.9-3.12, FFmpeg, and Git.

```bash
# Install PyTorch with GPU support first (NVIDIA example)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Then install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git@main
```

**Platform Notes:**
- **Apple Silicon (M1/M2/M3/M4)**: Just `pip install torch torchvision torchaudio` - MPS acceleration works automatically
- **AMD GPU (ROCm)**: Experimental. Use `--mode balanced` for best compatibility
- **CPU only**: Works but slow. Use `--accept-cpu-mode` to skip the GPU warning

### Prerequisites

- **Python 3.9-3.12** (3.13+ not compatible with openai-whisper)
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
# Basic
whisperjav video.mp4
whisperjav video.mp4 --mode balanced --sensitivity aggressive

# Output options
whisperjav video.mp4 --output-dir ./subtitles
whisperjav video.mp4 --subs-language english-direct

# Ensemble mode
whisperjav video.mp4 --ensemble --pass1-pipeline kotoba-faster-whisper --pass2-pipeline balanced

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

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - The underlying ASR model
- [stable-ts](https://github.com/jianfch/stable-ts) - Timestamp refinement
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized inference
- The testing community for feedback and bug reports

---

## Disclaimer

This tool generates accessibility subtitles. Users are responsible for compliance with applicable laws regarding the content they process.
