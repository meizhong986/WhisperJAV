# WhisperJAV - Japanese Adult Video Subtitle Generator

<p align="center">
  <img src="https://img.shields.io/badge/version-1.6.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.9+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License">
</p>

WhisperJAV is a subtitle generation tool optimized for Japanese Adult Videos (JAV). It uses custom enhancements specifically tailored for the audio characteristics, and sound patterns in JAV media.

## 🌟 Key Features

-   **Three Processing Modes**: Optimized pipelines for different content types and quality requirements.
-   **Japanese Language Processing**: Custom post-processing for natural dialogue segmentation.
-   **Scene Detection**: Automatic scene splitting for better transcription accuracy.
-   **VAD Integration**: Voice Activity Detection for improved speech recognition.
-   **Hallucination Removal**: Specialized filters for common JAV transcription errors.
-   **AI Translation**: Built-in subtitle translation powered by DeepSeek, Gemini, Claude, and more.
-   **GUI and CLI**: User-friendly interface and command-line options.
-   **Batch Processing**: Process multiple files with progress tracking.

## 📋 Table of Contents

-   [Installation](#-installation)
-   [Quick Start](#-quick-start)
-   [AI Translation](#-ai-translation)
-   [Processing Modes Guide](#-processing-modes-guide)
-   [Sensitivity Settings](#️-sensitivity-settings)
-   [Advanced Japanese Language Features](#-advanced-japanese-language-features)
-   [Usage Examples](#-usage-examples)
-   [Configuration](#️-configuration)
-   [GUI Interface](#️-gui-interface)
-   [Troubleshooting](#-troubleshooting)
-   [Contributing](#-contributing)
-   [License](#-license)
-   [Acknowledgments](#-acknowledgments)
-   [Disclaimer](#️-disclaimer)

## 🔧 Installation

### Prerequisites

Please see the details at the end of this readme for more details.

-   Python 3.9 - 3.12 (Python 3.13+ is not compatible with openai-whisper)
-   **GPU (Recommended for best performance)**:
    -   NVIDIA GPU (CUDA > 11.7) - RTX 20/30/40/50 series, Blackwell, etc.
    -   Apple Silicon (M1/M2/M3/M4/M5) - Native MPS acceleration
    -   AMD GPU (ROCm) - Limited support, see platform-specific notes
-   **PyTorch with GPU support** (see Platform-Specific Installation below)
-   FFmpeg installed and in your system's PATH
-   PIP and git installation packages

### Install from Source

```bash

# Standard installation (RECOMMENDED - use the latest commit from main)
pip install git+https://github.com/meizhong986/whisperjav.git@main



# For users with existing installations, Update:
pip install -U --no-deps git+https://github.com/meizhong986/whisperjav.git@main



### Platform-Specific Installation

WhisperJAV now supports multiple GPU platforms for optimal performance. Choose the installation method for your hardware:

#### Windows (NVIDIA GPU)

```bash
# For NVIDIA RTX 20/30/40/50 series and Blackwell GPUs
# Install PyTorch with CUDA 12.4 support (recommended for latest GPUs)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# For older GPUs or CUDA 12.1 drivers
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Then install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git@main
```

#### macOS (Apple Silicon)

```bash
# For M1/M2/M3/M4/M5 Macs - uses native Metal Performance Shaders (MPS)
pip install torch torchvision torchaudio

# Then install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git@main
```

**Note**: MPS acceleration is automatic on Apple Silicon. No additional configuration needed.

#### Linux (NVIDIA GPU)

```bash
# For NVIDIA GPUs on Linux
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Then install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git@main
```

#### Linux (AMD GPU - Experimental)

```bash
# AMD GPU support via ROCm is experimental
# Note: The faster-whisper backend has limited ROCm support
# Balanced mode works better with AMD GPUs

# Install PyTorch with ROCm (example for ROCm 6.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Then install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git@main

# Use balanced mode for best compatibility
whisperjav video.mp4 --mode balanced
```

#### CPU-Only Mode

```bash
# For systems without GPU (significantly slower)
pip install torch torchvision torchaudio

# Then install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git@main

# Use --accept-cpu-mode to skip GPU warning
whisperjav video.mp4 --accept-cpu-mode
```

### ⚠️ Performance Notes

- **NVIDIA GPU (CUDA)**: Best performance, ~5-10 minutes per hour of video
- **Apple Silicon (MPS)**: Good performance, ~8-15 minutes per hour of video
- **AMD GPU (ROCm)**: Limited support, use balanced mode, ~10-20 minutes per hour
- **CPU-Only**: Very slow, ~30-60 minutes per hour of video (6-10x slower than GPU)

**Important**: Always install GPU-enabled PyTorch BEFORE installing WhisperJAV. Otherwise, openai-whisper will automatically install a CPU-only version.

```

### Dependencies

The main dependencies will be automatically installed:

-   `openai-whisper` or `faster-whisper`
-   `stable-ts`
-   `torch` (with CUDA support)
-   `pysrt`
-   `tqdm`
-   `numpy`
-   `soundfile`

## 🚀 Quick Start

### Command Line

```bash
# Basic usage with default settings
whisperjav video.mp4

# Specify mode and output directory
whisperjav audio.wav --mode faster --output-dir ./subtitles

# Process multiple files with specific sensitivity
whisperjav *.mp3 --mode balanced --sensitivity aggressive

# Generate English subtitles
whisperjav video.mp4 --subs-language english-direct
```

### GUI

```bash
whisperjav-gui
```


## 🌐 AI Translation

WhisperJAV includes built-in AI-powered subtitle translation via `whisperjav-translate`.

### Quick Translation Setup

1. Get an API key from [DeepSeek](https://platform.deepseek.com/) (recommended, ~$0.10 per 100k tokens)
2. Set your API key:
   ```bash
   # Windows (PowerShell)
   $env:DEEPSEEK_API_KEY="sk-..."

   # Linux/Mac
   export DEEPSEEK_API_KEY="sk-..."
   ```

3. Translate subtitles:
   ```bash
   # Standalone translation
   whisperjav-translate -i movie.srt

   # Generate and translate in one command
   whisperjav video.mp4 --translate
   ```

### Translation Features

- **Multiple AI Providers**: DeepSeek (default), Gemini, Claude, GPT-4, OpenRouter
- **Smart Caching**: Instructions fetched from Gist with local caching
- **Settings File**: Save your preferences for repeated use
- **Tone Styles**: Standard or Pornify (explicit content) translation styles
- **Multiple Languages**: Translate to English, Spanish, Chinese, Indonesian
- **Metadata-aware prompts**: Optional movie title, plot, and actress name to improve translation
- **Sampling controls**: Temperature and top_p supported; pornify tone applies sensible defaults

### Translation Examples

```bash
# Translate to Spanish with pornify style
whisperjav-translate -i movie.srt -t spanish --tone pornify

# Use Gemini provider
whisperjav-translate -i movie.srt --provider gemini

# Configure translation preferences interactively
whisperjav-translate --configure

# View API key setup instructions
whisperjav-translate --print-env

# Advanced: Pass provider-specific options
whisperjav-translate -i movie.srt --provider-option temperature=0.7 --provider-option max_tokens=2000

# Generate subtitles and translate together
whisperjav video.mp4 --translate --translate-provider deepseek

# Use local custom instructions instead of defaults
whisperjav-translate -i movie.srt --instructions-file C:\path\to\my_instructions.txt

# Add helpful metadata context (optional)
whisperjav-translate -i movie.srt \
  --movie-title "JAV-123: After-work Massage" \
  --actress "Yua Mikami" \
  --movie-plot "Office worker gets an after-hours massage that turns intimate"

# Explicitly set sampling params (override tone defaults)
whisperjav-translate -i movie.srt --temperature 0.5 --top-p 0.85
```

### Configuration

WhisperJAV-Translate supports persistent settings to avoid repeating common flags:

```bash
# Interactive configuration wizard (RECOMMENDED)
whisperjav-translate --configure

# View API key setup instructions
whisperjav-translate --print-env

# Show current settings
whisperjav-translate --show-settings
```

Settings are stored in:
- **Windows**: `%AppData%\WhisperJAV\translate\settings.json`
- **Linux**: `~/.config/WhisperJAV/translate/settings.json`
- **Mac**: `~/Library/Application Support/WhisperJAV/translate/settings.json`

**Configuration Precedence** (highest to lowest):
1. CLI flags (e.g., `--provider gemini`)
2. Environment variables (e.g., `DEEPSEEK_API_KEY`)
3. Settings file
4. Built-in defaults

Instructions source precedence:
- Local file via `--instructions-file`
- Settings file mapping for source/tone
- Default remote Gist (with ETag cache) → local cache → bundled fallback

### Supported Providers

| Provider   | Cost | Setup | API Key Env Var |
|------------|------|-------|-----------------|
| **DeepSeek** (default) | ~$0.10/100k tokens | [platform.deepseek.com](https://platform.deepseek.com/) | `DEEPSEEK_API_KEY` |
| **Gemini** | Free tier available | [makersuite.google.com](https://makersuite.google.com/) | `GEMINI_API_KEY` |
| **Claude** | Pay-as-you-go | [console.anthropic.com](https://console.anthropic.com/) | `ANTHROPIC_API_KEY` |
| **GPT-4** | Pay-as-you-go | [platform.openai.com](https://platform.openai.com/) | `OPENAI_API_KEY` |
| **OpenRouter** | Varies | [openrouter.ai](https://openrouter.ai/) | `OPENROUTER_API_KEY` |

> **Provider SDKs:** PySubtrans only registers GPT and Gemini providers if their SDKs are installed. Run `pip install openai google-genai` (already covered by `requirements.txt`) or you will see errors like `Unknown translation provider: OpenAI`/`Gemini` even though the CLI recognizes the provider.

For detailed translation options, run: `whisperjav-translate --help`


## 📊 Processing Modes Guide

Choose the appropriate mode based on your content type and requirements:

| Mode     | Best For                                                                                             | Characteristics                                                                                     | Processing Speed | Accuracy      |
| :------- | :--------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- | :--------------- | :------------ |
| **Faster** | • Whole-file runs where speed matters                                                               | • Faster‑Whisper backend<br>• No scene splitting<br>• Internal VAD (Stable‑TS packed)<br>• Batched inference (higher throughput) | ⚡⚡⚡ Fast        | Adequate      |
| **Fast** | • Mixed content quality with variable audio                                                          | • Faster‑Whisper backend<br>• Scene detection enabled (mandatory splitting)<br>• Internal VAD (Stable‑TS packed)<br>• Non‑batched inference (batch_size=1) | ⚡⚡ Medium       | Satisfactory  |
| **Balanced** | • Max accuracy for dialogue timing and noisy audio                                                  | • Scene detection + separate VAD + WhisperPro (OpenAI Whisper)
| • Most accurate timestamps | ⚡ Slower        | Good          |


### Content-Specific Recommendations

| Genre                       | Recommended Mode | Recommended Sensitivity |
| :-------------------------- | :--------------- | :---------------------- |
| Drama/Dialogue Heavy        | balanced         | aggressive              |
| Group/3p/4p Scenes          | faster           | conservative            |
| Amateur/Homemade            | fast             | conservative            |
| Vintage (pre-2000)          | fast             | balanced                |
| ASMR/VR Content             | balanced         | aggressive              |
| Compilation/Omnibus         | faster           | conservative            |
| Heavy Background Music      | balanced         | conservative            |
| Outdoor/Public Scenes       | fast             | balanced                |

## 🎚️ Sensitivity Settings

The `sensitivity` parameter controls the trade-off between capturing detail and avoiding noise/hallucinations:

**Conservative**
-   **Fewer false positives**: Reduces hallucinated text and repetitions.
-   **Higher confidence threshold**: Only includes clearly spoken words.
-   **Best for**:
    -   Poor audio quality recordings
    -   Heavy background noise or music
    -   Vintage/degraded content
    -   Content with lots of non-speech sounds
-   **Trade-off**: May miss some quiet or unclear speech.

**Balanced (Default)**
-   **Optimal balance**: Good detection with reasonable filtering.
-   **Moderate thresholds**: Captures most speech while filtering obvious errors.
-   **Best for**:
    -   Standard quality recordings
    -   Mixed content types
    -   General-purpose transcription
    -   First-time users
-   **Trade-off**: A balanced approach to all aspects.

**Aggressive**
-   **Maximum detail capture**: Attempts to transcribe everything.
-   **Lower confidence threshold**: Includes uncertain segments.
-   **Best for**:
    -   High-quality audio
    -   ASMR or whisper content
    -   Content where every utterance matters
    -   Professional recordings with clear audio
-   **Trade-off**: May include more false positives and hallucinations.

### Sensitivity Selection Matrix

| Audio Quality | Background Noise | Speech Clarity | Recommended Sensitivity |
| :------------ | :--------------- | :------------- | :---------------------- |
| Poor          | High             | Unclear        | **Conservative** |
| Average       | Moderate         | Mixed          | **Balanced** |
| Excellent     | Low              | Clear          | **Aggressive** |
| Variable      | Variable         | Variable       | **Balanced** |

## 🗾 Advanced Japanese Language Features

WhisperJAV includes sophisticated Japanese language processing specifically optimized for adult content dialogue.

### Dialogue-Optimized Segmentation

The system uses advanced `stable-ts` regrouping algorithms customized for Japanese conversational patterns.
```python
# Automatic application of Japanese-specific rules:
# - Sentence-ending particles (ね, よ, わ, の, ぞ, ぜ, さ, か)
# - Polite forms (です, ます, でした, ましょう)
# - Question particles detection
# - Emotional expressions and interjections
# - Casual contractions (ちゃ, じゃ, きゃ)
```

### Specialized Pattern Recognition

-   **Aizuchi and Fillers**: Automatically identifies and handles:
    -   `あの`, `ええと`, `まあ`, `なんか` (filler words)
    -   `うん`, `はい`, `ええ`, `そう` (acknowledgments)
-   **Emotional Expressions**: Preserves important non-lexical vocalizations:
    -   `ああ`, `うう`, `はあ`, `ふう` (sighs, moans)
    -   Maintains timing for emotional context
-   **Dialect Support**: Recognizes common dialect patterns:
    -   Kansai-ben endings (`わ`, `で`, `ねん`, `や`)
    -   Feminine speech patterns (`かしら`, `わね`, `のよ`)
    -   Masculine speech patterns (`ぜ`, `ぞ`, `だい`)

### Custom Regrouping Strategies

The system automatically selects appropriate regrouping based on content:

```bash
# These are applied automatically based on mode and sensitivity:
--mode balanced      # Applies comprehensive regrouping
--sensitivity aggressive # Includes more nuanced patterns
```

### Timing Optimization for Natural Speech

-   **Gap-based merging**: Combines segments with natural speech pauses.
-   **Punctuation-aware splitting**: Respects Japanese punctuation (`。`, `、`, `！`, `？`).
-   **Maximum subtitle duration**: Ensures readability (default 7-8 seconds).
-   **Minimum duration filtering**: Removes micro-segments.

## 📖 Usage Examples

### Basic Transcription

```bash
# Generate Japanese subtitles (default)
whisperjav video.mp4

# Generate English translation
whisperjav video.mp4 --subs-language english-direct
```

### Batch Processing

```bash
# Process an entire directory
whisperjav /path/to/videos/*.mp4 --output-dir ./output

# Process with specific settings
whisperjav *.mp4 --mode balanced --sensitivity aggressive --output-dir ./subs
```

### Advanced Options

```bash
# Keep temporary files for debugging
whisperjav video.mp4 --keep-temp

# Enable all enhancement features
whisperjav video.mp4 --adaptive-classification --adaptive-audio-enhancement --smart-postprocessing

# Use a custom configuration file
whisperjav video.mp4 --config my_config.json

# Specify a different Whisper model (WiP)
whisperjav video.mp4 --model large-v2
```

### Output Options

```bash
# Save processing statistics to a file
whisperjav video.mp4 --stats-file stats.json

# Disable progress bars
whisperjav video.mp4 --no-progress

# Use a custom temporary directory (e.g., on a fast SSD)
whisperjav video.mp4 --temp-dir /fast/ssd/temp
```

## ⚙️ Configuration

### Configuration File Format (Work in Progress --subject to change)

Create a custom `config.json` to override default settings:

```json
{
  "modes": {
    "balanced": {
      "scene_detection": {
        "max_duration": 30.0,
        "min_duration": 0.2,
        "max_silence": 2.0
      },
      "vad_options": {
        "threshold": 0.4,
        "min_speech_duration_ms": 150
      }
    }
  },
  "sensitivity_profiles": {
    "aggressive": {
      "hallucination_threshold": 0.8,
      "repetition_threshold": 3,
      "min_confidence": 0.5
    }
  }
}
```

### Logprob Filter Controls

`common_transcriber_options` now accepts two extra keys per sensitivity profile:

- `logprob_margin` – gives short segments (≤1.6s) additional headroom by lowering the threshold when set (e.g., `0.2`).
- `drop_nonverbal_vocals` – when `true`, removes obvious nonverbal fillers such as `[music]`, `(moans)`, or pure `♪` notes before stitching.

Combine these with the existing `logprob_threshold` to dial in how aggressive scene-level filtering should be across modes.



## 🖥️ GUI Interface

The PyWebView-based GUI provides a modern, responsive interface for users who prefer not to use the command line.

### Features

-   Modern HTML/CSS/JS interface with professional look and feel
-   Drag-and-drop file and folder selection
-   Real-time progress monitoring and log streaming
-   Visual mode and sensitivity selection with descriptions
-   Advanced settings in tabbed interface
-   Keyboard shortcuts (Ctrl+O, Ctrl+R, F1, Esc, F5)
-   Console output display with real-time updates

### System Requirements

-   **Windows**: Requires WebView2 runtime (automatically installed with Microsoft Edge browser)
-   **macOS**: Uses native WebKit (built-in)
-   **Linux**: Uses GTK WebKit2

For detailed GUI usage instructions, see [GUI_USER_GUIDE.md](GUI_USER_GUIDE.md).

### Status of Adaptive Features (WIP)

The following optional features are present in the UI/CLI switches but are currently work in progress and not yet fully functional end-to-end:

- Adaptive scene classification (`--adaptive-classification`)
- Adaptive audio enhancement (`--adaptive-audio-enhancement`)
- Smart post‑processing (`--smart-postprocessing`)

You can toggle them, but expect incomplete behavior or no effect in some pipelines. We’ll remove this note once they’re production‑ready.

### GUI Quick Start

1.  Launch the GUI: `whisperjav-gui`
2.  Select files using the "Add Files" button or drag and drop.
3.  Choose your **Processing Mode** (Faster/Fast/Balanced).
4.  Select the **Sensitivity** (Conservative/Balanced/Aggressive).
5.  Choose the **Output Language** (Japanese/English).
6.  Click "Start" to begin processing.

## 🔍 Troubleshooting

### Common Issues

-   **Issue**: `FFmpeg not found`
    -   **Solution**: Install FFmpeg and ensure it's in your system's PATH.
        ```bash
        # Ubuntu/Debian
        sudo apt install ffmpeg
        # Windows (using Chocolatey)
        choco install ffmpeg
        # macOS (using Homebrew)
        brew install ffmpeg
        ```

-   **Issue**: Slow processing or "GPU Performance Warning"
    -   **Solution**: Check your GPU setup with diagnostics:
        ```bash
        whisperjav --check
        ```
    -   Common causes:
        - CPU-only PyTorch installed instead of GPU version
        - Missing or outdated GPU drivers
        - CUDA toolkit version mismatch
    -   **Fix for NVIDIA GPUs**:
        ```bash
        # Reinstall PyTorch with CUDA support
        pip uninstall torch torchvision torchaudio
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
        ```
    -   **Fix for Apple Silicon**:
        ```bash
        # Update to latest PyTorch for MPS support
        pip install --upgrade torch torchvision torchaudio
        ```

-   **Issue**: GPU not detected on Apple Silicon Mac
    -   **Solution**: Ensure you're running Python natively (not via Rosetta):
        ```bash
        # Check Python architecture
        python -c "import platform; print(platform.machine())"
        # Should show 'arm64', not 'x86_64'
        ```
    -   If using x86_64, reinstall Python for Apple Silicon

-   **Issue**: AMD GPU not working
    -   **Solution**: AMD GPU support is experimental. Use balanced mode:
        ```bash
        whisperjav video.mp4 --mode balanced
        ```
    -   Note: faster mode may not work with ROCm due to CTranslate2 limitations

-   **Issue**: `pkg_resources is deprecated` warning followed by `Unable to open file 'model.bin'` (faster/faster-balanced modes)
  -   **What it means**: the warning is emitted by `ctranslate2` and is harmless, but the crash indicates the Hugging Face cache stored a partial `model.bin` (often on Windows when Developer Mode/symlinks are disabled).
  -   **Fix**:
    1. Enable Windows *Developer Mode* (Settings → Privacy & Security → For Developers) **or** run the terminal as Administrator so Hugging Face can create symlinks.
    2. Remove the corrupted snapshot folder and let WhisperJAV re-download it (the app now retries automatically, but you can force it manually):
      ```powershell
      Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--Systran--faster-whisper-large-v2"
      ```
       (Change `large-v2` to the model you selected.)
    3. Re-run `whisperjav ... --mode faster` – the new build verifies the cache and re-downloads if the binary is missing.

### Performance Tips

-   **GPU Acceleration**:
    - **NVIDIA (CUDA)**: Best performance, 5-10 minutes per hour of video
    - **Apple Silicon (MPS)**: Good performance, 8-15 minutes per hour
    - **AMD (ROCm)**: Limited support, use balanced mode
    - Use `whisperjav --check` to verify GPU is detected correctly
-   **SSD Storage**: Use an SSD for temporary files via the `--temp-dir` argument for faster I/O.
-   **Batch Processing**: Process multiple files in one run to avoid reloading the model for each file.
-   **Memory Usage**: Close other memory-intensive applications when processing large files.
-   **CPU Mode**: If needed, use `--accept-cpu-mode` to skip GPU warning (processing will be 6-10x slower).

## 🤝 Contributing

We welcome contributions! Please see our `CONTRIBUTING.md` for details on how to get started.

### Development Setup

```bash
git clone [https://github.com/yourusername/whisperjav.git](https://github.com/yourusername/whisperjav.git)
cd whisperjav
# Install in editable mode with development dependencies
pip install -e .[dev]
# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   The [OpenAI Whisper](https://github.com/openai/whisper) team for the base ASR technology.
-   The [stable-ts](https://github.com/jianfch/stable-ts) project for enhanced timestamp features.
-   The [faster-whisper](https://github.com/guillaumekln/faster-whisper) project for optimized inference.
-   The JAV transcription community for their invaluable feedback and testing.

## ⚠️ Disclaimer

This tool is designed for creating accessibility subtitles and for use as a language-learning material. Users are solely responsible for compliance with all applicable local and international laws and regulations regarding the content they choose to process.


Prerequisites:
# ✅ Tools You MUST Install First (Prerequisites)

Install these in the order listed. If you already have them, ensure they are up-to-date.

---

## 🎮 NVIDIA CUDA Platform (Drivers, CUDA Toolkit, cuDNN)

**What you need**:  
Your **NVIDIA Graphics Card Drivers**, the **CUDA Toolkit**, and the **cuDNN** library.  
All three are essential for WhisperJAV to use your GPU.

### 🔧 How to install:

#### 1. NVIDIA Graphics Driver
- Ensure you have the latest drivers for your NVIDIA GPU.
- 📥 Download from: [https://www.nvidia.com/drivers](https://www.nvidia.com/drivers)

#### 2. CUDA Toolkit
- Open **Command Prompt (CMD)** and type:
  ```bash
  nvidia-smi
  ```
- Note the `CUDA Version:` (e.g., **12.3**).
- 📥 Go to: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- Select **Windows**, then choose a CUDA Toolkit version **equal to or lower than** what `nvidia-smi` showed.
- Download and install it.

#### 3. cuDNN (CUDA Deep Neural Network library)
- 📥 Go to: [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
- ⚠️ You need a **free NVIDIA Developer Program** account.
- Download the cuDNN version that **matches your installed CUDA Toolkit** (e.g., “cuDNN v9.x.x for CUDA 12.x”).
- Choose the “**Windows (x86_64) Zip**”.

**Extract and Copy (Crucial!)**
- Extract the cuDNN `.zip` file.
- You’ll find folders: `bin`, `include`, `lib`
- Copy **all contents** from cuDNN’s `bin` folder into your CUDA Toolkit’s `bin` folder:
  ```
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin
  ```
- Do the same for the `include` and `lib` folders.

📌 **Restart your PC** after copying cuDNN files.

---

## 🐍 Python 3.9 or Higher

### Download:
- 📥 [https://www.python.org/downloads/windows](https://www.python.org/downloads/windows)

### Install:
- During installation, **CHECK THE BOX**:  
  ✅ “Add Python.exe to PATH” (on the first screen).

---

## 🧬 Git for Windows

### Download:
- 📥 [https://git-scm.com/download/win](https://git-scm.com/download/win)

### Install:
- The default options are usually fine.

---

## 🎞 FFmpeg (For Video & Audio Processing)

### Download:
- 📥 [https://www.gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds)
- Download `ffmpeg-git-full.7z` or `.zip`.

### Extract & Move:
- Extract the archive.
- Rename the inner folder to `ffmpeg`.
- Move it to:
  ```
  C:\ffmpeg
  ```

### Add to PATH (Crucial!):

1. Open **Command Prompt as Administrator**
2. Paste and run:
   ```bash
   setx /M PATH "C:\ffmpeg\bin;%PATH%"
   ```

3. Close and reopen all Command Prompt / PowerShell windows.

### Verify:
- Open a new **regular** Command Prompt and type:
  ```bash
  ffmpeg -version
  ```
- You should see version info if installed correctly.
