# WhisperJAV - Japanese Adult Video Subtitle Generator

<p align="center">
  <img src="https://img.shields.io/badge/version-1.4.4-blue.svg" alt="Version">
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
-   **GUI and CLI**: User-friendly interface and command-line options.
-   **Batch Processing**: Process multiple files with progress tracking.

## 📋 Table of Contents

-   [Installation](#-installation)
-   [Quick Start](#-quick-start)
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
-   CUDA-capable GPU, drivers, CUDA Toolkit, cuDNN (CUDA > 11.7)
-   **CUDA-version** of pytorch, torchaudio and torchvision
-   FFmpeg installed and in your system's PATH
-   PIP and git installation packages

### Install from Source

```bash

# Standard installation (RECOMMENDED - use the latest commit from main)
pip install git+https://github.com/meizhong986/whisperjav.git@main



# For users with existing installations, Update:
pip install -U --no-deps git+https://github.com/meizhong986/whisperjav.git@main



### ⚠️ Important Note
Please make sure that you have installed cuda enabled pytorch, and pyaudio before installing whisperjav. Otherwise, openai-whisper will automatically installs a CPU torch version which is 8 times slower. You don't want that!!! 
Example for CUDA 12.4 torch 2.5.1 (the version WhisperJAv has been tested for): 
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

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


## 📊 Processing Modes Guide

Choose the appropriate mode based on your content type and requirements:

| Mode     | Best For                                                                                             | Characteristics                                                                                     | Processing Speed | Accuracy      |
| :------- | :--------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- | :--------------- | :------------ |
| **Faster** | • Complex multi-performer scenes<br>• Heavy background noise<br>• Mixed audio<br>• Moaning/non-speech | • Uses Faster-Whisper backend<br>• Direct transcription without chunking<br>• Lower memory usage         | ⚡⚡⚡ Fast        | Adequate      |
| **Fast** | • Mixed content quality<br>• Vintage/older content<br>• Amateur recordings<br>• Compilation videos      | • Standard Whisper with scene detection<br>• Mandatory scene splitting<br>• Better handling of quality | ⚡⚡ Medium       | Satisfactory  |
| **Balanced** | • Standard dialogue scenes<br>• Clear audio quality<br>• Modern HD content<br>• Single performer scenes | • Scene detection + VAD enhancement<br>• Best noise handling<br>• Most accurate timestamps          | ⚡ Slower        | Good          |


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



## 🖥️ GUI Interface (Work in Progress)

The GUI provides an intuitive interface for users who prefer not to use the command line.

### Features

-   Drag-and-drop file selection
-   Real-time progress monitoring
-   Visual mode and sensitivity selection
-   Advanced settings dialog
-   Console output display

### GUI Quick Start

1.  Launch the GUI: `python whisperjav_gui.py`
2.  Select files using the "Select File(s)" button.
3.  Choose your **Processing Mode** (Faster/Fast/Balanced).
4.  Select the **Sensitivity** (Conservative/Balanced/Aggressive).
5.  Choose the **Output Language** (Japanese/English).
6.  Click "START PROCESSING".

## 🔍 Troubleshooting

### Common Issues
        ```
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
-   **Issue**: Slow processing or GUI looks hanged
    -   **Solution**: Often it is caused by wrong pytorch version
        ```
        Remove CPU version of pytorch if exist
        ```

### Performance Tips

-   **GPU Acceleration**: Ensure CUDA is properly installed for a 3-5x speed improvement.
-   **SSD Storage**: Use an SSD for temporary files via the `--temp-dir` argument for faster I/O.
-   **Batch Processing**: Process multiple files in one run to avoid reloading the model for each file.
-   **Memory Usage**: Close other memory-intensive applications when processing large files.

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
