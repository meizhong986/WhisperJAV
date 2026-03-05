# WhisperJAV

**Japanese Adult Video Subtitle Generator with AI-powered transcription**

WhisperJAV is an open-source tool that generates accurate Japanese subtitles from video and audio files using OpenAI Whisper and other speech recognition models, with custom enhancements for Japanese language processing.

---

## Quick Start

### 1. Install

=== "Windows (Installer)"

    Download `WhisperJAV-x.x.x-Windows-x86_64.exe` from the [latest release](https://github.com/meizhong986/whisperjav/releases/latest). Run it — no admin rights required.

    [Full Windows installer guide](guides/installation_windows_standalone.md){ .md-button }

=== "Windows (Python)"

    ```bash
    git clone https://github.com/meizhong986/whisperjav.git
    cd whisperjav
    installer\install_windows.bat
    ```

    [Full Python install guide](guides/installation_windows_python.md){ .md-button }

=== "macOS"

    ```bash
    git clone https://github.com/meizhong986/whisperjav.git
    cd whisperjav
    python3 -m venv ~/venvs/whisperjav
    source ~/venvs/whisperjav/bin/activate
    ./installer/install_mac.sh
    ```

    [Full macOS guide](guides/installation_mac_apple_silicon.md){ .md-button }

=== "Linux"

    ```bash
    git clone https://github.com/meizhong986/whisperjav.git
    cd whisperjav
    ./installer/install_linux.sh
    ```

    [Full Linux guide](guides/installation_linux.md){ .md-button }

### 2. Launch the GUI

```bash
whisperjav-gui
```

Or use the Desktop shortcut (Windows installer).

### 3. Transcribe

1. Drag a video onto the app
2. Click **Start**
3. SRT subtitle file appears next to your video

That's it. For more control, see the [GUI User Guide](guides/gui_user_guide.md).

---

## Features at a Glance

| Feature | Description |
|---------|-------------|
| **Multiple pipelines** | Balanced, Fast, Faster, Fidelity, Transformers — trade speed for accuracy |
| **Ensemble mode** | Run two passes with different backends, merge for best results |
| **ChronosJAV** | Dedicated pipeline with anime-whisper and Kotoba models for anime/JAV content |
| **Qwen3-ASR** | Alternative ASR engine with strong Japanese performance |
| **AI Translation** | Translate subtitles using DeepSeek, Gemini, Claude, GPT, or local LLMs |
| **Speech enhancement** | ClearVoice denoising, BS-RoFormer vocal isolation, FFmpeg DSP chain |
| **Scene detection** | Auditok, Silero, Semantic — split long audio for better accuracy |
| **GUI + CLI** | Full-featured GUI for interactive use, CLI for automation and scripting |

---

## What's New

See the [latest release notes](https://github.com/meizhong986/whisperjav/releases/latest) for what's new in the current version.

---

## Getting Help

- [FAQ](faq.md) — common questions and answers
- [Upgrade Troubleshooting](UPGRADE_TROUBLESHOOTING.md) — fixing upgrade issues
- [GitHub Issues](https://github.com/meizhong986/whisperjav/issues) — bug reports and feature requests
