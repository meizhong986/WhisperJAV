# WhisperJAV v1.8.3 Release Notes

**Release Date:** February 2026
**Type:** Feature Preview + Bug Fixes
**Base:** v1.8.2-rc02

---

## A Note Before We Start

This release is a bit of a milestone. WhisperJAV started as a wrapper around OpenAI Whisper, but with v1.8.3 we're expanding beyond Whisper for the first time. The new Qwen3-ASR pipeline uses fundamentally different technology under the hood -- a Qwen2.5-based speech model with a dedicated forced aligner, rather than the Whisper encoder-decoder architecture.

We're shipping this as a **preview**. It works, and some of you will get better results with it, but it's not finished. Timestamps still need work, there are hallucination edge cases, and the vLLM backend isn't wired up yet. We'd rather get it in your hands early so you can tell us what works and what doesn't, instead of polishing it in isolation for another few months.

If you want to try it: it's available through **Ensemble mode in the GUI**, as the default Pass 2 pipeline. If you'd rather stick with what you know, the existing Whisper pipelines haven't changed.

---

## What's New: Qwen3-ASR Pipeline (Preview)

### What It Is

A new ASR engine based on [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), available as an alternative to Whisper. It uses a Qwen2.5 language model for text generation and a separate forced aligner for word-level timestamps.

### Three Input Modes

The Qwen pipeline supports three ways to process your audio:

| Mode | How It Works | Best For |
|------|-------------|----------|
| **Assembly** (default) | Generates text first, then aligns timestamps in a separate pass. Batches scenes up to 120s. | Most content. Decoupled design means each step can be optimized independently. |
| **Context-Aware** | Runs ASR and alignment together on full scenes (30-90s). | When you need the model to "see" more context around each utterance. |
| **VAD Slicing** | Pre-slices audio into short speech segments (up to 29s) using VAD, then transcribes each. | Legacy mode, similar to how Whisper pipelines work. |

### What It Solves

These were real problems reported by users on the Whisper pipelines, and the Qwen pipeline handles them differently:

- **N-gram repetition loops** -- Whisper sometimes gets stuck repeating phrases ("ああああああ..." or looping a sentence). Qwen has a `repetition_penalty` (default 1.1) plus a dynamic token budget (`max_tokens_per_audio_second=20.0`) that caps generation length per chunk.

- **Token budget overflow** -- Long scenes could cause Whisper to generate way too much text. The assembly mode decouples text generation from alignment, so each step has its own constraints.

- **Aligner collapses** -- When the forced aligner fails and assigns all words to timestamp 0.0 (or clusters them into a tiny window), the **alignment sentinel** detects this and recovers using proportional redistribution or VAD-guided placement.

### What Still Needs Work

Being honest about where we are:

- **Timestamp accuracy** -- The forced aligner sometimes drifts, especially on long scenes with background music. The assembly mode mitigates this with tighter scene boundaries (120s max), but it's not as precise as we want.
- **Hallucination** -- The Qwen model can hallucinate during silence or music-only sections. We've added Japanese post-processing to catch some of these, but it's not fully solved.
- **vLLM backend** -- Assembly mode is architecturally ready for vLLM (the text generation step can be swapped), but the integration isn't built yet. This would help with batch throughput.
- **MPS (Apple Silicon)** -- The Qwen pipeline currently runs on CPU on Macs. The underlying `transformers` library supports MPS, but `qwen_asr.py` doesn't detect it yet.

### How to Try It

1. Open the GUI (`whisperjav-gui`)
2. Select **Ensemble** pipeline
3. Pass 2 defaults to **Qwen3-ASR** -- just run it
4. Or in the Qwen tab, pick an input mode and tweak parameters

CLI users can use `--mode qwen` directly, but the GUI gives you more control over the parameters for this preview.

---

## What Else Changed

### Alignment Sentinel

A new module that watches for degenerate word timestamps and fixes them automatically. This runs on all three Qwen input modes:

- **Collapse detection** -- Finds words where timestamps are all 0.0 or clustered within a tiny window
- **Cluster collapse detection** -- Catches cases where words bunch up around a single point (degenerate-ratio check)
- **Recovery strategies:**
  - Strategy B: Proportional redistribution (spreads words evenly across the time window)
  - Strategy C: VAD-guided redistribution (uses voice activity data to place words where speech actually is)

### Japanese Post-Processing for Qwen

The Qwen model outputs Japanese text differently from Whisper (less punctuation, different segmentation). New processing specifically for Qwen output:

- Hierarchical linguistic splitting for unpunctuated Japanese text
- Particle-aware sentence boundaries
- Recursive pattern repetition detection (catches loops that the model-level penalty misses)
- Punctuation preservation through the assembly pipeline
- Compound particle handling, tiny fragment merging

### Adaptive Step-Down Architecture

When a scene is too long or complex for the primary grouping strategy, the pipeline now falls back gracefully:

- **Tier 1** -- Try the preferred group size (tighter boundaries)
- **Tier 2** -- If Tier 1 fails, relax to larger groups with proportional timing fallback

This replaced the old single-shot approach where a failure meant the whole scene got bad timestamps.

### Pipeline Analytics

A health check and metrics system that tracks what happened during processing:

- Per-group diagnostic detail records
- Sentinel activation counts and recovery success rates
- Timing source breakdown (aligned vs. interpolated vs. proportional)

Useful if you're comparing results across different settings, or reporting issues.

### Assembly Text Cleaner

A mid-pipeline text cleaning stage that runs between text generation and alignment (assembly mode only). Catches garbage that would confuse the aligner before it gets a chance to.

---

## Bug Fixes (since v1.8.2-rc02)

| What Was Broken | What Happened | Fixed |
|----------------|---------------|-------|
| Timestamp overflow past audio end | `_apply_timestamp_interpolation()` trailing gap had no upper bound, producing timestamps beyond the scene duration | Capped to group duration, added pre/post-offset clamping |
| Double-offset timestamp bug | Timestamps were being offset twice in certain code paths | Fixed coordinate transition logic |
| Temp file naming collision | Tier 1 and Tier 2 temp files could overwrite each other in step-down mode | Added tier tag to filenames |
| Chronological ordering after merge | Merged word lists could be out of order | Added defensive sort after merge |
| IndexError crash in JP postprocessor | Merge methods could crash on edge-case segment boundaries | Bounds checking added |
| Tiny segment merge eating dialogue | Zero-timestamp segments were falsely consumed during merge | Fixed merge predicate |
| Legacy sanitizer double-processing | Qwen output was being run through the Whisper-era sanitizer, mangling results | Disabled legacy sanitizer for Qwen pipeline |
| Python 3.9 accepted by preflight check | `preflight_check.py` said 3.9 was OK, but `pyproject.toml` requires 3.10+ | Threshold corrected to 3.10 |
| Colab notebook crash on None config | `colab_parallel` notebook hit `None` executable config | Added None handling |

---

## Known Issues

We're aware of these and working on them:

- **Qwen timestamp accuracy on long scenes** -- Scenes over 60s with background music can produce drifted timestamps. Assembly mode's 120s cap helps but doesn't fully solve it.
- **Qwen hallucination during silence** -- The model sometimes generates text during non-speech sections. The Japanese postprocessor catches some cases but not all.
- **Qwen on Apple Silicon** -- Falls back to CPU. MPS detection not yet implemented in qwen_asr.
- **Colab install script** -- Hardcodes `cu126` which is deprecated in PyTorch 2.7.x. May be intentional for certain LLM dependencies; under review.

---

## Installation Guide

### Important: We Recommend a Fresh Full Install

v1.8.3 introduces new dependencies from the Qwen3-ASR ecosystem (`qwen-asr`, `transformers`, `accelerate`, `hf_xet`, and others). If you're coming from v1.8.1 or earlier, a `--wheel-only` upgrade won't be enough -- you'll be missing packages that the Qwen pipeline needs.

**Our recommendation:** Do a full install. It takes longer, but it avoids half-working states where Whisper pipelines work but Qwen silently fails because a dependency is missing.

If you're already on v1.8.2-rc02 and just want the fixes, `whisperjav-upgrade` should work since the Qwen dependencies were already installed.

---

### Windows -- Standalone Installer (Most Users)

The easiest way. No Python knowledge needed.

1. **Download:** [WhisperJAV-1.8.3-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/download/v1.8.3/WhisperJAV-1.8.3-Windows-x86_64.exe) (~292 MB)
2. **Run the installer.** No admin rights required. Installs to `%LOCALAPPDATA%\WhisperJAV`.
3. **Wait 10-20 minutes.** It downloads and configures Python, PyTorch, FFmpeg, and all dependencies.
4. **Launch** from the Desktop shortcut.
5. **First run** downloads AI models (~3 GB, another 5-10 minutes).

**GPU auto-detection:** The installer checks your NVIDIA driver version and picks the right PyTorch:
- Driver 570+ gets CUDA 12.8 (optimal for RTX 20/30/40/50-series)
- Driver 450-569 gets CUDA 11.8 (broad compatibility)
- No NVIDIA GPU gets CPU-only mode

**Upgrading from v1.7.x?** Uninstall the old version first (Settings > Apps > WhisperJAV), then install fresh. Your models and output files are stored separately and won't be lost.

**Requires:** Windows 10/11 64-bit, 8 GB RAM minimum, Microsoft Edge WebView2 (for GUI).

---

### Windows -- Source Install (Developers)

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

The installer runs in 5 phases: PyTorch first (with GPU detection), then scientific stack, Whisper packages, audio/CLI tools, and optional extras. This order matters -- PyTorch must be installed before anything that depends on it, or you end up with CPU-only wheels.

**Manual alternative:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[all]"
```

For the full walkthrough, see [docs/guides/installation_windows_python.md](docs/guides/installation_windows_python.md).

---

### Linux (Ubuntu, Debian, Fedora, Arch)

**1. Install system packages first** -- these can't come from pip:

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

**NVIDIA GPU:** You need the NVIDIA driver (450+ or 570+) but NOT the CUDA Toolkit -- PyTorch bundles its own CUDA runtime.

**PEP 668 note:** If your distro's Python is "externally managed" (Ubuntu 24.04+, Fedora 38+), you'll need a virtual environment. The install script detects this and tells you what to do.

For the full walkthrough including Colab/Kaggle setup, headless servers, and systemd services, see [docs/guides/installation_linux.md](docs/guides/installation_linux.md).

---

### macOS (Apple Silicon)

New in v1.8.3: a dedicated macOS install script (`installer/install_mac.sh`) that replaces the old approach of running the Linux script on Mac.

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

### Google Colab / Kaggle

Use the one-click notebooks:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_parallel_expert.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_parallel_expert.ipynb)

---

### pip Install (Advanced)

If you prefer to manage everything yourself:

```bash
python -m venv whisperjav-env
source whisperjav-env/bin/activate  # Linux/Mac
# whisperjav-env\Scripts\activate   # Windows

# Install PyTorch FIRST
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # NVIDIA
# pip install torch torchaudio                                                    # Mac / CPU

# Install WhisperJAV with all extras
pip install "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git"
```

---

## System Requirements

| Requirement | Minimum | Recommended | For Qwen3-ASR |
|-------------|---------|-------------|---------------|
| **Python** | 3.10 | 3.11 or 3.12 | Same |
| **RAM** | 8 GB | 16 GB | 32 GB |
| **GPU VRAM** | 4 GB | 8 GB | 16+ GB |
| **Disk Space** | 8 GB | 15 GB (with models) | 50 GB (with Qwen models) |
| **OS** | Windows 10, macOS 13, Linux kernel 4.15+ | Latest versions | Same |

**GPU acceleration:**
- NVIDIA CUDA 11.8+ (Windows / Linux) -- Driver 450+ required
- Apple MPS (macOS Apple Silicon) -- Whisper pipelines only, Qwen runs on CPU for now
- CPU fallback on all platforms (slower, but works)

---

## Downloads

| File | Description |
|------|-------------|
| [WhisperJAV-1.8.3-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/download/v1.8.3/WhisperJAV-1.8.3-Windows-x86_64.exe) | Windows Standalone Installer (~292 MB) |
| [whisperjav-1.8.3-py3-none-any.whl](https://github.com/meizhong986/WhisperJAV/releases/download/v1.8.3/whisperjav-1.8.3-py3-none-any.whl) | Python Wheel (for upgrades) |

---

## Full Changelog

**42 commits since v1.8.2-rc02.** [v1.8.2-rc02...v1.8.3](https://github.com/meizhong986/WhisperJAV/compare/v1.8.2-rc02...v1.8.3)

---

*If you run into issues, please open a [GitHub issue](https://github.com/meizhong986/WhisperJAV/issues) with your platform, GPU, and the error output. That helps us fix things faster than a vague "it doesn't work".*
