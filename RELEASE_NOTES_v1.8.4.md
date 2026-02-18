# WhisperJAV v1.8.4 : Qwen stabilization + hardening

**Release Date:** February 2026

---

v1.8.3 shipped the Qwen3-ASR pipeline as a preview. This release stabilizes it. Most of the work here is invisible to you -- architectural cleanup, timestamp fixes, and making sure the three Qwen input modes (assembly, context-aware, VAD slicing) all go through the same hardening path instead of each having their own slightly-different copy.

If you've been using v1.8.3 and ran into timing issues, hallucination loops, or inconsistent results across Qwen input modes -- this release addresses those directly.

The existing Whisper pipelines (`faster`, `fast`, `balanced`, `fidelity`) are unchanged. If you're not using Qwen, this release is still worth taking for the scene detection improvements, the SRT translation GUI tab, and the bug fixes.

I also fixed an installer issue where the `enhance` extras (zipenhancer, modelscope, bs-roformer) were missing from some builds. If your v1.8.3 install was missing speech enhancement, a fresh v1.8.4 install will fix that.

---

## What Changed

| Area | What Changed |
|------|-------------|
| **Qwen assembly architecture** | Rebuilt around decoupled components (framers, generators, cleaners, aligners, hardening) with Protocol + Factory. Each step is independent, logged separately, with its own diagnostics. Replaces a monolithic 400-line method. |
| **Tighter scene/segment bounds** | Scene max: 120s/90s → 48s (all modes). VAD group max: 29s → 6s. Reduces forced aligner drift. Pipeline owns these defaults -- CLI/GUI only forward explicit overrides. |
| **Unified hardening** | All three Qwen modes now route through the same `harden_scene_result()`: timestamp resolution + boundary clamping + chronological sort. Previously, coupled modes were missing clamping and sorting. ~270 lines of duplicated code removed. |
| **Sentinel recovery fix** | Coupled modes were using `suppress_silence=True` for sentinel-recovered words, distorting redistributed timestamps. Now all modes use `suppress_silence=False`, matching assembly mode. |
| **Scene detection refactor** | Protocol + Factory with 4 backends (auditok, silero, semantic, none). All 7 pipelines + ensemble migrated to `SceneDetectorFactory`. Added `SafeSceneDetector` wrapper for crash recovery. |
| **GUI: SRT translation tab** | New tab for translating SRT files via AI providers (DeepSeek, Gemini, Claude, GPT, OpenRouter). Tab-aware file filtering, progress tracking, all CLI args available. |
| **SRT output to source folder** | SRT files now save next to the source video by default. Batch translate supports glob patterns: `whisperjav-translate -i "movies/*.srt"`. |
| **Local LLM reliability** | AVX2 pre-check (#157), steady-state speed gating with GOOD/MARGINAL/UNUSABLE classification (#149, #148), `ServerDiagnostics` return value. |
| **macOS GUI** | WebKit backend detection, updated `install_mac.sh` (#155). |
| **Installer fix** | Fallback requirements template was missing all enhance packages (modelscope, bs-roformer, onnxruntime, etc.) and qwen-asr. Rewritten to match pyproject.toml output. Fresh v1.8.4 install fixes missing speech enhancement. |

---

## Bug Fixes

| Issue | What Happened | Fixed |
|-------|---------------|-------|
| #125 | Cleanup crash on pipeline cancellation | Defensive cleanup handler |
| #163 | TEN speech segmenter timing drift | Config correction |
| #162 | ggml.dll not found in subprocess workers | DLL path forwarding |
| #149, #148 | Local LLM server startup failures, unreliable performance | AVX2 detection + viability gating |
| #157 | Crash on CPUs without AVX2 when loading llama.cpp | Pre-check with clear error message |
| F-01 | Orchestrator alignment batching sent wrong audio paths | Fixed batch construction |
| F-02 | Hardening speech regions array was stale after scene slicing | Fresh region extraction |
| Critical | Aligner adapter reading wrong attribute from `ForcedAlignResult` | Fixed attribute name |
| C1 | OOM from stale closure holding scene audio in memory | Closure cleanup |
| C2 | Wrong post-processor selected for Qwen output | Corrected processor routing |
| Installer | Enhance/qwen packages missing from fallback requirements template | Template rewritten |

---

## Logging and Diagnostics Improvements

Assembly mode now produces structured per-step logging:
- Step banners with scene index and duration
- Per-scene progress tracking across all orchestrator methods
- Enriched diagnostics (scene duration, input mode, word count, sentinel stats, timing sources, VAD regions)
- Phase 2 scene duration statistics
- Phase 5 assembly summary
- Batch summary logging from the text cleaner

These show up in `whisperjav.log` and in verbose console output. Useful for diagnosing why a particular scene produced bad subtitles.

---

## Breaking Changes

None for end users. If you have custom scripts that:
- Import `VAD_PARAMS` from `pass_worker.py` -- this was removed in a prior refactor
- Expect `--qwen-japanese-postprocess` to default to `True` -- now defaults to `False` (Qwen3 uses its own text cleaner instead)
- Expect `--qwen-max-group-duration` to default to 29 -- now defaults to 6 (pipeline-owned default)

---

## Installation Guide

> Important: Fresh Install Recommended
>
> v1.8.4 fixes an installer bug where enhance packages (zipenhancer, modelscope, bs-roformer) could be missing. If you had issues with speech enhancement on v1.8.3, do a fresh install rather than a wheel-only upgrade.

### Upgrading from v1.8.3

If your v1.8.3 install was working fine (including speech enhancement), this is a safe upgrade -- same dependency set. Use:

```bash
whisperjav-upgrade
```

Or for wheel-only (code changes only, no dependency reinstall):

```bash
whisperjav-upgrade --wheel-only
```

If speech enhancement was broken on your v1.8.3 install, do a full reinstall instead.

---

### Windows -- Standalone Installer (Most Users)

The easiest way. No Python knowledge needed.

Recommended: Uninstall the old version first (Settings > Apps > WhisperJAV), then install fresh. Your models and output files are stored separately and won't be lost.

1. **Download:** [WhisperJAV-1.8.4-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/download/v1.8.4/WhisperJAV-1.8.4-Windows-x86_64.exe)
2. **Run the installer.** No admin rights required. Installs to `%LOCALAPPDATA%\WhisperJAV`.
3. **Wait 10-20 minutes.** It downloads and configures Python, PyTorch, FFmpeg, and all dependencies.
4. **Launch** from the Desktop shortcut.
5. **First run** downloads models (~3 GB, another several minutes).

**GPU auto-detection:** The installer checks your NVIDIA driver version and picks the right PyTorch:
- Driver 570+ gets CUDA 12.8 (optimal for RTX 20/30/40/50-series)
- Driver 450-569 gets CUDA 11.8 (broad compatibility)
- No NVIDIA GPU gets CPU-only mode

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

### Google Colab / Kaggle

The notebooks are not updated yet for this release.

---

## Downloads

| File | Description |
|------|-------------|
| [WhisperJAV-1.8.4-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/download/v1.8.4/WhisperJAV-1.8.4-Windows-x86_64.exe) | Windows Standalone Installer |
| [whisperjav-1.8.4-py3-none-any.whl](https://github.com/meizhong986/WhisperJAV/releases/download/v1.8.4/whisperjav-1.8.4-py3-none-any.whl) | Python Wheel (for upgrades) |

---

## Full Changelog

**29 commits since v1.8.3.** [v1.8.3...v1.8.4](https://github.com/meizhong986/WhisperJAV/compare/v1.8.3...v1.8.4)

---

If you run into issues, please open a [GitHub issue](https://github.com/meizhong986/WhisperJAV/issues) with your platform, GPU, and the error output. That helps us fix things faster than a vague "it doesn't work" :)
