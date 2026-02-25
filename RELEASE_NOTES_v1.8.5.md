v1.8.4 was a stabilization release for the Qwen3-ASR pipeline. This release is about tuning -- finding the right defaults so the pipeline produces good subtitles out of the box without manual parameter tweaking.

The headline change: **frame-native output is now the default**. In testing across T1–T6 configurations and aggressive/conservative sensitivity variants, `regroup_mode=off` (one subtitle per temporal frame) consistently produced the best results. The previous default (`standard`) would sometimes merge across speaker turns or produce over-long segments. Frame-native mode preserves the natural speech boundaries from the VAD/framer stage, which turns out to be exactly what you want for conversational JAV dialogue.

The other defaults have been tuned to match: Silero VAD v6.2 replaces v4.x, sensitivity presets are recalibrated, and scene padding is tightened to reduce subtitle overlaps at scene boundaries.

---
### What's Changed

| Enhancement | Description |
|------|-------------|
| **Frame-native default** | `regroup_mode` changed from `standard` to `off` across all entry points (CLI, GUI, ensemble, pipeline). Each temporal frame becomes one subtitle line — no merging, splitting, or regrouping. Best overall quality in testing. |
| **Silero VAD v6.2** | New speech segmentation backend with better speech/non-speech discrimination. Replaces v4.x as the default for Qwen pipeline. Tuned defaults: threshold 0.35, padding 250ms. |
| **Sensitivity preset overhaul** | E1–E5 presets recalibrated for Silero v6.2 + vad-grouped framer. Inverted padding-sensitivity relationship (aggressive = less padding, conservative = more). GUI sliders for manual VAD threshold/padding override. |
| **Wall-clock 8s subtitle cap** | New safety cap splits any subtitle exceeding 8 seconds of wall-clock duration, even if speech time is shorter. Prevents unreadable long-on-screen subtitles caused by silence gaps. |
| **Scene padding reduction** | Semantic scene detection padding reduced from ±0.5s to ±0.25s. Halves subtitle overlap at scene boundaries (~1.0s → ~0.5s) while still buffering for soft Japanese consonant onsets. |
| **GUI parameter guide** | Interactive (?) help icon opens a detailed 5-tab modal explaining every Qwen parameter, with defaults, CLI equivalents, and "when to change" guidance. |
| **GUI output controls** | Regroup mode dropdown, chunk threshold slider, and post-processing preset selector added to the Qwen output tab. |
| **Timestamp architecture fixes** | Four gaps in the timestamp mode matrix fixed: vad_only no longer loads aligner (saves VRAM), aligner_only skips sentinel recovery, vad_fallback correctly maps speech regions, vad_only hardening preserves frame boundaries. |
| **Frame-native reconstruction** | `reconstruct_frame_native()` builds one segment per frame group, replacing word-level output for `regroup_mode=off`. Fixed `max_group_duration_s` priority so explicit values aren't overridden by segmenter config. |
| **Branch B fixes** | `split_frame_to_words()` for sentence-level splitting in vad_only mode. `REGROUP_VAD_ONLY` algorithm without gap heuristics (synthetic timestamps). Crash recovery fallback in orchestrator. |
| **stable-ts integration hardening** | `suppress_silence=False` prevents destructive timestamp shrinking on ForcedAligner output. `force_order=True` handles ~10-20ms aligner timestamp overlaps that caused `UnsortedException`. |
| **REGROUP_JAV tuning** | Japanese-tuned regrouping: gap-split threshold 0.5→1.5s, merge-by-gap post-pass, Japanese comma support, 8s subtitle cap. (Now available as `--qwen-regroup standard` rather than the default.) |

---

### Bug Fixes

| Issue | What Happened | Fixed |
|-------|---------------|-------|
| #167 | Double log output causing subprocess pipe hang | Duplicate handler removal |
| #173 | TEN VAD TypeError when GUI Customize modal passes string values | Type coercion in config |
| Timestamp G1 | vad_only mode wasted VRAM loading forced aligner unnecessarily | `aligner=None` when `timestamp_mode==VAD_ONLY` |
| Timestamp G2 | vad_fallback hardening ignored speech_regions, producing wrong timestamps | Rewritten with anchor+gap mapping |
| Timestamp G3 | aligner_only mode ran sentinel recovery, altering raw aligner output | `skip_recovery` flag for assessment-only diagnostics |
| Timestamp G4 | vad_only hardening destroyed frame boundaries set by framer | Made `_apply_vad_only_timestamps()` a no-op |
| GUI | Step-down default was `False` in some code paths (should be `True`) | Corrected getattr fallbacks in main.py and pass_worker.py |
| GUI | Timestamp mode dropdown labels were unclear | Clarified: "Aligner Only (no recovery on collapse)", "VAD Only (no aligner loaded)" |
| Aligner | ForcedAligner occasional ~10-20ms timestamp overlaps caused UnsortedException | `force_order=True` in reconstruction |
| Hardening | `suppress_silence` destroyed ForcedAligner timestamps via crude loudness quantizer | Set `suppress_silence=False` for aligned workflow |

### Breaking Changes

None for end users. If you have custom scripts that:
- Expect `--qwen-regroup` to default to `standard` — now defaults to `off` (frame-native). Pass `--qwen-regroup standard` explicitly to restore old behavior.
- Expect semantic scene padding of ±0.5s — now ±0.25s. This may slightly shift scene boundaries.

---

## Installation Guide

### Upgrading from v1.8.4

Same dependency set — safe upgrade:

```bash
whisperjav-upgrade
```

Or wheel-only (code changes only, no dependency reinstall):

```bash
whisperjav-upgrade --wheel-only
```

---

### Windows -- Standalone Installer (Most Users)

The easiest way. No Python knowledge needed.

Recommended: Uninstall the old version first (Settings > Apps > WhisperJAV), then install fresh. Your models and output files are stored separately and won't be lost.

1. **Download:** WhisperJAV-1.8.5-Windows-x86_64.exe from below
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

## Full Changelog

**33 commits since v1.8.4**

[View full comparison](https://github.com/meizhong986/WhisperJAV/compare/v1.8.4...v1.8.5)
