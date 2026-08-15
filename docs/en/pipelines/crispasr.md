# CrispASR Pipeline

CrispASR is a self-contained external ASR engine (a C++/ggml binary from the [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR) project). WhisperJAV shells out to it the same way Subtitle Edit does — WhisperJAV does **not** bundle the engine; you provide the executable.

---

## Backends

| Backend | Engine | Notes |
|---------|--------|-------|
| **parakeet** *(default)* | NVIDIA Parakeet | Native word timestamps; good speed/quality |
| **whispercpp** | whisper.cpp (large-v2) | Native word timestamps; alias of crispasr's `whisper` backend |
| **cohere** | Cohere Transcribe | Native word timestamps; open ggml asset (no token) |

All three need **no forced aligner**. The model GGUF auto-downloads on first use.

---

## Prerequisites

CrispASR needs its **external engine binary** (not shipped with WhisperJAV):

1. Download a release archive from [CrispStrobe/CrispASR releases](https://github.com/CrispStrobe/CrispASR/releases) matching your OS and GPU:
    - Windows + NVIDIA → `crispasr-windows-x86_64-cuda.zip`
    - Windows, no GPU → `crispasr-windows-x86_64-cpu.zip` (or `-vulkan`)
    - macOS → `crispasr-macos.tar.gz` · Linux → `crispasr-linux-x86_64.tar.gz`
2. Unzip it to a folder and note the path to the `crispasr` executable.

`ffmpeg` must be on `PATH` (already a WhisperJAV requirement). No HuggingFace token is needed for the curated backends.

---

## How to Use

### GUI (Transcription Tab)

1. Set **Mode** to **crispasr**
2. In the CrispASR panel: **Browse** to the `crispasr` executable, pick a **Backend**
3. Click **Start**

### GUI (Ensemble Tab)

1. Set a pass **Pipeline** to **CrispASR**
2. Choose the **Backend** in that pass's Model column; set the executable in the CrispASR panel
3. Click **Start**

### CLI

```bash
whisperjav video.mp4 --mode crispasr --crispasr-exe /path/to/crispasr --crispasr-backend parakeet
```

As an ensemble pass:

```bash
whisperjav video.mp4 --ensemble --pass1-pipeline balanced --pass2-pipeline crispasr --crispasr-exe /path/to/crispasr --crispasr-backend whispercpp
```

Pass advanced engine flags verbatim with `--crispasr-args "..."`.

---

## Where Models Are Downloaded

The engine self-manages models. On first use of a backend it downloads the GGUF (~150 MB – ~1 GB) from HuggingFace (`cstr/…-GGUF`) into the cache directory. WhisperJAV points `CRISPASR_CACHE_DIR` at `~/.cache/whisperjav/crispasr_models` unless you set it yourself. Requires outbound HTTPS and `curl`/`wget` on `PATH` (Windows also tries WinHTTP). Subsequent runs reuse the cache.

---

## Requirements

- WhisperJAV installed (no extra Python package needed — the pipeline is built in)
- `ffmpeg` on `PATH`
- The CrispASR engine binary (user-supplied, see Prerequisites)
- Network access on first run per backend

---

## Strengths and Limitations

**Strengths:**

- Alternative engines (Parakeet / whisper.cpp / Cohere) in one external provider
- Self-contained — handles its own VAD; no WhisperJAV scene/segmenter/enhancer tuning needed
- GPU variants (CUDA / Vulkan / CPU) chosen by which engine build you download

**Limitations:**

- The engine binary is not bundled — you download it once yourself
- Models pull from a mutable HuggingFace branch with no checksum (engine behavior)
- `whispercpp` requests large-v2 via the engine's auto mechanism — confirm it resolves on your build (override with `--crispasr-args -m <...>` if needed)
- `--subs-language direct-to-english` is not specially handled for CrispASR
- First run per backend pauses for the model download

---

## For Maintainers & Releases

- **Installer:** no change required. The pipeline is pure Python inside the `whisperjav` package; no new pip or system dependency (FFmpeg is already required). Do **not** bundle the engine binary (separate project; CUDA/Vulkan/CPU variant matrix).
- **GitHub release:** WhisperJAV's release assets are unchanged (the installer/wheel already contain the pipeline code). The engine binary is **not** a WhisperJAV asset — it lives in CrispStrobe/CrispASR releases. Models are in no release; they auto-download from HuggingFace at first use.
