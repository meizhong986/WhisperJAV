# Frequently Asked Questions

---

## General

### What video formats does WhisperJAV support?

Any format FFmpeg can read: MP4, MKV, AVI, MOV, WMV, FLV, WAV, MP3, FLAC, M4A, M4B, and many more. If FFmpeg can extract audio from it, WhisperJAV can process it.

### How long does transcription take?

Depends on video length, pipeline, and GPU:

| Video Length | Faster (GPU) | Balanced (GPU) | Ensemble (GPU) | Faster (CPU) |
|-------------|-------------|----------------|----------------|-------------|
| 30 min | ~1 min | ~2 min | ~4 min | ~10 min |
| 2 hours | ~3 min | ~5 min | ~10 min | ~40 min |

These are rough estimates. Actual times vary with GPU model and audio complexity.

### Can I use WhisperJAV without a GPU?

Yes. Go to **Advanced** → check **"Accept CPU-only mode"**. Use **Faster** mode for best CPU speed. It works, just significantly slower.

---

## Quality

### Which pipeline gives the best subtitles?

For anime/JAV content: **ChronosJAV** with anime-whisper, or **Ensemble** (Balanced + Qwen3-ASR with Smart Merge) for maximum accuracy.

For general Japanese content: **Balanced** mode is the best single-pass option.

### The subtitles have hallucinated text (random English phrases, URLs, etc.)

This is a known Whisper behavior on silent or very quiet sections. WhisperJAV includes a hallucination sanitizer that removes most of these. Try:

1. Use **Aggressive** sensitivity (captures more speech, leaving less "silence" for hallucinations)
2. Use **Ensemble** mode (two passes catch different hallucinations)
3. Enable a **Speech Enhancer** (ClearVoice or BS-RoFormer) to clean the audio first

### The timing is off / subtitles appear too early or late

Try a different **Scene Detector**:

- **Semantic** (default) — best for most content
- **Auditok** — better for content with clear silence between dialogue
- **Silero** — more aggressive splitting

Or try **ChronosJAV** pipeline, which uses TEN VAD for tighter timing.

---

## Translation

### Which translation provider is best?

For cost-effectiveness: **DeepSeek** offers excellent quality at low cost, especially for CJK languages.

For quality: **Claude** or **GPT-4** produce the most natural translations, but at higher cost.

For privacy: **Local LLM** runs entirely on your machine with no data leaving your computer.

### Translation fails with "API token limit" error

Your subtitle batches are too large for the model's context window. Try:

1. Reduce **Max Batch Size** in Advanced Settings (try 15 or 10)
2. Use a model with a larger context window
3. WhisperJAV auto-caps batch size for local LLMs

### Translation shows "Unknown provider: Gemini" on Linux

Install the missing dependency: `pip install google-api-core`. Fixed since v1.8.6.

---

## Installation

### The installer is stuck / taking very long

The first install downloads ~3-5 GB of packages. On slow connections this can take 30+ minutes. Check the install log in the installation directory for progress.

### "CUDA not available" but I have an NVIDIA GPU

1. Verify your NVIDIA driver version: `nvidia-smi`
2. Driver 450+ required for CUDA 11.8, 570+ for CUDA 12.8
3. You do NOT need to install the CUDA Toolkit — PyTorch bundles its own
4. Try reinstalling PyTorch with the correct CUDA version

### How do I upgrade?

```bash
whisperjav-upgrade
```

Or for code-only (faster): `whisperjav-upgrade --wheel-only`

See the [Upgrade Guide](UPGRADE.md) for details.

---

## Models & Cache

### Where are the AI models stored?

WhisperJAV downloads models on first use and caches them on disk. The cache locations follow each library's standard convention — WhisperJAV does not override them:

| Cache | Default path (Linux/macOS) | Default path (Windows) | Used by |
|-------|---------------------------|------------------------|---------|
| HuggingFace | `~/.cache/huggingface/` | `%USERPROFILE%\.cache\huggingface\` | Kotoba, anime-whisper, Qwen3-ASR, WhisperSeg, transformers backend |
| OpenAI Whisper | `~/.cache/whisper/` | `%USERPROFILE%\.cache\whisper\` | Standard `large-v2` / `large-v3` / `turbo` models |
| Faster-Whisper | (uses HuggingFace cache above) | (uses HuggingFace cache above) | `fidelity` / `balanced` / `fast` modes via faster-whisper |
| PyTorch Hub | `~/.cache/torch/hub/` | `%USERPROFILE%\.cache\torch\hub\` | Silero VAD weights |
| ModelScope | `~/.cache/modelscope/` | `%USERPROFILE%\.cache\modelscope\` | ZipEnhancer / ClearVoice (when used) |
| WhisperJAV-internal | `~/.whisperjav_cache/` | `%LOCALAPPDATA%\.whisperjav_cache\` | Update checker cache, numba JIT cache redirect |

A typical WhisperJAV install ends up with **5-15 GB** in HuggingFace + Whisper caches after a few sessions, depending on which models you've used.

### How do I move the cache to a different drive?

Set the relevant environment variable **before** launching WhisperJAV. Each library reads its own variable:

| Cache to relocate | Env variable | Example value |
|-------------------|--------------|---------------|
| HuggingFace (most models) | `HF_HOME` | `D:\AI\hf_cache` |
| OpenAI Whisper | `XDG_CACHE_HOME` | `D:\AI\cache` (Whisper cache becomes `D:\AI\cache\whisper`) |
| PyTorch Hub | `TORCH_HOME` | `D:\AI\torch` |
| ModelScope | `MODELSCOPE_CACHE` | `D:\AI\modelscope` |

**Windows GUI users**: set these via System Properties → Environment Variables → User variables, then restart WhisperJAV. The GUI inherits its environment from your user profile.

**CLI users**: export them in your shell profile (`.bashrc` / `.zshrc`) or pass per-command.

After relocating, your old caches are still on disk in the original locations — copy or move them to the new path so models don't re-download.

### How do I free up disk space after uninstall?

The standalone installer removes the WhisperJAV install directory, but model caches in your home / AppData persist (libraries place them outside the install dir by design). To fully clean up:

1. Delete the cache directories listed in the table above
2. Delete the WhisperJAV-internal cache: `~/.whisperjav_cache` (Linux/macOS) or `%LOCALAPPDATA%\.whisperjav_cache` (Windows)
3. If you set custom `HF_HOME` etc. env vars, clean those paths instead

Total disk reclaimed varies — usually 5-15 GB.

### What if I have only 4 GB VRAM?

Model recommendations for 4 GB VRAM:

| Pipeline | Model | VRAM | Notes |
|----------|-------|------|-------|
| **fast** | `turbo-int8` | ~3 GB | Recommended starting point |
| **faster** | `turbo-int8-batch8` | ~5 GB (peak) | May still work via fallback to CPU offload |
| **balanced** | `large-v2-int8` | ~6 GB | Likely too large; use `fast` instead |

Tips:
- Avoid running other GPU-heavy applications simultaneously (Chrome, games, video encoding)
- Translation with local LLMs needs separate VRAM (3-8 GB depending on model). On a 4 GB GPU, prefer cloud translation providers (DeepSeek, OpenRouter) for the translation step.
- Speech enhancers (`clearvoice`, `bs-roformer`, `zipenhancer`) each need 1-3 GB extra. Use `none` or `ffmpeg-dsp` (CPU-only) if VRAM is tight.

---

## Troubleshooting

### Processing fails with an error

1. Check the **Console** at the bottom of the GUI for the error message
2. Enable **Debug logging** in Advanced tab and retry — check `whisperjav.log`
3. Try a simpler pipeline (**Faster** mode) to isolate the issue
4. Report the error on [GitHub Issues](https://github.com/meizhong986/whisperjav/issues) with the error message and your system details

### The GUI won't start

- **Windows:** Ensure WebView2 is installed (comes with Windows 10 1803+ and Windows 11)
- **Linux:** Install `libwebkit2gtk-4.0-dev` (Ubuntu) or `webkit2gtk4.0-devel` (Fedora)
- **macOS:** WebKit is built-in, should work automatically
- Try launching from command line (`whisperjav-gui`) to see error output
