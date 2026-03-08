# v1.8.7-beta.1 — Network Resilience for Chinese Users

Pre-release for testing. Feedback welcome via [Issues](https://github.com/meizhong986/WhisperJAV/issues).

---

## What's New in beta.1

### Network Resilience for Users in China (#204)

WhisperJAV now handles unstable network conditions gracefully — especially for users in China who connect through VPN/proxy services like v2rayN, Clash, or corporate proxies.

**The problem:** HuggingFace Hub (where AI models are hosted) always contacts its servers to check for updates before loading a model — even when the model is already fully downloaded and cached on your machine. In China, VPN proxy tools often break SSL certificate validation, causing these checks to fail with `CERTIFICATE_VERIFY_FAILED` errors. The result: WhisperJAV refuses to start even though all required models are already on disk.

**The fix:** WhisperJAV now automatically detects SSL/network failures and uses a 3-step fallback strategy. No configuration needed — it works transparently at startup.

**What you'll see (3-step fallback):**

1. **Step 1 — Normal download from huggingface.co:** Works as before. If this succeeds, nothing changes.
2. **Step 2 — Local cache fallback:** If Step 1 fails with a network/SSL error, WhisperJAV checks your local model cache. If the model was downloaded before, it loads from cache and continues normally. You'll see a warning in the log, but processing is not interrupted.
3. **Step 3 — China mirror (hf-mirror.com):** If the model is not in your local cache, WhisperJAV automatically tries downloading from `hf-mirror.com`, the official HuggingFace mirror for China. This works even when your VPN blocks `huggingface.co`. Once downloaded, the model is cached locally for future use.

If all 3 steps fail, WhisperJAV shows a diagnostic summary with:
- The exact model name and download URLs (both huggingface.co and hf-mirror.com)
- Your local cache directory path, so you can download and place files manually
- The `HF_ENDPOINT` environment variable you can set as a permanent workaround

**Coverage:** This protection applies to all AI model downloads — Whisper models (faster-whisper), VAD models (NeMo/Silero), speech enhancement models (ZipEnhancer, BS-RoFormer, ClearVoice), and ensemble mode subprocess workers. Every entry point (CLI, GUI, ensemble workers) is protected.

**For users in China:** In most cases, WhisperJAV will now "just work" — even on first run — because the China mirror fallback handles the initial download automatically. No need to disconnect your VPN.

## Carried from beta.0

### Enhancements

- **Local LLM translation reliability** — Fixed token limit handling for small-context models (8K). Streaming is now correctly enabled for local LLM servers, and max_tokens is properly capped to prevent truncated translations. (#196)

- **Apple Silicon (MPS) support** — Metal GPU acceleration now works across more components:
  - Transformers pipeline detects and uses MPS instead of falling back to CPU (#198)
  - Speech enhancement backends (ZipEnhancer, BS-RoFormer) try MPS with automatic CPU fallback
  - Installer correctly installs PyTorch with MPS support on Apple Silicon (no longer pulls CPU-only wheels)
  - Prebuilt Metal wheel support for llama-cpp-python on Mac

- **VAD controls** — New `--vad-threshold` and `--speech-pad-ms` CLI flags for direct control over voice activity detection sensitivity, with per-pass overrides for ensemble mode. (#159)

- **Enhanced VAD for ChronosJAV pipelines** — Decoupled and Qwen pipelines now support speech segmentation with Silero v6.2 and TEN backends, giving these pipelines the same VAD quality as the classic balanced pipeline.

- **ZipEnhancer dependency fix** — The `enhance` extra now correctly declares all required ModelScope dependencies (einops, oss2, addict, attrs, datasets, etc.), eliminating manual pip installs after `pip install whisperjav[enhance]`.

### Bug Fixes

- **#196** — Local LLM translation failing with "No matches found" on 8K context models. Three root causes fixed: max_tokens cap, forced streaming, and streaming support flag in CustomClient.
- **#198** — Apple Silicon Macs running Transformers pipeline on CPU instead of MPS. MPS device detection added with proper float16 dtype handling.
- **#195** — UnicodeDecodeError when processing files with Japanese metadata (M4A/M4B). Added `errors='replace'` to FFmpeg subprocess calls. *(fix backported to v1.8.6)*

## Installation

```bash
pip install "whisperjav @ git+https://github.com/meizhong986/whisperjav.git@v1.8.7b1"
```

Or for Mac users:
```bash
curl -fsSL https://raw.githubusercontent.com/meizhong986/whisperjav/main/install_mac.sh | bash
```
