# v1.8.7-beta.1 — Network Resilience for Chinese Users

Pre-release for testing. Feedback welcome via [Issues](https://github.com/meizhong986/WhisperJAV/issues).

---

## What's New in beta.1

- **#204 — SSL/VPN resilience for HuggingFace downloads** — Users behind corporate proxies or Chinese VPN services (e.g., v2rayN) no longer get blocked by SSL certificate validation errors when models are already cached locally. A single startup-time monkeypatch transparently retries failed HuggingFace Hub downloads from local cache. Covers all download paths: faster-whisper models, NeMo VAD, speech enhancement models, and ensemble subprocess workers.

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
pip install "whisperjav @ git+https://github.com/meizhong986/whisperjav.git@v1.8.7-beta.1"
```

Or for Mac users:
```bash
curl -fsSL https://raw.githubusercontent.com/meizhong986/whisperjav/main/install_mac.sh | bash
```
