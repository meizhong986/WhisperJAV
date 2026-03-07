# WhisperJAV v1.8.7-beta — Release Notes

> **Pre-release** — Please mark as pre-release on GitHub. Normal users will not be prompted to upgrade.
> Tag: `v1.8.7b0`

---

## Summary

This beta release fixes three critical issues affecting Apple Silicon (M1/M2/M3) and local LLM translation users, and adds several frequently-requested CLI controls for VAD tuning.

---

## Bug Fixes

### 🔴 Fix #196 — Local LLM translation: "No matches found" / timeouts (3 root causes)

Affects users translating with `llama-cpp-python` (gemma, mistral, llama, qwen models).

**Root cause 1 — Context overflow → parser failure:**
Japanese/CJK subtitle lines consume ~300 BPE tokens each (byte-level encoding), but the previous code had no output token cap. The server filled all remaining context with output, got truncated mid-translation (`finish_reason='length'`), and the PySubtrans parser received no `#N Translation>` markers → "No matches found". Fixed by `compute_max_output_tokens()` which sets a JAV/CJK-tuned `max_tokens` cap on the server.

**Root cause 2 — Streaming silently disabled:**
`CustomClient.enable_streaming = stream_responses AND supports_streaming`. `CustomClient` (the local LLM client) does not declare `supports_streaming=True` internally — only cloud clients do. Without this flag in provider config, streaming was silently disabled even when `stream=True` was passed. Fixed by adding `'supports_streaming': True` to `local_provider_config`.

**Root cause 3 — Non-streaming HTTP timeouts on slow backends:**
Without streaming, the HTTP connection blocks until the last token is generated. On slow MPS/CPU inference (francetoastVN's M1 Mac: 2-3 t/s), every batch exceeded the read timeout. Fixed by forcing `stream=True` for the local LLM path — streaming delivers tokens incrementally, so no read timeout can fire. Output is identical.

**What's unchanged:** Cloud provider (DeepSeek, Claude, Gemini, OpenRouter) streaming remains user-controlled via `--stream`.

**Validation:** 28-test E2E suite covers full streaming chain through `CustomClient`, Metal/MPS detection, `max_tokens` math, CLI source, and GUI wiring. 28/28 pass.

---

### 🔴 Fix #198 — Apple Silicon: MPS (Metal) not used for Transformers pipeline

`TransformersASR._detect_device()` only checked CUDA → CPU. MPS was never checked. On any Apple Silicon Mac, `--device auto` always fell through to CPU even when Metal was available.

Fixed: `auto` now probes CUDA first, then MPS, then CPU. Explicit `--device mps` also handled with graceful fallback to CPU. MPS dtype correctly set to `float16` (bfloat16 is not supported on MPS).

---

### 🟡 Fix — Metal/MPS backend incorrectly labeled as CUDA

`ServerDiagnostics` (local LLM server diagnostics) was setting `using_cuda=True` for any GPU offloading, including Apple Silicon Metal. This caused the status summary to print "GPU: 33 layers on CUDA" on M1/M2 Macs.

Fixed: `ServerDiagnostics` now has a dedicated `using_metal` field. `_parse_server_stderr()` detects `ggml_metal_init` (Metal) vs `ggml_cuda_init` (CUDA) from llama-cpp logs. Status correctly shows "GPU: 33 layers on Metal/MPS". Edge cases tested: M1 Mac with "CUDA not available" text, M2 Mac with ggml_cuda_init appearing before metal init — both correctly detected as Metal.

---

## New Features

### ✨ `--vad-threshold` and `--speech-pad-ms` CLI flags (#159)

Override VAD sensitivity and speech padding directly from the command line, without editing presets:

```bash
# Lower threshold → capture more (breathy/quiet speech)
whisperjav video.mp4 --vad-threshold 0.25

# Increase padding → preserve trailing Japanese particles (ね, よ, わ)
whisperjav video.mp4 --speech-pad-ms 400

# Per-pass control in two-pass ensemble
whisperjav video.mp4 --pass1-vad-threshold 0.3 --pass2-speech-pad-ms 300
```

These flags have the highest priority — they override sensitivity presets and GUI sliders.

---

### ✨ `--enhance-for-vad` dual-track speech enhancement

New mode for Qwen and Decoupled pipelines: enhanced audio is used only for VAD framing while ASR reads from the original (non-enhanced) audio. This prevents enhancement artifacts from degrading ASR accuracy while still benefiting from cleaner VAD boundaries.

```bash
whisperjav video.mp4 --qwen-generator anime-whisper \
  --qwen-enhancer clearvoice --enhance-for-vad
```

Also available per-pass: `--pass1-enhance-for-vad`, `--pass2-enhance-for-vad`.

---

### ✨ VAD padding defaults tuned for JAV

Based on field testing, the default padding values have been increased to better capture Japanese trailing particles (ね, よ, わ, の) and soft speech onsets:

| Backend | Parameter | Old | New |
|---------|-----------|-----|-----|
| Silero v6.2 | `speech_pad_ms` | 250ms | 350ms |
| TEN | `start_pad_ms` | 0ms | 50ms |
| TEN | `end_pad_ms` | 100ms | 150ms |

---

### ✨ GUI streaming default True

Local LLM streaming is now enabled by default in both GUI integration paths (integrated pipeline and standalone translate). This eliminates the timeout-per-batch behavior that francetoastVN reported on M1 Mac.

---

## What's Unchanged

- All cloud provider (DeepSeek, Claude, Gemini, OpenRouter) behavior is unchanged
- The `--stream` flag still controls cloud provider streaming
- All existing CLI flags remain compatible

---

## Known Issues / Not Fixed in This Release

- **#197** — Worker crash 0xC0000005 (Windows native crash during large-v2 load): awaiting GPU specs from reporter. Likely VRAM exhaustion with ctranslate2. Not reproducible without hardware info.
- **#198 Bug 2** — `kotoba-whisper-bilingual-v1.0` with `task='translate'` produces 0 subtitles: this is a model limitation, not a code bug. The model does not reliably honour the Whisper translate task. Workaround: use `--subs-language native` + separate `--translate`, or use `openai/whisper-large-v3`.

---

## Upgrading

```bash
# From installed WhisperJAV
whisperjav-upgrade

# From source
git pull && pip install -e ".[all]"
```
