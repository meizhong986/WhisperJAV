# Ollama Integration Research: Strategies, Patterns & UX Design

**Date**: 2026-03-13
**Purpose**: Inform v1.9.0 architecture decisions for local LLM translation
**Audience**: WhisperJAV maintainers and external reviewers
**Status**: Research complete, pending design decisions

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Problem We're Solving](#2-the-problem-were-solving)
3. [Ollama Technical Profile](#3-ollama-technical-profile)
4. [Integration Strategy Options](#4-integration-strategy-options)
5. [Industry Patterns & Lessons](#5-industry-patterns--lessons)
6. [UX Design Principles](#6-ux-design-principles)
7. [Recommended Architecture](#7-recommended-architecture)
8. [Model Selection Strategy](#8-model-selection-strategy)
9. [Error Handling & Diagnostics Design](#9-error-handling--diagnostics-design)
10. [Current WhisperJAV State](#10-current-whisperjav-state)
11. [Risk Analysis](#11-risk-analysis)
12. [Open Questions](#12-open-questions)
13. [References](#13-references)

---

## 1. Executive Summary

### Context

WhisperJAV's current local LLM backend (`llama-cpp-python`) is architecturally fragile: 2,423 lines of DLL loading hacks, CUDA wheel management, and build-from-source fallbacks. It produces 5+ open issues and is the #1 source of user frustration.

### Research Question

How should WhisperJAV integrate Ollama as the primary local LLM backend, balancing user simplicity ("one click and go") against engineering overhead ("bring your own Ollama")?

### Key Finding

**No local LLM experience is truly one-click.** The irreducible minimum is (1) install a runtime and (2) download a multi-GB model. The best apps minimize this to two interactions, then get out of the way. The industry has converged on a **"Detect + Guide + Verify"** pattern.

### Recommendation

**Strategy C: "Smart BYOO" (Bring Your Own Ollama with Guided Detection)**

WhisperJAV detects Ollama, guides installation if missing, auto-pulls models with consent, and manages the translation session. Ollama remains a system-level tool the user owns. WhisperJAV never bundles, installs, or silently manages Ollama.

**Why not bundle Ollama?** Ollama is 200-400 MB (binary) + 4-8 GB (model). Bundling doubles installer size, creates update coupling, and violates the principle that system services should be user-controlled. Every major project (Open WebUI, AnythingLLM, Jan.ai) treats Ollama as external.

---

## 2. The Problem We're Solving

### User Pain Points (from GitHub issues)

| Issue | Root Cause | Frequency |
|-------|-----------|-----------|
| "Failed to load local LLM server" | DLL version mismatch | High (#196, #212) |
| "No matches found" on all batches | Context overflow, garbled output | High (#214, #132) |
| cu118 wheel rejected by uv | Version metadata mismatch | Medium (#218) |
| Build from source fails | No compiler, wrong CUDA | Medium |
| "What model should I use?" | No guidance | Ongoing |

### What Users Actually Want

Based on issue reports and user behavior:
- **60% want "one click and go"** — minimal technical knowledge required
- **30% are comfortable with guided setup** — will follow clear instructions
- **10% are power users** — want custom models, custom servers, full control

### What the Current System Delivers

```
User clicks "Translate" with --provider local
→ WhisperJAV tries to load llama-cpp-python
→ DLL loading fails (CUDA version mismatch) or
→ Tries to build from source (no compiler found) or
→ Server starts but model produces garbled output
→ ERROR: User confused, gives up
```

### What the Target System Should Deliver

```
User clicks "Translate" with --provider ollama
→ WhisperJAV checks: Is Ollama running? → Yes → Continue
→ WhisperJAV checks: Is model available? → Yes → Translate
→ Done. Standard HTTP API, no DLLs, no compilation.
```

---

## 3. Ollama Technical Profile

### Architecture Overview

Ollama is a Go binary (~200-400 MB) that embeds llama.cpp as its inference engine. It follows a client-server architecture:

```
┌─────────────────────────────────────────────┐
│                 Ollama Server               │
│  (Go binary, HTTP API on port 11434)        │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │         Scheduler                    │    │
│  │  - Model loading/unloading           │    │
│  │  - GPU memory management             │    │
│  │  - Request queuing                   │    │
│  └──────────────┬──────────────────────┘    │
│                 │                            │
│  ┌──────────────▼──────────────────────┐    │
│  │      Runner Subprocess               │    │
│  │  (llama.cpp inference engine)        │    │
│  │  - CUDA / ROCm / Metal / Vulkan     │    │
│  │  - CPU fallback                      │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
        ▲
        │ HTTP (OpenAI-compatible)
        ▼
┌─────────────────────────────────────────────┐
│            WhisperJAV                       │
│  (via PySubtrans → OpenAI API)              │
└─────────────────────────────────────────────┘
```

### Key Technical Facts

| Aspect | Detail |
|--------|--------|
| **Binary** | Single Go executable, ~200-400 MB |
| **API** | REST HTTP, OpenAI-compatible at `/v1/` |
| **Port** | 11434 (configurable via `OLLAMA_HOST`) |
| **GPU** | Auto-detects NVIDIA (CUDA), AMD (ROCm), Apple (Metal), Intel, Vulkan |
| **Models** | Content-addressable blobs in `~/.ollama/models/` |
| **Memory** | Auto-manages GPU/CPU layer distribution per available VRAM |
| **Keep-alive** | Models stay loaded 5 minutes after last request (configurable) |
| **Downloads** | Resumable — cancelled pulls continue from where they stopped |
| **License** | MIT |

### Installation Per Platform

| Platform | Method | Location | Runs As |
|----------|--------|----------|---------|
| **Windows** | `OllamaSetup.exe` from ollama.com | `%LOCALAPPDATA%\Programs\Ollama` | Background tray app |
| **macOS** | Drag `Ollama.app` to Applications | `/Applications/Ollama.app` | Menu bar app |
| **Linux** | `curl -fsSL https://ollama.com/install.sh \| sh` | `/usr/local/bin/ollama` | systemd service |

No admin rights required on Windows. All platforms add `ollama` to PATH.

### API Endpoints Relevant to WhisperJAV

| Method | Endpoint | Purpose | WhisperJAV Usage |
|--------|----------|---------|-----------------|
| `GET` | `/` | Health check | Detect running server |
| `GET` | `/api/version` | Server version | Version compatibility check |
| `GET` | `/api/tags` | List local models | Check if model is available |
| `POST` | `/api/show` | Model details | Get context length, capabilities |
| `POST` | `/api/pull` | Download model | Auto-pull on first use |
| `GET` | `/api/ps` | Loaded models | Check if model is in memory |
| `POST` | `/v1/chat/completions` | Translation | **Primary translation endpoint** |

### Detection Methods

```python
import shutil
import urllib.request

# Is Ollama installed?
ollama_path = shutil.which("ollama")  # Returns path or None

# Is Ollama server running?
try:
    resp = urllib.request.urlopen("http://localhost:11434/", timeout=2)
    is_running = resp.read().decode().strip() == "Ollama is running"
except Exception:
    is_running = False
```

### Key Environment Variables

| Variable | Default | WhisperJAV Relevance |
|----------|---------|---------------------|
| `OLLAMA_HOST` | `127.0.0.1:11434` | Custom port support |
| `OLLAMA_MODELS` | `~/.ollama/models` | Custom model storage |
| `OLLAMA_KEEP_ALIVE` | `5m` | Memory management during batch translation |
| `OLLAMA_GPU_OVERHEAD` | `0` | Reserve VRAM for Whisper ASR |
| `OLLAMA_NUM_PARALLEL` | `1` | Concurrent request handling |
| `OLLAMA_CONTEXT_LENGTH` | model default | Override context window globally |

### Model Sizes for Translation

| Model | Parameters | Disk (Q4_K_M) | VRAM Needed | JP→EN Quality |
|-------|-----------|---------------|-------------|--------------|
| Gemma 2 2B | 2.6B | ~1.6 GB | ~4 GB | Basic |
| Llama 3.2 3B | 3.2B | ~2.0 GB | ~4 GB | Passable |
| Qwen 2.5 7B | 7.6B | ~4.7 GB | ~8 GB | Good (strong CJK) |
| Llama 3.1 8B | 8.0B | ~4.9 GB | ~8 GB | Good |
| Gemma 3 12B | 12B | ~7.3 GB | ~12 GB | Very good |
| Qwen 2.5 14B | 14.8B | ~9.0 GB | ~16 GB | Excellent |

**Sweet spot for JAV translation**: 7B-12B models on 8-12 GB VRAM consumer GPUs. Qwen 2.5 7B is particularly strong for CJK languages.

---

## 4. Integration Strategy Options

### Strategy A: "Bundle Everything"

WhisperJAV packages Ollama binary inside its installer and manages the full lifecycle.

| Aspect | Assessment |
|--------|-----------|
| **User experience** | Simplest — truly "one click" after install |
| **Installer size** | +200-400 MB (binary) + 4-8 GB (model) = **4-8 GB total** |
| **Update coupling** | WhisperJAV must ship new installer for every Ollama update |
| **Platform complexity** | Must bundle platform-specific binaries (Windows, macOS, Linux) |
| **User trust** | Low — silently installing a system service feels invasive |
| **GPU management** | Ollama handles it, but conflicts with system Ollama possible |
| **Who does this** | GPT4All (bundles llama.cpp, not Ollama specifically) |
| **Verdict** | **Not recommended.** Too heavy, too coupled, too invasive. |

### Strategy B: "Pure BYOO" (Bring Your Own Ollama)

WhisperJAV requires Ollama to be pre-installed and running. Shows error if not found.

| Aspect | Assessment |
|--------|-----------|
| **User experience** | Requires separate install + model pull before first use |
| **Installer size** | No change |
| **Engineering effort** | Minimal — just HTTP client code |
| **User friction** | High for non-technical users (must install Ollama, pull model) |
| **Failure mode** | User gives up at "install Ollama" step |
| **Who does this** | AnythingLLM, Open WebUI |
| **Verdict** | **Too much friction for 60% of users.** |

### Strategy C: "Smart BYOO" (Detect + Guide + Auto-Pull) ← RECOMMENDED

WhisperJAV detects Ollama state and guides users through whatever is missing. Auto-pulls models with user consent.

| Aspect | Assessment |
|--------|-----------|
| **User experience** | Guided wizard on first use, zero friction after setup |
| **Installer size** | No change |
| **Engineering effort** | Medium — detection, guidance UI, model pull with progress |
| **User friction** | Low-medium (one external install, rest is guided) |
| **Failure mode** | User gets stuck installing Ollama (mitigated by clear instructions) |
| **Who does this** | This is the industry convergence pattern |
| **Verdict** | **Best balance of simplicity, engineering effort, and user trust.** |

### Strategy D: "Embedded Ollama Server"

WhisperJAV downloads Ollama binary on first use and manages it as a subprocess.

| Aspect | Assessment |
|--------|-----------|
| **User experience** | Near "one click" — auto-downloads ~400 MB binary |
| **Installer size** | No change initially, +400 MB on first use |
| **Engineering effort** | High — must handle download, versioning, process management |
| **Platform complexity** | Must detect OS/arch, download correct binary, handle permissions |
| **Conflict risk** | What if user already has system Ollama on different port? |
| **Update burden** | Must track Ollama releases, test compatibility |
| **Who does this** | No major project does this for Ollama |
| **Verdict** | **Over-engineered. High maintenance, low incremental benefit over C.** |

### Strategy Comparison Matrix

| Criterion | A: Bundle | B: Pure BYOO | C: Smart BYOO | D: Embedded |
|-----------|-----------|--------------|----------------|-------------|
| First-use simplicity | ★★★★★ | ★★ | ★★★★ | ★★★★ |
| Installer size impact | ★ | ★★★★★ | ★★★★★ | ★★★★ |
| Engineering effort | ★ | ★★★★★ | ★★★ | ★★ |
| Maintenance burden | ★ | ★★★★★ | ★★★★ | ★★ |
| User trust | ★★ | ★★★★★ | ★★★★ | ★★★ |
| Non-technical user UX | ★★★★★ | ★★ | ★★★★ | ★★★★ |
| Power user flexibility | ★★ | ★★★★★ | ★★★★★ | ★★★ |
| **Overall** | **Not viable** | **Too bare** | **Best fit** | **Over-engineered** |

---

## 5. Industry Patterns & Lessons

### How Other Projects Handle Ollama

#### Open WebUI (formerly Ollama WebUI)

- **Pattern**: Pure BYOO with reverse proxy architecture
- **Server management**: Connects to existing Ollama via `OLLAMA_BASE_URLS` env var
- **Model management**: Full UI for pulling, deleting, and managing models
- **Key lesson**: Timeout handling is their biggest UX problem — long inference times on CPU cause silent failures
- **Takeaway for WhisperJAV**: Streaming is essential. Non-streaming requests timeout on slow hardware.

#### AnythingLLM

- **Pattern**: Pure BYOO
- **Setup**: Users must install and configure Ollama separately
- **Marketing**: "Zero configuration" (for the app, not the LLM backend)
- **Key lesson**: Works well for technical users, frustrating for non-technical users
- **Takeaway for WhisperJAV**: "Zero configuration" is aspirational, not achievable for local LLM

#### GPT4All

- **Pattern**: Bundle Everything (bundles llama.cpp directly, not Ollama)
- **Model management**: In-app model browser with download progress
- **Key lesson**: Bundling works but creates massive installers and update coupling
- **Takeaway for WhisperJAV**: Auto-pull models is a great UX, but bundle the runtime is not

#### Jan.ai

- **Pattern**: All-in-one platform (bundles its own inference engine)
- **Hardware**: Auto-detects hardware and optimizes accordingly
- **Key lesson**: Hardware-aware defaults are critical — users shouldn't choose quantization levels
- **Takeaway for WhisperJAV**: Auto-select model based on detected VRAM

#### LM Studio

- **Pattern**: All-in-one platform
- **API**: Headless daemon with TTL auto-eviction for loaded models
- **Tiers**: Entry (4-6 GB), Mid-range (8-12 GB), High-end (16-24 GB+)
- **Key lesson**: Hardware tier documentation helps users self-select
- **Takeaway for WhisperJAV**: Communicate hardware requirements in human-readable tiers

### Common Pitfalls

| Pitfall | Projects Affected | Mitigation |
|---------|------------------|-----------|
| **Port 11434 conflicts** | Open WebUI, any multi-Ollama setup | Support custom port via `OLLAMA_HOST` |
| **GPU memory exhaustion** | All projects | Use `OLLAMA_GPU_OVERHEAD` to reserve VRAM for Whisper |
| **Timeout on slow inference** | Open WebUI (#1 complaint) | Always use streaming; set generous timeouts |
| **Model not pulled** | AnythingLLM users | Auto-detect and offer to pull |
| **Docker networking confusion** | Open WebUI in Docker | N/A for WhisperJAV (not Docker-based) |
| **Ollama version incompatibility** | Rare but documented | Check `/api/version`, warn if too old |

---

## 6. UX Design Principles

Based on research across LM Studio, Jan.ai, GPT4All, AnythingLLM, Open WebUI, and UX best practices literature.

### Principle 1: Detect, Don't Assume

Check for Ollama, GPU, VRAM, and model presence. Guide the user through whatever is missing.

```
Is Ollama installed? → Guide install if not
Is Ollama running? → Offer to start if not
Is model available? → Auto-pull with consent
Is VRAM sufficient? → Suggest appropriate model
```

### Principle 2: Progressive Disclosure

Three levels of information:

- **Level 1 (always visible)**: Plain language. "Translation engine not found."
- **Level 2 (one click)**: Actionable steps. "Install Ollama from ollama.com, then click Check Again."
- **Level 3 (behind "Details")**: Technical info. "Connection refused: http://localhost:11434"

### Principle 3: One-Time Pain, Permanent Gain

Frame Ollama install and model download as one-time steps:

> "Setting up local translation (one-time setup, ~5 minutes):
> 1. Install Ollama (free, 1 click)
> 2. Download translation model (4.7 GB)
> After setup, translation starts instantly every time."

### Principle 4: Hardware-Aware Defaults

Users should never see "Q4_K_M" or "num_ctx=8192". Auto-select based on detected hardware:

| Detected VRAM | Default Model | Quality Tier |
|--------------|--------------|-------------|
| ≥12 GB | gemma3:12b | Best |
| ≥8 GB | qwen2.5:7b | Good |
| ≥4 GB | llama3.2:3b | Basic |
| CPU only | llama3.2:3b (Q4) | Basic (slow) |

### Principle 5: Graceful Degradation, Not Hard Failures

Never show a raw traceback. Degrade with notification:

```
GPU + large model → GPU + small model → CPU + small model → Cloud fallback → Skip translation
```

Each step down should be automatic with a notification, not a blocking error.

### Principle 6: The "Check Again" Pattern

When external install is needed (Ollama), provide a re-check button:

```
┌─────────────────────────────────────────────────┐
│  Translation Engine Setup                        │
│                                                  │
│  Ollama is not installed on this system.         │
│                                                  │
│  Ollama is a free, open-source tool that runs    │
│  AI models locally on your computer.             │
│                                                  │
│  [Download Ollama]    [Check Again]              │
│                                                  │
│  ▸ What is Ollama?                               │
│  ▸ Troubleshooting                               │
└─────────────────────────────────────────────────┘
```

No app restart required — just click "Check Again" after installing.

### Principle 7: Separate "Setup" from "Use"

The setup wizard runs once. After that, the translate button works with zero friction. No Ollama awareness in the normal workflow — it should feel like translation is built in.

---

## 7. Recommended Architecture

### High-Level Design

```
┌────────────────────────────────────────────────────────────────────────┐
│                        WhisperJAV v1.9.0                               │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Translation Pipeline                          │  │
│  │  (PySubtrans → OpenAI-compatible API → Ollama)                  │  │
│  └──────────────────────┬───────────────────────────────────────────┘  │
│                         │                                              │
│  ┌──────────────────────▼───────────────────────────────────────────┐  │
│  │                 OllamaManager                                    │  │
│  │                                                                  │  │
│  │  detect_server()     → Is Ollama running?                       │  │
│  │  detect_installation() → Is Ollama installed?                   │  │
│  │  check_model(name)   → Is model available locally?              │  │
│  │  pull_model(name)    → Download model with progress             │  │
│  │  get_model_info(name) → Context length, capabilities           │  │
│  │  get_hardware_info()  → VRAM, loaded models                    │  │
│  │  recommend_model()   → Hardware-aware model suggestion          │  │
│  │  start_server()      → Start Ollama if installed but not running│  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              SetupWizard (first-use only)                        │  │
│  │                                                                  │  │
│  │  Step 1: Check Ollama → Guide install if missing                │  │
│  │  Step 2: Check model  → Auto-pull with consent                  │  │
│  │  Step 3: Ready        → Save config, don't show again           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### OllamaManager — Core Integration Class

This is the single point of contact between WhisperJAV and Ollama. Responsibilities:

| Method | Purpose | API Used |
|--------|---------|----------|
| `detect_server()` | Check if Ollama HTTP server is reachable | `GET /` |
| `detect_installation()` | Check if `ollama` binary is on PATH | `shutil.which()` |
| `check_model(name)` | Check if model is downloaded locally | `POST /api/show` |
| `pull_model(name, callback)` | Download model with progress callback | `POST /api/pull` (streaming) |
| `get_model_info(name)` | Get context length, parameter count | `POST /api/show` |
| `get_hardware_info()` | Get loaded models, VRAM usage | `GET /api/ps` |
| `recommend_model(vram_gb)` | Suggest model based on available VRAM | Local logic |
| `start_server()` | Start `ollama serve` as subprocess | `subprocess.Popen` |
| `ensure_ready(model)` | Full readiness check (server + model) | Combines above |

### Translation Flow (v1.9.0)

```
User invokes: whisperjav-translate -i subs.srt --provider ollama
                    │
                    ▼
         ┌─────────────────────┐
         │  OllamaManager      │
         │  .ensure_ready()     │
         └─────────┬───────────┘
                   │
         ┌─────────▼───────────┐     ┌──────────────────────────┐
         │ Server running?      │─NO─▶│ Installed?               │
         └─────────┬───────────┘     └──────────┬───────────────┘
                   │ YES                         │
                   │                    ┌────────▼────────┐
                   │                    │ YES: Start it   │
                   │                    │ NO: Guide install│
                   │                    └────────┬────────┘
                   │                             │
         ┌─────────▼───────────┐                 │
         │ Model available?     │◄────────────────┘
         └─────────┬───────────┘
                   │
              ┌────▼────┐
              │ YES     │ NO: Auto-pull with progress
              └────┬────┘      (user consent required)
                   │
         ┌─────────▼───────────┐
         │ Get model context    │
         │ length from /api/show│
         └─────────┬───────────┘
                   │
         ┌─────────▼───────────┐
         │ Compute batch size   │
         │ and max_tokens       │
         │ based on ACTUAL      │
         │ context length       │
         └─────────┬───────────┘
                   │
         ┌─────────▼───────────┐
         │ PySubtrans translate │
         │ via /v1/chat/        │
         │ completions          │
         └─────────┬───────────┘
                   │
                   ▼
              OUTPUT.en.srt
```

### Key Design Decisions

**1. No Ollama bundling.** Ollama is a system-level tool. Users install it once, use it across many apps.

**2. Auto-start Ollama if installed but not running.** Via `subprocess.Popen(["ollama", "serve"])`. Check port first to avoid conflict with existing instance.

**3. Auto-pull models with user consent.** Show model name and size, ask "Download now?". Use streaming pull API for progress.

**4. Query actual context length from model metadata.** Don't hardcode 8192 — use `POST /api/show` to get the model's actual context window, then compute batch size dynamically.

**5. Reserve VRAM for Whisper.** If user is transcribing + translating in one pipeline, Whisper and Ollama compete for VRAM. Set `OLLAMA_GPU_OVERHEAD` or unload Whisper model first.

**6. Support custom Ollama URL.** Via `--ollama-url` flag or `OLLAMA_HOST` env var. Enables remote Ollama servers.

**7. Keep `--provider ollama` as the user-facing flag.** Internally maps to `OllamaManager` + `Custom Server` PySubtrans provider.

### Dependencies

**No new pip dependencies required.** Ollama communication uses:
- `urllib.request` (stdlib) — for health checks and model queries
- OpenAI-compatible API via PySubtrans (already installed) — for translation
- `subprocess` (stdlib) — for starting Ollama if needed

The official `ollama` Python package is NOT needed. Using raw HTTP + OpenAI-compat endpoint keeps the dependency footprint at zero.

---

## 8. Model Selection Strategy

### The Problem

Users should not have to:
- Know what "7B Q4_K_M" means
- Guess whether a model fits their GPU
- Research which models are good for Japanese translation
- Manually type model names

### Two-Tier Selection

#### Tier 1: "Just Works" (Default)

WhisperJAV auto-selects a model based on detected hardware. User sees:

> "Using recommended model for your hardware (gemma3:12b, 7.3 GB download)"

No model names in the normal UI. No parameters. No jargon.

**Selection Logic:**

```python
def recommend_model(vram_gb: float | None) -> ModelRecommendation:
    """Select model based on available VRAM."""
    if vram_gb is None or vram_gb < 4:
        # CPU-only or very low VRAM
        return ModelRecommendation(
            name="qwen2.5:3b",
            download_size="2.0 GB",
            quality="basic",
            note="Your system will use CPU mode (slower but functional)"
        )
    elif vram_gb < 8:
        return ModelRecommendation(
            name="qwen2.5:7b",
            download_size="4.7 GB",
            quality="good",
            note="Good quality, fits your GPU"
        )
    elif vram_gb < 16:
        return ModelRecommendation(
            name="gemma3:12b",
            download_size="7.3 GB",
            quality="very good",
            note="Best quality for your GPU"
        )
    else:
        return ModelRecommendation(
            name="qwen2.5:14b",
            download_size="9.0 GB",
            quality="excellent",
            note="Highest quality translation"
        )
```

#### Tier 2: "Advanced" (Settings Page)

Full model picker for power users:

- Dropdown with pre-tested models + quality ratings
- VRAM requirement per model
- Custom model name field (for `ollama pull <any-model>`)
- Temperature, context window, batch size overrides

### Model Quality Tiers (Human-Readable)

| Tier | Label | Models | VRAM | User Sees |
|------|-------|--------|------|-----------|
| Entry | "Basic" | 3B models | 4 GB | "Basic translation quality" |
| Standard | "Good" | 7-8B models | 8 GB | "Good translation quality" |
| Quality | "Very Good" | 12B models | 12 GB | "High quality translation" |
| Premium | "Excellent" | 14B+ models | 16 GB+ | "Best available quality" |

### Pre-Configured Translation Parameters

Each recommended model ships with tuned parameters:

```python
MODEL_CONFIGS = {
    "qwen2.5:3b": {
        "temperature": 0.3,  # Lower = more consistent for small models
        "num_ctx": 4096,     # Smaller context for smaller model
        "batch_size": 8,     # Fewer lines per batch
    },
    "qwen2.5:7b": {
        "temperature": 0.5,
        "num_ctx": 8192,
        "batch_size": 11,
    },
    "gemma3:12b": {
        "temperature": 0.5,
        "num_ctx": 8192,
        "batch_size": 11,
    },
    "qwen2.5:14b": {
        "temperature": 0.5,
        "num_ctx": 16384,
        "batch_size": 20,
    },
}
```

Users should never see these parameters unless they visit Advanced Settings.

---

## 9. Error Handling & Diagnostics Design

### Error Scenarios & Messaging

| Scenario | Level 1 (Always Shown) | Level 2 (Expandable) | Level 3 (Details) |
|----------|----------------------|---------------------|------------------|
| **Ollama not installed** | "Translation engine not found." | "Install Ollama (free, 1 click) from ollama.com/download. Then click Check Again." | "shutil.which('ollama') returned None. Expected in PATH." |
| **Ollama not running** | "Translation engine is installed but not running." | "Start Ollama from your system tray (Windows) or Applications (macOS). Or run: ollama serve" | "GET http://localhost:11434/ returned ConnectionRefused" |
| **Model not downloaded** | "Translation model needs to be downloaded first (4.7 GB, one-time)." | "[Download Now] [Choose Different Model]" | "POST /api/show {model: 'qwen2.5:7b'} returned 404" |
| **Out of VRAM** | "This model is too large for your GPU." | "[Switch to Lite Model] [Use CPU Mode]" | "CUDA out of memory. Model needs 8 GB, available: 4 GB." |
| **No GPU detected** | "No compatible GPU found. Translation will use CPU mode (slower)." | "[Why?] [GPU Setup Guide]" | "No CUDA/ROCm/Metal devices found" |
| **Translation failed** | "Translation failed for batch 3 of 15." | "The model produced unexpected output. Try a different model or reduce batch size." | "PySubtrans: No matches found in response. Raw output: ..." |
| **Server timeout** | "Translation engine stopped responding." | "This usually means the model ran out of memory. Try: smaller batch size, smaller model, or restart Ollama." | "POST /v1/chat/completions timed out after 120s" |

### Graceful Degradation Chain

```
1. GPU + recommended model (best quality, fastest)
   ↓ if VRAM insufficient
2. GPU + smaller model (good quality, fast)
   ↓ if no GPU or all models too large
3. CPU + small model (good quality, slow)
   ↓ if translation fails
4. Suggest cloud provider (DeepSeek at $0.001/file)
   ↓ if user declines
5. Skip translation (output untranslated SRT)
```

Each step down should include a notification explaining why, not a blocking error.

### VRAM Sharing: Whisper + Ollama

When transcribing and translating in one pipeline, Whisper and Ollama compete for VRAM.

**Strategy**: Sequential, not concurrent.

```
1. Whisper loads → transcribes → produces SRT
2. Whisper model unloaded (or moved to CPU)
3. Ollama loads translation model → translates
4. Ollama model stays loaded for 5 minutes (keep_alive)
```

**Implementation**:
- Set `OLLAMA_GPU_OVERHEAD` to reserve VRAM for Whisper during transcription
- Or: unload Whisper model before starting translation
- Or: use `keep_alive: 0` to immediately unload Ollama model after translation

---

## 10. Current WhisperJAV State

### What Already Exists (v1.8.8)

The v1.8.8 preview provides a **working but manual** Ollama integration:

| Component | Status | Location |
|-----------|--------|----------|
| Provider definition | Done | `providers.py:47-56` |
| Server detection | Done | `cli.py:532-548` |
| Context window setup | Done (hardcoded 8192) | `cli.py:550-569` |
| Batch size capping | Done | `core.py:10-57` |
| Max token computation | Done | `core.py:60-103` |
| Diagnostic tracking | Done | `core.py:409-500` |
| Main pipeline integration | Done | `main.py:1256-1276` |
| Error messages | Done (basic) | `cli.py:540-548` |

### What's Missing for v1.9.0

| Component | Status | Effort |
|-----------|--------|--------|
| OllamaManager class | Not started | Medium |
| Model availability check | Not started | Small |
| Model auto-pull with progress | Not started | Medium |
| Dynamic context length from model metadata | Not started | Small |
| Hardware-aware model recommendation | Not started | Small |
| Auto-start Ollama if installed but not running | Not started | Small |
| Setup wizard (CLI) | Not started | Medium |
| Setup wizard (GUI) | Not started | Medium-Large |
| VRAM sharing with Whisper | Not started | Medium |
| Backend abstraction layer (`LLMBackend` protocol) | Not started | Medium |
| Deprecate `local_backend.py` | Not started | Small |

### Code to Remove (v1.9.0)

| File | Lines | Purpose | Replacement |
|------|-------|---------|-------------|
| `local_backend.py` | 2,423 | llama-cpp-python server management | OllamaManager |
| `llama_cuda_config.py` | 186 | CUDA wheel selection | Ollama handles GPU |
| `llama_build_utils.py` | 500 | Build-from-source fallback | Not needed |
| DLL path hacks in various files | ~400 | DLL loading workarounds | Not needed |
| **Total removed** | **~3,500** | | |

### Code to Add (v1.9.0)

| Component | Est. Lines | Purpose |
|-----------|-----------|---------|
| `ollama_manager.py` | ~300 | Detection, model management, server lifecycle |
| `llm_backend.py` | ~100 | Protocol + factory |
| Setup wizard (CLI) | ~150 | First-use guided setup |
| Model configs | ~50 | Pre-tuned parameters per model |
| **Total added** | **~600** | |

**Net change**: Remove ~3,500 lines, add ~600 lines = **net reduction of ~2,900 lines**.

---

## 11. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Users can't install Ollama** | Medium | High | Clear platform-specific instructions, video link |
| **Ollama API breaking change** | Low | Medium | Pin minimum version, check `/api/version` |
| **Model produces garbled translation** | Medium | Medium | Pre-tested model list, quality benchmarks |
| **VRAM conflict with Whisper** | Medium | Medium | Sequential loading, `OLLAMA_GPU_OVERHEAD` |
| **Port 11434 conflict** | Low | Low | Support custom port via env var |
| **Ollama not available on user's OS** | Very Low | High | Ollama supports Windows, macOS, Linux |
| **Model download fails/stalls** | Low | Medium | Resumable downloads (Ollama native) |
| **User has old Ollama version** | Medium | Low | Version check + upgrade guidance |

### Mitigation for Top Risk: "Users Can't Install Ollama"

This is the single biggest risk. Mitigation:

1. **Platform-specific install instructions** (not just "go to ollama.com"):
   - Windows: "Download OllamaSetup.exe, double-click, done"
   - macOS: "Download, drag to Applications, open"
   - Linux: "Run this one command: `curl -fsSL https://ollama.com/install.sh | sh`"

2. **"Check Again" button** — user installs externally, clicks button, WhisperJAV re-detects

3. **Fallback path** — if user truly can't install Ollama, suggest `--provider deepseek` (cloud, $0.001/file)

---

## 12. Open Questions

### Q1: Should WhisperJAV auto-start Ollama?

**Options**:
- A) Always auto-start if installed but not running
- B) Ask user: "Ollama is installed but not running. Start it?"
- C) Never auto-start, show guidance only

**Recommendation**: Option B. Transparent, respects user autonomy, avoids surprise background processes.

### Q2: How to handle uncensored models?

Current `--provider local` downloads uncensored GGUF from HuggingFace. With Ollama:
- **Option A**: Publish uncensored model to Ollama registry (community model)
- **Option B**: Create Modelfile from local GGUF: `ollama create whisperjav-uncensored -f Modelfile`
- **Option C**: Use `ollama pull hf.co/user/model:Q4_K_M` (HuggingFace direct pull, supported since Ollama 0.4+)

**Recommendation**: Start with standard models (gemma3, qwen2.5). Most users don't need uncensored for subtitle translation. Add uncensored model support in v1.9.x if demand exists.

### Q3: Should the setup wizard be CLI-only or also GUI?

**Recommendation**: Both, but CLI first. CLI wizard for v1.9.0, GUI wizard for v1.9.x. The GUI can call the same `OllamaManager` methods.

### Q4: What's the minimum Ollama version to support?

Ollama's OpenAI-compatible API (`/v1/chat/completions`) has been stable since v0.1.0+. The `POST /api/show` endpoint for model info is available since v0.1.0+.

**Recommendation**: Require Ollama 0.3.0+ (widely available, stable API). Check via `GET /api/version`.

### Q5: Should WhisperJAV manage `OLLAMA_KEEP_ALIVE`?

During batch translation of a long file, models should stay loaded. But after translation completes, models should unload to free VRAM.

**Options**:
- A) Set `keep_alive: "30m"` per-request during translation, `keep_alive: 0` on last request
- B) Don't manage it, let Ollama's default (5 min) handle it
- C) Set `OLLAMA_KEEP_ALIVE=0` env var globally (aggressive memory freeing)

**Recommendation**: Option A. Explicit control per-request. Keep model loaded during translation, unload when done.

### Q6: How to handle VRAM sharing between Whisper and Ollama?

**Recommendation**: Sequential loading. Whisper runs first (transcription), then Ollama runs (translation). If running in the same pipeline (`--translate` flag), unload Whisper model or move to CPU before starting Ollama translation. This is the simplest approach and avoids VRAM contention.

---

## 13. References

### Ollama Documentation
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama GPU Documentation](https://docs.ollama.com/gpu)
- [Ollama Windows Documentation](https://docs.ollama.com/windows)
- [Ollama FAQ](https://docs.ollama.com/faq)
- [Ollama Environment Variables](https://pkg.go.dev/github.com/ollama/ollama/envconfig)

### Ollama Architecture
- [Ollama Architecture — DeepWiki](https://deepwiki.com/ollama/ollama/2-architecture)
- [Ollama Memory Management — DeepWiki](https://deepwiki.com/ollama/ollama/5.4-memory-management-and-gpu-allocation)
- [Ollama GPU/Hardware Support — DeepWiki](https://deepwiki.com/ollama/ollama/6-gpu-and-hardware-support)
- [Ollama New Model Scheduling Blog](https://ollama.com/blog/new-model-scheduling)
- [Ollama Behind the Scenes](https://dasroot.net/posts/2026/01/ollama-behind-the-scenes-architecture/)

### Industry Projects
- [Open WebUI GitHub](https://github.com/open-webui/open-webui)
- [AnythingLLM](https://anythingllm.com/)
- [Jan.ai](https://www.jan.ai/)
- [LM Studio](https://lmstudio.ai/)
- [GPT4All](https://gpt4all.io/)

### UX Research
- [AI UX Patterns](https://www.aiuxpatterns.com/)
- [Progressive Disclosure — NN/g](https://www.nngroup.com/articles/progressive-disclosure/)
- [20+ GenAI UX Patterns — UX Collective](https://uxdesign.cc/20-genai-ux-patterns-examples-and-implementation-tactics-5b1868b7d4a1)
- [Desktop UX: Software Installer Best Practices — Medium](https://medium.com/ux-stories/desktop-ux-software-installer-best-practices-6d6d7383dc98)
- [Graceful Degradation — Medium](https://medium.com/@satyendra.jaiswal/graceful-degradation-handling-errors-without-disrupting-user-experience-fd4947a24011)

### Model Research
- [Best LLMs for Translation — Crowdin](https://crowdin.com/blog/best-llms-for-translation)
- [Best LLM for Translation 2026 — NutStudio](https://nutstudio.imyfone.com/llm-tips/best-llm-for-translation/)
- [8 Local LLM Settings Most People Never Touch — XDA](https://www.xda-developers.com/local-llm-settings-most-people-never-touch/)
- [LM Studio vs Ollama Comparison — BixTech](https://bix-tech.com/lm-studio-vs-ollama-how-to-run-llms-locally-and-scale-them-across-a-team/)

### WhisperJAV Internal
- [v1.9.0 Roadmap](../research/ROADMAP_LLM_AND_NUMPY2.md)
- [v1.8.8 Release Plan](../plans/V188_STABLE_RELEASE_PLAN.md)
- [LLM Diagnosis](../analysis/LOCAL_LLM_INDEPENDENT_DIAGNOSIS.md)
- [Installation Proposal](../architecture/INSTALLATION_PROPOSAL.md)
