# LLM Backend Architecture Best Practices

**Date**: 2026-03-11
**Purpose**: Research document analyzing how state-of-the-art open-source projects architect their LLM backends, with recommendations for WhisperJAV's local translation feature.

**WhisperJAV context**: Non-technical GUI users, Windows+NVIDIA primary, subtitle translation via OpenAI-compatible API, current llama-cpp-python approach is fragile due to compilation/DLL issues.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Industry Standard: OpenAI-Compatible API](#the-industry-standard-openai-compatible-api)
3. [Front-End / Back-End Separation Patterns](#front-end--back-end-separation-patterns)
4. [Process Isolation and Lifecycle Management](#process-isolation-and-lifecycle-management)
5. [Multi-Backend Abstraction Layers](#multi-backend-abstraction-layers)
6. [GPU Detection and Management](#gpu-detection-and-management)
7. [User Experience Patterns for Non-Technical Users](#user-experience-patterns-for-non-technical-users)
8. [Project-by-Project Analysis](#project-by-project-analysis)
9. [Anti-Patterns and What Doesn't Work](#anti-patterns-and-what-doesnt-work)
10. [Recommendations for WhisperJAV](#recommendations-for-whisperjav)

---

## Executive Summary

After analyzing 10+ major open-source LLM projects, clear patterns emerge:

1. **The OpenAI Chat Completions API is the universal contract.** Every successful project either implements it or consumes it. This is the only API format supported by dozens of both servers and clients.

2. **Process isolation is non-negotiable.** Every production-grade project runs the LLM engine as a separate process, communicating over HTTP. No successful project embeds the inference engine in-process for user-facing applications.

3. **Prebuilt binaries beat compilation.** Projects that require users to compile C++ code with GPU support (like llama-cpp-python) consistently generate the most support issues. Projects that ship prebuilt binaries with automatic GPU detection (Ollama, LM Studio) have the best user experience.

4. **The "delegate to external server" pattern dominates.** Rather than embedding LLM inference, successful applications detect whether an external LLM server (Ollama, LM Studio, etc.) is available and connect to it.

5. **Fallback chains are essential.** The best UX comes from: try local GPU → try local CPU → suggest cloud API → provide clear error with next steps.

---

## The Industry Standard: OpenAI-Compatible API

The OpenAI Chat Completions API has become the de facto standard for LLM interaction. It is the **only** API format that meets all three criteria simultaneously:

- Dozens of LLM **servers** implement it (Ollama, vLLM, LM Studio, LocalAI, llama.cpp server, Jan/Cortex, text-generation-webui)
- Dozens of LLM **client applications** consume it (Aider, Continue.dev, AnythingLLM, Open WebUI, LangChain, LlamaIndex)
- It works identically for **both local and cloud** models

### The Pattern

```python
from openai import OpenAI

# Cloud API
client = OpenAI(api_key="sk-...")

# Local Ollama — same code, different base_url
client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")

# Local LM Studio — same code
client = OpenAI(base_url="http://localhost:1234/v1/", api_key="lm-studio")

# Local llama.cpp server — same code
client = OpenAI(base_url="http://localhost:8080/v1/", api_key="none")
```

The key insight: **if your application speaks OpenAI Chat Completions, it works with every backend.** This is exactly what WhisperJAV already does via PySubtrans — the architecture is correct, the problem is the server-side (llama-cpp-python compilation).

### Relevant Endpoints

| Endpoint | Purpose | Universally Supported? |
|---|---|---|
| `POST /v1/chat/completions` | Chat/translation | Yes — every server |
| `GET /v1/models` | List available models | Yes — used for health/readiness |
| `GET /health` or `GET /` | Server liveness | Most servers |
| `POST /v1/embeddings` | Embeddings | Most servers |

**Sources:**
- [OpenAI-compatible API — LLM Inference Handbook (BentoML)](https://bentoml.com/llm/llm-inference-basics/openai-compatible-api)
- [Ollama OpenAI Compatibility](https://ollama.com/blog/openai-compatibility)
- [Local LLM Hosting Complete 2025 Guide](https://medium.com/@rosgluk/local-llm-hosting-complete-2025-guide-ollama-vllm-localai-jan-lm-studio-more-f98136ce7e4a)

---

## Front-End / Back-End Separation Patterns

### Pattern 1: "Delegate to External Server" (Aider, Continue.dev, Open WebUI)

The application assumes an LLM server already exists somewhere and connects to it via HTTP.

```
[Application] --HTTP/OpenAI API--> [External LLM Server]
                                    (Ollama, LM Studio, vLLM, cloud API)
```

**How it works:**
- Application configuration specifies a `base_url` and `model` name
- Application uses the `openai` Python library (or equivalent) to make requests
- Zero LLM engine code in the application itself
- User is responsible for running the server (or it's a cloud API)

**Who uses this:**
- **Aider**: Uses LiteLLM to abstract 100+ providers behind a unified `completion()` call. User configures model name and API key; Aider never touches inference.
- **Continue.dev**: Implements `ILLM` interface with `BaseLLM` base class. 40+ providers added by extending BaseLLM with provider-specific streaming. All message compilation, capability detection, and logging handled automatically.
- **Open WebUI**: Supports "Ollama and OpenAI-compatible Protocols" — connects to any server that speaks the standard API. Uses a "Pipelines" plugin system for extensibility.

**Pros:** Zero maintenance burden for inference engine, maximum flexibility, users can pick their preferred backend.
**Cons:** Requires user to set up an LLM server separately.

### Pattern 2: "Managed Subprocess" (AnythingLLM, Jan.ai)

The application ships a bundled LLM engine and manages it as a subprocess.

```
[Application Process]
    |
    ├── starts/stops --> [LLM Engine Subprocess] (llama.cpp, Cortex)
    |                         ↕ HTTP API
    └── sends requests -------┘
```

**How it works:**
- Application bundles a prebuilt binary (not Python bindings)
- On startup, spawns the engine as a subprocess with appropriate flags
- Communicates over localhost HTTP (OpenAI-compatible)
- Manages lifecycle: start, health check, graceful shutdown, crash recovery

**Who uses this:**
- **AnythingLLM Desktop**: Ships a built-in LLM engine. One-click install, no configuration needed. Downloads models on demand. Uses factory pattern (`getLLMProvider()`) for instantiation.
- **Jan.ai**: Powered by Cortex (C++ CLI), which runs llama.cpp under the hood. Supports multiple engine backends (llama.cpp, ONNX, TensorRT-LLM) loaded at runtime.
- **LM Studio**: Bundles llama.cpp as its engine with "llmster" daemon for headless operation. Manages concurrent requests via continuous batching.

**Pros:** "Just works" for users, no separate server setup.
**Cons:** Must bundle and maintain the binary, handle platform-specific builds.

### Pattern 3: "Hub and Spoke" (text-generation-webui)

The application IS the LLM server, with multiple backend loaders.

```
[text-generation-webui]
    ├── Transformers loader
    ├── ExLlamaV3 loader
    ├── llama.cpp loader
    └── TensorRT-LLM loader

    All expose: Gradio UI + OpenAI-compatible API
```

**How it works:**
- `shared.py` holds global state (model, tokenizer, args)
- `models.py` provides `load_model()` facade that dispatches to the correct backend
- Each loader implements model loading for its specific format
- A single API extension provides OpenAI-compatible endpoints

**Who uses this:**
- **text-generation-webui (oobabooga)**: Supports llama.cpp, Transformers, ExLlamaV3, TensorRT-LLM through a unified `load_model()` interface. Users select loader in the UI.

**Pros:** Maximum backend flexibility, power-user friendly.
**Cons:** Complex, maintenance-heavy, not suitable for embedding in other apps.

### Verdict for WhisperJAV

**Pattern 1 (delegate) + Pattern 2 (managed subprocess) hybrid** is the right approach:

1. Primary: Detect if Ollama/LM Studio is already running → use it (Pattern 1)
2. Fallback: If no server found, offer to use cloud API (Pattern 1)
3. Future: Optionally bundle a lightweight server binary (Pattern 2)

**Sources:**
- [Aider — Multi-Provider LLM Integration (DeepWiki)](https://deepwiki.com/Aider-AI/aider/6.3-multi-provider-llm-integration)
- [Continue.dev — LLM Abstraction Layer (DeepWiki)](https://deepwiki.com/continuedev/continue/4.1-extension-architecture)
- [AnythingLLM — LLM Provider Integration (DeepWiki)](https://deepwiki.com/Mintplex-Labs/anything-llm/5-vector-database-system)
- [oobabooga/text-generation-webui (DeepWiki)](https://deepwiki.com/oobabooga/text-generation-webui)
- [Jan.ai Desktop (GitHub)](https://github.com/janhq/jan)

---

## Process Isolation and Lifecycle Management

### Why Process Isolation Matters

Every production LLM project runs inference in a **separate process** from the application logic. Reasons:

1. **Crash isolation**: LLM inference engines can segfault (CUDA errors, memory corruption). A crash in a subprocess doesn't kill the main application.
2. **Memory management**: LLMs consume multi-GB of VRAM/RAM. A separate process allows clean release on shutdown.
3. **Language boundary**: Best inference engines are C/C++/Go. Separate process avoids Python GIL issues and complex FFI bindings.
4. **Restartability**: If the LLM server hangs, the application can kill and restart it without losing user state.

### Ollama's Lifecycle Management (Reference Implementation)

Ollama is the gold standard for LLM server lifecycle management:

```
Server Start
    → GPU Discovery (subprocess probe per backend: CUDA, ROCm, Metal, Vulkan)
    → Backend Selection (priority: CUDA/ROCm > Vulkan > CPU)
    → Library Loading (bundled GPU libs prepended to system PATH)
    → Model Loading (on first request, not startup)
    → Session Timer (5 min default idle timeout)
    → Expiry → Runner Close → Memory Free
```

Key design decisions:
- **Subprocess probes for GPU detection**: Ollama spawns a separate process per backend type (CUDA, ROCm, etc.) to test GPU availability. If the probe crashes (unsupported GPU), the main server is unaffected.
- **Deduplication**: Same physical GPU may be detected by multiple backends (CUDA and Vulkan). Ollama deduplicates by PCI ID, preferring CUDA/ROCm over Vulkan.
- **Bundled libraries**: Ollama prepends its own GPU libraries to the path, preventing version conflicts with system-installed CUDA.
- **Lazy model loading**: Models load on first request, not server startup. Reduces startup time.
- **Idle unloading**: After `sessionDuration` (default 5 min) with no requests, the model is unloaded to free VRAM.

### Health Check Patterns

Best practice from vLLM and LiteLLM:

| Probe | Endpoint | Purpose | When |
|---|---|---|---|
| **Liveness** | `GET /health` | "Is the process alive?" | Continuous |
| **Readiness** | `GET /v1/models` | "Is a model loaded and ready?" | Before sending requests |
| **Warmup** | `POST /v1/chat/completions` (1 token) | "Can it actually generate?" | After initial load |

LiteLLM recommends using `max_tokens=1` for health checks to minimize cost and latency.

### Lifecycle Management Code Pattern

```python
class LLMServerManager:
    """Manages an external LLM server process."""

    def start(self, model: str, gpu_layers: int = -1) -> str:
        """Start server, return base_url."""
        self._process = subprocess.Popen([...])
        self._wait_for_ready(timeout=120)  # Poll /v1/models
        return f"http://localhost:{self._port}/v1/"

    def health_check(self) -> bool:
        """Check if server is alive and model is loaded."""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            return resp.status_code == 200 and len(resp.json()["data"]) > 0
        except:
            return False

    def stop(self):
        """Graceful shutdown with force-kill fallback."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
```

**Sources:**
- [Ollama Architecture (DeepWiki)](https://deepwiki.com/ollama/ollama/2-architecture)
- [Ollama GPU Discovery and Backend Loading (DeepWiki)](https://deepwiki.com/ollama/ollama/6.1-gpu-discovery-and-backend-loading)
- [vLLM Model-Aware Readiness Probes](https://llm-d.ai/docs/usage/readiness-probes)
- [LiteLLM Health Checks](https://docs.litellm.ai/docs/proxy/health)

---

## Multi-Backend Abstraction Layers

### The Spectrum of Abstraction

Projects use different levels of abstraction depending on their needs:

#### Level 1: Configuration-Based Switching (AnythingLLM)

The simplest pattern. A factory function reads config and returns the right client:

```python
def get_llm_provider(config):
    """Factory pattern — config determines provider."""
    if config.provider == "ollama":
        return OllamaProvider(config.base_url, config.model)
    elif config.provider == "openai":
        return OpenAIProvider(config.api_key, config.model)
    elif config.provider == "local":
        return LocalProvider(config.model_path)
```

AnythingLLM uses this exact pattern with `getLLMProvider()`, supporting 30+ providers. Each provider implements a common interface for chat completion, streaming, and embedding.

**Key feature**: Workspace-level overrides — each workspace can use a different LLM provider/model, while a system-wide default exists.

#### Level 2: Unified SDK (LiteLLM, used by Aider)

A library that translates between your code and 100+ provider APIs:

```python
import litellm

# All these use the same function signature
response = litellm.completion(model="ollama/llama3.1", messages=[...])
response = litellm.completion(model="gpt-4o", messages=[...])
response = litellm.completion(model="claude-3-opus", messages=[...])
response = litellm.completion(model="deepseek/deepseek-chat", messages=[...])
```

LiteLLM handles authentication, streaming format differences, error normalization, cost tracking, and retry logic. Available as both:
- **Python SDK**: In-process library (simplest integration)
- **Proxy Server**: Standalone gateway with admin UI, budget tracking, rate limiting

#### Level 3: Interface + Base Class (Continue.dev)

A more formal OOP abstraction:

```typescript
interface ILLM {
    complete(prompt: string, options: CompletionOptions): AsyncGenerator<string>;
    chat(messages: Message[], options: ChatOptions): AsyncGenerator<string>;
    listModels(): Promise<string[]>;
    supportsFim(): boolean;  // Capability detection
}

class BaseLLM implements ILLM {
    // Common: token counting, message compilation, logging, error handling
    // Abstract: _streamChat(), _streamComplete() — provider-specific
}

class OllamaLLM extends BaseLLM { /* Ollama-specific streaming */ }
class OpenAILLM extends BaseLLM { /* OpenAI-specific streaming */ }
```

Continue.dev supports 40+ providers through this pattern. Adding a new provider requires only implementing the streaming methods — everything else (message compilation, capability detection, logging) is inherited.

### Recommendation for WhisperJAV

**Level 1 (factory pattern) is sufficient.** WhisperJAV's translation use case is simple: send chat completions, receive translated text. The existing PySubtrans integration already uses the OpenAI SDK, so the abstraction layer is:

```python
def get_translation_client(provider: str, config: dict) -> OpenAI:
    """Return an OpenAI-compatible client for the configured provider."""
    if provider == "ollama":
        return OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
    elif provider == "lm-studio":
        return OpenAI(base_url="http://localhost:1234/v1/", api_key="lm-studio")
    elif provider == "local-server":
        return OpenAI(base_url=config["base_url"], api_key="none")
    elif provider == "deepseek":
        return OpenAI(base_url="https://api.deepseek.com/v1", api_key=config["api_key"])
    # ... etc
```

No need for LiteLLM's complexity — WhisperJAV doesn't need cost tracking, rate limiting, or 100-provider support.

**Sources:**
- [AnythingLLM LLM Configuration Overview](https://docs.useanything.com/setup/llm-configuration/overview)
- [LiteLLM Getting Started](https://docs.litellm.ai/docs/)
- [Continue.dev Model Providers Overview](https://docs.continue.dev/customize/model-providers/overview)
- [LiteLLM — A Unified LLM API Gateway (Medium)](https://medium.com/@mrutyunjaya.mohapatra/litellm-a-unified-llm-api-gateway-for-enterprise-ai-de23e29e9e68)

---

## GPU Detection and Management

### How Ollama Does It (Best-in-Class)

Ollama's GPU detection is the industry reference implementation:

1. **Discovery phase**: For each supported backend (CUDA, ROCm, Metal, Vulkan), Ollama spawns a **bootstrap subprocess** that:
   - Sets `LD_LIBRARY_PATH`/`PATH` to bundled GPU libraries
   - Initializes the backend
   - Queries available devices (memory, compute capability, PCI ID)
   - Returns device info via HTTP (`GET /info` → `DeviceInfo[]`)
   - Has a timeout: 30s on Linux/macOS, 90s on Windows (Defender DLL scanning)

2. **Deep validation**: For CUDA and ROCm, sets `GGML_CUDA_INIT=1` to force immediate GPU initialization. If the subprocess crashes (unsupported architecture), it's caught cleanly.

3. **Deduplication**: Same physical GPU may appear in multiple backends. Ollama deduplicates by PCI ID, with priority: CUDA/ROCm > Vulkan > CPU.

4. **Library resolution**: Ollama prepends its bundled GPU libs to `PATH`/`LD_LIBRARY_PATH` before system libs, preventing version conflicts.

5. **Fallback**: If no GPU libraries found → CPU inference with `libggml-cpu`.

### The PyTorch Pattern (Used by Most Python Projects)

Most Python ML projects rely on PyTorch's device detection:

```python
import torch

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

**MPS (Apple Silicon) caveats:**
- Not all operators are implemented; PyTorch silently falls back to CPU for missing ops
- Can be slower than CPU for some workloads due to these fallbacks
- Profiling recommended: "always establish a CPU baseline before optimizing for MPS"

**ROCm (AMD) caveats:**
- `rocm_agent_enumerator` utility can crash if GPU permissions are wrong
- PyTorch ROCm wheels only support specific AMD GPU families
- Building ROCm from source on Windows is still experimental as of 2026

### The Prebuilt Binary Approach (Ollama, LM Studio)

The most reliable GPU management comes from projects that:
1. Ship prebuilt binaries for each GPU backend (CUDA 12, CUDA 13, ROCm, Vulkan, CPU)
2. Auto-detect hardware at runtime
3. Dynamically load the correct backend library

This avoids the compilation problem entirely — users never need to install CUDA Toolkit, Visual Studio, or compile anything.

### WhisperJAV Implication

The current approach (llama-cpp-python with CUDA compilation) is an anti-pattern. The fix is not better compilation — it's eliminating compilation entirely by delegating to a project that ships prebuilt GPU-accelerated binaries (Ollama).

**Sources:**
- [Ollama GPU Discovery (DeepWiki)](https://deepwiki.com/ollama/ollama/6.1-gpu-discovery-and-backend-loading)
- [Ollama Hardware Support](https://docs.ollama.com/gpu)
- [PyTorch MPS Backend Issues](https://github.com/pytorch/pytorch/issues/109457)
- [CodeProject.AI — GPU Support Configuration (DeepWiki)](https://deepwiki.com/codeproject/CodeProject.AI-Server/3.4-gpu-support-configuration)

---

## User Experience Patterns for Non-Technical Users

### Tier 1: "It Just Works" (Ollama Desktop, LM Studio, AnythingLLM)

These projects achieve the lowest friction by:

1. **Single installer**: One download, one click, done. No command line.
2. **Bundled GPU libraries**: No "install CUDA Toolkit" step.
3. **Auto GPU detection**: Application discovers hardware automatically.
4. **Model browser/downloader**: Users pick models from a gallery, not HuggingFace URLs.
5. **Health feedback**: Clear indication of what's running, what's loaded, GPU vs CPU status.

**Ollama Desktop** (v0.10.0, July 2025): First native GUI. Model dropdown, prompt box, streaming responses. Finds existing models from CLI. Drag-and-drop file upload. Alpha-grade but functional.

**LM Studio**: Polished desktop app. Model search and download built-in. One-click server start. Shows GPU utilization in real-time. "llmster" daemon for headless use.

**AnythingLLM Desktop**: Built-in LLM engine with zero configuration. Downloads popular models (Llama 3, Phi-3) on demand. CPU+GPU support. One-click install.

### Tier 2: "Guided Setup" (Jan.ai)

- Desktop app with setup wizard
- Prompts user to download a model on first launch
- Auto-detects GPU but shows the detection result
- Offers "recommended" model based on hardware

### Tier 3: "Technical but Documented" (text-generation-webui)

- Provides one-click installers (batch scripts) per platform
- User must select model loader
- Requires understanding of quantization formats
- Power-user focused

### Key UX Lessons for WhisperJAV

1. **Never ask users to compile anything.** The #1 source of support issues.
2. **Detect, don't ask.** Auto-detect GPU, VRAM, available servers. Present findings, not questions.
3. **Provide a "recommended" path.** "Ollama detected with 8GB VRAM → using Llama 3.1 8B" is better than a configuration form.
4. **Fail gracefully with actionable messages.** Not "ConnectionError" but "Local LLM server not found. Install Ollama (ollama.com) or use cloud translation (--provider deepseek)."
5. **Show progress.** Model downloads are multi-GB. Show download progress, estimated time, and disk space required.

### The "Detect External Server" Pattern

For applications that don't want to bundle an LLM engine, the best UX is:

```
1. Check localhost:11434 (Ollama) → found? Use it.
2. Check localhost:1234 (LM Studio) → found? Use it.
3. Check user-configured URL → found? Use it.
4. Not found → Show friendly message:
   "No local LLM server detected. Options:
    a) Install Ollama (free, recommended): ollama.com
    b) Use cloud translation: --provider deepseek ($0.001/subtitle file)"
```

This is exactly how AnythingLLM handles Ollama detection — auto-detect if possible, prompt for manual URL if not.

**Sources:**
- [Ollama v0.10.0 Native GUI (DatabaseMart)](https://www.databasemart.com/blog/intro-to-ollama-app-v0100)
- [LM Studio 0.4.0 (Blog)](https://lmstudio.ai/blog/0.4.0)
- [AnythingLLM Desktop](https://anythingllm.com/desktop)
- [AnythingLLM Built-in LLM (Docs)](https://docs.anythingllm.com/setup/llm-configuration/local/built-in)

---

## Project-by-Project Analysis

### Ollama

| Aspect | Detail |
|---|---|
| **What it is** | Local LLM server with model management |
| **Backend** | llama.cpp (Go wrapper around C/C++ engine) |
| **GPU support** | CUDA, ROCm, Metal, Vulkan — all auto-detected, prebuilt |
| **API** | OpenAI-compatible `/v1/chat/completions`, native `/api/chat` |
| **Process model** | Standalone server process, client connects via HTTP |
| **Model management** | Built-in registry, `ollama pull model_name` |
| **Binary size** | ~3-4GB with GPU libraries |
| **Key strength** | Zero-compilation GPU support, excellent auto-detection |
| **Key weakness** | Large binary size, no multi-GPU tensor parallelism |
| **Stars** | ~162k |
| **Release cadence** | Every 2-3 days |

**Architecture insight**: Ollama's Go binary bundles precompiled llama.cpp backends for every supported GPU. At startup, it probes each backend via subprocess and selects the best one. This eliminates compilation entirely — the user just downloads and runs.

**v0.17.5 (March 2026)**: Cloud model offloading, web search, multimodal, streaming tool calls, thinking models.

### Open WebUI

| Aspect | Detail |
|---|---|
| **What it is** | Web-based ChatGPT-like interface for local/cloud LLMs |
| **Backend** | None — connects to external servers (Ollama, OpenAI-compatible) |
| **Abstraction** | "Supports Ollama and OpenAI-compatible Protocols" |
| **Key pattern** | Pure frontend — delegates all inference to external servers |
| **Extensibility** | "Pipelines" plugin system, 9 vector DB backends via VectorDBBase |
| **Stars** | ~80k+ |

**Architecture insight**: Open WebUI is protocol-first — it doesn't care what's behind the API, only that it speaks OpenAI-compatible. This makes it work with Ollama, LM Studio, vLLM, cloud APIs, and anything else that implements the standard.

### text-generation-webui (oobabooga)

| Aspect | Detail |
|---|---|
| **What it is** | Gradio web UI for running LLMs locally |
| **Backends** | llama.cpp, Transformers, ExLlamaV3, TensorRT-LLM |
| **Architecture** | Hub-and-spoke: `shared.py` global state, `models.py` loader facade |
| **API** | OpenAI-compatible extension, ~85ms overhead |
| **Process model** | Single process, multiple backend loaders |
| **Key strength** | Maximum backend flexibility, power-user features |
| **Key weakness** | Complex, requires technical knowledge |
| **Stars** | ~45k+ |

**Architecture insight**: The `load_model()` function in `models.py` is a facade that dispatches to the correct backend loader based on model type. Each loader handles its own GPU configuration. Recent versions dropped ExLlamaV2 in favor of ExLlamaV3 for better quantization.

### LM Studio

| Aspect | Detail |
|---|---|
| **What it is** | Desktop app for running local LLMs |
| **Backend** | llama.cpp (embedded, custom "llm-engine" wrapper) |
| **API** | OpenAI-compatible REST API, streaming, embeddings, function calling |
| **Process model** | App + "llmster" daemon (headless mode) |
| **Key strength** | Polished UX, GPU visualization, concurrent request handling |
| **Key weakness** | Closed-source, not embeddable |
| **Notable** | Continuous batching, unified KV cache, concurrent inference |

**Architecture insight**: LM Studio 0.4.0 introduced concurrent inference via llama.cpp's continuous batching. The "llmster" daemon allows headless server operation on machines without displays — useful for home servers.

### Jan.ai / Cortex

| Aspect | Detail |
|---|---|
| **What it is** | Desktop app + Cortex engine (CLI alternative to Ollama) |
| **Backend** | Cortex (C++ CLI): llama.cpp, ONNX, TensorRT-LLM |
| **Multi-engine** | Runtime engine selection — swap backends without reinstall |
| **API** | OpenAI-compatible |
| **Key strength** | Multi-engine architecture, cross-platform |
| **Notable** | Cortex can be used independently of Jan |

### LocalAI

| Aspect | Detail |
|---|---|
| **What it is** | Self-hosted OpenAI-compatible API server |
| **Backend** | Multiple: llama.cpp, vLLM, Transformers, Diffusers |
| **Deployment** | Docker-first (not a desktop app) |
| **GPU support** | CUDA 12/13, ROCm, Intel SYCL, Vulkan, Metal, CPU |
| **Key strength** | Broadest backend and GPU support |
| **Key weakness** | Docker requirement, not suitable for desktop users |
| **Stars** | ~43k |

**Architecture insight**: LocalAI auto-detects GPU vendor and downloads the correct backend version. It supports parallel requests via both vLLM and llama.cpp backends. Model gallery provides one-click model installation.

### AnythingLLM

| Aspect | Detail |
|---|---|
| **What it is** | All-in-one AI desktop app with RAG |
| **Backend** | Built-in LLM engine + 30+ external provider support |
| **Abstraction** | Factory pattern `getLLMProvider()`, common interface |
| **Key feature** | Workspace-level provider switching (per-workspace LLM) |
| **Key strength** | Zero-config local mode, extensive provider support |
| **Stars** | ~40k+ |

**Architecture insight**: AnythingLLM's provider abstraction uses a connection-based architecture — each workspace can have its own LLM connection (provider + model + settings). The factory pattern with environment-based configuration makes adding new providers straightforward.

### Aider

| Aspect | Detail |
|---|---|
| **What it is** | AI pair programming in terminal |
| **Backend** | LiteLLM (100+ providers via unified SDK) |
| **Abstraction** | `litellm.completion()` — single function for all providers |
| **Key strength** | Works with any LLM, cloud or local |
| **Local support** | Ollama, LM Studio, llama.cpp server via OpenAI-compatible URL |

**Architecture insight**: Aider delegates all LLM complexity to LiteLLM. Adding a new provider means adding it to LiteLLM, not to Aider. This is the "thinnest possible" integration — just model name + API key.

### Continue.dev

| Aspect | Detail |
|---|---|
| **What it is** | AI coding assistant (VS Code / JetBrains) |
| **Abstraction** | `ILLM` interface + `BaseLLM` base class, 40+ providers |
| **Config** | YAML-based `config.yaml` with provider/model specification |
| **Key pattern** | Dual-layer: interface contract + shared base implementation |

**Architecture insight**: Continue.dev's abstraction is more formal than Aider's. The `BaseLLM` handles all common operations (token counting, message compilation, capability detection), while provider subclasses only implement streaming. New providers added with ~50 lines of code.

**Sources:**
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Open WebUI GitHub](https://github.com/open-webui/open-webui)
- [text-generation-webui GitHub](https://github.com/oobabooga/text-generation-webui)
- [LM Studio 0.4.0 Blog](https://lmstudio.ai/blog/0.4.0)
- [Jan.ai GitHub](https://github.com/janhq/jan)
- [LocalAI Features](https://localai.io/features/index.print)
- [AnythingLLM GitHub](https://github.com/Mintplex-Labs/anything-llm)
- [Aider GitHub](https://github.com/Aider-AI/aider)
- [Continue.dev Model Providers](https://docs.continue.dev/customize/model-providers/overview)

---

## Anti-Patterns and What Doesn't Work

### Anti-Pattern 1: Requiring Users to Compile C++ with GPU Support

**This is WhisperJAV's current problem.** llama-cpp-python requires:
- C++ compiler (MSVC on Windows, GCC on Linux)
- CUDA Toolkit (not just the driver) for GPU support
- Correct compiler version matching CUDA version
- CMAKE flags for architecture-specific optimization
- Pre-built wheels exist but often have DLL loading failures due to CUDA version mismatches

**Evidence**: The llama-cpp-python GitHub issues are dominated by build failures:
- RTX 50 Blackwell GPU users can't build due to CUDA architecture support (#2028)
- GCC version incompatibilities (#2081)
- "ggml.dll not found" DLL loading failures
- Metal performance regressions between versions

**Every project that has moved away from this pattern has seen reduced support burden.**

### Anti-Pattern 2: Embedding LLM Inference In-Process via Python Bindings

Running inference inside the application process (via Python C bindings) causes:
- Python GIL contention
- Application hangs during inference
- Crashes in C code kill the entire application
- Memory leaks from C++ objects not properly freed by Python GC
- Platform-specific DLL/dylib loading nightmares

**Better**: Always use a separate process with HTTP communication.

### Anti-Pattern 3: Building Your Own Model Registry/Downloader

Several projects tried to implement custom model download systems from HuggingFace. Problems:
- Authentication complexity (gated models)
- Partial download recovery
- Disk space management
- Model format compatibility checking

**Better**: Delegate to Ollama's model management (`ollama pull`) or use well-tested libraries (huggingface_hub).

### Anti-Pattern 4: Trying to Support Every GPU Backend in Your Own Code

LocalAI and text-generation-webui maintain support for CUDA, ROCm, Metal, Vulkan, and CPU. This is a massive maintenance burden that only makes sense if your project IS an LLM server.

**Better for application developers**: Delegate GPU management to the LLM server (Ollama handles all of this automatically).

### Anti-Pattern 5: Silent Fallback Without User Awareness

Some projects silently fall back from GPU to CPU when GPU initialization fails. Users then experience 10-50x slower performance without understanding why.

**Better**: Detect and report clearly: "GPU not available (reason: CUDA 11 detected, requires CUDA 12+). Running on CPU — expect slower translation. To use GPU: [actionable steps]."

### Anti-Pattern 6: Monolithic "All-in-One" Packaging

Bundling the LLM engine, model files, GPU libraries, and application into a single installer creates:
- Massive installer size (several GB)
- Update complexity (any component change requires full reinstall)
- Version conflicts between bundled and system libraries

**Better**: Separate concerns — application installer + instructions to install Ollama (or auto-detect existing installation).

**Sources:**
- [llama-cpp-python Build Issues (GitHub)](https://github.com/abetlen/llama-cpp-python/issues/2028)
- [llama-cpp-python GCC Issues (GitHub)](https://github.com/abetlen/llama-cpp-python/issues/2081)
- [Ollama vs llama.cpp Comparison (Openxcell)](https://www.openxcell.com/blog/llama-cpp-vs-ollama/)
- [Sidecar Architecture Pattern (MCP)](https://themlarchitect.wpcomstaging.com/blog/the-architectural-elegance-of-model-context-protocol-mcp/)

---

## Recommendations for WhisperJAV

### Recommended Architecture: "Detect + Delegate + Fallback"

```
User clicks "Translate" in GUI
    │
    ├── 1. Check for running Ollama (localhost:11434)
    │       → Found: Use Ollama with user's model
    │       → Offer to pull recommended model if none loaded
    │
    ├── 2. Check for running LM Studio (localhost:1234)
    │       → Found: Use LM Studio with loaded model
    │
    ├── 3. Check user-configured custom server URL
    │       → Found: Use custom OpenAI-compatible server
    │
    ├── 4. No local server found
    │       → Show friendly message with options:
    │         a) "Install Ollama (free, 1 click): ollama.com"
    │         b) "Use cloud translation (requires API key)"
    │         c) "Configure custom LLM server URL"
    │
    └── 5. Cloud fallback (DeepSeek, OpenAI, etc.)
            → Use existing cloud provider code path
```

### Why This Architecture

1. **Eliminates compilation entirely.** No more llama-cpp-python build issues, DLL problems, CUDA Toolkit requirements.

2. **Leverages existing ecosystem.** Ollama has 162k stars, ships prebuilt GPU-accelerated binaries for every platform, auto-detects hardware. Why duplicate this?

3. **Maintains current code structure.** PySubtrans already uses OpenAI-compatible API. Changing `base_url` from `localhost:8000` (current llama-cpp-python) to `localhost:11434` (Ollama) requires minimal code changes.

4. **Zero new dependencies.** The `openai` Python package (or raw `requests`) is all that's needed. No llama-cpp-python, no CUDA, no cmake.

5. **User choice.** Power users can use their preferred LLM server. Non-technical users get clear guidance.

### Implementation Phases

#### Phase 1: Add Ollama Support (Low effort, high impact)

- Add server detection: probe `localhost:11434`, `localhost:1234`, custom URL
- Add `--provider ollama` and `--provider lm-studio` options
- Implement health check + model listing via `/v1/models`
- Friendly error messages when no server found
- Update GUI to show detected server status
- Keep llama-cpp-python as deprecated fallback

#### Phase 2: Deprecate llama-cpp-python (Medium effort)

- Move llama-cpp-python to `[legacy]` extra
- Add first-run guide in GUI: "For local translation, install Ollama"
- Add Ollama setup instructions to documentation
- Remove llama_build_utils.py, llama_cuda_config.py complexity

#### Phase 3: Optional Server Bundling (Future, if needed)

- If user demand requires zero-external-dependency local translation:
  - Bundle `llama-server` binary (prebuilt from llama.cpp releases, ~100-400MB)
  - Manage as subprocess with lifecycle management
  - This is lighter than bundling Ollama (~3-4GB)
- Or: create an "Ollama installer" step in WhisperJAV's installer

### Server Detection Implementation Sketch

```python
"""LLM server detection for WhisperJAV translation."""

import requests
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DetectedServer:
    name: str           # "ollama", "lm-studio", "custom"
    base_url: str       # "http://localhost:11434/v1/"
    models: List[str]   # Available model names
    gpu_info: Optional[str]  # "NVIDIA RTX 4090" if detectable

PROBE_TARGETS = [
    ("ollama",    "http://localhost:11434"),
    ("lm-studio", "http://localhost:1234"),
]

def detect_servers(custom_url: Optional[str] = None) -> List[DetectedServer]:
    """Probe known LLM server ports and return available servers."""
    servers = []

    targets = list(PROBE_TARGETS)
    if custom_url:
        targets.append(("custom", custom_url))

    for name, base_url in targets:
        try:
            # Check /v1/models endpoint (OpenAI standard)
            resp = requests.get(f"{base_url}/v1/models", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])]
                servers.append(DetectedServer(
                    name=name,
                    base_url=f"{base_url}/v1/",
                    models=models,
                    gpu_info=None,  # Could probe Ollama-specific API
                ))
        except (requests.ConnectionError, requests.Timeout):
            continue  # Server not running, skip

    return servers

def select_best_server(
    servers: List[DetectedServer],
    preferred_provider: Optional[str] = None,
) -> Optional[DetectedServer]:
    """Select the best available server, respecting user preference."""
    if not servers:
        return None

    # If user specified a provider, prefer it
    if preferred_provider:
        for s in servers:
            if s.name == preferred_provider:
                return s

    # Default priority: ollama > lm-studio > custom
    return servers[0]
```

### Model Recommendation Strategy

When Ollama is detected but no suitable model is loaded:

```python
RECOMMENDED_MODELS = {
    "high_vram":  {"model": "llama3.1:8b", "min_vram_gb": 6, "desc": "Best quality"},
    "medium_vram": {"model": "llama3.2:3b", "min_vram_gb": 3, "desc": "Good balance"},
    "low_vram":   {"model": "phi3:mini",    "min_vram_gb": 2, "desc": "Lightweight"},
}

def suggest_model(available_vram_gb: float) -> str:
    """Suggest an Ollama model based on available VRAM."""
    if available_vram_gb >= 6:
        return "llama3.1:8b"
    elif available_vram_gb >= 3:
        return "llama3.2:3b"
    else:
        return "phi3:mini"
```

### What This Eliminates from WhisperJAV

Files/complexity that can be removed or deprecated:

| Current File | Purpose | Replacement |
|---|---|---|
| `translate/local_backend.py` (600+ lines) | llama-cpp-python server management | 50-line Ollama client |
| `translate/llama_build_utils.py` | Building llama-cpp-python from source | Not needed |
| `translate/llama_cuda_config.py` | CUDA version detection for wheels | Not needed |
| `translate/service.py` (partial) | FastAPI server for local LLM | Not needed |
| DLL path setup code | PyTorch CUDA lib path hacking | Not needed |
| HuggingFace wheel hosting | Pre-built wheel distribution | Not needed |
| `installer/core/registry.py` llm entries | llama-cpp-python, fastapi, uvicorn | Not needed |

**Estimated code reduction**: ~1500+ lines of fragile, platform-specific code replaced by ~100 lines of HTTP client code.

---

## Key Takeaways

1. **OpenAI-compatible API is the universal standard.** WhisperJAV already uses it. The problem is the server side, not the client side.

2. **Ollama is the right delegation target.** 162k stars, prebuilt GPU binaries for all platforms, automatic hardware detection, model management, zero compilation.

3. **Process isolation via HTTP is the proven pattern.** Every successful project does this. Never embed inference in your application process.

4. **Compilation is the enemy of user experience.** Projects that require C++ compilation with GPU support generate the most support issues. Eliminate it.

5. **The "detect external server" pattern is the lightest-weight integration.** Minimal code, maximum flexibility, zero dependency on inference engine internals.

6. **Fallback chains create resilient UX.** Local GPU → local CPU → cloud API → clear error message. Never leave the user stuck.
