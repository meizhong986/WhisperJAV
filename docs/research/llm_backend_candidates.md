# LLM Backend Candidates: Deep-Dive Evaluation

**Date**: 2026-03-11
**Context**: WhisperJAV currently uses llama-cpp-python (Python bindings for llama.cpp) as a local LLM server for subtitle translation. The current implementation is fragile and platform-dependent. This document evaluates alternatives.

**WhisperJAV constraints**:
- Users are non-technical GUI users (installer-based deployment)
- Primary platform: Windows + NVIDIA GPU; also Apple Silicon, AMD GPU, Linux
- Use case: short subtitle translation (not long-form generation)
- Current model: llama-8b GGUF (Q4, ~6GB VRAM)
- Translation uses OpenAI-compatible chat completions API via PySubtrans
- Must integrate with existing conda-constructor installer (~150MB)

---

## Table of Contents

1. [Comparison Matrix](#comparison-matrix)
2. [Candidate 1: Ollama](#1-ollama)
3. [Candidate 2: vLLM](#2-vllm)
4. [Candidate 3: llama-cpp-python (current)](#3-llama-cpp-python-current)
5. [Candidate 4: llama.cpp server (direct)](#4-llamacpp-server-direct)
6. [Candidate 5: LocalAI](#5-localai)
7. [Candidate 6: MLX (Apple only)](#6-mlx-apple-only)
8. [Candidate 7: LM Studio (headless)](#7-lm-studio-headless)
9. [Recommendations](#recommendations)
10. [Migration Path](#migration-path)

---

## Comparison Matrix

| Criteria | Ollama | vLLM | llama-cpp-python (current) | llama.cpp server | LocalAI | MLX | LM Studio |
|---|---|---|---|---|---|---|---|
| **Language** | Go | Python/C++ | Python/C++ | C/C++ | Go | Python/C++ | TypeScript/C++ |
| **Windows native** | Yes | No (WSL/Docker) | Yes (fragile) | Yes | No (Docker) | No | Yes |
| **macOS (Apple Silicon)** | Yes (Metal) | No | Yes (Metal) | Yes (Metal) | Docker only | Yes (native) | Yes |
| **Linux** | Yes | Yes | Yes | Yes | Yes | Limited | Yes |
| **CUDA** | Yes (auto) | Yes | Yes (DLL issues) | Yes | Yes | No | Yes |
| **Vulkan** | Experimental | No | No | Yes | No | No | Yes |
| **ROCm (AMD)** | Yes | Yes | Limited | Yes | Yes | No | Yes |
| **MPS (Apple)** | Yes (Metal) | No | Yes | Yes (Metal) | No | Yes (native) | Yes |
| **OpenAI API** | Yes (/v1/) | Yes | Yes | Yes | Yes | Via wrappers | Yes |
| **Model format** | GGUF (+registry) | safetensors | GGUF | GGUF | Multiple | MLX format | GGUF |
| **Binary size** | ~3-4GB (w/ GPU libs) | N/A (pip) | ~50MB wheel | ~100-400MB | ~2GB+ (Docker) | ~50MB (pip) | ~500MB+ |
| **GPU auto-detect** | Excellent | Good | Manual/fragile | Manual | Good | Automatic | Excellent |
| **Bundle-able** | Partial (zip) | No | Yes (wheel) | Yes (binary) | No | Yes (pip) | No (proprietary) |
| **GitHub stars** | ~162k | ~55k+ | ~8k | (part of 85k) | ~43k | ~20k | N/A (closed) |
| **Release cadence** | ~every 2 days | Weekly | Monthly | Daily | Monthly | Weekly | Monthly |
| **Integration effort** | Low | Very High | Zero (current) | Medium | High | Medium | Medium |
| **User complexity** | Low | Very High | Medium (hidden) | Low-Medium | High | Low (Mac only) | Medium |

---

## 1. Ollama

**Website**: https://ollama.com | **GitHub**: https://github.com/ollama/ollama

### Architecture

- Written in **Go**, ships as a single binary with bundled GPU runtime libraries
- Client-server model: `ollama serve` runs an HTTP server, CLI/API clients communicate via REST
- Uses llama.cpp under the hood but abstracts away all complexity
- Manages its own model registry with pull/push semantics (like Docker for models)
- Process model: single server process, manages model loading/unloading automatically

### Platform Support

| Platform | Status | GPU Backend |
|---|---|---|
| Windows x64 | Full support (installer + zip) | CUDA, Vulkan (experimental) |
| macOS Apple Silicon | Full support (app bundle) | Metal (native) |
| macOS Intel | CPU only | None |
| Linux x64 | Full support (script install) | CUDA, ROCm, Vulkan (experimental) |
| Docker | Full support | CUDA, ROCm |

### GPU Detection

Ollama has the most sophisticated GPU discovery system among all candidates:

- **Two-phase bootstrap**: Serial enumeration of GPU libraries, then parallel validation
- **Multi-backend discovery**: Detects CUDA, ROCm, Metal, and Vulkan simultaneously
- **Deduplication**: Same GPU detected by multiple backends (e.g., CUDA + Vulkan) is deduplicated with priority rules
- **Timeouts**: 30s on Linux/macOS, 90s on Windows (accounts for Defender DLL scanning)
- **Automatic fallback**: GPU failure gracefully falls back to CPU
- **VRAM detection**: Queries available VRAM and adjusts model layer offloading automatically

This is a **massive advantage** over llama-cpp-python, where GPU detection is manual and DLL loading is fragile.

### Dependencies

- **Zero Python dependencies** for the server itself (Go binary)
- Optional Python client library: `pip install ollama` (pure Python, minimal deps)
- GPU libraries are **bundled** in the distribution (no separate CUDA Toolkit install needed)
- No numpy, no torch dependency for the server

### API Compatibility

OpenAI-compatible API at `/v1/`:
- `POST /v1/chat/completions` - Chat completions (streaming supported)
- `GET /v1/models` - List models
- `POST /v1/embeddings` - Embeddings

Native Ollama API:
- `POST /api/pull` - Download model (with progress streaming)
- `POST /api/chat` - Chat
- `POST /api/generate` - Generate
- `GET /api/tags` - List local models
- `POST /api/create` - Create model from Modelfile
- `DELETE /api/delete` - Delete model

**Important**: OpenAI compatibility is marked "experimental" but widely used in production. No API key required (dummy key accepted if set by client).

### Model Format & Management

- **GGUF native**: Ollama's primary format is GGUF (same as WhisperJAV's current models)
- **Registry**: `ollama.com/library` hosts curated models (`ollama pull llama3.1:8b`)
- **Custom GGUF import**: Create a `Modelfile` with `FROM /path/to/model.gguf`, then `ollama create mymodel`
- **HuggingFace direct**: `ollama run hf.co/user/repo:Q4_K_M` pulls GGUF from HF Hub
- **Automatic quantization**: Can import safetensors and auto-quantize
- **Model storage**: `~/.ollama/models/` (configurable via `OLLAMA_MODELS` env var)

### Installation Complexity

**For end users**: Very low.
- Windows: `OllamaSetup.exe` (no admin required, installs to user profile)
- macOS: `Ollama.app` drag-and-drop
- Linux: `curl -fsSL https://ollama.com/install.sh | sh`
- Silent install: `/S` flag on Windows, or use standalone zip

**For bundling with WhisperJAV**:
- Standalone zip available: `ollama-windows-amd64.zip`
- **Size concern**: ~3-4GB on Windows (includes NVIDIA + AMD GPU libraries)
- Could potentially strip to NVIDIA-only (~1.5-2GB) but unsupported
- Alternative: Detect and download Ollama at first use (like current llama-cpp-python lazy install)

### Resource Requirements (8B Q4 model)

- **VRAM**: 5-6GB for full GPU offload
- **System RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~4.5GB for model file + ~3-4GB for Ollama itself
- **Inference speed**: 40-70 tokens/sec on RTX 3060 Ti / RTX 4060

### Maturity & Community

- **162k+ GitHub stars** (March 2026) - most popular local LLM tool
- **451+ releases**, ~every 2 days release cadence
- **Backed by Ollama Inc.** (funded startup)
- Extremely active community, extensive documentation
- Used by major projects (Open WebUI, Continue, LibreChat, etc.)

### Integration Effort for WhisperJAV

**Low to Medium**. Key changes:

1. **Server lifecycle**: Replace llama-cpp-python subprocess with Ollama subprocess management
   - Start: `ollama serve` (or check if already running on port 11434)
   - Health: `GET http://localhost:11434/` returns "Ollama is running"
   - Stop: Process termination (or leave running as user service)

2. **Model management**: Replace HuggingFace GGUF download with Ollama model pull
   - `ollama pull` or `POST /api/pull` with progress streaming
   - Custom models via Modelfile: `FROM /path/to/uncensored-model.gguf`
   - Or publish curated models to Ollama registry

3. **API integration**: Minimal change - already uses OpenAI-compatible API
   - Change base URL from `http://localhost:{port}/v1` to `http://localhost:11434/v1`
   - PySubtrans already supports OpenAI API format

4. **GPU detection**: Remove all CUDA DLL path hacking, AVX2 detection, wheel management
   - Ollama handles GPU detection internally
   - Diagnostics: Parse `ollama ps` output for GPU status

### Known Issues & Risks

- **Binary size**: 3-4GB standalone zip is too large to bundle with WhisperJAV installer
- **External dependency**: Users must install Ollama separately (or WhisperJAV must orchestrate installation)
- **Model registry dependency**: Ollama's model naming differs from raw GGUF files
- **Context window**: Default 2048 tokens, must be explicitly configured (llama.cpp defaults to model max)
- **Port conflicts**: Default port 11434 may conflict with other Ollama instances
- **Version pinning**: Rapid release cadence means potential breaking changes

---

## 2. vLLM

**Website**: https://docs.vllm.ai | **GitHub**: https://github.com/vllm-project/vllm

### Architecture

- Python/C++ hybrid, production-grade inference engine
- Optimized for high-throughput server workloads (continuous batching, PagedAttention)
- Designed for data center / cloud GPU deployment
- Process model: heavyweight Python process with CUDA kernels

### Platform Support

| Platform | Status | Notes |
|---|---|---|
| Linux x64 | Full support | Primary platform |
| Windows | **Not supported natively** | WSL2 or Docker only |
| macOS | **Not supported** | No Metal/MPS backend |
| Docker | Full support | NVIDIA GPU passthrough |

### GPU Support

- **CUDA**: Full support (primary backend)
- **ROCm**: Supported
- **MPS/Metal**: Not supported
- **Vulkan**: Not supported

### Dependencies

- Heavy Python dependency chain: PyTorch, numpy, transformers, etc.
- Requires CUDA Toolkit for compilation
- Python 3.9-3.12
- Flash Attention required for optimal performance

### API Compatibility

- OpenAI-compatible server mode
- Full streaming support
- Production-ready API

### Model Format

- Primarily **safetensors** (HuggingFace format)
- GGUF support added recently but not primary
- AWQ, GPTQ quantization support

### Resource Requirements

- Significantly higher than llama.cpp-based solutions
- Optimized for throughput, not single-user latency
- Minimum 16GB+ VRAM recommended

### Assessment for WhisperJAV

**REJECTED: Not suitable.**

- No native Windows support (deal-breaker for primary platform)
- No macOS support (deal-breaker for Apple Silicon users)
- Requires Docker/WSL2 - unacceptable for non-technical GUI users
- Massive dependency footprint
- Designed for server farms, not desktop applications
- Overkill for subtitle translation (short text, single user)

---

## 3. llama-cpp-python (current)

**GitHub**: https://github.com/abetlen/llama-cpp-python

### Architecture

- Python bindings (ctypes) for llama.cpp C++ library
- Includes a FastAPI-based OpenAI-compatible server
- Ships as pre-built wheels with platform-specific native libraries
- WhisperJAV manages the server as a subprocess

### Current Implementation in WhisperJAV

The current implementation in `whisperjav/translate/local_backend.py` is **extensive** (~800 lines) and handles:

- CUDA DLL path setup (Windows): Injects PyTorch's CUDA libs into DLL search path
- Linux CUDA library path setup
- CUDA version detection from PyTorch
- Pre-built wheel selection and download from HuggingFace
- Build-from-source fallback
- AVX2 CPU support detection
- Visual C++ Runtime detection
- Server subprocess lifecycle management
- Health check and diagnostics
- Model download from HuggingFace Hub
- VRAM detection and automatic model selection

### Known Failure Modes

1. **DLL Loading Failures (Windows)**: The most common issue. `llama.dll` fails to load because:
   - CUDA runtime libraries (cublas64_12.dll, cudart64_12.dll) not found
   - ctypes `winmode=0` vs default behavior conflicts
   - CUDA version mismatch between wheel and system
   - Missing Visual C++ 2015-2022 Redistributable
   - PyTorch CUDA libs path not in DLL search path

2. **Wheel Version Mismatch**: Pre-built wheels compiled for specific CUDA versions. System CUDA version may differ.

3. **Build-from-Source Failures**: Requires CMake, C++ compiler, CUDA Toolkit - most users don't have these.

4. **Platform-Specific Quirks**:
   - Windows: DLL search order is unpredictable
   - macOS: Metal support requires specific wheel variants
   - Linux: `libcudart.so.12` library path issues

5. **AVX2 Requirement**: Pre-built wheels use AVX2; older CPUs crash with `STATUS_ILLEGAL_INSTRUCTION`

### Why It's Fragile

The root cause is **the impedance mismatch between Python's ctypes and native C++ library loading**:

- Python wheels bundle `.dll`/`.so` files but have limited control over DLL search paths
- CUDA libraries may exist in multiple locations (PyTorch's, system CUDA Toolkit, conda env)
- Version conflicts are common and produce cryptic errors
- Each platform has different library loading semantics

WhisperJAV has ~300 lines of workaround code just for DLL loading (`_setup_pytorch_cuda_dll_paths`, `_setup_linux_cuda_library_paths`, `_diagnose_dll_failure`, `_check_cuda_toolkit_installed`, `_check_vc_runtime_installed`, `_check_cpu_avx2_support`).

### What Works Well

- Once working, inference performance is excellent
- Direct Python integration (no external process needed for library mode)
- OpenAI-compatible server mode
- GGUF format support matches WhisperJAV's model pipeline
- Mature project with good documentation

### Assessment

**The DLL/shared library loading problem is inherent to the Python bindings approach and cannot be fully solved.** Every CUDA version bump, every llama.cpp update, and every platform variation can break the loading chain. The workaround code in WhisperJAV is evidence of fighting against the architecture.

---

## 4. llama.cpp server (direct)

**GitHub**: https://github.com/ggml-org/llama.cpp (85k+ stars)

### Architecture

- Pure C/C++ with **zero external dependencies** (no Python runtime needed)
- `llama-server` binary provides HTTP server with OpenAI-compatible API
- Single binary deployment - no DLL loading issues because everything is statically linked
- Process model: single native process, managed via subprocess from Python

### Platform Support

| Platform | Status | Pre-built Binaries |
|---|---|---|
| Windows x64 | Full support | Yes (CUDA 12.4, CUDA 13.1, Vulkan, CPU) |
| macOS Apple Silicon | Full support | Yes (Metal native) |
| macOS Intel | CPU only | Yes |
| Linux x64 | Full support | Yes (CUDA, Vulkan, CPU) |

### GPU Support

- **CUDA**: Full support (pre-built binaries for CUDA 12.4 and 13.1)
- **Vulkan**: Full support (works on NVIDIA, AMD, Intel GPUs)
- **Metal**: Full support (Apple Silicon)
- **ROCm**: Supported (build from source or use HIP binaries)
- **SYCL**: Supported (Intel GPUs)

**Vulkan is a significant advantage**: It works across all GPU vendors (NVIDIA, AMD, Intel) without vendor-specific drivers beyond the standard Vulkan runtime. This could simplify AMD GPU support dramatically.

### Pre-built Binary Availability

Official releases on GitHub provide ZIP archives:
- `llama-b{build}-bin-win-cuda-cu12.4-x64.zip` (~373MB)
- `llama-b{build}-bin-win-cuda-cu13.1-x64.zip` (~384MB)
- `llama-b{build}-bin-win-vulkan-x64.zip` (smaller)
- `llama-b{build}-bin-win-cpu-x64.zip` (smallest)
- macOS and Linux variants also available

### API Compatibility

Full OpenAI-compatible API:
- `POST /v1/chat/completions` - Chat completions (streaming via SSE)
- `POST /v1/completions` - Text completions
- `POST /v1/embeddings` - Embeddings
- `GET /health` - Health check endpoint
- `GET /v1/models` - Model list

Additional llama.cpp-specific endpoints:
- `POST /slots` - KV cache slot management
- `GET /props` - Server properties
- `POST /tokenize` / `POST /detokenize`

Supports `response_format` for structured JSON output.

### Server Management

```bash
# Start server
llama-server -m model.gguf --port 8080 -ngl 999 --ctx-size 8192

# Health check
curl http://localhost:8080/health

# Chat completion
curl http://localhost:8080/v1/chat/completions -d '{...}'
```

Key parameters:
- `-ngl` / `--n-gpu-layers`: GPU layer offloading (999 = all layers)
- `--ctx-size`: Context window size
- `--port`: Server port
- `-m`: Model path
- `--list-devices`: Enumerate available GPU devices

### Installation Complexity

**Very low for pre-built binaries**:
- Download ZIP from GitHub releases
- Extract to directory
- Run `llama-server` with model path
- No installation step, no registry entries, no dependencies

**For WhisperJAV bundling**:
- Could ship the appropriate binary variant (~100-400MB per platform)
- Or download the correct binary at first use based on detected GPU
- Binary selection logic: detect CUDA version -> choose CUDA binary, else Vulkan, else CPU

### Resource Requirements (8B Q4 model)

- **VRAM**: 5-6GB for full GPU offload (same as llama-cpp-python - same engine)
- **System RAM**: 8GB minimum
- **Disk**: ~100-400MB for server binary + ~4.5GB for model
- **Inference speed**: Comparable to Ollama (both use llama.cpp under the hood)

### Maturity & Community

- **85k+ GitHub stars** (as part of llama.cpp project)
- **4000+ releases**, near-daily release cadence
- **1200+ contributors**
- The foundation that Ollama, LM Studio, and many others build upon
- Extremely active development

### Integration Effort for WhisperJAV

**Medium**. Key changes:

1. **Binary management**: Download and cache the correct `llama-server` binary
   - Detect GPU: CUDA version, Vulkan availability, Metal (macOS)
   - Download appropriate ZIP from GitHub releases
   - Extract to WhisperJAV data directory

2. **Server lifecycle**: Replace llama-cpp-python subprocess with llama-server subprocess
   - Start: `llama-server -m /path/to/model.gguf --port {port} -ngl 999`
   - Health: `GET http://localhost:{port}/health`
   - Stop: Process termination

3. **Model management**: Keep existing HuggingFace GGUF download pipeline
   - No model format change needed
   - No registry dependency

4. **API integration**: Minimal change
   - Same OpenAI-compatible API format
   - Change base URL

5. **GPU detection**: Simplified but still needed
   - Detect CUDA version for binary selection
   - Or use Vulkan binary as universal fallback

### Known Issues & Risks

- **Binary selection**: Must choose correct binary variant for user's system
- **No model management**: Must handle model download/caching ourselves (already do this)
- **Startup parsing**: Need to parse stdout/stderr for GPU layer info and readiness
- **Version pinning**: Rapid releases; need to pin to tested versions
- **Vulkan performance**: ~10-30% slower than CUDA for NVIDIA GPUs

### Key Advantage Over llama-cpp-python

**Eliminates the DLL loading problem entirely.** The server is a statically-linked binary that bundles its own CUDA/Vulkan runtime. No Python ctypes, no DLL search path hacking, no wheel version matching. If the binary runs, it works.

---

## 5. LocalAI

**Website**: https://localai.io | **GitHub**: https://github.com/mudler/LocalAI (43k+ stars)

### Architecture

- Written in **Go** with multiple backend bindings
- Docker-first deployment model
- Multi-modal: text, image, audio generation
- Uses llama.cpp, vLLM, Transformers, ExLlama as backends

### Platform Support

| Platform | Status | Notes |
|---|---|---|
| Linux | Full support | Primary platform |
| Windows | **Docker/WSL2 only** | No native binary |
| macOS | Docker only | No Metal support |

### Assessment for WhisperJAV

**REJECTED: Not suitable.**

- **Docker-only on Windows** - unacceptable for non-technical GUI users
- Requires 100GB+ disk space for container runtime
- No native Windows or macOS binary
- Overkill for subtitle translation (multi-modal AI platform)
- Higher RAM footprint (8-12GB baseline) than alternatives
- Complex configuration

While LocalAI is a capable platform, its Docker-first approach and lack of native Windows support make it unsuitable for WhisperJAV's target audience.

---

## 6. MLX (Apple Only)

**Website**: https://mlx-framework.org | **GitHub**: https://github.com/ml-explore/mlx (20k+ stars)

### Architecture

- Apple's native array framework for Apple Silicon
- `mlx-lm` package provides LLM inference and an OpenAI-compatible server
- Leverages Metal GPU and Neural Engine directly
- Uses MLX-format models (converted from safetensors/GGUF)
- Several community servers: vllm-mlx, oMLX, mlx-openai-server

### Platform Support

| Platform | Status |
|---|---|
| macOS Apple Silicon (M1+) | Full support (native) |
| macOS Intel | Not supported |
| Linux | CPU-only (experimental) |
| Windows | Not supported |

### Performance

- **vllm-mlx**: 2.17x faster than llama.cpp single-stream on M3 Ultra
- **Prefix caching**: 93.7% token savings in agentic workflows
- **M5 Neural Accelerators**: Up to 4x speedup for TTFT (latest hardware)
- Native Metal performance without translation layers

### Dependencies

- `pip install mlx-lm` (Python package)
- Requires macOS >= 14.0
- Apple Silicon only
- Moderate dependency chain (numpy, transformers, etc.)

### API Compatibility

- `mlx-lm` includes built-in OpenAI-compatible server
- Community projects (vllm-mlx, oMLX) add full API compatibility
- Streaming support

### Model Format

- MLX format (converted from safetensors)
- Large community model hub on HuggingFace (`mlx-community/`)
- GGUF not directly supported (needs conversion)

### Assessment for WhisperJAV

**Suitable as a Mac-specific backend complement, not a primary solution.**

- Only works on Apple Silicon - cannot replace the primary backend
- Best-in-class performance on Mac hardware
- Could be used as an optional backend when detected on macOS
- Model format difference (MLX vs GGUF) adds complexity
- Integration via OpenAI-compatible API is straightforward

**Recommendation**: Consider as a future enhancement for Mac users, not as part of the primary backend migration.

---

## 7. LM Studio (headless)

**Website**: https://lmstudio.ai | **Closed source**

### Architecture

- Desktop application with headless daemon mode (`llmster`)
- Built on llama.cpp (and MLX for Mac)
- TypeScript/Electron GUI + native inference backends
- CLI: `lms` commands for server/model management

### Platform Support

| Platform | Status |
|---|---|
| Windows x64 | Full support |
| macOS Apple Silicon | Full support (MLX + Metal) |
| macOS Intel | CPU only |
| Linux x64 | Full support |

### Key Features

- `lms daemon up` - Start headless daemon
- `lms server start` - Start API server
- `lms get <model>` - Download model
- OpenAI-compatible API
- Automatic GPU detection and optimization
- Polished model management

### Assessment for WhisperJAV

**Not recommended as a dependency.**

- **Closed source / proprietary** - cannot bundle or redistribute
- Users would need to install LM Studio separately
- No guarantee of API stability or long-term availability
- License restrictions unclear for embedding in other applications
- Good alternative for users who already have LM Studio installed

Could be supported as an optional "bring your own server" backend (user points WhisperJAV at LM Studio's API endpoint), but not as the primary solution.

---

## Recommendations

### Primary Recommendation: Ollama

**Ollama is the strongest candidate** for replacing llama-cpp-python, for the following reasons:

1. **GPU detection is solved**: Ollama's two-phase GPU discovery with multi-backend support eliminates the #1 pain point (DLL loading, CUDA detection)

2. **Cross-platform**: Native support on Windows, macOS (Metal), Linux with identical API

3. **OpenAI-compatible API**: Drop-in replacement for current API integration. PySubtrans needs zero changes.

4. **Model management**: Built-in model download/caching. Can import custom GGUF models via Modelfile.

5. **User experience**: Simple installation, automatic GPU utilization, no configuration needed

6. **Community momentum**: 162k+ stars, near-daily releases, strong ecosystem

7. **Zero Python dependencies for server**: Eliminates numpy conflicts, torch version mismatches, wheel compatibility issues

**Challenges to solve**:
- Binary size (~3-4GB) is too large to bundle in the installer
- Must orchestrate Ollama installation as a prerequisite or first-run step
- Custom uncensored models need Modelfile-based import

### Secondary Recommendation: llama.cpp server (direct)

**llama.cpp server is the strongest alternative** if Ollama's size/external dependency is unacceptable:

1. **Eliminates DLL loading**: Statically-linked binary - if it runs, it works

2. **Smaller footprint**: ~100-400MB per platform variant (vs 3-4GB for Ollama)

3. **Same GGUF models**: No model pipeline changes needed

4. **Bundle-able**: Can ship the binary with WhisperJAV installer or download at first use

5. **Vulkan support**: Universal GPU backend that works across NVIDIA/AMD/Intel

6. **Full control**: Direct subprocess management, no external service dependency

**Trade-offs vs Ollama**:
- Must handle binary variant selection (CUDA 12.4 vs 13.1 vs Vulkan vs CPU)
- No built-in model management (keep existing HuggingFace download)
- Less user-friendly error messages
- No automatic VRAM detection / layer offloading optimization

### Rejected Candidates

| Candidate | Rejection Reason |
|---|---|
| **vLLM** | No native Windows or macOS support. Docker/WSL2 only. Overkill for desktop. |
| **LocalAI** | Docker-only on Windows. 100GB+ disk. Too complex for end users. |
| **LM Studio** | Closed source, cannot bundle. Proprietary license. |

### Future Enhancement

| Candidate | Role |
|---|---|
| **MLX** | Optional Mac-specific backend for superior Apple Silicon performance |

---

## Migration Path

### Phase 1: Abstraction Layer (Low Risk)

Create a backend abstraction that allows swapping the LLM server implementation:

```python
# whisperjav/translate/llm_backend.py (new)

class LLMBackend(Protocol):
    def start(self, model: str, gpu_layers: int) -> ServerInfo: ...
    def stop(self) -> None: ...
    def health_check(self) -> HealthStatus: ...
    def get_api_base(self) -> str: ...

class LlamaCppPythonBackend(LLMBackend):
    """Current implementation - refactored from local_backend.py"""
    ...

class OllamaBackend(LLMBackend):
    """New Ollama-based backend"""
    ...

class LlamaCppServerBackend(LLMBackend):
    """Direct llama.cpp server binary backend"""
    ...
```

### Phase 2: Implement Ollama Backend

1. **Ollama detection**: Check if `ollama serve` is running or available
2. **Model import**: Create Modelfile from existing GGUF paths, `ollama create`
3. **Server lifecycle**: Start/stop/health check via Ollama API
4. **Diagnostics**: Parse `ollama ps` for GPU status, layer count, speed

### Phase 3: Implement llama.cpp Server Backend (Fallback)

1. **Binary download**: Detect GPU, download correct binary variant from GitHub releases
2. **Server lifecycle**: Subprocess management with `llama-server`
3. **Health check**: Poll `/health` endpoint
4. **GPU diagnostics**: Parse startup logs for layer offloading info

### Phase 4: Backend Selection Logic

```
1. If Ollama is installed and running -> use OllamaBackend
2. If Ollama is installed but not running -> start it, use OllamaBackend
3. If Ollama not installed -> try LlamaCppServerBackend (download binary)
4. Fallback -> LlamaCppPythonBackend (current, for backward compatibility)
5. User override: --llm-backend ollama|llama-server|llama-cpp-python
```

### Phase 5: Deprecate llama-cpp-python

Once Ollama and llama.cpp server backends are stable:
1. Remove llama-cpp-python from default installation
2. Keep as optional fallback for advanced users
3. Remove DLL workaround code (~300 lines)
4. Remove wheel management code (~200 lines)

### Estimated Code Impact

| Component | Lines Changed | Risk |
|---|---|---|
| New: `llm_backend.py` abstraction | +200 | Low |
| New: `ollama_backend.py` | +300 | Low |
| New: `llamacpp_server_backend.py` | +250 | Low |
| Modified: `local_backend.py` (refactor) | ~100 | Medium |
| Modified: `service.py` (backend selection) | ~50 | Low |
| Eventually removed: DLL workarounds | -500 | Low |

---

## Sources

### Ollama
- [Ollama Official Site](https://ollama.com)
- [Ollama GitHub](https://github.com/ollama/ollama) - 162k+ stars
- [GPU Discovery Architecture](https://deepwiki.com/ollama/ollama/6.1-gpu-discovery-and-backend-loading)
- [Hardware Support](https://docs.ollama.com/gpu)
- [OpenAI Compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Importing Models](https://docs.ollama.com/import)
- [Windows Documentation](https://docs.ollama.com/windows)
- [Integrating Into Desktop App - Issue #7419](https://github.com/ollama/ollama/issues/7419)
- [Windows Binary Size - Issue #9191](https://github.com/ollama/ollama/issues/9191)
- [Ollama Python Library](https://github.com/ollama/ollama-python)

### vLLM
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Installation Docs](https://docs.vllm.ai/en/latest/getting_started/installation/)
- [Docker Model Runner vLLM on Windows](https://www.docker.com/blog/docker-model-runner-vllm-windows/)
- [vLLM Windows CUDA RFC - Issue #14981](https://github.com/vllm-project/vllm/issues/14981)
- [vLLM-Windows Community Fork](https://github.com/SystemPanic/vllm-windows)

### llama-cpp-python
- [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python) - 8k stars
- [DLL Loading Fix - DEV Community](https://dev.to/jirenmaa/failed-to-load-shared-library-llamadll-could-not-find-llama-cpp-python-5e4l)
- [Failed to load shared library - Issue #208](https://github.com/abetlen/llama-cpp-python/issues/208)
- [llama.dll not found - Issue #1150](https://github.com/abetlen/llama-cpp-python/issues/1150)
- [DLL Loading Issues - Issue #1280](https://github.com/abetlen/llama-cpp-python/issues/1280)
- [Build with CUDA on Windows - Issue #1963](https://github.com/abetlen/llama-cpp-python/issues/1963)

### llama.cpp Server
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp) - 85k+ stars
- [Server Documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Pre-built Releases](https://github.com/ggml-org/llama.cpp/releases)
- [Pre-built CUDA Binaries](https://github.com/ai-dock/llama.cpp-cuda)
- [Vulkan Performance Discussion](https://github.com/ggml-org/llama.cpp/discussions/10879)
- [Vulkan Support in Ollama](https://www.phoronix.com/news/ollama-Experimental-Vulkan)

### LocalAI
- [LocalAI Website](https://localai.io)
- [LocalAI GitHub](https://github.com/mudler/LocalAI) - 43k stars
- [GPU Acceleration Docs](https://localai.io/features/gpu-acceleration/)
- [LocalAI vs Ollama Comparison](https://devtechinsights.com/self-hosted-ai-ollama-localai-2026/)

### MLX
- [MLX GitHub](https://github.com/ml-explore/mlx) - 20k stars
- [mlx-lm on PyPI](https://pypi.org/project/mlx-lm/)
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx)
- [oMLX](https://omlx.ai/)
- [Apple M5 Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)

### LM Studio
- [LM Studio Developer Docs](https://lmstudio.ai/docs/developer)
- [Headless Mode Docs](https://lmstudio.ai/docs/developer/core/headless)
- [LM Studio 0.4.0 Release](https://lmstudio.ai/blog/0.4.0)

### Comparison Articles
- [Ollama vs Llama.cpp - Openxcell (2026)](https://www.openxcell.com/blog/llama-cpp-vs-ollama/)
- [Local LLM Hosting Guide - Glukhov (2025)](https://www.glukhov.org/post/2025/11/hosting-llms-ollama-localai-jan-lmstudio-vllm-comparison/)
- [Complete Guide to Ollama Alternatives (2026)](https://localllm.in/blog/complete-guide-ollama-alternatives)
- [vLLM or llama.cpp - Red Hat Developer (2025)](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case)
- [Ollama VRAM Requirements Guide (2026)](https://localllm.in/blog/ollama-vram-requirements-for-local-llms)
