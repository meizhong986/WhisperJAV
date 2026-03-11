# WhisperJAV Roadmap: LLM Backend Migration & NumPy 2 Upgrade

**Date**: 2026-03-11
**Based on**: 4 research reports (llm_backend_best_practices.md, llm_backend_candidates.md, numpy2_modelscope_feasibility.md, numpy2_test_suite_report.md)

---

## Strategic Overview

Two major technical debts have been identified. The good news: they reinforce each other and can be addressed in a coordinated sequence.

| Problem | Current State | Target State |
|---------|--------------|--------------|
| **LLM backend** | llama-cpp-python (fragile DLL loading, CUDA compilation) | Ollama (prebuilt Go binary, zero Python deps) |
| **numpy pin** | `numpy<2.0` (outdated, blocks modern AI packages) | `numpy>=1.26.0` (lift pin, allow both 1.x and 2.x) |

**Key insight**: Migrating the LLM backend to Ollama *removes* it from Python dependency resolution entirely (Go binary, no pip). This makes the numpy 2 migration simpler — one fewer source of conflicts.

---

## Release Plan

### v1.8.8 — Stability + Groundwork (Near-term)

**Theme**: Fix immediate user-facing bugs, lay groundwork for the bigger changes.

#### A. Fix LLM Context Overflow Bug (Critical — #196, #212, #132)

This is the most-reported bug. Users get "No matches found" on long files because PySubtrans overflows the 8K context window with system instructions + conversation history + subtitle lines.

**Fix approach** (does not require backend migration):
1. Reduce system prompt token count (currently bloated)
2. Implement dynamic batch sizing — measure available context, adjust subtitle batch size accordingly
3. Add retry with smaller batch on "No matches found" error
4. These fixes work regardless of backend (llama-cpp-python, Ollama, cloud)

**Files**: `whisperjav/translate/core.py`, `whisperjav/translate/instructions.py`

#### B. Lift numpy<2 Pin

**Research finding**: The pin is unnecessary. ModelScope doesn't constrain numpy. ClearVoice fork already removed the pin. Only 1 line of WhisperJAV code needs fixing.

**Changes**:
| File | Change |
|------|--------|
| `pyproject.toml` | `numpy>=1.26.0,<2.0` → `numpy>=1.26.0` |
| `requirements.txt` | Same |
| `whisperjav/upgrade.py` | Same |
| `whisperjav/utils/metadata_manager.py` | Remove `np.int_` from isinstance check |
| `pyproject.toml` | Bump: `scipy>=1.12.0`, `numba>=0.59.0`, `librosa>=0.10.2`, `scikit-learn>=1.4.0`, `opencv-python>=4.9.0` |

**Risk**: Low. Allows pip to resolve numpy 1.x or 2.x based on environment. No forced upgrade.

#### C. Fix MPS Beam Search Crash (#198)

Force `num_beams=1` on MPS device in TransformersASR. The HuggingFace beam search implementation crashes on Apple Silicon with out-of-bounds index errors.

**File**: `whisperjav/modules/whisper_pro_asr.py`

#### D. Add `--provider ollama` (Preview)

Add Ollama as a recognized provider alongside the existing `custom` provider. This is a thin convenience layer — users can already use Ollama via `--provider custom --translate-server-address http://localhost:11434 --translate-endpoint /v1/chat/completions`. Making it a first-class provider improves discoverability.

**Changes**:
```python
# providers.py — add:
'ollama': {
    'pysubtrans_name': 'Custom Server',
    'model': 'llama3.1:8b',
    'env_var': None,
    'server_address': 'http://localhost:11434',
    'endpoint': '/v1/chat/completions',
},
```

Plus: auto-detection of running Ollama server, friendly error message if not found.

**This is NOT the full migration** — just making Ollama easily accessible while llama-cpp-python remains the `local` provider.

---

### v1.9.0 — LLM Backend Migration (Medium-term)

**Theme**: Replace llama-cpp-python with Ollama as the primary local LLM backend.

#### Phase 1: Backend Abstraction Layer

Create a clean interface that decouples translation from the LLM server implementation.

```python
# whisperjav/translate/llm_backend.py (new)
class LLMBackend(Protocol):
    """Interface for local LLM server backends."""
    def detect(self) -> bool: ...           # Is this backend available?
    def start(self, model: str) -> str: ... # Start server, return base_url
    def stop(self) -> None: ...             # Stop server
    def health_check(self) -> bool: ...     # Is server ready?
    def list_models(self) -> list[str]: ... # Available models
```

Refactor `local_backend.py` (2423 lines) into:
- `llm_backend.py` — Protocol + factory (~100 lines)
- `backends/llama_cpp_python.py` — Current implementation, refactored (~800 lines)
- `backends/ollama.py` — New Ollama backend (~200 lines)
- `backends/llama_server.py` — Direct llama.cpp server binary (~250 lines)

**What gets deleted**: DLL path hacking, CUDA wheel management, AVX2 detection, build-from-source logic — everything that makes the current approach fragile. Approximately **1500 lines** of workaround code.

#### Phase 2: Ollama Backend Implementation

The Ollama backend is simple because Ollama handles all the hard parts:

```
Detection:    GET http://localhost:11434/       → "Ollama is running"
Models:       GET http://localhost:11434/api/tags → list of pulled models
Model pull:   POST http://localhost:11434/api/pull → download with progress
Health:       GET http://localhost:11434/v1/models → model loaded and ready
Translation:  POST http://localhost:11434/v1/chat/completions → standard OpenAI API
```

**Key decisions**:
- **Don't bundle Ollama** (~3-4GB is too large). Instead, detect installed Ollama or guide user to install it.
- **Support custom GGUF models** via Ollama's Modelfile mechanism (`ollama create mymodel -f Modelfile`)
- **Auto-detect**: Probe localhost:11434 (Ollama), localhost:1234 (LM Studio), then user-configured URL
- **Context window**: Configure explicitly (Ollama defaults to 2048, our models support 8192+)

#### Phase 3: llama.cpp Server as Fallback

For users who can't or won't install Ollama, provide a lightweight fallback:
- Download pre-built `llama-server` binary from llama.cpp GitHub releases (~100-400MB)
- Detect GPU: CUDA version → pick CUDA binary; no CUDA → Vulkan binary; nothing → CPU binary
- Manage as subprocess, same OpenAI-compatible API
- Reuse existing GGUF model download pipeline

**Vulkan support** is the key differentiator here — it works across NVIDIA, AMD, and Intel GPUs without vendor-specific drivers.

#### Phase 4: Backend Selection Logic

```
User clicks "Translate" with --provider local (or GUI equivalent)
    │
    ├─ 1. Ollama running on localhost:11434?
    │     → Yes: Use OllamaBackend
    │     → Ollama installed but not running? Start it.
    │
    ├─ 2. LM Studio running on localhost:1234?
    │     → Yes: Use as OpenAI-compatible server
    │
    ├─ 3. Neither found?
    │     → Download llama-server binary (one-time, ~200MB)
    │     → Start LlamaCppServerBackend
    │
    ├─ 4. User override: --llm-backend ollama|llama-server|legacy
    │
    └─ 5. GUI: Show detected server status, offer Ollama install link
```

#### Phase 5: Deprecate llama-cpp-python

Once v1.9.0 is stable:
1. Move llama-cpp-python to `[legacy]` extra (not installed by default)
2. Remove from default requirements
3. Show deprecation warning if `--llm-backend legacy` is used
4. Full removal in v1.10.0

**Code impact**:

| Component | Lines | Change |
|-----------|-------|--------|
| New: `llm_backend.py` (protocol + factory) | +100 | New file |
| New: `backends/ollama.py` | +200 | New file |
| New: `backends/llama_server.py` | +250 | New file |
| Refactored: `backends/llama_cpp_python.py` | ~800 | From local_backend.py |
| Deleted: DLL workarounds, build utils | **-1500** | From local_backend.py, llama_build_utils.py, llama_cuda_config.py |
| Modified: `service.py`, `core.py` | ~100 | Backend selection |
| Modified: `providers.py` | +20 | Add ollama provider |
| Modified: GUI frontend | ~50 | Server status display |

**Net**: Remove ~1000 lines of fragile platform-specific code, replace with ~550 lines of clean HTTP client code.

---

### v1.9.x — Polish & Ecosystem (Follow-up)

#### numpy 2 Validation

After v1.8.8 lifts the pin and v1.9.0 removes llama-cpp-python from default deps:
- Run the test suite from `tests/research/test_numpy2_*.py`
- Test all 5 enhancement backends with numpy 2.2.x
- If all clear, consider requiring `numpy>=2.0` in v1.10.0

#### Optional: MLX Backend for Mac

For Apple Silicon users who want maximum performance:
- MLX is 2.17x faster than llama.cpp on M3 Ultra
- Could be an optional `--llm-backend mlx` for macOS
- Lower priority — Ollama with Metal is already good on Mac

#### Model Recommendations

Build intelligence into the model selection:
- Detect available VRAM → suggest appropriate model size
- 8GB+ VRAM → llama3.1:8b (best quality for translation)
- 4-6GB → llama3.2:3b (good balance)
- <4GB or CPU → phi3:mini or cloud provider suggestion

---

## User Experience Vision

### Before (v1.8.7 — Current)

```
User: Clicks "Translate" with local provider
→ WhisperJAV: Tries to load llama-cpp-python
→ DLL loading fails (CUDA version mismatch)
→ Tries to build from source (no compiler found)
→ ERROR: "Failed to load local LLM server"
→ User: Confused, gives up
```

### After (v1.9.0 — Target)

```
User: Clicks "Translate" with local provider
→ WhisperJAV: Checks localhost:11434... Ollama found!
→ Checks models... llama3.1:8b loaded, 8GB VRAM detected
→ Translation runs via standard OpenAI API
→ Done. 30 seconds for a subtitle file.

-- OR if Ollama not installed --

→ WhisperJAV: No local LLM server detected.
→ Shows friendly dialog:
  "For local translation, install Ollama (free, 1 click):
   → Download: ollama.com
   → Then run: ollama pull llama3.1:8b

   Or use cloud translation:
   → DeepSeek ($0.001/file): --provider deepseek"
```

---

## Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Ollama API breaking change | Low | Medium | Pin minimum Ollama version, version check on startup |
| Ollama not installed by user | High | Medium | Friendly guidance + llama-server fallback |
| llama-server binary variant wrong for user's GPU | Medium | Low | Vulkan as universal fallback, clear error message |
| numpy 2 breaks undiscovered transitive dep | Low | Medium | Keep `>=1.26.0` (not `>=2.0`), let pip resolve |
| ModelScope breaks with numpy 2 | Low | Low | Enhancement module has graceful fallback |
| Ollama context window default (2048) too small | High | High | Always set explicit `num_ctx` parameter |

---

## Dependency Impact

### What gets added
- Nothing to pip requirements (Ollama is external, `openai` package already installed)
- Optional: `ollama` Python client library (pure Python, minimal deps) — but `requests` works fine

### What gets removed (v1.9.0)
- `llama-cpp-python` (from default install)
- `fastapi` (was only for local LLM server)
- `uvicorn` (was only for local LLM server)
- `sse-starlette` (was only for local LLM server)
- `cmake` / build tools (no longer needed)

### What gets updated (v1.8.8)
- `numpy>=1.26.0` (remove `<2.0` upper bound)
- `scipy>=1.12.0`
- `numba>=0.59.0`
- `librosa>=0.10.2`
- `scikit-learn>=1.4.0`
- `opencv-python>=4.9.0`

---

## Implementation Priority

| Priority | Item | Release | Effort | Impact |
|----------|------|---------|--------|--------|
| **P0** | Fix LLM context overflow (#196/#212) | v1.8.8 | Medium | Critical — most-reported bug |
| **P0** | Fix MPS beam search (#198) | v1.8.8 | Low | Critical — Mac users blocked |
| **P1** | Lift numpy<2 pin | v1.8.8 | Low | High — unblocks ecosystem |
| **P1** | Add `--provider ollama` preview | v1.8.8 | Low | High — immediate relief for LLM users |
| **P2** | Backend abstraction layer | v1.9.0 | Medium | Foundation for migration |
| **P2** | Ollama backend implementation | v1.9.0 | Medium | The main migration |
| **P3** | llama.cpp server fallback | v1.9.0 | Medium | Safety net |
| **P3** | Deprecate llama-cpp-python | v1.9.0 | Low | Cleanup |
| **P4** | MLX backend for Mac | v1.9.x | Medium | Nice-to-have |
| **P4** | Require numpy>=2.0 | v1.10.0 | Low | Future cleanup |

---

## Success Metrics

| Metric | Current (v1.8.7) | Target (v1.9.0) |
|--------|-------------------|------------------|
| LLM-related GitHub issues | 5+ open (#196, #212, #132, #198-LLM, etc.) | 0 open |
| "DLL not found" / build failures | Recurring | Eliminated |
| Local translation setup steps | 0 (auto, but fails) | 1 (install Ollama) |
| Platform coverage (local LLM) | Windows+NVIDIA only (reliably) | Windows, Mac, Linux, AMD |
| numpy version | 1.26.x only | 1.26.x or 2.x (user's choice) |
| Lines of DLL/build workaround code | ~1500 | 0 |

---

## Open Questions

1. **Should WhisperJAV auto-install Ollama?** Or just detect + guide? Recommendation: detect + guide. Auto-installing a 3-4GB binary without consent is hostile UX.

2. **Custom uncensored models**: Current approach downloads a specific uncensored GGUF from HuggingFace. With Ollama, options are: (a) publish to Ollama registry, (b) create Modelfile from local GGUF, (c) use `ollama pull hf.co/user/model:Q4_K_M`. Need to decide which path.

3. **Ollama context window**: Default is 2048 tokens — too small for translation with system prompt. Must set `num_ctx: 8192` explicitly via API parameter or Modelfile. This is a must-fix before shipping.

4. **Backward compatibility**: Users who have working llama-cpp-python setups shouldn't be broken. Keep `--provider local` working via legacy backend, add `--provider ollama` as the recommended path.

---

## References

- `docs/research/llm_backend_best_practices.md` — Architecture patterns from 10+ projects
- `docs/research/llm_backend_candidates.md` — Detailed evaluation of 7 candidates
- `docs/research/numpy2_modelscope_feasibility.md` — ModelScope + numpy 2 analysis
- `docs/research/numpy2_test_suite_report.md` — Code audit + test suite
- `tests/research/test_numpy2_*.py` — Runnable numpy 2 compatibility tests
