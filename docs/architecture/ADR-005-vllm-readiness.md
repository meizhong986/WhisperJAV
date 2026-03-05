# ADR-005: vLLM Readiness — Qwen3-ASR Backend Strategy

**Created**: 2026-02-09
**Status**: INFORMATIONAL — Research & Planning
**Author**: Senior Architect
**Depends On**: ADR-004 (Dedicated QwenPipeline Architecture)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background](#2-background)
3. [Qwen3-ASR Dual Backend Architecture](#3-qwen3-asr-dual-backend-architecture)
4. [Transformers Backend (Current)](#4-transformers-backend-current)
5. [vLLM Backend (Future)](#5-vllm-backend-future)
6. [API Comparison](#6-api-comparison)
7. [Streaming Inference (vLLM Only)](#7-streaming-inference-vllm-only)
8. [vLLM Serving Mode](#8-vllm-serving-mode)
9. [Audio Limits and Constraints](#9-audio-limits-and-constraints)
10. [WhisperJAV Architectural Preparation](#10-whisperjav-architectural-preparation)
11. [Integration Pathway](#11-integration-pathway)
12. [Dependency and Installation Strategy](#12-dependency-and-installation-strategy)
13. [Risk Assessment](#13-risk-assessment)
14. [Decision](#14-decision)

---

## 1. Executive Summary

The official Qwen3-ASR Toolkit (`qwen-asr`) provides two inference backends: **HuggingFace Transformers** (current WhisperJAV default) and **vLLM** (high-throughput alternative). Both share a unified API through the `Qwen3ASRModel` class.

WhisperJAV's Qwen pipeline already has the architectural socket for vLLM integration: the **Assembly (decoupled) mode** separates text generation from forced alignment into VRAM-exclusive phases. Since the ForcedAligner is PyTorch-only, vLLM can only power the text generation phase — which is exactly what Assembly mode isolates.

This document records the technical findings from researching the Qwen3-ASR Toolkit's vLLM support and outlines the integration pathway when the time comes.

---

## 2. Background

Qwen3-ASR is a Large Audio Language Model (LALM) that performs ASR by encoding audio into the language model's embedding space. Unlike Whisper (encoder-decoder), Qwen3-ASR is a decoder-only transformer that benefits from LLM-optimized inference engines like vLLM.

The Qwen3-ASR Toolkit (`qwen-asr` package, version 0.0.6+) is maintained by the Qwen team and provides:
- `Qwen3ASRModel` — unified inference class with dual backend support
- `Qwen3ForcedAligner` — separate model for word-level timestamp alignment
- CLI tools for batch processing and serving

**Toolkit source**: `qwen_asr/` package
**Repository**: [Qwen/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)

---

## 3. Qwen3-ASR Dual Backend Architecture

The toolkit's `Qwen3ASRModel` class provides a unified interface with two construction paths:

```
Qwen3ASRModel
 ├── .from_pretrained(model_path, ...)   → Transformers backend
 └── .LLM(model_path, ...)              → vLLM backend
```

After construction, the same `.transcribe()` and `.align()` methods work identically regardless of backend. The dispatch happens internally:

```
Source: qwen_asr/inference/qwen3_asr.py

def _infer_asr(self, contexts, wavs, languages) -> List[str]:
    if self.backend == "transformers":
        return self._infer_asr_transformers(contexts, wavs, languages)
    if self.backend == "vllm":
        return self._infer_asr_vllm(contexts, wavs, languages)
    raise RuntimeError(f"Unknown backend: {self.backend}")
```

This design means WhisperJAV can switch backends with minimal code changes — the same `transcribe()` call signature works for both.

---

## 4. Transformers Backend (Current)

### Construction

```python
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",  # Optional
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
)
```

**Source**: `qwen_asr/inference/qwen3_asr.py` lines 175-224

### Inference

```python
# Internally calls model.generate(**inputs, max_new_tokens=...)
# Standard HuggingFace generate() with AutoProcessor for tokenization
text_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
decoded = self.processor.batch_decode(text_ids[:, inputs["input_ids"].shape[1]:])
```

### Characteristics

| Property | Value |
|----------|-------|
| VRAM usage | Full model + KV cache + audio encoder |
| Batch processing | Manual splitting at `max_inference_batch_size` |
| Timestamps | Via ForcedAligner (separate model) |
| Streaming | Not supported |
| Tensor parallelism | Via HuggingFace `device_map` |

### WhisperJAV Usage

WhisperJAV's `QwenASR` wrapper (`whisperjav/modules/qwen_asr.py`) uses this backend exclusively today. The wrapper adds:
- Decoupled loading (`load_model_text_only()` / `load_aligner_only()`)
- Generation safety controls (`repetition_penalty`, `max_tokens_per_audio_second`)
- VRAM lifecycle management (explicit load/unload for Assembly mode)

---

## 5. vLLM Backend (Future)

### Construction

```python
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.LLM(
    model="Qwen/Qwen3-ASR-1.7B",
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",  # Optional
    forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
    gpu_memory_utilization=0.8,
    max_inference_batch_size=-1,   # Unlimited for vLLM
    max_new_tokens=4096,
)
```

**Source**: `qwen_asr/inference/qwen3_asr.py` lines 226-288

### Model Registration

vLLM requires custom model architectures to be registered before use. The toolkit handles this automatically on import:

```python
# Source: qwen_asr/inference/qwen3_asr.py lines 49-54
try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "Qwen3ASRForConditionalGeneration",
        Qwen3ASRForConditionalGeneration,
    )
except:
    pass
```

### vLLM-Native Model Implementation

The toolkit includes a full vLLM-native model implementation:

**Source**: `qwen_asr/core/vllm_backend/qwen3_asr.py`

| Component | Class | Lines | Purpose |
|-----------|-------|-------|---------|
| Audio encoder | `Qwen3ASRAudioEncoder` | 298-531 | Mel-spectrogram → embeddings |
| Attention | `Qwen3ASRAudioAttention` | 157-227 | vLLM's `MMEncoderAttention` with TP |
| Main model | `Qwen3ASRForConditionalGeneration` | 692-875 | Full ASR model for vLLM |
| Processor | `Qwen3ASRMultiModalProcessor` | 622-684 | Audio input parsing |

This is not a simple wrapper around HuggingFace — it's a purpose-built implementation using vLLM's multimodal infrastructure (`SupportsMultiModal`, `SupportsPP`, `SupportsMRoPE`, `SupportsTranscription`).

### Inference

```python
# Internally constructs prompt dicts with multimodal data
inputs = {"prompt": prompt_str, "multi_modal_data": {"audio": [wav_tensor]}}
outputs = self.model.generate(
    [inputs],
    sampling_params=self.sampling_params,
    use_tqdm=False,
)
text = outputs[0].outputs[0].text
```

**SamplingParams** (created at construction):
```python
sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
```

- `temperature=0.0`: Deterministic greedy decoding (matches Whisper behavior)
- `max_tokens`: Dynamic per-call token budget

### Characteristics

| Property | Value |
|----------|-------|
| VRAM usage | Managed by vLLM's PagedAttention |
| Batch processing | Handled by vLLM's scheduler (unlimited batch) |
| Timestamps | Via ForcedAligner (separate model, PyTorch-only) |
| Streaming | Supported (incremental decoding) |
| Tensor parallelism | Native vLLM TP support |
| GPU memory utilization | Configurable via `gpu_memory_utilization` |

### Performance Claims

From the Qwen3-ASR README:
- Up to **2000x throughput** at concurrency 128 (0.6B model benchmark)
- Better batch scheduling than HuggingFace's manual splitting
- Specialized memory management via PagedAttention

---

## 6. API Comparison

### Construction

| Aspect | Transformers | vLLM |
|--------|-------------|------|
| Factory method | `Qwen3ASRModel.from_pretrained()` | `Qwen3ASRModel.LLM()` |
| Model loading | `AutoModel.from_pretrained()` | `vllm.LLM()` |
| Processor | `AutoProcessor.from_pretrained()` | Built into vLLM model |
| `backend` field | `"transformers"` | `"vllm"` |
| Default `max_new_tokens` | 512 | 4096 |
| Default `max_inference_batch_size` | 32 | -1 (unlimited) |
| Extra kwargs | HuggingFace kwargs | vLLM kwargs (`gpu_memory_utilization`, etc.) |

### Inference (after construction, identical API)

```python
# Both backends use the same call:
results = model.transcribe(
    audio=["path/to/audio.wav"],
    context=["optional context"],
    language=["Japanese"],
    return_time_stamps=True,
)

# Returns List[Qwen3ASRResult] with:
#   .text       — transcribed text
#   .language   — detected language
#   .words      — word-level timestamps (if return_time_stamps=True)
```

### Output Handling

| Aspect | Transformers | vLLM |
|--------|-------------|------|
| Text extraction | `processor.batch_decode()` on token IDs | `output.outputs[0].text` |
| Batch control | Manual split at `max_inference_batch_size` | vLLM scheduler handles batching |
| ForcedAligner | Same (PyTorch, separate model) | Same (PyTorch, separate model) |

### Generation Config

| Parameter | Transformers | vLLM |
|-----------|-------------|------|
| Max tokens | `max_new_tokens` in `generate()` | `SamplingParams.max_tokens` |
| Temperature | Via `GenerationConfig` | `SamplingParams.temperature` |
| Repetition penalty | Via `GenerationConfig` | `SamplingParams.repetition_penalty` |
| Top-k / Top-p | Via `GenerationConfig` | `SamplingParams.top_k` / `.top_p` |

---

## 7. Streaming Inference (vLLM Only)

Streaming is **exclusively available with the vLLM backend** and enables real-time incremental transcription. This is not currently needed by WhisperJAV (which processes pre-recorded files), but could enable future live captioning.

**Source**: `qwen_asr/inference/qwen3_asr.py` lines 584-830

### State Management

```python
@dataclass
class ASRStreamingState:
    unfixed_chunk_num: int        # Chunks before prefix prompt kicks in
    unfixed_token_num: int        # Tokens to rollback for prefix
    chunk_size_sec: float         # Chunk duration (e.g. 2.0s)
    chunk_size_samples: int       # Samples at 16kHz
    chunk_id: int                 # Current chunk counter
    buffer: np.ndarray            # Audio sample buffer
    audio_accum: np.ndarray       # All audio seen so far
    prompt_raw: str               # Base chat template prompt
    context: str                  # User-provided context string
    force_language: Optional[str] # Forced language
    language: str                 # Latest detected language
    text: str                     # Latest cumulative text
```

### Usage Pattern

```python
state = asr.init_streaming_state(
    unfixed_chunk_num=2,    # 2 chunks before stabilizing
    unfixed_token_num=5,    # 5-token rollback window
    chunk_size_sec=2.0,     # 2-second chunks
)

while has_audio:
    chunk = get_next_audio_chunk()
    asr.streaming_transcribe(chunk, state)
    print(f"[{state.language}] {state.text}")

asr.finish_streaming_transcribe(state)
print(f"Final: {state.text}")
```

### Constraints

- No ForcedAligner support (no word-level timestamps during streaming)
- Single stream only (no parallel streams in one model instance)
- Requires vLLM backend — Transformers backend raises `NotImplementedError`

---

## 8. vLLM Serving Mode

The toolkit includes an OpenAI-compatible serving endpoint.

**Source**: `qwen_asr/cli/serve.py`

### Starting the Server

```bash
qwen-asr-serve Qwen/Qwen3-ASR-1.7B \
    --gpu-memory-utilization 0.8 \
    --port 8000
```

This wraps `vllm serve` with the model registration step.

### Client API (OpenAI-compatible)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-ASR-1.7B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.wav"}}
      ]
    }]
  }'
```

### Relevance to WhisperJAV

Serving mode could enable a future **client-server architecture** where:
- The vLLM server runs persistently (no model load/unload per file)
- WhisperJAV sends audio clips via HTTP (OpenAI-compatible API)
- Multiple WhisperJAV instances share one GPU server

This is lower priority than direct vLLM integration but worth noting for multi-user or production deployments.

---

## 9. Audio Limits and Constraints

**Source**: `qwen_asr/inference/utils.py`

```python
SAMPLE_RATE = 16000
MAX_ASR_INPUT_SECONDS = 1200              # 20 minutes (text-only mode)
MAX_FORCE_ALIGN_INPUT_SECONDS = 180       # 3 minutes (with timestamps)
```

| Limit | Value | Applies To | Notes |
|-------|-------|-----------|-------|
| Max ASR input | 1200s (20 min) | Text-only generation | Both backends |
| Max aligner input | **180s** (3 min) | ForcedAligner timestamps | Hard limit; NOT 300s as some docs claim |
| Sample rate | 16000 Hz | All audio input | Resampled internally if different |
| Internal chunking | 180s | `split_audio_into_chunks()` | Applied before aligner call |

**Critical for WhisperJAV**: The ForcedAligner's **180-second hard limit** is the primary constraint driving the pipeline's scene detection and Assembly mode's 120-second max scene duration (60-second safety margin).

Both backends share the same ForcedAligner (PyTorch-only `Qwen3ForcedAligner`). Switching to vLLM for text generation does not change the aligner limits.

---

## 10. WhisperJAV Architectural Preparation

WhisperJAV's Qwen pipeline already has the structural foundation for vLLM integration through the **Assembly (decoupled) mode**.

### Assembly Mode VRAM Phases

```
Phase 5 Assembly Flow:
  Step 1: load_model_text_only()          ← vLLM replaces THIS
  Step 2: transcribe_text_only(batch)     ← vLLM replaces THIS
  Step 3: unload_model()                  ← VRAM freed
  Step 4: TextSanitizer.clean_batch()     ← CPU, no VRAM
  Step 5: load_aligner_only()             ← PyTorch ForcedAligner (stays)
  Step 6: align_standalone(batch)          ← PyTorch ForcedAligner (stays)
  Step 7: unload_model()                  ← VRAM freed
  Step 8: Reconstruct WhisperResult       ← CPU, no VRAM
```

Steps 1-3 are the **text generation phase** — this is where vLLM plugs in.
Steps 5-7 are the **alignment phase** — this stays on PyTorch/Transformers because the ForcedAligner has no vLLM implementation.

### Existing Code References

**QwenPipeline constructor** (`qwen_pipeline.py` line 60-69):
```
ASSEMBLY mode ... creates the architectural socket for future vLLM integration.
```

**QwenASR decoupled API** (`qwen_asr.py` lines 1080-1093):
```
The decoupled VRAM-exclusive phases allow ASR and forced alignment to run
as separate VRAM-exclusive phases. This allows:
  - Higher batch_size for text-only ASR (no aligner loaded)
  - Mid-pipeline sanitization between generation and alignment
  - Future vLLM integration (ASR supports vLLM, aligner does not)
```

### What Already Works

| Component | vLLM Ready? | Notes |
|-----------|------------|-------|
| Assembly mode pipeline | Yes | VRAM phases already decoupled |
| `transcribe_text_only()` | Needs adapter | Currently calls HuggingFace `.generate()` |
| `align_standalone()` | N/A | ForcedAligner is PyTorch-only, no change needed |
| TextSanitizer | Yes | CPU-only, backend-agnostic |
| Scene detection | Yes | Audio-only, backend-agnostic |
| VAD / Speech segmentation | Yes | Audio-only, backend-agnostic |
| Generation safety controls | Needs mapping | `repetition_penalty` → `SamplingParams` |
| Sentinel (collapse detection) | Yes | Operates on word dicts, backend-agnostic |

---

## 11. Integration Pathway

### Phase 1: Backend Selection (Minimal Change)

Add a `qwen_backend` parameter to `QwenPipeline` and `QwenASR`:

```python
# QwenASR.__init__()
self.backend = backend  # "transformers" (default) or "vllm"
```

In `load_model_text_only()`:
```python
if self.backend == "vllm":
    self.model = Qwen3ASRModel.LLM(
        model=self.model_id,
        gpu_memory_utilization=self.gpu_memory_utilization,
        max_new_tokens=self.max_new_tokens,
    )
else:
    self.model = Qwen3ASRModel.from_pretrained(
        self.model_id,
        dtype=self.dtype,
        device_map=self.device,
        max_new_tokens=self.max_new_tokens,
    )
```

**Impact**: Minimal. Only `load_model_text_only()` and `transcribe_text_only()` change. Alignment stays on Transformers.

### Phase 2: Generation Config Mapping

Map WhisperJAV's generation safety parameters to vLLM's `SamplingParams`:

| WhisperJAV Parameter | Transformers | vLLM SamplingParams |
|---------------------|-------------|-------------------|
| `repetition_penalty` | `GenerationConfig.repetition_penalty` | `SamplingParams.repetition_penalty` |
| `max_tokens_per_audio_second` | Dynamic `max_new_tokens` | Dynamic `SamplingParams.max_tokens` |
| Temperature | `GenerationConfig.temperature` | `SamplingParams.temperature` |

The toolkit's `Qwen3ASRModel.LLM()` already creates `SamplingParams` at construction:
```python
sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
```

WhisperJAV would need to either:
- Pass generation params through the constructor's `**kwargs`, or
- Override `self.model.sampling_params` before each call (for dynamic `max_tokens`)

### Phase 3: CLI and GUI Integration

```python
# CLI
parser.add_argument("--qwen-backend", choices=["transformers", "vllm"],
                    default="transformers")
parser.add_argument("--qwen-gpu-memory-utilization", type=float, default=0.8)

# GUI schema addition
"backend": {
    "type": "select",
    "label": "Inference Backend",
    "options": ["transformers", "vllm"],
    "default": "transformers",
}
```

### Phase 4: Coupled Modes (Optional, Lower Priority)

CONTEXT_AWARE and VAD_SLICING modes currently use coupled ASR+Aligner calls. For vLLM:
- Option A: Force Assembly mode when backend=vllm (simplest)
- Option B: Support coupled vLLM calls with separate aligner loading (complex, questionable value)

**Recommendation**: Phase 4 is Option A — when `backend="vllm"`, automatically use Assembly mode regardless of `input_mode` setting.

---

## 12. Dependency and Installation Strategy

### Current State

```toml
# WhisperJAV pyproject.toml
qwen = [
    "qwen-asr>=0.0.6",
    "whisperjav[huggingface]",
]
```

This installs `qwen-asr` without the vLLM extra. The Transformers backend works.

### Proposed Addition

```toml
# New extra for vLLM support
qwen-vllm = [
    "qwen-asr[vllm]>=0.0.6",
    "whisperjav[huggingface]",
]
```

### vLLM Version Pinning

The Qwen3-ASR toolkit pins vLLM strictly:
```toml
# qwen-asr pyproject.toml
vllm = ["vllm==0.14.0"]
```

This version pin means:
- WhisperJAV doesn't need to manage vLLM versions directly
- The `qwen-asr[vllm]` extra handles the correct version
- vLLM updates require a corresponding `qwen-asr` update

### Installation Sizes

| Extra | Additional Size | GPU Required |
|-------|----------------|-------------|
| `qwen` (current) | ~200MB (qwen-asr + transformers deps) | Yes (CUDA) |
| `qwen-vllm` (proposed) | ~800MB+ (vllm + CUDA kernels) | Yes (CUDA, Linux preferred) |

### Platform Considerations

- vLLM has strongest support on **Linux with CUDA**
- Windows support is experimental (WSL2 recommended)
- macOS is not supported by vLLM
- WhisperJAV's Windows installer would need to handle this gracefully

---

## 13. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| vLLM version conflicts with other WhisperJAV deps | MEDIUM | Separate `qwen-vllm` extra, version managed by `qwen-asr` |
| vLLM Windows compatibility | HIGH | Document Linux/WSL2 requirement; Transformers backend as fallback |
| ForcedAligner stays PyTorch-only | LOW | Assembly mode already decouples; no architectural change needed |
| Generation safety params may behave differently | MEDIUM | Test repetition_penalty and token budgets under vLLM; validate output quality |
| vLLM VRAM management conflicts with manual unload | MEDIUM | vLLM manages its own memory; may need to skip manual `torch.cuda.empty_cache()` |
| Streaming not needed yet | LOW | Don't implement until there's a use case (live captioning) |

---

## 14. Decision

### Current Decision: DEFER implementation, DOCUMENT readiness

The vLLM integration pathway is well-understood and architecturally prepared. Implementation is deferred because:

1. **No immediate throughput bottleneck** — WhisperJAV processes single files sequentially; vLLM's batch scheduling advantage is minimal for this workload.
2. **Platform risk** — vLLM's Windows support is experimental, and most WhisperJAV users are on Windows.
3. **Assembly mode maturity** — Assembly mode (the vLLM socket) is new (v1.8.8+) and needs production stabilization before adding another variable.
4. **Transformers backend works** — Current quality and speed are acceptable for the target use case.

### When to Revisit

Implement vLLM integration when any of these triggers occur:
- WhisperJAV adds **batch/queue processing** (multiple files) — vLLM's scheduler becomes valuable
- Users request **live captioning** — streaming is vLLM-only
- **Server deployment** is needed — vLLM serving mode enables multi-client architectures
- vLLM achieves **stable Windows support** — removes the platform risk
- Assembly mode is **proven stable** in production — safe to add vLLM as a variable

### What's Ready Today

- Architectural socket: Assembly mode's decoupled VRAM phases
- API knowledge: `Qwen3ASRModel.LLM()` construction and `SamplingParams` mapping
- Dependency strategy: Separate `qwen-vllm` extra with pinned version via `qwen-asr`
- Code locations: `load_model_text_only()` and `transcribe_text_only()` are the only methods that change

---

*This document is maintained as a living reference. Update when the Qwen3-ASR toolkit changes its vLLM interface or when WhisperJAV begins implementation.*
