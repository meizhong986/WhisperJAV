# ADR-003: vLLM Pipeline Architecture with Qwen3-ASR

**Status**: Draft v3 (Reviewed)
**Date**: 2026-01-31
**Author**: Architecture Team
**Branch**: dev_qwen17b
**Reviewers**: External (3 reviewers - consolidated feedback incorporated)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Qwen3-ASR Technical Deep Dive](#qwen3-asr-technical-deep-dive)
3. [Current WhisperJAV ASR Architecture](#current-whisperjav-asr-architecture)
4. [Revised Architecture: vLLM Pipeline](#revised-architecture-vllm-pipeline)
5. [Configuration Strategy](#configuration-strategy)
6. [Critical Design Contracts](#critical-design-contracts) ← NEW
7. [Integration Options (Reference)](#integration-options)
8. [Evaluation Matrix](#evaluation-matrix)
9. [Recommendations](#recommendations)
10. [Open Questions](#open-questions)
11. [References](#references)

---

## Executive Summary

### Context

WhisperJAV is creating a **new vLLM Pipeline** (`--mode vllm`) as a framework-specific pipeline optimized for vLLM-based ASR engines. **Qwen3-ASR** will be the first ASR engine in this pipeline, with the architecture designed to accommodate future vLLM-based ASR models.

**Key Design Decision**: This is a **framework-first** approach, not a model-first approach. The pipeline is named after the inference framework (vLLM), and specific ASR engines (Qwen3-ASR, future models) are pluggable components within it.

### Qwen3-ASR Advantages

- **State-of-the-art accuracy** among open-source ASR models
- **22-46% better Japanese WER** than Whisper-Large-v3
- **2x better singing voice** recognition (5.98% vs 13.58% WER)
- **52 language/dialect support** including Japanese
- **vLLM backend** for extremely high throughput (2000x at 128 concurrency)
- **Built-in forced alignment** for word/character-level timestamps

### Decision Drivers

1. **Future-Proofing**: vLLM is becoming the standard for LLM/multimodal inference
2. **Code Redundancy Acceptable**: Copy Fidelity pipeline as proven reference
3. **Framework Focus**: Pipeline named for vLLM, not specific ASR model
4. **Pragmatic Evolution**: Start simple, iterate toward v4 config system

### Scope

This ADR covers:
- Technical analysis of Qwen3-ASR capabilities
- vLLM Pipeline architecture and design
- Configuration strategy options
- Implementation path

---

## Qwen3-ASR Technical Deep Dive

### Model Family Overview

| Model | Parameters | Features | Use Case |
|-------|-----------|----------|----------|
| **Qwen3-ASR-1.7B** | ~2B | Full accuracy, all features | Quality-focused scenarios |
| **Qwen3-ASR-0.6B** | ~0.9B | 2000x throughput, streaming | High-throughput, real-time |
| **Qwen3-ForcedAligner-0.6B** | ~0.9B | NAR timestamp prediction | Post-hoc alignment |

### Language Support

**30 Core Languages**: Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Italian, Russian, Arabic, Vietnamese, Thai, Indonesian, Hindi, Turkish, Dutch, Polish, Swedish, Danish, Finnish, Czech, Greek, Hungarian, Romanian, Persian, Filipino, Malay, Macedonian, Cantonese

**22 Chinese Dialects**: Regional varieties including Sichuan, Shanghai, Cantonese, etc.

**English Accents**: UK, US, Australian, Indian, etc.

### Benchmark Performance (vs. Whisper)

#### Japanese ASR (Key for WhisperJAV)

| Dataset | Qwen3-ASR-1.7B | Whisper-Large-v3 | Improvement |
|---------|----------------|------------------|-------------|
| CommonVoice-ja | ~7% WER | ~9% WER | ~22% better |
| Fleurs-ja | 2.41% WER | 4.5% WER | ~46% better |

#### Chinese ASR

| Dataset | Qwen3-ASR-1.7B | Whisper-Large-v3 |
|---------|----------------|------------------|
| WenetSpeech (net) | **4.97%** | 9.86% |
| WenetSpeech (meeting) | **5.88%** | 19.11% |
| AISHELL-2 | **2.71%** | 5.06% |

#### English ASR

| Dataset | Qwen3-ASR-1.7B | Whisper-Large-v3 |
|---------|----------------|------------------|
| Librispeech (clean) | **1.63%** | 1.51% |
| Librispeech (other) | **3.38%** | 3.97% |
| GigaSpeech | **8.45%** | 9.76% |

#### Singing Voice (Critical for JAV BGM)

| Dataset | Qwen3-ASR-1.7B | Whisper-Large-v3 |
|---------|----------------|------------------|
| M4Singer | **5.98%** | 13.58% |
| MIR-1k-vocal | **6.25%** | 11.71% |

### Architecture Fundamentals

```
┌─────────────────────────────────────────────────────────────────┐
│                     Qwen3-ASR Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐   │
│   │   Audio      │───>│  Qwen3-Omni     │───>│   Text       │   │
│   │   Encoder    │    │  Foundation     │    │   Decoder    │   │
│   │              │    │  (Frozen)       │    │   (LLM)      │   │
│   └──────────────┘    └─────────────────┘    └──────────────┘   │
│         │                                            │          │
│         │            ┌─────────────────┐             │          │
│         └───────────>│  ForcedAligner  │<────────────┘          │
│                      │  (NAR, 0.6B)    │                        │
│                      │  Timestamps     │                        │
│                      └─────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Characteristics**:
- Built on Qwen3-Omni multimodal foundation
- Unified streaming/offline inference
- Non-autoregressive forced aligner for timestamps
- Audio encoder handles variable sample rates

### API Surface

#### Installation

```bash
# Minimal
pip install -U qwen-asr

# With vLLM (recommended for production)
pip install -U qwen-asr[vllm]

# Optional: FlashAttention 2
pip install -U flash-attn --no-build-isolation
```

#### Transformers Backend

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
)

results = model.transcribe(
    audio="path/to/audio.wav",  # or URL, base64, (np.ndarray, sr)
    language=None,               # Auto-detect or force
    return_time_stamps=True,
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
)

# Result structure
print(results[0].language)     # "Japanese"
print(results[0].text)         # Transcribed text
print(results[0].time_stamps)  # List of {text, start_time, end_time}
```

#### vLLM Backend (High Throughput)

```python
from qwen_asr import Qwen3ASRModel

if __name__ == '__main__':
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.7,
        max_inference_batch_size=128,
        max_new_tokens=4096,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
    )

    # Batch transcription
    results = model.transcribe(
        audio=["audio1.wav", "audio2.wav"],
        language=["Japanese", "Japanese"],
        return_time_stamps=True,
    )
```

#### Streaming (vLLM only)

```python
# Real-time PCM chunk processing
# Does NOT support batch inference or timestamps
# See: example_qwen3_asr_vllm_streaming.py
```

#### Direct Forced Alignment

```python
from qwen_asr import Qwen3ForcedAligner

aligner = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

# Align known text to audio
results = aligner.align(
    audio="audio.wav",
    text="Known transcript text here.",
    language="Japanese",
)

# Character-level timestamps
for segment in results[0]:
    print(f"{segment.text}: {segment.start_time}s - {segment.end_time}s")
```

### Input/Output Specifications

**Audio Input Formats**:
- Local file paths (`"path/to/audio.wav"`)
- HTTP/HTTPS URLs
- Base64-encoded audio data
- NumPy arrays: `(np.ndarray, sample_rate)` tuples
- Sample rate: Auto-resampled to 16kHz internally

**Output Structure**:
```python
@dataclass
class ASRResult:
    language: str           # Detected/specified language
    text: str              # Full transcription
    time_stamps: List[TimeStamp]  # Optional, when enabled

@dataclass
class TimeStamp:
    text: str              # Word or character
    start_time: float      # Seconds
    end_time: float        # Seconds
```

### VRAM Requirements

| Model | FP16 | BF16 | INT8 (estimate) |
|-------|------|------|-----------------|
| Qwen3-ASR-1.7B | ~4GB | ~4GB | ~2GB |
| Qwen3-ASR-0.6B | ~1.5GB | ~1.5GB | ~0.8GB |
| ForcedAligner-0.6B | ~1.5GB | ~1.5GB | ~0.8GB |

**Combined (ASR + Aligner)**: ~5-8GB GPU memory

### Inference Speed (vLLM)

| Model | TTFT | RTF | Throughput @128 concurrency |
|-------|------|-----|----------------------------|
| Qwen3-ASR-0.6B | 92ms | 0.064 | 2000x real-time |
| Qwen3-ASR-1.7B | ~150ms | ~0.1 | ~1200x real-time |

---

## Current WhisperJAV ASR Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     WhisperJAV Processing Flow                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────┐   ┌─────────────┐   ┌───────────┐   ┌─────────────┐   │
│  │  Media  │──>│   Audio     │──>│   Scene   │──>│   Speech    │   │
│  │  Input  │   │   Extract   │   │  Detect   │   │  Enhance    │   │
│  └─────────┘   │   (48kHz)   │   │  (Auditok)│   │  (Optional) │   │
│                └─────────────┘   └───────────┘   └─────────────┘   │
│                                        │                │           │
│                                        v                v           │
│                              ┌─────────────────────────────┐        │
│                              │  Per-Scene Processing Loop  │        │
│                              │  ┌───────────┐ ┌──────────┐ │        │
│                              │  │    VAD    │─│   ASR    │ │        │
│                              │  │ (Silero)  │ │(Whisper) │ │        │
│                              │  └───────────┘ └──────────┘ │        │
│                              └─────────────────────────────┘        │
│                                        │                            │
│                                        v                            │
│                              ┌─────────────────┐                    │
│                              │   SRT Stitch    │                    │
│                              │  (Time Offset)  │                    │
│                              └─────────────────┘                    │
│                                        │                            │
│                                        v                            │
│                              ┌─────────────────┐                    │
│                              │ Post-Processing │                    │
│                              │ (JP Sanitize)   │                    │
│                              └─────────────────┘                    │
│                                        │                            │
│                                        v                            │
│                              ┌─────────────────┐                    │
│                              │   Final SRT     │                    │
│                              └─────────────────┘                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### ASR Backend Interface Contract

```python
class ASRBackend(Protocol):
    """Interface contract for all ASR implementations."""

    def __init__(self, model_config: Dict, params: Dict, task: str, **kwargs):
        """
        Args:
            model_config: {"model_name": str, "device": str, "compute_type": str}
            params: {
                "decoder": {language, task, beam_size, ...},
                "vad": {threshold, min_speech_duration_ms, ...},
                "provider": {temperature, no_speech_threshold, ...}
            }
            task: "transcribe" | "translate"
        """
        ...

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Dict:
        """
        Returns: {
            "segments": [{"start_sec": float, "end_sec": float, "text": str, ...}],
            "text": str,
            "language": str
        }
        """
        ...

    def transcribe_to_srt(self, audio_path: Union[str, Path],
                          output_srt_path: Union[str, Path], **kwargs) -> Path:
        """Transcribe and save as SRT file."""
        ...

    def cleanup(self) -> None:
        """Release GPU/model resources."""
        ...

    def reset_statistics(self) -> None:
        """Reset per-file counters for batch processing."""
        ...
```

### Existing ASR Implementations

| Implementation | Backend | Use Case | Features |
|---------------|---------|----------|----------|
| `StableTSASR` | stable_whisper | Fast/Faster pipelines | VAD integration, JP regrouping |
| `WhisperProASR` | openai-whisper | Legacy | External VAD |
| `FasterWhisperProASR` | faster-whisper | Balanced pipeline | External segmenter, crash tracing |

### Configuration Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                   Config Resolution Order                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Component Presets (config/components/asr/*.py)   [HIGHEST]  │
│     └─ Pydantic models with sensitivity presets                  │
│                                                                  │
│  2. TranscriptionTuner (config/transcription_tuner.py)          │
│     └─ Pipeline + sensitivity → resolved config                  │
│                                                                  │
│  3. asr_config.json (config/asr_config.json)                    │
│     └─ Legacy JSON values                                        │
│                                                                  │
│  4. Module Defaults (modules/*.py)                    [LOWEST]  │
│     └─ Hardcoded fallbacks                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Timestamp Generation Flow

1. **ASR transcription** → Segment start/end times (seconds)
2. **Scene detection** → Assigns time offset per scene
3. **Per-scene transcription** → Relative timestamps (0-based)
4. **SRT Stitching** → Adds scene offset: `new_start = sub.start + offset`
5. **Output** → `HH:MM:SS,mmm` format

---

## Revised Architecture: vLLM Pipeline

### Design Philosophy

Based on architectural review, the chosen approach is to create a **dedicated vLLM Pipeline** that:

1. **Framework-First Naming**: `--mode vllm` (not `--mode qwen`)
2. **Code Redundancy Acceptable**: Copy Fidelity pipeline as a proven reference
3. **First ASR Engine**: Qwen3-ASR (with 0.6B and 1.7B variants)
4. **Future Extensibility**: Architecture supports additional vLLM-based ASR engines

### Pipeline Family Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WhisperJAV Pipeline Family (Updated)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                          BasePipeline                               │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│       │              │              │              │                         │
│       v              v              v              v                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐            │
│  │ Faster   │  │ Balanced │  │ Fidelity │  │   vLLM Pipeline  │  <── NEW   │
│  │ Pipeline │  │ Pipeline │  │ Pipeline │  │   (--mode vllm)  │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘            │
│       │              │              │              │                         │
│       v              v              v              v                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐            │
│  │StableTSAS│  │FasterWhis│  │WhisperPro│  │   vLLM ASR       │            │
│  │(stable-ts)│  │perProASR │  │   ASR    │  │   Engine         │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘            │
│                                                    │                         │
│                                          ┌────────┴────────┐                │
│                                          │  ASR Engines:   │                │
│                                          │  • Qwen3-ASR    │ <── First     │
│                                          │  • (Future...)  │                │
│                                          └─────────────────┘                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### vLLM Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         vLLM Pipeline (--mode vllm)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CLI: whisperjav video.mp4 --mode vllm [--model qwen-1.7b|qwen-0.6b]        │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        VLLMPipeline(BasePipeline)                      │  │
│  │                    (Copied from FidelityPipeline)                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       │  1. Audio Extraction (48kHz)                                        │
│       v                                                                      │
│  ┌───────────────┐                                                          │
│  │AudioExtractor │ ─────────────────────────────────────────┐               │
│  └───────────────┘                                           │               │
│       │                                                      │               │
│       │  2. Scene Detection (Auditok + VAD)                  │               │
│       v                                                      │               │
│  ┌───────────────────┐                                       │               │
│  │DynamicSceneDetect │ ───────────────────────────────┐      │               │
│  └───────────────────┘                                 │      │               │
│       │                                                │      │               │
│       │  3. Speech Enhancement (Optional)              │      │               │
│       v                                                │      │               │
│  ┌───────────────────┐                                 │      │               │
│  │ SpeechEnhancer    │ (ClearVoice/BS-RoFormer/None)  │      │               │
│  │ [Exclusive VRAM]  │                                 │      │               │
│  └───────────────────┘                                 │      │               │
│       │                                                │      │               │
│       │  ↓ VRAM Handoff (Destroy Enhancer)            │      │               │
│       │                                                │      │               │
│       │  4. vLLM ASR Transcription                     │      │               │
│       v                                                │      │               │
│  ┌─────────────────────────────────────────────────┐   │      │               │
│  │              vLLMASREngine (Abstract)           │   │      │               │
│  │  ┌─────────────────────────────────────────┐   │   │      │               │
│  │  │         Qwen3ASR (Concrete)             │   │   │      │               │
│  │  │  • vLLM.LLM() or qwen_asr.LLM()        │   │   │      │               │
│  │  │  • ForcedAligner for timestamps         │   │   │      │               │
│  │  │  • Batch scene transcription            │   │   │      │               │
│  │  │  • Models: qwen-1.7b (default), qwen-0.6b│  │   │      │               │
│  │  └─────────────────────────────────────────┘   │   │      │               │
│  │  ┌─────────────────────────────────────────┐   │   │      │               │
│  │  │       FutureASR (Placeholder)           │   │   │      │               │
│  │  │  • Next vLLM-based ASR model            │   │   │      │               │
│  │  └─────────────────────────────────────────┘   │   │      │               │
│  └─────────────────────────────────────────────────┘   │      │               │
│       │                                                │      │               │
│       │  5. SRT Stitching (Scene offsets)              │      │               │
│       v                                                v      v               │
│  ┌───────────────┐     ┌─────────────────────────────────────┐              │
│  │  SRTStitcher  │────>│           Final SRT                 │              │
│  └───────────────┘     └─────────────────────────────────────┘              │
│       │                                                                      │
│       │  6. Post-Processing (Japanese sanitization)                         │
│       v                                                                      │
│  ┌───────────────────┐                                                      │
│  │SRTPostProcessor   │ ─────> output/{basename}.ja.whisperjav.srt          │
│  └───────────────────┘                                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### File Structure

```
whisperjav/
├── pipelines/
│   ├── base_pipeline.py
│   ├── faster_pipeline.py
│   ├── fast_pipeline.py
│   ├── balanced_pipeline.py
│   ├── fidelity_pipeline.py        # Reference for vLLM pipeline
│   └── vllm_pipeline.py            # NEW - Copy from fidelity_pipeline.py
│
├── modules/
│   ├── asr/                        # NEW - ASR engine directory
│   │   ├── __init__.py            # Engine factory
│   │   ├── base.py                # VLLMASREngine protocol/base class
│   │   └── qwen.py                # Qwen3ASR implementation
│   ├── stable_ts_asr.py           # Existing
│   ├── faster_whisper_pro_asr.py  # Existing
│   └── whisper_pro_asr.py         # Existing
│
├── config/
│   ├── v4/                        # Future home for vLLM ecosystem
│   │   └── ecosystems/
│   │       └── vllm/              # NEW (Phase 2)
│   │           ├── ecosystem.yaml
│   │           └── models/
│   │               ├── qwen3-asr-1.7b.yaml
│   │               └── qwen3-asr-0.6b.yaml
│   └── ...
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pipeline Name | `vllm_pipeline.py` | Framework-first, not model-specific |
| CLI Flag | `--mode vllm` | Aligns with existing `--mode balanced`, etc. |
| Model Selection | `--model qwen-1.7b` | Extensible for future vLLM ASR engines |
| Default Model | `qwen-1.7b` | Quality-focused, fits 10-15GB VRAM |
| Base Reference | Copy FidelityPipeline | Proven VRAM management, scene handling |
| ASR Interface | `VLLMASREngine` base class | Common contract for all vLLM ASR engines |

---

## Configuration Strategy

### Option CS1: Start with v3-Style Config (Recommended for Phase 1)

Copy the FidelityPipeline's configuration approach using `resolved_config` dictionary structure.

```python
# vllm_pipeline.py
class VLLMPipeline(BasePipeline):
    def __init__(self, ..., resolved_config: Dict, ...):
        # V3 STRUCTURED CONFIG UNPACKING (same as FidelityPipeline)
        model_cfg = resolved_config["model"]
        params = resolved_config["params"]
        features = resolved_config["features"]
        task = resolved_config["task"]

        # vLLM-specific config
        vllm_opts = params.get("vllm", {})
        self.gpu_memory_utilization = vllm_opts.get("gpu_memory_utilization", 0.7)

        # ASR engine selection
        asr_engine = model_cfg.get("asr_engine", "qwen")
        asr_model = model_cfg.get("model_name", "Qwen/Qwen3-ASR-1.7B")

        # Store config for lazy ASR creation
        self._asr_config = {
            'engine': asr_engine,
            'model': asr_model,
            'params': params,
            'task': task,
        }
```

**Pros**:
- Fastest to implement
- Known working pattern from Fidelity
- Easy to iterate and debug

**Cons**:
- More code changes when adding new engines
- Not aligned with v4 YAML system long-term

### Option CS2: Design for v4 Compatibility (Phase 2)

Structure the config to be forward-compatible with the v4 YAML ecosystem.

```yaml
# config/v4/ecosystems/vllm/ecosystem.yaml
schemaVersion: v1
kind: Ecosystem

metadata:
  name: vllm
  displayName: "vLLM ASR Engines"
  description: "High-performance ASR via vLLM inference"

spec:
  provider: VLLMASRProvider
  commonDefaults:
    vllm.gpu_memory_utilization: 0.7
    vllm.max_inference_batch_size: 32
    aligner.enabled: true
```

```yaml
# config/v4/ecosystems/vllm/models/qwen3-asr-1.7b.yaml
schemaVersion: v1
kind: Model

metadata:
  name: qwen3-asr-1.7b
  ecosystem: vllm
  displayName: "Qwen3-ASR 1.7B"
  description: "High-accuracy Japanese ASR with singing voice support"
  tags: [japanese, singing, high-quality]

spec:
  model.id: "Qwen/Qwen3-ASR-1.7B"
  model.dtype: bfloat16
  aligner.id: "Qwen/Qwen3-ForcedAligner-0.6B"
  vllm.gpu_memory_utilization: 0.7
  decode.max_new_tokens: 4096
  decode.temperature: 0.01

presets:
  conservative:
    decode.temperature: 0.0
    vllm.gpu_memory_utilization: 0.6
  balanced: {}
  aggressive:
    decode.temperature: 0.05
    vllm.gpu_memory_utilization: 0.8

gui:
  model.dtype:
    widget: dropdown
    options: [bfloat16, float16]
  vllm.gpu_memory_utilization:
    widget: slider
    min: 0.3
    max: 0.95
    step: 0.05
```

**Pros**:
- Aligned with WhisperJAV's stated direction
- Adding new vLLM ASR engines = just add YAML file
- GUI auto-generation
- User-patchable without code changes

**Cons**:
- More upfront work
- Need to implement VLLMASRProvider in v4 system

### Recommended Configuration Path

**Phase 1 (Initial Implementation)**:
- Use v3-style `resolved_config` dictionary
- Copy FidelityPipeline's config unpacking pattern
- Quick iteration, known stability

**Phase 2 (After vLLM Pipeline Stable)**:
- Create `ecosystems/vllm/` in v4 system
- Migrate to YAML-driven configuration
- Add GUI integration

This hybrid approach follows the pragmatic principle: **"Start simple, iterate toward the ideal."**

---

## Critical Design Contracts

Based on external architecture review, the following contracts are **mandatory** for production stability. These address gaps identified by three independent reviewers.

### Contract 1: vLLM Batching Policy (REQUIRED)

vLLM batching is **request-driven**, not file-driven. Without explicit constraints, you will get:
- TTFT explosion
- Memory fragmentation
- Non-deterministic throughput

**Hard Batching Rules for VLLMPipeline:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    vLLM BATCHING CONTRACT                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Max audio per request:      30 seconds (configurable)           │
│  2. Max total batch duration:   120 seconds                          │
│  3. Duration variance rule:     Never mix scenes >2x apart          │
│  4. Long scene fallback:        Single-request mode for >60s        │
│  5. Default batch size:         32 scenes (configurable)            │
│                                                                      │
│  Config location: VLLMPipeline, NOT Qwen3ASR                        │
│  User override:   --vllm-batch-size, --vllm-max-scene-length        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation Pattern:**

```python
# vllm_pipeline.py
class VLLMPipeline(BasePipeline):
    # Batching constraints
    MAX_SCENE_DURATION = 30.0    # seconds
    MAX_BATCH_DURATION = 120.0   # seconds
    MAX_BATCH_SIZE = 32          # scenes
    DURATION_VARIANCE_RATIO = 2.0  # max ratio between scene lengths

    def _create_scene_batches(self, scenes: List[Scene]) -> List[List[Scene]]:
        """Group scenes into batches respecting constraints."""
        batches = []
        current_batch = []
        current_duration = 0.0

        for scene in sorted(scenes, key=lambda s: s.duration):
            # Long scene: process alone
            if scene.duration > self.MAX_SCENE_DURATION:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_duration = 0.0
                batches.append([scene])  # Single-scene batch
                continue

            # Check batch constraints
            if (current_duration + scene.duration > self.MAX_BATCH_DURATION or
                len(current_batch) >= self.MAX_BATCH_SIZE or
                (current_batch and scene.duration / current_batch[0].duration > self.DURATION_VARIANCE_RATIO)):
                batches.append(current_batch)
                current_batch = []
                current_duration = 0.0

            current_batch.append(scene)
            current_duration += scene.duration

        if current_batch:
            batches.append(current_batch)

        return batches
```

**Rationale**: This alone will prevent ~70% of future performance bugs (per reviewer feedback).

---

### Contract 2: ForcedAligner Lifecycle (REQUIRED)

The ForcedAligner is **NOT always needed**. It must be:
- **Lazy-loaded**: Only instantiated when timestamps are requested
- **Optional**: User can disable for speed runs
- **Detachable**: Can free VRAM independently of ASR model

**Lifecycle Pattern:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                 FORCEDALIGNER LIFECYCLE CONTRACT                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ASR Model (long-lived)         ForcedAligner (lazy, optional)      │
│  ┌─────────────────────┐        ┌─────────────────────────────┐     │
│  │   Qwen3ASR          │        │   Created ONLY when:        │     │
│  │   - Loaded once     │        │   1. return_time_stamps=True│     │
│  │   - Lives for batch │        │   2. User didn't --no-align │     │
│  │   - Text output     │        │   3. After text sanitization│     │
│  └─────────────────────┘        └─────────────────────────────┘     │
│           │                                  │                       │
│           v                                  v                       │
│   transcribe() → text          align(text, audio) → timestamps      │
│   (coarse or no timing)        (fine word/char timing)              │
│                                                                      │
│  Benefits:                                                           │
│  - Skip aligner for speed runs (--no-timestamps)                    │
│  - Run alignment AFTER text cleanup                                  │
│  - Free aligner VRAM aggressively (~1.5GB saved)                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation Pattern:**

```python
# modules/asr/qwen.py
class Qwen3ASR:
    def __init__(self, model_config: Dict, params: Dict, task: str):
        self._aligner = None  # Lazy - NOT loaded in __init__
        self._aligner_config = params.get("aligner", {})
        self._aligner_enabled = self._aligner_config.get("enabled", True)
        # ... load ASR model only

    def transcribe(self, audio_path, **kwargs) -> Dict:
        """Transcribe audio → text only (no timestamps)."""
        results = self.model.transcribe(audio=str(audio_path), ...)
        return {"text": results[0].text, "language": results[0].language}

    def align(self, audio_path, text: str, **kwargs) -> List[Dict]:
        """Align known text → word/char timestamps (lazy load aligner)."""
        if not self._aligner_enabled:
            return []  # Skip alignment

        if self._aligner is None:
            self._load_aligner()  # Lazy instantiation

        results = self._aligner.align(audio=str(audio_path), text=text, ...)
        return self._format_timestamps(results)

    def unload_aligner(self):
        """Explicitly free aligner VRAM."""
        if self._aligner is not None:
            del self._aligner
            self._aligner = None
            torch.cuda.empty_cache()
```

**CLI Flags:**

```bash
# Disable timestamps for speed (saves ~1.5GB VRAM, faster)
whisperjav video.mp4 --mode vllm --no-timestamps

# Force timestamps (default)
whisperjav video.mp4 --mode vllm --timestamps
```

---

### Contract 3: TimestampAdapter Layer (REQUIRED)

Qwen3-ASR produces **word/character-level timestamps**. WhisperJAV's existing post-processing expects **segment-level timestamps**. Without an adapter, you will get:
- Broken alignment after text cleanup
- Subtitle drift on long content
- Failed Japanese character span handling

**TimestampAdapter Responsibilities:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                 TIMESTAMP ADAPTER CONTRACT                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Qwen3 Output              TimestampAdapter           WhisperJAV    │
│  ┌─────────────┐           ┌─────────────┐           ┌───────────┐  │
│  │ Word/Char   │ ────────> │ Adapter     │ ────────> │ Segment   │  │
│  │ Timestamps  │           │ Layer       │           │ Timestamps│  │
│  └─────────────┘           └─────────────┘           └───────────┘  │
│                                   │                                  │
│  Responsibilities:                v                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 1. Word → Subtitle line grouping                              │  │
│  │    - Group words into natural subtitle lines                   │  │
│  │    - Respect max chars per line (~40 for Japanese)            │  │
│  │                                                                │  │
│  │ 2. Japanese character span merging                            │  │
│  │    - Handle particles (は, が, を) that bind to words        │  │
│  │    - Merge compound words (日本語 should be one span)         │  │
│  │                                                                │  │
│  │ 3. Preserve end times during text cleanup                     │  │
│  │    - After sanitizer removes hallucinations                   │  │
│  │    - Realign text to preserved end_time anchor                │  │
│  │                                                                │  │
│  │ 4. Fallback when ForcedAligner fails                          │  │
│  │    - Use coarse ASR timestamps if aligner errors              │  │
│  │    - Proportional distribution based on text length           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation Pattern:**

```python
# modules/asr/timestamp_adapter.py
class TimestampAdapter:
    """Bridge between Qwen3 word timestamps and WhisperJAV segments."""

    MAX_CHARS_PER_LINE = 40  # Japanese subtitle standard

    def word_to_segment(self, word_timestamps: List[Dict]) -> List[Dict]:
        """Convert word-level → segment-level timestamps."""
        segments = []
        current_segment = {"text": "", "start_sec": None, "end_sec": None}

        for word in word_timestamps:
            # Start new segment if needed
            if current_segment["start_sec"] is None:
                current_segment["start_sec"] = word["start_time"]

            # Check if adding word exceeds line limit
            if len(current_segment["text"]) + len(word["text"]) > self.MAX_CHARS_PER_LINE:
                current_segment["end_sec"] = word["start_time"]
                segments.append(current_segment)
                current_segment = {
                    "text": word["text"],
                    "start_sec": word["start_time"],
                    "end_sec": None
                }
            else:
                current_segment["text"] += word["text"]
                current_segment["end_sec"] = word["end_time"]

        if current_segment["text"]:
            segments.append(current_segment)

        return segments

    def preserve_timing_after_cleanup(self, original_segments: List[Dict],
                                       cleaned_text: str) -> List[Dict]:
        """Realign cleaned text to original timing anchors."""
        # Preserve end_time as anchor, redistribute text
        ...

    def fallback_proportional(self, text: str, duration: float) -> List[Dict]:
        """Proportional timestamp distribution when aligner fails."""
        ...
```

**Reviewer Quote**: "If you skip this, debugging subtitle drift will be hell."

---

### Contract 4: Error Recovery Strategy (RECOMMENDED)

vLLM can have initialization issues. Implement retry with fallback.

```python
# modules/asr/qwen.py
class Qwen3ASR:
    MAX_RETRIES = 3

    def _init_vllm(self, model_config, params):
        """Initialize vLLM backend with retry logic."""
        for attempt in range(self.MAX_RETRIES):
            try:
                self.model = Qwen3ASRModel.LLM(
                    model=self.model_name,
                    gpu_memory_utilization=self.gpu_util,
                    ...
                )
                self._backend = "vllm"
                return  # Success

            except (RuntimeError, torch.cuda.CudaError) as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"vLLM init failed (attempt {attempt+1}), retrying...")
                    torch.cuda.empty_cache()
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                else:
                    logger.warning("vLLM failed after retries, falling back to transformers")
                    self._init_transformers(model_config, params)
```

---

### Contract 5: Quantization Support (Phase 2)

For users with limited VRAM (8GB cards), support INT8 quantization:

```yaml
# Future config option
vllm:
  quantization: "int8"  # or "fp16", "bf16"
  gpu_memory_utilization: 0.85  # Can go higher with quantization
```

| Model | FP16 | INT8 (est.) | Target GPU |
|-------|------|-------------|------------|
| Qwen3-ASR-1.7B | ~4GB | ~2GB | RTX 2060/3060 (8GB) |
| Qwen3-ASR-0.6B | ~1.5GB | ~0.8GB | GTX 1660 (6GB) |

**Implementation**: Defer to Phase 2 after core pipeline is stable.

---

### TranscriptionTuner Integration

The existing `TranscriptionTuner` needs a new entry for `--mode vllm`:

```python
# config/transcription_tuner.py (conceptual update)
PIPELINE_CONFIGS = {
    'faster': {...},
    'balanced': {...},
    'fidelity': {...},
    'vllm': {                           # NEW
        'model': {
            'model_name': 'Qwen/Qwen3-ASR-1.7B',  # Default
            'device': 'cuda',
            'compute_type': 'bfloat16',
            'asr_engine': 'qwen',
        },
        'params': {
            'decoder': {'language': 'ja', 'task': 'transcribe'},
            'vllm': {'gpu_memory_utilization': 0.7},
            'aligner': {'enabled': True},
        },
        'features': {
            'scene_detection': {...},
            'post_processing': {...},
        },
    },
}
```

---

## Integration Options

### Option A: Dedicated Qwen Pipeline (New Pipeline Class)

**Architecture**: Create `QwenPipeline` as a new pipeline class alongside existing pipelines.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Option A: Dedicated Qwen Pipeline                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      BasePipeline                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│           │                    │                    │                │
│           v                    v                    v                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │FasterPipeline│    │BalancedPipe. │    │ QwenPipeline │ <─ NEW   │
│  │(faster-whsp) │    │(faster-whsp) │    │ (qwen-asr)   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                  │                   │
│                                                  v                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                        QwenPipeline                           │   │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────────┐   ┌──────────┐  │   │
│  │  │  Audio  │──>│ Scene   │──>│   Qwen3ASR  │──>│ForcedAln.│  │   │
│  │  │ Extract │   │ Detect  │   │  (vLLM/TF)  │   │(Optional)│  │   │
│  │  └─────────┘   └─────────┘   └─────────────┘   └──────────┘  │   │
│  │                                     │                         │   │
│  │                                     v                         │   │
│  │                          ┌─────────────────┐                  │   │
│  │                          │ SRT Generation  │                  │   │
│  │                          │ (Native stamps) │                  │   │
│  │                          └─────────────────┘                  │   │
│  │                                     │                         │   │
│  │                                     v                         │   │
│  │                          ┌─────────────────┐                  │   │
│  │                          │   Post-Process  │                  │   │
│  │                          │  (JP Sanitize)  │                  │   │
│  │                          └─────────────────┘                  │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**File Structure**:
```
whisperjav/
├── pipelines/
│   ├── base_pipeline.py
│   ├── faster_pipeline.py
│   ├── balanced_pipeline.py
│   └── qwen_pipeline.py          # NEW
├── modules/
│   ├── stable_ts_asr.py
│   ├── faster_whisper_pro_asr.py
│   └── qwen_asr.py               # NEW
├── config/
│   └── components/
│       └── asr/
│           ├── faster_whisper.py
│           └── qwen.py           # NEW
```

**Pros**:
- Clean separation of concerns
- No risk of breaking existing pipelines
- Can optimize for Qwen-specific features (batch scenes, native timestamps)
- Easy to A/B test against existing pipelines
- Simple mental model for users: `--mode qwen`

**Cons**:
- Code duplication (scene detection, stitching logic)
- Two separate upgrade/maintenance paths
- Configuration divergence risk
- Does not benefit from shared improvements automatically

**Implementation Effort**: Medium-High (2-3 weeks)

---

### Option B: Pluggable ASR Backend (New Backend in Existing Pipelines)

**Architecture**: Add `Qwen3ASR` as a new backend that existing pipelines can use.

```
┌─────────────────────────────────────────────────────────────────────┐
│               Option B: Pluggable ASR Backend                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     BalancedPipeline                          │   │
│  │  ┌─────────┐   ┌─────────┐   ┌────────────────────────────┐  │   │
│  │  │  Audio  │──>│ Scene   │──>│      ASR Backend Switch    │  │   │
│  │  │ Extract │   │ Detect  │   │   ┌────────────────────┐   │  │   │
│  │  └─────────┘   └─────────┘   │   │ FasterWhisperPro   │   │  │   │
│  │                               │   │     (default)      │   │  │   │
│  │                               │   └────────────────────┘   │  │   │
│  │                               │   ┌────────────────────┐   │  │   │
│  │                               │   │     Qwen3ASR       │   │  │   │
│  │                               │   │  (--backend qwen)  │   │  │   │
│  │                               │   └────────────────────┘   │  │   │
│  │                               │   ┌────────────────────┐   │  │   │
│  │                               │   │    StableTSASR     │   │  │   │
│  │                               │   │  (--backend stable)│   │  │   │
│  │                               │   └────────────────────┘   │  │   │
│  │                               └────────────────────────────┘  │   │
│  │                                            │                  │   │
│  │                                            v                  │   │
│  │                               ┌────────────────────┐          │   │
│  │                               │   Unified Result   │          │   │
│  │                               │   (Same format)    │          │   │
│  │                               └────────────────────┘          │   │
│  │                                            │                  │   │
│  │                                            v                  │   │
│  │                               ┌────────────────────┐          │   │
│  │                               │  SRT Stitch + PP   │          │   │
│  │                               └────────────────────┘          │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    ASR Backend Factory                        │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │  def create_asr_backend(name, config, params, task):    │ │   │
│  │  │      backends = {                                        │ │   │
│  │  │          "faster-whisper": FasterWhisperProASR,          │ │   │
│  │  │          "stable-ts": StableTSASR,                       │ │   │
│  │  │          "qwen": Qwen3ASR,  # NEW                        │ │   │
│  │  │      }                                                   │ │   │
│  │  │      return backends[name](config, params, task)         │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**File Structure**:
```
whisperjav/
├── pipelines/
│   ├── base_pipeline.py
│   ├── balanced_pipeline.py      # Modified: ASR backend selection
│   └── faster_pipeline.py        # Modified: ASR backend selection
├── modules/
│   ├── asr/
│   │   ├── __init__.py           # NEW: Factory
│   │   ├── base.py               # NEW: Protocol
│   │   ├── faster_whisper_pro.py # Moved/renamed
│   │   ├── stable_ts.py          # Moved/renamed
│   │   └── qwen.py               # NEW
│   └── ...
├── config/
│   └── components/
│       └── asr/
│           ├── faster_whisper.py
│           └── qwen.py           # NEW
```

**Pros**:
- Reuses all existing pipeline infrastructure
- Single maintenance path for scene detection, stitching, etc.
- Configuration follows existing patterns
- Easy to switch backends via CLI flag
- Minimal code duplication

**Cons**:
- Qwen3-ASR has different timestamp generation (ForcedAligner vs inline)
- May need interface adapter layer
- Backend-specific features harder to expose
- Risk of lowest-common-denominator interface

**Implementation Effort**: Medium (1-2 weeks)

---

### Option C: Hybrid Architecture (vLLM Server + Client Backend)

**Architecture**: Run Qwen3-ASR as a separate vLLM server, access via OpenAI-compatible API.

```
┌─────────────────────────────────────────────────────────────────────┐
│               Option C: vLLM Server Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    WhisperJAV Process                        │    │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────────────────────┐    │    │
│  │  │  Audio  │──>│ Scene   │──>│   QwenASRClient         │    │    │
│  │  │ Extract │   │ Detect  │   │   (HTTP/gRPC)           │    │    │
│  │  └─────────┘   └─────────┘   └───────────┬─────────────┘    │    │
│  │                                           │                  │    │
│  └───────────────────────────────────────────│──────────────────┘    │
│                                              │                       │
│                              HTTP/OpenAI API │                       │
│                                              v                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    vLLM Server (Separate Process)            │    │
│  │  ┌───────────────────────────────────────────────────────┐  │    │
│  │  │  qwen-asr-serve Qwen/Qwen3-ASR-1.7B                   │  │    │
│  │  │    --gpu-memory-utilization 0.8                       │  │    │
│  │  │    --host 0.0.0.0 --port 8000                         │  │    │
│  │  │                                                        │  │    │
│  │  │  ┌─────────────┐  ┌─────────────────┐                 │  │    │
│  │  │  │ Qwen3-ASR   │  │ ForcedAligner   │                 │  │    │
│  │  │  │   Model     │  │   (Integrated)  │                 │  │    │
│  │  │  └─────────────┘  └─────────────────┘                 │  │    │
│  │  │                                                        │  │    │
│  │  │  ┌─────────────────────────────────────────────────┐  │  │    │
│  │  │  │ Continuous Batching | CUDA Graphs | PagedAttention │ │    │
│  │  │  └─────────────────────────────────────────────────┘  │  │    │
│  │  │                                                        │  │    │
│  │  └───────────────────────────────────────────────────────┘  │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Alternative: Docker Container             │    │
│  │  ┌───────────────────────────────────────────────────────┐  │    │
│  │  │  docker run --gpus all -p 8000:80 qwenllm/qwen3-asr   │  │    │
│  │  └───────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Client Implementation**:
```python
class QwenASRClient:
    """HTTP client for vLLM-served Qwen3-ASR."""

    def __init__(self, base_url="http://localhost:8000/v1"):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

    def transcribe(self, audio_path, language=None):
        # Using OpenAI-compatible API
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        response = self.client.audio.transcriptions.create(
            model="Qwen/Qwen3-ASR-1.7B",
            file=audio_data,
        )
        return {"text": response.text, "language": "Japanese"}
```

**Pros**:
- Maximum throughput (vLLM optimizations)
- Process isolation (ASR crash doesn't kill WhisperJAV)
- Can scale independently (multiple GPU servers)
- Hot-swap models without restarting WhisperJAV
- Cloud deployment ready (K8s, load balancing)
- Simpler client code (HTTP)

**Cons**:
- Deployment complexity (two processes)
- Network latency overhead
- Forced aligner integration more complex
- Not suitable for single-user desktop use
- Harder to debug

**Implementation Effort**: High (3-4 weeks)

---

### Option D: Direct vLLM Integration (In-Process vLLM)

**Architecture**: Use vLLM's LLM class directly in WhisperJAV, benefiting from vLLM optimizations without server overhead.

```
┌─────────────────────────────────────────────────────────────────────┐
│               Option D: Direct vLLM Integration                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     WhisperJAV Process                        │   │
│  │                                                               │   │
│  │  ┌─────────┐   ┌─────────┐   ┌───────────────────────────┐   │   │
│  │  │  Audio  │──>│ Scene   │──>│      Qwen3ASR_vLLM        │   │   │
│  │  │ Extract │   │ Detect  │   │                           │   │   │
│  │  └─────────┘   └─────────┘   │  ┌─────────────────────┐  │   │   │
│  │                               │  │  vLLM.LLM()         │  │   │   │
│  │                               │  │  - Continuous Batch │  │   │   │
│  │                               │  │  - PagedAttention   │  │   │   │
│  │                               │  │  - CUDA Graphs      │  │   │   │
│  │                               │  └─────────────────────┘  │   │   │
│  │                               │                           │   │   │
│  │                               │  ┌─────────────────────┐  │   │   │
│  │                               │  │  ForcedAligner      │  │   │   │
│  │                               │  │  (Transformers)     │  │   │   │
│  │                               │  └─────────────────────┘  │   │   │
│  │                               │                           │   │   │
│  │                               └───────────────────────────┘   │   │
│  │                                            │                  │   │
│  │                                            v                  │   │
│  │                               ┌────────────────────┐          │   │
│  │                               │   SRT Generation   │          │   │
│  │                               │   (With Stamps)    │          │   │
│  │                               └────────────────────┘          │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               vLLM Direct Inference Pattern                   │   │
│  │  ┌───────────────────────────────────────────────────────┐   │   │
│  │  │  from vllm import LLM, SamplingParams                 │   │   │
│  │  │                                                        │   │   │
│  │  │  llm = LLM(model="Qwen/Qwen3-ASR-1.7B")              │   │   │
│  │  │                                                        │   │   │
│  │  │  # Batch all scenes for maximum throughput            │   │   │
│  │  │  conversations = [build_audio_message(scene)          │   │   │
│  │  │                   for scene in scenes]                 │   │   │
│  │  │                                                        │   │   │
│  │  │  outputs = llm.chat(conversations, sampling_params)   │   │   │
│  │  └───────────────────────────────────────────────────────┘   │   │
│  │                                                               │   │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Difference from Option B**: Uses `vllm.LLM()` directly instead of `qwen_asr.Qwen3ASRModel.from_pretrained()`, gaining vLLM's optimizations (continuous batching, PagedAttention, CUDA graphs) without server overhead.

**Pros**:
- vLLM optimizations without server complexity
- Can batch multiple scenes in single inference call
- Single process (simpler deployment)
- No network latency

**Cons**:
- vLLM startup overhead (CUDA compilation)
- Requires `__main__` guard for spawn
- May conflict with existing ctranslate2 usage
- Less isolation (crash affects WhisperJAV)

**Implementation Effort**: Medium (1-2 weeks)

---

## Evaluation Matrix

| Criterion | Weight | Option A (Dedicated) | Option B (Pluggable) | Option C (Server) | Option D (Direct vLLM) |
|-----------|--------|---------------------|---------------------|-------------------|------------------------|
| **Implementation Effort** | 15% | 2 | 4 | 1 | 4 |
| **Maintainability** | 20% | 2 | 5 | 3 | 4 |
| **Performance** | 20% | 4 | 3 | 5 | 5 |
| **Desktop UX** | 15% | 4 | 4 | 1 | 4 |
| **Extensibility** | 10% | 5 | 4 | 5 | 4 |
| **Risk** | 10% | 3 | 4 | 2 | 3 |
| **Feature Parity** | 10% | 5 | 3 | 4 | 4 |
| **WEIGHTED SCORE** | 100% | **3.15** | **3.85** | **2.95** | **4.10** |

### Scoring Notes

- **5 = Excellent**, 4 = Good, 3 = Acceptable, 2 = Poor, 1 = Very Poor
- **Implementation Effort**: Lower is harder (inverted)
- **Desktop UX**: Single-user, local-first experience
- **Risk**: Higher = more unknowns/breaking changes

---

## Recommendations

### Revised Recommendation: Dedicated vLLM Pipeline

Based on architectural review, the chosen approach is a **dedicated vLLM Pipeline** (`--mode vllm`) as a framework-specific pipeline with Qwen3-ASR as the first engine.

**Rationale**:
1. **Framework-first design** - Named for vLLM, extensible for future ASR engines
2. **Code redundancy acceptable** - Copy Fidelity as proven reference
3. **Clean separation** - No risk of breaking existing pipelines
4. **Future-proof** - vLLM is becoming the standard for multimodal inference
5. **Pragmatic** - Start simple with v3 config, migrate to v4 later

### Implementation Path

#### Phase 1: Foundation (Week 1)

**1.1 Create vLLM Pipeline** (copy from Fidelity)
```
whisperjav/pipelines/vllm_pipeline.py
```

**1.2 Create ASR Engine Module**
```python
# whisperjav/modules/asr/qwen.py
class Qwen3ASR:
    """Qwen3-ASR engine using vLLM or transformers backend."""

    def __init__(self, model_config: Dict, params: Dict, task: str):
        self.model_name = model_config.get("model_name", "Qwen/Qwen3-ASR-1.7B")
        self.task = task

        # Backend selection: vLLM (preferred) or transformers (fallback)
        backend = params.get("vllm", {}).get("backend", "vllm")

        if backend == "vllm":
            self._init_vllm(model_config, params)
        else:
            self._init_transformers(model_config, params)

        # ForcedAligner for timestamps
        if params.get("aligner", {}).get("enabled", True):
            self._init_aligner(params)

    def _init_vllm(self, model_config, params):
        from qwen_asr import Qwen3ASRModel
        vllm_opts = params.get("vllm", {})
        self.model = Qwen3ASRModel.LLM(
            model=self.model_name,
            gpu_memory_utilization=vllm_opts.get("gpu_memory_utilization", 0.7),
            max_inference_batch_size=vllm_opts.get("max_inference_batch_size", 32),
            max_new_tokens=vllm_opts.get("max_new_tokens", 4096),
        )
        self._backend = "vllm"

    def _init_transformers(self, model_config, params):
        import torch
        from qwen_asr import Qwen3ASRModel
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=params.get("max_inference_batch_size", 32),
            max_new_tokens=params.get("max_new_tokens", 4096),
        )
        self._backend = "transformers"

    def _init_aligner(self, params):
        import torch
        from qwen_asr import Qwen3ForcedAligner
        aligner_opts = params.get("aligner", {})
        self.aligner = Qwen3ForcedAligner.from_pretrained(
            aligner_opts.get("model", "Qwen/Qwen3-ForcedAligner-0.6B"),
            dtype=torch.bfloat16,
            device_map=aligner_opts.get("device_map", "cuda:0"),
        )

    def transcribe(self, audio_path, **kwargs) -> Dict:
        """Transcribe single audio file."""
        results = self.model.transcribe(
            audio=str(audio_path),
            language=kwargs.get("language"),
            return_time_stamps=True,
        )
        return self._format_result(results[0], audio_path)

    def transcribe_batch(self, audio_paths: List[Path], **kwargs) -> List[Dict]:
        """Batch transcribe for maximum throughput (vLLM backend)."""
        results = self.model.transcribe(
            audio=[str(p) for p in audio_paths],
            language=kwargs.get("language"),
            return_time_stamps=True,
        )
        return [self._format_result(r, p) for r, p in zip(results, audio_paths)]

    def cleanup(self):
        """Release GPU resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'aligner'):
            del self.aligner
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
```

**1.3 Update CLI** (`main.py`)
```python
# Add to mode choices
parser.add_argument('--mode', choices=['faster', 'balanced', 'fidelity', 'vllm'], ...)

# Add model selection for vLLM
parser.add_argument('--model', default='qwen-1.7b',
                    choices=['qwen-1.7b', 'qwen-0.6b'],
                    help='ASR model for vLLM pipeline')
```

#### Phase 2: Integration (Week 2)

**2.1 Pipeline Factory Update**
```python
PIPELINES = {
    'faster': FasterPipeline,
    'balanced': BalancedPipeline,
    'fidelity': FidelityPipeline,
    'vllm': VLLMPipeline,  # NEW
}
```

**2.2 TranscriptionTuner Extension**
- Add `vllm` mode configuration
- Map `--model qwen-1.7b` → `Qwen/Qwen3-ASR-1.7B`
- Map `--model qwen-0.6b` → `Qwen/Qwen3-ASR-0.6B`

**2.3 VRAM Sequencing** (inherit from Fidelity)
- Phase 1: Enhancer (exclusive VRAM)
- Destroy Enhancer, clear cache
- Phase 2: ASR (exclusive VRAM)
- Destroy ASR before return

#### Phase 3: Testing & Validation (Week 3)

**3.1 Benchmark Suite**
- Compare Qwen3-ASR vs Whisper on Japanese test files
- Measure timestamp accuracy
- VRAM profiling on 10-12GB GPUs

**3.2 Configuration**
- Add to v3 TranscriptionTuner (immediate)
- Plan v4 YAML migration (future)

**3.3 Documentation**
- Update CLAUDE.md with vLLM mode
- Add usage examples

### Fallback Strategy

If vLLM backend has issues (spawn errors, VRAM conflicts):
1. **Immediate**: Use `transformers` backend via `qwen_asr.Qwen3ASRModel.from_pretrained()`
2. **Config flag**: `--vllm-backend transformers`

### Model Selection Summary

| CLI Flag | Model | VRAM | Use Case |
|----------|-------|------|----------|
| `--model qwen-1.7b` (default) | Qwen/Qwen3-ASR-1.7B | ~4GB | Quality-focused |
| `--model qwen-0.6b` | Qwen/Qwen3-ASR-0.6B | ~1.5GB | Speed/memory-constrained |

---

### Quick Wins for Early Success (Reviewer Recommended)

Based on external review feedback, prioritize these for early validation:

1. **Start with 0.6B model first** - Lower risk, faster iteration cycles
   - Validate pipeline works end-to-end before optimizing for quality
   - 0.6B has 2000x throughput advantage for rapid testing

2. **Implement basic batch processing early** (2-4 scenes)
   - Demonstrate throughput gains over sequential processing
   - Validate batching contract before complex edge cases

3. **Compare timestamp accuracy early**
   - Benchmark ForcedAligner vs Whisper on Japanese test files
   - Identify any Japanese-specific issues before full integration

4. **Document VRAM usage patterns**
   - Profile with 10-12GB GPU (RTX 3060, T4)
   - Create reference table for user guidance

5. **Enhanced Benchmarking Suite**
   - Singing voice accuracy (critical for JAV BGM)
   - Background noise robustness
   - Speaker diarization (if multi-speaker content)
   - Japanese honorifics and contextual speech

---

## Ensemble Mode Integration

### Why Design for Ensemble Early

Based on past experience documented in the codebase, memory management and subprocess handling can be "tricky" with GPU-based pipelines. The existing ensemble architecture (`EnsembleOrchestrator` + `pass_worker.py`) was specifically designed to handle these challenges with the **"Drop-Box + Nuclear Exit" pattern**.

### Critical Insight

**vLLM has the same GPU memory cleanup issues as ctranslate2** (the backend for faster-whisper). This is documented in [vLLM Issue #1908](https://github.com/vllm-project/vllm/issues/1908).

**The existing WhisperJAV subprocess isolation pattern is perfectly suited for vLLM.**

### Ensemble Integration Requirements

#### 1. Register VLLMPipeline in pass_worker.py

```python
# ensemble/pass_worker.py
from whisperjav.pipelines.vllm_pipeline import VLLMPipeline

PIPELINE_CLASSES = {
    "balanced": BalancedPipeline,
    "fast": FastPipeline,
    "faster": FasterPipeline,
    "fidelity": FidelityPipeline,
    "vllm": VLLMPipeline,  # ADD THIS
    # ...
}

PIPELINE_BACKENDS = {
    # ...
    "vllm": "qwen_vllm",  # New backend type
}
```

#### 2. Environment Variables in Subprocess

```python
# ensemble/pass_worker.py - run_pass_worker()
def run_pass_worker(payload: WorkerPayload, result_file: str) -> None:
    # Existing subprocess markers
    os.environ['WHISPERJAV_SUBPROCESS_WORKER'] = '1'

    # vLLM-specific: Reduce memory overhead for multimodal inputs
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

    # vLLM-specific: Prevent tokenizer deadlocks
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
```

#### 3. No Special Cleanup Needed

The existing Nuclear Exit pattern handles vLLM perfectly:
- vLLM destructor never runs (os._exit skips it)
- GPU memory reclaimed by OS when process terminates
- No need for `destroy_model_parallel()` or similar hacks

### Ensemble Workflow with vLLM

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Ensemble Mode with vLLM Pipeline                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CLI: whisperjav video.mp4 --mode ensemble                          │
│       --pass1 vllm --pass2 fidelity                                 │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                  EnsembleOrchestrator                       │     │
│  └────────────────────────────────────────────────────────────┘     │
│       │                                                              │
│       ├── Pass 1: Spawn subprocess                                  │
│       │   ┌──────────────────────────────────────────────────┐      │
│       │   │  VLLMPipeline (Qwen3-ASR)                        │      │
│       │   │  - Fast, high accuracy                            │      │
│       │   │  - Result → Drop-Box                              │      │
│       │   │  - Nuclear Exit (os._exit(0))                    │      │
│       │   └──────────────────────────────────────────────────┘      │
│       │   └─ GPU memory freed by OS                                  │
│       │                                                              │
│       ├── Pass 2: Spawn subprocess                                  │
│       │   ┌──────────────────────────────────────────────────┐      │
│       │   │  FidelityPipeline (OpenAI Whisper)               │      │
│       │   │  - Different model perspective                    │      │
│       │   │  - Result → Drop-Box                              │      │
│       │   │  - Nuclear Exit (os._exit(0))                    │      │
│       │   └──────────────────────────────────────────────────┘      │
│       │   └─ GPU memory freed by OS                                  │
│       │                                                              │
│       └── Merge: MergeEngine                                        │
│           - pass1_primary / pass2_primary / union strategies         │
│           - Final merged SRT                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Potential Ensemble Use Cases

| Pass 1 | Pass 2 | Rationale |
|--------|--------|-----------|
| `vllm` | `fidelity` | Qwen3-ASR speed + Whisper complementary |
| `vllm` | `balanced` | Qwen3-ASR + faster-whisper comparison |
| `vllm --model qwen-0.6b` | `vllm --model qwen-1.7b` | Fast sweep + quality refinement |

---

## Open Questions

### Resolved Decisions

| Question | Decision |
|----------|----------|
| Pipeline naming | `--mode vllm` (framework-first) |
| Default model | `qwen-1.7b` (quality-focused) |
| Model support | Both 0.6B and 1.7B via `--model` flag |
| Configuration | Start with v3, migrate to v4 later |
| Translation | Continue with PySubtrans (Qwen-MT is API-only) |
| Code approach | Copy FidelityPipeline as reference |

### Technical Questions (Status After Review)

1. **VRAM Coexistence**: Can Qwen3-ASR (vLLM) and speech enhancer share GPU memory cleanly?
   - ✅ **ADDRESSED**: Existing Fidelity pattern (VRAM handoff) handles this
   - Mitigation: Enhancer cleanup before ASR load (existing pattern)

2. **Spawn Guard**: vLLM requires `if __name__ == '__main__':` - how does this interact with WhisperJAV's subprocess workers?
   - ✅ **ADDRESSED**: WhisperJAV already uses `spawn` context
   - Mitigation: Environment variables in pass_worker.py

3. **Timestamp Accuracy**: Does ForcedAligner match Whisper's word-level timing for Japanese?
   - ⚠️ **NEEDS VALIDATION**: Benchmark on sample JAV files
   - **NEW**: TimestampAdapter layer required (see Contract 3)
   - Note: ForcedAligner supports Japanese (one of 11 languages)

4. **Batch Scene Transcription**: Should we batch all scenes in one vLLM call?
   - ✅ **ADDRESSED**: Batching Contract defined (see Contract 1)
   - Max 32 scenes, max 120s batch duration
   - Configurable via `--vllm-batch-size`

5. **ForcedAligner VRAM**: Does ForcedAligner need to be separate from ASR model?
   - ✅ **ADDRESSED**: Lazy loading pattern (see Contract 2)
   - Can be disabled with `--no-timestamps` to save 1.5GB VRAM
   - Load only when needed, unload explicitly

### Architectural Questions (For Discussion)

1. **VLLMASREngine Base Class**: Should we create an abstract base class for vLLM ASR engines?
   - Pro: Clean contract for future engines
   - Con: Over-engineering for single engine initially
   - Recommendation: Defer until second engine is added

2. **v4 Config Migration Timing**: When should we migrate to YAML-driven config?
   - Option A: After vLLM pipeline is stable (Phase 2)
   - Option B: Immediately with vLLM pipeline (higher risk)
   - Recommendation: Option A (pragmatic)

3. **GUI Integration**: Should vLLM mode be exposed in GUI initially?
   - Recommendation: CLI-only for v1, add GUI after validation

---

## Appendix C: External Review Summary

### Reviewer Verdicts

| Reviewer | Verdict | Score |
|----------|---------|-------|
| **DeepSeek** | "Production-ready architecture" | 9/10 |
| **Gemini** | "Robust and ready for implementation" | Approve |
| **ChatGPT** | "Conditional approval - fix 3 risks first" | Conditional |

### Consensus Findings (All Three Agree)

1. ✅ Framework-first naming (`--mode vllm`) is correct
2. ✅ Nuclear Exit pattern is the right solution for vLLM
3. ✅ FidelityPipeline as reference is smart
4. ✅ Qwen3-ASR is legitimate for JAV domain
5. ⚠️ **Need explicit batching policy** → Contract 1
6. ⚠️ **ForcedAligner should be lazy/optional** → Contract 2

### Critical Gaps Identified

| Gap | Raised By | Solution |
|-----|-----------|----------|
| No batching policy | All Three | Contract 1: Hard batching rules |
| Aligner lifecycle undefined | ChatGPT + Gemini | Contract 2: Lazy loading pattern |
| Timestamp semantics mismatch | ChatGPT | Contract 3: TimestampAdapter layer |
| No error recovery | DeepSeek | Contract 4: Retry + fallback |
| No quantization support | DeepSeek | Contract 5: Phase 2 INT8 |

### DeepSeek-Specific Recommendations

- Start with 0.6B model first (lower risk)
- Implement basic batch processing early (2-4 scenes)
- Compare timestamp accuracy early
- Document VRAM usage patterns
- Enhanced benchmarking: singing voice, noise robustness, honorifics

### Key Architectural Quote (ChatGPT)

> "If you skip [the TimestampAdapter], debugging subtitle drift will be hell."

### Conditions for Implementation Approval

1. ✅ Explicit vLLM batching rules → **Added (Contract 1)**
2. ✅ ForcedAligner lifecycle clarified → **Added (Contract 2)**
3. ✅ TimestampAdapter layer defined → **Added (Contract 3)**

**Status**: All conditions satisfied. Ready for implementation.

---

## References

- [Qwen3-ASR GitHub Repository](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR-1.7B HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen3-ForcedAligner-0.6B HuggingFace](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- [Qwen3-ASR Technical Report (arXiv:2601.21337)](https://arxiv.org/abs/2601.21337)
- [vLLM Documentation - Multimodal Inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)
- [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [PyPI: qwen-asr Package](https://pypi.org/project/qwen-asr/)

---

## Appendix A: vLLM Best Practices Research

### Critical Discovery: vLLM Has Same GPU Cleanup Issues as ctranslate2

**Key Finding**: vLLM has [known GPU memory release issues](https://github.com/vllm-project/vllm/issues/1908) that are strikingly similar to the ctranslate2 problems WhisperJAV already handles.

From GitHub Issue #1908:
> "If you simply destruct the LLM object of vLLM, the memory is not freed. This persists with an explicit gc.collect() call."

**Good News**: WhisperJAV's existing **"Drop-Box + Nuclear Exit" pattern** is perfectly suited for vLLM integration!

### WhisperJAV's Existing Solution (pass_worker.py)

```python
# The "Nuclear Exit" prevents destructor crashes
def _write_dropbox_and_exit(result_file, result, tracer, exit_code):
    # Write result to Drop-Box pickle file
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)

    # NUCLEAR EXIT - Skip Python shutdown sequence
    # This prevents C++ destructor crashes (ctranslate2 AND vLLM!)
    os._exit(exit_code)
```

**Why This Works**:
1. Each pipeline pass runs in an isolated subprocess (`mp.get_context('spawn')`)
2. Results are written to a pickle file (the "Drop-Box")
3. Process exits via `os._exit(0)`, skipping Python's shutdown sequence
4. GPU memory is reclaimed by the OS when the process terminates
5. No destructor code runs = no crashes

### vLLM-Specific Requirements

#### 1. Multiprocessing Start Method

vLLM documentation ([Python Multiprocessing](https://docs.vllm.ai/en/stable/design/multiprocessing/)):
- **spawn**: Compatible with vLLM, requires `if __name__ == '__main__':` guard
- **fork**: Default on Linux, breaks CUDA initialization
- **forkserver**: Hybrid, has compatibility constraints

**WhisperJAV already uses spawn**: `mp_context = mp.get_context('spawn')` ✅

#### 2. CUDA Initialization

From vLLM docs:
> "To ensure vLLM initializes CUDA correctly, avoid calling `torch.cuda.set_device` before initializing vLLM."

**Mitigation**: Use `CUDA_VISIBLE_DEVICES` environment variable instead of programmatic device selection.

#### 3. Memory Overhead with Multiprocessing

From [Issue #16185](https://github.com/vllm-project/vllm/issues/16185):
> "With `VLLM_ENABLE_V1_MULTIPROCESSING=0` the issue completely disappears."

**Consideration**: May need to set this environment variable for audio multimodal inputs.

#### 4. Attempted Cleanup Approaches (from community)

```python
# Approaches that DON'T fully work:
from vllm.distributed.parallel_state import destroy_model_parallel

destroy_model_parallel()
del llm.llm_engine.model_executor.driver_worker
del llm
gc.collect()
torch.cuda.empty_cache()
ray.shutdown()  # If using Ray
```

**Conclusion**: Manual cleanup is unreliable. Subprocess isolation + Nuclear Exit is the correct pattern.

### vLLM Server Mode Considerations

If vLLM is used as a server (Option C), additional considerations apply:

#### Health Check Limitations

From [vLLM Readiness Probes](https://llm-d.ai/docs/usage/readiness-probes):
> "The vLLM `/health` endpoint only indicates that the server process is running, not that models are loaded and ready to serve."

**Required Health Checks**:
1. **Container Running**: Basic process health
2. **API Server Ready**: vLLM accepting connections
3. **Model Loaded**: Model ready for inference (requires custom probe)

#### Recommended Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Basic liveness |
| `/metrics` | Prometheus metrics |
| `/v1/models` | Verify model loaded |
| Custom readiness | Model inference test |

#### Monitoring Metrics (from [Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/))

- Request rate (success/failure)
- End-to-end latency (P50, P95, P99)
- Token throughput
- Active requests
- GPU cache usage

### Integration Pattern for vLLM Pipeline

Based on this research, the vLLM Pipeline should:

```
┌─────────────────────────────────────────────────────────────────────┐
│              vLLM Pipeline Integration Pattern                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. CLI invokes: whisperjav video.mp4 --mode vllm                   │
│                                                                      │
│  2. If ensemble mode: EnsembleOrchestrator creates subprocess       │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │  mp.get_context('spawn')  # Already used                 │     │
│     │  os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'     │     │
│     │  os.environ['WHISPERJAV_SUBPROCESS_WORKER'] = '1'       │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                      │
│  3. In subprocess: VLLMPipeline.process()                           │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │  a. Create Qwen3ASR (loads vLLM model)                  │     │
│     │  b. Process scenes (batch or sequential)                 │     │
│     │  c. ForcedAligner for timestamps                         │     │
│     │  d. NO explicit cleanup (let Nuclear Exit handle it)     │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                      │
│  4. Write results to Drop-Box pickle file                           │
│                                                                      │
│  5. Nuclear Exit: os._exit(0)                                       │
│     - Skips Python shutdown                                          │
│     - No vLLM destructor runs                                        │
│     - OS reclaims GPU memory                                         │
│                                                                      │
│  6. Parent process reads Drop-Box, continues with merge             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Environment Variables for vLLM

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | Reduce memory overhead |
| `TOKENIZERS_PARALLELISM` | `false` | Prevent deadlock warnings |
| `CUDA_VISIBLE_DEVICES` | `0` | Device selection (not torch.cuda.set_device) |
| `WHISPERJAV_SUBPROCESS_WORKER` | `1` | Mark subprocess context |

### Offline vs Server Mode Decision

| Criteria | Offline (In-Process) | Server Mode |
|----------|---------------------|-------------|
| Desktop single-user | ✅ Preferred | Overkill |
| Memory cleanup | Drop-Box + Nuclear Exit | Process isolation |
| Startup time | Per-invocation | Amortized |
| Health checks | Not needed | Required |
| Debugging | Easier | Harder |
| **Recommendation** | **For WhisperJAV** | For cloud/multi-user |

---

## Appendix B: Translation Strategy Research

### Qwen-MT Analysis

[Qwen-MT](https://qwenlm.github.io/blog/qwen-mt/) is Alibaba's dedicated translation model built on Qwen3, supporting 92 languages including Japanese.

**Key Finding: Qwen-MT is API-only** - No open-source weights are available for local deployment.

| Aspect | Qwen-MT |
|--------|---------|
| Model Access | API only (qwen-mt-turbo via DashScope) |
| Languages | 92 languages including Japanese |
| Cost | ~$0.5/million tokens |
| Local Deployment | **Not available** |
| Open Weights | **Not available** |

### Translation Options for WhisperJAV + Qwen3-ASR

Given that Qwen-MT is API-only, here are the viable translation strategies:

#### Option T1: Continue with Existing PySubtrans Pipeline (Recommended)

```
┌─────────────────────────────────────────────────────────────────┐
│           Qwen3-ASR → PySubtrans Translation                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────────┐   │
│  │  Qwen3-ASR  │──>│   Japanese  │──>│    PySubtrans        │   │
│  │  (ASR)      │   │   SRT       │   │ (DeepSeek/GPT/Gemini)│   │
│  └─────────────┘   └─────────────┘   └──────────────────────┘   │
│                                                │                 │
│                                                v                 │
│                                       ┌──────────────────────┐  │
│                                       │   English SRT        │  │
│                                       └──────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Pros**:
- Already implemented and tested
- Multiple provider options (DeepSeek, GPT-4, Gemini, Claude)
- Works offline with local LLMs
- No additional integration work

**Cons**:
- Separate ASR and translation models
- Not optimized for speech-to-text translation

#### Option T2: Qwen-MT API Integration

```
┌─────────────────────────────────────────────────────────────────┐
│           Qwen3-ASR → Qwen-MT API Translation                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────────┐   │
│  │  Qwen3-ASR  │──>│   Japanese  │──>│    Qwen-MT API       │   │
│  │  (Local)    │   │   SRT       │   │  (qwen-mt-turbo)     │   │
│  └─────────────┘   └─────────────┘   └──────────────────────┘   │
│                                                │                 │
│                                                v                 │
│                                       ┌──────────────────────┐  │
│                                       │   English SRT        │  │
│                                       └──────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Pros**:
- Same vendor ecosystem (Qwen ASR + Qwen MT)
- Potentially better Japanese translation quality
- Low cost ($0.5/million tokens)

**Cons**:
- Requires API key (DashScope account)
- Not fully local/offline
- Additional provider to maintain

#### Option T3: Local Qwen3 LLM for Translation

Use general-purpose Qwen3 models (available on HuggingFace) for translation:

```
┌─────────────────────────────────────────────────────────────────┐
│           Qwen3-ASR → Local Qwen3 Translation                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────────┐   │
│  │  Qwen3-ASR  │──>│   Japanese  │──>│    Qwen3-8B-Instruct │   │
│  │  (1.7B)     │   │   SRT       │   │    (Local LLM)       │   │
│  └─────────────┘   └─────────────┘   └──────────────────────┘   │
│                                                │                 │
│                                                v                 │
│                                       ┌──────────────────────┐  │
│                                       │   English SRT        │  │
│                                       └──────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Pros**:
- Fully local/offline
- Open-source weights on HuggingFace
- Same vendor ecosystem
- Qwen3 supports 119 languages with strong Japanese

**Cons**:
- General LLM, not translation-optimized
- Higher VRAM requirements (8B = ~16GB)
- May need fine-tuning for subtitle translation

### Translation Recommendation

For initial Qwen3-ASR integration, **Option T1 (PySubtrans)** is recommended:

1. **Zero additional integration work** - focus on ASR quality first
2. **Proven workflow** - PySubtrans handles subtitle-specific translation well
3. **Multiple fallback providers** - not locked to single vendor

**Future consideration**: If WhisperJAV moves to a fully Qwen-based stack, Option T3 (local Qwen3) provides a coherent ecosystem. This could be explored in a separate ADR.

---

## Appendix B: User Requirements Update

Based on user feedback:

| Requirement | User Response | Impact |
|-------------|---------------|--------|
| Target VRAM | 10-15GB (RTX 3060 to T4) | Both models viable |
| Model Support | Both 0.6B and 1.7B | Need `--model` selection |
| Translation | Research Qwen options | T1 (PySubtrans) recommended |

### VRAM Planning (10-15GB Target)

| Configuration | VRAM Estimate | Fits 10GB? | Fits 15GB? |
|--------------|---------------|------------|------------|
| Qwen3-ASR-0.6B only | ~1.5GB | Yes | Yes |
| Qwen3-ASR-1.7B only | ~4GB | Yes | Yes |
| 1.7B + ForcedAligner | ~5.5GB | Yes | Yes |
| 1.7B + Aligner + Enhancer (ClearVoice) | ~7GB | Yes | Yes |
| 1.7B + Aligner + Enhancer (BS-RoFormer) | ~9GB | Tight | Yes |

**Conclusion**: Both models fit comfortably within 10-15GB VRAM budget.

---

**Document Version**: 3.0 (Post-Review)
**Last Updated**: 2026-01-31
**Review Status**: External review complete, all conditions addressed
