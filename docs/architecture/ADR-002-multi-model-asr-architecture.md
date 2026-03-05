# ADR-002: Multi-Model ASR Architecture for WhisperJAV v2.0

**Status**: Draft
**Date**: 2025-12-30
**Author**: Solution Architecture Review
**Decision**: Proposed

---

## Executive Summary

This document analyzes the architectural requirements for integrating six new ASR models into WhisperJAV. Based on deep research, we recommend a **tiered pipeline architecture** that maximizes code reuse while properly handling the distinct requirements of each model family.

---

## Proposed Models

| # | Model | Architecture | Parameters | Key Strength |
|---|-------|--------------|------------|--------------|
| 1 | `reazon-research/japanese-wav2vec2-large-rs35kh` | Wav2Vec2 CTC | 319M | Best non-Whisper Japanese ASR |
| 2 | `sinhat98/w2v-bert-2.0-japanese-colab-CV16.0` | Wav2Vec2-BERT | 600M | BERT-enhanced efficiency |
| 3 | `TKU410410103/hubert-large-japanese-asr` | HuBERT CTC | 300M | Robust to background noise |
| 4 | `facebook/mms-1b-all` | MMS (Wav2Vec2) | 1B | Massively multilingual, very fast |
| 5 | `facebook/seamless-m4t-v2-large` | SeamlessM4T | 2.3B | Simultaneous ASR + Translation |
| 6 | `Qwen/Qwen2-Audio-7B-Instruct` | Qwen-Audio LLM | 7B | Instruction-following audio LLM |

---

## Architecture Analysis

### Model Family Classification

After deep research, the models fall into **three distinct architectural families**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ASR MODEL FAMILIES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FAMILY A: CTC-Based Models                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Wav2Vec2 (reazon-research)                                        │    │
│  │  • Wav2Vec2-BERT (sinhat98)                                          │    │
│  │  • HuBERT (TKU410410103)                                             │    │
│  │  • MMS (facebook/mms-1b-all)                                         │    │
│  │                                                                       │    │
│  │  Characteristics:                                                     │    │
│  │  - Connectionist Temporal Classification (CTC) loss                  │    │
│  │  - Uses Wav2Vec2ForCTC / HubertForCTC model classes                  │    │
│  │  - Requires AutoProcessor for feature extraction                     │    │
│  │  - Output: character/subword logits → argmax → decode                │    │
│  │  - NO native timestamps (requires CTC alignment post-processing)     │    │
│  │  - 16kHz audio input                                                 │    │
│  │  - HuggingFace pipeline("automatic-speech-recognition") SUPPORTED    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  FAMILY B: Sequence-to-Sequence Models                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Whisper (existing - kotoba-whisper, openai/whisper)               │    │
│  │  • SeamlessM4T (facebook/seamless-m4t-v2-large)                      │    │
│  │                                                                       │    │
│  │  Characteristics:                                                     │    │
│  │  - Encoder-decoder architecture                                      │    │
│  │  - Native timestamp support (segment-level or word-level)            │    │
│  │  - Built-in translation capability                                   │    │
│  │  - Chunked long-form processing                                      │    │
│  │  - Different model classes per architecture                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  FAMILY C: Audio Language Models (Audio-LLMs)                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Qwen2-Audio (Qwen/Qwen2-Audio-7B-Instruct)                        │    │
│  │                                                                       │    │
│  │  Characteristics:                                                     │    │
│  │  - Instruction-following paradigm (prompt-based)                     │    │
│  │  - Uses chat template (ChatML format)                                │    │
│  │  - Can perform multiple tasks: transcribe, describe, analyze         │    │
│  │  - Requires Qwen2AudioForConditionalGeneration                       │    │
│  │  - Very different API from traditional ASR                           │    │
│  │  - 7B parameters = significant VRAM requirements                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Model Research

### Family A: CTC-Based Models

#### 1. Wav2Vec2 Japanese (reazon-research/japanese-wav2vec2-large-rs35kh)

**Architecture**: Wav2Vec2 Large fine-tuned on ReazonSpeech v2.0 (35,000 hours)

**API Usage**:
```python
from transformers import AutoProcessor, Wav2Vec2ForCTC
import torch

model = Wav2Vec2ForCTC.from_pretrained(
    "reazon-research/japanese-wav2vec2-large-rs35kh",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to("cuda")

processor = AutoProcessor.from_pretrained(
    "reazon-research/japanese-wav2vec2-large-rs35kh"
)

# IMPORTANT: Requires 0.5s padding before inference
audio = np.pad(audio, pad_width=int(0.5 * 16_000))

input_values = processor(audio, sampling_rate=16_000, return_tensors="pt").input_values
logits = model(input_values.to("cuda")).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])
```

**Key Requirements**:
- Requires 0.5s audio padding (unique to this model)
- 16kHz sampling rate
- CTC decoding (no timestamps)
- Flash Attention 2 support

**Performance**: CER 11.00% on JSUT-BASIC5000

**Pipeline Compatibility**: ✅ Works with `pipeline("automatic-speech-recognition")`

---

#### 2. Wav2Vec2-BERT (sinhat98/w2v-bert-2.0-japanese-colab-CV16.0)

**Architecture**: Meta's Wav2Vec2-BERT hybrid, fine-tuned on Common Voice 16.0

**API Usage**: Same as Wav2Vec2 CTC pattern

**Key Requirements**:
- 16kHz sampling rate
- Based on ylacombe/w2v-bert-2.0

**Performance**: CER 31.71% (higher than Wav2Vec2 - may need verification)

**Pipeline Compatibility**: ✅ Should work (standard CTC architecture)

---

#### 3. HuBERT Japanese (TKU410410103/hubert-large-japanese-asr)

**Architecture**: HuBERT Large fine-tuned on ReazonSpeech + Common Voice

**API Usage**:
```python
from transformers import HubertForCTC, Wav2Vec2Processor

model = HubertForCTC.from_pretrained('TKU410410103/hubert-large-japanese-asr')
processor = Wav2Vec2Processor.from_pretrained("TKU410410103/hubert-large-japanese-asr")

inputs = processor(audio_array, sampling_rate=16_000, return_tensors="pt")
logits = model(inputs.input_values).logits
pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(pred_ids)
```

**Key Requirements**:
- **Outputs Hiragana only** (no Kanji)
- Uses `HubertForCTC` instead of `Wav2Vec2ForCTC`
- Uses `Wav2Vec2Processor` (shared with Wav2Vec2)

**Performance**: WER 22.71% / CER 9.40% on Common Voice 11.0

**Pipeline Compatibility**: ✅ Works with `pipeline("automatic-speech-recognition")`

**Note**: Hiragana-only output may require post-processing for kanji conversion.

---

#### 4. MMS (facebook/mms-1b-all)

**Architecture**: 1B parameter Wav2Vec2-based multilingual model with language adapters

**API Usage**:
```python
from transformers import Wav2Vec2ForCTC, AutoProcessor

model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")

# CRITICAL: Must load language adapter
processor.tokenizer.set_target_lang("jpn")  # ISO 639-3 code
model.load_adapter("jpn")

inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])
```

**Key Requirements**:
- **Requires language adapter loading** (not automatic)
- Uses ISO 639-3 language codes (`jpn` for Japanese)
- 1162 languages supported
- Very fast inference

**Pipeline Compatibility**: ⚠️ **Partial** - Standard pipeline doesn't handle adapter switching

**Note**: Custom wrapper needed for language adapter management.

---

### Family B: Sequence-to-Sequence Models

#### 5. SeamlessM4T (facebook/seamless-m4t-v2-large)

**Architecture**: UnitY2 architecture with 2.3B parameters

**API Usage**:
```python
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# Resample to 16kHz
audio, sr = torchaudio.load("audio.wav")
audio = torchaudio.functional.resample(audio, sr, 16_000)

# For ASR (transcription)
audio_inputs = processor(audios=audio, return_tensors="pt")
output_tokens = model.generate(**audio_inputs, tgt_lang="jpn", generate_speech=False)
transcription = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)

# For Translation (Japanese → English)
output_tokens = model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
translation = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
```

**Key Requirements**:
- Uses `SeamlessM4Tv2Model` (not compatible with standard ASR pipeline)
- Built-in translation to 96 languages
- Can output speech (T2S) - 35 languages
- 2.3B parameters = significant VRAM (~9GB FP16)

**Capabilities**:
- ✅ Speech-to-Text (ASR)
- ✅ Speech-to-Text Translation
- ✅ Text-to-Text Translation
- ✅ Text-to-Speech
- ✅ Speech-to-Speech Translation

**Pipeline Compatibility**: ❌ **No** - Requires custom wrapper

---

### Family C: Audio Language Models

#### 6. Qwen2-Audio (Qwen/Qwen2-Audio-7B-Instruct)

**Architecture**: 7B parameter Audio-LLM with instruction following

**API Usage**:
```python
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    device_map="auto"
)

# Load audio
audio, sr = librosa.load("audio.wav", sr=processor.feature_extractor.sampling_rate)

# Create conversation with instruction
conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "audio.wav"},
        {"type": "text", "text": "Transcribe this Japanese audio accurately."},
    ]},
]

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios = [audio]

inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
generate_ids = model.generate(**inputs, max_length=512)
response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
```

**Key Requirements**:
- **Instruction-based** - requires prompt engineering
- Uses ChatML format with `apply_chat_template()`
- 7B parameters = ~14GB VRAM (FP16)
- Can do multi-task: transcribe, translate, describe emotions

**Unique Capabilities**:
- "Transcribe this anime clip and describe the emotion"
- "What language is being spoken?"
- "Summarize what's being said"

**Pipeline Compatibility**: ❌ **No** - Completely different paradigm

**Note**: Japanese support needs verification (documentation only mentions English).

---

## Architectural Recommendation

### Option 1: Unified TransformersASR with Model Adapters (RECOMMENDED)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        UNIFIED ASR MODULE ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TransformersPipeline (existing)                                             │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      UnifiedASR (new)                                │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  Model Family Detection (auto-detect from model_id)         │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                            │                                         │    │
│  │         ┌──────────────────┼──────────────────┐                      │    │
│  │         ▼                  ▼                  ▼                      │    │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                │    │
│  │  │ WhisperASR  │   │   CTC_ASR   │   │ Seq2SeqASR  │                │    │
│  │  │ (existing)  │   │   (new)     │   │   (new)     │                │    │
│  │  │             │   │             │   │             │                │    │
│  │  │ • Whisper   │   │ • Wav2Vec2  │   │ • Seamless  │                │    │
│  │  │ • Kotoba    │   │ • HuBERT    │   │             │                │    │
│  │  │             │   │ • MMS       │   │             │                │    │
│  │  │             │   │ • w2v-BERT  │   │             │                │    │
│  │  └─────────────┘   └─────────────┘   └─────────────┘                │    │
│  │         │                  │                  │                      │    │
│  │         └──────────────────┼──────────────────┘                      │    │
│  │                            ▼                                         │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  Unified Output Format: List[{text, start, end}]            │    │    │
│  │  │  (CTC models require CTC alignment for timestamps)          │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  AudioLLMPipeline (new - separate)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Qwen2-Audio                                                       │    │
│  │  • Different paradigm: instruction → response                        │    │
│  │  • Custom prompts for different tasks                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 1: CTC ASR Backend (Highest Priority)

Create `whisperjav/modules/ctc_asr.py`:

```python
class CTCASR:
    """
    CTC-based ASR for Wav2Vec2, HuBERT, and MMS models.
    """

    SUPPORTED_MODELS = {
        "wav2vec2": ["reazon-research/japanese-wav2vec2-large-rs35kh"],
        "wav2vec2-bert": ["sinhat98/w2v-bert-2.0-japanese-colab-CV16.0"],
        "hubert": ["TKU410410103/hubert-large-japanese-asr"],
        "mms": ["facebook/mms-1b-all"],
    }

    def __init__(self, model_id: str, language: str = "ja", device: str = "auto"):
        self.model_id = model_id
        self.model_type = self._detect_model_type(model_id)
        self.language = language

        # Load appropriate model class
        if self.model_type == "hubert":
            from transformers import HubertForCTC
            self.model_class = HubertForCTC
        else:
            from transformers import Wav2Vec2ForCTC
            self.model_class = Wav2Vec2ForCTC

    def transcribe(self, audio_path: Path) -> List[Dict]:
        # Load and preprocess audio
        audio = self._load_audio(audio_path)

        # Model-specific preprocessing
        if "reazon-research" in self.model_id:
            audio = np.pad(audio, pad_width=int(0.5 * 16_000))  # 0.5s padding

        if self.model_type == "mms":
            self._load_language_adapter(self.language)

        # Forward pass
        inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt")
        logits = self.model(**inputs.to(self.device)).logits

        # CTC decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])

        # Generate timestamps using CTC alignment
        segments = self._ctc_alignment(logits, transcription)

        return segments

    def _ctc_alignment(self, logits, text) -> List[Dict]:
        """
        Generate timestamps from CTC logits.

        Options:
        1. Use ctc-segmentation library
        2. Use stable-ts alignment
        3. Simple duration-based estimation
        """
        # Implementation needed
        pass
```

#### Phase 2: SeamlessM4T Backend

Create `whisperjav/modules/seamless_asr.py`:

```python
class SeamlessASR:
    """
    SeamlessM4T ASR with built-in translation.
    """

    def __init__(
        self,
        model_id: str = "facebook/seamless-m4t-v2-large",
        source_lang: str = "jpn",
        target_lang: str = "jpn",  # Same = transcription, different = translation
        device: str = "auto"
    ):
        from transformers import SeamlessM4Tv2Model, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = SeamlessM4Tv2Model.from_pretrained(model_id)
        self.source_lang = source_lang
        self.target_lang = target_lang

    def transcribe(self, audio_path: Path) -> List[Dict]:
        audio = self._load_audio(audio_path)
        inputs = self.processor(audios=audio, return_tensors="pt")

        output_tokens = self.model.generate(
            **inputs,
            tgt_lang=self.target_lang,
            generate_speech=False
        )

        text = self.processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)

        # SeamlessM4T doesn't provide timestamps
        # Need to estimate or use alignment
        return [{"text": text, "start": 0.0, "end": self._get_duration(audio_path)}]
```

#### Phase 3: Audio-LLM Pipeline (Qwen2-Audio)

Create `whisperjav/pipelines/audio_llm_pipeline.py`:

```python
class AudioLLMPipeline(BasePipeline):
    """
    Instruction-following Audio LLM pipeline.

    Different from other pipelines - uses prompts for task specification.
    """

    DEFAULT_PROMPTS = {
        "transcribe": "Transcribe this Japanese audio accurately.",
        "transcribe_emotion": "Transcribe this audio and describe the speaker's emotion.",
        "translate": "Transcribe and translate this Japanese audio to English.",
        "summarize": "Summarize what is being said in this audio.",
    }

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-Audio-7B-Instruct",
        task: str = "transcribe",
        custom_prompt: Optional[str] = None,
        **kwargs
    ):
        self.model_id = model_id
        self.prompt = custom_prompt or self.DEFAULT_PROMPTS.get(task)

        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto"
        )
```

---

## CLI Integration

### Proposed CLI Arguments

```bash
# CTC Models
whisperjav video.mp4 --mode ctc --ctc-model reazon-research/japanese-wav2vec2-large-rs35kh
whisperjav video.mp4 --mode ctc --ctc-model TKU410410103/hubert-large-japanese-asr
whisperjav video.mp4 --mode ctc --ctc-model facebook/mms-1b-all --mms-lang jpn

# SeamlessM4T
whisperjav video.mp4 --mode seamless
whisperjav video.mp4 --mode seamless --seamless-translate eng  # Translate to English

# Audio-LLM (Qwen2-Audio)
whisperjav video.mp4 --mode audio-llm
whisperjav video.mp4 --mode audio-llm --llm-prompt "Transcribe and describe emotions"

# Ensemble with mixed models
whisperjav video.mp4 --ensemble \
    --pass1-pipeline ctc --pass1-model reazon-research/japanese-wav2vec2-large-rs35kh \
    --pass2-pipeline transformers --pass2-model kotoba-tech/kotoba-whisper-v2.2
```

---

## Key Technical Challenges

### 1. Timestamp Generation for CTC Models

CTC models don't provide native timestamps. Options:

| Approach | Pros | Cons |
|----------|------|------|
| **ctc-segmentation** | Accurate alignment | Requires additional dependency |
| **stable-ts alignment** | Already integrated | May not work well with non-Whisper |
| **Duration estimation** | Simple | Inaccurate for variable-rate speech |
| **Forced alignment (MFA)** | Gold standard | Complex, slow |

**Recommendation**: Use `ctc-segmentation` library for CTC models.

### 2. Hiragana-Only Output (HuBERT)

The HuBERT model outputs Hiragana only. Options:

| Approach | Pros | Cons |
|----------|------|------|
| **pykakasi** | Fast, pure Python | Basic accuracy |
| **mecab + unidic** | High accuracy | Heavy dependency |
| **LLM post-processing** | Contextual | Slow, expensive |
| **Leave as Hiragana** | Simplest | May not be acceptable |

**Recommendation**: Make configurable, default to pykakasi.

### 3. VRAM Management

| Model | Size | FP16 VRAM | Note |
|-------|------|-----------|------|
| Wav2Vec2 (reazon) | 319M | ~1.5GB | Lightweight |
| HuBERT | 300M | ~1.4GB | Lightweight |
| MMS | 1B | ~4GB | Medium |
| SeamlessM4T | 2.3B | ~9GB | Heavy |
| Qwen2-Audio | 7B | ~14GB | Very Heavy |
| Kotoba-Whisper | 1.5B | ~6GB | Current |

**Recommendation**: Maintain scope-based resource management pattern.

### 4. MMS Language Adapter Management

MMS requires explicit adapter loading per language. Need to:
- Track current loaded adapter
- Lazy-load adapters on language change
- Cache commonly used adapters

---

## Comparison Matrix

| Feature | Whisper | CTC (Wav2Vec2/HuBERT) | SeamlessM4T | Qwen2-Audio |
|---------|---------|------------------------|-------------|-------------|
| **Native Timestamps** | ✅ Yes | ❌ No (need alignment) | ❌ No | ❌ No |
| **Translation** | ✅ Yes | ❌ No | ✅ Yes (96 lang) | ✅ Prompt-based |
| **Long-form Chunking** | ✅ Built-in | ❌ Manual | ❌ Manual | ❌ Manual |
| **Japanese Quality** | ★★★★☆ | ★★★★★ | ★★★★☆ | ★★★☆☆ (TBD) |
| **Speed** | Medium | Fast | Medium | Slow |
| **VRAM** | 6GB | 1-4GB | 9GB | 14GB |
| **Pipeline Compatible** | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **Multi-task** | ❌ No | ❌ No | ❌ No | ✅ Yes |

---

## Implementation Priority

### Phase 1 (v2.0-alpha)
1. **CTCASR module** - Wav2Vec2/HuBERT/MMS support
2. **CTC timestamp alignment** - Using ctc-segmentation
3. **CLI integration** - `--mode ctc` with model selection

### Phase 2 (v2.0-beta)
4. **SeamlessASR module** - Translation integration
5. **Hiragana→Kanji conversion** - For HuBERT output
6. **Ensemble mixing** - CTC + Whisper combinations

### Phase 3 (v2.0-rc)
7. **AudioLLMPipeline** - Qwen2-Audio with prompts
8. **GUI integration** - Model selection in UI
9. **Performance benchmarks** - Compare all models

---

## Open Questions

1. **Japanese verification for Qwen2-Audio**: Documentation only mentions English. Need to test.
2. **Wav2Vec2-BERT quality**: CER 31.71% seems high - may need different model.
3. **ReazonSpeech NeMo vs Wav2Vec2**: NeMo has better docs but requires separate library.
4. **Scene detection compatibility**: Do CTC models benefit from scene detection?

---

## References

- [HuggingFace ASR Documentation](https://huggingface.co/docs/transformers/en/tasks/asr)
- [Wav2Vec2 Model Documentation](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2)
- [SeamlessM4T Paper](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/)
- [Qwen2-Audio GitHub](https://github.com/QwenLM/Qwen2-Audio)
- [CTC Segmentation](https://github.com/lumaku/ctc-segmentation)

---

## Decision

**Proceed with tiered architecture**:
1. Extend existing TransformersPipeline with model family detection
2. Create CTCASR module for CTC-based models
3. Create SeamlessASR module for SeamlessM4T
4. Create separate AudioLLMPipeline for Qwen2-Audio

This maximizes code reuse while properly handling each model family's unique requirements.

---

## Addendum: CTC-First Japanese Subtitling Architecture (v2.0 Focus)

### Scope Clarification

**Parked for later**: SeamlessM4T (Family B), Qwen2-Audio (Family C)

**Active focus**: CTC model family ONLY:
- `reazon-research/japanese-wav2vec2-large-rs35kh` (Primary candidate)
- `TKU410410103/hubert-large-japanese-asr` (Alternative)
- `facebook/mms-1b-all` with `jpn` adapter (Multilingual option)
- `sinhat98/w2v-bert-2.0-japanese-colab-CV16.0` (Needs validation)

### Use Case: Japanese JAV → English Subtitles

**Target workflow**:
```
Japanese Audio → CTC Transcription → Timestamped Japanese SRT → Translation → English SRT
```

**Fallback**: If translation quality is unreliable, output native Japanese SRT.

---

### CTC Timestamp Generation: Deep Analysis

#### The Core Problem

CTC models output character/token logits at each time step, then apply greedy decoding or beam search to produce text. The raw CTC output has **implicit temporal information** (each logit corresponds to a ~20ms audio frame), but standard decoding discards this.

#### Solution: CTC Segmentation (Self-Alignment)

The `ctc-segmentation` library (lumaku/ctc-segmentation) can extract timestamps from CTC inference:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CTC SEGMENTATION WORKFLOW                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. INFERENCE                                                            │
│     Audio (16kHz) → CTC Model → Logits (T × V matrix)                   │
│     T = time steps (~20ms each), V = vocabulary size                    │
│                                                                          │
│  2. GREEDY DECODE                                                        │
│     Logits → argmax → Token IDs → Processor.decode() → Text             │
│     (No timestamps at this stage)                                        │
│                                                                          │
│  3. CTC ALIGNMENT                                                        │
│     Logits + Text → ctc-segmentation → Character-level timestamps       │
│     Uses dynamic programming to find optimal alignment                   │
│                                                                          │
│  4. POST-PROCESSING                                                      │
│     Character timestamps → Word/Phrase segmentation → SRT segments      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: ctc-segmentation uses the SAME logits that produced the transcript, so it's essentially "self-alignment" - not dependent on a separate alignment model.

#### Code Implementation

```python
import ctc_segmentation
import torch
import numpy as np

def ctc_transcribe_with_timestamps(model, processor, audio, sample_rate=16000):
    """
    CTC transcription with timestamp extraction via ctc-segmentation.

    Returns:
        List[dict]: Segments with {text, start, end}
    """
    # 1. Prepare audio
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")

    # 2. Get logits (keep for alignment)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Convert to log probabilities for ctc-segmentation
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_np = log_probs.squeeze(0).cpu().numpy()

    # 3. Greedy decode to get transcript
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = processor.decode(predicted_ids[0])

    # 4. Prepare vocabulary mapping for ctc-segmentation
    vocab = processor.tokenizer.get_vocab()
    char_list = [None] * len(vocab)
    for char, idx in vocab.items():
        char_list[idx] = char

    # 5. Run ctc-segmentation
    config = ctc_segmentation.CtcSegmentationParameters(
        char_list=char_list,
        blank=processor.tokenizer.pad_token_id,  # CTC blank token
    )
    config.index_duration = audio.shape[0] / sample_rate / log_probs_np.shape[0]

    # Segment by sentences (or use custom segmentation)
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(
        config, [transcript]
    )

    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
        config, log_probs_np, ground_truth_mat
    )

    # 6. Extract segment boundaries
    segments = ctc_segmentation.determine_utterance_segments(
        config, utt_begin_indices, char_probs, timings, transcript
    )

    result = []
    for text, (start, end, score) in zip([transcript], segments):
        result.append({
            "text": text.strip(),
            "start": start,
            "end": end,
            "confidence": float(np.exp(score))  # Convert log prob to probability
        })

    return result
```

#### Why This Approach Works for Japanese

1. **Language-agnostic alignment**: ctc-segmentation aligns based on character/token positions in the logit matrix, not linguistic rules
2. **No separate alignment model needed**: Uses the same CTC logits that produced the transcript
3. **Self-consistent**: Transcript and timestamps come from the same model inference
4. **Japanese character support**: Works with hiragana, katakana, kanji (whatever the CTC model's vocabulary supports)

#### Comparison with WhisperX Approach

| Aspect | WhisperX | CTC-First (Proposed) |
|--------|----------|----------------------|
| **Primary transcription** | Whisper | CTC (Wav2Vec2/HuBERT) |
| **Timestamp source** | Whisper (rough) → CTC (refined) | CTC logits via ctc-segmentation |
| **Alignment model** | Separate Wav2Vec2 for alignment | Same model (self-alignment) |
| **Japanese alignment risk** | Uses jonatasgrosman (CER 20%) | Self-aligned (no external model) |
| **Hallucination handling** | VAD filtering | N/A (CTC doesn't hallucinate same way) |

**Critical difference**: WhisperX uses a DIFFERENT CTC model for alignment (jonatasgrosman), which introduces potential misalignment. Our approach uses the SAME model's logits, eliminating this risk.

---

### Proposed CTC Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CTC JAPANESE SUBTITLING PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT: Japanese Audio (JAV movie)                                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  STAGE 1: Audio Preprocessing (existing)                        │     │
│  │  - Scene detection (auditok/silero/semantic)                    │     │
│  │  - Optional speech enhancement (clearvoice/bs-roformer)         │     │
│  │  - VAD segmentation for long-form handling                      │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                            │                                             │
│                            ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  STAGE 2: CTC Transcription                                     │     │
│  │  ┌──────────────────────────────────────────────────────────┐  │     │
│  │  │  CTC Model (Wav2Vec2/HuBERT/MMS)                          │  │     │
│  │  │  - Load model + processor                                  │  │     │
│  │  │  - Model-specific preprocessing:                           │  │     │
│  │  │    • ReazonSpeech: 0.5s audio padding                     │  │     │
│  │  │    • MMS: load_adapter("jpn")                             │  │     │
│  │  │    • HuBERT: hiragana output                              │  │     │
│  │  │  - Forward pass → logits                                   │  │     │
│  │  │  - Greedy decode → Japanese text                          │  │     │
│  │  └──────────────────────────────────────────────────────────┘  │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                            │                                             │
│                            ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  STAGE 3: Timestamp Extraction (NEW)                            │     │
│  │  - Use ctc-segmentation with logits from Stage 2                │     │
│  │  - Character-level alignment → word/phrase boundaries           │     │
│  │  - Japanese-specific segmentation rules (particles, etc.)       │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                            │                                             │
│                            ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  STAGE 4: Post-Processing (existing + enhanced)                 │     │
│  │  - Japanese regrouping (reuse from stable_ts_asr.py)           │     │
│  │  - HuBERT: Optional hiragana → kanji conversion                │     │
│  │  - Hallucination removal (sanitizer.py)                         │     │
│  │  - SRT formatting                                               │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                            │                                             │
│                            ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  STAGE 5: Translation (existing - PySubtrans)                   │     │
│  │  - Japanese SRT → English SRT                                   │     │
│  │  - Uses DeepSeek/Gemini/Claude/GPT                             │     │
│  │  - Instruction files for context                                │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                            │                                             │
│                            ▼                                             │
│  OUTPUT: English SRT (or Japanese if translation disabled/failed)        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Implementation Files

```
whisperjav/
├── modules/
│   ├── ctc_asr/
│   │   ├── __init__.py
│   │   ├── base.py           # CTCASRBase protocol/interface
│   │   ├── wav2vec2.py       # Wav2Vec2ForCTC wrapper
│   │   ├── hubert.py         # HubertForCTC wrapper
│   │   ├── mms.py            # MMS with adapter management
│   │   └── timestamp.py      # ctc-segmentation wrapper
│   └── speech_segmentation/
│       └── backends/
│           └── ctc_segmentation.py  # Integrate with existing VAD structure
├── pipelines/
│   └── ctc_pipeline.py       # New pipeline using CTC models
└── config/
    └── v4/
        └── ecosystems/
            └── ctc/
                └── models/
                    ├── reazon-wav2vec2.yaml
                    ├── hubert-japanese.yaml
                    └── mms-japanese.yaml
```

---

### Dependencies

Add to requirements:
```
ctc-segmentation>=1.7.0
```

The library is pure Python with NumPy, minimal footprint.

---

### CLI Integration

```bash
# CTC mode with default model (ReazonSpeech)
whisperjav video.mp4 --mode ctc

# CTC with specific model
whisperjav video.mp4 --mode ctc --ctc-model reazon-research/japanese-wav2vec2-large-rs35kh
whisperjav video.mp4 --mode ctc --ctc-model TKU410410103/hubert-large-japanese-asr
whisperjav video.mp4 --mode ctc --ctc-model facebook/mms-1b-all

# CTC with translation
whisperjav video.mp4 --mode ctc --translate

# HuBERT with kanji conversion
whisperjav video.mp4 --mode ctc --ctc-model TKU410410103/hubert-large-japanese-asr --kanji-convert
```

---

### Advantages of CTC-First Approach

1. **Better Japanese transcription**: ReazonSpeech CER 11% vs Whisper-based models
2. **No hallucination risk**: CTC models don't "dream" content like seq2seq
3. **Self-consistent timestamps**: Same model produces text and timing
4. **Lighter VRAM**: Wav2Vec2 ~1.5GB vs Whisper ~6GB
5. **Faster inference**: CTC is single-pass, no autoregressive decoding
6. **Translation via existing PySubtrans**: No new translation infrastructure needed

### Limitations to Address

1. **Long-form handling**: CTC models typically have fixed context (~30s), need chunking
2. **HuBERT hiragana output**: May need kanji conversion for readability
3. **No built-in translation**: Relies on external translation (PySubtrans)
4. **Punctuation**: CTC models may not produce punctuation, need post-processing

---

## Summary: CTC vs WhisperX vs Whisper-Only

| Approach | Transcription | Timestamps | Translation | Japanese Quality | VRAM |
|----------|---------------|------------|-------------|------------------|------|
| **Whisper-Only** (current) | Whisper | Native (segment) | Built-in | Good | 6GB |
| **WhisperX** | Whisper | CTC alignment (external model) | Built-in | Good (transcription), Risky (alignment) | 8GB |
| **CTC-First** (proposed) | CTC (ReazonSpeech) | CTC self-alignment | PySubtrans | Excellent | 1.5GB |

### Key Architectural Decision

**CTC-First** is recommended for Japanese JAV subtitling because:

1. **Self-alignment eliminates misalignment risk**: WhisperX uses a different CTC model (jonatasgrosman, CER 20%) for alignment. Our approach uses the same model's logits for both transcription and timestamps.

2. **Better Japanese accuracy**: ReazonSpeech (CER 11%) outperforms most Whisper variants for Japanese.

3. **No hallucination**: CTC models can't "dream" non-existent content because they're constrained to their vocabulary at each frame.

4. **Lighter resource usage**: ~1.5GB VRAM vs ~6GB for Whisper models.

5. **Compatible with existing translation**: Japanese SRT → PySubtrans → English SRT uses the already-integrated translation module.

### Next Steps

1. **Validate ctc-segmentation with ReazonSpeech**: Quick proof-of-concept to verify timestamps quality
2. **Implement CTCPipeline**: New pipeline class following existing architecture patterns
3. **Add model configs**: YAML configurations for each CTC model
4. **Test with real JAV content**: Validate transcription quality and timestamp accuracy
5. **Integration**: CLI arguments, GUI model selection

---

## Appendix: ctc-segmentation Library Verification

Per library documentation:
- **Japanese support**: "For asian languages, no changes to CTC segmentation parameters necessary"
- **Performance**: ~400ms alignment time for 500 seconds of audio
- **Compatibility**: Works with Wav2Vec2, HuBERT, NeMo, SpeechBrain, ESPnet
- **Output**: Character-level alignments with confidence scores

Repository: https://github.com/lumaku/ctc-segmentation
License: BSD-3-Clause
Dependencies: NumPy only
