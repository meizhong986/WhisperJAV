# Qwen3-ASR Pipeline

Qwen3-ASR is an alternative ASR engine based on Alibaba's Qwen architecture, offering strong Japanese text quality with a different approach to speech recognition.

---

## Models

| Model | Size | Notes |
|-------|------|-------|
| **Qwen3-ASR 1.7B** | ~4 GB | Full model, best quality |
| **Qwen3-ASR 0.6B** | ~2 GB | Smaller, faster, slightly lower quality |

---

## How to Use

### GUI (Ensemble Tab)

1. Go to the **Ensemble** tab
2. Set **Pipeline** to **Qwen3-ASR**
3. Select model size
4. Click **Start**

### CLI

```bash
whisperjav video.mp4 --mode qwen
```

### As Ensemble Pass 2

Qwen3-ASR pairs well with Whisper-based pipelines:

1. **Pass 1:** Balanced (Whisper — good timing)
2. **Pass 2:** Qwen3-ASR (good text quality)
3. **Merge Strategy:** Smart Merge

---

## Requirements

- **HuggingFace extra** must be installed: `pip install whisperjav[huggingface]`
- Requires `transformers` and `accelerate` packages
- First run downloads the model from HuggingFace

---

## Strengths and Limitations

**Strengths:**

- Excellent Japanese text quality
- Good handling of casual/colloquial speech
- Strong punctuation and sentence structure

**Limitations:**

- Timing can differ from Whisper-based pipelines
- Apple Silicon: currently CPU-only (MPS not yet supported for forced aligner)
- Requires the HuggingFace extras installed
