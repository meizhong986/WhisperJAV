# ChronosJAV Pipeline

ChronosJAV is a dedicated pipeline for anime and JAV content, built around speech models specifically trained on Japanese anime and adult content dialogue.

Inspired by the temporal-awareness approach in [ChronusOmni](https://arxiv.org/abs/2512.09841) (Chen et al., 2025).

---

## Available Models

| Model | Size | Strengths |
|-------|------|-----------|
| **anime-whisper** | ~4 GB | Best quality for anime/JAV dialogue. Fine-tuned Whisper large-v3. |
| **Kotoba v2.1** | ~2 GB | Lighter weight with punctuation support. Good balance of speed and quality. |
| **Kotoba v2.0** | ~2 GB | Lighter weight, no punctuation. Fastest of the three. |

!!! tip
    Start with **anime-whisper** for best results. Switch to Kotoba if you need faster processing or have limited GPU memory.

---

## How to Use

### GUI

1. Go to the **Ensemble** tab
2. Set **Pipeline** to **ChronosJAV**
3. Select a **Model** from the dropdown
4. Click **Start**

### As Part of Ensemble

For maximum quality, combine ChronosJAV with another pipeline:

1. **Pass 1:** ChronosJAV with anime-whisper
2. **Pass 2:** Qwen3-ASR or Balanced
3. **Merge Strategy:** Smart Merge

---

## Technical Details

ChronosJAV uses different defaults than the standard Whisper pipelines:

| Setting | ChronosJAV Default | Standard Default |
|---------|-------------------|-----------------|
| **Decoding** | Greedy (beam=1) | Beam search (beam=5) |
| **Speech Segmenter** | TEN VAD | Silero v6.2 |
| **Timestamp Mode** | VAD-only | Full alignment |
| **Cleaner** | Passthrough | Standard sanitizer |

These defaults are optimized for anime/JAV content. The greedy decoding with TEN VAD segmentation produces tighter subtitle timing and eliminates oversized subtitle blocks.

---

## First Run

On first use, the model is downloaded from HuggingFace (~2-4 GB depending on model). This is a one-time download — subsequent runs use the cached model.

Models are cached in your HuggingFace cache directory:

- **Windows:** `C:\Users\<you>\.cache\huggingface\`
- **macOS/Linux:** `~/.cache/huggingface/`
