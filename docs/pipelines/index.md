# Pipelines

WhisperJAV offers multiple processing pipelines, each trading speed for accuracy. Choose based on your content and hardware.

---

## Pipeline Comparison

| Pipeline | Backend | Scene Detection | VAD | Speed | Accuracy | GPU Memory |
|----------|---------|-----------------|-----|-------|----------|------------|
| **Faster** | Faster-Whisper | No | No | Fastest | Good | ~2 GB |
| **Fast** | Whisper | Yes | No | Fast | Better | ~4 GB |
| **Balanced** | Whisper | Yes | Yes | Medium | Best (Whisper) | ~4 GB |
| **Fidelity** | Whisper | Yes | Full | Slow | Maximum | ~6 GB |
| **Transformers** | HuggingFace | Yes | Yes | Medium | Good | ~4 GB |
| **Qwen3-ASR** | Qwen3 | Assembly | Assembly | Medium | Excellent text | ~4-8 GB |
| **ChronosJAV** | anime-whisper / Kotoba | TEN VAD | TEN VAD | Medium | Best for anime/JAV | ~4-8 GB |

---

## Which Pipeline Should I Use?

| Scenario | Recommended Pipeline |
|----------|---------------------|
| First time, just want subtitles | **Balanced** (default) |
| Processing many files quickly | **Faster** |
| Anime or JAV content | **ChronosJAV** with anime-whisper |
| Maximum accuracy, don't mind waiting | **Ensemble**: Balanced + Qwen3-ASR with Smart Merge |
| No GPU / CPU only | **Faster** with CPU-only mode |
| Apple Silicon Mac | **Transformers** (MPS acceleration) |

---

## Ensemble Mode

Run two passes with different pipelines and merge the results. See [Ensemble Mode](ensemble.md) for details.

## Specialized Pipelines

- [ChronosJAV](chronosjav.md) — anime-whisper and Kotoba models for anime/JAV content
- [Qwen3-ASR](qwen3-asr.md) — alternative ASR engine with strong Japanese text quality
