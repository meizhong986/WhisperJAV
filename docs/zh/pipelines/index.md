# 处理管线

WhisperJAV 提供多种处理管线，在速度和准确度之间各有取舍。请根据你的内容类型和硬件条件进行选择。

---

## 处理管线对比

| 处理管线 | 后端 | 场景检测 | 语音活动检测 | 速度 | 准确度 | GPU 显存 |
|----------|------|----------|-------------|------|--------|----------|
| **Faster** | Faster-Whisper | 无 | 无 | 最快 | 良好 | ~2 GB |
| **Fast** | Whisper | 有 | 无 | 快 | 较好 | ~4 GB |
| **Balanced** | Whisper | 有 | 有 | 中等 | 最佳（Whisper） | ~4 GB |
| **Fidelity** | Whisper | 有 | 完整 | 慢 | 最高 | ~6 GB |
| **Transformers** | HuggingFace | 有 | 有 | 中等 | 良好 | ~4 GB |
| **Qwen3-ASR** | Qwen3 | Assembly | Assembly | 中等 | 文本质量优秀 | ~4-8 GB |
| **ChronosJAV** | anime-whisper / Kotoba | TEN VAD | TEN VAD | 中等 | 动漫/JAV 最佳 | ~4-8 GB |

---

## 我应该选择哪种处理管线？

| 使用场景 | 推荐处理管线 |
|----------|-------------|
| 首次使用，只需生成字幕 | **Balanced**（默认） |
| 需要快速处理大量文件 | **Faster** |
| 动漫或 JAV 内容 | **ChronosJAV** 搭配 anime-whisper |
| 追求最高准确度，不介意等待 | **集成模式**：Balanced + Qwen3-ASR 配合 Smart Merge |
| 无 GPU / 仅 CPU | **Faster** 并启用仅 CPU 模式 |
| Apple Silicon Mac | **Transformers**（MPS 加速） |

---

## 集成模式

使用两种不同的处理管线分别运行，然后合并结果。详情请参阅[集成模式](ensemble.md)。

## 专用处理管线

- [ChronosJAV](chronosjav.md) — 使用 anime-whisper 和 Kotoba 模型，专为动漫/JAV 内容优化
- [Qwen3-ASR](qwen3-asr.md) — 替代语音识别引擎，日语文本质量出色
