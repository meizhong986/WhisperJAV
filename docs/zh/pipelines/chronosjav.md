# ChronosJAV 处理管线

ChronosJAV 是一个专为动漫和 JAV 内容设计的处理管线，基于专门针对日语动漫和成人内容对话训练的语音模型构建。

灵感来自 [ChronusOmni](https://arxiv.org/abs/2512.09841)（Chen 等，2025）的时间感知方法。

---

## 可用模型

| 模型 | 大小 | 优势 |
|------|------|------|
| **anime-whisper** | ~4 GB | 动漫/JAV 对话的最佳质量。基于 Whisper large-v3 微调。 |
| **Kotoba v2.1** | ~2 GB | 更轻量，支持标点符号。速度与质量的良好平衡。 |
| **Kotoba v2.0** | ~2 GB | 更轻量，无标点符号。三者中速度最快。 |

!!! tip "建议"
    建议先使用 **anime-whisper** 以获得最佳效果。如果需要更快的处理速度或 GPU 显存有限，可切换至 Kotoba。

---

## 使用方法

### GUI

1. 进入 **Ensemble** 选项卡
2. 将 **Pipeline** 设置为 **ChronosJAV**
3. 从下拉菜单中选择一个**模型**
4. 点击 **Start**

### 作为集成模式的一部分

若要获得最高质量，可将 ChronosJAV 与另一个处理管线组合使用：

1. **Pass 1：** ChronosJAV 搭配 anime-whisper
2. **Pass 2：** Qwen3-ASR 或 Balanced
3. **合并策略：** Smart Merge

---

## 技术细节

ChronosJAV 使用与标准 Whisper 处理管线不同的默认设置：

| 设置项 | ChronosJAV 默认值 | 标准默认值 |
|--------|-------------------|------------|
| **解码方式** | 贪心解码（beam=1） | 集束搜索（beam=5） |
| **语音分割器** | TEN VAD | Silero v6.2 |
| **时间戳模式** | 仅 VAD | 完整对齐 |
| **清洗器** | 直通（Passthrough） | 标准清洗器 |

这些默认值专为动漫/JAV 内容优化。贪心解码搭配 TEN VAD 语音分割可产生更紧凑的字幕时间轴，并消除过大的字幕块。

---

## 首次运行

首次使用时，模型将从 HuggingFace 下载（约 2-4 GB，取决于模型）。这是一次性下载 — 后续运行将使用缓存的模型。

模型缓存在你的 HuggingFace 缓存目录中：

- **Windows：** `C:\Users\<you>\.cache\huggingface\`
- **macOS/Linux：** `~/.cache/huggingface/`
