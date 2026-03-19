# Qwen3-ASR 处理管线

Qwen3-ASR 是一个基于阿里巴巴 Qwen 架构的替代语音识别引擎，以不同的语音识别方法提供高质量的日语文本输出。

---

## 模型

| 模型 | 大小 | 备注 |
|------|------|------|
| **Qwen3-ASR 1.7B** | ~4 GB | 完整模型，最佳质量 |
| **Qwen3-ASR 0.6B** | ~2 GB | 更小、更快，质量略低 |

---

## 使用方法

### GUI（Ensemble 选项卡）

1. 进入 **Ensemble** 选项卡
2. 将 **Pipeline** 设置为 **Qwen3-ASR**
3. 选择模型大小
4. 点击 **Start**

### CLI

```bash
whisperjav video.mp4 --mode qwen
```

### 作为集成模式的 Pass 2

Qwen3-ASR 与基于 Whisper 的处理管线配合良好：

1. **Pass 1：** Balanced（Whisper — 良好的时间轴）
2. **Pass 2：** Qwen3-ASR（良好的文本质量）
3. **合并策略：** Smart Merge

---

## 系统要求

- 必须安装 **HuggingFace 扩展**：`pip install whisperjav[huggingface]`
- 需要 `transformers` 和 `accelerate` 包
- 首次运行时会从 HuggingFace 下载模型

---

## 优势与局限

**优势：**

- 出色的日语文本质量
- 对口语化/日常对话处理良好
- 标点符号和句子结构表现强

**局限：**

- 时间轴可能与基于 Whisper 的处理管线有差异
- Apple Silicon：目前仅支持 CPU（MPS 尚不支持强制对齐器）
- 需要安装 HuggingFace 扩展
