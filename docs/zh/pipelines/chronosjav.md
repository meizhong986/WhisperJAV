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
| **Cohere-Transcribe** *(推迟到 v1.9.0)* | ~4-8 GB | Cohere Labs 2B Conformer ASR。**目前在 GUI 下拉菜单中已禁用** — 该模型需要 transformers ≥ 5.4.0，与捆绑的 Qwen3-ASR fork 冲突。v1.9.0 将协同执行 transformers 升级并与现有生成器一起交付 Cohere。详细状态请参阅 [FAQ](../faq.md#cohere-transcribe预览--推迟到-v190)。 |

!!! tip "建议"
    建议先使用 **anime-whisper** 以获得最佳效果。如果需要更快的处理速度或 GPU 显存有限，可切换至 Kotoba。Cohere-Transcribe 计划在 v1.9.0 交付；v1.8.14 中下拉菜单的入口可见但已置灰，作为前向预告。

---

## 使用方法

### GUI

1. 进入 **专家模式**（Ensemble）选项卡
2. 将 **Pipeline** 设置为 **ChronosJAV**
3. 从下拉菜单中选择一个**模型**
4. 点击 **Start**

### 作为专家模式的一部分

若要获得最高质量，可将 ChronosJAV 与另一个处理管线组合使用：

1. **第一轮：** ChronosJAV 搭配 anime-whisper
2. **第二轮：** Qwen3-ASR 或 Balanced
3. **合并策略：** Smart Merge

---

## 技术细节

ChronosJAV 使用与标准 Whisper 处理管线不同的默认设置：

| 设置项 | ChronosJAV 默认值 | 标准默认值 |
|--------|-------------------|------------|
| **解码方式** | 贪心解码（beam=1） | 集束搜索（beam=5） |
| **语音分割器** | TEN VAD | Silero v6.2 |
| **时间戳模式** | 对齐器 + VAD 回退（1.10.0；之前为仅 VAD） | 完整对齐 |
| **清洗器** | 直通（Passthrough） | 标准清洗器 |

这些默认值专为动漫/JAV 内容优化。贪心解码搭配 TEN VAD 语音分割可产生更紧凑的字幕时间轴，并消除过大的字幕块。

---

## 字幕时间（1.10.0） {#subtitle-timing-1100}

ChronosJAV 的 ASR 模型只输出文字，不带时间。在 1.10.0 之前，Qwen3-ASR 和 anime-whisper 的字幕开始和结束时间取自语音分割器找到的那段语音（VAD 组），一个 VAD 组里的两句话（例如两个人说的话）会合成一条字幕。从 1.10.0 起，默认情况下：

1. **对齐器（aligner）** 为 ASR 模型写出的文字逐字标出时间。默认是 `wav2vec2-ja`（232 MB）。
2. **时间模型（timing model）** Whisper small（488 MB）在每个 VAD 组上运行，它自己写出的文字不使用。在 Whisper 开始一个新段落的地方，这个 VAD 组会在文字中离该处最近的标点（如 。？！…、 或右括号）或空格处切分成几条字幕。没有这样的标点就不切分。从不按字数或秒数切分；切分也从不让某一部分单独成为一条、随后被 WhisperJAV 的清理步骤删除的字幕（例如单独的喘息声或单独的「はい」），这样的部分会留在相邻的字幕里。每条字幕从对齐器标出的第一个字的时间开始，在下一条开始处结束；最后一条在这个 VAD 组结束处结束。对齐器没有给出时间的地方，时间模型会把开始时间往后移，移到 Whisper 听到语音开始的地方。

两者都在随 WhisperJAV 一起安装的 CrispASR 中运行，在 ChronosJAV 第一次需要时下载。它们在本安装的 CrispASR 运行的地方运行（见 [它们在哪里运行](crispasr.md#where-they-run)），因此在"通过 Vulkan 使用 GPU"或 Apple GPU 的安装上，它们会请求使用 GPU（两者都尚未试过），而 ASR 模型在 CPU 上运行。

| 选项 | GUI（Customize → Alignment） | 命令行 |
|---|---|---|
| 对齐器 | **Aligner Backend**：CrispASR (GGUF, recommended)、Qwen3 ForcedAligner (PyTorch)、None；**Aligner Model** | `--qwen-aligner wav2vec2-ja`（默认）、`qwen3-fa-gguf`、`parakeet-ctc-ja` 或 `Qwen/Qwen3-ForcedAligner-0.6B` |
| 不用对齐器 | Aligner Backend 选 **None**（运行时使用仅 VAD，并关闭 Step-Down） | `--qwen-timestamp-mode vad_only` |
| 时间模型 | **Timing Model (Whisper)**：Whisper small (recommended)、None (no timing model) | `--qwen-timing-model whisper-small`（默认）或 `none` |
| Step-Down | **Adaptive Step-Down**（默认关闭） | `--qwen-stepdown`（默认关闭） |

在命令行中专家模式的每一轮中，同样的选项写在 `--pass1-qwen-params` 里，例如 `'{"aligner_backend": "crispasr", "aligner_id": "qwen3-fa-gguf", "timing_model": "none"}'`。

- 不推荐 `parakeet-ctc-ja`：在我们的测试中，它给出的时间是均匀分布的。
- 不用对齐器时，时间模型仍会移动每条字幕的开始时间，但不切分。
- 当 **Subtitle Regrouping**（Output 标签页；`--qwen-regroup`）设为 Standard 或 Sentence Only 而不是 Off（默认）时，由重新分组来切分，开始时间取自对齐器；这时时间模型只移动对齐器没有给出时间的 VAD 组的开始时间，但它仍会运行。
- 想得到与 1.9.5 相同的 ChronosJAV 时间（取自 VAD 组）：`--qwen-timestamp-mode vad_only --qwen-timing-model none`，或在 GUI 中把 Aligner Backend 和 Timing Model 都选 None。
- **出问题时文字会保留。** 如果对齐器无法加载（例如第一次运行时没有网络），或对某个 VAD 组没有给出时间，这些字幕不切分，时间取自其 VAD 组；时间模型仍可能移动其开始时间。如果时间模型无法加载，则不切分，时间取自对齐器。运行汇总会说明是哪一部分以及原因。
- 如果对齐器把一个场景的词都挤在一起（坍塌检查），在默认设置下，这个场景的字幕时间取自其 VAD 组，开始时间由时间模型移动，不切分。Step-Down 会改为用最长 6 秒的 VAD 组重新处理这样的场景，它默认是关闭的。
- 使用 `--device cpu` 时，对齐器和时间模型不能被限制在 CPU 上：在 CrispASR 的版本会使用 GPU 的地方，它们仍会请求使用 GPU，运行开始时的提示框会说明这一点。在没有 GPU 的电脑上它们在 CPU 上运行；时间模型在那里需要多长时间尚未测量。
- 在《全裸导演》（The Naked Director）的 7 个场景上，以 Netflix 的日语字幕为参考测量：开始时间与参考相差 0.5 秒以内的字幕，Qwen3-ASR 为 78%（1.9.5：75%），anime-whisper 为 77%（1.9.5：63%）；文字得分与 1.9.5 相同。一条典型字幕的开始仍比参考晚约 0.06 到 0.09 秒；1.9.5 的则略早于参考。

---

## 首次运行

首次使用时，模型将从 HuggingFace 下载（约 2-4 GB，取决于模型）。这是一次性下载 — 后续运行将使用缓存的模型。1.10.0 的第一次 ChronosJAV 运行还会下载对齐器（232 MB）和时间模型（488 MB）。

模型缓存在你的 HuggingFace 缓存目录中：

- **Windows：** `C:\Users\<you>\.cache\huggingface\`
- **macOS/Linux：** `~/.cache/huggingface/`
