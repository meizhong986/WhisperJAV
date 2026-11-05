# ChronosJAV 处理管线

ChronosJAV 是一个为动漫和 JAV 内容设计的处理管线，运行两种 ASR 模型：Qwen3-ASR（其模型菜单中另有两个日语微调版本，其中一个用游戏语音微调），以及为动漫和视觉小说对白制作的 anime-whisper。

灵感来自 [ChronusOmni](https://arxiv.org/abs/2512.09841)（Chen 等，2025）的时间感知方法。

---

## 可用模型

在 **专家模式**（Ensemble Mode）标签页中，**处理管线**（Pipeline）菜单的 **ChronosJAV** 组有两个处理管线：**Qwen3-ASR** 和 **Anime-Whisper**。这一轮的 **模型**（Model）菜单提供：

| 处理管线 | 模型菜单中的选项 | 菜单标出的显存 |
|------|------|------|
| **Qwen3-ASR** | Qwen3-ASR-1.7B（默认） | 8 GB |
| | Qwen3-ASR-0.6B | 4 GB |
| | JA Anime-Galgame 1.7B（jaykwok 的微调模型） | 8 GB |
| | JA-tuned 1.7B（neosophie） | 8 GB |
| **Anime-Whisper** | anime-whisper（默认） | ~4 GB |
| | efwkjn/whisper-ja-anime-v0.3（实验性，#232） | ~3 GB |
| | Kotoba v2.0 | ~2 GB |
| | Kotoba v2.1 | ~2 GB |

!!! tip "建议"
    如果显卡显存不足 8 GB，请选择菜单标出显存更少的模型：Qwen3-ASR-0.6B（4 GB）或 Anime-Whisper 的模型（2 到 4 GB）。

---

## 使用方法

### GUI

1. 打开 **专家模式**（Ensemble Mode）选项卡（**基础模式**（Transcription Mode）选项卡中没有 ChronosJAV）
2. 在第一轮或第二轮的 **处理管线**（Pipeline）菜单中，选择 **ChronosJAV** 组里的 **Qwen3-ASR** 或 **Anime-Whisper**；这一轮的灵敏度和语音分割器会随之改为该处理管线自己的设置
3. 在这一轮的 **模型**（Model）菜单中选择模型
4. 点击 **开始**（Start）

命令行：`whisperjav video.mp4 --mode qwen`（Qwen3-ASR），或 `--mode qwen --qwen-generator anime-whisper`。

### 作为专家模式的一部分

ChronosJAV 可以作为第一轮、第二轮或两轮，另一轮可用任何其他处理管线。勾选 **第二轮**（Pass 2）时，两轮的字幕按 **合并**（Merge）中选择的策略合并（默认 **第一轮 + 用第二轮填补**（Pass 1 + Fill From Pass 2）；见[专家模式](ensemble.md)）。

---

## 技术细节

ChronosJAV 使用与标准 Whisper 处理管线不同的默认设置：

| 设置项 | ChronosJAV 默认值 | 标准默认值 |
|--------|-------------------|------------|
| **解码方式** | 贪心解码（beam=1） | 因处理管线而异（Balanced：集束搜索，2 个 beam） |
| **语音分割器** | FireRedVAD、TEN VAD 或 WhisperSeg，按轮次而定（见下文） | 因处理管线而异 |
| **时间戳模式** | 仅 VAD（从 1.10.1 起，与 1.9.5 相同；1.10.0：对齐器 + VAD 回退） | Whisper 模型自己给出的时间 |
| **文本清理器**（Text Cleaner） | Qwen3-ASR：Qwen3 AssemblyTextCleaner；anime-whisper：它自己的文本清理器 | — |
| **生成字幕后的清理** | ChronosJAV 自己的清理：删除「あ。」这类单独成行的字幕、只有呻吟或呼吸声的字幕，以及两个场景交界处的重叠等；没有幻觉过滤器 | 清洗器（sanitizer），含幻觉过滤器 |

ChronosJAV 的一轮开始时使用的语音分割器：在 **专家模式**（Ensemble Mode）标签页中，Qwen3-ASR 作为第一轮时为 FireRedVAD，Anime-Whisper 作为第一轮时为 WhisperSeg，两者作为第二轮时都为 TEN VAD。在命令行中：`--mode qwen` 为 WhisperSeg；专家模式运行与 GUI 相同。这一轮的 **语音分割器**（Speech Segmenter）菜单、`--pass1-speech-segmenter` / `--pass2-speech-segmenter` 或 `--qwen-segmenter` 可以更改它。

---

## 字幕时间 {#subtitle-timing-1100}

ChronosJAV 的 ASR 模型只输出文字，不带时间。从 1.10.1 起，默认情况下，Qwen3-ASR 和 anime-whisper 的字幕开始和结束时间取自语音分割器找到的那段语音（VAD 组），与 1.9.5 相同：不运行对齐器和计时模型，也不下载它们。一个 VAD 组里的两句话（例如两个人说的话）会合成一条字幕。

1.10.0 默认用另外两个步骤为文字标出时间；从 1.10.1 起它们只在选用时运行：

1. **对齐器（aligner）** 为 ASR 模型写出的文字逐字标出时间。没有指定其他对齐器时使用 `wav2vec2-ja`（232 MB）。
2. **计时模型（timing model）** Whisper small（488 MB）在每个 VAD 组上运行，它自己写出的文字不使用。在 Whisper 开始一个新段落的地方，这个 VAD 组会在文字中离该处最近的标点（如 。？！…、 或右括号）或空格处切分成几条字幕。没有这样的标点就不切分。从不按字数或秒数切分；切分也从不让某一部分单独成为一条、随后被 WhisperJAV 的清理步骤删除的字幕（例如单独的喘息声或单独的「はい」），这样的部分会留在相邻的字幕里。每条字幕从对齐器标出的第一个字的时间开始，在下一条开始处结束；最后一条在这个 VAD 组结束处结束。对齐器没有给出时间的地方，计时模型会把开始时间往后移，移到 Whisper 听到语音开始的地方。

两者都在随 WhisperJAV 一起安装的 CrispASR 中运行，在 ChronosJAV 第一次使用它们时下载。它们在本安装的 CrispASR 运行的地方运行（见 [它们在哪里运行](crispasr.md#where-they-run)），因此在"通过 Vulkan 使用 GPU"或 Apple GPU 的安装上，它们会请求使用 GPU（两者都尚未试过），而 ASR 模型在 CPU 上运行。

| 选项 | GUI（自定义参数 → 对齐；英文界面为 Customize Parameters → Alignment） | 命令行 |
|---|---|---|
| 仅 VAD（默认） | **时间戳模式**（Timestamp Mode）：仅 VAD（不加载对齐器）（VAD Only (no aligner loaded)） | `--qwen-timestamp-mode vad_only`（默认） |
| 对齐器 | **时间戳模式**：对齐器 + VAD 回退（Aligner + VAD Fallback）、对齐器 + 插值（Aligner + Interpolation）或仅对齐器（Aligner Only）；然后 **对齐器后端**（Aligner Backend）：CrispASR（GGUF，推荐）或 Qwen3 ForcedAligner（PyTorch），以及 **对齐模型**（Aligner Model） | `--qwen-timestamp-mode aligner_vad_fallback`（或 `aligner_interpolation`、`aligner_only`），配合 `--qwen-aligner wav2vec2-ja`（没有指定时使用）、`qwen3-fa-gguf`、`parakeet-ctc-ja` 或 `Qwen/Qwen3-ForcedAligner-0.6B`。只给 `--qwen-aligner` 而不给 `--qwen-timestamp-mode` 时，会开启 `aligner_vad_fallback` |
| 计时模型 | **计时模型（Whisper）**（Timing Model (Whisper)）：Whisper small、无（不用计时模型）（None (no timing model)；默认） | `--qwen-timing-model whisper-small`，或 `none`（默认） |
| Step-Down | **自适应逐级缩小**（Adaptive Step-Down；默认关闭；选对齐器模式时显示） | `--qwen-stepdown`（默认关闭） |

在命令行中专家模式的每一轮中，同样的选项写在 `--pass1-qwen-params` 里，例如用 `'{"timestamp_mode": "aligner_vad_fallback", "timing_model": "whisper-small"}'` 得到 1.10.0 的时间方式。只给 `aligner_id`，或给 `crispasr`、`qwen3` 作为 `aligner_backend`，而不给 `timestamp_mode` 时，也会开启 `aligner_vad_fallback`。

- 不推荐 `parakeet-ctc-ja`：在我们的测试中，它给出的时间是均匀分布的。
- 不用对齐器时，选用的计时模型仍会移动每条字幕的开始时间，但不切分。
- 当 **字幕重新分组**（Subtitle Regrouping；**输出**（Output）标签页；`--qwen-regroup`）设为 **标准**（Standard）或 **仅按句子**（Sentence Only）而不是 **关闭**（Off，默认）时，由重新分组来切分。在对齐器模式下，开始时间取自对齐器；选用的计时模型只移动对齐器没有给出时间的 VAD 组的开始时间，但它仍会运行。在仅 VAD 模式下，重新分组会切分每个 VAD 组的文字，并按文字长度分配该组的时间。
- 想得到与 1.10.0 相同的 ChronosJAV 时间（对齐器和计时模型）：`--qwen-timestamp-mode aligner_vad_fallback --qwen-timing-model whisper-small`，或在 GUI 中把 **时间戳模式**（Timestamp Mode）选 **对齐器 + VAD 回退**（Aligner + VAD Fallback）、**计时模型（Whisper）**（Timing Model (Whisper)）选 **Whisper small**。在 1.10.0 中从 ChronosJAV 的 **自定义参数**（Customize Parameters）窗口保存的 GUI 预设保留保存时的时间戳模式和计时模型；按 1.10.0 默认值保存的预设仍会运行对齐器和计时模型。
- **出问题时文字会保留。** 如果对齐器无法加载（例如第一次运行时没有网络），或对某个 VAD 组没有给出时间，这些字幕不切分，时间取自其 VAD 组；计时模型仍可能移动其开始时间。如果计时模型无法加载，则不切分，时间取自对齐器。运行汇总会说明是哪一部分以及原因。
- 使用 **对齐器 + VAD 回退**（Aligner + VAD Fallback）且其他设置为默认值时，如果对齐器把一个场景的词都挤在一起（坍塌检查），这个场景的字幕时间取自其 VAD 组，不切分；选用的计时模型会移动其开始时间。Step-Down 会改为用最长 6 秒的 VAD 组重新处理这样的场景，它默认是关闭的。
- 使用 `--device cpu` 时，选用的对齐器和计时模型不能被限制在 CPU 上：在 CrispASR 的版本会使用 GPU 的地方，它们仍会请求使用 GPU，运行开始时的提示框会说明这一点。在没有 GPU 的电脑上它们在 CPU 上运行；计时模型在那里需要多长时间尚未测量。
- 1.10.0 的时间方式（对齐器和计时模型），在《全裸导演》（The Naked Director）的 7 个场景上，以 Netflix 的日语字幕为参考测量：开始时间与参考相差 0.5 秒以内的字幕，Qwen3-ASR 为 78%（1.9.5：75%），anime-whisper 为 77%（1.9.5：63%）；文字得分与 1.9.5 相同。一条典型字幕的开始仍比参考晚约 0.06 到 0.09 秒；1.9.5 的字幕（与 1.10.1 的默认方式一样，按 VAD 组计时）则略早于参考；1.10.1 的默认方式尚未测量。

---

## 首次运行

首次使用时，模型将从 HuggingFace 下载（约 2-4 GB，取决于模型）。这是一次性下载 — 后续运行将使用缓存的模型。对齐器（232 MB）和计时模型（488 MB）只在运行选用它们时才下载（从 1.10.1 起；1.10.0 在第一次 ChronosJAV 运行时下载两者）。

模型缓存在你的 HuggingFace 缓存目录中：

- **Windows：** `C:\Users\<you>\.cache\huggingface\`
- **macOS/Linux：** `~/.cache/huggingface/`
