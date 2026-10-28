# 专家模式

专家模式在同一视频上运行两个不同的ASR 处理管线，并合并其输出以获得更高的准确率。这是 WhisperJAV 最强大的转写模式。

---

## 工作原理

```
Video → Pass 1 (Pipeline A) → SRT₁ ─┐
                                      ├→ Merge → Final SRT
Video → Pass 2 (Pipeline B) → SRT₂ ─┘
```

每个处理管线都有不同的优势。例如：

- **Whisper**（Balanced 模式）：出色的时间轴，良好的文本质量
- **Qwen3-ASR**：出色的文本质量，不同的时间轴方案

合并可以将两者的优势结合起来。

---

## 设置专家模式

1. 进入 **专家模式**（Ensemble）选项卡（第 3 个选项卡）
2. **第一轮** 始终处于激活状态 — 配置处理管线、灵敏度和选项
3. 勾选复选框以**启用第二轮**
4. 为第二轮配置不同的处理管线
5. 选择一种**合并策略**
6. 点击 **Start**

### Pass 配置

每一轮拥有相同的控件：

| 控件 | 说明 |
|------|------|
| **Pipeline** | ASR 后端（Balanced、Fast、Faster、Qwen3-ASR、ChronosJAV 等） |
| **Sensitivity** | 检测阈值（高灵敏、中灵敏、低灵敏） |
| **Scene Detector** | 音频场景检测方式（Auditok、Silero、Semantic、None） |
| **Speech Enhancer** | 音频预处理 / 语音增强（None、FFmpeg DSP、ZipEnhancer、htdemucs） |
| **Speech Segmenter** | 场景内的语音活动检测 / 语音分割器（Silero、TEN、None） |
| **Model** | 使用的模型（取决于处理管线） |

点击任意 Pass 上的 **Customize** 可进行精细参数调整。

---

## 合并策略

| 策略 | 适用场景 |
|------|----------|
| **Pass 1 Primary** | 当第一轮是可信基线时 — 用第二轮填补空缺 |
| **Smart Merge** | 通用场景 — 使用质量启发式方法从每一轮中选择最佳字幕 |
| **Full Merge** | 最大覆盖 — 合并所有字幕，解决重叠 |
| **Longest** | 当两轮重叠时，选择更长（更详细）的字幕 |
| **Pass 2 Primary** | 当第二轮是可信基线时 |
| **Overlap 30%** | 稳妥合并 — 要求 30% 的时间重叠才进行合并 |

!!! tip "推荐组合"
    **Balanced**（第一轮）+ **Qwen3-ASR**（第二轮）+ **Smart Merge** 对大多数内容来说是一个可靠的默认组合。

---

## BYOP：XXL Faster Whisper（v1.8.9+；1.10.0 起也可用于第一轮）

选择 **XXL Faster Whisper** 作为第一轮、第二轮或两者的处理管线，即可将 [PurfView 的 Faster Whisper XXL](https://github.com/Purfview/whisper-standalone-win) 作为外部程序使用。这是"自带处理管线"（Bring Your Own Pipeline）— 由你提供可执行文件。

### 设置方法

1. 从上方链接下载 Faster Whisper XXL
2. 在 专家模式（Ensemble）选项卡中，为第一轮、第二轮或两者选择 **XXL Faster Whisper**
3. 在这一轮的 **Model** 列中选择 XXL 运行的模型：Large V2（默认）、Large V3、Large V3 Turbo 或 Medium
4. 点击 **Browse** 指向你的 `faster-whisper-xxl.exe`
5. 添加任意额外参数（默认 `--verbose True --standard_asia`）。模型请在 Model 列中选择，不要写在这里：额外参数中的 `--model` 不会被使用，控制台会给出提示。

WhisperJAV 向 XXL 发送输入文件、输出目录、模型、语言和 SRT 输出格式（字幕要译成英文时还有 `--task translate`）；其他 XXL 参数请在额外参数字段中添加。如果 XXL 处理某个文件失败，控制台会说明是哪一轮失败以及原因；第一轮失败时该文件标记为失败，第二轮失败时，如果第一轮生成了字幕，该文件保留第一轮的字幕。

### CLI

```bash
# XXL 作为第一轮，Qwen3-ASR 作为第二轮
whisperjav video.mp4 --ensemble --pass1-pipeline xxl --pass1-model large-v2 --pass2-pipeline qwen --xxl-exe /path/to/faster-whisper-xxl.exe

# XXL 作为第二轮
whisperjav video.mp4 --ensemble --pass1-pipeline fidelity --pass2-pipeline xxl --pass2-model large-v2 --xxl-exe /path/to/faster-whisper-xxl.exe
```

`--pass1-model` / `--pass2-model` 选择 XXL 运行的模型（默认 `large-v2`）。额外参数使用 GUI 中 Extra Args 字段保存的内容；命令行没有对应的选项。

---

## 串行与并行批处理模式

在专家模式下处理多个文件时：

| 模式 | 行为 |
|------|------|
| **并行**（默认） | 所有第一轮任务先运行，然后所有第二轮，最后所有合并 |
| **串行** | 每个文件完整处理完毕（第一轮 → 第二轮 → 合并）后再处理下一个 |

**串行模式**适用于你希望随时查看已完成结果的场景。在 GUI 中勾选 **Serial** 复选框，或在 CLI 中使用 `--ensemble-serial` 来启用。

---

## 预设

保存你的专家模式配置以便复用：

1. 配置好每一轮、合并策略和参数
2. 点击 **Save Preset**
3. 输入名称（例如 "High Quality JAV"、"Quick Anime"）
4. 之后可从预设下拉菜单中加载

预设会保存所有 Pass 配置、合并策略和自定义参数，并在不同会话间持久化。

---

## 内联翻译

在合并策略之后勾选 **"AI-translate"**，即可自动翻译合并后的输出。在界面中选择翻译提供商和模型，或点击设置按钮进行完整配置。
