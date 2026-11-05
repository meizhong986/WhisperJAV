# 专家模式

专家模式在同一视频上先后运行一个或两个处理管线。勾选 **第二轮**（Pass 2）时，它按你选择的合并策略把两轮的字幕合并成一个文件。

---

## 工作原理

```
Video → Pass 1 (Pipeline A) → SRT₁ ─┐
                                      ├→ Merge → Final SRT
Video → Pass 2 (Pipeline B) → SRT₂ ─┘
```

每一轮用它自己的处理管线处理整个视频，并写出自己的字幕文件；合并再由这两个文件生成最终的字幕文件。

---

## 设置专家模式

1. 打开 **专家模式**（Ensemble Mode）选项卡（第 3 个选项卡）
2. **第一轮** 始终处于激活状态 — 配置处理管线、灵敏度和选项
3. 勾选复选框以**启用第二轮**
4. 为第二轮配置不同的处理管线
5. 选择一种**合并策略**
6. 点击 **Start**

### Pass 配置

每一轮拥有相同的控件：

| 控件 | 说明 |
|------|------|
| **Pipeline** | 这一轮的处理管线，分为六组：基于 Whisper（Balanced、Fast、Faster、Fidelity）、HuggingFace（Transformers (HF)）、ChronosJAV（Qwen3-ASR、Anime-Whisper）、外部 (BYOP)（XXL Faster Whisper）、GGUF via crispasr（parakeet、cohere），以及分隔线下方的 实验性，欢迎反馈（reazonspeech、kotoba、whisper-turbo、whisper-large、MOSS）。第一轮开始时为 cohere，第二轮为 parakeet。选择处理管线时，这一轮的灵敏度和语音分割器会随之设为它自己的值 |
| **Sensitivity** | 检测阈值（高灵敏、中灵敏、低灵敏） |
| **Scene Detector** | 音频场景检测方式（Auditok、Silero、Semantic、None） |
| **Speech Enhancer** | 音频预处理 / 语音增强（None、FFmpeg DSP、ZipEnhancer、htdemucs） |
| **Speech Segmenter** | 在每个场景中找出语音：Silero v4.0、Silero v3.1、Silero v6.2、TEN VAD、WhisperSeg (JA-ASMR)、FireRedVAD 或 无（None）。Balanced 时，这个菜单改为列出其内置 VAD 使用的 Silero 版本（内置 FW Silero VAD 3.1、4.0（默认）、6.2）。GGUF 处理管线选 无（None）时，每个场景整段交给模型 |
| **Model** | 使用的模型（取决于处理管线） |

点击某一轮的 **自定义参数**（Customize Parameters）可修改这一轮的详细设置；XXL Faster Whisper 的这个按钮是灰色的。

---

## 合并策略

| 策略（GUI） | 命令行 | 作用 |
|------|------|------|
| **第一轮 + 用第二轮填补**（Pass 1 + Fill From Pass 2，默认） | `pass1_primary` | 保留第一轮的全部字幕；第二轮的字幕只在不与任何第一轮字幕重叠时加入 |
| **智能合并（质量最佳）**（Smart Merge (Best Quality)） | `smart_merge` | 把每条第一轮字幕与和它重叠最多的第二轮字幕配对（重叠至少为较短一条的 30%），每对只保留一条：自身长度落在重叠部分里的比例更大的那条；比例相近时取文字更多的那条（由很少几种字符组成的行，如 ああああああああああ，按没有文字计）；再相同时取较短的那条。没有配对的字幕两轮都保留 |
| **完全合并（全部字幕行）**（Full Merge (All Lines)） | `full_merge` | 两轮的全部字幕按时间排列；重叠的字幕都保留 |
| **第二轮 + 用第一轮填补**（Pass 2 + Fill From Pass 1） | `pass2_primary` | 同"第一轮 + 用第二轮填补"，两轮互换 |
| **第一轮 + 填补（30% 重叠）**（Pass 1 + Fill (30% Overlap)） | `pass1_overlap` | 保留第一轮的全部字幕；第二轮的字幕只要与第一轮字幕的重叠不超过那条第一轮字幕长度的 30% 就加入 |
| **第二轮 + 填补（30% 重叠）**（Pass 2 + Fill (30% Overlap)） | `pass2_overlap` | 同上，两轮互换 |
| —（仅命令行） | `longest` | 与智能合并一样配对，保留文字更多的那条 |

无论哪种策略，合并后与前一条文字完全相同的字幕都会被删除。

---

## BYOP：XXL Faster Whisper（v1.8.9+；1.10.0 起也可用于第一轮）

选择 **XXL Faster Whisper** 作为第一轮、第二轮或两者的处理管线，即可将 [PurfView 的 Faster Whisper XXL](https://github.com/Purfview/whisper-standalone-win) 作为外部程序使用。菜单把它列在 **外部 (BYOP)**（External (BYOP)）组中：XXL 是一个需要你自己下载的独立程序，由 WhisperJAV 运行。

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

## 串行与批量模式

在专家模式下处理多个文件时：

| 模式 | 行为 |
|------|------|
| **批量** | 所有第一轮任务先运行，然后所有第二轮，最后所有合并；每一轮只为全部文件启动一次 |
| **串行**（默认） | 每个文件完整处理完毕（第一轮 → 第二轮 → 合并）后再处理下一个 |

**串行模式**从 1.10.1 起是默认模式，GUI 和命令行都是：一个文件处理完时它的字幕就已就绪，某个文件崩溃也不会丢失其他文件。在 GUI 中由 **逐个完成文件**（Finish each file）勾选框（默认勾选）开启；取消勾选，或在命令行中使用 `--no-ensemble-serial`，即为批量模式。

---

## 预设

一个预设保存一轮的设置：处理管线、灵敏度、场景检测器、语音增强器、语音分割器、模型，以及 **自定义参数**（Customize Parameters）窗口中的值。合并策略和另一轮不在其中。

1. 点击这一轮的 **自定义参数**（Customize Parameters），设好其中的值
2. 点击窗口底部的 **保存预设**（Save Preset），输入名称
3. 之后在同一窗口的 **加载预设...**（Load Preset...）菜单中选择它，可载入任一轮

预设会保留到下次启动。

---

## 内联翻译

在合并策略之后勾选 **"AI-translate"**，即可自动翻译合并后的输出。在界面中选择翻译提供商和模型，或点击设置按钮进行完整配置。
