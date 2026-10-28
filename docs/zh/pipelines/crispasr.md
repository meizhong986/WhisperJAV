# Parakeet-CrispASR 与 Cohere-CrispASR（GGUF）

Parakeet-CrispASR 和 Cohere-CrispASR 是两个处理管线，使用 GGUF 格式的 ASR 模型生成日语字幕。它们通过 CrispASR 运行；CrispASR 随每个 WhisperJAV 安装一起安装，无需另外下载任何程序。

- **仅限专家模式。** 只能在 **专家模式**（Ensemble Mode）选项卡中作为第一轮、第二轮或两者运行。没有对应的 `--mode` 值，也不出现在 Transcription Mode 选项卡中。
- **仅支持日语。** 选择其他源语言时，运行会停止并给出提示。
- **1.10.0 中为实验性功能（早期版本）。** 建议与你目前使用的处理管线的字幕对比，并告诉我们效果如何：https://github.com/meizhong986/WhisperJAV/issues

!!! warning "文件夹名称"
    程序中的提示原文：The GGUF (CrispASR) pipelines need folder names in plain Western letters (ASCII). When your user folder has Chinese, Japanese or Korean characters, WhisperJAV links or copies the model to a plain-named folder (the original stays where it is).

    意思是：GGUF（CrispASR）处理管线需要只由西文字母（ASCII）组成的文件夹名称。如果你的用户文件夹名称含有中文、日文或韩文字符，WhisperJAV 会把模型以链接或复制的方式放到一个名称只含 ASCII 字符的文件夹中（原文件保留在原处）。

    这一限制针对存放模型文件的文件夹：CrispASR 只接收这一个路径（音频在内存中交给它）。在 Windows 上，WhisperJAV 先尝试使用该文件夹的短名称（8.3 格式）；只有在没有短名称时，才把链接或副本放在 `C:\Users\Public\WhisperJAV\crispasr`（或 `C:\ProgramData\WhisperJAV\crispasr`），并在日志中说明。副本最多占用 2.4 GB，并保留供以后使用。已检查过一次：安装在含日文字符文件夹中的 CrispASR CPU 版能从那里加载模型并转写。WhisperJAV 本身以及 CrispASR 的 CUDA 版和 Vulkan 版尚未在这种文件夹中检查过。

!!! note "从 1.9.x 升级？"
    1.9.x 中需要自行下载 `crispasr` 程序的 CrispASR 模式已被移除。详见[从 1.9.x 升级](#upgrading-from-19x)。

---

## 它们在哪里运行 {#where-they-run}

每个 1.10.0 安装都包含 CrispASR，版本与安装所针对的加速器相符（见[是否加速](../getting-started/index.md#accelerated-or-not)）：

| 安装为哪种加速器制作 | CrispASR 版本 | GGUF 处理管线运行在 |
|---|---|---|
| NVIDIA GPU（CUDA），Windows 和 Linux x86-64 | CUDA 版 | NVIDIA GPU |
| 通过 Vulkan 使用 GPU（Windows） | Vulkan 版 | 通过 Vulkan 使用你的显卡——实验性，未经测试 |
| 自己的 PyTorch，Windows | Vulkan 版 | 通过 Vulkan 使用你的显卡——实验性，未经测试 |
| Apple GPU（Metal），Apple silicon Mac | macOS 版 | 请求使用 Apple GPU（Metal）——尚未在 Mac 上测试 |
| 自己的 PyTorch，Apple silicon Mac | macOS 版 | 请求使用 Apple GPU（Metal）——尚未在 Mac 上测试 |
| 仅 CPU；Linux 上的自己的 PyTorch；ARM64 Linux | CPU 版 | CPU（不加速），慢好几倍 |

处理器必须支持 AVX2（Intel 2013 年以后、AMD 2015 年以后的处理器）。不支持时，GGUF 处理管线会在读取任何音频之前停止并给出提示；其他所有处理管线照常工作。

每一轮开始时，日志会说明请求 CrispASR 在哪里运行，例如：

```
Parakeet-CrispASR asks CrispASR to use the NVIDIA GPU (CUDA) (CrispASR 0.8.44+cuda).
```

使用 Vulkan 版时为："Parakeet-CrispASR asks CrispASR's Vulkan build to use this PC's GPU (CrispASR 0.8.44+vulkan). GPUs through Vulkan are experimental in 1.10.0 and have not been tested on AMD or Intel graphics cards."（意思是：通过 Vulkan 使用 GPU 在 1.10.0 中为实验性，尚未在 AMD 或 Intel 显卡上测试。）WhisperJAV 目前无法读取 CrispASR 实际使用的设备，这一行只说明它请求了什么。

---

## 两个处理管线

| | Parakeet-CrispASR | Cohere-CrispASR |
|---|---|---|
| **模型** | `parakeet-tdt-0.6b-ja`（674 MB） | `cohere-asr-ja-v0.1`（默认）或 `cohere-asr-ja`（各 2,424 MB） |
| **默认语音分割器** | WhisperSeg | FireRedVAD |
| **默认灵敏度** | Balanced | Balanced |
| **VAD 组最长** | 6 秒，任何灵敏度下都一样 | 3 秒 |
| **字幕在哪里断开** | 只在语音的自然停顿处断开（见下文） | 只在句末断开（见下文） |
| **字幕时间** | 来自模型对其输出文字给出的时间 | 来自对齐器 `wav2vec2-ja`，它为模型写出的文字标出时间 |
| **音频分析报告** | 有 | 有 |

**VAD 组**：语音分割器找到并一次性交给模型的一段语音，长度为几秒。

Parakeet-CrispASR 使用 Fidelity 处理管线的语音分割器设置，另加 6 秒上限；Cohere-CrispASR 使用 Qwen3-ASR 处理管线的设置。

---

## 使用方法

### GUI

1. 打开 **专家模式**（Ensemble Mode）选项卡。
2. 在 **第一轮**（Pass 1）一行，打开 **Pipeline** 菜单，在 **GGUF** 分组下选择 **Parakeet-CrispASR** 或 **Cohere-CrispASR**。
3. 如果选的是 Cohere-CrispASR，可在 **Model** 列选择模型（默认 `cohere-asr-ja-v0.1`）。
4. 如需第二轮，勾选 **第二轮**（Pass 2），选择其处理管线，并在 **Merge** 中选择两者的合并方式。
5. 点击 **Start**。

当某一轮使用这两个处理管线之一时：

- **Sensitivity**、**Speech Segmenter**、**Model**、**Scene Detector**、**Speech Enhancer** 和 **Enhance for VAD only** 都可以设置，与其他 Pass 相同。语音分割器不能选 **None**：这两个处理管线只转写语音分割器找到的语音。
- **Customize Parameters** 打开三个选项卡：**Segmenter**、**Enhancer** 和 **Scene**，初始值就是这一轮实际使用的值。
- 这一轮一行下方会显示一个状态面板，说明模型和 CrispASR 将在哪里运行，例如：

  ```
  Model: Parakeet TDT 0.6B ja (q8_0) — not downloaded yet (674 MB); downloaded when the run starts
  GPU: asks for the NVIDIA GPU (CUDA)
  ```

  也可能显示 "CPU (not accelerated), several times slower: <原因>" 或 "CrispASR cannot run here: <原因>"。对 Cohere-CrispASR，面板还会列出它的对齐器，例如 "Aligner: wav2vec2 large XLSR-53 ja (q4_k) — not downloaded yet (232 MB); downloaded the first time it is used"。

### 命令行

单轮：

```bash
whisperjav video.mp4 --ensemble --pass1-pipeline parakeet-crispasr
```

两轮，第二轮使用 Cohere-CrispASR 的另一个模型：

```bash
whisperjav video.mp4 --ensemble --pass1-pipeline parakeet-crispasr --pass2-pipeline cohere-crispasr --pass2-model cohere-asr-ja
```

不指定 `--merge-strategy` 时，两轮按 `pass1_primary` 合并（GUI 中为“第一轮 + 用第二轮填补”，英文界面为 "Pass 1 + Fill From Pass 2"）。

---

## 设置

### 可以设置的项目

| 设置 | GUI | 命令行 | 说明 |
|---|---|---|---|
| 模型 | Model | `--pass1-model`、`--pass2-model` | `parakeet-tdt-0.6b-ja`；`cohere-asr-ja-v0.1`、`cohere-asr-ja` |
| 语音分割器 | Speech Segmenter | `--pass1-speech-segmenter` | 除 `none` 以外均可 |
| 灵敏度 | Sensitivity | `--pass1-sensitivity` | `conservative`、`balanced`（默认）、`aggressive` |
| 场景检测器 | Scene Detector | `--pass1-scene-detector` | `semantic`（默认，场景 12 到 48 秒）、`auditok`、`silero`（场景最长 48 秒）、`none`（每个文件一个场景） |
| 语音增强器 | Speech Enhancer | `--pass1-speech-enhancer` | `none`（默认），或其他 Pass 可用的语音增强器，例如 `ffmpeg-dsp`、`zipenhancer` 或 `htdemucs` |
| 仅用于 VAD 的增强 | Enhance for VAD only | `--pass1-enhance-for-vad` | 增强后的音频只交给语音分割器；CrispASR 听到的是原始音频 |
| 检测阈值 | Customize，Segmenter 选项卡 | `--pass1-vad-threshold` | 0 到 1；覆盖灵敏度设定的值 |
| 前后填充 | Customize，Segmenter 选项卡 | `--pass1-speech-pad-ms` | 每段语音前后各加的毫秒数。Cohere-CrispASR：前后都加。Parakeet-CrispASR：仅对有此设置的语音分割器生效；使用 FireRedVAD 或 TEN VAD 时会被忽略并给出警告 |
| 设备 | — | `--device auto`、`cuda` 或 `cpu` | 见下文 |
| 语言 | Source audio language | `--language japanese`（默认） | 其他语言会使运行停止 |

（第二轮有对应的 `--pass2-…` 参数。）

### 灵敏度

灵敏度选择语音分割器的预设。Aggressive 会接受语音分割器不太确定是否为语音的声音；Conservative 只接受它更有把握的声音。使用默认语音分割器时：

| | Conservative | Balanced | Aggressive |
|---|---|---|---|
| Parakeet-CrispASR（WhisperSeg）：检测阈值 | 0.45 | 0.35 | 0.25 |
| Parakeet-CrispASR：语音前后填充 | 200 ms | 300 ms | 400 ms |
| Cohere-CrispASR（FireRedVAD）：检测阈值 | 0.5 | 0.4 | 0.3 |
| Cohere-CrispASR：语音前后填充 | 100 ms | 100 ms | 100 ms |

检测阈值表示语音分割器要有多大把握才把一段声音当作语音，取值 0 到 1。无论灵敏度如何，VAD 组最长都保持 6 秒（Parakeet-CrispASR）或 3 秒（Cohere-CrispASR）。换用其他语音分割器时，Parakeet-CrispASR 使用 Fidelity 处理管线为它给出的值，Cohere-CrispASR 使用 Qwen3-ASR 处理管线为它给出的值；Parakeet-CrispASR 保留 6 秒上限。

不实际运行、只查看某次运行会用到的值：

```bash
whisperjav --ensemble --pass1-pipeline parakeet-crispasr --dump-params settings.json
```

然后查看 `settings.json` 中的 `crispasr_effective`。

### 设备

- `--device auto`（默认值，GUI 也使用它）：使用安装中 CrispASR 版本能用的 GPU（见[它们在哪里运行](#where-they-run)），否则使用 CPU。日志会说明用的是哪一个以及原因。
- `--device cuda`：需要 CUDA 版和它能用的 NVIDIA GPU；否则运行会在读取任何音频之前停止。
- `--device cpu`：ASR 模型始终使用 CPU。在 GUI 中，"Run on the CPU this time" 对一次运行起同样作用。Cohere-CrispASR 的对齐器不能被限制在 CPU 上：在 CrispASR 的版本会使用 GPU 的地方，它仍会请求使用 GPU，运行开始时的提示框会说明这一点。

`--device` 决定一次运行中所有处理管线的设备，包括语音增强器。

### 这两个处理管线不使用的设置

- 如果指定了 `--pass1-hf-params` 或 `--pass1-qwen-params`，日志会用一行列出它们（"Parakeet-CrispASR does not use: …"），运行继续。
- `--compute-type` 会被忽略并给出警告：CrispASR 的模型文件只有一种数值格式（8 位）。
- `--subs-language direct-to-english` 仍然生成日语字幕，并给出警告。之后可用 `--translate` 或 `whisperjav-translate` 翻译。

---

## 字幕是怎样生成的 {#how-the-subtitles-are-made}

对每个视频，两个处理管线都会：

1. 提取音频（16 kHz，单声道；选择了语音增强器时为 48 kHz，增强后得到 16 kHz 音频）；
2. 用场景检测器把音频分成场景（默认为 Semantic，12 到 48 秒）；
3. 如果选择了语音增强器，清理每个场景的音频；增强器在模型打开之前关闭，两者不会同时占用 GPU 显存；
4. 对每个场景运行语音分割器，把找到的语音组成 VAD 组；
5. 把每个 VAD 组单独交给模型；没有语音增强器时，模型每一轮只加载一次，有语音增强器时，每个视频的 ASR 步骤打开模型，结束后关闭；
6. 按下文各处理管线的规则，用每个 VAD 组的文字生成字幕；
7. 合并各场景，并使用与 Qwen3-ASR 处理管线相同的三个清理过滤器：删除「あ。」「うん。」这类单独成行的字幕、只有呻吟或呼吸声的字幕，以及两个场景交界处的重复和重叠；
8. 写出这一轮的字幕文件；若有另一轮，专家模式会把两者合并。

**Parakeet-CrispASR：字幕在哪里断开。** 在一个 VAD 组内，只有遇到以下自然停顿时才开始新的一条字幕：

- 句末标点之后（。？！… 等）；
- 至少 0.6 秒的停顿；
- 「、」或空格之后紧跟至少 0.4 秒的停顿。

它从不按字数或时长切分，因此没有上述停顿的语音会保持为一条字幕，最长与其 VAD 组相同（约 6 秒）。每条字幕从第一个词开始处开始，到最后一个说出的词结束处结束，时间取自模型给出的时间。

**Cohere-CrispASR：字幕在句末断开。** Cohere 写出一个 VAD 组的文字之后，对齐器 `wav2vec2-ja`（由 CrispASR 运行）找出这段文字中每个字在什么时候说出。在一个 VAD 组内，只有在文字的句末标点（。 ？ ！ … ? !）之后才开始新的一条字幕；像 3.5 这样的数字不会被切开。每条字幕从它第一个说出的字开始，在下一条开始处结束；最后一条在 VAD 组结束处结束。这里停顿不会切分：对齐器给出的是每个字开始的时刻，而不是声音结束的地方，拖长的声音看起来会像停顿。VAD 组最长 3 秒；一段语音连续不断时，约为 4.2 秒。

如果对齐器对某个 VAD 组没有给出时间，这个组保持为一条字幕，时间从语音分割器找到的语音开始处到结束处，运行汇总会统计这样的组。如果对齐器完全无法加载（例如第一次运行时没有网络），所有组都这样生成，运行汇总会说明；两种情况下文字都会保留。1.10.0 新增；尚未与参考字幕对比测量。

**音频分析报告。** 与 Qwen3-ASR 处理管线一样，两个处理管线都会为每个视频输出一份简短报告：哪些部分可能漏掉语音、哪些部分声学条件较差，并附上时间，方便你检查这些位置。报告需要 Semantic 场景检测器；使用其他场景检测器时，日志会用一行说明，不生成报告。

---

## 模型下载

- 处理管线第一次需要某个模型时，WhisperJAV 会在运行开始、读取任何音频之前下载它。下载前会先检查磁盘剩余空间是否足够（模型大小再加 10%）。
- 模型来自 Hugging Face：

| 模型 | Hugging Face 仓库 | 文件 |
|---|---|---|
| `parakeet-tdt-0.6b-ja` | `cstr/parakeet-tdt-0.6b-ja-GGUF` | `parakeet-tdt-0.6b-ja-q8_0.gguf` |
| `cohere-asr-ja-v0.1` | `TransWithAI/cohere-transcribe-ja-v0.1-GGUF` | `cohere-asr-ja-v0.1-q8_0.gguf` |
| `cohere-asr-ja` | `CKHO/cohere-asr-ja-GGUF` | `cohere-asr-ja-q8_0.gguf` |
| 对齐器 `wav2vec2-ja`（Cohere-CrispASR；也是 ChronosJAV 的默认对齐器） | `cstr/wav2vec2-large-xlsr-53-japanese-GGUF` | `wav2vec2-large-xlsr-53-japanese-q4_k.gguf` |

Cohere-CrispASR 的对齐器（232 MB）在这一轮第一次使用时下载，下载发生在这一轮内部，而不是运行开始之前。如果下载失败，这一轮会在没有对齐器的情况下继续：每个 VAD 组成为一条字幕，时间为该组的起止位置（见[字幕是怎样生成的](#how-the-subtitles-are-made)）。ChronosJAV 的其他对齐器和时间模型会在 `whisperjav --check` 中列出，也见 [ChronosJAV 页面](chronosjav.md#subtitle-timing-1100)。

- 文件保存在你的 Hugging Face 缓存文件夹中（Windows 默认 `C:\Users\<你的用户名>\.cache\huggingface\`，macOS 和 Linux 默认 `~/.cache/huggingface/`），之后的运行会直接使用。
- 首次使用前，WhisperJAV 会核对文件是否为它所需的版本。下载损坏的文件会被删除并重新下载一次。
- **无法访问 huggingface.co 时**，WhisperJAV 会改用镜像 hf-mirror.com 下载，与它的其他模型相同。也可以用 `--hf-endpoint <地址>` 指定其他镜像。
- 使用 `--offline`（GUI 中为 "Offline mode (downloaded Hugging Face models only)"）时不会下载任何内容。如果模型缺失，运行会在转写之前停止并说明原因；请先不加 `--offline` 运行一次。
- 如果下载或核对失败，运行会停止，显示 "CrispASR is not ready for this run" 及原因，不会转写任何内容。

**使用自己下载的模型文件。** 把文件（文件名保持不变）放到一个名称为纯英文字符的文件夹中，并把环境变量 `WHISPERJAV_CRISPASR_MODEL_DIR` 设为该文件夹。只有文件大小符合预期时 WhisperJAV 才会使用它，并在首次使用前核对；核对不通过时运行会停止，文件不会被删除。

---

## 检查你的环境

`whisperjav --check` 会列出 CrispASR 及其模型。在一切就绪的 NVIDIA GPU（CUDA）安装上显示：

```
✓ CrispASR: 0.8.44+cuda (CUDA build), GGUF pipelines: asks for the NVIDIA GPU (CUDA)
✓ CrispASR model parakeet-tdt-0.6b-ja: Parakeet TDT 0.6B ja (q8_0), for parakeet-crispasr: downloaded
✓ CrispASR model cohere-asr-ja-v0.1: Cohere Transcribe ja v0.1 (q8_0), for cohere-crispasr: downloaded
✓ CrispASR model cohere-asr-ja: Cohere ASR ja (q8_0), for cohere-crispasr: downloaded
✓ CrispASR model wav2vec2-ja: wav2vec2 large XLSR-53 ja (q4_k), for ChronosJAV and Cohere-CrispASR (aligner): downloaded
✓ CrispASR model parakeet-ctc-ja: Parakeet CTC 1.1B ja (q8_0), for ChronosJAV (aligner): downloaded
✓ CrispASR model qwen3-fa-gguf: Qwen3-ForcedAligner 0.6B (q4_k), for ChronosJAV (aligner): downloaded
✓ CrispASR model whisper-small: Whisper small, for ChronosJAV (timing model): downloaded
```

尚未下载的模型会显示 "not downloaded" 及其大小，第一次需要它的运行会下载它。

如果 Windows 上 CrispASR 的 Vulkan 版无法加载，提示会附加："With the GPU-through-Vulkan or Own PyTorch install on Windows this usually means the Vulkan runtime (vulkan-1.dll) is missing: update the graphics driver, or run the WhisperJAV installer again and choose CPU only."（意思是：缺少 Vulkan 运行库 vulkan-1.dll；请更新显卡驱动，或重新运行安装程序并选择仅 CPU。）

---

## 视频部分处理失败时

- 如果 CrispASR 在某个视频的部分 VAD 组上失败，这些部分没有字幕，运行汇总会在该视频下说明数量。如果所有 VAD 组都失败，该视频处理失败。
- 如果语音分割器在某个场景上失败，该场景的语音不会被转写，运行汇总中会注明。

---

## 从 1.9.x 升级 {#upgrading-from-19x}

1.9.x 有一个 CrispASR 模式，运行你自行下载的 `crispasr` 程序。1.10.0 移除了该模式：

- `--mode crispasr`、`--pass1-pipeline crispasr`、`--pass2-pipeline crispasr`、`--crispasr-exe`、`--crispasr-backend` 和 `--crispasr-args` 不再可用。运行会停止（退出码 2），并显示：

  ```
  --mode crispasr is no longer available: WhisperJAV 1.10 removed the "bring your own CrispASR" mode.
  CrispASR now runs in Ensemble Mode as two pipelines, Parakeet-CrispASR and Cohere-CrispASR, and WhisperJAV installs it itself.
  Example:
      whisperjav --ensemble --pass1-pipeline parakeet-crispasr <your video>
  In the GUI: open the Ensemble Mode tab and choose Parakeet-CrispASR or Cohere-CrispASR under GGUF in the Pass 1 or Pass 2 menu.
  ```

- GUI 中，Transcription Mode 选项卡 Mode 菜单里的 **crispasr** 项已删除，专家模式（Ensemble Mode）选项卡中带 Backend 选项的旧 **CrispASR** 处理管线也已删除。
- `whispercpp` 后端在新的处理管线中没有替代。
- 1.10.0 不再使用你下载的 `crispasr` 程序及其保存的模型（默认位于 `~/.cache/whisperjav/crispasr_models`），可以删除。

改用：`--ensemble --pass1-pipeline parakeet-crispasr` 替代 `--mode crispasr --crispasr-backend parakeet`；`--ensemble --pass1-pipeline cohere-crispasr` 替代 `--mode crispasr --crispasr-backend cohere`。

---

## 致谢

CrispASR 由 [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR) 开发。模型：NVIDIA 的 parakeet-tdt_ctc-0.6b-ja（CC-BY-4.0），由 cstr 转为 GGUF 格式；Cohere Transcribe 的日语微调版由 efwkjn 制作，由 TransWithAI 和 CKHO 转为 GGUF 格式。
