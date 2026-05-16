# CrispASR 处理管线

CrispASR 是一个自包含的外部语音识别引擎（来自 [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR) 项目的 C++/ggml 可执行程序）。WhisperJAV 像 Subtitle Edit 那样以子进程方式调用它——WhisperJAV **不** 捆绑该引擎，需要你自行提供可执行文件。

---

## 后端

| 后端 | 引擎 | 备注 |
|------|------|------|
| **parakeet**（默认） | NVIDIA Parakeet | 原生词级时间轴；速度/质量均衡 |
| **whispercpp** | whisper.cpp（large-v2） | 原生词级时间轴；crispasr `whisper` 后端的别名 |
| **cohere** | Cohere Transcribe | 原生词级时间轴；开放的 ggml 资源（无需 token） |

这三个后端均 **无需强制对齐器**。模型 GGUF 在首次使用时自动下载。

---

## 前置条件

CrispASR 需要其 **外部引擎可执行文件**（WhisperJAV 未随附）：

1. 从 [CrispStrobe/CrispASR releases](https://github.com/CrispStrobe/CrispASR/releases) 下载与你的操作系统和 GPU 匹配的发布压缩包：
    - Windows + NVIDIA → `crispasr-windows-x86_64-cuda.zip`
    - Windows，无 GPU → `crispasr-windows-x86_64-cpu.zip`（或 `-vulkan`）
    - macOS → `crispasr-macos.tar.gz` · Linux → `crispasr-linux-x86_64.tar.gz`
2. 解压到某个文件夹，并记下 `crispasr` 可执行文件的路径。

`ffmpeg` 必须在 `PATH` 中（WhisperJAV 本就要求）。所选后端均无需 HuggingFace token。

---

## 使用方法

### GUI（转录选项卡）

1. 将 **Mode** 设置为 **crispasr**
2. 在 CrispASR 面板中：**Browse** 选择 `crispasr` 可执行文件，并选择一个 **Backend**
3. 点击 **Start**

### GUI（Ensemble 选项卡）

1. 将某个 pass 的 **Pipeline** 设置为 **CrispASR**
2. 在该 pass 的 Model 列中选择 **Backend**；在 CrispASR 面板中设置可执行文件路径
3. 点击 **Start**

### CLI

```bash
whisperjav video.mp4 --mode crispasr --crispasr-exe /path/to/crispasr --crispasr-backend parakeet
```

作为集成模式的某个 pass：

```bash
whisperjav video.mp4 --ensemble --pass1-pipeline balanced --pass2-pipeline crispasr --crispasr-exe /path/to/crispasr --crispasr-backend whispercpp
```

使用 `--crispasr-args "..."` 可原样传递引擎的高级参数。

---

## 模型下载位置

引擎自行管理模型。首次使用某个后端时，会从 HuggingFace（`cstr/…-GGUF`）下载对应的 GGUF（约 150 MB – 1 GB）到缓存目录。除非你自行设置，否则 WhisperJAV 会将 `CRISPASR_CACHE_DIR` 指向 `~/.cache/whisperjav/crispasr_models`。这需要可访问外网 HTTPS，且 `PATH` 中有 `curl`/`wget`（Windows 还会尝试 WinHTTP）。之后的运行会复用缓存。

---

## 系统要求

- 已安装 WhisperJAV（无需额外 Python 包——该处理管线已内置）
- `PATH` 中有 `ffmpeg`
- CrispASR 引擎可执行文件（由用户自行提供，见前置条件）
- 每个后端首次运行时需要网络访问

---

## 优势与局限

**优势：**

- 在单一外部提供方中提供多种引擎（Parakeet / whisper.cpp / Cohere）
- 自包含——自行处理 VAD；无需调整 WhisperJAV 的场景检测/分段器/增强器
- GPU 变体（CUDA / Vulkan / CPU）取决于你下载的引擎构建版本

**局限：**

- 引擎可执行文件未捆绑——需自行下载一次
- 模型从可变的 HuggingFace 分支拉取且无校验和（引擎行为）
- `whispercpp` 通过引擎的 auto 机制请求 large-v2——请在你的构建版本上确认其可解析（必要时用 `--crispasr-args -m <...>` 覆盖）
- CrispASR 不对 `--subs-language direct-to-english` 做特殊处理
- 每个后端首次运行会因模型下载而暂停

---

## 面向维护者与发布

- **安装程序：** 无需改动。该处理管线是 `whisperjav` 包内的纯 Python 代码；无新增 pip 或系统依赖（FFmpeg 本就要求）。**不要** 捆绑引擎可执行文件（独立项目；存在 CUDA/Vulkan/CPU 变体矩阵）。
- **GitHub 发布：** WhisperJAV 的发布资源保持不变（安装程序/wheel 已包含该处理管线代码）。引擎可执行文件 **不是** WhisperJAV 的发布资源——它位于 CrispStrobe/CrispASR 的发布中。模型不在任何发布包内；它们在首次使用时从 HuggingFace 自动下载。
