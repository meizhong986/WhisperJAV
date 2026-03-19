# WhisperJAV Mac Apple Silicon 安装指南

**版本：** 1.8.9
**适用平台：** macOS 13 (Ventura) 或更高版本，搭载 Apple M1/M2/M3/M4/M5
**最后更新：** 2026-03-19

---

## 目录

1. [系统要求](#1-系统要求)
2. [前提条件](#2-前提条件)
3. [安装](#3-安装)
4. [MPS GPU 加速](#4-mps-gpu-加速)
5. [运行 WhisperJAV](#5-运行-whisperjav)
6. [处理管线模式选择](#6-处理管线模式选择)
7. [Mac 上的 Qwen 处理管线](#7-mac-上的-qwen-处理管线)
8. [GUI 应用程序](#8-gui-应用程序)
9. [本地大语言模型翻译](#9-本地大语言模型翻译)
10. [性能预期](#10-性能预期)
11. [故障排除](#11-故障排除)
12. [已知限制](#12-已知限制)

---

## 1. 系统要求

### 硬件

| 要求 | 最低 | 推荐 |
|------|------|------|
| 芯片 | Apple M1 | Apple M2 Pro 或更新 |
| 内存（RAM）| 16 GB | 32 GB 或更多 |
| 存储 | 15 GB 可用 | 30 GB 可用 |
| macOS | 13.0 (Ventura) | 14.0 (Sonoma) 或更新 |

**为什么内存很重要：** Apple Silicon 使用统一内存架构 -- GPU 和 CPU 共享同一内存。Whisper 模型根据大小需要 2-6 GB。总内存为 8 GB 时，只能运行 `small` 或 `base` 模型。16 GB 时可以运行 `large-v2` 但较为紧张。32 GB 可以舒适地运行所有模型大小。

### 软件

- Python 3.10、3.11 或 3.12（3.13+ 与 openai-whisper 不兼容）
- FFmpeg（用于音视频处理）
- Git（用于从源码安装包）
- Xcode Command Line Tools（用于编译 C 扩展）

---

## 2. 前提条件

### 步骤 1：安装 Xcode Command Line Tools

这是编译原生 Python 包（numpy、scipy 等）所必需的。

```bash
xcode-select --install
```

将出现一个对话框。点击"安装"并等待完成（可能需要 5-10 分钟）。

**验证：**
```bash
xcode-select -p
# 应输出：/Library/Developer/CommandLineTools
```

### 步骤 2：安装 Homebrew

如果尚未安装 Homebrew：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

安装后，按照屏幕上的指示将 Homebrew 添加到 PATH。对于 Apple Silicon Mac，Homebrew 安装到 `/opt/homebrew/`。您需要将其添加到 shell 配置文件中。

对于 **zsh**（macOS 默认）：
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
```

对于 **bash**：
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.bash_profile
source ~/.bash_profile
```

**验证：**
```bash
brew --version
```

### 步骤 3：安装 Python

macOS 自带的 Python 版本不适用。通过 Homebrew 安装 Python 3.12：

```bash
brew install python@3.12
```

**验证：**
```bash
python3 --version
# 应输出：Python 3.12.x
```

如果您倾向于使用版本管理工具，可以使用 `pyenv`：

```bash
brew install pyenv
pyenv install 3.12.8
pyenv global 3.12.8
```

使用 pyenv 时，将以下内容添加到 `~/.zshrc`：
```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

**重要：** 请勿使用 macOS 自带的系统 Python。它版本过旧且由操作系统管理。

### 步骤 4：安装 FFmpeg 和 PortAudio

```bash
brew install ffmpeg portaudio
```

- **FFmpeg** 用于音视频提取和格式转换。
- **PortAudio** 是 `pyaudio` 所需的系统库，`auditok` 使用它进行场景检测。

**验证：**
```bash
ffmpeg -version
```

### 步骤 5：安装 Git

Git 通常随 Xcode Command Line Tools 预装。验证：

```bash
git --version
```

如果未找到：
```bash
brew install git
```

---

## 3. 安装

### 步骤 1：创建虚拟环境

请始终使用虚拟环境。这可以防止与系统包和其他项目产生冲突。

```bash
# 创建虚拟环境
python3 -m venv ~/venvs/whisperjav

# 激活
source ~/venvs/whisperjav/bin/activate

# 验证您在 venv 中
which python
# 应输出：/Users/<您的用户名>/venvs/whisperjav/bin/python
```

每次打开新终端使用 WhisperJAV 时，都必须激活此环境：
```bash
source ~/venvs/whisperjav/bin/activate
```

### 步骤 2：升级 pip

```bash
pip install --upgrade pip
```

### 步骤 3：克隆仓库

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
```

### 步骤 4：安装支持 MPS 的 PyTorch

在 Apple Silicon 上，默认 PyPI 索引的 PyTorch 已包含 MPS（Metal Performance Shaders）支持。不要使用 CUDA 安装时使用的 `--index-url` 标志。

```bash
pip install torch torchaudio
```

**验证 MPS 支持：**
```bash
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    x = torch.ones(1, device='mps')
    print(f'MPS tensor test: {x} (SUCCESS)')
"
```

预期输出：
```
PyTorch version: 2.x.x
MPS available: True
MPS built: True
MPS tensor test: tensor([1.], device='mps:0') (SUCCESS)
```

如果 MPS 不可用，请参阅[故障排除](#11-故障排除)。

### 步骤 5：安装核心依赖项

```bash
# 科学计算栈（顺序重要：numpy 必须在 numba 之前）
pip install "numpy>=2.0.0" "scipy>=1.13.0" "numba>=0.60.0"

# 音频处理
pip install soundfile pydub "librosa>=0.11.0" pyloudnorm

# 字幕处理
pip install pysrt srt

# 工具库
pip install tqdm colorama requests aiofiles regex

# 配置
pip install "pydantic>=2.0,<3.0" "PyYAML>=6.0" jsonschema

# 语音活动检测
pip install pyaudio auditok "silero-vad>=6.2" ten-vad

# 性能
pip install "psutil>=5.9.0" "scikit-learn>=1.3.0"
```

### 步骤 6：安装 Whisper 包

```bash
# OpenAI Whisper（main 分支以获取最新修复）
pip install git+https://github.com/openai/whisper@main

# Stable-ts（自定义 fork）
pip install git+https://github.com/meizhong986/stable-ts-fix-setup.git@main

# FFmpeg Python 绑定（git 版本，PyPI 版本有构建问题）
pip install git+https://github.com/kkroening/ffmpeg-python.git

# Faster-Whisper（基于 CTranslate2 -- 在 Mac 上仅 CPU 运行，见已知限制）
pip install "faster-whisper>=1.1.0"
```

### 步骤 7：安装 HuggingFace 和 Qwen 支持

```bash
# HuggingFace Transformers 生态系统
pip install "huggingface-hub>=0.25.0" "transformers>=4.40.0" "accelerate>=0.26.0" hf_xet

# Qwen3-ASR（v1.8.9+）
pip install "qwen-asr>=0.0.6"
```

### 步骤 8：安装可选功能

**翻译模块：**
```bash
pip install "pysubtrans>=1.5.0" "openai>=1.35.0" "google-genai>=1.39.0"
```

**GUI：**
```bash
pip install "pywebview>=5.0.0"
```
注意：在 macOS 上，pywebview 使用原生 WebKit 引擎。不需要额外的依赖项。

**兼容性层：**
```bash
pip install "av>=13.0.0" "imageio>=2.31.0" "imageio-ffmpeg>=0.4.9" "httpx>=0.27.0" "websockets>=13.0" "soxr>=0.3.0"
```

### 步骤 9：安装 WhisperJAV

```bash
# 标准安装（不要让 pip 重新解析 torch）
pip install --no-deps .

# 或开发模式：
pip install --no-deps -e .
```

`--no-deps` 标志至关重要。它防止 pip 重新解析依赖项并可能将您支持 MPS 的 PyTorch 替换为不兼容的版本。

### 步骤 10：验证安装

```bash
python3 -c "import whisperjav; print(f'WhisperJAV {whisperjav.__version__} installed successfully')"
```

运行预检：
```bash
whisperjav --check
```

---

## 替代方案：使用安装脚本

您也可以使用提供的安装脚本，但请注意其在 Mac 上的限制：

```bash
cd whisperjav

# 先创建并激活 venv
python3 -m venv ~/venvs/whisperjav
source ~/venvs/whisperjav/bin/activate

# 使用 CPU 标志运行安装脚本
# （在 Mac 上，--cpu-only 会安装包含 MPS 支持的正确 PyTorch）
python install.py --cpu-only
```

`--cpu-only` 标志告诉安装程序跳过 NVIDIA GPU 检测并使用默认 PyTorch 索引，在 Apple Silicon 上这会包含 MPS 支持。尽管标志名称容易引起误解，但这是正确的行为。

**重要：** 脚本会报告 "No NVIDIA GPU detected - using CPU"。这在 Mac 上是正常的。运行时仍可使用 MPS 加速。

---

## 4. MPS GPU 加速

### 什么是 MPS？

MPS（Metal Performance Shaders）是 Apple 为 Apple Silicon 提供的 GPU 计算框架。它为 PyTorch 操作提供 GPU 加速，无需 CUDA。WhisperJAV 会自动检测并在可用时使用 MPS。

### 验证 MPS 是否工作

```bash
python3 -c "
from whisperjav.utils.device_detector import get_best_device, get_device_info
device = get_best_device()
print(f'Best device: {device}')
info = get_device_info()
print(f'MPS available: {info[\"mps\"][\"available\"]}')
print(f'MPS name: {info[\"mps\"][\"name\"]}')
"
```

预期输出：
```
Best device: mps
MPS available: True
MPS name: Apple Silicon (arm64)
```

### 各处理管线的 MPS 兼容性

| 处理管线模式 | MPS 支持 | 实际使用的设备 |
|-------------|----------|---------------|
| `balanced` | 否 | CPU（faster-whisper 使用 CTranslate2，不支持 MPS）|
| `fast` | 否 | CPU（使用 faster-whisper 后端）|
| `faster` | 否 | CPU（使用 faster-whisper 后端）|
| `transformers` | **是** | MPS（使用 HuggingFace Transformers）|
| `qwen` | 部分 | CPU（见[第 7 节](#7-mac-上的-qwen-处理管线)）|

---

## 5. 运行 WhisperJAV

### 基本 CLI 使用

```bash
# 先激活虚拟环境
source ~/venvs/whisperjav/bin/activate

# 基本转录（默认使用 balanced 模式 -- Mac 上为 CPU）
whisperjav video.mp4

# 使用 transformers 模式启用 MPS GPU 加速（Mac 上推荐）
whisperjav video.mp4 --mode transformers

# 调整灵敏度
whisperjav video.mp4 --mode transformers --sensitivity aggressive

# 指定模型大小（根据可用内存调整）
whisperjav video.mp4 --mode transformers --model large-v2

# 带翻译
whisperjav video.mp4 --mode transformers --translate
```

### Mac 推荐设置

对于 Apple Silicon Mac，推荐配置如下：

```bash
# 16 GB 内存 Mac
whisperjav video.mp4 --mode transformers --model medium

# 32 GB+ 内存 Mac
whisperjav video.mp4 --mode transformers --model large-v2

# 最高质量（32 GB+ 内存）
whisperjav video.mp4 --mode transformers --model large-v2 --sensitivity aggressive
```

---

## 6. 处理管线模式选择

本节说明在 Mac 上应使用哪种处理管线模式。

### GPU 加速：`--mode transformers`

`transformers` 处理管线使用 HuggingFace Transformers，完全支持 MPS。这是希望获得 GPU 加速的 Mac 用户的推荐模式。

```bash
whisperjav video.mp4 --mode transformers
```

### 最快 CPU 处理：`--mode faster`

如果您需要最快的纯 CPU 处理，`faster` 模式使用 CTranslate2 优化的 CPU 推理，配合 Apple Accelerate 框架。

```bash
whisperjav video.mp4 --mode faster
```

### 最高精度：`--mode balanced` 或 `--mode transformers`

`balanced` 模式提供完整的预处理管线（场景检测 + 语音活动检测），但语音识别通过 faster-whisper 在 CPU 上运行。`transformers` 模式的语音识别在 MPS GPU 上运行。要获得带 GPU 加速的最高精度：

```bash
whisperjav video.mp4 --mode transformers --sensitivity aggressive
```

### Qwen 处理管线：`--mode qwen`

详见[第 7 节](#7-mac-上的-qwen-处理管线)。

---

## 7. Mac 上的 Qwen 处理管线

### 当前状态（v1.8.9）

Qwen3-ASR 处理管线（`--mode qwen`）是 v1.8.9 的新功能，提供高质量的多语言语音识别。但在 Mac Apple Silicon 上存在一个已知限制：

**Qwen ASR 模块目前无法检测 MPS。** 在 Mac 上运行时，它会回退到 CPU 模式。这是 `qwen_asr.py` 中设备检测仅检查 CUDA 而跳过 MPS 的代码限制。

### 在 Mac 上运行 Qwen

尽管有 CPU 限制，Qwen 仍然可以在 Mac 上工作：

```bash
# Qwen 处理管线（将在 CPU 上运行）
whisperjav video.mp4 --mode qwen

# 使用 assembly 输入模式（推荐用于长内容）
whisperjav video.mp4 --mode qwen --input-mode assembly
```

### 内存注意事项

Qwen 模型较大。在 Mac 上：

| Mac 配置 | 可行性 |
|----------|--------|
| 8 GB 内存 | 不推荐 -- 模型可能无法放入内存 |
| 16 GB 内存 | 可行但紧张 -- 关闭其他应用程序 |
| 32 GB 内存 | 舒适 |
| 64 GB+ 内存 | 最优 |

Qwen 处理管线在 assembly 模式下使用解耦的文本生成和对齐。这意味着模型按顺序加载和卸载，与同时将两者保持在内存中相比，降低了峰值内存使用量。

### 未来 MPS 支持

Qwen 处理管线的 MPS 支持是已知的功能缺口。修复方案是更新 `qwen_asr.py` 中的 `_detect_device()` 以检查 `torch.backends.mps.is_available()`。这可能在未来的版本中解决。请查看发行说明获取更新信息。

---

## 8. GUI 应用程序

### 运行 GUI

```bash
source ~/venvs/whisperjav/bin/activate
whisperjav-gui
```

在 macOS 上，GUI 通过 pywebview 使用原生 WebKit 引擎（WKWebView）。这提供了原生外观的应用程序窗口，无需任何额外的运行时依赖项。

### 注意事项

- 窗口图标可能无法正确显示，因为 WhisperJAV 附带 `.ico` 格式图标（Windows 格式）。macOS 使用 `.icns` 格式。这仅是外观问题。
- GUI 启动一个本地 Web 服务器并在原生窗口中显示界面。它不是 Web 浏览器。
- 如果遇到渲染问题，可以设置 `WHISPERJAV_DEBUG=1` 启用带开发者工具的调试模式。

---

## 9. 本地大语言模型翻译

WhisperJAV 支持在 Apple Silicon 上使用带 Metal 后端的 llama-cpp-python 进行本地大语言模型字幕翻译。这允许无需 API 密钥的离线翻译。

### 安装本地大语言模型支持

在主安装过程中，可以添加本地大语言模型支持：

```bash
python install.py --local-llm
```

或单独安装：

```bash
# 安装脚本会检测 Apple Silicon 并使用 Metal 支持编译
# 这可能需要 10-15 分钟，因为需要从源码编译
CMAKE_ARGS="-DGGML_METAL=on" pip install "llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

或者，检查是否有预编译的 Metal wheel 可用：
```bash
python3 -c "
from whisperjav.translate.llama_build_utils import get_prebuilt_wheel_url
url, desc = get_prebuilt_wheel_url(verbose=True)
if url:
    print(f'Found: {desc}')
    print(f'URL: {url}')
else:
    print('No prebuilt wheel found -- will need to build from source')
"
```

### 使用本地大语言模型翻译

```bash
# 转录并使用本地大语言模型翻译
whisperjav video.mp4 --mode transformers --translate --translate-provider local
```

---

## 10. 性能预期

### 与 NVIDIA GPU（RTX 4090）的对比

| 指标 | RTX 4090 (CUDA) | M2 Pro (MPS) | M2 Pro (CPU) |
|------|-----------------|--------------|--------------|
| Whisper large-v2 (transformers) | ~10 倍实时速率 | ~2-3 倍实时速率 | ~0.5 倍实时速率 |
| Whisper large-v2 (faster-whisper) | ~15 倍实时速率 | 不适用（无 MPS）| ~2 倍实时速率 |
| 模型加载 | 3-5 秒 | 5-10 秒 | 5-10 秒 |
| 内存使用 | GPU 显存 | 统一内存 | 统一内存 |

"实时速率"指音频时长与处理时间的比值。3 倍实时速率意味着 60 分钟视频约需 20 分钟处理。

### 模型大小与内存

| 模型 | 大小 | 最低内存（Mac）| 推荐内存 |
|------|------|----------------|----------|
| tiny | 39 MB | 8 GB | 8 GB |
| base | 74 MB | 8 GB | 8 GB |
| small | 244 MB | 8 GB | 16 GB |
| medium | 769 MB | 16 GB | 16 GB |
| large-v2 | 1.55 GB | 16 GB | 32 GB |
| large-v3 | 1.55 GB | 16 GB | 32 GB |

这些数字包含了音频处理、语音活动检测和场景检测同时运行时的开销。

### 提升性能的技巧

1. **关闭其他应用程序**以为 GPU 释放内存。
2. **使用 `--mode transformers`** 以获得 MPS 加速。
3. 如果处理速度慢或内存不足，**使用较小的模型**。
4. **处理较短的视频**或使用 `--mode faster` 处理长时间批量任务（速度优先于 GPU 加速时）。
5. 遇到内存不足错误时，**启用 MPS 内存回退**：
   ```bash
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   whisperjav video.mp4 --mode transformers
   ```
   这允许 MPS 使用系统内存作为回退，以牺牲速度为代价防止崩溃。

---

## 11. 故障排除

### MPS 不可用

**症状：** `torch.backends.mps.is_available()` 返回 `False`。

**解决方案：**
1. 确认运行的是 macOS 13 (Ventura) 或更高版本。
2. 确认从默认索引安装 PyTorch（而非 CUDA 索引）：
   ```bash
   pip uninstall torch torchaudio
   pip install torch torchaudio
   ```
3. 验证您使用的是 Apple Silicon：
   ```bash
   uname -m
   # 应输出：arm64
   ```
4. 如果在 Rosetta 2（x86 模拟）下运行，MPS 不可用。确保使用原生 ARM64 Python：
   ```bash
   python3 -c "import platform; print(platform.machine())"
   # 应输出：arm64
   ```

### 安装后出现 "No module named 'whisperjav'"

**症状：** 运行 `whisperjav` 命令时出现导入错误。

**解决方案：**
1. 确认虚拟环境已激活：
   ```bash
   source ~/venvs/whisperjav/bin/activate
   ```
2. 验证包已安装：
   ```bash
   pip show whisperjav
   ```
3. 如果以开发模式安装，确保在仓库目录中：
   ```bash
   cd ~/path/to/whisperjav
   pip install --no-deps -e .
   ```

### 找不到 FFmpeg

**症状：** `FileNotFoundError: ffmpeg not found` 或 `FFmpeg not found in PATH`。

**解决方案：**
```bash
brew install ffmpeg
# 验证
which ffmpeg
ffmpeg -version
```

如果安装后 `which ffmpeg` 没有返回结果，可能需要重新加载 shell：
```bash
source ~/.zshrc
```

### NumPy 编译错误

**症状：** 安装 numpy 或 scipy 时出现编译错误。

**解决方案：**
1. 确认已安装 Xcode Command Line Tools：
   ```bash
   xcode-select --install
   ```
2. 尝试安装预编译 wheels：
   ```bash
   pip install --only-binary=:all: "numpy>=2.0.0"
   ```

### pyaudio / PortAudio 错误

**症状：** `ERROR: Could not build wheels for pyaudio` 或 `portaudio.h: No such file or directory`

**解决方案：** 在安装 pyaudio 之前安装 PortAudio 系统库：
```bash
brew install portaudio
pip install pyaudio
```

PortAudio 是 `auditok` 所需的库，`auditok` 用于场景检测。

### soundfile / libsndfile 错误

**症状：** `OSError: cannot load library 'libsndfile.dylib'`

**解决方案：**
```bash
brew install libsndfile
```

然后重新安装 soundfile：
```bash
pip uninstall soundfile
pip install soundfile
```

### MPS 内存不足

**症状：** `RuntimeError: MPS backend out of memory`

**解决方案：**
1. 使用较小的模型：
   ```bash
   whisperjav video.mp4 --mode transformers --model medium
   ```
2. 启用 MPS 内存回退：
   ```bash
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   ```
3. 关闭其他应用程序以释放内存。
4. 如果问题持续，回退到 CPU 模式：
   ```bash
   whisperjav video.mp4 --mode faster
   ```

### 性能低于预期

**症状：** 即使使用 MPS，处理速度仍然很慢。

**检查：**
1. 验证您使用的是 `--mode transformers`（而非 `balanced` 或 `faster`，它们在 Mac 上使用 CPU）：
   ```bash
   whisperjav video.mp4 --mode transformers
   ```
2. 验证 MPS 是否被使用（查看输出中的 "MPS device detected" 或使用 `--verbose` 运行）。
3. 在活动监视器中检查内存压力。如果系统正在使用交换空间，处理会非常慢。

### 安装了 Homebrew Python 但找不到

**症状：** 通过 Homebrew 安装 Python 后出现 `python3: command not found`。

**解决方案：**

在 Apple Silicon 上，Homebrew 安装到 `/opt/homebrew/`。确保它在您的 PATH 中：

```bash
# 添加到 ~/.zshrc
eval "$(/opt/homebrew/bin/brew shellenv)"
```

然后重新加载：
```bash
source ~/.zshrc
```

### PEP 668 错误（"externally-managed-environment"）

**症状：** `error: externally-managed-environment`

当尝试使用 Homebrew Python 全局安装包时会出现此错误。请始终使用虚拟环境：

```bash
python3 -m venv ~/venvs/whisperjav
source ~/venvs/whisperjav/bin/activate
# 然后在 venv 中安装
```

### 安装期间 Git 超时

**症状：** `Failed to connect to github.com port 443 after 21 ms`

**解决方案：**
1. 检查网络连接。
2. 如果在 VPN 或防火墙后，配置 Git 超时：
   ```bash
   git config --global http.connectTimeout 120
   git config --global http.timeout 300
   ```
3. 重试安装。

---

## 12. 已知限制

### CTranslate2 / faster-whisper 不支持 MPS

faster-whisper 后端（`balanced`、`fast` 和 `faster` 模式使用）依赖 CTranslate2，它仅支持 CUDA 和 CPU。在 Mac 上，这些模式会自动回退到 CPU。使用 `--mode transformers` 以获得 MPS 加速。

参考：[CTranslate2 Issue #1562](https://github.com/OpenNMT/CTranslate2/issues/1562)

### Qwen 处理管线在 CPU 上运行

截至 v1.8.9，Qwen ASR 模块（`qwen_asr.py`）不检测 MPS，在 Apple Silicon 上回退到 CPU。这是代码限制，而非框架限制 -- 底层的 `transformers` 库支持 MPS。预计将在未来版本中修复。

### 语音增强后端兼容性

语音增强后端（ClearVoice、BS-RoFormer）在 macOS 上的测试可能有限。如果遇到问题：

```bash
# 安装时跳过语音增强
python install.py --no-speech-enhancement

# 或在运行时跳过
whisperjav video.mp4 --mode transformers --no-enhance
```

### Mac 无独立安装程序

独立安装程序（conda-constructor `.exe`）仅适用于 Windows。Mac 用户必须使用本指南从源码安装。

### Apple Silicon 上的 ONNX Runtime

`onnxruntime` 包（增强 extra 使用）可能没有优化的 Apple Silicon 构建版本。如果安装失败，可以跳过 -- 它仅用于语音增强，转录功能不受影响：

```bash
pip install onnxruntime  # 如果失败，语音增强无法使用，但转录不受影响
```

### openai-whisper 的 MPS 稳定性

虽然 OpenAI Whisper 支持 MPS，但某些操作可能会为了数值稳定性而回退到 CPU。这由 PyTorch 自动处理，可能导致 GPU 利用率比 CUDA 略低。您可能会看到如下警告：
```
UserWarning: MPS: fallback to CPU for op 'aten::...'
```
这些是信息性提示，不表示错误。

---

## 快速参考卡片

```bash
# === 初始设置（一次性）===
xcode-select --install
brew install python@3.12 ffmpeg portaudio git
python3 -m venv ~/venvs/whisperjav
source ~/venvs/whisperjav/bin/activate
pip install --upgrade pip
pip install torch torchaudio

# === 安装 WhisperJAV ===
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
python install.py --cpu-only

# === 日常使用 ===
source ~/venvs/whisperjav/bin/activate

# Mac 推荐方式（MPS GPU 加速）
whisperjav video.mp4 --mode transformers

# GUI
whisperjav-gui

# 检查环境
whisperjav --check
```
