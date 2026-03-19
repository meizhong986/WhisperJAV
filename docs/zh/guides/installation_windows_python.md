# WhisperJAV v1.8.9 -- Windows 安装指南（Python 源码安装）

本指南面向有经验的 Python 开发者，介绍如何在 Windows 上从源码安装 WhisperJAV。如果您需要独立安装程序（无需 Python），请参阅 [Releases 页面](https://github.com/meizhong986/whisperjav/releases)。

---

## 1. 前提条件

### 必需软件

| 软件 | 版本 | 用途 | 下载 |
|------|------|------|------|
| **Python** | 3.10、3.11 或 3.12 | 运行时 | [python.org](https://www.python.org/downloads/) |
| **Git** | 任意近期版本 | 克隆仓库、安装基于 git 的包 | [git-scm.com](https://git-scm.com/download/win) |
| **FFmpeg** | 推荐 6.x 或 7.x | 音视频处理 | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) |

### GPU 加速所需

| 软件 | 版本 | 用途 | 下载 |
|------|------|------|------|
| **NVIDIA GPU 驱动程序** | 450+（CUDA 11.8）或 570+（CUDA 12.8）| GPU 计算 | [nvidia.com](https://www.nvidia.com/Download/index.aspx) |
| **Visual C++ Redistributable** | 2015-2022 (x64) | 原生库支持 | [microsoft.com](https://aka.ms/vs/17/release/vc_redist.x64.exe) |

### GUI 所需

| 软件 | 版本 | 用途 | 下载 |
|------|------|------|------|
| **Microsoft Edge WebView2** | 任意版本 | GUI 渲染引擎 | [microsoft.com](https://go.microsoft.com/fwlink/p/?LinkId=2124703) |

### Python 版本兼容性

- **Python 3.10-3.12：** 完全支持。
- **Python 3.9：** 不支持（因 `pysubtrans` 依赖项已移除支持）。
- **Python 3.13+：** 不支持（`openai-whisper` 无法在 3.13+ 上编译）。

### 验证前提条件

打开命令提示符或 PowerShell 并运行：

```cmd
python --version
git --version
ffmpeg -version
nvidia-smi
```

四个命令都应正常输出结果。如果没有 NVIDIA GPU，`nvidia-smi` 命令会失败，这是正常的 -- WhisperJAV 支持纯 CPU 运行。

### 安装 FFmpeg

FFmpeg 不包含在 Python 中，必须单独安装：

1. 从 [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) 下载 **essentials** 版本。
2. 解压归档文件（例如解压到 `C:\ffmpeg`）。
3. 将 `bin` 目录添加到系统 PATH：
   - 打开**系统属性** > **环境变量**
   - 在**系统变量**下，找到 **Path**，点击**编辑**
   - 点击**新建**，添加 `C:\ffmpeg\bin`
   - 点击**确定**关闭所有对话框
4. 打开新的命令提示符并验证：`ffmpeg -version`

或者，如果您使用包管理器：

```cmd
REM 使用 Chocolatey：
choco install ffmpeg

REM 使用 Scoop：
scoop install ffmpeg

REM 使用 winget：
winget install --id=Gyan.FFmpeg -e
```

---

## 2. 环境设置

**重要：** 请始终在虚拟环境或 conda 环境中安装 WhisperJAV。切勿安装到全局 Python。

### 选项 A：Python venv（推荐大多数用户使用）

```cmd
REM 创建虚拟环境
python -m venv whisperjav-env

REM 激活（命令提示符）
whisperjav-env\Scripts\activate

REM 激活（PowerShell）
whisperjav-env\Scripts\Activate.ps1

REM 验证您在 venv 中（应显示 venv 路径）
where python
```

### 选项 B：Conda / Miniconda

```cmd
REM 创建 Python 3.11 的 conda 环境
conda create -n whisperjav python=3.11 -y

REM 激活
conda activate whisperjav

REM 验证
python --version
```

### 选项 C：使用现有环境

如果您已有用于机器学习的虚拟环境（已安装 PyTorch），可以将 WhisperJAV 安装到其中。确保 PyTorch 是 CUDA 版本，而非纯 CPU 版本：

```cmd
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

如果输出 `CUDA: True`，可以跳过第 4 节中的 PyTorch 安装步骤。

---

## 3. 安装（自动化）

自动化安装程序会处理 GPU 检测、CUDA 选择和分阶段包安装。

### 步骤 1：克隆仓库

```cmd
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
```

### 步骤 2：激活您的环境

```cmd
REM venv：
whisperjav-env\Scripts\activate

REM conda：
conda activate whisperjav
```

### 步骤 3：运行安装程序

```cmd
REM 标准安装（自动检测 GPU）
python install.py

REM 或使用批处理包装器：
installer\install_windows.bat
```

两个命令执行的操作相同。`.bat` 包装器只是定位并运行 `install.py`。

### 安装程序选项

```
--cpu-only              安装仅 CPU 版本的 PyTorch（无 CUDA）
--cuda118               安装用于 CUDA 11.8 的 PyTorch（驱动程序 450+）
--cuda128               安装用于 CUDA 12.8 的 PyTorch（驱动程序 570+，默认）
--no-speech-enhancement 跳过语音增强包（更快的安装）
--minimal               最小化安装（仅转录功能，无 GUI/翻译/增强）
--dev                   以开发/可编辑模式安装（pip install -e）
--local-llm             安装本地大语言模型翻译（预编译 wheel）
--local-llm-build       安装本地大语言模型翻译（从源码编译）
--no-local-llm          跳过本地大语言模型安装
--skip-preflight        跳过磁盘空间和网络检查
--help                  显示所有选项
```

### 常用调用方式

```cmd
REM 标准安装（推荐）
python install.py

REM 强制使用 CUDA 11.8（较旧的 GPU 驱动程序）
python install.py --cuda118

REM 仅 CPU（无 NVIDIA GPU）
python install.py --cpu-only

REM 最小化安装，用于快速测试
python install.py --minimal

REM 开发者安装（可编辑模式）
python install.py --dev

REM 包含本地大语言模型的完整安装
python install.py --local-llm

REM 快速安装（跳过较慢的可选包）
python install.py --no-speech-enhancement --no-local-llm
```

### 安装程序执行的操作

安装程序按以下顺序执行：

1. **预检** -- 验证磁盘空间（8GB 可用）、网络连接、WebView2、VC++ Redistributable
2. **前提条件** -- 验证 Python 版本、FFmpeg、Git
3. **GPU 检测** -- 识别 NVIDIA GPU 和驱动程序版本，选择 CUDA 版本
4. **升级 pip** -- 将 pip 升级到最新版本
5. **PyTorch** -- 使用正确的 CUDA 索引 URL 安装 `torch` 和 `torchaudio`
6. **核心依赖项** -- numpy、scipy、numba、librosa、音频/字幕包
7. **Whisper 包** -- openai-whisper（来自 GitHub）、stable-ts（自定义 fork）、faster-whisper
8. **可选包** -- HuggingFace Transformers、Qwen3-ASR、翻译（pysubtrans、OpenAI、Gemini）、语音活动检测（Silero、TEN）、语音增强（ClearVoice、BS-RoFormer、ModelScope）
9. **GUI 包** -- PyWebView、pythonnet、pywin32
10. **WhisperJAV** -- 安装应用程序本身（使用 `--no-deps` 以保留分阶段环境）
11. **验证** -- 导入 whisperjav 并检查 torch CUDA 状态

### 安装时间

| 配置 | 大致时间 | 备注 |
|------|----------|------|
| 完整安装（含 GPU）| 10-20 分钟 | 取决于网速 |
| 最小化安装 | 5-10 分钟 | 仅转录功能 |
| 仅 CPU | 10-15 分钟 | 稍快（无 CUDA wheels）|

日志文件保存在仓库根目录的 `install_log.txt` 中。

---

## 4. 安装（手动）

如果您希望自行安装各个包，请按以下顺序执行。

### 步骤 1：升级 pip

```cmd
python -m pip install --upgrade pip
```

### 步骤 2：安装包含 CUDA 的 PyTorch

此步骤至关重要。您必须从正确的索引 URL 安装 PyTorch 以获得 GPU 支持。

```cmd
REM 用于 CUDA 12.8（驱动程序 570+，推荐用于新款 GPU）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

REM 用于 CUDA 11.8（驱动程序 450+，通用回退方案）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

REM 仅 CPU（无 NVIDIA GPU）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

验证安装：

```cmd
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

如果显示 `CUDA available: True`，说明 GPU 加速已启用。

### 步骤 3：安装核心依赖项

```cmd
REM 科学计算栈（numpy 必须在 numba 之前）
pip install "numpy>=2.0.0" "scipy>=1.13.0" "numba>=0.60.0"
pip install "librosa>=0.11.0" soundfile pydub pyloudnorm

REM 字幕处理
pip install pysrt srt

REM 工具库
pip install tqdm colorama requests aiofiles regex jsonschema
pip install "pydantic>=2.0,<3.0" "PyYAML>=6.0"

REM 语音活动检测
pip install "silero-vad>=6.2" auditok ten-vad

REM 性能
pip install "psutil>=5.9.0" "scikit-learn>=1.3.0"
```

### 步骤 4：安装 Whisper 包

这些包必须在 PyTorch 之后安装。它们依赖 `torch`，由于已安装了 CUDA 版本的 torch，pip 不会重新下载 CPU 版本。

```cmd
REM OpenAI Whisper（main 分支以获取最新修复）
pip install git+https://github.com/openai/whisper@main

REM Stable-ts（用于日语的自定义 fork）
pip install git+https://github.com/meizhong986/stable-ts-fix-setup.git@main

REM ffmpeg-python（必须使用 git 版本，PyPI tarball 有构建问题）
pip install git+https://github.com/kkroening/ffmpeg-python.git

REM Faster-Whisper（CTranslate2 后端）
pip install "faster-whisper>=1.1.0"
```

### 步骤 5：安装可选包

仅安装您需要的 extras：

```cmd
REM HuggingFace（Qwen3-ASR 和 kotoba-whisper 模型所需）
pip install "huggingface-hub>=0.25.0" "transformers>=4.40.0" "accelerate>=0.26.0" hf_xet

REM Qwen3-ASR（v1.8.9 新增，需要上述 HuggingFace 包）
pip install "qwen-asr>=0.0.6"

REM 翻译
pip install "pysubtrans>=1.5.0" "openai>=1.35.0" "google-genai>=1.39.0"

REM GUI（Windows）
pip install "pywebview>=5.0.0" "pythonnet>=3.0" "pywin32>=305"

REM 语音增强
pip install "modelscope>=1.20" oss2 addict "datasets>=2.14.0,<4.0" simplejson sortedcontainers packaging
pip install git+https://github.com/meizhong986/ClearerVoice-Studio.git#subdirectory=clearvoice
pip install bs-roformer-infer "onnxruntime>=1.16.0"

REM 兼容性（pyvideotrans 互操作）
pip install "av>=13.0.0" "imageio>=2.31.0" "imageio-ffmpeg>=0.4.9" "httpx>=0.27.0" "websockets>=13.0" "soxr>=0.3.0"

REM 分析/可视化
pip install matplotlib Pillow
```

### 步骤 6：安装 WhisperJAV

```cmd
REM 标准安装（从本地源码，保留分阶段依赖项）
pip install --no-deps .

REM 或开发/可编辑模式
pip install --no-deps -e .
```

`--no-deps` 标志是必需的。如果不使用它，pip 会重新解析所有依赖项，可能将您的 CUDA 版 PyTorch 替换为 CPU 版本。

### 步骤 7：验证

```cmd
python -c "import whisperjav; print(f'WhisperJAV {whisperjav.__version__}')"
whisperjav --help
```

---

## 5. 仅安装特定 Extras

如果您只需要某些功能，可以仅安装对应的 extras。但由于 GPU 锁定要求，您应始终先手动安装 PyTorch（上述步骤 2），然后使用 `--no-deps`：

```cmd
REM 错误做法：这会从 PyPI 拉取 CPU 版 PyTorch
pip install "whisperjav[cli]"

REM 正确做法：先安装 PyTorch，然后使用 no-deps 安装 WhisperJAV
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps -e "."
pip install "whisperjav[cli]" --no-deps
```

使用自动化安装程序的简洁方式，可以组合标志：

```cmd
REM 最小化安装（仅转录功能，无 GUI/翻译/增强）
python install.py --minimal

REM 无语音增强（更快的安装）
python install.py --no-speech-enhancement
```

### 可用 Extras

| Extra | 内容 | 使用场景 |
|-------|------|----------|
| `cli` | numpy、scipy、librosa、语音活动检测、scikit-learn | CLI 音频处理 |
| `gui` | pywebview、pythonnet、pywin32 | GUI 应用程序 |
| `translate` | pysubtrans、openai、google-genai | AI 字幕翻译 |
| `llm` | uvicorn、fastapi | 本地大语言模型服务器 |
| `enhance` | modelscope、clearvoice、bs-roformer | 语音增强 |
| `huggingface` | transformers、accelerate、hf_xet | HuggingFace 模型支持 |
| `qwen` | qwen-asr（+ huggingface 依赖项）| Qwen3-ASR 处理管线（v1.8.9+）|
| `analysis` | matplotlib、Pillow | 可视化工具 |
| `compatibility` | av、imageio、httpx、websockets、soxr | pyvideotrans 互操作 |
| `dev` | pytest、ruff、pre-commit | 开发工具 |
| `all` | 以上全部 | 完整安装 |
| `colab` | cli + translate + huggingface | Google Colab |
| `windows` | 同 `all` | Windows 完整体验 |

---

## 6. GPU 设置（CUDA）

### 确定您的 CUDA 版本

PyTorch 的 CUDA 版本取决于您的 NVIDIA 驱动程序版本，而非系统上安装的 CUDA Toolkit 版本。

```cmd
REM 检查您的驱动程序版本
nvidia-smi
```

查看输出头部中的 "Driver Version"：

```
+---------------------------+
| NVIDIA-SMI 570.xx.xx      |   <-- 这是您的驱动程序版本
| Driver Version: 570.xx.xx |
| CUDA Version: 12.8        |   <-- 支持的最高 CUDA 版本
+---------------------------+
```

### 驱动程序到 CUDA 的映射

| 驱动程序版本 | 推荐的 `--index-url` | 标志 |
|-------------|---------------------|------|
| 570+ | `https://download.pytorch.org/whl/cu128` | `--cuda128`（默认）|
| 450-569 | `https://download.pytorch.org/whl/cu118` | `--cuda118` |
| 低于 450 | `https://download.pytorch.org/whl/cpu` | `--cpu-only` |

### 验证 CUDA 是否正常工作

安装完成后，验证 CUDA 是否可用：

```cmd
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"
```

### 切换 CUDA 版本

如果安装了错误的 CUDA 版本，请卸载后重新安装：

```cmd
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### CUDA Toolkit（通常不需要）

WhisperJAV 不需要单独安装 CUDA Toolkit。PyTorch 捆绑了自己的 CUDA 运行时。仅在从源码编译包时（例如 `--local-llm-build`）才需要 CUDA Toolkit。

如有需要：[CUDA Toolkit 下载](https://developer.nvidia.com/cuda-downloads)

---

## 7. 运行 WhisperJAV

### CLI 使用

```cmd
REM 基本转录
whisperjav video.mp4

REM 选择模式
whisperjav video.mp4 --mode balanced    # 完整处理管线（推荐）
whisperjav video.mp4 --mode fast        # 场景检测 + 标准 Whisper
whisperjav video.mp4 --mode faster      # 直接使用 Faster-Whisper（最快）

REM 设置灵敏度
whisperjav video.mp4 --mode balanced --sensitivity aggressive

REM 带翻译
whisperjav video.mp4 --translate

REM 使用 Qwen3-ASR 处理管线（v1.8.9 新增）
whisperjav video.mp4 --mode qwen --input-mode assembly

REM 处理整个目录
whisperjav /path/to/videos/ --mode balanced

REM 查看所有选项
whisperjav --help
```

### GUI 使用

```cmd
REM 启动 GUI
whisperjav-gui
```

GUI 运行要求：
- Microsoft Edge WebView2 运行时
- 已安装 `[gui]` extra（默认安装中已包含）

### 翻译 CLI

```cmd
REM 翻译已有字幕
whisperjav-translate -i subtitles.srt

REM 查看翻译选项
whisperjav-translate --help
```

### 从源码运行（无需安装）

如果您正在开发且未运行 `pip install`：

```cmd
python -m whisperjav.main video.mp4 --mode balanced
python -m whisperjav.webview_gui.main
python -m whisperjav.translate.cli -i subtitles.srt
```

---

## 8. 更新到新版本

### 方法 1：Git Pull + 重新安装（开发模式）

如果以可编辑模式安装（`--dev`）：

```cmd
cd whisperjav
git pull
pip install --no-deps -e .
```

这会更新到最新代码而无需重新下载依赖项。如果新版本添加了新的依赖项，可能需要单独安装或重新运行 `python install.py`。

### 方法 2：完整重新安装

```cmd
cd whisperjav
git pull
python install.py
```

这会重新运行完整安装程序，按需升级各个包。

### 方法 3：升级命令（用于 pip 安装）

```cmd
REM 仅升级 WhisperJAV（不更改依赖项）
pip install -U --no-deps git+https://github.com/meizhong986/whisperjav.git

REM 升级所有依赖项（可能更改 PyTorch -- 请谨慎使用）
pip install -U "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git"
```

### 方法 4：内置升级工具

```cmd
REM 检查更新
whisperjav-upgrade --check

REM 交互式升级
whisperjav-upgrade

REM 仅升级包，跳过依赖项
whisperjav-upgrade --wheel-only
```

---

## 9. 环境变量

WhisperJAV 支持以下环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WHISPERJAV_DEBUG` | `0` | 设为 `1` 启用 GUI 调试模式（DevTools）|
| `WHISPERJAV_NO_ICON` | `0` | 设为 `1` 跳过图标加载（调试渲染问题）|
| `WHISPERJAV_CACHE_DIR` | `.whisperjav_cache` | 元数据缓存目录 |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace 模型缓存位置 |
| `TORCH_HOME` | `~/.cache/torch` | PyTorch 模型缓存位置 |
| `CUDA_VISIBLE_DEVICES` | 所有 GPU | 限制使用特定 GPU（例如 `0`）|

### 设置环境变量（Windows）

```cmd
REM 临时（仅当前会话）
set WHISPERJAV_DEBUG=1

REM 永久（PowerShell，用户范围）
[Environment]::SetEnvironmentVariable("WHISPERJAV_DEBUG", "1", "User")
```

---

## 10. 故障排除

### PyTorch / CUDA 问题

**问题：`torch.cuda.is_available()` 返回 `False`**

原因和解决方案：

1. **安装了错误的 PyTorch 版本（CPU 而非 CUDA）：**
   ```cmd
   pip uninstall torch torchaudio -y
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

2. **驱动程序版本对于所选 CUDA 版本过旧：**
   ```cmd
   REM 检查驱动程序版本
   nvidia-smi
   REM 如果驱动程序 < 570，使用 CUDA 11.8
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **未安装 NVIDIA 驱动程序：**
   从 [nvidia.com](https://www.nvidia.com/Download/index.aspx) 下载。

**问题：`RuntimeError: CUDA out of memory`**

- WhisperJAV 的 large-v2 模型需要约 3-4 GB GPU 显存。
- 关闭其他占用 GPU 的应用程序。
- 尝试使用较小的模型：`whisperjav video.mp4 --model medium`
- 如果有多个 GPU，使用 `CUDA_VISIBLE_DEVICES=0`。

### pip / 包安装问题

**问题：`pip install` 失败并提示 "Could not build wheels"**

```cmd
REM 升级 pip 和构建工具
pip install --upgrade pip setuptools wheel

REM 如需要，安装 Visual C++ Build Tools
REM 下载地址：https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

**问题：依赖项冲突**

```cmd
REM 在新 venv 中重新开始
deactivate
rmdir /s /q whisperjav-env
python -m venv whisperjav-env
whisperjav-env\Scripts\activate
python install.py
```

**问题：`pip install git+https://...` 超时失败**

这在防火墙或 VPN 后很常见。自动化安装程序会自动处理此问题，手动安装时：

```cmd
REM 为 Git 配置延长超时
git config --global http.connectTimeout 120
git config --global http.timeout 300
git config --global http.postBuffer 524288000

REM 重试安装
pip install --timeout 120 git+https://github.com/openai/whisper@main
```

**问题：`numpy` / `numba` 导入错误**

当 numba 在 numpy 之前安装时会出现此问题：

```cmd
pip uninstall numpy numba -y
pip install "numpy>=2.0.0"
pip install "numba>=0.60.0"
```

### FFmpeg 问题

**问题：`FFmpeg is not installed or not in PATH`**

```cmd
REM 验证 FFmpeg 是否可访问
ffmpeg -version

REM 如果未找到，添加到 PATH（示例）
set PATH=C:\ffmpeg\bin;%PATH%

REM 或通过包管理器安装
choco install ffmpeg
```

**问题：`ffmpeg-python` 导入错误**

PyPI 版本的 `ffmpeg-python` 有构建问题。请从 Git 安装：

```cmd
pip install git+https://github.com/kkroening/ffmpeg-python.git
```

### GUI 问题

**问题：GUI 窗口空白或无法打开**

1. 确认 WebView2 已安装：
   从 [microsoft.com](https://go.microsoft.com/fwlink/p/?LinkId=2124703) 下载

2. 检查 pythonnet 是否已安装：
   ```cmd
   pip install "pythonnet>=3.0"
   ```

3. 尝试以调试模式运行：
   ```cmd
   set WHISPERJAV_DEBUG=1
   whisperjav-gui
   ```

**问题：`ImportError: No module named 'webview'`**

```cmd
pip install "pywebview>=5.0.0"
```

### 语音增强问题

**问题：ModelScope 下载失败**

ModelScope 从中国 CDN 下载模型。如果您在中国境外，下载可能较慢。需要 `oss2` 包：

```cmd
pip install oss2 "modelscope>=1.20"
```

**问题：`datasets` 版本冲突**

```cmd
pip install "datasets>=2.14.0,<4.0"
```

`datasets>=4.0` 与 ModelScope 不兼容。

### Qwen3-ASR 问题（v1.8.9）

**问题：`ImportError: No module named 'qwen_asr'`**

```cmd
pip install "qwen-asr>=0.0.6"
```

注意：pip 包名为 `qwen-asr`（连字符），但 Python 导入名为 `qwen_asr`（下划线）。

**问题：Qwen 模型下载缓慢**

模型从 HuggingFace 下载。如需要可设置镜像：

```cmd
set HF_ENDPOINT=https://hf-mirror.com
```

### 通用技巧

1. **始终检查日志文件：** 运行 `python install.py` 后，检查仓库根目录的 `install_log.txt` 获取详细错误信息。

2. **清除 pip 缓存：** 如果包似乎已损坏：
   ```cmd
   pip cache purge
   ```

3. **检查正在使用哪个 Python：**
   ```cmd
   where python
   python -c "import sys; print(sys.executable)"
   ```
   确保指向您的 venv/conda Python，而非系统 Python。

4. **验证您的环境：**
   ```cmd
   python -c "
   import sys
   print(f'Python: {sys.version}')
   print(f'Executable: {sys.executable}')
   print(f'Prefix: {sys.prefix}')
   print(f'In venv: {sys.prefix != sys.base_prefix}')
   "
   ```

5. **首次转录需要额外时间：** WhisperJAV 在首次使用时会下载 AI 模型（~1-3 GB）。这是一次性下载，缓存在 `~/.cache/huggingface/` 中。

---

## 附录：磁盘空间要求

| 组件 | 大小 | 备注 |
|------|------|------|
| Python 包 | ~4-6 GB | PyTorch 最大（~2 GB）|
| Whisper 模型（large-v2）| ~3 GB | 首次使用时下载 |
| Qwen3-ASR 模型 | ~2-3 GB | 首次使用时下载（如使用 Qwen 模式）|
| 语音增强模型 | ~1 GB | 首次使用时下载 |
| **总计（建议可用空间）** | **~15 GB** | 包含临时文件余量 |

## 附录：完整包列表

完整的包及其版本列表请参阅：

- `pyproject.toml` -- Extras 和依赖项规格
- `whisperjav/installer/core/registry.py` -- 所有包的唯一真实来源
- 运行 `python -m whisperjav.installer.validation` 检查您的安装是否与注册表一致
