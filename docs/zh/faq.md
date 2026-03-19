# 常见问题

---

## 通用问题

### WhisperJAV 支持哪些视频格式？

支持 FFmpeg 可以读取的所有格式：MP4、MKV、AVI、MOV、WMV、FLV、WAV、MP3、FLAC、M4A、M4B 等等。只要 FFmpeg 能从中提取音频，WhisperJAV 就能处理。

### 转录需要多长时间？

取决于视频时长、处理管线和 GPU：

| 视频时长 | Faster（GPU） | Balanced（GPU） | 集成模式（GPU） | Faster（CPU） |
|----------|---------------|-----------------|-----------------|---------------|
| 30 分钟 | ~1 分钟 | ~2 分钟 | ~4 分钟 | ~10 分钟 |
| 2 小时 | ~3 分钟 | ~5 分钟 | ~10 分钟 | ~40 分钟 |

以上为粗略估计。实际时间因 GPU 型号和音频复杂度而异。

### 没有 GPU 也能使用 WhisperJAV 吗？

可以。进入 **Advanced**（高级）→ 勾选 **"Accept CPU-only mode"**（接受仅 CPU 模式）。使用 **Faster** 模式可获得最佳 CPU 速度。功能正常，只是明显较慢。

---

## 质量相关

### 哪种处理管线能生成最好的字幕？

动漫/JAV 内容：**ChronosJAV** 搭配 anime-whisper，或使用 **集成模式**（Balanced + Qwen3-ASR 配合 Smart Merge）以获得最高准确度。

通用日语内容：**Balanced** 模式是最佳的单次处理选择。

### 字幕中出现幻觉文本（随机英文短语、URL 等）

这是 Whisper 在静音或极安静片段上的已知行为。WhisperJAV 内置了幻觉过滤器，可移除大部分此类文本。你可以尝试：

1. 使用 **激进** 灵敏度（捕获更多语音，减少留给幻觉的"静音"空间）
2. 使用 **集成模式**（两次处理能捕获不同的幻觉）
3. 启用 **语音增强**（ClearVoice 或 BS-RoFormer）预先清理音频

### 时间轴偏移 / 字幕出现过早或过晚

尝试更换 **场景检测** 方式：

- **Semantic**（默认）— 适用于大多数内容
- **Auditok** — 更适合对话间有明显静音的内容
- **Silero** — 切分更为激进

或者尝试 **ChronosJAV** 处理管线，它使用 TEN VAD 实现更精确的时间轴。

---

## 翻译相关

### 哪个翻译提供商最好？

性价比最高：**DeepSeek** 提供优质翻译，费用低廉，尤其适合中日韩语言。

翻译质量最佳：**Claude** 或 **GPT-4** 能产出最自然的翻译，但费用较高。

注重隐私：**本地大语言模型** 完全在本机运行，数据不会离开你的电脑。

### 翻译失败并报 "API token limit" 错误

字幕批次对于模型的上下文窗口来说太大了。请尝试：

1. 在高级设置中减小 **最大批量大小**（尝试 15 或 10）
2. 使用上下文窗口更大的模型
3. WhisperJAV 会自动为本地大语言模型限制批量大小

### Linux 上翻译显示 "Unknown provider: Gemini"

安装缺失的依赖：`pip install google-api-core`。此问题已在 v1.8.6 中修复。

---

## 安装相关

### 安装程序卡住 / 耗时很长

首次安装需要下载约 3-5 GB 的软件包。网络较慢时可能需要 30 分钟以上。请查看安装目录中的安装日志了解进度。

### 提示 "CUDA not available"，但我有 NVIDIA GPU

1. 验证 NVIDIA 驱动版本：`nvidia-smi`
2. CUDA 11.8 需要驱动 450+，CUDA 12.8 需要驱动 570+
3. 你**不需要**安装 CUDA Toolkit —— PyTorch 自带 CUDA 运行时
4. 尝试重新安装对应 CUDA 版本的 PyTorch

### 如何升级？

```bash
whisperjav-upgrade
```

或仅更新代码（更快）：`whisperjav-upgrade --wheel-only`

详情请参阅[升级指南](UPGRADE.md)。

---

## 故障排除

### 处理时出错

1. 查看 GUI 底部的 **控制台** 获取错误信息
2. 在高级选项卡中启用 **调试日志** 并重试 —— 查看 `whisperjav.log`
3. 尝试更简单的处理管线（**Faster** 模式）以定位问题
4. 将错误信息和系统详情提交到 [GitHub Issues](https://github.com/meizhong986/whisperjav/issues)

### GUI 无法启动

- **Windows：** 确保已安装 WebView2（Windows 10 1803+ 和 Windows 11 已内置）
- **Linux：** 安装 `libwebkit2gtk-4.0-dev`（Ubuntu）或 `webkit2gtk4.0-devel`（Fedora）
- **macOS：** WebKit 已内置，应可自动运行
- 尝试从命令行启动（`whisperjav-gui`）以查看错误输出
