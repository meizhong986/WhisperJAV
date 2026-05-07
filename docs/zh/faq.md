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

## Cohere-Transcribe（预览）

### 使用 Cohere-Transcribe（预览，可选）

Cohere Transcribe-03-2026 在 v1.8.14 中作为 **ChronosJAV** 下拉菜单的第三种生成器提供，与 Qwen3-ASR 和 Anime-Whisper 并列。该模型在 HuggingFace 上受门控，需要一次性设置。Anime-Whisper 仍是日语调优的默认生成器；Cohere 为可选项。

**一次性设置**

1. 访问 <https://huggingface.co/CohereLabs/cohere-transcribe-03-2026> 并点击 *Agree and access repository*（同意并访问仓库）。
2. 在 <https://huggingface.co/settings/tokens> 创建一个 token（*Read* 权限即可）。
3. 在环境中设置 `HF_TOKEN`：
   - **Windows（持久化）：** `setx HF_TOKEN hf_xxxxxxxxxxxx` — 设置后需重启 GUI/终端才能生效。
   - **Windows（仅当前 PowerShell 会话）：** `$env:HF_TOKEN = "hf_xxxxxxxxxxxx"`
   - **Linux/macOS：** `export HF_TOKEN=hf_xxxxxxxxxxxx`
4. 在 GUI Ensemble 下拉菜单中选择 **ChronosJAV → Cohere-Transcribe (preview)**，或通过 CLI 运行：
   ```bash
   whisperjav --mode qwen --pass1-qwen-params '{"generator_backend":"cohere"}' video.mp4
   ```

**首次运行下载**

Cohere 权重总量约 3.85 GB（`model.safetensors` 原始 FP16 约 4.13 GB，外加分词器和自定义代码文件）。首次转录需先下载 10–30 分钟（每台机器仅一次；之后缓存在 HuggingFace 缓存目录中）。HuggingFace 使用 Xet（内容寻址）传输，会先把分块流到临时位置 — 缓存所在卷请预留**至少 5 GB 空闲空间**以保证下载稳定。

**显存需求**

FP16 约 4–8 GB。用于词级时间戳的 Qwen3-ForcedAligner 在 Cohere 卸载之后才依次加载，因此峰值显存仅由 Cohere 决定 — 在 8–10 GB 显卡上可舒适运行。

### 为什么 Cohere 不产生原生的词级时间戳？

Cohere Transcribe-03-2026 当前只输出文本；逐词时间戳在模型作者的[路线图](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/19)中但尚未实现。WhisperJAV 将 Cohere 与 Qwen3-ForcedAligner-0.6B（与 Qwen3-ASR 流水线使用的同一对齐器）配对，从 Cohere 的输出推导出词级时间戳。您可以通过自定义参数 → *Aligner Backend → None* 禁用对齐器，回退为仅基于 VAD 的段级时间 — 如果您只需要段级字幕并希望省去对齐器加载，可以这样做。

### Cohere 返回了空转录 / "gated" 错误

最常见的原因是启动 WhisperJAV 的环境中没有设置 `HF_TOKEN`。重新检查上述设置步骤，并在设置 token 后重启 GUI（特别是在 Windows 上使用 `setx` 时，它仅对新 shell 生效）。如果错误仍存在，请在 <https://huggingface.co/CohereLabs/cohere-transcribe-03-2026> 确认您的状态为 *Authorized* — 模型作者会手动审核访问。

### Cohere 首次运行时的常见错误

WhisperJAV 会沿异常链回溯并定制错误消息 — 请阅读控制台中的 `[ERROR]` 区块以获取可执行的诊断。最常见的原因：

**磁盘空间不足（`os error 112`、`no space left`、`CAS service error`）**

Cohere 下载在中途耗尽了缓存所在卷的空间。v1.8.14 在下载开始前会预检查空闲空间（至少约 5 GB），但如果您的缓存目录在接近写满的系统盘上，可以释放空间或将缓存重定向到其他盘：

- **Windows（持久化）**：`setx HUGGINGFACE_HUB_CACHE D:\hf_cache` — 之后重启 GUI/终端
- **Windows（仅当前会话）**：`$env:HUGGINGFACE_HUB_CACHE = "D:\hf_cache"`
- **macOS/Linux**：`export HUGGINGFACE_HUB_CACHE=/path/with/space`

重定向后重试 — 下载会在新的缓存位置重新开始。

**上次下载被中断（`Can't load the model... pytorch_model.bin`）**

之前失败的下载在缓存中留下了不完整的目录。请定位 HF 缓存下的 `models--CohereLabs--cohere-transcribe-03-2026/`（Windows 默认 `%USERPROFILE%\.cache\huggingface\hub\`）并删除它，然后重试。

**网络 / 代理问题**

如果出现 connection / timeout / SSL / proxy 错误，请检查网络。企业网络可能需要 `HTTPS_PROXY` / `HF_ENDPOINT`。中国用户：HuggingFace 经常被限流 — 请参阅下方专门的说明。

**身份验证失败 (401)**

您的 `HF_TOKEN` 已过期、被吊销或格式错误。在 <https://huggingface.co/settings/tokens> 重新创建一个 *Read* token 并重新设置 `HF_TOKEN`。

**老旧 transformers 版本上的加载器类问题**

WhisperJAV 根据 Cohere 的 `auto_map` 元数据使用 `AutoModelForSpeechSeq2Seq`（带生成能力的封装）。如果您所用的 transformers 版本不存在该类，WhisperJAV 将回退到 `AutoModel` 并记录警告 — 这会导致 `generate()` 在推理时失败。请升级 `transformers`（WhisperJAV 环境固定的 4.57.6 版本已包含该类）。

### 许可证和 trust_remote_code

- Cohere **模型代码**采用 **Apache-2.0** 许可证。
- 模型**权重**受 Cohere 的门控访问条款约束，您需在 HuggingFace 上接受。WhisperJAV 不再分发 Cohere 权重；它们在首次使用时从 HuggingFace 下载。
- 加载使用 `trust_remote_code=True`，因为 WhisperJAV 固定的 `transformers` 4.57.6 版本尚未原生暴露 Cohere 模型类 — 该类随模型仓库一起发布。这与 WhisperJAV 中 ZipEnhancer/ModelScope 的处理方式相同。

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
