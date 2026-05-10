# 常见问题

---

## 通用问题

### WhisperJAV 支持哪些视频格式？

支持 FFmpeg 可以读取的所有格式：MP4、MKV、AVI、MOV、WMV、FLV、WAV、MP3、FLAC、M4A、M4B 等等。只要 FFmpeg 能从中提取音频，WhisperJAV 就能处理。

### 转录需要多长时间？

取决于模式、GPU 等级以及音频的难度。下表给出的范围基于**中端 GPU**（RTX 3060 12 GB 或 Google Colab T4 免费额度）在典型日语对话内容上的实测数据。更高端的 GPU（RTX 3090/4080/4090）大约快 1.5-3 倍；仅 CPU 模式则明显更慢（通常是最慢 GPU 行的 5-10 倍）。

| 模式 | 30 分钟片段 | 2 小时影片 | 说明 |
|------|-------------|------------|------|
| **Faster** | 2-4 分钟 | 6-10 分钟 | 单次 turbo-int8-batch8；最快模式 |
| **Fast** | 3-5 分钟 | 8-15 分钟 | 单次 turbo-int8 |
| **Balanced** | 4-7 分钟 | 15-25 分钟 | 单次 large-v2-int8（推荐默认） |
| **Fidelity** | 10-15 分钟 | 40-50 分钟* | 单次 large-v2-fp16；质量高，但运行时间方差大 |
| **集成模式** | 15-25 分钟 | 60-90 分钟 | 双次（Pass 1 + Pass 2 顺序执行后合并） |

*Fidelity 在同一输入上的运行时间方差较大（约 4-5 倍），原因是 PyTorch cuDNN 自动调优——质量不受影响，仅墙钟时间会变化。

**音频难度倍率**：非常密集的低语内容（ASMR 风格、轻柔呻吟、气音等）会使处理时间增加到上表数值的 **1.2-1.5 倍**，尤其是在 Fidelity 和集成模式下——模型需要在边界帧上花费更多时间。一段 ASMR 较多的 2 小时影片用 Fidelity 处理可能超过 1 小时。

**仅 CPU 模式**：Faster 模式是没有 GPU 时唯一现实的选择。预计 30 分钟片段大约需要 30-60 分钟；对大多数用户而言，多小时内容在 CPU 上不实用。

### 没有 GPU 也能使用 WhisperJAV 吗？

可以。进入 **Advanced**（高级）→ 勾选 **"Accept CPU-only mode"**（接受仅 CPU 模式）。使用 **Faster** 模式可获得最佳 CPU 速度。功能正常，只是明显较慢。

---

## 质量相关

### 哪种处理管线能生成最好的字幕？

没有通用的"最佳"——这取决于内容风格以及您愿意花费的时间。来自实地反馈的常见建议：

- **动漫 / 戏剧化 JAV 对白**：**ChronosJAV** 搭配 anime-whisper。
- **通用日语对白，单次处理**：**Balanced** 是典型默认选择。
- **以墙钟时间换取最高准确度**：**集成模式**（Pass 1 = Balanced + Pass 2 = Qwen3-ASR 配合 Smart Merge），将两个模型的输出合并。
- **低语 / ASMR 较多的内容**：尝试 **Fidelity** 配合 **激进** 灵敏度；它能恢复其他模式可能漏掉的安静语音。

无论选哪种，建议先用 5-10 分钟具有代表性的片段进行对比——JAV 内容的质量因配音员音色差异较大，没有任何单一推荐能在所有内容库中都表现一致。

### 字幕中出现幻觉文本（随机英文短语、URL 等）

这是 Whisper 在静音或极安静片段上的已知行为。WhisperJAV 内置了幻觉过滤器，可移除大部分此类文本。你可以尝试：

1. 尝试不同的**灵敏度**设置——某些内容下 **激进** 模式能捕获 **保守** 模式作为静音丢弃的真实语音（Whisper 随后会用幻觉填充这些"静音"）；其他内容则相反。请用代表性片段在每种设置下试运行，找到最适合您内容库的设置。
2. 使用 **集成模式**——两个模型输出通常会对不同的幻觉产生分歧，因此合并步骤能去除比单次处理更多的幻觉。
3. 启用 **语音增强**（ClearVoice 或 BS-RoFormer）预先清理音频

### 时间轴偏移 / 字幕出现过早或过晚

尝试更换 **场景检测** 方式：

- **Semantic**（v1.8.13+ 默认）— 基于音频特征；对大多数内容是合理的起点。
- **Auditok** — 基于能量；通常更适合对话间存在明显静音的内容。
- **Silero** — 基于 VAD；切分更频繁，根据内容可能改善或损害时间轴。

或者尝试 **ChronosJAV** 处理管线，它使用 TEN VAD 实现更精确的时间轴。

---

## 翻译相关

### 哪个翻译提供商最好？

没有单一答案——每个都有取舍，且质量因内容而异。来自实地反馈的常见模式：

- **DeepSeek**（v1.8.14 默认 `deepseek-v4-flash`）：每 token 成本低；用户经常报告它在日译英/中文方向是不错的平衡。思考变体 `deepseek-v4-pro` 可通过 `--model deepseek-v4-pro` 使用。
- **Claude**（默认 `claude-3-5-haiku-20241022`）和 **OpenAI GPT**（默认 `gpt-4o-mini`）：每 token 成本较高；用户在涉及细微语义或上下文相关的段落时往往更偏爱它们。
- **Gemini**（`gemini-2.0-flash`）：中等成本；质量因内容风格而差异较大。
- **Ollama / 本地大语言模型**：完全在本机运行——数据不会离开您的电脑。质量取决于您运行的本地模型；推荐使用 `qwen2.5:7b-instruct` 或 `gemma3:12b` 以获得合理结果，避免使用思考型模型（其链式思维内容可能泄漏到 SRT 输出中）。
- **OpenRouter**：多个上游模型的路由器。如果您希望对比多个提供商而又不想维护多份 API 密钥，可以使用它。

如果不确定从哪里开始，可以将 30 行示例 SRT 同时通过 DeepSeek 和 Ollama 翻译，挑读起来更顺的版本。

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

## Cohere-Transcribe（预览 —— 推迟到 v1.9.0）

> **状态更新（v1.8.14）：** Cohere-Transcribe 预览**推迟到 v1.9.0**，下拉菜单中的入口已置灰。该模型需要 `transformers ≥ 5.4.0`，但 WhisperJAV 附带的 Qwen3-ASR fork 在 transformers 5.x 下会失败（`@check_model_inputs()` 装饰器 API 变更）。v1.9.0 将协同执行 transformers 升级、Qwen3-ASR fork 补丁以及 Cohere 交付。

> **已升级 transformers 用户的迁移说明：** 如果您最近运行过 `pip install transformers --upgrade`（或以其他方式将 transformers 5.x 安装到了 WhisperJAV 环境中），Qwen3-ASR 将无法加载。请使用 `pip install "transformers>=4.40.0,<5.0"` 恢复稳定，然后通过 `python -c "from qwen_asr import Qwen3ASRModel; print('OK')"` 验证 Qwen3-ASR 是否能正常加载。

下方的设置说明对 v1.9.0 恢复仍然有效 —— 保留作为前向参考。

### 使用 Cohere-Transcribe（预览，可选）—— v1.9.0

Cohere Transcribe-03-2026 在 v1.9.0 中将作为 **ChronosJAV** 下拉菜单的第三种生成器提供，与 Qwen3-ASR 和 Anime-Whisper 并列。该模型在 HuggingFace 上受门控，需要一次性设置。Anime-Whisper 仍是日语调优的默认生成器；Cohere 为可选项。

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
- **Linux：** 在 Ubuntu 上安装 `libwebkit2gtk-4.0-dev`（Ubuntu 24.04+ 改用 `libwebkit2gtk-4.1-dev`），在 Fedora 上安装 `webkit2gtk4.0-devel`
- **macOS：** WebKit 已内置，应可自动运行
- 尝试从命令行启动（`whisperjav-gui`）以查看错误输出
