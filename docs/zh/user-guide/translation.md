# AI 字幕翻译

WhisperJAV 可以使用 AI 语言模型将日语字幕翻译为其他语言。翻译既可以作为独立工具使用，也可以集成到转录处理管线中。

---

## 支持的提供商

| 提供商 | 类型 | 需要 API 密钥 | 适用场景 |
|--------|------|--------------|----------|
| **Ollama** | 本地 | 否 | 隐私、免费、易于配置。推荐用于本地翻译。 |
| **本地大语言模型** | 本地 | 否 | 旧版本地选项 (llama-cpp)。建议改用 Ollama。 |
| **DeepSeek** | 云端 | 是 | 性价比高，CJK 语言质量好 |
| **Gemini** | 云端 | 是 | 多语言支持好 |
| **Claude** | 云端 | 是 | 高质量 |
| **GPT** | 云端 | 是 | 广泛可用 |
| **OpenRouter** | 云端 | 是 | 可访问多种模型 |
| **GLM** | 云端 | 是 | 适合中文相关任务 |
| **Groq** | 云端 | 是 | 快速推理 |
| **Custom** | 云端 | 视情况 | 任何 OpenAI 兼容端点 |

---

## 配置提供商

### 云端提供商

1. 从提供商网站获取 API 密钥
2. 在 GUI 中从下拉菜单选择提供商
3. 在字段中输入你的 API 密钥
4. 点击 **Test Connection** 验证

!!! tip "提示"
    API 密钥保存在本地，除了提供商的 API 端点外不会发送到任何地方。

### Ollama（推荐用于本地翻译）

[Ollama](https://ollama.com/) 是运行本地翻译最简单的方式。安装 Ollama 后：

```bash
# CLI：使用 Ollama 翻译（自动检测 GPU，根据显存选择最佳模型）
whisperjav-translate -i subtitles.srt --provider ollama

# 使用指定模型
whisperjav-translate -i subtitles.srt --provider ollama --model gemma3:12b

# 列出本地可用的 Ollama 模型
whisperjav --list-ollama-models
```

OllamaManager 自动启动服务器、检测你的 GPU，并推荐模型：

| 显存 | 推荐模型 |
|------|----------|
| 仅 CPU | qwen2.5:3b |
| 8 GB | qwen2.5:7b |
| 12 GB | gemma3:12b |
| 16 GB+ | qwen2.5:14b |

### 本地大语言模型（旧版）

Local 提供商在你的机器上运行 llama-cpp 服务器。无需 API 密钥，但需要：

- 约 8GB 显存的 GPU
- 安装 `[llm]` extra（`pip install whisperjav[llm]`）
- GGUF 模型文件（首次使用时自动下载）

!!! note "说明"
    建议切换到 Ollama — 更易配置、更可靠，且支持更多模型。

---

## 翻译风格

| 风格 | 描述 |
|------|------|
| **Standard** | 干净、自然的翻译，适合一般受众 |
| **Adult-Explicit** | 针对 JAV 对白的专门指令和相应词汇 |

---

## 两种翻译方式

### 方式一：转录时同步翻译

在一个工作流程中完成转录和翻译：

1. 在 **Ensemble** 标签页配置你的转录设置
2. 在合并策略后面勾选 **"AI-translate"**
3. 选择提供商和模型
4. 点击 **Start**

转录完成后自动运行翻译。

### 方式二：翻译已有的 SRT

使用独立翻译标签页：

1. 进入 **AI SRT Translate**（标签页 4）
2. 添加你的 `.srt` 文件
3. 配置提供商、模型、目标语言和风格
4. 点击 **Start**

---

## 高级设置

| 设置 | 默认值 | 作用 |
|------|--------|------|
| **影片标题** | （空） | 为 AI 提供内容相关的上下文 |
| **演员姓名** | （空） | 帮助 AI 正确处理角色名称 |
| **剧情摘要** | （空） | 为更好的翻译提供额外上下文 |
| **场景阈值** | 60 秒 | 将字幕分组为场景进行批处理 |
| **最大批量大小** | 30 | 每次 API 调用的字幕数（越小越不容易超出 token 限制） |
| **最大重试次数** | 3 | API 调用失败的重试次数 |

!!! tip "提升翻译质量"
    填写影片标题和演员姓名字段可以显著提升翻译质量。AI 会利用这些上下文做出更好的用词选择，并一致地处理人名。

### 本地大语言模型的批量大小调优

本地大语言模型的上下文窗口比云端 API 有限。如果出现 **"Hit API token limit"** 或 **"No matches found in translation text"** 等错误，说明你的批量大小对于模型的上下文窗口来说太大了。

WhisperJAV 会根据上下文窗口大小自动调整批量大小，但你也可以手动设置：

```bash
# CLI：显式设置批量大小
whisperjav-translate -i subtitles.srt --provider local --max-batch-size 10

# 或持久化配置
whisperjav-translate --configure
# 当提示 "Max batch size" 时，输入你想要的值
```

**按模型上下文窗口推荐的批量大小：**

| 上下文窗口 | 自动上限 | 推荐手动值 | 备注 |
|-----------|----------|-----------|------|
| 8K (8192) | 11 | 8-12 | gemma-9b，小型模型 |
| 16K (16384) | 27 | 20-27 | 大多数中等规模模型 |
| 32K+ | 30 | 30 | 大上下文窗口模型、云端 API |

!!! note "说明"
    默认批量大小 30 是为具有 128K+ 上下文窗口的云端 API 设计的。对于本地模型，自动上限机制在大多数情况下可以自动处理。只有在仍然遇到 token 限制错误时才需要手动设置。

---

## CLI 翻译

```bash
# 使用 Ollama 翻译（本地，推荐）
whisperjav-translate -i subtitles.srt --provider ollama

# 使用 DeepSeek 翻译（云端）
whisperjav-translate -i subtitles.srt --provider deepseek --api-key YOUR_KEY

# 使用 Adult 风格翻译
whisperjav-translate -i subtitles.srt --provider gemini --tone adult

# 翻译为葡萄牙语
whisperjav-translate -i subtitles.srt --target-language portuguese

# 翻译为中文
whisperjav-translate -i subtitles.srt --target-language Chinese

# 本地大语言模型减小批量大小（适用于 8K 上下文窗口模型）
whisperjav-translate -i subtitles.srt --provider local --max-batch-size 10
```
