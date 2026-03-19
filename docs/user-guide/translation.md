# AI Subtitle Translation

WhisperJAV can translate Japanese subtitles to other languages using AI language models. Translation works as a standalone tool or integrated into the transcription pipeline.

---

## Supported Providers

| Provider | Type | API Key Required | Best For |
|----------|------|-----------------|----------|
| **Ollama** | Local | No | Privacy, no cost, easy setup. Recommended for local use. |
| **Local LLM** | Local | No | Legacy local option (llama-cpp). Consider Ollama instead. |
| **DeepSeek** | Cloud | Yes | Cost-effective, good CJK quality |
| **Gemini** | Cloud | Yes | Good multilingual support |
| **Claude** | Cloud | Yes | High quality |
| **GPT** | Cloud | Yes | Widely available |
| **OpenRouter** | Cloud | Yes | Access to many models |
| **GLM** | Cloud | Yes | Chinese-related tasks |
| **Groq** | Cloud | Yes | Fast inference |
| **Custom** | Cloud | Varies | Any OpenAI-compatible endpoint |

---

## Setting Up a Provider

### Cloud Providers

1. Get an API key from the provider's website
2. In the GUI, select the provider from the dropdown
3. Enter your API key in the field
4. Click **Test Connection** to verify

!!! tip
    API keys are saved locally and never sent anywhere except the provider's API endpoint.

### Ollama (Recommended for Local)

[Ollama](https://ollama.com/) is the easiest way to run local translation. Install Ollama, then:

```bash
# CLI: translate with Ollama (auto-detects GPU, picks best model for your VRAM)
whisperjav-translate -i subtitles.srt --provider ollama

# Use a specific model
whisperjav-translate -i subtitles.srt --provider ollama --model gemma3:12b

# List locally available Ollama models
whisperjav --list-ollama-models
```

OllamaManager auto-starts the server, detects your GPU, and recommends a model:

| VRAM | Recommended Model |
|------|-------------------|
| CPU only | qwen2.5:3b |
| 8 GB | qwen2.5:7b |
| 12 GB | gemma3:12b |
| 16 GB+ | qwen2.5:14b |

### Local LLM (Legacy)

The Local provider runs a llama-cpp server on your machine. No API key needed, but requires:

- A GPU with ~8GB VRAM
- The `[llm]` extra installed (`pip install whisperjav[llm]`)
- A GGUF model file (downloaded automatically on first use)

!!! note
    Consider switching to Ollama — it's easier to set up, more reliable, and supports more models.

---

## Translation Tone

| Tone | Description |
|------|-------------|
| **Standard** | Clean, natural translations suitable for general audiences |
| **Adult-Explicit** | Specialized instructions tuned for JAV dialogue with appropriate vocabulary |

---

## Two Ways to Translate

### Method 1: Translate During Transcription

Transcribe and translate in one workflow:

1. Set up your transcription on the **Ensemble** tab
2. Check **"AI-translate"** after the merge strategy
3. Select provider and model
4. Click **Start**

Translation runs automatically after transcription completes.

### Method 2: Translate an Existing SRT

Use the standalone translation tab:

1. Go to **AI SRT Translate** (Tab 4)
2. Add your `.srt` file
3. Configure provider, model, target language, and tone
4. Click **Start**

---

## Advanced Settings

| Setting | Default | What It Does |
|---------|---------|-------------|
| **Movie Title** | (empty) | Gives the AI context about the content |
| **Actress Names** | (empty) | Helps the AI handle character names correctly |
| **Plot Summary** | (empty) | Additional context for better translation |
| **Scene Threshold** | 60 sec | Groups subtitles into scenes for batch processing |
| **Max Batch Size** | 30 | Subtitles per API call (lower = fewer token issues) |
| **Max Retries** | 3 | Retry count for failed API calls |

!!! tip "Improving Translation Quality"
    Filling in the Movie Title and Actress Names fields significantly improves translation quality. The AI uses this context to make better word choices and handle names consistently.

### Batch Size Tuning for Local LLMs

Local LLMs have limited context windows compared to cloud APIs. If you see errors like **"Hit API token limit"** or **"No matches found in translation text"**, your batch size is too large for your model's context window.

WhisperJAV auto-adjusts the batch size based on context window size, but you can also set it manually:

```bash
# CLI: set batch size explicitly
whisperjav-translate -i subtitles.srt --provider local --max-batch-size 10

# Or configure it persistently
whisperjav-translate --configure
# When prompted for "Max batch size", enter your preferred value
```

**Recommended batch sizes by model context window:**

| Context Window | Auto-Cap | Recommended Manual | Notes |
|---------------|----------|-------------------|-------|
| 8K (8192) | 11 | 8–12 | gemma-9b, small models |
| 16K (16384) | 27 | 20–27 | Most mid-range models |
| 32K+ | 30 | 30 | Large context models, cloud APIs |

!!! note
    The default batch size of 30 is designed for cloud APIs with 128K+ context windows. For local models, the auto-cap handles most cases automatically. Only set it manually if you still see token limit errors.

---

## CLI Translation

```bash
# Translate with Ollama (local, recommended)
whisperjav-translate -i subtitles.srt --provider ollama

# Translate with DeepSeek (cloud)
whisperjav-translate -i subtitles.srt --provider deepseek --api-key YOUR_KEY

# Translate with adult tone
whisperjav-translate -i subtitles.srt --provider gemini --tone adult

# Translate to Portuguese
whisperjav-translate -i subtitles.srt --target-language portuguese

# Translate to Chinese
whisperjav-translate -i subtitles.srt --target-language Chinese

# Local LLM with reduced batch size (for 8K context models)
whisperjav-translate -i subtitles.srt --provider local --max-batch-size 10
```
