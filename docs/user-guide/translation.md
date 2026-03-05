# AI Subtitle Translation

WhisperJAV can translate Japanese subtitles to other languages using AI language models. Translation works as a standalone tool or integrated into the transcription pipeline.

---

## Supported Providers

| Provider | Type | API Key Required | Best For |
|----------|------|-----------------|----------|
| **Local LLM** | Local | No | Privacy, no cost, offline use |
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

### Local LLM

The Local provider runs a llama-cpp server on your machine. No API key needed, but requires:

- A GPU with ~8GB VRAM
- The `[llm]` extra installed (`pip install whisperjav[llm]`)
- A GGUF model file (downloaded automatically on first use)

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

---

## CLI Translation

```bash
# Translate with DeepSeek
whisperjav-translate -i subtitles.srt --provider deepseek --api-key YOUR_KEY

# Translate with adult tone
whisperjav-translate -i subtitles.srt --provider gemini --tone adult

# Translate to Chinese
whisperjav-translate -i subtitles.srt --target-language Chinese
```
