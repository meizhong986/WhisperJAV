Hi @KenZP12, thanks for reporting.

The urllib3/chardet warning is harmless — it just means those packages are newer than what `requests` was tested against. It shouldn't cause any failures on its own.

Could you share the **actual error** you're seeing when translation fails? The full traceback from the console would help a lot. Specifically:

1. Which translation provider are you using? (local LLM, DeepSeek, Gemini, etc.)
2. What's the error message after the urllib3 warning?

If you're using `--provider local`, there's a known issue where the batch size can exceed the model's context window. Workaround: reduce the batch size in your translation settings:

```bash
whisperjav-translate -i file.srt --provider local --max-batch-size 10
```

A fix for auto-adjusting this is planned for the next release.
