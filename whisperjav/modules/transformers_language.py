"""The `language` that transformers mode passes to a Whisper model's generate().

A standard Whisper model reads `language` as the language of the speech, so Direct to English (task "translate")
passes the speech's language, "ja". Kotoba bilingual reads it as the language of the text it writes: its model card
translates Japanese speech to English text with {"language": "en", "task": "translate"}, and English speech to
Japanese text with {"language": "ja", "task": "translate"}
(https://huggingface.co/kotoba-tech/kotoba-whisper-bilingual-v1.0, "Transformers Usage"). Passing "ja" with
"translate" asked it for Japanese text (#469).

Kept apart from transformers_asr.py so it can be tested without importing torch.
"""

# Model names (the part after the last "/") whose `language` is the language of the output text.
OUTPUT_LANGUAGE_MODELS = ("kotoba-whisper-bilingual",)


def generation_language(model_id: str, language: str, task: str) -> str:
    """The `language` for generate(): "en" for Direct to English on a model listed above, else `language`."""
    name = (model_id or "").replace("\\", "/").rstrip("/").split("/")[-1].lower()
    if task == "translate" and name.startswith(OUTPUT_LANGUAGE_MODELS):
        return "en"
    return language
