"""The language transformers mode passes to generate() (#469): Kotoba bilingual takes the language of its output."""
from whisperjav.modules.transformers_language import generation_language

BILINGUAL = "kotoba-tech/kotoba-whisper-bilingual-v1.0"


def test_bilingual_direct_to_english_passes_en():
    # Model card: Japanese speech -> English text = {"language": "en", "task": "translate"}.
    assert generation_language(BILINGUAL, "ja", "translate") == "en"


def test_bilingual_japanese_subtitles_unchanged():
    # Model card: Japanese ASR = {"language": "ja", "task": "transcribe"}.
    assert generation_language(BILINGUAL, "ja", "transcribe") == "ja"


def test_bilingual_matched_by_name_whatever_the_path():
    assert generation_language("kotoba-tech/kotoba-whisper-bilingual-v1.0/", "ja", "translate") == "en"
    assert generation_language(r"C:\models\kotoba-whisper-bilingual-v1.0", "ja", "translate") == "en"  # a local copy
    assert generation_language("someone/Kotoba-Whisper-Bilingual-v1.1", "ja", "translate") == "en"


def test_standard_whisper_models_keep_the_speech_language():
    for model in ("openai/whisper-large-v3", "kotoba-tech/kotoba-whisper-v2.0", "efwkjn/whisper-ja-anime-v0.3"):
        assert generation_language(model, "ja", "translate") == "ja"
        assert generation_language(model, "ja", "transcribe") == "ja"


def test_empty_model_id():
    assert generation_language("", "ja", "translate") == "ja"
    assert generation_language(None, "ja", "translate") == "ja"
