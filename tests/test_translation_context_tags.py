"""Context metadata must not become subtitle text when a model repeats it."""

from types import SimpleNamespace

from PySubtrans.Options import Options
from PySubtrans.Translation import Translation
from PySubtrans.TranslationParser import TranslationParser

from whisperjav.translate.core import (
    apply_translation_context_filter,
    clean_resumed_translation_tags,
    strip_translation_context_tags,
)
from whisperjav.translate.providers import PROVIDER_CONFIGS


def test_repeated_tags_are_removed_before_pysubtrans_parses_cues():
    raw = (
        "#21\nTranslation>\nFirst <i>emphasis</i>\n"
        "<summary>First batch summary</summary>\n<scene>First scene</scene>\n\n"
        "#22\nTranslation>\nSecond\n"
        "<summary>Latest batch summary</summary>\n<scene>Latest scene</scene>"
    )
    assert "<summary>" in Translation({"text": raw}).text

    class Client:
        def RequestTranslation(self, *args, **kwargs):
            return Translation({"text": raw, "model": "local-model"})

    translator = SimpleNamespace(client=Client())
    assert apply_translation_context_filter(translator)
    result = translator.client.RequestTranslation(object(), streaming_callback=lambda _: None)

    assert "#21\nTranslation>\nFirst <i>emphasis</i>" in result.text
    assert "#22\nTranslation>\nSecond" in result.text
    assert "<summary>" not in result.text
    assert "<scene>" not in result.text
    assert result.summary == "Latest batch summary"
    assert result.scene == "Latest scene"
    assert result.content["model"] == "local-model"
    assert "<summary>" not in result.content["text"]

    parser = TranslationParser("Translation", Options({}))
    lines = parser.ProcessTranslation(result)
    assert [(line.number, line.text) for line in lines] == [
        (21, "First <i>emphasis</i>"),
        (22, "Second"),
    ]


def test_incomplete_tag_cannot_consume_the_next_numbered_cue():
    raw = "#1\nTranslation>\nHello <summary>unfinished\n#2\nTranslation>\nWorld</summary>"
    cleaned, context, count = strip_translation_context_tags(raw)
    assert (cleaned, context, count) == (raw, {}, 0)


def test_resumed_project_lines_are_cleaned_before_they_are_skipped():
    translated = SimpleNamespace(text="Hello\n<summary>Notes</summary>\n<scene>Scene</scene>")
    original = SimpleNamespace(translation=translated.text)
    batch = SimpleNamespace(translated=[translated], originals=[original])
    project = SimpleNamespace(
        subtitles=SimpleNamespace(scenes=[SimpleNamespace(batches=[batch])]),
        needs_writing=False,
    )

    assert clean_resumed_translation_tags(project) == 1
    assert translated.text == "Hello"
    assert original.translation == "Hello"
    assert project.needs_writing


def test_deepseek_direct_default_uses_current_flash_name():
    assert PROVIDER_CONFIGS["deepseek"]["model"] == "deepseek-flash"
