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


# --- Markdown code fences around the answer (bug report 2026-11-05: Gemma 3 through Ollama) ---

FENCED = (
    "```\n"
    "#1\nTranslation>\nHello.\n\n"
    "#2\nTranslation>\nGood night, while she sleeps.\n\n\n\n"
    "```"
)


def _filtered(raw):
    class Client:
        def RequestTranslation(self, *args, **kwargs):
            return Translation({"text": raw, "model": "gemma3:27b"})

    translator = SimpleNamespace(client=Client())
    assert apply_translation_context_filter(translator)
    return translator.client.RequestTranslation(object(), streaming_callback=lambda _: None)


def test_code_fences_are_removed_before_pysubtrans_parses_cues():
    result = _filtered(FENCED)
    assert "```" not in result.text
    parser = TranslationParser("Translation", Options({}))
    lines = parser.ProcessTranslation(result)
    assert [(line.number, line.text) for line in lines] == [(1, "Hello."), (2, "Good night, while she sleeps.")]
    assert parser.errors == []
    assert result.content["model"] == "gemma3:27b"


def test_without_the_filter_the_fence_fails_the_check():
    """The reporter's failure, kept as the reason for the filter."""
    parser = TranslationParser("Translation", Options({}))
    parser.ProcessTranslation(Translation({"text": FENCED}))
    assert parser.errors


def test_a_fence_with_a_language_word_and_the_tags_are_both_removed():
    raw = ("```text\n#1\nTranslation>\nHello.\n<summary>S</summary>\n<scene>Sc</scene>\n```\n")
    result = _filtered(raw)
    assert "```" not in result.text and "<summary>" not in result.text
    assert result.summary == "S" and result.scene == "Sc"


def test_backticks_inside_a_line_are_kept():
    raw = "#1\nTranslation>\nHe typed ``` in the chat."
    result = _filtered(raw)
    assert result.text == raw
