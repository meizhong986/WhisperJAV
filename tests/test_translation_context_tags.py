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


# --- 1.10.3 (owner W752 1, #460): a block closed by another of the three tags ---

def test_a_mismatched_pair_is_removed_and_its_value_kept_under_the_opening_tag():
    # #460 (2026-11-17): ten subtitles kept "<scene>...</summary>".
    raw = "#5\nTranslation>\nGood morning.\n<scene>A morning at the office</summary>"
    cleaned, context, count = strip_translation_context_tags(raw)
    assert cleaned.rstrip() == "#5\nTranslation>\nGood morning."
    assert context == {"scene": "A morning at the office"} and count == 1


def test_every_combination_of_the_three_tags_is_removed():
    for opening in ("summary", "scene", "synopsis"):
        for closing in ("summary", "scene", "synopsis"):
            raw = f"Hello\n<{opening}>note</{closing}>"
            assert strip_translation_context_tags(raw)[0].rstrip() == "Hello", (opening, closing)


def test_a_mismatched_pair_cannot_consume_the_next_numbered_cue():
    raw = "#1\nTranslation>\nHello <scene>unfinished\n#2\nTranslation>\nWorld</summary>"
    assert strip_translation_context_tags(raw) == (raw, {}, 0)


def test_the_mismatched_pair_is_removed_before_pysubtrans_parses_cues():
    raw = ("#1\nTranslation>\nFirst\n<scene>Scene one</summary>\n\n"
           "#2\nTranslation>\nSecond\n<summary>Latest</summary>")
    result = _filtered(raw)
    assert "<scene>" not in result.text and "</summary>" not in result.text
    parser = TranslationParser("Translation", Options({}))
    assert [(line.number, line.text) for line in parser.ProcessTranslation(result)] == [(1, "First"), (2, "Second")]


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


# --- 1.10.3 (owner W752 1 (b), W764 2: texts sheet item 8): the end warning names lines still holding a tag ---

def _srt(tmp_path, texts):
    path = tmp_path / "out.english.srt"
    path.write_text("".join(f"{n}\n00:00:0{n},000 --> 00:00:0{n},900\n{t}\n\n" for n, t in enumerate(texts, 1)),
                    encoding="utf-8")
    return path


def test_the_end_check_names_a_subtitle_with_a_tag_left(tmp_path):
    from whisperjav.translate.core import srt_format_marker_entries
    path = _srt(tmp_path, ["Hello.", "Good night. <scene>unfinished note", "A note</summary>", "Bye."])
    assert srt_format_marker_entries(path) == [2, 3]


def test_the_end_check_passes_a_clean_file(tmp_path):
    from whisperjav.translate.core import srt_format_marker_entries
    assert srt_format_marker_entries(_srt(tmp_path, ["Hello.", "<i>Good night.</i>", "Bye."])) == []


def test_the_end_warning_text_and_the_retry_check():
    """The warning's list of what it looks for gains the tags (the approved wording); the retry check and the retry's
    instruction to the model do not change."""
    from whisperjav.translate.core import END_LABELS_TEXT, FORMAT_LABELS_TEXT, format_marker_lines
    assert END_LABELS_TEXT == ("Translation>, Original>, the same labels in another language such as 翻译>, "
                               "or a #number line, or a <summary>, <scene> or <synopsis> tag")
    assert FORMAT_LABELS_TEXT == ("Translation>, Original>, the same labels in another language such as 翻译>, "
                                  "or a #number line")
    lines = [SimpleNamespace(number=1, text="Hello <scene>note")]
    assert format_marker_lines(lines) == []
