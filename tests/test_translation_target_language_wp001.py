"""WP-001 / #347 (owner, 2026-10-06): the instructions name the target language. Bundled files carry {LANG};
WhisperJAV fills it in before PySubtrans reads the file, and 1.9.4 reads the bundled files first (the Gists stay
unchanged for 1.9.3). The end-to-end check of what the model receives is in the commit message."""
from pathlib import Path

import pytest

from whisperjav.translate import core
from whisperjav.translate.instructions import get_instruction_content

TONES = ("standard", "pornify", "contextual")


@pytest.mark.parametrize("tone", TONES)
def test_bundled_file_is_used_and_names_the_language(tone, monkeypatch):
    import whisperjav.translate.instructions as ins
    monkeypatch.setattr(ins, "fetch_from_gist", lambda *a, **k: pytest.fail("Gist must not be fetched"))
    text = get_instruction_content(tone)
    assert text.count("{LANG}") == 6


@pytest.mark.parametrize("tone", TONES)
def test_fill_for_chinese_and_english(tone, tmp_path):
    src = tmp_path / f"{tone}.txt"
    src.write_text(get_instruction_content(tone), encoding="utf-8")
    zh = Path(core.fill_target_language(src, "chinese")).read_text(encoding="utf-8")
    assert "{LANG}" not in zh
    assert "Translate into Chinese. Write every translated line, and the <summary> and <scene> tags, in Chinese. " \
           "Do not answer in English." in zh
    en = Path(core.fill_target_language(src, "english")).read_text(encoding="utf-8")
    assert "{LANG}" not in en and "Do not answer in English" not in en
    assert "Translate into English." in en


def test_a_users_own_file_without_the_tag_is_passed_unchanged(tmp_path):
    own = tmp_path / "mine.txt"
    own.write_text("### instructions\nmy rules\n", encoding="utf-8")
    assert core.fill_target_language(own, "chinese") == str(own)


def test_language_names():
    assert core.target_language_name("chinese") == "Chinese"
    assert core.target_language_name("") == "the target language"
