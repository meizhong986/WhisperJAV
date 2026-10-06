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


def test_runs_sharing_the_file_never_see_it_empty_and_never_fail(tmp_path):
    """Review finding: two runs of the same tone share the temp instruction file. A plain rewrite could be read
    empty (literal {LANG} to the model); on Windows a file another run has open cannot be replaced at all.
    write_shared_text leaves a file that already holds the text alone, and falls back to a private copy."""
    import threading
    target = tmp_path / "instructions_standard.txt"
    text = get_instruction_content("standard")
    errors, bad, stop = [], [], threading.Event()

    def writer():
        try:
            while not stop.is_set():
                used = core.write_shared_text(target, text)
                if Path(used).read_text(encoding="utf-8") != text:
                    bad.append("writer got wrong text")
        except Exception as e:          # a failing writer must fail the test, not die silently
            errors.append(repr(e))

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for t in threads:
        t.start()
    try:
        for _ in range(300):
            try:
                seen = target.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            if seen != text:
                bad.append(len(seen))
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert errors == [] and bad == []


def test_a_file_held_open_by_another_run_is_not_replaced(tmp_path):
    target = tmp_path / "instructions_standard.txt"
    target.write_text("old text", encoding="utf-8")
    with open(target, encoding="utf-8") as held:          # another run is reading it
        used = core.write_shared_text(target, "new text")
        assert Path(used).read_text(encoding="utf-8") == "new text"
        assert held.read() == "old text"


def test_the_same_text_is_not_rewritten(tmp_path):
    import os
    target = tmp_path / "instructions_standard.txt"
    core.write_shared_text(target, "same")
    os.utime(target, (1, 1))
    assert core.write_shared_text(target, "same") == str(target)
    assert target.stat().st_mtime == 1


def test_different_texts_get_different_filled_copies(tmp_path):
    a, b = tmp_path / "a" / "instructions_standard.txt", tmp_path / "b" / "instructions_standard.txt"
    a.parent.mkdir(); b.parent.mkdir()
    a.write_text("### instructions\nA {LANG}\n", encoding="utf-8")
    b.write_text("### instructions\nB {LANG}\n", encoding="utf-8")
    assert core.fill_target_language(a, "chinese") != core.fill_target_language(b, "chinese")
