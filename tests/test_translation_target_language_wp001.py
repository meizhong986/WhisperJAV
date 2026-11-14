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


def _runs_at_once(target, texts, rounds=200):
    """Each text is one run of the same tone: it asks write_shared_text for the file and reads the path it is given,
    as PySubtrans then does, ``rounds`` times; all runs start together. Returns (errors, wrong reads)."""
    import threading
    errors, bad = [], []
    start = threading.Barrier(len(texts))

    def run(text):
        try:
            start.wait()
            for _ in range(rounds):
                used = core.write_shared_text(target, text)
                if Path(used).read_text(encoding="utf-8") != text:
                    bad.append(Path(used).name)
        except Exception as e:          # a failing run must fail the test, not die silently
            errors.append(repr(e))

    threads = [threading.Thread(target=run, args=(text,)) for text in texts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors, bad


def test_runs_sharing_the_file_never_see_it_empty_and_never_fail(tmp_path):
    """Review finding: two runs of the same tone share the temp instruction file. A plain rewrite could be read
    empty (literal {LANG} to the model); on Windows a file another run has open cannot be replaced at all.
    write_shared_text leaves a file that already holds the text alone, and falls back to a private copy.
    Here: two runs of the same text start together, creating the file, and each reads the path it was given; no
    read is refused and none gets other text (before 1.10.2, on Windows: refused in 1, 2 and 0 of 20 rounds in three
    measurements, 2026-11-13; a chance race). (The
    test's earlier form also polled the file's name from outside any run; on Windows that read is refused at the
    instant any file is put in place under the name, with or without this fix.)"""
    text = get_instruction_content("standard")
    assert _runs_at_once(tmp_path / "instructions_standard.txt", [text, text]) == ([], [])


def test_runs_with_different_texts_never_get_each_others_file(tmp_path):
    """1.10.2 (owner W633 3): two runs of the same tone whose instruction texts differ (an install of 1.9.3 or older,
    or the Gist's text when a bundled file is missing; the temp folder is shared by every run on the computer).
    Before, both used the one name instructions_<tone>.txt and replaced the file in turn (20 of 20 rounds failed,
    2026-11-13). Now each text has its own name; this guards that two texts never share one."""
    text = get_instruction_content("standard")
    assert _runs_at_once(tmp_path / "instructions_standard.txt", [text, text + "\nAnother copy.\n"]) == ([], [])


def test_a_run_keeps_its_text_when_another_text_is_written_later(tmp_path):
    """1.10.2 (owner W633 3), the real window: the command line gets the file once and PySubtrans reads it again for
    every media file. Another run writing another text of the same tone meanwhile must not change it (before: the
    first run's later files got the other text)."""
    target = tmp_path / "instructions_standard.txt"
    mine = core.write_shared_text(target, "TEXT A")
    core.write_shared_text(target, "TEXT B")
    assert Path(mine).read_text(encoding="utf-8") == "TEXT A"


def test_a_text_with_cr_lf_is_matched_and_leaves_no_copies(tmp_path):
    target = tmp_path / "instructions_standard.txt"
    text = "line one\r\nline two\r\n"
    first = core.write_shared_text(target, text)
    assert core.write_shared_text(target, text) == first
    assert Path(first).read_bytes() == text.encode("utf-8")
    assert [p.name for p in tmp_path.iterdir()] == [Path(first).name]


def test_a_file_of_this_name_edited_by_hand_is_repaired(tmp_path):
    target = tmp_path / "instructions_standard.txt"
    shared = core.shared_text_path(target, "the text")
    shared.write_text("edited", encoding="utf-8")
    assert core.write_shared_text(target, "the text") == str(shared)
    assert shared.read_text(encoding="utf-8") == "the text"
    assert [p.name for p in tmp_path.iterdir()] == [shared.name]


def test_a_file_held_open_by_another_run_is_not_replaced(tmp_path):
    target = tmp_path / "instructions_standard.txt"
    shared = core.shared_text_path(target, "new text")
    shared.write_text("old text", encoding="utf-8")      # e.g. a file someone edited by hand
    with open(shared, encoding="utf-8") as held:          # another run is reading it
        used = core.write_shared_text(target, "new text")
        assert Path(used).read_text(encoding="utf-8") == "new text"
        assert held.read() == "old text"


def test_the_same_text_is_not_rewritten(tmp_path):
    import os
    target = tmp_path / "instructions_standard.txt"
    used = core.write_shared_text(target, "same")
    os.utime(used, (1, 1))
    assert core.write_shared_text(target, "same") == used
    assert Path(used).stat().st_mtime == 1


def test_each_text_has_its_own_shared_file_and_none_is_replaced(tmp_path, monkeypatch):
    """1.10.2 (owner W633 3): the name carries the text's code; a shared file that holds the text is used as it is,
    and the write never calls os.replace (which replaces a file another run may have open)."""
    import os
    monkeypatch.setattr(os, "replace", lambda *a, **k: pytest.fail("a shared file must not be replaced"))
    target = tmp_path / "instructions_standard.txt"
    a, b = core.write_shared_text(target, "text A"), core.write_shared_text(target, "text B")
    assert a != b and Path(a).parent == Path(b).parent == tmp_path
    assert Path(a).name.startswith("instructions_standard_") and Path(a).suffix == ".txt"
    assert Path(a).read_text(encoding="utf-8") == "text A" and Path(b).read_text(encoding="utf-8") == "text B"
    assert core.write_shared_text(target, "text A") == a
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([Path(a).name, Path(b).name])   # no copies left


def test_different_texts_get_different_filled_copies(tmp_path):
    a, b = tmp_path / "a" / "instructions_standard.txt", tmp_path / "b" / "instructions_standard.txt"
    a.parent.mkdir(); b.parent.mkdir()
    a.write_text("### instructions\nA {LANG}\n", encoding="utf-8")
    b.write_text("### instructions\nB {LANG}\n", encoding="utf-8")
    assert core.fill_target_language(a, "chinese") != core.fill_target_language(b, "chinese")
