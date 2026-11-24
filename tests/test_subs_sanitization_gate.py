"""Subtitle sanitization gate (1.10.3, owner W753 CL2, W754; default strict, W756 5; named W772).

strict keeps today's behaviour; relaxed keeps a line holding a relaxed term (気持ちいい, イク, じゃあ, ...) where the
hallucination filter, the assembly cleaner's hallucination list or the non-linguistic filter would remove it. Lines of
sounds only stay removed in both; the non-verbal filter's lone tokens (a lone はい or うん) are not reached.
"""
from pathlib import Path

import pytest

from whisperjav.modules.subtitle_pipeline.cleaners import subs_sanitization_gate as g
from whisperjav.modules.subtitle_pipeline.cleaners.nonlinguistic_utterance_filter import _should_remove_entry
from whisperjav.modules.subtitle_pipeline.cleaners.nonverbal_line_filter import NonverbalLineFilter

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def relaxed(monkeypatch):
    monkeypatch.setenv(g.ENV, "relaxed")


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv(g.ENV, "strict")


def test_strict_is_the_default_and_an_unknown_value_is_strict(monkeypatch):
    monkeypatch.delenv(g.ENV, raising=False)
    assert g.current() == "strict"
    monkeypatch.setenv(g.ENV, "loose")
    assert g.current() == "strict"
    monkeypatch.setenv(g.ENV, " Relaxed ")
    assert g.current() == "relaxed"


def test_a_term_counts_at_the_start_of_the_line_or_after_a_punctuation_mark():
    assert g.holds_relaxed_term("気持ちいい！")
    assert g.holds_relaxed_term("気持ちいい♡")
    assert g.holds_relaxed_term("イ、イク…")
    assert g.holds_relaxed_term("イクイク…")
    assert g.holds_relaxed_term("い、いっちゃ…")       # a line Phase 8 removed in the owner's test of 2026-11-17
    assert g.holds_relaxed_term("いっちゃう")
    assert not g.holds_relaxed_term("はぁ…んっ")
    assert not g.holds_relaxed_term("")


def test_a_term_inside_a_run_of_other_kana_does_not_count():
    # Review of 2026-11-24: the hallucination list holds loops such as くいくい, which contain いく.
    for loop in ("くいくい", "くいくいくいくい", "あんないくそ"):
        assert not g.holds_relaxed_term(loop), loop


def test_strict_changes_nothing(strict):
    assert g.relaxed_keeps("気持ちいい") is False
    assert _should_remove_entry("じゃあ…") is True                 # removed today (Phase 8)


def test_relaxed_keeps_word_lines_the_sound_filter_removes(relaxed):
    for line in ("じゃあ…", "あ、こっち。", "あ、すご…は、はぁ…"):
        assert _should_remove_entry(line) is False, line


def test_relaxed_still_removes_sounds_and_the_lone_tokens(relaxed):
    for line in ("はぁ…", "んっ、んっ", "ちゅぱ"):
        assert _should_remove_entry(line) is True, line
    for token in ("はい。", "うん。", "え?", "あ。"):
        assert NonverbalLineFilter.is_nonverbal_line(token) is True, token


def _remover(phrases):
    from whisperjav.modules.hallucination_remover import HallucinationRemover
    r = HallucinationRemover.__new__(HallucinationRemover)       # no list download: the lists are set here
    r.primary_language = "ja"
    r._exact_lists = {"japanese": set(phrases)}
    r._normalized_exact_lists = {}
    r._regex_patterns = []
    return r


def test_the_hallucination_list_follows_the_setting(monkeypatch):
    r = _remover({"気持ちいい", "ご視聴ありがとうございました", "くいくい"})
    monkeypatch.setenv(g.ENV, "strict")
    assert r.remove_hallucinations("気持ちいい", "ja")[0] == ""
    assert r.remove_hallucinations("気持ちいい♡", "ja")[0] == ""                   # the emoji stage
    monkeypatch.setenv(g.ENV, "relaxed")
    assert r.remove_hallucinations("気持ちいい", "ja") == ("気持ちいい", [])
    assert r.remove_hallucinations("気持ちいい♡", "ja") == ("気持ちいい♡", [])      # before every stage
    assert r.remove_hallucinations("ご視聴ありがとうございました", "ja")[0] == ""   # not a relaxed term: removed
    assert r.remove_hallucinations("くいくい", "ja")[0] == ""                      # a loop, not words: removed


def test_the_assembly_cleaner_list_follows_the_setting(monkeypatch):
    from whisperjav.modules.assembly_text_cleaner import AssemblyCleanerConfig, AssemblyTextCleaner
    cleaner = AssemblyTextCleaner(config=AssemblyCleanerConfig(enabled=True), language="ja")
    cleaner._hallucination_set = {"気持ちいい。", "ご視聴ありがとうございました。"}
    monkeypatch.setenv(g.ENV, "strict")
    assert cleaner._stage_hallucination_filter("気持ちいい。")[0] == ""
    monkeypatch.setenv(g.ENV, "relaxed")
    assert cleaner._stage_hallucination_filter("気持ちいい。")[0] == "気持ちいい。"
    assert cleaner._stage_hallucination_filter("ご視聴ありがとうございました。")[0] == ""


def test_the_switch_is_in_help_and_sets_the_environment():
    src = (REPO / "whisperjav" / "main.py").read_text(encoding="utf-8")
    assert 'add_argument("--subs-sanitization-gate", choices=["strict", "relaxed"], default="strict"' in src
    assert ("os.environ[SUBS_SANITIZATION_GATE_ENV] = getattr(args, 'subs_sanitization_gate', None) or \"strict\""
            in src)
    assert "--cleanup-gating" not in src                       # the name of the 82nd session's draft is gone (W772)


def test_the_gui_passes_relaxed_only_and_shows_the_new_name(monkeypatch):
    api_src = (REPO / "whisperjav" / "webview_gui" / "api.py").read_text(encoding="utf-8")
    assert api_src.count('if options.get("subs_sanitization_gate") == "relaxed":') == 3
    assert api_src.count('if config.get("subs_sanitization_gate") == "relaxed":') == 1
    assert api_src.count('args += ["--subs-sanitization-gate", "relaxed"]') == 4
    assert '"subs_sanitization_gate":    "subsSanitizationGate"' in api_src
    js = (REPO / "whisperjav" / "webview_gui" / "assets" / "app.js").read_text(encoding="utf-8")
    assert js.count("subs_sanitization_gate: document.getElementById('subsSanitizationGate')?.value || 'strict',") == 2
    html = (REPO / "whisperjav" / "webview_gui" / "assets" / "index.html").read_text(encoding="utf-8")
    assert '<option value="strict" selected>Strict (default)</option>' in html
    assert ">Subtitle sanitization gate:</label>" in html
    assert "Same as --subs-sanitization-gate." in html
    zh = (REPO / "whisperjav" / "webview_gui" / "assets" / "i18n_zh.js").read_text(encoding="utf-8")
    assert '"Subtitle sanitization gate:": "字幕清理关卡：",' in zh


def test_a_remembered_relaxed_comes_back(tmp_path, monkeypatch):
    """Review of 2026-11-24: load_gui_settings reads back only the keys of DEFAULT_GUI_SETTINGS."""
    from whisperjav.settings import gui_settings as s
    assert s.DEFAULT_GUI_SETTINGS["subs_sanitization_gate"] == "strict"
    path = tmp_path / "gui_settings.json"                      # a test folder: the user's settings are never read
    monkeypatch.setattr(s, "get_gui_settings_path", lambda: path)
    assert s.save_gui_settings(dict(s.DEFAULT_GUI_SETTINGS, subs_sanitization_gate="relaxed"))
    assert s.load_gui_settings()["subs_sanitization_gate"] == "relaxed"
