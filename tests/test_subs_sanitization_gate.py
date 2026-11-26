"""Subtitle sanitization gate (1.10.3; owner W753 CL2, W754, W756 5, W772; decided W789-W803).

strict (default) is the clean-up as before and does not read the pass-through file. relaxed lets more terms through: a
line made only of the owner's pass-through words, that a clean-up step would remove, is not removed by that step and
leaves it as it came in; a line the step keeps or only changes, and a loop of the words, get the step's own result;
the CPS check (characters per second) and the repetition rule run as in strict. Each step is tested through its own
code, with the shipped lists, offline, and with the list cache in a test folder (never the user's).
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pysrt
import pytest

from whisperjav.modules import subs_sanitization_gate as g

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "whisperjav" / "data" / "hallucination_filters" / "passthrough_v01.json"
HIS_WORDS = ["気持ちいい", "きもちいい", "気持ち良い", "キモチイイ", "イク", "いく", "逝く"]   # owner W790 2
GIST = ("https://gist.githubusercontent.com/meizhong986/4882bdb3f4f5aa4034a112cebd2e0845/raw/"
        "WhisperJAV_sanitization_passthrough_v01.json")                                    # owner W799 10


@pytest.fixture(autouse=True)
def fresh_gate(monkeypatch, tmp_path):
    """Each test: no remembered words, counts or pass; the list cache in a test folder; no network."""
    import requests
    import whisperjav.modules.hallucination_remover as hr

    def offline(*a, **k):
        raise requests.ConnectionError("offline test")

    monkeypatch.setattr(requests, "get", offline)
    monkeypatch.setattr(hr, "CACHE_DIR", tmp_path / "cache")
    g._reset_for_tests()
    yield
    g._reset_for_tests()


@pytest.fixture
def relaxed(monkeypatch):
    monkeypatch.setenv(g.ENV, "relaxed")


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv(g.ENV, "strict")


def _gate_lines(caplog):
    return [r.getMessage() for r in caplog.records if "Subtitle sanitization gate" in r.getMessage()]


# --- the setting ---------------------------------------------------------------------------------------------------

def test_strict_is_the_default_and_an_unknown_value_is_strict(monkeypatch):
    monkeypatch.delenv(g.ENV, raising=False)
    assert g.current() == "strict"
    monkeypatch.setenv(g.ENV, "loose")
    assert g.current() == "strict"
    monkeypatch.setenv(g.ENV, " Relaxed ")
    assert g.current() == "relaxed"


def test_strict_does_not_read_the_pass_through_file(strict, monkeypatch, caplog):
    def must_not_load():
        raise AssertionError("the pass-through file was read in strict")

    monkeypatch.setattr(g, "load_passthrough", must_not_load)
    with caplog.at_level("DEBUG"):
        assert not g.spares("気持ちいい")
        assert g.words("japanese") == ()
        g.note(g.HALLUCINATION_FILTER)
        g.report_file()
    assert _gate_lines(caplog) == []


# --- the data file and its loader -----------------------------------------------------------------------------------

def test_the_shipped_file_holds_his_words_and_the_approved_use_line():
    data = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert g.validate(data) is None
    assert data["japanese"] == HIS_WORDS
    assert data["english"] == [] and data["chinese"] == [] and data["korean"] == []
    assert "で" not in data["japanese"]                                                   # owner W790 1
    assert data["meta"]["use"] == ("Words that --subs-sanitization-gate relaxed lets through. One word per entry; "
                                   "list each written form (気持ちいい, きもちいい).")              # owner W802 4


def test_the_shipped_file_is_package_data_and_the_address_is_his_gist():
    from whisperjav.config.sanitization_constants import HallucinationConstants
    from whisperjav.data import hallucination_filters as data
    toml = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "hallucination_filters/*.json" in toml
    assert data.get_bundled_passthrough_path() == SHIPPED
    assert HallucinationConstants().PASSTHROUGH_LIST_URL == GIST


@pytest.mark.parametrize("data, why", [
    ([], "not a JSON object"),
    ({"japanese": "イク"}, "is not a list"),
    ({"japanese": ["イク", ""]}, "not a word"),
    ({"japanese": ["イク", "  "]}, "not a word"),
    ({"japanese": ["イク", 3]}, "not a word"),
    ({"japanese": ["イク", "♡"]}, "not a word"),                     # nothing left once normalized
])
def test_a_broken_file_is_refused(data, why):
    assert why in g.validate(data)


class _Response:
    def __init__(self, body: bytes):
        self.content = body

    def raise_for_status(self):
        pass


def _serve(monkeypatch, body: bytes):
    import requests
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: _Response(body))


def test_offline_the_shipped_copy_is_read_and_not_cached(relaxed, tmp_path):
    assert set(g.words("japanese")) == {g.normalize(w) for w in HIS_WORDS}
    assert not (tmp_path / "cache" / g.PASSTHROUGH_CACHE).exists()


def test_a_good_gist_copy_is_used_and_cached(relaxed, monkeypatch, tmp_path):
    _serve(monkeypatch, json.dumps({"japanese": ["イク", "じゃあ"]}).encode("utf-8"))
    assert g.spares("じゃあ")
    cached = json.loads((tmp_path / "cache" / g.PASSTHROUGH_CACHE).read_text(encoding="utf-8"))
    assert cached == {"japanese": ["イク", "じゃあ"]}


def test_a_gist_copy_that_is_not_json_is_refused_with_one_line(relaxed, monkeypatch, caplog):
    _serve(monkeypatch, b"{not json")
    with caplog.at_level("WARNING"):
        assert g.spares("イク")                                       # the shipped copy is read next
    assert _gate_lines(caplog) == [
        "Subtitle sanitization gate: the pass-through words from the gist are refused (not valid JSON)"]


def test_a_broken_cache_is_refused_once_and_the_next_source_read(relaxed, tmp_path, caplog):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / g.PASSTHROUGH_CACHE).write_text('{"japanese": [""]}', encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert g.spares("イク")
    assert _gate_lines(caplog) == [
        "Subtitle sanitization gate: the pass-through words from the cache are refused "
        "(\"japanese\" holds an entry that is not a word: '')"]


def test_a_local_file_address_is_read(relaxed, monkeypatch, tmp_path):
    import whisperjav.config.sanitization_constants as sc
    local = tmp_path / "mine.json"
    local.write_text(json.dumps({"japanese": ["はぁ"]}), encoding="utf-8")
    monkeypatch.setattr(sc, "HallucinationConstants", lambda: SimpleNamespace(PASSTHROUGH_LIST_URL=str(local)))
    assert g.spares("はぁ…") and not g.spares("イク")


def test_no_usable_copy_runs_relaxed_as_strict_and_says_so_once(relaxed, monkeypatch, caplog):
    import whisperjav.config.sanitization_constants as sc
    monkeypatch.setattr(sc, "HallucinationConstants", lambda: SimpleNamespace(PASSTHROUGH_LIST_URL="/no/such/file.json"))
    with caplog.at_level("INFO"):
        assert not g.spares("気持ちいい")
        assert not g.spares("イク")
        g.report_file()
        g.report_file()
    assert _gate_lines(caplog) == [
        "Subtitle sanitization gate: the pass-through words could not be read; relaxed runs as strict in this run."]


def test_strict_reads_the_hallucination_lists_as_before(strict, tmp_path):
    """HEAD's loader: the two lists' own cache names; the pass-through file is not read in strict."""
    from whisperjav.config.sanitization_constants import HallucinationConstants
    from whisperjav.modules.hallucination_remover import HallucinationRemover
    HallucinationRemover(HallucinationConstants(), "ja")
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == ["filter_list.json", "regexp_patterns.json"]


# --- the one test: made only of the words, and not a loop (owner W799 1, W802 1, W803 N14) ---------------------------

@pytest.mark.parametrize("line", [
    "気持ちいい", "気持ちいい。", "気持ちいい!!", "気持ちいい♡", "気持ちいい❤️", "「気持ちいい」", "…気持ちいい…",
    "気持ち いい", "気持ち、いい",                                   # owner W799 4: look-alikes pass
    "イク", "イク…", "イク!!", "イク、、", "「イク」", "(イク)", "★イク★", "イク♡", "ｲｸ", "イクイク", "イクーッ",
    "イクッイクッ", "イク〜〜〜〜!!", "イクーーーー", "いくいく", "イクいく",                 # owner W799 5: words mix
    "逝く", "キモチイイ", "気持ち良い", "イク\nイク",
])
def test_lines_made_only_of_his_words_are_spared(relaxed, line):
    assert g.spares(line), line


@pytest.mark.parametrize("line", [
    "あっ、イク", "あっイク", "もうイク", "もう、イク", "イク🐯 Sound Hodori", "「イク、字幕 山田」",
    "いくら", "イクラ", "くいくい", "あんないくそ", "いっちゃう", "気持ちいいね", "おうくん、気持ちいいね",
    "で、チャンネル登録お願いします", "はぁ…", "", "!!", "♡",
])
def test_other_lines_are_not_spared(relaxed, line):
    assert not g.spares(line), line


@pytest.mark.parametrize("line", [
    "気持ちいい!!" * 30, "イク!!" * 50, "イク!!イク!!イク!!", "イク" * 100, "気持ちいい" * 50, "イク、イク、イク、",
])
def test_a_loop_of_his_words_is_not_spared(relaxed, line):
    """Owner W803 N14: a loop is removed by the repetition rule, as in strict."""
    assert g.is_words(line) and g.is_loop(line) and not g.spares(line), line[:20]


def test_a_word_with_a_space_or_a_mark_inside_matches(relaxed, monkeypatch):
    """Owner W799 1: the word and the line are normalized the same way."""
    monkeypatch.setattr(g, "_words", {"english": (g.normalize("feels good"),),
                                      "japanese": (g.normalize("気持ち、いい"),)})
    assert g.spares("Feels good!", g.ENGLISH) and g.spares("feels good", g.ENGLISH)
    assert not g.spares("Oh, it feels good.", g.ENGLISH)
    assert g.spares("気持ちいい") and g.spares("気持ち いい")


def test_korean_chinese_and_english_have_no_words_yet(relaxed):
    assert not g.spares("イク", g.key_for("ko"))
    assert not g.spares("イク", g.key_for("zh"))
    assert not g.spares("yes", g.ENGLISH)
    assert g.key_for("ja") == g.key_for("japanese") == g.key_for("xx") == "japanese"
    assert g.key_for("zh-tw") == "chinese"


def test_the_test_is_fast_on_a_long_line(relaxed):
    import time
    began = time.time()
    assert not g.is_words("イクイクッ" * 4000 + "ら")
    assert time.time() - began < 2.0


# --- the hallucination filter (legacy pipelines and the GGUF pipelines but MOSS-CrispASR) ----------------------------

def _remover():
    from whisperjav.config.sanitization_constants import HallucinationConstants
    from whisperjav.modules.hallucination_remover import HallucinationRemover
    return HallucinationRemover(HallucinationConstants(), "ja")       # offline: the shipped lists


DECORATED = ["気持ちいい", "気持ちいい。", "気持ちいい!", "気持ちいい!!", "気持ちいい♡", "「気持ちいい」",
             "イク!!", "いく!!", "イクっ!!", "イク、、", "「イク」", "(イク)", "★イク★", "イク♡", "イク〜〜〜〜!!"]
STAY_REMOVED = ["気持ちいい!!" * 30, "イク!!" * 50, "イク!!イク!!イク!!", "イク🐯 Sound Hodori", "「イク、字幕 山田」",
                "で、チャンネル登録お願いします", "ご視聴ありがとうございました", "くいくい"]


def test_strict_removes_these_lines(strict):
    r = _remover()
    for line in DECORATED + STAY_REMOVED:
        assert r.remove_hallucinations(line, "ja")[0] == "", line[:20]


def test_relaxed_keeps_lines_of_his_words_as_written(relaxed, caplog):
    r = _remover()
    for line in DECORATED:
        assert r.remove_hallucinations(line, "ja") == (line, []), line
    for line in STAY_REMOVED:
        assert r.remove_hallucinations(line, "ja")[0] == "", line[:20]
    with caplog.at_level("INFO"):
        g.report_file()
    assert _gate_lines(caplog) == [f"Subtitle sanitization gate (relaxed): lines kept for their pass-through words: "
                                   f"hallucination filter {len(DECORATED)}"]


def test_relaxed_changes_no_line_strict_keeps(monkeypatch):
    """For every line: relaxed gives strict's result, unless strict removes it and it is made only of his words."""
    lines = DECORATED + STAY_REMOVED + [
        "あっ、イク", "もうイク!!", "イク", "イク…", "それでいいよ、もう少しだけ待って", "(笑)気持ちいい", "気持ちいいね♪",
        "ありがとう、ありがとう、ありがとう、ありがとう", "イク〜〜〜〜〜〜〜〜〜〜〜", "はぁ!!", "イクwww", "www",
    ]
    monkeypatch.setenv(g.ENV, "strict")
    r = _remover()
    before = {line: r.remove_hallucinations(line, "ja") for line in lines}
    monkeypatch.setenv(g.ENV, "relaxed")
    for line in lines:
        after = r.remove_hallucinations(line, "ja")
        if before[line][0] == "" and g.spares(line):
            assert after == (line, []), line[:20]
        else:
            assert after == before[line], line[:20]


# --- the text cleaner (ChronosJAV with Qwen3-ASR, decoupled, the CrispASR text cleaners) ----------------------------

def test_the_text_cleaner(monkeypatch):
    from whisperjav.modules.assembly_text_cleaner import AssemblyCleanerConfig, AssemblyTextCleaner

    def stage3(text):
        c = AssemblyTextCleaner(config=AssemblyCleanerConfig(enabled=True), language="ja")
        c._hallucination_set = {"気持ちいい!", "おうくん、気持ちいいね。", "ご視聴ありがとうございました。",
                                "気持ちいい!気持ちいい!気持ちいい!気持ちいい!"}
        c._hallucination_key = "japanese"
        return c._stage_hallucination_filter(text)[0]

    monkeypatch.setenv(g.ENV, "strict")
    assert stage3("気持ちいい!") == ""
    monkeypatch.setenv(g.ENV, "relaxed")
    assert stage3("気持ちいい!") == "気持ちいい!"
    assert stage3("おうくん、気持ちいいね。") == ""
    assert stage3("ご視聴ありがとうございました。") == ""
    assert stage3("気持ちいい!気持ちいい!気持ちいい!気持ちいい!") == ""                    # a loop


# --- the two line filters, MOSS-CrispASR's rules, the split check (through their SRT paths) --------------------------

def _srt(path, items):
    subs = pysrt.SubRipFile()
    for n, (start_ms, end_ms, text) in enumerate(items, 1):
        subs.append(pysrt.SubRipItem(n, pysrt.SubRipTime(milliseconds=start_ms),
                                     pysrt.SubRipTime(milliseconds=end_ms), text))
    subs.save(str(path), encoding="utf-8")


def _texts(path):
    return [s.text for s in pysrt.open(str(path), encoding="utf-8")]


def _lines_srt(tmp_path, name, lines):
    path = tmp_path / name
    _srt(path, [(i * 2000, i * 2000 + 1500, t) for i, t in enumerate(lines)])
    return path


def test_the_sound_line_filter_keeps_a_listed_word_in_relaxed_only(tmp_path, monkeypatch):
    from whisperjav.modules.subtitle_pipeline.cleaners.nonlinguistic_utterance_filter import (
        NonlinguisticUtteranceFilter)
    lines = ["はぁ…", "ちゅぱ", "気持ちいい"]
    monkeypatch.setenv(g.ENV, "strict")
    path = _lines_srt(tmp_path, "strict.srt", lines)
    NonlinguisticUtteranceFilter().filter_srt_file(path)
    assert _texts(path) == ["気持ちいい"]                    # his words are speech to the filter in both settings
    monkeypatch.setenv(g.ENV, "relaxed")
    monkeypatch.setattr(g, "_words", {"japanese": ("はぁ",)})         # a word he might add
    path = _lines_srt(tmp_path, "relaxed.srt", lines)
    NonlinguisticUtteranceFilter().filter_srt_file(path)
    assert _texts(path) == ["はぁ…", "気持ちいい"]


def test_the_lone_token_filter_keeps_a_listed_word_in_relaxed_only(tmp_path, monkeypatch):
    from whisperjav.modules.subtitle_pipeline.cleaners.nonverbal_line_filter import NonverbalLineFilter
    lines = ["はい。", "うん。", "イク"]
    monkeypatch.setenv(g.ENV, "relaxed")
    path = _lines_srt(tmp_path, "a.srt", lines)
    NonverbalLineFilter().filter_srt_file(path)
    assert _texts(path) == ["イク"]                           # no lone token is his word today
    monkeypatch.setattr(g, "_words", {"japanese": ("はい",)})
    path = _lines_srt(tmp_path, "b.srt", lines)
    NonverbalLineFilter().filter_srt_file(path)
    assert _texts(path) == ["はい。", "イク"]


def test_moss_interjection_rules_through_its_clean_up(tmp_path, monkeypatch):
    from whisperjav.crispasr.cleanup import INTERJECTIONS, SrtCleanup
    lines = ["ああ…", "うん", "あっ、気持ちいい"]
    monkeypatch.setenv(g.ENV, "strict")
    path = _lines_srt(tmp_path, "strict.srt", lines)
    SrtCleanup([INTERJECTIONS]).run(path, tmp_path / "raw", "s", len(lines))
    assert _texts(path) == ["あっ、気持ちいい"]
    monkeypatch.setenv(g.ENV, "relaxed")
    monkeypatch.setattr(g, "_words", {"japanese": ("ああ",)})
    path = _lines_srt(tmp_path, "relaxed.srt", lines)
    SrtCleanup([INTERJECTIONS]).run(path, tmp_path / "raw", "r", len(lines))
    assert _texts(path) == ["ああ…", "あっ、気持ちいい"]


def test_the_gguf_clean_up_keeps_his_words_in_relaxed(tmp_path, monkeypatch):
    from whisperjav.crispasr.cleanup import SRT_STAGES, SrtCleanup
    lines = ["気持ちいい!!", "イク!!", "「イク」", "イク!!" * 50, "ご視聴ありがとうございました", "それでいいよ、もう少しだけ待って"]
    monkeypatch.setenv(g.ENV, "strict")
    path = _lines_srt(tmp_path, "strict.srt", lines)
    SrtCleanup(SRT_STAGES).run(path, tmp_path / "raw", "s", len(lines))
    assert _texts(path) == ["それでいいよ、もう少しだけ待って"]
    monkeypatch.setenv(g.ENV, "relaxed")
    path = _lines_srt(tmp_path, "relaxed.srt", lines)
    SrtCleanup(SRT_STAGES).run(path, tmp_path / "raw", "r", len(lines))
    assert _texts(path) == ["気持ちいい!!", "イク!!", "「イク」", "それでいいよ、もう少しだけ待って"]


def test_the_split_check_agrees_with_the_two_filters(tmp_path, monkeypatch):
    """ChronosJAV's split check (orchestrator.line_filter_test) drops what the Phase 8 filters drop, in both settings."""
    from whisperjav.modules.subtitle_pipeline.cleaners.nonlinguistic_utterance_filter import (
        NonlinguisticUtteranceFilter)
    from whisperjav.modules.subtitle_pipeline.cleaners.nonverbal_line_filter import NonverbalLineFilter
    from whisperjav.modules.subtitle_pipeline.orchestrator import line_filter_test
    lines = ["イクッ…はぁ", "はぁ…", "あ。", "はい。", "気持ちいい", "じゃあ…", "ちゅぱ"]
    for state in ("strict", "relaxed"):
        monkeypatch.setenv(g.ENV, state)
        monkeypatch.setattr(g, "_words", {"japanese": ("はぁ", "はい")})
        removed = line_filter_test()
        for line in lines:
            path = _lines_srt(tmp_path, f"{state}.srt", [line])
            NonverbalLineFilter().filter_srt_file(path)
            NonlinguisticUtteranceFilter().filter_srt_file(path)
            assert removed(line) == (not _texts(path)), (state, line)


# --- the CPS check: the same in both settings (owner W802 2, W803 N3, N11) -----------------------------------------

def test_the_cps_check_is_the_same_in_strict_and_relaxed(tmp_path, monkeypatch):
    from whisperjav.config.sanitization_config import SanitizationConfig
    from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer

    def run(state):
        monkeypatch.setenv(g.ENV, state)
        g._reset_for_tests()
        src = tmp_path / state / "x_stitched.srt"
        src.parent.mkdir()
        _srt(src, [(0, 6000, "イク"), (7000, 10700, "息子"), (11000, 13000, "それでいいよ、もう少しだけ待って")])
        s = SubtitleSanitizer(SanitizationConfig(primary_language="ja"))
        s.config.preserve_original_file = True
        return _texts(s.process(src).sanitized_path)

    assert run("strict") == run("relaxed") == ["それでいいよ、もう少しだけ待って"]


# --- the English clean-up -------------------------------------------------------------------------------------------

def test_the_english_clean_up(tmp_path, monkeypatch):
    from whisperjav.modules.subtitle_sanitizer_english import EnglishSubtitleCleaner

    def run(state):
        monkeypatch.setenv(g.ENV, state)
        monkeypatch.setattr(g, "_words", {"english": ("yes",)})              # English words he might add (W789 R7)
        src = tmp_path / f"{state}.srt"
        _srt(src, [(0, 1000, "(yes)"), (2000, 3000, "(applause)"), (4000, 5000, "Thanks for watching"),
                   (6000, 7000, "Yes!"), (8000, 8010, "yes")])                # the last one far too fast (CPS)
        c = EnglishSubtitleCleaner(str(src), str(tmp_path / f"out_{state}"), hallucination_list_url="")
        c.hallucinations = {"yes", "thanksforwatching"}
        clean_path, _log = c.clean()
        return _texts(clean_path)

    assert run("strict") == []
    assert run("relaxed") == ["(yes)", "Yes!"]


# --- the console lines (owner W799 6, 7; W790 7) ------------------------------------------------------------------

def test_relaxed_says_per_file_how_many_lines_each_step_kept(relaxed, caplog):
    with caplog.at_level("INFO"):
        g.note(g.HALLUCINATION_FILTER)
        g.note(g.HALLUCINATION_FILTER)
        g.note(g.NONLINGUISTIC)
        g.report_file()
        g.report_file()
    assert _gate_lines(caplog) == [
        "Subtitle sanitization gate (relaxed): lines kept for their pass-through words: hallucination filter 2, "
        "non-linguistic filter 1",
        "Subtitle sanitization gate (relaxed): lines kept for their pass-through words: none",
    ]


def test_in_an_ensemble_pass_the_lines_name_the_pass(relaxed, monkeypatch, caplog):
    g.set_pass(2)
    _serve(monkeypatch, b"{not json")
    with caplog.at_level("INFO"):
        g.note(g.TEXT_CLEANER)
        g.report_file()
    assert _gate_lines(caplog) == [
        "Subtitle sanitization gate, pass 2: the pass-through words from the gist are refused (not valid JSON)",
        "Subtitle sanitization gate (relaxed, pass 2): lines kept for their pass-through words: text cleaner 1",
    ]


def test_the_pass_worker_tells_the_gate_its_pass():
    """Read, not imported (importing the pass worker loads the ASR stack)."""
    src = (REPO / "whisperjav" / "ensemble" / "pass_worker.py").read_text(encoding="utf-8")
    start = src.index("pass_number = payload.pass_number")
    assert src.index("subs_sanitization_gate.set_pass(pass_number)") > start
    assert src.index("subs_sanitization_gate.set_pass(pass_number)") < src.index("_run_xxl_pass(\n", start)


def test_strict_says_nothing(strict, caplog):
    with caplog.at_level("INFO"):
        g.note(g.HALLUCINATION_FILTER)
        g.report_file()
    assert _gate_lines(caplog) == []


# --- the switch, the GUI, the remembered setting ----------------------------------------------------------------

def test_the_switch_and_its_approved_help_text():
    src = (REPO / "whisperjav" / "main.py").read_text(encoding="utf-8")
    assert 'add_argument("--subs-sanitization-gate", choices=["strict", "relaxed"], default="strict"' in src
    assert "from whisperjav.modules.subs_sanitization_gate import ENV as SUBS_SANITIZATION_GATE_ENV" in src
    assert ("os.environ[SUBS_SANITIZATION_GATE_ENV] = getattr(args, 'subs_sanitization_gate', None) or \"strict\""
            in src)
    for piece in ('help="How strictly the subtitle clean-up treats words that are also common "',      # owner W802 3
                  '"hallucinations. strict (default): the clean-up as before. relaxed: lets more "',
                  '"terms through, such as 気持ちいい and イク, that strict can remove.")'):
        assert piece in src, piece
    assert "--cleanup-gating" not in src


def test_no_v1_or_v3_leftovers_and_no_reading_speed():
    assert not (REPO / "whisperjav" / "modules" / "subtitle_pipeline" / "cleaners" / "subs_sanitization_gate.py").exists()
    for path in (REPO / "whisperjav").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for gone in ("RELAXED_TERMS", "relaxed_keeps", "gate.keeps(", "gate.holds_word(", "taken_entries(",
                     "load_passthrough_list", "READING_SPEED"):
            assert gone not in text, (path, gone)
    gate_src = (REPO / "whisperjav" / "modules" / "subs_sanitization_gate.py").read_text(encoding="utf-8")
    assert "reading-speed" not in gate_src and "reading speed" not in gate_src             # owner W803 N3


def test_the_gui_passes_relaxed_only_and_shows_the_approved_texts():
    api_src = (REPO / "whisperjav" / "webview_gui" / "api.py").read_text(encoding="utf-8")
    assert api_src.count('args += ["--subs-sanitization-gate", "relaxed"]') == 4
    js = (REPO / "whisperjav" / "webview_gui" / "assets" / "app.js").read_text(encoding="utf-8")
    assert js.count("subs_sanitization_gate: document.getElementById('subsSanitizationGate')?.value || 'strict',") == 2
    html = (REPO / "whisperjav" / "webview_gui" / "assets" / "index.html").read_text(encoding="utf-8")
    tip = ("How strictly the subtitle clean-up treats words that are also common hallucinations. Strict (default): "
           "the clean-up as before. Relaxed: lets more terms through, such as 気持ちいい and イク, that strict can "
           "remove. Same as --subs-sanitization-gate.")
    assert f'title="{tip}">Subtitle sanitization gate:</label>' in html
    assert '<option value="strict" selected>Strict (default)</option>' in html
    assert '<option value="relaxed">Relaxed: let more terms through</option>' in html               # owner W803 3
    zh = (REPO / "whisperjav" / "webview_gui" / "assets" / "i18n_zh.js").read_text(encoding="utf-8")
    assert '"Relaxed: let more terms through": "宽松：放行更多词语",' in zh
    assert (f'"{tip}": "字幕清理如何处理那些同时也是常见幻觉的词语。严格（默认）：清理与以前相同。宽松：放行更多词语，'
            f'例如 気持ちいい 和 イク，这些词在严格模式下可能被删除。与 --subs-sanitization-gate 相同。",') in zh


def test_a_remembered_relaxed_comes_back(tmp_path, monkeypatch):
    from whisperjav.settings import gui_settings as s
    assert s.DEFAULT_GUI_SETTINGS["subs_sanitization_gate"] == "strict"
    path = tmp_path / "gui_settings.json"                      # a test folder: the user's settings are never read
    monkeypatch.setattr(s, "get_gui_settings_path", lambda: path)
    assert s.save_gui_settings(dict(s.DEFAULT_GUI_SETTINGS, subs_sanitization_gate="relaxed"))
    assert s.load_gui_settings()["subs_sanitization_gate"] == "relaxed"
