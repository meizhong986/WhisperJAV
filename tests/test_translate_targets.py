#!/usr/bin/env python3
"""The target-language lists must agree, and translated files must be named correctly.

SUPPORTED_TARGETS in whisperjav/translate/providers.py is the source of truth. main.py's
--translate-target choices and both GUI dropdowns are held to it here.

Naming is exercised through whisperjav/translate/output_naming.py, which cli.py and
service.py both call. Earlier versions of two tests in this file could not fail for the
regressions they named: one asserted that a literal string appeared in service.py (a
narrowed list such as `SUPPORTED_TARGETS - {'french'}` satisfies that substring), and the
GUI check was subset-only, so a language added to SUPPORTED_TARGETS and forgotten in
index.html still passed.

Imports only providers.py and output_naming.py -- no ASR module, and not cli.py, whose
module-level argument parsing exits on import.
"""
import ast
import re
from pathlib import Path

import pytest

from whisperjav.translate.output_naming import (
    SOURCE_LANGUAGE_SUFFIXES,
    generate_output_path,
    strip_language_suffix,
)
from whisperjav.translate.providers import SUPPORTED_TARGETS

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "whisperjav" / "main.py"
INDEX_HTML = ROOT / "whisperjav" / "webview_gui" / "assets" / "index.html"
TRANSLATE = ROOT / "whisperjav" / "translate"


def _main_translate_target_choices():
    """The choices= list attached to --translate-target in whisperjav/main.py.

    Parsed from source rather than imported: importing whisperjav.main loads torch,
    stable_whisper, transformers, faster_whisper and whisper.
    """
    src = MAIN.read_text(encoding="utf-8")
    i = src.index('"--translate-target"')
    m = re.search(r"choices=(\[[^\]]*\])", src[i:i + 1200], re.S)
    assert m, "could not find choices= for --translate-target"
    return set(ast.literal_eval(m.group(1)))


def _gui_options(select_id):
    html = INDEX_HTML.read_text(encoding="utf-8")
    i = html.index('id="%s"' % select_id)
    block = html[i:html.index("</select>", i)]
    return set(re.findall(r'<option value="([a-z]+)"', block))


# --- the four lists must agree -------------------------------------------------------

def test_cli_offers_every_supported_target():
    missing = SUPPORTED_TARGETS - _main_translate_target_choices()
    assert not missing, (
        "--translate-target rejects target(s) the translation layer supports: %s"
        % sorted(missing))


def test_cli_offers_nothing_the_translation_layer_would_refuse():
    extra = _main_translate_target_choices() - SUPPORTED_TARGETS
    assert not extra, (
        "--translate-target accepts target(s) service.py would raise on: %s" % sorted(extra))


def test_french_specifically():
    """The originally reported defect."""
    assert "french" in SUPPORTED_TARGETS
    assert "french" in _main_translate_target_choices()


@pytest.mark.parametrize("select_id", ["translatorTargetLang", "translationTargetLang"])
def test_gui_dropdowns_match_supported_targets_exactly(select_id):
    """Both directions: a missing option is as much a defect as an unsupported one."""
    offered = _gui_options(select_id)
    assert offered == SUPPORTED_TARGETS, (
        "%s is out of step: missing %s, extra %s"
        % (select_id, sorted(SUPPORTED_TARGETS - offered), sorted(offered - SUPPORTED_TARGETS)))


# --- output naming: exercise the code, do not grep it --------------------------------

@pytest.mark.parametrize("target", sorted(SUPPORTED_TARGETS))
def test_an_existing_target_suffix_is_replaced_not_stacked(target):
    out = Path(generate_output_path("/media/SONE-853.french.srt", target)).name
    assert out == "SONE-853.%s.srt" % target


@pytest.mark.parametrize("suffix", sorted(SOURCE_LANGUAGE_SUFFIXES))
def test_source_language_suffixes_are_replaced(suffix):
    out = Path(generate_output_path("/media/SONE-853.%s.srt" % suffix, "english")).name
    assert out == "SONE-853.english.srt"


@pytest.mark.parametrize("target", sorted(SUPPORTED_TARGETS))
def test_every_target_survives_a_round_trip(target):
    """Translate to X, feed the result back, translate to English."""
    first = generate_output_path("/media/SONE-853.srt", target)
    assert Path(first).name == "SONE-853.%s.srt" % target
    second = generate_output_path(first, "english")
    assert Path(second).name == "SONE-853.english.srt", (
        "round trip through %s stacked suffixes: %s" % (target, Path(second).name))


def test_an_unrelated_dotted_stem_is_left_alone():
    assert Path(generate_output_path("/media/SONE-853.v2.srt", "english")).name \
        == "SONE-853.v2.english.srt"
    assert strip_language_suffix("SONE-853.v2") == "SONE-853.v2"


def test_a_stem_with_no_dot_is_left_alone():
    assert strip_language_suffix("SONE-853") == "SONE-853"


# --- neither naming path may keep its own list again ---------------------------------

def test_both_naming_paths_use_the_shared_helper():
    """cli.py knew only japanese/english/ja/en/jp; service.py lacked portuguese and french."""
    cli_src = (TRANSLATE / "cli.py").read_text(encoding="utf-8")
    svc_src = (TRANSLATE / "service.py").read_text(encoding="utf-8")
    assert "from .output_naming import generate_output_path" in cli_src
    assert "from .output_naming import strip_language_suffix" in svc_src
    for src, name in ((cli_src, "cli.py"), (svc_src, "service.py")):
        assert "'japanese', 'english', 'ja', 'en', 'jp'" not in src, (
            "%s has a hand-kept language list again" % name)
