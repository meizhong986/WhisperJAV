#!/usr/bin/env python3
"""The target-language lists must agree (C2, the French defect).

SUPPORTED_TARGETS in whisperjav/translate/providers.py is the source of truth -- it is what
the translation layer accepts and what `whisperjav-translate --target` offers. main.py's
--translate-target choices were hand-kept and had drifted: french was missing, so the GUI
(whose dropdown offers French) emitted --translate-target french and the run died with
"invalid choice: 'french'" and exit 2.

Reads main.py's argparse choices out of the source rather than importing and building the
parser, so nothing heavy is imported.
"""
import ast
import re
from pathlib import Path

import pytest

from whisperjav.translate.providers import SUPPORTED_TARGETS

MAIN = Path(__file__).resolve().parents[1] / "whisperjav" / "main.py"
INDEX_HTML = (Path(__file__).resolve().parents[1] / "whisperjav" / "webview_gui"
              / "assets" / "index.html")


def _main_translate_target_choices():
    """The choices= list attached to --translate-target in whisperjav/main.py."""
    src = MAIN.read_text(encoding="utf-8")
    i = src.index('"--translate-target"')
    m = re.search(r"choices=(\[[^\]]*\])", src[i:i + 1200], re.S)
    assert m, "could not find choices= for --translate-target"
    return set(ast.literal_eval(m.group(1)))


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
    """The reported defect."""
    assert "french" in SUPPORTED_TARGETS
    assert "french" in _main_translate_target_choices()


def test_gui_dropdowns_offer_only_supported_targets():
    """A dropdown option the CLI rejects is a guaranteed failed run."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    for select_id in ("translatorTargetLang", "translationTargetLang"):
        i = html.index('id="%s"' % select_id)
        block = html[i:html.index("</select>", i)]
        offered = set(re.findall(r'<option value="([a-z]+)"', block))
        assert offered <= SUPPORTED_TARGETS, (
            "%s offers unsupported target(s): %s"
            % (select_id, sorted(offered - SUPPORTED_TARGETS)))


def test_output_suffix_stripping_covers_every_target():
    """service.py strips an existing language suffix before writing the new one."""
    from whisperjav.translate import service
    src = Path(service.__file__).read_text(encoding="utf-8")
    assert "_language_suffixes = SUPPORTED_TARGETS" in src, (
        "the suffix list is hand-kept again; it must derive from SUPPORTED_TARGETS")
