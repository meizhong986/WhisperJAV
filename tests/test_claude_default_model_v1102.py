"""The Claude model WhisperJAV uses by default, and the GUI's Claude menu (1.10.2, owner W622 2, W624 A, W625 1).

Until 1.10.1 the default was claude-3-5-haiku-20241022 and the GUI offered it with claude-3-5-sonnet-20241022 and
claude-3-opus-20240229. None of the three was in Anthropic's model list for the owner's key on 2026-11-12; the one
real run with claude-haiku-5-5 ended with exit 0, 6 of 6 lines. The default is named in WhisperJAV's code (not
PySubtrans's own default, W624 A); both translation windows offer Haiku 5.5, Sonnet 5.5, Opus 5.5, Haiku first.
"""
import re
from pathlib import Path

from whisperjav.translate.providers import PROVIDER_CONFIGS

APP_JS = Path(__file__).resolve().parents[1] / "whisperjav" / "webview_gui" / "assets" / "app.js"
MENU = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]


def test_the_default_claude_model_is_haiku_5_5():
    assert PROVIDER_CONFIGS["claude"]["model"] == "claude-haiku-5-5"


def test_both_translation_windows_offer_the_three_current_models_haiku_first():
    menus = re.findall(r"^\s*claude:\s*\[([^\]]*)\]", APP_JS.read_text(encoding="utf-8"), flags=re.M)
    assert len(menus) == 2, menus  # the Translate tab and the translation settings window
    for menu in menus:
        assert re.findall(r"'([^']*)'", menu) == MENU


def test_the_menu_starts_with_the_default():
    assert MENU[0] == PROVIDER_CONFIGS["claude"]["model"]
