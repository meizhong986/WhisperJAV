"""The GUI console shows an API key masked wherever it prints a command (1.10.3, owner W751 B1 5; #304: a user's
key was posted with a pasted console). The process still gets the real key: only the shown text changes.

Run with: python -m pytest tests/test_masked_key_in_console.py -q
"""
from pathlib import Path

from whisperjav.webview_gui.api import KEY_FLAGS, MASKED_KEY, display_command

REPO = Path(__file__).resolve().parents[1]
KEY = "not-a-real-key-only-for-this-test"   # plain words: a key-shaped string would stop the commit check


def test_the_key_after_its_flag_is_masked():
    args = ["movie.mp4", "--ensemble", "--translate-api-key", KEY, "--translate-provider", "deepseek"]
    shown = display_command("whisperjav.main", args)
    assert KEY not in shown
    assert f"--translate-api-key {MASKED_KEY} --translate-provider deepseek" in shown
    assert args[3] == KEY                                   # the arguments themselves are not changed


def test_the_key_written_with_an_equals_sign_is_masked():
    shown = display_command("whisperjav.main", ["a.mp4", f"--translate-api-key={KEY}"])
    assert KEY not in shown and shown.endswith(f"--translate-api-key={MASKED_KEY}")


def test_both_key_flags_are_masked():
    assert set(KEY_FLAGS) == {"--translate-api-key", "--api-key"}
    shown = display_command("whisperjav.translate", ["-i", "a.srt", "--api-key", KEY])
    assert KEY not in shown


def test_other_arguments_are_shown_as_before():
    args = ["C:/My Videos/a b.mp4", "--mode", "qwen", "--pass1-qwen-params", '{"x": 1}']
    assert display_command("whisperjav.main", args) == (
        'whisperjav.main "C:/My Videos/a b.mp4" --mode qwen --pass1-qwen-params "{"x": 1}"')


def test_a_command_without_a_key_is_unchanged():
    assert display_command("whisperjav.main", ["a.mp4", "--mode", "balanced"]) == "whisperjav.main a.mp4 --mode balanced"


def test_every_printed_command_goes_through_the_mask():
    api = (REPO / "whisperjav/webview_gui/api.py").read_text(encoding="utf-8")
    assert api.count('cmd_display = display_command("whisperjav.main", args)') == 3
    assert '" ".join(quote_arg(a) for a in args)' not in api
