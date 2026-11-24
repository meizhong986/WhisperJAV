"""The console's Copy button copies the console's text with every API key masked (1.10.3, owner W749 4; texts sheet
item 9). The page passes the console's lines and the values of its two key boxes to api.py console_text_for_copy.

The fake keys are built at run time from plain words, so that no key-shaped string is written in this file (the commit
check stops those).
"""
from whisperjav.webview_gui.api import MASKED_KEY, WhisperJAVAPI, mask_keys

SK_SHAPED = "sk-" + "x" * 24          # the shape of an OpenAI, DeepSeek, Anthropic or OpenRouter key
GROQ_SHAPED = "gsk_" + "y" * 24
GOOGLE_SHAPED = "AIza" + "z" * 35


def test_the_value_after_a_key_flag_is_masked_in_both_forms():
    text = ("whisperjav-translate -i a.srt --api-key typed-value-one --provider deepseek\n"
            "whisperjav a.mp4 --translate-api-key=typed-value-two --translate")
    out = mask_keys(text)
    assert "typed-value-one" not in out and "typed-value-two" not in out
    assert f"--api-key {MASKED_KEY} --provider deepseek" in out
    assert f"--translate-api-key={MASKED_KEY} --translate" in out


def test_a_quoted_value_is_masked_whole():
    out = mask_keys('x --api-key "two words" --model m')
    assert out == f"x --api-key {MASKED_KEY} --model m"


def test_the_keys_typed_in_the_gui_are_masked_wherever_they_appear():
    typed = "my own typed key value"
    out = mask_keys(f"Error from provider: key {typed} was refused", [typed, "", "short"])
    assert out == f"Error from provider: key {MASKED_KEY} was refused"


def test_short_values_are_not_taken_for_keys():
    assert mask_keys("pass 1 of 2 done", ["1", "of 2"]) == "pass 1 of 2 done"


def test_key_shaped_strings_are_masked():
    out = mask_keys(f"a {SK_SHAPED} b {GROQ_SHAPED} c {GOOGLE_SHAPED} d")
    assert out == f"a {MASKED_KEY} b {MASKED_KEY} c {MASKED_KEY} d"


def test_ordinary_console_text_is_kept():
    text = ("Audio check: no problems found (3.6 s).\nPass 1: cohere-crispasr, task-ask-for-more, sk-short\n"
            "[1/2] film.mp4 done")
    assert mask_keys(text) == text


def test_the_api_method_returns_the_masked_text():
    api = WhisperJAVAPI.__new__(WhisperJAVAPI)      # no window, no state needed
    result = api.console_text_for_copy(f"line {SK_SHAPED}", None)
    assert result == {"success": True, "text": f"line {MASKED_KEY}"}
