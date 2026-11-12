r"""XXL's extra arguments keep their backslashes (1.10.2, still-open item 8 of the 1.10.1 notes).

shlex.split (POSIX rules) read a backslash as an escape: --model_dir C:\Models\xxl arrived as C:Modelsxxl. The
split now treats a backslash as an ordinary character; quotes still group and are removed, an unclosed quote still
fails, and '#' starts no comment.
"""
import pytest

from whisperjav.byop.xxl_runner import _split, strip_model_args


@pytest.mark.parametrize("text, tokens", [
    (r"--model_dir C:\Models\xxl", ["--model_dir", r"C:\Models\xxl"]),
    (r'--output_dir "D:\My Subs\out" --verbose True', ["--output_dir", r"D:\My Subs\out", "--verbose", "True"]),
    (r"--model_dir 'E:\a b\c'", ["--model_dir", r"E:\a b\c"]),
    (r"\\server\share\models", [r"\\server\share\models"]),
    ("--verbose True --standard_asia", ["--verbose", "True", "--standard_asia"]),
    ("--initial_prompt #1 take", ["--initial_prompt", "#1", "take"]),
    ("   ", []),
    ("", []),
])
def test_split_keeps_backslashes(text, tokens):
    assert _split(text) == tokens


def test_an_unclosed_quote_still_fails():
    with pytest.raises(ValueError):
        _split('--output_dir "D:\\no end')


def test_the_model_is_still_removed_after_the_split():
    kept, removed = strip_model_args(_split(r"--model large-v3 --model_dir C:\Models\xxl"))
    assert kept == ["--model_dir", r"C:\Models\xxl"] and removed == ["--model large-v3"]
