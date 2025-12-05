import sys
import types

if 'stable_whisper' not in sys.modules:
    stub = types.ModuleType('stable_whisper')

    class _DummyResult:
        pass

    def _unavailable(*args, **kwargs):
        raise RuntimeError("stable_whisper is not available in test environment")

    stub.WhisperResult = _DummyResult
    stub.load_model = _unavailable
    stub.load_faster_whisper = _unavailable
    sys.modules['stable_whisper'] = stub

from whisperjav.modules.repetition_cleaner import RepetitionCleaner
from whisperjav.config.sanitization_constants import RepetitionConstants


def _make_cleaner():
    return RepetitionCleaner(RepetitionConstants())


def test_whitespace_separated_repetitions_are_collapsed():
    cleaner = _make_cleaner()
    noisy_text = "あ\nあ\nあ\nあ\nあ\n"
    cleaned, modifications = cleaner.clean_repetitions(noisy_text)
    assert cleaned.strip() in {"ああ", "あ"}
    assert modifications, "Expected at least one repetition-cleaning modification"


def test_regular_text_is_unchanged():
    cleaner = _make_cleaner()
    sentence = "おはようございます。"
    cleaned, modifications = cleaner.clean_repetitions(sentence)
    assert cleaned == sentence
    assert modifications == []
