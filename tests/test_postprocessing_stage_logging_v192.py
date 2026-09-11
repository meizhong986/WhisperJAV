"""Post-processing prints progress on the Japanese path (v1.9.2, issue #372).

Issue #372 reports a batch that stops before every file is finished, with nothing
on screen to say where it stopped. Post-processing was a silent stretch: the
pipeline printed "Post-processing subtitles" once and the sanitizer then said
nothing at all until the next file, because every marker inside it was at DEBUG,
and the markers that did exist sat on the English-only path. A Japanese run --
which is every JAV run -- printed none of them.

These tests run the real Japanese post-processor over a small SRT file and check
the lines appear at INFO, in order.

The whisperjav logger sets propagate = False (whisperjav/utils/logger.py:80) so
that dependencies calling logging.basicConfig() cannot duplicate its output.
pytest's caplog fixture reads the root logger, so it sees nothing from it. These
tests attach their own handler to the whisperjav logger instead.
"""

import logging

import pytest

from whisperjav.modules.srt_postprocessing import SRTPostProcessor

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:03,000
こんにちは、元気ですか

2
00:00:04,000 --> 00:00:06,500
今日はいい天気ですね

3
00:00:07,000 --> 00:00:09,000
また明日会いましょう
"""


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def captured_info():
    """Records at INFO or above from the whisperjav logger."""
    logger = logging.getLogger("whisperjav")
    handler = _Collector()
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


@pytest.fixture
def japanese_srt(tmp_path):
    path = tmp_path / "scene_stitched.srt"
    path.write_text(SAMPLE_SRT, encoding="utf-8")
    return path


def _messages(records):
    return [r.getMessage() for r in records if r.levelno >= logging.INFO]


def _index_of(messages, fragment):
    for i, message in enumerate(messages):
        if fragment in message:
            return i
    raise AssertionError(f"no line containing {fragment!r}. Lines were:\n  " + "\n  ".join(messages))


class TestJapanesePostProcessingLogging:
    def test_each_file_step_is_announced_at_info(self, japanese_srt, captured_info):
        processor = SRTPostProcessor(language="ja")
        processor.process(japanese_srt)

        messages = _messages(captured_info)

        reading = _index_of(messages, "reading")
        saving_original = _index_of(messages, "saving a copy of the original")
        writing = _index_of(messages, f"writing {japanese_srt.stem}")
        cleaned = _index_of(messages, "cleaned")

        assert reading < saving_original < writing < cleaned, (
            "the progress lines must follow the order of the work:\n  "
            + "\n  ".join(messages)
        )

    def test_the_completion_line_gives_the_counts(self, japanese_srt, captured_info):
        """A reader needs to see that something came out, not only that it ran."""
        processor = SRTPostProcessor(language="ja")
        processor.process(japanese_srt)

        messages = _messages(captured_info)
        cleaned = messages[_index_of(messages, "cleaned")]

        assert "cleaned 3 subtitles down to" in cleaned, cleaned

    def test_the_non_linguistic_step_is_announced_with_its_count(self, japanese_srt, captured_info):
        """This step rewrites the file in place and used to say nothing at all.

        Its own message only printed when it dropped something, so on a file
        where it dropped nothing the run looked stalled.
        """
        processor = SRTPostProcessor(language="ja")
        processor.process(japanese_srt)

        messages = _messages(captured_info)
        looking = _index_of(messages, "looking for non-linguistic lines")
        dropped = _index_of(messages, "dropped")

        assert looking < dropped, messages
        assert "non-linguistic lines" in messages[dropped], messages[dropped]

    def test_every_progress_line_says_what_it_belongs_to(self, japanese_srt, captured_info):
        """The lines share one prefix so they can be told apart in a long log."""
        processor = SRTPostProcessor(language="ja")
        processor.process(japanese_srt)

        progress_lines = [m for m in _messages(captured_info) if m.startswith("Post-processing:")]

        assert len(progress_lines) >= 5, (
            "expected a line for reading, saving the original, writing the result, and "
            f"the two non-linguistic lines. Got: {progress_lines}"
        )

    def test_a_korean_file_is_also_announced(self, tmp_path, captured_info):
        """Korean and Chinese take the same path; only the extra Japanese-only
        non-linguistic step is skipped."""
        path = tmp_path / "korean_stitched.srt"
        path.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\n안녕하세요 잘 지내세요\n", encoding="utf-8"
        )

        SRTPostProcessor(language="ko").process(path)

        messages = _messages(captured_info)
        _index_of(messages, "reading")
        _index_of(messages, "cleaned")
        assert not any("non-linguistic" in m for m in messages), (
            "the non-linguistic filter is Japanese-only and must not be announced "
            f"on a Korean file: {messages}"
        )
