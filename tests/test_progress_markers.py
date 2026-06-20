from whisperjav.utils.progress_markers import (
    parse_progress_marker,
    parse_transcribe_marker,
    progress_marker,
    transcribe_marker,
)


def test_plain_marker_roundtrip():
    line = transcribe_marker("My Video.mp4", 4, 20)
    assert line == "Transcribing [4/20]: My Video.mp4"
    parsed = parse_transcribe_marker(line)
    assert parsed == {"current": 4, "total": 20, "pass_number": None, "pass_total": None, "name": "My Video.mp4"}


def test_pass_marker_roundtrip():
    line = transcribe_marker("clip.mkv", 4, 20, pass_number=1, pass_total=2)
    assert line == "Transcribing [pass 1/2] [4/20]: clip.mkv"
    parsed = parse_transcribe_marker(line)
    assert parsed == {"current": 4, "total": 20, "pass_number": 1, "pass_total": 2, "name": "clip.mkv"}


def test_pass_total_one_omits_pass_label():
    line = transcribe_marker("a.mp4", 1, 3, pass_number=1, pass_total=1)
    assert line == "Transcribing [1/3]: a.mp4"
    assert parse_transcribe_marker(line)["pass_number"] is None


def test_name_with_brackets_and_colon():
    line = transcribe_marker("a [b] c: d.mp4", 2, 5)
    parsed = parse_transcribe_marker(line)
    assert parsed["name"] == "a [b] c: d.mp4"
    assert parsed["current"] == 2 and parsed["total"] == 5


def test_trailing_whitespace_stripped():
    parsed = parse_transcribe_marker("Transcribing [1/2]: name.mp4\r\n")
    assert parsed["name"] == "name.mp4"


def test_non_marker_returns_none():
    assert parse_transcribe_marker("Some other log line") is None
    assert parse_transcribe_marker("[INFO] starting") is None


def test_progress_marker_roundtrip():
    assert progress_marker(67) == "Transcribing-progress [67]"
    assert parse_progress_marker("Transcribing-progress [67]") == 67
    assert parse_progress_marker("Transcribing-progress [0]") == 0


def test_progress_marker_rejects_non_markers():
    assert parse_progress_marker("Transcribing [4/20]: x.mp4") is None
    assert parse_progress_marker("random log line") is None
    # the two marker types must not cross-parse
    assert parse_transcribe_marker("Transcribing-progress [50]") is None


def test_marker_after_prefix_is_rejected():
    assert parse_transcribe_marker("DEBUG: Transcribing [1/5]: foo.mp4") is None
    assert parse_transcribe_marker("2026-06-19 Transcribing [2/3]: bar.mp4") is None
