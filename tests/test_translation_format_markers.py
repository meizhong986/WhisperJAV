"""Answer-format labels must not stay in translated subtitles unnoticed (#460).

A model that repeats "Translation>" inside a subtitle's text gives PySubtrans no
error, so its retry never ran. WhisperJAV makes these labels an error in
PySubtrans's own check; the retry then asks again. If the retry still holds
them, the text is kept and one console line names the subtitles (W279).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from types import SimpleNamespace

from PySubtrans.Options import Options
from PySubtrans.SubtitleError import TranslationValidationError
from PySubtrans.TranslationParser import TranslationParser
from PySubtrans.Translation import Translation

from whisperjav.translate.core import (
    apply_format_marker_check,
    format_marker_lines,
    translate_subtitle,
)

# The reporter's samples (#460), unchanged.
SAMPLES = {
    "normal": "#21\nTranslation>\nHello.",
    "text_before_marker": "#21\nHello.\nTranslation>\nHello.",
    "empty_after_marker": "#21\nHello.\nTranslation>",
    "extra_marker": "#21\nTranslation>\nHello.\nTranslation>",
    "extra_marker_and_text": "#21\nTranslation>\nHello.\nTranslation>\nHello.",
    "wrapped_cues": (
        "<Translation>\n"
        "#11\nHello.\n#12\nGood morning.\n"
        "#13\nThank you.\n#14\nGoodbye.\n"
        "</Translation>"
    ),
}


def _checked_parser():
    class Client:
        def GetParser(self, task_type="Translation"):
            return TranslationParser(task_type, Options({}))

    translator = SimpleNamespace(client=Client())
    assert apply_format_marker_check(translator)
    return translator.client.GetParser("Translation")


def _marker_errors(parser):
    return [e for e in parser.errors if isinstance(e, TranslationValidationError) and "answer-format" in str(e)]


def test_normal_answer_gives_no_marker_error():
    parser = _checked_parser()
    lines = parser.ProcessTranslation(Translation({"text": SAMPLES["normal"]}))
    assert [(line.number, line.text) for line in lines] == [(21, "Hello.")]
    assert parser.errors == []


def test_each_malformed_answer_is_an_error_naming_its_subtitle():
    expected = {
        "text_before_marker": 21,
        "empty_after_marker": 21,
        "extra_marker": 21,
        "extra_marker_and_text": 21,
        "wrapped_cues": 11,
    }
    for name, number in expected.items():
        parser = _checked_parser()
        parser.ProcessTranslation(Translation({"text": SAMPLES[name]}))
        errors = _marker_errors(parser)
        assert len(errors) == 1, name
        assert f"#{number}" in str(errors[0]), name
        assert [line.number for line in errors[0].lines] == [number], name


def test_ordinary_text_with_hash_or_angle_bracket_is_not_a_marker():
    lines = [
        SimpleNamespace(number=1, text="No.1 #fan"),
        SimpleNamespace(number=2, text="A > B"),
        SimpleNamespace(number=3, text="The Translation> label mid-line"),
        SimpleNamespace(number=4, text="#1 fan\nsecond row"),
        SimpleNamespace(number=5, text=None),
        SimpleNamespace(number=6, text="Ranking\n#１"),  # full-width digit
    ]
    assert format_marker_lines(lines) == []


def test_labels_are_found_with_windows_line_ends_and_another_task_type():
    lines = [
        SimpleNamespace(number=1, text="Hello.\r\n#12\r\nGood morning."),
        SimpleNamespace(number=2, text="Hello.\nSubtitle>\nHello."),
    ]
    assert format_marker_lines(lines) == [1]
    assert format_marker_lines(lines, "Subtitle") == [1, 2]


# --- end to end: the real PySubtrans against a stand-in OpenAI-compatible server ---

SOURCE_SRT = (
    "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nおはよう\n\n"
    "3\n00:00:05,000 --> 00:00:06,000\nありがとう\n"
)
MARKED = ("#1\nTranslation>\nHello.\nTranslation>\nHello.\n\n"
          "#2\nTranslation>\nGood morning.\n\n#3\nTranslation>\nThank you.")
CLEAN = "#1\nTranslation>\nHello.\n\n#2\nTranslation>\nGood morning.\n\n#3\nTranslation>\nThank you."


def _serve(answers):
    """Answer each chat request with the next scripted text; record the requests."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            text = answers[min(len(requests), len(answers)) - 1]
            reply = json.dumps({
                "id": "x", "object": "chat.completion", "model": "stand-in",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


def _translate(tmp_path, answers):
    source = tmp_path / "film.ja.srt"
    source.write_text(SOURCE_SRT, encoding="utf-8")
    output = tmp_path / "film.english.srt"
    server, requests = _serve(answers)
    try:
        result = translate_subtitle(
            input_path=str(source),
            output_path=output,
            provider_config={
                "pysubtrans_name": "Custom Server",
                "whisperjav_provider": "custom",
                "server_address": f"http://127.0.0.1:{server.server_address[1]}",
                "endpoint": "/v1/chat/completions",
            },
            model="stand-in",
            api_key="",
            target_lang="english",
            instruction_file=str(files("whisperjav.translate") / "defaults" / "standard.txt"),
        )
    finally:
        server.shutdown()
    return result, output, requests


def test_marked_answer_is_retried_and_the_clean_retry_is_saved(tmp_path, capsys):
    result, output, requests = _translate(tmp_path, [MARKED, CLEAN])

    assert result == output
    assert len(requests) == 2
    assert "answer-format labels" in requests[1]["messages"][-1]["content"]
    text = output.read_text(encoding="utf-8")
    assert "Translation>" not in text
    assert text.count("Hello.") == 1
    assert "still contain answer-format" not in capsys.readouterr().err


def test_retry_with_labels_keeps_the_text_and_names_the_subtitles(tmp_path, capsys):
    result, output, requests = _translate(tmp_path, [MARKED, MARKED])

    assert result == output
    assert len(requests) == 2
    text = output.read_text(encoding="utf-8")
    assert "Hello.\nTranslation>\nHello." in text
    err = capsys.readouterr().err
    warning = [line for line in err.splitlines() if "still contain answer-format" in line]
    assert len(warning) == 1
    assert "1 translated subtitles" in warning[0] and ": #1 " in warning[0]
    assert output.name in warning[0]
    # the subtitle named is the SRT entry that holds the label
    entries = {block.split("\n", 1)[0]: block for block in text.strip().split("\n\n")}
    assert "Translation>" in entries["1"]
    assert "Translation>" not in entries["2"] + entries["3"]


def test_console_names_the_saved_entry_when_an_earlier_line_is_untranslated(tmp_path, capsys):
    # #1 never comes back, so the saved SRT starts at the model's #2: its entry 1.
    dropped = "#2\nTranslation>\nGood morning.\nTranslation>\nGood morning.\n\n#3\nTranslation>\nThank you."
    result, output, requests = _translate(tmp_path, [dropped, dropped])

    text = output.read_text(encoding="utf-8")
    entries = {block.split("\n", 1)[0]: block for block in text.strip().split("\n\n")}
    labelled = [number for number, block in entries.items() if "Translation>" in block]
    assert labelled == ["1"]
    warning = [line for line in capsys.readouterr().err.splitlines() if "still contain answer-format" in line]
    assert len(warning) == 1
    assert ": #1 " in warning[0]
