#!/usr/bin/env python3
"""
Tests for the start-up check that the local translation server can answer a CHAT
request -- in the shape translation actually sends.

Everything the readiness check did before used /v1/completions. A build can serve
that and fail /v1/chat/completions every time: they are different routes with
different response models. That is exactly what happened on Colab -- the server
reported healthy, and about 90 seconds later translation failed with four server
errors and nothing translated (owner's report, 2026-09-16).

Owner, 2026-09-17, agreeing to add this: it is the one measurement that separates
"this build is broken" from everything else, and it turns a long mystery into an
immediate, explainable message.

It WARNS; it does not stop the run (owner, same day: "soften it to a warning for
this release"). The check has never met a real llama-cpp server, so a mistake in
it would stop runs that were going to work -- worse than the fault it reports.
The repair for the known fault is the shim applied when the server starts, which
does not depend on this check at all.

Streaming matters here (added 2026-09-17 after review). llama-cpp-python
validates a single chat reply against a response model and sends streamed chunks
without validating them, so a build can fail one and serve the other. The check
therefore asks in whichever shape the caller will use: the translate CLI streams,
the GUI and --translate path do not. Asking in the wrong shape would report a
fault that is not there, or miss one that is.

These run against a stub HTTP server, so no model is loaded and nothing is
downloaded.

Run with: pytest tests/test_local_chat_check.py -v
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from whisperjav.translate.local_backend import _verify_chat_completion, _wait_for_server

# The real failure, verbatim in shape: the server rejects its own reply because
# a required field on the assistant message was never set.
REFUSAL_ERROR = {
    "error": {
        "message": ("2 validation errors:\n"
                    "{'type': 'missing', 'loc': ('response', "
                    "'CreateChatCompletionResponse', 'choices', 0, 'message', "
                    "'refusal'), 'msg': 'Field required'}"),
        "type": "internal_server_error",
    }
}

GOOD_REPLY = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [{"index": 0,
                 "message": {"role": "assistant", "content": "OK"},
                 "finish_reason": "stop"}],
}

GOOD_COMPLETION = {
    "id": "cmpl-test",
    "object": "text_completion",
    "choices": [{"index": 0, "text": "1 2 3", "finish_reason": "stop"}],
    "usage": {"completion_tokens": 5},
}

STREAM_CHUNK = {
    "id": "chatcmpl-test",
    "object": "chat.completion.chunk",
    "choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": None}],
}


def _serve(chat_behaviour, stream_behaviour=None):
    """
    Start a one-off HTTP server standing in for llama-cpp-python's.

    ``chat_behaviour`` answers a non-streaming /v1/chat/completions and returns
    (status, body). ``stream_behaviour`` answers a streaming one and returns
    (status, list-of-SSE-lines); when it is None the streaming path reuses the
    non-streaming behaviour, which is what a healthy build does.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Phase 1 of the readiness check: is the HTTP server up?
            payload = json.dumps({"data": [{"id": "stub-model"}]}).encode("utf-8")
            self._send(200, payload, "application/json")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                request = json.loads(body or b"{}")
            except ValueError:
                request = {}

            if self.path.endswith("/completions") and not self.path.endswith("/chat/completions"):
                # Phases 2 and 3: plain completions, which always work here.
                self._send(200, json.dumps(GOOD_COMPLETION).encode("utf-8"),
                           "application/json")
                return

            if request.get("stream"):
                # A default-healthy stream, independent of the single-reply
                # behaviour: on the real server these are two different paths,
                # and only the single reply is validated against a response
                # model. A test for a build broken in its stream passes one in.
                behaviour = stream_behaviour or (
                    lambda: (200, [f"data: {json.dumps(STREAM_CHUNK)}", "data: [DONE]"]))
                status, lines = behaviour()
                if status != 200:
                    self._send(status, json.dumps(lines).encode("utf-8"),
                               "application/json")
                    return
                payload = ("\n\n".join(lines) + "\n\n").encode("utf-8")
                self._send(200, payload, "text/event-stream")
                return

            status, reply = chat_behaviour()
            self._send(status, json.dumps(reply).encode("utf-8"), "application/json")

        def _send(self, status, payload, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("localhost", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture
def server_factory():
    servers = []

    def start(chat_behaviour, stream_behaviour=None):
        server = _serve(chat_behaviour, stream_behaviour)
        servers.append(server)
        return server.server_address[1]

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


class TestAServerThatWorks:
    def test_a_good_chat_reply_passes(self, server_factory):
        port = server_factory(lambda: (200, GOOD_REPLY))
        ok, error = _verify_chat_completion(port, timeout=10)
        assert ok is True
        assert error is None

    def test_a_good_streamed_reply_passes(self, server_factory):
        port = server_factory(lambda: (200, GOOD_REPLY))
        ok, error = _verify_chat_completion(port, timeout=10, stream=True)
        assert ok is True
        assert error is None


class TestAServerThatDoesNot:
    def test_the_refusal_defect_is_caught_and_explained(self, server_factory):
        """The exact fault from the owner's Colab run."""
        port = server_factory(lambda: (500, REFUSAL_ERROR))
        ok, error = _verify_chat_completion(port, timeout=10)
        assert ok is False
        assert "refused a chat request" in error
        assert "500" in error
        # It must say what it means for the user, not just repeat the HTTP code.
        assert "nothing" in error and "translated" in error
        # And it must recognise this particular fault.
        assert "rejects its own chat replies" in error

    def test_an_ordinary_server_error_is_reported_without_the_special_hint(
            self, server_factory):
        port = server_factory(lambda: (500, {"error": {"message": "model not loaded"}}))
        ok, error = _verify_chat_completion(port, timeout=10)
        assert ok is False
        assert "refused a chat request" in error
        assert "rejects its own chat replies" not in error

    def test_a_reply_with_no_choices_is_caught(self, server_factory):
        port = server_factory(lambda: (200, {"id": "x", "choices": []}))
        ok, error = _verify_chat_completion(port, timeout=10)
        assert ok is False
        assert "no content" in error

    def test_an_empty_body_is_caught_and_named(self, server_factory):
        # A 2xx with nothing in it used to come back as a JSON decode error.
        port = server_factory(lambda: (204, None))
        ok, error = _verify_chat_completion(port, timeout=10)
        assert ok is False
        assert "empty body" in error

    def test_a_stream_that_sends_nothing_is_caught(self, server_factory):
        port = server_factory(lambda: (200, GOOD_REPLY),
                              stream_behaviour=lambda: (200, ["data: [DONE]"]))
        ok, error = _verify_chat_completion(port, timeout=10, stream=True)
        assert ok is False
        assert "without sending any reply" in error

    def test_a_server_that_is_not_there_is_reported(self):
        # Nothing is listening on this port.
        ok, error = _verify_chat_completion(59999, timeout=2)
        assert ok is False
        assert "could not answer a chat request" in error


class TestTheShapeAskedForIsTheShapeThatWillBeUsed:
    """
    The two chat paths can differ, so checking the wrong one is worse than
    useless. These are the two builds that made the distinction necessary.
    """

    def test_a_build_broken_only_when_not_streaming_does_not_stop_a_streaming_run(
            self, server_factory):
        port = server_factory(lambda: (500, REFUSAL_ERROR))  # single reply fails
        assert _verify_chat_completion(port, timeout=10)[0] is False
        assert _verify_chat_completion(port, timeout=10, stream=True)[0] is True

    def test_a_build_broken_only_when_streaming_is_caught_by_a_streaming_run(
            self, server_factory):
        port = server_factory(lambda: (200, GOOD_REPLY),
                              stream_behaviour=lambda: (500, REFUSAL_ERROR))
        assert _verify_chat_completion(port, timeout=10)[0] is True
        assert _verify_chat_completion(port, timeout=10, stream=True)[0] is False


class TestItIsWiredIntoTheReadinessCheck:
    """
    Run the readiness check itself, not a search of its source: a stub answers
    /v1/models and /v1/completions, so only the chat phase is in question.

    Owner, 2026-09-17: *"soften it to a warning for this release."* The check has
    never met a real llama-cpp server, so a mistake in it would stop runs that
    were going to work -- worse than the fault it reports. It therefore WARNS and
    the server still comes up. The repair for the known fault is the shim,
    applied when the server starts, which does not depend on this check.
    """

    def test_a_server_whose_chat_works_is_reported_ready(self, server_factory):
        port = server_factory(lambda: (200, GOOD_REPLY))
        ready, error, diagnostics = _wait_for_server(port, max_wait=20)
        assert ready is True
        assert error is None
        assert diagnostics is not None

    def test_a_server_whose_chat_fails_is_still_reported_ready(self, server_factory):
        # The Colab case. It is named in the log, and the run goes on.
        port = server_factory(lambda: (500, REFUSAL_ERROR))
        ready, error, _ = _wait_for_server(port, max_wait=20)
        assert ready is True
        assert error is None

    def test_the_reason_is_put_in_the_log_where_the_user_will_find_it(
            self, server_factory, caplog):
        import logging

        port = server_factory(lambda: (500, REFUSAL_ERROR))
        with caplog.at_level(logging.WARNING, logger="whisperjav"):
            _wait_for_server(port, max_wait=20)

        warnings = " ".join(r.getMessage() for r in caplog.records
                            if r.levelno >= logging.WARNING)
        # The diagnosis must survive the softening -- it is the whole value.
        assert "rejects its own chat replies" in warnings
        assert "Continuing anyway" in warnings

    def test_a_healthy_server_produces_no_such_warning(self, server_factory, caplog):
        import logging

        port = server_factory(lambda: (200, GOOD_REPLY))
        with caplog.at_level(logging.WARNING, logger="whisperjav"):
            _wait_for_server(port, max_wait=20)

        warnings = " ".join(r.getMessage() for r in caplog.records
                            if r.levelno >= logging.WARNING)
        assert "start-up chat check" not in warnings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
