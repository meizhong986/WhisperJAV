#!/usr/bin/env python3
"""
Tests for the start-up check that the local translation server can answer a CHAT
request -- the shape translation actually sends.

Everything the readiness check did before used /v1/completions. A build can serve
that and fail /v1/chat/completions every time: they are different routes with
different response models. That is exactly what happened on Colab -- the server
reported healthy, and about 90 seconds later translation failed with four server
errors and nothing translated (owner's report, 2026-09-16).

Owner, 2026-09-17, agreeing to add this: it is the one measurement that separates
"this build is broken" from everything else, and it turns a long mystery into an
immediate, explainable stop.

These run against a stub HTTP server, so no model is loaded and nothing is
downloaded.

Run with: pytest tests/test_local_chat_check.py -v
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from whisperjav.translate.local_backend import _verify_chat_completion

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


def _serve(behaviour):
    """Start a one-off HTTP server that answers /v1/chat/completions."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            status, body = behaviour()
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
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

    def start(behaviour):
        server = _serve(behaviour)
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

    def test_a_server_that_is_not_there_is_reported(self):
        # Nothing is listening on this port.
        ok, error = _verify_chat_completion(59999, timeout=2)
        assert ok is False
        assert "could not answer a chat request" in error


class TestItIsWiredIntoTheReadinessCheck:
    def test_wait_for_server_runs_it_before_reporting_ready(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "whisperjav" / "translate"
                  / "local_backend.py").read_text(encoding="utf-8")
        wait = source[source.index("def _wait_for_server("):]
        assert "_verify_chat_completion(port" in wait
        # And a failure there must stop the server being reported as ready.
        assert "if not chat_ok:" in wait
        assert wait.index("_verify_chat_completion(port") < wait.index(
            "Server ready and speed measured")
