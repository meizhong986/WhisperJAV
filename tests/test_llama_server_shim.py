#!/usr/bin/env python3
"""
Tests for the llama-cpp-python server shim (whisperjav/translate/llama_server_shim.py).

Some prebuilt CUDA builds of llama-cpp-python declare `refusal` as a required key
on a chat completion's assistant message while never setting it, so the server
fails to validate its own reply and every non-streaming chat completion comes back
as HTTP 500. The shim fills the key in, but only on a build that requires it.

No such build can be installed here, so the affected build is simulated by making
the installed type require the key. What these tests establish is that the repair
fires exactly when the key is required, fills it in, and leaves the reply and the
streaming path alone -- not that a real affected build now works.

Run with: pytest tests/test_llama_server_shim.py -v
"""

import subprocess
import sys
from pathlib import Path

import pytest

llama_cpp = pytest.importorskip("llama_cpp", reason="llama-cpp-python is not installed")
from llama_cpp import llama_types  # noqa: E402

from whisperjav.translate.llama_server_shim import install_refusal_fix  # noqa: E402

SHIM_PATH = (Path(__file__).resolve().parents[1]
             / "whisperjav" / "translate" / "llama_server_shim.py")

REPLY = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [{"index": 0,
                 "message": {"role": "assistant", "content": "hello"},
                 "finish_reason": "stop"}],
}


def _fake_create_chat_completion(self, *args, **kwargs):
    if kwargs.get("stream"):
        return iter([{"choices": [{"delta": {"content": "hi"}}]}])
    import copy
    return copy.deepcopy(REPLY)


@pytest.fixture
def llama_cpp_state():
    """Restore whatever the shim changes, whichever way the test went."""
    original_method = llama_cpp.Llama.create_chat_completion
    original_keys = llama_types.ChatCompletionResponseMessage.__required_keys__
    try:
        yield
    finally:
        llama_cpp.Llama.create_chat_completion = original_method
        llama_types.ChatCompletionResponseMessage.__required_keys__ = original_keys


def _require_refusal():
    keys = set(llama_types.ChatCompletionResponseMessage.__required_keys__)
    llama_types.ChatCompletionResponseMessage.__required_keys__ = frozenset(keys | {"refusal"})


def _do_not_require_refusal():
    keys = set(llama_types.ChatCompletionResponseMessage.__required_keys__)
    llama_types.ChatCompletionResponseMessage.__required_keys__ = frozenset(keys - {"refusal"})


class TestRepairGate:
    """The repair must fire on an affected build and only on an affected build."""

    def test_declines_when_the_key_is_not_required(self, llama_cpp_state):
        _do_not_require_refusal()
        before = llama_cpp.Llama.create_chat_completion
        assert install_refusal_fix() is False
        assert llama_cpp.Llama.create_chat_completion is before

    def test_fires_when_the_key_is_required(self, llama_cpp_state):
        _require_refusal()
        before = llama_cpp.Llama.create_chat_completion
        assert install_refusal_fix() is True
        assert llama_cpp.Llama.create_chat_completion is not before


class TestRepairedReply:
    """What the server hands to FastAPI must now satisfy the response model."""

    def test_missing_key_is_filled_in(self, llama_cpp_state):
        llama_cpp.Llama.create_chat_completion = _fake_create_chat_completion
        _require_refusal()
        assert install_refusal_fix() is True

        dummy = object()
        result = llama_cpp.Llama.create_chat_completion(dummy, messages=[])
        message = result["choices"][0]["message"]
        assert message["refusal"] is None

    def test_reply_text_is_not_altered(self, llama_cpp_state):
        llama_cpp.Llama.create_chat_completion = _fake_create_chat_completion
        _require_refusal()
        install_refusal_fix()

        dummy = object()
        result = llama_cpp.Llama.create_chat_completion(dummy, messages=[])
        message = result["choices"][0]["message"]
        assert message["content"] == "hello"
        assert message["role"] == "assistant"
        assert result["id"] == "chatcmpl-test"

    def test_an_existing_value_is_kept(self, llama_cpp_state):
        def with_refusal(self, *args, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": None,
                                             "refusal": "I cannot help with that"}}]}

        llama_cpp.Llama.create_chat_completion = with_refusal
        _require_refusal()
        install_refusal_fix()

        dummy = object()
        result = llama_cpp.Llama.create_chat_completion(dummy, messages=[])
        assert result["choices"][0]["message"]["refusal"] == "I cannot help with that"

    def test_streaming_reply_passes_through(self, llama_cpp_state):
        """
        The server's streaming branch calls the same callable unbound, so the
        wrapper is on that path too; it must hand the iterator back untouched.
        """
        llama_cpp.Llama.create_chat_completion = _fake_create_chat_completion
        _require_refusal()
        install_refusal_fix()

        dummy = object()
        chunks = llama_cpp.Llama.create_chat_completion(dummy, messages=[], stream=True)
        assert next(chunks)["choices"][0]["delta"]["content"] == "hi"

    def test_applying_it_twice_is_harmless(self, llama_cpp_state):
        llama_cpp.Llama.create_chat_completion = _fake_create_chat_completion
        _require_refusal()
        install_refusal_fix()
        install_refusal_fix()

        dummy = object()
        message = llama_cpp.Llama.create_chat_completion(dummy, messages=[])["choices"][0]["message"]
        assert message["refusal"] is None
        assert message["content"] == "hello"


class TestShimProcess:
    """The shim replaces `-m llama_cpp.server`, so it must behave like it."""

    def test_the_shim_file_is_where_the_launcher_looks_for_it(self):
        assert SHIM_PATH.is_file()

    def test_running_it_as_a_script_does_not_import_whisperjav(self):
        """
        The server subprocess must not depend on the translate package (and its
        dependencies) being importable, so the shim is run by path, not with -m.
        """
        # Executed with a __name__ other than "__main__", so the module body runs
        # (its imports are what is being measured) but the server is not started.
        probe = (
            "import sys;"
            "src = open(r'%s', encoding='utf-8').read();"
            "exec(compile(src, 'shim', 'exec'), {'__name__': 'not_main'});"
            "print('whisperjav' in sys.modules)"
        ) % SHIM_PATH
        result = subprocess.run([sys.executable, "-c", probe],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("False"), result.stdout

    def test_it_accepts_the_servers_own_options(self):
        """
        `--help` is llama-cpp-python's own; it prints an emoji, which a non-UTF-8
        Windows console cannot encode (true of `-m llama_cpp.server` as well), so
        the encoding is forced for this check.
        """
        env = {**dict(__import__("os").environ), "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run([sys.executable, str(SHIM_PATH), "--help"],
                                capture_output=True, text=True, timeout=120, env=env)
        assert result.returncode == 0, result.stderr
        for option in ("--model", "--n_gpu_layers", "--n_ctx", "--host", "--port"):
            assert option in result.stdout, option
