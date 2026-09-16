"""
Start llama-cpp-python's OpenAI-compatible server, with one compatibility repair.

Why this module exists
----------------------
Local translation talks to a llama-cpp-python server that WhisperJAV starts on
this machine. Some prebuilt CUDA builds of llama-cpp-python -- the ones the Colab
and Windows installers fall back to when no pinned wheel matches the running
Python -- declare ``refusal`` as a REQUIRED field on the assistant message of a
chat completion, while their own code never puts it there. FastAPI then rejects
the server's own reply and every non-streaming chat completion comes back as
HTTP 500::

    'msg': 'Field required',
    'loc': ('response', 'CreateChatCompletionResponse', 'choices', 0, 'message', 'refusal')

What the user sees is four retries and then "Failed to communicate with server
after 3 retries", with no translation produced.

Filling that field in before the reply is validated costs nothing. The repair is
applied only when the installed build actually requires the field, so a build
without the defect is left exactly as it is.

This module is started as ``python -m whisperjav.translate.llama_server_shim``
with the same arguments llama-cpp-python's own server takes.
"""

import sys


def install_refusal_fix() -> bool:
    """
    Add the missing ``refusal`` key to chat completions, if this build needs it.

    Returns True when the repair was applied, False when the installed
    llama-cpp-python does not have the defect (or is not importable).
    """
    try:
        import llama_cpp
        from llama_cpp import llama_types
    except Exception:
        return False

    message_type = getattr(llama_types, "ChatCompletionResponseMessage", None)
    required_keys = getattr(message_type, "__required_keys__", frozenset())
    if "refusal" not in required_keys:
        return False

    original = llama_cpp.Llama.create_chat_completion

    def create_chat_completion(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # A streaming request returns an iterator of chunks, which the server
        # sends without validating; only the single-response form needs this.
        if isinstance(result, dict):
            for choice in result.get("choices", []):
                message = choice.get("message")
                if isinstance(message, dict):
                    message.setdefault("refusal", None)
        return result

    llama_cpp.Llama.create_chat_completion = create_chat_completion
    return True


def main() -> None:
    if install_refusal_fix():
        print("[LOCAL-LLM] Applied the missing-'refusal' repair to this "
              "llama-cpp-python build", file=sys.stderr)
    from llama_cpp.server.__main__ import main as server_main
    server_main()


if __name__ == "__main__":
    main()
