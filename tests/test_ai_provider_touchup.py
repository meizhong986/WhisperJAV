"""Cloud AI providers: current default models, and Claude requests the current API accepts.

Model IDs were read from each provider's model page on 2026-11-10.
"""
from types import SimpleNamespace

import pytest

from whisperjav.translate.core import apply_claude_request_patch
from whisperjav.translate.providers import PROVIDER_CONFIGS

# Retired or shut down by the provider. None may be a default or a dropdown choice.
RETIRED = {
    "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229",
    "anthropic/claude-3.5-sonnet", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "openai/gpt-4o",
    "glm-4-flash", "glm-4", "glm-4-plus",
    "llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768",
}

class TestDefaults:
    def test_no_retired_default(self):
        for key, config in PROVIDER_CONFIGS.items():
            assert config["model"] not in RETIRED, key

    def test_claude_output_room_for_thinking(self):
        # Current Claude models think before answering, and the thinking counts against max_tokens.
        assert PROVIDER_CONFIGS["claude"]["max_tokens"] >= 16000


class _FakeSdk:
    """Stands in for `anthropic.Anthropic` and records each call's keyword arguments."""

    def __init__(self):
        self.calls = []
        self.messages = SimpleNamespace(create=self._record("create"), stream=self._record("stream"))

    def _record(self, name):
        def call(**kwargs):
            self.calls.append((name, kwargs))
            return name
        return call


class _FakeAnthropicClient:
    """Mirrors PySubtrans's `AnthropicClient`: `_get_client_response` calls `self.client.messages`."""

    def __init__(self):
        self.client = _FakeSdk()

    stream = False

    def _get_client_response(self, request, temperature):
        kwargs = {"model": request, "temperature": temperature, "max_tokens": 16000, "top_p": 0.9}
        if self.stream:
            return self.client.messages.stream(**kwargs)
        return self.client.messages.create(**kwargs)


class TestClaudeRequestPatch:
    def _patched(self, model):
        translator = SimpleNamespace(client=_FakeAnthropicClient())
        assert apply_claude_request_patch(translator, model)
        return translator.client

    @pytest.mark.parametrize("stream", [False, True])
    def test_sampling_fields_removed(self, stream):
        client = self._patched("claude-haiku-5-5")
        client.stream = stream
        sdk = client.client
        client._get_client_response("claude-haiku-5-5", 0.5)
        name, kwargs = sdk.calls[0]
        assert name == ("stream" if stream else "create")
        assert "temperature" not in kwargs and "top_p" not in kwargs
        assert kwargs["max_tokens"] == 16000

    def test_low_effort_on_current_models(self):
        client = self._patched("claude-sonnet-5-5")
        client._get_client_response("claude-sonnet-5-5", 0.5)
        assert client.client.calls[0][1]["output_config"] == {"effort": "low"}

    def test_no_effort_on_models_without_it(self):
        # Claude Haiku 4.5 rejects `effort`.
        client = self._patched("claude-haiku-4-5")
        client._get_client_response("claude-haiku-4-5", 0.5)
        assert "output_config" not in client.client.calls[0][1]

    def test_sdk_client_wrapped_once(self):
        client = self._patched("claude-haiku-5-5")
        client._get_client_response("claude-haiku-5-5", 0.5)
        wrapped = client.client
        client._get_client_response("claude-haiku-5-5", 0.5)
        assert client.client is wrapped
        assert len(wrapped.calls) == 2  # attributes other than `messages` reach the SDK client

    def test_missing_hook_is_reported(self):
        assert not apply_claude_request_patch(SimpleNamespace(client=None), "claude-haiku-5-5")
