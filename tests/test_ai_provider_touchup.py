"""Cloud AI providers: current default models, and Claude requests the current API accepts.

Model IDs were read from each provider's model page on 2026-11-10.
"""
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from whisperjav.translate.core import apply_claude_request_patch
from whisperjav.translate.providers import PROVIDER_CONFIGS

APP_JS = Path(__file__).resolve().parents[1] / "whisperjav" / "webview_gui" / "assets" / "app.js"

# Retired or shut down by the provider. None may be a default or a dropdown choice.
RETIRED = {
    "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229",
    "anthropic/claude-3.5-sonnet", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "openai/gpt-4o",
    "glm-4-flash", "glm-4", "glm-4-plus",
    "llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768",
}


def _gui_model_lists() -> dict:
    """Parse the `providerModels` table out of app.js.

    Returns:
        A dict of provider key to its list of model IDs.
    """
    text = APP_JS.read_text(encoding="utf-8")
    block = re.search(r"providerModels: \{(.*?)\n    \}", text, re.S).group(1)
    lists = {}
    for key, body in re.findall(r"^\s*(\w+): \[(.*?)\]", block, re.M | re.S):
        lists[key] = re.findall(r"'([^']+)'", body)
    return lists


class TestDefaults:
    def test_no_retired_default(self):
        for key, config in PROVIDER_CONFIGS.items():
            assert config["model"] not in RETIRED, key

    def test_claude_output_room_for_thinking(self):
        # Current Claude models think before answering, and the thinking counts against max_tokens.
        assert PROVIDER_CONFIGS["claude"]["max_tokens"] >= 16000


class TestGuiModelLists:
    def test_one_table(self):
        # The settings dialog reuses the main table, so the two cannot drift apart.
        assert APP_JS.read_text(encoding="utf-8").count("providerModels: {") == 1

    def test_no_retired_choice(self):
        for key, models in _gui_model_lists().items():
            assert not RETIRED & set(models), key

    def test_first_choice_is_the_default(self):
        for key, models in _gui_model_lists().items():
            if models:
                assert models[0] == PROVIDER_CONFIGS[key]["model"], key


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


class TestConnectionCheck:
    """The GUI's Test Connection button must ask each provider, not OpenAI."""

    @pytest.fixture
    def api(self):
        from whisperjav.webview_gui.api import WhisperJAVAPI
        return WhisperJAVAPI()

    @pytest.fixture
    def sent(self, monkeypatch):
        """Record the request urlopen gets, and answer it with the status in `sent["status"]`."""
        import io
        import urllib.error
        import urllib.request

        sent = {"status": 200}

        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def urlopen(req, timeout=None):
            sent["url"] = req.full_url
            sent["headers"] = {k.lower(): v for k, v in req.header_items()}
            if sent["status"] != 200:
                raise urllib.error.HTTPError(req.full_url, sent["status"], "err", None, io.BytesIO())
            return _Response()

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        return sent

    @pytest.mark.parametrize("provider, url, auth", [
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/models", ("x-goog-api-key", "k")),
        ("groq", "https://api.groq.com/openai/v1/models", ("authorization", "Bearer k")),
        ("glm", "https://open.bigmodel.cn/api/paas/v4/models", ("authorization", "Bearer k")),
        ("gpt", "https://api.openai.com/v1/models", ("authorization", "Bearer k")),
        ("deepseek", "https://api.deepseek.com/models", ("authorization", "Bearer k")),
    ])
    def test_each_provider_asks_its_own_server(self, api, sent, provider, url, auth):
        assert api.test_provider_connection(provider, "k")["success"]
        assert sent["url"] == url
        assert sent["headers"][auth[0]] == auth[1]

    def test_gemini_bad_key_is_invalid(self, api, sent):
        sent["status"] = 400
        assert api.test_provider_connection("gemini", "k")["error"] == "Invalid API key"

    def test_claude_looks_up_the_default_model(self, api, monkeypatch):
        import anthropic

        calls = []

        class _Client:
            def __init__(self, **kwargs):
                calls.append(kwargs["api_key"])
                self.models = SimpleNamespace(retrieve=calls.append)

        monkeypatch.setattr(anthropic, "Anthropic", _Client)
        assert api.test_provider_connection("claude", "k")["success"]
        assert calls == ["k", PROVIDER_CONFIGS["claude"]["model"]]

    def test_claude_bad_key_is_invalid(self, api, monkeypatch):
        import anthropic
        import httpx

        request = httpx.Request("GET", "https://api.anthropic.com/v1/models/x")
        error = anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None)

        def retrieve(model):
            raise error

        monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: SimpleNamespace(models=SimpleNamespace(retrieve=retrieve)))
        assert api.test_provider_connection("claude", "k")["error"] == "Invalid API key"
