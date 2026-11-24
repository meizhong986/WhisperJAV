"""The translation model rule (1.10.3, owner W624 A): at the start of a translation the chosen model is looked up in the
provider's own list of models for the key; when it is missing, one line names the models the key can use, and the
translation goes on. The GUI's Test Connection fills the model menu from the same list.

No network: urllib's urlopen is replaced. No key-shaped strings (the commit check stops those).
"""
import io
import json
import urllib.request
from pathlib import Path

import pytest

from whisperjav.translate import providers as p

KEY = "plain words standing in for a key"


@pytest.fixture(autouse=True)
def fresh_state():
    p._LISTS_READ.clear()
    p._LINES_SHOWN.clear()
    yield
    p._LISTS_READ.clear()
    p._LINES_SHOWN.clear()


def answer(payload):
    """A stand-in urlopen that answers `payload` and records the requests."""
    seen = []

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(request, timeout=None):
        seen.append(request.full_url)
        return Response(json.dumps(payload).encode("utf-8"))
    return urlopen, seen


def test_each_provider_is_asked_at_its_own_list():
    assert p.models_list_request("gemini", KEY)[0] == p.GEMINI_MODELS_URL + "?pageSize=1000"
    assert p.models_list_request("claude", KEY)[0] == p.ANTHROPIC_MODELS_URL + "?limit=1000"
    assert p.models_list_request("openrouter", KEY)[0] == p.OPENROUTER_MODELS_URL
    assert p.models_list_request("deepseek", KEY)[0] == "https://api.deepseek.com/models"
    assert p.models_list_request("groq", KEY)[0] == "https://api.groq.com/openai/v1/models"
    assert p.models_list_request("ollama", KEY) is None and p.models_list_request("custom", KEY) is None


def test_the_chat_models_are_read_from_both_answer_forms():
    gemini = {"models": [{"name": "models/gemini-3.6-flash", "supportedGenerationMethods": ["generateContent"]},
                         {"name": "models/text-embedding-9", "supportedGenerationMethods": ["embedContent"]}]}
    assert p.model_ids_from_list("gemini", gemini) == ["gemini-3.6-flash"]
    openai_form = {"data": [{"id": "gpt-6-luna"}, {"id": "whisper-1"}, {"id": "text-embedding-3-small"},
                            {"id": "gpt-6.1-sol"}]}
    assert p.model_ids_from_list("gpt", openai_form) == ["gpt-6-luna", "gpt-6.1-sol"]


def test_a_listed_model_or_an_unread_list_says_nothing():
    assert p.missing_model_line("deepseek", "deepseek-flash", ["deepseek-flash", "deepseek-v4-pro"]) is None
    assert p.missing_model_line("deepseek", "deepseek-flash", None) is None
    assert p.missing_model_line("deepseek", "deepseek-flash", []) is None


def test_a_missing_model_is_named_with_the_models_of_its_family():
    ids = ["claude-haiku-5-5", "claude-opus-5-5", "claude-sonnet-5-5"]
    line = p.missing_model_line("claude", "claude-3-5-haiku-20241022", ids, flag="--translate-model")
    assert line == ("Translation: Claude does not list the model 'claude-3-5-haiku-20241022' for this API key, so "
                    "the translation may fail. The models it lists include: claude-haiku-5-5, claude-opus-5-5, "
                    "claude-sonnet-5-5. Choose one with --translate-model, or in the model menu of the GUI.")


def test_a_name_the_list_leaves_out_is_not_called_missing():
    assert p.missing_model_line("gpt", "gpt-6-search-preview", ["gpt-6-luna"]) is None


def test_a_long_list_is_cut_and_counted():
    ids = [f"vendor/model-{n:02d}" for n in range(30)] + ["other/x"]
    line = p.missing_model_line("openrouter", "vendor/retired", ids)
    assert "vendor/model-11 (and 18 more)" in line and "other/x" not in line


def test_the_list_is_read_once_per_key_and_the_line_shown_once(monkeypatch):
    urlopen, seen = answer({"data": [{"id": "deepseek-flash"}]})
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    first = p.missing_model_line_once("deepseek", "deepseek-v4-flash", KEY)
    assert first.startswith("Translation: DeepSeek does not list the model 'deepseek-v4-flash'")
    assert p.missing_model_line_once("deepseek", "deepseek-v4-flash", KEY) is None
    assert len(seen) == 1
    assert p.fetch_model_ids("deepseek", KEY, cache=False) == ["deepseek-flash"] and len(seen) == 2


def test_no_network_or_no_key_says_nothing(monkeypatch):
    def refuse(request, timeout=None):
        raise OSError("no network")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert p.missing_model_line_once("gemini", "gemini-2.0-flash", KEY) is None
    assert p.fetch_model_ids("gemini", "") is None
    assert p.fetch_model_ids("ollama", KEY) is None


def test_both_translation_starts_apply_the_rule():
    root = Path(__file__).resolve().parents[1] / "whisperjav"
    cli = (root / "translate" / "cli.py").read_text(encoding="utf-8")
    service = (root / "translate" / "service.py").read_text(encoding="utf-8")
    assert "missing_model_line_once(provider_name, model, api_key, flag='--model')" in cli
    assert "missing_model_line_once(provider, resolved_model, resolved_api_key, flag='--translate-model')" in service
    api = (root / "webview_gui" / "api.py").read_text(encoding="utf-8")
    assert '"models": fetch_model_ids(provider, key, cache=False) or [],' in api
    assert '"provider_name": PROVIDER_NAMES.get(provider, provider),' in api
    js = (root / "webview_gui" / "assets" / "app.js").read_text(encoding="utf-8")
    assert js.count("const listedBy = result.provider_name || provider;") == 2


def test_every_provider_has_the_name_the_gui_shows():
    """1.10.3 (owner W772): the lines a user reads name the provider as the GUI's menu does."""
    assert set(p.PROVIDER_NAMES) == set(p.PROVIDER_CONFIGS)
    html = (Path(__file__).resolve().parents[1] / "whisperjav" / "webview_gui" / "assets" / "index.html"
            ).read_text(encoding="utf-8")
    for key, name in p.PROVIDER_NAMES.items():
        assert f'<option value="{key}">{name}</option>' in html, key
