"""Cloud AI providers: current default models.

Model IDs were read from each provider's model page on 2026-11-10.
"""
import re
from pathlib import Path

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
