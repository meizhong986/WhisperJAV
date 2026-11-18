"""Cloud AI providers: current default models.

Model IDs were read from each provider's model page on 2026-11-10.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

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


# Runs the real modelOptionsHtml from app.js against the real table and prints each provider's option values.
_OPTIONS_JS = r"""
const s = require("fs").readFileSync(process.argv[1], "utf8");
const table = eval("(" + s.match(/providerModels: (\{[\s\S]*?\r?\n    \}),/)[1] + ")");
const body = s.match(/modelOptionsHtml\(provider\) \{([\s\S]*?)\r?\n    \},/)[1];
const tm = {providerModels: table, modelOptionsHtml: new Function("provider", body)};
const out = {};
for (const key of Object.keys(table)) {
    out[key] = [...tm.modelOptionsHtml(key).matchAll(/value="([^"]*)"/g)].map(m => m[1]);
}
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_dropdown_lists_the_default_under_its_own_name_too():
    # A remembered pick of the default model must still be an option, so it is restored exactly.
    result = subprocess.run(["node", "-e", _OPTIONS_JS, str(APP_JS)], capture_output=True, text=True, check=True)
    lists = _gui_model_lists()
    for key, values in json.loads(result.stdout).items():
        assert values == [""] + lists[key], key
