"""Cloud AI providers: current default models.

Model IDs were read from each provider's model page on 2026-11-10.
"""
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
