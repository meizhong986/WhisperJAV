"""
Provider configurations for translation services.
"""

PROVIDER_CONFIGS = {
    'deepseek': {
        'pysubtrans_name': 'DeepSeek',
        # v1.8.14 (#325): DeepSeek announced new model names 2026-05-06.
        # 'deepseek-chat' / 'deepseek-reasoner' deprecate 2026-07-24, replaced
        # by 'deepseek-v4-flash' (non-thinking, was deepseek-chat) and
        # 'deepseek-v4-pro' (thinking, was deepseek-reasoner).
        # Source: https://api-docs.deepseek.com/zh-cn/
        # Users wanting the thinking model can override via: --model deepseek-v4-pro
        # v1.9.5 (PR #457): DeepSeek's API page (read 2026-10-12) names the flash
        # model 'deepseek-flash'; 'deepseek-v4-flash' is still accepted but its
        # model is retired. Flash can think; core.py switches thinking off for it.
        'model': 'deepseek-flash',
        'env_var': 'DEEPSEEK_API_KEY',
        'api_base': 'https://api.deepseek.com'
    },
    'openrouter': {
        'pysubtrans_name': 'OpenRouter',
        # v1.8.14 (#325): OpenRouter lags the upstream DeepSeek catalog, so
        # deepseek-chat was kept as the routed default until v4-flash appeared.
        # v1.9.2: verified present in OpenRouter's public catalog
        # (GET https://openrouter.ai/api/v1/models lists deepseek/deepseek-v4-flash),
        # and upstream deprecated deepseek-chat on 2026-07-24, so the wait is over.
        # Matches the direct-API default. Override with --model if you prefer
        # another route, e.g. deepseek/deepseek-v4-pro for the thinking model.
        'model': 'deepseek/deepseek-v4-flash',
        'env_var': 'OPENROUTER_API_KEY',
        'api_base': 'https://openrouter.ai/api/v1'
    },
    'gemini': {
        'pysubtrans_name': 'Gemini',
        # WP-002 (#291, #346): gemini-2.0-flash is shut down; Google's deprecations page (read 2026-10-06)
        # names gemini-3.6-flash as its replacement.
        'model': 'gemini-3.6-flash',
        'env_var': 'GEMINI_API_KEY'
    },
    'claude': {
        'pysubtrans_name': 'Claude',
        # The Claude 3 models are retired. Haiku 5.5 is the cheapest current model.
        # Current models think before answering, and the thinking counts against max_tokens.
        # PySubtrans' own default of 4096 can cut a batch short, so this raises it.
        'model': 'claude-haiku-5-5',
        'max_tokens': 16000,
        'env_var': 'ANTHROPIC_API_KEY'
    },
    'gpt': {
        'pysubtrans_name': 'OpenAI',
        # OpenAI removes gpt-4o-mini on 2027-01-14 and names gpt-6-luna as its replacement.
        'model': 'gpt-6-luna',
        'env_var': 'OPENAI_API_KEY'
    },
    'glm': {
        'pysubtrans_name': 'Custom Server',  # Custom Server avoids Responses API misrouting (#178)
        # Zhipu's model list no longer has the bare glm-4 names. glm-4.7-flash is the free model.
        'model': 'glm-4.7-flash',
        'env_var': 'GLM_API_KEY',
        'server_address': 'https://open.bigmodel.cn',
        'endpoint': '/api/paas/v4/chat/completions',
    },
    'groq': {
        'pysubtrans_name': 'Custom Server',  # Custom Server avoids Responses API misrouting (#178)
        # Groq shut the Llama 3 models down on 2026-08-16 and names gpt-oss-120b as the replacement.
        'model': 'openai/gpt-oss-120b',
        'env_var': 'GROQ_API_KEY',
        'server_address': 'https://api.groq.com',
        'endpoint': '/openai/v1/chat/completions',
    },
    'ollama': {
        'pysubtrans_name': 'Custom Server',  # Uses OpenAI-compatible /v1/chat/completions
        'model': 'gemma3:12b',         # Default; OllamaManager.recommend_model() overrides at runtime
        'env_var': None,               # No API key needed
        'server_address': 'http://localhost:11434',
        'endpoint': '/v1/chat/completions',
        'supports_conversation': True,
        'supports_system_messages': True,
        'supports_streaming': True,
    },
    'custom': {
        'pysubtrans_name': 'Custom Server',  # Custom Server avoids Responses API misrouting (#178)
        'model': '',                   # User provides via --translate-model
        'env_var': None               # API key optional, provided via --translate-api-key
    }
}

# Every entry carries the name the user types after --translate-provider.
# 'pysubtrans_name' cannot serve that purpose: it is the name of PySubtrans's
# client class, and ollama, glm, groq and custom all use 'Custom Server'.
# Code that has to tell the providers apart -- which server failed, whose log to
# point at -- reads 'whisperjav_provider'.
for _provider_key, _provider_config in PROVIDER_CONFIGS.items():
    _provider_config.setdefault('whisperjav_provider', _provider_key)


# v1.10.0 (REQ2, owner W22 (1), W42 (2)): the 'local' provider -- the llama-cpp-python
# server WhisperJAV started itself -- is removed; Ollama is the only local translator.
# A saved 'local' choice is read as 'ollama' with one console line (settings.py);
# 'local' typed on a command line stops with an error. Texts approved by the owner (W43).
REMOVED_LOCAL_PROVIDER = 'local'
LOCAL_MODEL_NAMES = {'llama-8b', 'gemma-9b', 'llama-3b', 'auto'}  # the removed provider's models
SAVED_LOCAL_NOTICE = ("Local translation with llama-cpp was removed in 1.10; "
                      "your saved choice now uses Ollama.")


def removed_local_provider_error(flag: str = '--provider') -> str:
    """The error for 'local' typed after a provider flag (owner's text M5, W43)."""
    return (f"Error: {flag} local (llama-cpp) was removed in WhisperJAV 1.10. "
            f"Use {flag} ollama for local translation.")


SUPPORTED_SOURCES = {'japanese', 'korean', 'chinese', 'english'}
# The source of truth for translation targets. main.py's --translate-target choices, the two
# GUI dropdowns and the output-suffix stripping in service.py all derive from or are pinned to
# this set by tests/test_translate_targets.py -- add a language here and add it there too.
#
# v1.9.3: italian (#351), thai (#268) and korean added. There is no allow-list to satisfy on
# the library side: PySubtrans substitutes target_language into the prompt as a free string
# (Options.py:301-304), so a target costs nothing but the choice itself. Quality per language
# is the model's, not WhisperJAV's.
SUPPORTED_TARGETS = {'english', 'chinese', 'indonesian', 'portuguese', 'spanish', 'french',
                     'italian', 'thai', 'korean'}
