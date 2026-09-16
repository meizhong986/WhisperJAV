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
        'model': 'deepseek-v4-flash',
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
        'model': 'gemini-2.0-flash',
        'env_var': 'GEMINI_API_KEY'
    },
    'claude': {
        'pysubtrans_name': 'Claude',
        'model': 'claude-3-5-haiku-20241022',
        'env_var': 'ANTHROPIC_API_KEY'
    },
    'gpt': {
        'pysubtrans_name': 'OpenAI',
        'model': 'gpt-4o-mini',
        'env_var': 'OPENAI_API_KEY'
    },
    'glm': {
        'pysubtrans_name': 'Custom Server',  # Custom Server avoids Responses API misrouting (#178)
        'model': 'glm-4-flash',
        'env_var': 'GLM_API_KEY',
        'server_address': 'https://open.bigmodel.cn',
        'endpoint': '/api/paas/v4/chat/completions',
    },
    'groq': {
        'pysubtrans_name': 'Custom Server',  # Custom Server avoids Responses API misrouting (#178)
        'model': 'llama-3.3-70b-versatile',
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
    # DEPRECATED in v1.8.10. Will be removed in v1.9.0.
    # Users should migrate to 'ollama' provider.
    'local': {
        'pysubtrans_name': 'Local',  # Marker for local LLM bypass
        'model': 'llama-8b',          # Default: good quality, 6GB VRAM
        'env_var': None               # No API key needed
    },
    'custom': {
        'pysubtrans_name': 'Custom Server',  # Custom Server avoids Responses API misrouting (#178)
        'model': '',                   # User provides via --translate-model
        'env_var': None               # API key optional, provided via --translate-api-key
    }
}

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
