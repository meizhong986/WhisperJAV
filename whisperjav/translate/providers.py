"""
Provider configurations for translation services.
"""

PROVIDER_CONFIGS = {
    'deepseek': {
        'pysubtrans_name': 'DeepSeek',
        'model': 'deepseek-chat',
        'env_var': 'DEEPSEEK_API_KEY',
        'api_base': 'https://api.deepseek.com'
    },
    'openrouter': {
        'pysubtrans_name': 'OpenRouter',
        'model': 'deepseek/deepseek-chat',
        'env_var': 'OPENROUTER_API_KEY',
        'api_base': 'https://openrouter.ai/api/v1'
    },
    'gemini': {
        'pysubtrans_name': 'Gemini',
        'model': 'gemini-2.0-flash-exp',
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
    }
}

SUPPORTED_SOURCES = {'japanese', 'korean', 'chinese'}
SUPPORTED_TARGETS = {'english', 'chinese', 'indonesian', 'spanish'}
