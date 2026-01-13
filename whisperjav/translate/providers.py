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
        'pysubtrans_name': 'OpenAI',  # GLM uses OpenAI-compatible API
        'model': 'glm-4-flash',
        'env_var': 'GLM_API_KEY',
        'api_base': 'https://open.bigmodel.cn/api/paas/v4'  # Zhipu AI endpoint
    },
    'groq': {
        'pysubtrans_name': 'OpenAI',  # Groq uses OpenAI-compatible API
        'model': 'llama-3.3-70b-versatile',
        'env_var': 'GROQ_API_KEY',
        'api_base': 'https://api.groq.com/openai/v1'
    },
    'local': {
        'pysubtrans_name': 'Local',  # Marker for local LLM bypass
        'model': 'llama-8b',          # Default: good quality, 6GB VRAM
        'env_var': None               # No API key needed
    }
}

SUPPORTED_SOURCES = {'japanese', 'korean', 'chinese'}
SUPPORTED_TARGETS = {'english', 'chinese', 'indonesian', 'spanish'}
