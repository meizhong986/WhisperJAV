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
        # 1.10.2 (owner W622 2, W624 A): Claude Haiku 5.5. Anthropic's model list for a key (2026-11-12) no longer
        # held claude-3-5-haiku-20241022, the default until 1.10.1; claude-haiku-5-5 translated a test file, exit 0.
        # Named here, not left to PySubtrans, whose own defaults are fixed names too (W623 A).
        'model': 'claude-haiku-5-5',
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

# 1.10.3 (owner W772, texts sheet D2): each provider's name as the GUI's provider menus show it (index.html), for the
# lines a user reads; the keys are the names typed after --translate-provider.
PROVIDER_NAMES = {
    'deepseek': 'DeepSeek', 'openrouter': 'OpenRouter', 'gemini': 'Gemini', 'claude': 'Claude', 'gpt': 'GPT',
    'glm': 'GLM (Zhipu AI)', 'groq': 'Groq', 'ollama': 'Ollama', 'custom': 'Custom (OpenAI-compatible)',
}


# The GUI's Test Connection checks a key at the provider's own models list (#465). Until 1.10.1 every provider
# without 'api_base' was checked at OpenAI's address, so a working Gemini, Claude, GLM or Groq key showed
# "Invalid API key". Each address below answered a made-up key with its invalid-key status on 2026-11-06:
# Gemini 400 API_KEY_INVALID; Anthropic, Zhipu (GLM), Groq, DeepSeek and OpenAI 401. OpenRouter's models list is
# public (200 for any key), so its key is checked at /key (401 for a made-up key). Groq's firewall answers Python's
# default User-Agent with 403 "error code: 1010", so every check names WhisperJAV as its User-Agent.
OPENAI_MODELS_URL = 'https://api.openai.com/v1/models'
GEMINI_MODELS_URL = 'https://generativelanguage.googleapis.com/v1beta/models'
ANTHROPIC_MODELS_URL = 'https://api.anthropic.com/v1/models'
OPENROUTER_KEY_URL = 'https://openrouter.ai/api/v1/key'
ANTHROPIC_VERSION = '2023-06-01'


def _user_agent() -> str:
    try:
        from whisperjav.__version__ import __version__
        return f'WhisperJAV/{__version__}'
    except Exception:
        return 'WhisperJAV'


def key_check_request(provider: str, key: str):
    """(url, headers) that check `key` at `provider`'s own service."""
    agent = {'User-Agent': _user_agent()}
    if provider == 'gemini':
        return GEMINI_MODELS_URL, {'x-goog-api-key': key, **agent}
    if provider == 'claude':
        return ANTHROPIC_MODELS_URL, {'x-api-key': key, 'anthropic-version': ANTHROPIC_VERSION, **agent}
    if provider == 'openrouter':
        return OPENROUTER_KEY_URL, {'Authorization': f'Bearer {key}', **agent}
    config = PROVIDER_CONFIGS.get(provider) or {}
    base = config.get('api_base')
    if not base and config.get('server_address'):
        # OpenAI-compatible servers (glm, groq): the chat endpoint's folder holds the models list.
        endpoint = config.get('endpoint', '')
        base = config['server_address'].rstrip('/') + endpoint.rsplit('/chat/completions', 1)[0]
    if not base:
        return OPENAI_MODELS_URL, {'Authorization': f'Bearer {key}', **agent}
    return f"{base.rstrip('/')}/models", {'Authorization': f'Bearer {key}', **agent}


def is_invalid_key_response(provider: str, status: int, body: str = '') -> bool:
    """True when the provider's answer means the key itself is wrong (Gemini answers 400, not 401)."""
    if status == 401:
        return True
    return provider == 'gemini' and status == 400 and ('API_KEY_INVALID' in body or 'API key not valid' in body)


# 1.10.3 (owner W624 A; scope record of 1.10.2, section 2): the translation model rule. WhisperJAV's fixed defaults
# went stale three times (Gemini 2.0 Flash #291/#346, DeepSeek #325, Claude 3.5 Haiku). At the start of a translation
# the chosen model is looked up in the provider's own list of models for the key; when it is missing, one line says so
# and names the models the key can use. The GUI's Test Connection fills the model menu from the same list. The list is
# read with the key only (no translation is sent); when it cannot be read (no network, a provider without a list),
# nothing is said and the translation goes on as before. Ollama and custom servers are not looked up.
OPENROUTER_MODELS_URL = 'https://openrouter.ai/api/v1/models'
NO_MODEL_LIST = frozenset({'ollama', 'custom'})
# Entries of a models list that are not chat models (OpenAI's and Groq's lists carry them too).
_NOT_CHAT = ('embedding', 'embed', 'tts', 'whisper', 'dall-e', 'moderation', 'transcribe', 'realtime', 'image',
             'audio', 'search', 'guard', 'davinci', 'babbage')
MODELS_SHOWN = 12


def models_list_request(provider: str, key: str):
    """(url, headers) of `provider`'s list of models for `key`, or None for a provider without one."""
    if provider in NO_MODEL_LIST or provider not in PROVIDER_CONFIGS:
        return None
    if provider == 'openrouter':
        return OPENROUTER_MODELS_URL, {'Authorization': f'Bearer {key}', 'User-Agent': _user_agent()}
    url, headers = key_check_request(provider, key)
    if provider == 'gemini':
        url += '?pageSize=1000'
    elif provider == 'claude':
        url += '?limit=1000'
    return url, headers


def model_ids_from_list(provider: str, payload) -> list:
    """The chat models a models-list answer names, in the form --model takes."""
    ids = []
    if provider == 'gemini':
        for entry in (payload or {}).get('models') or []:
            methods = entry.get('supportedGenerationMethods') or ['generateContent']
            name = str(entry.get('name') or '')
            if name and 'generateContent' in methods:
                ids.append(name[len('models/'):] if name.startswith('models/') else name)
    else:
        for entry in (payload or {}).get('data') or []:
            if entry.get('id'):
                ids.append(str(entry['id']))
    return sorted({i for i in ids if not any(word in i.lower() for word in _NOT_CHAT)})


# One read per provider and key in a process (a run that translates several files asks once); the key itself is
# not kept, only a digest of it.
_LISTS_READ: dict = {}
_LINES_SHOWN: set = set()


def fetch_model_ids(provider: str, key: str, timeout: float = 10.0, cache: bool = True):
    """The chat models `provider` lists for `key`, or None when the list cannot be read. `cache=False` (the GUI's
    Test Connection) reads the list again."""
    request = models_list_request(provider, key) if key or provider == 'openrouter' else None
    if request is None:
        return None
    import hashlib
    import json
    import urllib.request
    seen = (provider, hashlib.sha256((key or '').encode('utf-8')).hexdigest())
    if cache and seen in _LISTS_READ:
        return _LISTS_READ[seen]
    url, headers = request
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
            ids = model_ids_from_list(provider, json.loads(response.read().decode('utf-8', 'replace')))
    except Exception:  # noqa: BLE001 - no list: nothing is said, the translation goes on
        ids = None
    _LISTS_READ[seen] = ids
    return ids


def missing_model_line_once(provider: str, model: str, key: str, flag: str = '--model'):
    """missing_model_line for the start of a translation, read from the live list, at most once per process."""
    line = missing_model_line(provider, model, fetch_model_ids(provider, key), flag)
    if line is None or (provider, model) in _LINES_SHOWN:
        return None
    _LINES_SHOWN.add((provider, model))
    return line


def _family(model: str) -> str:
    """The part of a model name that names its maker or family: before '/' (OpenRouter), else before the first '-'."""
    return model.split('/', 1)[0] if '/' in model else model.split('-', 1)[0]


def missing_model_line(provider: str, model: str, ids, flag: str = '--model'):
    """The line printed at the start of a translation when `provider` does not list `model` for the key, or None.

    `ids`: the provider's list (fetch_model_ids); None or empty means the list could not be read: nothing is said."""
    if not ids or not model or model in ids:
        return None
    if any(word in model.lower() for word in _NOT_CHAT):
        return None      # the list leaves such names out, so it cannot say this one is missing
    family = [i for i in ids if _family(i) == _family(model)]
    shown = family if family else ids
    more = len(shown) - MODELS_SHOWN
    names = ', '.join(shown[:MODELS_SHOWN]) + (f' (and {more} more)' if more > 0 else '')
    name = PROVIDER_NAMES.get(provider, provider)
    return (f"Translation: {name} does not list the model '{model}' for this API key, so the translation may "
            f"fail. The models it lists include: {names}. Choose one with {flag}, or in the model menu of the GUI.")


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
