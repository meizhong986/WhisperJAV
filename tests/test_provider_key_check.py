"""The GUI's Test Connection checks a key at the provider's own service (#465)."""
from whisperjav.translate.providers import (
    ANTHROPIC_MODELS_URL, GEMINI_MODELS_URL, OPENAI_MODELS_URL, OPENROUTER_KEY_URL, is_invalid_key_response,
    key_check_request,
)


def _without_agent(headers):
    assert headers.get('User-Agent', '').startswith('WhisperJAV')
    return {k: v for k, v in headers.items() if k != 'User-Agent'}


def test_gemini_and_claude_use_their_own_service_and_header():
    url, headers = key_check_request('gemini', 'K')
    assert url == GEMINI_MODELS_URL and _without_agent(headers) == {'x-goog-api-key': 'K'}
    url, headers = key_check_request('claude', 'K')
    assert url == ANTHROPIC_MODELS_URL
    assert headers['x-api-key'] == 'K' and 'anthropic-version' in headers and 'Authorization' not in headers


def test_glm_and_groq_use_their_server_not_openai():
    url, headers = key_check_request('glm', 'K')
    assert url == 'https://open.bigmodel.cn/api/paas/v4/models' and _without_agent(headers) == {'Authorization': 'Bearer K'}
    url, headers = key_check_request('groq', 'K')
    assert url == 'https://api.groq.com/openai/v1/models' and _without_agent(headers) == {'Authorization': 'Bearer K'}


def test_openrouter_checks_the_key_not_the_public_model_list():
    url, headers = key_check_request('openrouter', 'K')
    assert url == OPENROUTER_KEY_URL and _without_agent(headers) == {'Authorization': 'Bearer K'}


def test_providers_with_api_base_unchanged():
    assert key_check_request('deepseek', 'K')[0] == 'https://api.deepseek.com/models'
    url, headers = key_check_request('gpt', 'K')
    assert url == OPENAI_MODELS_URL and _without_agent(headers) == {'Authorization': 'Bearer K'}


def test_no_provider_with_a_key_is_checked_at_openai_except_openai():
    for provider in ('deepseek', 'openrouter', 'gemini', 'claude', 'glm', 'groq'):
        assert 'api.openai.com' not in key_check_request(provider, 'K')[0], provider


def test_invalid_key_answers():
    assert is_invalid_key_response('claude', 401)
    gemini_bad_key = '{"error": {"status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}]}}'
    assert is_invalid_key_response('gemini', 400, gemini_bad_key)
    assert not is_invalid_key_response('gemini', 400, '{"error": {"message": "model not found"}}')
    assert not is_invalid_key_response('groq', 400, 'API_KEY_INVALID')
    assert not is_invalid_key_response('gemini', 403, '')
