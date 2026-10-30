"""
When translation fails against a server, WhisperJAV says where that server's own
log is. Which server it is depends on the provider the user chose.

These tests exist because the first version of this advice keyed on
``pysubtrans_name``, which is the name of PySubtrans's *client class*: ollama,
glm, groq and custom are all 'Custom Server' there, so the ollama branch was
unreachable and every one of them got the custom-server line. The provider the
user chose is now carried in 'whisperjav_provider'.

v1.10.0: the 'local' provider (the llama-cpp server WhisperJAV started itself) is
removed, and with it the tests of its log advice.
"""

import pytest

from whisperjav.translate.core import server_log_hint
from whisperjav.translate.providers import PROVIDER_CONFIGS


def advice(config, verbose=False):
    return " ".join(server_log_hint(config, verbose=verbose))


class TestEveryProviderIsIdentifiable:
    def test_each_shipped_provider_carries_its_own_name(self):
        for key, config in PROVIDER_CONFIGS.items():
            assert config["whisperjav_provider"] == key

    def test_the_pysubtrans_class_name_cannot_tell_them_apart(self):
        # The reason 'whisperjav_provider' had to be added: these four providers
        # are indistinguishable by the field the advice used to read.
        shared = {PROVIDER_CONFIGS[k]["pysubtrans_name"]
                  for k in ("ollama", "glm", "groq", "custom")}
        assert shared == {"Custom Server"}

    def test_the_removed_local_provider_is_not_offered(self):
        assert "local" not in PROVIDER_CONFIGS


class TestTheAdviceMatchesTheServerInUse:
    def test_ollama_is_pointed_at_ollamas_log(self):
        text = advice(PROVIDER_CONFIGS["ollama"])
        assert "ollama logs" in text

    def test_ollama_verbose_keeps_the_per_platform_detail(self):
        text = advice(PROVIDER_CONFIGS["ollama"], verbose=True)
        assert "journalctl" in text

    def test_a_server_the_user_named_is_theirs(self):
        text = advice(PROVIDER_CONFIGS["custom"])
        assert "the address you gave" in text

    def test_a_cloud_service_has_no_local_log(self):
        for key in ("deepseek", "openrouter", "gemini", "claude", "gpt", "glm", "groq"):
            text = advice(PROVIDER_CONFIGS[key])
            assert "cloud service" in text, key
            assert "ollama" not in text.lower(), key

    def test_nobody_but_ollama_is_told_to_run_ollama_logs(self):
        for key in PROVIDER_CONFIGS:
            if key == "ollama":
                continue
            assert "ollama logs" not in advice(PROVIDER_CONFIGS[key]), key


class TestARedirectedCloudProviderIsNotCalledCloud:
    def test_a_cloud_provider_pointed_at_this_machine_is_the_users_server(self):
        # --translate-provider deepseek --translate-endpoint http://localhost:8080/v1
        # is not talking to DeepSeek any more.
        config = dict(PROVIDER_CONFIGS["deepseek"])
        config.pop("api_base", None)
        config["pysubtrans_name"] = "Custom Server"
        config["server_address"] = "http://localhost:8080"
        assert "the address you gave" in advice(config)

    def test_the_real_cloud_endpoint_is_still_cloud(self):
        assert "cloud service" in advice(PROVIDER_CONFIGS["deepseek"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
