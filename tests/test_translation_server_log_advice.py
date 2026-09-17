"""
When translation fails against a server, WhisperJAV says where that server's own
log is. Which server it is depends on the provider the user chose.

These tests exist because the first version of this advice keyed on
``pysubtrans_name``, which is the name of PySubtrans's *client class*: local,
ollama, glm, groq and custom are all 'Custom Server' there, so the local and
ollama branches were unreachable and every one of them got the custom-server
line. The provider the user chose is now carried in 'whisperjav_provider'.
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

    def test_the_local_marker_does_not_survive_to_the_translator(self):
        # The registry says 'Local', but a local run never reaches
        # translate_subtitle with that config: cli.py and service.py build a
        # fresh one naming the llama-cpp server they just started, and PySubtrans
        # has to be told 'Custom Server' to talk to it. That is what made the
        # local branch of the advice unreachable.
        assert PROVIDER_CONFIGS["local"]["pysubtrans_name"] == "Local"

        # A wiring guard, not a behaviour test: the two places that build that
        # config cannot be imported here (both run argparse at import time) and
        # cannot be exercised without starting a real llama-cpp server, so the
        # only thing that can be checked cheaply is that the marker is still
        # written where the advice will look for it.
        from pathlib import Path

        translate = Path(__file__).resolve().parents[1] / "whisperjav" / "translate"
        for name in ("cli.py", "service.py"):
            source = (translate / name).read_text(encoding="utf-8")
            marker = source.index("local_provider_config = {")
            block = source[marker:marker + 400]
            assert "'pysubtrans_name': 'Custom Server'" in block, name
            assert "'whisperjav_provider': 'local'" in block, name


class TestTheAdviceMatchesTheServerInUse:
    def test_local_is_pointed_at_whisperjavs_own_log(self):
        # The config cli.py and service.py build for a local run: PySubtrans
        # sees a custom server, but WhisperJAV started it.
        config = {
            "pysubtrans_name": "Custom Server",
            "whisperjav_provider": "local",
            "server_address": "http://127.0.0.1:8080",
        }
        text = advice(config)
        assert "local translation server" in text
        assert "ollama" not in text.lower()

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


class TestTheLogItPointsAtIsStillThere:
    def test_naming_the_local_log_asks_for_it_to_be_kept(self):
        # Advice naming a file that stop_local_server deletes a moment later is
        # advice the user cannot act on.
        from whisperjav.translate import local_backend

        local_backend._keep_server_log = False
        advice({"whisperjav_provider": "local"})
        assert local_backend._keep_server_log is True

    def test_a_kept_log_survives_stopping_the_server(self, tmp_path):
        from whisperjav.translate import local_backend

        log = tmp_path / "llm_server.log"
        log.write_text("the reason it failed", encoding="utf-8")

        local_backend._server_process = None
        local_backend._server_stderr_path = str(log)
        local_backend._keep_server_log = True
        try:
            local_backend.stop_local_server()
            assert log.exists()
        finally:
            local_backend._server_stderr_path = None
            local_backend._keep_server_log = False

    def test_an_ordinary_run_still_cleans_its_log_up(self, tmp_path):
        from whisperjav.translate import local_backend

        log = tmp_path / "llm_server.log"
        log.write_text("nothing went wrong", encoding="utf-8")

        local_backend._server_process = None
        local_backend._server_stderr_path = str(log)
        local_backend._keep_server_log = False
        try:
            local_backend.stop_local_server()
            assert not log.exists()
        finally:
            local_backend._server_stderr_path = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
