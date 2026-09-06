"""#415: offline mode and cache-first Hugging Face loading.

Light tests (no ML imports) for the helper module and the resilience wrapper,
plus the real CLI for the flag and the environment it sets.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from whisperjav.utils import offline_mode as om


class TestHelper:
    def test_flag_detection(self):
        assert om.offline_requested(["--offline"])
        assert om.offline_requested(["x.mp4", "--mode", "balanced", "--offline"])
        assert not om.offline_requested(["x.mp4", "--offline-ish"])
        # argparse accepts unambiguous prefixes; the pre-import scan must too
        assert om.offline_requested(["--offl"])
        assert om.offline_requested(["--off"])
        assert not om.offline_requested(["--o"])
        assert not om.offline_requested(["offline"])

    def test_enable_sets_the_hub_variable(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        assert not om.is_offline()
        om.enable_offline_mode()
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert om.is_offline() and om.hub_offline()

    @pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
    def test_either_hub_variable_counts_as_offline(self, monkeypatch, var):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        monkeypatch.setenv(var, "1")
        assert om.is_offline()


class _Loader:
    """Records from_pretrained calls; raises `exc` for the first N local calls."""

    def __init__(self, fail_local_times=0, exc=OSError("not in cache")):
        self.calls = []
        self._fail_local = fail_local_times
        self._exc = exc

    def from_pretrained(self, model_id, **kw):
        self.calls.append((model_id, kw))
        if kw.get("local_files_only") and self._fail_local > 0:
            self._fail_local -= 1
            raise self._exc
        return ("loaded", model_id, kw)


class TestLoadCachedFirst:
    def test_cached_model_never_touches_the_hub(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        ld = _Loader()
        out = om.load_cached_first(ld, "org/model", dtype="x")
        assert out[0] == "loaded"
        assert ld.calls == [("org/model", {"local_files_only": True, "dtype": "x"})]

    def test_missing_model_falls_back_to_a_normal_load(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        ld = _Loader(fail_local_times=1)
        om.load_cached_first(ld, "org/model")
        assert [c[1].get("local_files_only") for c in ld.calls] == [True, None]

    def test_offline_mode_names_the_model_and_makes_no_second_attempt(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        ld = _Loader(fail_local_times=1)
        with pytest.raises(OSError) as ei:
            om.load_cached_first(ld, "org/model")
        assert len(ld.calls) == 1
        assert "Offline mode: 'org/model' is not in the local Hugging Face cache" in str(ei.value)
        assert isinstance(ei.value.__cause__, OSError)

    def test_non_cache_failures_are_not_retried_online(self, monkeypatch):
        # A corrupt weight file or a bad dtype must surface from the first attempt,
        # not be swallowed and re-run as a full online load.
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        ld = _Loader(fail_local_times=1, exc=RuntimeError("corrupt safetensors"))
        with pytest.raises(RuntimeError):
            om.load_cached_first(ld, "org/model")
        assert len(ld.calls) == 1


class TestResilienceWrapperUnderOffline:
    def test_wrapper_passes_straight_through_when_hub_is_offline(self, monkeypatch):
        import huggingface_hub.constants as c
        from whisperjav.utils import model_loader as ml
        monkeypatch.setattr(c, "HF_HUB_OFFLINE", True)
        calls = []

        def original(*a, **kw):
            calls.append(kw)
            raise ConnectionError("offline")

        wrapped = ml._make_resilient_wrapper(original, "hf_hub_download")
        with pytest.raises(ConnectionError):
            wrapped("org/model", "file.json")
        # exactly one call, and no local_files_only / endpoint fallback attempts
        assert calls == [{}]

    def test_wrapper_still_falls_back_when_online(self, monkeypatch):
        import huggingface_hub.constants as c
        from whisperjav.utils import model_loader as ml
        monkeypatch.setattr(c, "HF_HUB_OFFLINE", False)
        calls = []

        def original(*a, **kw):
            calls.append(kw)
            if kw.get("local_files_only"):
                return "cached"
            raise ConnectionError("connection timed out")

        wrapped = ml._make_resilient_wrapper(original, "hf_hub_download")
        assert wrapped("org/model", "file.json") == "cached"
        assert calls == [{}, {"local_files_only": True}]


@pytest.mark.slow
class TestCli:
    def _run(self, *args, timeout=600):
        return subprocess.run(
            [sys.executable, "-m", "whisperjav.main", *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )

    def test_flag_is_registered(self):
        p = self._run("--help")
        assert "--offline" in p.stdout
        assert self._run("--offline", "--help").returncode == 0

    def test_flag_sets_offline_mode_inside_the_process(self, tmp_path):
        out = tmp_path / "dump.json"
        p = self._run("--offline", "--dump-params", str(out), "--mode", "faster")
        assert out.exists(), p.stderr[-2000:]
        dump = json.loads(out.read_text(encoding="utf-8"))
        assert dump["cli_args"]["offline"] is True
        assert dump["offline_mode"] is True
        # The discriminating variable: huggingface_hub froze HF_HUB_OFFLINE at import,
        # so this is True only if the scan ran BEFORE the hub was imported.
        assert dump["hub_constant_offline"] is True
        assert "Offline mode: using downloaded Hugging Face models only" in (p.stdout + p.stderr)

    def test_abbreviated_flag_also_sets_offline_mode(self, tmp_path):
        out = tmp_path / "dump.json"
        p = self._run("--offl", "--dump-params", str(out), "--mode", "faster")
        assert out.exists(), p.stderr[-2000:]
        dump = json.loads(out.read_text(encoding="utf-8"))
        assert dump["cli_args"]["offline"] is True
        assert dump["offline_mode"] is True
        assert dump["hub_constant_offline"] is True

    def test_without_flag_offline_is_off(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        out = tmp_path / "dump.json"
        p = self._run("--dump-params", str(out), "--mode", "faster")
        assert out.exists(), p.stderr[-2000:]
        dump = json.loads(out.read_text(encoding="utf-8"))
        assert dump["cli_args"]["offline"] is False
        assert dump["offline_mode"] is False
        assert dump["hub_constant_offline"] is False
