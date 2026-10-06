"""#438: the installer "installed" PyTorch in 6 s from a damaged uv cache and PyTorch then failed to import; clearing
%LOCALAPPDATA%\\uv\\cache fixed it (reporter). install_pytorch now checks the import in a fresh process and, under uv,
downloads PyTorch once more past the cache. The template is loaded as in test_post_install_verification_v192.py; the
installer and the import check are stand-ins, nothing is installed."""
import re
import types
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).parent.parent / "installer" / "templates" / "post_install.py.template"


def _load():
    source = re.sub(r"\{\{[A-Z_]+\}\}", "1.9.4", TEMPLATE.read_text(encoding="utf-8"))
    module = types.ModuleType("post_install_under_test")
    module.__file__ = str(TEMPLATE)
    exec(compile(source, str(TEMPLATE), "exec"), module.__dict__)
    return module


@pytest.fixture
def m(tmp_path, monkeypatch):
    mod = _load()
    lines = []
    monkeypatch.setattr(mod, "LOG_FILE", str(tmp_path / "log.txt"))
    monkeypatch.setattr(mod, "log", lines.append)
    monkeypatch.setattr(mod, "log_section", lines.append)
    monkeypatch.setattr(mod, "verify_existing_torch_stack", lambda: (False, None))
    plan = mod.TorchInstallPlan(pip_args=["install", "torch", "torchaudio"], pip_command="pip3 install torch torchaudio",
                                description="Install PyTorch", uses_gpu=True, target_label="cu128",
                                driver_requirement=(570, 0, 0), driver_detected=(570, 0, 0), gpu_name="GPU", reason="")
    monkeypatch.setattr(mod, "select_torch_install_plan", lambda d: plan)
    monkeypatch.setattr(mod, "create_gpu_constraints", lambda pins: True)
    mod._lines = lines
    return mod


def _driver(mod):
    return object()


def test_import_failure_under_uv_triggers_one_fresh_download(m, monkeypatch):
    calls, checks = [], iter([(False, "OSError: [WinError 126] ... c10.dll"), (True, "")])
    monkeypatch.setattr(m, "USE_UV", True)
    monkeypatch.setattr(m, "run_pip", lambda args, desc, **k: calls.append(args) or True)
    monkeypatch.setattr(m, "torch_imports_in_fresh_process", lambda: next(checks))
    assert m.install_pytorch(_driver(m)) is None
    assert calls == [["install", "torch", "torchaudio"], ["install", "torch", "torchaudio", "--reinstall", "--no-cache"]]


def test_still_broken_after_the_fresh_download_stops_with_the_cache_advice(m, monkeypatch):
    monkeypatch.setattr(m, "USE_UV", True)
    monkeypatch.setattr(m, "run_pip", lambda args, desc, **k: True)
    monkeypatch.setattr(m, "torch_imports_in_fresh_process", lambda: (False, "ImportError: DLL load failed"))
    reason = m.install_pytorch(_driver(m))
    assert reason == "PyTorch installed but does not import: ImportError: DLL load failed"
    assert any("uv\\cache" in str(l) for l in m._lines)


def test_a_clean_install_is_not_downloaded_twice(m, monkeypatch):
    calls = []
    monkeypatch.setattr(m, "USE_UV", True)
    monkeypatch.setattr(m, "run_pip", lambda args, desc, **k: calls.append(args) or True)
    monkeypatch.setattr(m, "torch_imports_in_fresh_process", lambda: (True, ""))
    assert m.install_pytorch(_driver(m)) is None
    assert len(calls) == 1
