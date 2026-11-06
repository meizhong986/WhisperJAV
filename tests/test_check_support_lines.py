"""whisperjav --check names what support asks for first (1.10.2, owner W563): version, where PyTorch is loaded
from (#502), PYTHONPATH and the user's own package folder, and which addresses answer (#495)."""
import site
import sys
import sysconfig
import time
import types
import urllib.request

from whisperjav.utils import preflight_check as pc


def _results(checker, name):
    return [r for r in checker.results if r.name == name]


def _setup(monkeypatch, tmp_path, *, own_has_torch, torch_in):
    own = tmp_path / "own-site-packages"
    own.mkdir()
    if own_has_torch:
        (own / "torch").mkdir()
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(own), "platlib": str(own)})
    fake = types.ModuleType("torch")
    fake.__file__ = str((own if torch_in == "own" else tmp_path / "elsewhere") / "torch" / "__init__.py")
    fake.__version__ = "2.10.0+cpu"
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    checker = pc.PreflightChecker()
    checker._check_install_python()
    return checker


def test_502_another_pytorch_ahead_of_the_installed_one_is_a_warning(monkeypatch, tmp_path):
    checker = _setup(monkeypatch, tmp_path, own_has_torch=True, torch_in="elsewhere")
    [loc] = _results(checker, "PyTorch location")
    assert loc.status == pc.CheckStatus.WARN
    assert "2.10.0+cpu" in loc.message and "ahead of the PyTorch in this install's own packages" in loc.message
    assert _results(checker, "WhisperJAV")[0].status == pc.CheckStatus.INFO


def test_pytorch_in_the_own_packages_passes(monkeypatch, tmp_path):
    checker = _setup(monkeypatch, tmp_path, own_has_torch=True, torch_in="own")
    assert _results(checker, "PyTorch location")[0].status == pc.CheckStatus.PASS


def test_pytorch_elsewhere_with_none_in_the_install_is_only_information(monkeypatch, tmp_path):
    # A venv with system packages, a per-user Python, Own PyTorch put in the user folder: not #502's case.
    checker = _setup(monkeypatch, tmp_path, own_has_torch=False, torch_in="elsewhere")
    [loc] = _results(checker, "PyTorch location")
    assert loc.status == pc.CheckStatus.INFO and "instead" not in loc.message


def test_pythonpath_is_a_warning_and_the_user_folder_is_named(monkeypatch, tmp_path):
    own = tmp_path / "own"
    own.mkdir()
    user_site = tmp_path / "user-site"
    user_site.mkdir()
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(own), "platlib": str(own)})
    monkeypatch.setenv("PYTHONPATH", r"D:\old\libs")
    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(user_site))
    monkeypatch.setattr(sys, "path", [str(user_site), str(own)] + sys.path)
    checker = pc.PreflightChecker()
    checker._check_install_python()
    assert r"D:\old\libs" in _results(checker, "PYTHONPATH")[0].message
    assert str(user_site) in _results(checker, "User package folder")[0].message


def test_unreachable_addresses_are_named_with_the_proxy_in_use(monkeypatch):
    monkeypatch.setattr(pc, "_address_answers",
                        lambda url: "timed out" if "pypi" in url or "releases/download" in url else None)
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"https": "http://127.0.0.1:7890"})
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    checker = pc.PreflightChecker()
    checker._check_network()
    [net] = _results(checker, "Network")
    assert net.status == pc.CheckStatus.WARN
    assert "pypi.org, GitHub release downloads" in net.message and "127.0.0.1:7890" in net.message
    assert "huggingface.co" in net.message


def test_hf_endpoint_is_tested_instead_of_huggingface(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    names = [name for name, _url in pc._network_targets()]
    assert names[-1] == "hf-mirror.com" and "huggingface.co" not in names


def test_all_addresses_answer(monkeypatch):
    monkeypatch.setattr(pc, "_address_answers", lambda url: None)
    checker = pc.PreflightChecker()
    checker._check_network()
    assert _results(checker, "Network")[0].status == pc.CheckStatus.PASS


def test_a_hanging_address_does_not_hold_the_check(monkeypatch):
    monkeypatch.setattr(pc, "NETWORK_TIMEOUT_S", 0)
    monkeypatch.setattr(pc, "_address_answers", lambda url: time.sleep(30) if "pypi" in url else None)
    started = time.monotonic()
    answers = dict(pc._probe_all([("pypi.org", "https://pypi.org/simple/"), ("x", "https://x/")]))
    assert time.monotonic() - started < 10
    assert answers["pypi.org"].startswith("no answer within") and answers["x"] is None
