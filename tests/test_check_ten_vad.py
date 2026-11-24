"""whisperjav --check says whether TEN VAD loads (1.10.3, owner W753 CL5; wording W764 2, texts sheet item 4)."""
import sys
import types

from whisperjav.utils import preflight_check
from whisperjav.utils.preflight_check import CheckStatus, PreflightChecker


def _ten_result(checker):
    return [r for r in checker.results if r.name == "TEN VAD"]


def _fake_ten_vad(monkeypatch, error=None):
    module = types.ModuleType("ten_vad")

    class TenVad:
        def __init__(self, hop_size, threshold):
            if error:
                raise error

    module.TenVad = TenVad
    monkeypatch.setitem(sys.modules, "ten_vad", module)


def test_loads(monkeypatch):
    _fake_ten_vad(monkeypatch)
    checker = PreflightChecker()
    checker._check_ten_vad()
    (result,) = _ten_result(checker)
    assert (result.status, result.message) == (CheckStatus.PASS, "loads")


def test_cannot_load_on_linux_names_the_error_and_the_packages(monkeypatch):
    _fake_ten_vad(monkeypatch, OSError("libc++.so.1: cannot open shared object file: No such file or directory"))
    monkeypatch.setattr(preflight_check.sys, "platform", "linux")
    checker = PreflightChecker()
    checker._check_ten_vad()
    (result,) = _ten_result(checker)
    assert result.status == CheckStatus.FAIL and result.fatal is False
    assert result.message == ("cannot load (libc++.so.1: cannot open shared object file: No such file or directory). "
                              "On Ubuntu or Debian: sudo apt-get install -y libc++1 libc++abi1")


def test_cannot_load_elsewhere_names_the_error_only(monkeypatch):
    _fake_ten_vad(monkeypatch, ImportError("DLL load failed"))
    monkeypatch.setattr(preflight_check.sys, "platform", "win32")
    checker = PreflightChecker()
    checker._check_ten_vad()
    (result,) = _ten_result(checker)
    assert result.message == "cannot load (DLL load failed)"
