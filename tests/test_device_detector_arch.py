"""#411: a GPU the installed PyTorch build has no kernels for is not a usable GPU.

Covers the pure rule, the detector that every device choice derives from, and
the --check suite, with a stand-in torch so no real CUDA is needed.
"""
from __future__ import annotations

import sys
import types

import pytest

from whisperjav.utils import device_detector as dd

CU128_LIST = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


class TestRule:
    @pytest.mark.parametrize("capability,expected", [
        ((6, 1), False),   # GTX 1060 (#411)
        ((7, 0), False),   # V100 (#333)
        ((7, 5), True),    # RTX 20
        ((8, 6), True),    # RTX 30
        ((8, 9), True),    # RTX 40: same major as sm_86, higher minor
        ((12, 0), True),   # RTX 50
        ((13, 0), False),  # a future major with no binary and no PTX
    ])
    def test_cu128_wheel(self, capability, expected):
        assert dd.cuda_build_supports_device(capability, CU128_LIST) is expected

    def test_ptx_forward_compatibility(self):
        assert dd.cuda_build_supports_device((9, 0), ["compute_75"]) is True
        assert dd.cuda_build_supports_device((7, 0), ["compute_75"]) is False

    def test_unknown_list_is_not_judged(self):
        assert dd.cuda_build_supports_device((6, 1), []) is True
        assert dd.cuda_build_supports_device((6, 1), None) is True
        assert dd.cuda_build_supports_device((6, 1), ["weird"]) is True
        assert dd.cuda_build_supports_device((6, 1), ["SM_61", "gfx90a"]) is True   # unknown kinds are not judged

    @pytest.mark.parametrize("capability,arch_list,expected", [
        ((9, 0), ["sm_90a"], True),                 # Hopper arch-specific cubin
        ((12, 0), ["sm_100f", "sm_120a"], True),    # family/arch-specific Blackwell names
        ((10, 0), ["sm_90", "sm_100a"], True),
        ((6, 1), ["sm_90a", "sm_120a"], False),
    ])
    def test_arch_specific_suffixes_are_stripped_like_torch(self, capability, arch_list, expected):
        assert dd.cuda_build_supports_device(capability, arch_list) is expected


def _fake_torch(capability, arch_list, name="NVIDIA GeForce GTX 1060 6GB"):
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda i=0: name,
        get_device_capability=lambda i=0: capability,
        get_arch_list=lambda: list(arch_list),
        device_count=lambda: 1,
    )
    return types.SimpleNamespace(cuda=cuda, version=types.SimpleNamespace(cuda="12.8"))


class TestDetector:
    def test_unsupported_card_is_reported_as_no_gpu_with_reason(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch((6, 1), CU128_LIST))
        ok, name = dd._check_cuda_available()
        assert ok is False and name is None
        assert "compute capability 6.1" in dd.CUDA_UNUSABLE_REASON
        assert "sm_75" in dd.CUDA_UNUSABLE_REASON
        assert dd.get_best_device() != "cuda"

    def test_supported_card_is_unchanged(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch((8, 6), CU128_LIST, "NVIDIA GeForce RTX 3060"))
        ok, name = dd._check_cuda_available()
        assert ok is True and name == "NVIDIA GeForce RTX 3060"
        assert dd.CUDA_UNUSABLE_REASON is None


class TestCheckSuite:
    def test_check_reports_the_build_mismatch_as_fatal(self, monkeypatch):
        from whisperjav.utils.preflight_check import CheckStatus, PreflightChecker
        monkeypatch.setitem(sys.modules, "torch", _fake_torch((6, 1), CU128_LIST))
        checker = PreflightChecker()
        checker._check_cuda_availability()
        r = checker.results[-1]
        assert r.name == "CUDA Availability"
        assert r.status == CheckStatus.FAIL and r.fatal
        assert "not supported by this PyTorch build" in r.message
        assert any("6.1" in d for d in r.details)

    def test_check_passes_for_a_supported_card(self, monkeypatch):
        from whisperjav.utils.preflight_check import CheckStatus, PreflightChecker
        monkeypatch.setitem(sys.modules, "torch", _fake_torch((8, 6), CU128_LIST, "RTX 3060"))
        checker = PreflightChecker()
        checker._check_cuda_availability()
        assert checker.results[-1].status == CheckStatus.PASS


class TestCpuConsent:
    @pytest.mark.parametrize("argv,expected", [
        (["x.mp4", "--accept-cpu-mode"], True),
        (["x.mp4", "--device", "cpu"], True),
        (["x.mp4", "--device=cpu"], True),
        (["x.mp4", "--device", "cuda"], False),
        (["x.mp4", "--device"], False),
        (["x.mp4"], False),
    ])
    def test_command_line_answers(self, argv, expected):
        from whisperjav.utils.preflight_check import cpu_consent_in_argv
        assert cpu_consent_in_argv(argv) is expected


class TestStartupGate:
    """The unconditional gate every run passes through (owner, 2026-09-06): an
    unusable GPU stops the run and ASKS proceed-or-abort; nothing continues on its
    own; where nobody can answer, it aborts and says how to answer."""

    def _gate(self, monkeypatch, interactive, answer=None):
        from whisperjav.utils import preflight_check as pf
        monkeypatch.delenv(pf.CPU_ACCEPTED_ENV, raising=False)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch((6, 1), CU128_LIST))
        monkeypatch.setattr(pf, "_stdin_is_interactive", lambda: interactive)
        asked = []
        monkeypatch.setattr(pf, "_ask", lambda prompt: (asked.append(prompt), answer or "")[1])
        return pf, asked

    def test_console_user_is_asked_and_no_means_abort(self, monkeypatch, capsys):
        pf, asked = self._gate(monkeypatch, interactive=True, answer="")
        with pytest.raises(SystemExit) as ei:
            pf.enforce_gpu_requirement(accept_cpu_mode=False)
        assert ei.value.code == 1
        assert asked == ["Continue on the CPU anyway? [y/N] "]
        out = capsys.readouterr().out
        assert "compute capability 6.1" in out and "Aborted. Nothing was processed." in out

    def test_console_user_can_choose_to_continue_on_cpu(self, monkeypatch, capsys):
        pf, asked = self._gate(monkeypatch, interactive=True, answer="y")
        assert pf.enforce_gpu_requirement(accept_cpu_mode=False) is True
        assert asked and "Continuing on the CPU" in capsys.readouterr().out

    def test_yes_is_remembered_for_the_run_and_its_workers(self, monkeypatch, capsys):
        # The check runs at import time and again in main(), and every spawned
        # worker re-runs whisperjav.main's module level: one answer must serve all.
        import os
        pf, asked = self._gate(monkeypatch, interactive=True, answer="y")
        assert pf.enforce_gpu_requirement(accept_cpu_mode=False) is True
        assert os.environ.get(pf.CPU_ACCEPTED_ENV) == "1"
        assert pf.enforce_gpu_requirement(accept_cpu_mode=False) is True
        assert len(asked) == 1
        # a worker with no console inherits the answer and does not abort
        monkeypatch.setattr(pf, "_stdin_is_interactive", lambda: False)
        assert pf.enforce_gpu_requirement(accept_cpu_mode=False) is True
        assert len(asked) == 1

    def test_accept_cpu_mode_is_remembered_too(self, monkeypatch):
        import os
        pf, asked = self._gate(monkeypatch, interactive=False)
        assert pf.enforce_gpu_requirement(accept_cpu_mode=True) is True
        assert os.environ.get(pf.CPU_ACCEPTED_ENV) == "1"

    def test_gui_child_marker_means_nobody_can_answer(self, monkeypatch):
        from whisperjav.utils import preflight_check as pf
        monkeypatch.setenv("WHISPERJAV_NO_CONSOLE", "1")
        monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
        assert pf._stdin_is_interactive() is False

    def test_no_console_aborts_and_says_how_to_answer(self, monkeypatch, capsys):
        pf, asked = self._gate(monkeypatch, interactive=False)
        with pytest.raises(SystemExit) as ei:
            pf.enforce_gpu_requirement(accept_cpu_mode=False)
        assert ei.value.code == 1 and asked == []
        out = capsys.readouterr().out
        assert "--accept-cpu-mode" in out and "Accept CPU-only mode" in out

    def test_no_timeout_auto_continue_on_this_path(self, monkeypatch):
        from whisperjav.utils import preflight_check as pf
        pf_, _ = self._gate(monkeypatch, interactive=False)
        waited = []
        monkeypatch.setattr(pf, "_wait_for_keypress_with_timeout", lambda t: waited.append(t) or True)
        with pytest.raises(SystemExit):
            pf.enforce_gpu_requirement(accept_cpu_mode=False, timeout_seconds=30)
        assert waited == []

    def test_accept_cpu_mode_answers_in_advance(self, monkeypatch):
        pf, asked = self._gate(monkeypatch, interactive=False)
        assert pf.enforce_gpu_requirement(accept_cpu_mode=True) is True
        assert asked == []

    def test_supported_card_passes_silently(self, monkeypatch, capsys):
        from whisperjav.utils import preflight_check as pf
        monkeypatch.setitem(sys.modules, "torch", _fake_torch((8, 6), CU128_LIST, "RTX 3060"))
        assert pf.enforce_gpu_requirement(accept_cpu_mode=False) is True
        assert capsys.readouterr().out == ""
