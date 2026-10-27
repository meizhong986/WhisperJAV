"""#411: a GPU the installed PyTorch build has no kernels for is not a usable GPU; and the start of
each run and --check by the install record (design 22.5, owner W234 D-A).

Covers the pure rule, the detector that every device choice derives from, the --check suite and
the start-of-run check, with a stand-in torch so no real CUDA is needed.
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


def _fake_torch(capability, arch_list, name="NVIDIA GeForce GTX 1060 6GB", hip=None, mps=False):
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda i=0: name,
        get_device_capability=lambda i=0: capability,
        get_arch_list=lambda: list(arch_list),
        device_count=lambda: 1,
    )
    backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    return types.SimpleNamespace(cuda=cuda, backends=backends, __version__="2.11.0+cu128",
                                 version=types.SimpleNamespace(cuda="12.8", hip=hip))


def _no_gpu_torch(cuda_build="", mps=False):
    cuda = types.SimpleNamespace(is_available=lambda: False)
    backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    return types.SimpleNamespace(cuda=cuda, backends=backends, __version__="2.11.0",
                                 version=types.SimpleNamespace(cuda=cuda_build or None, hip=None))


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


# --- the record of the install --------------------------------------------------------------

def _record(monkeypatch, value):
    """Stand in for the install record: a value, or None for no record."""
    from whisperjav.installer.core import accelerators
    monkeypatch.setattr(accelerators, "read_record",
                        lambda prefix=None: (value is not None, value))


class TestCheckSuite:
    """--check's Accelerators group (design 22.5.5; owner W232 2: exit 1 only when the install does
    not fit the computer)."""

    def _checker(self, monkeypatch, torch_module, record, ct2=False, smi=""):
        from whisperjav.utils import preflight_check as pf
        monkeypatch.setitem(sys.modules, "torch", torch_module)
        _record(monkeypatch, record)
        monkeypatch.setattr(pf, "_ctranslate2_sees_cuda", lambda: ct2)
        monkeypatch.setattr(pf, "_nvidia_smi_gpu_name", lambda: smi)
        checker = pf.PreflightChecker()
        checker._check_cuda_availability()
        return pf, {r.name: r for r in checker.results}, checker.results

    def test_unusable_card_on_an_nvidia_install_is_the_one_failure(self, monkeypatch):
        pf, by_name, _ = self._checker(monkeypatch, _fake_torch((6, 1), CU128_LIST), "cu128")
        r = by_name["PyTorch"]
        assert r.status == pf.CheckStatus.FAIL and r.fatal
        assert "the NVIDIA GPU cannot be used" in r.message and "6.1" in r.message
        assert r.details[0] == "1. Run the WhisperJAV installer again"
        assert by_name["Accelerator"].message == "Accelerator of this install: NVIDIA GPU (CUDA 12.8)."

    def test_driver_problem_on_an_nvidia_install_names_the_driver(self, monkeypatch):
        pf, by_name, _ = self._checker(monkeypatch, _no_gpu_torch(cuda_build="13.0"), "cu130")
        r = by_name["PyTorch"]
        assert r.status == pf.CheckStatus.FAIL and r.fatal
        assert r.details[0].startswith("1. Update the NVIDIA driver (580 or newer)")
        assert "pip" not in " ".join(r.details)

    @pytest.mark.parametrize("record", ["cpu", "vulkan", "own-pytorch", None])
    def test_no_gpu_on_other_installs_is_info_not_a_failure(self, monkeypatch, record):
        pf, by_name, results = self._checker(monkeypatch, _no_gpu_torch(), record)
        assert by_name["PyTorch"].status == pf.CheckStatus.INFO
        assert by_name["PyTorch"].message.endswith("CPU (not accelerated)")
        assert not [r for r in results if r.status == pf.CheckStatus.FAIL]

    def test_a_supported_card_passes(self, monkeypatch):
        pf, by_name, _ = self._checker(monkeypatch, _fake_torch((8, 6), CU128_LIST, "RTX 3060"), "cu128", ct2=True)
        assert by_name["PyTorch"].status == pf.CheckStatus.PASS
        assert by_name["PyTorch"].message == "PyTorch 2.11.0+cu128: NVIDIA GPU (CUDA 12.8), RTX 3060"
        assert by_name["CTranslate2"].message == "CTranslate2 (balanced, fast, faster): NVIDIA GPU (CUDA)"
        assert by_name["ONNX Runtime"].message.endswith("CPU (not accelerated) in every install")

    def test_rocm_own_pytorch_is_info_and_not_tested(self, monkeypatch):
        pf, by_name, _ = self._checker(
            monkeypatch, _fake_torch((9, 0), [], "AMD Radeon RX 7900", hip="6.2"), "own-pytorch")
        assert by_name["PyTorch"].status == pf.CheckStatus.INFO
        assert by_name["PyTorch"].message == "PyTorch 2.11.0+cu128 (your own): AMD GPU (ROCm), AMD Radeon RX 7900 — not tested"
        assert by_name["CTranslate2"].message.endswith("CPU (not accelerated)")

    def test_cpu_only_install_on_a_pc_with_an_nvidia_gpu_is_a_warning(self, monkeypatch):
        pf, _, results = self._checker(monkeypatch, _no_gpu_torch(), "cpu", smi="NVIDIA GeForce RTX 4070")
        warn = [r for r in results if r.status == pf.CheckStatus.WARN]
        assert len(warn) == 1 and "RTX 4070" in warn[0].message and "made for the CPU only" in warn[0].message

    def test_no_pip_command_for_pytorch_anywhere_in_the_checks(self):
        from pathlib import Path
        from whisperjav.utils import preflight_check as pf
        text = Path(pf.__file__).read_text(encoding="utf-8")
        for removed in ("pip install torch torchvision torchaudio", "pip install --upgrade torch",
                        "pip uninstall torch"):
            assert removed not in text
        assert "pip install fireredvad" in text  # not a PyTorch command; stays


# --- the start of each run ------------------------------------------------------------------

def _flat(out):
    """The box's text as one line: the box wraps long sentences."""
    return " ".join(out.replace("║", " ").split())


class TestStartOfRun:
    """design 22.5.2 (owner W234 D-A): by the install record, silent / a notice / the one stop.
    Nothing is ever asked."""

    def _run(self, monkeypatch, torch_module, record, device=None):
        import builtins
        from whisperjav.utils import preflight_check as pf
        monkeypatch.setitem(sys.modules, "torch", torch_module)
        _record(monkeypatch, record)
        monkeypatch.setattr(pf, "_gguf_where", lambda best: f"GGUF[{best}]")
        monkeypatch.setattr(pf, "cublas12_files_missing", lambda: [])
        monkeypatch.setattr(pf, "_ctranslate2_sees_cuda", lambda: False)
        monkeypatch.setattr(builtins, "input", lambda *a: pytest.fail("the start of a run asked a question"))
        return pf.start_of_run_check(device=device)

    def test_nvidia_install_with_unusable_gpu_stops_with_status_3(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as ei:
            self._run(monkeypatch, _fake_torch((6, 1), CU128_LIST), "cu128")
        assert ei.value.code == 3
        out = _flat(capsys.readouterr().out)
        assert "This install is made for an NVIDIA GPU, but the GPU cannot be used now" in out
        assert "6.1" in out and "WhisperJAV 1.9.5 does" in out   # the old card: the owner's W43 line
        assert "--device cpu" in out and "Nothing was processed." in out
        assert "╔" in out and "╚" in out

    def test_nvidia_install_with_unusable_gpu_and_device_cpu_goes_on(self, monkeypatch, capsys):
        assert self._run(monkeypatch, _fake_torch((6, 1), CU128_LIST), "cu130", device="cpu") == "notice"
        out = _flat(capsys.readouterr().out)
        assert "made for: NVIDIA GPU (CUDA 13.0)" in out and "as asked with --device cpu" in out

    def test_nvidia_install_with_a_working_gpu_is_silent(self, monkeypatch, capsys):
        assert self._run(monkeypatch, _fake_torch((8, 6), CU128_LIST, "RTX 3060"), "cu128") == "silent"
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("record,name", [
        ("vulkan", "GPU through Vulkan (AMD, Intel or older NVIDIA card)"),
        ("cpu", "CPU only (not accelerated)"),
    ])
    def test_other_installs_get_the_notice_and_go_on(self, monkeypatch, capsys, record, name):
        assert self._run(monkeypatch, _no_gpu_torch(), record) == "notice"
        out = _flat(capsys.readouterr().out)
        assert f"This install of WhisperJAV is made for: {name}. On this computer:" in out
        assert "GGUF[cpu]" in out and "Other pipelines: the CPU (not accelerated)" in out
        assert "?" not in out  # no question

    def test_apple_install_is_silent_with_the_apple_gpu_and_a_notice_without(self, monkeypatch, capsys):
        assert self._run(monkeypatch, _no_gpu_torch(mps=True), "metal") in ("silent", "notice")
        monkeypatch.setattr(dd, "get_best_device", lambda prefer_cpu=False: "mps")
        from whisperjav.utils import preflight_check as pf
        monkeypatch.setattr(pf, "get_best_device", lambda: "mps")
        assert self._run(monkeypatch, _no_gpu_torch(mps=True), "metal") == "silent"
        capsys.readouterr()
        monkeypatch.setattr(pf, "get_best_device", lambda: "cpu")
        assert self._run(monkeypatch, _no_gpu_torch(), "metal") == "notice"
        assert "made for: Apple GPU (Metal)" in _flat(capsys.readouterr().out)

    def test_own_pytorch_on_rocm_names_its_gpu(self, monkeypatch, capsys):
        assert self._run(monkeypatch, _fake_torch((9, 0), [], "AMD Radeon RX 7900", hip="6.2"),
                         "own-pytorch") == "notice"
        out = capsys.readouterr().out.replace("\n", " ").replace("║", " ")
        out = " ".join(out.split())
        assert "made for: Own PyTorch" in out
        assert "ask for its AMD GPU (ROCm), AMD Radeon RX 7900 — not tested" in out
        assert "balanced, fast and faster on the CPU (not accelerated)" in out

    def test_no_record_is_judged_by_pytorch_and_never_stops(self, monkeypatch, capsys):
        # report v2 3.3, restored after review finding 4
        assert self._run(monkeypatch, _fake_torch((8, 6), CU128_LIST, "RTX 3060"), None) == "silent"
        assert self._run(monkeypatch, _fake_torch((6, 1), CU128_LIST), None) == "notice"
        assert self._run(monkeypatch, _no_gpu_torch(cuda_build="12.8"), None) == "notice"
        assert "No GPU that WhisperJAV can use was found" in _flat(capsys.readouterr().out)

    def test_the_notice_and_the_stop_reach_the_log_file(self, monkeypatch, tmp_path):
        import logging
        from whisperjav.utils.logger import logger as wj_logger
        log = tmp_path / "run.log"
        handler = logging.FileHandler(log, encoding="utf-8")
        wj_logger.addHandler(handler)
        try:
            self._run(monkeypatch, _no_gpu_torch(), "cpu")
            with pytest.raises(SystemExit):
                self._run(monkeypatch, _fake_torch((6, 1), CU128_LIST), "cu128")
        finally:
            wj_logger.removeHandler(handler)
            handler.close()
        text = log.read_text(encoding="utf-8")
        assert "made for: CPU only (not accelerated)" in text and "the GPU cannot be used now" in text

    def test_gguf_line_follows_the_installed_crispasr_build(self, monkeypatch):
        from whisperjav.crispasr import install as crisp
        from whisperjav.utils import preflight_check as pf
        monkeypatch.setattr(crisp, "crispasr_installed", lambda: True)
        for build, best, expected in [("vulkan", "cpu", pf.GGUF_VULKAN_WHERE), ("cuda", "cuda", pf.GGUF_CUDA_WHERE),
                                      ("cuda", "cpu", pf.GGUF_CPU_WHERE), ("cpu", "cpu", pf.GGUF_CPU_WHERE)]:
            monkeypatch.setattr(crisp, "installed_build", lambda b=build: ("0.8.44", b))
            assert pf._gguf_where(best) == expected

    def test_the_question_and_its_consent_mechanisms_are_gone(self):
        from pathlib import Path
        from whisperjav.utils import preflight_check as pf
        for name in ("enforce_gpu_requirement", "cpu_consent_in_argv", "CPU_ACCEPTED_ENV",
                     "_ask_to_continue_on_cpu", "CPU_QUESTION"):
            assert not hasattr(pf, name), name
        main_src = (Path(pf.__file__).parents[1] / "main.py").read_text(encoding="utf-8")
        assert "enforce_gpu_requirement" not in main_src and "cpu_consent_in_argv" not in main_src
        assert main_src.index("start_of_run_check(device=args.device)") > main_src.index(
            "logger = setup_logger(\"whisperjav\", log_level, args.log_file)")
