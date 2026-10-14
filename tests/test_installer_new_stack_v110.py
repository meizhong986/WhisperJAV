"""v1.10.0 item 2 (new stack), the Windows installer's decisions -- spec checks S5 and part of S4.

Old card (compute capability below 7.5) -> the owner's three choices, answered with no time
limit (W36 (1), W37 (1), (2)); driver older than 580 on a supported card -> stop and ask for an
update (W36 (2)); the cuBLAS 12 phase keeps CUDA 13.0 when the load check passes and moves this
PC to the CUDA 12.8 build, with the owner's notice, when it fails (W37 (3), W42 (4), W43).

The template is loaded as in test_installer_torch_cache_438.py. Nothing is installed: input(),
the installer commands and the fresh-process steps are stand-ins.
"""
import re
import types
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).parent.parent / "installer" / "templates" / "post_install.py.template"


def _load():
    source = re.sub(r"\{\{[A-Z_]+\}\}", "1.10.0", TEMPLATE.read_text(encoding="utf-8"))
    module = types.ModuleType("post_install_under_test")
    module.__file__ = str(TEMPLATE)
    exec(compile(source, str(TEMPLATE), "exec"), module.__dict__)
    return module


@pytest.fixture
def m(tmp_path, monkeypatch):
    mod = _load()
    lines = []
    monkeypatch.setattr(mod, "LOG_FILE", str(tmp_path / "log.txt"))
    monkeypatch.setattr(mod, "STACK_FILE", str(tmp_path / "whisperjav_stack.txt"))
    monkeypatch.setattr(mod, "log", lambda msg: lines.append(str(msg)))
    monkeypatch.setattr(mod, "log_section", lambda msg: lines.append(str(msg)))
    mod._lines = lines
    return mod


def _answers(monkeypatch, *typed):
    """input() returns these in turn; records every prompt it was given."""
    seq = iter(typed)
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return next(seq)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


def _info(m, driver=(610, 47, 0), cc=(8, 6), name="NVIDIA GeForce RTX 3060"):
    return m.DriverInfo(driver, name, "nvidia-smi", cc)


# --- E: old card -----------------------------------------------------------------------

@pytest.mark.parametrize("cc", [(6, 1), (7, 0)])
def test_old_card_shows_the_owners_warning_and_waits_for_a_valid_answer(m, monkeypatch, cc):
    prompts = _answers(monkeypatch, "", "x", "3")  # junk is asked again, with no time limit
    assert m.timed_input is not None
    monkeypatch.setattr(m, "timed_input", lambda *a, **k: pytest.fail("the old-card question must not time out"))
    gate, reason = m.gpu_support_gate(_info(m, cc=cc, name="NVIDIA GeForce GTX 1080"))
    assert gate == "stop" and "answer 3" in reason
    assert len(prompts) == 3
    text = "\n".join(m._lines)
    assert (f"Your graphics card, NVIDIA GeForce GTX 1080 (compute capability {cc[0]}.{cc[1]}), is older than "
            "WhisperJAV 1.10 supports. 1.10 needs a GTX 16-series, RTX 20-series or newer card.") in text
    assert ("Choose: 1 = my card is identified wrongly (show where to report it, then stop); "
            "2 = install for the CPU only (much slower); 3 = stop.") in text
    assert "WhisperJAV 1.9.5 still supports this card: https://github.com/meizhong986/WhisperJAV/releases/tag/v1.9.5." in text
    assert m._stopped_on_purpose is True


def test_old_card_answer_1_shows_where_to_report_then_stops(m, monkeypatch):
    _answers(monkeypatch, "1")
    gate, _ = m.gpu_support_gate(_info(m, cc=(6, 1)))
    assert gate == "stop"
    assert any("https://github.com/meizhong986/WhisperJAV/issues" in l and "report" in l for l in m._lines)


def test_old_card_answer_2_installs_for_the_cpu_only(m, monkeypatch):
    _answers(monkeypatch, "2")
    gate, reason = m.gpu_support_gate(_info(m, cc=(6, 1)))
    assert (gate, reason) == ("cpu", None)
    assert m._stopped_on_purpose is False
    plan = m.select_torch_install_plan(_info(m, cc=(6, 1)), force_cpu=True)
    assert plan.uses_gpu is False and "torch==2.11.0" in plan.pip_command


def test_old_card_with_no_console_stops(m, monkeypatch):
    def no_console(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_console)
    gate, _ = m.gpu_support_gate(_info(m, cc=(6, 1)))
    assert gate == "stop"


def test_card_7_5_is_supported_and_asks_nothing(m, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no question for a supported card"))
    assert m.gpu_support_gate(_info(m, cc=(7, 5), name="NVIDIA GeForce GTX 1660")) == ("gpu", None)


def test_unreadable_compute_capability_gives_no_warning(m, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no question when the card cannot be read"))
    assert m.gpu_support_gate(_info(m, cc=None)) == ("gpu", None)


def test_no_nvidia_gpu_is_left_to_the_cpu_path(m, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no question without an NVIDIA GPU"))
    assert m.gpu_support_gate(m.DriverInfo(None, None, "none")) == ("gpu", None)


# --- F: old driver ---------------------------------------------------------------------

def test_driver_566_03_stops_with_the_owners_text_and_waits_for_enter(m, monkeypatch):
    prompts = _answers(monkeypatch, "")
    gate, reason = m.gpu_support_gate(_info(m, driver=(566, 3, 0)))
    assert gate == "stop" and "566.03" in reason
    assert len(prompts) == 1  # one Enter, no time limit
    assert ("Your NVIDIA driver (566.03) is older than WhisperJAV 1.10 needs (580 or newer). Update it from "
            "https://www.nvidia.com/Download/index.aspx, restart the PC, and run this installer again. "
            "Nothing has been installed. Press Enter to close.") in "\n".join(m._lines)
    assert m._stopped_on_purpose is True


def test_driver_579_99_also_stops(m, monkeypatch):
    _answers(monkeypatch, "")
    assert m.gpu_support_gate(_info(m, driver=(579, 99, 0)))[0] == "stop"


def test_driver_580_gets_pytorch_2_11_0_for_cuda_13(m, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no question for driver 580"))
    assert m.gpu_support_gate(_info(m, driver=(580, 0, 0))) == ("gpu", None)
    plan = m.select_torch_install_plan(_info(m, driver=(580, 0, 0)))
    assert plan.uses_gpu is True and plan.target_label == "CUDA 13.0"
    assert plan.pip_args == ["install", "torch==2.11.0", "torchaudio==2.11.0", "torchvision==0.26.0",
                             "numpy>=2.5,<2.6", "--index-url", "https://download.pytorch.org/whl/cu130"]


def test_old_card_is_asked_before_the_driver(m, monkeypatch):
    _answers(monkeypatch, "3")
    gate, reason = m.gpu_support_gate(_info(m, driver=(470, 0, 0), cc=(6, 1)))
    assert gate == "stop" and "compute capability 6.1" in reason


# --- D: the cuBLAS 12 phase ------------------------------------------------------------

def _phase_stubs(m, monkeypatch, check_ok, tmp_path=None):
    calls = []
    # Never the real environment's folder: the fallback deletes the placed DLLs there.
    ct2 = (tmp_path or Path(m.LOG_FILE).parent) / "ctranslate2"
    ct2.mkdir(exist_ok=True)
    for name in ("cublas64_12.dll", "cublasLt64_12.dll"):
        (ct2 / name).write_bytes(b"placed")
    monkeypatch.setattr(m, "ctranslate2_package_dir", lambda: str(ct2))
    m._ct2_dir = ct2
    monkeypatch.setattr(m, "run_pip", lambda args, desc, **k: calls.append(("pip", list(args))) or True)
    monkeypatch.setattr(m, "uninstall_package", lambda name, desc: calls.append(("uninstall", name)) or True)
    steps = {"place": (True, "OK: placed"), "check": (check_ok, "OK" if check_ok else "FAILED: status 3")}
    monkeypatch.setattr(m, "run_cublas12_step", lambda step: calls.append(("step", step)) or steps[step])
    monkeypatch.setattr(m, "torch_imports_in_fresh_process", lambda: (True, ""))
    monkeypatch.setattr(m, "torch_facts_in_fresh_process",
                        lambda: {"torch": "2.11.0+cu128", "torchaudio": "2.11.0+cu128",
                                 "torchvision": "0.26.0+cu128", "cuda": "12.8", "cuda_available": True})
    monkeypatch.setattr(m, "create_gpu_constraints", lambda pins: calls.append(("pins", dict(pins))) or True)
    return calls


def test_cublas12_check_passes_keeps_cuda_13(m, monkeypatch):
    calls = _phase_stubs(m, monkeypatch, check_ok=True)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no notice when the check passes"))
    assert m.cublas12_phase() is None
    assert ("pip", ["install", "nvidia-cublas-cu12==12.8.4.1", "--progress-bar", "on"]) in calls
    assert ("step", "place") in calls and ("step", "check") in calls
    assert Path(m.STACK_FILE).read_text(encoding="utf-8").strip() == "cu130"


def test_cublas12_check_fails_shows_notice_then_installs_cuda_12_8(m, monkeypatch):
    calls = _phase_stubs(m, monkeypatch, check_ok=False)
    prompts = _answers(monkeypatch, "")
    assert m.cublas12_phase() is None
    text = "\n".join(m._lines)
    assert ("WARNING: This PC could not load the CUDA 12 library (cublas64_12.dll) that Balanced, Fast, Faster and "
            "Kotoba need next to CUDA 13. WhisperJAV will now install its CUDA 12.8 build instead of the standard "
            "CUDA 13 build. Please report this so it can be fixed: https://github.com/meizhong986/WhisperJAV/issues "
            "— attach install_log_v1.10.0.txt. Press Enter to continue.") in text
    assert len(prompts) == 1  # waits for Enter, no time limit
    # The notice comes before the CUDA 12.8 install starts (W42 (4)).
    notice_at = next(i for i, l in enumerate(m._lines) if l.startswith("WARNING: This PC could not load"))
    install_at = next(i for i, l in enumerate(m._lines) if "Installing the CUDA 12.8 build" in l)
    assert notice_at < install_at
    torch_installs = [c for c in calls if c[0] == "pip" and any(a.startswith("torch==") for a in c[1])]
    assert torch_installs and "https://download.pytorch.org/whl/cu128" in torch_installs[0][1]
    # Only the three PyTorch packages are replaced; nothing they depend on (review D3).
    assert "--no-deps" in torch_installs[0][1]
    assert not any(a.startswith("numpy") for a in torch_installs[0][1])
    assert ("uninstall", "nvidia-cublas-cu12") in calls
    assert not list(m._ct2_dir.glob("cublas*64_12.dll"))  # placed copies removed
    assert any(c[0] == "pip" and "ctranslate2==4.8.3" in c[1] and "--no-deps" in c[1] for c in calls)
    assert Path(m.STACK_FILE).read_text(encoding="utf-8").strip() == "cu128"


def test_test_switch_reaches_the_check_process(m, monkeypatch):
    """The installer passes its environment on, so WHISPERJAV_TEST_FORCE_CUBLAS12_FAIL=1 works (S4)."""
    seen = {}

    def fake_run(cmd, **kw):
        seen["env"] = kw.get("env")
        seen["cmd"] = cmd
        return types.SimpleNamespace(returncode=1, stdout="FAILED: failed on purpose\n", stderr="")

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    ok, detail = m.run_cublas12_step("check")
    assert ok is False and detail == "FAILED: failed on purpose"
    assert seen["cmd"][1:] == ["-m", "whisperjav.installer.cublas12", "check"]
    assert seen["env"] is None  # inherits os.environ, including the test switch


# --- review fixes (2026-10-14) ---------------------------------------------------------

def test_a_failed_cublas12_download_stops_and_does_not_fall_back(m, monkeypatch):
    """D2: a download failure is an ordinary failure, not "this PC cannot load cuBLAS 12"."""
    calls = _phase_stubs(m, monkeypatch, check_ok=True)
    monkeypatch.setattr(m, "run_pip", lambda args, desc, **k: calls.append(("pip", list(args))) or False)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no fallback notice for a download failure"))
    reason = m.cublas12_phase()
    assert reason and "nvidia-cublas-cu12" in reason
    assert not Path(m.STACK_FILE).exists()
    assert not any(c[0] == "step" for c in calls)


def test_a_failed_placing_stops_and_does_not_fall_back(m, monkeypatch):
    calls = _phase_stubs(m, monkeypatch, check_ok=True)
    monkeypatch.setattr(m, "run_cublas12_step",
                        lambda step: calls.append(("step", step)) or (False, "FAILED: ctranslate2 is not installed"))
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no fallback notice for a placing failure"))
    reason = m.cublas12_phase()
    assert reason and "placing cuBLAS 12" in reason
    assert ("step", "check") not in calls
    assert not Path(m.STACK_FILE).exists()


@pytest.mark.parametrize("cc_env, drv_env, expect_cc, expect_driver", [
    ("6.1", "", (6, 1), (610, 47, 0)),
    ("", "566.03", (8, 6), (566, 3, 0)),
    ("x", "", (8, 6), (610, 47, 0)),  # malformed switch is ignored
])
def test_switches_make_the_old_card_and_driver_stops_testable(m, monkeypatch, cc_env, drv_env, expect_cc, expect_driver):
    monkeypatch.setenv("WHISPERJAV_TEST_FORCE_COMPUTE_CAP", cc_env)
    monkeypatch.setenv("WHISPERJAV_TEST_FORCE_DRIVER", drv_env)
    info = m.apply_gpu_test_switches(_info(m))
    assert info.compute_capability == expect_cc and info.driver_version == expect_driver
    if cc_env or drv_env:
        assert any("TEST SWITCH" in l for l in m._lines)


# --- install.py (source installer): Linux on CUDA 12.8 (owner, W45) ---------------------

def _load_install_py(monkeypatch, platform_name):
    import importlib.util
    import sys as _sys
    monkeypatch.setattr(_sys, "platform", platform_name)
    path = Path(__file__).parent.parent / "install.py"
    spec = importlib.util.spec_from_file_location("install_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.log = lambda msg: None
    return mod


class _GPU:
    def __init__(self, driver, cc=(8, 6), name="NVIDIA GeForce RTX 3060"):
        self.driver_version, self.compute_capability, self.name = driver, cc, name


@pytest.mark.parametrize("platform_name, default, floor", [("linux", "cu128", 570), ("win32", "cu130", 580)])
def test_install_py_default_build_and_driver_floor_per_platform(monkeypatch, platform_name, default, floor):
    ins = _load_install_py(monkeypatch, platform_name)
    assert ins.DEFAULT_CUDA == default and ins.MIN_DRIVER[0] == floor
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert ins.gpu_support_gate(_GPU((floor, 0, 0))) == "gpu"
    assert ins.gpu_support_gate(_GPU((floor - 1, 99, 0))) == "stop"


def test_install_py_linux_driver_575_installs_cuda_12_8(monkeypatch):
    ins = _load_install_py(monkeypatch, "linux")
    import types as _t

    gpu = _t.SimpleNamespace(detected=True, name="NVIDIA GeForce RTX 3060", driver_version=(575, 0, 0),
                             cuda_version=None, compute_capability=(8, 6), message="")
    monkeypatch.setattr(ins, "detect_gpu", lambda: gpu)
    monkeypatch.setattr(ins, "_INSTALLER_AVAILABLE", True)
    args = _t.SimpleNamespace(cpu_only=False, cuda=None, force_cuda=None)
    assert ins.detect_cuda_version(args) == "cu128"


# --- D1 (owner W46 (1)): a deliberate stop is not a failure ------------------------------

def test_a_deliberate_stop_writes_the_stop_marker_not_the_failure_marker(m, monkeypatch, tmp_path):
    marker = tmp_path / "INSTALLATION_STOPPED_v1.10.0.txt"
    monkeypatch.setattr(m, "STOPPED_MARKER", str(marker))
    m.create_stopped_file("NVIDIA driver 566.03 is older than 580")
    text = marker.read_text(encoding="utf-8")
    assert "was not installed: the installer stopped" in text and "566.03" in text
    source = TEMPLATE.read_text(encoding="utf-8")
    gate_block = source[source.index("gate, gate_reason = gpu_support_gate(driver_info)"):][:300]
    assert "create_stopped_file(gate_reason)" in gate_block and "return 0" in gate_block
    assert "create_failure_file(gate_reason)" not in gate_block


def test_nsis_shows_the_stop_dialog_before_the_failure_dialog():
    nsis = (TEMPLATE.parent / "custom_template.nsi.tmpl.template").read_text(encoding="utf-8")
    stop_at = nsis.index(r'FileExists} "$INSTDIR\INSTALLATION_STOPPED_v{{VERSION}}.txt"')
    fail_at = nsis.index(r'FileExists} "$INSTDIR\INSTALLATION_FAILED_v{{VERSION}}.txt"')
    post_at = nsis.index("_nsis.py\" post_install")
    assert post_at < stop_at < fail_at
