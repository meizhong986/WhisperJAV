"""v1.10.0 item 2 (new stack), the Windows installer's decisions -- spec checks S5 and part of S4.

What the install is made for (design 22.6.2): an NVIDIA GPU 1.10 cannot use (old card, old
driver, not reachable) -> the window with the reason first and no time limit (W37 (2), D-C); no
NVIDIA card -> the window, CPU only after 20 s (R3); a second run -> the recorded accelerator;
the cuBLAS 12 phase keeps CUDA 13.0 when the load check passes and moves this PC to the CUDA
12.8 build, with the owner's notice, when it fails (W37 (3), W42 (4), W43).

The template is loaded as in test_installer_torch_cache_438.py. Nothing is installed: input(),
the installer commands and the fresh-process steps are stand-ins.
"""
import re
import types
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).parent.parent / "installer" / "templates" / "post_install.py.template"
# v1.10.0 (item 2f): the plans read PyTorch's lines from the shipped locked files.
LOCKED = Path(__file__).parent.parent / "whisperjav" / "installer" / "locked"


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
    monkeypatch.setattr(mod, "locked_requirements_path", lambda v: str(LOCKED / f"requirements_{v}.txt"))
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


# --- E, F and the window (design 22.6.2; owner W234 D-C, R3): what the install is made for -----

def _plan(m, info, adapters=(), recorded=None):
    case, lines = m.nvidia_case(info, list(adapters))
    return case, m.window_plan(case, lines, list(adapters), recorded)


@pytest.mark.parametrize("cc", [(6, 1), (7, 0)])
def test_old_card_shows_the_owners_lines_first_and_waits_without_a_time_limit(m, cc):
    case, plan = _plan(m, _info(m, cc=cc, name="NVIDIA GeForce GTX 1080"))
    assert case == "old_card"
    assert plan["lines"][0] == (f"Your graphics card, NVIDIA GeForce GTX 1080 (compute capability {cc[0]}.{cc[1]}), "
                                "is older than WhisperJAV 1.10 supports. 1.10 needs a GTX 16-series, RTX 20-series "
                                "or newer card.")
    assert plan["lines"][1] == ("WhisperJAV 1.9.5 still supports this card: "
                                "https://github.com/meizhong986/WhisperJAV/releases/tag/v1.9.5.")
    assert plan["timeout"] is None and plan["preselected"] == "stop" and plan["wrongly"] is True


def test_old_driver_gets_the_driver_line_first_and_waits(m):
    case, plan = _plan(m, _info(m, driver=(566, 3, 0)))
    assert case == "old_driver"
    assert plan["lines"][0].startswith("Your NVIDIA driver (566.03) is older than WhisperJAV 1.10 needs (580 or newer).")
    assert "Press Enter" not in plan["lines"][0]       # 22.13 detail 9: no "Press Enter" in a window
    assert plan["timeout"] is None and plan["preselected"] == "stop" and not plan["wrongly"]


def test_driver_579_99_is_old_and_580_is_supported(m):
    assert m.nvidia_case(_info(m, driver=(579, 99, 0)), [])[0] == "old_driver"
    assert m.nvidia_case(_info(m, driver=(580, 0, 0)), [])[0] == "supported"


def test_old_card_is_found_before_the_driver(m):
    assert m.nvidia_case(_info(m, driver=(470, 0, 0), cc=(6, 1)), [])[0] == "old_card"


def test_card_7_5_and_unreadable_capability_are_supported_with_no_window(m):
    for info in (_info(m, cc=(7, 5), name="NVIDIA GeForce GTX 1660"), _info(m, cc=None)):
        case, plan = _plan(m, info)
        assert case == "supported" and plan is None


def test_no_nvidia_card_gets_the_window_with_cpu_only_after_20_seconds(m):
    case, plan = _plan(m, m.DriverInfo(None, None, "none"), adapters=["AMD Radeon RX 6600"])
    assert case == "none"
    assert plan["title"] == "WhisperJAV — no supported NVIDIA GPU found"
    assert plan["lines"][0] == ("WhisperJAV found no NVIDIA graphics card that version 1.10 supports (graphics "
                                "found: AMD Radeon RX 6600). What should this install be made for?")
    assert [label for _k, label in plan["choices"]] == [
        "AMD graphics card", "Intel graphics card", "NVIDIA graphics card older than WhisperJAV 1.10 supports",
        "I have my own PyTorch for my GPU (advanced)", "None, or use the CPU only"]
    # the pre-selection is what the timeout installs (review finding 8 of W236 I2)
    assert plan["preselected"] == "5" and plan["timeout"] == 20
    assert plan["footer"] == "Without an answer, the CPU-only version is installed in 20 seconds."


def test_an_nvidia_card_windows_lists_but_the_installer_cannot_reach(m):
    case, plan = _plan(m, m.DriverInfo(None, None, "none"), adapters=["NVIDIA GeForce RTX 4060 Laptop GPU", "Intel UHD"])
    assert case == "not_reachable"
    assert plan["lines"][0].startswith("Windows lists an NVIDIA graphics card (NVIDIA GeForce RTX 4060 Laptop GPU), "
                                       "but WhisperJAV cannot reach it")
    assert "the NVIDIA chip may be switched off" in plan["lines"][0]       # the Optimus case (#200)
    assert plan["timeout"] is None and plan["preselected"] == "stop"


@pytest.mark.parametrize("recorded,adapters,key", [
    ("vulkan", ["Intel Arc A770"], "2"), ("vulkan", ["AMD Radeon"], "1"), ("own-pytorch", [], "4"),
    ("cpu", [], "5"),
])
def test_a_second_run_keeps_the_recorded_accelerator_by_default(m, recorded, adapters, key):
    case, plan = _plan(m, m.DriverInfo(None, None, "none"), adapters, recorded)
    assert plan["title"] == "WhisperJAV — this install's accelerator"
    assert plan["preselected"] == key and plan["timeout"] == 20
    assert m.accelerator_for_answer(plan["preselected"], vulkan_present=True, plan=plan) == recorded


@pytest.mark.parametrize("info,adapters,recorded,key", [
    (dict(cc=(6, 1), name="NVIDIA GeForce GTX 1080"), ["NVIDIA GeForce GTX 1080"], "vulkan", "3"),
    (dict(cc=(6, 1), name="NVIDIA GeForce GTX 1080"), [], "own-pytorch", "4"),
    (dict(driver=(566, 3, 0)), [], "cpu", "5"),
    (None, ["NVIDIA GeForce RTX 4060 Laptop GPU", "Intel UHD"], "vulkan", "2"),   # 1, 2, 3 all give vulkan
])
def test_a_second_run_on_an_unusable_nvidia_gpu_still_keeps_the_record(m, info, adapters, recorded, key):
    # Review P7 finding 1: the record comes before the unusable-GPU case (22.6.6, detail 10).
    driver = _info(m, **info) if info is not None else m.DriverInfo(None, None, "none")
    case, plan = _plan(m, driver, adapters, recorded)
    assert case in ("old_card", "old_driver", "not_reachable")
    assert plan["title"] == "WhisperJAV — this install's accelerator"
    assert plan["preselected"] == key and plan["timeout"] == 20
    assert plan["lines"][-1].startswith("This install is made for: ")
    assert len(plan["lines"]) > 1                      # the reason stays above T25r
    assert m.accelerator_for_answer(plan["preselected"], vulkan_present=True, plan=plan) == recorded


@pytest.mark.parametrize("info,adapters", [
    (dict(driver=(566, 3, 0)), []), (None, ["NVIDIA GeForce RTX 4060 Laptop GPU"]),
])
def test_old_driver_and_not_reachable_do_not_say_the_card_is_unsupported(m, info, adapters):
    # Review P7 finding 8: the card may be supported; only T25's question follows the reason.
    driver = _info(m, **info) if info is not None else m.DriverInfo(None, None, "none")
    _case, plan = _plan(m, driver, adapters)
    assert plan["lines"][-1] == "What should this install be made for?"
    assert not any("found no NVIDIA graphics card" in l for l in plan["lines"])


def test_a_second_run_with_a_supported_nvidia_gpu_offers_it_and_keeps_it(m):
    case, plan = _plan(m, _info(m, driver=(580, 0, 0)), recorded="cu130")
    assert plan["choices"][0] == ("nvidia", "NVIDIA GPU (CUDA)") and plan["preselected"] == "nvidia"
    assert m.accelerator_for_answer("nvidia", True, plan) == "cu130"
    case, plan = _plan(m, _info(m, driver=(580, 0, 0)), recorded="own-pytorch")
    assert plan["preselected"] == "4"   # an Own PyTorch install is not switched to CUDA (review finding 2)


@pytest.mark.parametrize("answer,vulkan,expected", [
    ("1", True, "vulkan"), ("2", True, "vulkan"), ("3", True, "vulkan"), ("1", False, "cpu"),
    ("4", True, "own-pytorch"), ("5", True, "cpu"), ("stop", True, "stop"), ("wrongly", True, "wrongly"),
])
def test_answers_give_the_accelerator(m, answer, vulkan, expected):
    assert m.accelerator_for_answer(answer, vulkan) == expected
    if answer == "1" and not vulkan:
        assert any("Vulkan (vulkan-1.dll) was not found" in l for l in m._lines)


def test_choosing_stop_or_identified_wrongly_is_a_stop_on_purpose(m, monkeypatch):
    monkeypatch.setattr(m, "windows_display_adapters", lambda: [])
    for answer in ("stop", "wrongly"):
        m._stopped_on_purpose = False
        monkeypatch.setattr(m, "ask_accelerator", lambda plan, a=answer: a)
        accelerator, reason = m.choose_install_accelerator(_info(m, cc=(6, 1)))
        assert accelerator == "stop" and m._stopped_on_purpose is True
    assert any("https://github.com/meizhong986/WhisperJAV/issues" in l and "report" in l for l in m._lines)


def test_a_supported_nvidia_gpu_on_a_first_install_asks_nothing(m, monkeypatch):
    monkeypatch.setattr(m, "windows_display_adapters", lambda: ["NVIDIA GeForce RTX 3060"])
    monkeypatch.setattr(m, "ask_accelerator", lambda plan: pytest.fail("no window for a supported NVIDIA GPU"))
    assert m.choose_install_accelerator(_info(m, driver=(580, 0, 0))) == ("cu130", None)


def test_no_window_means_the_console_form_with_the_same_default(m, monkeypatch):
    monkeypatch.setattr(m, "_ask_in_window", lambda plan: (_ for _ in ()).throw(RuntimeError("no display")))
    monkeypatch.setattr(m, "timed_input", lambda prompt, timeout, default: default)
    case, plan = _plan(m, m.DriverInfo(None, None, "none"))
    assert m.ask_accelerator(plan) == "5"
    case, plan = _plan(m, _info(m, cc=(6, 1)))
    _answers(monkeypatch, "x", "7")      # junk is asked again; no time limit
    assert m.ask_accelerator(plan) == "wrongly"


def test_cpu_and_vulkan_get_pytorchs_cpu_build_and_own_pytorch_none(m):
    plan = m.select_torch_install_plan(_info(m, cc=(6, 1)), force_cpu=True)
    assert plan.uses_gpu is False and plan.variant == "cpu"
    assert plan.pip_args[:2] == ["install", "--no-deps"]
    assert any(a.startswith("torch==2.11.0+cpu") for a in plan.pip_args)
    assert plan.pip_args[-2:] == ["--index-url", "https://download.pytorch.org/whl/cpu"]
    vulkan, error = m.plan_pytorch(m.DriverInfo(None, None, "none"), accelerator="vulkan")
    assert error is None and vulkan.variant == "vulkan"
    assert any(a.startswith("torch==2.11.0+cpu") for a in vulkan.pip_args)
    assert vulkan.pip_args[-2:] == ["--index-url", "https://download.pytorch.org/whl/cpu"]
    assert m.plan_pytorch(accelerator="own-pytorch") == (None, None)


def test_driver_580_gets_pytorch_2_11_0_for_cuda_13(m, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no question for driver 580"))
    plan = m.select_torch_install_plan(_info(m, driver=(580, 0, 0)))
    assert plan.uses_gpu is True and plan.target_label == "CUDA 13.0"
    # v1.10.0 (item 2f): the PyTorch lines of the cu130 locked file, --no-deps; numpy
    # and the rest come from the locked file in phase 4. Item 3 (owner W78 C2): torch
    # and torchaudio only; torchvision left with ClearVoice.
    assert plan.variant == "cu130"
    assert plan.pip_args[:2] == ["install", "--no-deps"]
    assert plan.pip_args[-2:] == ["--index-url", "https://download.pytorch.org/whl/cu130"]
    names = {a.split("=")[0] for a in plan.pip_args[2:-2]}
    assert names == {"torch", "torchaudio"}
    assert "torch==2.11.0+cu130" in plan.pip_args and "torchaudio==2.11.0+cu130" in plan.pip_args


def test_the_record_is_the_accelerator_chosen(m):
    m.write_stack_file("vulkan")
    assert Path(m.STACK_FILE).read_text(encoding="utf-8").strip() == "vulkan"
    assert m.read_stack_file() == "vulkan"
    source = TEMPLATE.read_text(encoding="utf-8")
    phase = source[source.index("# === Phase 5.1"):source.index("# === Phase 5.5")]
    assert "if not nvidia_install:\n        write_stack_file(accelerator)" in phase


def test_own_pytorch_skips_pytorch_and_shows_the_command(m, tmp_path, monkeypatch):
    source = TEMPLATE.read_text(encoding="utf-8")
    assert "pytorch_error = install_pytorch(driver_info, plan=torch_plan) if torch_plan else None" in source
    monkeypatch.setattr(m.sys, "prefix", str(tmp_path))
    text = m.own_pytorch_next_step_text()
    assert text.startswith("WhisperJAV is installed without PyTorch, as you chose.")
    assert f'"{tmp_path}\\python.exe" -m pip install torch torchaudio --index-url' in text
    assert "OWN_PYTORCH_NEXT_STEP.txt" in text and "WhisperJAV will never replace it." in text


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
                                 "cuda": "12.8", "cuda_available": True})
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
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no question for a driver at the floor"))
    assert ins.gpu_support_gate(_GPU((floor, 0, 0))) == "gpu"
    # Below the floor (design 22.6.4): the driver line first, then the one question; "stop" stops,
    # "CPU only" installs for the CPU.
    stop, cpu = ("6", "5") if platform_name == "win32" else ("3", "2")
    monkeypatch.setattr("builtins.input", lambda *a: stop)
    assert ins.gpu_support_gate(_GPU((floor - 1, 99, 0))) == "stop"
    monkeypatch.setattr("builtins.input", lambda *a: cpu)
    assert ins.gpu_support_gate(_GPU((floor - 1, 99, 0))) == "cpu"


def test_install_py_linux_driver_575_installs_cuda_12_8(monkeypatch):
    ins = _load_install_py(monkeypatch, "linux")
    import types as _t

    gpu = _t.SimpleNamespace(detected=True, name="NVIDIA GeForce RTX 3060", driver_version=(575, 0, 0),
                             cuda_version=None, compute_capability=(8, 6), message="")
    monkeypatch.setattr(ins, "detect_gpu", lambda: gpu)
    monkeypatch.setattr(ins, "_INSTALLER_AVAILABLE", True)
    monkeypatch.setattr(ins.ACC, "read_record", lambda prefix=None: (False, None))
    args = _t.SimpleNamespace(accelerator=None, cpu_only=False, cuda=None, force_cuda=None)
    assert ins.choose_accelerator(args) == "cu128"


# --- D1 (owner W46 (1)): a deliberate stop is not a failure ------------------------------

def test_a_deliberate_stop_writes_the_stop_marker_not_the_failure_marker(m, monkeypatch, tmp_path):
    marker = tmp_path / "INSTALLATION_STOPPED_v1.10.0.txt"
    monkeypatch.setattr(m, "STOPPED_MARKER", str(marker))
    m.create_stopped_file("NVIDIA driver 566.03 is older than 580")
    text = marker.read_text(encoding="utf-8")
    assert "was not installed: the installer stopped" in text and "566.03" in text
    source = TEMPLATE.read_text(encoding="utf-8")
    gate_block = source[source.index("accelerator, stop_reason = choose_install_accelerator(driver_info)"):][:300]
    assert "create_stopped_file(stop_reason)" in gate_block and "return 0" in gate_block
    assert "create_failure_file(stop_reason)" not in gate_block


def test_nsis_shows_the_stop_dialog_before_the_failure_dialog():
    nsis = (TEMPLATE.parent / "custom_template.nsi.tmpl.template").read_text(encoding="utf-8")
    stop_at = nsis.index(r'FileExists} "$INSTDIR\INSTALLATION_STOPPED_v{{VERSION}}.txt"')
    fail_at = nsis.index(r'FileExists} "$INSTDIR\INSTALLATION_FAILED_v{{VERSION}}.txt"')
    post_at = nsis.index(r'start "" /wait "$CMD_EXE" /D /C ""$INSTDIR\pkgs\post_install.bat""')
    assert post_at < stop_at < fail_at


def test_replacing_an_install_keeps_its_record_for_the_second_run_window():
    # Owner W242 1 (design 22.6.6): the NSIS replace removes the whole folder; the record is kept
    # aside before that and put back before post_install reads it.
    nsi = (Path(__file__).resolve().parents[1] / "installer" / "templates" /
           "custom_template.nsi.tmpl.template").read_text(encoding="utf-8")
    remove = nsi[nsi.index("Function RemoveExistingWhisperJAV"):]
    remove = remove[:remove.index("FunctionEnd")]
    keep = remove.index('CopyFiles /SILENT "$INSTDIR\\whisperjav_stack.txt" "$PLUGINSDIR\\whisperjav_stack_kept.txt"')
    assert keep < remove.index("Uninstall-WhisperJAV.exe\" /S") < remove.index('RMDir /r "$INSTDIR"')
    put_back = nsi.index('CopyFiles /SILENT "$PLUGINSDIR\\whisperjav_stack_kept.txt" "$INSTDIR\\whisperjav_stack.txt"')
    assert put_back < nsi.index('${If} ${FileExists} "$INSTDIR\\pkgs\\post_install.bat"')
    assert nsi.index("InitPluginsDir") < nsi.index("Function RemoveExistingWhisperJAV")
