"""install.py: one accelerator per install (design section 22.6; owner W234 D-A, D-E, W238).

install.py is loaded from its file, as in test_installer_new_stack_v110.py. Nothing is installed:
input(), detection and the environment are stand-ins.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(monkeypatch, platform_name="win32"):
    monkeypatch.setattr(sys, "platform", platform_name)
    spec = importlib.util.spec_from_file_location("install_under_test_acc", REPO / "install.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lines = []
    mod.log = lambda msg="": lines.append(str(msg))
    mod._lines = lines
    return mod


def _args(**kw):
    base = dict(accelerator=None, cpu_only=False, cuda=None, force_cuda=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _no_gpu(ins, monkeypatch, record=None):
    monkeypatch.setattr(ins, "_INSTALLER_AVAILABLE", True)
    monkeypatch.setattr(ins, "detect_gpu", lambda: types.SimpleNamespace(
        detected=False, cuda_version=None, name=None, driver_version=None, message=""))
    monkeypatch.setattr(ins.ACC, "read_record", lambda prefix=None: (record is not None, record))
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))


def _answers(monkeypatch, *answers):
    queue = list(answers)
    asked = []

    def fake_input(prompt=""):
        asked.append(prompt)
        if not queue:
            pytest.fail(f"asked once more than expected: {prompt}")
        return queue.pop(0)
    monkeypatch.setattr("builtins.input", fake_input)
    return asked


# --- --accelerator answers in advance; the old options keep working (D-E) ---------------------

@pytest.mark.parametrize("kw,platform_name,expected", [
    (dict(accelerator="cpu"), "win32", "cpu"),
    (dict(cpu_only=True), "linux", "cpu"),
    (dict(accelerator="own-pytorch"), "linux", "own-pytorch"),
    (dict(accelerator="nvidia"), "linux", "cu128"),
    (dict(accelerator="nvidia"), "win32", "cu130"),
    (dict(force_cuda="cu128"), "win32", "cu128"),     # the Optimus case, as before
    (dict(cuda="12.8"), "win32", "cu128"),
])
def test_answered_in_advance(monkeypatch, kw, platform_name, expected):
    ins = _load(monkeypatch, platform_name)
    _no_gpu(ins, monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("answered in advance: no question"))
    assert ins.choose_accelerator(_args(**kw)) == expected


def test_vulkan_on_windows_checks_for_the_vulkan_runtime(monkeypatch):
    ins = _load(monkeypatch, "win32")
    _no_gpu(ins, monkeypatch)
    monkeypatch.setattr(ins, "_vulkan_runtime_present", lambda: True)
    assert ins.choose_accelerator(_args(accelerator="vulkan")) == "vulkan"
    monkeypatch.setattr(ins, "_vulkan_runtime_present", lambda: False)
    assert ins.choose_accelerator(_args(accelerator="vulkan")) == "cpu"
    assert ins.ACC.NO_VULKAN_TEXT in " ".join(ins._lines)


@pytest.mark.parametrize("platform_name,system", [("linux", "Linux"), ("darwin", "macOS")])
def test_vulkan_outside_windows_is_refused(monkeypatch, platform_name, system):
    ins = _load(monkeypatch, platform_name)
    _no_gpu(ins, monkeypatch)
    assert ins.choose_accelerator(_args(accelerator="vulkan")) == "stop"
    assert ins.ACC.VULKAN_NOT_HERE_TEXT.format(system=system) in " ".join(ins._lines)


@pytest.mark.parametrize("kw", [dict(accelerator="cpu"), dict(cpu_only=True)])
def test_cpu_on_an_apple_silicon_mac_gives_the_apple_install(monkeypatch, kw):
    # The Mac guide tells users install.py --cpu-only (review finding 6; 22.13 detail 3).
    ins = _load(monkeypatch, "darwin")
    _no_gpu(ins, monkeypatch)
    assert ins.choose_accelerator(_args(**kw)) == "metal"
    assert ins.ACC.MAC_CPU_TEXT in " ".join(ins._lines)


# --- the record, the automatic choices, the question -----------------------------------------

def test_a_second_run_takes_the_record(monkeypatch):
    ins = _load(monkeypatch, "win32")
    _no_gpu(ins, monkeypatch, record="vulkan")
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("the record is the answer"))
    assert ins.choose_accelerator(_args()) == "vulkan"


def test_apple_silicon_is_automatic(monkeypatch):
    ins = _load(monkeypatch, "darwin")
    _no_gpu(ins, monkeypatch)
    monkeypatch.setattr(ins.platform_module, "machine", lambda: "arm64")
    assert ins.choose_accelerator(_args()) == "metal"


@pytest.mark.parametrize("answer,expected", [("1", "vulkan"), ("2", "vulkan"), ("3", "vulkan"),
                                             ("4", "own-pytorch"), ("5", "cpu"), ("6", "stop")])
def test_windows_without_a_supported_nvidia_gpu_asks_the_windows_question(monkeypatch, answer, expected):
    ins = _load(monkeypatch, "win32")
    _no_gpu(ins, monkeypatch)
    monkeypatch.setattr(ins, "_vulkan_runtime_present", lambda: True)
    asked = _answers(monkeypatch, answer)
    assert ins.choose_accelerator(_args()) == expected
    assert asked[0].startswith("Type 1 = AMD, 2 = Intel, 3 = older NVIDIA card, 4 = my own PyTorch")


@pytest.mark.parametrize("answer,expected", [("1", "own-pytorch"), ("2", "cpu"), ("3", "stop")])
def test_linux_without_a_supported_nvidia_gpu_asks_the_linux_question(monkeypatch, answer, expected):
    ins = _load(monkeypatch, "linux")
    _no_gpu(ins, monkeypatch)
    asked = _answers(monkeypatch, "9", answer)   # an invalid answer is asked again, no time limit
    assert ins.choose_accelerator(_args()) == expected
    assert asked[0].startswith("No supported NVIDIA GPU found. What should this install be made for?")


def test_no_console_installs_cpu_only_without_an_nvidia_gpu(monkeypatch):
    ins = _load(monkeypatch, "linux")
    _no_gpu(ins, monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(EOFError()))
    assert ins.choose_accelerator(_args()) == "cpu"


def test_an_old_card_offers_the_accelerators_and_stops_without_a_console(monkeypatch):
    ins = _load(monkeypatch, "win32")
    old = types.SimpleNamespace(name="NVIDIA GeForce GTX 1060", compute_capability=(6, 1), driver_version=(580, 0, 0))
    asked = _answers(monkeypatch, "7")
    assert ins.gpu_support_gate(old) == "stop"
    joined = " ".join(ins._lines)
    assert "is older than WhisperJAV 1.10 supports" in joined and "1.9.5 still supports this card" in joined
    assert "7 = my card is identified wrongly" in asked[0]
    assert ins.ACC.REPORT_IT_TEXT in joined
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(EOFError()))
    assert ins.gpu_support_gate(old) == "stop"


# --- the lock extra and Own PyTorch's uv sync ------------------------------------------------

def _sync_cmd(ins, monkeypatch, accelerator, target_env=None, is_external=False):
    import subprocess
    seen = {}

    class P:
        stdout = iter(())
        returncode = 0

        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **k: seen.setdefault("cmd", cmd) and P())
    assert ins.run_uv_sync(["cli"], accelerator, target_env=target_env, is_external=is_external)
    return seen["cmd"]


@pytest.mark.parametrize("platform_name,accelerator,extra", [
    ("win32", "vulkan", "vulkan"), ("win32", "own-pytorch", "vulkan"), ("linux", "own-pytorch", "cpu"),
    ("darwin", "metal", "cpu"), ("win32", "cu130", "cu130"), ("linux", "cpu", "cpu"),
])
def test_uv_sync_takes_the_accelerators_lock_extra(monkeypatch, platform_name, accelerator, extra):
    ins = _load(monkeypatch, platform_name)
    cmd = _sync_cmd(ins, monkeypatch, accelerator, target_env="X", is_external=True)
    assert cmd[cmd.index("cli") + 2] == extra or ["--extra", extra] == cmd[cmd.index("cli") + 1:cmd.index("cli") + 3]


def test_own_pytorch_never_installs_pytorch(monkeypatch):
    ins = _load(monkeypatch, "linux")
    cmd = _sync_cmd(ins, monkeypatch, "own-pytorch", target_env="X", is_external=True)
    joined = " ".join(cmd)
    assert "--inexact" in cmd
    assert "--no-install-package torch --no-install-package torchaudio" in joined


def test_own_pytorch_without_pytorch_in_the_environment_is_refused(monkeypatch):
    ins = _load(monkeypatch, "linux")
    assert ins.own_pytorch_version(None) == ""    # no active environment: uv would make a new .venv
    monkeypatch.setattr(ins, "_run_in_env", lambda python, *a: (False, "No module named 'torch'"))
    assert ins.own_pytorch_version("/env") == ""
    monkeypatch.setattr(ins, "_run_in_env", lambda python, *a: (True, "2.9.1+rocm6.3"))
    assert ins.own_pytorch_version("/env") == "2.9.1+rocm6.3"


def test_unsupported_computers_are_refused_before_preflight():
    src = (REPO / "install.py").read_text(encoding="utf-8")
    body = src[src.index("def main():"):]
    assert body.index("refusal = SUPPORTED.refusal()") < body.index("run_preflight_checks()")
    assert body.index("refusal = SUPPORTED.refusal()") < body.index("run_uv_sync(")


def test_help_shows_accelerator_and_hides_the_old_options():
    import subprocess
    # The child writes UTF-8 whatever the console (Git Bash gives cp1252; review P7 finding 16).
    r = subprocess.run([sys.executable, str(REPO / "install.py"), "--help"], capture_output=True, text=True,
                       encoding="utf-8", timeout=120, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0
    usage = r.stdout.split("WhisperJAV Source Installation Script —")[0]
    assert "--accelerator {nvidia,vulkan,own-pytorch,cpu}" in usage
    for hidden in ("--cpu-only", "--cuda ", "--force-cuda"):
        assert hidden not in usage


def test_the_shell_scripts_check_the_computer_first():
    linux = (REPO / "installer" / "install_linux.sh").read_text(encoding="utf-8")
    assert linux.index("whisperjav/installer/core/supported.py") < linux.index("check_venv_requirement\n")
    mac = (REPO / "installer" / "install_mac.sh").read_text(encoding="utf-8")
    main = mac[mac.index("# Main Execution"):]
    assert main.index("check_supported_mac") < main.index("check_xcode_cli_tools")
    from whisperjav.installer.core import supported
    assert supported.INTEL_MAC_TEXT in mac
    assert supported.OLD_MACOS_TEXT.split("{version}")[1] in mac
    colab = (REPO / "installer" / "install_colab.sh").read_text(encoding="utf-8")
    assert supported.NOTEBOOK_TEXT in colab
    assert colab.index(supported.NOTEBOOK_TEXT) < colab.index("pip install") if "pip install" in colab else True
