"""The run side of one accelerator per install (design section 22.3, 22.5.4; owner W234, W237 1).

- PyTorch missing: the probe at the top of whisperjav.main, exit status 4, no traceback.
- --device governs every pipeline; --hf-device and --qwen-device are hidden names for it.
- balanced, fast and faster use the GPU only when CTranslate2 itself sees a CUDA device.

No ASR module is imported here (CLAUDE.md "Testing"); the modules are checked by their source.
"""

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from whisperjav.utils import device_detector as dd
from whisperjav.utils import pytorch_probe

REPO = Path(__file__).resolve().parents[1]

BLOCK_TORCH = r"""
import sys, importlib.abc, runpy
class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return None
sys.meta_path.insert(0, _NoTorch())
import importlib.util
_orig = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **k: None if name == "torch" else _orig(name, *a, **k)
sys.argv = ["whisperjav"] + sys.argv[1:]
runpy.run_module("whisperjav.main", run_name="__main__")
"""


# --- PyTorch missing (review finding 1 of W236 I2) -----------------------------------------

@pytest.mark.parametrize("argv", [["--help"], ["--check"], ["some.mp4"]])
def test_without_pytorch_every_command_says_what_to_do_and_exits_4(argv, tmp_path):
    # UTF-8 mode from the start: otherwise whisperjav.main relaunches itself on Windows (#190), and
    # the relaunched process would not have the torch blocker.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", BLOCK_TORCH, *argv], capture_output=True, text=True,
                       encoding="utf-8", cwd=str(tmp_path), env=env, timeout=300)
    assert r.returncode == pytorch_probe.EXIT_PYTORCH_MISSING == 4, r.stderr[-2000:]
    assert "PyTorch, which WhisperJAV needs, is missing or damaged in this install" in r.stderr
    assert "Nothing was processed." in r.stderr
    assert "Traceback" not in r.stderr


def test_own_pytorch_install_without_pytorch_gets_the_command(monkeypatch):
    from whisperjav.installer.core import accelerators
    monkeypatch.setattr(pytorch_probe.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(accelerators, "read_record", lambda prefix=None: (True, "own-pytorch"))
    text = pytorch_probe.pytorch_problem()
    assert text.startswith("This install uses your own PyTorch, and none is installed in WhisperJAV's Python")
    assert f'"{sys.executable}" -m pip install torch torchaudio --index-url' in text


def test_a_damaged_pytorch_is_named_with_its_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def failing(name, *a, **k):
        if name == "torch":
            raise OSError("[WinError 126] fbgemm.dll could not be loaded")
        return real_import(name, *a, **k)
    monkeypatch.setattr(pytorch_probe.importlib.util, "find_spec", lambda name: object())
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", failing)
    text = pytorch_probe.pytorch_problem()
    assert "missing or damaged in this install (OSError: [WinError 126] fbgemm.dll" in text


def test_with_pytorch_the_probe_is_silent():
    assert pytorch_probe.pytorch_problem() == ""


def test_the_probe_runs_before_the_pipeline_imports():
    src = (REPO / "whisperjav" / "main.py").read_text(encoding="utf-8")
    assert src.index("_stop_if_pytorch_missing()") < src.index("from whisperjav.pipelines.faster_pipeline import")


# --- --device governs every pipeline (owner W237 1) -----------------------------------------

@pytest.mark.parametrize("run,setting,expected", [
    ("cpu", "cuda", "cpu"),      # --device cpu is absolute ("Run on the CPU this time")
    ("cpu", None, "cpu"),
    (None, "cuda", "cuda"),      # a pass's own Customize device stands with auto
    ("auto", "cpu", "cpu"),
    ("cuda", None, "cuda"),
    (None, None, "auto"),
    (None, "auto", "auto"),
])
def test_pass_device(run, setting, expected):
    assert dd.pass_device(run, setting) == expected


@pytest.mark.parametrize("device,hf,qwen,expected,notes", [
    (None, "cpu", None, "cpu", 0),       # an old command: --hf-device cpu sets the run's device
    (None, None, "cuda", "cuda", 0),
    ("cpu", "cuda", None, "cpu", 1),     # --device holds, with one line
    ("cuda", "cuda", "cuda", "cuda", 0),
    (None, None, None, None, 0),
])
def test_old_device_options_are_names_for_device(device, hf, qwen, expected, notes):
    result, lines = dd.resolve_run_device(device, hf, qwen)
    assert result == expected and len(lines) == notes
    if notes:
        assert lines[0] == "Note: --hf-device cuda is ignored: --device cpu sets every pipeline's device."


def test_help_shows_device_for_every_pipeline_and_hides_the_old_options():
    r = subprocess.run([sys.executable, "-m", "whisperjav.main", "--help"], capture_output=True, text=True,
                       encoding="utf-8", timeout=300, cwd=str(REPO))
    assert r.returncode == 0
    flat = " ".join(r.stdout.split())
    assert "Where this run's ASR and other models run, for every pipeline." in flat
    for hidden in ("--hf-device", "--qwen-device", "--accept-cpu-mode"):
        assert hidden not in r.stdout


def test_transformers_and_chronosjav_choose_through_the_device_detector():
    for module in ("transformers_asr.py", "qwen_asr.py"):
        src = (REPO / "whisperjav" / "modules" / module).read_text(encoding="utf-8")
        body = src[src.index("def _detect_device"):src.index("def _detect_dtype")]
        auto = body[:body.index('elif self.device_request == "cuda"')]
        assert "get_best_device()" in auto and "torch.cuda.is_available()" not in auto, module


def test_ensemble_passes_take_the_runs_device():
    src = (REPO / "whisperjav" / "ensemble" / "pass_worker.py").read_text(encoding="utf-8")
    assert 'hf_defaults["hf_device"] = pass_device(pass_config.get("device"), hf_defaults.get("hf_device"))' in src
    assert '"device": pass_device(pass_config.get("device"), qwen_defaults.get("qwen_device")),' in src


# --- CTranslate2 decides whether it can use the GPU (design 22.3.3) --------------------------

def _ct2(monkeypatch, best, ct2_count, record="cpu"):
    from whisperjav.installer.core import accelerators
    monkeypatch.setattr(dd, "get_best_device", lambda prefer_cpu=False: best)
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: ct2_count))
    monkeypatch.setattr(accelerators, "read_record", lambda prefix=None: (True, record))


def test_ctranslate2_uses_cuda_when_it_sees_cuda(monkeypatch):
    _ct2(monkeypatch, "cuda", 1, record="cu130")
    assert dd.ctranslate2_device("auto") == "cuda"
    assert dd.ctranslate2_device(None) == "cuda"


def test_rocm_pytorch_does_not_send_ctranslate2_to_the_gpu(monkeypatch, caplog):
    _ct2(monkeypatch, "cuda", 0, record="own-pytorch")
    with caplog.at_level("WARNING", logger="whisperjav"):
        assert dd.ctranslate2_device("auto") == "cpu"
    assert dd.OWN_PYTORCH_CTRANSLATE2_TEXT in caplog.text


def test_the_amd_text_shows_only_on_an_own_pytorch_install(monkeypatch, caplog):
    # Linux ARM64 with an NVIDIA GPU: PyTorch sees CUDA, CTranslate2 does not (review finding 11).
    _ct2(monkeypatch, "cuda", 0, record="cu128")
    with caplog.at_level("WARNING", logger="whisperjav"):
        assert dd.ctranslate2_device("auto") == "cpu"
    assert dd.OWN_PYTORCH_CTRANSLATE2_TEXT not in caplog.text


def test_an_explicit_request_is_passed_as_given(monkeypatch):
    _ct2(monkeypatch, "cuda", 0)
    assert dd.ctranslate2_device("cuda") == "cuda"
    assert dd.ctranslate2_device("cpu") == "cpu"


def test_the_resolver_and_the_modules_use_the_ctranslate2_rule():
    resolver = (REPO / "whisperjav" / "config" / "resolver_v3.py").read_text(encoding="utf-8")
    assert 'CTRANSLATE2_DEVICE_PROVIDERS = {"faster_whisper", "stable_ts"}' in resolver
    assert 'selected_device = ctranslate2_device("auto", best=get_best_device())' in resolver
    fw = (REPO / "whisperjav" / "modules" / "faster_whisper_pro_asr.py").read_text(encoding="utf-8")
    assert 'self.device = ctranslate2_device(model_config.get("device"))' in fw
    st = (REPO / "whisperjav" / "modules" / "stable_ts_asr.py").read_text(encoding="utf-8")
    assert 'ctranslate2_device("auto") if turbo_mode else get_best_device()' in st
