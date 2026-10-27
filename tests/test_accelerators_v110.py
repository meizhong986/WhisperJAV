"""The accelerator table and the supported-computers check (design section 22.1.2, 22.6.8; owner W234, W238).

Standard library only on the module side; nothing here imports torch.
"""

import importlib.util
from pathlib import Path

import pytest

from whisperjav.installer.core import accelerators as acc
from whisperjav.installer.core import supported

REPO = Path(__file__).resolve().parents[1]


# --- the accelerator table ---------------------------------------------------------

def test_the_five_accelerators_and_their_record_values():
    assert acc.VALUES == ("cu130", "cu128", "vulkan", "metal", "own-pytorch", "cpu")
    names = {a.value: a.name for a in acc.ACCELERATORS}
    assert names == {
        "cu130": "NVIDIA GPU (CUDA 13.0)",
        "cu128": "NVIDIA GPU (CUDA 12.8)",
        "vulkan": "GPU through Vulkan (AMD, Intel or older NVIDIA card)",
        "metal": "Apple GPU (Metal)",
        "own-pytorch": "Own PyTorch",
        "cpu": "CPU only (not accelerated)",
    }


def test_lock_extras_are_the_pyproject_extras():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    for extra in acc.LOCK_EXTRAS:
        assert f"\n{extra} = [" in text
    assert '[{ extra = "cu130" }, { extra = "cu128" }, { extra = "vulkan" }, { extra = "cpu" }]' in text


@pytest.mark.parametrize("value,platform,extra", [
    ("cu130", "win32", "cu130"), ("cu128", "linux", "cu128"), ("vulkan", "win32", "vulkan"),
    ("metal", "darwin", "cpu"), ("cpu", "linux", "cpu"),
    ("own-pytorch", "win32", "vulkan"), ("own-pytorch", "linux", "cpu"), ("own-pytorch", "darwin", "cpu"),
])
def test_lock_extra_per_platform(value, platform, extra):
    assert acc.lock_extra(value, platform) == extra
    assert acc.requirements_file(value, platform) == f"requirements_{extra}.txt"


def test_only_own_pytorch_installs_no_pytorch_and_only_cuda_is_nvidia():
    assert [a.value for a in acc.ACCELERATORS if not acc.installs_pytorch(a.value)] == ["own-pytorch"]
    assert [a.value for a in acc.ACCELERATORS if a.nvidia] == ["cu130", "cu128"]


def test_record_round_trip_unknown_and_missing(tmp_path):
    assert acc.read_record(str(tmp_path)) == (False, None)
    for value in acc.VALUES:
        acc.write_record(value, str(tmp_path))
        assert acc.read_record(str(tmp_path)) == (True, value)
    (tmp_path / acc.STACK_FILE).write_text("cu118\n", encoding="utf-8")
    assert acc.read_record(str(tmp_path)) == (True, None)
    with pytest.raises(ValueError):
        acc.write_record("cu118", str(tmp_path))


def test_cublas12_reads_and_writes_through_the_table(tmp_path):
    from whisperjav.installer import cublas12
    assert not hasattr(cublas12, "STACKS")
    cublas12.write_stack("vulkan", str(tmp_path))
    assert cublas12.read_stack(str(tmp_path)) == "vulkan"
    cublas12.write_stack("own-pytorch", str(tmp_path))
    assert cublas12.read_stack(str(tmp_path)) == "own-pytorch"


def test_every_locked_file_holds_one_crispasr_line_per_platform():
    locked = REPO / "whisperjav" / "installer" / "locked"
    expected = {
        "cu130": ["aarch64", "x86_64", "AMD64"],
        "cu128": ["aarch64", "x86_64", "AMD64"],
        "vulkan": ["AMD64"],
        "cpu": ["one line, four platforms"],
    }
    for extra in acc.LOCK_EXTRAS:
        lines = [l for l in (locked / f"requirements_{extra}.txt").read_text(encoding="utf-8").splitlines()
                 if l.startswith("crispasr")]
        assert len(lines) == len(expected[extra]), (extra, lines)
    vulkan = (locked / "requirements_vulkan.txt").read_text(encoding="utf-8")
    assert "crispasr-0.8.44%2Bvulkan-py3-none-win_amd64.whl" in vulkan and "+cpu" not in vulkan.split("torch==")[0]


# --- the supported-computers check -------------------------------------------------

def _refusal(**kw):
    base = dict(notebook=False, windows_arm=False, rosetta=False)
    base.update(kw)
    return supported.refusal(**base)


@pytest.mark.parametrize("kw,text_start", [
    (dict(system="linux", machine="x86_64", glibc="2.31"), "WhisperJAV 1.10.0 does not support this Linux. This is due to its system library glibc 2.31"),
    (dict(system="linux", machine="x86_64", glibc="2.34"), "WhisperJAV 1.10.0 does not support this Linux. This is due to its system library glibc 2.34"),
    (dict(system="linux", machine="aarch64", glibc=""), "WhisperJAV 1.10.0 does not support this Linux. This is due to its system library: it has no glibc"),
    (dict(system="linux", machine="armv7l", glibc="2.39"), "WhisperJAV 1.10.0 does not support this computer (Linux, armv7l)"),
    (dict(system="darwin", machine="x86_64"), "WhisperJAV 1.10.0 does not support Intel Macs."),
    (dict(system="darwin", machine="x86_64", rosetta=True), "WhisperJAV 1.10.0 does not support this Python on an Apple silicon Mac."),
    (dict(system="darwin", machine="arm64", mac_version="13.6.1"), "WhisperJAV 1.10.0 does not support macOS 13.6.1."),
    (dict(system="win32", machine="AMD64", windows_major=10, windows_arm=True), "WhisperJAV 1.10.0 does not support Windows on ARM."),
    (dict(system="win32", machine="AMD64", windows_major=6), "WhisperJAV 1.10.0 does not support this version of Windows."),
    (dict(system="linux", machine="x86_64", glibc="2.39", notebook=True), "WhisperJAV 1.10.0 does not support Google Colab or Kaggle notebooks."),
])
def test_refused_computers_in_the_owners_form(kw, text_start):
    text = _refusal(**kw)
    assert text.startswith(text_start)
    assert "This is due to" in text and text.endswith("Nothing was installed.")


@pytest.mark.parametrize("kw", [
    dict(system="linux", machine="x86_64", glibc="2.35"),
    dict(system="linux", machine="aarch64", glibc="2.36"),
    dict(system="darwin", machine="arm64", mac_version="14.0"),
    dict(system="darwin", machine="arm64", mac_version="15.3"),
    dict(system="win32", machine="AMD64", windows_major=10),
])
def test_supported_computers_pass(kw):
    assert _refusal(**kw) == ""


def test_this_pc_is_supported_and_the_file_runs_alone():
    assert supported.refusal() == ""
    lines = [l.strip() for l in Path(supported.__file__).read_text(encoding="utf-8").splitlines()]
    imports = [l for l in lines if l.startswith(("import ", "from "))]
    assert imports and not [l for l in imports if "whisperjav" in l or "__future__" in l]
    # Python 3.6 syntax: the shell scripts run it with any python3


def test_the_module_does_not_need_the_package():
    spec = importlib.util.spec_from_file_location("supported_alone", supported.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.refusal(system="win32", machine="AMD64", windows_major=10, windows_arm=False,
                          notebook=False) == ""
