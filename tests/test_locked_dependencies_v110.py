"""v1.10.0, item 2f: one locked dependency list (uv.lock) for development, source installs, the installer
and upgrades (owner W53-W56). Spec: docs/requirements/v1.10.x/specs/2026-10-14-item02f-locked-dependencies.md.

Nothing here installs anything or touches a real environment: paths are tmp_path or the tracked locked files,
subprocess is replaced by stand-ins, and source-install detection is patched (lesson of 2026-10-14).
"""
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
TEMPLATE = ROOT / "installer" / "templates" / "post_install.py.template"
LOCKED = ROOT / "whisperjav" / "installer" / "locked"
VARIANTS = ("cu130", "cu128", "cpu")


def _load_post_install():
    source = re.sub(r"\{\{[A-Z_]+\}\}", "1.10.0", TEMPLATE.read_text(encoding="utf-8"))
    module = types.ModuleType("post_install_under_test")
    module.__file__ = str(TEMPLATE)
    exec(compile(source, str(TEMPLATE), "exec"), module.__dict__)
    return module


@pytest.fixture
def pi(tmp_path, monkeypatch):
    mod = _load_post_install()
    lines = []
    monkeypatch.setattr(mod, "LOG_FILE", str(tmp_path / "log.txt"))
    monkeypatch.setattr(mod, "log", lambda msg: lines.append(str(msg)))
    monkeypatch.setattr(mod, "log_section", lambda msg: lines.append(str(msg)))
    monkeypatch.setattr(mod, "locked_requirements_path", lambda v: str(LOCKED / f"requirements_{v}.txt"))
    mod._lines = lines
    return mod


def _conda_meta(prefix: Path, name: str, version: str, files):
    meta = prefix / "conda-meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / f"{name}-{version}-0.json").write_text(
        json.dumps({"name": name, "version": version, "files": files}), encoding="utf-8")


def _conda_python_package(prefix: Path, conda_name: str, conda_version: str, pypi_name: str, pypi_version: str,
                          egg_info: bool = False):
    """A conda package with its Python package's metadata folder, as conda lays it out."""
    if egg_info:
        folder = f"Lib/site-packages/{pypi_name}-{pypi_version}-py3.13.egg-info"
        meta_file, meta_name = f"{folder}/PKG-INFO", "PKG-INFO"
    else:
        folder = f"Lib/site-packages/{pypi_name.replace('-', '_')}-{pypi_version}.dist-info"
        meta_file, meta_name = f"{folder}/METADATA", "METADATA"
    (prefix / folder).mkdir(parents=True, exist_ok=True)
    (prefix / meta_file).write_text(f"Metadata-Version: 2.1\nName: {pypi_name}\nVersion: {pypi_version}\n\nbody\n",
                                    encoding="utf-8")
    _conda_meta(prefix, conda_name, conda_version,
                [f"Lib/site-packages/{pypi_name.replace('-', '_')}/__init__.py", meta_file])


def _with_prefix(pi, prefix):
    pi.sys.prefix = str(prefix)
    try:
        return pi.conda_python_packages()
    finally:
        pi.sys.prefix = sys.prefix


# --- the installer (post_install template) ----------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_pytorch_comes_from_the_locked_file_of_its_build(pi, variant):
    args = pi.torch_pip_args(variant)
    assert args[:2] == ["install", "--no-deps"]
    assert args[-2:] == ["--index-url", f"https://download.pytorch.org/whl/{variant}"]
    # Item 3 (owner W78 C2): torchvision left with ClearVoice.
    assert {pi.requirement_name(a) for a in args[2:-2]} == {"torch", "torchaudio"}
    assert any(a.startswith(f"torch==2.11.0+{variant}") for a in args)


def test_phase_4_leaves_out_pytorch_and_what_conda_owns(pi, tmp_path):
    _conda_python_package(tmp_path, "packaging", "26.3", "packaging", "26.3")
    _conda_meta(tmp_path, "tzdata", "2026c", ["share/zoneinfo/UTC"])  # time-zone data, not the Python package
    found, unidentified = _with_prefix(pi, tmp_path)
    assert found == {"packaging": ("packaging", "26.3")} and unidentified == []

    locked = pi.locked_requirement_lines("cu130")
    assert pi.check_conda_packages(locked, found, unidentified) is True
    lines = pi.release_pinned_install_lines(locked, found)
    names = {pi.requirement_name(l) for l in lines}
    assert not names & {"torch", "torchaudio", "torchvision", "packaging"}
    assert "tzdata" in names and "numpy" in names and "faster-whisper" in names
    # Item 3 (owner W78 C2): torchvision is no longer locked; torch, torchaudio, packaging left out.
    assert len(lines) == len(locked) - 3
    # W58 texts 1 and 3
    assert "Keeping conda's packaging 26.3: the same version this release is pinned to." in pi._lines
    assert "Checked 1 package that conda installed against the pinned list: all match." in pi._lines


def test_conda_packages_are_named_as_pip_names_them(pi, tmp_path):
    """D1: the PyPI name and version come from the metadata folder, not the conda name."""
    _conda_python_package(tmp_path, "python-build", "1.6.1", "build", "1.6.1")
    _conda_python_package(tmp_path, "msgpack-python", "1.1.2", "msgpack", "1.1.2")
    _conda_python_package(tmp_path, "setuptools", "84.0.0", "setuptools", "84.0.0", egg_info=True)
    _conda_meta(tmp_path, "python", "3.13.16", ["Lib/site-packages/README.txt", "python.exe"])
    _conda_meta(tmp_path, "oddpkg", "1.0", ["Lib/site-packages/odd.py"])
    found, unidentified = _with_prefix(pi, tmp_path)
    assert found == {"build": ("build", "1.6.1"), "msgpack": ("msgpack", "1.1.2"),
                     "setuptools": ("setuptools", "84.0.0")}
    assert unidentified == ["oddpkg"]


def test_a_conda_package_at_another_version_stops_the_install(pi, tmp_path):
    """D1 / W58 text 2: setuptools 84.0.0 from conda against the lock's 81.0.0 stops phase 4."""
    _conda_python_package(tmp_path, "setuptools", "84.0.0", "setuptools", "84.0.0", egg_info=True)
    found, unidentified = _with_prefix(pi, tmp_path)
    assert pi.check_conda_packages(pi.locked_requirement_lines("cu130"), found, unidentified) is False
    assert ("Stopped: conda installed setuptools 84.0.0, but this release is pinned to setuptools 81.0.0. "
            "Please report this so it can be fixed: https://github.com/meizhong986/WhisperJAV/issues "
            "— attach log.txt.") in pi._lines
    assert not any(l.startswith("Checked ") for l in pi._lines)


def test_a_check_that_compared_nothing_says_so(pi, tmp_path):
    _conda_meta(tmp_path, "oddpkg", "1.0", ["Lib/site-packages/odd.py"])
    found, unidentified = _with_prefix(pi, tmp_path)
    assert pi.check_conda_packages(pi.locked_requirement_lines("cpu"), found, unidentified) is True
    assert "Checked 0 packages that conda installed against the pinned list: all match." in pi._lines
    assert any(l.startswith("conda's oddpkg put Python files here without package metadata") for l in pi._lines)


def test_phase_4_stops_before_pip_when_conda_differs():
    source = TEMPLATE.read_text(encoding="utf-8")
    check = source.index("if not check_conda_packages(locked_lines, conda_packages, conda_unidentified):")
    assert check < source.index('["install", "--no-deps", "-r", install_file, "--progress-bar", "on"]')


def test_installer_tools_are_pinned_exactly():
    """D2 (owner W57): lief and nvidia-ml-py are installer tools, pinned here, not in the lock."""
    source = TEMPLATE.read_text(encoding="utf-8")
    assert 'NVIDIA_ML_PY_REQUIREMENT = "nvidia-ml-py==13.615.71"' in source
    assert 'LIEF_REQUIREMENT = "lief==1.0.0"' in source
    assert '"install", "lief"]' not in source and '"nvidia-ml-py",' not in source
    for variant in VARIANTS:
        names = {re.split(r"[=;@\s]", l, maxsplit=1)[0].lower()
                 for l in (LOCKED / f"requirements_{variant}.txt").read_text(encoding="utf-8").splitlines()}
        assert not names & {"lief", "nvidia-ml-py"}


@pytest.mark.parametrize("build, variant", [("13.0", "cu130"), ("12.8", "cu128"), ("", "cpu"), ("12.6", None)])
def test_the_variant_named_is_the_build_actually_installed(pi, build, variant):
    assert pi.variant_of_cuda_build(build) == variant


def test_the_console_line_is_the_owners_and_names_the_selected_build(pi):
    """W56: "Installing release-pinned dependencies (CUDA 13.0)..." with the build actually selected."""
    source = TEMPLATE.read_text(encoding="utf-8")
    assert 'log(f"Installing release-pinned dependencies ({VARIANT_LABELS[variant]})...")' in source
    assert "variant = torch_plan.variant if torch_plan else variant_of_cuda_build(" in source
    assert pi.VARIANT_LABELS == {"cu130": "CUDA 13.0", "cu128": "CUDA 12.8", "cpu": "CPU"}


def test_no_range_install_is_left_in_the_installer():
    source = TEMPLATE.read_text(encoding="utf-8")
    for gone in ("TORCH_PACKAGES", "NUMPY_REQUIREMENT", "CU128_COMMAND", "CPU_FALLBACK_COMMAND",
                 "git_packages", 'requirements_v{{VERSION}}.txt"'):
        assert gone not in source, gone
    assert '["install", "--no-deps", "-r", install_file, "--progress-bar", "on"]' in source


# --- install.py (source installs) ---------------------------------------------------------------------------

def _load_install_py(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("install_under_test", ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "log", lambda *a, **k: None)
    return module


class _FakeProcess:
    returncode = 0
    stdout = iter(())

    def wait(self, timeout=None):
        return 0


@pytest.mark.parametrize("cuda, extra", [("cu130", "cu130"), ("cu128", "cu128"), ("cpu", "cpu"), ("metal", "cpu")])
def test_install_py_syncs_frozen_with_the_builds_extra(monkeypatch, cuda, extra):
    install = _load_install_py(monkeypatch)
    seen = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: seen.append(cmd) or _FakeProcess())
    assert install.run_uv_sync(["all"], cuda, target_env=None) is True
    cmd = seen[0]
    assert cmd[cmd.index("sync") + 1] == "--frozen"
    assert cmd[cmd.index("all") + 1:cmd.index("all") + 3] == ["--extra", extra]
    assert "--index" not in cmd and "--index-strategy" not in cmd


# Platforms as (sys.platform, platform.machine()). PyTorch 2.11.0 has no Intel-Mac or Windows-on-ARM wheel.
@pytest.mark.parametrize("system, machine, build, held", [
    ("win32", "AMD64", "cu130", True), ("win32", "AMD64", "cu128", True), ("win32", "AMD64", "cpu", True),
    ("win32", "ARM64", "cpu", False),
    ("linux", "x86_64", "cu128", True), ("linux", "aarch64", "cu128", True), ("linux", "x86_64", "cpu", True),
    ("darwin", "arm64", "metal", True), ("darwin", "x86_64", "cpu", False),
])
def test_install_py_knows_which_builds_the_lock_holds(monkeypatch, system, machine, build, held):
    install = _load_install_py(monkeypatch)
    monkeypatch.setattr(install.sys, "platform", system)
    monkeypatch.setattr(install, "IS_LINUX", system == "linux")
    monkeypatch.setattr(install.platform_module, "machine", lambda: machine)
    assert install.lock_has_pytorch_for_this_pc(build) is held


@pytest.mark.parametrize("system, machine, build, text", [
    ("linux", "x86_64", "cu130",
     "CUDA 13.0 is not offered on Linux in WhisperJAV 1.10.0. Use --accelerator nvidia (CUDA 12.8 on Linux) or --accelerator cpu."),
    ("darwin", "arm64", "cu128",
     "macOS has no CUDA builds. Run without --accelerator; WhisperJAV installs for the Apple GPU."),
    ("darwin", "x86_64", "cpu",
     "PyTorch has no CPU build for this computer (darwin, x86_64) in WhisperJAV 1.10.0, so it cannot be "
     "installed here. Nothing was installed."),
    ("win32", "AMD64", "cu130", ""),
    ("linux", "x86_64", "cu128", ""),
])
def test_install_py_stops_unsupported_combinations_clearly(monkeypatch, system, machine, build, text):
    """D3 (owner W57) / W58 text 10."""
    install = _load_install_py(monkeypatch)
    monkeypatch.setattr(install.sys, "platform", system)
    monkeypatch.setattr(install, "IS_LINUX", system == "linux")
    monkeypatch.setattr(install.platform_module, "machine", lambda: machine)
    assert install.unsupported_build_message(build) == text


def test_install_py_stops_before_uv_sync_and_names_the_build_in_its_hint():
    source = (ROOT / "install.py").read_text(encoding="utf-8")
    assert source.index("unsupported = unsupported_build_message(cuda_version)") < source.index(
        "success = run_uv_sync(extras, cuda_version")
    # W58 text 9: the build actually chosen
    assert ('log(f"    You can install it later: uv sync --frozen --extra all "\n'
            '                    f"--extra {ACC.lock_extra(cuda_version)}{own}")') in source
    assert "uv sync --extra enhance" not in source


# --- whisperjav-upgrade ---------------------------------------------------------------------------------------

@pytest.fixture
def upgrade(monkeypatch):
    import whisperjav.upgrade as up
    monkeypatch.setattr(up, "_find_source_root", lambda: None)
    return up


def test_source_upgrade_syncs_frozen_with_the_recorded_build(upgrade, monkeypatch, tmp_path, capsys):
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="Already up to date.", stderr="")

    monkeypatch.setattr(upgrade.subprocess, "run", fake_run)
    monkeypatch.setattr(upgrade.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(upgrade, "read_installed_stack", lambda install_dir=None: "cu128")
    monkeypatch.setattr(upgrade, "_detect_target_env_for_upgrade", lambda: (None, False))
    monkeypatch.setattr(upgrade, "unsupported_source_build_message", lambda root, cuda: "")
    assert upgrade._run_uv_sync_upgrade(tmp_path, extras="all") is True
    sync = next(c for c in seen if c[:2] == ["uv", "sync"])
    assert sync == ["uv", "sync", "--frozen", "--extra", "all", "--extra", "cu128"]
    assert "Upgrading with release-pinned dependencies (NVIDIA GPU (CUDA 12.8))..." in capsys.readouterr().out


@pytest.mark.parametrize("system, machine, build, text", [
    ("linux", "x86_64", "cu130",
     "This installation uses CUDA 13.0, which WhisperJAV 1.10.0 does not offer on Linux. Nothing was changed. "
     "Run install.py again with --accelerator nvidia (CUDA 12.8 on Linux)."),
    ("darwin", "x86_64", "cpu",
     "PyTorch has no CPU build for this computer (darwin, x86_64) in WhisperJAV 1.10.0, so it cannot be "
     "upgraded here. Nothing was changed."),
    ("darwin", "arm64", "metal", ""),
    ("win32", "AMD64", "cu130", ""),
])
def test_source_upgrade_stops_unsupported_combinations_before_git_pull(upgrade, monkeypatch, system, machine,
                                                                        build, text):
    """D3 (owner W57): checked against the repository's own uv.lock, before anything changes."""
    monkeypatch.setattr(upgrade.sys, "platform", system)
    monkeypatch.setattr(upgrade.platform, "machine", lambda: machine)
    assert upgrade.unsupported_source_build_message(ROOT, build) == text


def test_source_upgrade_does_not_pull_for_an_unsupported_build(upgrade, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(upgrade.subprocess, "run", lambda *a, **k: pytest.fail("nothing may run"))
    monkeypatch.setattr(upgrade, "_detect_cuda_for_source", lambda: "cu130")
    monkeypatch.setattr(upgrade, "unsupported_source_build_message", lambda root, cuda: "stop reason")
    assert upgrade._run_uv_sync_upgrade(tmp_path, extras="all") is False
    assert "stop reason" in capsys.readouterr().out


def _installer_install(tmp_path):
    install = tmp_path / "WhisperJAV"
    (install / "Scripts").mkdir(parents=True)
    (install / "python.exe").touch()
    (install / "Scripts" / "pip.exe").touch()
    return install


def _locked_lines(stack):
    return [l.strip() for l in (LOCKED / f"requirements_{stack}.txt").read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def test_installer_upgrade_installs_the_locked_file_without_conda_packages(upgrade, monkeypatch, tmp_path, capsys):
    install = _installer_install(tmp_path)
    _conda_python_package(install, "packaging", "26.3", "packaging", "26.3")
    seen = []

    def fake_run(cmd, **kw):
        if "-r" in cmd:
            seen.append(("rest", Path(cmd[cmd.index("-r") + 1]).read_text(encoding="utf-8").split("\n")))
        else:
            seen.append(("cmd", list(cmd)))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(upgrade.subprocess, "run", fake_run)
    lines = _locked_lines("cpu")
    assert upgrade.install_locked_requirements(install, "cpu", lines) is True
    rest = next(v for k, v in seen if k == "rest")
    names = {upgrade._requirement_name(l) for l in rest if l}
    assert not names & {"torch", "torchaudio", "torchvision", "packaging"}
    torch_cmd = next(v for k, v in seen if k == "cmd")
    assert "--no-deps" in torch_cmd and torch_cmd[-2:] == ["--index-url", "https://download.pytorch.org/whl/cpu"]
    assert not (install / "requirements_upgrade_cpu.txt").exists()
    out = capsys.readouterr().out
    assert "Upgrading with release-pinned dependencies (CPU only (not accelerated))..." in out
    assert f"Installed the release's pinned packages ({len(lines)})." in out  # W58 text 6


def test_an_unrecorded_build_stops_the_installer_upgrade(upgrade, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(upgrade.subprocess, "run", lambda *a, **k: pytest.fail("nothing may be installed"))
    assert upgrade.upgrade_package_with_extras(tmp_path, "all") is False
    assert ("This installation does not record its PyTorch build (whisperjav_stack.txt), so the upgrade cannot "
            "choose its pinned packages. Nothing was changed. Run the WhisperJAV 1.10 installer again to repair "
            "it.") in capsys.readouterr().out  # W58 text 5


# --- the upgrade's pre-check (D5) ------------------------------------------------------------------------------

def test_the_pre_check_runs_the_install_commands_as_a_dry_run(upgrade, monkeypatch, tmp_path, capsys):
    """W57 INPUT3: the pre-check must agree with the pinned install -- same lines, same commands, plus --dry-run."""
    install = _installer_install(tmp_path)
    _conda_python_package(install, "packaging", "26.3", "packaging", "26.3")
    text = (LOCKED / "requirements_cu130.txt").read_text(encoding="utf-8")
    monkeypatch.setattr(upgrade, "fetch_release_locked_file", lambda source, stack: text)
    dry, real = [], []

    def record(store):
        def run(cmd, **kw):
            if "-r" in cmd:
                store.append([*cmd[:cmd.index("-r") + 1],
                              Path(cmd[cmd.index("-r") + 1]).read_text(encoding="utf-8")])
            else:
                store.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return run

    monkeypatch.setattr(upgrade, "_run_bounded", lambda cmd, timeout: record(dry)(cmd))
    lines = upgrade.prepare_release_pins(install, "cu130")
    assert lines == _locked_lines("cu130")
    monkeypatch.setattr(upgrade.subprocess, "run", record(real))
    assert upgrade.install_locked_requirements(install, "cu130", lines) is True
    assert [[a for a in c if a != "--dry-run"] for c in dry] == real
    assert all("--dry-run" in c for c in dry)
    out = capsys.readouterr().out
    assert "Checking that this release's pinned packages are available for this PC (NVIDIA GPU (CUDA 13.0))..." in out
    assert f"All {len(lines)} pinned packages are available." in out  # W58 text 4
    assert "Checked 1 package that conda installed against the pinned list: all match." in out


def test_an_unavailable_package_stops_before_any_change(upgrade, monkeypatch, tmp_path, capsys):
    install = _installer_install(tmp_path)
    monkeypatch.setattr(upgrade, "fetch_release_locked_file",
                        lambda source, stack: (LOCKED / "requirements_cpu.txt").read_text(encoding="utf-8"))
    pip_err = ("ERROR: Could not find a version that satisfies the requirement numpy==9.9.9 (from versions: 2.5.2)\n"
               "ERROR: No matching distribution found for numpy==9.9.9\n")
    monkeypatch.setattr(upgrade, "_run_bounded", lambda cmd, timeout: subprocess.CompletedProcess(cmd, 1, "", pip_err))
    assert upgrade.prepare_release_pins(install, "cpu") is None
    assert ("Upgrade stopped before any change: numpy is not available for this PC "
            "(ERROR: No matching distribution found for numpy==9.9.9).") in capsys.readouterr().out


def test_a_conda_package_at_another_version_stops_the_upgrade_before_any_change(upgrade, monkeypatch, tmp_path,
                                                                                 capsys):
    install = _installer_install(tmp_path)
    _conda_python_package(install, "packaging", "26.2", "packaging", "26.2")
    monkeypatch.setattr(upgrade, "fetch_release_locked_file",
                        lambda source, stack: (LOCKED / "requirements_cpu.txt").read_text(encoding="utf-8"))
    monkeypatch.setattr(upgrade, "_run_bounded", lambda *a, **k: pytest.fail("no dry run after a stop"))
    assert upgrade.prepare_release_pins(install, "cpu") is None
    assert "Stopped: conda installed packaging 26.2, but this release is pinned to packaging 26.3." in \
        capsys.readouterr().out


def test_a_missing_pinned_list_stops_before_any_change(upgrade, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(upgrade, "fetch_release_locked_file", lambda source, stack: None)
    assert upgrade.prepare_release_pins(_installer_install(tmp_path), "cu128") is None
    assert "Upgrade stopped before any change: the new release's pinned package list for NVIDIA GPU (CUDA 12.8) " \
           "could not be read" in capsys.readouterr().out


def test_the_pinned_list_is_read_from_the_release_tag_a_wheel_or_a_folder(upgrade, monkeypatch, tmp_path):
    import zipfile
    import urllib.request
    rel = "whisperjav/installer/locked/requirements_cu130.txt"
    wheel = tmp_path / "whisperjav-1.10.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr(rel, "numpy==2.5.2\n")
    assert upgrade.fetch_release_locked_file(str(wheel), "cu130") == "numpy==2.5.2\n"
    assert upgrade.fetch_release_locked_file(str(ROOT), "cu130") == (LOCKED / "requirements_cu130.txt").read_text(
        encoding="utf-8")
    asked = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"torch==2.11.0+cu130\n"

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: asked.append(req.full_url) or _Response())
    assert upgrade.fetch_release_locked_file("git+https://github.com/meizhong986/whisperjav.git@v1.10.1",
                                             "cu130") == "torch==2.11.0+cu130\n"
    assert asked == [f"https://raw.githubusercontent.com/meizhong986/whisperjav/v1.10.1/{rel}"]
    assert upgrade.fetch_release_locked_file(str(tmp_path / "nothing"), "cu130") is None


# --- --extras on an installer install (D4) ---------------------------------------------------------------------

def test_extras_on_an_installer_install_stops_and_help_says_so(upgrade, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(upgrade.sys, "argv", ["whisperjav-upgrade", "--yes", "--extras", "cli,gui"])
    monkeypatch.setattr(upgrade, "detect_installation", lambda: tmp_path)
    monkeypatch.setattr(upgrade, "get_current_version", lambda d: "1.10.0")
    monkeypatch.setattr(upgrade, "check_network", lambda: pytest.fail("stops before the network"))
    assert upgrade.main() == 1
    assert ("--extras applies to source installs only. This installation came from the Windows installer, so an "
            "upgrade installs the release's full pinned package set. Run again without --extras.") in \
        capsys.readouterr().out  # W58 text 8
    monkeypatch.setattr(upgrade.sys, "argv", ["whisperjav-upgrade", "--help"])
    with pytest.raises(SystemExit):
        upgrade.parse_args()
    assert ("Extras to upgrade (source installs only; installer installs always get the release's full pinned "
            "set).") in " ".join(capsys.readouterr().out.split())


def test_step_2_is_labelled_by_what_it_does(upgrade):
    source = Path(upgrade.__file__).read_text(encoding="utf-8")
    assert 'print_step(2, total_steps, "Placing the CUDA 12 library for faster-whisper...")' in source  # W58 text 7
    assert 'print_success("Nothing to do for this PyTorch build.")' in source
    assert "Fixing package versions" not in source and not hasattr(upgrade, "check_upgrade_compatibility")


def test_the_newest_numpy_step_is_gone(upgrade):
    assert not hasattr(upgrade, "fix_package_versions") and not hasattr(upgrade, "FIX_PACKAGES")


# --- build_release.py --------------------------------------------------------------------------------------

def test_the_build_copies_the_locked_files_under_versioned_names(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_release_under_test", ROOT / "installer" / "build_release.py")
    build_release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_release)
    builder = build_release.ReleaseBuilder()
    builder.generated_dir = tmp_path
    assert builder.generate_from_templates() is True
    for variant in VARIANTS:
        copied = tmp_path / f"requirements_v{builder.version}_{variant}.txt"
        assert copied.read_bytes() == (LOCKED / f"requirements_{variant}.txt").read_bytes()
    assert not list(tmp_path.glob(f"requirements_v{builder.version}.txt"))


def test_construct_ships_the_three_files_and_pins_the_conda_part():
    text = (ROOT / "installer" / "templates" / "construct.yaml.template").read_text(encoding="utf-8")
    for variant in VARIANTS:
        assert f"  - requirements_v{{{{VERSION}}}}_{variant}.txt" in text
    for spec in ("python=3.13.16", "openssl=3.6.2", "pip=26.2.1", "git=2.56.0", "ffmpeg=7.1.1", "menuinst=2.5.2"):
        assert f"  - {spec}" in text, spec


def test_the_build_environment_pins_uv_and_constructor():
    text = (ROOT / "installer" / "build_env.yml").read_text(encoding="utf-8")
    for spec in ("constructor=3.17.3", "uv=0.9.26", "conda-standalone=26.5.2", "nsis=3.12"):
        assert f"  - {spec}" in text, spec
    bat = (ROOT / "installer" / "templates" / "build_installer.bat.template").read_text(encoding="utf-8")
    assert 'set "UV_REQUIRED=uv 0.9.26"' in bat and "%USERPROFILE%\\.local\\bin" not in bat
