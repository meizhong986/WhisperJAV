"""v1.10.0, item 2f: one locked dependency list (uv.lock) for development, source installs, the installer
and upgrades (owner W53-W56). Spec: docs/requirements/v1.10.x/specs/2026-10-14-item02f-locked-dependencies.md.

Nothing here installs anything or touches a real environment: paths are tmp_path or the tracked locked files,
subprocess is replaced by stand-ins, and source-install detection is patched (lesson of 2026-10-14).
"""
import json
import re
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


# --- the installer (post_install template) ----------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_pytorch_comes_from_the_locked_file_of_its_build(pi, variant):
    args = pi.torch_pip_args(variant)
    assert args[:2] == ["install", "--no-deps"]
    assert args[-2:] == ["--index-url", f"https://download.pytorch.org/whl/{variant}"]
    assert {pi.requirement_name(a) for a in args[2:-2]} == {"torch", "torchaudio", "torchvision"}
    assert any(a.startswith(f"torch==2.11.0+{variant}") for a in args)


def test_phase_4_leaves_out_pytorch_and_what_conda_owns(pi, tmp_path):
    _conda_meta(tmp_path, "packaging", "26.3", ["Lib/site-packages/packaging/__init__.py"])
    _conda_meta(tmp_path, "tzdata", "2026c", ["share/zoneinfo/UTC"])  # time-zone data, not the Python package
    pi.sys.prefix = str(tmp_path)
    try:
        owned = pi.conda_python_packages()
    finally:
        pi.sys.prefix = sys.prefix
    assert owned == {"packaging": "26.3"}

    locked = pi.locked_requirement_lines("cu130")
    lines = pi.release_pinned_install_lines(locked, owned)
    names = {pi.requirement_name(l) for l in lines}
    assert not names & {"torch", "torchaudio", "torchvision", "packaging"}
    assert "tzdata" in names and "numpy" in names and "faster-whisper" in names
    assert len(lines) == len(locked) - 4
    assert any(l.startswith("Kept conda's packaging 26.3") for l in pi._lines)


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
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: seen.append(cmd) or _FakeProcess())
    assert install.run_uv_sync(["all"], cuda, target_env=None) is True
    cmd = seen[0]
    assert cmd[cmd.index("sync") + 1] == "--frozen"
    assert cmd[cmd.index("all") + 1:cmd.index("all") + 3] == ["--extra", extra]
    assert "--index" not in cmd and "--index-strategy" not in cmd


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
    assert upgrade._run_uv_sync_upgrade(tmp_path, extras="all") is True
    sync = next(c for c in seen if c[:2] == ["uv", "sync"])
    assert sync == ["uv", "sync", "--frozen", "--extra", "all", "--extra", "cu128"]
    assert "Upgrading with release-pinned dependencies (CUDA 12.8)..." in capsys.readouterr().out


def test_installer_upgrade_installs_the_locked_file_without_conda_packages(upgrade, monkeypatch, tmp_path, capsys):
    install = tmp_path / "WhisperJAV"
    (install / "Scripts").mkdir(parents=True)
    (install / "python.exe").touch()
    (install / "Scripts" / "pip.exe").touch()
    _conda_meta(install, "packaging", "26.3", ["Lib/site-packages/packaging/__init__.py"])
    monkeypatch.setattr(upgrade, "_installed_locked_file", lambda py, stack: LOCKED / f"requirements_{stack}.txt")
    seen = []

    def fake_run(cmd, **kw):
        if "-r" in cmd:
            seen.append(("rest", Path(cmd[cmd.index("-r") + 1]).read_text(encoding="utf-8").split()))
        else:
            seen.append(("cmd", list(cmd)))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(upgrade.subprocess, "run", fake_run)
    assert upgrade.install_locked_requirements(install, "cpu") is True
    rest = next(v for k, v in seen if k == "rest")
    names = {upgrade._requirement_name(l) for l in rest}
    assert not names & {"torch", "torchaudio", "torchvision", "packaging"}
    torch_cmd = next(v for k, v in seen if k == "cmd")
    assert "--no-deps" in torch_cmd and torch_cmd[-2:] == ["--index-url", "https://download.pytorch.org/whl/cpu"]
    assert not (install / "requirements_upgrade_cpu.txt").exists()
    assert "Upgrading with release-pinned dependencies (CPU)..." in capsys.readouterr().out


def test_an_unrecorded_build_stops_the_installer_upgrade(upgrade, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(upgrade.subprocess, "run", lambda *a, **k: pytest.fail("nothing may be installed"))
    assert upgrade.upgrade_package_with_extras(tmp_path, "all") is False
    assert "not recorded" in capsys.readouterr().out


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
