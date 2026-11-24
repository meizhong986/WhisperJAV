"""The upgrade gives pip a temporary folder inside the install folder (1.10.3, owner W764).

Before, pip moved the running Scripts\\whisperjav-upgrade.exe into Windows' temporary folder while replacing
WhisperJAV: across drives that failed halfway (WinError 5) and WhisperJAV no longer started; on the same drive the old
launcher stayed in the temporary folder. Every test works in pytest's temporary folder; nothing here runs pip.
"""
import subprocess
import sys
import time
from pathlib import Path

from whisperjav import upgrade


def test_pip_env_puts_pips_temporary_folder_inside_the_install(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERJAV_TEST_KEEP", "1")
    env = upgrade.pip_env(tmp_path)
    folder = tmp_path / upgrade.PIP_TEMP_NAME
    assert folder.is_dir()
    assert env["TMPDIR"] == env["TEMP"] == env["TMP"] == str(folder)
    assert env["WHISPERJAV_TEST_KEEP"] == "1"          # the rest of the environment is passed on


def test_the_wheel_step_runs_pip_with_that_folder(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(upgrade, "_find_pip", lambda install_dir: tmp_path / "Scripts" / "pip.exe")
    monkeypatch.setattr(upgrade.subprocess, "run", fake_run)
    assert upgrade.upgrade_package_wheel_only(tmp_path) is True
    assert seen["env"]["TEMP"] == str(tmp_path / upgrade.PIP_TEMP_NAME)


def test_the_pinned_step_and_the_rollback_run_pip_with_that_folder(tmp_path, monkeypatch):
    envs = []

    def fake_run(cmd, **kwargs):
        envs.append(kwargs.get("env"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(upgrade, "_find_pip", lambda install_dir: tmp_path / "Scripts" / "pip.exe")
    monkeypatch.setattr(upgrade, "_split_pins", lambda install_dir, lines: (lines, []))
    monkeypatch.setattr(upgrade.subprocess, "run", fake_run)
    assert upgrade.install_locked_requirements(tmp_path, "cu130", ["packaging==26.3"]) is True
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    for name in ("requirements.txt", "metadata.json", "freeze.txt"):
        (snapshot / name).write_text("packaging==26.3\n", encoding="utf-8")
    upgrade.rollback_to_snapshot(tmp_path, snapshot)
    assert envs and all(env and env["TEMP"] == str(tmp_path / upgrade.PIP_TEMP_NAME) for env in envs)


def test_the_folder_is_removed_at_the_end(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(upgrade.subprocess, "Popen", lambda *a, **k: started.append(a))
    upgrade.pip_env(tmp_path)
    (tmp_path / upgrade.PIP_TEMP_NAME / "pip-uninstall-x").mkdir()
    upgrade.remove_pip_temp(tmp_path)
    assert not (tmp_path / upgrade.PIP_TEMP_NAME).exists()
    assert started == []                                # nothing left to remove later


def test_a_folder_still_in_use_is_removed_by_a_process_of_its_own(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(upgrade.shutil, "rmtree", lambda path, ignore_errors=False: None)   # as if a file is in use
    monkeypatch.setattr(upgrade.subprocess, "Popen", lambda cmd, **k: started.append((cmd, k)))
    upgrade.pip_env(tmp_path)
    upgrade.remove_pip_temp(tmp_path, later=False)      # at the start: nothing goes on running
    assert started == []
    upgrade.remove_pip_temp(tmp_path)                   # at the end
    (cmd, kwargs), = started
    assert cmd[:2] == [sys.executable, "-c"] and cmd[3] == str(tmp_path / upgrade.PIP_TEMP_NAME)
    assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL


def test_the_remover_removes_the_folder_once_it_can(tmp_path):
    """The remover's own code, run for real on a folder nothing holds."""
    folder = tmp_path / upgrade.PIP_TEMP_NAME
    (folder / "pip-uninstall-x").mkdir(parents=True)
    (folder / "pip-uninstall-x" / "whisperjav-upgrade.exe").write_bytes(b"MZ")
    proc = subprocess.Popen([sys.executable, "-c", upgrade._REMOVE_WHEN_FREE, str(folder)])
    try:
        proc.wait(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert not folder.exists()
