"""On an install made with the Windows installer the upgrade tool neither suggests nor tries a rollback (1.10.3, owner
W753 CL4; wording W764 2, texts sheet item 6). The rollback cannot work there: the install's pip freeze names conda's
packages by build folders that do not exist (upgrade test of 2026-11-05, C5).

main() runs for real; every step that would touch pip, the network or the install is replaced. Nothing is installed.
"""
import subprocess
import sys

import pytest

from whisperjav import upgrade


@pytest.fixture
def install(tmp_path, monkeypatch):
    """A stand-in install folder and every outside step of main() replaced; returns (folder, calls)."""
    calls = []
    folder = tmp_path / "WhisperJAV"
    folder.mkdir()
    monkeypatch.setattr(upgrade, "IS_WINDOWS", True)
    monkeypatch.setattr(upgrade, "_find_source_root", lambda: None)
    monkeypatch.setattr(upgrade, "detect_installation", lambda: folder)
    monkeypatch.setattr(upgrade, "get_current_version", lambda d: "1.10.2")
    monkeypatch.setattr(upgrade, "check_network", lambda: True)
    monkeypatch.setattr(upgrade, "resolve_upgrade_source", lambda v: ("git+https://example.invalid/w.git@v1.10.3", "v1.10.3"))
    monkeypatch.setattr(upgrade, "read_installed_stack", lambda d: "cu130")
    monkeypatch.setattr(upgrade, "prepare_release_pins", lambda d, s, check_availability=True: ["packaging==26.3"])
    monkeypatch.setattr(upgrade, "create_upgrade_snapshot", lambda d: d / ".whisperjav_snapshots" / "snap")
    monkeypatch.setattr(upgrade, "rollback_to_snapshot", lambda d, s: calls.append("rollback") or True)
    monkeypatch.setattr(upgrade, "list_snapshots", lambda d: calls.append("list") or [])
    monkeypatch.setattr(upgrade, "place_cublas12_after_upgrade", lambda d: None)
    monkeypatch.setattr(upgrade, "update_launcher", lambda d: True)
    monkeypatch.setattr(upgrade, "cleanup_old_files", lambda d: 0)
    monkeypatch.setattr(upgrade, "cleanup_old_snapshots", lambda d, keep=3: 0)
    monkeypatch.setattr("atexit.register", lambda *a, **k: None)
    return folder, calls


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["whisperjav-upgrade", *argv])
    return upgrade.main()


def test_an_installer_install_is_recognised(install):
    folder, _ = install
    assert upgrade.made_by_windows_installer(folder) is False
    (folder / "conda-meta").mkdir()
    assert upgrade.made_by_windows_installer(folder) is True


def test_a_failed_upgrade_gives_the_guidance_and_tries_no_rollback(install, monkeypatch, capsys):
    folder, calls = install
    (folder / "conda-meta").mkdir()
    monkeypatch.setattr(upgrade, "upgrade_package_with_extras", lambda d, e, l=None: False)
    assert run(monkeypatch, "--yes") == 1
    out = capsys.readouterr().out
    assert upgrade.INSTALLER_UPGRADE_FAILED_TEXT in out
    assert "Attempting rollback" not in out and calls == []


def test_on_another_install_the_failed_upgrade_still_rolls_back(install, monkeypatch, capsys):
    _, calls = install
    monkeypatch.setattr(upgrade, "upgrade_package_with_extras", lambda d, e, l=None: False)
    assert run(monkeypatch, "--yes") == 1
    assert calls == ["rollback"]
    assert upgrade.INSTALLER_UPGRADE_FAILED_TEXT not in capsys.readouterr().out


def test_a_successful_upgrade_does_not_suggest_rollback(install, monkeypatch, capsys):
    folder, _ = install
    (folder / "conda-meta").mkdir()
    monkeypatch.setattr(upgrade, "upgrade_package_with_extras", lambda d, e, l=None: True)
    monkeypatch.setattr(upgrade, "verify_installation", lambda d: (True, "1.10.3"))
    assert run(monkeypatch, "--yes") == 0
    out = capsys.readouterr().out
    assert "Upgrade complete!" in out and "--rollback" not in out and "Rollback snapshot" not in out


def test_rollback_on_an_installer_install_says_what_works(install, monkeypatch, capsys):
    folder, calls = install
    (folder / "conda-meta").mkdir()
    assert run(monkeypatch, "--rollback", "latest", "--yes") == 1
    captured = capsys.readouterr()
    assert upgrade.INSTALLER_ROLLBACK_TEXT in captured.out + captured.err
    assert calls == []                                  # no snapshot looked up, nothing installed


def _one_snapshot(folder):
    return [{"name": "snap", "path": folder / "snap", "whisperjav_version": "1.10.2", "timestamp": "t"}]


def test_list_snapshots_on_an_installer_install_offers_no_rollback(install, monkeypatch, capsys):
    folder, _ = install
    (folder / "conda-meta").mkdir()
    monkeypatch.setattr(upgrade, "list_snapshots", lambda d: _one_snapshot(folder))
    assert run(monkeypatch, "--list-snapshots") == 0
    out = capsys.readouterr().out
    assert "To rollback" not in out and upgrade.INSTALLER_ROLLBACK_TEXT in out


def test_list_snapshots_on_another_install_still_names_the_rollback(install, monkeypatch, capsys):
    folder, _ = install
    monkeypatch.setattr(upgrade, "list_snapshots", lambda d: _one_snapshot(folder))
    assert run(monkeypatch, "--list-snapshots") == 0
    assert "To rollback: whisperjav-upgrade --rollback <snapshot_name>" in capsys.readouterr().out


@pytest.mark.parametrize("installer, promised", [(True, False), (False, True)])
def test_the_confirmation_promises_a_rollback_only_where_it_works(install, monkeypatch, capsys, installer, promised):
    folder, _ = install
    if installer:
        (folder / "conda-meta").mkdir()
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")    # the user answers no: nothing runs
    assert run(monkeypatch) == 0
    out = capsys.readouterr().out
    assert "Upgrade cancelled." in out
    assert ("A snapshot will be created for rollback if needed." in out) is promised
