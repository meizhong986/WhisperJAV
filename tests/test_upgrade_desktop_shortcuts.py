"""The upgrade removes only this install's "WhisperJAV v..." desktop shortcuts (1.10.3).

Before 1.10.3 it removed every desktop shortcut named "WhisperJAV v...", those of other installs too.

The Windows shell stays closed to tests (conftest.py, after real shortcuts were lost to tests on 2026-10-27 and
2026-11-09). These tests give update_desktop_shortcut a stand-in shell instead: its shortcuts are small text files in a
desktop folder under pytest's temporary folder, and it refuses to name the user's desktop, so the function cannot reach
it even by mistake. The function itself is the real one, taken when this file is loaded (conftest replaces it in every
test).
"""
import os

import pytest

from whisperjav import upgrade

REAL_UPDATE = upgrade.update_desktop_shortcut   # before conftest's stand-in is put in place for each test


class _Link:
    """A stand-in shortcut: the file holds "target|start-in folder"."""

    def __init__(self, path):
        self.path = path
        self.TargetPath = self.WorkingDirectory = self.Arguments = self.IconLocation = self.Description = ""
        if os.path.exists(path):
            self.TargetPath, self.WorkingDirectory = open(path, encoding="utf-8").read().split("|")

    def Save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"{self.TargetPath}|{self.WorkingDirectory}")


class _Shell:
    def CreateShortcut(self, path):
        return _Link(path)

    def SpecialFolders(self, name):
        raise AssertionError("the test's shell names no desktop: the function must use the folder it is given")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    win32com_client = pytest.importorskip("win32com.client")
    monkeypatch.setattr(upgrade.platform, "system", lambda: "Windows")
    monkeypatch.setattr(win32com_client, "Dispatch", lambda name: _Shell())
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    installs = {}
    for name in ("WhisperJAV", "WhisperJAV2", "Other"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "WhisperJAV-GUI.exe").write_bytes(b"")
        installs[name] = folder
    return desktop, installs


def _make_link(path, target, working_dir):
    link = _Link(str(path))
    link.TargetPath, link.WorkingDirectory = str(target), str(working_dir)
    link.Save()


def test_only_this_installs_shortcuts_go(setup):
    desktop, installs = setup
    this, other, third = installs["WhisperJAV"], installs["WhisperJAV2"], installs["Other"]
    _make_link(desktop / "WhisperJAV v1.10.2.lnk", this / "WhisperJAV-GUI.exe", this)
    _make_link(desktop / "WhisperJAV v1.10.1.lnk", other / "WhisperJAV-GUI.exe", other)   # its folder name shares the prefix
    _make_link(desktop / "WhisperJAV v1.10.0.lnk", third / "WhisperJAV-GUI.exe", third)
    _make_link(desktop / "My notes.lnk", this / "WhisperJAV-GUI.exe", this)                # not a "WhisperJAV v" name

    assert REAL_UPDATE(this, "1.10.3", desktop=str(desktop)) is True

    assert sorted(os.listdir(desktop)) == ["My notes.lnk", "WhisperJAV v1.10.0.lnk", "WhisperJAV v1.10.1.lnk",
                                           "WhisperJAV v1.10.3.lnk"]
    new = _Link(str(desktop / "WhisperJAV v1.10.3.lnk"))
    assert (new.TargetPath, new.WorkingDirectory) == (str(this / "WhisperJAV-GUI.exe"), str(this))
    kept = _Link(str(desktop / "WhisperJAV v1.10.1.lnk"))
    assert kept.TargetPath == str(other / "WhisperJAV-GUI.exe")


def test_a_shortcut_of_this_install_by_its_start_in_folder_goes(setup):
    """An older shortcut whose target is outside the install (a .bat elsewhere) but which starts in the install."""
    desktop, installs = setup
    this = installs["WhisperJAV"]
    _make_link(desktop / "WhisperJAV v1.9.5.lnk", desktop / "start.bat", this)

    assert REAL_UPDATE(this, "1.10.3", desktop=str(desktop)) is True

    assert sorted(os.listdir(desktop)) == ["WhisperJAV v1.10.3.lnk"]


def test_the_check_of_one_shortcut(setup):
    desktop, installs = setup
    this, other = installs["WhisperJAV"], installs["WhisperJAV2"]
    shell = _Shell()
    _make_link(desktop / "a.lnk", this / "WhisperJAV-GUI.exe", desktop)
    _make_link(desktop / "b.lnk", other / "WhisperJAV-GUI.exe", other)
    assert upgrade._shortcut_opens_install(shell, str(desktop / "a.lnk"), this) is True
    assert upgrade._shortcut_opens_install(shell, str(desktop / "b.lnk"), this) is False
    assert upgrade._shortcut_opens_install(shell, str(desktop / "missing.lnk"), this) is False
