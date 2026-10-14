"""v1.10.0 item 2e (spec H, check S8): whisperjav-upgrade takes the newest release of its own
series -- v1.10.* for 1.10 -- never GitHub's main branch, and keeps the PC's stack.

No network: the releases list and the stack file are stand-ins.
"""
import pytest

import whisperjav.upgrade as upgrade


def _rel(tag, prerelease=False, draft=False):
    return {"tag_name": tag, "prerelease": prerelease, "draft": draft}


RELEASES = [
    _rel("v1.11.0"), _rel("v2.0.0"),
    _rel("v1.10.3", prerelease=True), _rel("v1.10.4", draft=True),
    _rel("v1.10.2"), _rel("v1.10.0"), _rel("v1.10.1"),
    _rel("v1.9.5"), _rel("v1.9.4"),
]


@pytest.fixture(autouse=True)
def _never_a_source_install(monkeypatch):
    # The repository folder looks like a source install; that path runs a real git pull.
    monkeypatch.setattr(upgrade, "_find_source_root", lambda: None)


def test_newest_release_of_the_series_only():
    assert upgrade.newest_release_in_series("1.10.0", RELEASES) == "v1.10.2"


def test_later_series_and_pre_releases_and_drafts_are_never_chosen():
    tag = upgrade.newest_release_in_series("1.10.0", RELEASES)
    assert tag not in ("v1.11.0", "v2.0.0", "v1.10.3", "v1.10.4")


def test_a_195_install_is_not_offered_110():
    assert upgrade.newest_release_in_series("1.9.5", RELEASES) == "v1.9.5"


def test_no_release_in_the_series():
    assert upgrade.newest_release_in_series("1.12.0", RELEASES) is None


def test_resolved_source_is_a_tag_never_main(monkeypatch):
    monkeypatch.delenv("WHISPERJAV_UPGRADE_REPO", raising=False)
    monkeypatch.setattr(upgrade, "_fetch_releases", lambda timeout=15: RELEASES)
    source, why = upgrade.resolve_upgrade_source("1.10.0")
    assert source == "git+https://github.com/meizhong986/whisperjav.git@v1.10.2"
    assert "@main" not in source and "v1.10.2" in why


def test_unreadable_releases_list_does_not_fall_back_to_main(monkeypatch):
    monkeypatch.delenv("WHISPERJAV_UPGRADE_REPO", raising=False)
    monkeypatch.setattr(upgrade, "_fetch_releases", lambda timeout=15: None)
    source, why = upgrade.resolve_upgrade_source("1.10.0")
    assert source is None and "releases list" in why


def test_override_still_wins(monkeypatch):
    monkeypatch.setenv("WHISPERJAV_UPGRADE_REPO", "D:/wheels/whisperjav-1.10.1-py3-none-any.whl")
    monkeypatch.setattr(upgrade, "_fetch_releases", lambda timeout=15: pytest.fail("not read with an override"))
    assert upgrade.resolve_upgrade_source("1.10.0")[0] == "D:/wheels/whisperjav-1.10.1-py3-none-any.whl"


def test_default_source_is_no_longer_main(monkeypatch):
    assert "@main" not in upgrade.GITHUB_GIT_URL


@pytest.mark.parametrize("recorded", ["cu130", "cu128", "cpu"])
def test_source_install_keeps_the_recorded_stack(monkeypatch, recorded):
    monkeypatch.setattr(upgrade, "read_installed_stack", lambda install_dir=None: recorded)
    monkeypatch.setattr(upgrade.sys, "platform", "win32")
    monkeypatch.setattr(upgrade.subprocess, "run",
                        lambda *a, **k: pytest.fail("the recorded stack is used, nothing is detected"))
    assert upgrade._detect_cuda_for_source() == recorded


def test_stack_file_round_trip(tmp_path):
    from whisperjav.installer.cublas12 import read_stack, write_stack
    assert read_stack(str(tmp_path)) is None
    write_stack("cu128", str(tmp_path))
    assert read_stack(str(tmp_path)) == "cu128"
    assert upgrade.read_installed_stack(tmp_path) == "cu128"
    with pytest.raises(ValueError):
        write_stack("cu118", str(tmp_path))


def test_cublas12_is_placed_again_only_on_a_cuda_13_windows_install(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(upgrade, "IS_WINDOWS", True)
    monkeypatch.setattr(upgrade, "_find_python", lambda d: tmp_path / "python.exe")

    class R:
        returncode, stdout, stderr = 0, "OK: placed", ""

    monkeypatch.setattr(upgrade.subprocess, "run", lambda cmd, **k: calls.append(cmd) or R())
    for stack in ("cu128", "cpu", None):
        monkeypatch.setattr(upgrade, "read_installed_stack", lambda d=None, s=stack: s)
        upgrade.place_cublas12_after_upgrade(tmp_path)
    assert calls == []
    monkeypatch.setattr(upgrade, "read_installed_stack", lambda d=None: "cu130")
    upgrade.place_cublas12_after_upgrade(tmp_path)
    # First: is nvidia-cublas-cu12 still there (a source install's uv sync removes it)? Then place.
    assert calls[0][1:] == ["-c", "import nvidia.cublas"]
    assert calls[-1][1:] == ["-m", "whisperjav.installer.cublas12", "place"]


def test_a_removed_cublas12_package_is_installed_again_before_placing(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(upgrade, "IS_WINDOWS", True)
    monkeypatch.setattr(upgrade, "_find_python", lambda d: tmp_path / "python.exe")
    monkeypatch.setattr(upgrade, "read_installed_stack", lambda d=None: "cu130")
    monkeypatch.setattr(upgrade.shutil, "which", lambda name: None)

    class R:
        def __init__(self, rc):
            self.returncode, self.stdout, self.stderr = rc, "", ""

    def fake_run(cmd, **k):
        calls.append(cmd)
        return R(1 if cmd[1:] == ["-c", "import nvidia.cublas"] else 0)

    monkeypatch.setattr(upgrade.subprocess, "run", fake_run)
    upgrade.place_cublas12_after_upgrade(tmp_path)
    assert calls[1][-1] == "nvidia-cublas-cu12==12.8.4.1"
    assert calls[-1][1:] == ["-m", "whisperjav.installer.cublas12", "place"]


@pytest.mark.parametrize("platform_name, expected", [("linux", "cu128"), ("win32", "cu130")])
def test_without_a_record_linux_gets_cuda_12_8(monkeypatch, platform_name, expected):
    """Owner, W45: Linux stays on CUDA 12.8 for 1.10.0."""
    monkeypatch.setattr(upgrade, "read_installed_stack", lambda install_dir=None: None)
    monkeypatch.setattr(upgrade.sys, "platform", platform_name)

    class R:
        returncode, stdout, stderr = 0, "610.47\n", ""

    monkeypatch.setattr(upgrade.subprocess, "run", lambda *a, **k: R())
    assert upgrade._detect_cuda_for_source() == expected
