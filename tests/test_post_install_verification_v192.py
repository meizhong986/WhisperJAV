"""Behaviour of Phase 6 of the Windows installer's post-install script (v1.9.2).

Phase 6 checks that the packages WhisperJAV needs can actually be imported. Until
v1.9.2 it printed a line per failure and the install then reported success anyway,
so a user whose install could not run was told it had worked. It now ends the
install instead.

That makes the difference between "this package does not import" and "this check
did not finish in time" matter, because only the first is evidence of a broken
install. Commit 63936e1 records that the earlier 30 s limit caused false failures
on real installs; a timeout therefore counts as a warning, not a failure.

These tests load the template, substitute the version placeholders, and run
``run_post_install_verification()`` with the import check replaced, so they test
what the function decides rather than what the file contains. The string-level
guards live in ``tests/test_installer_comprehensive.py``.
"""

import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATE = PROJECT_ROOT / "installer" / "templates" / "post_install.py.template"

# The seven packages Phase 6 treats as core, by the display name it logs.
CORE_DISPLAY_NAMES = {
    "PyTorch", "OpenAI Whisper", "Stable-TS", "Faster-Whisper",
    "PyWebView", "SRT", "PyYAML",
}


def _load_template_module():
    """Exec the install-script template as a module, placeholders filled in."""
    source = TEMPLATE.read_text(encoding="utf-8")
    source = re.sub(r"\{\{[A-Z_]+\}\}", "1.9.2", source)

    module = types.ModuleType("post_install_under_test")
    module.__file__ = str(TEMPLATE)
    exec(compile(source, str(TEMPLATE), "exec"), module.__dict__)
    return module


@pytest.fixture
def phase6(tmp_path, monkeypatch):
    """The verification function, with everything it touches under our control."""
    module = _load_template_module()

    # Somewhere harmless to write the install log
    monkeypatch.setattr(module, "LOG_FILE", str(tmp_path / "install_log.txt"))

    lines = []
    monkeypatch.setattr(module, "log", lines.append)
    monkeypatch.setattr(module, "log_section", lines.append, raising=False)

    # A prefix with the four entry-point executables present
    prefix = tmp_path / "prefix"
    (prefix / "Scripts").mkdir(parents=True)
    for exe in ("whisperjav.exe", "whisperjav-gui.exe",
                "whisperjav-translate.exe", "whisperjav-upgrade.exe"):
        (prefix / "Scripts" / exe).write_bytes(b"x" * 2048)
    (prefix / "python.exe").write_bytes(b"x")
    monkeypatch.setattr(module, "sys", types.SimpleNamespace(prefix=str(prefix)))

    # Everything else Phase 6 calls that is not the import check
    monkeypatch.setattr(module, "check_ffmpeg", lambda: True)
    monkeypatch.setattr(module, "get_gpu_vram_mb", lambda: ("Test GPU", 12288))
    monkeypatch.setattr(module, "get_installation_size_gb", lambda: 9.0)

    class _FakeCompleted:
        returncode = 0
        stdout = "whisperjav 1.9.2"
        stderr = ""

    fake_subprocess = types.SimpleNamespace(
        run=lambda *a, **k: _FakeCompleted(),
        TimeoutExpired=subprocess.TimeoutExpired,
        CalledProcessError=subprocess.CalledProcessError,
    )
    monkeypatch.setattr(module, "subprocess", fake_subprocess)

    def set_imports(behaviour):
        """behaviour: display name -> 'ok' | 'fail' | 'timeout' (default 'ok')."""
        def fake_verify_import(module_name, display_name):
            outcome = behaviour.get(display_name, "ok")
            if outcome == "ok":
                return True, "1.0.0", False
            if outcome == "timeout":
                return False, "check did not finish within 120s", True
            return False, f"No module named '{module_name}'", False
        monkeypatch.setattr(module, "verify_import", fake_verify_import)

    return types.SimpleNamespace(module=module, lines=lines, set_imports=set_imports)


class TestPhase6Outcome:
    def test_a_healthy_install_reports_no_failures(self, phase6):
        phase6.set_imports({})
        passes, fails, warnings, failures = phase6.module.run_post_install_verification()

        assert fails == 0, f"a healthy install must not fail; failures={failures}"
        assert failures == []
        assert passes > 0

    @pytest.mark.parametrize("display", sorted(CORE_DISPLAY_NAMES))
    def test_any_core_package_that_will_not_import_is_a_failure(self, phase6, display):
        phase6.set_imports({display: "fail"})
        _passes, fails, _warnings, failures = phase6.module.run_post_install_verification()

        assert fails >= 1, f"{display} failing to import must fail the install"
        assert any(display in item for item in failures), (
            f"the failure list must name {display}, so the marker file can say what "
            f"went wrong. Got: {failures}"
        )

    def test_faster_whisper_is_one_of_the_core_checks(self, phase6):
        """v1.9.2: the installer did not install faster-whisper at all.

        Balanced, Fast, Faster and Kotoba all need it, so a missing
        faster-whisper must end the install rather than print one line.
        """
        phase6.set_imports({"Faster-Whisper": "fail"})
        _passes, fails, _warnings, failures = phase6.module.run_post_install_verification()

        assert fails >= 1
        assert any("Faster-Whisper" in item for item in failures), failures

    def test_a_timed_out_check_is_a_warning_not_a_failure(self, phase6):
        """A check that ran out of time has not shown that anything is wrong.

        The first import of torch off a freshly written site-packages, with
        antivirus reading every file, is slow. Commit 63936e1 raised this limit
        from 30 s to 120 s precisely because it was causing false failures, and
        since v1.9.2 a false failure aborts the install.
        """
        phase6.set_imports({"PyTorch": "timeout", "Faster-Whisper": "timeout"})
        _passes, fails, warnings, failures = phase6.module.run_post_install_verification()

        assert fails == 0, (
            f"a slow machine must not be told its install is broken; failures={failures}"
        )
        assert any("PyTorch" in w for w in warnings), warnings
        assert any("Faster-Whisper" in w for w in warnings), warnings

    def test_an_optional_package_that_is_absent_is_only_a_warning(self, phase6):
        """llama-cpp-python, ClearVoice and Transformers are optional features."""
        phase6.set_imports({
            "llama-cpp-python": "fail", "ClearVoice": "fail", "Transformers": "fail",
        })
        _passes, fails, warnings, failures = phase6.module.run_post_install_verification()

        assert fails == 0, f"optional packages must never end the install; {failures}"
        assert any("ClearVoice" in w for w in warnings), warnings

    def test_a_missing_critical_entry_point_is_a_failure(self, phase6, tmp_path):
        """Without whisperjav-gui.exe the desktop shortcut points at nothing."""
        phase6.set_imports({})
        (tmp_path / "prefix" / "Scripts" / "whisperjav-gui.exe").unlink()

        _passes, fails, _warnings, failures = phase6.module.run_post_install_verification()

        assert fails >= 1
        assert any("whisperjav-gui.exe" in item for item in failures), failures

    def test_a_missing_optional_entry_point_is_only_a_warning(self, phase6, tmp_path):
        phase6.set_imports({})
        (tmp_path / "prefix" / "Scripts" / "whisperjav-translate.exe").unlink()

        _passes, fails, warnings, failures = phase6.module.run_post_install_verification()

        assert fails == 0, failures
        assert any("whisperjav-translate.exe" in w for w in warnings), warnings

    def test_the_core_package_list_still_names_all_seven(self, phase6):
        """A package quietly dropped from the core list stops being checked."""
        checked = []

        def recording_verify_import(module_name, display_name):
            checked.append(display_name)
            return True, "1.0.0", False

        phase6.module.verify_import = recording_verify_import
        phase6.module.run_post_install_verification()

        missing = CORE_DISPLAY_NAMES - set(checked)
        assert not missing, f"these core packages are no longer verified: {missing}"
