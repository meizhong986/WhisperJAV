"""
Pytest configuration for WhisperJAV tests.

Registers custom markers used across test suites.
"""

import pytest


@pytest.fixture(autouse=True)
def _never_touch_real_desktop_shortcuts(monkeypatch):
    """No test may create or delete a real desktop shortcut (1.10.1 install test, finding 6).

    whisperjav.upgrade.update_desktop_shortcut() deletes every "WhisperJAV v*.lnk" on the user's real desktop and
    writes a new one; a test run on 2026-10-27 left "WhisperJAV v1.8.0.lnk" there, pointing into a pytest temp folder.
    Every test gets a stand-in that touches nothing and reports success; a test may still patch it further.

    A stand-in on the module is not enough: a second copy of whisperjav.upgrade (a test that re-imported it) still held
    the real function, and on 2026-11-09 a run deleted the real "WhisperJAV v1.10.1.lnk" through it. So the Windows
    shell itself is closed to tests as well: every copy of the function needs win32com.client.Dispatch("WScript.Shell")
    to find the desktop, and gets an error instead (the function then reports failure and touches nothing).
    """
    monkeypatch.setattr("whisperjav.upgrade.update_desktop_shortcut", lambda *args, **kwargs: True)
    try:
        import win32com.client
    except ImportError:
        return

    def _no_windows_shell(*args, **kwargs):
        raise RuntimeError("tests may not use the Windows shell (desktop shortcuts)")

    monkeypatch.setattr(win32com.client, "Dispatch", _no_windows_shell)


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests (may be slow)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (longer than 60 seconds)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
