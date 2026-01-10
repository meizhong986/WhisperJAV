#!/usr/bin/env python3
"""
Frontend validation tests for WhisperJAV GUI Translation Console Output.

Tests that translation output is properly displayed in the GUI console by
verifying that all translation status polling loops call fetchLogs().

This test was created to prevent regression of the bug where
TranslateIntegrationManager.startStatusPolling() did not call fetchLogs(),
causing PySubTrans output to be captured but never displayed.

Run with: pytest tests/test_translation_console_output.py -v
"""

import re
import pytest
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def assets_dir():
    """Get the path to the webview_gui assets directory."""
    repo_root = Path(__file__).parent.parent
    return repo_root / "whisperjav" / "webview_gui" / "assets"


@pytest.fixture
def app_js(assets_dir):
    """Load the app.js content."""
    js_path = assets_dir / "app.js"
    assert js_path.exists(), f"app.js not found at {js_path}"
    return js_path.read_text(encoding="utf-8")


# =============================================================================
# Test: TranslatorManager (Standalone Translation Tab)
# =============================================================================

class TestTranslatorManager:
    """Tests for TranslatorManager translation console output."""

    def test_translator_manager_has_fetch_logs_method(self, app_js):
        """Test that TranslatorManager has a fetchLogs method."""
        # Find TranslatorManager definition
        assert "const TranslatorManager = {" in app_js, "TranslatorManager not found"

        # Find fetchLogs method within TranslatorManager
        # Look for pattern: async fetchLogs() or fetchLogs: async
        translator_section = app_js[app_js.find("const TranslatorManager = {"):]
        translator_section = translator_section[:translator_section.find("const TranslateIntegrationManager")]

        assert "fetchLogs" in translator_section, \
            "TranslatorManager should have a fetchLogs method"

    def test_translator_manager_polling_calls_fetch_logs(self, app_js):
        """Test that TranslatorManager.startStatusPolling calls fetchLogs."""
        # Find TranslatorManager section
        translator_section = app_js[app_js.find("const TranslatorManager = {"):]
        translator_section = translator_section[:translator_section.find("const TranslateIntegrationManager")]

        # Find startStatusPolling method definition (with opening brace)
        polling_match = re.search(r'startStatusPolling\(\)\s*\{', translator_section)
        assert polling_match, "TranslatorManager.startStatusPolling method not found"

        # Get the method body - find the setInterval block
        method_start = polling_match.start()
        method_section = translator_section[method_start:method_start + 2000]  # Get enough content

        # Check that fetchLogs is called within the setInterval callback
        assert "this.fetchLogs()" in method_section, \
            "TranslatorManager.startStatusPolling should call this.fetchLogs()"


# =============================================================================
# Test: TranslateIntegrationManager (Ensemble Mode Integration)
# =============================================================================

class TestTranslateIntegrationManager:
    """Tests for TranslateIntegrationManager translation console output."""

    def test_integration_manager_has_fetch_logs_method(self, app_js):
        """Test that TranslateIntegrationManager has a fetchLogs method."""
        # Find TranslateIntegrationManager definition
        assert "const TranslateIntegrationManager = {" in app_js, \
            "TranslateIntegrationManager not found"

        # Find fetchLogs method within TranslateIntegrationManager
        integration_section = app_js[app_js.find("const TranslateIntegrationManager = {"):]
        integration_section = integration_section[:integration_section.find("const TranslationSettingsModal")]

        assert "fetchLogs" in integration_section, \
            "TranslateIntegrationManager should have a fetchLogs method"

    def test_integration_manager_polling_calls_fetch_logs(self, app_js):
        """
        Test that TranslateIntegrationManager.startStatusPolling calls fetchLogs.

        This is the critical test - this was the bug that caused PySubTrans
        output to not be displayed in the GUI console when using Ensemble Mode
        with translation enabled.
        """
        # Find TranslateIntegrationManager section
        integration_section = app_js[app_js.find("const TranslateIntegrationManager = {"):]
        integration_section = integration_section[:integration_section.find("const TranslationSettingsModal")]

        # Find startStatusPolling method definition (with opening brace)
        polling_match = re.search(r'startStatusPolling\(\)\s*\{', integration_section)
        assert polling_match, "TranslateIntegrationManager.startStatusPolling method not found"

        # Get the method body
        method_start = polling_match.start()
        method_section = integration_section[method_start:method_start + 2000]  # Get enough content

        # Check that fetchLogs is called within the setInterval callback
        assert "this.fetchLogs()" in method_section, \
            "TranslateIntegrationManager.startStatusPolling should call this.fetchLogs() " \
            "to display PySubTrans output in the console"


# =============================================================================
# Test: Backend API has required methods
# =============================================================================

class TestBackendApiMethods:
    """Tests that backend API has required translation log methods."""

    @pytest.fixture
    def api_py(self):
        """Load the api.py content."""
        repo_root = Path(__file__).parent.parent
        api_path = repo_root / "whisperjav" / "webview_gui" / "api.py"
        assert api_path.exists(), f"api.py not found at {api_path}"
        return api_path.read_text(encoding="utf-8")

    def test_api_has_get_translation_logs(self, api_py):
        """Test that API has get_translation_logs method."""
        assert "def get_translation_logs" in api_py, \
            "API should have get_translation_logs method"

    def test_api_has_translation_log_queue(self, api_py):
        """Test that API has translation log queue."""
        assert "_translate_log_queue" in api_py, \
            "API should have _translate_log_queue for buffering translation output"

    def test_api_streams_translation_output(self, api_py):
        """Test that API has method to stream translation output."""
        assert "_stream_translation_output" in api_py, \
            "API should have _stream_translation_output method"


# =============================================================================
# Test: Core translation connects event wrappers to signals
# =============================================================================

class TestCoreTranslationEventWiring:
    """
    Tests that core.py properly wires PySubTrans event handlers.

    This is the CRITICAL backend test - the root cause of the bug was that
    the event wrappers were replaced but never connected to the Blinker signals.
    Without connect_default_loggers(), the wrappers sit idle and never receive
    any events, so translation output never reaches stderr.
    """

    @pytest.fixture
    def core_py(self):
        """Load the core.py content."""
        repo_root = Path(__file__).parent.parent
        core_path = repo_root / "whisperjav" / "translate" / "core.py"
        assert core_path.exists(), f"core.py not found at {core_path}"
        return core_path.read_text(encoding="utf-8")

    def test_core_connects_event_wrappers(self, core_py):
        """
        Test that core.py calls connect_default_loggers() after setting wrappers.

        The PySubTrans TranslationEvents class has:
        1. _default_*_wrapper attributes that hold wrapper functions
        2. connect_default_loggers() method that connects them to Blinker signals

        If we replace the wrappers but don't call connect_default_loggers(),
        the new wrappers are never connected and events go nowhere.
        """
        # Find the emit_raw_output block
        assert "if emit_raw_output:" in core_py, \
            "core.py should have emit_raw_output conditional block"

        # Check that wrapper assignments exist
        assert "_default_error_wrapper = _make_wrapper()" in core_py, \
            "core.py should set _default_error_wrapper"
        assert "_default_warning_wrapper = _make_wrapper()" in core_py, \
            "core.py should set _default_warning_wrapper"
        assert "_default_info_wrapper = _make_wrapper()" in core_py, \
            "core.py should set _default_info_wrapper"

        # CRITICAL: Check that connect_default_loggers() is called
        # This was the missing piece that caused the bug!
        assert "connect_default_loggers()" in core_py, \
            "core.py MUST call translator.events.connect_default_loggers() " \
            "to connect the wrappers to the Blinker signals. Without this, " \
            "event wrappers are replaced but never receive any events."

    def test_wrapper_connects_after_assignment(self, core_py):
        """
        Test that connect_default_loggers() is called AFTER wrapper assignments.

        The order matters: first replace the wrappers, then connect them.
        """
        wrapper_pos = core_py.find("_default_info_wrapper = _make_wrapper()")
        connect_pos = core_py.find("connect_default_loggers()")

        assert wrapper_pos > 0, "Wrapper assignment not found"
        assert connect_pos > 0, "connect_default_loggers() not found"
        assert connect_pos > wrapper_pos, \
            "connect_default_loggers() should be called AFTER wrapper assignments"


class TestConsoleOutputConsistency:
    """Tests for consistent console output handling across translation managers."""

    def test_both_managers_use_console_manager(self, app_js):
        """Test that both translation managers use ConsoleManager for output."""
        # TranslatorManager
        translator_section = app_js[app_js.find("const TranslatorManager = {"):]
        translator_section = translator_section[:translator_section.find("const TranslateIntegrationManager")]
        assert "ConsoleManager" in translator_section, \
            "TranslatorManager should use ConsoleManager"

        # TranslateIntegrationManager
        integration_section = app_js[app_js.find("const TranslateIntegrationManager = {"):]
        integration_section = integration_section[:integration_section.find("const TranslationSettingsModal")]
        assert "ConsoleManager" in integration_section, \
            "TranslateIntegrationManager should use ConsoleManager"

    def test_fetch_logs_uses_correct_api_call(self, app_js):
        """Test that fetchLogs methods use the correct API call."""
        # Both fetchLogs implementations should call get_translation_logs
        fetch_logs_pattern = r'async fetchLogs\(\)[^}]+pywebview\.api\.get_translation_logs'
        matches = re.findall(fetch_logs_pattern, app_js, re.DOTALL)

        assert len(matches) >= 2, \
            "Both TranslatorManager and TranslateIntegrationManager should have " \
            "fetchLogs methods that call pywebview.api.get_translation_logs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
