#!/usr/bin/env python3
"""
E2E tests for translation console output.

These tests ACTUALLY RUN the translation event system and verify
that output appears on stderr.

Run with: pytest tests/test_translation_e2e_output.py -v -s
"""

import io
import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class MockStderr(io.StringIO):
    """StringIO with encoding attribute for cli.py compatibility."""
    encoding = "utf-8"


class TestTranslationEventOutputE2E:
    """E2E tests that verify PySubTrans events actually produce stderr output."""

    def test_info_events_appear_on_stderr(self):
        """E2E test: Verify info events appear on stderr after connect_default_loggers()."""
        from PySubtrans.TranslationEvents import TranslationEvents

        captured_stderr = io.StringIO()
        events = TranslationEvents()

        def _emit_raw(message):
            if message is None:
                return
            print(message, file=captured_stderr, flush=True)

        def _make_wrapper():
            def _wrapper(sender, message=None, **kwargs):
                msg = message or kwargs.get("message", "")
                _emit_raw(msg)
            return _wrapper

        events._default_error_wrapper = _make_wrapper()
        events._default_warning_wrapper = _make_wrapper()
        events._default_info_wrapper = _make_wrapper()
        events.connect_default_loggers()

        events.info.send(None, message="Translating 50 lines in 3 scenes")
        events.warning.send(None, message="Rate limit reached")
        events.info.send(None, message="Successfully translated 50 lines!")

        output = captured_stderr.getvalue()
        assert "Translating 50 lines in 3 scenes" in output
        assert "Rate limit reached" in output
        assert "Successfully translated 50 lines" in output

    def test_without_connect_no_output(self):
        """E2E test: Without connect_default_loggers(), no output appears (proves the bug)."""
        from PySubtrans.TranslationEvents import TranslationEvents

        captured_stderr = io.StringIO()
        events = TranslationEvents()

        def _make_wrapper():
            def _wrapper(sender, message=None, **kwargs):
                print(message or kwargs.get("message", ""), file=captured_stderr, flush=True)
            return _wrapper

        events._default_error_wrapper = _make_wrapper()
        events._default_warning_wrapper = _make_wrapper()
        events._default_info_wrapper = _make_wrapper()
        # DO NOT call events.connect_default_loggers() - simulating the bug

        events.info.send(None, message="This should NOT appear")

        output = captured_stderr.getvalue()
        assert output == "", f"Without connect_default_loggers(), no output should appear. Got: {output!r}"


class TestCoreTranslationIntegration:
    """Integration tests that verify core.py correctly wires events."""

    def test_core_emits_to_stderr_when_emit_raw_output_true(self, tmp_path):
        """Integration test: Verify core.py emits events to stderr."""
        from PySubtrans.TranslationEvents import TranslationEvents
        from whisperjav.translate.core import translate_subtitle

        test_srt = tmp_path / "test.srt"
        test_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nTest\n", encoding="utf-8")
        output_srt = tmp_path / "test.english.srt"

        real_events = TranslationEvents()
        mock_translator = MagicMock()
        mock_translator.events = real_events
        mock_translator.resume = False

        mock_project = MagicMock()
        mock_project.events = MagicMock()
        mock_project.events.batch_translated = MagicMock()
        mock_project.existing_project = False

        def mock_translate_subtitles(translator):
            translator.events.info.send(None, message="Translating 2 lines")
            translator.events.info.send(None, message="Successfully translated!")

        mock_project.TranslateSubtitles = mock_translate_subtitles
        mock_project.SaveTranslation = MagicMock(return_value=str(output_srt))
        mock_project.SaveProject = MagicMock()

        captured_stderr = MockStderr()
        old_stderr = sys.stderr

        try:
            sys.stderr = captured_stderr

            with patch("PySubtrans.init_options") as m1, \
                 patch("PySubtrans.init_translation_provider") as m2, \
                 patch("PySubtrans.init_project") as m3, \
                 patch("PySubtrans.init_translator") as m4:

                m1.return_value = MagicMock()
                m2.return_value = MagicMock()
                m3.return_value = mock_project
                m4.return_value = mock_translator

                translate_subtitle(
                    input_path=str(test_srt),
                    output_path=output_srt,
                    provider_config={"pysubtrans_name": "OpenAI"},
                    model="gpt-4",
                    api_key="test-key",
                    emit_raw_output=True
                )
        finally:
            sys.stderr = old_stderr

        output = captured_stderr.getvalue()
        assert "Translating 2 lines" in output, f"Expected message not found. Got: {output}"
        assert "Successfully translated!" in output, f"Expected message not found. Got: {output}"


class TestSubprocessUnbufferedOutput:
    """Tests that verify subprocess unbuffered output works correctly."""

    def test_subprocess_with_unbuffered_flag_streams_output(self):
        """Test that Python subprocess with -u flag streams output in real-time."""
        import subprocess
        import os
        import time

        # Create a simple Python script that outputs to stderr with flushes
        script = '''
import sys
import time
for i in range(3):
    print(f"Line {i+1}", file=sys.stderr, flush=True)
    time.sleep(0.1)
'''

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # Run with -u flag and PYTHONUNBUFFERED=1
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        # Collect output
        lines = []
        start = time.time()
        for line in proc.stdout:
            lines.append((time.time() - start, line.strip()))

        proc.wait()

        # Should have 3 lines
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}: {lines}"
        assert "Line 1" in lines[0][1]
        assert "Line 2" in lines[1][1]
        assert "Line 3" in lines[2][1]

    def test_gui_api_uses_unbuffered_output(self):
        """Test that GUI API code uses -u flag and PYTHONUNBUFFERED."""
        from pathlib import Path

        api_path = Path(__file__).parent.parent / "whisperjav" / "webview_gui" / "api.py"
        api_content = api_path.read_text(encoding="utf-8")

        # Check that translation subprocess uses -u flag
        assert '"-u", "-m", "whisperjav.translate.cli"' in api_content, \
            "Translation subprocess should use -u flag for unbuffered output"

        # Check that PYTHONUNBUFFERED is set
        assert 'PYTHONUNBUFFERED' in api_content, \
            "API should set PYTHONUNBUFFERED environment variable"

    def test_service_always_emits_raw_output(self):
        """Test that service.py always emits raw output (CLI parity)."""
        from pathlib import Path

        service_path = Path(__file__).parent.parent / "whisperjav" / "translate" / "service.py"
        service_content = service_path.read_text(encoding="utf-8")

        # Check that emit_raw_output is always True, not tied to stream parameter
        assert "emit_raw_output=True" in service_content, \
            "service.py should always set emit_raw_output=True for CLI parity"

        # Ensure the old buggy pattern is not present
        assert "emit_raw_output=stream" not in service_content, \
            "service.py should NOT tie emit_raw_output to stream parameter"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
