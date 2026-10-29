"""CrispASR's own log lines (owner W349, W350): without --debug, the GPU library's DEBUG messages (the "CUDA graph
warmup" lines) are dropped; its INFO, WARN and ERROR messages, and everything with --debug, reach the console."""

import types

import pytest

from whisperjav.crispasr import native_log
from whisperjav.crispasr.native_log import GGML_LOG_LEVEL_CONT, GGML_LOG_LEVEL_DEBUG, should_show

INFO, WARN, ERROR = 2, 3, 4


class TestShouldShow:
    def test_debug_messages_are_dropped_without_debug(self):
        assert should_show(GGML_LOG_LEVEL_DEBUG, INFO, debug=False) is False

    @pytest.mark.parametrize("level", [INFO, WARN, ERROR])
    def test_other_messages_pass_without_debug(self, level):
        assert should_show(level, GGML_LOG_LEVEL_DEBUG, debug=False) is True

    def test_a_continuation_follows_the_message_it_continues(self):
        assert should_show(GGML_LOG_LEVEL_CONT, GGML_LOG_LEVEL_DEBUG, debug=False) is False
        assert should_show(GGML_LOG_LEVEL_CONT, WARN, debug=False) is True

    @pytest.mark.parametrize("level", [GGML_LOG_LEVEL_DEBUG, INFO, WARN, ERROR, GGML_LOG_LEVEL_CONT])
    def test_everything_passes_with_debug(self, level):
        assert should_show(level, GGML_LOG_LEVEL_DEBUG, debug=True) is True


class TestRegistration:
    def test_no_ggml_library_leaves_ggml_alone_and_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.setattr(native_log, "_handler", None)
        fake = types.SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
        assert native_log.quiet_debug_lines(fake) is False
        assert native_log._handler is None

    def test_the_installed_crispasr_takes_the_handler_once(self, monkeypatch):
        crispasr = pytest.importorskip("crispasr")
        if native_log._ggml_base(native_log.Path(crispasr.__file__).parent) is None:
            pytest.skip("this crispasr build has no ggml-base library")
        assert native_log.quiet_debug_lines(crispasr) is True
        first = native_log._handler
        assert first is not None
        assert native_log.quiet_debug_lines(crispasr) is True
        assert native_log._handler is first   # registered once per process


@pytest.mark.parametrize("module", ["engine", "aligner", "timer"])
def test_each_place_that_loads_crispasr_registers_the_handler(module):
    from pathlib import Path

    source = (Path(native_log.__file__).parent / f"{module}.py").read_text(encoding="utf-8")
    assert "native_log.quiet_debug_lines(crispasr)" in source
