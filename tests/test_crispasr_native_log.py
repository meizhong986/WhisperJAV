"""CrispASR's own log lines (owner W349, W350, W509 console item 6): without --debug, its DEBUG messages (the "CUDA
graph warmup" lines) and its INFO messages (the start-up details) are dropped; its WARN and ERROR messages, and
everything with --debug, reach the console."""

import types

import pytest

from whisperjav.crispasr import native_log
from whisperjav.crispasr.native_log import GGML_LOG_LEVEL_CONT, GGML_LOG_LEVEL_DEBUG, should_show

INFO, WARN, ERROR = 2, 3, 4


class TestShouldShow:
    @pytest.mark.parametrize("level", [GGML_LOG_LEVEL_DEBUG, INFO])
    def test_debug_and_info_messages_are_dropped_without_debug(self, level):
        assert should_show(level, WARN, debug=False) is False

    @pytest.mark.parametrize("level", [WARN, ERROR])
    def test_warnings_and_errors_pass_without_debug(self, level):
        assert should_show(level, GGML_LOG_LEVEL_DEBUG, debug=False) is True

    def test_a_continuation_follows_the_message_it_continues(self):
        assert should_show(GGML_LOG_LEVEL_CONT, GGML_LOG_LEVEL_DEBUG, debug=False) is False
        assert should_show(GGML_LOG_LEVEL_CONT, INFO, debug=False) is False
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


def test_the_handler_asks_the_console_not_the_log_file():
    """A log file records DEBUG lines (W509, way B); CrispASR's flood must still need --debug on the console."""
    from pathlib import Path

    source = Path(native_log.__file__).read_text(encoding="utf-8")
    assert "console_shows_debug()" in source and "isEnabledFor" not in source


def test_whisper_log_set_is_tried_first(monkeypatch):
    calls = []

    class Lib:
        @property
        def whisper_log_set(self):
            calls.append("whisper_log_set")
            return lambda handler, user: None

    monkeypatch.setattr(native_log, "_handler", None)
    monkeypatch.setattr(native_log.ctypes, "CDLL", lambda path: Lib())
    fake = types.SimpleNamespace(__file__="x/__init__.py",
                                 _binding=types.SimpleNamespace(_find_lib=lambda: "crispasr.dll"))
    assert native_log.quiet_debug_lines(fake) is True
    assert calls == ["whisper_log_set"]
    monkeypatch.setattr(native_log, "_handler", None)


class TestTheHandler:
    """The registered handler itself (review F5): what reaches the console, what reaches the log file."""

    @pytest.fixture
    def handler(self, monkeypatch, tmp_path, capsys):
        import logging

        from whisperjav.utils.logger import setup_logger

        captured = {}

        def set_log(callback, _user):
            captured["callback"] = callback

        set_log.argtypes = None
        monkeypatch.setattr(native_log, "_handler", None)
        monkeypatch.setattr(native_log, "_last_level", 0)
        monkeypatch.setattr(native_log, "_last_shown", True)
        monkeypatch.setattr(native_log, "_pending", [])
        monkeypatch.setattr(native_log, "_set_log_function", lambda module: set_log)
        log_path = tmp_path / "run.log"
        setup_logger("whisperjav", "INFO", str(log_path))
        assert native_log.quiet_debug_lines(types.SimpleNamespace()) is True

        def send(level, text):
            captured["callback"](level, text.encode("utf-8"), None)

        def file_text():
            for h in logging.getLogger("whisperjav").handlers:
                if isinstance(h, logging.FileHandler):
                    h.flush()
            return log_path.read_text(encoding="utf-8")

        yield send, file_text, capsys
        for h in list(logging.getLogger("whisperjav").handlers):
            h.close()
        setup_logger("whisperjav", "INFO", None)
        monkeypatch.setattr(native_log, "_handler", None)

    def test_an_info_line_goes_to_the_log_file_not_the_console(self, handler):
        send, file_text, capsys = handler
        send(INFO, "whisper_init_with_params_no_state: use gpu = 1\n")
        assert "use gpu" not in capsys.readouterr().err
        assert "CrispASR: whisper_init_with_params_no_state: use gpu = 1" in file_text()

    def test_an_info_line_in_parts_is_one_line_of_the_log_file(self, handler):
        send, file_text, capsys = handler
        send(INFO, "whisper_model_load: loading")
        send(GGML_LOG_LEVEL_CONT, " model ... done\n")
        assert capsys.readouterr().err == ""
        assert "CrispASR: whisper_model_load: loading model ... done" in file_text()

    @pytest.mark.parametrize("line", ["whisper_backend_init_gpu: no GPU found\n", "DECODER FAILED\n"])
    def test_no_gpu_found_and_decoder_failed_stay_on_the_console(self, handler, line):
        send, _file_text, capsys = handler
        send(INFO, line)
        assert line in capsys.readouterr().err

    def test_a_debug_line_goes_nowhere(self, handler):
        send, file_text, capsys = handler
        send(GGML_LOG_LEVEL_DEBUG, "ggml_backend_cuda_graph_compute: CUDA graph warmup complete\n")
        send(GGML_LOG_LEVEL_CONT, "more\n")
        assert capsys.readouterr().err == ""
        assert "CUDA graph" not in file_text() and "more" not in file_text()

    def test_a_warning_reaches_the_console(self, handler):
        send, _file_text, capsys = handler
        send(WARN, "crispasr: WARNING: something\n")
        assert "WARNING: something" in capsys.readouterr().err

    @pytest.mark.parametrize("line", ["whisper_backend_init_gpu: no GPU found\n"])
    def test_a_line_on_the_console_is_in_the_log_file_too(self, handler, line):
        send, file_text, capsys = handler
        send(INFO, line)
        assert line in capsys.readouterr().err
        assert "CrispASR: whisper_backend_init_gpu: no GPU found" in file_text()
