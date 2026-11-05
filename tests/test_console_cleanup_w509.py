"""The console clean-up of 1.10.1 (owner W509: way B and items 1-6 of
docs/plans/design/2026-11-04-console-output-cleanup-proposal.md).

- Way B: the log file records every line of WhisperJAV's logger, DEBUG included, whatever the console shows; each
  Ensemble pass writes its own file beside it.
- Item 1: ChronosJAV's steps print one line at every tenth of their VAD groups; the semantic engine's step and
  diagnosis lines are DEBUG, its environment warning stays a warning.
- Item 3: transformers' "Loading weights" bar only with --debug (download bars stay).
- Item 4: libraries' INFO lines (through a later logging.basicConfig) and their known start-up warnings stay off
  the console without --debug; other warnings show.
- Item 5: ChronosJAV writes its scene SRT files itself (stable-ts prints "Saved: <path>" for each).
"""

import importlib
import logging
from pathlib import Path

import pytest

wj_logger = importlib.import_module("whisperjav.utils.logger")   # the module (the package exports a Logger "logger")

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_logging():
    """Keep the root logger and the loggers these tests touch as they were."""
    root = logging.getLogger()
    saved_root = (list(root.handlers), root.level)
    names = ["whisperjav_test_w509", "httpx", "httpcore", "fireredvad", "nagisa_like", *wj_logger._KNOWN_LIBRARY_LINES]
    saved = {n: (logging.getLogger(n).level, list(logging.getLogger(n).filters)) for n in names}
    saved_filters = dict(wj_logger._quiet_state["filters"])
    yield
    root.handlers[:] = saved_root[0]
    root.setLevel(saved_root[1])
    for n, (level, filters) in saved.items():
        lg = logging.getLogger(n)
        lg.setLevel(level)
        lg.filters[:] = filters
    wj_logger._quiet_state["filters"] = saved_filters


class TestWayB:
    def test_the_log_file_records_debug_lines_the_console_does_not_show(self, tmp_path, capsys, clean_logging):
        log = tmp_path / "run.log"
        lg = wj_logger.setup_logger("whisperjav_test_w509", "INFO", str(log))
        lg.debug("a debug line")
        lg.info("an info line")
        for h in lg.handlers:
            h.flush()
        out = capsys.readouterr().out
        assert "an info line" in out and "a debug line" not in out
        text = log.read_text(encoding="utf-8")
        assert "a debug line" in text and "an info line" in text
        assert wj_logger.console_shows_debug("whisperjav_test_w509") is False
        for h in lg.handlers:
            h.close()

    def test_debug_on_the_console_is_still_debug(self, clean_logging):
        wj_logger.setup_logger("whisperjav_test_w509", "DEBUG", None)
        assert wj_logger.console_shows_debug("whisperjav_test_w509") is True

    def test_without_a_log_file_nothing_changes(self, capsys, clean_logging):
        lg = wj_logger.setup_logger("whisperjav_test_w509", "INFO", None)
        assert lg.level == logging.INFO
        lg.debug("not shown")
        assert "not shown" not in capsys.readouterr().out

    @pytest.mark.parametrize("log_file, n, expected", [
        ("D:/x/run.log", 1, "run.pass1.log"), ("D:/x/run.txt", 2, "run.pass2.txt"), ("D:/x/run", 1, "run.pass1.log"),
    ])
    def test_each_pass_writes_beside_the_runs_file(self, log_file, n, expected):
        assert Path(wj_logger.pass_log_file(log_file, n)).name == expected
        assert wj_logger.pass_log_file(None, n) is None

    def test_a_spawned_worker_gets_the_consoles_level_not_the_files(self, tmp_path, clean_logging):
        """Review F2: with a log file the logger is at DEBUG; balanced's ASR worker took that as its console level."""
        lg = wj_logger.setup_logger("whisperjav_test_w509", "INFO", str(tmp_path / "run.log"))
        try:
            assert lg.level == logging.DEBUG
            assert wj_logger.console_level_name("whisperjav_test_w509") == "INFO"
            assert wj_logger.has_log_file("whisperjav_test_w509") is True
        finally:
            for h in lg.handlers:
                h.close()
        wj_logger.setup_logger("whisperjav_test_w509", "DEBUG", None)
        assert wj_logger.console_level_name("whisperjav_test_w509") == "DEBUG"
        source = (REPO / "whisperjav" / "modules" / "asr_worker_proxy.py").read_text(encoding="utf-8")
        assert "log_level = console_level_name()" in source and 'getLogger("whisperjav").level' not in source

    def test_a_log_file_that_cannot_be_opened_does_not_stop_the_run(self, tmp_path, capsys, clean_logging):
        blocker = tmp_path / "a_file"
        blocker.write_text("x", encoding="utf-8")
        lg = wj_logger.setup_logger("whisperjav_test_w509", "INFO", str(blocker / "sub" / "run.log"))
        assert "could not be opened" in capsys.readouterr().out
        assert lg.level == logging.INFO and not wj_logger.has_log_file("whisperjav_test_w509")

    def test_the_passes_get_their_file(self):
        orch = (REPO / "whisperjav" / "ensemble" / "orchestrator.py").read_text(encoding="utf-8")
        worker = (REPO / "whisperjav" / "ensemble" / "pass_worker.py").read_text(encoding="utf-8")
        main = (REPO / "whisperjav" / "main.py").read_text(encoding="utf-8")
        assert "log_file=pass_log_file(self.log_file, pass_number)" in orch
        assert 'setup_logger("whisperjav", payload.log_level, payload.log_file)' in worker
        assert "log_file=args.log_file" in main


class TestLibraries:
    def test_nagisas_basicconfig_is_undone_after_the_import(self, capsys, clean_logging, monkeypatch):
        """Review F3: the root logger is left alone; only the handler an import adds is taken away again."""
        monkeypatch.setattr(wj_logger, "console_shows_debug", lambda name="whisperjav": False)
        root = logging.getLogger()
        root.handlers[:] = []
        root.setLevel(logging.WARNING)
        with wj_logger.keep_root_logger():
            logging.basicConfig(level=logging.INFO, format="%(message)s")   # what nagisa does at import
        assert root.handlers == [] and root.level == logging.WARNING
        logging.getLogger("nagisa_like").info("HTTP Request: GET ...")
        logging.getLogger("nagisa_like").warning("a real warning")
        err = capsys.readouterr().err
        assert "HTTP Request" not in err and err.count("a real warning") == 1

    def test_a_library_with_its_own_handler_prints_each_warning_once(self, capsys, clean_logging, monkeypatch):
        """Review F3: huggingface_hub has its own handler and propagates; a handler on the root printed it twice."""
        import sys

        monkeypatch.setattr(wj_logger, "console_shows_debug", lambda name="whisperjav": False)
        root = logging.getLogger()
        root.handlers[:] = []
        wj_logger.quiet_library_loggers(debug=False)
        hub = logging.getLogger("nagisa_like")
        own = logging.StreamHandler(sys.stderr)
        hub.addHandler(own)
        try:
            with wj_logger.keep_root_logger():
                logging.basicConfig(level=logging.INFO, format="%(message)s")
            hub.warning("a hub warning")
        finally:
            hub.removeHandler(own)
        assert capsys.readouterr().err.count("a hub warning") == 1

    def test_with_debug_the_import_keeps_its_handler(self, clean_logging, monkeypatch):
        monkeypatch.setattr(wj_logger, "console_shows_debug", lambda name="whisperjav": True)
        root = logging.getLogger()
        root.handlers[:] = []
        with wj_logger.keep_root_logger():
            logging.basicConfig(level=logging.INFO, format="%(message)s")
        assert len(root.handlers) == 1

    def test_the_qwen3_asr_imports_are_wrapped(self):
        source = (REPO / "whisperjav" / "modules" / "qwen_asr.py").read_text(encoding="utf-8")
        assert source.count("with keep_root_logger():") == 3
        assert "lastResort" not in (REPO / "whisperjav" / "utils" / "logger.py").read_text(encoding="utf-8")

    def test_known_lines_are_filtered_other_warnings_show(self, capsys, clean_logging):
        root = logging.getLogger()
        root.handlers[:] = []
        wj_logger.quiet_library_loggers(debug=False)
        rope = logging.getLogger("transformers.modeling_rope_utils")
        rope.warning("Unrecognized keys in `rope_parameters` for 'rope_type'='default': {...}")
        rope.warning("something else about rope")
        err = capsys.readouterr().err
        assert "Unrecognized keys" not in err and "something else about rope" in err

    def test_the_generation_flags_line_only_for_qwen3_asrs_flags(self, capsys, clean_logging):
        root = logging.getLogger()
        root.handlers[:] = []
        wj_logger.quiet_library_loggers(debug=False)
        gen = logging.getLogger("transformers.generation.configuration_utils")
        gen.warning("The following generation flags are not valid and may be ignored: ['temperature']. "
                    "Set `TRANSFORMERS_VERBOSITY=info` for more details.")
        gen.warning("The following generation flags are not valid and may be ignored: ['top_p', 'top_k']. "
                    "Set `TRANSFORMERS_VERBOSITY=info` for more details.")
        err = capsys.readouterr().err
        assert "['temperature']" not in err and "['top_p', 'top_k']" in err

    def test_fireredvad_only_its_short_segment_line_is_filtered(self, capsys, clean_logging):
        root = logging.getLogger()
        root.handlers[:] = []
        wj_logger.quiet_library_loggers(debug=False)
        logging.getLogger("fireredvad.core.vad_postprocessor").warning(
            "Unexpected short speech segment, check vad_postprocessor.py")
        logging.getLogger("fireredvad.vad").warning("Too long input, split every 18000 frames")
        err = capsys.readouterr().err
        assert "Unexpected short speech segment" not in err and "Too long input" in err
        assert logging.getLogger("fireredvad").level == logging.NOTSET   # the logger is not silenced

    def test_with_debug_every_line_shows(self, clean_logging):
        wj_logger.quiet_library_loggers(debug=False)
        wj_logger.quiet_library_loggers(debug=True)
        for name in wj_logger._KNOWN_LIBRARY_LINES:
            assert not any(isinstance(f, wj_logger._KnownLinesFilter) for f in logging.getLogger(name).filters)

    def test_quieting_is_not_done_at_import(self):
        """The translate command, the updater and bench call logging.basicConfig themselves."""
        source = (REPO / "whisperjav" / "utils" / "logger.py").read_text(encoding="utf-8")
        assert "\nlogger = setup_logger()" in source
        assert "quiet_library_loggers(" not in source.split("def setup_logger")[1]


class TestProgress:
    def test_one_line_at_every_tenth(self, caplog):
        from whisperjav.modules.subtitle_pipeline.orchestrator import _StepProgress

        caplog.set_level(logging.INFO, logger="whisperjav")
        lg = logging.getLogger("whisperjav")
        lg.addHandler(caplog.handler)
        try:
            p = _StepProgress("Text generation", "transcribed", 95)
            for _ in range(95):
                p.advance(1)
            p.advance(5)   # past the end: nothing more
        finally:
            lg.removeHandler(caplog.handler)
        lines = [r.getMessage() for r in caplog.records if "VAD groups" in r.getMessage()]
        assert lines[0] == "Text generation: 10 of 95 VAD groups transcribed (10%)"
        assert lines[-1] == "Text generation: 95 of 95 VAD groups transcribed (100%)"
        assert len(lines) == 10

    def test_no_line_without_groups(self, caplog):
        from whisperjav.modules.subtitle_pipeline.orchestrator import _StepProgress

        p = _StepProgress("Alignment", "done", 0)
        p.advance(3)
        assert not [r for r in caplog.records if "VAD groups" in r.getMessage()]

    @pytest.mark.parametrize("source, noun", [("vad-grouped", "VAD groups"), ("full-scene", "scenes"),
                                              ("srt-source", "frames"), ("manual", "frames")])
    def test_the_noun_follows_the_framer(self, source, noun):
        """Review F4: "VAD groups" only where a frame is one."""
        from types import SimpleNamespace

        from whisperjav.modules.subtitle_pipeline.orchestrator import frame_noun

        assert frame_noun([[], [SimpleNamespace(source=source)]]) == noun
        assert frame_noun([]) == "frames"

    def test_the_noun_is_in_the_line(self, caplog):
        from whisperjav.modules.subtitle_pipeline.orchestrator import _StepProgress

        caplog.set_level(logging.INFO, logger="whisperjav")
        lg = logging.getLogger("whisperjav")
        lg.addHandler(caplog.handler)
        try:
            _StepProgress("Text generation", "transcribed", 2, "scenes").advance(2)
        finally:
            lg.removeHandler(caplog.handler)
        assert "Text generation: 2 of 2 scenes transcribed (100%)" in [r.getMessage() for r in caplog.records]

    def test_only_what_is_done_is_counted(self):
        """Review F4: text generation counts the frames sent to the model (not those under 0.2 s, nor those with
        their own text); alignment says "done", not "aligned"."""
        source = (REPO / "whisperjav" / "modules" / "subtitle_pipeline" / "orchestrator.py").read_text(
            encoding="utf-8")
        assert "if f.text is None and f.duration >= MIN_MODEL_AUDIO_S" in source
        assert "progress.advance(len(gen_indices))" in source
        assert '_StepProgress("Alignment", "done",' in source and '_StepProgress("Alignment", "aligned"' not in source


class TestTexts:
    def test_semantic_engine_lines_are_debug_its_warning_stays(self):
        from whisperjav.modules.scene_detection_backends.semantic_adapter import _EngineLog

        seen = []

        class Target:
            def log(self, level, msg, *a, **k):
                seen.append((level, msg))

        log = _EngineLog(Target())
        log.log(logging.INFO, "[1/5 diag] Step 1: sf.info(x.wav)...")
        log.log(logging.INFO, "[Semantic Diag] *** WARNING: librosa 0.9 is NOT compatible")
        log.log(logging.ERROR, "engine failed")
        assert seen == [(logging.DEBUG, "[1/5 diag] Step 1: sf.info(x.wav)..."),
                        (logging.WARNING, "[Semantic Diag] *** WARNING: librosa 0.9 is NOT compatible"),
                        (logging.ERROR, "engine failed")]

    def test_chronosjav_writes_its_scene_files_itself(self):
        source = (REPO / "whisperjav" / "pipelines" / "qwen_pipeline.py").read_text(encoding="utf-8")
        assert "to_srt_vtt(str(" not in source and "to_srt_vtt(\n                    str(" not in source
        assert source.count("result.to_srt_vtt(word_level=False, segment_level=True, strip=True)") == 2
        assert "SRTPostProcessor(" not in source    # the unused one that loaded the hallucination lists
        assert "[QwenPipeline PID" not in source
        assert "aligner limit=180s" not in source

    def test_the_bars_follow_the_console(self):
        source = (REPO / "whisperjav" / "modules" / "qwen_asr.py").read_text(encoding="utf-8")
        assert source.count("disable=not console_shows_debug()") == 2
        assert source.count("hide_weight_loading_bar()") == 3   # Qwen3-ASR twice, the PyTorch aligner
        for other in (REPO / "whisperjav" / "modules" / "subtitle_pipeline" / "generators" / "anime_whisper.py",
                      REPO / "whisperjav" / "modules" / "transformers_asr.py"):
            assert "hide_weight_loading_bar()" in other.read_text(encoding="utf-8"), other.name

    def test_xxl_lines_have_no_worker_prefix(self):
        source = (REPO / "whisperjav" / "ensemble" / "pass_worker.py").read_text(encoding="utf-8")
        # The INFO lines (the console); the warning, error and debug lines keep the process number for diagnosis.
        for line in ('"Pass %s: BYOP XXL (exe=%s, model=%s)"', '"Pass %s [XXL] processing %s"',
                     '"Pass %s [XXL] %s -> %d subtitles in %.1fs"', '"Pass %s [XXL] done: %d completed, %d failed'):
            assert line in source, line

    def test_the_log_file_help_says_what_is_in_it(self):
        source = (REPO / "whisperjav" / "main.py").read_text(encoding="utf-8")
        assert "Not in the " in source and "run.log -> run.pass1.log" in source

    def test_the_weight_bar_hook_keeps_other_bars(self, monkeypatch):
        hf_logging = pytest.importorskip("transformers.utils.logging")
        if not hasattr(hf_logging, "set_tqdm_hook"):
            pytest.skip("this transformers has no tqdm hook")
        monkeypatch.setattr(wj_logger, "console_shows_debug", lambda name="whisperjav": False)
        previous = hf_logging.set_tqdm_hook(None)
        try:
            wj_logger.hide_weight_loading_bar()
            hook = hf_logging.set_tqdm_hook(None)
            made = []
            factory = lambda *a, **k: made.append(k.get("desc")) or "bar"   # noqa: E731
            assert isinstance(hook(factory, (), {"desc": "Loading weights"}), hf_logging.EmptyTqdm)
            assert hook(factory, (), {"desc": "Downloading"}) == "bar" and made == ["Downloading"]
        finally:
            hf_logging.set_tqdm_hook(previous)
