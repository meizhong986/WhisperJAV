#!/usr/bin/env python3
"""
A shortfall inside a two-pass run reaches the run summary.

Cross-cutting rule 1 of the agreed error-handling table (owner, 2026-09-17):
anything that quietly fell short is named in the RUN SUMMARY rather than left in
a log line. On the plain single-file path that was straightforward. A two-pass
run is harder and matters more: each pass runs in its own process, so a note made
inside a pass has to survive being carried back before anyone can be told.

It also matters more because two-pass is currently the only way to choose a
clean-up from the command line -- there is no top-level --speech-enhancer.

The route, and what each test below pins:

  pipeline  -> metadata["summary"]["degradations"]   (every pipeline writes it)
  worker    -> FileResult.degradations               (survives the process)
  orchestra -> summary["degradations"], per pass     (labelled, only used passes)
  main.py   -> classify_output(degraded=...)         -> "suspect" in the summary

Run with: pytest tests/test_ensemble_carries_degradations.py -v
"""

from pathlib import Path

import pytest

from whisperjav.ensemble.pass_worker import FileResult
from whisperjav.utils.run_outcome import classify_output

SHORTFALL = "2 of 40 scenes went through without being cleaned up by htdemucs"


class TestTheWorkerCanCarryThem:
    def test_the_result_has_somewhere_to_put_them(self):
        result = FileResult(basename="clip", status="completed",
                            degradations=[SHORTFALL])
        assert result.degradations == [SHORTFALL]

    def test_it_defaults_to_empty_rather_than_none(self):
        # Everything downstream iterates it; None would be a crash on the
        # ordinary path where nothing went wrong.
        assert FileResult(basename="clip", status="completed").degradations == []

    def test_each_result_gets_its_own_list(self):
        # A shared default would leak one file's shortfalls onto every other.
        first = FileResult(basename="a", status="completed")
        second = FileResult(basename="b", status="completed")
        first.degradations.append(SHORTFALL)
        assert second.degradations == []

    def test_it_survives_the_trip_through_the_dropbox(self):
        # The worker hands results back as plain dicts via __dict__.
        carried = FileResult(basename="clip", status="completed",
                             degradations=[SHORTFALL]).__dict__
        assert carried["degradations"] == [SHORTFALL]


class TestTheOrchestratorGathersThem:
    """
    Runs the real merge/metadata method with plain dicts standing in for the
    worker results, so the gathering logic itself is exercised.
    """

    @staticmethod
    def merge(pass1, pass2, pass2_config=None, tmp_path=None):
        from whisperjav.ensemble.orchestrator import EnsembleOrchestrator

        # __init__ wires up real processing machinery; only the two small
        # methods under test are wanted here.
        orchestrator = EnsembleOrchestrator.__new__(EnsembleOrchestrator)
        # The attributes the merge path reads; everything else it needs comes
        # in as an argument.
        orchestrator.temp_dir = tmp_path
        orchestrator.output_dir = tmp_path
        media_info = {"path": str(tmp_path / "clip.mp4"), "basename": "clip"}

        metadata, _row = orchestrator._process_single_file_merge(
            media_info=media_info,
            pass1_results={"clip": pass1},
            pass2_results={"clip": pass2} if pass2 else {},
            pass1_config={"pipeline": "fidelity"},
            pass2_config=pass2_config,
            merge_strategy="pass1_primary",
            pass_languages={1: "ja", 2: "ja"},
        )
        return metadata

    @staticmethod
    def completed_pass(tmp_path, name, degradations=()):
        srt = tmp_path / name
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nline\n", encoding="utf-8")
        return {
            "basename": "clip",
            "status": "completed",
            "srt_path": str(srt),
            "subtitles": 1,
            "processing_time": 1.0,
            "degradations": list(degradations),
        }

    def test_a_shortfall_in_pass_one_is_reported_and_labelled(self, tmp_path):
        metadata = self.merge(
            self.completed_pass(tmp_path, "p1.srt", [SHORTFALL]),
            None, tmp_path=tmp_path)
        assert metadata["summary"]["degradations"] == [f"pass 1: {SHORTFALL}"]

    def test_nothing_is_reported_when_nothing_fell_short(self, tmp_path):
        metadata = self.merge(self.completed_pass(tmp_path, "p1.srt"),
                              None, tmp_path=tmp_path)
        assert metadata["summary"]["degradations"] == []

    def test_both_passes_are_reported_and_told_apart(self, tmp_path):
        metadata = self.merge(
            self.completed_pass(tmp_path, "p1.srt", ["pass one trouble"]),
            self.completed_pass(tmp_path, "p2.srt", ["pass two trouble"]),
            pass2_config={"pipeline": "qwen"},
            tmp_path=tmp_path)
        assert metadata["summary"]["degradations"] == [
            "pass 1: pass one trouble",
            "pass 2: pass two trouble",
        ]

    def test_a_failed_pass_two_contributes_nothing(self, tmp_path):
        # Its output is not used, so its shortfalls are not the user's problem;
        # the file is already marked degraded for the failure itself.
        metadata = self.merge(
            self.completed_pass(tmp_path, "p1.srt", ["pass one trouble"]),
            {"basename": "clip", "status": "failed", "error": "boom",
             "processing_time": 0.0, "degradations": ["pass two trouble"]},
            pass2_config={"pipeline": "qwen"},
            tmp_path=tmp_path)
        assert metadata["summary"]["degradations"] == ["pass 1: pass one trouble"]

    def test_an_older_result_without_the_field_does_not_crash(self, tmp_path):
        stale = self.completed_pass(tmp_path, "p1.srt")
        del stale["degradations"]
        metadata = self.merge(stale, None, tmp_path=tmp_path)
        assert metadata["summary"]["degradations"] == []


class TestTheUserIsToldAtTheEnd:
    """The last hop: what main.py does with what the orchestrator gathered."""

    @staticmethod
    def outcome(reasons, srt):
        # The same combination main.py performs for an ensemble file.
        return classify_output("clip.mp4", srt, 300.0,
                               degraded=bool(reasons),
                               degraded_reason="; ".join(reasons))

    @pytest.fixture
    def srt(self, tmp_path):
        path = tmp_path / "clip.ja.srt"
        lines = []
        for n in range(30):
            start, end = n * 10, n * 10 + 9
            lines.append(
                f"{n + 1}\n"
                f"00:{start // 60:02d}:{start % 60:02d},000 --> "
                f"00:{end // 60:02d}:{end % 60:02d},000\nline {n + 1}\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_a_shortfall_alone_makes_the_file_suspect(self, srt):
        result = self.outcome([f"pass 1: {SHORTFALL}"], srt)
        assert result.state == "suspect"
        assert "pass 1" in result.detail
        assert "htdemucs" in result.detail

    def test_a_clean_two_pass_run_is_still_done(self, srt):
        assert self.outcome([], srt).state == "done"

    def test_a_failed_pass_two_and_a_shortfall_are_both_named(self, srt):
        result = self.outcome(
            ["pass 2 failed; output is pass 1 alone", f"pass 1: {SHORTFALL}"], srt)
        assert result.state == "suspect"
        assert "pass 2 failed" in result.detail
        assert "cleaned up" in result.detail

    def test_the_subtitles_are_still_delivered(self, srt):
        result = self.outcome([f"pass 1: {SHORTFALL}"], srt)
        assert result.output == str(srt)
        assert result.subtitle_count == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
