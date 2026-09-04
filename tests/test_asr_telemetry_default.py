"""Where the #394 telemetry file goes, and how each execution path decides it.

Telemetry is on by default (owner decision, 2026-09-04). These tests pin the
one rule that places the file, and the way the sync, async and ensemble paths
hand their settings to the pipeline.
"""

from pathlib import Path

import pytest

from whisperjav.utils.asr_telemetry import (
    TELEMETRY_SUBDIR,
    TELEMETRY_SUFFIX,
    resolve_telemetry_path,
)


class TestResolveTelemetryPath:
    def test_default_is_raw_subs_next_to_the_outputs(self, tmp_path):
        p = resolve_telemetry_path(None, True, tmp_path / "out", "MOVIE-001")
        assert p == tmp_path / "out" / TELEMETRY_SUBDIR / f"MOVIE-001{TELEMETRY_SUFFIX}"

    def test_disabled_is_none(self, tmp_path):
        assert resolve_telemetry_path(None, False, tmp_path, "x") is None
        assert resolve_telemetry_path(str(tmp_path), False, tmp_path, "x") is None

    def test_pass_tag_is_part_of_the_name(self, tmp_path):
        p = resolve_telemetry_path(None, True, tmp_path, "MOVIE-001", tag="pass2")
        assert p.name == f"MOVIE-001.pass2{TELEMETRY_SUFFIX}"

    def test_directory_override_gets_one_file_per_media(self, tmp_path):
        d = tmp_path / "tele"
        d.mkdir()
        assert resolve_telemetry_path(str(d), True, tmp_path / "out", "a") == d / f"a{TELEMETRY_SUFFIX}"
        # a path without a suffix is treated as a directory even if it does not exist yet
        nd = tmp_path / "later"
        assert resolve_telemetry_path(str(nd), True, tmp_path / "out", "a") == nd / f"a{TELEMETRY_SUFFIX}"

    def test_file_override_is_used_as_given_and_tagged_per_pass(self, tmp_path):
        f = tmp_path / "run.jsonl"
        assert resolve_telemetry_path(str(f), True, tmp_path / "out", "a") == f
        assert resolve_telemetry_path(str(f), True, tmp_path / "out", "a", tag="pass1") == tmp_path / "run.pass1.jsonl"

    def test_no_output_dir_and_no_override_is_none(self):
        assert resolve_telemetry_path(None, True, None, "a") is None


class TestRecordsReachDiskAsTheyHappen:
    """A run that dies mid-way must still leave what it had recorded."""

    def test_each_scene_is_on_disk_before_finalize(self, tmp_path):
        import json

        from whisperjav.utils.asr_telemetry import AsrTelemetry

        t = AsrTelemetry(tmp_path / "sub" / "m.asr_telemetry.jsonl", "m")
        t.record_scene(index=1, audio_duration_s=28.0, wall_s=1.0, segments=[], speech_detected=True, produced_output=True)
        t.record_scene(index=2, audio_duration_s=28.0, wall_s=1.2, segments=[], speech_detected=True, produced_output=False)
        # no finalize() yet -- simulate a crash here
        lines = (tmp_path / "sub" / "m.asr_telemetry.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert [json.loads(l)["scene"] for l in lines] == [1, 2]
        # finalize only logs; it must not duplicate or truncate
        assert t.finalize() == tmp_path / "sub" / "m.asr_telemetry.jsonl"
        assert len((tmp_path / "sub" / "m.asr_telemetry.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    def test_a_previous_runs_file_is_replaced_not_appended_to(self, tmp_path):
        from whisperjav.utils.asr_telemetry import AsrTelemetry

        p = tmp_path / "m.asr_telemetry.jsonl"
        p.write_text('{"scene": 99}\n', encoding="utf-8")
        t = AsrTelemetry(p, "m")
        t.record_scene(index=1, audio_duration_s=1.0, wall_s=0.1)
        assert p.read_text(encoding="utf-8").count("\n") == 1

    def test_unwritable_destination_never_raises(self, tmp_path):
        from whisperjav.utils.asr_telemetry import AsrTelemetry

        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        t = AsrTelemetry(blocker / "child" / "m.jsonl", "m")
        t.record_scene(index=1, audio_duration_s=1.0, wall_s=0.1)
        assert t.finalize() is None

    def test_write_is_an_alias_for_finalize(self):
        from whisperjav.utils.asr_telemetry import AsrTelemetry

        assert AsrTelemetry.write is AsrTelemetry.finalize


class TestPassWorkerConfiguresThePipeline:
    def test_points_at_the_files_pass_output_folder(self, tmp_path):
        from whisperjav.ensemble.pass_worker import _configure_telemetry

        class _P:
            pass

        p = _P()
        dest = tmp_path / "movies"
        _configure_telemetry(p, {"asr_telemetry": None, "asr_telemetry_enabled": True}, 2, dest)
        assert p.asr_telemetry_enabled is True
        assert p.asr_telemetry_tag == "pass2"
        assert Path(p.asr_telemetry_path) == dest / TELEMETRY_SUBDIR
        # and the pipeline-side rule then yields the per-pass file beside the pass output
        assert resolve_telemetry_path(p.asr_telemetry_path, p.asr_telemetry_enabled, None,
                                      "MOVIE", p.asr_telemetry_tag) == dest / TELEMETRY_SUBDIR / f"MOVIE.pass2{TELEMETRY_SUFFIX}"

    def test_user_override_and_disable_are_honoured(self, tmp_path):
        from whisperjav.ensemble.pass_worker import _configure_telemetry

        class _P:
            pass

        p = _P()
        _configure_telemetry(p, {"asr_telemetry": str(tmp_path / "t"), "asr_telemetry_enabled": False}, 1, tmp_path)
        assert p.asr_telemetry_enabled is False
        assert p.asr_telemetry_path == str(tmp_path / "t")

    def test_missing_keys_mean_on_by_default(self, tmp_path):
        from whisperjav.ensemble.pass_worker import _configure_telemetry

        class _P:
            pass

        p = _P()
        _configure_telemetry(p, {}, 1, tmp_path)
        assert p.asr_telemetry_enabled is True


class TestCliFlags:
    def test_flags_are_registered(self):
        import subprocess
        import sys

        r = subprocess.run([sys.executable, "-m", "whisperjav.main", "--help"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=180)
        assert "--asr-telemetry" in r.stdout and "--no-asr-telemetry" in r.stdout
        ok = subprocess.run([sys.executable, "-m", "whisperjav.main", "--no-asr-telemetry", "--help"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        assert ok.returncode == 0

    def test_version_is_1_9_2(self):
        from whisperjav.__version__ import __version__, __version_info__

        assert __version__ == "1.9.2"
        assert __version_info__["patch"] == 2
