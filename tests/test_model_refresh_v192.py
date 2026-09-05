"""
v1.9.2 CFF1: periodic recogniser refresh between scenes.

Owner decisions (2026-09-05): budget in minutes of SCENE audio, default 20 (D2);
Balanced hosts its CTranslate2 model in a child process and a refresh is a fresh
child (D5); Fidelity reloads in-process.

Layers covered:
1. ``ModelRefreshPolicy`` — pure accounting.
2. ``RemoteFasterWhisperASR`` against a fake recogniser loaded in a real spawned
   worker (``tests/fake_asr_for_proxy.py``): handshake, per-scene facts, refresh
   → new PID, statistics carried across generations, learned compute type, a
   native worker death, and the give-up rule.
3. The CLI flag (parse, validation, ``--dump-params`` echo).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from whisperjav.utils.model_refresh import (
    DEFAULT_MODEL_REFRESH_AUDIO_MINUTES,
    ModelRefreshPolicy,
)


# ─── 1. policy ───────────────────────────────────────────────────────────────

class TestPolicy:
    def test_default_is_twenty_minutes(self):
        assert DEFAULT_MODEL_REFRESH_AUDIO_MINUTES == 20.0

    def test_zero_disables(self):
        p = ModelRefreshPolicy.from_minutes(0)
        p.record(10_000)
        assert not p.enabled and not p.due()

    def test_due_when_scene_audio_reaches_budget(self):
        p = ModelRefreshPolicy.from_minutes(1)      # 60 s
        p.record(25); p.record(25)
        assert not p.due()
        p.record(10)                                 # 60 s exactly
        assert p.due()

    def test_reset_starts_a_new_generation(self):
        p = ModelRefreshPolicy.from_minutes(1)
        p.record(90)
        assert p.due() and p.epoch == 1 and p.refresh_count == 0
        p.reset()
        assert not p.due() and p.epoch == 2 and p.refresh_count == 1
        assert p.consumed_audio_s == 0.0

    def test_garbage_input_is_ignored(self):
        p = ModelRefreshPolicy.from_minutes("nope")
        assert not p.enabled
        q = ModelRefreshPolicy.from_minutes(1)
        q.record(None); q.record(-5); q.record("x")
        assert q.consumed_audio_s == 0.0


# ─── 2. worker proxy with a fake recogniser ───────────────────────────────────

FAKE = "tests.fake_asr_for_proxy:FakeASR"


def _proxy(tmp_path: Path, minutes: float, **kw):
    from whisperjav.modules.asr_worker_proxy import RemoteFasterWhisperASR
    cfg = {
        "model_config": {"model_name": "fake-large", "compute_type": "float16", "device": "cpu"},
        "params": {"decoder": {}, "provider": {}, "vad": {}, "speech_segmenter": {"backend": "none"}},
        "task": "transcribe",
        "tracer": None,
    }
    return RemoteFasterWhisperASR(cfg, ModelRefreshPolicy.from_minutes(minutes), asr_class=FAKE,
                                  log_level="WARNING", **kw)


def _pid_in(srt: Path) -> int:
    text = srt.read_text(encoding="utf-8")
    return int(text.split("pid=")[1].split()[0])


@pytest.mark.slow
class TestRemoteASR:
    def test_handshake_and_surface(self, tmp_path):
        asr = _proxy(tmp_path, 20)
        try:
            assert asr.model_name == "fake-large"
            assert asr.compute_type == "int8"           # learned from the worker
            assert asr.get_segmenter_name() == "fake-seg"
            assert asr.epoch == 1 and asr.refresh_count == 0
            out = asr.transcribe_to_srt(tmp_path / "a.wav", tmp_path / "a.srt", task="transcribe")
            assert out.exists()
            assert _pid_in(out) == asr.worker_pid != None
            assert asr.get_last_vad_segments() == [{"start_sec": 0.0, "end_sec": 1.0}]
            assert asr.get_last_decode_stats()[0]["pid"] == asr.worker_pid
            assert asr.get_filter_statistics()["logprob_filtered"] == 1
        finally:
            asr.shutdown()

    def test_refresh_replaces_the_worker_and_keeps_statistics(self, tmp_path):
        asr = _proxy(tmp_path, 1)   # 60 s of scene audio
        try:
            asr.transcribe_to_srt(tmp_path / "s1.wav", tmp_path / "s1.srt")
            asr.record_audio(40)
            pid1 = _pid_in(tmp_path / "s1.srt")
            asr.transcribe_to_srt(tmp_path / "s2.wav", tmp_path / "s2.srt")   # budget not yet spent
            asr.record_audio(30)                                              # 70 s >= 60 s
            assert _pid_in(tmp_path / "s2.srt") == pid1
            assert asr.policy.due()
            asr.transcribe_to_srt(tmp_path / "s3.wav", tmp_path / "s3.srt")   # refresh happens first
            pid3 = _pid_in(tmp_path / "s3.srt")
            assert pid3 != pid1
            assert asr.epoch == 2 and asr.refresh_count == 1
            assert not asr.policy.due()
            # 2 scenes on generation 1 + 1 on generation 2
            assert asr.get_filter_statistics()["logprob_filtered"] == 3
            # the learned compute type is carried into the new generation
            assert asr.compute_type == "int8"
        finally:
            asr.shutdown()

    def test_reset_statistics_spans_generations(self, tmp_path):
        asr = _proxy(tmp_path, 1)
        try:
            asr.transcribe_to_srt(tmp_path / "s1.wav", tmp_path / "s1.srt")
            asr.record_audio(120)
            asr.transcribe_to_srt(tmp_path / "s2.wav", tmp_path / "s2.srt")   # refreshed
            assert asr.get_filter_statistics()["logprob_filtered"] == 2
            asr.reset_statistics()
            assert asr.get_filter_statistics().get("logprob_filtered", 0) == 0
        finally:
            asr.shutdown()

    def test_ordinary_scene_error_does_not_kill_the_worker(self, tmp_path):
        asr = _proxy(tmp_path, 20)
        try:
            pid = asr.worker_pid
            with pytest.raises(RuntimeError, match="synthetic scene failure"):
                asr.transcribe_to_srt(tmp_path / "RAISE.wav", tmp_path / "r.srt")
            assert asr.get_last_vad_segments() == []
            asr.transcribe_to_srt(tmp_path / "ok.wav", tmp_path / "ok.srt")
            assert _pid_in(tmp_path / "ok.srt") == pid
        finally:
            asr.shutdown()

    def test_native_death_fails_the_scene_then_respawns(self, tmp_path):
        from whisperjav.modules.asr_worker_proxy import AsrWorkerDied
        asr = _proxy(tmp_path, 20)
        try:
            pid = asr.worker_pid
            asr.transcribe_to_srt(tmp_path / "before.wav", tmp_path / "before.srt")
            assert asr.get_filter_statistics()["logprob_filtered"] == 1
            with pytest.raises(AsrWorkerDied) as ei:
                asr.transcribe_to_srt(tmp_path / "CRASH.wav", tmp_path / "c.srt")
            assert ei.value.exit_code == 3
            asr.transcribe_to_srt(tmp_path / "ok.wav", tmp_path / "ok.srt")   # fresh worker
            assert _pid_in(tmp_path / "ok.srt") != pid
            # the restarted worker is a new generation for telemetry, but not a refresh
            assert asr.epoch == 2 and asr.refresh_count == 0
            assert asr.policy.consumed_audio_s == 0.0
            # the dead generation's completed scene still counts in the file totals
            assert asr.get_filter_statistics()["logprob_filtered"] == 2
        finally:
            asr.shutdown()

    def test_gives_up_after_consecutive_deaths(self, tmp_path):
        from whisperjav.modules.asr_worker_proxy import AsrWorkerDied
        asr = _proxy(tmp_path, 20, max_consecutive_deaths=2)
        try:
            with pytest.raises(AsrWorkerDied):
                asr.transcribe_to_srt(tmp_path / "CRASH1.wav", tmp_path / "c1.srt")
            with pytest.raises(RuntimeError, match="died 2 times in a row"):
                asr.transcribe_to_srt(tmp_path / "CRASH2.wav", tmp_path / "c2.srt")
            with pytest.raises(RuntimeError, match="giving up"):
                asr.transcribe_to_srt(tmp_path / "ok.wav", tmp_path / "ok.srt")
            # the next FILE starts clean: the pipeline calls reset_statistics per file
            asr.reset_statistics()
            asr.transcribe_to_srt(tmp_path / "next_file.wav", tmp_path / "nf.srt")
            assert (tmp_path / "nf.srt").exists()
        finally:
            asr.shutdown()

    def test_shutdown_ends_the_worker(self, tmp_path):
        asr = _proxy(tmp_path, 20)
        proc = asr._proc
        asr.shutdown()
        assert not proc.is_alive()
        assert proc.exitcode == 0
        asr.shutdown()   # idempotent

    def test_bad_model_fails_at_construction(self, tmp_path):
        from whisperjav.modules.asr_worker_proxy import RemoteFasterWhisperASR
        cfg = {"model_config": {}, "params": {}, "task": "transcribe"}
        with pytest.raises(RuntimeError, match="could not load the model"):
            RemoteFasterWhisperASR(cfg, ModelRefreshPolicy.from_minutes(20),
                                   asr_class="tests.fake_asr_for_proxy:DoesNotExist", log_level="WARNING")


# ─── 3. CLI ──────────────────────────────────────────────────────────────────

def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "whisperjav.main", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )


@pytest.mark.slow
class TestCli:
    def test_flag_parses_and_is_documented(self):
        proc = _run("--model-refresh-audio-minutes", "5", "--help")
        assert proc.returncode == 0
        assert "--model-refresh-audio-minutes" in proc.stdout
        assert "Default: 20" in proc.stdout

    def test_negative_is_rejected_at_startup(self, tmp_path):
        proc = _run("--model-refresh-audio-minutes", "-1", "--dump-params", str(tmp_path / "d.json"),
                    "--mode", "balanced")
        assert proc.returncode == 2
        assert "--model-refresh-audio-minutes" in proc.stderr

    def test_dump_echoes_default_and_override(self, tmp_path):
        out = tmp_path / "d.json"
        _run("--dump-params", str(out), "--mode", "balanced")
        assert json.loads(out.read_text(encoding="utf-8"))["cli_args"]["model_refresh_audio_minutes"] == 20.0
        _run("--dump-params", str(out), "--mode", "fidelity", "--model-refresh-audio-minutes", "0")
        assert json.loads(out.read_text(encoding="utf-8"))["cli_args"]["model_refresh_audio_minutes"] == 0.0
