"""
Child-process recogniser for the Balanced pipeline (v1.9.2, owner CFF1 / D5).

Why a child process
-------------------
The Balanced pipeline's CTranslate2 model must never be destroyed inside the main
process: deleting or garbage-collecting it crashes natively on Windows
(0xC0000409 / 0xC0000005 — #125, commit 0874e91, the 2026-09-04 async death), which
is why ``balanced_pipeline.py`` keeps one immortal instance per process. A periodic
"unload and reload fresh" (owner CFF1) is therefore done the only safe way the code
base already relies on for ensemble passes: the model lives in a spawned worker that
ends with ``os._exit`` (no destructor), and a *refresh* is a fresh worker.

What this module provides
-------------------------
``RemoteFasterWhisperASR`` exposes exactly the members the Balanced scene loop uses
on ``FasterWhisperProASR`` (``transcribe_to_srt``, ``get_last_vad_segments``,
``get_last_decode_stats``, ``get_filter_statistics``, ``get_segmenter_name``,
``reset_statistics``, ``model_name``, ``compute_type``) plus ``record_audio`` (feeds
the refresh budget) and ``shutdown``. The worker serves every file of a batch until
the budget is spent, so today's model reuse across files is preserved; the refresh
happens between two scenes, never inside one.

Telemetry, the #394 streak, the progress bar and metadata stay in the parent — the
worker only transcribes and reports per-scene facts back over a Pipe.

Import policy: this module must stay light (no torch / faster-whisper at import
time). The heavy imports happen inside the worker entry function only.
"""
from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from whisperjav.utils.logger import logger
from whisperjav.utils.model_refresh import ModelRefreshPolicy

# "module:Class" of the recogniser the worker hosts. Tests inject a fake.
DEFAULT_ASR_CLASS = "whisperjav.modules.faster_whisper_pro_asr:FasterWhisperProASR"

# Consecutive worker deaths (or failed restarts) tolerated before the file is failed.
MAX_CONSECUTIVE_WORKER_DEATHS = 3


class AsrWorkerDied(RuntimeError):
    """The recogniser worker exited while a request was outstanding."""

    def __init__(self, exit_code: Optional[int], during: str):
        self.exit_code = exit_code
        self.during = during
        super().__init__(
            f"ASR worker process died (exit code {exit_code}) during {during}"
        )


def _load_class(spec: str):
    module_name, _, class_name = spec.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def _asr_worker_main(conn, payload: Dict[str, Any]) -> None:  # pragma: no cover - runs in a child
    """Worker entry point: load the recogniser once, serve requests, never destruct.

    Protocol (parent -> worker -> parent):
        ("transcribe", audio_path, srt_path, task) -> ("ok", facts) | ("err", message)
        ("reset_stats",)                            -> ("ok", None)
        ("exit",)                                   -> process ends with os._exit(0)
    First message from the worker is ("ready", info) or ("init_error", message).
    """
    # Same bootstrap as the ensemble pass worker (pass_worker.run_pass_worker).
    os.environ["WHISPERJAV_SUBPROCESS_WORKER"] = "1"
    try:
        from whisperjav.utils.model_loader import patch_hf_hub_downloads
        patch_hf_hub_downloads()
    except Exception:  # noqa: BLE001 - resilience patch is best-effort
        pass
    from whisperjav.utils.logger import setup_logger
    setup_logger("whisperjav", payload.get("log_level") or "INFO")
    from whisperjav.utils.parameter_tracer import NullTracer, create_tracer
    trace_path = payload.get("trace_file_path")
    tracer = create_tracer(trace_path, append=True) if trace_path else NullTracer()

    try:
        asr_cls = _load_class(payload.get("asr_class") or DEFAULT_ASR_CLASS)
        asr = asr_cls(
            model_config=payload["model_config"],
            params=payload["params"],
            task=payload["task"],
            tracer=tracer,
        )
    except Exception as exc:  # noqa: BLE001 - reported to the parent, then exit
        try:
            conn.send(("init_error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        os._exit(1)

    def _call(name: str, default):
        fn = getattr(asr, name, None)
        try:
            return fn() if callable(fn) else default
        except Exception:  # noqa: BLE001 - diagnostics must not fail a scene
            return default

    conn.send((
        "ready",
        {
            "pid": os.getpid(),
            "model_name": getattr(asr, "model_name", None),
            "compute_type": getattr(asr, "compute_type", None),
            "segmenter_name": _call("get_segmenter_name", "none"),
        },
    ))

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break  # parent is gone
        cmd = msg[0]
        if cmd == "transcribe":
            _, audio_path, srt_path, task = msg
            t0 = time.time()
            try:
                asr.transcribe_to_srt(audio_path, srt_path, task=task)
                conn.send((
                    "ok",
                    {
                        "wall_s": time.time() - t0,
                        "vad_segments": _call("get_last_vad_segments", []),
                        "decode_stats": _call("get_last_decode_stats", []),
                        "filter_statistics": _call("get_filter_statistics", {}),
                    },
                ))
            except Exception as exc:  # noqa: BLE001 - one scene failing must not kill the worker
                conn.send(("err", f"{type(exc).__name__}: {exc}"))
        elif cmd == "reset_stats":
            _call("reset_statistics", None)
            conn.send(("ok", None))
        elif cmd == "exit":
            break

    # Nuclear exit: skip Python shutdown so the CTranslate2 destructor never runs.
    try:
        tracer.close()
    except Exception:
        pass
    try:
        logging.shutdown()
    except Exception:
        pass
    os._exit(0)


class RemoteFasterWhisperASR:
    """Drop-in for ``FasterWhisperProASR`` inside ``BalancedPipeline``'s scene loop,
    backed by a child process that is replaced when the refresh budget is spent."""

    def __init__(
        self,
        asr_config: Dict[str, Any],
        policy: ModelRefreshPolicy,
        *,
        trace_file_path: Optional[str] = None,
        log_level: Optional[str] = None,
        asr_class: str = DEFAULT_ASR_CLASS,
        max_consecutive_deaths: int = MAX_CONSECUTIVE_WORKER_DEATHS,
    ):
        cfg = dict(asr_config)
        tracer = cfg.pop("tracer", None)
        if trace_file_path is None and tracer is not None and hasattr(tracer, "output_path"):
            trace_file_path = str(tracer.output_path)
        self._model_config: Dict[str, Any] = dict(cfg["model_config"])
        self._params = cfg["params"]
        self._task = cfg.get("task", "transcribe")
        self._trace_file_path = trace_file_path
        if log_level is None:
            lvl = logging.getLogger("whisperjav").level
            log_level = logging.getLevelName(lvl) if lvl else "INFO"
        self._log_level = log_level
        self._asr_class = asr_class
        self._max_deaths = max(1, int(max_consecutive_deaths))

        self.policy = policy
        self.model_name: str = self._model_config.get("model_name", "large-v2")
        self.compute_type: str = self._model_config.get("compute_type", "auto")
        self._segmenter_name: str = "none"

        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None
        self._consecutive_deaths = 0
        self._last_vad_segments: List[Dict] = []
        self._last_decode_stats: List[Dict] = []
        self._filter_stats_done: Dict[str, int] = {}   # from retired workers
        self._filter_stats_live: Dict[str, int] = {}   # from the current worker

        self._start()  # eager, like FasterWhisperProASR: a bad model fails now

    # ------------------------------------------------------------------ lifecycle
    @property
    def epoch(self) -> int:
        return self.policy.epoch

    @property
    def refresh_count(self) -> int:
        return self.policy.refresh_count

    @property
    def worker_pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _payload(self) -> Dict[str, Any]:
        return {
            "model_config": self._model_config,
            "params": self._params,
            "task": self._task,
            "trace_file_path": self._trace_file_path,
            "log_level": self._log_level,
            "asr_class": self._asr_class,
        }

    def _start(self) -> None:
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=_asr_worker_main,
            args=(child_conn, self._payload()),
            name="whisperjav-asr-worker",
            daemon=True,
        )
        proc.start()
        child_conn.close()
        self._proc, self._conn = proc, parent_conn
        kind, info = self._recv("model load")
        if kind == "init_error":
            self._proc.join(timeout=5)
            raise RuntimeError(f"ASR worker could not load the model: {info}")
        if kind != "ready":
            raise RuntimeError(f"ASR worker sent an unexpected first message: {kind!r}")
        if info.get("model_name"):
            self.model_name = info["model_name"]
        learned = info.get("compute_type")
        if learned and learned != self._model_config.get("compute_type"):
            # e.g. the int8 VRAM fallback fired: start the next generation there
            # instead of re-failing float16 on every refresh.
            self._model_config["compute_type"] = learned
        if learned:
            self.compute_type = learned
        self._segmenter_name = info.get("segmenter_name") or "none"
        logger.info(
            "ASR worker ready (pid %s, generation %d): %s on %s",
            proc.pid, self.policy.epoch, self.model_name, self.compute_type,
        )

    def _stop(self, timeout_s: float = 30.0) -> None:
        proc, conn = self._proc, self._conn
        self._proc, self._conn = None, None
        if conn is not None:
            try:
                if proc is not None and proc.is_alive():
                    conn.send(("exit",))
            except Exception:  # noqa: BLE001
                pass
        if proc is not None:
            proc.join(timeout=timeout_s)
            if proc.is_alive():
                logger.warning("ASR worker %s did not exit in %.0fs; terminating", proc.pid, timeout_s)
                proc.terminate()
                proc.join(timeout=5)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def _retire_live_stats(self) -> None:
        for k, v in (self._filter_stats_live or {}).items():
            self._filter_stats_done[k] = self._filter_stats_done.get(k, 0) + int(v or 0)
        self._filter_stats_live = {}

    def _refresh(self) -> None:
        logger.info(
            "Model refresh %d: unloading the recogniser after %.1f min of scene audio "
            "(budget %.0f min) and loading a fresh instance (scene boundary).",
            self.policy.refresh_count + 1, self.policy.consumed_minutes, self.policy.budget_minutes,
        )
        self._retire_live_stats()
        self._stop()
        self.policy.reset()
        # v1.9.2: clear the per-scene stats BEFORE the restart. _start() can raise
        # (the fresh model may not load), and that exception propagates out of
        # transcribe_to_srt -- so without this the caller would attribute the
        # PREVIOUS scene's decode statistics to the scene that failed.
        self._last_decode_stats = []
        self._last_vad_segments = []
        self._start()

    def shutdown(self) -> None:
        """End the worker (no destructor runs in this process). Idempotent."""
        self._retire_live_stats()
        self._stop()

    # ------------------------------------------------------------------ transport
    def _recv(self, during: str, timeout: Optional[float] = None) -> Tuple[str, Any]:
        deadline = None if timeout is None else time.time() + timeout
        while True:
            try:
                if self._conn.poll(1.0):
                    return self._conn.recv()
            except (EOFError, OSError):
                raise AsrWorkerDied(self._exit_code(), during)
            if not self._alive():
                # Drain anything that was written just before the exit.
                try:
                    if self._conn.poll(0):
                        return self._conn.recv()
                except (EOFError, OSError):
                    pass
                raise AsrWorkerDied(self._exit_code(), during)
            if deadline is not None and time.time() > deadline:
                raise RuntimeError(f"ASR worker did not respond within {timeout:.0f}s during {during}")

    def _request(self, msg: tuple, during: str) -> Tuple[str, Any]:
        try:
            self._conn.send(msg)
        except (EOFError, OSError, BrokenPipeError):
            raise AsrWorkerDied(self._exit_code(), during)
        return self._recv(during)

    def _exit_code(self) -> Optional[int]:
        """Exit code of a worker that just died (reaped first, so it is not None)."""
        proc = self._proc
        if proc is None:
            return None
        try:
            proc.join(timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
        return proc.exitcode

    def _on_death(self, exc: AsrWorkerDied) -> None:
        self._consecutive_deaths += 1
        self._last_vad_segments = []
        self._last_decode_stats = []
        # The dead generation's completed scenes still count in the file totals.
        self._retire_live_stats()
        self._stop(timeout_s=5.0)
        # The restart is a new instance: new generation for telemetry, budget
        # restarts — but it is not a refresh.
        self.policy.reset(refresh=False)
        logger.error(
            "%s (%d consecutive). The next scene starts a fresh worker.",
            exc, self._consecutive_deaths,
        )

    # ------------------------------------------------------------------ ASR surface
    def transcribe_to_srt(self, audio_path: Union[str, Path], output_srt_path: Union[str, Path], **kwargs) -> Path:
        task = kwargs.get("task", self._task)
        if self._consecutive_deaths >= self._max_deaths:
            raise RuntimeError(
                f"ASR worker died {self._consecutive_deaths} times in a row; giving up on this file"
            )
        if self._alive() and self.policy.due():
            self._refresh()
        elif not self._alive():
            # Restart after a death. A restart that cannot load the model counts
            # like a death, so a permanently unloadable model gives up after
            # max_consecutive_deaths instead of retrying a model load per scene.
            try:
                self._start()
            except AsrWorkerDied as exc:
                self._on_death(exc)
                raise
            except RuntimeError:
                self._consecutive_deaths += 1
                self._stop(timeout_s=5.0)
                raise
        try:
            kind, data = self._request(
                ("transcribe", str(audio_path), str(output_srt_path), task), "transcription"
            )
        except AsrWorkerDied as exc:
            self._on_death(exc)
            if self._consecutive_deaths >= self._max_deaths:
                raise RuntimeError(
                    f"ASR worker died {self._consecutive_deaths} times in a row; giving up on this file"
                ) from exc
            raise
        self._consecutive_deaths = 0
        if kind == "err":
            self._last_vad_segments = []
            self._last_decode_stats = []
            raise RuntimeError(data)
        self._last_vad_segments = list(data.get("vad_segments") or [])
        self._last_decode_stats = list(data.get("decode_stats") or [])
        self._filter_stats_live = dict(data.get("filter_statistics") or {})
        return Path(output_srt_path)

    def record_audio(self, audio_duration_s: float) -> None:
        """Feed the refresh budget with the scene just handed to the recogniser."""
        self.policy.record(audio_duration_s)

    def get_last_vad_segments(self) -> List[Dict]:
        return list(self._last_vad_segments)

    def get_last_decode_stats(self) -> List[Dict]:
        return list(self._last_decode_stats)

    def get_filter_statistics(self) -> Dict[str, int]:
        out = dict(self._filter_stats_done)
        for k, v in (self._filter_stats_live or {}).items():
            out[k] = out.get(k, 0) + int(v or 0)
        return out

    def get_segmenter_name(self) -> str:
        return self._segmenter_name

    def reset_statistics(self) -> None:
        """Per-file reset (the pipeline calls this at the start of each file):
        filter statistics and the consecutive-death counter, so one bad file cannot
        make the proxy give up on every later file of the batch."""
        self._filter_stats_done = {}
        self._filter_stats_live = {}
        self._consecutive_deaths = 0
        if self._alive():
            try:
                self._request(("reset_stats",), "statistics reset")
            except AsrWorkerDied as exc:
                self._on_death(exc)
