"""Per-scene ASR telemetry, for diagnosing #394.

Why this exists
---------------
The reports on #394 describe the recogniser entering a state where it returns
nothing for the rest of a run. Everything WhisperJAV records today describes the
*aftermath* — empty scenes, a short subtitle file — and nothing describes the
approach to it. The most striking unexploited detail in the whole issue is that
the failure is preceded by a slowdown: @daoran9 measured the median ASR call
going from **1.05 s to 31.65 s** before the empty scenes began.

That matters because it discriminates between explanations. A model that is
merely "stuck" returns nothing *quickly*. Thirty-fold slowdown followed by
silence is the signature of something accumulating — repeated decoder fallback,
a growing cache, or memory pressure — and those leave different fingerprints.

What is captured, and why each field earns its place
----------------------------------------------------
``temperature``
    faster-whisper raises the sampling temperature and retries when a decode
    trips its compression-ratio or logprob guards. A temperature above the first
    configured value therefore *is* a fallback event. If fallbacks cluster in the
    scenes preceding the collapse, the pathology is in decoding, not in the
    model's state.
``compression_ratio`` / ``avg_logprob`` / ``no_speech_prob``
    The three gates that trigger those retries and that decide whether a window
    is discarded as silence. Recording them shows *which* gate is firing.
``wall_s`` and ``rtf``
    The slowdown itself, per scene, against constant-length input.
``cuda_used_mb`` / ``cuda_allocated_mb`` / ``cuda_reserved_mb`` / ``rss_mb``
    ``cuda_used_mb`` is device-wide (``torch.cuda.mem_get_info``), which is
    the only one of these that can see CTranslate2's own arena -- the
    ``allocated``/``reserved`` pair reports PyTorch's caching allocator alone,
    and Balanced's recogniser does not allocate through it. If the device-wide
    figure or the process RSS grows monotonically up to the failure, that is
    close to conclusive and points upstream rather than at our parameters.

Each record is appended to the file the moment its scene finishes, so a run
that crashes, hangs, or is killed still leaves everything up to that scene on
disk. That is the whole point of keeping it on by default: the record has to
exist before anyone knows the run was one of the bad ones.

The output is a JSONL file — one object per scene — so a reporter can attach it
and it can be plotted directly.

Where it goes, and when
-----------------------
On by default (owner decision, 2026-09-03): a run that later turns out to be
#394 has to have recorded its approach *before* anyone knew to ask. The file
lives in ``raw_subs/`` next to the outputs — the folder that already holds the
per-run artefacts users attach to bug reports — as
``<name>.asr_telemetry.jsonl`` (``<name>.pass1.asr_telemetry.jsonl`` inside an
ensemble run, one per pass). ``--asr-telemetry PATH`` moves it (a directory,
or a file path for a single input); ``--no-asr-telemetry`` switches it off.
``resolve_telemetry_path`` is the one place that rule lives.

This module is diagnostic only. It changes no behaviour and must never raise
into the pipeline: a telemetry failure has to stay a telemetry failure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from whisperjav.utils.logger import logger

TELEMETRY_SUFFIX = ".asr_telemetry.jsonl"
TELEMETRY_SUBDIR = "raw_subs"


def resolve_telemetry_path(
    override: Optional[str],
    enabled: bool,
    output_dir: Optional[Path],
    basename: str,
    tag: Optional[str] = None,
) -> Optional[Path]:
    """Where this media's telemetry file goes, or None when telemetry is off.

    ``override`` is the user's ``--asr-telemetry`` value: a directory (or a
    path without a suffix) gets one file per media; a file path is used as
    given. ``tag`` distinguishes passes inside an ensemble run (``pass1``).
    Without an override the file goes to ``<output_dir>/raw_subs/``.
    Never raises.
    """
    if not enabled:
        return None
    try:
        name = f"{basename}.{tag}{TELEMETRY_SUFFIX}" if tag else f"{basename}{TELEMETRY_SUFFIX}"
        if override:
            p = Path(override)
            if p.is_dir() or not p.suffix:
                return p / name
            if tag:
                return p.with_name(f"{p.stem}.{tag}{p.suffix}")
            return p
        if output_dir is None:
            return None
        return Path(output_dir) / TELEMETRY_SUBDIR / name
    except Exception:  # noqa: BLE001 - a path problem must not abort a run
        return None


def _memory_snapshot() -> dict[str, Optional[float]]:
    """Best-effort memory reading. Never raises, never imports heavily."""
    snap: dict[str, Optional[float]] = {
        "cuda_used_mb": None,
        "cuda_allocated_mb": None,
        "cuda_reserved_mb": None,
        "rss_mb": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            snap["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1048576, 1)
            snap["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1048576, 1)
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                snap["cuda_used_mb"] = round((total_b - free_b) / 1048576, 1)
            except Exception:  # noqa: BLE001
                snap["cuda_used_mb"] = None
    except Exception:  # noqa: BLE001 - diagnostics must not break the run
        pass
    try:
        import os

        import psutil

        snap["rss_mb"] = round(psutil.Process(os.getpid()).memory_info().rss / 1048576, 1)
    except Exception:  # noqa: BLE001
        pass
    return snap


def summarise_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce one scene's raw segment dicts to the fields that discriminate.

    Accepts the shape produced by ``dataclasses.asdict`` on a faster-whisper
    ``Segment``. Missing keys are tolerated so this keeps working if the upstream
    dataclass changes.
    """
    def _vals(key):
        out = []
        for s in segments or []:
            v = s.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    temps = _vals("temperature")
    logprobs = _vals("avg_logprob")
    compression = _vals("compression_ratio")
    no_speech = _vals("no_speech_prob")

    return {
        "n_segments": len(segments or []),
        "max_temperature": max(temps) if temps else None,
        "fallback_segments": sum(1 for t in temps if t > 0.0) if temps else 0,
        "min_avg_logprob": min(logprobs) if logprobs else None,
        "mean_avg_logprob": round(sum(logprobs) / len(logprobs), 4) if logprobs else None,
        "max_compression_ratio": max(compression) if compression else None,
        "max_no_speech_prob": max(no_speech) if no_speech else None,
    }


class AsrTelemetry:
    """One record per scene, appended to a JSONL file as each scene finishes.

    The file is truncated on the first record and every later record is
    appended and flushed immediately, so an interrupted run leaves a partial
    but valid file. ``write()`` (alias ``finalize()``) only logs the trend and
    the path; it does not hold anything back for the end.
    """

    def __init__(self, output_path: Path, media_name: str = ""):
        self.output_path = Path(output_path)
        self.media_name = media_name
        self.records: list[dict[str, Any]] = []
        self._t0 = time.time()
        self._file_ready = False   # truncated / created on first record
        self._write_failed = False  # warn once, then keep the run going

    def _append(self, rec: dict[str, Any]) -> None:
        """Append one record to disk now. Never raises; warns once on failure."""
        if self._write_failed:
            return
        try:
            if not self._file_ready:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_path.write_text("", encoding="utf-8")
                self._file_ready = True
            with open(self.output_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail the run
            self._write_failed = True
            logger.warning("Could not write ASR telemetry to %s: %s", self.output_path, exc)

    def record_scene(
        self,
        *,
        index: int,
        audio_duration_s: float,
        wall_s: float,
        segments: Optional[list[dict[str, Any]]] = None,
        speech_detected: bool = False,
        produced_output: bool = False,
    ) -> None:
        """Record one scene. Never raises."""
        try:
            rec: dict[str, Any] = {
                "media": self.media_name,
                "scene": index,
                "elapsed_s": round(time.time() - self._t0, 2),
                "audio_duration_s": round(float(audio_duration_s or 0), 3),
                "wall_s": round(float(wall_s), 3),
                "speech_detected": bool(speech_detected),
                "produced_output": bool(produced_output),
            }
            if rec["audio_duration_s"] > 0:
                rec["rtf"] = round(rec["wall_s"] / rec["audio_duration_s"], 3)
            rec.update(summarise_segments(segments or []))
            rec.update(_memory_snapshot())
            self.records.append(rec)
            self._append(rec)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ASR telemetry: could not record scene %s: %s", index, exc)

    def trend_summary(self, window: int = 10) -> Optional[str]:
        """Compare the first and last *window* scenes.

        This is the shape of #394: healthy early, degrading late. A single line
        the user can paste is worth more than a file they have to be persuaded
        to attach.
        """
        try:
            timed = [r for r in self.records if r.get("rtf") is not None]
            if len(timed) < 2 * window:
                return None
            head, tail = timed[:window], timed[-window:]

            def _avg(rows, key):
                vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
                return sum(vals) / len(vals) if vals else None

            def _fmt(v, unit=""):
                return f"{v:.2f}{unit}" if isinstance(v, (int, float)) else "n/a"

            parts = [
                f"RTF {_fmt(_avg(head, 'rtf'))} -> {_fmt(_avg(tail, 'rtf'))}",
                f"fallback segs/scene {_fmt(_avg(head, 'fallback_segments'))} -> "
                f"{_fmt(_avg(tail, 'fallback_segments'))}",
            ]
            for key, label, unit in (
                ("cuda_reserved_mb", "CUDA reserved", " MB"),
                ("rss_mb", "RSS", " MB"),
            ):
                a, b = _avg(head, key), _avg(tail, key)
                if a is not None and b is not None:
                    parts.append(f"{label} {_fmt(a, unit)} -> {_fmt(b, unit)}")
            return "; ".join(parts)
        except Exception:  # noqa: BLE001
            return None

    def finalize(self) -> Optional[Path]:
        """Log the trend and the path. Every record is already on disk.

        Returns the path, or None if nothing was recorded or the file could
        not be written. Safe to call from an error path.
        """
        if not self.records or self._write_failed or not self._file_ready:
            return None
        try:
            trend = self.trend_summary()
            if trend:
                logger.info("ASR telemetry trend (first 10 scenes -> last 10): %s", trend)
            logger.info("ASR telemetry: %s (%d scenes)", self.output_path, len(self.records))
        except Exception:  # noqa: BLE001
            pass
        return self.output_path

    # Kept for callers written against the buffered version.
    write = finalize
