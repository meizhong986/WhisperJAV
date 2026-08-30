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
``cuda_allocated_mb`` / ``cuda_reserved_mb`` / ``rss_mb``
    If memory grows monotonically up to the failure, that is close to
    conclusive and points upstream at CTranslate2 rather than at our parameters.

The output is a JSONL file — one object per scene — so a reporter can attach it
and it can be plotted directly. Off unless ``--asr-telemetry`` is passed.

This module is diagnostic only. It changes no behaviour and must never raise
into the pipeline: a telemetry failure has to stay a telemetry failure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from whisperjav.utils.logger import logger


def _memory_snapshot() -> dict[str, Optional[float]]:
    """Best-effort memory reading. Never raises, never imports heavily."""
    snap: dict[str, Optional[float]] = {
        "cuda_allocated_mb": None,
        "cuda_reserved_mb": None,
        "rss_mb": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            snap["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1048576, 1)
            snap["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1048576, 1)
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
    """Collects one record per scene and writes them as JSONL."""

    def __init__(self, output_path: Path, media_name: str = ""):
        self.output_path = Path(output_path)
        self.media_name = media_name
        self.records: list[dict[str, Any]] = []
        self._t0 = time.time()

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

    def write(self) -> Optional[Path]:
        """Write the JSONL file. Returns the path, or None on failure."""
        if not self.records:
            return None
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as fh:
                for rec in self.records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad. A malformed path raises ValueError rather than
            # OSError, and a record holding something unserialisable raises
            # TypeError — neither should turn a diagnostic aid into a run failure.
            logger.warning("Could not write ASR telemetry to %s: %s", self.output_path, exc)
            return None

        trend = self.trend_summary()
        if trend:
            logger.info("ASR telemetry trend (first 10 scenes -> last 10): %s", trend)
        logger.info("ASR telemetry written to %s (%d scenes). Please attach this "
                    "file if you are reporting issue #394.",
                    self.output_path, len(self.records))
        return self.output_path
