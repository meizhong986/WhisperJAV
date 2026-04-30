"""Backend runner with per-backend timeout isolation.

Pattern adapted from tests/test_speech_segmentation_visual.py. Each backend
gets its own segmenter instance and runs in a daemon thread with join(timeout).
A backend crash or hang does not kill the tool — the report captures the
failure and the tool proceeds to the next backend.

Sensitivity presets flow through ensemble.pass_worker.resolve_qwen_sensitivity —
the same function both qwen and legacy pipeline override paths use — so the
measured behaviour matches what production would produce.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .models import BackendReport, VadSegment

logger = logging.getLogger("whisperjav.vad_gt")

VALID_SENSITIVITIES = ("conservative", "balanced", "aggressive")


class BackendRunner:
    """Runs a named speech segmenter backend on a single audio array.

    Attributes:
        sensitivity: One of VALID_SENSITIVITIES. Used to resolve preset params
            via ConfigManager through ensemble.pass_worker.resolve_qwen_sensitivity.
        timeout_sec: Hard limit on segmentation wall-clock time.
    """

    def __init__(self, sensitivity: str = "aggressive", timeout_sec: int = 300):
        if sensitivity not in VALID_SENSITIVITIES:
            raise ValueError(
                f"sensitivity must be one of {VALID_SENSITIVITIES}, got {sensitivity!r}"
            )
        if timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be > 0, got {timeout_sec}")
        self.sensitivity = sensitivity
        self.timeout_sec = int(timeout_sec)
        self._availability_cache: Dict[str, Tuple[bool, str]] = {}

    # ------------------------------------------------------------------
    # Availability + params
    # ------------------------------------------------------------------

    def check_available(self, backend: str) -> Tuple[bool, str]:
        """Cached availability lookup. Returns (available, install_hint)."""
        if backend not in self._availability_cache:
            from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory
            self._availability_cache[backend] = SpeechSegmenterFactory.is_backend_available(backend)
        return self._availability_cache[backend]

    def resolve_params(self, backend: str) -> Dict[str, Any]:
        """Resolve sensitivity preset for a backend.

        Uses the same path as ensemble.pass_worker — guarantees we measure the
        same configuration production would produce for --sensitivity <s>.
        Returns {} for 'none' or unknown backends (tolerated by factory).
        """
        if backend == "none":
            return {}
        try:
            from whisperjav.ensemble.pass_worker import resolve_qwen_sensitivity
            return resolve_qwen_sensitivity(backend, self.sensitivity, None) or {}
        except Exception as e:
            logger.warning("Could not resolve sensitivity for %s: %s", backend, e)
            return {}

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        backend: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> BackendReport:
        """Run one backend on `audio`. Never raises; always returns a report.

        Ordering of failure paths:
            1. Dependencies missing → available=False, success=False
            2. Factory cannot create (bad config) → success=False, error=msg
            3. Backend thread hangs past timeout → success=False, error="Timeout..."
            4. Backend raises → success=False, error=exception msg
            5. Backend returns empty segments → success=True, num_segments=0
            6. Otherwise → success=True with segments populated

        Always calls segmenter.cleanup() in finally.
        """
        # Step 1 — availability
        available, hint = self.check_available(backend)
        if not available:
            return BackendReport(
                name=backend,
                display_name=backend,
                available=False,
                success=False,
                error=f"Not available: {hint}",
                processing_time_sec=0.0,
            )

        params = self.resolve_params(backend)
        q: "queue.Queue[BackendReport]" = queue.Queue()

        def _worker() -> None:
            from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory

            segmenter = None
            try:
                try:
                    segmenter = SpeechSegmenterFactory.create(backend, config=params)
                except Exception as e:
                    q.put(BackendReport(
                        name=backend,
                        display_name=backend,
                        available=True,
                        success=False,
                        error=f"Factory create failed: {e}",
                        processing_time_sec=0.0,
                        parameters=params,
                    ))
                    return

                display = getattr(segmenter, "display_name", backend)
                t0 = time.time()
                try:
                    result = segmenter.segment(audio, sample_rate=sample_rate)
                except Exception as e:
                    logger.debug("Backend %s segment() raised: %s\n%s",
                                 backend, e, traceback.format_exc())
                    q.put(BackendReport(
                        name=backend,
                        display_name=display,
                        available=True,
                        success=False,
                        error=f"segment() raised: {e}",
                        processing_time_sec=time.time() - t0,
                        parameters=params,
                    ))
                    return
                elapsed = time.time() - t0

                # Convert SpeechSegment → our VadSegment (decoupled from whisperjav internals)
                vad_segs = [
                    VadSegment(
                        start_sec=float(s.start_sec),
                        end_sec=float(s.end_sec),
                        confidence=float(getattr(s, "confidence", 1.0)),
                        metadata=dict(getattr(s, "metadata", {}) or {}),
                    )
                    for s in getattr(result, "segments", [])
                ]

                q.put(BackendReport(
                    name=backend,
                    display_name=display,
                    available=True,
                    success=True,
                    error=None,
                    processing_time_sec=elapsed,
                    parameters=params,
                    segments=vad_segs,
                    num_segments=len(vad_segs),
                    coverage_ratio=float(getattr(result, "speech_coverage_ratio", 0.0)),
                ))
            finally:
                if segmenter is not None:
                    try:
                        segmenter.cleanup()
                    except Exception as e:
                        logger.debug("cleanup() for %s raised: %s", backend, e)

        th = threading.Thread(target=_worker, daemon=True, name=f"vad-runner-{backend}")
        th.start()
        th.join(timeout=self.timeout_sec)

        if th.is_alive():
            # Hang — cannot forcibly kill a Python thread; return failure and let the
            # daemon thread eventually exit when the process dies or the model releases.
            return BackendReport(
                name=backend,
                display_name=backend,
                available=True,
                success=False,
                error=f"Timeout after {self.timeout_sec}s",
                processing_time_sec=float(self.timeout_sec),
                parameters=params,
            )

        # Thread finished; retrieve result (should always have one entry)
        try:
            return q.get_nowait()
        except queue.Empty:
            return BackendReport(
                name=backend,
                display_name=backend,
                available=True,
                success=False,
                error="No result returned (worker exited without emitting)",
                processing_time_sec=0.0,
                parameters=params,
            )

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def run_many(
        self,
        backends: list[str],
        audio: np.ndarray,
        sample_rate: int,
        progress_cb=None,
    ) -> Dict[str, BackendReport]:
        """Run multiple backends sequentially.

        Args:
            backends: list of backend names.
            audio, sample_rate: same across all.
            progress_cb: optional callback(backend_name, report) after each run.

        Returns:
            Ordered dict {backend_name: BackendReport}.
        """
        out: Dict[str, BackendReport] = {}
        for b in backends:
            logger.info("Running backend: %s", b)
            rep = self.run(b, audio, sample_rate)
            out[b] = rep
            if progress_cb is not None:
                try:
                    progress_cb(b, rep)
                except Exception:
                    logger.debug("progress_cb raised; ignoring", exc_info=True)
        return out
