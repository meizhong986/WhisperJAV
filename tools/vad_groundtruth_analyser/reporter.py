"""Output serialisation: JSON + CSV + HTML.

JSON is the canonical artefact (schema_version pinned in AnalysisReport).
CSV is one-row-per-backend, convenient for cross-file aggregation.
HTML is rendered via renderer.render_html.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from .models import AnalysisReport, GtSegment

logger = logging.getLogger("whisperjav.vad_gt")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Handle numpy scalars and anything else that falls through."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def write_json(report: AnalysisReport, path: Path) -> None:
    """Write report as UTF-8 JSON (ensure_ascii=False for Japanese text)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            asdict(report), f,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    logger.info("Wrote JSON: %s", path)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_HEADERS = [
    "media_file",
    "media_duration_sec",
    "sensitivity",
    "ground_truth_available",
    "backend",
    "available",
    "success",
    "error",
    "processing_time_sec",
    "num_segments",
    "coverage_pct",
    # GT-only metrics (empty in GT-less mode)
    "frame_f1",
    "frame_precision",
    "frame_recall",
    "iou_mean",
    "iou_median",
    "onset_drift_mean_ms",
    "offset_drift_mean_ms",
    "missed_speech_pct",
    "false_alarm_pct",
]


def write_csv(report: AnalysisReport, path: Path) -> None:
    """One row per backend. GT-less rows leave metric columns empty.

    Cross-file aggregation: `cat *.vad_analysis.csv` works because every row
    includes media_file + backend columns.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gt_avail = report.ground_truth is not None

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for name, rep in report.backends.items():
            row = {
                "media_file": report.media_file,
                "media_duration_sec": f"{report.audio_duration_sec:.3f}",
                "sensitivity": report.sensitivity,
                "ground_truth_available": "yes" if gt_avail else "no",
                "backend": name,
                "available": "yes" if rep.available else "no",
                "success": "yes" if rep.success else "no",
                "error": rep.error or "",
                "processing_time_sec": f"{rep.processing_time_sec:.3f}",
                "num_segments": rep.num_segments,
                "coverage_pct": f"{rep.coverage_ratio * 100:.2f}",
            }
            m = rep.metrics
            if m is not None:
                row.update({
                    "frame_f1":             f"{m.frame_f1:.4f}",
                    "frame_precision":      f"{m.frame_precision:.4f}",
                    "frame_recall":         f"{m.frame_recall:.4f}",
                    "iou_mean":             f"{m.iou_mean:.4f}",
                    "iou_median":           f"{m.iou_median:.4f}",
                    "onset_drift_mean_ms":  f"{m.onset_drift_mean_ms:.1f}",
                    "offset_drift_mean_ms": f"{m.offset_drift_mean_ms:.1f}",
                    "missed_speech_pct":    f"{m.missed_speech_pct:.2f}",
                    "false_alarm_pct":      f"{m.false_alarm_pct:.2f}",
                })
            else:
                for k in ("frame_f1", "frame_precision", "frame_recall",
                          "iou_mean", "iou_median",
                          "onset_drift_mean_ms", "offset_drift_mean_ms",
                          "missed_speech_pct", "false_alarm_pct"):
                    row[k] = ""
            writer.writerow(row)
    logger.info("Wrote CSV: %s", path)


# ---------------------------------------------------------------------------
# HTML — thin delegate to renderer
# ---------------------------------------------------------------------------

def write_html(
    report: AnalysisReport,
    waveform: np.ndarray,
    path: Path,
    title: Optional[str] = None,
    waveform_points: int = 10000,
    gt_segments: Optional[List[GtSegment]] = None,
) -> None:
    from .renderer import render_html
    render_html(
        report=report,
        waveform=waveform,
        output_path=Path(path),
        title=title,
        waveform_points=waveform_points,
        gt_segments=gt_segments,
    )
