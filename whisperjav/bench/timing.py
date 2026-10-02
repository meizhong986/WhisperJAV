"""
Subtitle timing against a reference: how early or late lines start and end.

Built for the 1.9.4 ChronosJAV timing requirement (T1). Each reference line is
matched to a WhisperJAV line with the existing matcher (time overlap + text
similarity). For every matched pair:

    start error = hyp.start - ref.start   (positive: the line starts late)
    end error   = hyp.end   - ref.end     (positive: the line ends late / lingers)

Two runs are compared only on the reference lines BOTH runs matched, so a run
cannot improve its score by dropping hard lines. Reference lines the base run
matched and the candidate did not are reported as lost.

Pure stdlib + existing bench modules; runs anywhere.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from whisperjav.bench.loader import SubtitleEntry, _parse_srt_file, load_ground_truth
from whisperjav.bench.matcher import match_subtitles


@dataclass
class TimingMetrics:
    """Start/end errors over a set of matched reference lines (seconds)."""

    n_lines: int
    start_abs_median: float
    start_abs_p90: float
    start_signed_mean: float
    end_abs_median: float
    end_abs_p90: float
    end_signed_mean: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _entries_to_dicts(entries: list[SubtitleEntry]) -> list[dict]:
    return [{"start": e.start, "end": e.end, "text": e.text} for e in entries]


def _ref_key(ref: dict) -> tuple[float, float, str]:
    return (ref["start"], ref["end"], ref["text"])


def match_errors(ref_subs: list[dict], hyp_subs: list[dict]) -> dict[tuple, tuple[float, float]]:
    """Map each matched reference line (by key) to its (start error, end error)."""
    matched = match_subtitles(ref_subs, hyp_subs)["matched"]
    return {
        _ref_key(ref): (hyp["start"] - ref["start"], hyp["end"] - ref["end"])
        for ref, hyp in matched
    }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an ascending list (fraction in 0..1)."""
    if not sorted_values:
        return 0.0
    rank = math.ceil(fraction * len(sorted_values) - 1e-9)
    rank = min(len(sorted_values), max(1, rank))
    return sorted_values[rank - 1]


def _median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def summarize(errors: list[tuple[float, float]]) -> TimingMetrics:
    """Summarise a list of (start error, end error) pairs."""
    starts = [s for s, _ in errors]
    ends = [e for _, e in errors]
    start_abs = sorted(abs(v) for v in starts)
    end_abs = sorted(abs(v) for v in ends)
    n = len(errors)
    return TimingMetrics(
        n_lines=n,
        start_abs_median=round(_median(start_abs), 3),
        start_abs_p90=round(_percentile(start_abs, 0.9), 3),
        start_signed_mean=round(sum(starts) / n, 3) if n else 0.0,
        end_abs_median=round(_median(end_abs), 3),
        end_abs_p90=round(_percentile(end_abs, 0.9), 3),
        end_signed_mean=round(sum(ends) / n, 3) if n else 0.0,
    )


@dataclass
class ClipComparison:
    """Base vs candidate on one clip, over the reference lines both matched."""

    n_ref: int
    n_matched_base: int
    n_matched_cand: int
    n_common: int
    n_lost: int                 # matched by base, not by candidate
    base: TimingMetrics
    cand: TimingMetrics
    base_errors: list[tuple[float, float]]
    cand_errors: list[tuple[float, float]]


def compare_clip(ref_subs: list[dict], base_subs: list[dict], cand_subs: list[dict]) -> ClipComparison:
    """Compare two runs' timing on one clip, on common matched reference lines."""
    base = match_errors(ref_subs, base_subs)
    cand = match_errors(ref_subs, cand_subs)
    common = [k for k in (_ref_key(r) for r in ref_subs) if k in base and k in cand]
    base_errors = [base[k] for k in common]
    cand_errors = [cand[k] for k in common]
    return ClipComparison(
        n_ref=len(ref_subs),
        n_matched_base=len(base),
        n_matched_cand=len(cand),
        n_common=len(common),
        n_lost=sum(1 for k in base if k not in cand),
        base=summarize(base_errors),
        cand=summarize(cand_errors),
        base_errors=base_errors,
        cand_errors=cand_errors,
    )


def load_srt(path: Path, reference: bool = False) -> list[dict]:
    """Load an SRT as a list of {'start','end','text'} dicts ([] if missing)."""
    path = Path(path)
    if reference:
        return _entries_to_dicts(load_ground_truth(path))
    return _entries_to_dicts(_parse_srt_file(path)) if path.exists() else []
