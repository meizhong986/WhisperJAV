"""Two test runs (base, candidate) against the same reference, clip by clip.

Text: per clip, the change in edits (wrong + missing + extra), missed lines and suspected hallucination lines.
Timing: per clip and pooled, on the reference lines whose start (end) error was measured in BOTH runs, the median
|error| of each run; a clip counts as better (worse) when the candidate's median is lower (higher) by more than
`tolerance` seconds. Lines measured in one run only are counted, not compared.
"""
from __future__ import annotations

import statistics

from .evaluate import ClipResult

TOLERANCE = 0.02


def _both(b: ClipResult, c: ClipResult, attr: str):
    bm = {r.idx: getattr(r, attr) for r in b.ref if getattr(r, attr) is not None}
    cm = {r.idx: getattr(r, attr) for r in c.ref if getattr(r, attr) is not None}
    common = sorted(set(bm) & set(cm))
    return [abs(bm[k]) for k in common], [abs(cm[k]) for k in common], len(bm) - len(common), len(cm) - len(common)


def compare(base: list[ClipResult], cand: list[ClipResult], tolerance: float = TOLERANCE) -> dict:
    cm = {c.clip: c for c in cand}
    if sorted(cm) != sorted(b.clip for b in base):
        raise ValueError(f"base and candidate cover different clips: {sorted(b.clip for b in base)} vs {sorted(cm)}")
    rows = []
    pooled = {"start": ([], []), "end": ([], [])}
    for b in base:
        c = cm[b.clip]
        if [(r.start, r.end, r.text) for r in b.ref] != [(r.start, r.end, r.text) for r in c.ref]:
            raise ValueError(f"{b.clip}: base and candidate were scored against different references")
        row = {"clip": b.clip,
               "d_edits": c.edits - b.edits,
               "d_plain": c.plain_distance - b.plain_distance,
               "d_missed": sum(r.missed for r in c.ref) - sum(r.missed for r in b.ref),
               "d_halluc": sum(t.hallucination for t in c.test) - sum(t.hallucination for t in b.test),
               "d_misheard": sum(t.misheard for t in c.test) - sum(t.misheard for t in b.test)}
        for edge in ("start", "end"):
            eb, ec, only_b, only_c = _both(b, c, f"{edge}_err")
            pooled[edge][0].extend(eb)
            pooled[edge][1].extend(ec)
            mb = statistics.median(eb) if eb else None
            mc = statistics.median(ec) if ec else None
            row[f"{edge}_n_both"], row[f"{edge}_only_base"], row[f"{edge}_only_cand"] = len(eb), only_b, only_c
            row[f"{edge}_med_base"], row[f"{edge}_med_cand"] = mb, mc
            row[f"{edge}_verdict"] = ("n/a" if mb is None else "better" if mc < mb - tolerance
                                      else "worse" if mc > mb + tolerance else "same")
        rows.append(row)
    out = {"tolerance_s": tolerance, "clips": rows,
           "d_edits": sum(r["d_edits"] for r in rows), "d_plain": sum(r["d_plain"] for r in rows),
           "d_missed": sum(r["d_missed"] for r in rows), "d_halluc": sum(r["d_halluc"] for r in rows),
           "d_misheard": sum(r["d_misheard"] for r in rows),
           "text_clips_better": sum(r["d_edits"] < 0 for r in rows),
           "text_clips_worse": sum(r["d_edits"] > 0 for r in rows)}
    for edge in ("start", "end"):
        eb, ec = pooled[edge]
        out[f"{edge}_n_both"] = len(eb)
        out[f"{edge}_med_base"] = statistics.median(eb) if eb else None
        out[f"{edge}_med_cand"] = statistics.median(ec) if ec else None
        out[f"{edge}_within_base"] = sum(x <= 0.5 for x in eb) / len(eb) if eb else None
        out[f"{edge}_within_cand"] = sum(x <= 0.5 for x in ec) / len(ec) if ec else None
        out[f"{edge}_clips_better"] = sum(r[f"{edge}_verdict"] == "better" for r in rows)
        out[f"{edge}_clips_worse"] = sum(r[f"{edge}_verdict"] == "worse" for r in rows)
    return out
