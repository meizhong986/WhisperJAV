#!/usr/bin/env python3
"""Score a candidate SRT against a ground-truth SRT (anime-whisper VAD tuning).

Pure / no GPU. Produces a metric vector that separates the three failure modes
identified in the manual T3 analysis, plus segmentation + timing metrics:

  Content accuracy (character-level, jiwer, alignment-free):
    cer        overall character error rate (lower better)
    del_rate   deletions / GT chars   -> MISSING content (recall gaps)
    ins_rate   insertions / GT chars  -> HALLUCINATION
    sub_rate   substitutions / GT chars -> MISHEARING
    char_recall (GT chars - deletions) / GT chars

  Segmentation granularity:
    n_subs, seg_ratio (= n_subs / n_gt), mean_dur, mean_dur_gt

  Timing / coverage (time-interval overlap, segmentation-agnostic):
    time_recall     GT speech-seconds overlapped by any candidate / GT speech-seconds
    time_precision  candidate speech-seconds overlapping GT / candidate speech-seconds

Text is concatenated in start-time order and compared at character level, so the
score is robust to the candidate and GT having different segment boundaries.

Usage:
    python -m tools.vad_hypothesis_suite.score GROUND_TRUTH.srt CANDIDATE.srt [--name NAME] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import jiwer
import pysrt

# Content characters kept for character-error-rate: hiragana, katakana (incl.
# prolonged-sound mark ー and halfwidth katakana), CJK ideographs, iteration mark
# 々, and latin/digits. Everything else (punctuation, spaces, LTR/RTL marks, the
# GT's "． " line separators) is stripped so CER reflects content, not styling.
_KEEP_RE = re.compile(
    r"[^"
    r"぀-ゟ"   # hiragana
    r"゠-ヿ"   # katakana (includes ー U+30FC)
    r"ㇰ-ㇿ"   # katakana phonetic ext
    r"ｦ-ﾟ"   # halfwidth katakana
    r"一-鿿"   # CJK unified
    r"㐀-䶿"   # CJK ext A
    r"々"          # 々 iteration mark
    r"0-9A-Za-z"
    r"]"
)

Interval = Tuple[float, float, str]


def normalize_ja(text: str) -> str:
    """Reduce text to bare content characters for a fair character-level CER."""
    return _KEEP_RE.sub("", (text or "")).lower()


def load_srt(path: str | Path) -> List[Interval]:
    """Return [(start_s, end_s, text), ...] sorted by start time."""
    subs = pysrt.open(str(path), encoding="utf-8")
    out: List[Interval] = []
    for s in subs:
        start = s.start.ordinal / 1000.0
        end = s.end.ordinal / 1000.0
        out.append((start, end, s.text or ""))
    out.sort(key=lambda x: x[0])
    return out


def _merge_intervals(subs: List[Interval]) -> List[Tuple[float, float]]:
    """Union of [start,end] spans (handles overlaps) for coverage math."""
    spans = sorted((s, e) for s, e, _ in subs if e > s)
    if not spans:
        return []
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def _total(spans: List[Tuple[float, float]]) -> float:
    return sum(e - s for s, e in spans)


def _intersection(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    """Total overlap seconds between two sorted interval unions."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def content_metrics(gt: List[Interval], cand: List[Interval]) -> Dict[str, float]:
    gt_text = normalize_ja("".join(t for _, _, t in gt))
    cand_text = normalize_ja("".join(t for _, _, t in cand))
    n_gt_chars = max(1, len(gt_text))
    out = jiwer.process_characters(gt_text, cand_text)
    return {
        "gt_chars": len(gt_text),
        "cand_chars": len(cand_text),
        "cer": round(out.cer, 4),
        "sub_rate": round(out.substitutions / n_gt_chars, 4),
        "del_rate": round(out.deletions / n_gt_chars, 4),
        "ins_rate": round(out.insertions / n_gt_chars, 4),
        "char_recall": round((n_gt_chars - out.deletions) / n_gt_chars, 4),
    }


def timing_metrics(gt: List[Interval], cand: List[Interval]) -> Dict[str, float]:
    gt_spans = _merge_intervals(gt)
    cand_spans = _merge_intervals(cand)
    gt_total = _total(gt_spans) or 1e-9
    cand_total = _total(cand_spans) or 1e-9
    inter = _intersection(gt_spans, cand_spans)
    gt_dur = [e - s for s, e, _ in gt]
    cand_dur = [e - s for s, e, _ in cand]
    return {
        "n_subs": len(cand),
        "n_subs_gt": len(gt),
        "seg_ratio": round(len(cand) / max(1, len(gt)), 3),
        "mean_dur": round(sum(cand_dur) / max(1, len(cand_dur)), 3),
        "mean_dur_gt": round(sum(gt_dur) / max(1, len(gt_dur)), 3),
        "time_recall": round(inter / gt_total, 4),
        "time_precision": round(inter / cand_total, 4),
        "gt_speech_s": round(gt_total, 1),
        "cand_speech_s": round(cand_total, 1),
    }


def region_recall(gt: List[Interval], cand: List[Interval], bin_s: float = 20.0,
                  horizon_s: float | None = None) -> List[Dict[str, float]]:
    """Per-time-bin GT coverage — localizes recall gaps (e.g. 1:16-1:39)."""
    if horizon_s is None:
        horizon_s = max((e for _, e, _ in gt), default=0.0)
    cand_spans = _merge_intervals(cand)
    bins = []
    t = 0.0
    while t < horizon_s:
        lo, hi = t, min(t + bin_s, horizon_s)
        gt_here = _merge_intervals([(s, e, "") for s, e, _ in gt if e > lo and s < hi])
        gt_here = [(max(s, lo), min(e, hi)) for s, e in gt_here]
        gt_amt = _total(gt_here)
        if gt_amt > 0.05:
            cov = _intersection(gt_here, cand_spans)
            bins.append({
                "bin": f"{int(lo//60):d}:{int(lo%60):02d}-{int(hi//60):d}:{int(hi%60):02d}",
                "gt_s": round(gt_amt, 1),
                "recall": round(cov / gt_amt, 3),
            })
        t += bin_s
    return bins


def score_srt(gt_path: str | Path, cand_path: str | Path, name: str = "") -> Dict:
    gt = load_srt(gt_path)
    cand = load_srt(cand_path)
    m: Dict = {"name": name or Path(cand_path).stem}
    m.update(content_metrics(gt, cand))
    m.update(timing_metrics(gt, cand))
    m["region_recall"] = region_recall(gt, cand)
    return m


def _main(argv=None):
    ap = argparse.ArgumentParser(description="Score candidate SRT vs ground-truth SRT")
    ap.add_argument("ground_truth")
    ap.add_argument("candidate")
    ap.add_argument("--name", default="")
    ap.add_argument("--json", action="store_true", help="emit full JSON incl. region recall")
    args = ap.parse_args(argv)

    m = score_srt(args.ground_truth, args.candidate, args.name)
    if args.json:
        print(json.dumps(m, ensure_ascii=False, indent=2))
        return 0
    print(f"# {m['name']}")
    print(f"  CER {m['cer']:.3f}   sub {m['sub_rate']:.3f}  del {m['del_rate']:.3f}  ins {m['ins_rate']:.3f}"
          f"   char_recall {m['char_recall']:.3f}")
    print(f"  segs {m['n_subs']}/{m['n_subs_gt']} (ratio {m['seg_ratio']})   "
          f"mean_dur {m['mean_dur']}s (gt {m['mean_dur_gt']}s)")
    print(f"  time_recall {m['time_recall']:.3f}   time_precision {m['time_precision']:.3f}")
    worst = sorted(m["region_recall"], key=lambda b: b["recall"])[:5]
    if worst:
        print("  weakest regions: " + ", ".join(f"{b['bin']}={b['recall']:.2f}" for b in worst))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
