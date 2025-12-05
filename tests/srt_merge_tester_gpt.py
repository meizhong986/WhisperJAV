#!/usr/bin/env python3
"""
SRT Merge Tester Utility

Usage:
    python srt_merge_tester.py input1.srt input2.srt merged.srt <option>

Options (string):
    base1_fill2         - first SRT as base, fill gaps from second
    base2_fill1         - second SRT as base, fill gaps from first
    combine_both        - combine both regardless of overlap
    base1_fill2_30pct   - base1, fill from 2, allow 30% overlap
    base2_fill1_30pct   - base2, fill from 1, allow 30% overlap

Notes & assumptions:
 - Expects UTF-8 encoded .srt files. Will raise on decoding errors.
 - Uses the `srt` package (pip install srt) for parsing and rendering.
 - Timing tolerance: 0.5 seconds for matching start/end times unless exact match required.
 - For "30pct" modes, an overlap is allowed when overlap_duration < 0.3 * min(duration_a, duration_b).

The tool performs multiple validations and prints a summary with PASS/FAIL and details.
"""

import argparse
import sys
import srt
import datetime
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("srt_merge_tester")

# Global tolerances
TIME_TOLERANCE_SECONDS = 0.5


def load_srt_file(path: str) -> List[srt.Subtitle]:
    """Load an SRT file as UTF-8 and parse into srt.Subtitle list.
    Will raise UnicodeDecodeError if not valid UTF-8.
    """
    logger.info(f"Loading SRT: {path}")
    with open(path, 'r', encoding='utf-8', errors='strict') as f:
        raw = f.read()
    subs = list(srt.parse(raw))
    # Normalize: sort by start
    subs.sort(key=lambda x: x.start)
    logger.info(f"  -> {len(subs)} subtitles parsed")
    return subs


def subtitle_key(sub: srt.Subtitle) -> Tuple[float,float,str]:
    """Return a simple comparable key (start_seconds, end_seconds, normalized_text)"""
    start = sub.start.total_seconds()
    end = sub.end.total_seconds()
    text = " ".join(sub.content.split())
    return (start, end, text)


def find_matching_sub(target: srt.Subtitle, candidates: List[srt.Subtitle], time_tol: float = TIME_TOLERANCE_SECONDS) -> List[srt.Subtitle]:
    """Return candidate subs that match target by text and timing within tolerance."""
    matches = []
    tstart = target.start.total_seconds()
    tend = target.end.total_seconds()
    ttext = " ".join(target.content.split())
    for c in candidates:
        cstart = c.start.total_seconds()
        cend = c.end.total_seconds()
        ctext = " ".join(c.content.split())
        if ttext == ctext and abs(cstart - tstart) <= time_tol and abs(cend - tend) <= time_tol:
            matches.append(c)
    return matches


def duration(sub: srt.Subtitle) -> float:
    return (sub.end - sub.start).total_seconds()


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return overlap duration in seconds between intervals."""
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def build_timeline(subs: List[srt.Subtitle]) -> List[Tuple[float,float,srt.Subtitle]]:
    """Return list of (start, end, subtitle) for quick scanning."""
    return [(s.start.total_seconds(), s.end.total_seconds(), s) for s in subs]


def check_no_overlap(subs: List[srt.Subtitle], allow_overlap: bool=False, allow_pairs: List[Tuple[int,int]] = None) -> List[str]:
    """Check for overlapping subtitles within a list. Returns list of error messages if overlaps found.
    allow_overlap unused in simple mode; allow_pairs can contain indices of allowed overlapping pairs (not implemented by index here).
    """
    errs = []
    if not subs:
        return errs
    sorted_subs = sorted(subs, key=lambda s: s.start)
    for i in range(len(sorted_subs)-1):
        cur = sorted_subs[i]
        nxt = sorted_subs[i+1]
        if cur.end > nxt.start + datetime.timedelta(seconds=1e-9):
            errs.append(f"Overlap detected between #{i+1} ({cur.start} -> {cur.end}) and #{i+2} ({nxt.start} -> {nxt.end})")
    return errs


def find_gaps(base: List[srt.Subtitle], total_range: Tuple[float,float]=None) -> List[Tuple[float,float]]:
    """Return list of gaps (start,end) in base timeline. If total_range provided, consider from total_range[0] to total_range[1]."""
    if not base:
        if total_range:
            return [total_range]
        return []
    intervals = sorted([(s.start.total_seconds(), s.end.total_seconds()) for s in base], key=lambda x: x[0])
    gaps = []
    if total_range:
        cur = total_range[0]
        end_limit = total_range[1]
    else:
        cur = intervals[0][0]
        end_limit = intervals[-1][1]
    for st, en in intervals:
        if st > cur:
            gaps.append((cur, st))
        cur = max(cur, en)
    if cur < end_limit:
        gaps.append((cur, end_limit))
    return gaps


def subtitle_in_interval(sub: srt.Subtitle, interval: Tuple[float,float]) -> bool:
    st = sub.start.total_seconds()
    en = sub.end.total_seconds()
    return st >= interval[0] - TIME_TOLERANCE_SECONDS and en <= interval[1] + TIME_TOLERANCE_SECONDS


def assert_base_fill(base: List[srt.Subtitle], filler: List[srt.Subtitle], merged: List[srt.Subtitle], allow_30pct: bool=False) -> Tuple[bool, List[str]]:
    """Validate base-fill merge: all base subs must be present in merged; filler subs that fall in gaps should be present; no unexpected deletions.
    Returns (ok, messages)
    """
    msgs = []
    ok = True
    # Check base subs preserved
    for i, b in enumerate(base):
        matches = find_matching_sub(b, merged)
        if not matches:
            ok = False
            msgs.append(f"Base subtitle #{i+1} not found in merged (start={b.start}, end={b.end}, text={b.content!r})")
    # Identify gaps in base across the combined span of base and filler
    min_time = min((base[0].start.total_seconds() if base else filler[0].start.total_seconds()),
                   (filler[0].start.total_seconds() if filler else base[0].start.total_seconds())) if (base or filler) else 0
    max_time = max((base[-1].end.total_seconds() if base else filler[-1].end.total_seconds()),
                   (filler[-1].end.total_seconds() if filler else base[-1].end.total_seconds())) if (base or filler) else 0
    gaps = find_gaps(base, total_range=(min_time, max_time))
    # For each filler sub that lies (fully) in a gap, ensure it exists in merged
    for j, f in enumerate(filler):
        fstart = f.start.total_seconds(); fend = f.end.total_seconds()
        in_gap = any(fstart + 1e-9 >= g[0] - TIME_TOLERANCE_SECONDS and fend <= g[1] + TIME_TOLERANCE_SECONDS for g in gaps)
        if in_gap:
            matches = find_matching_sub(f, merged)
            if not matches:
                ok = False
                msgs.append(f"Filler subtitle #{j+1} that falls in base-gap not found in merged (start={f.start}, end={f.end}, text={f.content!r})")
    # If allow_30pct, allow some overlaps between base and filler to remain in merged; else ensure there is no unexpected overlap introduced
    if not allow_30pct:
        # detect overlaps in merged that are not present in base
        merged_overlaps = check_no_overlap(merged)
        if merged_overlaps:
            ok = False
            msgs.append("Merged contains overlaps (not allowed in this mode):")
            msgs.extend(merged_overlaps[:10])
    else:
        # Validate that any overlaps between base and filler in merged are below 30% threshold
        # Build timeline arrays
        base_tl = build_timeline(base)
        filler_tl = build_timeline(filler)
        for bstart, bend, bsub in base_tl:
            for fstart, fend, fsub in filler_tl:
                ov = interval_overlap(bstart, bend, fstart, fend)
                if ov > 0:
                    min_dur = min(bend - bstart, fend - fstart)
                    if ov >= 0.3 * min_dur - 1e-9:
                        msgs.append(f"Source overlap >=30% between base sub ({bsub.start}->{bsub.end}) and filler ({fsub.start}->{fsub.end}) -- expected such overlaps to be avoided or handled. overlap={ov:.3f}s, min_dur={min_dur:.3f}s")
        # Now check merged overlaps and whether those correspond to permitted overlaps
        merged_tl = build_timeline(merged)
        # brute-force check any pair overlaps in merged
        for i in range(len(merged_tl)):
            for j in range(i+1, len(merged_tl)):
                astart, aend, a = merged_tl[i]
                bstart, bend, b = merged_tl[j]
                ov = interval_overlap(astart, aend, bstart, bend)
                if ov > 0:
                    # allow if corresponds to base/filler short overlap
                    # find if this pair corresponds to a base and a filler original pair
                    corresponds = False
                    for bstart_o, bend_o, bsub_o in base_tl:
                        for fstart_o, fend_o, fsub_o in filler_tl:
                            if (abs(astart - bstart_o) <= TIME_TOLERANCE_SECONDS and abs(aend - bend_o) <= TIME_TOLERANCE_SECONDS and abs(bstart - fstart_o) <= TIME_TOLERANCE_SECONDS and abs(bend - fend_o) <= TIME_TOLERANCE_SECONDS) or (
                               abs(astart - fstart_o) <= TIME_TOLERANCE_SECONDS and abs(aend - fend_o) <= TIME_TOLERANCE_SECONDS and abs(bstart - bstart_o) <= TIME_TOLERANCE_SECONDS and abs(bend - bend_o) <= TIME_TOLERANCE_SECONDS):
                                # this merged overlap is exactly the two original overlapping subs
                                if ov < 0.3 * min(duration(a), duration(b)):
                                    corresponds = True
                    if not corresponds:
                        msgs.append(f"Merged contains overlap between ({a.start}->{a.end}) and ({b.start}->{b.end}) not matching an allowed base/filler short overlap (ov={ov:.3f}s)")
    return (ok, msgs)


def assert_combine_both(a: List[srt.Subtitle], b: List[srt.Subtitle], merged: List[srt.Subtitle]) -> Tuple[bool, List[str]]:
    msgs = []
    ok = True
    # Check every subtitle from A and B appears in merged (within tolerance)
    for i, sub in enumerate(a):
        matches = find_matching_sub(sub, merged)
        if not matches:
            ok = False
            msgs.append(f"Source A subtitle #{i+1} not found in merged: start={sub.start}, end={sub.end}, text={sub.content!r}")
    for i, sub in enumerate(b):
        matches = find_matching_sub(sub, merged)
        if not matches:
            ok = False
            msgs.append(f"Source B subtitle #{i+1} not found in merged: start={sub.start}, end={sub.end}, text={sub.content!r}")
    return (ok, msgs)


def general_validations(merged: List[srt.Subtitle]) -> List[str]:
    msgs = []
    # Check sequential ordering and indices
    for i, s in enumerate(merged, start=1):
        if s.index != i:
            msgs.append(f"Subtitle numbering inconsistent at position {i}: file index {s.index}")
    # Check no overlapping subtitles (hard rule unless combine mode) - report but not fail here
    overlaps = check_no_overlap(merged)
    if overlaps:
        msgs.append("Overlaps detected in merged (details):")
        msgs.extend(overlaps[:20])
    return msgs


def summarize_and_exit(results: dict):
    """Print summary and exit with code 0 on success, 2 on failures."""
    print("\n===== MERGE TEST SUMMARY =====")
    for k, v in results.items():
        status = "PASS" if v['ok'] else "FAIL"
        print(f"{k}: {status}")
        if v.get('messages'):
            for m in v['messages'][:10]:
                print(f"  - {m}")
            if len(v['messages']) > 10:
                print(f"  ... and {len(v['messages'])-10} more messages")
    overall_ok = all(v['ok'] for v in results.values())
    print("==============================")
    if overall_ok:
        print("All checks passed ✅")
        sys.exit(0)
    else:
        print("Some checks failed ❌")
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="SRT Merge Test Utility")
    parser.add_argument('srt1', help='First input SRT file (base or source A)')
    parser.add_argument('srt2', help='Second input SRT file (filler or source B)')
    parser.add_argument('merged', help='Merged result SRT file to validate')
    parser.add_argument('mode', choices=['base1_fill2','base2_fill1','combine_both','base1_fill2_30pct','base2_fill1_30pct'], help='Merge mode used to produce merged.srt')
    args = parser.parse_args()

    try:
        s1 = load_srt_file(args.srt1)
        s2 = load_srt_file(args.srt2)
        sm = load_srt_file(args.merged)
    except UnicodeDecodeError as e:
        logger.error("File is not valid UTF-8. Ensure files are saved in UTF-8 encoding.")
        logger.error(str(e))
        sys.exit(3)
    except Exception as e:
        logger.error(f"Error reading/parsing srt files: {e}")
        sys.exit(4)

    results = {}

    # General validations
    results['general'] = {'ok': True, 'messages': []}
    gen_msgs = general_validations(sm)
    if gen_msgs:
        results['general']['ok'] = False
        results['general']['messages'] = gen_msgs

    # Mode-specific checks
    if args.mode == 'combine_both':
        ok, msgs = assert_combine_both(s1, s2, sm)
        results['combine_both'] = {'ok': ok, 'messages': msgs}
    elif args.mode == 'base1_fill2':
        ok, msgs = assert_base_fill(s1, s2, sm, allow_30pct=False)
        results['base1_fill2'] = {'ok': ok, 'messages': msgs}
    elif args.mode == 'base2_fill1':
        ok, msgs = assert_base_fill(s2, s1, sm, allow_30pct=False)
        results['base2_fill1'] = {'ok': ok, 'messages': msgs}
    elif args.mode == 'base1_fill2_30pct':
        ok, msgs = assert_base_fill(s1, s2, sm, allow_30pct=True)
        results['base1_fill2_30pct'] = {'ok': ok, 'messages': msgs}
    elif args.mode == 'base2_fill1_30pct':
        ok, msgs = assert_base_fill(s2, s1, sm, allow_30pct=True)
        results['base2_fill1_30pct'] = {'ok': ok, 'messages': msgs}

    summarize_and_exit(results)

if __name__ == '__main__':
    main()
