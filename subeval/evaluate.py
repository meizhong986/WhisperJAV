"""Indicators for one clip (a reference SRT and a test SRT), all derived from one time-aware character alignment.
Definitions are in definitions.py (the same text goes into every report) and METHOD.md."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .align import align, levenshtein
from .normalize import normalizer
from .srt import Cue

WITHIN = 0.5  # seconds, for "ends/starts within 0.5 s"


@dataclass
class RefLine:
    idx: int  # position in time order
    number: int  # the block's number in the SRT file (its position among read blocks if it has none)
    start: float
    end: float
    text: str
    n: int
    hits: int = 0
    wrong: int = 0
    missing: int = 0
    test_lines: list[int] = field(default_factory=list)  # our lines holding its hit or wrong characters
    home: int | None = None
    missed: bool = False
    missed_kind: str = ""  # "nothing" (every character missing) or "wrong only" (paired, but no character right)
    start_class: str = ""  # ok / lost / wrong / misplaced / uncertain ("" for missed or empty lines)
    end_class: str = ""
    lead_lost: int = 0
    tail_lost: int = 0
    start_err: float | None = None
    end_err: float | None = None


@dataclass
class TestLine:
    idx: int
    number: int
    start: float
    end: float
    text: str
    n: int
    hits: int = 0
    wrong: int = 0
    extra: int = 0
    ref_lines: list[int] = field(default_factory=list)
    hallucination: bool = False  # no character paired with any reference character
    misheard: bool = False  # paired characters, but none of them right


def _streams(cues, norm):
    chars, owner, lines = [], [], []
    for k, c in enumerate(cues):
        t = norm(c.text)
        chars.append(t)
        owner += [k] * len(t)
        lines.append((c.start, c.end))
    return "".join(chars), owner, lines


def _classify(ref: list[RefLine], tests: list[TestLine], ops, r_owner, t_owner):
    for r in ref:
        r.hits = r.wrong = r.missing = 0
        r.test_lines, r.home, r.missed, r.missed_kind = [], None, False, ""
        r.start_class = r.end_class = ""
        r.lead_lost = r.tail_lost = 0
        r.start_err = r.end_err = None
    for t in tests:
        t.hits = t.wrong = t.extra = 0
        t.ref_lines, t.hallucination, t.misheard = [], False, False
    status = {}   # reference char index -> (kind, test line or None)
    hits_in = {}  # (ref line, test line) -> hits
    first_hit_of_test, last_hit_of_test = {}, {}  # test line -> reference line of its first / last hit
    first_hit_of_ref, last_hit_of_ref = {}, {}    # reference line -> test line holding its first / last hit
    for kind, i, j in ops:
        if kind == "extra":
            tests[t_owner[j]].extra += 1
            continue
        r = ref[r_owner[i]]
        if kind == "missing":
            r.missing += 1
            status[i] = ("missing", None)
            continue
        tl = t_owner[j]
        status[i] = (kind, tl)
        if kind == "hit":
            r.hits += 1
            tests[tl].hits += 1
            hits_in[(r.idx, tl)] = hits_in.get((r.idx, tl), 0) + 1
            first_hit_of_test.setdefault(tl, r.idx)
            last_hit_of_test[tl] = r.idx
            first_hit_of_ref.setdefault(r.idx, tl)
            last_hit_of_ref[r.idx] = tl
        else:
            r.wrong += 1
            tests[tl].wrong += 1
        if tl not in r.test_lines:
            r.test_lines.append(tl)
        if r.idx not in tests[tl].ref_lines:
            tests[tl].ref_lines.append(r.idx)
    pos = 0
    for r in ref:
        first, last = pos, pos + r.n - 1
        pos += r.n
        if r.n == 0:
            continue
        if r.hits == 0:
            r.missed = True
            r.missed_kind = "wrong only" if r.wrong else "nothing"
            continue
        # home: our line with the most hit characters of this line; ties -> the earliest such line
        r.home = min((tl for tl in r.test_lines if hits_in.get((r.idx, tl))),
                     key=lambda tl: (-hits_in[(r.idx, tl)], tl))
        for edge, idxs in (("start", range(first, last + 1)), ("end", range(last, first - 1, -1))):
            kind, tl = status[idxs[0]]
            cls = ("lost" if kind == "missing" else "wrong" if kind == "wrong"
                   else "ok" if tl == r.home else "misplaced")
            lost = 0
            for k in idxs:
                if status[k][0] != "missing":
                    break
                lost += 1
            if edge == "start":
                r.start_class, r.lead_lost = cls, lost
            else:
                r.end_class, r.tail_lost = cls, lost
    for t in tests:
        t.hallucination = t.n > 0 and t.hits == 0 and t.wrong == 0
        t.misheard = t.n > 0 and t.hits == 0 and t.wrong > 0
    # timing: a test line's start is measured when its first hit is the first hit of that reference line (the test
    # line opens the reference line); its end when its last hit is the last hit of that reference line. A test-line
    # edge that falls inside a reference line has no reference edge to be measured against and is not timed.
    for tl, ri in first_hit_of_test.items():
        if first_hit_of_ref[ri] == tl:
            ref[ri].start_err = tests[tl].start - ref[ri].start
    for tl, ri in last_hit_of_test.items():
        if last_hit_of_ref[ri] == tl:
            ref[ri].end_err = tests[tl].end - ref[ri].end


@dataclass
class ClipResult:
    clip: str
    ref: list[RefLine]
    test: list[TestLine]
    n_ref_chars: int
    plain_distance: int
    edits: int
    wrong: int
    missing: int
    extra: int
    ambiguous_ref_lines: list[int]  # reference lines with any indicator differing under the mirror alignment
    ambiguous_test_lines: list[int]
    ambiguity: dict  # indicator -> reference (or test) line indexes whose value differs under the mirror
    params: dict


AMBIGUITY_FIELDS = {"missed": ("missed", "missed_kind"), "start": ("start_class", "lead_lost"),
                    "end": ("end_class", "tail_lost"), "timing": ("start_err", "end_err")}


def _line_signature(r: RefLine):
    return {k: tuple(getattr(r, a) for a in attrs) for k, attrs in AMBIGUITY_FIELDS.items()}


def evaluate_clip(ref_cues: list[Cue], test_cues: list[Cue], clip: str = "", window: float = 1.0,
                  profile: str = "ja-v1") -> ClipResult:
    """Align forward and mirror; every indicator is read from the forward alignment, except the start and end
    classes, which are kept only where both alignments agree ("uncertain" otherwise), with the lower of the two
    lost-character counts. (With equal-cost alternatives the forward tie order moves losses towards line starts and
    the mirror order towards line ends, so either one alone would bias these two indicators.)"""
    norm = normalizer(profile)
    ref_cues = sorted(ref_cues, key=lambda c: (c.start, c.end))
    test_cues = sorted(test_cues, key=lambda c: (c.start, c.end))
    rs, r_owner, r_lines = _streams(ref_cues, norm)
    ts, t_owner, t_lines = _streams(test_cues, norm)
    ref = [RefLine(k, c.number, c.start, c.end, c.text, len(norm(c.text))) for k, c in enumerate(ref_cues)]
    tests = [TestLine(k, c.number, c.start, c.end, c.text, len(norm(c.text))) for k, c in enumerate(test_cues)]
    ops_m, _ = align(rs, r_owner, r_lines, ts, t_owner, t_lines, window, "mirror")
    _classify(ref, tests, ops_m, r_owner, t_owner)
    sig_r = [_line_signature(r) for r in ref]
    edges_m = [(r.start_class, r.lead_lost, r.end_class, r.tail_lost) for r in ref]
    sig_t = [(t.hallucination, t.misheard) for t in tests]
    ops, edits = align(rs, r_owner, r_lines, ts, t_owner, t_lines, window, "forward")
    _classify(ref, tests, ops, r_owner, t_owner)
    amb = {k: [r.idx for r, s in zip(ref, sig_r) if _line_signature(r)[k] != s[k]] for k in AMBIGUITY_FIELDS}
    amb["hallucination"] = [t.idx for t, s in zip(tests, sig_t) if (t.hallucination, t.misheard) != s]
    amb_r = sorted({i for k in AMBIGUITY_FIELDS for i in amb[k]})
    for r, (sc, ll, ec, tl) in zip(ref, edges_m):
        if r.missed or r.n == 0:
            continue
        if r.start_class != sc:
            r.start_class = "uncertain"
        r.lead_lost = min(r.lead_lost, ll)
        if r.end_class != ec:
            r.end_class = "uncertain"
        r.tail_lost = min(r.tail_lost, tl)
    count = lambda k: sum(1 for o in ops if o[0] == k)
    return ClipResult(clip=clip, ref=ref, test=tests, n_ref_chars=len(rs), plain_distance=levenshtein(rs, ts),
                      edits=edits, wrong=count("wrong"), missing=count("missing"), extra=count("extra"),
                      ambiguous_ref_lines=amb_r, ambiguous_test_lines=amb["hallucination"], ambiguity=amb,
                      params={"window_s": window, "profile": profile, "within_s": WITHIN})


def _median(v):
    return statistics.median(v) if v else None


def _mean(v):
    return statistics.fmean(v) if v else None


def summarize(results: list[ClipResult]) -> dict:
    """Pooled indicators over clips (sums of counts; medians/means over all pooled lines)."""
    R = sum(c.n_ref_chars for c in results)
    refs = [r for c in results for r in c.ref]
    tests = [t for c in results for t in c.test]
    live = [r for r in refs if r.n > 0]
    delivered = [r for r in live if not r.missed]
    st = [r.start_err for r in refs if r.start_err is not None]
    en = [r.end_err for r in refs if r.end_err is not None]
    cls = lambda attr, v: sum(1 for r in delivered if getattr(r, attr) == v)
    out = {
        "clips": len(results),
        "ref_lines": len(refs), "ref_lines_empty": len(refs) - len(live), "ref_chars": R,
        "test_lines": len(tests), "test_lines_empty": sum(1 for t in tests if t.n == 0),
        "cer_plain": sum(c.plain_distance for c in results) / R if R else None,
        "cer": sum(c.edits for c in results) / R if R else None,
        "wrong_rate": sum(c.wrong for c in results) / R if R else None,
        "missing_rate": sum(c.missing for c in results) / R if R else None,
        "extra_rate": sum(c.extra for c in results) / R if R else None,
        "edits": sum(c.edits for c in results),
        "missed_lines": sum(1 for r in live if r.missed),
        "missed_lines_short": sum(1 for r in live if r.missed and r.n <= 2),
        "missed_nothing": sum(1 for r in live if r.missed_kind == "nothing"),
        "missed_wrong_only": sum(1 for r in live if r.missed_kind == "wrong only"),
        "hallucination_lines": sum(1 for t in tests if t.hallucination),
        "hallucination_chars": sum(t.n for t in tests if t.hallucination),
        "misheard_test_lines": sum(1 for t in tests if t.misheard),
        "single_hit_ref_lines": sum(1 for r in live if r.n >= 4 and r.hits == 1),
        "single_hit_test_lines": sum(1 for t in tests if t.n >= 4 and t.hits == 1),
        "delivered_lines": len(delivered),
    }
    for edge in ("start", "end"):
        for v in ("ok", "lost", "wrong", "misplaced", "uncertain"):
            out[f"{edge}_{v}"] = cls(f"{edge}_class", v)
        lost = [getattr(r, "lead_lost" if edge == "start" else "tail_lost") for r in delivered]
        out[f"{edge}_lost_chars"] = sum(lost)
    for name, v in (("start", st), ("end", en)):
        out[f"{name}_measured"] = len(v)
        out[f"{name}_abs_median"] = _median([abs(x) for x in v])
        out[f"{name}_signed_mean"] = _mean(v)
        out[f"{name}_signed_median"] = _median(v)
        out[f"{name}_within_share"] = (sum(1 for x in v if abs(x) <= WITHIN) / len(v)) if v else None
    out["ambiguous_ref_lines"] = sum(len(c.ambiguous_ref_lines) for c in results)
    out["ambiguous_test_lines"] = sum(len(c.ambiguous_test_lines) for c in results)
    for k in list(AMBIGUITY_FIELDS) + ["hallucination"]:
        out[f"ambiguous_{k}"] = sum(len(c.ambiguity[k]) for c in results)
    return out
