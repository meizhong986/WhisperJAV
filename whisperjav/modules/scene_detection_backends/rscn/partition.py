"""R-SCN partition: from the measures, the speech stretches and the black stretches to scenes that divide the audio
exactly, each with its marks. Design: docs/plans/design/2026-11-02-rscn-design.md sections 5-7.

Owner's rules applied here: scenes of min_scene_s-max_scene_s, preferably preferred_min_s-preferred_max_s, with no
length cost inside that range (W451 C2); cuts at the strongest changes, placed in a pause at most max_cut_shift_s away
(W451 C2); a cut forced by the length is marked (W451 C2); digital silence handed on as its own scenes (W450 2d,
W451 C1); no overlap (W450 2c); confirmed black picture is a fixed cut (W450 2g; v6 7.3).
Correctness rules (not settings): every sample in exactly one scene; checked before anything is written.
"""

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from .analysis import Measures, digital_silence, quiet_stretches, strength
from .picture import BlackStretch
from .settings import RSCNSettings
from .speech import SpeechStretch


@dataclass
class Candidate:
    at_s: float           # where the cut would go (after placement)
    origin_s: float       # where the change (or pause, or grid point) was
    strength: float
    kind: str             # change_peak | pause | speech_gap | black | grid
    placement: str        # pause | speech_gap | no_pause | in_speech
    in_speech: bool = False

    @property
    def shift_s(self) -> float:
        return self.at_s - self.origin_s

    @property
    def is_change(self) -> bool:
        """Only a change peak or a black stretch is a change; a pause or a grid point is a place to cut when the
        length needs one, its strength only a preference among such places."""
        return self.kind in ("change_peak", "black")


@dataclass
class Cut:
    at_s: float
    reason: str           # change | to_fit_length | major_break | digital_silence | equal_split
    strength: float = 0.0
    placement: str = "fixed"
    shift_s: float = 0.0
    kind: str = ""


@dataclass
class Act:
    index: int
    start_s: float
    end_s: float
    candidates: int = 0
    cuts: List[Cut] = field(default_factory=list)


@dataclass
class ScenePlan:
    start_sample: int
    end_sample: int
    kind: str                         # audio | digital_silence
    act: Optional[int]
    start_cut: Cut
    end_cut: Cut
    duration_exception: Optional[str] = None
    marks: Dict[str, Any] = field(default_factory=dict)


class _Speech:
    """Speech stretches with fast 'is this moment in speech' and 'how much speech in this span' answers."""

    def __init__(self, stretches: Optional[Sequence[SpeechStretch]], guard_s: float):
        self.on = stretches is not None
        self.items = list(stretches or [])
        self.starts = [x.start_s for x in self.items]
        self.guard = guard_s

    # The stretches are sorted and do not overlap (speech.merge), so their ends rise too: only the last stretch that
    # starts before a moment can reach it.

    def contains(self, t: float) -> bool:
        """True when ``t`` is inside a speech stretch widened by the guard on each side."""
        k = bisect.bisect_right(self.starts, t + self.guard) - 1
        return k >= 0 and t <= self.items[k].end_s + self.guard

    def overlaps(self, a: float, b: float) -> bool:
        """True when any speech stretch overlaps [a, b]."""
        k = bisect.bisect_left(self.starts, b) - 1
        return k >= 0 and self.items[k].end_s > a

    def seconds_in(self, a: float, b: float) -> float:
        k = bisect.bisect_right(self.starts, b)
        total = 0.0
        for x in reversed(self.items[:k]):
            if x.end_s <= a:
                break
            total += max(0.0, min(b, x.end_s) - max(a, x.start_s))
        return total

    def gaps(self, a: float, b: float) -> List[float]:
        """Middles of the gaps between speech stretches inside [a, b] that are wider than two guards."""
        inside = [x for x in self.items if x.end_s > a and x.start_s < b]
        edges = [a] + [v for x in inside for v in (x.start_s, x.end_s)] + [b]
        out = []
        for i in range(0, len(edges) - 1, 2):
            g0, g1 = edges[i], edges[i + 1]
            if g1 - g0 > 2 * self.guard + 0.05:
                out.append((g0 + g1) / 2)
        return out


# ---------------------------------------------------------------------------------------------------------------------

def plan(native_samples: np.ndarray, rate: int, s: RSCNSettings, m: Measures,
         speech: Optional[Sequence[SpeechStretch]], black: Sequence[BlackStretch]) -> Tuple[List[ScenePlan], Dict]:
    """The scenes of one file. ``native_samples`` are the WAV's own samples (for digital silence and the sample
    bounds); ``m`` the measures at the analysis rate; ``speech`` the speech stretches (None: no speech measure);
    ``black`` the black stretches in WAV time. Returns the scenes and a record of the decisions."""
    n = len(native_samples)
    if n == 0:
        raise ValueError("the audio is empty")
    duration = n / rate
    sp = _Speech(speech, s.speech_guard_s)
    record: Dict[str, Any] = {"regions": [], "acts": [], "black": [], "digital_silence": []}

    zero = digital_silence(native_samples, rate, s.digital_silence_min_s)
    record["digital_silence"] = [{"start_s": round(a / rate, 3), "end_s": round(b / rate, 3)} for a, b in zero]

    # Fixed pieces: audio regions between digital-silence runs, and the runs themselves.
    pieces: List[Tuple[int, int, str]] = []
    cursor = 0
    for a, b in zero:
        if a > cursor:
            pieces.append((cursor, a, "audio"))
        pieces.append((a, b, "digital_silence"))
        cursor = b
    if cursor < n:
        pieces.append((cursor, n, "audio"))
    pieces_with_audio = _absorb_tiny(pieces, int(round(s.tiny_region_s * rate)))

    scenes: List[ScenePlan] = []
    act_index = 0
    previous_cut = Cut(0.0, "file_start")
    for k_piece, (a, b, kind, tiny_audio) in enumerate(pieces_with_audio):
        boundary_after = Cut(b / rate, "file_end" if b == n else "digital_silence")
        if kind == "digital_silence":
            parts = max(1, math.ceil((b - a) / rate / s.max_scene_s - 1e-9))
            edges = [a + ((b - a) * k) // parts for k in range(parts + 1)]
            for k in range(parts):
                end_cut = boundary_after if k == parts - 1 else Cut(edges[k + 1] / rate, "equal_split")
                one = ScenePlan(edges[k], edges[k + 1], "digital_silence", None, previous_cut, end_cut,
                                duration_exception="digital_silence")
                one.marks["short_audio_inside_s"] = round(tiny_audio / rate, 4) if (tiny_audio and k == 0) else 0.0
                scenes.append(one)
                previous_cut = end_cut
            continue
        ta, tb = a / rate, b / rate
        silence_before = k_piece > 0
        silence_after = k_piece < len(pieces_with_audio) - 1
        region_cuts, region_record, acts = _region(ta, tb, s, m, sp, black, act_index, silence_before, silence_after)
        record["regions"].append(region_record)
        record["black"].extend(region_record.pop("black"))
        bounds = [a] + [int(round(c.at_s * rate)) for c in region_cuts] + [b]
        cuts = [previous_cut] + region_cuts + [boundary_after]
        act_of = _act_lookup(acts)
        for k in range(len(bounds) - 1):
            if bounds[k + 1] <= bounds[k]:
                raise AssertionError("R-SCN: two cuts fell on the same sample")
            mid = (bounds[k] + bounds[k + 1]) / 2 / rate
            scenes.append(ScenePlan(bounds[k], bounds[k + 1], "audio", act_of(mid), cuts[k], cuts[k + 1]))
        record["acts"].extend({"act": x.index, "start_s": round(x.start_s, 3), "end_s": round(x.end_s, 3),
                               "candidates": x.candidates} for x in acts)
        act_index += len(acts)
        previous_cut = boundary_after

    _check(scenes, n)
    for sc in scenes:
        _mark(sc, rate, s, m, sp, duration)
    return scenes, record


def _absorb_tiny(pieces: List[Tuple[int, int, str]], tiny: int) -> List[Tuple[int, int, str, int]]:
    """An audio stretch shorter than ``tiny`` samples next to digital silence (a click between two runs of zeros)
    joins that silence's scene, its length kept as a mark, rather than making a scene of a few samples. Returns
    (start, end, kind, samples of audio absorbed)."""
    out: List[List[Any]] = []
    for a, b, kind in pieces:
        if kind == "audio" and b - a < tiny and out and out[-1][2] == "digital_silence":
            out[-1][1] = b
            out[-1][3] += b - a
            continue
        if kind == "digital_silence" and out and out[-1][2] == "audio" and out[-1][1] - out[-1][0] < tiny:
            small = out.pop()
            if out and out[-1][2] == "digital_silence":
                out[-1][1] = b
                out[-1][3] += small[1] - small[0]
            else:
                out.append([small[0], b, "digital_silence", small[1] - small[0]])
            continue
        if kind == "digital_silence" and out and out[-1][2] == "digital_silence":
            out[-1][1] = b
            continue
        out.append([a, b, kind, 0])
    return [(a, b, kind, extra) for a, b, kind, extra in out]


def _act_lookup(acts: List[Act]):
    starts = [x.start_s for x in acts]

    def find(t: float) -> Optional[int]:
        k = bisect.bisect_right(starts, t) - 1
        return acts[k].index if k >= 0 else None
    return find


def _rows(t0: float, t1: float, m: Measures) -> Tuple[int, int]:
    total = len(m.coarse)
    r0 = min(total, max(0, int(math.floor(t0 / m.coarse_hop_s + 1e-9))))
    r1 = min(total, max(r0, int(math.ceil(t1 / m.coarse_hop_s - 1e-9))))
    return r0, r1


def _region(ta: float, tb: float, s: RSCNSettings, m: Measures, sp: _Speech, black: Sequence[BlackStretch],
            first_act: int, silence_before: bool, silence_after: bool) -> Tuple[List[Cut], Dict, List[Act]]:
    """The cuts inside one audio region (between digital-silence runs), in time order."""
    r0, r1 = _rows(ta, tb, m)
    curve, per_scale = strength(m.coarse[r0:r1], s)
    coarsest = per_scale[int(np.argmax(s.novelty_scales_s))] if per_scale else curve
    hop = m.coarse_hop_s

    def value(arr: np.ndarray, t0: float, t1: Optional[float] = None) -> float:
        if len(arr) == 0:
            return 0.0
        i0 = min(len(arr) - 1, max(0, int(round(t0 / hop)) - r0))
        if t1 is None:
            return float(arr[i0])
        i1 = min(len(arr), max(i0 + 1, int(round(t1 / hop)) - r0 + 1))
        return float(np.max(arr[i0:i1]))

    f0 = max(0, int(math.floor(ta / m.fine_hop_s)))
    f1 = min(len(m.fine_level_db), max(f0 + 1, int(math.ceil(tb / m.fine_hop_s))))
    fine = m.fine_level_db[f0:f1]
    quiet = [(f0 * m.fine_hop_s + a, f0 * m.fine_hop_s + b) for a, b in quiet_stretches(fine, m.fine_hop_s, s)]

    # Black stretches inside the region (design 7): a stretch next to digital silence is that silence's fade and adds
    # nothing; a confirmed one is a fixed cut (major break) when it leaves at least min_scene_s on each side, else a
    # candidate like an unconfirmed one (no scene below the minimum because of the picture).
    seen: List[BlackStretch] = []
    confirmed: List[Tuple[float, float, BlackStretch]] = []
    open_black: List[BlackStretch] = []
    for bs in sorted(black, key=lambda b: b.start_s):
        if bs.end_s <= ta or bs.start_s >= tb:
            continue
        seen.append(bs)
        lo, hi = max(ta, bs.start_s), min(tb, bs.end_s)
        if (silence_before and lo - ta <= s.black_silence_merge_s) or \
                (silence_after and tb - hi <= s.black_silence_merge_s):
            bs.confirmed, bs.why = False, "next to digital silence: part of that silence's break"
            continue
        w0, w1 = max(ta, lo - s.black_confirm_margin_s), min(tb, hi + s.black_confirm_margin_s)
        speech_free = (not sp.on) or not sp.overlaps(w0, w1)
        change = value(coarsest, w0, w1)
        bs.confirmed = speech_free and change >= s.black_confirm_change
        bs.why = (("no speech measure; " if not sp.on else ("no speech; " if speech_free else "speech over it; "))
                  + f"coarse change {change:.3f} {'>=' if change >= s.black_confirm_change else '<'} "
                  f"{s.black_confirm_change:g}")
        if bs.confirmed:
            confirmed.append((_quietest(lo, hi, m), change, bs))
        else:
            open_black.append(bs)
    fixed: List[Cut] = []
    last = ta
    for at, change, bs in confirmed:
        if at - last >= s.min_scene_s and tb - at >= s.min_scene_s:
            fixed.append(Cut(at, "major_break", change, "black", 0.0, "black"))
            last = at
        else:
            bs.why += "; it would leave a scene shorter than the minimum, so it is a candidate"
            open_black.append(bs)
    black_record = [bs.as_dict() for bs in seen]

    edges = [ta] + [c.at_s for c in fixed] + [tb]
    acts = [Act(first_act + k, edges[k], edges[k + 1]) for k in range(len(edges) - 1)]
    cuts: List[Cut] = []
    for k, act in enumerate(acts):
        if k > 0:
            cuts.append(fixed[k - 1])
        cuts.extend(_act_cuts(act, s, m, sp, curve, r0, quiet, open_black, value))
    region_record = {"start_s": round(ta, 3), "end_s": round(tb, 3), "quiet_stretches": len(quiet),
                     "strength_max": round(float(curve.max()) if len(curve) else 0.0, 4),
                     "strength_p95": round(float(np.percentile(curve, 95)) if len(curve) else 0.0, 4),
                     "black": black_record}
    return cuts, region_record, acts


def _quietest(a: float, b: float, m: Measures) -> float:
    f0 = max(0, int(math.floor(a / m.fine_hop_s)))
    f1 = min(len(m.fine_level_db), max(f0 + 1, int(math.ceil(b / m.fine_hop_s))))
    if f1 <= f0:
        return (a + b) / 2
    k = int(np.argmin(m.fine_level_db[f0:f1]))
    return min(b, max(a, (f0 + k + 0.5) * m.fine_hop_s))


def _act_cuts(act: Act, s: RSCNSettings, m: Measures, sp: _Speech, curve: np.ndarray, r0: int,
              quiet: List[Tuple[float, float]], open_black: List[BlackStretch], value) -> List[Cut]:
    """The cuts inside one act, by dynamic programming over the placed candidates (design 5.3-5.4)."""
    ta, tb = act.start_s, act.end_s
    length = tb - ta
    if length < s.min_scene_s or length <= 0:
        return []
    lo, hi = ta + s.min_scene_s, tb - s.min_scene_s

    # Safe points: quiet stretches (three points each) and gaps between speech stretches, not in speech.
    safe: List[Tuple[float, str]] = []
    for qa, qb in quiet:
        if qb <= ta or qa >= tb:
            continue
        inner = min(0.2, (qb - qa) / 4)
        for p in (qa + inner, (qa + qb) / 2, qb - inner):
            if ta < p < tb and not sp.contains(p):
                safe.append((p, "pause"))
    if sp.on:
        safe.extend((p, "speech_gap") for p in sp.gaps(ta, tb) if ta < p < tb)
    safe.sort()
    safe_t = [p for p, _ in safe]

    def place(t: float) -> Tuple[float, str, bool]:
        if safe_t:
            k = bisect.bisect_left(safe_t, t)
            best = None
            for j in (k - 1, k):
                if 0 <= j < len(safe_t) and abs(safe_t[j] - t) <= s.max_cut_shift_s:
                    if best is None or abs(safe_t[j] - t) < abs(safe_t[best] - t):
                        best = j
            if best is not None:
                return safe_t[best], safe[best][1], False
        f0 = max(0, int(math.floor((t - s.max_cut_shift_s) / m.fine_hop_s)))
        f1 = min(len(m.fine_level_db), int(math.ceil((t + s.max_cut_shift_s) / m.fine_hop_s)) + 1)
        p = t
        if f1 > f0:
            # The lowest level in reach, preferring points outside speech; among points within the tolerance of
            # that level, the one nearest the change.
            levels = m.fine_level_db[f0:f1]
            times = (f0 + np.arange(f1 - f0) + 0.5) * m.fine_hop_s
            usable = (times > ta) & (times < tb)
            free = usable & np.array([not sp.contains(float(q)) for q in times]) if sp.items else usable
            pool = free if free.any() else usable
            if pool.any():
                floor = float(levels[pool].min())
                near = pool & (levels <= floor + s.placement_level_tolerance_db)
                idx = np.flatnonzero(near)
                p = float(times[idx[np.argmin(np.abs(times[idx] - t))]])
        p = min(tb - 1e-3, max(ta + 1e-3, p))
        inside = sp.contains(p)
        return p, ("in_speech" if inside else "no_pause"), inside

    raw: List[Tuple[float, float, str]] = []   # (origin time, strength, kind)
    i0 = max(0, int(round(ta / m.coarse_hop_s)) - r0)
    i1 = min(len(curve), int(round(tb / m.coarse_hop_s)) - r0)
    if i1 - i0 >= 3:
        distance = max(1, round(s.peak_min_distance_s / m.coarse_hop_s))
        peaks, _ = signal.find_peaks(curve[i0:i1], distance=distance, prominence=s.peak_prominence)
        raw.extend(((r0 + i0 + int(p)) * m.coarse_hop_s, float(curve[i0 + int(p)]), "change_peak") for p in peaks)
    raw.extend((p, value(curve, p), kind) for p, kind in safe)
    for bs in open_black:
        if bs.end_s > ta and bs.start_s < tb:
            mid = _quietest(max(ta, bs.start_s), min(tb, bs.end_s), m)
            # Black picture is evidence of a break of its own (W440): its strength gets a bonus.
            bonus = s.black_strength_bonus * s.change_threshold
            raw.append((mid, value(curve, max(ta, bs.start_s), min(tb, bs.end_s)) + bonus, "black"))
    t = ta + s.grid_step_s
    while t < tb:
        raw.append((t, value(curve, t), "grid"))
        t += s.grid_step_s

    placed: List[Candidate] = []
    for origin, strong, kind in raw:
        if kind in ("pause", "speech_gap"):
            at, how, inside = origin, kind, False
        else:
            at, how, inside = place(origin)
        if lo - s.max_cut_shift_s <= at <= hi + s.max_cut_shift_s and ta < at < tb:
            placed.append(Candidate(at, origin, strong, kind, how, inside))
    threshold = s.change_threshold
    ranked = sorted(placed, key=lambda c: (c.is_change and c.strength >= threshold, not c.in_speech, c.is_change,
                                           c.strength, c.kind != "grid"), reverse=True)
    chosen: List[Candidate] = []
    chosen_t: List[float] = []
    for c in ranked:
        k = bisect.bisect_left(chosen_t, c.at_s)
        if (k > 0 and c.at_s - chosen_t[k - 1] < s.candidate_merge_s) or \
                (k < len(chosen_t) and chosen_t[k] - c.at_s < s.candidate_merge_s):
            continue
        chosen_t.insert(k, c.at_s)
        chosen.insert(k, c)
    act.candidates = len(chosen)

    picked = _choose(ta, tb, chosen, s)
    if picked is None:
        parts = max(2, math.ceil(length / s.max_scene_s))
        return [Cut(ta + length * k / parts, "equal_split") for k in range(1, parts)]
    cuts = []
    for c in picked:
        reason = "change" if c.is_change and c.strength >= threshold else "to_fit_length"
        cuts.append(Cut(c.at_s, reason, c.strength, c.placement, c.shift_s, c.kind))
    return cuts


def length_cost(L: float, s: RSCNSettings) -> float:
    """0 inside the preferred range (W451 C2); outside it, length_weight times the square of how far into the room
    between the range and the bound the scene reaches (1.0 at min_scene_s or at max_scene_s). Both sides weigh the
    same: with the defaults a scene of 70 s or 420 s costs 8; 100 s costs 1.28 and 390 s 2.0."""
    if L < s.preferred_min_s and s.preferred_min_s > s.min_scene_s:
        return s.length_weight * ((s.preferred_min_s - L) / (s.preferred_min_s - s.min_scene_s)) ** 2
    if L > s.preferred_max_s and s.max_scene_s > s.preferred_max_s:
        return s.length_weight * ((L - s.preferred_max_s) / (s.max_scene_s - s.preferred_max_s)) ** 2
    return 0.0


def cut_cost(c: Candidate, s: RSCNSettings) -> float:
    r = c.strength / s.change_threshold
    if c.is_change and r >= 1:
        cost = -s.change_reward * min(r, s.change_reward_cap)
    else:
        cost = s.forced_cut_cost - s.forced_strength_bonus * min(r, 1.0)
    if c.in_speech:
        cost += s.in_speech_cost
    return cost


def _choose(ta: float, tb: float, cands: List[Candidate], s: RSCNSettings) -> Optional[List[Candidate]]:
    """Exact choice over the candidates: minimum of the scenes' length costs plus the cuts' costs, every scene within
    [min_scene_s, max_scene_s]. None when no such division exists."""
    eps = 1e-6
    pos = [ta] + [c.at_s for c in cands] + [tb]
    m = len(pos)
    best = [math.inf] * m
    prior = [-1] * m
    best[0] = 0.0
    costs = [0.0] + [cut_cost(c, s) for c in cands] + [0.0]
    for j in range(1, m):
        first = bisect.bisect_left(pos, pos[j] - s.max_scene_s - eps, 0, j)
        last = bisect.bisect_right(pos, pos[j] - s.min_scene_s + eps, 0, j)
        bj = math.inf
        pj = -1
        for i in range(first, last):
            if best[i] == math.inf:
                continue
            v = best[i] + length_cost(pos[j] - pos[i], s)
            if v < bj:
                bj, pj = v, i
        if pj >= 0:
            best[j] = bj + costs[j]
            prior[j] = pj
    if prior[-1] < 0:
        return None
    out = []
    k = prior[-1]
    while k > 0:
        out.append(cands[k - 1])
        k = prior[k]
    out.reverse()
    return out


def _check(scenes: List[ScenePlan], n: int) -> None:
    """Every sample in exactly one scene (correctness rule, not a setting)."""
    at = 0
    for sc in scenes:
        if sc.start_sample != at or sc.end_sample <= sc.start_sample:
            raise AssertionError(f"R-SCN: scenes leave a gap or overlap at sample {at}")
        at = sc.end_sample
    if at != n:
        raise AssertionError(f"R-SCN: the scenes end at sample {at}, the audio at {n}")


def _mark(sc: ScenePlan, rate: int, s: RSCNSettings, m: Measures, sp: _Speech, duration: float) -> None:
    a, b = sc.start_sample / rate, sc.end_sample / rate
    L = b - a
    if sc.kind == "audio" and L < s.min_scene_s:
        if sc.start_cut.reason == "file_start" and sc.end_cut.reason == "file_end":
            sc.duration_exception = "short_file"
        else:
            sc.duration_exception = "short_act"
    r0, r1 = _rows(a, b, m)
    levels = m.coarse_level_db[r0:r1]
    mean_db = float(10 * np.log10(np.mean(np.power(10.0, levels / 10.0)))) if len(levels) else None
    share = (sp.seconds_in(a, b) / L) if (sp.on and L > 0) else None
    sc.marks = {
        **sc.marks,
        "rscn_kind": sc.kind,
        "act": sc.act,
        "start_cut": sc.start_cut.reason,
        "end_cut": sc.end_cut.reason,
        "end_strength": round(sc.end_cut.strength, 4),
        "end_shift_s": round(sc.end_cut.shift_s, 3),
        "end_placement": sc.end_cut.placement,
        "end_candidate": sc.end_cut.kind,
        "speech_share": None if share is None else round(share, 4),
        "little_or_no_dialogue": None if share is None else share < s.little_speech_share,
        "duration_exception": sc.duration_exception,
        "mean_level_db": None if mean_db is None else round(mean_db, 2),
    }
