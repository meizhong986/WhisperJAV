"""Line and character accounting against the ground truth (owner's questions W1-W5, 2026-10-05).

Base: every ground-truth line and character of the 7 reference clips (305 lines, 3,115 characters after the
cer_trace.py normalisation). Each clip's whole ground-truth text is aligned with our whole text (character edit
operations); every ground-truth character is then exactly one of: correct, wrong (substituted), missing (deleted).
Our characters with no counterpart are "extra".

Per ground-truth line:
  missing line         none of its characters appear in our text (all deleted)
  mostly missing line  80 % or more of its characters deleted (includes the missing lines)
  produced line        every other line; its characters are split into correct / wrong / missing
For missing and wrong characters in produced lines: position in the line (first quarter / middle half / last
quarter, by character index) against the share of all produced-line characters in that position; and the very
first and very last character of each line.
Usage: python line_accounting.py <run> [<run> ...]   prints one block per run and a summary table."""
import collections
import re
import sys
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from whisperjav.bench.timing import load_srt
from whisperjav.bench.timing_cli import clip_id_of, find_hyp

REF = Path(__file__).resolve().parents[4] / "test_media" / "Ground_Truths" / "Netflix"


def norm(t):
    t = re.sub(r"[（(][^）)]*[）)]", "", t)
    t = unicodedata.normalize("NFKC", t)
    return "".join(ch for ch in t if unicodedata.category(ch)[0] in "LN")


def position(k, n):
    frac = (k + 0.5) / n
    return "first quarter" if frac < 0.25 else ("last quarter" if frac > 0.75 else "middle half")


summary = []
for arg in sys.argv[1:]:
    run = Path(arg)
    name = run.parent.name if run.name.startswith("run") else run.name
    n_lines = n_chars = 0
    lines_missing = lines_mostly = 0
    chars_in_missing_lines = chars_in_mostly = 0
    prod = collections.Counter()                  # correct / wrong / missing in produced lines
    pos_all, pos_missing, pos_wrong = collections.Counter(), collections.Counter(), collections.Counter()
    first_last = collections.Counter()            # (kind, 'first'/'last')
    first_last_all = collections.Counter()
    extra = our_chars = our_lines = 0
    for rp in sorted(REF.glob("*.srt")):
        refs = sorted(load_srt(rp, reference=True), key=lambda r: r["start"])
        hyps = sorted(load_srt(find_hyp(run, clip_id_of(rp))), key=lambda h: h["start"])
        our_lines += len(hyps)
        line_of, idx_in, len_of = [], [], []
        for li, r in enumerate(refs):
            t = norm(r["text"])
            for k in range(len(t)):
                line_of.append(li)
                idx_in.append(k)
                len_of.append(len(t))
        g = "".join(norm(r["text"]) for r in refs)
        h = "".join(norm(x["text"]) for x in hyps)
        our_chars += len(h)
        state = ["correct"] * len(g)
        for op in Levenshtein.editops(g, h):
            if op.tag == "delete":
                state[op.src_pos] = "missing"
            elif op.tag == "replace":
                state[op.src_pos] = "wrong"
            else:
                extra += 1
        per_line = collections.defaultdict(list)
        for i, st in enumerate(state):
            per_line[line_of[i]].append(i)
        for li, r in enumerate(refs):
            idxs = per_line.get(li, [])
            n = len(idxs)
            if n == 0:
                continue                           # a line with no letters after normalisation
            n_lines += 1
            n_chars += n
            d = sum(state[i] == "missing" for i in idxs)
            if d == n:
                lines_missing += 1
                chars_in_missing_lines += n
            if d >= 0.8 * n:
                lines_mostly += 1
                chars_in_mostly += n
            if d == n:
                continue
            for i in idxs:
                st = state[i]
                prod[st] += 1
                p = position(idx_in[i], len_of[i])
                pos_all[p] += 1
                end = "first" if idx_in[i] == 0 else ("last" if idx_in[i] == len_of[i] - 1 else None)
                if end:
                    first_last_all[end] += 1
                if st == "missing":
                    pos_missing[p] += 1
                    if end:
                        first_last[("missing", end)] += 1
                elif st == "wrong":
                    pos_wrong[p] += 1
                    if end:
                        first_last[("wrong", end)] += 1

    pc = sum(prod.values())
    print(f"\n=== {name}  (our lines {our_lines}, our characters {our_chars})")
    print(f"Ground truth: {n_lines} lines, {n_chars} characters")
    print(f"  missing lines (no character present): {lines_missing} = {lines_missing / n_lines:.1%} of lines, "
          f"holding {chars_in_missing_lines} characters = {chars_in_missing_lines / n_chars:.1%} of all characters")
    print(f"  mostly missing lines (>= 80 %):       {lines_mostly} = {lines_mostly / n_lines:.1%} of lines, "
          f"{chars_in_mostly / n_chars:.1%} of characters")
    print(f"  produced lines: {n_lines - lines_missing}, {pc} characters: correct {prod['correct'] / pc:.1%}, "
          f"wrong {prod['wrong'] / pc:.1%}, missing {prod['missing'] / pc:.1%}")
    print(f"  extra characters (ours, no counterpart): {extra}")
    for kind, ctr in (("missing", pos_missing), ("wrong", pos_wrong)):
        tot = max(1, sum(ctr.values()))
        parts = "  ".join(f"{p} {ctr[p] / tot:.0%} (expected {pos_all[p] / max(1, pc):.0%})"
                          for p in ("first quarter", "middle half", "last quarter"))
        fl = (f"first character of line {first_last[(kind, 'first')] / max(1, first_last_all['first']):.0%} missing-or-"
              if False else "")
        print(f"  {kind} chars in produced lines by position: {parts}")
        print(f"    rate at the line's first character {first_last[(kind, 'first')] / max(1, first_last_all['first']):.1%}, "
              f"last character {first_last[(kind, 'last')] / max(1, first_last_all['last']):.1%}, "
              f"all characters {ctr_total / pc:.1%}" if (ctr_total := sum(ctr.values())) is not None else "")
    summary.append((name, our_lines, lines_missing / n_lines, chars_in_missing_lines / n_chars,
                    prod["correct"] / pc, prod["wrong"] / pc, prod["missing"] / pc, extra,
                    pos_missing["first quarter"] / max(1, sum(pos_missing.values())),
                    pos_missing["last quarter"] / max(1, sum(pos_missing.values())),
                    pos_wrong["first quarter"] / max(1, sum(pos_wrong.values())),
                    pos_wrong["last quarter"] / max(1, sum(pos_wrong.values())),
                    pos_all["first quarter"] / max(1, pc), pos_all["last quarter"] / max(1, pc)))

print("\n| Run | Our lines | Missing lines | Chars in missing lines | Produced lines: correct | wrong | missing "
      "| Extra chars | Missing in first / last quarter | Wrong in first / last quarter | Expected first / last |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in summary:
    print(f"| {r[0]} | {r[1]} | {r[2]:.1%} | {r[3]:.1%} | {r[4]:.1%} | {r[5]:.1%} | {r[6]:.1%} | {r[7]} "
          f"| {r[8]:.0%} / {r[9]:.0%} | {r[10]:.0%} / {r[11]:.0%} | {r[12]:.0%} / {r[13]:.0%} |")
