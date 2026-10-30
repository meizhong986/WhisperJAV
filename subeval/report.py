"""Reports: a Markdown report (definitions included, so it stands on its own), a JSON file with every number down to
each line, and a plain-text pairing record that shows, for every reference line, the test lines its characters
were paired with and how it was classified."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from . import __version__
from .definitions import DEFINITIONS
from .evaluate import WITHIN, ClipResult


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _f(v, fmt="{:.3f}"):
    return "-" if v is None else fmt.format(v)


def _pct(v):
    return "-" if v is None else f"{v:.0%}"


def _ts(s):
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


SUMMARY_ROWS = [
    ("CER (plain)", "cer_plain", "{:.3f}"), ("CER", "cer", "{:.3f}"), ("Wrong", "wrong_rate", "{:.3f}"),
    ("Missing", "missing_rate", "{:.3f}"), ("Extra", "extra_rate", "{:.3f}"),
    ("Reference lines (empty after normalisation)", None, None),
    ("Missed lines (nothing / wrong only; 1-2 characters)", None, None), ("Test lines (empty after normalisation)", None, None),
    ("Suspected hallucination lines (their characters)", None, None), ("Misheard test lines (paired, no character right)", "misheard_test_lines", "{}"),
    ("Lines decided by a single hit, 4+ characters (reference / test)", None, None),
    ("Start ok / lost / wrong / misplaced / uncertain", None, None), ("Start lost characters", "start_lost_chars", "{}"),
    ("End ok / lost / wrong / misplaced / uncertain", None, None), ("End lost characters", "end_lost_chars", "{}"),
    ("Starts measured", "start_measured", "{}"), ("Start median |error| (s)", "start_abs_median", "{:.3f}"),
    ("Start signed mean (s)", "start_signed_mean", "{:+.3f}"), (f"Starts within {WITHIN} s", "start_within_share", None),
    ("Ends measured", "end_measured", "{}"), ("End median |error| (s)", "end_abs_median", "{:.3f}"),
    ("End signed mean (s)", "end_signed_mean", "{:+.3f}"), (f"Ends within {WITHIN} s", "end_within_share", None),
    ("Ambiguity: lines whose missed / start / end / timing / hallucination value differs", None, None),
]


def summary_cells(s: dict) -> list[tuple[str, str]]:
    out = []
    for label, key, fmt in SUMMARY_ROWS:
        if key is None:
            v = {
                "Reference lines (empty after normalisation)": f"{s['ref_lines']} ({s['ref_lines_empty']})",
                "Test lines (empty after normalisation)": f"{s['test_lines']} ({s['test_lines_empty']})",
                "Missed lines (nothing / wrong only; 1-2 characters)": (f"{s['missed_lines']} ({s['missed_nothing']} / {s['missed_wrong_only']}; "
                             f"{s['missed_lines_short']})"),
                "Lines decided by a single hit, 4+ characters (reference / test)": f"{s['single_hit_ref_lines']} / {s['single_hit_test_lines']}",
                "Suspected hallucination lines (their characters)":
                    f"{s['hallucination_lines']} ({s['hallucination_chars']})",
                "Start ok / lost / wrong / misplaced / uncertain": " / ".join(
                    str(s[f"start_{v}"]) for v in ("ok", "lost", "wrong", "misplaced", "uncertain")),
                "End ok / lost / wrong / misplaced / uncertain": " / ".join(
                    str(s[f"end_{v}"]) for v in ("ok", "lost", "wrong", "misplaced", "uncertain")),
                "Ambiguity: lines whose missed / start / end / timing / hallucination value differs":
                    " / ".join(str(s[f"ambiguous_{k}"])
                                    for k in ("missed", "start", "end", "timing", "hallucination")),
            }[label]
        elif fmt is None:
            v = _pct(s[key])
        else:
            v = _f(s[key], fmt)
        out.append((label, v))
    return out


def markdown(title: str, files: list[tuple[str, str, str]], results: list[ClipResult], summary: dict,
             per_clip: list[tuple[str, dict]], problems: list[str], params: dict) -> str:
    L = [f"# {title}", "", f"subeval {__version__}; parameters: {json.dumps(params)}", ""]
    L += ["## Inputs", "", "| Clip | Reference (SHA-256) | Test (SHA-256) |", "|---|---|---|"]
    for clip, ref, test in files:
        L.append(f"| {clip} | {Path(ref).name} ({sha256(ref)[:16]}) | {Path(test).name} ({sha256(test)[:16]}) |")
    L += ["", "Reading problems: " + ("none" if not problems else ""), *[f"- {p}" for p in problems], ""]
    L += ["## Results, all clips together", "", "| Indicator | Value |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in summary_cells(summary)]
    L += ["", "## Results per clip", ""]
    L.append("Clips: " + "; ".join(f"C{k} = {clip}" for k, (clip, _) in enumerate(per_clip, 1)))
    L += ["", "| Indicator | " + " | ".join(f"C{k}" for k in range(1, len(per_clip) + 1)) + " |",
          "|---" * (len(per_clip) + 1) + "|"]
    cols = [summary_cells(s) for _, s in per_clip]
    for row, (label, _) in enumerate(summary_cells(summary)):
        L.append(f"| {label} | " + " | ".join(col[row][1] for col in cols) + " |")
    L += ["", DEFINITIONS.format(version=__version__, profile=params["profile"], window=params["window_s"],
                                  within=WITHIN)]
    return "\n".join(L)


def pairing_record(results: list[ClipResult]) -> str:
    L = []
    for c in results:
        L += [f"===== {c.clip}", ""]
        for r in c.ref:
            tl = ", ".join(f"T{c.test[k].number}" for k in r.test_lines) or "-"
            cls = ("empty after normalisation" if r.n == 0 else f"MISSED ({r.missed_kind})" if r.missed else
                   f"home T{c.test[r.home].number}; start {r.start_class}"
                   + (f" ({r.lead_lost} lost)" if r.lead_lost else "")
                   + f"; end {r.end_class}" + (f" ({r.tail_lost} lost)" if r.tail_lost else ""))
            timing = ("" if r.start_err is None else f"; start err {r.start_err:+.3f}") + \
                     ("" if r.end_err is None else f"; end err {r.end_err:+.3f}")
            L.append(f"R{r.number} [{_ts(r.start)} - {_ts(r.end)}] {r.text!r}")
            L.append(f"    chars {r.n}: hit {r.hits}, wrong {r.wrong}, missing {r.missing}; in {tl}; {cls}{timing}")
        L += ["", "-- test lines"]
        for t in c.test:
            flag = "  SUSPECTED HALLUCINATION" if t.hallucination else "  MISHEARD" if t.misheard else ""
            refs = ", ".join(f"R{c.ref[k].number}" for k in t.ref_lines) or "-"
            L.append(f"T{t.number} [{_ts(t.start)} - {_ts(t.end)}] {t.text!r}")
            L.append(f"    chars {t.n}: hit {t.hits}, wrong {t.wrong}, extra {t.extra}; holds {refs}{flag}")
        if c.ambiguous_ref_lines:
            L.append("    ambiguous reference lines (mirror alignment differs): "
                     + ", ".join(f"R{c.ref[k].number}" for k in c.ambiguous_ref_lines))
        L.append("")
    return "\n".join(L)


def to_json(summary: dict, results: list[ClipResult], extra: dict | None = None) -> str:
    data = {"subeval": __version__, "summary": summary,
            "clips": [dataclasses.asdict(c) for c in results], **(extra or {})}
    return json.dumps(data, ensure_ascii=False, indent=1)
