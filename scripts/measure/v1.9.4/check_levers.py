"""For every step-3 run: which settings differ from its baseline, (1) in the recorded segmenter and framer settings
(framing.jsonl, compared with the 7-clip baseline run) and (2) at the calls that use them (calls.jsonl, compared with
the one-clip trace of the baseline). The applied-shift fields are left out of (2) because they depend on the audio.
A correct run differs in the changed setting only (lead0 changes nothing here; its value is checked in run.log).
Usage: python check_levers.py <levers folder> <baseline folder> <trace folder>"""
import json
import re
import sys
from pathlib import Path

lev_root, base_root, trace_root = (Path(a) for a in sys.argv[1:4])
SKIP_ATTR = {"_model", "_model_path", "_wjdiag_wrapped", "_wj_record_wrapped", "_wj_record_last", "_session",
             "_wjdiag_last"}


def recorded(run):
    for line in (run / "framing.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        if d.get("kind") == "segmenter":
            a = {k: v for k, v in d["attrs"].items() if k not in SKIP_ATTR}
            a.update({f"framer.{k}": v for k, v in d["framer"].items() if k not in SKIP_ATTR})
            return a
    return {}


def traced(run):
    vals = {}
    for line in (run / "calls.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        call = d.pop("call")
        if call == "tracer_added":
            continue
        d.pop("pid", None)
        for k, v in (d.pop("values", None) or {}).items():
            d[k] = v
        for k, v in d.items():
            if not k.startswith("largest_"):
                vals.setdefault(f"{call.split('.')[-1]}.{k}", set()).add(json.dumps(v))
    return vals


def diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


for run in sorted(p.parent for p in lev_root.glob("*/run1/DONE")):
    name = run.parent.name
    base = name.split("__")[0]
    d1 = diff(recorded(base_root / base / "run1"), recorded(run))
    tb = trace_root / base / "run1"
    d2 = diff(traced(tb), traced(run)) if (tb / "calls.jsonl").exists() else "no baseline trace"
    if isinstance(d2, dict):  # keys the baseline trace never had (traced since): shown as "new", not a difference
        new = {k: sorted(v[1]) for k, v in d2.items() if v[0] is None}
        d2 = {k: (sorted(v[0] or []), sorted(v[1] or [])) for k, v in d2.items() if v[0] is not None}
        if new:
            d2["new"] = new
    log = (run / "run.log").read_text(encoding="utf-8", errors="replace")
    lead = sorted(set(re.findall(r"lead(?:ing)?[-_ ]?(?:in )?silence[^\n]{0,60}", log, re.I)))[:2]
    print(f"- {name}\n    recorded: {d1 or 'no difference'}\n    traced:   {d2 or 'no difference'}"
          + (f"\n    log: {lead}" if name.endswith("lead0") or name.startswith("anime") else ""))
