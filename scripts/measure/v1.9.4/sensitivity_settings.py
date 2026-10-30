"""The recorded segmenter and framer settings of today's three sensitivities, side by side, per model and segmenter
(framing.jsonl "segmenter" record of each baseline run; recorder-internal attributes left out). Lists only the
settings that differ between sensitivities.
Usage: python sensitivity_settings.py <baseline folder>"""
import json
import sys
from pathlib import Path

SKIP = {"_model", "_model_path", "_wjdiag_wrapped", "_wj_record_wrapped", "_wj_record_last", "_session",
        "_wjdiag_last"}
SENS = ("conservative", "balanced", "aggressive")


def recorded(run):
    for line in (run / "framing.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        if d.get("kind") == "segmenter":
            a = {k: v for k, v in d["attrs"].items() if k not in SKIP}
            a.update({f"framer.{k}": v for k, v in d["framer"].items() if k not in SKIP})
            return a
    return {}


base = Path(sys.argv[1])
for gen in ("qwen3", "anime"):
    for seg in ("ten", "firered-vad", "whisperseg", "silero"):
        vals = [recorded(base / f"{gen}_{seg}_{s}" / "run1") for s in SENS]
        keys = sorted(set().union(*vals))
        diff = [k for k in keys if len({json.dumps(v.get(k), sort_keys=True, default=str) for v in vals}) > 1]
        print(f"\n{gen} {seg}: {len(diff)} settings differ")
        for k in diff:
            print(f"  {k}: " + " / ".join(str(v.get(k)) for v in vals))
