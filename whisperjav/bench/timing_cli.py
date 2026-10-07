"""
CLI for subtitle timing against reference (ground-truth) subtitles.

Pairs every reference SRT in --ref-dir with the one WhisperJAV SRT in each run
folder whose name starts with the clip id. The clip id is the reference file
name without ".ja.srt" / ".srt".

  # One run on its own (every matched line):
  python -m whisperjav.bench.timing_cli --ref-dir test_media/Ground_Truths/Netflix --base out/run1

  # Two runs compared on the reference lines both matched
  # (run 1 vs run 2 of the same version = run-to-run variation):
  python -m whisperjav.bench.timing_cli --ref-dir test_media/Ground_Truths/Netflix \
      --base out/run1 --cand out/run2 -o timing.json

Errors are in seconds: start/end error = WhisperJAV time - reference time
(positive = late). "abs" values are absolute errors; "signed mean" shows a
consistent lean (e.g. lines that end late).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from whisperjav.bench.timing import compare_clip, load_srt, match_errors, summarize


def clip_id_of(ref_path: Path) -> str:
    name = ref_path.name
    for suffix in (".ja.srt", ".srt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return ref_path.stem


def find_hyp(run_dir: Path, clip_id: str) -> Path:
    """The single SRT in run_dir whose name starts with '<clip_id>.'"""
    hits = sorted(p for p in run_dir.glob("*.srt") if p.name.startswith(clip_id + "."))
    if len(hits) != 1:
        raise SystemExit(
            f"{run_dir}: expected exactly one SRT for clip '{clip_id}', found {len(hits)}: "
            + ", ".join(p.name for p in hits)
        )
    return hits[0]


def _load_hyp(run_dir: Path, clip_id: str) -> list[dict]:
    path = find_hyp(run_dir, clip_id)
    subs = load_srt(path)
    if not subs and path.stat().st_size > 0:
        raise SystemExit(f"{path}: file is not empty but no subtitle lines could be read")
    return subs


def _fmt(m: dict) -> str:
    return (f"n={m['n_lines']:>4}  start |med| {m['start_abs_median']:6.3f} p90 {m['start_abs_p90']:6.3f} "
            f"mean {m['start_signed_mean']:+6.3f}   end |med| {m['end_abs_median']:6.3f} "
            f"p90 {m['end_abs_p90']:6.3f} mean {m['end_signed_mean']:+6.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-dir", type=Path, required=True, help="Folder of reference SRTs")
    parser.add_argument("--base", type=Path, required=True, help="Run folder (base)")
    parser.add_argument("--cand", type=Path, help="Second run folder, compared with --base on common lines")
    parser.add_argument("-o", "--output", type=Path, help="Write the full result as JSON")
    args = parser.parse_args(argv)

    refs = sorted(args.ref_dir.glob("*.srt"))
    if not refs:
        raise SystemExit(f"No reference SRTs in {args.ref_dir}")

    result: dict = {"ref_dir": str(args.ref_dir), "base": str(args.base),
                    "cand": str(args.cand) if args.cand else None, "clips": {}}
    pooled_base: list[tuple[float, float]] = []
    pooled_cand: list[tuple[float, float]] = []
    totals = {"n_ref": 0, "n_matched_base": 0, "n_matched_cand": 0, "n_common": 0, "n_lost": 0}

    for ref_path in refs:
        clip_id = clip_id_of(ref_path)
        ref = load_srt(ref_path, reference=True)
        base = _load_hyp(args.base, clip_id)
        if args.cand:
            c = compare_clip(ref, base, _load_hyp(args.cand, clip_id))
            pooled_base += c.base_errors
            pooled_cand += c.cand_errors
            for key in totals:
                totals[key] += getattr(c, key)
            result["clips"][clip_id] = {
                "n_ref": c.n_ref, "n_matched_base": c.n_matched_base, "n_matched_cand": c.n_matched_cand,
                "n_common": c.n_common, "n_lost": c.n_lost,
                "base": c.base.as_dict(), "cand": c.cand.as_dict(),
                "base_errors": c.base_errors, "cand_errors": c.cand_errors,
            }
            print(f"{clip_id}  (ref {c.n_ref}, common {c.n_common}, lost {c.n_lost})")
            print(f"  base  {_fmt(c.base.as_dict())}")
            print(f"  cand  {_fmt(c.cand.as_dict())}")
        else:
            errors = list(match_errors(ref, base).values())
            pooled_base += errors
            totals["n_ref"] += len(ref)
            totals["n_matched_base"] += len(errors)
            m = summarize(errors)
            result["clips"][clip_id] = {"n_ref": len(ref), "base": m.as_dict(), "base_errors": errors}
            print(f"{clip_id}  (ref {len(ref)}, matched {len(errors)})")
            print(f"  base  {_fmt(m.as_dict())}")

    result["totals"] = totals
    result["pooled_base"] = summarize(pooled_base).as_dict()
    print("\nALL CLIPS POOLED")
    print(f"  base  {_fmt(result['pooled_base'])}")
    if args.cand:
        result["pooled_cand"] = summarize(pooled_cand).as_dict()
        print(f"  cand  {_fmt(result['pooled_cand'])}")
        print(f"  lines lost (matched by base, not by cand): {totals['n_lost']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nWritten: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
