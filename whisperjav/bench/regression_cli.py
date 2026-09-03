"""
CLI for the accuracy regression gate.

Three subcommands:

  score     Score pipeline SRT outputs against ground truth, write metrics JSON.
  baseline  Same as score, but write in baseline format (checked into bench/baselines/).
  gate      Score outputs, compare to a baseline, exit 1 on any violation.

The corpus manifest (YAML) maps clip ids to ground-truth SRTs; hypothesis SRTs
are found in --hyp-dir as <clip_id>.srt (whisperjav's default output naming can
be symlinked or copied into that shape by the runner script).

Examples:

  # After running the corpus through a pipeline on a GPU machine:
  python -m whisperjav.bench.regression_cli baseline \
      --manifest bench/corpus.yaml --hyp-dir out/v1.8.14_balanced \
      --label "v1.8.14 balanced/balanced" -o bench/baselines/balanced.json

  # Gate a candidate build against that baseline:
  python -m whisperjav.bench.regression_cli gate \
      --manifest bench/corpus.yaml --hyp-dir out/candidate_balanced \
      --baseline bench/baselines/balanced.json

Scoring and gating are CPU-only and dependency-light; they run in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from whisperjav.bench.regression import (
    ClipMetrics,
    build_baseline,
    gate,
    load_baseline,
    score_srt_files,
)


def _load_manifest(path: Path) -> dict:
    import yaml  # lazy: PyYAML is already a whisperjav dependency

    with open(path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    if not manifest or "clips" not in manifest:
        raise SystemExit(f"Manifest {path} missing 'clips' list")
    return manifest


def _score_corpus(manifest_path: Path, hyp_dir: Path) -> dict[str, ClipMetrics]:
    manifest = _load_manifest(manifest_path)
    root = manifest_path.parent
    results: dict[str, ClipMetrics] = {}
    missing = []
    for clip in manifest["clips"]:
        clip_id = clip["id"]
        gt_path = root / clip["ground_truth"]
        hyp_path = hyp_dir / f"{clip_id}.srt"
        if not hyp_path.exists():
            missing.append(clip_id)
            continue
        results[clip_id] = score_srt_files(gt_path, hyp_path)
    if missing:
        print(f"WARNING: no hypothesis SRT for clips: {', '.join(missing)}",
              file=sys.stderr)
    if not results:
        raise SystemExit("No clips scored - is --hyp-dir correct?")
    return results


def _print_table(results: dict[str, ClipMetrics]) -> None:
    header = f"{'clip':<28} {'CER':>6} {'recall':>7} {'F1':>6} {'IoU':>6} {'n_hyp/n_gt':>11} {'rep':>5}"
    print(header)
    print("-" * len(header))
    for clip_id, m in sorted(results.items()):
        print(f"{clip_id:<28} {m.cer:>6.3f} {m.recall:>7.3f} {m.f1:>6.3f} "
              f"{m.mean_iou:>6.3f} {m.n_hyp:>5}/{m.n_gt:<5} {m.repetition_ratio:>5.2f}")


def _get_version() -> str:
    try:
        from whisperjav.__version__ import __version__
        return __version__
    except Exception:
        return "unknown"


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whisperjav-accuracy-gate")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, required=True,
                        help="Corpus manifest YAML (see bench/corpus.example.yaml)")
    common.add_argument("--hyp-dir", type=Path, required=True,
                        help="Directory with <clip_id>.srt pipeline outputs")

    p_score = sub.add_parser("score", parents=[common],
                             help="Score outputs, print table, optionally write JSON")
    p_score.add_argument("-o", "--output", type=Path, default=None)

    p_base = sub.add_parser("baseline", parents=[common],
                            help="Score outputs and write a baseline JSON")
    p_base.add_argument("-o", "--output", type=Path, required=True)
    p_base.add_argument("--label", default="")

    p_gate = sub.add_parser("gate", parents=[common],
                            help="Score outputs and compare against a baseline")
    p_gate.add_argument("--baseline", type=Path, required=True)
    p_gate.add_argument("--thresholds", type=Path, default=None,
                        help="Optional YAML overriding default thresholds/hard bounds")

    args = parser.parse_args(argv)
    results = _score_corpus(args.manifest, args.hyp_dir)
    _print_table(results)

    if args.command == "score":
        if args.output:
            payload = {cid: m.as_dict() for cid, m in results.items()}
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nMetrics written to {args.output}")
        return 0

    if args.command == "baseline":
        baseline = build_baseline(results, label=args.label,
                                  whisperjav_version=_get_version())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print(f"\nBaseline written to {args.output}")
        return 0

    # gate
    baseline = load_baseline(args.baseline)
    thresholds = hard_bounds = None
    if args.thresholds:
        import yaml

        with open(args.thresholds, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        thresholds = cfg.get("thresholds")
        raw_bounds = cfg.get("hard_bounds")
        if raw_bounds:
            hard_bounds = {k: (v.get("min"), v.get("max"))
                           for k, v in raw_bounds.items()}

    violations = gate(results, baseline, thresholds, hard_bounds)
    if violations:
        print(f"\nACCURACY GATE FAILED - {len(violations)} violation(s):",
              file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    print(f"\nAccuracy gate passed ({len(results)} clips vs "
          f"baseline '{baseline.get('label', '')}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
