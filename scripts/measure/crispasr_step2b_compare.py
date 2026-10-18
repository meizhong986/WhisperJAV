"""CrispASR step 2b: does each scored run repeat itself? A rerun batch against the first batch, file by file (the
step 2 report's item 10 (b)).

For each configuration folder and film found in both batches, byte for byte:
    the pipeline's own SRT          <config>/<film>.ja.whisperjav.srt
    every variant's SRT             <config>/lines-*/<film>.lines-*.srt
    the cut points                  <config>/<film>.cuts.json
    the program's JSON output       <config>/temp/crispasr_out/<film's run folder>/out.json; when the bytes differ,
                                    also with the field that names the model file left out
and, from both batch.json files, each run's exit status, seconds, and the program's GPU memory (peak - before).

Usage (from the repository root, with WJ110's python):
    python scripts/measure/crispasr_step2b_compare.py --first <batch> --rerun <batch>

Writes <rerun>/compare.json and <rerun>/compare.md, and prints the table. Exit status 0 when every compared file is
the same, 1 otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def same_bytes(a: Path, b: Path):
    if not (a.is_file() and b.is_file()):
        return None
    return a.read_bytes() == b.read_bytes()


def program_output(config_folder: Path, film: str):
    from whisperjav.crispasr.runner import film_run_dir
    return film_run_dir(config_folder / "temp", film) / "out.json"


def without_model_path(path: Path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc.get("crispasr", {}).pop("model", None)
    return doc


def batch_runs(batch: Path) -> dict:
    try:
        runs = json.loads((batch / "batch.json").read_text(encoding="utf-8"))["runs"]
    except (OSError, ValueError, KeyError):
        return {}
    return {(r["configuration"], Path(r["film"]).stem): r for r in runs}


def compare(first: Path, rerun: Path) -> list:
    rows = []
    first_runs, rerun_runs = batch_runs(first), batch_runs(rerun)
    for config in sorted(p for p in rerun.iterdir() if p.is_dir() and (first / p.name).is_dir()):
        for run_json in sorted(config.glob("*.run.json")):
            film = run_json.name[:-len(".run.json")]
            a, b = first / config.name, config
            variants = sorted(p.relative_to(b) for p in b.glob(f"lines-*/{film}.lines-*.srt"))
            variant_same = {str(v): same_bytes(a / v, b / v) for v in variants}
            out_a, out_b = program_output(a, film), program_output(b, film)
            out_same = same_bytes(out_a, out_b)
            row = {"configuration": config.name, "film": film,
                   "own_srt_same": same_bytes(a / f"{film}.ja.whisperjav.srt", b / f"{film}.ja.whisperjav.srt"),
                   "variants": len(variants), "variants_same": sum(1 for v in variant_same.values() if v),
                   "variant_files": variant_same,
                   "cuts_same": same_bytes(a / f"{film}.cuts.json", b / f"{film}.cuts.json"),
                   "program_output_same": out_same,
                   "program_output_same_without_model_path": (without_model_path(out_a) == without_model_path(out_b))
                   if out_same is False else out_same}
            for key, runs in (("first", first_runs), ("rerun", rerun_runs)):
                r = runs.get((config.name, film), {})
                program = (r.get("gpu") or {}).get("program") or {}
                row[key] = {"exit_status": r.get("exit_status"), "seconds": r.get("seconds"),
                            "program_seconds": program.get("seconds"), "program_mib": program.get("program_mib")}
            rows.append(row)
    return rows


def all_same(row: dict) -> bool:
    return bool(row["own_srt_same"] and row["cuts_same"] and row["variants_same"] == row["variants"]
                and row["program_output_same"])


def table(rows: list) -> str:
    def yn(v):
        return {True: "same", False: "DIFFERENT", None: "missing"}[v]

    head = ("| configuration | film | own SRT | variants | cut points | program output | exit (first, rerun) "
            "| seconds (first, rerun) | program GPU MiB (first, rerun) |\n|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in rows:
        out = yn(r["program_output_same"])
        if r["program_output_same"] is False and r["program_output_same_without_model_path"]:
            out = "same but the model path"
        lines.append(f"| {r['configuration']} | {r['film']} | {yn(r['own_srt_same'])} | "
                     f"{r['variants_same']} of {r['variants']} same | {yn(r['cuts_same'])} | {out} | "
                     f"{r['first']['exit_status']}, {r['rerun']['exit_status']} | "
                     f"{r['first']['seconds']}, {r['rerun']['seconds']} | "
                     f"{r['first']['program_mib']}, {r['rerun']['program_mib']} |")
    same = sum(all_same(r) for r in rows)
    lines.append(f"\n{same} of {len(rows)} runs repeated every compared file byte for byte.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crispasr_step2b_compare.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", required=True, type=Path, help="the first batch")
    ap.add_argument("--rerun", required=True, type=Path, help="the rerun batch")
    a = ap.parse_args(argv)
    rows = compare(a.first, a.rerun)
    text = table(rows)
    (a.rerun / "compare.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (a.rerun / "compare.md").write_text(f"# Rerun against the first batch\n\nfirst: {a.first}\nrerun: {a.rerun}\n\n"
                                        f"{text}\n", encoding="utf-8")
    print(text)
    return 0 if rows and all(all_same(r) for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
