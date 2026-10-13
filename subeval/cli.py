"""Command line.

  python -m subeval score REF.srt TEST.srt [--out DIR]
  python -m subeval score --ref-dir DIR --test-dir DIR [--out DIR]
  python -m subeval compare --ref-dir DIR --base-dir DIR --cand-dir DIR [--out DIR]

Folder mode pairs each reference file with its test file by clip id: the reference file name without ".ja.srt" or
".srt"; the test folder must hold exactly one .srt whose name starts with "<clip id>.". Anything else is an error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .compare import TOLERANCE, compare
from .evaluate import evaluate_clip, summarize
from .report import markdown, pairing_record, to_json
from .normalize import unbalanced_brackets
from .srt import read_srt


def clip_id(ref: Path) -> str:
    for suffix in (".ja.srt", ".srt"):
        if ref.name.endswith(suffix):
            return ref.name[: -len(suffix)]
    return ref.stem


def pair_folder(ref_dir, test_dir) -> list[tuple[str, Path, Path]]:
    refs = sorted(Path(ref_dir).glob("*.srt"))
    if not refs:
        raise SystemExit(f"no .srt files in {ref_dir}")
    out = []
    for ref in refs:
        cid = clip_id(ref)
        hits = sorted(p for p in Path(test_dir).glob("*.srt") if p.name.startswith(cid + "."))
        if len(hits) != 1:
            raise SystemExit(f"{test_dir}: expected exactly one .srt for clip {cid!r}, found {[p.name for p in hits]}")
        out.append((cid, ref, hits[0]))
    return out


def score_files(files, window=1.0, profile="ja-v1"):
    """files: [(clip id, reference path, test path)] -> (clip results, reading problems)."""
    results, problems = [], []
    for cid, ref, test in files:
        r, t = read_srt(ref), read_srt(test)
        problems += [f"{ref}: {p}" for p in r.problems] + [f"{test}: {p}" for p in t.problems]
        problems += [f"{ref}: block {c.number}: unbalanced bracket, the note may be damaged and is compared as "
                     f"speech: {c.text!r}" for c in r.cues if unbalanced_brackets(c.text)]
        results.append(evaluate_clip(r.cues, t.cues, cid, window, profile))
    return results, problems


def _write(out, title, files, results, problems, params, extra=None):
    summary = summarize(results)
    per_clip = [(c.clip, summarize([c])) for c in results]
    md = markdown(title, [(c, str(r), str(t)) for c, r, t in files], results, summary, per_clip, problems, params)
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(md, encoding="utf-8")
        (out / "result.json").write_text(to_json(summary, results, extra), encoding="utf-8")
        (out / "pairs.txt").write_text(pairing_record(results), encoding="utf-8")
    return md, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="subeval", description="Evaluate test subtitles against reference subtitles "
                                 "(text, missed lines, suspected hallucination lines, line openings/endings, "
                                 "timing). Definitions: subeval/METHOD.md and every report.")
    ap.add_argument("--version", action="version", version=f"subeval {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--window", type=float, default=1.0,
                        help="largest time gap (s) between two lines whose characters may be paired (default 1.0)")
    common.add_argument("--profile", default="ja-v1", help="text normalisation profile (default ja-v1)")
    common.add_argument("--out", help="folder for report.md, result.json and pairs.txt; without it the report "
                                      "is printed")
    s = sub.add_parser("score", parents=[common], help="one test run against the reference")
    s.add_argument("ref", nargs="?", help="reference .srt (file mode)")
    s.add_argument("test", nargs="?", help="test .srt (file mode)")
    s.add_argument("--ref-dir")
    s.add_argument("--test-dir")
    c = sub.add_parser("compare", parents=[common], help="two test runs against the same reference, clip by clip")
    c.add_argument("--ref-dir", required=True)
    c.add_argument("--base-dir", required=True)
    c.add_argument("--cand-dir", required=True)
    c.add_argument("--tolerance", type=float, default=TOLERANCE,
                   help=f"timing difference (s) below which a clip counts as the same (default {TOLERANCE})")
    a = ap.parse_args(argv)
    params = {"window_s": a.window, "profile": a.profile}
    if a.cmd == "score":
        if a.ref and a.test:
            files = [(clip_id(Path(a.ref)), Path(a.ref), Path(a.test))]
        elif a.ref_dir and a.test_dir:
            files = pair_folder(a.ref_dir, a.test_dir)
        else:
            ap.error("score needs REF TEST, or --ref-dir and --test-dir")
        results, problems = score_files(files, a.window, a.profile)
        md, _ = _write(a.out, "subeval report", files, results, problems, params)
        if not a.out:
            print(md)
        return 0
    fb, fc = pair_folder(a.ref_dir, a.base_dir), pair_folder(a.ref_dir, a.cand_dir)
    rb, pb = score_files(fb, a.window, a.profile)
    rc, pc = score_files(fc, a.window, a.profile)
    cmp = compare(rb, rc, a.tolerance)
    text = json.dumps(cmp, indent=1)
    if a.out:
        _write(Path(a.out) / "base", "subeval report: base", fb, rb, pb, params)
        _write(Path(a.out) / "cand", "subeval report: candidate", fc, rc, pc, params)
        (Path(a.out) / "compare.json").write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
