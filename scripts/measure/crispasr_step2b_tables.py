"""CrispASR step 2b: the report's tables, from what the batch wrote (CrispASR master plan 4.5).

Reads, in <batch>:
    summary.json       subeval scores per folder, from summarize_f0.py --batch <batch> --pipelines <folders>
    batch.json         per run: exit status, seconds, GPU figures (crispasr_step2b_batch.py)
    <config>/<film>.run.json   per run: slices, scenes, the program run, subtitles, notes (crispasr_step2b.py)
and, with --context <batch> (an earlier batch scored by summarize_f0.py on the same films), its pipelines' rows for
comparison. Writes <batch>/tables.md: text first, timing second, with the full line and character breakdown.

Usage (from the repository root):
    python scripts/measure/crispasr_step2b_tables.py --batch <batch> [--context <batch> --context-pipelines a b]
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "measure"))
import crispasr_step2b as step2b  # noqa: E402

FILMS = ("The.Naked.Director.S01E04.Scene.4", "The.Naked.Director.S02E02.Scene.5.Bubble.More")
SHORT = {FILMS[0]: "S01E04 Sc.4", FILMS[1]: "S02E02 Sc.5"}


def pct(v):
    return "-" if v is None else f"{100 * v:.1f}%"


def sec(v):
    return "-" if v is None else f"{v:.3f}"


def num(v):
    return "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]


def text_row(label, s):
    return [label, pct(s.get("cer")), pct(s.get("cer_plain")), pct(s.get("wrong_rate")), pct(s.get("missing_rate")),
            pct(s.get("extra_rate")), f"{num(s.get('missed_lines'))} ({num(s.get('missed_nothing'))}/"
            f"{num(s.get('missed_wrong_only'))})", num(s.get("hallucination_lines")), num(s.get("misheard_test_lines")),
            f"{num(s.get('test_lines'))}/{num(s.get('ref_lines'))}", num(s.get("ambiguous_ref_lines"))]


TEXT_HEAD = ["", "CER", "CER (plain)", "wrong", "missing", "extra", "missed lines (nothing/wrong only)",
             "susp. halluc. lines", "misheard test lines", "test/ref lines", "ambiguous ref lines"]


def timing_row(label, s):
    def classes(edge):
        return "/".join(num(s.get(f"{edge}_{v}")) for v in ("ok", "lost", "wrong", "misplaced", "uncertain"))
    return [label, sec(s.get("start_abs_median")), sec(s.get("end_abs_median")), pct(s.get("start_within_share")),
            pct(s.get("end_within_share")), sec(s.get("start_signed_mean")), sec(s.get("end_signed_mean")),
            f"{num(s.get('start_measured'))}/{num(s.get('end_measured'))}", classes("start"), classes("end")]


TIMING_HEAD = ["", "start median abs err s", "end median abs err s", "starts within 0.5 s", "ends within 0.5 s",
               "start signed mean s", "end signed mean s", "measured starts/ends",
               "start ok/lost/wrong/misplaced/uncertain", "end ok/lost/wrong/misplaced/uncertain"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crispasr_step2b_tables.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, type=Path)
    ap.add_argument("--context", type=Path, help="an earlier summarize_f0.py batch on the same films")
    ap.add_argument("--context-pipelines", nargs="*", default=["chronos_qwen3", "fidelity", "balanced"])
    a = ap.parse_args(argv)
    batch = a.batch
    scores = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
    runs = {(r["configuration"], r["film"]): r
            for r in json.loads((batch / "batch.json").read_text(encoding="utf-8"))["runs"]}
    sp = scores["pipelines"]
    configs = [(p, m, s, step2b.config_label(m, s)) for p, m, s in step2b.STEP_2B_CONFIGURATIONS]

    out = [f"# CrispASR step 2b tables — {batch}", "",
           f"subeval {scores.get('subeval')}, window {scores.get('window')} s; rates are shares of the reference "
           f"characters of both films ({FILMS[0]}, {FILMS[1]}).", ""]

    # 1. The program runs
    rows = []
    for _, _, _, label in configs:
        for film in FILMS:
            rj = batch / label / f"{film}.run.json"
            r = json.loads(rj.read_text(encoding="utf-8")) if rj.is_file() else {}
            pr, sl = r.get("program_run") or {}, r.get("slices") or {}
            g = ((runs.get((label, film + ".mkv")) or {}).get("gpu") or {})
            gp = g.get("program") or {}
            rows.append([label, SHORT[film], num(r.get("audio_seconds")), num(r.get("scenes")),
                         f"{num(sl.get('count'))} / {num(sl.get('seconds'))}", num(pr.get("seconds")),
                         num(pr.get("segments")), num(pr.get("warnings")),
                         f"{len(r.get('notes') or [])} / {len(r.get('degradations') or [])}",
                         f"{num(gp.get('before_mib'))} → {num(gp.get('peak_mib'))}",
                         num(gp.get("program_mib")), num(gp.get("utilisation_median_pct")),
                         (r.get("program") or {}).get("build") or "-",
                         "ok" if not r.get("error") else f"FAILED: {r['error'].get('message')}"])
    out += ["## 1. The 12 program runs", ""]
    out += table(["configuration", "film", "audio s", "scenes", "slices / s", "program s", "segments", "warnings",
                  "notes / degradations", "GPU MiB before → peak", "program MiB", "GPU use median %", "build",
                  "result"], rows)

    # 2. Text and 3. timing, the pipeline's own SRT, both films, with context rows
    own = [(label, sp.get(label, {}).get("all", {})) for _, _, _, label in configs]
    context = []
    if a.context:
        cs = json.loads((a.context / "summary.json").read_text(encoding="utf-8"))["pipelines"]
        context = [(f"context: {p} ({a.context.name})", cs[p]["all"]) for p in a.context_pipelines if p in cs]
    out += ["", "## 2. Text, both films: each configuration's own SRT", ""]
    out += table(TEXT_HEAD, [text_row(l, s) for l, s in own + context])
    out += ["", "## 3. Timing, both films: each configuration's own SRT", ""]
    out += table(TIMING_HEAD, [timing_row(l, s) for l, s in own + context])

    # 4. Per film, own SRT
    out += ["", "## 4. Per film: each configuration's own SRT", ""]
    rows = []
    for _, _, _, label in configs:
        for film in FILMS:
            s = sp.get(label, {}).get("films", {}).get(film, {})
            rows.append([label, SHORT[film], pct(s.get("cer")), pct(s.get("wrong_rate")), pct(s.get("missing_rate")),
                         pct(s.get("extra_rate")), num(s.get("missed_lines")), num(s.get("hallucination_lines")),
                         f"{num(s.get('test_lines'))}/{num(s.get('ref_lines'))}", sec(s.get("start_abs_median")),
                         sec(s.get("end_abs_median"))])
    for name, _ in context:
        pass
    out += table(["configuration", "film", "CER", "wrong", "missing", "extra", "missed lines", "susp. halluc. lines",
                  "test/ref lines", "start median abs err s", "end median abs err s"], rows)

    # 5. Line building and post-processing variants, both films
    out += ["", "## 5. Line building and post-processing: every variant, both films", "",
            "Each variant is built from the same program run as its configuration's own SRT. \"own\" marks the "
            "variant with the pipeline's default settings.", ""]
    rows = []
    for p, _, _, label in configs:
        rj = batch / label / f"{FILMS[0]}.run.json"
        own_variant = (json.loads(rj.read_text(encoding="utf-8")).get("pipeline_srt") or {}).get("settings_variant") \
            if rj.is_file() else None
        for name, _, _, _ in step2b.variants_for(p):
            s = sp.get(f"{label}/{name}", {}).get("all", {})
            rows.append([label, name + (" (own)" if name == own_variant else ""), pct(s.get("cer")),
                         pct(s.get("wrong_rate")), pct(s.get("missing_rate")), pct(s.get("extra_rate")),
                         num(s.get("missed_lines")), num(s.get("hallucination_lines")),
                         f"{num(s.get('test_lines'))}/{num(s.get('ref_lines'))}", sec(s.get("start_abs_median")),
                         sec(s.get("end_abs_median")), pct(s.get("start_within_share")), pct(s.get("end_within_share"))])
    out += table(["configuration", "variant", "CER", "wrong", "missing", "extra", "missed lines",
                  "susp. halluc. lines", "test/ref lines", "start median abs err s", "end median abs err s",
                  "starts within 0.5 s", "ends within 0.5 s"], rows)

    # 6. Line shape: how long the lines are, against the reference
    out += ["", "## 6. Line shape, both films: characters and seconds per line", "",
            "Characters are those of the subtitle text without spaces and line breaks; seconds are end - start.", ""]
    rows = [shape_row("reference (Netflix)", [REPO / "test_media" / "Ground_Truths" / "Netflix" / n
                                              for n in ("The.Naked.Director.S01E04.Scene.4.ja.srt",
                                                        "The.Naked.Director.S02E02.Scene.5.Bubble.More.srt")])]
    for p, _, _, label in configs:
        rows.append(shape_row(f"{label} (own)", [batch / label / f"{f}.ja.whisperjav.srt" for f in FILMS]))
        if "segments" in step2b.LINE_MODES[p]:
            v = step2b.variant_name("segments", "text+overlap")
            rows.append(shape_row(f"{label} / {v}", [batch / label / v / f"{f}.{v}.srt" for f in FILMS]))
    if a.context:
        for p in a.context_pipelines:
            srts = sorted((a.context / p).glob("*.srt"))
            if srts:
                rows.append(shape_row(f"context: {p} ({a.context.name})", srts))
    out += table(["SRT", "lines", "mean characters", "lines over 30 characters", "mean s", "median s",
                  "lines over 7 s", "longest s"], rows)

    (batch / "tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"written {batch / 'tables.md'}")
    return 0


def shape_row(label, paths):
    import srt
    import statistics
    lines = []
    for path in paths:
        lines += [(len("".join(s.content.split())), (s.end - s.start).total_seconds())
                  for s in srt.parse(Path(path).read_text(encoding="utf-8", errors="replace"))]
    if not lines:
        return [label, "0", "-", "-", "-", "-", "-", "-"]
    chars, secs = [c for c, _ in lines], [d for _, d in lines]
    return [label, str(len(lines)), f"{statistics.mean(chars):.1f}", str(sum(c > 30 for c in chars)),
            f"{statistics.mean(secs):.2f}", f"{statistics.median(secs):.2f}", str(sum(d > 7 for d in secs)),
            f"{max(secs):.2f}"]


if __name__ == "__main__":
    sys.exit(main())
