"""Preset audit (owner, 2026-10-09): ChronosJAV (Qwen3-ASR, anime-whisper) x WhisperSeg / TEN / FireRedVAD x
conservative / balanced / aggressive, through BOTH routes a user takes:
  ensemble   the command the GUI Ensemble tab runs for an uncustomized pass (no settings passed; the
             same command as run_seg_levers.py's base)
  modeqwen   python -m whisperjav.main --mode qwen --qwen-generator G --qwen-segmenter S --qwen-sensitivity X
             (--mode qwen reads --qwen-sensitivity; a plain --sensitivity is accepted and ignored there)
Each run is the real product on one 40 s reference clip, with the segmenter recorder (every scalar of the live
segmenter object) and the call tracer (threshold and pads at the calls that use them; FireRedVAD's upstream
postprocessor settings; anime-whisper's lead-in silence). Nothing is passed but the route's own options.

The EXPECTED values below are typed by hand from the sources, not computed by the product's own functions:
  - the YAML presets (config/v4/ecosystems/tools/{whisperseg,ten,firered-vad}-speech-segmentation.yaml);
  - Qwen3-ASR with WhisperSeg: threshold 0.25 and longest segment 4 s at every sensitivity (owner 2026-10-05,
    config/qwen3_whisperseg_vad.py), end pad 0 (owner 2026-10-09);
  - anime-whisper with WhisperSeg: its table (config/anime_whisper_vad.py), owner 2026-10-05/06;
  - TEN and FireRedVAD in ChronosJAV: longest segment 4 s (owner 2026-10-08), the rest of their presets;
  - the pipeline values (pads, frame gap, group cap): Qwen3-ASR 100/100 ms (end 0 with WhisperSeg), 0.3 s, 3.0 s
    (QwenPipeline defaults); anime-whisper from its table at every segmenter;
  - anime-whisper lead-in 200 ms (owner 2026-10-06).
Also: the Customize window's defaults (webview_gui/api.py get_qwen_schema, what Apply would send) against the run.

Usage: python audit_presets.py run <out folder> [name ...]     (resumable; DONE marker)
       python audit_presets.py table <out folder>              (markdown on stdout)"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable
CLIP = REPO / "test_media" / "Ground_Truths" / "Netflix" / "The.Naked.Director.S01E04.Scene.1.mkv"
INSTRUMENTS = REPO / "scripts" / "measure" / "record_and_trace"
SENS = ("conservative", "balanced", "aggressive")
GENS = ("qwen3", "anime")
SEGS = ("whisperseg", "ten", "firered-vad")

# ---- expected, typed from the sources ------------------------------------------------------------
YAML = {  # segmenter -> sensitivity -> values from the YAML (spec < preset) that reach the backend
    "whisperseg": {"conservative": dict(threshold=0.45, min_speech_duration_ms=150, min_silence_duration_ms=200),
                   "balanced": dict(threshold=0.35, min_speech_duration_ms=100, min_silence_duration_ms=100),
                   "aggressive": dict(threshold=0.25, min_speech_duration_ms=80, min_silence_duration_ms=80)},
    "ten": {"conservative": dict(threshold=0.42, min_speech_duration_ms=200, min_silence_duration_ms=200, hop_size=256),
            "balanced": dict(threshold=0.32, min_speech_duration_ms=150, min_silence_duration_ms=150, hop_size=256),
            "aggressive": dict(threshold=0.22, min_speech_duration_ms=100, min_silence_duration_ms=100, hop_size=256)},
    "firered-vad": {
        "conservative": dict(threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=250, smooth_window_size=5),
        "balanced": dict(threshold=0.4, min_speech_duration_ms=200, min_silence_duration_ms=200, smooth_window_size=5),
        "aggressive": dict(threshold=0.3, min_speech_duration_ms=150, min_silence_duration_ms=150, smooth_window_size=5)},
}
ANIME_TABLE = {  # config/anime_whisper_vad.py
    "conservative": dict(chunk_threshold_s=0.3, max_group_duration_s=3.0, start_pad_ms=100, end_pad_ms=100,
                         ws=dict(threshold=0.35, dip_search_from=0.3, dip_accept_any=True)),
    "balanced": dict(chunk_threshold_s=0.25, max_group_duration_s=2.5, start_pad_ms=50, end_pad_ms=50,
                     ws=dict(threshold=0.30, dip_search_from=0.3, dip_accept_any=True)),
    "aggressive": dict(chunk_threshold_s=0.2, max_group_duration_s=2.0, start_pad_ms=0, end_pad_ms=30,
                       ws=dict(threshold=0.15, segmentation_decoder="offline", grow_floor=0.15, gap_merge_ms=350,
                               split_smooth_ms=120, speech_start_threshold=0.30, neg_threshold=0.01,
                               force_split_mode="chop", min_silence_duration_ms=80)),
}


def expected(gen, seg, sens):
    e = dict(YAML[seg][sens])
    e["max_speech_duration_s"] = 4.0
    if gen == "qwen3":
        e.update(chunk_threshold_s=0.3, max_group_duration_s=3.0, start_pad_ms=100, end_pad_ms=100)
        if seg == "whisperseg":
            e.update(threshold=0.25, end_pad_ms=0)
    else:
        t = ANIME_TABLE[sens]
        e.update({k: t[k] for k in ("chunk_threshold_s", "max_group_duration_s", "start_pad_ms", "end_pad_ms")})
        if seg == "whisperseg":
            e.update(t["ws"])
        e["lead_in_ms"] = 200
    return e


# ---- runs ----------------------------------------------------------------------------------------
def command(route, gen, seg, sens, out):
    if route == "ensemble":
        model = {"qwen3": "Qwen/Qwen3-ASR-1.7B", "anime": "litagin/anime-whisper"}[gen]
        qp = {"framer": "vad-grouped"} if gen == "qwen3" else {
            "framer": "vad-grouped", "generator_backend": "anime-whisper", "timestamp_mode": "vad_only",
            "assembly_cleaner": "passthrough", "stepdown": False}
        return [PY, "-m", "whisperjav.main", str(CLIP), "--ensemble", "--pass1-pipeline", "qwen",
                "--pass1-sensitivity", sens, "--pass1-model", model, "--pass1-qwen-params", json.dumps(qp),
                "--pass1-scene-detector", "semantic", "--pass1-speech-segmenter", seg,
                "--merge-strategy", "pass1_primary", "--output-dir", str(out), "--subs-language", "native",
                "--language", "japanese"]
    return [PY, "-m", "whisperjav.main", str(CLIP), "--mode", "qwen",
            "--qwen-generator", {"qwen3": "qwen3", "anime": "anime-whisper"}[gen], "--qwen-segmenter", seg,
            "--qwen-sensitivity", sens, "--output-dir", str(out), "--subs-language", "native", "--language", "japanese"]


def names():
    return [(r, g, s, x) for r in ("ensemble", "modeqwen") for g in GENS for s in SEGS for x in SENS]


def run(root, only):
    for route, gen, seg, sens in names():
        name = f"{route}_{gen}_{seg}_{sens}"
        if only and name not in only:
            continue
        out = root / name
        if (out / "DONE").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        for f in ("framing.jsonl", "calls.jsonl"):
            (out / f).unlink(missing_ok=True)
        cmd = command(route, gen, seg, sens, out)
        (out / "command.json").write_text(json.dumps(cmd, indent=1), encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(INSTRUMENTS), WJ_RECORD_OUT=str(out / "framing.jsonl"),
                   WJ_TRACE_OUT=str(out / "calls.jsonl"))
        t0 = time.monotonic()
        with open(out / "run.log", "w", encoding="utf-8", errors="replace") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode
        status = {"run": name, "exit": rc, "seconds": round(time.monotonic() - t0, 1),
                  "n_srt": len(list(out.glob("*.srt")))}
        print(json.dumps(status), flush=True)
        if rc == 0:
            (out / "DONE").write_text(json.dumps(status), encoding="utf-8")
    print("finished", flush=True)


# ---- what the run used ---------------------------------------------------------------------------
def actual(out, seg):
    """Values as the run used them. Segmenter object scalars (recorder; every instance must agree), then the
    values at the calls that use them (tracer), which win where both exist."""
    a, objs, longest = {}, [], 0.0
    for line in (out / "framing.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("kind") == "segmenter":
            objs.append(d["attrs"])
        elif d.get("kind") == "frame_call":
            for s in d["segments"]:
                longest = max(longest, s["e"] - s["s"])
    distinct = {json.dumps({k: v for k, v in o.items() if not k.startswith("_")}, sort_keys=True) for o in objs}
    if objs:
        a.update({k: v for k, v in objs[-1].items() if not k.startswith("_")})
    a["_segmenter_objects"] = len(objs)
    a["_objects_disagree"] = len(distinct) > 1
    a["_longest_segment_s"] = round(longest, 3)
    calls = [json.loads(x) for x in (out / "calls.jsonl").read_text(encoding="utf-8").splitlines()] \
        if (out / "calls.jsonl").exists() else []
    used = {}
    for c in calls:
        n = c["call"]
        if n == "ten_vad.TenVad.__init__":
            used.setdefault("threshold", set()).add(c["threshold"])
            used.setdefault("hop_size", set()).add(c["hop_size"])
        elif n.startswith("fireredvad.FireRedVad.from_pretrained"):
            v = c["values"]
            used.setdefault("threshold", set()).add(v.get("prob_threshold"))
            used.setdefault("smooth_window_size", set()).add(v.get("smooth_window_size"))
            used.setdefault("min_speech_duration_ms", set()).add(v.get("min_speech_frame", 0) * 10)
            used.setdefault("min_silence_duration_ms", set()).add(v.get("min_silence_frame", 0) * 10)
            used.setdefault("max_speech_duration_s", set()).add(v.get("max_speech_frame", 0) / 100)
        elif n.startswith("WhisperSegSpeechSegmenter._probs"):
            used.setdefault("threshold", set()).add(c["threshold"])
            used.setdefault("_offset_threshold_used", set()).add(c["offset_threshold_used"])
            used.setdefault("_decoder_used", set()).add(c["decoder"])
        elif n.endswith(("_apply_padding", "_pad_and_convert")):
            used.setdefault("start_pad_ms", set()).add(c["start_pad_ms_read"])
            used.setdefault("end_pad_ms", set()).add(c["end_pad_ms_read"])
        elif n == "AnimeWhisperGenerator.with_leading_silence":
            used.setdefault("lead_in_ms", set()).add(c["silence_added_ms"])
    for k, v in used.items():
        a[k] = sorted(v, key=str)[0] if len(v) == 1 else f"VARIES {sorted(v, key=str)}"
    return a


def dialog(gen, seg, sens):
    """The Customize window's defaults for this pass (what Apply sends)."""
    sys.path.insert(0, str(REPO))
    from whisperjav.webview_gui.api import WhisperJAVAPI
    api = WhisperJAVAPI.__new__(WhisperJAVAPI)
    sch = WhisperJAVAPI.get_qwen_schema(api, sens, {"qwen3": "qwen3", "anime": "anime-whisper"}[gen], seg)["schema"]
    aud, gen_s = sch["audio"], sch.get("generation", {})
    m = {"threshold": ("vad_threshold", 1), "max_speech_duration_s": ("max_speech_duration", 1),
         "start_pad_ms": ("vad_start_pad", 1), "end_pad_ms": ("vad_end_pad", 1),
         "chunk_threshold_s": ("chunk_threshold_ms", 1000), "max_group_duration_s": ("max_group_duration", 1)}
    d = {k: (aud[key]["default"] / div if div != 1 else aud[key]["default"]) for k, (key, div) in m.items()
         if key in aud}
    if "leading_silence_ms" in gen_s:
        d["lead_in_ms"] = gen_s["leading_silence_ms"]["default"]
    if seg == "whisperseg":
        for k, key in (("segmentation_decoder", "vad_decoder"), ("grow_floor", "vad_grow_floor"),
                       ("gap_merge_ms", "vad_gap_merge_ms")):
            if key in aud:
                d[k] = aud[key]["default"]
    return d


def same(x, y):
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        return abs(x - y) < 1e-6
    return x == y


def table(root):
    rows, problems = [], []
    full = {}
    for gen in GENS:
        for seg in SEGS:
            for sens in SENS:
                exp = expected(gen, seg, sens)
                acts = {}
                for route in ("ensemble", "modeqwen"):
                    out = root / f"{route}_{gen}_{seg}_{sens}"
                    acts[route] = actual(out, seg) if (out / "DONE").exists() else None
                dlg = dialog(gen, seg, sens)
                full[f"{gen}_{seg}_{sens}"] = {"expected": exp, "dialog": dlg, **acts}
                cells = []
                for k, v in exp.items():
                    got = {r: (a.get(k, "missing") if a else "not run") for r, a in acts.items()}
                    ok = all(same(g, v) for g in got.values())
                    dk = dlg.get(k, "—")
                    dok = dk == "—" or same(dk, v)
                    if not ok or not dok:
                        problems.append(f"| {gen} | {seg} | {sens} | {k} | {v} | {got['ensemble']} | "
                                        f"{got['modeqwen']} | {dk} |")
                    cells.append(f"{k}={v}" + ("" if ok and dok else " **≠**"))
                ex = []
                for r, a in acts.items():
                    if a:
                        if a["_objects_disagree"]:
                            problems.append(f"| {gen} | {seg} | {sens} | segmenter objects disagree ({r}) | | | | |")
                        # FireRedVAD and WhisperSeg cut before their pads are added; TEN cuts after.
                        lim = 4.0 + (exp["start_pad_ms"] + exp["end_pad_ms"]) / 1000 if seg != "ten" else 4.0
                        if a["_longest_segment_s"] > lim + 0.02:
                            problems.append(f"| {gen} | {seg} | {sens} | longest segment ({r}) | ≤ {lim:.2f} | "
                                            f"{a['_longest_segment_s']} | | |")
                        ex.append(f"{r}: longest {a['_longest_segment_s']} s, objects {a['_segmenter_objects']}"
                                  + (f", offset thr {a.get('_offset_threshold_used')}, decoder {a.get('_decoder_used')}"
                                     if seg == "whisperseg" else ""))
                rows.append(f"| {gen} | {seg} | {sens} | {'; '.join(cells)} | {' / '.join(ex)} |")
    (root / "audit_full.json").write_text(json.dumps(full, indent=1, default=str), encoding="utf-8")
    print("## Every expected value, checked in both routes and in the Customize window "
          "(**≠** = a route or the window differs)\n")
    print("| Model | Segmenter | Sensitivity | Expected = used | Notes (per route) |\n|---|---|---|---|---|")
    print("\n".join(rows))
    print("\n## Differences\n")
    print("| Model | Segmenter | Sensitivity | Setting | Expected | Ensemble run | --mode qwen run | Window |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(problems) if problems else "| none |||||||")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(Path(sys.argv[2]), set(sys.argv[3:]))
    else:
        table(Path(sys.argv[2]))
