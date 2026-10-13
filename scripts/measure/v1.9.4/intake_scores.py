"""REQ2 ordering recovery (owner, 2026-10-08): how much audio each run let in, measured from the segmenter recorder,
without the ASR. A sensitivity is an ordered setting: aggressive must let in at least as much audio as balanced, and
balanced at least as much as conservative.

Source per run: framing.jsonl (scripts/measure/segmenter_recorder). Each "scenes" record lists a clip's scenes
(path, start, end, duration on the clip timeline); each "frame_call" record is one scene, in the same order, with
  segments  the segmenter's speech regions after padding and splitting (scene time)
  frames    the audio windows sent to the speech recogniser after grouping (scene time; a frame can span the gap
            between grouped segments)
Both are moved to the clip timeline (scene start + time) and merged (union; scenes overlap by about 0.4 s).

Measures per run, pooled over the 7 clips (times in seconds):
  audio in      length of the union of frames (what the recogniser hears); also given for segments
  clip time     sum of clip durations (end of the last scene)
  speech time   length of the union of reference-line spans (start to end of each reference SRT line)
  speech in     length of (frames ∩ reference lines) / speech time        -> the segmenter's own coverage
  non-speech in length of (frames − reference lines)                      -> audio let in outside the lines
Reference-line spans are subtitle times, not exact speech; they include the subtitler's lead and lag.

Usage: python intake_scores.py <levers folder> <baseline folder> [--json out.json]   (markdown on stdout)"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from subeval.cli import clip_id  # noqa: E402
from subeval.srt import read_srt  # noqa: E402
from run_seg_levers import SEG, lever_table  # noqa: E402

REF = REPO / "test_media" / "Ground_Truths" / "Netflix"
SENS = ("conservative", "balanced", "aggressive")
MISMATCH = [0.0]  # largest scene-duration difference between the scene list and its framing call


def union(iv):
    out = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def length(iv):
    return sum(e - s for s, e in iv)


def intersect(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append([s, e])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def reference_spans():
    out = {}
    for ref in sorted(REF.glob("*.srt")):
        out[clip_id(ref)] = union([(c.start, c.end) for c in read_srt(ref).cues])
    return out


def run_intervals(run):
    """{clip: {"segments": [...], "frames": [...], "clip_end": t}} on the clip timeline."""
    scenes, calls = [], []
    for line in (run / "framing.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        if d.get("kind") == "scenes":
            scenes += d["scenes"]
        elif d.get("kind") == "frame_call":
            calls.append(d)
    if len(scenes) != len(calls):
        raise ValueError(f"{run}: {len(scenes)} scenes but {len(calls)} framing calls")
    out = {}
    for (path, start, end, dur), call in zip(scenes, calls):
        MISMATCH[0] = max(MISMATCH[0], abs(float(call["dur"]) - dur))
        if abs(float(call["dur"]) - dur) > 0.1:  # last scene of a clip can be ~0.05 s short (end of the audio)
            raise ValueError(f"{run}: scene {Path(path).name} lasts {dur} s but its framing call {call['dur']} s")
        clip = Path(path).name.split("_scene_")[0]
        c = out.setdefault(clip, {"segments": [], "frames": [], "clip_end": 0.0})
        c["segments"] += [(start + g["s"], start + g["e"]) for g in call["segments"]]
        c["frames"] += [(start + f[0], start + f[1]) for f in call["frames"]]
        c["clip_end"] = max(c["clip_end"], end)
    return out


def intake(run, refs):
    iv = run_intervals(run)
    if sorted(iv) != sorted(refs):
        raise ValueError(f"{run}: clips {sorted(iv)} differ from the references {sorted(refs)}")
    t = {"clip": 0.0, "speech": 0.0, "audio_in": 0.0, "seg_in": 0.0, "speech_in": 0.0, "seg_speech_in": 0.0}
    for clip, c in iv.items():
        fr, sg, rf = union(c["frames"]), union(c["segments"]), refs[clip]
        t["clip"] += c["clip_end"]
        t["speech"] += length(rf)
        t["audio_in"] += length(fr)
        t["seg_in"] += length(sg)
        t["speech_in"] += length(intersect(fr, rf))
        t["seg_speech_in"] += length(intersect(sg, rf))
    t["nonspeech_in"] = t["audio_in"] - t["speech_in"]
    t["audio_share"] = t["audio_in"] / t["clip"]
    t["speech_share"] = t["speech_in"] / t["speech"]
    t["seg_speech_share"] = t["seg_speech_in"] / t["speech"]
    return t


def threshold_of(combo, lever, table):
    """The run's segmenter threshold if the run differs from its base in the threshold only, else None."""
    gen, seg, sens = combo.split("_", 2)
    if lever == "baseline":
        return SEG[seg][sens][0] if seg in SEG else None
    pp, qx = table[f"{combo}__{lever}"][3:5]
    return pp["threshold"] if set(pp) == {"threshold"} and not qx else None


def main():
    lev_root, base_root = Path(sys.argv[1]), Path(sys.argv[2])
    refs = reference_spans()
    table = lever_table()
    rows = {}
    for p in sorted(base_root.glob("*/run1/DONE")):
        rows[(p.parent.parent.name, "baseline")] = intake(p.parent, refs)
    for p in sorted(lev_root.glob("*/run1/DONE")):
        combo, lever = p.parent.parent.name.split("__")
        rows[(combo, lever)] = intake(p.parent, refs)
    any_row = next(iter(rows.values()))
    print(f"Clip time {any_row['clip']:.1f} s; reference speech time {any_row['speech']:.1f} s "
          f"({any_row['speech'] / any_row['clip']:.1%}). Largest scene/framing-call duration difference: "
          f"{MISMATCH[0]:.3f} s.\n")

    print("### Today's settings by sensitivity (conservative / balanced / aggressive)\n")
    print("| Model, segmenter | Audio in (share of clip) | Speech in (frames) | Speech in (segments) "
          "| Non-speech in (s) | Ordered? |")
    print("|---|---|---|---|---|---|")
    for gs in sorted({c.rsplit("_", 1)[0] for c, lv in rows if lv == "baseline"}):
        t = [rows.get((f"{gs}_{s}", "baseline")) for s in SENS]
        if None in t:
            continue
        a = [x["audio_in"] for x in t]
        ok = "yes" if a[0] <= a[1] <= a[2] else "**no**"
        print(f"| {gs} | " + " / ".join(f"{x['audio_share']:.1%}" for x in t) + " | "
              + " / ".join(f"{x['speech_share']:.1%}" for x in t) + " | "
              + " / ".join(f"{x['seg_speech_share']:.1%}" for x in t) + " | "
              + " / ".join(f"{x['nonspeech_in']:.0f}" for x in t) + f" | {ok} |")

    print("\n### Threshold alone (runs that differ from their base in the threshold only)\n")
    print("| Model, segmenter, other settings of | Threshold → audio in / speech in / non-speech in (s) "
          "| Lower threshold never lets in less? |")
    print("|---|---|---|")
    for combo in sorted({c for c, _ in rows}):
        pts = sorted((threshold_of(combo, lv, table), r) for (c, lv), r in rows.items()
                     if c == combo and threshold_of(combo, lv, table) is not None)
        if len(pts) < 2:
            continue
        mono = all(pts[i][1]["audio_in"] >= pts[i + 1][1]["audio_in"] - 1e-6 for i in range(len(pts) - 1))
        print(f"| {combo} | " + "; ".join(f"{th:.2f} → {r['audio_share']:.1%} / {r['speech_share']:.1%} / "
                                          f"{r['nonspeech_in']:.0f}" for th, r in pts)
              + f" | {'yes' if mono else '**no**'} |")

    print("\n### Every other setting's effect on audio in (change against its base)\n")
    print("| Run | Audio in | Speech in | Non-speech in (s) |")
    print("|---|---|---|---|")
    for (combo, lever), r in sorted(rows.items()):
        if lever == "baseline" or threshold_of(combo, lever, table) is not None:
            continue
        b = rows[(combo, "baseline")]
        print(f"| {combo} {lever} | {100 * (r['audio_share'] - b['audio_share']):+.1f} pt "
              f"| {100 * (r['speech_share'] - b['speech_share']):+.1f} pt "
              f"| {r['nonspeech_in'] - b['nonspeech_in']:+.0f} |")

    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        out.write_text(json.dumps({f"{c}__{lv}": r for (c, lv), r in rows.items()}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
