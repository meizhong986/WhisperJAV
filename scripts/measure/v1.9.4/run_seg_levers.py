"""REQ2 step 3 (2026-10-08): one setting changed per run, for TEN and FireRedVAD with Qwen3-ASR and anime-whisper at
every sensitivity, WhisperSeg with Qwen3-ASR at conservative and aggressive, and the anime-whisper lead-in silence
switched off for Silero. The base command is the baseline command of run_chronosjav_reference_runs.py (GUI Ensemble
pass 1, default timestamp mode); the changed value goes in through the same routes the GUI uses:
  --pass1-params       threshold, neg_threshold, min_silence_duration_ms, max_speech_duration_s, smooth_window_size
  --pass1-qwen-params  vad_end_pad, max_group_duration, chunk_threshold, leading_silence_ms
Every run carries the recorder and the call tracer (record_and_trace), so check_levers.py can show that the changed
value, and only it, reached the call. Resumable (DONE marker).
Usage: python run_seg_levers.py <out folder> [--list] [run name ...]"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable
CLIPS = sorted(str(p) for p in (REPO / "test_media" / "Ground_Truths" / "Netflix").glob("*.mkv"))
INSTRUMENTS = REPO / "scripts" / "measure" / "record_and_trace"
GEN = {
    "qwen3": ["--pass1-model", "Qwen/Qwen3-ASR-1.7B"],
    "anime": ["--pass1-model", "litagin/anime-whisper"],
}
QP = {
    "qwen3": {"framer": "vad-grouped"},
    "anime": {"framer": "vad-grouped", "generator_backend": "anime-whisper", "timestamp_mode": "vad_only",
              "assembly_cleaner": "passthrough", "stepdown": False},
}
# Baseline values as traced/recorded on 2026-10-08 (trace_table.md, baseline_scores.md)
SEG = {  # segmenter -> sensitivity -> (threshold, min silence ms, longest segment s)
    "ten": {"conservative": (0.42, 200, 6.0), "balanced": (0.32, 150, 5.0), "aggressive": (0.22, 100, 4.0)},
    "firered-vad": {"conservative": (0.5, 250, 7.0), "balanced": (0.4, 200, 6.0), "aggressive": (0.3, 150, 5.0)},
}
GROUP = {  # generator -> sensitivity -> (group cap s, group gap s)
    "qwen3": {s: (3.0, 0.3) for s in ("conservative", "balanced", "aggressive")},
    "anime": {"conservative": (3.0, 0.3), "balanced": (2.5, 0.25), "aggressive": (2.0, 0.2)},
}


def lever_table():
    runs = {}  # name -> (generator, segmenter, sensitivity, pass1-params, extra qwen-params)
    for gen in ("qwen3", "anime"):
        for seg in ("ten", "firered-vad"):
            for sens in ("conservative", "balanced", "aggressive"):
                thr, minsil, mx = SEG[seg][sens]
                cap, gap = GROUP[gen][sens]
                base = f"{gen}_{seg}_{sens}"
                lv = {
                    "thr_up": ({"threshold": round(thr + 0.10, 2)}, {}),
                    "thr_dn": ({"threshold": round(thr - 0.10, 2)}, {}),
                    "minsil_half": ({"min_silence_duration_ms": minsil // 2}, {}),
                    f"max{3 if sens == 'aggressive' else 4}": (
                        {"max_speech_duration_s": 3.0 if sens == "aggressive" else 4.0}, {}),
                    "endpad0": ({}, {"vad_end_pad": 0}),
                    "grp_less": ({}, {"max_group_duration": max(1.5, cap - 1.0)}),
                    "gap_half": ({}, {"chunk_threshold": round(gap / 2, 3)}),
                }
                if seg == "firered-vad":
                    lv["smooth3"] = ({"smooth_window_size": 3}, {})
                if gen == "anime":
                    lv["lead0"] = ({}, {"leading_silence_ms": 0})
                for k, (pp, qx) in lv.items():
                    runs[f"{base}__{k}"] = (gen, seg, sens, pp, qx)
    for sens in ("conservative", "aggressive"):
        base = f"qwen3_whisperseg_{sens}"
        for k, (pp, qx) in {"neg020": ({"neg_threshold": 0.20}, {}), "thr_up": ({"threshold": 0.35}, {}),
                            "minsil_half": ({"min_silence_duration_ms": 100 if sens == "conservative" else 40}, {}),
                            "endpad0": ({}, {"vad_end_pad": 0})}.items():
            runs[f"{base}__{k}"] = ("qwen3", "whisperseg", sens, pp, qx)
    for sens in ("conservative", "balanced", "aggressive"):
        runs[f"anime_silero_{sens}__lead0"] = ("anime", "silero", sens, {}, {"leading_silence_ms": 0})
    # Second round: values between the ones tried, and the best single changes together (chosen from
    # levers_scores.md: text kept first, then timing).
    T, MS, MX = "threshold", "min_silence_duration_ms", "max_speech_duration_s"
    combos = {
        "qwen3_ten_aggressive": {"c_thr027": ({T: 0.27}, {}), "c_thr042": ({T: 0.42}, {}),
                                 "c_thr032_grp2": ({T: 0.32}, {"max_group_duration": 2.0}),
                                 "c_thr037": ({T: 0.37}, {})},
        "qwen3_ten_balanced": {"c_thr037": ({T: 0.37}, {}), "c_thr042_ms75": ({T: 0.42, MS: 75}, {}),
                               "c_thr042_grp2": ({T: 0.42}, {"max_group_duration": 2.0}),
                               "c_thr047": ({T: 0.47}, {})},
        "anime_ten_aggressive": {"c_thr027": ({T: 0.27}, {}), "c_thr027_ms50": ({T: 0.27, MS: 50}, {})},
        "anime_ten_balanced": {"c_thr037": ({T: 0.37}, {}), "c_thr042_ms75": ({T: 0.42, MS: 75}, {}),
                               "c_thr042_grp15": ({T: 0.42}, {"max_group_duration": 1.5})},
        "anime_ten_conservative": {"c_thr047": ({T: 0.47}, {}),
                                   "c_ms100_gap015": ({MS: 100}, {"chunk_threshold": 0.15}),
                                   "c_thr037": ({T: 0.37}, {})},
        # Confirming runs (owner, 2026-10-08: "run the two confirming runs first"): conservative between 0.32 and 0.42.
        "qwen3_ten_conservative": {"c_thr037": ({T: 0.37}, {})},
        "qwen3_firered-vad_aggressive": {"c_max4": ({MX: 4.0}, {}), "c_max4_thr04": ({MX: 4.0, T: 0.4}, {})},
        "qwen3_firered-vad_balanced": {"c_max5": ({MX: 5.0}, {})},
        "qwen3_firered-vad_conservative": {"c_max5": ({MX: 5.0}, {}), "c_max6": ({MX: 6.0}, {})},
        "anime_firered-vad_aggressive": {"c_max4": ({MX: 4.0}, {}), "c_max4_thr04": ({MX: 4.0, T: 0.4}, {})},
        "anime_firered-vad_balanced": {"c_max5": ({MX: 5.0}, {}),
                                       "c_thr05_gap0125": ({T: 0.5}, {"chunk_threshold": 0.125})},
        "anime_firered-vad_conservative": {"c_max5": ({MX: 5.0}, {}), "c_max6": ({MX: 6.0}, {})},
        "qwen3_whisperseg_conservative": {"c_endpad0_neg020": ({"neg_threshold": 0.20}, {"vad_end_pad": 0}),
                                          "c_endpad50": ({}, {"vad_end_pad": 50})},
        "qwen3_whisperseg_aggressive": {"c_endpad0_neg020": ({"neg_threshold": 0.20}, {"vad_end_pad": 0}),
                                        "c_endpad50": ({}, {"vad_end_pad": 50})},
    }
    for base, lv in combos.items():
        gen, seg, sens = base.split("_", 2)
        for k, (pp, qx) in lv.items():
            runs[f"{base}__{k}"] = (gen, seg, sens, pp, qx)
    # Grid (owner, 2026-10-08, ordering recovery): each sensitivity's other settings with each sensitivity's
    # threshold, so the threshold's effect is separated from the rest. Cells not already covered by a run above.
    grid = {"qwen3_ten_conservative": [0.22], "anime_ten_conservative": [0.22], "anime_ten_aggressive": [0.42],
            "qwen3_firered-vad_conservative": [0.3], "qwen3_firered-vad_aggressive": [0.5],
            "anime_firered-vad_conservative": [0.3], "anime_firered-vad_aggressive": [0.5]}
    for base, thrs in grid.items():
        gen, seg, sens = base.split("_", 2)
        for t in thrs:
            runs[f"{base}__g_thr{int(round(t * 100)):03d}"] = (gen, seg, sens, {T: t}, {})
    # Does the longest segment explain the later line ends at threshold 0.22 with conservative's other settings?
    for gen in ("qwen3", "anime"):
        runs[f"{gen}_ten_conservative__g_thr022_max4"] = (gen, "ten", "conservative", {T: 0.22, MX: 4.0}, {})
    # TEN longest-segment split fixed in ten.py (owner, 2026-10-08: "FIX IT"). Run into a separate folder:
    # today's settings on the fixed code, and the two open checks he approved ("3- ok").
    for gen in ("qwen3", "anime"):
        for sens in ("conservative", "balanced", "aggressive"):
            runs[f"{gen}_ten_{sens}__fix"] = (gen, "ten", sens, {}, {})
    runs["qwen3_ten_aggressive__fix_ms200"] = ("qwen3", "ten", "aggressive", {"min_speech_duration_ms": 200}, {})
    runs["qwen3_firered-vad_aggressive__ms250"] = ("qwen3", "firered-vad", "aggressive",
                                                   {"min_speech_duration_ms": 250}, {})
    runs["anime_ten_conservative__fix_g_thr022_max4"] = ("anime", "ten", "conservative", {T: 0.22, MX: 4.0}, {})
    runs["anime_ten_conservative__fix_g_thr022_max4_ep30"] = ("anime", "ten", "conservative", {T: 0.22, MX: 4.0},
                                                             {"vad_end_pad": 30})
    # Check after the product change (owner, 2026-10-08: TEN and FireRedVAD 4 s for ChronosJAV): today's
    # conservative command, no setting passed; the 4 s must come from the product defaults.
    for gen in ("qwen3", "anime"):
        for seg in ("ten", "firered-vad"):
            runs[f"{gen}_{seg}_conservative__product4s"] = (gen, seg, "conservative", {}, {})
    # Check after the product change (owner, 2026-10-09, "E- implement both"): anime-whisper + Silero, no setting
    # passed; the lead-in must now be off by default (output equal to the __lead0 runs). And Qwen3-ASR + WhisperSeg
    # balanced with end pad 0, the one sensitivity not measured with it at today's settings.
    for sens in ("conservative", "balanced", "aggressive"):
        runs[f"anime_silero_{sens}__productlead"] = ("anime", "silero", sens, {}, {})
    runs["qwen3_whisperseg_balanced__endpad0"] = ("qwen3", "whisperseg", "balanced", {}, {"vad_end_pad": 0})
    # After his "0 at all three" (2026-10-09): the same, no setting passed; must equal __endpad0.
    runs["qwen3_whisperseg_balanced__productpad"] = ("qwen3", "whisperseg", "balanced", {}, {})
    return runs


def main():
    out_root = Path(sys.argv[1])
    table = lever_table()
    args = sys.argv[2:]
    if args[:1] == ["--list"]:
        for k, v in table.items():
            print(k, json.dumps(v[3]), json.dumps(v[4]))
        print(len(table), "runs")
        return
    names = args or list(table)
    for name in names:
        gen, seg, sens, pp, qx = table[name]
        out = out_root / name / "run1"
        if (out / "DONE").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "-m", "whisperjav.main", *CLIPS, "--ensemble", "--pass1-pipeline", "qwen",
               "--pass1-sensitivity", sens, *GEN[gen], "--pass1-qwen-params", json.dumps({**QP[gen], **qx})]
        if pp:
            cmd += ["--pass1-params", json.dumps(pp)]
        cmd += ["--pass1-scene-detector", "semantic", "--pass1-speech-segmenter", seg,
                "--merge-strategy", "pass1_primary", "--output-dir", str(out), "--subs-language", "native",
                "--language", "japanese"]
        (out / "command.json").write_text(json.dumps(cmd, indent=1), encoding="utf-8")
        for f in ("framing.jsonl", "calls.jsonl"):
            (out / f).unlink(missing_ok=True)
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


if __name__ == "__main__":
    main()
