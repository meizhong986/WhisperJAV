#!/usr/bin/env python3
"""Prepare reference benchmark clips + ground-truth SRTs from Netflix MKV sources.

For each (name, source, start, end):
  1. clip the MKV with -c copy (lossless), keeping video + primary audio + ONLY
     the Japanese SDH subtitle track (detected by language=jpn + title contains
     'SDH'). Timestamps reset to 0 so the clip is self-contained.
  2. extract the SDH track from the clip -> raw SRT (0-based, aligned to clip audio).
  3. transform SDH -> ground truth: strip bracketed sound/speaker cues
     (（）()【】［］〈〉«»), music (♪), and bidi format marks; drop entries that
     become empty (sound-only cues); renumber.
  4. verify 0-based alignment (first/last within clip duration) and report.
Writes/updates test_media/reference_benchmarks/manifest.json.

Usage:
    python -m tools.vad_hypothesis_suite.prep_benchmark [--only NAME[,NAME]]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import pysrt

ROOT = r"D:\Git\WhisperJav_V1_Minami_Edition"
BENCH_DIR = os.path.join(ROOT, "test_media", "reference_benchmarks")

# (name, source mkv, start, end)
SOURCES = [
    ("S02E02_more",
     r"F:\The.Naked.Director\The.Naked.Director.S02E02\The.Naked.Director.S02E02.More.More.More.1080p.NF.WEB-DL.DDP5.1.Atmos.x264-TEPES.mkv",
     "00:19:40.420", "00:21:57.840"),
    ("S02E04_dream",
     r"F:\The.Naked.Director\The.Naked.Director.S02E04\The.Naked.Director.S02E04.Our.Dream.1080p.NF.WEB-DL.DDP5.1.Atmos.x264-TEPES.mkv",
     "00:36:38.540", "00:40:07.780"),
    ("S02E05_bubble",
     r"F:\The.Naked.Director\The.Naked.Director.S02E05\The.Naked.Director.S02E05.The.Bubble.Bursts.1080p.NF.WEB-DL.DDP5.1.Atmos.x264-TEPES.mkv",
     "00:00:23.600", "00:04:10.416"),
]

# Original benchmark (already prepared) — recorded in the manifest for the sweep.
ORIGINAL = {
    "name": "S01E04_scene4",
    "media": os.path.join(ROOT, "test_media", "293sec-The.Naked.Director.S01E04.Scene4.mkv"),
    "gt": os.path.join(ROOT, "test_media", "1815acceptance", "T3",
                       "Ground_Truth-293sec-The.Naked.Director.S01E04.Scene4-sanitized.srt"),
    "note": "original manually-sanitized benchmark",
}

_BIDI = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮⁦⁧⁨⁩"), None)
_BRACKETS = re.compile(r"（[^（）]*）|\([^()]*\)|【[^【】]*】|［[^［］]*］|〈[^〈〉]*〉|«[^«»]*»|♪[^♪]*♪|♪")


def strip_sdh(text: str) -> str:
    """Remove SDH annotations; return bare dialogue (may be empty)."""
    t = (text or "").translate(_BIDI)
    prev = None
    while prev != t:                      # iterate to clear nested brackets
        prev = t
        t = _BRACKETS.sub("", t)
    return re.sub(r"[ 　]+", " ", t.replace("\n", " ")).strip()


def _ffprobe_sdh_index(source: str) -> int:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index:stream_tags=language,title",
         "-of", "json", source], text=True, encoding="utf-8")
    for st in json.loads(out)["streams"]:
        tags = st.get("tags", {})
        if tags.get("language") == "jpn" and "SDH" in (tags.get("title") or ""):
            return st["index"]
    raise RuntimeError(f"No 'jpn ... SDH' subtitle track in {source}")


def _duration(path: str) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], text=True, encoding="utf-8")
    return float(out.strip())


def prep_one(name, source, start, end) -> dict:
    out_dir = os.path.join(BENCH_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    clip = os.path.join(out_dir, "clip.mkv")
    raw = os.path.join(out_dir, "clip.jpn_sdh.raw.srt")
    gt = os.path.join(out_dir, "clip.jpn_sdh.gt.srt")

    sdh_idx = _ffprobe_sdh_index(source)
    if not os.path.exists(clip):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", start, "-to", end, "-i", source,
             "-map", "0:v:0", "-map", "0:a:0", "-map", f"0:{sdh_idx}",
             "-c", "copy", "-avoid_negative_ts", "make_zero", clip],
            check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip, "-map", "0:s:0", raw], check=True)

    subs = pysrt.open(raw, encoding="utf-8")
    kept = []
    dropped = 0
    for s in subs:
        txt = strip_sdh(s.text)
        if not txt:
            dropped += 1
            continue
        s.text = txt
        kept.append(s)
    for i, s in enumerate(kept, 1):
        s.index = i
    pysrt.SubRipFile(items=kept).save(gt, encoding="utf-8")

    dur = _duration(clip)
    first = kept[0].start.ordinal / 1000.0 if kept else 0.0
    last = kept[-1].end.ordinal / 1000.0 if kept else 0.0
    aligned = last <= dur + 1.0 and first >= 0.0
    return {
        "name": name, "media": clip, "gt": gt, "source": source,
        "clip": [start, end], "duration": round(dur, 2),
        "sdh_track_index": sdh_idx, "raw_entries": len(subs),
        "gt_entries": len(kept), "sdh_cues_dropped": dropped,
        "first_sub_s": round(first, 2), "last_sub_s": round(last, 2),
        "aligned": aligned,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)
    only = set(args.only.split(",")) if args.only else None

    entries = [ORIGINAL]
    for name, source, start, end in SOURCES:
        if only and name not in only:
            continue
        print(f">>> {name} ...", flush=True)
        rec = prep_one(name, source, start, end)
        entries.append(rec)
        flag = "OK" if rec["aligned"] else "!! ALIGN?"
        print(f"    {flag}  dur={rec['duration']}s  gt={rec['gt_entries']} entries "
              f"(-{rec['sdh_cues_dropped']} cues)  subs {rec['first_sub_s']}..{rec['last_sub_s']}s")

    manifest = os.path.join(BENCH_DIR, "manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"benchmarks": entries}, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {manifest} ({len(entries)} benchmarks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
