#!/usr/bin/env python3
"""Measure semantic-scene boundary DAMAGE on the reference clips (v2).

v1 (measure_cuts.py) assumed the fixed +/-0.35s pad and scored cut placement
only. With v7.2's adaptive pads the honest KPI is what actually harms the
output at each final boundary:

  - duplicated SPEECH: seconds of above-floor audio inside the overlap
    window [cut - next.start_pad, cut + prev.end_pad] that both adjacent
    scenes receive (duplicate ASR text -> resolver clipping -> early ends);
  - pad overshoot INTO speech per side: max(0, pad - silence_extent).

Both are reported for the ACTUAL adaptive pads and for the legacy fixed
0.35/0.35 pads at the SAME cuts, so the pad change is isolated from the cut
change. Uses the vendor's own components so numbers match the pipeline.
"""
import json
import sys
import numpy as np

sys.path.insert(0, r"D:\Git\WhisperJav_V1_Minami_Edition")
from whisperjav.vendor.semantic_audio_clustering import (
    SegmentationConfig, StreamFeatureExtractor, SemanticSegmenter,
    compute_silence_floor, smoothed_rms, compute_adaptive_pads,
)

MANIFEST = r"D:\Git\WhisperJav_V1_Minami_Edition\test_media\reference_benchmarks\manifest.json"
benches = json.load(open(MANIFEST, encoding="utf-8"))["benchmarks"]

FIXED_PAD = 0.35


def speech_seconds(rms, times, floor, t0, t1):
    """Seconds of above-floor audio inside [t0, t1]."""
    if t1 <= t0:
        return 0.0
    i0 = int(np.searchsorted(times, t0))
    i1 = int(np.searchsorted(times, t1))
    i0 = min(max(i0, 0), len(rms))
    i1 = min(max(i1, 0), len(rms))
    if i1 <= i0:
        return 0.0
    frame_dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.032
    return float(np.sum(rms[i0:i1] > floor)) * frame_dt


all_sil, all_L, all_R = [], [], []
dup_new, dup_old = [], []
over_new, over_old = [], []

for b in benches:
    cfg = SegmentationConfig(min_duration=12, max_duration=48)
    ext = StreamFeatureExtractor(cfg)
    features, times, duration = ext.extract(b["media"])
    floor = compute_silence_floor(features, cfg)
    rms = smoothed_rms(features, cfg)
    seg = SemanticSegmenter(cfg)
    segments = seg.segment(features, times, duration, silence_floor=floor)

    print(f"\n=== {b['name']}  dur={duration:.0f}s  {len(segments)} scenes ===")
    for prev, curr in zip(segments, segments[1:]):
        cut = curr["start"]
        idx = min(max(int(np.searchsorted(times, cut)), 0), len(rms) - 1)
        is_sil = rms[idx] <= floor
        # silence extents around the cut
        L = idx
        while L > 0 and rms[L - 1] <= floor:
            L -= 1
        R = idx
        while R < len(rms) - 1 and rms[R + 1] <= floor:
            R += 1
        left_w = float(times[idx] - times[L])
        right_w = float(times[R] - times[idx])

        # actual adaptive pads for the two scenes meeting at this cut
        _, prev_end_pad = compute_adaptive_pads(prev["start"], prev["end"], rms, times, floor)
        curr_start_pad, _ = compute_adaptive_pads(curr["start"], curr["end"], rms, times, floor)

        d_new = speech_seconds(rms, times, floor, cut - curr_start_pad, cut + prev_end_pad)
        d_old = speech_seconds(rms, times, floor, cut - FIXED_PAD, cut + FIXED_PAD)
        o_new = max(0.0, curr_start_pad - left_w) + max(0.0, prev_end_pad - right_w)
        o_old = max(0.0, FIXED_PAD - left_w) + max(0.0, FIXED_PAD - right_w)

        all_sil.append(is_sil); all_L.append(left_w); all_R.append(right_w)
        dup_new.append(d_new); dup_old.append(d_old)
        over_new.append(o_new); over_old.append(o_old)
        print(
            f"  cut@{int(cut//60)}:{cut%60:05.2f} sil={str(is_sil):5s} "
            f"L={left_w:5.2f}s R={right_w:5.2f}s pads=({curr_start_pad:.2f},{prev_end_pad:.2f}) "
            f"dupSpeech new={d_new:.2f}s old={d_old:.2f}s"
        )

n = len(all_sil)
aL, aR = np.array(all_L), np.array(all_R)
dn, do = np.array(dup_new), np.array(dup_old)
on, oo = np.array(over_new), np.array(over_old)
print(f"\n================ SUMMARY over {n} final cuts ({len(benches)} clips) ================")
print(f"cut lands in silence:                      {100*np.mean(all_sil):.0f}%")
print(f"full 0.35s silence before cut (pad-ready): {100*np.mean(aL >= 0.35):.0f}%")
print(f"median silence: L={np.median(aL):.2f}s R={np.median(aR):.2f}s")
print(f"DUPLICATED SPEECH per boundary:  adaptive={np.mean(dn):.3f}s mean / {np.median(dn):.3f}s median"
      f"   fixed-0.35={np.mean(do):.3f}s mean / {np.median(do):.3f}s median")
print(f"PAD OVERSHOOT into speech:       adaptive={np.mean(on):.3f}s mean"
      f"   fixed-0.35={np.mean(oo):.3f}s mean")
