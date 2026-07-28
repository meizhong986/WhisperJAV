#!/usr/bin/env python3
"""Measure semantic-scene cut safety on the reference clips.

For each semantic cut boundary, report:
  - RMS at the snapped point vs the calibrated silence threshold (did it hit silence?)
  - silence WIDTH around the cut (how far RMS stays below the silence threshold),
    per side, vs the 0.35s padding.
Uses the vendor's own extractor/segmenter/classifier so numbers match the pipeline.
"""
import json
import sys
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, r"D:\Git\WhisperJav_V1_Minami_Edition")
from whisperjav.vendor.semantic_audio_clustering import (
    SegmentationConfig, StreamFeatureExtractor, SemanticSegmenter,
    AdaptiveClassifier, FeatureRegistry,
)

PAD = 0.35
MANIFEST = r"D:\Git\WhisperJav_V1_Minami_Edition\test_media\reference_benchmarks\manifest.json"
benches = json.load(open(MANIFEST, encoding="utf-8"))["benchmarks"]

all_left, all_right, all_is_sil, all_rms_ratio = [], [], [], []
for b in benches:
    media = b["media"]
    # qwen safe_chunking uses min 12 / max 48
    cfg = SegmentationConfig(min_duration=12, max_duration=48)
    ext = StreamFeatureExtractor(cfg)
    features, times, duration = ext.extract(media)
    clf = AdaptiveClassifier(cfg); clf.calibrate(features)
    sil = clf.stats["rms_base"] * cfg.silence_threshold_multiplier
    seg = SemanticSegmenter(cfg)
    segments = seg.segment(features, times, duration)
    rms = median_filter(features[FeatureRegistry.RMS, :], size=cfg.rms_smoothing_window)

    bounds = sorted({s["start"] for s in segments if s["start"] > 0.05 and s["start"] < duration - 0.05})
    print(f"\n=== {b['name']}  dur={duration:.0f}s  {len(bounds)} internal cuts  "
          f"silence_thresh={sil:.4f} ===")
    for cut in bounds:
        idx = int(np.searchsorted(times, cut))
        idx = min(max(idx, 0), len(rms) - 1)
        rms_b = rms[idx]
        is_sil = rms_b <= sil
        L = idx
        while L > 0 and rms[L - 1] <= sil:
            L -= 1
        R = idx
        while R < len(rms) - 1 and rms[R + 1] <= sil:
            R += 1
        left_w = times[idx] - times[L]
        right_w = times[R] - times[idx]
        all_left.append(left_w); all_right.append(right_w)
        all_is_sil.append(is_sil); all_rms_ratio.append(rms_b / sil)
        flag = "" if (is_sil and left_w >= PAD and right_w >= PAD) else "  <-- pad overshoots speech"
        print(f"  cut@{int(cut//60)}:{cut%60:05.2f}  rms/thr={rms_b/sil:4.2f}  "
              f"silence L={left_w:.2f}s R={right_w:.2f}s{flag}")

n = len(all_is_sil)
al, ar = np.array(all_left), np.array(all_right)
print(f"\n================ SUMMARY over {n} cuts ({len(benches)} clips) ================")
print(f"cut lands below silence threshold:        {100*np.mean(all_is_sil):.0f}%")
print(f"silence >= 0.35s on BOTH sides (pad safe): {100*np.mean((al>=PAD)&(ar>=PAD)):.0f}%")
print(f"pad OVERSHOOTS speech on >=1 side:         {100*np.mean((al<PAD)|(ar<PAD)):.0f}%")
print(f"total silence width < 0.70s (overlap zone hits speech): {100*np.mean(al+ar<2*PAD):.0f}%")
print(f"median silence half-width: L={np.median(al):.2f}s  R={np.median(ar):.2f}s")
print(f"median rms/threshold at cut: {np.median(all_rms_ratio):.2f}")
