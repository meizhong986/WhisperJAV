#!/usr/bin/env python3
"""Load the reference-benchmark manifest (media + ground-truth per clip)."""

import json
import os

ROOT = r"D:\Git\WhisperJav_V1_Minami_Edition"
MANIFEST = os.path.join(ROOT, "test_media", "reference_benchmarks", "manifest.json")

# Focused cross-clip VALIDATION subset (names must exist in configs.CONFIGS):
# the round 1-3 winners + baseline + one refuted config as a control. Used for
# the multi-clip run so we validate ~7 configs x 4 clips (~20 min) instead of
# the full 21-config grid x 4.
VALIDATION = [
    "B0_baseline_aggressive",   # baseline
    "H3_thr_0.15",              # threshold win (single knob)
    "R2_thr_0.12",              # threshold optimum
    "R2_thr0.15_end30",         # best CER combo
    "R2_thr0.12_end30",         # best recall combo
    "R3_thr0.15_ms6_end30",     # best-CER + long-context combo
    "H1_maxspeech_2.0",         # CONTROL: refuted (should stay worse cross-clip)
]


def load_benchmarks():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)["benchmarks"]
