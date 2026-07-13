#!/usr/bin/env python3
"""Hypothesis / knob grid for the anime-whisper VAD tuning sweep.

Design: one-factor-at-a-time (OFAT) around the AGGRESSIVE baseline, which the T3
scoring showed is the better of the two shipped presets (lower CER, lower
deletions, more segments). Each config changes exactly ONE knob vs the baseline
so the effect on the metric vector is attributable.

Reference benchmark (fixed): the 293s "Naked Director S01E04 Scene4" clip + its
sanitized ground-truth SRT.
"""

import os

_ROOT = r"D:\Git\WhisperJav_V1_Minami_Edition"
REFERENCE_MEDIA = os.path.join(_ROOT, r"test_media\293sec-The.Naked.Director.S01E04.Scene4.mkv")
GROUND_TRUTH = os.path.join(
    _ROOT, r"test_media\1815acceptance\T3\Ground_Truth-293sec-The.Naked.Director.S01E04.Scene4-sanitized.srt"
)

# Every run is anime-whisper, semantic scene, aggressive preset. The aggressive
# per-sensitivity table supplies chunk 0.20 / max_group 2.0 / threshold 0.25 /
# pad 0/0; max_speech comes from the WhisperSeg YAML aggressive preset (=4.0).
BASE_ARGS = [
    "--mode", "qwen",
    "--qwen-generator", "anime-whisper",
    "--qwen-sensitivity", "aggressive",
    "--qwen-scene", "semantic",
]

# name -> (hypothesis label, extra CLI args, human note)
CONFIGS = [
    # Baseline (should reproduce pass2 ~35 segments; also a determinism check).
    ("B0_baseline_aggressive", "baseline", [],
     "aggressive preset as-is (max_speech 4, chunk 0.20, thr 0.25, pad 0/0)"),

    # H1 - max_speech is the binding cap on subtitle length. Lower -> shorter,
    #      more granular subs (target GT mean 2.0s). Strongest granularity lever.
    ("H1_maxspeech_3.0", "H1 granularity/max_speech", ["--qwen-max-speech-duration", "3.0"],
     "force-split at 3.0s"),
    ("H1_maxspeech_2.5", "H1 granularity/max_speech", ["--qwen-max-speech-duration", "2.5"],
     "force-split at 2.5s"),
    ("H1_maxspeech_2.0", "H1 granularity/max_speech", ["--qwen-max-speech-duration", "2.0"],
     "force-split at 2.0s (~GT mean)"),

    # H2 - chunk_threshold (frame gap). Natural inter-line gaps are ~0.12s; the
    #      0.20 baseline merges them. Lower -> split on natural pauses.
    ("H2_chunk_0.15", "H2 granularity/chunk_threshold", ["--qwen-chunk-threshold", "0.15"],
     "group gap 0.15s"),
    ("H2_chunk_0.10", "H2 granularity/chunk_threshold", ["--qwen-chunk-threshold", "0.10"],
     "group gap 0.10s (below the ~0.12s natural gaps)"),

    # H3 - VAD threshold. Recover soft moans/interjections in the weak regions.
    ("H3_thr_0.20", "H3 recall/threshold", ["--qwen-vad-threshold", "0.20"],
     "threshold 0.20"),
    ("H3_thr_0.15", "H3 recall/threshold", ["--qwen-vad-threshold", "0.15"],
     "threshold 0.15 (aggressive soft-speech capture)"),

    # H4 - end pad. Aggressive runs 0/0 -> may clip soft onsets/particles. The
    #      scene-overlap resolver now absorbs the overlap risk of a small pad.
    ("H4_endpad_30", "H4 edges/end_pad", ["--qwen-vad-end-pad", "30"],
     "end pad 30ms"),
    ("H4_endpad_50", "H4 edges/end_pad", ["--qwen-vad-end-pad", "50"],
     "end pad 50ms"),

    # H5 - are the recall gaps a scene-detection problem (not VAD)? Swap the
    #      scene detector; if the weak regions recover, the gap is upstream.
    ("H5_scene_auditok", "H5 recall/scene_detector", ["--qwen-scene", "auditok"],
     "auditok scene detection instead of semantic"),
    ("H5_scene_min6", "H5 recall/scene_bounds", ["--qwen-scene-min-duration", "6"],
     "semantic min scene 12->6s (finer chunking)"),

    # ---- Round 2: map the non-monotonic threshold curve + combine H3+H4 ----
    ("R2_thr_0.10", "R2 threshold-curve", ["--qwen-vad-threshold", "0.10"], "threshold 0.10"),
    ("R2_thr_0.12", "R2 threshold-curve", ["--qwen-vad-threshold", "0.12"], "threshold 0.12"),
    ("R2_thr_0.18", "R2 threshold-curve", ["--qwen-vad-threshold", "0.18"], "threshold 0.18"),
    ("R2_thr0.15_end30", "R2 combine H3+H4",
     ["--qwen-vad-threshold", "0.15", "--qwen-vad-end-pad", "30"], "thr 0.15 + end pad 30"),
    ("R2_thr0.12_end30", "R2 combine H3+H4",
     ["--qwen-vad-threshold", "0.12", "--qwen-vad-end-pad", "30"], "thr 0.12 + end pad 30"),
    # Counter-test: RAISING max_speech (longer context) — does accuracy improve?
    ("R2_maxspeech_6", "R2 counter/max_speech", ["--qwen-max-speech-duration", "6"],
     "max_speech 6 (longer context, fewer/longer subs)"),

    # ---- Round 3: stack recall (low thr) + accuracy (long context + end pad) ----
    ("R3_thr0.12_ms6_end30", "R3 combine",
     ["--qwen-vad-threshold", "0.12", "--qwen-max-speech-duration", "6", "--qwen-vad-end-pad", "30"],
     "thr 0.12 + max_speech 6 + end pad 30"),
    ("R3_thr0.12_ms5_end30", "R3 combine",
     ["--qwen-vad-threshold", "0.12", "--qwen-max-speech-duration", "5", "--qwen-vad-end-pad", "30"],
     "thr 0.12 + max_speech 5 + end pad 30"),
    ("R3_thr0.15_ms6_end30", "R3 combine",
     ["--qwen-vad-threshold", "0.15", "--qwen-max-speech-duration", "6", "--qwen-vad-end-pad", "30"],
     "thr 0.15 + max_speech 6 + end pad 30"),
]


def build_command(python_exe: str, media: str, cfg_args, out_dir: str, tmp_dir: str):
    """Assemble the whisperjav CLI command for one config."""
    return [
        python_exe, "-m", "whisperjav.main", media,
        *BASE_ARGS, *cfg_args,
        "--output-dir", out_dir,
        "--temp-dir", tmp_dir,
        "--output-format", "srt",
    ]
