"""MEASUREMENTS section 2.12 (1.9.3 defaults against the final 1.9.4 code, the release-note timing figures) scored
again with subeval, to see whether the published conclusions hold under the corrected method.
Usage: python recheck_212_subeval.py <measure folder>   (the folder holding req2_baseline, req2_confirm, step2)"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from subeval.cli import pair_folder, score_files  # noqa: E402
from subeval.compare import compare  # noqa: E402
from subeval.evaluate import summarize  # noqa: E402

REF = REPO / "test_media" / "Ground_Truths" / "Netflix"
M = Path(sys.argv[1])
ROWS = [("Qwen3-ASR balanced", "req2_baseline/qwen3_whisperseg/run1", "step2/f_q/run1"),
        ("anime-whisper conservative", "req2_confirm/a_cons_old6/run1", "step2/f_ac2/run1"),
        ("anime-whisper balanced", "req2_confirm/a_bal_old5/run1", "step2/f_ab/run1"),
        ("anime-whisper aggressive", "req2_baseline/anime_whisperseg/run1", "step2/f_aa/run1"),
        # Qwen3-ASR + WhisperSeg end pad 0 at every sensitivity (owner, 2026-10-09): the final balanced setting.
        ("Qwen3-ASR balanced, end pad 0", "req2_baseline/qwen3_whisperseg/run1",
         "req2_e_check/qwen3_whisperseg_balanced__endpad0/run1"),
        ("Qwen3-ASR balanced, end pad 100 → 0", "step2/f_q/run1", "req2_e_check/qwen3_whisperseg_balanced__endpad0/run1")]
print("| Model, sensitivity | End median, both measured (s) | Ends ≤ 0.5 s, both | Start median, both (s) "
      "| End mean (s) | CER plain | CER | Missed lines | Halluc. lines | Δ edits (clips better/worse) "
      "| End clips better/worse |")
print("|---" * 11 + "|")
for label, b, c in ROWS:
    rb, _ = score_files(pair_folder(REF, M / b))
    rc, _ = score_files(pair_folder(REF, M / c))
    sb, sc, k = summarize(rb), summarize(rc), compare(rb, rc)
    print(f"| {label} | {k['end_med_base']:.3f} → {k['end_med_cand']:.3f} (n {k['end_n_both']}) "
          f"| {k['end_within_base']:.0%} → {k['end_within_cand']:.0%} "
          f"| {k['start_med_base']:.3f} → {k['start_med_cand']:.3f} "
          f"| {sb['end_signed_mean']:+.3f} → {sc['end_signed_mean']:+.3f} "
          f"| {sb['cer_plain']:.3f} → {sc['cer_plain']:.3f} | {sb['cer']:.3f} → {sc['cer']:.3f} "
          f"| {sb['missed_lines']} → {sc['missed_lines']} | {sb['hallucination_lines']} → {sc['hallucination_lines']} "
          f"| {k['d_edits']:+d} ({k['text_clips_better']}/{k['text_clips_worse']}) "
          f"| {k['end_clips_better']}/{k['end_clips_worse']} |", flush=True)
