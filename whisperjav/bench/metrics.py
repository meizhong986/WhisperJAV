"""
CER (Character Error Rate) and timing accuracy metrics for benchmarking.

CER is computed at character level using edit distance, suitable for Japanese
where word boundaries are ambiguous.

Timing accuracy uses IoU (Intersection over Union) of matched subtitle time ranges.
"""

import unicodedata
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Text normalization (pre-CER)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for CER comparison.

    - Strip whitespace
    - Normalize fullwidth/halfwidth forms (NFKC)
    - Remove common punctuation that doesn't affect meaning
    """
    text = unicodedata.normalize("NFKC", text)
    # Remove whitespace
    text = "".join(text.split())
    # Remove punctuation that varies between transcription systems
    remove_chars = set("。、！？「」『』（）()…・〜～.,!?\"' ")
    text = "".join(c for c in text if c not in remove_chars)
    return text


# ---------------------------------------------------------------------------
# Edit distance (character-level Levenshtein)
# ---------------------------------------------------------------------------

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, deletion, substitution
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# ---------------------------------------------------------------------------
# CER computation
# ---------------------------------------------------------------------------

def compute_cer(hypothesis: str, reference: str) -> float:
    """
    Compute Character Error Rate.

    CER = edit_distance(hyp, ref) / len(ref)

    Both strings are normalized before comparison.
    Returns 0.0 if reference is empty (nothing to compare against).
    """
    hyp = normalize_text(hypothesis)
    ref = normalize_text(reference)

    if not ref:
        return 0.0 if not hyp else 1.0

    distance = _levenshtein_distance(hyp, ref)
    return distance / len(ref)


def compute_cer_from_segments(
    hyp_texts: List[str],
    ref_texts: List[str],
) -> float:
    """
    Compute global CER by concatenating all texts in order.

    This avoids alignment artifacts from per-subtitle CER averaging.
    """
    hyp_combined = "".join(hyp_texts)
    ref_combined = "".join(ref_texts)
    return compute_cer(hyp_combined, ref_combined)


# ---------------------------------------------------------------------------
# Timing accuracy (IoU)
# ---------------------------------------------------------------------------

def compute_iou(
    start1: float, end1: float,
    start2: float, end2: float,
) -> float:
    """
    Compute Intersection over Union for two time ranges.

    Returns 0.0 if no overlap, 1.0 for perfect alignment.
    """
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0.0, intersection_end - intersection_start)

    union = max(end1, end2) - min(start1, start2)
    if union <= 0:
        return 0.0

    return intersection / union


def compute_timing_score(
    matched_pairs: List[Tuple[dict, dict]],
) -> float:
    """
    Compute mean IoU across matched subtitle pairs.

    Args:
        matched_pairs: List of (gt_sub, test_sub) dicts, each with
                       'start' and 'end' keys (seconds as float).

    Returns:
        Mean IoU (0.0 to 1.0). Returns 0.0 if no pairs.
    """
    if not matched_pairs:
        return 0.0

    total_iou = 0.0
    for gt, test in matched_pairs:
        iou = compute_iou(gt["start"], gt["end"], test["start"], test["end"])
        total_iou += iou

    return total_iou / len(matched_pairs)


def compute_timing_offsets(
    matched_pairs: List[Tuple[dict, dict]],
) -> dict:
    """
    Compute timing offset statistics for matched pairs.

    Returns dict with:
        start_offset_mean_ms: mean(test_start - gt_start)
        end_offset_mean_ms: mean(test_end - gt_end)
        start_offset_abs_mean_ms: mean(|test_start - gt_start|)
        end_offset_abs_mean_ms: mean(|test_end - gt_end|)
    """
    if not matched_pairs:
        return {
            "start_offset_mean_ms": 0.0,
            "end_offset_mean_ms": 0.0,
            "start_offset_abs_mean_ms": 0.0,
            "end_offset_abs_mean_ms": 0.0,
        }

    start_offsets = []
    end_offsets = []

    for gt, test in matched_pairs:
        start_offsets.append((test["start"] - gt["start"]) * 1000)
        end_offsets.append((test["end"] - gt["end"]) * 1000)

    n = len(matched_pairs)
    return {
        "start_offset_mean_ms": sum(start_offsets) / n,
        "end_offset_mean_ms": sum(end_offsets) / n,
        "start_offset_abs_mean_ms": sum(abs(o) for o in start_offsets) / n,
        "end_offset_abs_mean_ms": sum(abs(o) for o in end_offsets) / n,
    }
