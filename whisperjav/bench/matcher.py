"""
Subtitle matching: align test subtitles to ground-truth subtitles.

Uses temporal overlap + text similarity to find the best match for each GT
subtitle. Produces matched pairs, missed GT subs, and hallucinated test subs.
"""

import difflib
from typing import Dict, List

from whisperjav.bench.metrics import normalize_text


# ---------------------------------------------------------------------------
# Subtitle matching
# ---------------------------------------------------------------------------

def match_subtitles(
    gt_subs: List[dict],
    test_subs: List[dict],
    min_overlap_sec: float = 0.1,
    min_text_similarity: float = 0.2,
) -> Dict[str, list]:
    """
    Match test subtitles to ground-truth subtitles.

    Algorithm:
        1. For each GT subtitle, find test subtitles with temporal overlap
        2. Among overlapping candidates, pick highest text similarity
        3. Each test sub can only match one GT sub (greedy, GT-order)

    Args:
        gt_subs: Ground truth subtitles, each with 'start', 'end', 'text'.
        test_subs: Test subtitles, same format.
        min_overlap_sec: Minimum temporal overlap to consider a candidate.
        min_text_similarity: Minimum SequenceMatcher ratio to accept a match.

    Returns:
        Dict with keys:
            matched: List of (gt_sub, test_sub) tuples
            missed: List of gt_subs with no match
            hallucinated: List of test_subs with no match
    """
    used_test_indices = set()
    matched = []
    missed = []

    for gt in gt_subs:
        gt_start = gt["start"]
        gt_end = gt["end"]
        gt_text = normalize_text(gt["text"])

        best_match = None
        best_score = -1.0
        best_idx = -1

        for t_idx, test in enumerate(test_subs):
            if t_idx in used_test_indices:
                continue

            # Check temporal overlap
            overlap_start = max(gt_start, test["start"])
            overlap_end = min(gt_end, test["end"])
            overlap = overlap_end - overlap_start

            if overlap < min_overlap_sec:
                continue

            # Text similarity
            test_text = normalize_text(test["text"])
            similarity = difflib.SequenceMatcher(
                None, gt_text, test_text,
            ).ratio()

            if similarity > best_score:
                best_score = similarity
                best_match = test
                best_idx = t_idx

        if best_match is not None and best_score >= min_text_similarity:
            matched.append((gt, best_match))
            used_test_indices.add(best_idx)
        else:
            missed.append(gt)

    # Hallucinated: test subs that weren't matched to any GT
    hallucinated = [
        test_subs[i] for i in range(len(test_subs))
        if i not in used_test_indices
    ]

    return {
        "matched": matched,
        "missed": missed,
        "hallucinated": hallucinated,
    }


def match_subtitles_by_scene(
    gt_subs: List[dict],
    test_subs: List[dict],
    scene_boundaries: List[dict],
) -> Dict[int, Dict[str, list]]:
    """
    Match subtitles within each scene's time range.

    Args:
        gt_subs: All ground-truth subtitles.
        test_subs: All test subtitles.
        scene_boundaries: List of dicts with 'start_sec' and 'end_sec'.

    Returns:
        Dict mapping scene_index to match_subtitles() result.
    """
    results = {}

    for idx, scene in enumerate(scene_boundaries):
        scene_start = scene["start_sec"]
        scene_end = scene["end_sec"]

        # Filter subs within this scene's time range
        # A subtitle belongs to a scene if its midpoint falls within the range
        scene_gt = [
            s for s in gt_subs
            if _midpoint(s) >= scene_start and _midpoint(s) < scene_end
        ]
        scene_test = [
            s for s in test_subs
            if _midpoint(s) >= scene_start and _midpoint(s) < scene_end
        ]

        results[idx] = match_subtitles(scene_gt, scene_test)

    return results


def _midpoint(sub: dict) -> float:
    """Midpoint of a subtitle's time range."""
    return (sub["start"] + sub["end"]) / 2
