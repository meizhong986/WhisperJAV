"""
Sub provenance traceability and timing source analytics.

For each test subtitle, traces the full provenance chain:
  sub -> scene -> VAD group -> timing source -> GT match

Provides aggregate analytics: % by timing source, accuracy per source,
out-of-bounds and regression counts.
"""

from typing import Dict, List, Optional

from whisperjav.bench.loader import SceneBoundary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_scene_for_sub(
    sub_start: float,
    sub_end: float,
    scene_boundaries: List[SceneBoundary],
) -> Optional[int]:
    """
    Find the scene index whose [start_sec, end_sec) contains the sub's midpoint.

    Returns scene index or None if no scene contains the midpoint.
    """
    midpoint = (sub_start + sub_end) / 2
    for b in scene_boundaries:
        if b.start_sec <= midpoint < b.end_sec:
            return b.index
    return None


def _find_vad_group(
    sub_rel_start: float,
    sub_rel_end: float,
    vad_regions: List[dict],
) -> Optional[int]:
    """
    Find the VAD group index whose time range overlaps the sub's scene-relative range.

    VAD regions are expected as dicts with 'start' and 'end' keys (scene-relative seconds).
    Returns 0-based index or None if no overlap.
    """
    for i, region in enumerate(vad_regions):
        r_start = region.get("start", 0.0)
        r_end = region.get("end", 0.0)
        # Check overlap
        if sub_rel_start < r_end and sub_rel_end > r_start:
            return i
    return None


def _classify_scene_timing(scene_diag: Optional[dict]) -> tuple:
    """
    Classify a scene's timing source from its diagnostics.

    Returns:
        (timing_source: str, sentinel_status: str)

    timing_source is one of:
        "aligner", "sentinel_vad_guided", "sentinel_proportional",
        "mixed", "unknown"
    sentinel_status is one of:
        "OK", "COLLAPSED", "n/a"
    """
    if not scene_diag:
        return ("unknown", "n/a")

    sentinel = scene_diag.get("sentinel", {})
    status = sentinel.get("status", "n/a")
    recovery = sentinel.get("recovery")

    if status == "COLLAPSED" and recovery:
        strategy = recovery.get("strategy", "unknown")
        return (f"sentinel_{strategy}", "COLLAPSED")

    if status == "OK":
        ts = scene_diag.get("timing_sources", {})
        aligner_count = ts.get("aligner_native", 0)
        vad_fb = ts.get("vad_fallback", 0)
        interpolated = ts.get("interpolated", 0)

        if vad_fb > 0 or interpolated > 0:
            return ("mixed", "OK")
        if aligner_count > 0:
            return ("aligner", "OK")
        # Status OK but no timing source info
        return ("aligner", "OK")

    # Status exists but is neither OK nor COLLAPSED, or no sentinel key
    if status == "n/a":
        return ("unknown", "n/a")

    return ("unknown", status)


# ---------------------------------------------------------------------------
# Build provenance
# ---------------------------------------------------------------------------

def build_sub_provenance(
    test_subs: List[dict],
    scene_boundaries: List[SceneBoundary],
    scene_diagnostics: Dict[int, dict],
    gt_match_map: Dict[int, tuple],
    temporal_order: dict,
) -> List[dict]:
    """
    Build a provenance record for each test subtitle.

    Args:
        test_subs: Final SRT subs as dicts with 'start', 'end', 'text', 'index'.
        scene_boundaries: Scene boundary objects from loader.
        scene_diagnostics: Phase 2 per-scene diagnostics (scene_index -> dict).
        gt_match_map: Mapping of test sub index -> (gt_sub dict, IoU float).
        temporal_order: Result from analyze_temporal_order().

    Returns:
        List of provenance dicts, one per test subtitle.
    """
    # Build lookup sets for regressions and overlaps by sub index
    regression_indices = set()
    for r in temporal_order.get("regressions", []):
        regression_indices.add(r["curr_index"])

    overlap_indices = set()
    for o in temporal_order.get("overlaps", []):
        overlap_indices.add(o["curr_index"])

    # Build scene boundary lookup for quick access
    scene_map = {b.index: b for b in scene_boundaries}

    provenances = []
    for sub in test_subs:
        sub_start = sub["start"]
        sub_end = sub["end"]
        sub_index = sub["index"]
        sub_duration = sub_end - sub_start

        # --- Scene attribution ---
        scene_idx = _find_scene_for_sub(sub_start, sub_end, scene_boundaries)

        scene_start = 0.0
        scene_end = 0.0
        scene_rel_start = 0.0
        scene_rel_end = 0.0
        out_of_bounds = False

        if scene_idx is not None and scene_idx in scene_map:
            boundary = scene_map[scene_idx]
            scene_start = boundary.start_sec
            scene_end = boundary.end_sec
            scene_duration = boundary.duration_sec
            scene_rel_start = sub_start - scene_start
            scene_rel_end = sub_end - scene_start

            # Out of bounds: scene-relative times violate [0, scene_duration]
            out_of_bounds = (scene_rel_start < -0.05 or scene_rel_end > scene_duration + 0.05)

        # --- VAD group attribution ---
        vad_group_index = None
        vad_group_start = None
        vad_group_end = None

        if scene_idx is not None:
            diag = scene_diagnostics.get(scene_idx)
            if diag:
                vad_regions = diag.get("vad_regions", [])
                if vad_regions:
                    vad_group_index = _find_vad_group(
                        scene_rel_start, scene_rel_end, vad_regions,
                    )
                    if vad_group_index is not None and vad_group_index < len(vad_regions):
                        vad_group_start = vad_regions[vad_group_index].get("start")
                        vad_group_end = vad_regions[vad_group_index].get("end")

        # --- Timing source classification ---
        scene_diag = scene_diagnostics.get(scene_idx) if scene_idx is not None else None
        timing_source, sentinel_status = _classify_scene_timing(scene_diag)

        # --- GT matching ---
        gt_match = gt_match_map.get(sub_index)
        gt_match_index = None
        gt_iou = None
        if gt_match is not None:
            gt_sub, iou = gt_match
            gt_match_index = gt_sub.get("index")
            gt_iou = iou

        # --- Flags ---
        has_regression = sub_index in regression_indices
        has_overlap = sub_index in overlap_indices

        provenances.append({
            "sub_index": sub_index,
            "text": sub["text"],
            "start": sub_start,
            "end": sub_end,
            "duration": round(sub_duration, 3),

            # Scene attribution
            "scene_index": scene_idx,
            "scene_start": round(scene_start, 3),
            "scene_end": round(scene_end, 3),
            "scene_relative_start": round(scene_rel_start, 3),
            "scene_relative_end": round(scene_rel_end, 3),

            # VAD group
            "vad_group_index": vad_group_index,
            "vad_group_start": vad_group_start,
            "vad_group_end": vad_group_end,

            # Timing source
            "timing_source": timing_source,
            "sentinel_status": sentinel_status,

            # GT matching
            "gt_match_index": gt_match_index,
            "gt_iou": gt_iou,

            # Flags
            "has_regression": has_regression,
            "has_overlap": has_overlap,
            "out_of_scene_bounds": out_of_bounds,
        })

    return provenances


# ---------------------------------------------------------------------------
# Timing source analytics
# ---------------------------------------------------------------------------

def compute_timing_source_analytics(
    provenances: List[dict],
    iou_good: float = 0.7,
    iou_acceptable: float = 0.5,
) -> dict:
    """
    Compute aggregate statistics grouped by timing source.

    Args:
        provenances: List of provenance dicts from build_sub_provenance().
        iou_good: IoU threshold for "good" timing (>= this).
        iou_acceptable: IoU threshold for "acceptable" timing (>= this).

    Returns:
        Dict with total counts, per-source stats, and flag counts.
    """
    total_subs = len(provenances)
    total_matched = sum(1 for p in provenances if p["gt_iou"] is not None)

    # Group by timing source
    by_source: Dict[str, list] = {}
    for p in provenances:
        source = p["timing_source"]
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(p)

    source_stats = {}
    for source, group in sorted(by_source.items()):
        count = len(group)
        matched = [p for p in group if p["gt_iou"] is not None]
        matched_count = len(matched)

        if matched:
            ious = [p["gt_iou"] for p in matched]
            mean_iou = sum(ious) / len(ious)
            good_count = sum(1 for iou in ious if iou >= iou_good)
            acceptable_count = sum(1 for iou in ious if iou >= iou_acceptable)
            good_pct = good_count / matched_count * 100
            acceptable_pct = acceptable_count / matched_count * 100
        else:
            mean_iou = None
            good_pct = 0.0
            acceptable_pct = 0.0

        source_stats[source] = {
            "count": count,
            "pct": count / total_subs * 100 if total_subs > 0 else 0.0,
            "matched_count": matched_count,
            "mean_iou": round(mean_iou, 3) if mean_iou is not None else None,
            "good_pct": round(good_pct, 1),
            "acceptable_pct": round(acceptable_pct, 1),
        }

    return {
        "total_subs": total_subs,
        "total_matched": total_matched,
        "by_timing_source": source_stats,
        "out_of_bounds_count": sum(1 for p in provenances if p["out_of_scene_bounds"]),
        "regression_count": sum(1 for p in provenances if p["has_regression"]),
        "overlap_count": sum(1 for p in provenances if p["has_overlap"]),
    }
