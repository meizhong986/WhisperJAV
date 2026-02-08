"""
Report generation for Qwen pipeline benchmarking.

Produces:
- Console summary table (CER, timing IoU, subtitle counts, processing time)
- Worst-scenes ranking
- Pipeline events summary (sentinel collapses, sanitization)
- Per-scene drill-down with matched subtitles and word timestamps
- JSON report for programmatic consumption
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

from whisperjav.bench.loader import SubtitleEntry, SceneBoundary, TestResult
from whisperjav.bench.matcher import match_subtitles, match_subtitles_by_scene
from whisperjav.bench.metrics import (
    analyze_temporal_order,
    compute_cer_from_segments,
    compute_iou,
    compute_timing_offsets,
    compute_timing_score,
)
from whisperjav.bench.provenance import (
    build_sub_provenance,
    compute_timing_source_analytics,
)


# ---------------------------------------------------------------------------
# Analysis: compute all metrics
# ---------------------------------------------------------------------------

def _subs_to_dicts(subs: List[SubtitleEntry]) -> List[dict]:
    """Convert SubtitleEntry list to the dict format expected by matcher."""
    return [
        {"start": s.start, "end": s.end, "text": s.text, "index": s.index}
        for s in subs
    ]


def _scene_boundaries_to_dicts(boundaries: List[SceneBoundary]) -> List[dict]:
    """Convert SceneBoundary list to dict format for matcher."""
    return [
        {"start_sec": b.start_sec, "end_sec": b.end_sec}
        for b in boundaries
    ]


def analyze(
    gt_subs: List[SubtitleEntry],
    tests: List[TestResult],
) -> dict:
    """
    Run full analysis: matching, CER, timing for all tests.

    Returns a structured dict with:
        ground_truth: {subtitle_count, duration_sec}
        tests: [{name, input_mode, cer, timing_iou, ...}, ...]
        per_scene: [{scene_index, boundaries, tests: [{name, cer, timing_iou}, ...]}, ...]
        worst_scenes: [(scene_index, max_cer), ...] sorted worst-first
    """
    gt_dicts = _subs_to_dicts(gt_subs)
    gt_duration = gt_subs[-1].end if gt_subs else 0.0

    # Use the first test with scene boundaries for per-scene analysis
    scene_boundaries = []
    for t in tests:
        if t.scene_boundaries:
            scene_boundaries = t.scene_boundaries
            break
    scene_boundary_dicts = _scene_boundaries_to_dicts(scene_boundaries)

    result = {
        "ground_truth": {
            "subtitle_count": len(gt_subs),
            "duration_sec": round(gt_duration, 1),
        },
        "tests": [],
        "per_scene": [],
        "worst_scenes": [],
    }

    # Per-test analysis
    test_analyses = []
    for t in tests:
        test_dicts = _subs_to_dicts(t.final_srt)

        # Global matching + metrics
        match_result = match_subtitles(gt_dicts, test_dicts)
        matched = match_result["matched"]

        # CER: concatenate matched text in time order
        gt_texts = [pair[0]["text"] for pair in matched]
        test_texts = [pair[1]["text"] for pair in matched]
        global_cer = compute_cer_from_segments(test_texts, gt_texts)

        # Timing
        timing_iou = compute_timing_score(matched)
        timing_offsets = compute_timing_offsets(matched)

        # Pipeline events from master metadata
        stages = t.master_metadata.get("stages", {})
        asr_stage = stages.get("asr", {})
        san_stage = stages.get("sanitisation", {})
        san_stats = san_stage.get("stats", {})

        # Temporal ordering integrity
        temporal_order = analyze_temporal_order(test_dicts)

        test_analysis = {
            "name": t.name,
            "input_mode": t.input_mode,
            "model_id": t.model_id,
            "cer": global_cer,
            "timing_iou": timing_iou,
            "timing_offsets": timing_offsets,
            "temporal_order": temporal_order,
            "subtitle_count": len(t.final_srt),
            "matched_count": len(matched),
            "missed_count": len(match_result["missed"]),
            "hallucinated_count": len(match_result["hallucinated"]),
            "processing_time_sec": t.processing_time_sec,
            "sentinel": {
                "collapses": asr_stage.get("alignment_collapses", 0),
                "recoveries": asr_stage.get("alignment_recoveries", 0),
            },
            "sanitization": {
                "hallucinations_removed": san_stats.get("removed_hallucinations", 0),
                "repetitions_removed": san_stats.get("removed_repetitions", 0),
                "cps_filtered": san_stats.get("cps_filtered", 0),
                "empty_removed": san_stats.get("empty_removed", 0),
            },
        }

        # Per-scene matching + metrics
        scene_metrics = {}
        if scene_boundary_dicts:
            scene_matches = match_subtitles_by_scene(
                gt_dicts, test_dicts, scene_boundary_dicts,
            )
            for s_idx, s_match in scene_matches.items():
                s_matched = s_match["matched"]
                s_gt_texts = [p[0]["text"] for p in s_matched]
                s_test_texts = [p[1]["text"] for p in s_matched]
                s_cer = compute_cer_from_segments(s_test_texts, s_gt_texts) if s_gt_texts else None
                s_iou = compute_timing_score(s_matched) if s_matched else None

                scene_metrics[s_idx] = {
                    "cer": s_cer,
                    "timing_iou": s_iou,
                    "matched": len(s_matched),
                    "missed": len(s_match["missed"]),
                    "hallucinated": len(s_match["hallucinated"]),
                    "match_detail": s_match,  # Keep for drill-down
                }

        test_analysis["scene_metrics"] = scene_metrics

        # --- Sub provenance traceability ---
        # Build GT match map: test sub index -> (gt_sub, iou)
        gt_match_map = {}
        for gt_sub, test_sub in matched:
            iou = compute_iou(
                gt_sub["start"], gt_sub["end"],
                test_sub["start"], test_sub["end"],
            )
            gt_match_map[test_sub["index"]] = (gt_sub, iou)

        provenances = build_sub_provenance(
            test_dicts,
            t.scene_boundaries,
            t.scene_diagnostics,
            gt_match_map,
            temporal_order,
        )
        analytics = compute_timing_source_analytics(provenances)

        test_analysis["provenance"] = provenances
        test_analysis["timing_analytics"] = analytics

        test_analyses.append(test_analysis)

    result["tests"] = test_analyses

    # Build per-scene cross-test view
    for s_idx, boundary in enumerate(scene_boundaries):
        scene_entry = {
            "scene_index": s_idx,
            "start_sec": boundary.start_sec,
            "end_sec": boundary.end_sec,
            "duration_sec": boundary.duration_sec,
            "tests": [],
        }
        for ta in test_analyses:
            sm = ta["scene_metrics"].get(s_idx)
            if sm:
                scene_entry["tests"].append({
                    "name": ta["name"],
                    "cer": sm["cer"],
                    "timing_iou": sm["timing_iou"],
                })
        result["per_scene"].append(scene_entry)

    # Worst scenes: sort by max CER across tests (scenes where any test struggles)
    worst = []
    for scene_entry in result["per_scene"]:
        cers = [
            t["cer"] for t in scene_entry["tests"]
            if t["cer"] is not None
        ]
        if cers:
            worst.append((scene_entry["scene_index"], max(cers)))

    worst.sort(key=lambda x: x[1], reverse=True)
    result["worst_scenes"] = worst[:10]  # Top 10

    return result


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def _fmt_time(sec: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    if sec < 0:
        return "0:00"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _fmt_time_precise(sec: float) -> str:
    """Format seconds as HH:MM:SS.fff."""
    if sec < 0:
        return "00:00:00.000"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _fmt_pct(val: Optional[float]) -> str:
    """Format a ratio as percentage string."""
    if val is None:
        return "  n/a "
    return f"{val * 100:5.1f}%"


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_summary(
    analysis: dict,
    benchmark_name: str = "",
    file=None,
):
    """Print the console summary table."""
    out = file or sys.stdout
    gt = analysis["ground_truth"]
    tests = analysis["tests"]

    # Header
    title = "Qwen Pipeline Benchmark"
    if benchmark_name:
        title += f": {benchmark_name}"
    print(f"\n{title}", file=out)
    print(
        f"Ground Truth: {gt['subtitle_count']} subtitles, "
        f"{gt['duration_sec']}s",
        file=out,
    )

    # Summary table
    sep = "=" * 78
    print(sep, file=out)

    # Column headers
    header = (
        f"{'Test':<20} {'Mode':<14} {'CER':>6} {'Timing':>7} "
        f"{'Subs':>5} {'Match':>5} {'Time':>6}"
    )
    print(header, file=out)
    print("-" * 78, file=out)

    for t in tests:
        name = t["name"][:19]
        mode = t["input_mode"][:13]
        cer = _fmt_pct(t["cer"])
        timing = _fmt_pct(t["timing_iou"])
        subs = str(t["subtitle_count"])
        matched = str(t["matched_count"])
        time_str = f"{t['processing_time_sec']:.0f}s" if t["processing_time_sec"] else "n/a"

        print(
            f"{name:<20} {mode:<14} {cer:>6} {timing:>7} "
            f"{subs:>5} {matched:>5} {time_str:>6}",
            file=out,
        )

    print(sep, file=out)

    # Worst scenes
    worst = analysis.get("worst_scenes", [])
    per_scene = {s["scene_index"]: s for s in analysis.get("per_scene", [])}
    if worst:
        print("\nWorst scenes by CER:", file=out)
        for s_idx, max_cer in worst[:5]:
            scene = per_scene.get(s_idx, {})
            start = scene.get("start_sec", 0)
            end = scene.get("end_sec", 0)

            # Per-test CER for this scene
            test_cers = []
            for t_entry in scene.get("tests", []):
                label = t_entry["name"].split(":")[0] if ":" in t_entry["name"] else t_entry["name"][:6]
                cer_val = t_entry.get("cer")
                if cer_val is not None:
                    test_cers.append(f"{label}={cer_val*100:.1f}%")

            print(
                f"  Scene {s_idx} ({_fmt_time(start)}-{_fmt_time(end)}): "
                + " ".join(test_cers),
                file=out,
            )

    # Temporal ordering integrity
    any_order_issues = any(
        not t["temporal_order"]["is_monotonic"] or t["temporal_order"]["overlap_count"] > 0
        for t in tests
    )
    if any_order_issues:
        print("\nTemporal ordering:", file=out)
        for t in tests:
            to = t["temporal_order"]
            issues = []
            if not to["is_monotonic"]:
                issues.append(
                    f"{to['regression_count']} regressions "
                    f"(max {to['max_regression_sec']:.1f}s back)"
                )
            if to["overlap_count"] > 0:
                issues.append(
                    f"{to['overlap_count']} overlaps "
                    f"({to['total_overlap_sec']:.1f}s total)"
                )
            if issues:
                print(f"  {t['name']}: {', '.join(issues)}", file=out)
                # Show each regression detail
                for reg in to["regressions"]:
                    print(
                        f"    Sub #{reg['prev_index']} ({_fmt_time_precise(reg['prev_start'])}) "
                        f"-> #{reg['curr_index']} ({_fmt_time_precise(reg['curr_start'])}): "
                        f"-{reg['regression_sec']:.1f}s",
                        file=out,
                    )
            else:
                print(f"  {t['name']}: OK", file=out)

    # Pipeline events
    print("\nPipeline events:", file=out)
    for t in tests:
        events = []
        col = t["sentinel"]["collapses"]
        rec = t["sentinel"]["recoveries"]
        if col > 0:
            events.append(f"{col} collapses, {rec} recovered")

        san = t["sanitization"]
        hall = san["hallucinations_removed"]
        rep = san["repetitions_removed"]
        if hall > 0:
            events.append(f"{hall} hallucinations removed")
        if rep > 0:
            events.append(f"{rep} repetitions removed")

        if events:
            print(f"  {t['name']}: {', '.join(events)}", file=out)
        else:
            print(f"  {t['name']}: (clean run)", file=out)

    print(file=out)


# ---------------------------------------------------------------------------
# Timing source analytics
# ---------------------------------------------------------------------------

def print_timing_analytics(
    analysis: dict,
    file=None,
):
    """Print timing source analytics for each test."""
    out = file or sys.stdout
    tests = analysis.get("tests", [])

    has_analytics = any(t.get("timing_analytics") for t in tests)
    if not has_analytics:
        return

    print("Timing Source Analytics:", file=out)
    for t in tests:
        analytics = t.get("timing_analytics")
        if not analytics:
            continue

        total = analytics["total_subs"]
        matched = analytics["total_matched"]
        print(
            f"  {t['name']}: {t['input_mode']} "
            f"({total} subs, {matched} matched)",
            file=out,
        )

        for source, stats in sorted(analytics["by_timing_source"].items()):
            count = stats["count"]
            pct = stats["pct"]
            mean_iou = stats["mean_iou"]
            good = stats["good_pct"]
            acceptable = stats["acceptable_pct"]

            iou_str = f"{mean_iou:.2f}" if mean_iou is not None else " n/a"
            print(
                f"    {source:<22} {count:3d} ({pct:4.1f}%) "
                f"| IoU: {iou_str} "
                f"| Good: {good:5.1f}% "
                f"| Accept: {acceptable:5.1f}%",
                file=out,
            )

        oob = analytics["out_of_bounds_count"]
        reg = analytics["regression_count"]
        ovlp = analytics["overlap_count"]
        if oob or reg or ovlp:
            parts = []
            if oob:
                parts.append(f"Out of bounds: {oob}")
            if reg:
                parts.append(f"Regressions: {reg}")
            if ovlp:
                parts.append(f"Overlaps: {ovlp}")
            print(f"    {' | '.join(parts)}", file=out)

    print(file=out)


# ---------------------------------------------------------------------------
# Traceability table
# ---------------------------------------------------------------------------

def print_traceability_table(
    analysis: dict,
    test_index: int,
    file=None,
):
    """
    Print full per-sub provenance traceability table for a single test.

    Args:
        analysis: Full analysis dict from analyze().
        test_index: 0-based index into analysis["tests"].
        file: Output stream (default: stdout).
    """
    out = file or sys.stdout
    tests = analysis.get("tests", [])
    if test_index < 0 or test_index >= len(tests):
        print(f"Test index {test_index} out of range.", file=out)
        return

    t = tests[test_index]
    provenances = t.get("provenance", [])
    if not provenances:
        print(f"No provenance data for {t['name']}.", file=out)
        return

    print(f"\nSub Provenance: {t['name']}: {t['input_mode']}", file=out)

    header = (
        f"{'#':>4}  {'Start->End':<19} {'Dur':>5}  "
        f"{'Scn':>3}  {'SceneRange':<21} {'Rel.Start':>9}  "
        f"{'VAD':>3}  {'Source':<22} {'GT#':>4}  {'IoU':>5}  {'Flags'}"
    )
    sep_line = "=" * len(header)
    print(sep_line, file=out)
    print(header, file=out)
    print(sep_line, file=out)

    for p in provenances:
        idx = p["sub_index"]
        start_str = _fmt_time_precise(p["start"])[3:]  # Skip leading "00:" for MM:SS.fff
        end_str = _fmt_time_precise(p["end"])[3:]
        time_range = f"{start_str}->{end_str}"
        dur_str = f"{p['duration']:.1f}s"

        scn = f"{p['scene_index']}" if p["scene_index"] is not None else "?"
        if p["scene_index"] is not None:
            scene_range = (
                f"{_fmt_time_precise(p['scene_start'])[3:]}->"
                f"{_fmt_time_precise(p['scene_end'])[3:]}"
            )
        else:
            scene_range = "?"

        rel_start = f"{p['scene_relative_start']:.1f}s" if p["scene_index"] is not None else "?"

        vad = f"{p['vad_group_index']}" if p["vad_group_index"] is not None else "-"
        source = p["timing_source"]

        gt_idx = f"{p['gt_match_index']}" if p["gt_match_index"] is not None else "-"
        iou_str = f"{p['gt_iou']:.2f}" if p["gt_iou"] is not None else "  -  "

        # Flags
        flags = []
        if p["has_regression"]:
            flags.append("REGR")
        if p["out_of_scene_bounds"]:
            flags.append("OOB")
        if p["has_overlap"]:
            flags.append("OVLP")
        flag_str = ",".join(flags)

        print(
            f"{idx:>4}  {time_range:<19} {dur_str:>5}  "
            f"{scn:>3}  {scene_range:<21} {rel_start:>9}  "
            f"{vad:>3}  {source:<22} {gt_idx:>4}  {iou_str:>5}  {flag_str}",
            file=out,
        )

    print(sep_line, file=out)

    # Summary line
    total = len(provenances)
    reg_count = sum(1 for p in provenances if p["has_regression"])
    oob_count = sum(1 for p in provenances if p["out_of_scene_bounds"])
    ovlp_count = sum(1 for p in provenances if p["has_overlap"])
    print(
        f"  {total} subs | {reg_count} regressions | "
        f"{oob_count} out-of-bounds | {ovlp_count} overlaps",
        file=out,
    )
    print(file=out)


# ---------------------------------------------------------------------------
# Scene drill-down
# ---------------------------------------------------------------------------

def print_scene_detail(
    analysis: dict,
    scene_index: int,
    gt_subs: List[SubtitleEntry],
    tests: List[TestResult],
    file=None,
):
    """Print detailed drill-down for a specific scene."""
    out = file or sys.stdout
    per_scene = analysis.get("per_scene", [])

    # Find scene info
    scene_info = None
    for s in per_scene:
        if s["scene_index"] == scene_index:
            scene_info = s
            break

    if scene_info is None:
        print(f"Scene {scene_index} not found in analysis.", file=out)
        return

    start = scene_info["start_sec"]
    end = scene_info["end_sec"]
    dur = scene_info["duration_sec"]

    print(
        f"\nScene {scene_index} ({_fmt_time_precise(start)} - "
        f"{_fmt_time_precise(end)}, {dur:.1f}s)",
        file=out,
    )
    print("-" * 60, file=out)

    # Ground truth subtitles in this scene
    gt_in_scene = [
        s for s in gt_subs
        if (s.start + s.end) / 2 >= start and (s.start + s.end) / 2 < end
    ]
    print(f"Ground Truth ({len(gt_in_scene)} subtitles):", file=out)
    for s in gt_in_scene[:10]:  # Show first 10
        text_preview = s.text[:50]
        suffix = "..." if len(s.text) > 50 else ""
        print(
            f"  {s.index:3d}. [{_fmt_time(s.start)}-{_fmt_time(s.end)}] "
            f"{text_preview}{suffix}",
            file=out,
        )
    if len(gt_in_scene) > 10:
        print(f"  ... ({len(gt_in_scene) - 10} more)", file=out)

    # Per-test detail
    for t_idx, t in enumerate(tests):
        test_analysis = analysis["tests"][t_idx]
        sm = test_analysis["scene_metrics"].get(scene_index)
        if sm is None:
            continue

        cer_str = _fmt_pct(sm["cer"])
        iou_str = _fmt_pct(sm["timing_iou"])

        print(
            f"\n{t.name} (CER={cer_str.strip()}, Timing={iou_str.strip()})",
            file=out,
        )

        # Sentinel info from diagnostics (Phase 2)
        diag = t.scene_diagnostics.get(scene_index)
        if diag:
            sentinel = diag.get("sentinel", {})
            status = sentinel.get("status", "n/a")
            assessment = sentinel.get("assessment", {})
            recovery = sentinel.get("recovery")

            if status == "COLLAPSED" and recovery:
                strategy = recovery.get("strategy", "unknown")
                cov = assessment.get("coverage_ratio", 0)
                cps = assessment.get("aggregate_cps", 0)
                span = assessment.get("word_span_sec", 0)
                print(
                    f"  Sentinel: COLLAPSED -> recovered ({strategy})",
                    file=out,
                )
                print(
                    f"    coverage={cov:.3f}, CPS={cps:.1f}, span={span:.2f}s",
                    file=out,
                )
            elif status == "OK":
                print("  Sentinel: OK", file=out)

            # Timing sources
            ts = diag.get("timing_sources")
            if ts:
                parts = []
                for key in ("aligner_native", "vad_fallback", "interpolated"):
                    val = ts.get(key, 0)
                    if val > 0:
                        parts.append(f"{key}={val}")
                if parts:
                    print(f"  Timing sources: {', '.join(parts)}", file=out)

        # Temporal ordering issues in this scene
        to = test_analysis.get("temporal_order", {})
        scene_regs = [
            r for r in to.get("regressions", [])
            if (r["curr_start"] >= start and r["curr_start"] < end)
            or (r["prev_start"] >= start and r["prev_start"] < end)
        ]
        scene_overlaps = [
            o for o in to.get("overlaps", [])
            if (o["curr_start"] >= start and o["curr_start"] < end)
            or (o["prev_start"] >= start and o["prev_start"] < end)
        ]
        if scene_regs or scene_overlaps:
            print("  Temporal ordering issues:", file=out)
            for r in scene_regs:
                print(
                    f"    REGRESSION: #{r['prev_index']} "
                    f"({_fmt_time_precise(r['prev_start'])}) -> "
                    f"#{r['curr_index']} ({_fmt_time_precise(r['curr_start'])}): "
                    f"-{r['regression_sec']:.1f}s",
                    file=out,
                )
            for o in scene_overlaps:
                print(
                    f"    Overlap: #{o['prev_index']} -> #{o['curr_index']}: "
                    f"{o['overlap_sec']:.3f}s",
                    file=out,
                )

        # Raw / clean text
        raw = t.raw_texts.get(scene_index)
        if raw:
            preview = raw[:80].replace("\n", " ")
            suffix = "..." if len(raw) > 80 else ""
            print(f"  Raw text: {preview}{suffix}", file=out)

        clean = t.clean_texts.get(scene_index)
        if clean and clean != raw:
            preview = clean[:80].replace("\n", " ")
            suffix = "..." if len(clean) > 80 else ""
            print(f"  Clean text: {preview}{suffix}", file=out)

        # Matched subtitles
        match_detail = sm.get("match_detail")
        if match_detail:
            matched_pairs = match_detail["matched"]
            if matched_pairs:
                print(f"  Matched subtitles ({len(matched_pairs)}):", file=out)
                for gt_sub, test_sub in matched_pairs[:8]:  # Show first 8
                    iou = compute_iou(
                        gt_sub["start"], gt_sub["end"],
                        test_sub["start"], test_sub["end"],
                    )
                    text_preview = test_sub["text"][:30]
                    suffix = "..." if len(test_sub["text"]) > 30 else ""
                    ok = "ok" if iou > 0.7 else "!!"
                    print(
                        f"    GT#{gt_sub['index']:d} <> T#{test_sub['index']:d} "
                        f"IoU={iou:.2f} \"{text_preview}{suffix}\" {ok}",
                        file=out,
                    )
                if len(matched_pairs) > 8:
                    print(f"    ... ({len(matched_pairs) - 8} more)", file=out)

            missed = match_detail["missed"]
            if missed:
                print(f"  Missed GT subs: {len(missed)}", file=out)

            hallucinated = match_detail["hallucinated"]
            if hallucinated:
                print(f"  Hallucinated test subs: {len(hallucinated)}", file=out)

        # Word timestamps
        words = t.word_timestamps.get(scene_index)
        if words:
            print(f"  Word timestamps ({len(words)} words):", file=out)
            # Show first few words on one line
            word_strs = []
            for w in words[:12]:
                word = w.get("word", "?")
                ws = w.get("start", 0)
                we = w.get("end", 0)
                word_strs.append(f"{word}[{ws:.2f}-{we:.2f}]")
            line = "    " + " ".join(word_strs)
            if len(words) > 12:
                line += " ..."
            print(line, file=out)

        # Mini-provenance table for subs in this scene
        scene_provs = [
            p for p in test_analysis.get("provenance", [])
            if p["scene_index"] == scene_index
        ]
        if scene_provs:
            print("  Sub provenance in scene:", file=out)
            print(
                f"    {'#':>4}  {'RelStart':>8}  {'VAD':>3}  "
                f"{'Source':<22} {'GT#':>4}  {'IoU':>5}  {'Flags'}",
                file=out,
            )
            for p in scene_provs[:15]:
                rel_s = f"{p['scene_relative_start']:.1f}s"
                vad = f"{p['vad_group_index']}" if p["vad_group_index"] is not None else "-"
                gt_idx = f"{p['gt_match_index']}" if p["gt_match_index"] is not None else "-"
                iou_str = f"{p['gt_iou']:.2f}" if p["gt_iou"] is not None else "  -  "
                flags = []
                if p["has_regression"]:
                    flags.append("REGR")
                if p["out_of_scene_bounds"]:
                    flags.append("OOB")
                if p["has_overlap"]:
                    flags.append("OVLP")
                flag_str = ",".join(flags)
                print(
                    f"    {p['sub_index']:>4}  {rel_s:>8}  {vad:>3}  "
                    f"{p['timing_source']:<22} {gt_idx:>4}  {iou_str:>5}  {flag_str}",
                    file=out,
                )
            if len(scene_provs) > 15:
                print(f"    ... ({len(scene_provs) - 15} more)", file=out)


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def write_json_report(analysis: dict, output_path: Path):
    """
    Write full analysis as JSON.

    Strips non-serializable match_detail from scene_metrics to keep
    the JSON clean (the raw matching data is large).
    """
    # Deep copy and clean up for serialization
    clean = _prepare_json(analysis)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(clean, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _prepare_json(analysis: dict) -> dict:
    """Prepare analysis dict for JSON serialization."""
    import copy
    clean = copy.deepcopy(analysis)

    # Remove match_detail from scene_metrics (too verbose for JSON)
    # Remove provenance list (too verbose), keep timing_analytics
    for test in clean.get("tests", []):
        for s_idx, sm in test.get("scene_metrics", {}).items():
            if "match_detail" in sm:
                # Keep counts but remove raw pairs
                del sm["match_detail"]
        if "provenance" in test:
            del test["provenance"]

    return clean
