"""
Pipeline Analytics — post-run health check and metrics aggregation.

Reads per-scene diagnostics JSONs and the final stitched SRT to produce
a comprehensive analytics report. Designed to be called automatically by
the pipeline after Phase 8, but also usable standalone for post-hoc
analysis of existing temp directories.

Usage (standalone):
    from whisperjav.modules.pipeline_analytics import compute_analytics, print_summary
    analytics = compute_analytics(raw_subs_dir, srt_path)
    print_summary(analytics)

Usage (pipeline integration):
    Called automatically at end of QwenPipeline.process() before cleanup.
"""

import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from whisperjav.utils.logger import logger

# ---------------------------------------------------------------------------
# SRT timestamp parsing
# ---------------------------------------------------------------------------

_SRT_TS_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _parse_srt_timestamp(ts: str) -> float:
    """Parse 'HH:MM:SS,mmm' → seconds as float."""
    m = _SRT_TS_RE.match(ts.strip())
    if not m:
        return 0.0
    h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mi * 60 + s + ms / 1000.0


def _format_duration(seconds: float) -> str:
    """Format seconds → 'HH:MM:SS' or 'MM:SS' string."""
    if seconds < 0:
        seconds = 0.0
    total_sec = int(seconds)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def _parse_srt(srt_path: Path) -> List[Dict]:
    """
    Parse an SRT file into a list of subtitle dicts.

    Returns:
        List of {'index': int, 'start': float, 'end': float, 'text': str}
    """
    if not srt_path.exists():
        return []

    text = srt_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    subs = []
    # Split on blank lines to get blocks
    blocks = re.split(r"\n\s*\n", text)

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        # Find the timestamp line (contains ' --> ')
        ts_line_idx = None
        for i, line in enumerate(lines):
            if " --> " in line:
                ts_line_idx = i
                break

        if ts_line_idx is None:
            continue

        # Parse timestamps
        ts_parts = lines[ts_line_idx].split(" --> ")
        if len(ts_parts) != 2:
            continue

        start = _parse_srt_timestamp(ts_parts[0])
        end = _parse_srt_timestamp(ts_parts[1])

        # Parse index (line before timestamp)
        idx = 0
        if ts_line_idx > 0:
            try:
                idx = int(lines[ts_line_idx - 1].strip())
            except ValueError:
                pass

        # Text is everything after the timestamp line
        sub_text = "\n".join(lines[ts_line_idx + 1:]).strip()

        subs.append({
            "index": idx,
            "start": start,
            "end": end,
            "text": sub_text,
        })

    return subs


# ---------------------------------------------------------------------------
# Diagnostics loading
# ---------------------------------------------------------------------------

def _load_diagnostics(raw_subs_dir: Path) -> List[Dict]:
    """Load all scene_NNNN_diagnostics.json files, sorted by scene index."""
    if not raw_subs_dir.exists():
        return []

    diag_files = sorted(raw_subs_dir.glob("scene_*_diagnostics.json"))
    scenes = []

    for f in diag_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            scenes.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Analytics: Failed to load %s: %s", f.name, e)

    return scenes


# ---------------------------------------------------------------------------
# Metric computation: Scenes & Audio
# ---------------------------------------------------------------------------

def _compute_scene_metrics(scenes: List[Dict]) -> Dict:
    """Compute audio/scene health metrics."""
    if not scenes:
        return {
            "scene_count": 0,
            "total_audio_duration_sec": 0.0,
            "scene_duration_min": 0.0,
            "scene_duration_max": 0.0,
            "scene_duration_mean": 0.0,
            "scene_duration_median": 0.0,
            "total_vad_speech_sec": 0.0,
            "speech_ratio": 0.0,
        }

    durations = [s.get("scene_duration_sec", 0.0) for s in scenes]
    total_audio = sum(durations)

    # VAD speech duration: sum of all VAD region durations across scenes
    total_vad = 0.0
    for s in scenes:
        for region in s.get("vad_regions", []):
            r_start = region.get("start", 0.0)
            r_end = region.get("end", 0.0)
            if r_end > r_start:
                total_vad += r_end - r_start

    return {
        "scene_count": len(scenes),
        "total_audio_duration_sec": round(total_audio, 3),
        "scene_duration_min": round(min(durations), 3) if durations else 0.0,
        "scene_duration_max": round(max(durations), 3) if durations else 0.0,
        "scene_duration_mean": round(statistics.mean(durations), 3) if durations else 0.0,
        "scene_duration_median": round(statistics.median(durations), 3) if durations else 0.0,
        "total_vad_speech_sec": round(total_vad, 3),
        "speech_ratio": round(total_vad / total_audio, 4) if total_audio > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Metric computation: Alignment & Step-Down
# ---------------------------------------------------------------------------

def _compute_alignment_metrics(scenes: List[Dict]) -> Dict:
    """Compute step-down/alignment health metrics from group_details."""
    # Aggregate step-down counters
    t1_total = t1_accepted = t1_collapsed = 0
    t2_total = t2_accepted = t2_collapsed = 0

    # Aggregate from existing stepdown summary (works with v1.0.0 too)
    for s in scenes:
        sd = s.get("stepdown")
        if sd and sd.get("enabled"):
            t1_total += sd.get("tier1_groups", 0)
            t1_accepted += sd.get("tier1_accepted", 0)
            t1_collapsed += sd.get("tier1_collapsed", 0)
            t2_total += sd.get("tier2_groups", 0)
            t2_accepted += sd.get("tier2_accepted", 0)
            t2_collapsed += sd.get("tier2_collapsed", 0)

    total_groups = t1_total + t2_total
    total_collapsed = t1_collapsed + t2_collapsed
    collapse_rate = total_collapsed / total_groups if total_groups > 0 else 0.0

    # Recovery strategy breakdown from sentinel
    recovery_vad_guided = 0
    recovery_proportional = 0
    for s in scenes:
        sentinel = s.get("sentinel", {})
        recovery = sentinel.get("recovery")
        if recovery:
            strategy = recovery.get("strategy", "unknown")
            recovered = recovery.get("groups_recovered", 0)
            if "vad" in strategy:
                recovery_vad_guided += recovered
            elif "proportional" in strategy:
                recovery_proportional += recovered

    # Per-group detail analytics (v1.1.0 — graceful when absent)
    trigger_counts = {}
    outcome_counts = {}
    group_count = 0

    for s in scenes:
        for gd in s.get("group_details", []):
            group_count += 1
            outcome = gd.get("outcome", "unknown")
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

            for trigger in gd.get("sentinel_triggers", []):
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

    return {
        "tier1_total": t1_total,
        "tier1_accepted": t1_accepted,
        "tier1_collapsed": t1_collapsed,
        "tier1_acceptance_rate": round(
            t1_accepted / t1_total, 4) if t1_total > 0 else 0.0,
        "tier2_total": t2_total,
        "tier2_accepted": t2_accepted,
        "tier2_collapsed": t2_collapsed,
        "tier2_acceptance_rate": round(
            t2_accepted / t2_total, 4) if t2_total > 0 else 0.0,
        "total_groups": total_groups,
        "total_collapsed": total_collapsed,
        "collapse_rate": round(collapse_rate, 4),
        "recovery_vad_guided": recovery_vad_guided,
        "recovery_proportional": recovery_proportional,
        "trigger_counts": trigger_counts,
        "outcome_counts": outcome_counts,
        "group_details_available": group_count > 0,
    }


# ---------------------------------------------------------------------------
# Metric computation: Subtitle Output
# ---------------------------------------------------------------------------

def _compute_subtitle_metrics(
    subs: List[Dict],
    total_audio_duration_sec: float,
) -> Dict:
    """Compute subtitle output quality metrics."""
    if not subs:
        return {
            "total_subs": 0,
            "total_sub_duration_sec": 0.0,
            "sub_coverage_ratio": 0.0,
            "sub_density_per_min": 0.0,
            "sub_duration_mean": 0.0,
            "sub_duration_median": 0.0,
            "short_subs_count": 0,
            "short_subs_pct": 0.0,
            "large_gaps": [],
            "max_gap_sec": 0.0,
        }

    total_subs = len(subs)
    durations = [max(0.0, s["end"] - s["start"]) for s in subs]
    total_sub_dur = sum(durations)

    # Short subs (< 0.3 seconds)
    short_count = sum(1 for d in durations if d < 0.3)

    # Gaps between consecutive subs
    large_gaps = []
    max_gap = 0.0
    for i in range(1, len(subs)):
        gap = subs[i]["start"] - subs[i - 1]["end"]
        if gap > max_gap:
            max_gap = gap
        if gap > 15.0:
            large_gaps.append({
                "after_sub": subs[i - 1]["index"],
                "before_sub": subs[i]["index"],
                "gap_sec": round(gap, 3),
                "at_time": _format_duration(subs[i - 1]["end"]),
            })

    # Sort gaps by size descending, keep top 10
    large_gaps.sort(key=lambda g: g["gap_sec"], reverse=True)
    large_gaps = large_gaps[:10]

    sub_density = (total_subs / (total_audio_duration_sec / 60.0)
                   if total_audio_duration_sec > 0 else 0.0)

    return {
        "total_subs": total_subs,
        "total_sub_duration_sec": round(total_sub_dur, 3),
        "sub_coverage_ratio": round(
            total_sub_dur / total_audio_duration_sec, 4
        ) if total_audio_duration_sec > 0 else 0.0,
        "sub_density_per_min": round(sub_density, 1),
        "sub_duration_mean": round(
            statistics.mean(durations), 3) if durations else 0.0,
        "sub_duration_median": round(
            statistics.median(durations), 3) if durations else 0.0,
        "short_subs_count": short_count,
        "short_subs_pct": round(
            short_count / total_subs * 100, 1) if total_subs > 0 else 0.0,
        "large_gaps": large_gaps,
        "max_gap_sec": round(max_gap, 3),
    }


# ---------------------------------------------------------------------------
# Metric computation: Timing Sources
# ---------------------------------------------------------------------------

def _compute_timing_metrics(scenes: List[Dict]) -> Dict:
    """Compute timing source breakdown across all scenes."""
    aligner_native = 0
    vad_fallback = 0
    interpolated = 0
    total_segments = 0

    for s in scenes:
        ts = s.get("timing_sources", {})
        aligner_native += ts.get("aligner_native", 0)
        vad_fallback += ts.get("vad_fallback", 0)
        interpolated += ts.get("interpolated", 0)
        total_segments += ts.get("total_segments", 0)

    return {
        "aligner_native": aligner_native,
        "vad_fallback": vad_fallback,
        "interpolated": interpolated,
        "total_segments": total_segments,
        "aligner_native_pct": round(
            aligner_native / total_segments * 100, 1
        ) if total_segments > 0 else 0.0,
        "vad_fallback_pct": round(
            vad_fallback / total_segments * 100, 1
        ) if total_segments > 0 else 0.0,
        "interpolated_pct": round(
            interpolated / total_segments * 100, 1
        ) if total_segments > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Health Scorecard
# ---------------------------------------------------------------------------

def _compute_health_scorecard(
    alignment: Dict,
    timing: Dict,
    subtitle: Dict,
    scene: Dict,
) -> List[Dict]:
    """Compute traffic-light health indicators."""
    indicators = []

    # 1. Collapse rate
    cr = alignment["collapse_rate"] * 100  # to percentage
    if cr < 5:
        level = "GREEN"
    elif cr < 20:
        level = "YELLOW"
    else:
        level = "RED"
    indicators.append({
        "name": "Collapse rate",
        "value": f"{cr:.1f}%",
        "level": level,
    })

    # 2. Aligner native percentage
    an_pct = timing["aligner_native_pct"]
    if an_pct > 90:
        level = "GREEN"
    elif an_pct > 70:
        level = "YELLOW"
    else:
        level = "RED"
    indicators.append({
        "name": "Aligner native",
        "value": f"{an_pct:.1f}%",
        "level": level,
    })

    # 3. Speech ratio
    sr = scene["speech_ratio"] * 100
    if 25 <= sr <= 70:
        level = "GREEN"
    elif 15 <= sr <= 85:
        level = "YELLOW"
    else:
        level = "RED"
    indicators.append({
        "name": "Speech ratio",
        "value": f"{sr:.1f}%",
        "level": level,
    })

    # 4. Max gap
    mg = subtitle["max_gap_sec"]
    detail = ""
    if subtitle["large_gaps"]:
        top = subtitle["large_gaps"][0]
        detail = f" (sub {top['after_sub']}->{top['before_sub']})"
    if mg < 30:
        level = "GREEN"
    elif mg < 60:
        level = "YELLOW"
    else:
        level = "RED"
    indicators.append({
        "name": "Max gap",
        "value": f"{mg:.1f}s{detail}",
        "level": level,
    })

    # 5. Short subs percentage
    ss_pct = subtitle["short_subs_pct"]
    if ss_pct < 5:
        level = "GREEN"
    elif ss_pct < 15:
        level = "YELLOW"
    else:
        level = "RED"
    indicators.append({
        "name": "Short subs (<0.3s)",
        "value": f"{ss_pct:.1f}%",
        "level": level,
    })

    return indicators


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

_LEVEL_SYMBOLS = {"GREEN": "+", "YELLOW": "~", "RED": "!"}


def print_summary(analytics: Dict, title: str = "") -> None:
    """Print concise health-check summary to console."""
    scene = analytics.get("scene", {})
    alignment = analytics.get("alignment", {})
    subtitle = analytics.get("subtitle", {})
    timing = analytics.get("timing", {})
    health = analytics.get("health", [])

    header = " PIPELINE ANALYTICS"
    if title:
        header += f" - {title}"

    bar = "=" * 60
    logger.info("")
    logger.info(bar)
    logger.info(header)
    logger.info(bar)

    # Audio row
    total_dur = _format_duration(scene.get("total_audio_duration_sec", 0))
    sc = scene.get("scene_count", 0)
    mean_s = scene.get("scene_duration_mean", 0)
    median_s = scene.get("scene_duration_median", 0)
    logger.info(
        " Audio       %s total | %d scenes | mean %.1fs | median %.1fs",
        total_dur, sc, mean_s, median_s,
    )

    # Speech row
    vad_dur = _format_duration(scene.get("total_vad_speech_sec", 0))
    sr_pct = scene.get("speech_ratio", 0) * 100
    logger.info(
        " Speech      %s VAD | %.1f%% speech ratio",
        vad_dur, sr_pct,
    )

    # Subtitle row
    total_subs = subtitle.get("total_subs", 0)
    sub_dur = _format_duration(subtitle.get("total_sub_duration_sec", 0))
    cov_pct = subtitle.get("sub_coverage_ratio", 0) * 100
    density = subtitle.get("sub_density_per_min", 0)
    logger.info(
        " Subtitles   %d subs | %s duration | %.1f%% coverage | %.1f/min",
        total_subs, sub_dur, cov_pct, density,
    )

    # Step-Down row
    t1_tot = alignment.get("tier1_total", 0)
    t1_acc = alignment.get("tier1_accepted", 0)
    t2_tot = alignment.get("tier2_total", 0)
    t2_acc = alignment.get("tier2_accepted", 0)
    rv = alignment.get("recovery_vad_guided", 0)
    rp = alignment.get("recovery_proportional", 0)
    if t1_tot > 0 or t2_tot > 0:
        t1_pct = t1_acc / t1_tot * 100 if t1_tot > 0 else 0
        t2_pct = t2_acc / t2_tot * 100 if t2_tot > 0 else 0
        logger.info(
            " Step-Down   T1: %d/%d accepted (%.1f%%)  T2: %d/%d accepted (%.1f%%)",
            t1_acc, t1_tot, t1_pct, t2_acc, t2_tot, t2_pct,
        )
        logger.info(
            "             Recovery: %d vad_guided, %d proportional",
            rv, rp,
        )

    # Trigger breakdown (v1.1.0)
    triggers = alignment.get("trigger_counts", {})
    if triggers:
        parts = [f"{k}={v}" for k, v in
                 sorted(triggers.items(), key=lambda x: -x[1])]
        logger.info(" Triggers    %s", ", ".join(parts))

    # Timing row
    an_pct = timing.get("aligner_native_pct", 0)
    ip_pct = timing.get("interpolated_pct", 0)
    vf_pct = timing.get("vad_fallback_pct", 0)
    logger.info(
        " Timing      %.1f%% aligner | %.1f%% interpolated | %.1f%% vad_fallback",
        an_pct, ip_pct, vf_pct,
    )

    # Health scorecard
    if health:
        logger.info("")
        for ind in health:
            sym = _LEVEL_SYMBOLS.get(ind["level"], "?")
            logger.info(
                " [%s] %s: %s",
                sym, ind["name"], ind["value"],
            )

    logger.info(bar)
    logger.info("")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def save_analytics(analytics: Dict, output_path: Path) -> None:
    """Save full analytics JSON to disk."""
    try:
        output_path.write_text(
            json.dumps(analytics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Analytics saved: %s", output_path.name)
    except OSError as e:
        logger.warning("Failed to save analytics: %s", e)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_analytics(
    raw_subs_dir: Path,
    srt_path: Path,
    title: Optional[str] = None,
) -> Dict:
    """
    Compute pipeline analytics from diagnostics and SRT.

    Args:
        raw_subs_dir: Directory containing scene_NNNN_diagnostics.json files.
        srt_path: Path to the final (or stitched) SRT file.
        title: Optional title for the analytics report (e.g., media filename).

    Returns:
        Analytics dict with keys: scene, alignment, subtitle, timing, health.
    """
    # Load data
    scenes = _load_diagnostics(raw_subs_dir)
    subs = _parse_srt(srt_path)

    if not scenes:
        logger.debug("Analytics: No diagnostics files found in %s", raw_subs_dir)

    # Compute metrics
    scene_metrics = _compute_scene_metrics(scenes)
    alignment_metrics = _compute_alignment_metrics(scenes)
    subtitle_metrics = _compute_subtitle_metrics(
        subs, scene_metrics["total_audio_duration_sec"],
    )
    timing_metrics = _compute_timing_metrics(scenes)
    health = _compute_health_scorecard(
        alignment_metrics, timing_metrics, subtitle_metrics, scene_metrics,
    )

    analytics = {
        "title": title or srt_path.stem,
        "scene": scene_metrics,
        "alignment": alignment_metrics,
        "subtitle": subtitle_metrics,
        "timing": timing_metrics,
        "health": health,
    }

    return analytics
