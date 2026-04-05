#!/usr/bin/env python3
"""
WhisperJAV Forensic CSV Generator — Adapted for S0104/R1 test
================================================================
Produces a detailed forensic CSV comparing every ground truth subtitle against
a WhisperJAV pass output. Each GT subtitle gets one row with full pipeline
trace metadata: scene coverage, VAD coverage, raw model output, inline filter
status, sanitizer status, and a mechanically-determined loss stage.

Usage:
    python forensic_csv_generator_s0104.py --pass-number 1 --output S0104_R1_PASS1_FORENSIC.csv
    python forensic_csv_generator_s0104.py --pass-number 2 --output S0104_R1_PASS2_FORENSIC.csv

Adapted from the T6 forensic script for the 293sec-S01E04-scene4 test media.
Key differences from original:
  - Basename: 293sec-S01E04-scene4 (not SONE-853)
  - GT file: Ground-Truth-Netflix-Reference-Subs.srt
  - Transcribe JSON format: segments at group level (not nested under 'result')
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_BASE_DIR = Path(r"F:\MEDIA_DLNA\SONE-853\S0104\R1")
BASENAME = "293sec-S01E04-scene4"
GT_FILENAME = "Ground-Truth-Netflix-Reference-Subs.srt"
MATCH_TOLERANCE_SEC = 3.0


# ============================================================================
# SRT Parsing
# ============================================================================

def parse_srt_time(time_str: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    time_str = time_str.strip()
    match = re.match(r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})', time_str)
    if not match:
        return 0.0
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt_file(path: Path) -> List[Dict]:
    """Parse SRT file into list of {index, start, end, start_sec, end_sec, text}."""
    if not path.exists():
        return []

    content = path.read_text(encoding='utf-8-sig')
    blocks = re.split(r'\n\s*\n', content.strip())
    subs = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue

        time_match = re.match(
            r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})',
            lines[1].strip()
        )
        if not time_match:
            continue

        start_str, end_str = time_match.groups()
        text = '\n'.join(lines[2:]).strip()

        subs.append({
            'index': idx,
            'start': start_str,
            'end': end_str,
            'start_sec': parse_srt_time(start_str),
            'end_sec': parse_srt_time(end_str),
            'text': text,
        })

    return subs


# ============================================================================
# Artifact Parsing
# ============================================================================

def parse_artifacts_srt(path: Path) -> List[Dict]:
    """Parse the sanitization artifacts SRT. Returns list of removed/modified entries."""
    if not path.exists():
        return []

    content = path.read_text(encoding='utf-8-sig')
    blocks = re.split(r'\n\s*\n', content.strip())
    artifacts = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        time_match = re.match(
            r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})',
            lines[1].strip()
        )
        if not time_match:
            continue

        start_str, end_str = time_match.groups()
        body = '\n'.join(lines[2:]).strip()

        reason = ""
        original = ""
        if body.startswith('[REMOVED'):
            reason_match = re.match(r'\[REMOVED\s*-\s*(.+?)\]', body)
            if reason_match:
                reason = reason_match.group(1).strip()
            orig_match = re.search(r'Original:\s*(.+)', body)
            if orig_match:
                original = orig_match.group(1).strip()
        elif body.startswith('[TIMING MODIFIED'):
            reason = "timing_modified"
            orig_match = re.search(r'Original:\s*(.+)', body)
            if orig_match:
                original = orig_match.group(1).strip()
        elif body.startswith('[MODIFIED'):
            reason = "modified"
            orig_match = re.search(r'Original:\s*(.+)', body)
            if orig_match:
                original = orig_match.group(1).strip()
        elif body.startswith('[SANITIZATION SUMMARY]'):
            continue

        artifacts.append({
            'start_sec': parse_srt_time(start_str),
            'end_sec': parse_srt_time(end_str),
            'reason': reason,
            'original': original,
            'body': body,
        })

    return artifacts


# ============================================================================
# Scene & Transcribe JSON Loading
# ============================================================================

def load_master_json(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_semantic_json(path: Path) -> Optional[List[Dict]]:
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('segments', [])


def load_transcribe_json(path: Path) -> Optional[List[Dict]]:
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception):
        return None


# ============================================================================
# Matching Logic
# ============================================================================

def find_covering_scene(timestamp_sec: float, scenes: List[Dict]) -> Optional[Dict]:
    for scene in scenes:
        s = scene['start_time_seconds']
        e = scene['end_time_seconds']
        if s <= timestamp_sec <= e:
            return scene
    return None


def find_semantic_segment(timestamp_sec: float, semantic_segments: List[Dict]) -> Optional[Dict]:
    if not semantic_segments:
        return None
    for seg in semantic_segments:
        ts = seg.get('timestamps', {})
        if ts.get('start', 0) <= timestamp_sec <= ts.get('end', 0):
            return seg
    return None


def find_matching_sub(gt_start_sec: float, subs: List[Dict], tolerance: float) -> Optional[Dict]:
    best = None
    best_dist = tolerance + 1
    for sub in subs:
        dist = abs(sub['start_sec'] - gt_start_sec)
        if dist <= tolerance and dist < best_dist:
            best = sub
            best_dist = dist
    return best


def find_artifact_at(gt_start_sec: float, artifacts: List[Dict], tolerance: float) -> Optional[Dict]:
    best = None
    best_dist = tolerance + 1
    for art in artifacts:
        dist = abs(art['start_sec'] - gt_start_sec)
        if dist <= tolerance and dist < best_dist:
            best = art
            best_dist = dist
    return best


def find_raw_sub_at(gt_start_sec: float, raw_subs: List[Dict], tolerance: float) -> Optional[Dict]:
    best = None
    best_dist = tolerance + 1
    for sub in raw_subs:
        dist = abs(sub['start_sec'] - gt_start_sec)
        if dist <= tolerance and dist < best_dist:
            best = sub
            best_dist = dist
    return best


def find_vad_group_and_segment(
    gt_start_sec: float,
    scene: Dict,
    scene_srts_dir: Path,
    tolerance: float = 5.0
) -> Tuple[Optional[int], Optional[Dict], Optional[str]]:
    """
    Find the VAD group and closest model segment for a GT timestamp within a scene.

    This version handles BOTH transcribe.json formats:
      - Old format: group['result']['segments'] (T6 style)
      - New format: group['segments'] (S0104 style, segments directly on group)
    """
    scene_idx = scene['scene_index']
    scene_start = scene['start_time_seconds']

    json_path = scene_srts_dir / f"{BASENAME}_scene_{scene_idx:04d}.transcribe.json"
    groups = load_transcribe_json(json_path)

    if not groups:
        return None, None, None

    gt_relative = gt_start_sec - scene_start

    best_group_idx = None
    best_segment = None
    best_raw_text = None
    best_dist = tolerance + 1

    for g_idx, group in enumerate(groups):
        group_start = group.get('group_start_sec', 0)
        group_end = group.get('group_end_sec', 0)

        # Handle both formats: group['result']['segments'] or group['segments']
        if 'result' in group:
            result = group['result']
            segments = result.get('segments', [])
            result_text = result.get('text', '')
        else:
            segments = group.get('segments', [])
            result_text = ''

        # Check if GT falls within this group's time range (with tolerance)
        if gt_relative >= group_start - tolerance and gt_relative <= group_end + tolerance:
            # Find closest segment within this group
            for seg in segments:
                seg_start = group_start + seg.get('start', 0)
                seg_end = group_start + seg.get('end', 0)
                seg_mid = (seg_start + seg_end) / 2
                dist = abs(gt_relative - seg_mid)

                if dist < best_dist:
                    best_dist = dist
                    best_group_idx = g_idx
                    best_segment = {
                        'no_speech_prob': seg.get('no_speech_prob'),
                        'avg_logprob': seg.get('avg_logprob'),
                        'compression_ratio': seg.get('compression_ratio'),
                        'temperature': seg.get('temperature'),
                        'text': seg.get('text', ''),
                    }
                    best_raw_text = seg.get('text', '')

            # If no segments in this group but group covers the time, record the group
            if not segments and best_group_idx is None:
                if group_start - tolerance <= gt_relative <= group_end + tolerance:
                    best_group_idx = g_idx
                    best_raw_text = result_text or ''

    return best_group_idx, best_segment, best_raw_text


# ============================================================================
# Loss Stage Classification
# ============================================================================

def classify_loss_stage(
    has_match: bool,
    scene: Optional[Dict],
    vad_group_idx: Optional[int],
    best_segment: Optional[Dict],
    raw_model_text: Optional[str],
    in_raw_srt: bool,
    sanitizer_removed: bool,
) -> str:
    """Mechanically determine which pipeline stage caused the loss."""

    if has_match:
        return "Captured"

    if scene is None:
        return "Skipped_by_scenedetect"

    if scene.get('no_speech_detected', False):
        if vad_group_idx is not None:
            if raw_model_text:
                return "Dropped_internal_threshold"
            else:
                return "Empty_transcriber_results"
        return "Skipped_by_VAD"

    if vad_group_idx is None:
        return "Skipped_by_VAD"

    if best_segment is None and not raw_model_text:
        return "Empty_transcriber_results"

    if raw_model_text:
        if sanitizer_removed:
            return "Removed_by_sanitization"
        if in_raw_srt:
            return "Removed_by_sanitization"
        return "Dropped_internal_threshold"

    return "Missed_Unknown_Cause"


# ============================================================================
# Main Generator
# ============================================================================

def generate_forensic_csv(pass_number: int, output_path: Path, base_dir: Path = None):
    """Generate the forensic CSV for a given pass."""

    BASE_DIR = base_dir or DEFAULT_BASE_DIR
    GT_SRT = BASE_DIR / GT_FILENAME

    print(f"Generating forensic CSV for Pass {pass_number}...")
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Basename: {BASENAME}")

    # --- Load all data sources ---

    gt_subs = parse_srt_file(GT_SRT)
    print(f"  Ground truth: {len(gt_subs)} subtitles")

    pass_dir = BASE_DIR / "TEMP" / f"pass{pass_number}_worker"

    # Final SRT
    final_srt_path = BASE_DIR / f"{BASENAME}.ja.pass{pass_number}.srt"
    final_subs = parse_srt_file(final_srt_path)
    print(f"  Pass {pass_number} final SRT: {len(final_subs)} subtitles")

    # Raw SRT (before sanitization = the stitched SRT)
    raw_srt_path = pass_dir / f"{BASENAME}_stitched.srt"
    raw_subs = parse_srt_file(raw_srt_path)
    print(f"  Pass {pass_number} raw/stitched SRT: {len(raw_subs)} subtitles")

    # Artifacts
    artifacts_path = pass_dir / "raw_subs" / f"{BASENAME}_stitched.artifacts.srt"
    if not artifacts_path.exists():
        artifacts_path = BASE_DIR / "raw_subs" / f"{BASENAME}_stitched.artifacts.srt"
    artifacts = parse_artifacts_srt(artifacts_path)
    print(f"  Artifacts: {len(artifacts)} entries")

    # Master JSON
    master_json_path = pass_dir / f"{BASENAME}_master.json"
    master = load_master_json(master_json_path)
    scenes = master.get('scenes_detected', [])
    print(f"  Scenes: {len(scenes)}")

    # Semantic JSON
    semantic_json_path = pass_dir / "scenes" / f"{BASENAME}_semantic.json"
    semantic_segments = load_semantic_json(semantic_json_path)
    has_semantic = semantic_segments is not None
    print(f"  Semantic data: {'YES' if has_semantic else 'NO'}")

    # Scene SRTs directory
    scene_srts_dir = pass_dir / "scene_srts"

    # Config info
    config = master.get('config', {})
    vad_params = config.get('vad_params', {})
    pipeline_opts = config.get('pipeline_options', {})
    provider = pipeline_opts.get('provider', {})
    print(f"  VAD params: threshold={vad_params.get('threshold')}, "
          f"min_speech={vad_params.get('min_speech_duration_ms')}ms")
    print(f"  Logprob threshold: {provider.get('logprob_threshold')}")
    print(f"  No speech threshold: {provider.get('no_speech_threshold')}")

    # --- Build CSV rows ---

    rows = []
    stats = {
        'Captured': 0,
        'Skipped_by_scenedetect': 0,
        'Skipped_by_VAD': 0,
        'Dropped_internal_threshold': 0,
        'Empty_transcriber_results': 0,
        'Removed_by_sanitization': 0,
        'Missed_Unknown_Cause': 0,
    }

    for gt in gt_subs:
        gt_start = gt['start_sec']
        gt_end = gt['end_sec']

        # Find matching final subtitle
        match = find_matching_sub(gt_start, final_subs, MATCH_TOLERANCE_SEC)
        has_match = match is not None

        # Find covering scene
        scene = find_covering_scene(gt_start, scenes)

        # Find semantic classification
        sem_seg = find_semantic_segment(gt_start, semantic_segments) if has_semantic else None

        # Find VAD group and model segment data
        vad_group_idx = None
        best_segment = None
        raw_model_text = None

        if scene is not None:
            vad_group_idx, best_segment, raw_model_text = find_vad_group_and_segment(
                gt_start, scene, scene_srts_dir
            )

        # Check raw SRT
        raw_match = find_raw_sub_at(gt_start, raw_subs, MATCH_TOLERANCE_SEC)
        in_raw_srt = raw_match is not None

        # Check artifacts
        artifact = find_artifact_at(gt_start, artifacts, MATCH_TOLERANCE_SEC)
        sanitizer_removed = artifact is not None

        # Classify loss stage
        loss_stage = classify_loss_stage(
            has_match, scene, vad_group_idx, best_segment,
            raw_model_text, in_raw_srt, sanitizer_removed
        )
        stats[loss_stage] = stats.get(loss_stage, 0) + 1

        # Build row
        row = {
            'GT_Sub_Number': gt['index'],
            'GT_Start_Time': gt['start'],
            'GT_End_Time': gt['end'],
            'GT_Text': gt['text'],

            'Has_Corresponding_Sub': 'YES' if has_match else 'NO',
            'Pass_Final_Text': match['text'] if match else '',

            'Scene_Number': scene['scene_index'] if scene else '',
            'Scene_Start': f"{scene['start_time_seconds']:.2f}" if scene else '',
            'Scene_End': f"{scene['end_time_seconds']:.2f}" if scene else '',
            'Scene_Duration': f"{scene['duration_seconds']:.2f}" if scene else '',
            'Scene_No_Speech_Flag': str(scene.get('no_speech_detected', False)).upper() if scene else '',

            'Semantic_Label': sem_seg['context']['label'] if sem_seg else '',
            'Semantic_Confidence': sem_seg['context']['confidence'] if sem_seg else '',
            'Semantic_Loudness_dB': sem_seg['context']['loudness_db'] if sem_seg else '',

            'VAD_Group_Number': vad_group_idx if vad_group_idx is not None else '',

            'Model_No_Speech_Prob': f"{best_segment['no_speech_prob']:.4f}" if best_segment and best_segment.get('no_speech_prob') is not None else '',
            'Model_Avg_Logprob': f"{best_segment['avg_logprob']:.4f}" if best_segment and best_segment.get('avg_logprob') is not None else '',
            'Model_Compression_Ratio': f"{best_segment['compression_ratio']:.4f}" if best_segment and best_segment.get('compression_ratio') is not None else '',
            'Model_Temperature': f"{best_segment['temperature']}" if best_segment and best_segment.get('temperature') is not None else '',

            'Raw_Model_Text': raw_model_text or '',

            'In_Raw_SRT': 'YES' if in_raw_srt else 'NO',
            'Raw_SRT_Text': raw_match['text'] if raw_match else '',

            'Sanitizer_Removed': 'YES' if sanitizer_removed else 'NO',
            'Sanitizer_Reason': artifact['reason'] if artifact else '',

            'Loss_Stage': loss_stage,
        }

        rows.append(row)

    # --- Write CSV ---

    fieldnames = [
        'GT_Sub_Number', 'GT_Start_Time', 'GT_End_Time', 'GT_Text',
        'Has_Corresponding_Sub', 'Pass_Final_Text',
        'Scene_Number', 'Scene_Start', 'Scene_End', 'Scene_Duration', 'Scene_No_Speech_Flag',
        'Semantic_Label', 'Semantic_Confidence', 'Semantic_Loudness_dB',
        'VAD_Group_Number',
        'Model_No_Speech_Prob', 'Model_Avg_Logprob', 'Model_Compression_Ratio', 'Model_Temperature',
        'Raw_Model_Text',
        'In_Raw_SRT', 'Raw_SRT_Text',
        'Sanitizer_Removed', 'Sanitizer_Reason',
        'Loss_Stage',
    ]

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Written: {output_path} ({len(rows)} rows)")
    print(f"\n  Loss Stage Distribution:")
    for stage, count in sorted(stats.items(), key=lambda x: -x[1]):
        pct = count / len(rows) * 100 if rows else 0
        print(f"    {stage:35s}: {count:4d} ({pct:5.1f}%)")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WhisperJAV Forensic CSV Generator — S0104/R1')
    parser.add_argument('--pass-number', type=int, required=True, choices=[1, 2],
                        help='Pass number (1 or 2)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory for test artifacts (default: S0104/R1)')
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else DEFAULT_BASE_DIR
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path

    generate_forensic_csv(args.pass_number, output_path, base_dir)
