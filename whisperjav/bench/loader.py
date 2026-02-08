"""
Load pipeline test artifacts from a temp directory.

Auto-detects input mode from _master.json, loads scene boundaries,
raw/clean texts, word timestamps, scene SRTs, and Phase 2 diagnostics.

Handles both assembly and coupled-mode naming conventions.
Graceful degradation: missing artifacts → empty dicts/lists (not errors).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubtitleEntry:
    """A single subtitle with timing and text."""

    index: int
    start: float  # seconds
    end: float  # seconds
    text: str


@dataclass
class SceneBoundary:
    """Scene time range from scene detection."""

    index: int
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass
class TestResult:
    """All artifacts from a single pipeline test run."""

    name: str  # User-provided label (e.g., "T1: vad_slicing")
    path: Path  # Path to test temp directory
    input_mode: str  # "assembly", "context_aware", "vad_slicing"
    model_id: str  # e.g., "Qwen/Qwen3-ASR-1.7B"
    basename: str  # Media basename from _master.json
    final_srt: List[SubtitleEntry]
    scene_boundaries: List[SceneBoundary]
    scene_srts: Dict[int, List[SubtitleEntry]]  # scene_index → subs
    raw_texts: Dict[int, str]  # scene_index → raw ASR text
    clean_texts: Dict[int, str]  # scene_index → cleaned text (assembly only)
    word_timestamps: Dict[int, List[dict]]  # scene_index → word dicts
    raw_aligner_timestamps: Dict[int, List[dict]]  # scene_index → raw aligner (coupled only)
    master_metadata: dict  # Full _master.json content
    scene_diagnostics: Dict[int, dict] = field(default_factory=dict)  # Phase 2 (optional)
    processing_time_sec: float = 0.0


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def _parse_srt_file(srt_path: Path) -> List[SubtitleEntry]:
    """
    Parse an SRT file into SubtitleEntry objects.

    Uses pysrt (already a project dependency) for robust parsing.
    Falls back to empty list on any error.
    """
    if not srt_path.exists():
        return []

    try:
        import pysrt
        subs = pysrt.open(str(srt_path), encoding="utf-8")
        entries = []
        for sub in subs:
            entries.append(SubtitleEntry(
                index=sub.index,
                start=sub.start.ordinal / 1000.0,
                end=sub.end.ordinal / 1000.0,
                text=sub.text,
            ))
        return entries
    except Exception as e:
        logger.warning("Failed to parse SRT %s: %s", srt_path.name, e)
        return []


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    """Load a JSON file, returning None on failure."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load JSON %s: %s", path.name, e)
        return None


def _load_json_list(path: Path) -> Optional[list]:
    """Load a JSON file expected to contain a list."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return None
    except Exception as e:
        logger.warning("Failed to load JSON list %s: %s", path.name, e)
        return None


# ---------------------------------------------------------------------------
# Scene index extraction
# ---------------------------------------------------------------------------

_SCENE_INDEX_RE = re.compile(r"scene_(\d{4})")


def _extract_scene_index(filename: str) -> Optional[int]:
    """Extract scene index from a filename containing scene_NNNN."""
    m = _SCENE_INDEX_RE.search(filename)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Master metadata discovery
# ---------------------------------------------------------------------------

def _find_master_json(test_dir: Path) -> Optional[Path]:
    """
    Find the *_master.json file in the test directory.

    Expects exactly one. If multiple, takes the first by name.
    """
    candidates = sorted(test_dir.glob("*_master.json"))
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning(
            "Multiple _master.json files found in %s, using %s",
            test_dir, candidates[0].name,
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Scene boundary loading
# ---------------------------------------------------------------------------

def _load_scene_boundaries(test_dir: Path) -> List[SceneBoundary]:
    """Load scene boundaries from scenes/*_semantic.json."""
    scenes_dir = test_dir / "scenes"
    if not scenes_dir.is_dir():
        return []

    # Find the semantic JSON (should be exactly one)
    candidates = sorted(scenes_dir.glob("*_semantic.json"))
    if not candidates:
        return []

    data = _load_json(candidates[0])
    if not data or "segments" not in data:
        return []

    boundaries = []
    for seg in data["segments"]:
        ts = seg.get("timestamps", {})
        idx = seg.get("segment_index", len(boundaries))
        boundaries.append(SceneBoundary(
            index=idx,
            start_sec=ts.get("start", 0.0),
            end_sec=ts.get("end", 0.0),
            duration_sec=ts.get("duration", 0.0),
        ))

    return boundaries


# ---------------------------------------------------------------------------
# Scene SRT loading
# ---------------------------------------------------------------------------

def _load_scene_srts(test_dir: Path) -> Dict[int, List[SubtitleEntry]]:
    """Load per-scene SRT files from scene_srts/ directory."""
    scene_srts_dir = test_dir / "scene_srts"
    if not scene_srts_dir.is_dir():
        return {}

    result = {}
    for srt_file in sorted(scene_srts_dir.glob("*_scene_????.srt")):
        idx = _extract_scene_index(srt_file.name)
        if idx is not None:
            entries = _parse_srt_file(srt_file)
            if entries:
                result[idx] = entries

    return result


# ---------------------------------------------------------------------------
# Raw text loading (mode-aware)
# ---------------------------------------------------------------------------

def _load_raw_texts(
    raw_subs_dir: Path, input_mode: str, basename: str,
) -> Dict[int, str]:
    """
    Load raw ASR text per scene.

    Assembly mode:  scene_NNNN_assembly_raw.txt
    Coupled modes:  {basename}_scene_NNNN_qwen_master.txt
    """
    if not raw_subs_dir.is_dir():
        return {}

    result = {}

    if input_mode == "assembly":
        for f in sorted(raw_subs_dir.glob("scene_????_assembly_raw.txt")):
            idx = _extract_scene_index(f.name)
            if idx is not None:
                try:
                    result[idx] = f.read_text(encoding="utf-8")
                except Exception:
                    pass
    else:
        # Coupled modes: {basename}_scene_NNNN_qwen_master.txt
        for f in sorted(raw_subs_dir.glob("*_qwen_master.txt")):
            idx = _extract_scene_index(f.name)
            if idx is not None:
                try:
                    result[idx] = f.read_text(encoding="utf-8")
                except Exception:
                    pass

    return result


def _load_clean_texts(raw_subs_dir: Path) -> Dict[int, str]:
    """Load cleaned ASR text (assembly mode only): scene_NNNN_assembly_clean.txt."""
    if not raw_subs_dir.is_dir():
        return {}

    result = {}
    for f in sorted(raw_subs_dir.glob("scene_????_assembly_clean.txt")):
        idx = _extract_scene_index(f.name)
        if idx is not None:
            try:
                result[idx] = f.read_text(encoding="utf-8")
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# Word timestamp loading (mode-aware)
# ---------------------------------------------------------------------------

def _load_word_timestamps(
    raw_subs_dir: Path, input_mode: str, basename: str,
) -> Dict[int, List[dict]]:
    """
    Load merged word-level timestamps per scene.

    Assembly mode:  scene_NNNN_assembly_merged.json
    Coupled modes:  {basename}_scene_NNNN_qwen_merged.json
    """
    if not raw_subs_dir.is_dir():
        return {}

    result = {}

    if input_mode == "assembly":
        pattern = "scene_????_assembly_merged.json"
    else:
        pattern = "*_qwen_merged.json"

    for f in sorted(raw_subs_dir.glob(pattern)):
        idx = _extract_scene_index(f.name)
        if idx is not None:
            words = _load_json_list(f)
            if words is not None:
                result[idx] = words

    return result


def _load_raw_aligner_timestamps(
    raw_subs_dir: Path, basename: str,
) -> Dict[int, List[dict]]:
    """
    Load raw ForcedAligner timestamps (coupled modes only).

    File: {basename}_scene_NNNN_qwen_timestamps.json
    """
    if not raw_subs_dir.is_dir():
        return {}

    result = {}
    for f in sorted(raw_subs_dir.glob("*_qwen_timestamps.json")):
        idx = _extract_scene_index(f.name)
        if idx is not None:
            data = _load_json_list(f)
            if data is not None:
                result[idx] = data

    return result


# ---------------------------------------------------------------------------
# Diagnostics loading (Phase 2 — optional/graceful)
# ---------------------------------------------------------------------------

def _load_scene_diagnostics(raw_subs_dir: Path) -> Dict[int, dict]:
    """
    Load Phase 2 per-scene diagnostics: scene_NNNN_diagnostics.json.

    Returns empty dict if files don't exist (pre-Phase 2 test runs).
    """
    if not raw_subs_dir.is_dir():
        return {}

    result = {}
    for f in sorted(raw_subs_dir.glob("scene_????_diagnostics.json")):
        idx = _extract_scene_index(f.name)
        if idx is not None:
            data = _load_json(f)
            if data is not None:
                result[idx] = data

    return result


# ---------------------------------------------------------------------------
# Final SRT discovery
# ---------------------------------------------------------------------------

def _find_final_srt(test_dir: Path, master_metadata: dict) -> Optional[Path]:
    """
    Find the final SRT file.

    Strategy:
    1. Check srt_path / output_files.final_srt from master metadata
    2. Search test_dir for *.ja.whisperjav.srt
    3. Search test_dir parent for *.ja.whisperjav.srt
    """
    # Try paths from metadata
    for key in ("srt_path", ):
        path_str = master_metadata.get(key)
        if path_str:
            p = Path(path_str)
            if p.exists():
                return p

    output_files = master_metadata.get("output_files", {})
    final_path = output_files.get("final_srt")
    if final_path:
        p = Path(final_path)
        if p.exists():
            return p

    # Search in test directory and parent
    for search_dir in (test_dir, test_dir.parent):
        candidates = sorted(search_dir.glob("*.whisperjav.srt"))
        if candidates:
            return candidates[0]

    return None


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_test_result(test_dir: Path, name: str) -> TestResult:
    """
    Load all artifacts from a pipeline test output directory.

    Args:
        test_dir: Path to the test temp directory containing _master.json
                  and subdirectories (scenes/, raw_subs/, scene_srts/).
        name: User-provided label for this test (e.g., "T1: vad_slicing").

    Returns:
        TestResult with all discoverable artifacts loaded.

    Raises:
        FileNotFoundError: If _master.json cannot be found.
        ValueError: If _master.json is missing required fields.
    """
    test_dir = Path(test_dir).resolve()

    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # --- Master metadata ---
    master_path = _find_master_json(test_dir)
    if master_path is None:
        raise FileNotFoundError(
            f"No *_master.json found in {test_dir}. "
            "Is this a WhisperJAV temp directory?"
        )

    master_metadata = _load_json(master_path)
    if master_metadata is None:
        raise ValueError(f"Failed to parse {master_path}")

    # Extract key fields
    basename = master_metadata.get("basename", "")
    stages = master_metadata.get("stages", {})
    asr_stage = stages.get("asr", {})
    input_mode = asr_stage.get("input_mode", "unknown")
    model_id = asr_stage.get("model_id", master_metadata.get("model_id", "unknown"))
    total_time = master_metadata.get("total_time_sec", 0.0)

    if not basename:
        # Fallback: derive from master filename (remove _master.json suffix)
        basename = master_path.stem.replace("_master", "")

    raw_subs_dir = test_dir / "raw_subs"

    # --- Load all artifacts ---
    scene_boundaries = _load_scene_boundaries(test_dir)
    scene_srts = _load_scene_srts(test_dir)
    raw_texts = _load_raw_texts(raw_subs_dir, input_mode, basename)
    clean_texts = _load_clean_texts(raw_subs_dir) if input_mode == "assembly" else {}
    word_timestamps = _load_word_timestamps(raw_subs_dir, input_mode, basename)
    raw_aligner = _load_raw_aligner_timestamps(raw_subs_dir, basename) if input_mode != "assembly" else {}
    scene_diagnostics = _load_scene_diagnostics(raw_subs_dir)

    # --- Final SRT ---
    final_srt_path = _find_final_srt(test_dir, master_metadata)
    final_srt = _parse_srt_file(final_srt_path) if final_srt_path else []

    if not final_srt:
        logger.warning("No final SRT found for test '%s' in %s", name, test_dir)

    # --- Summary ---
    n_scenes = len(scene_boundaries)
    n_subs = len(final_srt)
    n_scene_srts = len(scene_srts)
    n_raw = len(raw_texts)
    n_words = len(word_timestamps)
    n_diag = len(scene_diagnostics)

    logger.info(
        "Loaded test '%s': mode=%s, %d scenes, %d subs, "
        "%d scene_srts, %d raw_texts, %d word_ts, %d diagnostics",
        name, input_mode, n_scenes, n_subs,
        n_scene_srts, n_raw, n_words, n_diag,
    )

    return TestResult(
        name=name,
        path=test_dir,
        input_mode=input_mode,
        model_id=model_id,
        basename=basename,
        final_srt=final_srt,
        scene_boundaries=scene_boundaries,
        scene_srts=scene_srts,
        raw_texts=raw_texts,
        clean_texts=clean_texts,
        word_timestamps=word_timestamps,
        raw_aligner_timestamps=raw_aligner,
        master_metadata=master_metadata,
        scene_diagnostics=scene_diagnostics,
        processing_time_sec=total_time,
    )


def load_ground_truth(srt_path: Path) -> List[SubtitleEntry]:
    """
    Load ground-truth SRT file.

    Args:
        srt_path: Path to the ground-truth SRT file.

    Returns:
        List of SubtitleEntry objects.

    Raises:
        FileNotFoundError: If SRT file doesn't exist.
        ValueError: If SRT file is empty or unparseable.
    """
    srt_path = Path(srt_path).resolve()

    if not srt_path.exists():
        raise FileNotFoundError(f"Ground truth SRT not found: {srt_path}")

    entries = _parse_srt_file(srt_path)
    if not entries:
        raise ValueError(f"Ground truth SRT is empty or unparseable: {srt_path}")

    logger.info(
        "Loaded ground truth: %d subtitles, %.1fs duration",
        len(entries),
        entries[-1].end if entries else 0.0,
    )

    return entries
