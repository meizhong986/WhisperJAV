#!/usr/bin/env python3
"""
HuggingFace Transformers Chunked Long-Form ASR Reference Implementation

Uses kotoba-tech/kotoba-whisper-v2.0 for Japanese audio transcription.
Produces SRT subtitle files from audio input.

Based on: https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0#chunked-long-form

Usage:
    python reference_implementation_kotoba_transformers.py audio.wav
    python reference_implementation_kotoba_transformers.py video.mp4 --output subtitles.srt
    python reference_implementation_kotoba_transformers.py audio.mp3 --timestamps word
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import pipeline


# ============== Constants ==============

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus', '.aac'}

DEFAULT_MODEL_ID = "kotoba-tech/kotoba-whisper-v2.0"
DEFAULT_CHUNK_LENGTH = 15  # Optimal for distil-large-v3
DEFAULT_STRIDE = None  # None = use default (chunk_length_s / 6)
DEFAULT_BATCH_SIZE = 16
DEFAULT_LANGUAGE = "ja"
DEFAULT_TASK = "transcribe"

# Model capabilities for word timestamps
# Distilled models (2-layer decoder) don't support word timestamps
WORD_TIMESTAMP_SUPPORT = {
    "openai/whisper-large-v3": True,
    "openai/whisper-large-v3-turbo": True,
    "openai/whisper-large-v2": True,
    "kotoba-tech/kotoba-whisper-v2.0": False,  # Distilled, 2-layer decoder
    "kotoba-tech/kotoba-whisper-v2.1": False,
    "kotoba-tech/kotoba-whisper-v2.2": False,
    "kotoba-tech/kotoba-whisper-bilingual-v1.0": False,
}


# ============== Data Classes for Modular Pipeline ==============

@dataclass
class Segment:
    """A single subtitle segment with accurate timing."""
    index: int
    text: str
    start: float  # Accurate start time in seconds
    end: float    # Accurate end time (not overrun!)
    confidence: Optional[float] = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "confidence": self.confidence
        }


@dataclass
class ASRResult:
    """Raw output from ASR transcriber - saved for re-processing."""

    # Metadata
    audio_path: str
    model_id: str
    timestamp_type: str  # "segment" or "word"
    transcription_time: float

    # Full text
    full_text: str

    # Raw chunks from pipeline (preserved exactly)
    raw_chunks: List[Dict]  # [{"text": "...", "timestamp": (start, end)}, ...]

    # Word-level data (if available)
    words: Optional[List[Dict]] = None  # [{"word": "...", "start": 0.0, "end": 0.5}, ...]

    def save(self, path: Path):
        """Save raw result to JSON for later re-processing."""
        data = {
            "audio_path": str(self.audio_path),
            "model_id": self.model_id,
            "timestamp_type": self.timestamp_type,
            "transcription_time": self.transcription_time,
            "full_text": self.full_text,
            "raw_chunks": self.raw_chunks,
            "words": self.words
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Raw ASR saved to: {path}")

    @classmethod
    def load(cls, path: Path) -> "ASRResult":
        """Load saved raw result."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            audio_path=data["audio_path"],
            model_id=data["model_id"],
            timestamp_type=data["timestamp_type"],
            transcription_time=data["transcription_time"],
            full_text=data["full_text"],
            raw_chunks=data["raw_chunks"],
            words=data.get("words")
        )


@dataclass
class RefinedResult:
    """Refined/aligned output ready for SRT generation."""

    refiner_name: str  # Which refiner was used
    segments: List[Segment]  # Final aligned segments

    # Quality metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_srt(self) -> str:
        """Convert refined segments to SRT format."""
        srt_lines = []
        for seg in self.segments:
            srt_lines.append(str(seg.index))
            srt_lines.append(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}")
            srt_lines.append(seg.text)
            srt_lines.append("")
        return "\n".join(srt_lines)


# ============== Refiner Interface ==============

class BaseRefiner(ABC):
    """Base class for all refinement strategies."""

    name: str = "base"

    @abstractmethod
    def refine(self, raw: ASRResult) -> RefinedResult:
        """
        Take raw ASR output and produce refined segments.

        Args:
            raw: Raw ASR result with words/segments

        Returns:
            RefinedResult with accurately timed segments
        """
        pass

    def validate(self, result: RefinedResult) -> List[str]:
        """Validate refined result, return list of warnings."""
        warnings = []
        for seg in result.segments:
            if seg.duration <= 0:
                warnings.append(f"Segment {seg.index}: invalid duration {seg.duration:.2f}s")
            if seg.duration > 15:
                warnings.append(f"Segment {seg.index}: very long duration {seg.duration:.1f}s")
        return warnings


class PassthroughRefiner(BaseRefiner):
    """No refinement - just convert raw chunks to segments."""

    name = "none"

    def refine(self, raw: ASRResult) -> RefinedResult:
        """Pass through raw chunks without modification."""
        segments = []
        for idx, chunk in enumerate(raw.raw_chunks, 1):
            text = chunk.get("text", "").strip()
            timestamp = chunk.get("timestamp")

            if not text or not timestamp:
                continue

            start, end = timestamp
            if start is None:
                start = 0.0
            if end is None:
                end = start + 2.0

            segments.append(Segment(
                index=idx,
                text=text,
                start=float(start),
                end=float(end)
            ))

        return RefinedResult(
            refiner_name=self.name,
            segments=segments,
            metrics={"total_segments": len(segments)}
        )


class JapaneseRuleRefiner(BaseRefiner):
    """
    Enhanced Japanese linguistic segmentation with priority-based splitting.

    Features:
    - Priority-ordered break point detection
    - Sentence-final particles (終助詞): ね、よ、わ、ぞ、etc.
    - Polite/formal endings: ます、です、ました、でした
    - Imperative forms: ろ、せろ、しろ
    - Sentence starters: でも、しかし、だから
    - Hybrid fallback: char count OR duration threshold
    - Proportional timing for marker-less text
    """

    name = "japanese_rules"

    # ===== Japanese Linguistic Markers =====

    # Priority 1: Strong punctuation (definite sentence end)
    SENTENCE_ENDERS = {"。", "？", "！", "?", "!", "…"}

    # Priority 5: Clause markers (weak, needs context)
    CLAUSE_MARKERS = {"、", ","}

    # Priority 3: Sentence-final particles (終助詞)
    FINAL_PARTICLES = {
        "ね", "よ", "わ", "の", "さ", "な", "ぞ", "ぜ",
        "かな", "けど", "けれど", "から", "って", "ってば",
        "のに", "もの", "もん", "かしら",
    }

    # Priority 3: Polite/formal endings
    POLITE_ENDINGS = {
        "ます", "です", "ました", "でした",
        "ません", "ではない", "じゃない",
        "ございます", "いたします",
    }

    # Priority 3: Imperative endings (命令形)
    IMPERATIVE_ENDINGS = {
        "ろ", "れ", "せろ", "しろ", "けろ",
        "なさい", "ください", "てくれ", "てよ",
    }

    # Priority 4: Sentence starters (split BEFORE these)
    SENTENCE_STARTERS = {
        "でも", "しかし", "だから", "それで", "じゃあ", "まあ",
        "えっと", "ええと", "あの", "ほら", "ねえ", "おい",
        "それから", "そして", "だって", "なので", "ところで",
    }

    # Common interjections that indicate segment boundaries
    INTERJECTIONS = {
        "はい", "うん", "ええ", "ああ", "おお", "えー",
        "ごめん", "ごめんなさい", "すみません", "すいません",
        "ありがとう", "オーケー", "OK",
    }

    # ===== Configuration =====

    DEFAULT_CONFIG = {
        "max_segment_duration": 8.0,      # Hard limit - always split above this
        "force_split_duration": 5.0,      # Trigger evaluation at this duration
        "max_chars_per_segment": 40,      # Hard limit for character count
        "force_split_chars": 35,          # Trigger evaluation at this count
        "min_gap_for_break": 0.3,         # Gap threshold for sentence boundary
        "min_gap_for_clause": 0.15,       # Gap threshold for clause boundary
        "min_segment_duration": 0.5,      # Avoid flash subtitles
        "speech_rate_chars_per_sec": 8,   # Japanese speech rate estimate
    }

    def __init__(self, **kwargs):
        self.config = {**self.DEFAULT_CONFIG, **kwargs}

    def refine(self, raw: ASRResult) -> RefinedResult:
        """Apply Japanese rules to create natural segments."""

        if raw.words:
            segments = self._refine_from_words(raw.words)
        else:
            segments = self._refine_from_segments(raw.raw_chunks)

        metrics = self._calculate_metrics(segments)

        return RefinedResult(
            refiner_name=self.name,
            segments=segments,
            metrics=metrics
        )

    def _refine_from_words(self, words: List[Dict]) -> List[Segment]:
        """Refine using word-level timestamps."""
        segments = []
        current_words = []
        current_start = None
        current_end = None
        segment_index = 1

        for i, word_data in enumerate(words):
            word = word_data.get("text", word_data.get("word", "")).strip()
            timestamp = word_data.get("timestamp")

            if not word:
                continue

            if timestamp and isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                word_start, word_end = timestamp
            else:
                continue

            if word_start is None or word_end is None:
                continue

            gap_before = 0.0
            if current_end is not None:
                gap_before = word_start - current_end

            should_break = False

            if current_words:
                current_text = "".join(current_words)
                current_duration = word_end - current_start

                if self._should_break_after(current_text, gap_before):
                    should_break = True
                elif current_duration > self.config["max_segment_duration"]:
                    should_break = True
                elif len(current_text) >= self.config["max_chars_per_segment"]:
                    should_break = True
                elif self._is_sentence_starter(word):
                    should_break = True

            if should_break and current_words:
                segments.append(Segment(
                    index=segment_index,
                    text="".join(current_words),
                    start=current_start,
                    end=current_end
                ))
                segment_index += 1
                current_words = []
                current_start = None
                current_end = None

            current_words.append(word)
            if current_start is None:
                current_start = word_start
            current_end = word_end

        if current_words:
            segments.append(Segment(
                index=segment_index,
                text="".join(current_words),
                start=current_start,
                end=current_end
            ))

        return segments

    def _refine_from_segments(self, raw_chunks: List[Dict]) -> List[Segment]:
        """
        Refine segment-level timestamps with enhanced splitting.

        Uses hybrid threshold (char count OR duration) to trigger splitting,
        then applies priority-based break detection.
        """
        segments = []
        segment_index = 1

        for chunk in raw_chunks:
            text = chunk.get("text", "").strip()
            timestamp = chunk.get("timestamp")

            if not text or not timestamp:
                continue

            start, end = timestamp
            if start is None:
                start = 0.0
            if end is None:
                end = start + 2.0

            duration = end - start
            char_count = len(text)

            # Hybrid threshold check
            needs_split = (
                duration > self.config["force_split_duration"] or
                char_count > self.config["force_split_chars"]
            )

            if needs_split:
                sub_segments = self._split_long_segment(text, start, end)
                for sub_text, sub_start, sub_end in sub_segments:
                    if sub_text.strip():
                        segments.append(Segment(
                            index=segment_index,
                            text=sub_text.strip(),
                            start=sub_start,
                            end=sub_end
                        ))
                        segment_index += 1
            else:
                # Estimate better end time based on speech rate
                estimated_duration = char_count / self.config["speech_rate_chars_per_sec"]
                estimated_end = min(start + estimated_duration + 0.3, end)

                segments.append(Segment(
                    index=segment_index,
                    text=text,
                    start=start,
                    end=estimated_end
                ))
                segment_index += 1

        return segments

    def _split_long_segment(self, text: str, start: float, end: float) -> List[tuple]:
        """
        Split a long segment using priority-ordered break points.

        Priority order:
        1. Strong punctuation (。？！)
        2. Interjections (はい, ごめん, etc.)
        3. Sentence-final particles/polite endings
        4. Sentence starters (split BEFORE)
        5. Clause markers (、) - only if segment still too long
        6. Proportional fallback
        """
        duration = end - start
        char_count = len(text)

        if char_count == 0:
            return [(text, start, end)]

        # Find all break point candidates with priorities
        candidates = self._find_break_candidates(text)

        if not candidates:
            # No linguistic markers - use proportional split
            return self._proportional_split(text, start, end)

        # Select optimal split points
        selected = self._select_optimal_splits(text, candidates)

        if not selected:
            return self._proportional_split(text, start, end)

        # Calculate timing for each sub-segment
        return self._calculate_sub_timings(text, start, end, selected)

    def _find_break_candidates(self, text: str) -> List[tuple]:
        """
        Find all potential break points with priority scores.

        Returns: List of (position, priority, type) tuples
        Lower priority number = stronger break point
        """
        candidates = []
        text_len = len(text)

        for i in range(text_len):
            char = text[i]
            remaining = text[i+1:] if i+1 < text_len else ""
            prefix = text[:i+1]

            # Priority 1: Strong punctuation
            if char in self.SENTENCE_ENDERS:
                candidates.append((i + 1, 1, "punctuation"))
                continue

            # Priority 2: Interjections (check full words)
            for intj in self.INTERJECTIONS:
                if prefix.endswith(intj):
                    # Make sure it's not part of a longer word
                    pre_intj = prefix[:-len(intj)]
                    if not pre_intj or not self._is_continuation_char(pre_intj[-1]):
                        candidates.append((i + 1, 2, f"interjection:{intj}"))
                        break

            # Priority 3: Polite endings
            for ending in self.POLITE_ENDINGS:
                if prefix.endswith(ending):
                    candidates.append((i + 1, 3, f"polite:{ending}"))
                    break

            # Priority 3: Imperative endings
            for ending in self.IMPERATIVE_ENDINGS:
                if prefix.endswith(ending):
                    candidates.append((i + 1, 3, f"imperative:{ending}"))
                    break

            # Priority 3: Final particles
            for particle in self.FINAL_PARTICLES:
                if prefix.endswith(particle):
                    candidates.append((i + 1, 3, f"particle:{particle}"))
                    break

            # Priority 4: Sentence starters (split BEFORE)
            for starter in self.SENTENCE_STARTERS:
                if remaining.startswith(starter):
                    candidates.append((i + 1, 4, f"starter:{starter}"))
                    break

            # Priority 5: Clause markers
            if char in self.CLAUSE_MARKERS:
                candidates.append((i + 1, 5, "clause"))

        return candidates

    def _is_continuation_char(self, char: str) -> bool:
        """Check if character suggests word continuation (hiragana/katakana)."""
        if not char:
            return False
        code = ord(char)
        # Hiragana: 3040-309F, Katakana: 30A0-30FF
        return (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF)

    def _select_optimal_splits(self, text: str, candidates: List[tuple]) -> List[int]:
        """
        Select optimal split points to create balanced segments.

        Strategy:
        - Prefer high-priority breaks (lower number)
        - Aim for ~35 chars per segment
        - Ensure minimum segment length
        """
        if not candidates:
            return []

        text_len = len(text)
        target_len = self.config["force_split_chars"]
        min_len = 5  # Minimum segment length

        # Sort by position
        sorted_candidates = sorted(candidates, key=lambda x: x[0])

        selected = []
        last_split = 0

        for pos, priority, break_type in sorted_candidates:
            segment_len = pos - last_split
            remaining_len = text_len - pos

            # Skip if too close to last split
            if segment_len < min_len:
                continue

            # Skip if remaining would be too short (unless it's a high priority break)
            if remaining_len < min_len and priority > 2:
                continue

            # Accept high-priority breaks almost always
            if priority <= 2:
                selected.append(pos)
                last_split = pos
            # Accept medium priority if segment is getting long
            elif priority <= 3 and segment_len >= target_len * 0.7:
                selected.append(pos)
                last_split = pos
            # Accept low priority only if segment is too long
            elif segment_len >= target_len:
                selected.append(pos)
                last_split = pos

        return selected

    def _calculate_sub_timings(self, text: str, start: float, end: float, splits: List[int]) -> List[tuple]:
        """Calculate timing for each sub-segment based on character positions."""
        duration = end - start
        char_count = len(text)

        if char_count == 0:
            return [(text, start, end)]

        chars_per_second = char_count / duration if duration > 0 else self.config["speech_rate_chars_per_sec"]

        results = []
        prev_pos = 0

        for split_pos in splits:
            sub_text = text[prev_pos:split_pos]
            if sub_text.strip():
                sub_start = start + (prev_pos / chars_per_second) if chars_per_second > 0 else start
                sub_end = start + (split_pos / chars_per_second) if chars_per_second > 0 else end
                # Add small buffer but don't exceed chunk end
                sub_end = min(sub_end + 0.1, end)
                results.append((sub_text.strip(), sub_start, sub_end))
            prev_pos = split_pos

        # Add final segment
        if prev_pos < char_count:
            sub_text = text[prev_pos:]
            if sub_text.strip():
                sub_start = start + (prev_pos / chars_per_second) if chars_per_second > 0 else start
                results.append((sub_text.strip(), sub_start, end))

        return results if results else [(text, start, end)]

    def _proportional_split(self, text: str, start: float, end: float) -> List[tuple]:
        """
        Last-resort: split at character intervals when no linguistic markers found.
        """
        duration = end - start
        char_count = len(text)
        target_chars = self.config["force_split_chars"]

        if char_count <= target_chars:
            return [(text, start, end)]

        # Calculate number of segments needed
        num_segments = (char_count + target_chars - 1) // target_chars
        chars_per_segment = char_count // num_segments

        results = []
        chars_per_second = char_count / duration if duration > 0 else self.config["speech_rate_chars_per_sec"]

        for i in range(num_segments):
            seg_start_char = i * chars_per_segment
            seg_end_char = min((i + 1) * chars_per_segment, char_count)

            # Extend last segment to include remaining chars
            if i == num_segments - 1:
                seg_end_char = char_count

            sub_text = text[seg_start_char:seg_end_char]
            sub_start = start + (seg_start_char / chars_per_second) if chars_per_second > 0 else start
            sub_end = start + (seg_end_char / chars_per_second) if chars_per_second > 0 else end

            if i == num_segments - 1:
                sub_end = end

            if sub_text.strip():
                results.append((sub_text.strip(), sub_start, sub_end))

        return results if results else [(text, start, end)]

    def _should_break_after(self, text: str, gap_after: float) -> bool:
        """Priority-based break detection for word-level refinement."""
        if not text:
            return False

        # Priority 1: Strong punctuation
        if text[-1] in self.SENTENCE_ENDERS:
            return True

        # Priority 2: Significant gap (speaker pause)
        if gap_after >= self.config["min_gap_for_break"]:
            return True

        # Priority 3: Polite endings
        for ending in self.POLITE_ENDINGS:
            if text.endswith(ending):
                return True

        # Priority 3: Final particles (with small gap)
        for particle in self.FINAL_PARTICLES:
            if text.endswith(particle) and gap_after >= 0.1:
                return True

        # Priority 5: Clause marker with pause
        if text[-1] in self.CLAUSE_MARKERS:
            if gap_after >= self.config["min_gap_for_clause"]:
                return True

        return False

    def _is_sentence_starter(self, text: str) -> bool:
        """Check if text starts a new sentence."""
        for starter in self.SENTENCE_STARTERS:
            if text.startswith(starter):
                return True
        return False

    def _calculate_metrics(self, segments: List[Segment]) -> Dict[str, Any]:
        """Calculate quality metrics for refined output."""
        if not segments:
            return {"total_segments": 0}

        durations = [seg.duration for seg in segments]
        gaps = []
        for i in range(1, len(segments)):
            gap = segments[i].start - segments[i-1].end
            gaps.append(gap)

        long_segments = [d for d in durations if d > self.config["max_segment_duration"]]

        return {
            "total_segments": len(segments),
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "avg_gap": sum(gaps) / len(gaps) if gaps else 0,
            "total_gaps": len([g for g in gaps if g > 0.1]),
            "long_segments": len(long_segments),
        }


# ============== Audio Extraction ==============

def is_video_file(path: Path) -> bool:
    """Check if file is a video format."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_audio_file(path: Path) -> bool:
    """Check if file is an audio format."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """
    Extract audio from video using ffmpeg.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted audio

    Returns:
        Path to extracted WAV file
    """
    output_path = output_dir / f"{video_path.stem}_extracted.wav"

    print(f"  Extracting audio from video...")
    print(f"  Input:  {video_path}")
    print(f"  Output: {output_path}")

    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vn',                    # No video
        '-acodec', 'pcm_s16le',   # PCM 16-bit
        '-ar', '16000',           # 16kHz sample rate
        '-ac', '1',               # Mono
        '-y',                     # Overwrite
        str(output_path)
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            text=True
        )
        print(f"  Audio extracted successfully!")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: ffmpeg failed: {e.stderr}", file=sys.stderr)
        raise RuntimeError(f"Failed to extract audio: {e.stderr}")
    except FileNotFoundError:
        print("  ERROR: ffmpeg not found. Please install ffmpeg.", file=sys.stderr)
        raise RuntimeError("ffmpeg not found in PATH")


# ============== SRT Formatting ==============

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds (float)

    Returns:
        SRT formatted timestamp string
    """
    if seconds is None:
        seconds = 0.0

    # Handle negative values
    seconds = max(0.0, seconds)

    # Calculate components
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def chunks_to_srt(chunks: list) -> str:
    """
    Convert pipeline chunks to SRT format string.

    Args:
        chunks: List of dicts with 'text' and 'timestamp' keys

    Returns:
        SRT formatted string
    """
    srt_lines = []
    subtitle_index = 1

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        timestamp = chunk.get("timestamp")

        # Skip empty segments
        if not text:
            continue

        # Handle timestamp tuple
        if timestamp and isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
            start, end = timestamp
            # Handle None values
            if start is None:
                start = 0.0
            if end is None:
                # Estimate end time if missing
                end = start + 2.0
        else:
            # Skip segments without valid timestamps
            continue

        # Add SRT entry
        srt_lines.append(str(subtitle_index))
        srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        srt_lines.append(text)
        srt_lines.append("")  # Blank line between entries

        subtitle_index += 1

    return "\n".join(srt_lines)


# ============== Device & Dtype Detection ==============

def detect_device(requested: str) -> str:
    """
    Detect best available device.

    Args:
        requested: 'auto', 'cuda', or 'cpu'

    Returns:
        Device string for PyTorch
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"
    elif requested == "cuda":
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    else:
        return "cpu"


def detect_dtype(requested: str, device: str) -> torch.dtype:
    """
    Detect best dtype for device.

    Args:
        requested: 'auto', 'float16', 'bfloat16', or 'float32'
        device: Target device string

    Returns:
        PyTorch dtype
    """
    if requested == "auto":
        if "cuda" in device:
            # bfloat16 is generally better for modern GPUs
            return torch.bfloat16
        else:
            return torch.float32
    elif requested == "float16":
        return torch.float16
    elif requested == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float32


# ============== Pipeline Creation ==============

def create_pipeline(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    attn_implementation: str,
    batch_size: int
):
    """
    Create and configure the ASR pipeline.

    Args:
        model_id: HuggingFace model ID
        device: Target device
        torch_dtype: Data type for model
        attn_implementation: Attention implementation (sdpa, flash_attention_2, eager)
        batch_size: Batch size for chunk processing

    Returns:
        Configured pipeline object
    """
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    print(f"  Model:    {model_id}")
    print(f"  Device:   {device}")
    print(f"  Dtype:    {torch_dtype}")
    print(f"  Attention: {attn_implementation}")
    print(f"  Batch:    {batch_size}")

    # Build model_kwargs
    model_kwargs = {}
    if "cuda" in device and attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    start_time = time.time()

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs if model_kwargs else None,
        batch_size=batch_size
    )

    load_time = time.time() - start_time
    print(f"  Loaded in {load_time:.1f}s")

    return pipe


# ============== Transcription ==============

def transcribe(
    audio_path: Path,
    pipe,
    chunk_length_s: int,
    stride_length_s: float,
    language: str,
    task: str,
    timestamp_type: str,
    beam_size: int = 5,
    temperature: float = 0.0,
    compression_ratio_threshold: float = 2.4,
    logprob_threshold: float = -1.0,
    no_speech_threshold: float = 0.6,
    condition_on_previous: bool = True
) -> dict:
    """
    Run transcription with chunked long-form algorithm.

    Args:
        audio_path: Path to audio file
        pipe: ASR pipeline
        chunk_length_s: Chunk length in seconds
        stride_length_s: Overlap between chunks (handles speech at boundaries)
        language: Language code (e.g., 'ja')
        task: 'transcribe' or 'translate'
        timestamp_type: 'segment' or 'word'
        beam_size: Beam size for decoding (higher = more accurate)
        temperature: Sampling temperature (0 = deterministic)
        compression_ratio_threshold: Filter high compression segments
        logprob_threshold: Filter low confidence segments
        no_speech_threshold: Threshold for non-speech detection
        condition_on_previous: Condition on previous text for coherence

    Returns:
        Pipeline result dict with 'text' and 'chunks'
    """
    # Calculate effective stride
    if stride_length_s is None:
        stride_length_s = chunk_length_s / 6  # Default: 1/6 of chunk length

    print(f"\n{'='*60}")
    print("TRANSCRIPTION")
    print(f"{'='*60}")
    print(f"  Audio:     {audio_path.name}")
    print(f"  Chunk:     {chunk_length_s}s")
    print(f"  Stride:    {stride_length_s:.1f}s (overlap at boundaries)")
    print(f"  Language:  {language}")
    print(f"  Task:      {task}")
    print(f"  Timestamps: {timestamp_type}")
    print(f"  Beam size: {beam_size}")
    print(f"  Temperature: {temperature}")

    # Get audio duration for progress estimation
    duration = get_audio_duration(audio_path)
    if duration > 0:
        estimated_chunks = int(duration / chunk_length_s) + 1
        print(f"  Duration:  {duration:.1f}s (~{estimated_chunks} chunks)")

    # Configure generate_kwargs with accuracy parameters
    generate_kwargs = {
        "language": language,
        "task": task,
        "num_beams": beam_size,
        "temperature": temperature,
        "compression_ratio_threshold": compression_ratio_threshold,
        "logprob_threshold": logprob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "condition_on_prev_tokens": condition_on_previous,
    }

    # Configure return_timestamps
    if timestamp_type == "word":
        return_timestamps = "word"
    else:
        return_timestamps = True  # Segment-level

    print(f"\n  Processing...")
    start_time = time.time()

    result = pipe(
        str(audio_path),
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        return_timestamps=return_timestamps,
        generate_kwargs=generate_kwargs
    )

    process_time = time.time() - start_time

    # Print stats
    print(f"\n  Completed in {process_time:.1f}s")
    if duration > 0:
        rtf = process_time / duration
        print(f"  Real-time factor: {rtf:.2f}x")

    num_chunks = len(result.get("chunks", []))
    print(f"  Segments: {num_chunks}")

    return result


# ============== Output ==============

def print_config_summary(args, device: str, torch_dtype: torch.dtype):
    """Print configuration summary."""
    stride_display = f"{args.stride}s" if args.stride else f"auto ({args.chunk_length/6:.1f}s)"

    print(f"\n{'='*60}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Input:      {args.input}")
    print(f"  Output:     {args.output}")
    print(f"  Model:      {args.model_id}")
    print(f"  Device:     {device}")
    print(f"  Dtype:      {torch_dtype}")
    print(f"  Chunk:      {args.chunk_length}s")
    print(f"  Stride:     {stride_display}")
    print(f"  Batch:      {args.batch_size}")
    print(f"  Language:   {args.language}")
    print(f"  Task:       {args.task}")
    print(f"  Timestamps: {args.timestamps}")
    if args.attn:
        print(f"  Attention:  {args.attn}")


def print_final_stats(total_time: float, input_path: Path, output_path: Path, num_segments: int):
    """Print final statistics."""
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Segments:   {num_segments}")
    print(f"  Output:     {output_path}")

    # Show output file size
    if output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        print(f"  Size:       {size_kb:.1f} KB")


# ============== Refiner Selection ==============

def get_refiner(name: str) -> BaseRefiner:
    """Get refiner instance by name."""
    refiners = {
        "none": PassthroughRefiner,
        "japanese_rules": JapaneseRuleRefiner,
    }
    if name not in refiners:
        raise ValueError(f"Unknown refiner: {name}. Available: {list(refiners.keys())}")
    return refiners[name]()


# ============== Main ==============

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HuggingFace Transformers Chunked Long-Form ASR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.wav
  %(prog)s video.mp4 --output subtitles.srt
  %(prog)s audio.mp3 --model-id openai/whisper-large-v3
  %(prog)s long_audio.wav --timestamps word --batch-size 1
        """
    )

    # Positional
    parser.add_argument(
        "input",
        type=Path,
        help="Audio or video file to transcribe"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output SRT file path (default: <input>.srt)"
    )

    # Model
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})"
    )

    # Processing
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=DEFAULT_CHUNK_LENGTH,
        help=f"Chunk length in seconds (default: {DEFAULT_CHUNK_LENGTH})"
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=DEFAULT_STRIDE,
        help="Stride/overlap in seconds (default: chunk_length/6). Handles speech at chunk boundaries."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for parallel processing (default: {DEFAULT_BATCH_SIZE})"
    )

    # Language
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE})"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["transcribe", "translate"],
        default=DEFAULT_TASK,
        help=f"Task type (default: {DEFAULT_TASK})"
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Data type (default: auto)"
    )
    parser.add_argument(
        "--attn",
        type=str,
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
        help="Attention implementation (default: sdpa)"
    )

    # Timestamps
    parser.add_argument(
        "--timestamps",
        type=str,
        choices=["segment", "word"],
        default="segment",
        help="Timestamp granularity (default: segment)"
    )

    # Accuracy tuning
    accuracy_group = parser.add_argument_group("Accuracy Tuning")
    accuracy_group.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (higher = more accurate, slower). Default: 5"
    )
    accuracy_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0 = greedy/deterministic). Default: 0.0"
    )
    accuracy_group.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=2.4,
        help="Threshold for filtering high compression ratio segments. Default: 2.4"
    )
    accuracy_group.add_argument(
        "--logprob-threshold",
        type=float,
        default=-1.0,
        help="Log probability threshold for filtering low confidence. Default: -1.0"
    )
    accuracy_group.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.6,
        help="Threshold for detecting non-speech segments. Default: 0.6"
    )
    accuracy_group.add_argument(
        "--condition-on-previous",
        action="store_true",
        default=True,
        help="Condition on previous text (improves coherence). Default: True"
    )

    # Refinement options (modular pipeline)
    refine_group = parser.add_argument_group("Refinement (Modular Pipeline)")
    refine_group.add_argument(
        "--refiner",
        type=str,
        choices=["none", "japanese_rules"],
        default="japanese_rules",
        help="Refinement strategy for timestamp alignment. Default: japanese_rules"
    )
    refine_group.add_argument(
        "--save-raw",
        action="store_true",
        default=False,
        help="Save raw ASR output to JSON for re-processing"
    )
    refine_group.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Path to save raw ASR output (default: <input>_raw.json)"
    )
    refine_group.add_argument(
        "--refine-only",
        type=Path,
        default=None,
        help="Skip ASR, refine from saved raw JSON file"
    )

    return parser.parse_args()


def main():
    """Main entry point with modular refinement pipeline."""
    args = parse_args()
    total_start = time.time()

    # Set default output path
    if args.output is None:
        args.output = args.input.with_suffix(".srt")

    # ========== REFINE-ONLY MODE ==========
    # Skip ASR entirely, load from saved JSON
    if args.refine_only:
        print(f"\n{'='*60}")
        print("REFINE-ONLY MODE")
        print(f"{'='*60}")

        if not args.refine_only.exists():
            print(f"ERROR: Raw JSON file not found: {args.refine_only}", file=sys.stderr)
            sys.exit(1)

        print(f"  Loading raw ASR from: {args.refine_only}")
        raw_result = ASRResult.load(args.refine_only)
        print(f"  Model: {raw_result.model_id}")
        print(f"  Timestamp type: {raw_result.timestamp_type}")
        print(f"  Chunks: {len(raw_result.raw_chunks)}")

        # Apply refinement
        refiner = get_refiner(args.refiner)
        print(f"\n  Applying refiner: {refiner.name}")
        refined = refiner.refine(raw_result)

        # Validate
        warnings = refiner.validate(refined)
        if warnings:
            print(f"\n  Validation warnings:")
            for w in warnings[:5]:  # Show first 5
                print(f"    - {w}")

        # Write SRT
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(refined.to_srt(), encoding="utf-8")

        # Print metrics
        print(f"\n{'='*60}")
        print("REFINEMENT COMPLETE")
        print(f"{'='*60}")
        print(f"  Refiner:    {refined.refiner_name}")
        print(f"  Segments:   {refined.metrics.get('total_segments', 0)}")
        print(f"  Avg duration: {refined.metrics.get('avg_duration', 0):.2f}s")
        print(f"  Output:     {args.output}")

        total_time = time.time() - total_start
        print(f"  Total time: {total_time:.1f}s")
        return

    # ========== NORMAL MODE: ASR + REFINEMENT ==========

    # Validate input file
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Word-level timestamps require batch_size=1
    if args.timestamps == "word" and args.batch_size > 1:
        print(f"NOTE: Word-level timestamps require batch_size=1. Adjusting from {args.batch_size} to 1.")
        args.batch_size = 1

    # Check word timestamp support for model
    if args.timestamps == "word":
        supports_word = WORD_TIMESTAMP_SUPPORT.get(args.model_id)
        if supports_word is False:
            print(f"\nWARNING: Model '{args.model_id}' may not support word-level timestamps.")
            print("  Distilled models have 2-layer decoders (vs 12 in original).")
            print("  This may cause IndexError in _extract_token_timestamps.")
            print("  Consider using: openai/whisper-large-v3-turbo")
            print("  Falling back to segment timestamps.\n")
            args.timestamps = "segment"

    # Detect device and dtype
    device = detect_device(args.device)
    torch_dtype = detect_dtype(args.dtype, device)

    # Print configuration
    print_config_summary(args, device, torch_dtype)

    # Handle video files
    audio_path = args.input
    temp_dir = None

    if is_video_file(args.input):
        print(f"\n{'='*60}")
        print("VIDEO PROCESSING")
        print(f"{'='*60}")
        temp_dir = Path(tempfile.mkdtemp(prefix="kotoba_asr_"))
        try:
            audio_path = extract_audio(args.input, temp_dir)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    elif not is_audio_file(args.input):
        print(f"WARNING: Unknown file type: {args.input.suffix}")
        print("  Attempting to process as audio...")

    try:
        # Create pipeline
        pipe = create_pipeline(
            model_id=args.model_id,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn,
            batch_size=args.batch_size
        )

        # Run transcription
        transcribe_start = time.time()
        result = transcribe(
            audio_path=audio_path,
            pipe=pipe,
            chunk_length_s=args.chunk_length,
            stride_length_s=args.stride,
            language=args.language,
            task=args.task,
            timestamp_type=args.timestamps,
            beam_size=args.beam_size,
            temperature=args.temperature,
            compression_ratio_threshold=args.compression_ratio_threshold,
            logprob_threshold=args.logprob_threshold,
            no_speech_threshold=args.no_speech_threshold,
            condition_on_previous=args.condition_on_previous
        )
        transcribe_time = time.time() - transcribe_start

        # ========== BUILD ASRResult ==========
        chunks = result.get("chunks", [])

        raw_result = ASRResult(
            audio_path=str(audio_path),
            model_id=args.model_id,
            timestamp_type=args.timestamps,
            transcription_time=transcribe_time,
            full_text=result.get("text", ""),
            raw_chunks=chunks,
            words=None  # Transformers pipeline puts word data in chunks for word mode
        )

        # Save raw output if requested
        if args.save_raw:
            raw_path = args.raw_output or args.input.parent / f"{args.input.stem}_raw.json"
            raw_result.save(raw_path)

        # ========== APPLY REFINEMENT ==========
        print(f"\n{'='*60}")
        print("REFINEMENT")
        print(f"{'='*60}")

        refiner = get_refiner(args.refiner)
        print(f"  Refiner: {refiner.name}")

        refined = refiner.refine(raw_result)

        # Validate and warn
        warnings = refiner.validate(refined)
        if warnings:
            print(f"  Warnings: {len(warnings)}")
            for w in warnings[:3]:
                print(f"    - {w}")

        # Print metrics
        print(f"  Segments: {refined.metrics.get('total_segments', 0)}")
        print(f"  Avg duration: {refined.metrics.get('avg_duration', 0):.2f}s")
        print(f"  Max duration: {refined.metrics.get('max_duration', 0):.2f}s")

        # ========== OUTPUT ==========
        srt_content = refined.to_srt()

        # Write output
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(srt_content, encoding="utf-8")

        # Print final stats
        total_time = time.time() - total_start
        print_final_stats(total_time, args.input, args.output, len(refined.segments))

    finally:
        # Cleanup temp files
        if temp_dir and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
