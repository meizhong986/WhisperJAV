#!/usr/bin/env python3
"""
Manifold Surgical CTC - Reference Implementation

A proof-of-concept gap-filler subsystem that uses CTC-based ASR models
to recover speech that Whisper systematically misses in challenging audio.

Design Philosophy:
- This is NOT an ASR replacement - it's a "speech recall probe"
- Operates ONLY on gaps where Whisper failed
- Uses enhanced audio (post speech-enhancement)
- Never overrides Whisper output, only fills silence
- Precision > Recall (aggressive filtering)

The "manifold surgical" name reflects:
- Operating on a different acoustic manifold (enhanced vs raw)
- Surgical precision in targeting specific gaps
- CTC models tolerate enhancement artifacts better than Whisper

Usage:
    python surgical_ctc_reference_implementation.py \
        --source-audio INPUT/source_audio.wav \
        --whisper-srt INPUT/whisper_transcript.srt \
        --enhancer clearvoice \
        --ctc-model facebook/mms-1b-all \
        --output-dir OUTPUT/results

Author: WhisperJAV Project
License: MIT
"""

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Protocol, runtime_checkable
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Gap:
    """Represents a temporal gap in the Whisper transcript."""
    start: float  # seconds
    end: float    # seconds
    duration: float = field(init=False)
    source: str = "auto"  # "auto" (from SRT analysis) or "manual"

    def __post_init__(self):
        self.duration = self.end - self.start


@dataclass
class CTCResult:
    """Result from CTC transcription of a gap."""
    gap: Gap
    raw_text: str
    filtered_text: Optional[str] = None
    confidence: Optional[float] = None
    was_filtered: bool = False
    filter_reason: Optional[str] = None
    enhanced_audio_path: Optional[str] = None


@dataclass
class EvaluationReport:
    """Summary of the surgical CTC run."""
    total_gaps_detected: int = 0
    gaps_processed: int = 0
    gaps_with_output: int = 0
    gaps_filtered_out: int = 0
    total_duration_seconds: float = 0.0
    ctc_model_used: str = ""
    enhancer_used: str = ""
    filter_settings: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CTC Backend Protocol (for swappable backends)
# =============================================================================

@runtime_checkable
class CTCBackend(Protocol):
    """Protocol for CTC ASR backends."""

    def transcribe(self, audio_path: Path, language: str = "ja") -> Tuple[str, Optional[float]]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file (WAV, 16kHz mono)
            language: Language code

        Returns:
            Tuple of (transcription_text, confidence_score)
        """
        ...

    def cleanup(self) -> None:
        """Release resources."""
        ...


# =============================================================================
# CTC Backend Implementations
# =============================================================================

class MMSBackend:
    """
    MMS (Massively Multilingual Speech) CTC backend.

    Uses facebook/mms-1b-all (the full multilingual model with language adapters).

    Why MMS for gap-filling:
    - Weak language model = doesn't hallucinate structure
    - High phonetic recall = catches fragments
    - Trained on augmented audio = tolerates enhancement artifacts

    Note: facebook/mms-300m is the base pretrained model WITHOUT language adapters.
          Use facebook/mms-1b-all for actual ASR with Japanese support.
    """

    # Map of common model IDs to the correct MMS model
    MODEL_ALIASES = {
        "facebook/mms-300m": "facebook/mms-1b-all",  # Redirect to full model
        "mms-300m": "facebook/mms-1b-all",
        "mms": "facebook/mms-1b-all",
    }

    def __init__(self, model_id: str = "facebook/mms-1b-all", device: str = "auto"):
        # Redirect common mistakes to correct model
        if model_id in self.MODEL_ALIASES:
            logger.warning(
                f"Model '{model_id}' redirected to 'facebook/mms-1b-all' "
                f"(the base model doesn't have language adapters)"
            )
            model_id = self.MODEL_ALIASES[model_id]

        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None
        self._torch_device = None

    def _resolve_device(self):
        """Get torch device."""
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _load_model(self):
        """Lazy load the MMS model with Japanese adapter."""
        if self._model is not None:
            return

        logger.info(f"Loading MMS model: {self.model_id}")

        import torch
        from transformers import Wav2Vec2ForCTC, AutoProcessor

        self._torch_device = self._resolve_device()

        # Load processor and model
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id)

        # Load Japanese language adapter
        try:
            self._processor.tokenizer.set_target_lang("jpn")
            self._model.load_adapter("jpn")
            logger.info("Loaded Japanese adapter for MMS")
        except Exception as e:
            logger.error(f"Failed to load Japanese adapter: {e}")
            raise RuntimeError(
                f"Could not load Japanese adapter for MMS. "
                f"Make sure you're using 'facebook/mms-1b-all' which has language adapters."
            )

        self._model = self._model.to(self._torch_device)
        self._model.eval()

    def transcribe(self, audio_path: Path, language: str = "ja") -> Tuple[str, Optional[float]]:
        """Transcribe audio using MMS with confidence estimation."""
        self._load_model()

        import torch
        import librosa

        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

            # Process audio
            inputs = self._processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )

            input_values = inputs.input_values.to(self._torch_device)

            with torch.no_grad():
                outputs = self._model(input_values)
                logits = outputs.logits

            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            text = self._processor.decode(predicted_ids[0])

            # Compute confidence from softmax probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values.squeeze()

            # Filter out very low probability frames (likely blank/padding)
            if max_probs.dim() == 0:
                # Single frame case
                confidence = float(max_probs.cpu())
            else:
                significant_probs = max_probs[max_probs > 0.1]
                if len(significant_probs) > 0:
                    confidence = float(significant_probs.mean().cpu())
                else:
                    confidence = float(max_probs.mean().cpu())

            return text.strip(), confidence

        except Exception as e:
            logger.error(f"MMS transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return "", None

    def cleanup(self):
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ReazonSpeechBackend:
    """
    ReazonSpeech Wav2Vec2 CTC backend.

    Uses reazon-research/reazonspeech-wav2vec2-large-rs35kh.

    Why ReazonSpeech:
    - Trained on 35,000 hours of Japanese audio
    - CER 11% on Japanese benchmarks
    - CTC architecture = enhancement tolerant
    """

    def __init__(self, model_id: str = "reazon-research/reazonspeech-wav2vec2-large-rs35kh", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _resolve_device(self):
        """Get torch device."""
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        logger.info(f"Loading ReazonSpeech model: {self.model_id}")

        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        device = self._resolve_device()

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id).to(device)
        self._model.eval()

    def transcribe(self, audio_path: Path, language: str = "ja") -> Tuple[str, Optional[float]]:
        """Transcribe audio using ReazonSpeech."""
        self._load_model()

        import torch
        import numpy as np

        try:
            # Load audio
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

            # ReazonSpeech requires 0.5s padding
            audio = np.pad(audio, pad_width=int(0.5 * 16000))

            device = self._resolve_device()

            # Process
            inputs = self._processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )

            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                logits = self._model(input_values).logits

            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            text = self._processor.decode(predicted_ids[0])

            # Compute confidence from logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            confidence = float(max_probs.mean().cpu())

            return text.strip(), confidence

        except Exception as e:
            logger.error(f"ReazonSpeech transcription failed: {e}")
            return "", None

    def cleanup(self):
        """Release resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Gap Detection
# =============================================================================

def parse_srt_timestamp(ts: str) -> float:
    """Convert SRT timestamp to seconds."""
    # Format: HH:MM:SS,mmm
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", ts.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {ts}")

    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def parse_srt(srt_path: Path) -> List[Tuple[float, float, str]]:
    """
    Parse SRT file into list of (start, end, text) tuples.

    Handles various SRT formatting quirks:
    - Windows/Unix line endings
    - Extra whitespace around timestamps
    - Missing sequence numbers
    - Empty text blocks
    """
    segments = []

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Split into subtitle blocks (one or more blank lines)
    blocks = re.split(r"\n{2,}", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        # Find timestamp line and its index
        timestamp_idx = None
        timestamp_line = None
        for idx, line in enumerate(lines):
            if "-->" in line:
                timestamp_idx = idx
                timestamp_line = line
                break

        if timestamp_line is None:
            continue

        # Parse timestamps
        parts = timestamp_line.split("-->")
        if len(parts) != 2:
            continue

        try:
            start = parse_srt_timestamp(parts[0])
            end = parse_srt_timestamp(parts[1])
        except ValueError:
            continue

        # Get text: everything after the timestamp line
        text_lines = lines[timestamp_idx + 1:]
        text = "\n".join(text_lines).strip()

        segments.append((start, end, text))

    return sorted(segments, key=lambda x: x[0])


def _split_long_gap(start: float, end: float, max_gap: float, min_gap: float) -> List[Gap]:
    """
    Split a long gap into chunks of max_gap duration.

    Args:
        start: Gap start time
        end: Gap end time
        max_gap: Maximum chunk duration
        min_gap: Minimum chunk duration (final chunk must meet this)

    Returns:
        List of Gap objects
    """
    gaps = []
    current = start

    while current < end:
        chunk_end = min(current + max_gap, end)
        chunk_duration = chunk_end - current

        # Only add if meets minimum duration
        if chunk_duration >= min_gap:
            gaps.append(Gap(start=current, end=chunk_end, source="auto"))

        current = chunk_end

    return gaps


def detect_gaps_from_srt(
    srt_path: Path,
    audio_duration: float,
    min_gap: float = 0.3,
    max_gap: float = 3.0,
    split_long_gaps: bool = True
) -> List[Gap]:
    """
    Detect gaps in Whisper transcript where speech may have been missed.

    Args:
        srt_path: Path to Whisper's SRT output
        audio_duration: Total audio duration in seconds
        min_gap: Minimum gap duration to consider (seconds)
        max_gap: Maximum gap duration to consider (seconds)
        split_long_gaps: If True, split gaps longer than max_gap into chunks.
                        If False, ignore gaps longer than max_gap entirely.

    Returns:
        List of Gap objects

    Behavior:
        - Gaps shorter than min_gap are ignored
        - Gaps longer than max_gap are split into max_gap chunks (if split_long_gaps=True)
        - If SRT has no segments, entire audio is treated as gaps and chunked
    """
    segments = parse_srt(srt_path)
    gaps = []

    if not segments:
        # No segments = entire audio is potential gap(s)
        # Split into chunks if longer than max_gap
        logger.warning("SRT has no segments - treating entire audio as gap(s)")
        if split_long_gaps and audio_duration > max_gap:
            gaps.extend(_split_long_gap(0.0, audio_duration, max_gap, min_gap))
        elif audio_duration >= min_gap:
            gaps.append(Gap(start=0.0, end=min(audio_duration, max_gap), source="auto"))
        return gaps

    # Gap before first segment
    first_start = segments[0][0]
    if first_start >= min_gap:
        if split_long_gaps and first_start > max_gap:
            gaps.extend(_split_long_gap(0.0, first_start, max_gap, min_gap))
        elif first_start <= max_gap:
            gaps.append(Gap(start=0.0, end=first_start, source="auto"))

    # Gaps between segments
    for i in range(len(segments) - 1):
        current_end = segments[i][1]
        next_start = segments[i + 1][0]
        gap_duration = next_start - current_end

        if gap_duration >= min_gap:
            if split_long_gaps and gap_duration > max_gap:
                # Split long mid-file gaps into chunks
                gaps.extend(_split_long_gap(current_end, next_start, max_gap, min_gap))
            elif gap_duration <= max_gap:
                gaps.append(Gap(start=current_end, end=next_start, source="auto"))
            # If not splitting and gap > max_gap, it's skipped (original behavior)

    # Gap after last segment
    last_end = segments[-1][1]
    trailing_duration = audio_duration - last_end
    if trailing_duration >= min_gap:
        if split_long_gaps and trailing_duration > max_gap:
            gaps.extend(_split_long_gap(last_end, audio_duration, max_gap, min_gap))
        elif trailing_duration <= max_gap:
            gaps.append(Gap(start=last_end, end=audio_duration, source="auto"))

    logger.info(f"Detected {len(gaps)} gaps from SRT analysis")
    return gaps


def parse_manual_gaps(gaps_path: Path) -> List[Gap]:
    """
    Parse manually specified gaps file.

    Format (SRT-like):
        1
        00:00:05,000 --> 00:00:08,500

        2
        00:01:23,000 --> 00:01:25,000

    Or simple format:
        5.0 - 8.5
        83.0 - 85.0
    """
    gaps = []

    with open(gaps_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try SRT format first
    if "-->" in content:
        blocks = re.split(r"\n\s*\n", content.strip())
        for block in blocks:
            for line in block.split("\n"):
                if "-->" in line:
                    parts = line.split("-->")
                    if len(parts) == 2:
                        try:
                            start = parse_srt_timestamp(parts[0])
                            end = parse_srt_timestamp(parts[1])
                            gaps.append(Gap(start=start, end=end, source="manual"))
                        except ValueError:
                            continue
    else:
        # Simple format: start - end
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = re.match(r"([\d.]+)\s*[-–]\s*([\d.]+)", line)
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                gaps.append(Gap(start=start, end=end, source="manual"))

    logger.info(f"Parsed {len(gaps)} manual gaps")
    return gaps


# =============================================================================
# Audio Processing
# =============================================================================

def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Failed to get audio duration: {e}")
        return 0.0


def extract_segment(
    audio_path: Path,
    start: float,
    end: float,
    output_path: Path,
    padding: float = 0.0
) -> bool:
    """
    Extract audio segment using ffmpeg.

    Args:
        audio_path: Source audio file
        start: Start time in seconds
        end: End time in seconds
        output_path: Output WAV file path
        padding: Padding to add on both sides (seconds)

    Returns:
        True if successful
    """
    # Apply padding
    padded_start = max(0, start - padding)
    padded_end = end + padding
    duration = padded_end - padded_start

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-ss", str(padded_start),
        "-t", str(duration),
        "-ac", "1",           # Mono
        "-ar", "16000",       # 16kHz
        "-acodec", "pcm_s16le",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg extraction failed: {e.stderr.decode()}")
        return False


def enhance_audio(
    audio_path: Path,
    output_path: Path,
    enhancer_name: str,
    enhancer_model: Optional[str] = None
) -> bool:
    """
    Apply speech enhancement to audio segment.

    Args:
        audio_path: Input audio file
        output_path: Output enhanced audio file
        enhancer_name: Enhancer backend name
        enhancer_model: Optional model variant

    Returns:
        True if successful
    """
    if enhancer_name == "none":
        # Just copy the file
        import shutil
        shutil.copy(audio_path, output_path)
        return True

    try:
        import librosa
        import soundfile as sf

        # Import WhisperJAV's speech enhancement
        from whisperjav.modules.speech_enhancement.factory import SpeechEnhancerFactory

        # Check availability
        available, hint = SpeechEnhancerFactory.is_backend_available(enhancer_name)
        if not available:
            logger.error(f"Enhancer '{enhancer_name}' not available: {hint}")
            return False

        # Create enhancer
        kwargs = {}
        if enhancer_model:
            kwargs["model"] = enhancer_model

        enhancer = SpeechEnhancerFactory.create(enhancer_name, **kwargs)

        # Get the preferred sample rate for this enhancer
        preferred_sr = enhancer.get_preferred_sample_rate()

        # Load audio at the enhancer's preferred sample rate
        audio_array, sr = librosa.load(str(audio_path), sr=preferred_sr, mono=True)

        # Enhance - backends expect (audio_array, sample_rate)
        result = enhancer.enhance(audio_array, sr)

        if result.success:
            # Save enhanced audio to output path
            # Resample to 16kHz if needed (for CTC input)
            output_sr = 16000
            if result.sample_rate != output_sr:
                enhanced_audio = librosa.resample(
                    result.audio,
                    orig_sr=result.sample_rate,
                    target_sr=output_sr
                )
            else:
                enhanced_audio = result.audio

            sf.write(str(output_path), enhanced_audio, output_sr)
            enhancer.cleanup()
            return True
        else:
            logger.error(f"Enhancement failed: {result.error_message}")
            enhancer.cleanup()
            return False

    except ImportError as e:
        logger.error(f"WhisperJAV speech enhancement module not available: {e}")
        logger.info("Falling back to no enhancement")
        import shutil
        shutil.copy(audio_path, output_path)
        return True
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Filtering
# =============================================================================

# Common noise patterns that should be rejected
NOISE_PATTERNS = [
    r"^[あァアうウ]+$",           # Just vowels repeated
    r"^[んンー]+$",               # Just 'n' or prolongation
    r"^[。、！？]+$",             # Just punctuation
    r"^(.)\1{3,}$",              # Any character repeated 4+ times
    r"^\s*$",                     # Empty/whitespace
]

NOISE_REGEXES = [re.compile(p) for p in NOISE_PATTERNS]


def is_japanese_char(char: str) -> bool:
    """
    Check if a character is Japanese (hiragana, katakana, or kanji).

    Unicode ranges:
    - Hiragana: U+3040-U+309F
    - Katakana: U+30A0-U+30FF
    - Katakana Phonetic Extensions: U+31F0-U+31FF
    - CJK Unified Ideographs (Kanji): U+4E00-U+9FFF
    - CJK Extension A: U+3400-U+4DBF
    - Halfwidth Katakana: U+FF65-U+FF9F
    - Japanese punctuation: U+3000-U+303F
    """
    code = ord(char)
    return (
        (0x3040 <= code <= 0x309F) or  # Hiragana
        (0x30A0 <= code <= 0x30FF) or  # Katakana
        (0x31F0 <= code <= 0x31FF) or  # Katakana extensions
        (0x4E00 <= code <= 0x9FFF) or  # CJK Unified (Kanji)
        (0x3400 <= code <= 0x4DBF) or  # CJK Extension A
        (0xFF65 <= code <= 0xFF9F) or  # Halfwidth Katakana
        (0x3000 <= code <= 0x303F)     # Japanese punctuation
    )


def is_unicode_ligature(char: str) -> bool:
    """
    Check if a character is a Unicode ligature (common garbage from MMS).

    Unicode ligatures (U+FB00-U+FB4F) are often produced by MMS when
    processing noise/non-speech audio.
    """
    code = ord(char)
    return 0xFB00 <= code <= 0xFB4F


def calculate_japanese_ratio(text: str) -> float:
    """
    Calculate the ratio of Japanese characters in text.

    Args:
        text: Input text

    Returns:
        Ratio of Japanese characters (0.0 to 1.0)
    """
    if not text:
        return 0.0

    japanese_count = sum(1 for char in text if is_japanese_char(char))
    return japanese_count / len(text)


def contains_unicode_ligatures(text: str) -> bool:
    """
    Check if text contains Unicode ligatures (MMS garbage indicator).

    When MMS produces ligatures like 'fi' (U+FB01), 'ffi' (U+FB03), etc.,
    it's typically outputting garbage on non-speech audio.
    """
    return any(is_unicode_ligature(char) for char in text)


def filter_ctc_output(
    text: str,
    min_chars: int = 2,
    max_chars: int = 100,
    confidence: Optional[float] = None,
    min_confidence: float = 0.3,
    min_japanese_ratio: float = 0.5,
    require_japanese: bool = True
) -> Tuple[bool, Optional[str], str]:
    """
    Filter CTC output to reject noise/garbage.

    Args:
        text: Raw CTC transcription
        min_chars: Minimum character count
        max_chars: Maximum character count (suspiciously long)
        confidence: Confidence score (if available)
        min_confidence: Minimum confidence threshold
        min_japanese_ratio: Minimum ratio of Japanese characters (0.0-1.0)
        require_japanese: If True, reject outputs without any Japanese chars

    Returns:
        Tuple of (passed, filtered_text, reason)
        - passed: True if text passed filtering
        - filtered_text: The text if passed, None if filtered
        - reason: Reason for filtering (empty if passed)
    """
    # Strip whitespace
    text = text.strip()

    # Length check
    if len(text) < min_chars:
        return False, None, f"too_short ({len(text)} < {min_chars})"

    if len(text) > max_chars:
        return False, None, f"too_long ({len(text)} > {max_chars})"

    # Unicode ligature check (strong garbage indicator from MMS)
    if contains_unicode_ligatures(text):
        return False, None, "contains_ligatures"

    # Japanese character check
    jp_ratio = calculate_japanese_ratio(text)

    if require_japanese and jp_ratio == 0.0:
        return False, None, "no_japanese_chars"

    if jp_ratio < min_japanese_ratio:
        return False, None, f"low_japanese_ratio ({jp_ratio:.2f} < {min_japanese_ratio})"

    # Noise pattern check
    for i, regex in enumerate(NOISE_REGEXES):
        if regex.match(text):
            return False, None, f"noise_pattern_{i}"

    # Confidence check
    if confidence is not None and confidence < min_confidence:
        return False, None, f"low_confidence ({confidence:.3f} < {min_confidence})"

    return True, text, ""


# =============================================================================
# Main Processing
# =============================================================================

def create_ctc_backend(backend_name: str, model_id: str, device: str) -> CTCBackend:
    """
    Factory function for CTC backends.

    Args:
        backend_name: Backend type ("mms", "reazonspeech")
        model_id: Model identifier
        device: Device to use

    Returns:
        CTCBackend instance
    """
    if backend_name == "mms":
        return MMSBackend(model_id=model_id, device=device)
    elif backend_name == "reazonspeech":
        return ReazonSpeechBackend(model_id=model_id, device=device)
    else:
        raise ValueError(f"Unknown CTC backend: {backend_name}")


def process_gaps(
    gaps: List[Gap],
    source_audio: Path,
    ctc_backend: CTCBackend,
    enhancer_name: str,
    enhancer_model: Optional[str],
    gap_padding: float,
    filter_settings: Dict[str, Any],
    work_dir: Path
) -> List[CTCResult]:
    """
    Process all gaps through the surgical CTC pipeline.
    """
    results = []

    for i, gap in enumerate(gaps):
        logger.info(f"Processing gap {i+1}/{len(gaps)}: {gap.start:.2f}s - {gap.end:.2f}s ({gap.duration:.2f}s)")

        # Extract segment
        segment_path = work_dir / f"gap_{i:04d}_raw.wav"
        if not extract_segment(source_audio, gap.start, gap.end, segment_path, padding=gap_padding):
            logger.warning(f"Failed to extract gap {i+1}")
            continue

        # Enhance segment
        enhanced_path = work_dir / f"gap_{i:04d}_enhanced.wav"
        if not enhance_audio(segment_path, enhanced_path, enhancer_name, enhancer_model):
            logger.warning(f"Failed to enhance gap {i+1}")
            enhanced_path = segment_path  # Fall back to raw

        # Transcribe with CTC
        raw_text, confidence = ctc_backend.transcribe(enhanced_path)

        # Filter
        passed, filtered_text, reason = filter_ctc_output(
            raw_text,
            min_chars=filter_settings.get("min_chars", 2),
            max_chars=filter_settings.get("max_chars", 100),
            confidence=confidence,
            min_confidence=filter_settings.get("min_confidence", 0.3),
            min_japanese_ratio=filter_settings.get("min_japanese_ratio", 0.5),
            require_japanese=filter_settings.get("require_japanese", True)
        )

        result = CTCResult(
            gap=gap,
            raw_text=raw_text,
            filtered_text=filtered_text,
            confidence=confidence,
            was_filtered=not passed,
            filter_reason=reason if not passed else None,
            enhanced_audio_path=str(enhanced_path) if enhanced_path.exists() else None
        )

        results.append(result)

        if passed:
            logger.info(f"  -> Output: '{filtered_text}'")
        else:
            logger.info(f"  -> Filtered ({reason}): '{raw_text}'")

    return results


def generate_report(
    results: List[CTCResult],
    gaps: List[Gap],
    ctc_model: str,
    enhancer: str,
    filter_settings: Dict[str, Any]
) -> EvaluationReport:
    """Generate evaluation report."""
    report = EvaluationReport(
        total_gaps_detected=len(gaps),
        gaps_processed=len(results),
        gaps_with_output=sum(1 for r in results if r.filtered_text),
        gaps_filtered_out=sum(1 for r in results if r.was_filtered),
        total_duration_seconds=sum(g.duration for g in gaps),
        ctc_model_used=ctc_model,
        enhancer_used=enhancer,
        filter_settings=filter_settings
    )
    return report


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(output_path: Path, results: List[CTCResult]) -> int:
    """
    Write accepted CTC results to SRT file.

    Args:
        output_path: Path to output SRT file
        results: List of CTCResult objects

    Returns:
        Number of subtitles written
    """
    # Filter to only accepted (non-filtered) results with text
    accepted = [r for r in results if not r.was_filtered and r.filtered_text]

    if not accepted:
        logger.warning("No accepted results to write to SRT")
        return 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, result in enumerate(accepted, start=1):
            start_ts = format_srt_timestamp(result.gap.start)
            end_ts = format_srt_timestamp(result.gap.end)

            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{result.filtered_text}\n")
            f.write("\n")

    return len(accepted)


def write_outputs(
    output_dir: Path,
    audio_basename: str,
    gaps: List[Gap],
    results: List[CTCResult],
    report: EvaluationReport
):
    """
    Write all output files with audio basename prefix.

    Args:
        output_dir: Output directory
        audio_basename: Base name of the source audio file (without extension)
        gaps: List of detected gaps
        results: List of CTC results
        report: Evaluation report
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # detected_gaps.json
    gaps_data = [asdict(g) for g in gaps]
    with open(output_dir / f"{audio_basename}.detected_gaps.json", "w", encoding="utf-8") as f:
        json.dump(gaps_data, f, indent=2, ensure_ascii=False)

    # ctc_raw_transcripts.json
    raw_data = [
        {
            "gap_index": i,
            "start": r.gap.start,
            "end": r.gap.end,
            "raw_text": r.raw_text,
            "confidence": r.confidence
        }
        for i, r in enumerate(results)
    ]
    with open(output_dir / f"{audio_basename}.ctc_raw_transcripts.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)

    # ctc_filtered_transcripts.json
    filtered_data = [
        {
            "gap_index": i,
            "start": r.gap.start,
            "end": r.gap.end,
            "text": r.filtered_text,
            "confidence": r.confidence,
            "was_filtered": r.was_filtered,
            "filter_reason": r.filter_reason
        }
        for i, r in enumerate(results)
    ]
    with open(output_dir / f"{audio_basename}.ctc_filtered_transcripts.json", "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    # SRT output with accepted results
    srt_path = output_dir / f"{audio_basename}.surgical_ctc.srt"
    srt_count = write_srt(srt_path, results)
    logger.info(f"SRT written with {srt_count} accepted segments: {srt_path}")

    # surgical_evaluation.txt
    with open(output_dir / f"{audio_basename}.surgical_evaluation.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MANIFOLD SURGICAL CTC - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"CTC Model: {report.ctc_model_used}\n")
        f.write(f"Enhancer: {report.enhancer_used}\n")
        f.write(f"Filter Settings: {report.filter_settings}\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total gaps detected: {report.total_gaps_detected}\n")
        f.write(f"Gaps processed: {report.gaps_processed}\n")
        f.write(f"Gaps with output: {report.gaps_with_output}\n")
        f.write(f"Gaps filtered out: {report.gaps_filtered_out}\n")
        f.write(f"Total gap duration: {report.total_duration_seconds:.2f}s\n")
        f.write(f"SRT segments written: {srt_count}\n\n")

        if report.gaps_processed > 0:
            yield_rate = (report.gaps_with_output / report.gaps_processed) * 100
            f.write(f"Yield rate: {yield_rate:.1f}%\n\n")

        f.write("DETAILED RESULTS\n")
        f.write("-" * 40 + "\n")
        for i, r in enumerate(results):
            f.write(f"\nGap {i+1}: {r.gap.start:.2f}s - {r.gap.end:.2f}s ({r.gap.duration:.2f}s)\n")
            f.write(f"  Raw: '{r.raw_text}'\n")
            if r.was_filtered:
                f.write(f"  Status: FILTERED ({r.filter_reason})\n")
            else:
                f.write(f"  Status: ACCEPTED -> '{r.filtered_text}'\n")
            if r.confidence is not None:
                f.write(f"  Confidence: {r.confidence:.3f}\n")

    logger.info(f"Output written to: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manifold Surgical CTC - Gap-filler reference implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Whisper SRT for gap detection
  python surgical_ctc_reference_implementation.py \\
      --source-audio input.wav \\
      --whisper-srt whisper_output.srt \\
      --enhancer clearvoice \\
      --ctc-model facebook/mms-1b-all

  # Using manual gap specification
  python surgical_ctc_reference_implementation.py \\
      --source-audio input.wav \\
      --manual-gaps gaps.srt \\
      --enhancer none \\
      --ctc-model facebook/mms-1b-all

  # Testing ReazonSpeech backend
  python surgical_ctc_reference_implementation.py \\
      --source-audio input.wav \\
      --whisper-srt whisper_output.srt \\
      --ctc-backend reazonspeech \\
      --ctc-model reazon-research/reazonspeech-wav2vec2-large-rs35kh
        """
    )

    # Input/Output
    parser.add_argument(
        "--source-audio",
        type=Path,
        required=True,
        help="Source audio file (WAV, MP4, etc.)"
    )
    parser.add_argument(
        "--whisper-srt",
        type=Path,
        help="Whisper's SRT output for automatic gap detection"
    )
    parser.add_argument(
        "--manual-gaps",
        type=Path,
        help="Manually specified gaps file (SRT format or simple 'start - end' format)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Output directory (default: ./results)"
    )

    # CTC Backend
    parser.add_argument(
        "--ctc-backend",
        type=str,
        choices=["mms", "reazonspeech"],
        default="mms",
        help="CTC backend to use (default: mms)"
    )
    parser.add_argument(
        "--ctc-model",
        type=str,
        default="facebook/mms-1b-all",
        help="CTC model identifier (default: facebook/mms-1b-all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'cpu', or device index (default: auto)"
    )

    # Enhancement
    parser.add_argument(
        "--enhancer",
        type=str,
        choices=["none", "clearvoice", "bs-roformer", "zipenhancer"],
        default="none",
        help="Speech enhancer backend (default: none)"
    )
    parser.add_argument(
        "--enhancer-model",
        type=str,
        help="Enhancer model variant (optional)"
    )

    # Gap Detection
    parser.add_argument(
        "--min-gap-duration",
        type=float,
        default=0.3,
        help="Minimum gap duration in seconds (default: 0.3)"
    )
    parser.add_argument(
        "--max-gap-duration",
        type=float,
        default=3.0,
        help="Maximum gap duration in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--gap-padding",
        type=float,
        default=0.15,
        help="Padding to add around gaps in seconds (default: 0.15)"
    )
    parser.add_argument(
        "--split-long-gaps",
        action="store_true",
        default=True,
        help="Split gaps longer than max-gap-duration into chunks (default: True)"
    )
    parser.add_argument(
        "--no-split-long-gaps",
        action="store_false",
        dest="split_long_gaps",
        help="Ignore gaps longer than max-gap-duration instead of splitting"
    )

    # Filtering
    parser.add_argument(
        "--min-output-chars",
        type=int,
        default=2,
        help="Minimum characters in CTC output (default: 2)"
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=100,
        help="Maximum characters in CTC output (default: 100)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--min-japanese-ratio",
        type=float,
        default=0.5,
        help="Minimum ratio of Japanese characters (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--no-require-japanese",
        action="store_true",
        help="Allow outputs without any Japanese characters (not recommended)"
    )

    # Misc
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary audio files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not args.source_audio.exists():
        logger.error(f"Source audio not found: {args.source_audio}")
        sys.exit(1)

    if not args.whisper_srt and not args.manual_gaps:
        logger.error("Must provide either --whisper-srt or --manual-gaps")
        sys.exit(1)

    if args.whisper_srt and not args.whisper_srt.exists():
        logger.error(f"Whisper SRT not found: {args.whisper_srt}")
        sys.exit(1)

    if args.manual_gaps and not args.manual_gaps.exists():
        logger.error(f"Manual gaps file not found: {args.manual_gaps}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("MANIFOLD SURGICAL CTC - Reference Implementation")
    logger.info("=" * 60)
    logger.info(f"Source audio: {args.source_audio}")
    logger.info(f"CTC backend: {args.ctc_backend}")
    logger.info(f"CTC model: {args.ctc_model}")
    logger.info(f"Enhancer: {args.enhancer}")

    # Get audio duration
    audio_duration = get_audio_duration(args.source_audio)
    if audio_duration <= 0:
        logger.error("Could not determine audio duration")
        sys.exit(1)
    logger.info(f"Audio duration: {audio_duration:.2f}s")

    # Detect gaps
    if args.manual_gaps:
        gaps = parse_manual_gaps(args.manual_gaps)
    else:
        gaps = detect_gaps_from_srt(
            args.whisper_srt,
            audio_duration,
            min_gap=args.min_gap_duration,
            max_gap=args.max_gap_duration,
            split_long_gaps=args.split_long_gaps
        )

    if not gaps:
        logger.info("No gaps detected - nothing to process")
        sys.exit(0)

    logger.info(f"Gaps to process: {len(gaps)}")

    # Create CTC backend
    ctc_backend = create_ctc_backend(args.ctc_backend, args.ctc_model, args.device)

    # Filter settings
    filter_settings = {
        "min_chars": args.min_output_chars,
        "max_chars": args.max_output_chars,
        "min_confidence": args.min_confidence,
        "min_japanese_ratio": args.min_japanese_ratio,
        "require_japanese": not args.no_require_japanese
    }

    # Process gaps
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)

        results = process_gaps(
            gaps=gaps,
            source_audio=args.source_audio,
            ctc_backend=ctc_backend,
            enhancer_name=args.enhancer,
            enhancer_model=args.enhancer_model,
            gap_padding=args.gap_padding,
            filter_settings=filter_settings,
            work_dir=work_dir
        )

        # Generate report
        report = generate_report(
            results=results,
            gaps=gaps,
            ctc_model=args.ctc_model,
            enhancer=args.enhancer,
            filter_settings=filter_settings
        )

        # Extract audio basename (without extension)
        audio_basename = args.source_audio.stem

        # Write outputs
        write_outputs(args.output_dir, audio_basename, gaps, results, report)

        # Optionally keep temp files
        if args.keep_temp:
            import shutil
            temp_output = args.output_dir / "temp_audio"
            shutil.copytree(work_dir, temp_output)
            logger.info(f"Temporary files saved to: {temp_output}")

    # Cleanup
    ctc_backend.cleanup()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Gaps processed: {report.gaps_processed}")
    logger.info(f"Gaps with output: {report.gaps_with_output}")
    logger.info(f"Gaps filtered: {report.gaps_filtered_out}")
    if report.gaps_processed > 0:
        logger.info(f"Yield rate: {(report.gaps_with_output / report.gaps_processed) * 100:.1f}%")
    logger.info(f"Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
