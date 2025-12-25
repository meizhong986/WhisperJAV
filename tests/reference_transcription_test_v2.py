#!/usr/bin/env python3
"""
Clean Reference Implementation for ASR Testing.

This script provides a minimal, direct implementation that bypasses all
v1.7.3 pipeline complexity to test the raw ASR components.

Flow:
1. Accept media file
2. Convert to 16kHz mono WAV (using ffmpeg)
3. Run Silero VAD v3.1 directly
4. For each VAD group, run faster-whisper directly
5. Create SRT output

Purpose: Isolate whether missing subs are caused by:
- Core components (VAD + faster-whisper)
- v1.7.3 pipeline orchestration (scene detection, config layers, etc.)

Usage:
    python reference_transcription_test.py --audio input.wav --output output.srt
    python reference_transcription_test.py --audio input.wav --compute-type int8
    python reference_transcription_test.py --audio input.wav --config preset_name
"""

import argparse
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("reference_test")


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

@dataclass
class TestPreset:
    """Configuration preset for testing."""
    name: str
    description: str

    # VAD version: "v3.1", "v4.0", "latest" (defaults to latest from torch hub)
    vad_version: str = "v3.1"

    # VAD parameters
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30

    # VAD grouping
    chunk_threshold_s: float = 4.0  # Gap threshold for grouping segments

    # Whisper parameters
    model_name: str = "large-v2"
    compute_type: str = "float16"
    device: str = "cuda"

    # Decoder parameters
    language: str = "ja"
    task: str = "transcribe"
    beam_size: int = 5
    patience: float = 1.0
    temperature: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Quality thresholds
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

    # Word timestamps
    word_timestamps: bool = True


# Preset configurations matching different scenarios
PRESETS = {
    "default": TestPreset(
        name="default",
        description="faster-whisper defaults",
    ),

    "v171_match": TestPreset(
        name="v171_match",
        description="EXACT parameters from v1.7.1 standalone asr_config.json",
        vad_threshold=0.05,  # From silero_vad_options.aggressive.threshold
        min_speech_duration_ms=30,  # From silero_vad_options.aggressive.min_speech_duration_ms
        min_silence_duration_ms=300,  # From silero_vad_options.aggressive.min_silence_duration_ms
        speech_pad_ms=450,  # From silero_vad_options.aggressive.speech_pad_ms
        chunk_threshold_s=0.950,
        compute_type="int8_float16",  # From models.faster-whisper-large-v2-int8.compute_type
        beam_size=2,  # From common_decoder_options.aggressive.beam_size
        patience=2.9,  # From common_decoder_options.aggressive.patience
        temperature=[0.0, 0.3],  # From common_transcriber_options.aggressive.temperature
        compression_ratio_threshold=3.0,  # From common_transcriber_options.aggressive
        logprob_threshold=-2.5,  # From common_transcriber_options.aggressive
        no_speech_threshold=0.22,  # From common_transcriber_options.aggressive
    ),

    "v173_aggressive": TestPreset(
        name="v173_aggressive",
        description="EXACT parameters from current v1.7.3 asr_config.json",
        vad_threshold=0.158,  # From silero_vad_options.aggressive.threshold
        min_speech_duration_ms=30,  # From silero_vad_options.aggressive
        min_silence_duration_ms=300,  # From silero_vad_options.aggressive
        speech_pad_ms=700,  # From silero_vad_options.aggressive.speech_pad_ms
        chunk_threshold_s=4.0,  # From silero_vad_options.aggressive
        compute_type="float16",
        beam_size=3,  # From common_decoder_options.aggressive.beam_size
        patience=1.6,  # From common_decoder_options.aggressive.patience
        temperature=[0.0],  # From common_transcriber_options.aggressive (single temp, no fallback)
        compression_ratio_threshold=3.0,  # From common_transcriber_options.aggressive
        logprob_threshold=-2.5,  # From common_transcriber_options.aggressive
        no_speech_threshold=0.22,  # From common_transcriber_options.aggressive
    ),

    "v171_latest_vad": TestPreset(
        name="v171_latest_vad",
        description="v1.7.1 ASR params with LATEST Silero VAD (v6.x from torch hub)",
        vad_version="latest",  # Use latest Silero from torch hub (currently v6.1)
        vad_threshold=0.05,  # From v1.7.1 silero_vad_options.aggressive.threshold
        min_speech_duration_ms=30,
        min_silence_duration_ms=300,
        speech_pad_ms=600,
        chunk_threshold_s=4.0,
        compute_type="int8_float16",  # Match v1.7.1 faster-whisper-large-v2-int8
        beam_size=2,  # v1.7.1 ASR params
        patience=2.9,  # v1.7.1 ASR params
        temperature=[0.0, 0.3],
        compression_ratio_threshold=3.0,
        logprob_threshold=-2.5,
        no_speech_threshold=0.22,
    ),

    "compute_int8": TestPreset(
        name="compute_int8",
        description="int8 compute type test",
        vad_threshold=0.187,
        min_speech_duration_ms=30,
        min_silence_duration_ms=300,
        speech_pad_ms=500,
        compute_type="int8",
        beam_size=3,
        patience=1.6,
        temperature=[0.0, 0.3],
    ),

    "compute_int8_float16": TestPreset(
        name="compute_int8_float16",
        description="int8_float16 compute type test",
        vad_threshold=0.187,
        min_speech_duration_ms=30,
        min_silence_duration_ms=300,
        speech_pad_ms=500,
        compute_type="int8_float16",
        beam_size=3,
        patience=1.6,
        temperature=[0.0, 0.3],
    ),

    "compute_auto": TestPreset(
        name="compute_auto",
        description="auto compute type (let ctranslate2 decide)",
        vad_threshold=0.187,
        min_speech_duration_ms=30,
        min_silence_duration_ms=300,
        speech_pad_ms=500,
        compute_type="auto",
        beam_size=3,
        patience=1.6,
        temperature=[0.0, 0.3],
    ),

    "vad_no_padding": TestPreset(
        name="vad_no_padding",
        description="VAD with no speech padding",
        vad_threshold=0.187,
        min_speech_duration_ms=30,
        min_silence_duration_ms=0,
        speech_pad_ms=0,
        chunk_threshold_s=4.0,
        compute_type="float16",
        beam_size=3,
        patience=1.6,
        temperature=[0.0, 0.3],
    ),
}


# =============================================================================
# AUDIO PROCESSING
# =============================================================================

def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
    """Convert audio/video to 16kHz mono WAV using ffmpeg."""
    logger.info(f"Converting {input_path.name} to {sample_rate}Hz mono WAV...")

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",  # Mono
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    logger.info(f"Converted to: {output_path}")
    return output_path


def load_audio(wav_path: Path) -> Tuple[np.ndarray, int]:
    """Load WAV file as numpy array."""
    import wave

    with wave.open(str(wav_path), 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    logger.info(f"Loaded audio: {len(audio)/sample_rate:.1f}s @ {sample_rate}Hz")
    return audio, sample_rate


# =============================================================================
# SILERO VAD v3.1 - DIRECT IMPLEMENTATION
# =============================================================================

class SileroVADv31:
    """Direct Silero VAD v3.1 implementation (no WhisperJAV wrappers)."""

    SAMPLE_RATE = 16000

    def __init__(self):
        logger.info("Loading Silero VAD v3.1 from torch hub...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad:v3.1',
            model='silero_vad',
            force_reload=False
        )
        self.get_speech_timestamps = self.utils[0]
        logger.info("Silero VAD v3.1 loaded successfully")

    def detect_speech(
        self,
        audio: np.ndarray,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> List[Dict[str, int]]:
        """
        Detect speech segments in audio.

        Returns list of dicts with 'start' and 'end' keys (in samples).
        """
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Run VAD
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.SAMPLE_RATE,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        logger.info(f"VAD detected {len(speech_timestamps)} speech segments")
        return speech_timestamps

    def group_segments(
        self,
        segments: List[Dict[str, int]],
        chunk_threshold_s: float = 4.0,
    ) -> List[List[Dict[str, int]]]:
        """
        Group speech segments by gap threshold.

        Segments with gaps > chunk_threshold_s become separate groups.
        """
        if not segments:
            return []

        groups = []
        current_group = [segments[0]]

        for seg in segments[1:]:
            gap_s = (seg['start'] - current_group[-1]['end']) / self.SAMPLE_RATE

            if gap_s > chunk_threshold_s:
                groups.append(current_group)
                current_group = [seg]
            else:
                current_group.append(seg)

        groups.append(current_group)
        logger.info(f"Grouped into {len(groups)} groups (threshold: {chunk_threshold_s}s)")
        return groups


class SileroVADLatest:
    """Latest Silero VAD from torch hub (no version specified = latest, currently v6.x)."""

    SAMPLE_RATE = 16000

    def __init__(self):
        logger.info("Loading Silero VAD LATEST from torch hub (no version = latest)...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',  # No version = latest
            model='silero_vad',
            force_reload=False
        )
        self.get_speech_timestamps = self.utils[0]
        # Try to get version info
        try:
            version = getattr(self.model, '__version__', 'unknown')
            logger.info(f"Silero VAD loaded successfully (version: {version})")
        except:
            logger.info("Silero VAD LATEST loaded successfully")

    def detect_speech(
        self,
        audio: np.ndarray,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> List[Dict[str, int]]:
        """Detect speech segments in audio."""
        audio_tensor = torch.from_numpy(audio).float()

        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.SAMPLE_RATE,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        logger.info(f"VAD detected {len(speech_timestamps)} speech segments")
        return speech_timestamps

    def group_segments(
        self,
        segments: List[Dict[str, int]],
        chunk_threshold_s: float = 4.0,
    ) -> List[List[Dict[str, int]]]:
        """Group speech segments by gap threshold."""
        if not segments:
            return []

        groups = []
        current_group = [segments[0]]

        for seg in segments[1:]:
            gap_s = (seg['start'] - current_group[-1]['end']) / self.SAMPLE_RATE

            if gap_s > chunk_threshold_s:
                groups.append(current_group)
                current_group = [seg]
            else:
                current_group.append(seg)

        groups.append(current_group)
        logger.info(f"Grouped into {len(groups)} groups (threshold: {chunk_threshold_s}s)")
        return groups


class SileroVADv40:
    """Silero VAD v4.0 implementation."""

    SAMPLE_RATE = 16000

    def __init__(self):
        logger.info("Loading Silero VAD v4.0 from torch hub...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad:v4.0',
            model='silero_vad',
            force_reload=False
        )
        self.get_speech_timestamps = self.utils[0]
        logger.info("Silero VAD v4.0 loaded successfully")

    def detect_speech(
        self,
        audio: np.ndarray,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> List[Dict[str, int]]:
        """Detect speech segments in audio."""
        audio_tensor = torch.from_numpy(audio).float()

        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.SAMPLE_RATE,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        logger.info(f"VAD detected {len(speech_timestamps)} speech segments")
        return speech_timestamps

    def group_segments(
        self,
        segments: List[Dict[str, int]],
        chunk_threshold_s: float = 4.0,
    ) -> List[List[Dict[str, int]]]:
        """Group speech segments by gap threshold."""
        if not segments:
            return []

        groups = []
        current_group = [segments[0]]

        for seg in segments[1:]:
            gap_s = (seg['start'] - current_group[-1]['end']) / self.SAMPLE_RATE

            if gap_s > chunk_threshold_s:
                groups.append(current_group)
                current_group = [seg]
            else:
                current_group.append(seg)

        groups.append(current_group)
        logger.info(f"Grouped into {len(groups)} groups (threshold: {chunk_threshold_s}s)")
        return groups


def create_vad(version: str = "v3.1"):
    """Factory function to create the appropriate VAD based on version."""
    version = version.lower()
    if version == "v3.1":
        return SileroVADv31()
    elif version == "v4.0":
        return SileroVADv40()
    elif version == "latest":
        return SileroVADLatest()
    else:
        logger.warning(f"Unknown VAD version '{version}', defaulting to v3.1")
        return SileroVADv31()


# =============================================================================
# FASTER-WHISPER - DIRECT IMPLEMENTATION
# =============================================================================

class DirectWhisperTranscriber:
    """Direct faster-whisper implementation (no WhisperJAV wrappers)."""

    def __init__(
        self,
        model_name: str = "large-v2",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        from faster_whisper import WhisperModel

        logger.info(f"Loading Whisper model: {model_name} ({compute_type} on {device})...")
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper model loaded successfully")

    def transcribe_segment(
        self,
        audio: np.ndarray,
        offset_s: float = 0.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Transcribe an audio segment.

        Args:
            audio: Audio samples (16kHz float32)
            offset_s: Time offset to add to timestamps
            **kwargs: Whisper transcription parameters

        Returns:
            List of segment dicts with start, end, text
        """
        segments_gen, info = self.model.transcribe(
            audio,
            **kwargs
        )

        results = []
        for seg in segments_gen:
            results.append({
                'start': seg.start + offset_s,
                'end': seg.end + offset_s,
                'text': seg.text.strip(),
            })

        return results


# =============================================================================
# SRT GENERATION
# =============================================================================

def format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: List[Dict[str, Any]], output_path: Path, preset: Optional[TestPreset] = None) -> int:
    """Write segments to SRT file. Returns subtitle count."""
    subtitle_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            if not seg['text']:
                continue
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")
            subtitle_count = i

        # Append preset info as final subtitle entry
        if preset:
            subtitle_count += 1
            # Use a timestamp far in the future so it doesn't interfere
            f.write(f"{subtitle_count}\n")
            f.write(f"99:59:59,000 --> 99:59:59,999\n")
            f.write(f"[PRESET: {preset.name}] "
                    f"vad={preset.vad_version} "
                    f"threshold={preset.vad_threshold} "
                    f"speech_pad={preset.speech_pad_ms}ms "
                    f"compute={preset.compute_type} "
                    f"beam={preset.beam_size} "
                    f"patience={preset.patience} "
                    f"temp={preset.temperature} "
                    f"no_speech={preset.no_speech_threshold}\n\n")

    logger.info(f"Wrote {subtitle_count} subtitles to {output_path}")
    return subtitle_count


# =============================================================================
# MAIN TRANSCRIPTION PIPELINE
# =============================================================================

def run_reference_transcription(
    audio_path: Path,
    output_path: Path,
    preset: TestPreset,
) -> Dict[str, Any]:
    """
    Run the clean reference transcription pipeline.

    Returns metrics dict.
    """
    start_time = time.time()

    # Step 1: Convert to WAV if needed
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        if audio_path.suffix.lower() != '.wav':
            wav_path = tmp_path / "audio.wav"
            convert_to_wav(audio_path, wav_path)
        else:
            wav_path = audio_path

        # Step 2: Load audio
        audio, sample_rate = load_audio(wav_path)
        audio_duration = len(audio) / sample_rate

        # Step 3: Run Silero VAD
        logger.info("=" * 60)
        logger.info(f"STEP 1: Running Silero VAD {preset.vad_version}")
        logger.info("=" * 60)
        logger.info(f"  vad_version: {preset.vad_version}")
        logger.info(f"  threshold: {preset.vad_threshold}")
        logger.info(f"  min_speech_duration_ms: {preset.min_speech_duration_ms}")
        logger.info(f"  min_silence_duration_ms: {preset.min_silence_duration_ms}")
        logger.info(f"  speech_pad_ms: {preset.speech_pad_ms}")

        vad = create_vad(preset.vad_version)
        speech_segments = vad.detect_speech(
            audio,
            threshold=preset.vad_threshold,
            min_speech_duration_ms=preset.min_speech_duration_ms,
            min_silence_duration_ms=preset.min_silence_duration_ms,
            speech_pad_ms=preset.speech_pad_ms,
        )

        # Step 4: Group segments
        logger.info("=" * 60)
        logger.info("STEP 2: Grouping VAD segments")
        logger.info("=" * 60)
        logger.info(f"  chunk_threshold_s: {preset.chunk_threshold_s}")

        groups = vad.group_segments(speech_segments, preset.chunk_threshold_s)

        # Step 5: Load Whisper and transcribe
        logger.info("=" * 60)
        logger.info("STEP 3: Running faster-whisper transcription")
        logger.info("=" * 60)
        logger.info(f"  model: {preset.model_name}")
        logger.info(f"  compute_type: {preset.compute_type}")
        logger.info(f"  device: {preset.device}")
        logger.info(f"  language: {preset.language}")
        logger.info(f"  beam_size: {preset.beam_size}")
        logger.info(f"  patience: {preset.patience}")
        logger.info(f"  temperature: {preset.temperature}")
        logger.info(f"  no_speech_threshold: {preset.no_speech_threshold}")

        whisper = DirectWhisperTranscriber(
            model_name=preset.model_name,
            device=preset.device,
            compute_type=preset.compute_type,
        )

        all_segments = []

        for i, group in enumerate(groups):
            # Get audio for this group (with some padding)
            group_start = group[0]['start']
            group_end = group[-1]['end']

            # Add small padding
            pad_samples = int(0.1 * sample_rate)
            start_sample = max(0, group_start - pad_samples)
            end_sample = min(len(audio), group_end + pad_samples)

            group_audio = audio[start_sample:end_sample]
            offset_s = start_sample / sample_rate

            logger.info(f"Transcribing group {i+1}/{len(groups)}: {offset_s:.1f}s - {end_sample/sample_rate:.1f}s")

            # Transcribe
            segments = whisper.transcribe_segment(
                group_audio,
                offset_s=offset_s,
                language=preset.language,
                task=preset.task,
                beam_size=preset.beam_size,
                patience=preset.patience,
                temperature=preset.temperature,
                compression_ratio_threshold=preset.compression_ratio_threshold,
                log_prob_threshold=preset.logprob_threshold,
                no_speech_threshold=preset.no_speech_threshold,
                word_timestamps=preset.word_timestamps,
            )

            all_segments.extend(segments)
            logger.info(f"  -> {len(segments)} segments")

        # Step 6: Write SRT
        logger.info("=" * 60)
        logger.info("STEP 4: Writing SRT output")
        logger.info("=" * 60)

        # Sort by start time
        all_segments.sort(key=lambda x: x['start'])

        # Filter empty segments
        all_segments = [s for s in all_segments if s['text'].strip()]

        subtitle_count = write_srt(all_segments, output_path, preset)

    # Calculate metrics
    total_time = time.time() - start_time
    total_speech_duration = sum(
        (s['end'] - s['start']) for s in all_segments
    )

    metrics = {
        'preset': preset.name,
        'audio_duration_s': audio_duration,
        'vad_segments': len(speech_segments),
        'vad_groups': len(groups),
        'subtitle_count': subtitle_count,
        'total_speech_duration_s': total_speech_duration,
        'avg_subtitle_duration_s': total_speech_duration / subtitle_count if subtitle_count else 0,
        'processing_time_s': total_time,
        'output_path': str(output_path),
    }

    # Print summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")

    return metrics


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean reference transcription test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets available:
  default           - faster-whisper defaults
  v171_match        - Parameters matching v1.7.1 debug output
  v173_aggressive   - v1.7.3 aggressive sensitivity defaults
  compute_int8      - int8 compute type test
  compute_int8_float16 - int8_float16 compute type test
  compute_auto      - auto compute type (let ctranslate2 decide)
  vad_no_padding    - VAD with no speech padding

Examples:
  python reference_transcription_test.py --audio test.wav --preset v171_match
  python reference_transcription_test.py --audio test.wav --compute-type int8
  python reference_transcription_test.py --audio test.wav --list-presets
"""
    )

    parser.add_argument(
        "--audio", "-a",
        type=Path,
        help="Input audio/video file"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output SRT file (default: <input>_reference.srt)"
    )

    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()),
        default="v171_match",
        help="Configuration preset (default: v171_match)"
    )

    parser.add_argument(
        "--compute-type",
        choices=["auto", "int8", "int8_float16", "float16", "float32"],
        help="Override compute type"
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        help="Override beam size"
    )

    parser.add_argument(
        "--patience",
        type=float,
        help="Override patience"
    )

    parser.add_argument(
        "--vad-threshold",
        type=float,
        help="Override VAD threshold"
    )

    parser.add_argument(
        "--vad-version",
        choices=["v3.1", "v4.0", "latest"],
        help="Override VAD version (v3.1, v4.0, or latest from torch hub)"
    )

    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit"
    )

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print("\nAvailable presets:\n")
        for name, preset in PRESETS.items():
            print(f"  {name}:")
            print(f"    {preset.description}")
            print(f"    compute_type: {preset.compute_type}")
            print(f"    beam_size: {preset.beam_size}, patience: {preset.patience}")
            print(f"    vad_threshold: {preset.vad_threshold}")
            print()
        return

    # Validate input
    if not args.audio:
        parser.error("--audio is required (or use --list-presets)")

    if not args.audio.exists():
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    # Get preset and apply overrides
    preset = PRESETS[args.preset]

    # Create a modified copy if overrides provided
    if args.compute_type:
        preset = TestPreset(**{**preset.__dict__, 'compute_type': args.compute_type})
    if args.beam_size:
        preset = TestPreset(**{**preset.__dict__, 'beam_size': args.beam_size})
    if args.patience:
        preset = TestPreset(**{**preset.__dict__, 'patience': args.patience})
    if args.vad_threshold:
        preset = TestPreset(**{**preset.__dict__, 'vad_threshold': args.vad_threshold})
    if args.vad_version:
        preset = TestPreset(**{**preset.__dict__, 'vad_version': args.vad_version})

    # Set output path
    output_path = args.output or args.audio.with_suffix('.reference.srt')

    # Run transcription
    logger.info(f"Running reference transcription with preset: {preset.name}")
    logger.info(f"Input: {args.audio}")
    logger.info(f"Output: {output_path}")

    try:
        metrics = run_reference_transcription(args.audio, output_path, preset)

        print("\n" + "=" * 60)
        print("REFERENCE TRANSCRIPTION COMPLETE")
        print("=" * 60)
        print(f"Subtitles: {metrics['subtitle_count']}")
        print(f"VAD segments: {metrics['vad_segments']}")
        print(f"VAD groups: {metrics['vad_groups']}")
        print(f"Total speech: {metrics['total_speech_duration_s']:.1f}s")
        print(f"Processing time: {metrics['processing_time_s']:.1f}s")
        print(f"Output: {metrics['output_path']}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
