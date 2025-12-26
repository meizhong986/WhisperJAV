"""
FFmpeg-based DSP audio preprocessing.

Applies traditional signal processing filters before AI enhancement.
All filters run in a single FFmpeg pass for efficiency.

FFmpeg Version Requirements:
    - Minimum: FFmpeg 4.2 (for all effects including deesser)
    - Most effects work with FFmpeg 3.1+
    - Filter introduction dates:
        - loudnorm, dynaudnorm, acompressor: FFmpeg 3.1 (June 2016)
        - highpass, lowpass, volume: FFmpeg 2.x
        - afftdn (denoise): FFmpeg 4.1 (November 2018)
        - deesser: FFmpeg 4.2 (August 2019)

Features:
- loudnorm: EBU R128 loudness normalization (-16 LUFS)
- normalize: Dynamic audio normalization
- compress: Dynamic range compression
- denoise: FFT-based noise reduction (afftdn)
- highpass: Remove low frequency rumble (<80Hz)
- lowpass: Remove high frequency hiss (>8kHz)
- deess: Reduce sibilant sounds (s, sh, ch)
- amplify: Volume boost (+3dB)

Usage:
    # Single effect
    backend = FFmpegDSPBackend(effects=["loudnorm"])
    result = backend.enhance(audio_path, sample_rate)

    # Multiple effects (chained)
    backend = FFmpegDSPBackend(effects=["highpass", "denoise", "loudnorm"])
    result = backend.enhance(audio_path, sample_rate)

CLI (Ensemble mode):
    whisperjav video.mp4 --ensemble --pass1-speech-enhancer ffmpeg-dsp:loudnorm,denoise

References:
    - FFmpeg Audio Filters: https://ffmpeg.org/ffmpeg-filters.html#Audio-Filters
    - EBU R128 Loudness: https://tech.ebu.ch/docs/r/r128.pdf
"""

from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import subprocess
import tempfile
import time
import logging
import numpy as np

from ..base import EnhancementResult, load_audio_to_array, create_failed_result

logger = logging.getLogger("whisperjav")


# Available DSP effects with their FFmpeg filter strings
AVAILABLE_EFFECTS: Dict[str, Dict[str, str]] = {
    "loudnorm": {
        "filter": "loudnorm=I=-16:TP=-1.5:LRA=11",
        "display_name": "Loudness Normalize (EBU R128)",
        "description": "Normalize loudness to broadcast standard (-16 LUFS)",
    },
    "normalize": {
        "filter": "dynaudnorm=f=150:g=15",
        "display_name": "Dynamic Normalize",
        "description": "Normalize audio dynamics while preserving natural sound",
    },
    "compress": {
        "filter": "acompressor=threshold=-20dB:ratio=4:attack=5:release=50",
        "display_name": "Dynamic Range Compression",
        "description": "Reduce volume difference between loud and quiet parts",
    },
    "denoise": {
        "filter": "afftdn=nf=-25",
        "display_name": "Noise Reduction",
        "description": "Remove background noise using FFT denoising",
    },
    "highpass": {
        "filter": "highpass=f=80",
        "display_name": "High-pass Filter (80Hz)",
        "description": "Remove low rumble and hum below 80Hz",
    },
    "lowpass": {
        "filter": "lowpass=f=8000",
        "display_name": "Low-pass Filter (8kHz)",
        "description": "Remove high frequency hiss above 8kHz",
    },
    "deess": {
        "filter": "deesser=i=0.4:m=0.5:f=0.5:s=o",
        "display_name": "De-esser",
        "description": "Reduce harsh sibilant sounds (s, sh, ch)",
    },
    "amplify": {
        "filter": "volume=3dB",
        "display_name": "Amplify (+3dB)",
        "description": "Increase volume by 3 decibels",
    },
}


def get_available_effects() -> Dict[str, Dict[str, str]]:
    """Return dictionary of available DSP effects with metadata."""
    return AVAILABLE_EFFECTS.copy()


def get_effect_names() -> List[str]:
    """Return list of available effect names."""
    return list(AVAILABLE_EFFECTS.keys())


class FFmpegDSPBackend:
    """
    DSP preprocessing using FFmpeg audio filters.

    Implements SpeechEnhancer protocol but designed for
    lightweight preprocessing before AI enhancement.

    Unlike AI enhancers (ZipEnhancer, ClearVoice), this backend:
    - Uses no GPU (CPU-only via FFmpeg)
    - Has zero model loading time
    - Can chain multiple effects efficiently
    - Works with any sample rate

    Example:
        # Create with specific effects
        backend = FFmpegDSPBackend(effects=["loudnorm", "compress"])

        # Enhance audio
        result = backend.enhance("/path/to/audio.wav", sample_rate=16000)

        # Cleanup (no-op for FFmpeg, but call for protocol compliance)
        backend.cleanup()
    """

    def __init__(
        self,
        effects: Optional[List[str]] = None,
        model: Optional[str] = None,  # Ignored, for protocol compatibility
        **kwargs
    ):
        """
        Initialize FFmpeg DSP backend.

        Args:
            effects: List of effect names to apply (e.g., ["loudnorm", "compress"])
                    If None or empty, defaults to ["loudnorm"]
            model: Ignored parameter for protocol compatibility
            **kwargs: Additional parameters (ignored)
        """
        # Parse effects from model parameter if provided (for factory compatibility)
        if model and not effects:
            effects = model.split(",")

        self.effects = effects or ["loudnorm"]
        self._validate_effects()

        logger.debug(f"FFmpegDSPBackend initialized with effects: {self.effects}")

    def _validate_effects(self) -> None:
        """Validate that all requested effects exist."""
        invalid = [e for e in self.effects if e not in AVAILABLE_EFFECTS]
        if invalid:
            logger.warning(
                f"Unknown DSP effects ignored: {invalid}. "
                f"Available: {list(AVAILABLE_EFFECTS.keys())}"
            )
            self.effects = [e for e in self.effects if e in AVAILABLE_EFFECTS]

        if not self.effects:
            logger.warning("No valid effects specified, defaulting to loudnorm")
            self.effects = ["loudnorm"]

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "ffmpeg"

    @property
    def display_name(self) -> str:
        """Human-readable name for GUI."""
        if not self.effects:
            return "FFmpeg DSP (No effects)"
        effect_names = [AVAILABLE_EFFECTS[e]["display_name"].split()[0] for e in self.effects]
        return f"FFmpeg DSP ({', '.join(effect_names)})"

    @property
    def is_lightweight(self) -> bool:
        """FFmpeg DSP uses no GPU, always lightweight."""
        return True

    def get_preferred_sample_rate(self) -> int:
        """Return 48kHz (v1.7.4+ contract: scene files are always 48kHz).

        FFmpeg DSP benefits from higher sample rates:
        - More frequency bins for afftdn noise estimation
        - Deesser can target full sibilant range (4-12kHz)
        - Better loudnorm LUFS measurement accuracy
        """
        return 48000

    def get_output_sample_rate(self) -> int:
        """Return 16kHz (v1.7.4+ contract: output is always 16kHz for VAD/ASR)."""
        return 16000

    def get_supported_models(self) -> List[str]:
        """Return available effect combinations as 'models'."""
        return list(AVAILABLE_EFFECTS.keys())

    def _build_filter_chain(self) -> str:
        """Build FFmpeg filter chain string from effects list."""
        filters = [AVAILABLE_EFFECTS[e]["filter"] for e in self.effects]
        return ",".join(filters)

    def enhance(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int,
        **kwargs
    ) -> EnhancementResult:
        """
        Apply DSP effects to audio using FFmpeg.

        Args:
            audio: Audio data as numpy array or path to audio file
            sample_rate: Sample rate of input audio
            **kwargs: Additional parameters (ignored)

        Returns:
            EnhancementResult with processed audio

        Note:
            If FFmpeg fails, returns original audio with success=False
            for graceful degradation.
        """
        start_time = time.time()

        # Handle file path input
        if isinstance(audio, (str, Path)):
            input_path = Path(audio)
            if not input_path.exists():
                return create_failed_result(
                    audio=np.zeros(1, dtype=np.float32),
                    sample_rate=sample_rate,
                    method=self.name,
                    error_message=f"Input file not found: {input_path}",
                    processing_time_sec=time.time() - start_time,
                )
            is_file_input = True
        else:
            # numpy array input - need to write to temp file
            is_file_input = False
            try:
                audio_data, actual_sr = load_audio_to_array(audio, sample_rate)
            except Exception as e:
                return create_failed_result(
                    audio=audio if isinstance(audio, np.ndarray) else np.zeros(1, dtype=np.float32),
                    sample_rate=sample_rate,
                    method=self.name,
                    error_message=f"Failed to load audio: {e}",
                    processing_time_sec=time.time() - start_time,
                )

        try:
            # Build filter chain
            filter_chain = self._build_filter_chain()
            logger.debug(f"FFmpeg filter chain: {filter_chain}")

            # Create temp files for processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as in_tmp:
                input_temp = Path(in_tmp.name)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as out_tmp:
                output_temp = Path(out_tmp.name)

            try:
                # Write input if numpy array
                if not is_file_input:
                    import soundfile as sf
                    sf.write(str(input_temp), audio_data, sample_rate)
                    ffmpeg_input = str(input_temp)
                else:
                    ffmpeg_input = str(input_path)

                # Run FFmpeg
                cmd = [
                    "ffmpeg", "-y",
                    "-i", ffmpeg_input,
                    "-af", filter_chain,
                    "-ar", str(sample_rate),  # Preserve sample rate
                    "-ac", "1",  # Force mono
                    str(output_temp)
                ]

                logger.debug(f"FFmpeg command: {' '.join(cmd)}")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

                if result.returncode != 0:
                    error_msg = result.stderr[:500] if result.stderr else "Unknown FFmpeg error"
                    logger.warning(f"FFmpeg failed: {error_msg}")

                    # Return original audio on failure
                    if is_file_input:
                        import soundfile as sf
                        original_audio, _ = sf.read(str(input_path), dtype='float32')
                    else:
                        original_audio = audio_data

                    return create_failed_result(
                        audio=original_audio,
                        sample_rate=sample_rate,
                        method=self.name,
                        error_message=f"FFmpeg error: {error_msg}",
                        processing_time_sec=time.time() - start_time,
                    )

                # Read output
                import soundfile as sf
                enhanced_audio, output_sr = sf.read(str(output_temp), dtype='float32')

                # Ensure mono
                if enhanced_audio.ndim > 1:
                    enhanced_audio = np.mean(enhanced_audio, axis=1)

                processing_time = time.time() - start_time

                logger.debug(
                    f"FFmpeg DSP complete: {len(self.effects)} effects in {processing_time:.2f}s"
                )

                return EnhancementResult(
                    audio=enhanced_audio.astype(np.float32),
                    sample_rate=output_sr,
                    method=self.name,
                    parameters={
                        "effects": self.effects,
                        "filter_chain": filter_chain,
                    },
                    processing_time_sec=processing_time,
                    metadata={
                        "input_samples": len(audio_data) if not is_file_input else None,
                        "output_samples": len(enhanced_audio),
                        "effects_applied": len(self.effects),
                    },
                    success=True,
                    error_message=None,
                )

            finally:
                # Cleanup temp files
                input_temp.unlink(missing_ok=True)
                output_temp.unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            return create_failed_result(
                audio=audio_data if not is_file_input else np.zeros(1, dtype=np.float32),
                sample_rate=sample_rate,
                method=self.name,
                error_message="FFmpeg processing timed out after 5 minutes",
                processing_time_sec=time.time() - start_time,
            )
        except Exception as e:
            logger.warning(f"FFmpeg DSP error: {e}")
            return create_failed_result(
                audio=audio_data if not is_file_input else np.zeros(1, dtype=np.float32),
                sample_rate=sample_rate,
                method=self.name,
                error_message=str(e),
                processing_time_sec=time.time() - start_time,
            )

    def cleanup(self) -> None:
        """
        Release resources.

        FFmpeg backend has no persistent resources, but implements
        cleanup() for protocol compliance.
        """
        pass  # No resources to release

    def __repr__(self) -> str:
        return f"FFmpegDSPBackend(effects={self.effects})"
