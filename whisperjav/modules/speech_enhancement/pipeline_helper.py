"""
Pipeline integration helper for speech enhancement.

Provides a clean interface for pipelines to integrate speech enhancement
without extensive code changes. Handles:
- Determining extraction sample rate based on enhancer
- Enhancing scene audio files
- Resampling to 16kHz for VAD/ASR
- Graceful degradation on failure
- Resource cleanup

Usage in pipelines:
    from whisperjav.modules.speech_enhancement.pipeline_helper import (
        create_enhancer_from_config,
        get_extraction_sample_rate,
        enhance_scenes,
    )

    # In __init__
    self.speech_enhancer = create_enhancer_from_config(resolved_config)
    extraction_sr = get_extraction_sample_rate(self.speech_enhancer)
    self.audio_extractor = AudioExtractor(sample_rate=extraction_sr)

    # After scene detection, before transcription
    if self.speech_enhancer:
        scene_paths = enhance_scenes(
            scene_paths, self.speech_enhancer, self.temp_dir, logger
        )
        self.speech_enhancer.cleanup()  # Free GPU before ASR
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import soundfile as sf

from .factory import SpeechEnhancerFactory
from .base import SpeechEnhancer, resample_audio

logger = logging.getLogger("whisperjav")

# Target sample rate for VAD/ASR
TARGET_SAMPLE_RATE = 16000


def create_enhancer_from_config(
    resolved_config: Dict[str, Any],
    **overrides
) -> Optional[SpeechEnhancer]:
    """
    Create a speech enhancer from resolved pipeline configuration.

    Args:
        resolved_config: V3 resolved config dict with 'params' key
        **overrides: Override specific enhancer parameters

    Returns:
        SpeechEnhancer instance, or None if enhancement disabled

    Example:
        enhancer = create_enhancer_from_config(resolved_config)
        if enhancer:
            # Enhancement is enabled
            extraction_sr = enhancer.get_preferred_sample_rate()
    """
    params = resolved_config.get("params", {})
    enhancer_config = params.get("speech_enhancer", {})

    # Merge overrides
    if overrides:
        enhancer_config = {**enhancer_config, **overrides}

    # Get backend name
    backend = enhancer_config.get("backend", "none")

    # If no backend or "none", return None
    if not backend or backend == "none":
        logger.debug("Speech enhancement disabled (backend='none')")
        return None

    # Check availability
    available, hint = SpeechEnhancerFactory.is_backend_available(backend)
    if not available:
        logger.warning(
            f"Speech enhancer '{backend}' not available: {hint}. "
            "Continuing without enhancement."
        )
        return None

    # Create enhancer
    try:
        enhancer = SpeechEnhancerFactory.create(backend, config=enhancer_config)
        logger.info(f"Speech enhancer created: {enhancer.display_name}")
        return enhancer
    except Exception as e:
        logger.warning(f"Failed to create speech enhancer: {e}. Continuing without enhancement.")
        return None


def create_enhancer_direct(
    backend: str,
    model: Optional[str] = None,
    **kwargs
) -> Optional[SpeechEnhancer]:
    """
    Create a speech enhancer directly (for TransformersPipeline).

    Args:
        backend: Enhancer backend name ("none", "clearvoice", "bs-roformer")
        model: Optional model variant
        **kwargs: Additional parameters

    Returns:
        SpeechEnhancer instance, or None if enhancement disabled
    """
    if not backend or backend == "none":
        return None

    # Check availability
    available, hint = SpeechEnhancerFactory.is_backend_available(backend)
    if not available:
        logger.warning(
            f"Speech enhancer '{backend}' not available: {hint}. "
            "Continuing without enhancement."
        )
        return None

    # Create enhancer
    try:
        params = {**kwargs}
        if model:
            params["model"] = model
        enhancer = SpeechEnhancerFactory.create(backend, config=params)
        logger.info(f"Speech enhancer created: {enhancer.display_name}")
        return enhancer
    except Exception as e:
        logger.warning(f"Failed to create speech enhancer: {e}. Continuing without enhancement.")
        return None


def get_extraction_sample_rate(enhancer: Optional[SpeechEnhancer]) -> int:
    """
    Get the sample rate to use for audio extraction.

    If an enhancer is active, returns its preferred rate.
    Otherwise returns 16kHz (standard for VAD/ASR).

    Args:
        enhancer: SpeechEnhancer instance or None

    Returns:
        Sample rate in Hz
    """
    if enhancer is None:
        return TARGET_SAMPLE_RATE
    return enhancer.get_preferred_sample_rate()


def enhance_scenes(
    scene_paths: List[Tuple[Path, float, float, float]],
    enhancer: SpeechEnhancer,
    temp_dir: Path,
    progress_callback: Optional[callable] = None,
) -> List[Tuple[Path, float, float, float]]:
    """
    Enhance scene audio files and resample to 16kHz for ASR.

    This is the main integration point for pipelines. It:
    1. Creates an 'enhanced_scenes' directory
    2. For each scene: enhance audio, resample to 16kHz, save
    3. Returns new scene paths pointing to enhanced files
    4. On failure: logs warning, returns original scene unchanged

    Args:
        scene_paths: List of (scene_path, start_sec, end_sec, duration_sec)
        enhancer: Active SpeechEnhancer instance
        temp_dir: Temporary directory (enhanced_scenes will be created here)
        progress_callback: Optional callback(scene_num, total_scenes, scene_name)

    Returns:
        List of (enhanced_scene_path, start_sec, end_sec, duration_sec)
        Same structure as input, but paths point to enhanced files

    Note:
        If enhancement fails for a scene, the original scene is used.
        This ensures graceful degradation.
    """
    if not scene_paths:
        return scene_paths

    total_scenes = len(scene_paths)
    enhancer_sr = enhancer.get_preferred_sample_rate()
    enhanced_dir = temp_dir / "enhanced_scenes"
    enhanced_dir.mkdir(exist_ok=True)

    enhanced_paths = []
    enhancement_start = time.time()

    logger.info(
        f"Enhancing {total_scenes} scenes with {enhancer.display_name} "
        f"(input: {enhancer_sr}Hz → output: {TARGET_SAMPLE_RATE}Hz)"
    )

    for idx, (scene_path, start_sec, end_sec, dur_sec) in enumerate(scene_paths):
        scene_num = idx + 1

        if progress_callback:
            progress_callback(scene_num, total_scenes, scene_path.name)

        enhanced_path = enhanced_dir / f"{scene_path.stem}_enhanced.wav"

        try:
            # Load scene audio
            audio_data, actual_sr = sf.read(str(scene_path), dtype='float32')

            # Convert stereo to mono if needed
            if audio_data.ndim > 1:
                import numpy as np
                audio_data = np.mean(audio_data, axis=1)

            # Resample to enhancer's preferred rate if needed
            if actual_sr != enhancer_sr:
                audio_data = resample_audio(audio_data, actual_sr, enhancer_sr)

            # Enhance
            result = enhancer.enhance(audio_data, enhancer_sr)

            if result.success:
                enhanced_audio = result.audio
                output_sr = result.sample_rate

                # Resample to 16kHz for ASR if needed
                if output_sr != TARGET_SAMPLE_RATE:
                    enhanced_audio = resample_audio(
                        enhanced_audio, output_sr, TARGET_SAMPLE_RATE
                    )

                # Save enhanced audio
                sf.write(str(enhanced_path), enhanced_audio, TARGET_SAMPLE_RATE)

                enhanced_paths.append((enhanced_path, start_sec, end_sec, dur_sec))
                logger.debug(
                    f"Scene {scene_num}/{total_scenes} enhanced: "
                    f"{result.processing_time_sec:.2f}s"
                )
            else:
                # Enhancement failed - use original
                logger.warning(
                    f"Scene {scene_num} enhancement failed: {result.error_message}. "
                    "Using original."
                )
                enhanced_paths.append((scene_path, start_sec, end_sec, dur_sec))

        except Exception as e:
            logger.warning(
                f"Scene {scene_num} enhancement error: {e}. Using original."
            )
            enhanced_paths.append((scene_path, start_sec, end_sec, dur_sec))

    total_time = time.time() - enhancement_start
    logger.info(f"Enhancement complete: {total_scenes} scenes in {total_time:.1f}s")

    return enhanced_paths


def enhance_single_audio(
    audio_path: Path,
    enhancer: SpeechEnhancer,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Enhance a single audio file (for pipelines without scene detection).

    Args:
        audio_path: Path to input audio
        enhancer: Active SpeechEnhancer instance
        output_path: Optional output path (default: same dir with _enhanced suffix)

    Returns:
        Path to enhanced audio (at 16kHz), or original path if enhancement fails
    """
    if output_path is None:
        output_path = audio_path.parent / f"{audio_path.stem}_enhanced.wav"

    try:
        enhancer_sr = enhancer.get_preferred_sample_rate()

        # Load audio
        audio_data, actual_sr = sf.read(str(audio_path), dtype='float32')

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            import numpy as np
            audio_data = np.mean(audio_data, axis=1)

        # Resample to enhancer's preferred rate if needed
        if actual_sr != enhancer_sr:
            audio_data = resample_audio(audio_data, actual_sr, enhancer_sr)

        # Enhance
        result = enhancer.enhance(audio_data, enhancer_sr)

        if result.success:
            enhanced_audio = result.audio
            output_sr = result.sample_rate

            # Resample to 16kHz for ASR if needed
            if output_sr != TARGET_SAMPLE_RATE:
                enhanced_audio = resample_audio(
                    enhanced_audio, output_sr, TARGET_SAMPLE_RATE
                )

            # Save enhanced audio
            sf.write(str(output_path), enhanced_audio, TARGET_SAMPLE_RATE)

            logger.info(
                f"Audio enhanced: {audio_path.name} → {output_path.name} "
                f"({result.processing_time_sec:.2f}s)"
            )
            return output_path
        else:
            logger.warning(
                f"Audio enhancement failed: {result.error_message}. Using original."
            )
            return audio_path

    except Exception as e:
        logger.warning(f"Audio enhancement error: {e}. Using original.")
        return audio_path
