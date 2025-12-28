"""
Pipeline integration helper for speech enhancement.

Provides a clean interface for pipelines to integrate speech enhancement
without extensive code changes. Handles:
- Extracting audio at consistent 48kHz for scene files
- Enhancing scene audio files
- Resampling to 16kHz for VAD/ASR
- Graceful degradation on failure
- Resource cleanup

CONTRACTS (v1.7.4+):
    Input:  Scene files are ALWAYS 48kHz mono (SCENE_EXTRACTION_SR)
    Output: Enhanced files are ALWAYS 16kHz mono (TARGET_SAMPLE_RATE)

Usage in pipelines:
    from whisperjav.modules.speech_enhancement.pipeline_helper import (
        create_enhancer_from_config,
        SCENE_EXTRACTION_SR,
        enhance_scenes,
    )

    # In __init__ - ALWAYS use 48kHz for scene extraction
    self.audio_extractor = AudioExtractor(sample_rate=SCENE_EXTRACTION_SR)

    # After scene detection - ALWAYS run enhancement (even for "none" backend)
    enhancer = create_enhancer_from_config(resolved_config)
    scene_paths = enhance_scenes(scene_paths, enhancer, self.temp_dir)
    enhancer.cleanup()  # Free GPU before ASR
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import gc
import logging
import time
import soundfile as sf

from .factory import SpeechEnhancerFactory
from .base import SpeechEnhancer, resample_audio

logger = logging.getLogger("whisperjav")

# Contract: Scene files are ALWAYS extracted at 48kHz mono
SCENE_EXTRACTION_SR = 48000

# Contract: Enhanced files for VAD/ASR are ALWAYS 16kHz mono
TARGET_SAMPLE_RATE = 16000


def create_enhancer_from_config(
    resolved_config: Dict[str, Any],
    **overrides
) -> SpeechEnhancer:
    """
    Create a speech enhancer from resolved pipeline configuration.

    ALWAYS returns an enhancer (v1.7.4+ clean contract).
    The "none" backend performs 48kHz→16kHz resampling without processing.

    Args:
        resolved_config: V3 resolved config dict with 'params' key
        **overrides: Override specific enhancer parameters

    Returns:
        SpeechEnhancer instance (never None - "none" backend is valid)

    Example:
        enhancer = create_enhancer_from_config(resolved_config)
        scene_paths = enhance_scenes(scene_paths, enhancer, temp_dir)
        enhancer.cleanup()
    """
    params = resolved_config.get("params", {})
    enhancer_config = params.get("speech_enhancer", {})

    # Merge overrides
    if overrides:
        enhancer_config = {**enhancer_config, **overrides}

    # Get backend name (default: none = passthrough with resampling)
    backend = enhancer_config.get("backend", "none")

    # Normalize empty backend to "none"
    if not backend:
        backend = "none"

    # Check availability (except for "none" which is always available)
    if backend != "none":
        available, hint = SpeechEnhancerFactory.is_backend_available(backend)
        if not available:
            logger.warning(
                f"Speech enhancer '{backend}' not available: {hint}. "
                "Falling back to 'none' (passthrough with resampling)."
            )
            backend = "none"

    # Create enhancer - ALWAYS succeeds (none is always available)
    try:
        enhancer = SpeechEnhancerFactory.create(backend, config=enhancer_config)
        logger.info(f"Speech enhancer created: {enhancer.display_name}")
        return enhancer
    except Exception as e:
        logger.warning(f"Failed to create speech enhancer '{backend}': {e}. Falling back to 'none'.")
        # Fallback to none backend - guaranteed to work
        return SpeechEnhancerFactory.create("none", config={})


def create_enhancer_direct(
    backend: str,
    model: Optional[str] = None,
    **kwargs
) -> SpeechEnhancer:
    """
    Create a speech enhancer directly (for TransformersPipeline).

    ALWAYS returns an enhancer (v1.7.4+ clean contract).
    The "none" backend performs 48kHz→16kHz resampling without processing.

    Args:
        backend: Enhancer backend name ("none", "clearvoice", "bs-roformer", "ffmpeg-dsp")
        model: Optional model variant
        **kwargs: Additional parameters

    Returns:
        SpeechEnhancer instance (never None - "none" backend is valid)
    """
    # Normalize empty backend to "none"
    if not backend:
        backend = "none"

    # Check availability (except for "none" which is always available)
    if backend != "none":
        available, hint = SpeechEnhancerFactory.is_backend_available(backend)
        if not available:
            logger.warning(
                f"Speech enhancer '{backend}' not available: {hint}. "
                "Falling back to 'none' (passthrough with resampling)."
            )
            backend = "none"

    # Create enhancer - ALWAYS succeeds (none is always available)
    try:
        params = {**kwargs}
        if model:
            params["model"] = model
        enhancer = SpeechEnhancerFactory.create(backend, config=params)
        logger.info(f"Speech enhancer created: {enhancer.display_name}")
        return enhancer
    except Exception as e:
        logger.warning(f"Failed to create speech enhancer '{backend}': {e}. Falling back to 'none'.")
        return SpeechEnhancerFactory.create("none", config={})


def get_extraction_sample_rate(enhancer: Optional[SpeechEnhancer] = None) -> int:
    """
    Get the sample rate to use for audio extraction.

    ALWAYS returns SCENE_EXTRACTION_SR (48kHz) per v1.7.4+ clean contract.
    The enhancer parameter is ignored but kept for backwards compatibility.

    Args:
        enhancer: Ignored (kept for backwards compatibility)

    Returns:
        SCENE_EXTRACTION_SR (48000 Hz)
    """
    return SCENE_EXTRACTION_SR


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
        f"(input: {enhancer_sr}Hz -> output: {TARGET_SAMPLE_RATE}Hz)"
    )

    # Pre-warm the enhancer model before starting progress display
    # This ensures model download/loading happens before the progress bar starts
    if hasattr(enhancer, '_ensure_initialized'):
        enhancer._ensure_initialized()

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

        finally:
            # Aggressive memory cleanup for 8GB VRAM GPUs
            # PyTorch's memory caching allocator holds onto CUDA memory between
            # loop iterations, causing OOM on scene 8+ if not explicitly released.
            # This cleanup ensures bounded memory usage regardless of scene count.
            try:
                # Delete references to large arrays/tensors
                try:
                    del audio_data
                except NameError:
                    pass
                try:
                    del enhanced_audio
                except NameError:
                    pass
                try:
                    del result
                except NameError:
                    pass

                # Force Python garbage collection to release tensor references
                gc.collect()

                # Return PyTorch's cached memory to CUDA driver
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug(f"Scene {scene_num}/{total_scenes}: CUDA cache cleared")
                except ImportError:
                    pass  # torch not available, skip CUDA cleanup
                except Exception as cuda_err:
                    # CUDA context may be corrupted - log but continue
                    logger.debug(f"Scene {scene_num}/{total_scenes}: CUDA cache clear failed: {cuda_err}")

            except Exception as cleanup_err:
                # Non-critical, log and continue processing
                logger.debug(f"Scene {scene_num}/{total_scenes}: cleanup exception: {cleanup_err}")

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
                f"Audio enhanced: {audio_path.name} -> {output_path.name} "
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

    finally:
        # Memory cleanup after enhancement
        try:
            try:
                del audio_data
            except NameError:
                pass
            try:
                del enhanced_audio
            except NameError:
                pass
            try:
                del result
            except NameError:
                pass
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        except Exception:
            pass
