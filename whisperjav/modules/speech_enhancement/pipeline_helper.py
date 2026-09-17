"""
Pipeline integration helper for speech enhancement.

Provides a clean interface for pipelines to integrate speech enhancement
without extensive code changes. Handles:
- Dynamic extraction SR: 16kHz when enhancer is "none", 48kHz for real enhancers
- Enhancing scene audio files (when a real enhancer is configured)
- Resampling to 16kHz for VAD/ASR (when extracting at 48kHz)
- Graceful degradation on failure
- Resource cleanup

==============================================================================
AUDIO SAMPLE RATE CONTRACT (v1.8.0+)
==============================================================================

EXTRACTION (v1.8.5+):
  - If enhancer backend is "none" (default) → extract at 16kHz (direct ASR path)
  - If a real enhancer is configured → extract at 48kHz (enhancer needs high-SR)
  - get_extraction_sample_rate() returns the correct SR based on backend name
  - FFmpeg handles all extraction sample rate conversion

SCENE DETECTION (Auditok):
  - Receives audio at extraction SR (NO resampling needed)
  - auditok handles any sample rate natively via sampling_rate parameter
  - Scene files saved at extraction SR
  - Note: v1.8.0 removed unnecessary 48kHz→16kHz resample that caused
    10-30 minute "hangs" on 2+ hour files (Issue #129)

SCENE DETECTION (Silero Pass 2):
  - Silero VAD requires 16kHz internally
  - _detect_pass2_silero() handles its own resampling per-region
  - Only small region chunks (~30-90s) are resampled, not full file

VAD/ASR:
  - If input is not 16kHz → resample to 16kHz (ONLY resample point)
  - Silero VAD requires 16kHz
  - Whisper ASR requires 16kHz

==============================================================================

CONTRACTS (v1.8.5+):
    When enhancer backend is "none" (passthrough):
        - Extract at 16kHz (TARGET_SAMPLE_RATE) — scenes go directly to VAD/ASR
        - Skip enhance_scenes() entirely — no enhanced_scenes/ folder created
        - No disk I/O or CPU resampling overhead

    When a real enhancer is configured (clearvoice, bs-roformer, etc.):
        - Extract at 48kHz (SCENE_EXTRACTION_SR) — enhancers need high-SR input
        - enhance_scenes() runs the full pipeline (enhance → resample → save)
        - Output: Enhanced files at 16kHz mono (TARGET_SAMPLE_RATE)

Usage in pipelines:
    from whisperjav.modules.speech_enhancement.pipeline_helper import (
        create_enhancer_from_config,
        get_extraction_sample_rate,
        is_passthrough_backend,
        enhance_scenes,
    )

    # In __init__ - use dynamic extraction SR based on enhancer backend
    extraction_sr = get_extraction_sample_rate(backend_name)
    self.audio_extractor = AudioExtractor(sample_rate=extraction_sr)

    # After scene detection - conditionally run enhancement
    if is_passthrough_backend(backend_name):
        # Scenes already at 16kHz — skip enhancement entirely
        pass
    else:
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

from .factory import FATAL_WHEN_UNAVAILABLE, SpeechEnhancerFactory
from .base import SpeechEnhancer, SpeechEnhancerUnavailable, resample_audio

logger = logging.getLogger("whisperjav")

# Extraction SR when a real enhancer is configured (enhancers need 48kHz input)
SCENE_EXTRACTION_SR = 48000

# Target SR for VAD/ASR — also the extraction SR when enhancer is "none"
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
            if backend in FATAL_WHEN_UNAVAILABLE:
                # Chosen because the audio needs it. Handing back the original
                # audio would produce a poor subtitle file from a run that
                # exited 0, with only a warning in the log to explain it.
                raise SpeechEnhancerUnavailable(
                    f"The '{backend}' audio clean-up was asked for, but it is "
                    f"not installed on this machine. {hint}"
                )
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
    except SpeechEnhancerUnavailable:
        raise
    except Exception as e:
        if backend in FATAL_WHEN_UNAVAILABLE:
            raise SpeechEnhancerUnavailable(
                f"The '{backend}' audio clean-up could not be started: {e}"
            )
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
            if backend in FATAL_WHEN_UNAVAILABLE:
                # Chosen because the audio needs it. Handing back the original
                # audio would produce a poor subtitle file from a run that
                # exited 0, with only a warning in the log to explain it.
                raise SpeechEnhancerUnavailable(
                    f"The '{backend}' audio clean-up was asked for, but it is "
                    f"not installed on this machine. {hint}"
                )
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
    except SpeechEnhancerUnavailable:
        raise
    except Exception as e:
        if backend in FATAL_WHEN_UNAVAILABLE:
            raise SpeechEnhancerUnavailable(
                f"The '{backend}' audio clean-up could not be started: {e}"
            )
        logger.warning(f"Failed to create speech enhancer '{backend}': {e}. Falling back to 'none'.")
        return SpeechEnhancerFactory.create("none", config={})


def get_extraction_sample_rate(enhancer_backend: Optional[str] = None) -> int:
    """
    Get the sample rate for audio extraction based on enhancer backend.

    - "none" or empty: 16kHz (no enhancement needed, extract at ASR target SR)
    - Any real backend: 48kHz (enhancers work best with high-SR input)

    Note: If a real backend (e.g. "clearvoice") is configured but unavailable
    at runtime, create_enhancer_from_config/create_enhancer_direct will fall
    back to NullSpeechEnhancer.  In that case scenes are already at 48kHz,
    so enhance_scenes() still runs the 48→16 kHz resample — same as before.
    The 16kHz shortcut only fires when the user *explicitly* configures "none".

    Args:
        enhancer_backend: Backend name string, or None for passthrough

    Returns:
        TARGET_SAMPLE_RATE (16000) for passthrough, SCENE_EXTRACTION_SR (48000) otherwise
    """
    if is_passthrough_backend(enhancer_backend):
        return TARGET_SAMPLE_RATE
    return SCENE_EXTRACTION_SR


def is_passthrough_backend(backend_name: Optional[str]) -> bool:
    """Whether the named backend is a no-op passthrough (no audio processing)."""
    return not backend_name or backend_name == "none"


def _scene_at_target_rate(scene_path: Path, destination: Path) -> Path:
    """
    Write a 16kHz mono copy of one scene and return its path.

    Everything that leaves this module is paired by position with everything
    else, and the dual-track path refuses two tracks recorded at different
    rates, so a scene that skips the resampling step fails the whole file with a
    message about sample rates rather than about the clean-up that actually
    failed.

    Returns the original path if the copy cannot be made -- at that point there
    is nothing better to hand back.
    """
    import numpy as np

    try:
        audio_data, actual_sr = sf.read(str(scene_path), dtype='float32')

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if actual_sr != TARGET_SAMPLE_RATE:
            audio_data = resample_audio(audio_data, actual_sr, TARGET_SAMPLE_RATE)

        sf.write(str(destination), audio_data, TARGET_SAMPLE_RATE)
        return destination
    except Exception as e:
        logger.warning(
            "Could not resample scene %s to %dHz: %s. Using it as it is.",
            scene_path.name, TARGET_SAMPLE_RATE, e,
        )
        return scene_path


def resample_scenes(
    scene_paths: List[Tuple[Path, float, float, float]],
    temp_dir: Path,
) -> List[Tuple[Path, float, float, float]]:
    """
    Resample scene audio files from extraction SR (48kHz) to 16kHz without enhancement.

    Used by the dual-track ``--enhance-for-vad`` mode: the original (non-enhanced)
    scenes are resampled to 16kHz so they can be fed to the ASR generator, while
    the enhanced copies are used only for VAD framing.

    Args:
        scene_paths: List of (scene_path, start_sec, end_sec, duration_sec)
                     at extraction sample rate (typically 48kHz).
        temp_dir: Temporary directory (``resampled_scenes/`` will be created here).

    Returns:
        List of (resampled_path, start_sec, end_sec, duration_sec).
        Same structure as input, but paths point to 16kHz mono WAV files.
    """
    if not scene_paths:
        return scene_paths

    resampled_dir = temp_dir / "resampled_scenes"
    resampled_dir.mkdir(exist_ok=True)

    resampled_paths = []
    for scene_path, start_sec, end_sec, dur_sec in scene_paths:
        resampled_path = _scene_at_target_rate(
            scene_path, resampled_dir / f"{scene_path.stem}_resampled.wav")
        resampled_paths.append((resampled_path, start_sec, end_sec, dur_sec))

    logger.info("Resampled %d scenes to %dHz (dual-track ASR path)", len(resampled_paths), TARGET_SAMPLE_RATE)
    return resampled_paths


def enhance_scenes(
    scene_paths: List[Tuple[Path, float, float, float]],
    enhancer: SpeechEnhancer,
    temp_dir: Path,
    progress_callback: Optional[callable] = None,
    degradations: Optional[List[str]] = None,
) -> List[Tuple[Path, float, float, float]]:
    """
    Enhance scene audio files and resample to 16kHz for ASR.

    This is the main integration point for pipelines. It:
    1. Creates an 'enhanced_scenes' directory
    2. For each scene: enhance audio, resample to 16kHz, save
    3. Returns new scene paths pointing to enhanced files
    4. On failure of one scene: logs a warning, uses that scene unenhanced

    Args:
        scene_paths: List of (scene_path, start_sec, end_sec, duration_sec)
        enhancer: Active SpeechEnhancer instance
        temp_dir: Temporary directory (enhanced_scenes will be created here)
        progress_callback: Optional callback(scene_num, total_scenes, scene_name)
        degradations: Optional list. A plain-language line is appended to it for
            anything the user should be told about in the run summary -- today,
            scenes that went through without being cleaned up.

    Returns:
        List of (enhanced_scene_path, start_sec, end_sec, duration_sec)
        Same structure as input, but paths point to enhanced files

    Raises:
        SpeechEnhancerUnavailable: if the clean-up cannot run at all -- it is not
            installed, it could not start, or it failed on EVERY scene. The
            owner's rule (2026-09-17): a component the user chose that cannot
            install or start is a failure and the run stops. A clean-up that ran
            and achieved nothing is the same thing by a different route, so it is
            treated the same (agreed error-handling table, 2026-09-17).

    Note:
        If enhancement fails for SOME scenes, a 16kHz copy of each failed
        original is used and the run goes on -- only all-fail is fatal (owner,
        2026-09-17: "only all-fail is fatal, agreed"). The count of scenes that
        went through unenhanced is returned to the caller through
        ``degradations`` so it can reach the run summary instead of living in a
        log line nobody reads.

        The fallback is a copy rather than the scene itself because every path
        returned here is at 16kHz, and the dual-track path pairs these scenes
        with originals at that rate and refuses two tracks recorded at different
        rates.
    """
    if not scene_paths:
        return scene_paths

    total_scenes = len(scene_paths)
    enhancer_sr = enhancer.get_preferred_sample_rate()
    enhanced_dir = temp_dir / "enhanced_scenes"
    enhanced_dir.mkdir(exist_ok=True)

    enhanced_paths = []
    failed_scenes = []
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
                # Enhancement failed on this scene - use the original, at the
                # rate the enhanced scenes come back at, so the lists stay
                # comparable.
                logger.warning(
                    f"Scene {scene_num} enhancement failed: {result.error_message}. "
                    "Using original."
                )
                failed_scenes.append((scene_num, result.error_message))
                enhanced_paths.append((
                    _scene_at_target_rate(
                        scene_path,
                        enhanced_dir / f"{scene_path.stem}_original_{TARGET_SAMPLE_RATE}.wav"),
                    start_sec, end_sec, dur_sec))

        except SpeechEnhancerUnavailable:
            # The clean-up itself cannot run. Carrying on scene by scene would
            # repeat the same failure for every scene and end with a subtitle
            # file made from audio the user asked to have cleaned up.
            raise
        except Exception as e:
            logger.warning(
                f"Scene {scene_num} enhancement error: {e}. Using original."
            )
            failed_scenes.append((scene_num, str(e)))
            enhanced_paths.append((
                _scene_at_target_rate(
                    scene_path,
                    enhanced_dir / f"{scene_path.stem}_original_{TARGET_SAMPLE_RATE}.wav"),
                start_sec, end_sec, dur_sec))

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

    if failed_scenes:
        # Owner, 2026-09-17: "only all-fail is fatal, agreed." A clean-up that
        # ran and cleaned up nothing is the same as one that could not run, so
        # it is refused the same way. Anything short of that is a shortfall the
        # user is told about, not a reason to throw away the transcription.
        if len(failed_scenes) == total_scenes:
            first_reason = failed_scenes[0][1]
            raise SpeechEnhancerUnavailable(
                f"{enhancer.display_name} failed on every one of the "
                f"{total_scenes} pieces of this audio, so nothing was cleaned "
                f"up. The first failure was: {first_reason}. The run has stopped "
                f"rather than give you subtitles made from audio you asked to "
                f"have cleaned up. Choose a different clean-up, or none."
            )

        note = (f"{len(failed_scenes)} of {total_scenes} scenes went through "
                f"without being cleaned up by {enhancer.display_name} "
                f"(first: {failed_scenes[0][1]})")
        logger.warning(note)
        if degradations is not None:
            degradations.append(note)

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

    except SpeechEnhancerUnavailable:
        raise
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
