"""
Backend execution module for the pipeline analysis test suite.

Runs scene detection and speech segmentation backends on real audio,
producing BackendRunResult instances with unified SegmentInfo data.

This is the ONLY module that imports from WhisperJAV.
"""

import logging
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import AudioInfo, BackendRunResult, SegmentInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend name validation
# ---------------------------------------------------------------------------


def _suggest_backend_name(
    name: str, known_names: List[str]
) -> Optional[str]:
    """Find closest match for a mistyped backend name using edit distance.

    Returns the best match if edit distance <= 3, else None.
    """
    if not known_names:
        return None

    import difflib

    matches = difflib.get_close_matches(name, known_names, n=1, cutoff=0.5)
    return matches[0] if matches else None


def validate_backend_names(
    names: List[str],
    backend_type: str,
) -> List[str]:
    """Validate backend names against known registries.

    Prints warnings for unrecognized names with "did you mean?" suggestions.
    Returns only valid names.

    Args:
        names: User-provided backend names
        backend_type: "scene" or "seg"

    Returns:
        List of valid backend names (unrecognized names are excluded)
    """
    if backend_type == "scene":
        known = [b["name"] for b in get_available_scene_backends()]
        type_label = "scene detection"
    else:
        known = [b["name"] for b in get_available_seg_backends()]
        type_label = "speech segmentation"

    valid = []
    for name in names:
        if name in known:
            valid.append(name)
        else:
            suggestion = _suggest_backend_name(name, known)
            msg = f"  Warning: '{name}' is not a known {type_label} backend."
            if suggestion:
                msg += f" Did you mean '{suggestion}'?"
            else:
                msg += f" Known backends: {', '.join(known)}"
            print(msg)

    return valid


# ---------------------------------------------------------------------------
# Audio utilities (adapted from test_speech_segmentation_player.py)
# ---------------------------------------------------------------------------


def extract_audio(
    media_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
) -> bool:
    """Extract audio from media file using FFmpeg."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(output_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        return result.returncode == 0
    except Exception as e:
        logger.error("Error extracting audio: %s", e)
        return False


def load_audio(
    audio_path: Path,
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Load audio file and return (audio_data, sample_rate).

    Args:
        audio_path: Path to audio file
        target_sr: If specified, resample to this rate.

    Returns:
        Tuple of (audio_data as float32 ndarray, sample_rate)
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required: pip install soundfile")

    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")

    # Convert stereo to mono if needed
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if requested
    if target_sr is not None and target_sr != sample_rate:
        try:
            import librosa

            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=target_sr
            )
            sample_rate = target_sr
        except ImportError:
            # Fallback: linear interpolation
            ratio = target_sr / sample_rate
            new_length = int(len(audio_data) * ratio)
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            audio_data = np.interp(
                indices, np.arange(len(audio_data)), audio_data
            ).astype(np.float32)
            sample_rate = target_sr

    return audio_data, sample_rate


def prepare_audio(
    media_path: Path,
    sample_rate: int = 16000,
) -> Tuple[AudioInfo, np.ndarray]:
    """Prepare audio from media file — extract if needed, load, return info + data.

    Args:
        media_path: Path to media file (video or audio)
        sample_rate: Target sample rate

    Returns:
        Tuple of (AudioInfo, audio_data as numpy array)
    """
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    is_audio = media_path.suffix.lower() in audio_extensions

    if is_audio:
        audio_path = media_path
        is_extracted = False
    else:
        print(f"  Extracting audio from video @ {sample_rate} Hz...")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = Path(tmp.name)
        tmp.close()

        if not extract_audio(media_path, audio_path, sample_rate):
            raise RuntimeError(f"Failed to extract audio from {media_path}")
        is_extracted = True

    audio_data, sr = load_audio(audio_path, target_sr=sample_rate)
    duration = len(audio_data) / sr

    info = AudioInfo(
        path=audio_path,
        sample_rate=sr,
        duration_sec=duration,
        num_samples=len(audio_data),
        source_media=media_path,
        is_extracted=is_extracted,
    )

    return info, audio_data


def cleanup_audio(audio_info: AudioInfo) -> None:
    """Clean up extracted audio file if we created one."""
    if audio_info.is_extracted and audio_info.path.exists():
        try:
            audio_info.path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Production config loading
# ---------------------------------------------------------------------------


def load_production_vad_config(sensitivity: str) -> Dict[str, Any]:
    """Load VAD parameters from production Pydantic presets.

    Args:
        sensitivity: One of 'conservative', 'balanced', 'aggressive'

    Returns:
        Dict of VAD parameters matching production config
    """
    try:
        from whisperjav.config.components.base import get_vad_registry

        registry = get_vad_registry()
        silero_component = registry.get("silero")

        if silero_component and sensitivity in silero_component.presets:
            preset = silero_component.presets[sensitivity]
            return preset.model_dump()
        else:
            print(
                f"  Warning: silero/{sensitivity} preset not found in registry"
            )
            return {}
    except Exception as e:
        print(f"  Warning: Failed to load production config: {e}")
        return {}


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------


def get_available_scene_backends() -> List[Dict[str, Any]]:
    """Get list of available scene detection backends from factory."""
    try:
        from whisperjav.modules.scene_detection_backends.factory import (
            SceneDetectorFactory,
        )

        return SceneDetectorFactory.get_available_backends()
    except ImportError:
        logger.warning("Could not import SceneDetectorFactory")
        return []


def get_available_seg_backends() -> List[Dict[str, Any]]:
    """Get list of available speech segmentation backends from factory."""
    try:
        from whisperjav.modules.speech_segmentation.factory import (
            SpeechSegmenterFactory,
        )

        return SpeechSegmenterFactory.get_available_backends()
    except ImportError:
        logger.warning("Could not import SpeechSegmenterFactory")
        return []


# ---------------------------------------------------------------------------
# Scene detection runner
# ---------------------------------------------------------------------------


def run_scene_detection(
    backend_name: str,
    audio_path: Path,
    timeout_sec: int = 120,
    **kwargs: Any,
) -> BackendRunResult:
    """Run a single scene detection backend.

    Args:
        backend_name: Scene detector name (auditok, silero, semantic, none)
        audio_path: Path to audio WAV file
        timeout_sec: Maximum execution time
        **kwargs: Extra parameters passed to the scene detector

    Returns:
        BackendRunResult with scene boundaries as SegmentInfo list
    """
    from whisperjav.modules.scene_detection_backends.factory import (
        SceneDetectorFactory,
    )

    # Check availability
    available, hint = SceneDetectorFactory.is_backend_available(backend_name)
    if not available:
        return BackendRunResult(
            backend_name=backend_name,
            backend_type="scene_detection",
            display_name=backend_name,
            available=False,
            success=False,
            error=f"Not available: {hint}",
            processing_time_sec=0.0,
            segments=[],
        )

    result_queue: queue.Queue[BackendRunResult] = queue.Queue()

    def _run() -> None:
        # Create a temp dir for scene WAV extraction (required by API)
        with tempfile.TemporaryDirectory(prefix="pa_scene_") as tmp_dir:
            try:
                detector = SceneDetectorFactory.create(
                    backend_name, **kwargs
                )
                display_name = detector.display_name

                media_basename = audio_path.stem
                start_time = time.time()
                result = detector.detect_scenes(
                    audio_path, Path(tmp_dir), media_basename
                )
                elapsed = time.time() - start_time

                # Convert SceneInfo -> SegmentInfo
                segments = [
                    SegmentInfo(
                        start_sec=scene.start_sec,
                        end_sec=scene.end_sec,
                        metadata={
                            "detection_pass": scene.detection_pass,
                            **(scene.metadata or {}),
                        },
                    )
                    for scene in result.scenes
                ]

                parameters = result.parameters or {}

                detector.cleanup()

                result_queue.put(
                    BackendRunResult(
                        backend_name=backend_name,
                        backend_type="scene_detection",
                        display_name=display_name,
                        available=True,
                        success=True,
                        error=None,
                        processing_time_sec=elapsed,
                        segments=segments,
                        parameters=parameters,
                        method=result.method,
                    )
                )
            except Exception as e:
                result_queue.put(
                    BackendRunResult(
                        backend_name=backend_name,
                        backend_type="scene_detection",
                        display_name=backend_name,
                        available=True,
                        success=False,
                        error=str(e),
                        processing_time_sec=0.0,
                        segments=[],
                    )
                )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        return BackendRunResult(
            backend_name=backend_name,
            backend_type="scene_detection",
            display_name=backend_name,
            available=True,
            success=False,
            error=f"Timeout after {timeout_sec}s",
            processing_time_sec=float(timeout_sec),
            segments=[],
        )

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return BackendRunResult(
            backend_name=backend_name,
            backend_type="scene_detection",
            display_name=backend_name,
            available=True,
            success=False,
            error="No result returned",
            processing_time_sec=0.0,
            segments=[],
        )


# ---------------------------------------------------------------------------
# Speech segmentation runner
# ---------------------------------------------------------------------------


def run_speech_segmentation(
    backend_name: str,
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    timeout_sec: int = 120,
    vad_config: Optional[Dict[str, Any]] = None,
) -> BackendRunResult:
    """Run a single speech segmentation backend.

    Args:
        backend_name: Segmenter name (silero-v4.0, ten, nemo-lite, etc.)
        audio_data: Audio as numpy float32 array
        sample_rate: Sample rate of audio_data
        timeout_sec: Maximum execution time
        vad_config: Optional production config dict (for Silero backends)

    Returns:
        BackendRunResult with speech segments as SegmentInfo list
    """
    from whisperjav.modules.speech_segmentation.factory import (
        SpeechSegmenterFactory,
    )

    # Check availability
    available, hint = SpeechSegmenterFactory.is_backend_available(backend_name)
    if not available:
        return BackendRunResult(
            backend_name=backend_name,
            backend_type="speech_segmentation",
            display_name=backend_name,
            available=False,
            success=False,
            error=f"Not available: {hint}",
            processing_time_sec=0.0,
            segments=[],
        )

    result_queue: queue.Queue[BackendRunResult] = queue.Queue()

    def _run() -> None:
        try:
            if vad_config:
                segmenter = SpeechSegmenterFactory.create(
                    backend_name, config=vad_config
                )
            else:
                segmenter = SpeechSegmenterFactory.create(backend_name)

            display_name = segmenter.display_name

            # Capture parameters before running
            parameters: Dict[str, Any] = {}
            if hasattr(segmenter, "_get_parameters"):
                parameters = segmenter._get_parameters()

            start_time = time.time()
            result = segmenter.segment(audio_data, sample_rate=sample_rate)
            elapsed = time.time() - start_time

            # Merge result parameters
            if hasattr(result, "parameters") and result.parameters:
                parameters.update(result.parameters)

            # Convert SpeechSegment -> SegmentInfo
            segments = [
                SegmentInfo(
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    confidence=seg.confidence,
                    metadata=seg.metadata if hasattr(seg, "metadata") else {},
                )
                for seg in result.segments
            ]

            method = result.method if hasattr(result, "method") else backend_name

            segmenter.cleanup()

            result_queue.put(
                BackendRunResult(
                    backend_name=backend_name,
                    backend_type="speech_segmentation",
                    display_name=display_name,
                    available=True,
                    success=True,
                    error=None,
                    processing_time_sec=elapsed,
                    segments=segments,
                    parameters=parameters,
                    method=method,
                )
            )
        except Exception as e:
            result_queue.put(
                BackendRunResult(
                    backend_name=backend_name,
                    backend_type="speech_segmentation",
                    display_name=backend_name,
                    available=True,
                    success=False,
                    error=str(e),
                    processing_time_sec=0.0,
                    segments=[],
                )
            )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        return BackendRunResult(
            backend_name=backend_name,
            backend_type="speech_segmentation",
            display_name=backend_name,
            available=True,
            success=False,
            error=f"Timeout after {timeout_sec}s",
            processing_time_sec=float(timeout_sec),
            segments=[],
        )

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return BackendRunResult(
            backend_name=backend_name,
            backend_type="speech_segmentation",
            display_name=backend_name,
            available=True,
            success=False,
            error="No result returned",
            processing_time_sec=0.0,
            segments=[],
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# Silero-family backend names that accept production VAD config
_SILERO_BACKENDS = {
    "silero",
    "silero-v4.0",
    "silero-v3.1",
    "silero-v6.2",
}


def run_all_backends(
    audio_info: AudioInfo,
    audio_data: np.ndarray,
    scene_backends: Optional[List[str]] = None,
    seg_backends: Optional[List[str]] = None,
    timeout_sec: int = 120,
    sensitivity: Optional[str] = None,
    user_specified_scene: bool = False,
    user_specified_seg: bool = False,
) -> Dict[str, BackendRunResult]:
    """Run all specified scene detection and speech segmentation backends.

    Args:
        audio_info: Audio metadata
        audio_data: Audio as numpy array
        scene_backends: Scene detector names (default: auditok, silero)
        seg_backends: Speech segmenter names (default: silero-v4.0, ten)
        timeout_sec: Timeout per backend
        sensitivity: Production VAD preset name (applies to Silero segmenters)
        user_specified_scene: True if user explicitly passed --scene-backends
        user_specified_seg: True if user explicitly passed --seg-backends

    Returns:
        Dict mapping "scene:<name>" or "seg:<name>" to BackendRunResult
    """
    # Default behavior: if neither type is specified, run both defaults.
    # If user specified only one type, run ONLY that type (no uninvited defaults).
    if scene_backends is None and not user_specified_scene:
        if not user_specified_seg:
            scene_backends = ["auditok", "silero"]
        else:
            scene_backends = []
    elif scene_backends is None:
        scene_backends = []

    if seg_backends is None and not user_specified_seg:
        if not user_specified_scene:
            seg_backends = ["silero-v4.0", "ten"]
        else:
            seg_backends = []
    elif seg_backends is None:
        seg_backends = []

    # Validate backend names and filter out unrecognized ones
    if scene_backends:
        scene_backends = validate_backend_names(scene_backends, "scene")
    if seg_backends:
        seg_backends = validate_backend_names(seg_backends, "seg")

    # Load production VAD config if sensitivity specified
    vad_config = None
    if sensitivity:
        print(f"  Loading production config for sensitivity: {sensitivity}")
        vad_config = load_production_vad_config(sensitivity)
        if vad_config:
            for key, value in sorted(vad_config.items()):
                print(f"    {key}: {value}")
        else:
            print("  Warning: Failed to load config, using backend defaults")
        print()

    results: Dict[str, BackendRunResult] = {}

    # --- Scene detection ---
    if scene_backends:
        print("Running scene detection backends:")
        for name in scene_backends:
            print(f"  {name}...", end=" ", flush=True)
            result = run_scene_detection(
                name, audio_info.path, timeout_sec=timeout_sec
            )
            key = f"scene:{name}"
            results[key] = result

            if result.success:
                coverage = (
                    sum(s.duration_sec for s in result.segments)
                    / audio_info.duration_sec
                    if audio_info.duration_sec > 0
                    else 0
                )
                print(
                    f"OK ({len(result.segments)} scenes, "
                    f"{coverage:.1%} coverage, "
                    f"{result.processing_time_sec:.2f}s)"
                )
            elif not result.available:
                error_msg = result.error or ""
                if "Unknown backend" in error_msg:
                    print("SKIP (unknown backend)")
                else:
                    print(f"SKIP (not installed)")
            elif result.error and "Timeout" in result.error:
                print(f"TIMEOUT ({timeout_sec}s)")
            else:
                print(f"FAIL: {result.error}")
        print()

    # --- Speech segmentation ---
    if seg_backends:
        print("Running speech segmentation backends:")
        if vad_config:
            print(
                f"  (production config applied to Silero backends, "
                f"sensitivity={sensitivity})"
            )
        for name in seg_backends:
            print(f"  {name}...", end=" ", flush=True)
            # Only apply vad_config to Silero-family backends
            backend_config = (
                vad_config if name in _SILERO_BACKENDS else None
            )
            result = run_speech_segmentation(
                name,
                audio_data,
                sample_rate=audio_info.sample_rate,
                timeout_sec=timeout_sec,
                vad_config=backend_config,
            )
            key = f"seg:{name}"
            results[key] = result

            if result.success:
                coverage = (
                    sum(s.duration_sec for s in result.segments)
                    / audio_info.duration_sec
                    if audio_info.duration_sec > 0
                    else 0
                )
                print(
                    f"OK ({len(result.segments)} segments, "
                    f"{coverage:.1%} coverage, "
                    f"{result.processing_time_sec:.2f}s)"
                )
            elif not result.available:
                error_msg = result.error or ""
                if "Unknown backend" in error_msg:
                    print("SKIP (unknown backend)")
                else:
                    print(f"SKIP (not installed)")
            elif result.error and "Timeout" in result.error:
                print(f"TIMEOUT ({timeout_sec}s)")
            else:
                print(f"FAIL: {result.error}")
        print()

    return results
