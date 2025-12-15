"""
ZipEnhancer speech enhancement backend.

ZipEnhancer is a lightweight, high-quality speech enhancement model from
Alibaba's Speech Lab. It uses only 2.04M parameters while achieving SOTA
PESQ scores (3.69 on DNS2020).

Features:
- 50x smaller than MossFormer2 (~2MB vs ~100MB)
- Native 16kHz (no resampling needed for VAD/ASR)
- Two inference modes: torch (GPU) and ONNX (CPU/GPU)
- ICASSP 2025 paper

Model Options:
- "torch" (default): Uses ModelScope pipeline with PyTorch backend
- "onnx": Uses ONNX runtime with STFT/ISTFT processing

References:
- https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base
- https://arxiv.org/abs/2501.05183

Installation: pip install modelscope>=1.20
"""

from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import time
import os
import numpy as np
import logging

from ..base import (
    EnhancementResult,
    load_audio_to_array,
    create_failed_result,
    resample_audio,
)

logger = logging.getLogger("whisperjav")

# ZipEnhancer configuration
ZIPENHANCER_SAMPLE_RATE = 16000
MODELSCOPE_MODEL_ID = "iic/speech_zipenhancer_ans_multiloss_16k_base"

# STFT parameters for ONNX mode (must match training config)
ONNX_N_FFT = 400
ONNX_HOP_SIZE = 100
ONNX_WIN_SIZE = 400
ONNX_COMPRESS_FACTOR = 0.3

# Chunking configuration for long audio
MAX_CHUNK_DURATION = 10.0  # seconds
OVERLAP_DURATION = 0.5  # 500ms overlap for crossfade


class ZipEnhancerBackend:
    """
    ZipEnhancer speech enhancement backend.

    A lightweight alternative to ClearVoice with better quality scores
    and dramatically lower VRAM usage (~50x smaller model).

    Two model options:
    - "torch": ModelScope pipeline (requires GPU, better quality)
    - "onnx": ONNX runtime (works on CPU, faster inference)

    Example:
        # Torch mode (default)
        enhancer = ZipEnhancerBackend(model="torch")
        result = enhancer.enhance(audio_array, sample_rate=16000)

        # ONNX mode
        enhancer = ZipEnhancerBackend(model="onnx")
        result = enhancer.enhance(audio_array, sample_rate=16000)

        # Always cleanup when done
        enhancer.cleanup()
    """

    def __init__(
        self,
        model: str = "torch",
        device: Optional[str] = None,
        chunk_duration: float = MAX_CHUNK_DURATION,
        **kwargs
    ):
        """
        Initialize ZipEnhancer backend.

        Args:
            model: Model variant - "torch" (default) or "onnx"
            device: Device to use ("cuda", "cpu", "gpu", or None for auto)
            chunk_duration: Max seconds per chunk for long audio (default: 10.0)
            **kwargs: Additional parameters (ignored)
        """
        # Normalize model name
        self.model_variant = model.lower() if model else "torch"
        if self.model_variant not in ("torch", "onnx"):
            logger.warning(f"Unknown model variant '{model}', using 'torch'")
            self.model_variant = "torch"

        self.device = device
        self.chunk_duration = chunk_duration

        # Runtime state
        self._pipeline = None  # For torch mode
        self._onnx_session = None  # For ONNX mode
        self._onnx_model_path = None
        self._initialized = False

        logger.debug(
            f"ZipEnhancerBackend configured: model={self.model_variant}, "
            f"device={device}, chunk_duration={chunk_duration}"
        )

    @property
    def name(self) -> str:
        return "zipenhancer"

    @property
    def display_name(self) -> str:
        suffix = " (ONNX)" if self.model_variant == "onnx" else ""
        return f"ZipEnhancer 16kHz{suffix}"

    def _ensure_initialized(self) -> bool:
        """
        Lazy initialization of model.

        Returns:
            True if initialized successfully, False otherwise
        """
        if self._initialized:
            return True

        try:
            if self.model_variant == "onnx":
                return self._init_onnx()
            else:
                return self._init_torch()
        except Exception as e:
            logger.error(f"Failed to initialize ZipEnhancer: {e}")
            return False

    def _init_torch(self) -> bool:
        """Initialize using ModelScope pipeline (torch backend)."""
        try:
            import torch
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            # Set torch threads for better CPU performance
            torch.set_num_threads(8)
            torch.set_num_interop_threads(8)

            logger.info("Loading ZipEnhancer via ModelScope (torch)...")

            # Determine device
            device = self.device
            if device is None:
                device = "gpu" if torch.cuda.is_available() else "cpu"
            elif device == "cuda":
                device = "gpu"  # ModelScope uses "gpu" not "cuda"

            self._pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model=MODELSCOPE_MODEL_ID,
                device=device
            )

            self._initialized = True
            logger.info(f"ZipEnhancer (torch) loaded successfully on {device}")
            return True

        except ImportError as e:
            logger.error(
                f"ModelScope not installed: {e}. "
                "Install with: pip install modelscope>=1.20"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ModelScope pipeline: {e}")
            return False

    def _init_onnx(self) -> bool:
        """Initialize using ONNX runtime."""
        try:
            import onnxruntime as ort
            from modelscope import snapshot_download

            logger.info("Loading ZipEnhancer ONNX model...")

            # Download ONNX model from the dedicated ONNX repository
            # The main model (iic/...) doesn't include ONNX, so we use the ONNX-specific model
            ONNX_MODEL_ID = "manyeyes/ZipEnhancer-se-16k-base-onnx"

            try:
                model_dir = snapshot_download(ONNX_MODEL_ID)
            except Exception as e:
                logger.error(f"Failed to download ONNX model: {e}")
                raise

            # Find ONNX model file in the directory
            onnx_model_path = os.path.join(model_dir, "model.onnx")
            if not os.path.exists(onnx_model_path):
                # Try alternative name
                onnx_model_path = os.path.join(model_dir, "onnx_model.onnx")

            if not os.path.exists(onnx_model_path):
                # List directory contents for debugging
                contents = os.listdir(model_dir) if os.path.exists(model_dir) else []
                raise FileNotFoundError(
                    f"ONNX model not found in {model_dir}. "
                    f"Directory contents: {contents}"
                )

            # Configure ONNX providers
            providers = ['CPUExecutionProvider']
            if self.device != 'cpu':
                # Try CUDA first if available
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

            self._onnx_session = ort.InferenceSession(
                onnx_model_path,
                providers=providers
            )
            self._onnx_model_path = onnx_model_path

            self._initialized = True
            logger.info(f"ZipEnhancer ONNX loaded successfully (providers: {providers})")
            return True

        except ImportError as e:
            logger.error(
                f"Required package not installed: {e}. "
                "Install with: pip install onnxruntime modelscope>=1.20"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")
            return False

    def enhance(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int,
        **kwargs
    ) -> EnhancementResult:
        """
        Enhance audio using ZipEnhancer.

        Args:
            audio: Audio data as numpy array (float32, mono), or path to file
            sample_rate: Sample rate of input audio
            **kwargs: Additional parameters (ignored)

        Returns:
            EnhancementResult with enhanced audio at 16kHz

        Note:
            On failure, returns original audio with success=False
            for graceful degradation.
        """
        start_time = time.time()

        # Load audio if path provided
        try:
            audio_data, actual_sr = load_audio_to_array(audio, sample_rate)
        except Exception as e:
            return create_failed_result(
                audio=np.zeros(1, dtype=np.float32),
                sample_rate=sample_rate,
                method="zipenhancer",
                error_message=f"Failed to load audio: {e}",
                processing_time_sec=time.time() - start_time,
            )

        # Initialize model if needed
        if not self._ensure_initialized():
            return create_failed_result(
                audio=audio_data,
                sample_rate=actual_sr,
                method="zipenhancer",
                error_message="Failed to initialize ZipEnhancer model",
                processing_time_sec=time.time() - start_time,
            )

        try:
            # Resample to 16kHz if needed
            if actual_sr != ZIPENHANCER_SAMPLE_RATE:
                audio_for_model = resample_audio(
                    audio_data, actual_sr, ZIPENHANCER_SAMPLE_RATE
                )
                logger.debug(
                    f"Resampled {actual_sr}Hz -> {ZIPENHANCER_SAMPLE_RATE}Hz for ZipEnhancer"
                )
            else:
                audio_for_model = audio_data

            # Process audio (with chunking for long audio)
            duration = len(audio_for_model) / ZIPENHANCER_SAMPLE_RATE

            if duration > self.chunk_duration:
                logger.debug(
                    f"Audio duration {duration:.1f}s exceeds {self.chunk_duration}s, "
                    "using chunked processing"
                )
                enhanced = self._process_chunked(audio_for_model)
            else:
                enhanced = self._process_single(audio_for_model)

            processing_time = time.time() - start_time

            return EnhancementResult(
                audio=enhanced,
                sample_rate=ZIPENHANCER_SAMPLE_RATE,
                method="zipenhancer",
                parameters={
                    "model": self.model_variant,
                    "chunk_duration": self.chunk_duration,
                    "input_sr": actual_sr,
                    "output_sr": ZIPENHANCER_SAMPLE_RATE,
                },
                processing_time_sec=processing_time,
                metadata={
                    "input_samples": len(audio_data),
                    "output_samples": len(enhanced),
                    "duration_sec": duration,
                    "chunked": duration > self.chunk_duration,
                },
                success=True,
                error_message=None,
            )

        except Exception as e:
            logger.warning(f"ZipEnhancer enhancement failed: {e}")
            return create_failed_result(
                audio=audio_data,
                sample_rate=actual_sr,
                method="zipenhancer",
                error_message=str(e),
                processing_time_sec=time.time() - start_time,
            )

    def _process_single(self, audio: np.ndarray) -> np.ndarray:
        """Process a single audio chunk."""
        if self.model_variant == "onnx":
            return self._process_onnx(audio)
        else:
            return self._process_torch(audio)

    def _process_torch(self, audio: np.ndarray) -> np.ndarray:
        """
        Process using ModelScope pipeline (torch backend).

        ModelScope pipeline expects file path, so we write to temp file.
        """
        import tempfile
        import soundfile as sf

        # Pipeline expects file path, so write temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)

        try:
            sf.write(str(temp_path), audio, ZIPENHANCER_SAMPLE_RATE)
            result = self._pipeline(str(temp_path))

            # Extract enhanced audio from result
            if isinstance(result, dict):
                if 'output_pcm' in result:
                    # PCM output - could be bytes (int16) or numpy array
                    pcm_data = result['output_pcm']
                    if isinstance(pcm_data, bytes):
                        # Raw PCM bytes - decode as int16 and normalize to float32
                        enhanced = np.frombuffer(pcm_data, dtype=np.int16)
                        enhanced = enhanced.astype(np.float32) / 32768.0
                    elif isinstance(pcm_data, np.ndarray):
                        if pcm_data.dtype == np.int16:
                            enhanced = pcm_data.astype(np.float32) / 32768.0
                        else:
                            enhanced = pcm_data.astype(np.float32)
                    else:
                        enhanced = np.array(pcm_data, dtype=np.float32)
                elif 'output_wav' in result:
                    # Path to output file
                    enhanced, _ = sf.read(result['output_wav'], dtype='float32')
                else:
                    # Try first array-like value
                    for key, value in result.items():
                        if isinstance(value, (np.ndarray, list)):
                            enhanced = np.array(value, dtype=np.float32)
                            break
                    else:
                        logger.warning(f"Unexpected result format: {result.keys()}")
                        return audio
            elif hasattr(result, 'output_wav'):
                enhanced, _ = sf.read(result.output_wav, dtype='float32')
            else:
                logger.warning(f"Unexpected result type: {type(result)}")
                return audio

            # Ensure 1D
            if enhanced.ndim > 1:
                enhanced = enhanced.flatten()

            return enhanced.astype(np.float32)

        finally:
            temp_path.unlink(missing_ok=True)

    def _process_onnx(self, audio: np.ndarray) -> np.ndarray:
        """
        Process using ONNX runtime with STFT/ISTFT.

        The ONNX model expects STFT magnitude and phase as inputs,
        and outputs enhanced magnitude and phase.
        """
        import torch
        from modelscope.models.audio.ans.zipenhancer import mag_pha_stft, mag_pha_istft
        from modelscope.utils.audio.audio_utils import audio_norm

        # Normalize audio
        audio = audio_norm(audio).astype(np.float32)

        # Convert to torch tensor with batch dimension [1, samples]
        noisy_wav = torch.from_numpy(np.reshape(audio, [1, audio.shape[0]]))

        # Compute normalization factor
        norm_factor = torch.sqrt(noisy_wav.shape[1] / torch.sum(noisy_wav ** 2.0))
        noisy_audio = noisy_wav * norm_factor

        # Compute STFT
        noisy_amp, noisy_pha, _ = mag_pha_stft(
            noisy_audio,
            ONNX_N_FFT,
            ONNX_HOP_SIZE,
            ONNX_WIN_SIZE,
            compress_factor=ONNX_COMPRESS_FACTOR,
            center=True
        )

        # Run ONNX inference
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        input_names = [inp.name for inp in self._onnx_session.get_inputs()]
        ort_inputs = {
            input_names[0]: to_numpy(noisy_amp),
            input_names[1]: to_numpy(noisy_pha),
        }

        ort_outs = self._onnx_session.run(None, ort_inputs)

        # Convert outputs back to torch tensors
        amp_g = torch.from_numpy(ort_outs[0])
        pha_g = torch.from_numpy(ort_outs[1])

        # Inverse STFT
        wav = mag_pha_istft(
            amp_g,
            pha_g,
            ONNX_N_FFT,
            ONNX_HOP_SIZE,
            ONNX_WIN_SIZE,
            compress_factor=ONNX_COMPRESS_FACTOR,
            center=True
        )

        # Denormalize
        wav = wav / norm_factor

        # Convert to numpy
        enhanced = to_numpy(wav).squeeze()

        return enhanced.astype(np.float32)

    def _process_chunked(self, audio: np.ndarray) -> np.ndarray:
        """Process long audio in overlapping chunks with crossfade."""
        chunk_samples = int(self.chunk_duration * ZIPENHANCER_SAMPLE_RATE)
        overlap_samples = int(OVERLAP_DURATION * ZIPENHANCER_SAMPLE_RATE)
        hop_samples = chunk_samples - overlap_samples

        # Process chunks
        enhanced_chunks = []
        positions = []

        for start in range(0, len(audio), hop_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            # Skip very short chunks (less than 0.5 sec)
            if len(chunk) < ZIPENHANCER_SAMPLE_RATE // 2:
                break

            enhanced_chunk = self._process_single(chunk)
            enhanced_chunks.append(enhanced_chunk)
            positions.append(start)

        if not enhanced_chunks:
            return audio

        # Crossfade stitch
        return self._crossfade_stitch(
            enhanced_chunks, positions, overlap_samples, len(audio)
        )

    def _crossfade_stitch(
        self,
        chunks: List[np.ndarray],
        positions: List[int],
        overlap_samples: int,
        total_length: int
    ) -> np.ndarray:
        """Stitch chunks together with crossfade overlap."""
        if len(chunks) == 1:
            return chunks[0]

        # Create output buffer
        output = np.zeros(total_length, dtype=np.float32)
        weight = np.zeros(total_length, dtype=np.float32)

        for chunk, pos in zip(chunks, positions):
            chunk_len = len(chunk)

            # Create crossfade weights
            fade = np.ones(chunk_len, dtype=np.float32)

            # Fade in at start (except first chunk)
            if pos > 0:
                fade_len = min(overlap_samples, chunk_len // 2)
                fade[:fade_len] = np.linspace(0, 1, fade_len)

            # Fade out at end (except last chunk)
            if pos + chunk_len < total_length:
                fade_len = min(overlap_samples, chunk_len // 2)
                fade[-fade_len:] = np.linspace(1, 0, fade_len)

            # Apply weighted overlap-add
            end_pos = min(pos + chunk_len, total_length)
            actual_len = end_pos - pos

            output[pos:end_pos] += chunk[:actual_len] * fade[:actual_len]
            weight[pos:end_pos] += fade[:actual_len]

        # Normalize by weight
        weight = np.maximum(weight, 1e-8)  # Avoid division by zero
        output /= weight

        return output

    def get_preferred_sample_rate(self) -> int:
        """ZipEnhancer requires 16kHz input."""
        return ZIPENHANCER_SAMPLE_RATE

    def get_output_sample_rate(self) -> int:
        """ZipEnhancer outputs 16kHz."""
        return ZIPENHANCER_SAMPLE_RATE

    def cleanup(self) -> None:
        """Release resources."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        if self._onnx_session is not None:
            del self._onnx_session
            self._onnx_session = None

        self._initialized = False

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        import gc
        gc.collect()

        logger.debug("ZipEnhancer resources released")

    def get_supported_models(self) -> List[str]:
        """Return list of supported model variants."""
        return ["torch", "onnx"]

    @property
    def is_lightweight(self) -> bool:
        """
        ZipEnhancer is lightweight (~500MB VRAM with 2.04M params).

        Can co-exist with VAD/ASR models and optionally skip disk
        intermediates for faster streaming processing.
        """
        return True

    def __repr__(self) -> str:
        return f"ZipEnhancerBackend(model={self.model_variant})"

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
