#!/usr/bin/env python3
"""
QualityAware ASR engine leveraging brouhaha VAD/SNR/C50 for quality-aware routing
and optional enhancement before Whisper transcription. Designed to mirror
WhisperProASR's high-level contract (transcribe, transcribe_to_srt).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import logging
import json

import numpy as np
import soundfile as sf
import torch
import whisper
import srt
import datetime

from pyannote.audio import Model, Inference

from whisperjav.utils.logger import logger
from whisperjav.modules.audio_quality_router import select_route_for_segment
from whisperjav.modules.audio_enhancement import run_chain


class QualityAwareASR:
    """Whisper ASR with brouhaha-based quality routing and optional enhancement."""

    def __init__(self, model_config: Dict, params: Dict, task: str):
        # Whisper model/device
        self.model_name = model_config.get("model_name", "large-v2")
        self.device = model_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Params
        self._decoder_params = params.get("decoder", {})
        self._provider_params = params.get("provider", {})
        self._qa_params = params.get("quality_aware", {})  # optional section

        # Whisper parameters consolidated
        self.whisper_params: Dict[str, Any] = {}
        self.whisper_params.update(self._decoder_params)
        self.whisper_params.update(self._provider_params)
        self.whisper_params["task"] = task

        # Optional language override for consistency with existing flows
        if "language" not in self.whisper_params:
            self.whisper_params["language"] = "ja"

        # Load quality routing config (JSON) if provided
        self.routing_cfg = self._load_routing_cfg(self._qa_params.get("routing_config_path"))

        # HF token optional – if missing, we degrade gracefully
        self.hf_token = self._qa_params.get("hf_token")

        # Initialize models
        self._init_models()

    def _init_models(self) -> None:
        logger.debug(f"Loading Whisper model: {self.model_name} on {self.device}")
        self.whisper_model = whisper.load_model(self.model_name, device=self.device)

        # Load brouhaha model if possible
        self.brouhaha_inference: Inference | None = None
        try:
            snr_model = Model.from_pretrained("pyannote/brouhaha", use_auth_token=self.hf_token)
            self.brouhaha_inference = Inference(snr_model)
            logger.debug("Loaded brouhaha model for VAD/SNR/C50")
        except Exception as e:
            logger.warning(f"brouhaha model not available; proceeding without quality routing: {e}")

    def _load_routing_cfg(self, path: Union[str, Path, None]) -> Dict[str, Any]:
        if not path:
            # Reasonable defaults aligned with template
            return {
                "VAD_MIN": 0.5,
                "SNR_CLEAN": 7.5,
                "C50_CLEAN": 0.5,
                "MIN_COVERAGE": 0.4,
                "SCORE": {"w_snr": 0.7, "w_c50": 0.3, "thresh": 6.0},
                "CHAINS": {
                    "noisy": ["denoise", "norm"],
                    "reverberant": ["dereverb", "denoise", "norm"],
                    "music": ["separate", "denoise", "norm"],
                    "low_energy": ["agc", "norm"],
                    "unknown": ["denoise", "norm"],
                },
            }
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load routing config {path}: {e}; using defaults")
            return {}

    def _prepare_whisper_params(self) -> Dict[str, Any]:
        params = self.whisper_params.copy()
        # Normalize temperature list to tuple for whisper
        if isinstance(params.get("temperature"), list):
            params["temperature"] = tuple(params["temperature"])  # type: ignore
        params.setdefault("verbose", None)
        return params

    def _brouhaha_analyze(self, waveform: np.ndarray, sample_rate: int) -> List[Tuple[float, Tuple[float, float, float]]]:
        """Run brouhaha inference over the whole segment; return frames as (time, (vad, snr, c50))."""
        if self.brouhaha_inference is None:
            return []
        # pyannote Inference expects dict with waveform, sample_rate
        result = self.brouhaha_inference({"waveform": waveform[np.newaxis, :], "sample_rate": sample_rate})
        # Normalize to list of (time, (vad, snr, c50))
        frames: List[Tuple[float, Tuple[float, float, float]]] = []
        for t, triple in result:
            vad, snr, c50 = triple
            frames.append((float(t), (float(vad), float(snr), float(c50))))
        return frames

    def _route_and_enhance(self, waveform: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Decide direct vs enhance and apply chain if needed. Returns possibly modified audio and a metadata dict."""
        meta: Dict[str, Any] = {"routed": False, "decision": None}
        frames = self._brouhaha_analyze(waveform, sample_rate)
        if not frames:
            # No brouhaha available; pass-through
            return waveform, sample_rate, meta

        metrics, decision = select_route_for_segment(frames, 0.0, float(len(waveform) / sample_rate), self.routing_cfg)
        meta["routed"] = True
        meta["metrics"] = metrics.__dict__
        meta["decision"] = {"route": decision.route, "label": decision.label, "reasons": decision.reasons}

        if decision.route == "enhance":
            steps = (decision.params or {}).get("chain", [])
            try:
                enhanced_wav, enhanced_sr = run_chain(waveform, sample_rate, steps)
                meta["enhancement_chain"] = steps
                return enhanced_wav, enhanced_sr, meta
            except Exception as e:
                logger.warning(f"Enhancement failed ({steps}); falling back to direct: {e}")
                return waveform, sample_rate, meta
        return waveform, sample_rate, meta

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Transcribe a file with quality-aware routing and optional enhancement."""
        audio_path = Path(audio_path)
        # Load audio to mono float32
        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        routed_wav, routed_sr, meta = self._route_and_enhance(data, sr)

        params = self._prepare_whisper_params()
        try:
            result = self.whisper_model.transcribe(routed_wav, **params)
        except Exception as e:
            logger.error(f"QualityAwareASR transcribe failed: {e}")
            # Minimal fallback
            result = self.whisper_model.transcribe(routed_wav, task=params.get("task", "transcribe"), language=params.get("language", "ja"), verbose=None)

        # Pack minimal contract: segments, text, language, plus optional meta
        out = {
            "segments": result.get("segments", []),
            "text": result.get("text", ""),
            "language": params.get("language", "ja"),
        }
        if meta.get("routed"):
            out["quality_meta"] = meta
        return out

    def transcribe_to_srt(self, audio_path: Union[str, Path], output_srt_path: Union[str, Path], **kwargs) -> Path:
        audio_path = Path(audio_path)
        output_srt_path = Path(output_srt_path)

        result = self.transcribe(audio_path, **kwargs)
        segments = result.get("segments", []) or []

        srt_subs: List[srt.Subtitle] = []
        for idx, seg in enumerate(segments, 1):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            srt_subs.append(
                srt.Subtitle(
                    index=idx,
                    start=datetime.timedelta(seconds=start),
                    end=datetime.timedelta(seconds=end),
                    content=text,
                )
            )

        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(srt_subs))

        logger.debug(f"Saved SRT to: {output_srt_path}")
        return output_srt_path
