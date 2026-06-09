#!/usr/bin/env python3
"""
SenseVoice ASR Module for WhisperJAV (issue #350).

Wraps the FunASR ``AutoModel`` running iic/SenseVoiceSmall — a fast,
non-autoregressive multilingual ASR model (~234M params, ~1GB VRAM) with
strong Japanese support and built-in audio-event/emotion tagging.

SenseVoice is NOT a Whisper-style model: a single ``generate()`` call returns
one (or a few VAD-merged) utterance(s) of text with rich special tokens
(``<|ja|><|HAPPY|><|Speech|>...``) rather than per-segment timestamps. We strip
those tokens via FunASR's ``rich_transcription_postprocess`` and return a flat
``List[{text, start, end}]`` — the same contract as TransformersASR — so the
pipeline's scene-detection + SRT-stitching machinery provides timing.

funasr is an OPTIONAL heavy dependency. It is imported lazily inside
``load_model`` so importing this module (and running other WhisperJAV modes)
never requires it. If it is missing, a clear ``pip install funasr`` message is
raised only when SenseVoice is actually used.

Reference: https://github.com/FunAudioLLM/SenseVoice
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import contextlib
import logging
import time

from whisperjav.utils.logger import logger


# Benign funasr root-logger lines emitted during vad_segment-mode diarization
# (no punc_model) even when the run succeeds. Matched as substrings.
_BENIGN_FUNASR_DIARIZE_MSGS = (
    "Missing punc_model, which is required by spk_model",
    "No timestamp found in ASR result. Speaker diarization relies on timestamps",
)


class _DropMessagesFilter(logging.Filter):
    """logging.Filter that drops records whose message contains any needle."""

    def __init__(self, needles):
        super().__init__()
        self._needles = tuple(needles)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(n in msg for n in self._needles)


@contextlib.contextmanager
def _suppress_root_log_messages(needles):
    """Temporarily filter specific benign messages off the ROOT logger.

    funasr logs some diarization lines via bare ``logging.error(...)`` (root
    logger), so per-logger level tweaks can't silence them. We attach a filter
    to the root logger AND its handlers for the duration of the block, then
    remove it — without changing levels (real errors still propagate).
    """
    root = logging.getLogger()
    flt = _DropMessagesFilter(needles)
    root.addFilter(flt)
    handlers = list(root.handlers)
    for h in handlers:
        h.addFilter(flt)
    try:
        yield
    finally:
        root.removeFilter(flt)
        for h in handlers:
            try:
                h.removeFilter(flt)
            except Exception:
                pass


# Map WhisperJAV language names/codes to SenseVoice language codes.
# SenseVoice supports: zh (Chinese), en (English), yue (Cantonese),
# ja (Japanese), ko (Korean), and "auto" / "nospeech".
_SENSEVOICE_LANG_MAP = {
    "japanese": "ja", "ja": "ja", "jp": "ja",
    "english": "en", "en": "en",
    "chinese": "zh", "zh": "zh", "mandarin": "zh",
    "cantonese": "yue", "yue": "yue",
    "korean": "ko", "ko": "ko",
    "auto": "auto", "": "auto", None: "auto",
}


class SenseVoiceASR:
    """SenseVoice (FunASR) ASR backend.

    Lazily loads the FunASR AutoModel. ``transcribe(audio_path)`` returns a list
    of ``{text, start, end}`` segments whose timestamps span the supplied audio
    clip (SenseVoice does not emit reliable intra-clip timestamps; per-scene
    granularity comes from the pipeline's scene detection).
    """

    DEFAULT_MODEL_ID = "iic/SenseVoiceSmall"
    DEFAULT_VAD_MODEL = "fsmn-vad"
    DEFAULT_LANGUAGE = "ja"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        language: str = DEFAULT_LANGUAGE,
        use_itn: bool = True,
        vad_model: Optional[str] = DEFAULT_VAD_MODEL,
        vad_max_segment_ms: int = 60000,
        merge_vad: bool = True,
        merge_length_s: int = 15,
        batch_size_s: int = 60,
        ban_emo_unk: bool = False,
        task: str = "transcribe",
        # Speaker diarization (funasr 1.3.9+): spk_model (cam++) clusters speaker
        # embeddings per VAD segment so generate() returns sentence_info with
        # per-sentence {spk, start, end}. Requires vad_model on.
        #
        # spk_mode="vad_segment" (NOT punc_segment): SenseVoice is non-
        # autoregressive and emits NO word timestamps, so funasr's punc_segment
        # mode can't run — it logs "No timestamps in ASR result (e.g.
        # SenseVoice), falling back to vad_segment" and downgrades anyway. We
        # request vad_segment directly so the punc_model is never needed (its
        # text isn't even used in vad_segment mode), which removes the
        # "length mismatch between punc and timestamp" warning spam and avoids
        # loading/downloading ct-punc for nothing. ITN (use_itn) is a separate
        # SenseVoice decode flag and still formats the per-speaker text.
        diarize: bool = False,
        spk_model: str = "cam++",
        punc_model: Optional[str] = None,
        spk_mode: str = "vad_segment",
        # Force a fixed speaker count. 0/None = cam++ auto-estimates via spectral
        # eigengap (often collapses to 1 speaker on monologue-ish/breathy audio).
        # Set 2+ when you KNOW how many speakers a scene has and auto-detect
        # under-segments (passed to generate() as preset_spk_num → oracle_num).
        spk_num: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            model_id: FunASR/ModelScope model id (default iic/SenseVoiceSmall).
            device: 'auto', 'cuda', 'cuda:N', or 'cpu'.
            language: source language name or code; mapped to SenseVoice codes.
            use_itn: inverse text normalization (punctuation/numbers).
            vad_model: FunASR VAD model for long-audio chunking ('fsmn-vad'),
                       or None to disable internal VAD.
            vad_max_segment_ms: max single VAD segment length (ms). FunASR's
                vad_kwargs.max_single_segment_time; the tutorial recommends
                60000 (60s) for long audio. SenseVoice truncates segments longer
                than this, so don't set it below your longest expected utterance.
            merge_vad: merge short VAD segments before decoding.
            merge_length_s: target merged-segment length (s) when merge_vad.
            batch_size_s: dynamic batch size in audio-seconds.
            task: 'transcribe' or 'translate'. SenseVoice has no native
                  translation; 'translate' is accepted but only transcribes
                  (a warning is logged). Direct-to-English should use another
                  backend.
        """
        self.model_id = model_id
        self.device_request = device
        self.language = self._normalize_language(language)
        self.use_itn = bool(use_itn)
        self.vad_model = vad_model
        self.vad_max_segment_ms = int(vad_max_segment_ms)
        self.merge_vad = bool(merge_vad)
        self.merge_length_s = int(merge_length_s)
        self.batch_size_s = int(batch_size_s)
        self.ban_emo_unk = bool(ban_emo_unk)
        self.task = task
        self.diarize = bool(diarize)
        self.spk_model = spk_model
        self.punc_model = punc_model
        self.spk_mode = spk_mode
        # 0 → None (auto). Only forwarded when diarizing.
        self.spk_num = int(spk_num) if spk_num else None

        if self.diarize and not self.vad_model:
            # The spk path runs through inference_with_vadres — VAD must be on.
            logger.warning(
                "Diarization requires VAD; re-enabling fsmn-vad (was disabled)."
            )
            self.vad_model = self.DEFAULT_VAD_MODEL

        if task == "translate":
            logger.warning(
                "SenseVoice does not support translation natively; it will "
                "transcribe in the source language. For direct-to-English use "
                "--mode qwen or --mode transformers instead."
            )

        # Lazily-loaded FunASR model + postprocess fn
        self.model = None
        self._rich_postprocess = None
        self._device = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_language(language: Optional[str]) -> str:
        key = language.lower() if isinstance(language, str) else language
        return _SENSEVOICE_LANG_MAP.get(key, "auto")

    def _detect_device(self) -> str:
        if self.device_request and self.device_request != "auto":
            # 'cuda' -> 'cuda:0' to match FunASR's expected format
            if self.device_request == "cuda":
                return "cuda:0"
            return self.device_request
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _quiet_funasr_logging() -> None:
        """Raise FunASR/modelscope loggers to WARNING to de-clutter the console.

        FunASR emits per-call INFO lines (rtf_avg, time_speech, load_data,
        extract_feat) on every generate(); with one call per scene this floods
        the GUI console. We keep WARNING/ERROR so real problems still surface.
        No-op when WhisperJAV is running at DEBUG (power users want the firehose).
        """
        import logging
        if logging.getLogger("whisperjav").isEnabledFor(logging.DEBUG):
            return
        for name in ("funasr", "modelscope"):
            try:
                logging.getLogger(name).setLevel(logging.WARNING)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    def load_model(self) -> None:
        """Lazily construct the FunASR AutoModel. Idempotent."""
        if self.model is not None:
            logger.debug("SenseVoice model already loaded")
            return

        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError as e:
            raise ImportError(
                "SenseVoice requires the 'funasr' package, which is not installed.\n"
                "Install it with:\n\n"
                "    pip install funasr\n\n"
                "(funasr also pulls in modelscope/onnxruntime; first run will "
                "download the SenseVoiceSmall model ~1GB.)\n"
                f"Original import error: {e}"
            ) from e

        self._rich_postprocess = rich_transcription_postprocess
        self._device = self._detect_device()

        # Quiet FunASR's chatty per-call INFO logging (rtf_avg / time_speech /
        # load_data / extract_feat lines) so the GUI console stays readable.
        # Skipped when WhisperJAV is in DEBUG so power users can still see it.
        self._quiet_funasr_logging()

        logger.info("Loading SenseVoice ASR model (FunASR)...")
        logger.info(f"  Model:    {self.model_id}")
        logger.info(f"  Device:   {self._device}")
        logger.info(f"  Language: {self.language}")
        logger.info(f"  VAD:      {self.vad_model or 'disabled'}")

        model_kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "trust_remote_code": False,  # use the funasr-packaged model.py
            "device": self._device,
            "disable_update": True,      # don't phone home for version checks
            # Suppress FunASR's per-call tqdm progress bars (rtf_avg/load_data/
            # extract_feat/[====] bars). These redraw once per scene and clutter
            # the GUI console without adding info. Normal log lines are kept.
            #
            # EXCEPTION: in diarization mode we transcribe the WHOLE file in one
            # generate() call (no per-scene loop to report progress), so funasr's
            # internal per-VAD-chunk bar is the only sign of life — keep it on.
            "disable_pbar": not self.diarize,
        }
        if self.vad_model:
            model_kwargs["vad_model"] = self.vad_model
            model_kwargs["vad_kwargs"] = {"max_single_segment_time": self.vad_max_segment_ms}

        if self.diarize:
            # Speaker diarization: spk_model (cam++) clusters speaker embeddings
            # per VAD segment. punc_model is left off for SenseVoice — vad_segment
            # mode doesn't use it (see __init__ note). ITN still formats the text.
            model_kwargs["spk_model"] = self.spk_model
            model_kwargs["spk_mode"] = self.spk_mode
            if self.punc_model:
                model_kwargs["punc_model"] = self.punc_model
            logger.info(
                f"  Diarization: ON (spk={self.spk_model}, punc={self.punc_model}, "
                f"mode={self.spk_mode})"
            )

        start = time.time()
        try:
            self.model = AutoModel(**model_kwargs)
        except Exception as e:
            logger.error(f"Failed to load SenseVoice model '{self.model_id}': {e}")
            raise
        logger.info(f"  Loaded in {time.time() - start:.1f}s")

    # ------------------------------------------------------------------ #
    # Transcription
    # ------------------------------------------------------------------ #
    def transcribe(
        self,
        audio_path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Transcribe one audio file/clip.

        Returns a list of ``{text, start, end}`` segments. Because SenseVoice
        returns merged-utterance text without reliable intra-clip timestamps,
        the segment(s) span the full clip duration; finer timing comes from the
        pipeline's scene splitting.
        """
        audio_path = Path(audio_path)
        if self.model is None:
            self.load_model()

        if progress_callback:
            progress_callback(0.0, f"Transcribing: {audio_path.name}")

        duration = self._probe_duration(audio_path)

        logger.debug(f"SenseVoice transcribing: {audio_path.name} "
                     f"(lang={self.language}, dur={duration:.2f}s)")

        def _generate():
            gen_kwargs = dict(
                input=str(audio_path),
                cache={},
                language=self.language,
                use_itn=self.use_itn,
                batch_size_s=self.batch_size_s,
                merge_vad=self.merge_vad,
                merge_length_s=self.merge_length_s,
                ban_emo_unk=self.ban_emo_unk,
            )
            # Force a fixed speaker count (cam++ auto-estimate often collapses to
            # 1 on monologue-ish/breathy audio). funasr maps preset_spk_num →
            # cluster oracle_num. Only meaningful while diarizing.
            if self.diarize and self.spk_num:
                gen_kwargs["preset_spk_num"] = self.spk_num
            if self.diarize:
                # In vad_segment mode (our SenseVoice setup, no punc_model) funasr
                # unconditionally logs `ERROR:root:Missing punc_model, which is
                # required by spk_model.` even though diarization then completes
                # fine via the vad_segment path. It's a bare logging.error on the
                # ROOT logger, so our funasr/modelscope quieting doesn't catch it.
                # Filter only that exact benign line for the duration of the call
                # so users don't see a scary ERROR for a working run.
                with _suppress_root_log_messages(_BENIGN_FUNASR_DIARIZE_MSGS):
                    return self.model.generate(**gen_kwargs)
            return self.model.generate(**gen_kwargs)

        try:
            res = _generate()
        except Exception as e:
            # funasr's diarization/punc alignment can throw on some audio
            # (e.g. "'>' not supported between float and NoneType" when the
            # punc/timestamp lengths mismatch). Rather than fail the whole file,
            # fall back to a plain (non-diarized) decode for this call.
            if self.diarize:
                logger.warning(
                    f"Diarized generate() failed for {audio_path.name} ({e}); "
                    f"retrying without diarization."
                )
                self._disable_diarization_runtime()
                try:
                    res = _generate()
                except Exception as e2:
                    logger.error(f"SenseVoice generate() failed for {audio_path.name}: {e2}")
                    raise
            else:
                logger.error(f"SenseVoice generate() failed for {audio_path.name}: {e}")
                raise

        segments = self._result_to_segments(res, duration)

        if progress_callback:
            progress_callback(1.0, "Transcription complete")

        logger.debug(f"SenseVoice produced {len(segments)} segment(s) for {audio_path.name}")
        return segments

    def _result_to_segments(self, res: Any, duration: float) -> List[Dict[str, Any]]:
        """Convert FunASR generate() output into {text, start, end} segments.

        FunASR returns a list of dicts (one per input). Each dict's 'text'
        carries SenseVoice rich tokens which we strip via
        rich_transcription_postprocess. If a per-utterance 'timestamp' field is
        present (some FunASR builds expose it), we use it; otherwise we span the
        clip.
        """
        if not res:
            return []

        # Diarization path: each result carries 'sentence_info' — a list of
        # {text, start, end, spk, timestamp} with REAL per-sentence timestamps
        # and speaker ids. Prefer it when present; one segment per sentence.
        if self.diarize:
            diar_segments = self._sentence_info_to_segments(res)
            if diar_segments:
                return diar_segments
            logger.warning(
                "Diarization on but no sentence_info returned; "
                "falling back to clip-span segmentation."
            )

        segments: List[Dict[str, Any]] = []
        n = len(res)
        for i, item in enumerate(res):
            raw_text = item.get("text", "") if isinstance(item, dict) else str(item)
            text = self._rich_postprocess(raw_text).strip() if raw_text else ""
            if not text:
                continue

            # Prefer explicit per-utterance timestamps when available. Accept
            # only a flat 2-number [start, end] pair — NOT word-level timestamp
            # lists (which output_timestamp=True produces: [[s,e],[s,e],...],
            # and which can coincidentally have len==2). Anything else falls
            # through to even clip-spanning.
            ts = item.get("timestamp") if isinstance(item, dict) else None
            start = end = None
            if (ts and isinstance(ts, (list, tuple)) and len(ts) == 2
                    and all(isinstance(x, (int, float)) for x in ts)):
                start, end = float(ts[0]) / 1000.0, float(ts[1]) / 1000.0
            if start is None or end is None:
                # Spread evenly across the clip when multiple items, else span all.
                start = (duration * i / n) if n > 1 else 0.0
                end = (duration * (i + 1) / n) if n > 1 else (duration or start + 2.0)

            segments.append({"text": text, "start": float(start), "end": float(end)})

        return segments

    def _disable_diarization_runtime(self) -> None:
        """Turn diarization off on the live model after a funasr diarize failure.

        Detaches spk_model/punc_model from the loaded AutoModel so a retry
        decodes plainly, and clears self.diarize so _result_to_segments uses the
        clip-span path. Idempotent and best-effort.
        """
        self.diarize = False
        for attr in ("spk_model", "punc_model"):
            try:
                if getattr(self.model, attr, None) is not None:
                    setattr(self.model, attr, None)
            except Exception:
                pass

    def _sentence_info_to_segments(self, res: Any) -> List[Dict[str, Any]]:
        """Convert funasr diarization 'sentence_info' into speaker-tagged segments.

        Each sentence_info entry has start/end (milliseconds), an integer ``spk``
        id, and the text under ``sentence`` (vad_segment mode, the SenseVoice
        path) or ``text`` (punc_segment mode). We strip rich tokens and prefix
        the speaker as ``[spkN] text`` so the label survives into the SRT.
        Returns [] if no usable sentence_info is present.
        """
        segments: List[Dict[str, Any]] = []
        for item in res:
            if not isinstance(item, dict):
                continue
            sinfo = item.get("sentence_info")
            if not sinfo or not isinstance(sinfo, (list, tuple)):
                continue
            for sent in sinfo:
                if not isinstance(sent, dict):
                    continue
                # funasr writes the per-sentence text under DIFFERENT keys
                # depending on spk_mode: "sentence" in vad_segment mode (the
                # SenseVoice path — auto_model.py:842) vs "text" in punc_segment
                # mode (timestamp_sentence). Accept both.
                raw_text = sent.get("sentence") or sent.get("text") or ""
                text = self._rich_postprocess(raw_text).strip() if raw_text else ""
                if not text:
                    continue
                # funasr sentence_info start/end are in milliseconds.
                start = float(sent.get("start", 0) or 0) / 1000.0
                end = float(sent.get("end", 0) or 0) / 1000.0
                if end <= start:
                    end = start + 0.1
                spk = sent.get("spk")
                if spk is not None:
                    text = f"[spk{spk}] {text}"
                segments.append({"text": text, "start": start, "end": end})
        return segments

    @staticmethod
    def _probe_duration(audio_path: Path) -> float:
        """Best-effort audio duration in seconds (0.0 if unknown)."""
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            if info.samplerate:
                return float(info.frames) / float(info.samplerate)
        except Exception as e:
            logger.debug(f"Could not probe duration for {audio_path}: {e}")
        return 0.0

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def unload_model(self) -> None:
        """Free the model and GPU memory."""
        if self.model is not None:
            logger.debug("Unloading SenseVoice model...")
            try:
                del self.model
            except Exception as e:
                logger.warning(f"Error deleting SenseVoice model: {e}")
            finally:
                self.model = None
            import gc
            try:
                gc.collect()
            except Exception:
                pass
            # CUDA cache cleanup is handled by the caller (safe_cuda_cleanup).
            logger.debug("SenseVoice model unloaded")

    def cleanup(self) -> None:
        """Alias for unload_model()."""
        self.unload_model()

    def __del__(self):
        try:
            self.unload_model()
        except Exception:
            pass
