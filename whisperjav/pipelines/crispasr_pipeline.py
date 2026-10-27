#!/usr/bin/env python3
"""The shared chain of the CrispASR pipelines (1.10.0; route B, code structure S2, owner W175-W176).

Two pipelines for the user, Parakeet-CrispASR and Cohere-CrispASR (Ensemble passes only, W118 2), one
chain in the code. Each pipeline is a small subclass that names itself, its models and its row of the
per-model table (ENGINE); everything else is here (route B design, sections 3-4):

  1. readiness, once per pass: CrispASR, the device (--device, W194 1), the model file (downloaded by
     WhisperJAV, SHA-256 checked, staged under a plain-ASCII name when needed);
  2. a mono WAV of the film (16 kHz; 48 kHz for a speech enhancer);
  3. scenes: the pass's scene detector, semantic by default with ChronosJAV's 12-48 s (W202 5d,
     W214 2), auditok and silero with their own minimum and 48 s (design 22.9.1); the audio analytics
     with semantic only (W202 5b, W211 C2, T55); then the pass's speech enhancer, if any, as the qwen
     pipeline runs it (one track, or enhanced audio for the speech segmenter only, W232 8);
  4. the speech segmenter, scene by scene, through ChronosJAV's VadGroupedFramer, with the settings of
     the pipeline the owner named (W199 DIR1, W202): fidelity's for Parakeet, the qwen pipeline's for
     Cohere (whisperjav/crispasr/pass_settings.py); it is released before the model runs;
  5. each VAD group's audio, cut from its scene's audio in memory, goes to CrispASR on its own, one
     call per group (Parakeet option P2, Cohere option C1); the model is opened once per pass (W202 5a);
  6. each group's result is taken apart and its subtitles built by the row's line rule
     (whisperjav/crispasr/lines.py): Parakeet cut only at natural cues (W203, W204), Cohere one
     subtitle per group;
  7. scene SRTs, stitched; the three filters (nonverbal, nonlinguistic, scene overlap; W129 6);
     <name>.ja.whisperjav.srt.

Times are the engine's own (W93 c5, W116): Parakeet's token times; Cohere's VAD group edges.
"""

import dataclasses
import gc
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from whisperjav.crispasr import install, lines
from whisperjav.crispasr.engine import SAMPLE_RATE, BindingEngine, CrispSegment
from whisperjav.crispasr.errors import CrispASRRunError, CrispASRSetupError
from whisperjav.crispasr.manifest import JAPANESE_NAMES, ModelFile, model_for
from whisperjav.crispasr.pass_settings import SegmenterSettings, segmenter_settings
from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.utils.logger import logger


@dataclass(frozen=True)
class EngineRow:
    """A pipeline's row of the per-model table (route B design 6)."""
    line_rule: str          # lines.LINE_RULES: "tokens" (Parakeet, P2) or "group" (Cohere, C1)
    audio_analytics: bool   # both pipelines today (Cohere W202 5b, Parakeet W211 C2)


@dataclass
class _Scene:
    index: int
    path: Path
    start: float  # media-file time, seconds: the scene detector's padded start
    end: float


@dataclass
class _Group:
    scene: int
    start: float  # scene time, seconds
    end: float


class CrispASRPipeline(BasePipeline):
    """The shared chain; subclasses set PIPELINE_NAME, LABEL and ENGINE."""

    PIPELINE_NAME = ""
    LABEL = ""
    ENGINE: EngineRow = EngineRow("group", False)

    # ChronosJAV's semantic scenes (qwen_pipeline.py, safe chunking; W202 5d). auditok and silero keep
    # their own minimum, with the same maximum: a 12 s minimum makes them drop a stretch of speech
    # shorter than 12 s between long silences (design 22.9.1; owner W232 14).
    SCENE_MIN_S = 12
    SCENE_MAX_S = 48
    SCENE_DETECTOR = "semantic"  # the default (owner W214 2)
    # T55 (design 22.10).
    ANALYTICS_NEED_SEMANTIC = ("{label}: audio analytics need the semantic scene detector; this pass uses "
                               "{method}, so they are not made.")
    MIN_GROUP_S = 0.1  # VadGroupedFramer drops shorter groups; nothing shorter is sent to CrispASR

    def __init__(self,
                 output_dir: str = "./output",
                 temp_dir: str = "./temp",
                 keep_temp_files: bool = False,
                 save_metadata_json: bool = False,
                 progress_display=None,
                 model_name: Optional[str] = None,
                 speech_segmenter: Optional[str] = None,
                 language: str = "ja",
                 subs_language: str = "native",
                 sensitivity: Optional[str] = None,
                 device: Optional[str] = None,
                 vad_threshold: Optional[float] = None,
                 speech_pad_ms: Optional[int] = None,
                 cues: Optional[lines.CueSettings] = None,
                 scene_detector: Optional[str] = None,
                 speech_enhancer: Optional[str] = None,
                 speech_enhancer_model: Optional[str] = None,
                 enhance_for_vad: bool = False,
                 scene_params: Optional[Dict[str, Any]] = None,
                 segmenter_overrides: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Args:
            model_name: a short name from the manifest; None means this pipeline's default.
            speech_segmenter: the speech segmenter; None means this pipeline's default. "none" is
                refused: the pipeline transcribes the VAD groups the segmenter finds.
            language: the source language; Japanese only.
            subs_language: "direct-to-english" still writes Japanese, with one console line.
            sensitivity: the pass's sensitivity; None means balanced (W135 2, W202 2).
            device: --device (None, auto, cuda, cpu; W194 1).
            vad_threshold, speech_pad_ms: --passN-vad-threshold and --passN-speech-pad-ms (W208), applied
                as fidelity or the qwen pipeline applies them (pass_settings.segmenter_settings).
            cues: the pause values of the "tokens" line rule (W208); None means lines.DEFAULT_CUES.
            scene_detector: auditok, silero, semantic or none; None means semantic (design 22.9.1, W214 2).
            speech_enhancer, speech_enhancer_model: the pass's speech enhancer; None or "none" means none.
            enhance_for_vad: the enhanced audio goes to the speech segmenter only, the original to
                CrispASR (W232 8), as the qwen pipeline's dual track.
            scene_params, segmenter_overrides: the pass's Customize values (Scene and Segmenter tabs).
            **kwargs: pass settings this pipeline does not use arrive here and are not used.
        """
        super().__init__(output_dir=output_dir, temp_dir=temp_dir, keep_temp_files=keep_temp_files,
                         save_metadata_json=save_metadata_json, **kwargs)
        from whisperjav.utils.progress_display import DummyProgress

        self.progress = progress_display or DummyProgress()
        if str(language or "ja").strip().lower() not in JAPANESE_NAMES:
            raise ValueError(f"{self.LABEL} transcribes Japanese only; the source language is '{language}'.")
        self.lang_code = "ja"
        self.subs_language = subs_language
        if subs_language == "direct-to-english":
            logger.warning(f"{self.LABEL} cannot translate: it writes Japanese subtitles. Translate them "
                           f"afterwards with --translate or whisperjav-translate.")
        self.model: ModelFile = model_for(self.PIPELINE_NAME, model_name)
        if speech_segmenter is not None and str(speech_segmenter).strip().lower() == "none":
            raise ValueError(f"{self.LABEL} needs a speech segmenter: it transcribes the VAD groups it finds.")
        self.segmenter: SegmenterSettings = segmenter_settings(self.PIPELINE_NAME, speech_segmenter, sensitivity,
                                                               vad_threshold=vad_threshold,
                                                               speech_pad_ms=speech_pad_ms)
        if segmenter_overrides:
            # The Customize Segmenter tab, on top of the resolved settings (as the qwen branch does).
            overrides = dict(segmenter_overrides)
            self.segmenter = dataclasses.replace(
                self.segmenter,
                max_group_duration_s=overrides.pop("max_group_duration_s", self.segmenter.max_group_duration_s),
                chunk_threshold_s=overrides.pop("chunk_threshold_s", self.segmenter.chunk_threshold_s),
                config={**self.segmenter.config, **overrides})
        self.speech_segmenter = self.segmenter.backend
        self.scene_detector = str(scene_detector or self.SCENE_DETECTOR).strip().lower()
        self.scene_params: Dict[str, Any] = dict(scene_params or {})
        self.enhancer_backend = str(speech_enhancer or "none").strip().lower()
        self.enhancer_model = speech_enhancer_model
        self.enhance_for_vad = bool(enhance_for_vad)
        self.sensitivity = self.segmenter.sensitivity
        self.device = device
        self.cues: lines.CueSettings = cues or lines.DEFAULT_CUES

        self.audio_extractor = None
        self.asr: Optional[BindingEngine] = None  # opened once per pass; BasePipeline.cleanup() closes it
        self._runtime: Optional[install.RuntimeChoice] = None
        self._model_path: Optional[Path] = None
        self._setup_error: Optional[Exception] = None
        self._device_line_shown = False
        self.degradations: List[str] = []
        self.notes: List[str] = []

        logger.info(f"{self.LABEL} ready: model {self.model.label}, speech segmenter {self.speech_segmenter} "
                    f"({self.sensitivity}), VAD groups of at most {self.segmenter.max_group_duration_s:g} s")

    def get_mode_name(self) -> str:
        return self.PIPELINE_NAME

    # -----------------------------------------------------------------------------------------------
    # The run
    # -----------------------------------------------------------------------------------------------

    def process(self, media_info: Dict) -> Dict:
        started = time.time()
        # The worker reuses one pipeline object for every file of the pass.
        self.degradations = []
        self.notes = []
        input_file = Path(media_info["path"])
        basename = media_info["basename"]
        md = self._new_metadata(input_file, basename)

        self.progress.set_current_step("Preparing CrispASR", 1, 6)
        runtime, model_path = self._ready()
        if not self._device_line_shown:
            logger.info(runtime.device_line(self.LABEL))
            self._device_line_shown = True
        if runtime.note_in_summary:
            self.notes.append(runtime.device_line(self.LABEL))
        md["config"]["crispasr"].update({"crispasr_version": runtime.version, "build": runtime.build,
                                         "device": runtime.device, "use_gpu": runtime.use_gpu,
                                         "model_path": str(model_path)})

        self.progress.set_current_step("Extracting audio", 2, 6)
        wav, duration = self._extract_audio(input_file, basename)
        md["input_info"] = {"processed_audio_file": str(wav), "audio_duration_seconds": duration}

        self.progress.set_current_step("Detecting scenes", 3, 6)
        scenes = self._detect_scenes(wav, basename, md)
        md["stages"]["scene_detection"] = {"scenes": len(scenes), "method": self.scene_detector}
        # The pass's speech enhancer (design 22.9.1): the scenes CrispASR reads, and the scenes the
        # speech segmenter reads (the enhanced ones with --passN-enhance-for-vad).
        scenes, vad_scenes = self._enhance(scenes, md)

        self.progress.set_current_step("Finding speech", 4, 6)
        groups, failed_scenes = self._speech_groups(vad_scenes)
        if failed_scenes:
            self.notes.append(f"{self.LABEL}: the speech segmenter failed on {failed_scenes} scene(s); "
                              f"their speech was not transcribed.")
        speech_s = sum(g.end - g.start for g in groups)
        md["stages"]["groups"] = {"count": len(groups), "seconds": round(speech_s, 2),
                                  "failed_scenes": failed_scenes}
        logger.info(f"{self.LABEL}: {len(groups)} VAD groups over {len(scenes)} scenes, "
                    f"{speech_s:.0f} s of speech in {duration:.0f} s")

        self.progress.set_current_step("Transcribing with CrispASR", 5, 6)
        results: List[Tuple[_Group, List[CrispSegment]]] = []
        try:
            if groups:
                results = self._transcribe(scenes, groups, md)
            else:
                logger.info(f"{self.LABEL}: no speech was found in {basename}")
        finally:
            if self.enhancer_backend not in ("", "none") and self.asr is not None and self.asr.is_open:
                # With a speech enhancer the model is opened for each file's ASR step and closed after
                # it, also when the step fails, so the enhancer and the CrispASR model never hold GPU
                # memory together (W217 19; review P7 finding 15).
                self.asr.close()

        self.progress.set_current_step("Building subtitles", 6, 6)
        stitched, count_raw = self._build_subtitles(scenes, results, basename, md)
        if self.save_metadata_json:
            self._write_group_record(basename, scenes, results)
        stats, count = self._postprocess(stitched, count_raw)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_srt = self.output_dir / f"{basename}.{self.lang_code}.whisperjav.srt"
        if count > 0:
            shutil.copy2(stitched, final_srt)
        else:
            final_srt.write_text("", encoding="utf-8")

        total = time.time() - started
        md["output_files"] = {"final_srt": str(final_srt), "stitched_srt": str(stitched)}
        md["srt_path"] = str(final_srt)
        md["stages"]["postprocessing"] = stats
        md["summary"] = {
            "final_subtitles_refined": count,
            "final_subtitles_raw": count_raw,
            "quality_metrics": {"hallucinations_removed": 0, "repetitions_removed": 0,
                                "duration_adjustments": stats.get("overlap_shifted", 0),
                                "empty_removed": stats.get("empty_removed", 0), "cps_filtered": 0,
                                "logprob_filtered": 0, "nonverbal_filtered": stats.get("nonverbal_dropped", 0)},
            "total_processing_time_seconds": round(total, 2),
            "degradations": list(self.degradations),
            "notes": list(self.notes),
        }
        md["total_time_sec"] = round(total, 2)
        md["metadata_master"]["updated_at"] = datetime.now().isoformat() + "Z"

        if self.save_metadata_json:
            self.metadata_manager.save_master_metadata(md, basename)
        self.cleanup_temp_files(basename)
        logger.info(f"{self.LABEL}: {count} subtitles in {final_srt} ({total:.1f} s)")
        return md

    def _new_metadata(self, input_file: Path, basename: str) -> Dict:
        now = datetime.now().isoformat() + "Z"
        return {
            "metadata_master": {"structure_version": "1.0.0", "created_at": now, "updated_at": now},
            "input_file": str(input_file),
            "basename": basename,
            "pipeline": self.PIPELINE_NAME,
            "config": {"crispasr": {"model": self.model.name, "repo": self.model.repo_id,
                                    "revision": self.model.revision, "backend": self.model.backend,
                                    "line_rule": self.ENGINE.line_rule,
                                    "cues": self.cues.as_dict() if self.ENGINE.line_rule == "tokens" else None,
                                    "speech_segmenter_settings": self.segmenter.as_dict(),
                                    "threads": None}},
            "stages": {},
            "output_files": {},
            "summary": {},
        }

    # -----------------------------------------------------------------------------------------------
    # 1. Readiness, once per pass
    # -----------------------------------------------------------------------------------------------

    def _ready(self) -> Tuple[install.RuntimeChoice, Path]:
        """CrispASR, the device and the model file, made ready once per pass.

        A setup failure is kept and raised again for the pass's other files, so a failed 2.4 GB
        download is not repeated for every film.
        """
        if self._setup_error is not None:
            raise self._setup_error
        try:
            if self._runtime is None:
                self._runtime = install.prepare_runtime(self.device, self.model)
            if self._model_path is None or not Path(self._model_path).is_file():
                path = install.ensure_model(self.model)
                self._model_path = install.stage_model(path, self.temp_dir, self.model.sha256)
        except CrispASRSetupError as e:
            self._setup_error = e
            raise
        return self._runtime, self._model_path

    def _engine(self, md: Dict) -> BindingEngine:
        """The engine, opened at the first file's ASR step and kept for the pass (W202 5a)."""
        if self._setup_error is not None:
            raise self._setup_error
        if self.asr is None:
            self.asr = BindingEngine(self._model_path, self.model.backend, self._runtime.use_gpu,
                                     label=self.model.label)
        if not self.asr.is_open:
            opened = time.time()
            try:
                self.asr.open()
            except CrispASRSetupError as e:
                self._setup_error = e
                self.asr = None
                raise
            md["stages"].setdefault("asr", {})["open_seconds"] = round(time.time() - opened, 2)
        md["config"]["crispasr"]["threads"] = self.asr.n_threads
        return self.asr

    # -----------------------------------------------------------------------------------------------
    # 2-4. Audio, scenes, speech
    # -----------------------------------------------------------------------------------------------

    def _extract_audio(self, input_file: Path, basename: str) -> Tuple[Path, float]:
        from whisperjav.modules.audio_extraction import AudioExtractor
        from whisperjav.modules.audio_integrity import apply_to_run as apply_audio_integrity

        if self.audio_extractor is None:
            # 16 kHz, or 48 kHz for a speech enhancer (it resamples its output to 16 kHz).
            from whisperjav.modules.speech_enhancement import get_extraction_sample_rate
            self.audio_extractor = AudioExtractor(sample_rate=get_extraction_sample_rate(self.enhancer_backend))
        wav, duration = self.audio_extractor.extract(input_file, self.temp_dir / f"{basename}_extracted.wav")
        apply_audio_integrity(self.audio_extractor.last_integrity, self.degradations,
                              in_summary=self.audio_integrity_in_summary)
        return Path(wav), float(duration)

    def scene_detector_kwargs(self) -> Dict[str, Any]:
        """The scene detector's settings for this pass (design 22.9.1): semantic with ChronosJAV's
        12-48 s; auditok and silero with their own minimum and 48 s; none, one scene per file. The
        Customize Scene tab's values on top."""
        kwargs: Dict[str, Any] = {"method": self.scene_detector}
        if self.scene_detector == "semantic":
            kwargs.update(min_duration=self.SCENE_MIN_S, max_duration=self.SCENE_MAX_S)
        elif self.scene_detector in ("auditok", "silero"):
            kwargs.update(max_duration=self.SCENE_MAX_S)
        kwargs.update(self.scene_params)
        return kwargs

    def _detect_scenes(self, wav: Path, basename: str, md: Dict) -> List[_Scene]:
        """The pass's scenes (scene_detector_kwargs); the audio analytics with semantic scenes."""
        from whisperjav.modules.scene_detection_backends import SceneDetectorFactory

        scenes_dir = self.temp_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(**self.scene_detector_kwargs())
        try:
            result = detector.detect_scenes(Path(wav), scenes_dir, basename)
            tuples = result.to_legacy_tuples()
            method = getattr(result, "method", self.scene_detector)
        finally:
            detector.cleanup()
        scenes = [_Scene(i, Path(p), float(s), float(e)) for i, (p, s, e, _d) in enumerate(tuples)]
        if self.ENGINE.audio_analytics and self.scene_detector != "semantic":
            # The analytics need the semantic detector's scenes (design 21.4): one line per file (T55).
            logger.info(self.ANALYTICS_NEED_SEMANTIC.format(label=self.LABEL, method=self.scene_detector))
        elif self.ENGINE.audio_analytics:
            # As the qwen pipeline: which scenes look acoustically difficult. Prints only; never raises.
            from whisperjav.modules.analytics import report as report_audio_analytics

            analytics = report_audio_analytics(wav, [(sc.index, sc.start, sc.end) for sc in scenes],
                                               scene_method=method, vad_threshold=None)
            if analytics is not None:
                md.setdefault("audio_analytics", analytics.to_dict())
        return scenes

    def _enhance(self, scenes: List[_Scene], md: Dict) -> Tuple[List[_Scene], List[_Scene]]:
        """The pass's speech enhancer, as the qwen pipeline runs it (qwen_pipeline.py, phase 3):
        (the scenes CrispASR reads, the scenes the speech segmenter reads), both at 16 kHz.

        One track: the enhanced scenes for both. --passN-enhance-for-vad: the enhanced scenes for the
        speech segmenter, the original scenes resampled to 16 kHz for CrispASR. The enhancer is
        released before the speech segmenter and the CrispASR model run (W217 19).
        """
        from whisperjav.modules.speech_enhancement import (create_enhancer_direct, enhance_scenes,
                                                           is_passthrough_backend, resample_scenes)

        if is_passthrough_backend(self.enhancer_backend):
            if self.enhance_for_vad:
                logger.warning(f"{self.LABEL}: --enhance-for-vad does nothing without a speech enhancer.")
            return scenes, scenes
        started = time.time()
        tuples = [(sc.path, sc.start, sc.end, sc.end - sc.start) for sc in scenes]
        from whisperjav.utils.device_detector import pass_device
        # The enhancer runs on PyTorch and follows --device as every part does (22.3.1; P7 finding 3).
        enhancer = create_enhancer_direct(backend=self.enhancer_backend, model=self.enhancer_model,
                                          device=pass_device(self.device))
        try:
            enhanced = enhance_scenes(tuples, enhancer, self.temp_dir, degradations=self.degradations)
        finally:
            enhancer.cleanup()
            del enhancer
            from whisperjav.utils.gpu_utils import safe_cuda_cleanup
            safe_cuda_cleanup()
        enhanced_scenes = [_Scene(sc.index, Path(t[0]), sc.start, sc.end) for sc, t in zip(scenes, enhanced)]
        asr_scenes = enhanced_scenes
        if self.enhance_for_vad:
            originals = resample_scenes(tuples, self.temp_dir)
            asr_scenes = [_Scene(sc.index, Path(t[0]), sc.start, sc.end) for sc, t in zip(scenes, originals)]
        md["stages"]["enhancement"] = {"backend": self.enhancer_backend, "dual_track": self.enhance_for_vad,
                                       "time_sec": round(time.time() - started, 2)}
        return asr_scenes, enhanced_scenes

    @staticmethod
    def _read_scene(path: Path):
        """A scene WAV as mono float32 at 16 kHz."""
        import numpy as np
        import soundfile as sf

        audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        if rate != SAMPLE_RATE:
            raise CrispASRRunError(f"the scene audio {path.name} is at {rate} Hz, not {SAMPLE_RATE} Hz")
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _speech_groups(self, scenes: List[_Scene]) -> Tuple[List[_Group], int]:
        """The VAD groups of every scene, in scene time; the segmenter is released before returning.

        Returns the groups and the number of scenes the segmenter failed on.
        """
        from whisperjav.modules.subtitle_pipeline.framers.vad_grouped import VadGroupedFramer

        framer = VadGroupedFramer(segmenter_backend=self.segmenter.backend,
                                  max_group_duration_s=self.segmenter.max_group_duration_s,
                                  chunk_threshold_s=self.segmenter.chunk_threshold_s,
                                  min_frame_duration_s=self.MIN_GROUP_S,
                                  segmenter_config=dict(self.segmenter.config))
        groups: List[_Group] = []
        failed = 0
        try:
            for scene in scenes:
                try:
                    audio = self._read_scene(scene.path)
                    result = framer.frame(audio, SAMPLE_RATE)
                except Exception as e:
                    failed += 1
                    logger.warning(f"{self.LABEL}: speech segmentation failed for scene {scene.index}: {e}")
                    continue
                if result.metadata.get("segmenter_error"):
                    # The segmenter caught its own failure and returned an empty result: not silence.
                    failed += 1
                    logger.warning(f"{self.LABEL}: speech segmentation failed for scene {scene.index}: "
                                   f"{result.metadata['segmenter_error']}")
                    continue
                clip_end = len(audio) / SAMPLE_RATE
                for frame in result.frames:
                    start, end = max(0.0, frame.start), min(clip_end, frame.end)
                    if end - start >= self.MIN_GROUP_S:
                        groups.append(_Group(scene.index, start, end))
        finally:
            try:
                framer.cleanup()
            except Exception as e:
                logger.debug(f"{self.LABEL}: segmenter cleanup: {e}")
            del framer
            gc.collect()
        return groups, failed

    # -----------------------------------------------------------------------------------------------
    # 5. One CrispASR call per VAD group
    # -----------------------------------------------------------------------------------------------

    def _transcribe(self, scenes: List[_Scene], groups: List[_Group],
                    md: Dict) -> List[Tuple[_Group, List[CrispSegment]]]:
        """Each group's audio, cut from its scene's audio in memory, to CrispASR on its own."""
        engine = self._engine(md)
        scene_of = {sc.index: sc for sc in scenes}
        progress = self._progress_logger()
        results: List[Tuple[_Group, List[CrispSegment]]] = []
        failed, without_text = 0, 0
        first_reason: Optional[str] = None
        began = time.time()
        loaded_scene, audio = None, None
        for i, g in enumerate(groups):
            segments: List[CrispSegment] = []
            try:
                if loaded_scene != g.scene:
                    audio, loaded_scene = None, None
                    audio = self._read_scene(scene_of[g.scene].path)
                    loaded_scene = g.scene
                piece = audio[int(g.start * SAMPLE_RATE):int(g.end * SAMPLE_RATE)].copy()
                if len(piece) < int(self.MIN_GROUP_S * SAMPLE_RATE):
                    raise CrispASRRunError("the VAD group's audio is shorter than 0.1 s")
                segments = engine.transcribe(piece)
            except (CrispASRRunError, ValueError, OSError, RuntimeError) as e:
                failed += 1
                first_reason = first_reason or str(e)
                logger.debug(f"{self.LABEL}: VAD group {i} (scene {g.scene}, {g.start:.2f}-{g.end:.2f} s) failed: {e}")
            else:
                if not any(seg.text.strip() for seg in segments):
                    without_text += 1
            results.append((g, segments))
            progress(i + 1, len(groups))
        md["stages"].setdefault("asr", {}).update({"groups": len(groups), "without_text": without_text,
                                                   "failed": failed, "seconds": round(time.time() - began, 2)})
        if failed == len(groups):
            raise CrispASRRunError(f"the CrispASR call failed for every VAD group of this file: {first_reason}")
        if failed:
            self.degradations.append(f"{self.LABEL}: {failed} of {len(groups)} VAD group(s) could not be "
                                     f"transcribed ({first_reason})")
        if without_text:
            self.notes.append(f"{self.LABEL}: {without_text} of {len(groups)} VAD group(s) gave no text")
        return results

    def _progress_logger(self) -> Callable[[int, int], None]:
        """Console lines at every tenth of the groups: in an Ensemble pass, the log is the progress display."""
        state = {"next": 10}

        def report(done: int, total: int) -> None:
            percent = int(done * 100 / max(1, total))
            if percent >= state["next"] or done >= total:
                logger.info(f"{self.LABEL}: {done} of {total} VAD groups transcribed ({percent}%)")
                state["next"] = (percent // 10 + 1) * 10

        return report

    # -----------------------------------------------------------------------------------------------
    # 6-7. Subtitles per scene, stitching, post-processing
    # -----------------------------------------------------------------------------------------------

    def _build_subtitles(self, scenes: List[_Scene], results: List[Tuple[_Group, List[CrispSegment]]],
                         basename: str, md: Dict) -> Tuple[Path, int]:
        """Each group's subtitles by the row's line rule; one SRT per scene in scene time; stitched."""
        import srt
        from whisperjav.modules.srt_stitching import SRTStitcher

        scene_of = {sc.index: sc for sc in scenes}
        stats = lines.LineStats()
        by_scene: Dict[int, List[lines.Subtitle]] = {}
        for g, segments in results:
            scene = scene_of[g.scene]
            subs = lines.build(self.ENGINE.line_rule, scene.start + g.start, scene.start + g.end, segments, stats,
                               self.cues)
            by_scene.setdefault(g.scene, []).extend(subs)

        scene_srts_dir = self.temp_dir / "scene_srts"
        scene_srts_dir.mkdir(parents=True, exist_ok=True)
        scene_srt_info: List[Tuple[Path, float]] = []
        total = 0
        for index in sorted(by_scene):
            subs = sorted(by_scene[index], key=lambda s: s[1])
            if not subs:
                continue
            scene = scene_of[index]
            entries = [srt.Subtitle(index=n + 1, start=timedelta(seconds=max(0.0, s - scene.start)),
                                    end=timedelta(seconds=max(0.0, e - scene.start)), content=text)
                       for n, (text, s, e) in enumerate(subs)]
            scene_srt = scene_srts_dir / f"{basename}_scene_{index:04d}.srt"
            scene_srt.write_text(srt.compose(entries), encoding="utf-8")
            scene_srt_info.append((scene_srt, scene.start))
            total += len(entries)

        md["stages"]["lines"] = {"subtitles": total, "scenes_with_lines": len(scene_srt_info),
                                 "clamped_times": stats.clamped_times,
                                 "segments_without_token_times": stats.fallback_segments}
        stitched = self.temp_dir / f"{basename}_stitched.srt"
        count = 0
        if scene_srt_info:
            count = SRTStitcher().stitch(scene_srt_info, stitched)
        else:
            stitched.write_text("", encoding="utf-8")
        return stitched, count

    def _postprocess(self, stitched: Path, count: int) -> Tuple[Dict[str, int], int]:
        """The three filters, in place: nonverbal, nonlinguistic, scene overlap (W129 6)."""
        stats: Dict[str, int] = {}
        if count == 0:
            return stats, 0
        from whisperjav.modules.subtitle_pipeline.cleaners.nonlinguistic_utterance_filter import (
            NonlinguisticUtteranceFilter)
        from whisperjav.modules.subtitle_pipeline.cleaners.nonverbal_line_filter import NonverbalLineFilter
        from whisperjav.modules.subtitle_pipeline.cleaners.scene_overlap_resolver import SceneOverlapResolver

        r = NonverbalLineFilter().filter_srt_file(stitched)
        stats["nonverbal_dropped"] = r.get("dropped_nonverbal", 0)
        stats["empty_removed"] = r.get("dropped_empty", 0)
        count = r.get("final_count", count)
        if count:
            r = NonlinguisticUtteranceFilter().filter_srt_file(stitched)
            stats["nonlinguistic_dropped"] = r.get("dropped_nonlinguistic", 0)
            stats["empty_removed"] += r.get("dropped_empty", 0)
            count = r.get("final_count", count)
        if count:
            r = SceneOverlapResolver().resolve_srt_file(stitched)
            stats["overlap_nested_dropped"] = r.get("dropped_nested", 0)
            stats["overlap_shifted"] = r.get("shifted_starts", 0)
            count = r.get("final_count", count)
        return stats, count

    def _write_group_record(self, basename: str, scenes: List[_Scene],
                            results: List[Tuple[_Group, List[CrispSegment]]]) -> None:
        """With --debug: one JSON line per VAD group, with CrispASR's result in group time (owner O12).

        In the output folder's raw_subs/, because a pass deletes its own temp folder at its end.
        """
        scene_of = {sc.index: sc for sc in scenes}
        folder = self.output_dir / "raw_subs"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            # Named after the pipeline too: both passes of an Ensemble run share the basename.
            with open(folder / f"{basename}.{self.PIPELINE_NAME}.groups.jsonl", "w", encoding="utf-8") as fh:
                for g, segments in results:
                    scene = scene_of[g.scene]
                    fh.write(json.dumps({
                        "scene": g.scene, "start": round(scene.start + g.start, 3),
                        "end": round(scene.start + g.end, 3),
                        "segments": [{"text": s.text, "start": s.start, "end": s.end,
                                      "words": [[w.text, w.start, w.end, w.confidence] for w in s.words],
                                      "tokens": [[t.text, t.start, t.end, t.confidence] for t in s.tokens]}
                                     for s in segments],
                    }, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"{self.LABEL}: the per-group record could not be written ({e})")
