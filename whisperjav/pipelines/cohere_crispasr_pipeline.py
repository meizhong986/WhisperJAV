#!/usr/bin/env python3
"""Cohere-CrispASR: a Japanese fine-tune of Cohere Transcribe, run by CrispASR's own program (1.10.0, item 4).

An Ensemble Mode pass pipeline only (W118 2): ``whisperjav --ensemble --pass1-pipeline cohere-crispasr
--pass1-model <name>``. It has no --mode value and no entry in the Transcription Mode tab. Its model is
one of two Japanese fine-tunes (W118 4); the step 2b measurement picks the default, so until then a run
names its model.

One run, stage by stage (master plan 3.2):
  1. the program and the model are made ready -- downloaded if missing -- before any audio is read,
     so a failure costs nothing (plan 1.5, 2.6);
  2. a 16 kHz mono WAV of the film;
  3. semantic scenes of 12-48 s, as ChronosJAV makes them;
  4. the speech segmenter, scene by scene, groups speech into pieces of at most 30 s; it is then
     released, so the program gets the GPU memory;
  5. this pipeline's SLICE RULE turns the groups into cut points in film time;
  6. one program run over the whole film reads the cut points (--vad-import) and writes -ojf JSON;
  7. each segment is tied to its slice, and this pipeline's PARSER turns its tokens into words;
  8. scene by scene, this pipeline's LINE BUILDING makes the subtitle lines; the scene SRTs are stitched;
  9. this pipeline's POST-PROCESSING, then <name>.ja.whisperjav.srt.

What it shares with the other CrispASR pipelines is only what CrispASR's formats make the same for every
engine (owner decision Q20 = A, W118 1): the program runner, the JSON reader, the cut-point writer and the
downloads, in whisperjav/crispasr/. Every Cohere-specific rule is in this file.

Times are the engine's own (W116): Cohere's token times are estimated by CrispASR from the decoder (DTW);
they are only clamped to their slice, and no stable-ts silence adjustment moves them. Each segment ends
at its slice's end. Speech-bounded slices keep the model out of stretches without speech, where CrispASR
reports that Cohere invents text (W113 X3); quiet audio that is not silent is unknown U10.
"""

import gc
import math
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from whisperjav.crispasr import install
from whisperjav.crispasr.manifest import ModelFile, model_for
from whisperjav.crispasr.ojf import (CrispSegment, CutPoint, header_mismatch, parse_ojf, place_in_slices,
                                     read_ojf, write_vad_import)
from whisperjav.crispasr.runner import (CrispASRRunError, CrispOutcome, CrispRunRequest, film_run_dir,
                                        run_crispasr_json)
from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.utils.logger import logger

_JAPANESE = frozenset({"ja", "jp", "jpn", "japanese"})


@dataclass
class _Scene:
    index: int
    path: Path
    start: float  # film time, seconds: the scene detector's padded start
    end: float


@dataclass
class _Group:
    scene: int    # the scene it was found in
    start: float  # film time, seconds
    end: float
    pauses: List[float] = field(default_factory=list)  # film-time middles of the gaps inside the group


def _count_srt(path: Path) -> int:
    import srt
    try:
        return len(list(srt.parse(Path(path).read_text(encoding="utf-8", errors="replace"))))
    except Exception:
        return 0


class CohereCrispASRPipeline(BasePipeline):
    """Cohere-CrispASR, an Ensemble pass pipeline (master plan 3.1, 3.3)."""

    PIPELINE_NAME = "cohere-crispasr"
    LABEL = "Cohere-CrispASR"

    # Slice rule (W113 X3, plan 3.4): speech-bounded slices of at most 30 s, Cohere's own window.
    MAX_SLICE_S = 30.0
    # "windows" -- fixed windows over the scenes -- is the step 2b comparison, reached only through this
    # constructor argument; it is no user setting (plan 3.7).
    SLICE_FORMS = ("speech", "windows")
    MIN_SLICE_S = 0.1     # shorter groups are dropped, as the vad-grouped framer does
    SPLIT_MARGIN_S = 0.5  # a split never lands closer than this to a slice's ends
    # The speech segmenter's settings (plan 3.4): ChronosJAV's pads with WhisperSeg, and the framer's
    # default gap. Step 2b confirms them.
    SEGMENTER_START_PAD_MS = 100
    SEGMENTER_END_PAD_MS = 0
    SEGMENTER_CHUNK_THRESHOLD_S = 1.0

    # Line building (W113 X3, plan 3.3): Japanese line splitting with REGROUP_JAV over the folded tokens.
    LINE_MODES = ("regroup",)

    # Post-processing (plan 3.6): the two text-only filters, and the scene-overlap resolver, which moves
    # times only where two lines overlap. Provisional until step 2b scores the candidates (Q2 (2)).
    POSTPROCESS_STEPS = ("nonverbal", "nonlinguistic", "overlap")
    DEFAULT_POSTPROCESS = ("nonverbal", "nonlinguistic", "overlap")

    CLAMP_TOLERANCE_S = 0.05  # a time moved further than this into its slice counts as clamped

    def __init__(self,
                 output_dir: str = "./output",
                 temp_dir: str = "./temp",
                 keep_temp_files: bool = False,
                 save_metadata_json: bool = False,
                 progress_display=None,
                 model_name: Optional[str] = None,
                 speech_segmenter: Optional[str] = "whisperseg",
                 language: str = "ja",
                 subs_language: str = "native",
                 slice_form: str = "speech",
                 line_mode: str = "regroup",
                 postprocess: Optional[Tuple[str, ...]] = None,
                 **kwargs):
        """
        Args:
            model_name: a short name from the manifest (plan 3.3). This pipeline has no default model
                until step 2b picks one, so a run must name it.
            speech_segmenter: WhisperJAV's speech segmenter for the slices; default WhisperSeg. "none"
                is refused: the slices follow speech (Q14).
            language: the source language; Japanese only (Q19).
            subs_language: "direct-to-english" still writes Japanese, with one console line (Q19).
            slice_form, line_mode, postprocess: the step 2b comparisons (plan 3.4, 3.3, 3.6).
            **kwargs: pass settings this pipeline does not use (sensitivity, scene detector, speech
                enhancer, ...) arrive here and are not used.
        """
        super().__init__(output_dir=output_dir, temp_dir=temp_dir, keep_temp_files=keep_temp_files,
                         save_metadata_json=save_metadata_json, **kwargs)
        from whisperjav.utils.progress_display import DummyProgress

        self.progress = progress_display or DummyProgress()

        if str(language or "ja").strip().lower() not in _JAPANESE:
            raise ValueError(f"{self.LABEL} transcribes Japanese only; the source language is '{language}'.")
        self.lang_code = "ja"
        self.subs_language = subs_language
        if subs_language == "direct-to-english":
            logger.warning(f"{self.LABEL} cannot translate: it writes Japanese subtitles. Translate them "
                           f"afterwards with --translate or whisperjav-translate.")

        # The models this pipeline offers, and those measured in step 2b but not yet offered (plan 3.3);
        # main.py lets only the offered ones through the command line.
        self.model: ModelFile = model_for(self.PIPELINE_NAME, model_name, include_measurement=True)

        if slice_form not in self.SLICE_FORMS:
            raise ValueError(f"{self.LABEL}: unknown slice form '{slice_form}' ({', '.join(self.SLICE_FORMS)})")
        self.slice_form = slice_form
        self.speech_segmenter = (speech_segmenter or "whisperseg").strip().lower()
        if self.slice_form == "speech" and self.speech_segmenter == "none":
            raise ValueError(f"{self.LABEL} needs a speech segmenter: its slices follow speech.")
        if line_mode not in self.LINE_MODES:
            raise ValueError(f"{self.LABEL}: unknown line mode '{line_mode}' ({', '.join(self.LINE_MODES)})")
        self.line_mode = line_mode
        steps = self.DEFAULT_POSTPROCESS if postprocess is None else tuple(postprocess)
        unknown = [s for s in steps if s not in self.POSTPROCESS_STEPS]
        if unknown:
            raise ValueError(f"{self.LABEL}: unknown post-processing step(s) {unknown} "
                             f"({', '.join(self.POSTPROCESS_STEPS)})")
        self.postprocess = steps

        self.audio_extractor = None
        self._program: Optional[install.ProgramChoice] = None
        self._model_path: Optional[Path] = None
        self._setup_error: Optional[Exception] = None
        self._program_note_shown = False
        # The last film's scenes, slices and placed segments, for the step 2b measurement script.
        self.last_run: Optional[Dict] = None
        self.degradations: List[str] = []
        self.notes: List[str] = []

        logger.info(f"{self.LABEL} ready: model {self.model.label}, speech segmenter {self.speech_segmenter}, "
                    f"slices {self.slice_form} (at most {self.MAX_SLICE_S:.0f} s), lines {self.line_mode}")

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
        program, model_path = self._ready()
        if program.note:
            # Plan 2.6: one console line per pass; a note in each file's summary only for a fallback
            # that was not expected (the GPU build failed its start check, PyTorch's files were missing).
            if not self._program_note_shown:
                logger.info(program.note)
                self._program_note_shown = True
            if program.note_in_summary:
                self.notes.append(program.note)
        md["config"]["crispasr"].update({"build": program.build.key, "program": str(program.exe),
                                         "gpu_devices": list(program.check.gpu_devices),
                                         "model_path": str(model_path)})

        self.progress.set_current_step("Extracting audio", 2, 6)
        wav, duration = self._extract_audio(input_file, basename)
        md["input_info"] = {"processed_audio_file": str(wav), "audio_duration_seconds": duration}

        self.progress.set_current_step("Detecting scenes", 3, 6)
        scenes = self._detect_scenes(wav, basename)
        md["stages"]["scene_detection"] = {"scenes": len(scenes)}

        self.progress.set_current_step("Finding speech", 4, 6)
        if self.slice_form == "speech":
            groups, unsegmented = self._speech_groups(scenes)
            if unsegmented:
                self.notes.append(f"{self.LABEL}: the speech segmenter failed on {unsegmented} scene(s); "
                                  f"their speech was not transcribed.")
            cuts, cut_scene = self._speech_slices(groups, duration)
        else:
            cuts, cut_scene = self._window_slices(scenes, duration)
        md["stages"]["slices"] = {"form": self.slice_form, "count": len(cuts),
                                  "seconds": round(sum(c.duration for c in cuts), 2)}
        logger.info(f"{self.LABEL}: {len(cuts)} slices over {len(scenes)} scenes, "
                    f"{sum(c.duration for c in cuts):.0f} s of {duration:.0f} s")

        self.progress.set_current_step("Transcribing with CrispASR", 5, 6)
        run_dir = film_run_dir(self.temp_dir, basename)
        segments: List[Tuple[int, CrispSegment]] = []
        if cuts:
            segments = self._transcribe(program, model_path, wav, duration, cuts, run_dir, md)
        else:
            logger.info(f"{self.LABEL}: no speech was found in {basename}")

        # Kept for the step 2b measurement script, which rebuilds the lines with other settings from
        # this same program run (plan 3.6: every candidate is scored from the same runs).
        self.last_run = {"basename": basename, "scenes": scenes, "cuts": cuts, "cut_scene": cut_scene,
                         "segments": segments}
        self.progress.set_current_step("Building subtitles", 6, 6)
        stitched = self._build_lines(scenes, cuts, cut_scene, segments, basename, md)
        count_raw = _count_srt(stitched)
        stats = self._postprocess(stitched)
        count = _count_srt(stitched)

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
        if not self.keep_temp_files:
            shutil.rmtree(run_dir, ignore_errors=True)
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
                                    "speech_segmenter": self.speech_segmenter, "slice_form": self.slice_form,
                                    "line_mode": self.line_mode, "postprocess": list(self.postprocess)}},
            "stages": {},
            "output_files": {},
            "summary": {},
        }

    # -----------------------------------------------------------------------------------------------
    # 1. Program and model
    # -----------------------------------------------------------------------------------------------

    def _ready(self) -> Tuple[install.ProgramChoice, Path]:
        """The program build for this PC and the model file, made ready once per pass.

        A setup failure is kept and raised again for the pass's other files, so a failed 2.4 GB
        download is not repeated for every film.
        """
        if self._setup_error is not None:
            raise self._setup_error
        try:
            # Gone since they were prepared (deleted, or taken by an antivirus): prepare them again.
            if self._program is not None and not Path(self._program.exe).is_file():
                self._program = None
            if self._model_path is not None and not Path(self._model_path).is_file():
                self._model_path = None
            if self._program is None:
                self._program = install.prepare_program()
            if self._model_path is None:
                self._model_path = install.ensure_model(self.model)
        except install.CrispASRSetupError as e:
            self._setup_error = e
            raise
        return self._program, self._model_path

    # -----------------------------------------------------------------------------------------------
    # 2-4. Audio, scenes, speech
    # -----------------------------------------------------------------------------------------------

    def _extract_audio(self, input_file: Path, basename: str) -> Tuple[Path, float]:
        from whisperjav.modules.audio_extraction import AudioExtractor
        from whisperjav.modules.audio_integrity import apply_to_run as apply_audio_integrity

        if self.audio_extractor is None:
            self.audio_extractor = AudioExtractor(sample_rate=16000)
        wav, duration = self.audio_extractor.extract(input_file, self.temp_dir / f"{basename}_extracted.wav")
        apply_audio_integrity(self.audio_extractor.last_integrity, self.degradations,
                              in_summary=self.audio_integrity_in_summary)
        return Path(wav), float(duration)

    def _detect_scenes(self, wav: Path, basename: str) -> List[_Scene]:
        """Semantic scenes with safe chunking, 12-48 s, as ChronosJAV makes them (qwen_pipeline.py)."""
        from whisperjav.modules.scene_detection_backends import SceneDetectorFactory

        scenes_dir = self.temp_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(method="semantic", min_duration=12,
                                                                       max_duration=48)
        try:
            result = detector.detect_scenes(Path(wav), scenes_dir, basename)
            tuples = result.to_legacy_tuples()
        finally:
            detector.cleanup()
        return [_Scene(i, Path(p), float(s), float(e)) for i, (p, s, e, _d) in enumerate(tuples)]

    def _speech_groups(self, scenes: List[_Scene]) -> Tuple[List[_Group], int]:
        """Speech groups of every scene, in film time; the segmenter is released before returning.

        Returns the groups and the number of scenes the segmenter failed on.
        """
        from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory

        pads_s = (self.SEGMENTER_START_PAD_MS + self.SEGMENTER_END_PAD_MS) / 1000.0
        segmenter = SpeechSegmenterFactory.create(
            self.speech_segmenter,
            max_group_duration_s=self.MAX_SLICE_S,
            chunk_threshold_s=self.SEGMENTER_CHUNK_THRESHOLD_S,
            start_pad_ms=self.SEGMENTER_START_PAD_MS,
            end_pad_ms=self.SEGMENTER_END_PAD_MS,
            # A padded segment then still fits the window; the slice rule splits anything longer.
            max_speech_duration_s=self.MAX_SLICE_S - pads_s,
        )
        groups: List[_Group] = []
        failed = 0
        try:
            for scene in scenes:
                try:
                    result = segmenter.segment(scene.path, sample_rate=16000)
                except Exception as e:
                    failed += 1
                    logger.warning(f"{self.LABEL}: speech segmentation failed for scene {scene.index}: {e}")
                    continue
                if getattr(result, "error", None):
                    # The segmenter caught its own failure and returned an empty result: not silence.
                    failed += 1
                    continue
                for g in result.groups or []:
                    if not g:
                        continue
                    pauses = [scene.start + (a.end_sec + b.start_sec) / 2.0
                              for a, b in zip(g, g[1:]) if b.start_sec > a.end_sec]
                    groups.append(_Group(scene.index, scene.start + g[0].start_sec, scene.start + g[-1].end_sec,
                                         pauses))
        finally:
            try:
                segmenter.cleanup()
            except Exception as e:
                logger.debug(f"{self.LABEL}: segmenter cleanup: {e}")
            del segmenter
            gc.collect()
        return groups, failed

    # -----------------------------------------------------------------------------------------------
    # 5. This pipeline's slice rule (plan 3.4)
    # -----------------------------------------------------------------------------------------------

    def _speech_slices(self, groups: List[_Group], audio_s: float) -> Tuple[List[CutPoint], List[int]]:
        """Cut points from speech groups: at most 30 s each, sorted, not overlapping, inside the audio.

        - Groups shorter than 0.1 s are dropped.
        - Groups that overlap -- at a scene edge, where neighbouring scenes share up to 0.7 s -- are
          merged; the merged slice belongs to the scene of its earliest group.
        - Anything over 30 s is split at the pause nearest its middle, or at 30 s when it has no pause.
        Returns the cut points and, for each, the scene it belongs to.
        """
        items = sorted(((max(0.0, g.start), min(audio_s, g.end), g.scene, g.pauses) for g in groups),
                       key=lambda t: (t[0], t[1]))
        merged: List[list] = []
        for start, end, scene, pauses in items:
            if end - start < self.MIN_SLICE_S:
                continue
            if merged and start < merged[-1][1]:
                last = merged[-1]
                last[1] = max(last[1], end)
                last[3] = last[3] + list(pauses)
            else:
                merged.append([start, end, scene, list(pauses)])
        cuts: List[CutPoint] = []
        cut_scene: List[int] = []
        for start, end, scene, pauses in merged:
            for s, e in self._split(start, end, pauses):
                cuts.append(CutPoint(s, e))
                cut_scene.append(scene)
        return cuts, cut_scene

    def _split(self, start: float, end: float, pauses: List[float]) -> List[Tuple[float, float]]:
        """Split [start, end] into pieces of at most MAX_SLICE_S (the second study's prepare_cuts, 5.2)."""
        out: List[Tuple[float, float]] = []
        todo = [(start, end)]
        while todo:
            s, e = todo.pop(0)
            if e - s <= self.MAX_SLICE_S + 1e-9:
                out.append((s, e))
                continue
            inside = [p for p in pauses if s + self.SPLIT_MARGIN_S < p < e - self.SPLIT_MARGIN_S]
            middle = (s + e) / 2.0
            if inside:
                cut = min(inside, key=lambda p: abs(p - middle))
            elif e - (s + self.MAX_SLICE_S) >= self.MIN_SLICE_S:
                cut = s + self.MAX_SLICE_S
            else:
                # A cut at the limit would leave a sliver shorter than MIN_SLICE_S, which the program may
                # skip -- and a skipped slice shifts every later chunk_id. Both halves are then between
                # half the limit and the limit.
                cut = middle
            todo[0:0] = [(s, cut), (cut, e)]
        return out

    def _window_slices(self, scenes: List[_Scene], audio_s: float) -> Tuple[List[CutPoint], List[int]]:
        """Fixed windows of at most 30 s over the scenes (step 2b's second slice form).

        Where two scenes overlap, the overlap is cut at its middle, never merged (plan 3.4); each scene
        is then cut into equal windows of at most 30 s.
        """
        spans: List[list] = []
        for scene in sorted(scenes, key=lambda sc: sc.start):
            start, end = max(0.0, scene.start), min(audio_s, scene.end)
            if spans and start < spans[-1][1]:
                middle = (start + spans[-1][1]) / 2.0
                spans[-1][1] = middle
                start = middle
            if end - start >= self.MIN_SLICE_S:
                spans.append([start, end, scene.index])
        cuts: List[CutPoint] = []
        cut_scene: List[int] = []
        for start, end, scene in spans:
            if end - start < self.MIN_SLICE_S:
                continue
            n = max(1, math.ceil((end - start) / self.MAX_SLICE_S - 1e-9))
            step = (end - start) / n
            for i in range(n):
                cuts.append(CutPoint(start + i * step, end if i == n - 1 else start + (i + 1) * step))
                cut_scene.append(scene)
        return cuts, cut_scene

    # -----------------------------------------------------------------------------------------------
    # 6-7. One program run, then this pipeline's parser
    # -----------------------------------------------------------------------------------------------

    def _transcribe(self, program: install.ProgramChoice, model_path: Path, wav: Path, duration: float,
                    cuts: List[CutPoint], run_dir: Path, md: Dict) -> List[Tuple[int, CrispSegment]]:
        """Run the program once over the film and return its segments, each tied to its slice."""
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        install.link_or_copy(wav, run_dir / "audio.wav", what="a copy of the audio")
        write_vad_import(run_dir / "cuts.json", cuts, audio_seconds=duration)
        # A model path the program may not open is staged once for the pass, beside the films' run
        # folders (plan 1.7), and named relative to this film's folder.
        model_arg = install.stage_model(model_path, run_dir, stage_dir=run_dir.parent)
        request = CrispRunRequest(exe=program.exe, run_dir=run_dir, backend=self.model.backend, model=model_arg,
                                  extra_args=tuple(program.build.extra_args) + tuple(self.model.extra_args))
        logger.info(f"{self.LABEL}: CrispASR {program.build.label} transcribes {len(cuts)} slices "
                    f"with {self.model.label}")
        result = run_crispasr_json(request, on_progress=self._progress_logger())
        md["stages"]["crispasr"] = {"outcome": result.outcome.value, "exit_code": result.exit_code,
                                    "seconds": round(result.seconds, 1), "warnings": result.warning_count,
                                    "log": str(result.log_path)}
        log_hint = (f"; its log is {result.log_path}" if self.keep_temp_files
                    else "; run with --keep-temp to keep its log")

        if result.download_lines:
            # The program fetched something by itself: WhisperJAV's command line missed a switch (plan 4.3).
            self.notes.append(f"{self.LABEL}: the CrispASR program downloaded a file by itself "
                              f"({result.download_lines[0]}){log_hint}")
        if result.warning_count:
            self.notes.append(f"{self.LABEL}: the CrispASR program printed {result.warning_count} "
                              f"warning(s){log_hint}")

        if result.outcome == CrispOutcome.NO_SPEECH:
            logger.info(f"{self.LABEL}: the CrispASR program found no speech in the slices")
            return []
        if result.outcome == CrispOutcome.CRASHED and result.json_path is not None:
            self.notes.append(f"{self.LABEL}: the CrispASR program ended with {result.meaning} after writing "
                              f"its complete output; that output was used{log_hint}")
        elif result.outcome != CrispOutcome.OK:
            logger.error(f"{self.LABEL}: the CrispASR program's last output:\n{result.stderr_tail}")
            if result.outcome == CrispOutcome.NO_OUTPUT:
                raise CrispASRRunError(f"the CrispASR program wrote no output{log_hint}")
            raise CrispASRRunError(f"the CrispASR program failed: {result.meaning}{log_hint}")

        doc = read_ojf(result.json_path)
        if doc is None:
            raise CrispASRRunError(f"the CrispASR program's output could not be read{log_hint}")
        transcript = parse_ojf(doc)
        mismatch = header_mismatch(transcript, self.model.backend, model_arg, language=request.language)
        if mismatch:
            raise CrispASRRunError(f"the CrispASR output is not this run's: {mismatch}")
        placement = place_in_slices(transcript.segments, cuts)
        with_text = {i for i, seg in placement.placed if seg.text.strip()}
        no_text = len(cuts) - len(with_text)
        md["stages"]["crispasr"].update({"segments": len(transcript.segments), "retied": placement.retied,
                                         "dropped": placement.dropped, "malformed": transcript.malformed,
                                         "slices_without_text": no_text})
        if no_text:
            # Plan 2.6: slices with no text are counted in a note (an empty segment is the program's
            # answer to silence, runs M1 and M2).
            self.notes.append(f"{self.LABEL}: {no_text} of {len(cuts)} slice(s) gave no text")
        if placement.retied:
            self.notes.append(f"{self.LABEL}: {placement.retied} segment(s) named the wrong slice and were "
                              f"placed by their time")
        if placement.dropped:
            self.degradations.append(f"{self.LABEL}: {placement.dropped} segment(s) of the CrispASR output lay "
                                     f"outside every slice and were dropped")
        return placement.placed

    def _progress_logger(self) -> Callable[[int, int], None]:
        """Console lines at every tenth of the slices: in an Ensemble pass, the log is the progress display."""
        state = {"next": 10}

        def report(done: int, total: int) -> None:
            percent = int(done * 100 / max(1, total))
            if percent >= state["next"] or done >= total:
                logger.info(f"{self.LABEL}: {done} of {total} slices transcribed ({percent}%)")
                state["next"] = (percent // 10 + 1) * 10

        return report

    def _word_dicts(self, seg: CrispSegment, cut: CutPoint) -> Tuple[List[Dict], int]:
        """This pipeline's parser (plan 3.5): one segment's tokens -> word dicts in film time.

        Cohere's "words" list holds the whole segment as one word, so its tokens are read instead:
        - blank tokens are dropped;
        - a zero-length token is folded into the next one (CrispASR gives many: 24 of 54 in run S1);
        - times are clamped to the slice.
        A segment with text but no usable tokens is shared out over its own times (split_frame_to_words),
        so no text is lost. Returns the word dicts and how many times had to be moved into the slice.
        """
        clamped = 0

        def clamp(t: float) -> float:
            nonlocal clamped
            c = min(max(t, cut.start), cut.end)
            if abs(c - t) > self.CLAMP_TOLERANCE_S:
                clamped += 1
            return c

        words: List[Dict] = []
        tokens = [t for t in seg.tokens if t.text.strip()]
        pending = ""
        for i, t in enumerate(tokens):
            if t.end <= t.start and i + 1 < len(tokens):
                pending += t.text
                continue
            start = clamp(t.start)
            words.append({"word": pending + t.text, "start": start, "end": max(start, clamp(t.end))})
            pending = ""
        if not words and seg.text.strip():
            from whisperjav.modules.subtitle_pipeline.reconstruction import split_frame_to_words

            start = clamp(seg.start)
            words = split_frame_to_words(seg.text.strip(), start, max(start, clamp(seg.end)))
        return words, clamped

    # -----------------------------------------------------------------------------------------------
    # 8. This pipeline's line building, per scene, then stitching
    # -----------------------------------------------------------------------------------------------

    def _build_lines(self, scenes: List[_Scene], cuts: List[CutPoint], cut_scene: List[int],
                     segments: List[Tuple[int, CrispSegment]], basename: str, md: Dict) -> Path:
        """Subtitle lines scene by scene, in scene-relative times; then the stitched SRT of the film.

        Each slice belongs to one scene (the scene of its earliest speech group), so its words are
        counted once even where neighbouring scenes overlap. A slice merged across a scene edge keeps
        its words in the earlier scene, past that scene's end.
        """
        from whisperjav.modules.srt_stitching import SRTStitcher

        by_scene: Dict[int, List[Tuple[CrispSegment, CutPoint]]] = {}
        for slice_index, seg in segments:
            by_scene.setdefault(cut_scene[slice_index], []).append((seg, cuts[slice_index]))
        scene_of = {sc.index: sc for sc in scenes}

        scene_srts_dir = self.temp_dir / "scene_srts"
        scene_srts_dir.mkdir(parents=True, exist_ok=True)
        scene_srt_info: List[Tuple[Path, float]] = []
        clamped_total = 0
        for scene_index in sorted(by_scene):
            scene = scene_of[scene_index]
            scene_srt = scene_srts_dir / f"{basename}_scene_{scene_index:04d}.srt"
            written, clamped = self._write_regrouped_lines(by_scene[scene_index], scene, scene_srt)
            clamped_total += clamped
            if written:
                scene_srt_info.append((scene_srt, scene.start))

        md["stages"]["lines"] = {"mode": self.line_mode, "scenes_with_lines": len(scene_srt_info),
                                 "clamped_times": clamped_total}
        if clamped_total:
            self.notes.append(f"{self.LABEL}: {clamped_total} word time(s) lay outside their slice and were "
                              f"moved to its edge")

        stitched = self.temp_dir / f"{basename}_stitched.srt"
        if scene_srt_info:
            SRTStitcher().stitch(scene_srt_info, stitched)
        else:
            stitched.write_text("", encoding="utf-8")
        return stitched

    def _write_regrouped_lines(self, items: List[Tuple[CrispSegment, CutPoint]], scene: _Scene,
                               scene_srt: Path) -> Tuple[bool, int]:
        """REGROUP_JAV over the scene's words, with the engine's own times (suppress_silence off)."""
        from whisperjav.modules.subtitle_pipeline.reconstruction import REGROUP_JAV, reconstruct_from_words

        words: List[Dict] = []
        clamped = 0
        for seg, cut in sorted(items, key=lambda item: item[0].start):
            seg_words, n = self._word_dicts(seg, cut)
            clamped += n
            for w in seg_words:
                words.append({"word": w["word"], "start": w["start"] - scene.start, "end": w["end"] - scene.start})
        if not words:
            return False, clamped
        # stable-ts refuses words whose starts go backwards. Sorting them would reorder the text -- and
        # Cohere's estimated token times can step back -- so the engine's order is kept and only a start
        # that would precede the one before it is held back.
        previous = 0.0
        for w in words:
            w["start"] = max(w["start"], previous)
            w["end"] = max(w["end"], w["start"])
            previous = w["start"]
        try:
            result = reconstruct_from_words(words, scene.path, suppress_silence=False, regroup=REGROUP_JAV)
        except Exception as e:
            # As the subtitle orchestrator does: one retry without regrouping.
            logger.warning(f"{self.LABEL}: line building failed for scene {scene.index} ({e}); "
                           f"retrying without regrouping")
            result = reconstruct_from_words(words, scene.path, suppress_silence=False, regroup=False)
        if not getattr(result, "segments", None):
            return False, clamped
        result.to_srt_vtt(str(scene_srt), word_level=False, segment_level=True, strip=True)
        return scene_srt.exists() and scene_srt.stat().st_size > 0, clamped

    # -----------------------------------------------------------------------------------------------
    # 9. This pipeline's post-processing (plan 3.6)
    # -----------------------------------------------------------------------------------------------

    def _postprocess(self, stitched: Path) -> Dict[str, int]:
        """Text-only filters, then the scene-overlap resolver; each step chosen by self.postprocess."""
        stats: Dict[str, int] = {}
        if _count_srt(stitched) == 0:
            return stats
        if "nonverbal" in self.postprocess:
            from whisperjav.modules.subtitle_pipeline.cleaners.nonverbal_line_filter import NonverbalLineFilter

            r = NonverbalLineFilter().filter_srt_file(stitched)
            stats["nonverbal_dropped"] = r.get("dropped_nonverbal", 0)
            stats["empty_removed"] = stats.get("empty_removed", 0) + r.get("dropped_empty", 0)
        if "nonlinguistic" in self.postprocess:
            from whisperjav.modules.subtitle_pipeline.cleaners.nonlinguistic_utterance_filter import (
                NonlinguisticUtteranceFilter)

            r = NonlinguisticUtteranceFilter().filter_srt_file(stitched)
            stats["nonlinguistic_dropped"] = r.get("dropped_nonlinguistic", 0)
            stats["empty_removed"] = stats.get("empty_removed", 0) + r.get("dropped_empty", 0)
        if "overlap" in self.postprocess:
            from whisperjav.modules.subtitle_pipeline.cleaners.scene_overlap_resolver import SceneOverlapResolver

            r = SceneOverlapResolver().resolve_srt_file(stitched)
            stats["overlap_nested_dropped"] = r.get("dropped_nested", 0)
            stats["overlap_shifted"] = r.get("shifted_starts", 0)
        return stats
