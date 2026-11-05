"""R-SCN as a WhisperJAV scene-detector backend (``rscn``). Design: docs/plans/design/2026-11-02-rscn-design.md.

detect_scenes(audio_path, output_dir, media_basename, media=None):
  1. reads the extracted WAV (its own rate: 16 kHz, or 48 kHz with a speech enhancer);
  2. measures the sound on a 16 kHz copy (analysis.py);
  3. runs the speech measure on that copy, in pieces (speech.py; first default FireRedVAD);
  4. with a picture (settings.picture and the media record), finds black stretches (picture.py);
  5. plans the scenes (partition.py), checks that every sample is in exactly one scene;
  6. writes each scene's WAV at the WAV's own rate and a run record <basename>_rscn.json in output_dir.
A failure raises SceneDetectionError: the file fails with a plain message, never a silent fall-back (v6 11.6). The
constructor only checks the settings (a ValueError there is not caught by the factory's fall-back to auditok).
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from whisperjav.utils.logger import logger

from ..base import SceneDetectionError, SceneDetectionResult, SceneInfo
from ..utils import save_scene_wav
from . import analysis, partition, picture
from .settings import PREFIX, RSCNSettings
from .speech import SpeechMeasure, SpeechMeasureError, create_speech_measure

VERSION = "rscn-whisperjav-1"


class RSCNSceneDetector:
    """Scenes cut where the sound's characteristics change, within the pipeline's length bounds (owner W431, W450,
    W451). Settings: the pipeline's preset with the pass's ``rscn_`` Customize values."""

    def __init__(self, pipeline: Optional[str] = None, settings: Optional[RSCNSettings] = None,
                 speech_measure: Optional[SpeechMeasure] = None, **kwargs: Any):
        self.pipeline = pipeline
        self.settings = settings or RSCNSettings.for_pipeline(pipeline, kwargs)
        # Values for the other detectors (the Customize Scene tab's keys) are not R-SCN's: named once, not used.
        ignored = sorted(k for k in kwargs if not str(k).startswith(PREFIX))
        if ignored:
            logger.info(f"R-SCN does not use: {', '.join(ignored)}")
        self._speech_measure = speech_measure  # injected (tests, or a caller with its own); else built per settings

    @property
    def name(self) -> str:
        return "rscn"

    @property
    def display_name(self) -> str:
        return "R-SCN (changes in the sound, picture breaks)"

    def detect_scenes(self, audio_path: Path, output_dir: Path, media_basename: str, media: Any = None,
                      **kwargs: Any) -> SceneDetectionResult:
        started = time.time()
        s = self.settings
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        try:
            import soundfile as sf
            # 16-bit WAVs (WhisperJAV's extraction) are read as 16-bit: half the memory of float32 for a long film;
            # exact zeros stay exact.
            subtype = sf.info(str(audio_path)).subtype
            samples, rate = sf.read(str(audio_path), dtype="int16" if subtype == "PCM_16" else "float32",
                                    always_2d=False)
        except Exception as e:
            raise SceneDetectionError(f"R-SCN could not read the audio {audio_path.name}: {e}") from e
        if samples.ndim > 1:
            samples = samples.mean(axis=1).astype(np.float32)
        if samples.dtype == np.int16:
            to_float = lambda x: x.astype(np.float32) / 32768.0  # noqa: E731
        else:
            to_float = lambda x: x  # noqa: E731
        if len(samples) == 0:
            raise SceneDetectionError(f"R-SCN: the audio {audio_path.name} is empty")
        duration = len(samples) / rate
        timings: Dict[str, float] = {}

        t = time.time()
        analysis_audio = analysis.to_analysis_rate(to_float(samples), rate, s.analysis_rate)
        m = analysis.measures(analysis_audio, s)
        timings["measures_s"] = time.time() - t

        speech_record: Dict[str, Any] = {"backend": "none"}
        stretches = None
        measure = self._speech_measure
        own_measure = False
        if measure is None:
            measure = create_speech_measure(s.speech_measure_name, s.speech_measure_piece_s, s.speech_measure_params,
                                            overlap_s=s.speech_measure_overlap_s)
            own_measure = measure is not None
        if measure is not None:
            t = time.time()
            try:
                speech_audio = analysis_audio if s.analysis_rate == 16000 else \
                    analysis.to_analysis_rate(to_float(samples), rate, 16000)
                result = measure.measure(speech_audio)
            except SpeechMeasureError as e:
                raise SceneDetectionError(f"R-SCN: {e}. Set rscn_speech_measure to another speech segmenter "
                                          f"backend, or to none.") from e
            finally:
                if own_measure:
                    measure.close()
            stretches = result.stretches
            speech_record = result.as_dict()
            timings["speech_measure_s"] = time.time() - t

        black = []
        try:
            use_picture, why = picture.decide(s.picture_mode, media)
            if use_picture:
                black, timings["picture_s"] = picture.black_stretches(media, s, duration)
        except picture.PictureError as e:
            if s.picture_mode == "on":
                raise SceneDetectionError(f"R-SCN: {e}") from e
            # auto (proposal): a picture that cannot be read does not cost the file its subtitles.
            logger.warning(f"R-SCN: {e}; the scenes are found from the sound alone for this file.")
            use_picture, why = False, f"picture analysis failed: {e}"
        if use_picture:
            logger.info(f"R-SCN: {len(black)} black picture stretch(es) of at least {s.black_min_s:g} s found.")
        elif s.picture_mode == "auto" and media is not None:
            logger.info(f"R-SCN: {why}.")

        try:
            plans, record = partition.plan(samples, rate, s, m, stretches, black)
        except (AssertionError, ValueError) as e:
            raise SceneDetectionError(f"R-SCN could not divide {audio_path.name}: {e}") from e

        output_dir.mkdir(parents=True, exist_ok=True)
        scenes = []
        for i, p in enumerate(plans):
            path = save_scene_wav(samples[p.start_sample:p.end_sample], rate, i, output_dir, media_basename)
            scenes.append(SceneInfo(start_sec=p.start_sample / rate, end_sec=p.end_sample / rate, scene_path=path,
                                    detection_pass=0, metadata=dict(p.marks)))
        elapsed = time.time() - started
        audio_scenes = [sc for sc in scenes if sc.metadata.get("rscn_kind") == "audio"]
        lengths = [sc.duration_sec for sc in audio_scenes]
        forced = sum(1 for p in plans if p.end_cut.reason in ("to_fit_length", "equal_split"))
        logger.info(f"R-SCN: {len(scenes)} scenes over {duration:.1f} s"
                    + (f" (audio scenes {min(lengths):.1f}-{max(lengths):.1f} s)" if lengths else "")
                    + f"; {len(record['digital_silence'])} digital silence, "
                    f"{sum(1 for b in black if b.confirmed)} picture break(s), {forced} cut(s) to fit the length; "
                    f"{elapsed:.1f} s")

        run_record = {
            "schema": VERSION,
            "audio": {"path": str(audio_path), "rate": int(rate), "samples": int(len(samples)),
                      "duration_s": round(duration, 6)},
            "pipeline": self.pipeline,
            "settings": s.as_dict(),
            "media": media.as_dict() if hasattr(media, "as_dict") else None,
            "picture": {"mode": s.picture_mode, "used": use_picture, "why": why,
                        "black": record.get("black", [])},
            "speech_measure": speech_record,
            "digital_silence": record["digital_silence"],
            "regions": record["regions"],
            "acts": record["acts"],
            "scenes": [{"index": i, "start_sample": p.start_sample, "end_sample": p.end_sample,
                        "start_s": round(p.start_sample / rate, 6), "end_s": round(p.end_sample / rate, 6),
                        "duration_s": round((p.end_sample - p.start_sample) / rate, 6),
                        "end_cut_at_s": round(p.end_cut.at_s, 3), **p.marks}
                       for i, p in enumerate(plans)],
            "validation": {"every_sample_in_one_scene": True, "samples": int(len(samples)),
                           "scene_samples": int(sum(p.end_sample - p.start_sample for p in plans))},
            "timings_s": {k: round(v, 3) for k, v in {**timings, "total_s": elapsed}.items()},
        }
        record_path = output_dir / f"{media_basename}_rscn.json"
        try:
            record_path.write_text(json.dumps(run_record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        except OSError as e:
            logger.warning(f"R-SCN: the run record could not be written ({e})")
        return SceneDetectionResult(scenes=scenes, method=self.name, audio_duration_sec=duration,
                                    parameters={**s.as_dict(), "record": str(record_path)},
                                    processing_time_sec=elapsed)

    def cleanup(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"RSCNSceneDetector(pipeline={self.pipeline!r})"
