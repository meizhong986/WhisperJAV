#!/usr/bin/env python3
"""Parakeet-CrispASR: NVIDIA's Japanese Parakeet TDT, run through CrispASR (1.10.0).

An Ensemble Mode pass pipeline only (W118 2): ``whisperjav --ensemble --pass1-pipeline parakeet-crispasr``.
It has no --mode value and no entry in the Transcription Mode tab.

The chain is shared with Cohere-CrispASR (whisperjav/pipelines/crispasr_pipeline.py; code structure S2,
W175). What is Parakeet's own (route B design 6):
  - the speech segmenter settings of the fidelity pipeline: WhisperSeg at balanced by default, its
    preset by sensitivity, the longest VAD group and speech segment 6 s (W199 DIR1, W202 1-2);
  - each VAD group's tokens become one or more subtitles, cut only at natural cues -- sentence-end
    marks, pauses, 、 and spaces followed by a pause -- with Parakeet's own token times (option P2,
    W175; W203, W204);
  - no audio analytics, as fidelity.
"""

from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline, EngineRow


class ParakeetCrispASRPipeline(CrispASRPipeline):
    """Parakeet-CrispASR, an Ensemble pass pipeline."""

    PIPELINE_NAME = "parakeet-crispasr"
    LABEL = "Parakeet-CrispASR"
    ENGINE = EngineRow(line_rule="tokens", audio_analytics=False)
