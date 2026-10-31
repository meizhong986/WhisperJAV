#!/usr/bin/env python3
"""Kotoba-CrispASR: Kotoba-Whisper v2.0, a Japanese distilled Whisper, run through CrispASR (1.10.1).

An Ensemble Mode pass pipeline only (W118 2): ``whisperjav --ensemble --pass1-pipeline kotoba-crispasr``.
Asked for by the owner in W397, kept in W401. Models: the full-size whisper.cpp file (f16, the default, W400)
or q5_0, from kotoba-tech/kotoba-whisper-v2.0-ggml.

The chain is shared with the other CrispASR pipelines (whisperjav/pipelines/crispasr_pipeline.py). What is
Kotoba's own (W401, pre-approved choices of step 2): Cohere-CrispASR's settings -- FireRedVAD at balanced, VAD
groups of at most 3.0 s, gap 0.3 s, pads 100/100 ms; each VAD group's text timed by the CrispASR aligner
(wav2vec2-ja) and cut at sentence-end marks, a group the aligner gives nothing for one subtitle at its edges;
the audio analytics.
"""

from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline, EngineRow


class KotobaCrispASRPipeline(CrispASRPipeline):
    """Kotoba-CrispASR, an Ensemble pass pipeline."""

    PIPELINE_NAME = "kotoba-crispasr"
    LABEL = "Kotoba-CrispASR"
    ENGINE = EngineRow(line_rule="group", audio_analytics=True, aligner="wav2vec2-ja")
