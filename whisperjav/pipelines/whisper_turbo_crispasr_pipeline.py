#!/usr/bin/env python3
"""Whisper-Turbo-CrispASR: OpenAI Whisper large-v3-turbo, run through CrispASR (1.10.1).

An Ensemble Mode pass pipeline only (W118 2): ``whisperjav --ensemble --pass1-pipeline whisper-turbo-crispasr``.
Asked for by the owner in W397, kept in W401. Models: the full-size whisper.cpp file (f16, the default, W400)
or q8_0, from ggerganov/whisper.cpp.

The chain is shared with the other CrispASR pipelines (whisperjav/pipelines/crispasr_pipeline.py). What is its
own (W401, pre-approved choices of step 2): Cohere-CrispASR's settings -- FireRedVAD at balanced, VAD groups of
at most 3.0 s, gap 0.3 s, pads 100/100 ms; each VAD group's text timed by the CrispASR aligner (wav2vec2-ja) and
cut at sentence-end marks, a group the aligner gives nothing for one subtitle at its edges; the audio analytics.
"""

from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline, EngineRow


class WhisperTurboCrispASRPipeline(CrispASRPipeline):
    """Whisper-Turbo-CrispASR, an Ensemble pass pipeline."""

    PIPELINE_NAME = "whisper-turbo-crispasr"
    LABEL = "Whisper-Turbo-CrispASR"
    ENGINE = EngineRow(line_rule="group", audio_analytics=True, aligner="wav2vec2-ja")
