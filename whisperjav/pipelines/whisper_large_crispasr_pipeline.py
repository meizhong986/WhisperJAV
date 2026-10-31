#!/usr/bin/env python3
"""Whisper-Large-CrispASR: OpenAI Whisper large-v2 or large-v3, run through CrispASR (1.10.1).

An Ensemble Mode pass pipeline only (W118 2): ``whisperjav --ensemble --pass1-pipeline whisper-large-crispasr``.
Asked for by the owner in W397 ("whisper-large-v3 / v2"), kept in W401. Models, from ggerganov/whisper.cpp:
large-v2 full size (f16, the default) or q8_0; large-v3 full size (f16) or q5_0 (the repository has no large-v3
q8_0). large-v2 is the default: on the Netflix reference clip it wrote fewer invented lines and had the lower
character error rate of the two (2026-10-31; the manifest's PIPELINE_DEFAULT_MODEL).

The chain is shared with the other CrispASR pipelines (whisperjav/pipelines/crispasr_pipeline.py). What is its
own (W401, pre-approved choices of step 2): Cohere-CrispASR's settings -- FireRedVAD at balanced, VAD groups of
at most 3.0 s, gap 0.3 s, pads 100/100 ms; each VAD group's text timed by the CrispASR aligner (wav2vec2-ja) and
cut at sentence-end marks, a group the aligner gives nothing for one subtitle at its edges; the audio analytics.
"""

from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline, EngineRow


class WhisperLargeCrispASRPipeline(CrispASRPipeline):
    """Whisper-Large-CrispASR, an Ensemble pass pipeline."""

    PIPELINE_NAME = "whisper-large-crispasr"
    LABEL = "Whisper-Large-CrispASR"
    ENGINE = EngineRow(line_rule="group", audio_analytics=True, aligner="wav2vec2-ja")
