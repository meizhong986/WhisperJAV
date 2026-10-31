#!/usr/bin/env python3
"""MOSS-CrispASR: OpenMOSS's MOSS-Transcribe-Diarize 0.9B, run through CrispASR (1.10.1).

An Ensemble Mode pass pipeline only (W118 2): ``whisperjav --ensemble --pass1-pipeline moss-crispasr``.
Planned for 1.10.2 (W113, W114), brought into 1.10.1 by the owner (W401: "please add moss-transcriber"). Models:
the full-size file (f16, the default, W400) or q8_0, from cstr/MOSS-Transcribe-Diarize-GGUF.

Why the Diarize model (W401, pre-approved choices of step 2): MOSS-Transcribe-preview-2B, the other MOSS-Transcribe
model, is English only (its model card: "Language: English"); on the Netflix reference clip it wrote English for
81 of 82 VAD groups (2026-10-31). MOSS-Transcribe-Diarize transcribes 50+ languages, Japanese among them. No
diarization in 1.10.x (W113 X7): its speaker labels are not used ("Speaker labels may be filtered out or omitted").
Not solved yet: the labels ("[S01]") can stay inside the text, and on non-speech it can repeat a line to its limit.

The chain is shared with the other CrispASR pipelines (whisperjav/pipelines/crispasr_pipeline.py). What is its
own (W401, pre-approved choices of step 2): Cohere-CrispASR's settings -- FireRedVAD at balanced, VAD groups of
at most 3.0 s, gap 0.3 s, pads 100/100 ms; each VAD group's text timed by the CrispASR aligner (wav2vec2-ja) and
cut at sentence-end marks, a group the aligner gives nothing for one subtitle at its edges; the audio analytics.
"""

from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline, EngineRow


class MossCrispASRPipeline(CrispASRPipeline):
    """MOSS-CrispASR, an Ensemble pass pipeline."""

    PIPELINE_NAME = "moss-crispasr"
    LABEL = "MOSS-CrispASR"
    ENGINE = EngineRow(line_rule="group", audio_analytics=True, aligner="wav2vec2-ja")
