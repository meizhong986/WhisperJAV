#!/usr/bin/env python3
"""MOSS-CrispASR: OpenMOSS's MOSS-Transcribe-Diarize 0.9B, run through CrispASR (1.10.1).

An Ensemble Mode pass pipeline only (W118 2): ``whisperjav --ensemble --pass1-pipeline moss-crispasr``.
Planned for 1.10.2 (W113, W114), brought into 1.10.1 by the owner (W401: "please add moss-transcriber"). Models:
the full-size file (f16, the default, W400) or q8_0, from cstr/MOSS-Transcribe-Diarize-GGUF.

Why the Diarize model (W401, pre-approved choices of step 2): MOSS-Transcribe-preview-2B, the other MOSS-Transcribe
model, is English only (its model card: "Language: English"); on the Netflix reference clip it wrote English for
81 of 82 VAD groups (2026-10-31). MOSS-Transcribe-Diarize transcribes 50+ languages, Japanese among them. No
diarization in 1.10.x (W113 X7): its speaker labels are not used ("Speaker labels may be filtered out or omitted").
Not solved yet: the labels ("[S01]") can stay inside the text.

Token limit (owner W406 i1): each call may write at most 65,536 / 3,000 tokens per second of the VAD group's audio
(about 21.8; at least 32), the rate of the owner's reference script (65,536 tokens per 50-minute chunk). Without it
the model repeated a line to its own limit on non-speech (one 4 s group: 517 characters). On the Netflix reference
clip: character error rate 1.506 -> 0.823, longest call 30.9 s -> 2.6 s (D:\\WJ110_measure\\moss_ref_2026-10-31).

The chain is shared with the other CrispASR pipelines (whisperjav/pipelines/crispasr_pipeline.py). What is its
own (W401, pre-approved choices of step 2): Cohere-CrispASR's settings -- FireRedVAD at balanced, VAD groups of
at most 3.0 s, gap 0.3 s, pads 100/100 ms; each VAD group's text timed by the CrispASR aligner (wav2vec2-ja) and
cut at sentence-end marks, a group the aligner gives nothing for one subtitle at its edges; the audio analytics.
"""

from whisperjav.crispasr import cleanup as crispasr_cleanup
from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline, EngineRow


class MossCrispASRPipeline(CrispASRPipeline):
    """MOSS-CrispASR, an Ensemble pass pipeline."""

    PIPELINE_NAME = "moss-crispasr"
    LABEL = "MOSS-CrispASR"
    # Post-processing: the three filters of 1.10.0 only, until the pipeline is hardened (owner W421 i3).
    ENGINE = EngineRow(line_rule="group", audio_analytics=True, aligner="wav2vec2-ja",
                       tokens_per_second=65536 / 3000, cleanup=crispasr_cleanup.EARLIER_STAGES)
