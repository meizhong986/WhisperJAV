#!/usr/bin/env python3
"""MOSS-CrispASR: OpenMOSS's MOSS-Transcribe-Diarize 0.9B, run through CrispASR (1.10.1).

An Ensemble Mode pass pipeline only (W118 2): ``whisperjav --ensemble --pass1-pipeline moss-crispasr``.
Planned for 1.10.2 (W113, W114), brought into 1.10.1 by the owner (W401: "please add moss-transcriber"). Models:
the full-size file (f16, the default, W400) or q8_0, from cstr/MOSS-Transcribe-Diarize-GGUF.

Why the Diarize model (W401, pre-approved choices of step 2): MOSS-Transcribe-preview-2B, the other MOSS-Transcribe
model, is English only (its model card: "Language: English"); on the Netflix reference clip it wrote English for
81 of 82 VAD groups (2026-10-31). MOSS-Transcribe-Diarize transcribes 50+ languages, Japanese among them. No
diarization in 1.10.x (W113 X7): its speaker labels are not used.

Built after the owner's reference script (W423-W428; design docs/plans/design/2026-11-01-moss-crispasr-reference-way-
design.md, section 9.4). His findings (W425): MOSS works best with longer audio that starts with speech, and
relevant hotwords help; clips of 70 s and more were transcribed very well. So, through the shared chain
(whisperjav/pipelines/crispasr_pipeline.py):

  - scenes: the semantic scene detector, scenes of 90-240 s (W482; crispasr/pass_settings.py; before it silero
    scenes up to 420 s, W426 4);
  - the speech segmenter FireRedVAD builds the VAD groups MOSS hears, one per call, each starting at speech: speech
    segments less than 5 s apart are one group, a group is at most 240 s, a short group followed by a longer silence
    stays alone (no shortest group), at every sensitivity (W482, W428 I1 2); all in the pass's Customize Segmenter
    tab. Before W482: Silero VAD v6.2 at 0.7, groups of 70-120 s (W426 1-3);
  - the reference's instruction (English, "Language hint: Japanese.", the hotwords) through CrispASR's set_ask,
    adjustable in the pass's MOSS tab or --passN-params (W428 I1 4; crispasr/moss_prompt.py);
  - a token limit per call by the stretch's length, at the reference's rate (65,536 tokens per 50 minutes, about
    21.8 a second; owner W406 i1): without it the model can repeat a line to its limit on non-speech;
  - the model's own segment times (W411): each segment MOSS writes is one subtitle (the "segments" line rule),
    speaker labels dropped; --passN-word-timestamps aligner re-times them with the CrispASR aligner on request;
  - post-processing: the three filters of 1.10.0 only (owner W421 i3).

The reference's sampling (--do-sample --temperature 0.7) is the preset since 1.10.1 with crispasr 0.8.50: sampling
on, temperature 0.7, top-k 50, seed 42; greedy and beam size are choices in the MOSS tab (owner W492 M1, W501, W502;
crispasr/moss_prompt.py). Measured on 2026-11-04 (docs/plans/design/2026-11-03-moss-m1-m3-research.md section 6):
on loop-prone JAV audio sampling cut the call time by 74-85 % against greedy; on the Netflix clip its text was
0.015-0.050 worse in plain CER.
"""

from whisperjav.crispasr import cleanup as crispasr_cleanup
from whisperjav.crispasr.moss_prompt import TOKENS_PER_SECOND, MossSettings
from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline, EngineRow


class MossCrispASRPipeline(CrispASRPipeline):
    """MOSS-CrispASR, an Ensemble pass pipeline."""

    PIPELINE_NAME = "moss-crispasr"
    LABEL = "MOSS-CrispASR"
    MODEL_SETTINGS = MossSettings
    # Post-processing: the three filters of 1.10.0 only, until the pipeline is hardened (owner W421 i3).
    ENGINE = EngineRow(line_rule="segments", audio_analytics=True, tokens_per_second=TOKENS_PER_SECOND,
                       cleanup=crispasr_cleanup.EARLIER_STAGES)
