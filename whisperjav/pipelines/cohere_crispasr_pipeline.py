#!/usr/bin/env python3
"""Cohere-CrispASR: a Japanese fine-tune of Cohere Transcribe, run through CrispASR (1.10.0).

An Ensemble Mode pass pipeline only (W118 2): ``whisperjav --ensemble --pass1-pipeline cohere-crispasr``.
It has no --mode value and no entry in the Transcription Mode tab. Its model is one of two Japanese
fine-tunes (W118 4): cohere-asr-ja-v0.1 by default (W129 1), or cohere-asr-ja (--pass1-model).

The chain is shared with Parakeet-CrispASR (whisperjav/pipelines/crispasr_pipeline.py; code structure
S2, W175). What is Cohere's own (route B design 6):
  - the speech segmenter settings of the qwen pipeline with Qwen3-ASR: FireRedVAD at balanced by
    default, its threshold by sensitivity, VAD groups of at most 3.0 s, gap 0.3 s, longest speech
    segment 4.0 s, pads 100/100 ms (W199 DIR1, W202 3-4);
  - each VAD group becomes one subtitle with the group's start and end, where the speech segmenter
    found the speech (option C1, W175); Cohere's estimated token times are kept, not used for cutting;
  - the audio analytics, as the qwen pipeline (W202 5b); no CPS start retimer (W202 5c).
"""

from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline, EngineRow


class CohereCrispASRPipeline(CrispASRPipeline):
    """Cohere-CrispASR, an Ensemble pass pipeline."""

    PIPELINE_NAME = "cohere-crispasr"
    LABEL = "Cohere-CrispASR"
    ENGINE = EngineRow(line_rule="group", audio_analytics=True)
