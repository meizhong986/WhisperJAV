#!/usr/bin/env python3
"""ReazonSpeech-CrispASR: ReazonSpeech's Japanese NeMo transducer (reazonspeech-nemo-v2), run through CrispASR (1.10.1).

An Ensemble Mode pass pipeline only (W118 2): ``whisperjav --ensemble --pass1-pipeline reazonspeech-crispasr``.
Asked for by the owner in W397, kept in W401. Models: the full-size file (f16, the default, W400) or q8_0.

The chain is shared with the other CrispASR pipelines (whisperjav/pipelines/crispasr_pipeline.py). What is
ReazonSpeech's own (W401, pre-approved choices of step 2):
  - the speech segmenter settings of Parakeet-CrispASR (TEN VAD at aggressive by default, VAD groups of at most
    6 s): CrispASR opens the model as the parakeet family, and both are transducers;
  - its text is timed by the CrispASR aligner (wav2vec2-ja) and cut as Cohere-CrispASR's is: its tokens carry
    one time each (start = end, measured 2026-10-31), so Parakeet's token rule would end each subtitle where
    its last token starts;
  - the audio analytics, as the other CrispASR pipelines.
"""

from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline, EngineRow


class ReazonSpeechCrispASRPipeline(CrispASRPipeline):
    """ReazonSpeech-CrispASR, an Ensemble pass pipeline."""

    PIPELINE_NAME = "reazonspeech-crispasr"
    LABEL = "ReazonSpeech-CrispASR"
    ENGINE = EngineRow(line_rule="group", audio_analytics=True, aligner="wav2vec2-ja")
