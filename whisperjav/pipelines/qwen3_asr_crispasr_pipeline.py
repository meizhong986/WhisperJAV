#!/usr/bin/env python3
"""Qwen3-ASR-CrispASR: Qwen3-ASR-1.7B as a GGUF file, run through CrispASR (1.10.2).

An Ensemble Mode pass pipeline only (W118 2): ``whisperjav --ensemble --pass1-pipeline qwen3-asr-crispasr``.
Added by the owner in W574, experimental, on the findings of the GGUF model test tool (tools/crispasr_model_test.py,
owner W572-W573; docs/plans/design/2026-11-06-haas-crispasr-first-measurements.md section 10). It runs the model that
ChronosJAV's Qwen3-ASR runs through PyTorch (Qwen/Qwen3-ASR-1.7B), from the GGUF file cstr/qwen3-asr-1.7b-GGUF f16. It
is not part of ChronosJAV (W114): no timing model, no step-down, no Phase 8.

The chain is shared with the other CrispASR pipelines (whisperjav/pipelines/crispasr_pipeline.py). What is its own is
what the test tool found and measured it with:
  - CrispASR 0.8.50 opens it as the qwen3 family; its words and tokens carry no times and each call of up to 30 s
    gives one segment over the whole call, so it is built as Cohere-CrispASR is: ChronosJAV's text cleaner on each
    VAD group's text, then the CrispASR aligner (wav2vec2-ja) times it, cut at sentence-end marks; a VAD group the
    aligner gives nothing for stays one subtitle at the VAD group's edges (the tool's proposed row);
  - Parakeet-CrispASR's speech segmenter settings (TEN VAD at aggressive by default, VAD groups of at most 6 s), on
    whose VAD groups the tool measured it; other settings were not measured;
  - the GGUF clean-up and the audio analytics, as the other CrispASR pipelines.
With --passN-word-timestamps model the aligner is off: each VAD group is one subtitle at its edges (the model gives
no times of its own), which is what the tool measured.
"""

from whisperjav.crispasr import cleanup as crispasr_cleanup
from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline, EngineRow


class Qwen3AsrCrispASRPipeline(CrispASRPipeline):
    """Qwen3-ASR-CrispASR, an Ensemble pass pipeline."""

    PIPELINE_NAME = "qwen3-asr-crispasr"
    LABEL = "Qwen3-ASR-CrispASR"
    ENGINE = EngineRow(line_rule="group", audio_analytics=True, aligner="wav2vec2-ja",
                       text_cleanup=crispasr_cleanup.TEXT_STAGES)
    # --passN-word-timestamps model: no aligner and so no text cleaner (it runs only before the aligner); one subtitle
    # per VAD group at its edges.
    ENGINE_WITHOUT_ALIGNER = EngineRow(line_rule="group", audio_analytics=True)
