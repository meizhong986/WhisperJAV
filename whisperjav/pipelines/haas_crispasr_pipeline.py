#!/usr/bin/env python3
"""Haas-CrispASR: Haas-ja-v1.0, a Japanese Parakeet-TDT model by third-intelligence, run through CrispASR (1.10.2).

An Ensemble Mode pass pipeline only (W118 2): ``whisperjav --ensemble --pass1-pipeline haas-crispasr``.
Asked for in issue #504; added by the owner in W574, experimental, on the findings of the GGUF model test tool
(tools/crispasr_model_test.py, owner W568-W573; docs/plans/design/2026-11-06-haas-crispasr-first-measurements.md).
Models: the full-size file (f16, the default, as W400) or q8_0; both scored the same on the Netflix clips.

The chain is shared with the other CrispASR pipelines (whisperjav/pipelines/crispasr_pipeline.py). What is Haas's own
is what the test tool found and measured it with:
  - CrispASR 0.8.50 opens it as the parakeet family, and every token comes back with a start and an end, so it is
    built as Parakeet-CrispASR is: its own token times, cut at sentence-end marks and pauses, no aligner;
  - Parakeet-CrispASR's speech segmenter settings (TEN VAD at aggressive by default, VAD groups of at most 6 s), the
    conditions it was measured in; other settings were not measured;
  - the GGUF clean-up and the audio analytics, as the other CrispASR pipelines.
"""

from whisperjav.pipelines.crispasr_pipeline import CrispASRPipeline, EngineRow


class HaasCrispASRPipeline(CrispASRPipeline):
    """Haas-CrispASR, an Ensemble pass pipeline."""

    PIPELINE_NAME = "haas-crispasr"
    LABEL = "Haas-CrispASR"
    ENGINE = EngineRow(line_rule="tokens", audio_analytics=True)
