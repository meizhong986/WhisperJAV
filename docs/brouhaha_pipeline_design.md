# Brouhaha Pipeline & QualityAware ASR

This document outlines the architectural additions:

- A new ASR engine `QualityAwareASR` that uses brouhaha (VAD/SNR/C50) to route segments to enhancement or direct ASR.
- A new pipeline `BrouhahaPipeline` mirroring the balanced pipeline but using `QualityAwareASR`.

## Contracts
- `QualityAwareASR.transcribe(path) -> Dict` returns {segments, text, language, [quality_meta]}.
- `QualityAwareASR.transcribe_to_srt(path, out_path) -> Path` writes SRT similar to WhisperProASR.

## Config
- Optional `params.quality_aware.routing_config_path` to point to a JSON like `whisperjav/config/quality_routing.template.json`.
- Optional `params.quality_aware.hf_token` for private models if needed.

## Integration
- New pipeline class `BrouhahaPipeline` under `whisperjav/pipelines/brouhaha_pipeline.py`.
- To enable as a CLI mode, wire it into `main.py` mode selection (not yet done in code here to avoid conflicts).

## Enhancement registry
- Implement your enhancement steps, then register them by name in `audio_enhancement.registry`.
- The router selects a chain via config and `run_chain` executes it.

## Next steps
- Decide on enhancement implementations: rnnoise/nara_wpe/demucs/pyloudnorm.
- Wire a new CLI mode (e.g., --mode brouhaha) and add unit tests for routing decisions.
