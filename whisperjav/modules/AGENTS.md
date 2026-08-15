# AGENTS.md — ASR / Engines area dossier

> Loaded when working under `whisperjav/modules/` (ASR engines). Stores decisions,
> invariants, and gotchas — **not** code. Inducted 2026-06-20 from primary sources.
> Cross-area contracts: `docs/architecture/MODULE_CONTRACTS.md` (C2 = ASR return shape).
> Disciplines: `.claude/agents/_disciplines.md` — **NEVER heavy-import these modules.**

## Engine inventory

| File | Backend | Returns |
|------|---------|---------|
| `whisper_pro_asr.py` (`WhisperProASR`) | OpenAI Whisper (PyTorch) + external segmenter | **Dict** (`:339`) |
| `faster_whisper_pro_asr.py` (`FasterWhisperProASR`) | faster-whisper / ctranslate2 + external segmenter | **Dict** (`:553`) |
| `stable_ts_asr.py` (`StableTSASR`) | Whisper or faster-whisper via stable-ts | **WhisperResult** (`:477`) |
| `qwen_asr.py` (`QwenASR`) | Qwen3-ASR (transformers) + optional ForcedAligner | WhisperResult |
| `transformers_asr.py` (`TransformersASR`) | HF transformers (kotoba) chunked long-form | (see file) |
| `kotoba_faster_whisper_asr.py` | kotoba faster-whisper / ctranslate2 | (see file) |
| `segment_filters.py` | post-transcription logprob/nonverbal filter (shared) | — |

Constructor convention (Dict/WhisperResult engines): `__init__(model_config: Dict, params: Dict, task: str, tracer=None)`.

## Invariants / red-lines (evidence-cited)

- **Silero VAD parameter firewall** — `whisper_pro_asr.py:68-94`, `faster_whisper_pro_asr.py:96-122`.
  When the segmenter backend is **not** `silero*`, resolver-produced Silero `vad_params` are
  **blanked** (`faster_whisper_pro_asr.py:103-110`) before merging, else they contaminate the
  segmenter factory. A second guard (`:91-94`) only merges `vad_params` for silero backends.
  **Do not "simplify" this away** — it is the fix for a documented contamination class.
- **post_model_filter default differs by engine** — ON for `WhisperProASR` (`:117-127`),
  OFF for `FasterWhisperProASR` (`:145-155`). Rationale in comments (R5/R6 forensic findings:
  the gate helps Whisper, hurts faster-whisper whose Layer-1 filter already cleans output).
- **No `chunk_length` / overlap hardcoded in the Pro ASR wrappers** — segmentation is delegated
  to `SpeechSegmenterFactory`. (Note: the *aggressive faster-whisper preset* does pin
  `chunk_length=30` in the config layer — removing it empirically yields 100% empty output;
  see `memory/MEMORY.md`. That pin lives in config, not here.)
- **compute_type selection** (`config/resolver_v3.py:~134-170`): Pascal→float32, Blackwell→float16
  (int8_float16 buggy in ct2), other CUDA→float16, CPU/MPS→"auto".

## Gotchas

- **MPS + ctranslate2 unsupported** → silent downgrade to CPU (`faster_whisper_pro_asr.py:51-61`).
- **float16 invalid on CPU** → fallback to "auto" (`faster_whisper_pro_asr.py:74-79`).
- **stable-ts silently drops params** not in the active backend's allow-set
  (`stable_ts_asr.py:341-356`) — masks config typos.
- **Qwen >180s** internally chunks; boundary timestamp continuity not guaranteed — prefer
  WhisperJAV scene detection for long files (`qwen_asr.py:176-183`).
- **VAD segments arrive grouped** (`List[List[Dict]]`); ASR flattens for transcription, keeps
  grouping only for viz via `_last_vad_segments`.

## Tests that pin behavior (source of truth)

`tests/test_qwen_asr.py` (init, device/dtype fallback, WhisperResult return, translation→transcribe
fallback), `tests/test_qwen_pipeline_integration.py`, `tests/test_qwen_japanese_postprocess.py`,
`tests/test_speech_segmentation.py`, `tests/test_hardening.py`, `tests/test_vad_failover.py`.

## Active context

CrispASR (`pipelines/crispasr_*`) is an **external subprocess** provider, not an in-process engine
— it does NOT belong here; see `memory/project_v190_crispasr_test_loop_resume_pointer.md`.
v1.9.x theme = ASR backend expansion (Cohere, FireRedVAD, etc.).
