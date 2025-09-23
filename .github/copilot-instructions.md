# AI coding guide for WhisperJAV (work with confidence)

Use this repo-specific context to be productive immediately. Keep changes small, respect existing patterns, and test via the CLI.

## Architecture, at a glance
- Entry points: `whisperjav/main.py` (CLI orchestration), `whisperjav/cli.py` (pre-flight wrapper), GUI in `whisperjav/gui/` (`whisperjav-gui`).
- Pipelines (`whisperjav/pipelines/`):
  - `faster_pipeline.py` → Faster-Whisper via Stable-TS wrapper (turbo mode, no scene splitting).
  - `fast_pipeline.py` → Standard Whisper via Stable-TS + mandatory audio scene detection.
  - `balanced_pipeline.py` → WhisperPro (OpenAI Whisper) + scene detection + VAD (most accurate).
- Core modules (`whisperjav/modules/`):
  - ASR: `stable_ts_asr.py` (backend filter/param normalization, Japanese regrouping); `whisper_pro_asr.py`.
  - Audio I/O: `audio_extraction.py` (FFmpeg), scene splitting: `scene_detection.py` (incl. `DynamicSceneDetector`).
  - Post: `srt_postprocessing.py` routes to Japanese/English cleaners; `subtitle_sanitizer*.py` hold rules.
  - Stitch/merge: `srt_stitching.py`; metadata: `utils/metadata_manager.py`; progress: `utils/progress_*`.

## Config flow and conventions
- Central config: `whisperjav/config/asr_config.json` describes models, pipelines (faster/fast/balanced), and per-sensitivity parameter packs (common/engine/vad/stable-ts).
- Builder: `config/transcription_tuner.py` resolves mode+sensitivity to a structured dict: `{ model, params, features, task }`; pipelines pass this into ASR classes.
- Language: `--subs-language english-direct` forces model switch from `turbo` to `large-v2` for translation (see pipelines).
- Paths: use `pathlib.Path`. Temp artifacts live under `--temp-dir`; final SRT → `output/{basename}.{ja|en}.whisperjav.srt`; raw artifacts go in `output/raw_subs`.
- Logging: `whisperjav/utils/logger.py`; avoid bare prints. Use unified progress adapters in `utils/unified_progress.py` via pipeline constructors.

## ASR integration patterns (important)
- Stable-TS wrapper (`StableTSASR`):
  - Filters params per backend (standard vs faster-whisper) and coerces types for CTranslate2 (e.g., lists not tuples, ints for `no_repeat_ngram_size`).
  - Always sets `task` and `language`; VAD flags (`vad`, `vad_threshold`) come from `stable_ts_vad_options`.
  - Suppresses noisy warnings/output; pre-caches Silero VAD: `snakers4/silero-vad:v4.0` (standard) or latest (turbo).
- Balanced pipeline uses `WhisperProASR` + separate VAD; fast/faster pipelines use `StableTSASR`.

## Developer workflow (Windows-friendly)
- Env: Python 3.9–3.12, CUDA-enabled PyTorch, FFmpeg in PATH. `openai-whisper` is pinned to main; `stable-ts` uses a custom fork.
- Install (dev): `pip install -e . -U`  (ensure CUDA torch installed first). Run:
  - CLI: `whisperjav <file> --mode balanced --sensitivity aggressive` (or `python -m whisperjav.main ...`).
  - GUI: `whisperjav-gui`.
- Tests: `pytest` under `whisperjav/tests/` (example present: `test_async_cancellation.py`).
- Lint/format (optional but used in docs): ruff (`python -m ruff check whisperjav/; python -m ruff format whisperjav/`).

## When adding features
- Plug into a pipeline: subclass `BasePipeline` or extend `DynamicSceneDetector`; respect `progress_reporter` and `keep_temp_files`.
- Scene detectors must return: `(scene_path: Path, start_s, end_s, duration_s)`; stitcher expects a list of scene SRTs with offsets.
- Post-processing: return `(processed_path, stats)` from `SRTPostProcessor.process`; keep Japanese vs English branching.
- Preserve naming and folders (`raw_subs`, `..._stitched.srt`), and update `MetadataManager` stages for observability.

## Gotchas
- CUDA is enforced in `main.py` (`enforce_cuda_requirement()`); Python 3.13 is not supported by `openai-whisper`.
- Faster mode (turbo) can’t translate; pipelines switch to `large-v2` when `english-direct` is requested.
- Stable-TS can spawn FFmpeg; wrapper mitigates console spam with `suppress_output()` and in-memory audio loading.

If any of the above feels off or you need deeper examples (e.g., param maps or progress wiring), tell me which file/flow to expand and I’ll refine this doc.