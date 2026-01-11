# AI Coding Guide for WhisperJAV

## MASTER INSTRUCTIONS :

### Evidence-First Debugging (Mandatory)
- If the user says they already tried a fix, do **not** repeat the same root-cause claim; ask for (or create) a minimal check and update the diagnosis from its output.
- For runtime/library errors, do **not** assert a root cause without either (a) the exact error line/stack trace, or (b) direct verification via workspace logs/tools.
- When suggesting installs, always include a verification command; any non-zero exit is a hard stop (surface stdout/stderr tail and propose the next action).
- Before concluding, restate the key user-provided facts you’re relying on, then state what new evidence you need.
- Debug-output contract: request exactly **one** artifact (error line, log excerpt, or specific command output) and base the next step only on that artifact.




## 🧭 Architecture Map
- CLI orchestrator `whisperjav/main.py` (also `cli.py`) wires argument parsing, preflight checks, pipeline invocation, and translation toggles.
- PyWebView GUI lives in `whisperjav/webview_gui/` (`main.py`, `api.py`, `assets/`); GUI launches pipelines via the same CLI APIs and requires WebView2 on Windows.
- Pipelines under `whisperjav/pipelines/` inherit `BasePipeline` and select ASR implementations in `whisperjav/modules/` (e.g., `stable_ts_asr.py`, `whisper_pro_asr.py`, `scene_detection.py`, `subtitle_sanitizer.py`).
- Two-pass/ensemble logic is spec'd in `docs/ENSEMBLE_*` and executed through CLI flags (`--ensemble`, `--pass*-pipeline`, merge strategies).
- Translation stack (`whisperjav/translate/cli.py`, `core.py`, `providers.py`) wraps PySubtrans, manages provider credentials, and caches instruction presets in `%AppData%/WhisperJAV/translate/cache`.

## ⚙️ Config & Pipeline Rules
- Treat `whisperjav/config/v4` as the source of truth: YAML ecosystems/models/presets feed `ConfigManager` (`manager.py`) and expose GUI schemas through `gui_api.py`.
- When adding or tweaking models, edit YAML under `config/v4/ecosystems/**` and let `ConfigManager().get_model_config(...)` or `list_models()` drive runtime choices—never hardcode defaults in Python.
- Sensitivity/mode knobs must go through `TranscriptionTuner` (`transcription_tuner.py`) so presets remain consistent across CLI, GUI, and translation flows.
- Pipelines must accept `**kwargs` so new config keys can be threaded through without breaking legacy constructors; resolve final kwargs with the tuner/config manager before instantiating ASR modules.

## 🧪 Development Workflow
- Install editable with GPU-ready PyTorch first, then `pip install -e .[dev]`; CLI entry points `whisperjav`, `whisperjav-gui`, and `whisperjav-translate` map to `whisperjav/main.py`, `webview_gui/main.py`, and `translate/cli.py`.
- Run tests via `python -m pytest tests/` or target suites like `python -m pytest tests/test_config_v4.py -k transformers`; many tests inject `sys.path.insert(0, Path(__file__).parents[1])` so keep relative imports stable.
- Lint/format with `python -m ruff check whisperjav/` and `python -m ruff format whisperjav/`; translation/GUI JS lives outside Ruff’s scope.
- Build the Windows installer pipeline from `installer/` using `python installer/build_release.py`, which regenerates `generated/*.bat` and constructor specs before invoking `build_installer_*.bat`.

## 💻 Coding Patterns & Conventions
- Use `pathlib.Path` for every filesystem interaction (temp dirs, cache, FFmpeg paths) and respect the `temp_dir` resolved from config; never shell out with raw strings.
- Route logging through `whisperjav.utils.logger` (structured, colorized output) and emit CLI progress via `progress_aggregator.py`; only CLI user prompts may use `print`.
- Translation progress must go to `stderr`, while resulting subtitle/translation paths go to `stdout` to keep scripts composable.
- Don’t import `torch` at module import time in utilities—delay until needed so `whisperjav --help` stays fast and skips CUDA probing already handled in `main.py`.
- When touching subtitle post-processing, keep Japanese-specific regrouping and hallucination filters in `modules/stable_ts_asr.py` and `modules/subtitle_sanitizer.py` synchronized.

## 🔌 External Integrations
- Expect FFmpeg on PATH and pre-installed PyTorch with CUDA/MPS; `whisperjav/utils/preflight_check.py` handles enforcement, so reuse its helpers instead of duplicating GPU checks.
- PySubtrans may return `Path` or `str`; normalize with `Path(value)` before downstream use to avoid Windows-only bugs.
- New translation providers or AI instructions belong in `whisperjav/translate/providers.py` and `translate/instructions.py`, which also handle caching + Gist fetch fallbacks.

## ⚠️ Gotchas
- CUDA detection runs on first `torch` import; guard imports inside functions when adding utilities or tests.
- Scene detection/VAD temp artifacts must respect the configured `temp_dir` or `%TEMP%/whisperjav`; leaking files breaks GUI cleanup.
- Ensemble merges expect strictly monotonic timestamps; when altering `modules/srt_postprocessing.py`, rerun `tests/test_tab_spacing.py` and `tests/ensemble_test_output` fixtures.
- GUI builds require WebView2 and proper icons from `webview_gui/assets`; keep new static assets referenced in `installer/create_icon.py`.