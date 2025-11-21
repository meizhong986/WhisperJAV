# AI Coding Guide for WhisperJAV

## 🧠 Project Architecture
- **Entry Points**: 
  - `whisperjav/main.py`: CLI entry, orchestrates pipelines and async processing.
  - `whisperjav/webview_gui/main.py`: GUI entry point.
- **Pipelines** (`whisperjav/pipelines/`): 
  - All pipelines inherit from `BasePipeline`.
  - `BalancedPipeline` & `FasterPipeline` use `FasterWhisperProASR`.
  - `FidelityPipeline` uses `WhisperProASR`.
  - `FastPipeline` uses `StableTSASR`.
- **Configuration** (`whisperjav/config/`):
  - **v2.0 Architecture**: Centralized in `ConfigManager` (`manager.py`) handling `asr_config.json` (v4.3 schema).
  - **Resolution**: Use `TranscriptionTuner` (`transcription_tuner.py`) to resolve parameters based on mode/sensitivity.
  - **Validation**: `ConfigManager` enforces schema validation.
- **Translation** (`whisperjav/translate/`):
  - `cli.py` drives `whisperjav-translate`.
  - `core.translate_subtitle` wraps PySubtrans.
  - Providers defined in `providers.py`.

## 🛠️ Development Workflow
- **Installation**: `pip install -e .` (requires pre-installed PyTorch with CUDA).
- **Testing**: 
  - Run all: `python -m pytest tests/`
  - Specific: `python -m pytest tests/test_balanced_pipeline.py`
  - **Note**: Tests often require `sys.path.insert(0, ...)` to resolve `whisperjav` in dev mode.
- **Linting**: `python -m ruff check whisperjav/`
- **Installer**: 
  - Scripts in `installer/`.
  - Build release: `python installer/build_release.py`.

## 📝 Coding Conventions
- **Filesystem**: Always use `pathlib.Path`, never raw strings for paths.
- **Logging**: Use `whisperjav.utils.logger`. Do not use `print()` except for CLI output.
- **Configuration**: 
  - **Do not** read `asr_config.json` directly. Use `ConfigManager` or `TranscriptionTuner`.
  - Respect `**kwargs` in Pipeline `__init__` to handle evolving config parameters gracefully.
- **Translation**:
  - Progress output goes to `stderr`, result paths to `stdout`.
  - Instructions cached in `%AppData%/WhisperJAV/translate/cache/`.

## ⚠️ Gotchas
- **CUDA**: `torch` import triggers CUDA check. Bypass for `--help` is handled in `main.py`.
- **Windows Paths**: `PySubtrans` may return `str` or `Path`; handle both.
- **Temp Files**: Respect `temp_dir` config; default is `%TEMP%/whisperjav`.