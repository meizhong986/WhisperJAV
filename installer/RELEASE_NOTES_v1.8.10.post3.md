**Quality Hardening + Crash Fixes + Post-Processing Overhaul**
22 commits since v1.8.10. Three crash fixes, full ASR/VAD parameter retune backed by
forensic analysis, post-processing pipeline hardened, GUI customize modal audited.

---

## Crash Fixes (3)

- **device="auto" crash in fidelity pipeline** — OpenAI Whisper ASR modules received `device="auto"` which is not a valid torch device string. Now resolved to actual device before passing to the backend. (`17bc2e0`)

- **length_penalty crash in OpenAI Whisper** — A parameter retune accidentally set `length_penalty=-0.5`, but OpenAI Whisper requires values in 0-1 range. Reverted to `None` (default behavior). (`ccd469f`)

- **Single-segment crash** — The post-model segment filter crashed when scene detection produced a single segment spanning the entire audio (no splits). Now handles this edge case. (`75b8e12`)

## Bug Fixes (8)

- **XXL model flag ignored** — `--pass2-model` was not forwarded to the XXL subprocess, so it always used large-v3 regardless of what the user specified. (#272) (`e50df15`)

- **Translation temp file overwrite** — AI SRT Translate created a temp file in the same folder as the input SRT, overwriting existing subtitles. Save path is now respected throughout the process. (#259) (`cfcbddf`)

- **numba cache hang on conda installs** — Semantic scene detection hung indefinitely on conda-constructor installs because numba couldn't write its JIT cache to read-only site-packages. Added `ensure_numba_cache_dir()` to redirect cache to user-writable `%LOCALAPPDATA%\whisperjav\numba_cache`. (#267) (`28dff5b`)

- **`--scene-detection-method` ignored** — The CLI flag had no effect in standard pipeline modes. One-line injection fix. (#269) (`4a600b2`)

- **Hallucination filter not bundled in wheel** — `filter_list_v08.json` was missing from pip-installed wheels. Stale gist URLs updated. Added user-visible INFO logging for filter activity. (#265) (`1bfe548`)

- **Sensitivity presets ignored for non-Silero segmenters** — Legacy pipeline only applied sensitivity presets to the Silero VAD backend. TEN and other segmenters now receive correct sensitivity-adjusted parameters. (`6109c0c`)

- **GUI customize modal: 11 fixes** — Wrong min/max/step values, missing parameters, incorrect data types in the "customize parameters" modal for the ensemble tab. Config standardized across all sources. (`a0eb82e`)

- **Colab/numpy2 migration** — Removed venv creation in Colab (reuse system torch), fixed llama-cpp filename, numpy 2.x compatibility updates. (#231) (`a6b473d`)

## Quality Improvements

### ASR Parameter Retune (forensic-verified)

All three sensitivity levels (aggressive/balanced/conservative) retuned across both balanced (Faster-Whisper) and fidelity (OpenAI Whisper) pipelines. Changes verified against ground truth using the forensic analysis methodology.

| Parameter | Aggressive | Balanced | Conservative |
|-----------|-----------|----------|-------------|
| beam_size | 4 → 2 | unchanged | 3 → 2 |
| best_of | 3 → 2 | 1 → 2 | 1 → 2 |
| patience | 2.5 → 2.0 | 2.0 → 1.6 | 1.5 → 1.2 |
| no_speech_threshold | 0.22 → 0.77 | 0.70 → 0.65 | 0.74 → 0.46 |
| logprob_threshold | -2.5 → -1.00 | -1.2 → -1.00 | -1.0 → -0.80 |
| temperature | [0.0, 0.3] → [0.0, 0.17] | [0.0, 0.1] → [0.0] | unchanged |
| no_repeat_ngram_size | 2 → 3 | 2 → 3 | 2 → 3 |
| max_initial_timestamp | None → 0.0 | None → 0.0 | None → 0.0 |

### Silero VAD v3.1/v4.0 Retune (anti mega-group)

Thresholds raised and group limits dramatically reduced to prevent Silero v3.1/v4.0 from creating mega-groups (15-69s segments) that degrade Whisper model performance.

| Parameter | Aggressive | Balanced | Conservative |
|-----------|-----------|----------|-------------|
| threshold | 0.05 → 0.18 | 0.18 → 0.28 | 0.35 → 0.41 |
| max_speech_duration_s | 14 → 8 | 11 → 7 | 9 → 6 |
| max_group_duration_s | 29 → 10 | 29 → 9 | 29 → 8 |
| neg_threshold | 0.15 → removed | 0.15 → removed | 0.15 → removed |

`neg_threshold` has been removed from all presets — it was never passed to the Silero v3.1/v4.0 API and had zero runtime effect. Silero v6.2 auto-calculates it when not supplied.

### Post-Processing Pipeline

- **Regex sanitization enabled** — Was previously disabled. Now catches hallucination patterns via regex matching. (`9c2dc8a`)
- **Punctuation normalization** — Full-width/half-width consistency in output subtitles. (`9c2dc8a`)
- **Post-merge deduplication** — Duplicate subtitles created during ensemble merge are now removed. (`9c2dc8a`)
- **New hallucination patterns** — 次回予告/映像特典 closing variants, multi-word English patterns. (`80a2ef6`)
- **Slow CPS removal** — Short hallucination labels (e.g., "息子" at 0.54 characters/second) now detected and removed. (`756da9d`)

### Post-Model Logprob Gate

The post-model logprob gate is now configurable per-pipeline:
- **Balanced pipeline**: gate OFF (fewer false drops, let more segments through)
- **Fidelity pipeline**: gate ON (strict filtering for higher confidence)

## New Features

- **TEN VAD backend hardening** — Silence merging, max_speech splitting, long-segment control. The TEN backend is now production-ready with the same quality controls as Silero. (`7375400`)

- **Forensic analysis tools** — `tools/forensic_csv_generator.py` for generating per-subtitle forensic CSVs from test runs, and `tools/fw_diagnostic_suite.py` for systematic Faster-Whisper parameter testing. Developer/diagnostic tools with documentation. (`78ad155`)

---

## How to Upgrade

**From v1.8.10 (or any v1.8.x):**

```
pip install -U --no-deps "whisperjav @ git+https://github.com/meizhong986/whisperjav.git@v1.8.10.post3"
```

**Windows .exe installer users:**

Download the updated installer from Assets below, or run `whisperjav-upgrade` from the WhisperJAV terminal.

### Fresh Install

#### Windows — Standalone Installer (.exe)

1. Download **WhisperJAV-1.8.10.post3-Windows-x86_64.exe** from the Assets below
2. Run the installer (no admin rights required)
3. Wait 10-20 minutes for setup to complete
4. Launch from the Desktop shortcut

Installs to `%LOCALAPPDATA%\WhisperJAV`. A desktop shortcut is created automatically. Your GPU is detected automatically.

#### macOS

Requires [Git](https://git-scm.com/downloads). Open Terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.10.post3
installer/install_mac.sh
```

After installation, double-click **WhisperJAV.command** to launch the GUI.

#### Linux

Requires Git and Python 3.10-3.12. Open a terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.10.post3
installer/install_linux.sh
```

After installation, launch the GUI with `./WhisperJAV.sh`.

#### Windows — Source Install

Requires [Git](https://git-scm.com/downloads) and [Python 3.10-3.12](https://www.python.org/downloads/). Open a terminal and run:

```
cd %USERPROFILE%
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.10.post3
installer\install_windows.bat
```

After installation, double-click **WhisperJAV.bat** to launch the GUI.

## Compatibility

Same as v1.8.10 — no dependency changes.

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 - 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |
| Ollama | 0.3.0+ recommended |

## Known Issues

- **Semantic scene detection hang (#267)** — The numba cache redirect fix resolves most cases, but one user reports the hang persists on a conda-constructor install. If you experience this, try setting the environment variable `NUMBA_CACHE_DIR=%LOCALAPPDATA%\whisperjav\numba_cache` manually before launching, or use auditok scene detection as a workaround (`--scene-detection-method auditok`).

- **Qwen3-ASR crash (#280)** — The third-party `qwen_asr` library may crash with `TypeError: check_model_inputs()` if your installed `transformers` version is incompatible. This is an upstream issue. Workaround: `pip install transformers==4.48.0` (or the version that `qwen_asr` was tested with).

- **Ollama thinking models (#271)** — Models with "thinking" capabilities (qwen3.5, gemma4, qwen3) include chain-of-thought in their output, breaking the translation format. Use instruct-tuned models instead: `gemma3:12b`, `qwen2.5:7b-instruct`, `translategemma:27b`.

- **Apple Silicon MPS + whisper-large-v3-turbo** — Produces garbage output on MPS for this specific model. Use `--hf-device cpu` or the default kotoba model. (#198, #227)

## What's Next (v1.9.0)

- Full Ollama migration — remove llama-cpp-python dependency entirely
- Standalone subtitle merge CLI tool (`whisperjav-merge`) (#230)
- Chinese GUI (partial i18n) (#175, #180)
- Speaker diarization (#248, #252)
- Translation model compatibility layer — detect thinking models vs instruct models (#271)

---

## Technical Details

### New Developer Tools

- `tools/forensic_csv_generator.py` — Generate per-subtitle forensic CSVs from test runs. Maps each ground truth subtitle to pipeline stages (scene detection → VAD → ASR → sanitizer) to identify where and why subtitles are lost. Invoke with `--base-dir` flag.

- `tools/fw_diagnostic_suite.py` — Systematic Faster-Whisper parameter testing. Run isolated parameter sweeps against audio segments to measure individual parameter effects on transcription quality.

### Config Consistency

All 5 config resolution paths synchronized: Pydantic presets, V4 YAML, legacy schemas, legacy mapper fallbacks, and GUI defaults. This eliminates the class of bugs where different code paths used different default values.
