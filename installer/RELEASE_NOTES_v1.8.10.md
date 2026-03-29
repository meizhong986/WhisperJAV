# WhisperJAV v1.8.10

**Quality + Ollama + Stability** — Transcription accuracy significantly improved through ground-truth-validated parameter tuning. Ollama is now a first-class translation provider in the GUI. Several bugs fixed across the board.

## What's New

### Transcription Quality Improvement

- **Aggressive sensitivity retune** — I ran the aggressive preset against Netflix ground truth (68 subtitles, 293s clip) and found that several parameters were working against each other. Coverage improved from 76.5% to 92.6% after four iterations of testing. If you use aggressive mode, this is a meaningful accuracy upgrade.

- **Key parameter changes** — `no_speech_threshold` was inverted (the "aggressive" value was actually the most restrictive). Fixed across all four ASR backends. `compression_ratio_threshold` tightened from 3.0 to 2.6 to reject hallucination loops while keeping legitimate Japanese dialogue. `condition_on_previous_text` enabled for aggressive mode to help recognize repeated dialogue. Decode capacity increased (beam_size, best_of) so the model has more room to find correct transcriptions in difficult audio.

- **Diagnostic JSON per scene** — Full Whisper transcribe() results are now saved as JSON alongside each scene SRT. This includes per-segment no_speech_prob, compression_ratio, and temperature — useful for understanding why specific segments were captured or missed. Saved automatically, no configuration needed.

### Ollama GUI Integration

- **Ollama as a first-class provider** — Select "Ollama" from the provider dropdown in the Ensemble tab or Standalone Translation tab. No more workarounds with "Custom Server" or manual endpoint configuration. (#132, #128)

- **Three-state onboarding** — When you select Ollama, the GUI detects your setup and shows the right panel: not installed (download link), no model (recommended model with copyable `ollama pull` command), or connected (green status dot).

- **In-GUI model download** — Selecting an uninstalled model opens a confirmation popup with the model name and download size. Click "Download" to pull it directly. Progress streams to the terminal.

- **VRAM-aware model recommendations** — The GUI detects your GPU VRAM and recommends the best model: gemma3:4b for 4GB, qwen2.5:7b for 8GB, gemma3:12b for 12GB, qwen2.5:14b for 16GB+.

- **Automatic VRAM cleanup** — When ensemble translation finishes, the Ollama model is automatically unloaded from VRAM. Previously it stayed loaded indefinitely, blocking other GPU workloads.

- **Auto-start server** — If Ollama is installed but not running, the GUI starts it automatically when you select the Ollama provider.

- **llama-cpp deprecated** — The "Local LLM" option is renamed to "llama-cpp (deprecated)". I plan to remove it entirely in v1.9.0 in favor of Ollama.

### Enhance for VAD

- **Checkbox in ensemble UI** — The "Enhance for VAD only" option (use enhanced audio for speech detection but original audio for transcription) is now accessible as a checkbox below each enhancer dropdown in the ensemble tab. Previously this was only available inside the Qwen Customize Parameters modal. Works for all pipelines. (#253)

## Bug Fixes

- **XXL exe path lost on restart** — The BYOP panel's XXL executable path was not restored when reopening the app. Caused by a race condition between DOMContentLoaded and pywebview API readiness. Fixed by reloading BYOP preferences after pywebview is ready.

- **XXL --model hardcoded** — The `--model` flag was hardcoded in the XXL runner, preventing users from changing it. Moved to user-editable Extra Args field (default: `--model large-v3`).

- **Silero VAD crashes in Colab/Kaggle** — `torch.hub.load()` calls `input()` for interactive trust confirmation, which crashes with EOFError in non-interactive environments. Fixed with `trust_repo=True`. (#253)

- **Config contamination between runs** — VAD parameters from one run could leak into the next due to mutable default dictionaries. Added a contamination firewall that deep-copies config at pipeline entry.

- **GUI ensemble presets not applied** — Switching between pipelines of the same type (e.g., balanced to fast, both using Faster-Whisper) did not refresh the preset values. The GUI now always reloads presets on pipeline change.

- **Colab/Kaggle notebook fixes** — Added `llvmlite>=0.46.0` to install cells and `TORCH_HUB_TRUST_REPO=1` environment variable. (#253, #231)

- **Translation diagnostic hardening** — Fixed 12+ issues in translation pipeline: truthful statistics, debug wiring, temperature clobber, Qwen3 thinking model support, ground-truth detection, instruction delivery, PySubtrans integration diagnostics, and Ollama server-side log guidance.

- **Output directory not honored** — Fixed output path handling and added artifact cleanup safety guard.

## How to Upgrade

**Windows installer users:**
```
whisperjav-upgrade
```

**Source install (git) users:**
```
git pull && uv sync
```
Or simply: `whisperjav-upgrade`

**Colab / Kaggle users:**
Re-run your install cell — it always pulls the latest version.

**pip users:**
```
pip install -U "whisperjav @ git+https://github.com/meizhong986/whisperjav.git@v1.8.10"
```

## Compatibility

Same as v1.8.9 — no dependency changes.

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 - 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |
| Ollama | 0.3.0+ recommended |

## Known Issues

- **Apple Silicon MPS + whisper-large-v3-turbo** — Produces garbage output on MPS for this specific model. Use `--hf-device cpu` or the default kotoba model. (#198, #227)

- **Ollama download progress** — The download progress bar in the popup is indeterminate (pulsing). Real progress is shown in the terminal. A hint in the popup directs you to check there.

## What's Next (v1.9.0)

- Full Ollama migration — remove llama-cpp-python dependency entirely
- Standalone subtitle merge CLI tool (`whisperjav-merge`) (#230)
- Chinese GUI (partial i18n) (#175, #180)
- Speaker diarization (#248, #252)

## Technical Details

<details>
<summary>Click to expand — for power users and contributors</summary>

### Aggressive Preset Changes (all backends)

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| no_speech_threshold | 0.22 (OpenAI) / 0.22 (Faster) | 0.60 / 0.55 | Was inverted — lower = more restrictive, not more aggressive |
| compression_ratio_threshold | 3.0 | 2.6 | Rejects hallucination loops (ratio ~10) while keeping legitimate repetition (ratio 2.0-2.5) |
| beam_size | 2 (OpenAI) / 3 (Faster) | 5 / 4 | More decode capacity for difficult audio |
| best_of | 1 / 3 | 3 / 3 | More sampling diversity |
| condition_on_previous_text | False | True | Helps recognize dialogue echoes and repeated phrases |
| hallucination_silence_threshold | 2.5 (OpenAI) | None (disabled) | Was suppressing real speech near silence gaps |
| temperature | [0.0, 0.2, 0.4] | [0.0, 0.15, 0.3, 0.5] | 4-step fallback with finer initial steps |

### New Developer Tools

- `scripts/whisper_param_tuner.py` — Standalone CLI for testing Whisper parameters against ground truth SRT in seconds (vs 8+ min full pipeline). Supports `--sweep` for grid search, `--gt-focus` for specific subtitle IDs, both OpenAI and Faster-Whisper backends.

- Diagnostic JSON — Saved to `{scene_name}_diagnostic.json` alongside scene SRTs. Contains full segment data including no_speech_prob, compression_ratio, temperature per segment.

### Ollama GUI Architecture

- `OllamaStateManager` — centralized state machine (CHECKING / NOT_INSTALLED / NO_MODEL / READY)
- `ProviderUIManager` — centralized provider-change handler for both tabs
- `OllamaPullModal` — download confirmation + progress + error handling
- Curated models loaded from `whisperjav/config/ollama_models.json` (not hardcoded in JS)
- New API methods: `detect_ollama()`, `start_ollama_server()`, `list_ollama_models()`, `recommend_ollama_model()`, `pull_ollama_model()`, `get_ollama_curated_models()`

### Files Changed (39 commits)

Major files modified:
- `whisperjav/config/components/asr/` — all 4 ASR preset files (aggressive retune)
- `whisperjav/modules/whisper_pro_asr.py` — diagnostic JSON save
- `whisperjav/modules/faster_whisper_pro_asr.py` — diagnostic JSON save
- `whisperjav/modules/stable_ts_asr.py` — diagnostic JSON save + trust_repo
- `whisperjav/modules/speech_segmentation/backends/silero.py` — trust_repo
- `whisperjav/webview_gui/assets/app.js` — Ollama integration, XXL fixes, enhance-for-VAD
- `whisperjav/webview_gui/assets/index.html` — Ollama panels, enhance-for-VAD checkbox
- `whisperjav/webview_gui/api.py` — 6 new Ollama API methods
- `whisperjav/translate/` — 8 files for translation diagnostic hardening
- `scripts/whisper_param_tuner.py` — new utility

</details>
