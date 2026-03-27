# WhisperJAV v1.8.10

**Ollama GUI Integration** — Ollama is now a first-class translation provider in the GUI with full onboarding, model management, and VRAM cleanup.

If you're on v1.8.9 and using Ollama for translation (or want to start), this is the upgrade to get. The GUI now handles the entire Ollama workflow — detection, model downloads, translation, and VRAM cleanup — without ever touching a terminal.

## What's New

### Ollama GUI Integration

- **Ollama as a first-class provider** — Select "Ollama" from the provider dropdown in both the Ensemble tab (Tab 3) and Standalone Translation tab (Tab 4). No more workarounds with "Custom Server" or manual endpoint configuration. (#132, #128)

- **Three-state onboarding panels** — When you select Ollama, the GUI detects your setup and shows the right panel:
  - **Not installed**: Download link and "Check Again" button
  - **No model**: Recommended model with copyable `ollama pull` command
  - **Connected**: Green dot with "Ollama server connected" status

- **Smart model dropdown with optgroups** — The model dropdown is split into two groups:
  - **Installed Models**: All models currently on your machine (with download sizes)
  - **Top Recommendations**: Curated models you haven't installed yet (duplicates omitted)
  - At-rest state shows "-- Select a model --" instead of a misleading pre-selection

- **In-GUI model download** — Selecting an uninstalled model opens a confirmation popup showing the model name and download size. Click "Download" to pull the model directly from the GUI. Progress streams to the terminal; the popup auto-closes on success and the model is auto-selected.

- **Curated model list in JSON config** — The 5 recommended models (gemma3:4b, qwen2.5:7b, gemma3:12b, qwen2.5:14b, qwen2.5:3b) are defined in `whisperjav/config/ollama_models.json`. Updatable without code changes.

- **VRAM-aware model recommendations** — OllamaManager detects your GPU VRAM and recommends the best model: gemma3:4b for 4GB, qwen2.5:7b for 8GB, gemma3:12b for 12GB, qwen2.5:14b for 16GB+.

- **Auto-start server** — If Ollama is installed but not running, the GUI starts it automatically when you select the Ollama provider.

### VRAM Management

- **Automatic model unload after translation** — When ensemble translation completes (all files processed), the Ollama model is automatically unloaded from VRAM via `POST /api/generate {"keep_alive": 0}`. Previously, the model stayed in VRAM indefinitely, blocking other GPU workloads. The console shows `[OLLAMA] Model X unloaded from VRAM` on success.

### Provider Management

- **Provider dropdown redesign** — Both translation dropdowns (Ensemble + SRT tabs) now use optgroup-based selects with "Local LLM" and "Cloud AI" groups. Blank "-- Select a Provider --" placeholder prevents false readiness signals.

- **llama-cpp deprecated** — The "Local LLM" option is renamed to "llama-cpp (deprecated)" with a visible deprecation hint: "This backend is being retired in v1.9.0. Consider switching to Ollama." Deprecation warnings also emit on CLI (`--provider local`).

- **Provider selection persists** — Your provider choice is saved to localStorage and restored on next launch, so you don't re-select Ollama every time.

### Ollama Backend

- **Settings persistence** — `ollama_url` is saved to the settings file with `OLLAMA_HOST` env var fallback. Configurable in Translation Settings modal.

- **New API methods**: `detect_ollama()`, `start_ollama_server()`, `list_ollama_models()`, `recommend_ollama_model()`, `pull_ollama_model()`, `get_ollama_curated_models()`.

- **New CLI flags**: `--ollama-url URL`, `--list-ollama-models`, `--yes` (auto-confirm model pulls).

- **Gemma 3 4B model config** — Added gemma3:4b (128K context, 2.5GB download) to OllamaManager's model configs. VRAM recommendation tiers updated.

## Bug Fixes

- **Ollama model stays in VRAM after translation** — Model now unloaded automatically after ensemble translation completes. Best-effort (never fails the run).

- **Selecting uninstalled Ollama model fails silently** — Previously, the dropdown showed all curated models as equally selectable. Selecting an uninstalled model would fail at runtime with "Model not available locally." Now triggers the download confirmation popup.

- **"Ollama ready" gave false confidence** — Status text changed to "Ollama server connected" to accurately reflect what is verified (server connectivity, not model readiness).

- **Hardcoded Ollama model lists** — Model names and VRAM labels were duplicated in 3 places in the JS. Now driven from a single JSON config file via the backend.

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
Re-run your install cell -- it always pulls the latest version.

**pip users:**
```
pip install -U "whisperjav @ git+https://github.com/meizhong986/whisperjav.git"
```

## Compatibility

Same as v1.8.9 -- no dependency changes.

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 - 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |
| Ollama | 0.3.0+ recommended |

## Known Issues

- **Ollama pull progress not shown in popup** — The download progress bar in the popup is indeterminate (pulsing). Real progress is shown in the terminal from which WhisperJAV was launched. A hint in the popup directs users to check the terminal.

- **Apple Silicon MPS + whisper-large-v3-turbo** — Produces garbage output on MPS for this specific model. Use `--hf-device cpu` or the default kotoba model. (#198, #227)

## What's Next (v1.9.0)

- Full Ollama migration -- remove llama-cpp-python dependency entirely
- Standalone subtitle merge CLI tool (`whisperjav-merge`) (#230)
- Chinese GUI (partial i18n) (#175, #180)

## Technical Details

<details>
<summary>Click to expand -- for power users and contributors</summary>

### Files Changed (11 files, +1151 / -56 lines)

| File | Change |
|------|--------|
| `whisperjav/config/ollama_models.json` | **New** -- curated model list (5 models with sizes) |
| `whisperjav/main.py` | Post-translation Ollama model unload, --translate-provider help text |
| `whisperjav/translate/cli.py` | Deprecation warning for --provider local |
| `whisperjav/translate/ollama_manager.py` | gemma3:4b config, unload_model() method |
| `whisperjav/translate/providers.py` | DEPRECATED comment on local provider |
| `whisperjav/translate/service.py` | Deprecation warning for local provider |
| `whisperjav/translate/settings.py` | ollama_url in settings + OLLAMA_HOST env var |
| `whisperjav/webview_gui/api.py` | 6 new Ollama API methods, provider persistence |
| `whisperjav/webview_gui/assets/app.js` | OllamaStateManager, ProviderUIManager, OllamaPullModal, dropdown refactor |
| `whisperjav/webview_gui/assets/index.html` | Onboarding panels, pull modal, optgroup dropdowns |
| `whisperjav/webview_gui/assets/style.css` | Theme-aware Ollama styles, pull modal styles |

### New OllamaManager Methods
- `unload_model(name)` -- POST /api/generate with keep_alive=0
- `recommend_model()` updated tiers: 4GB -> gemma3:4b

### GUI Architecture
- `OllamaStateManager` -- centralized state machine (CHECKING / NOT_INSTALLED / NO_MODEL / READY)
- `ProviderUIManager` -- centralized provider-change handler for both tabs
- `OllamaPullModal` -- download confirmation + progress + error handling
- Curated models loaded from backend JSON config (not hardcoded in JS)

</details>
