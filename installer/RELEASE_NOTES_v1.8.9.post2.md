# WhisperJAV v1.8.9 — Hotfix 2

**Hotfix release** — fixes 3 additional bugs discovered after hotfix 1.

If you're on v1.8.9 or v1.8.9-hotfix1 and experiencing crashes on CPU, GUI startup failures, or XXL errors with Chinese filenames, this release fixes all three.

## Bug Fixes (Hotfix 2)

- **CPU users crash on startup** — v1.8.9's quality improvement changed the default compute type to `float16`, which is only supported on NVIDIA GPUs. All CPU-only users, Apple Silicon (MPS) users, and anyone without a CUDA GPU would crash immediately with "target device or backend do not support efficient float16 computation". The resolver now uses device-aware selection: `float16` on CUDA, `auto` on everything else. (#241)

- **GUI fails to start on Windows 11** — The persistent WebView2 storage folder could become corrupted, causing access violations and localhost timeouts on launch. Switched to `private_mode=True` which eliminates disk caching entirely. All user settings persist via the Python backend (not browser storage), so nothing is lost. (#240)

- **XXL crashes with Chinese filenames** — When processing files with Chinese characters in the path, the XXL subprocess runner would crash with `UnicodeDecodeError` while reading stderr, even though the transcription completed successfully and the SRT was written. Fixed by using lossy decoding (`errors="replace"`) for diagnostic stderr output.

## Bug Fixes (Hotfix 1)

- **Ollama translation returns 404** — The API endpoint path was being appended twice (`/v1/chat/completions/v1/chat/completions`), causing every `--provider ollama` translation to fail with HTTP 404. (#132)

- **GUI shows old interface after upgrade** — WebView2 cached the previous version's HTML/CSS/JS to disk. After upgrading, users saw the old GUI without v1.8.9 changes. (Superseded by the `private_mode=True` fix above.) (#236)

- **Icon setting crashes on first launch** — On 64-bit Windows, the taskbar icon function could fail with `OverflowError` because Win32 API functions were called without proper type declarations. (#235)

- **Portuguese translation target** — Added Portuguese/Brazilian as a translation target language. Available in both CLI (`--translate-target portuguese`) and GUI. (#238)

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
pip install -U "whisperjav @ git+https://github.com/meizhong986/whisperjav.git"
```

## Compatibility

Same as v1.8.9 — no dependency changes.

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 - 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |

## Full Changelog

For the complete list of v1.8.9 features (XXL BYOP, quality improvements, OllamaManager, etc.), see the [v1.8.9 release notes](https://github.com/meizhong986/WhisperJAV/releases/tag/v1.8.9).
