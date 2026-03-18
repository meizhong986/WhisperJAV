# WhisperJAV v1.8.9 — Hotfix 1

**Hotfix release** — fixes 3 bugs in v1.8.9 and adds Portuguese translation.

If you upgraded to v1.8.9 and something isn't working right, this release addresses the most common issues reported.

## Bug Fixes

- **Ollama translation returns 404** — The API endpoint path was being appended twice (`/v1/chat/completions/v1/chat/completions`), causing every `--provider ollama` translation to fail with HTTP 404. Fixed. (#132)

- **GUI shows old interface after upgrade** — WebView2 cached the previous version's HTML/CSS/JS to disk. After upgrading, users saw the old GUI with no BYOP panel or other v1.8.9 changes. The GUI now detects version changes and clears the WebView2 cache on first launch. Your settings (Local Storage) are preserved. (#236)

- **Icon setting crashes on first launch** — On 64-bit Windows, the taskbar icon function could fail with `OverflowError: int too long to convert` because Win32 API functions were called without proper type declarations. Non-blocking (the app still works), but now fixed properly. (#235)

## New

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
