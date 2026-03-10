# WhisperJAV v1.8.7 Release Notes

## Summary

v1.8.7 is a stability and robustness release focused on installer reliability, dependency completeness, and bug fixes across transcription, translation, and speech enhancement.

## Highlights

- **Installer robustness**: Fixed constraint file failures for CPU and GPU installs, added proxy detection for users behind corporate/regional firewalls, and completed all missing transitive dependencies
- **China network resilience** (#204): 3-step fallback for model downloads — normal HF Hub, local cache, then hf-mirror.com community mirror
- **Apple Silicon (MPS)** (#198): Device detection for MPS in TransformersASR, deferred generate_kwargs to model defaults
- **Repetition hallucination fix** (#209): 4 regex gaps patched in RepetitionCleaner plus generic safety net
- **Translation fix** (#143): Custom provider no longer crashes when API key env var is None

## Bug Fixes

### Installer
- **fix: GPU constraint file fails with local version suffixes** — `torch==2.10.0+cpu` and `torch==2.4.0+cu128` now correctly stripped to base version before writing constraints, preventing uv resolution failures
- **fix: `--no-deps` for git packages** — openai-whisper, stable-ts, ffmpeg-python, and clearvoice now install without dependency resolution, eliminating the risk of torch CUDA→CPU replacement during Phase 3.5
- **fix: missing transitive dependencies** — Added tiktoken, more-itertools (openai-whisper deps) and 8 ClearVoice deps (gdown, joblib, opencv-python, python-speech-features, rotary-embedding-torch, scenedetect, torchinfo, yamlargparse) to requirements.txt for --no-deps safety
- **feat: torchvision in Phase 3** — Now installed alongside torch/torchaudio from the correct CUDA index, required by ClearVoice
- **fix: proxy detection for bundled git** (#210) — Reads Windows system proxy from registry and propagates to bundled git; fixes DNS resolution failures for users in China and corporate networks
- **fix: broadened git network error patterns** — Now catches DNS resolution, SSL certificate, and connection refused errors (not just timeouts)
- **fix: mac installer auto-clone** — `curl | bash` standalone installs now auto-clone the repo

### Transcription
- **fix(#198): MPS device support** — TransformersASR now detects and uses Apple Silicon Metal/MPS acceleration
- **fix(#198): deferred generate_kwargs** — HF Transformers beam search crash on MPS fixed by deferring to model defaults
- **fix(#209): repetition cleaner** — Fixed 4 regex gaps (dakuten variants, phrase length, separator handling) plus added generic safety net for uncaught repetitions

### Translation
- **fix(#143): custom provider NoneType crash** — `os.getenv(None)` no longer crashes when API key env var is not configured; defaults to "not-needed" for local providers like Ollama

### Speech Enhancement
- **fix: ZipEnhancer GUI option** — Removed colon suffix from GUI option values
- **fix: setuptools pin** — `setuptools>=61.0,<82` prevents pkg_resources removal breaking ModelScope

### Other
- **feat: `--vad-threshold` and `--speech-pad-ms` CLI flags** (#159)
- **feat: `--stream` flag in main.py CLI**
- **feat: `--enhance-for-vad` wired into GUI** (Qwen passes)
- **fix: `errors='replace'` in audio extraction** (#195)
- **fix: repeated 'Process Completed' messages after LLM translation**

## Dependency Changes

| Change | Package | Extra | Reason |
|--------|---------|-------|--------|
| **Added** | tiktoken>=0.7.0 | core | openai-whisper tokenizer (was transitive, now explicit) |
| **Added** | more-itertools>=10.0 | core | openai-whisper dep (was transitive, now explicit) |
| **Added** | torchvision | core | ClearVoice requirement; installed from CUDA index |
| **Added** | gdown | enhance | ClearVoice model downloads |
| **Added** | joblib | enhance | ClearVoice transitive dep |
| **Added** | opencv-python>=4.0 | enhance | ClearVoice requirement |
| **Added** | python-speech-features>=0.6 | enhance | ClearVoice feature extraction |
| **Added** | rotary-embedding-torch>=0.8 | enhance | ClearVoice model architecture |
| **Added** | scenedetect>=0.6 | enhance | ClearVoice requirement |
| **Added** | torchinfo | enhance | ClearVoice model utility |
| **Added** | yamlargparse | enhance | ClearVoice configuration |

Total dependencies: 60 → 70

## Known Issues

- **Group A (#196/#198/#132)**: LLM translation "No matches found" on long files — deferred to v1.8.8. Root cause analysis complete (cascading token overflow). Workaround: use shorter batch sizes or cloud translation providers.
- **numpy<2.0 pin**: Still required for ModelScope compatibility. Migration to numpy 2.x planned for v1.8.8+.

## Compatibility

```
Python:       3.10, 3.11, 3.12
PyTorch:      2.4.0 - 2.9.x
CTranslate2:  4.5.0 - 4.7.x (requires cuDNN 9)
CUDA:         12.1+ (12.4+ recommended)
cuDNN:        9.x
NumPy:        1.26.x
```

## Upgrade

```bash
# Windows installer users: download new .exe from Releases
# pip users:
pip install --upgrade "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git@v1.8.7"
# GUI upgrade:
whisperjav-upgrade
```
