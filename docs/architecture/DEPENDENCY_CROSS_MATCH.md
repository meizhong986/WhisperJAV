# WhisperJAV Dependency Cross-Match Table

**Generated**: 2026-03-11
**Sources checked**: pyproject.toml, registry.py, requirements.txt, requirements.txt.template

## How to Read This Table

- **pip name**: The `pip install` package name
- **import name**: The Python import name (only shown if different from pip name)
- **extra**: Which pyproject.toml extra group the package belongs to
- **pyproject.toml**: Version constraint in pyproject.toml
- **registry.py**: Version constraint in whisperjav/installer/core/registry.py
- **requirements.txt**: Version constraint in requirements.txt (legacy)
- **template**: Version constraint in installer/templates/requirements.txt.template
- **status**: `SYNC` = all sources agree, `N/A` = not present in that source (by design)

## Declared Dependencies (in pyproject.toml + registry)

### Core Dependencies (`dependencies` / `Extra.CORE`)

| # | pip name | import name | pyproject.toml | registry.py | requirements.txt | template | status |
|---|----------|-------------|----------------|-------------|------------------|----------|--------|
| 1 | openai-whisper | whisper | git (openai/whisper) | git (openai/whisper) | git (openai/whisper) | N/A (git phase) | SYNC |
| 2 | stable-ts | stable_whisper | git (meizhong986/stable-ts) | git (meizhong986/stable-ts) | git (meizhong986/stable-ts) | N/A (git phase) | SYNC |
| 3 | faster-whisper | faster_whisper | >=1.1.0 | >=1.1.0 | >=1.1.0 | >=1.1.0 | SYNC |
| 4 | ffmpeg-python | | git (kkroening/ffmpeg-python) | git (kkroening/ffmpeg-python) | git (kkroening/ffmpeg-python) | N/A (git phase) | SYNC |
| 5 | pysrt | | (none) | (none) | (none) | (none) | SYNC |
| 6 | srt | | (none) | (none) | (none) | (none) | SYNC |
| 7 | tqdm | | (none) | (none) | (none) | (none) | SYNC |
| 8 | colorama | | (none) | (none) | (none) | (none) | SYNC |
| 9 | requests | | (none) | (none) | (none) | (none) | SYNC |
| 10 | aiofiles | | (none) | (none) | (none) | (none) | SYNC |
| 11 | regex | | (none) | (none) | (none) | (none) | SYNC |
| 12 | tiktoken | | >=0.7.0 | >=0.7.0 | N/A | N/A | SYNC |
| 13 | more-itertools | more_itertools | >=10.0 | >=10.0 | N/A | N/A | SYNC |
| 14 | pydantic | | >=2.0,<3.0 | >=2.0,<3.0 | >=2.0,<3.0 | >=2.0,<3.0 | SYNC |
| 15 | PyYAML | yaml | >=6.0 | >=6.0 | N/A | >=6.0 | SYNC |
| 16 | jsonschema | | (none) | (none) | (none) | (none) | SYNC |
| 17 | torch | | (none, INDEX_URL) | (none, INDEX_URL) | N/A (separate) | N/A (separate) | SYNC |
| 18 | torchaudio | | (none, INDEX_URL) | (none, INDEX_URL) | N/A (separate) | N/A (separate) | SYNC |
| 19 | torchvision | | (none, INDEX_URL) | (none, INDEX_URL) | N/A (separate) | N/A (separate) | SYNC |

### CLI Extra (`[cli]` / `Extra.CLI`)

| # | pip name | import name | pyproject.toml | registry.py | requirements.txt | template | status |
|---|----------|-------------|----------------|-------------|------------------|----------|--------|
| 20 | soundfile | | (none) | (none) | (none) | (none) | SYNC |
| 21 | pydub | | (none) | (none) | (none) | (none) | SYNC |
| 22 | numpy | | >=2.0.0 | >=2.0.0 | >=2.0.0 | >=2.0.0 | SYNC |
| 23 | scipy | | >=1.13.0 | >=1.13.0 | >=1.13.0 | >=1.13.0 | SYNC |
| 24 | librosa | | >=0.11.0 | >=0.11.0 | >=0.11.0 | >=0.11.0 | SYNC |
| 25 | pyloudnorm | | (none) | (none) | (none) | (none) | SYNC |
| 26 | auditok | | (none) | (none) | (none) | (none) | SYNC |
| 27 | silero-vad | silero_vad | >=6.2 | >=6.2 | >=6.2 | >=6.2 | SYNC |
| 28 | ten-vad | ten_vad | (none) | (none) | (none) | (none) | SYNC |
| 29 | numba | | >=0.60.0 | >=0.60.0 | >=0.60.0 | >=0.60.0 | SYNC |
| 30 | psutil | | >=5.9.0 | >=5.9.0 | N/A | >=5.9.0 | SYNC |
| 31 | scikit-learn | sklearn | >=1.4.0 | >=1.4.0 | N/A | >=1.4.0 | SYNC |

### GUI Extra (`[gui]` / `Extra.GUI`)

| # | pip name | import name | pyproject.toml | registry.py | requirements.txt | template | status |
|---|----------|-------------|----------------|-------------|------------------|----------|--------|
| 32 | pywebview | webview | >=5.0.0 | >=5.0.0 | >=5.0.0 | >=5.0.0 | SYNC |
| 33 | pythonnet | clr | >=3.0 (win32) | >=3.0 (win32) | >=3.0 (win32) | >=3.0 (win32) | SYNC |
| 34 | pywin32 | win32com | >=305 (win32) | >=305 (win32) | >=305 (win32) | >=305 (win32) | SYNC |

### Translate Extra (`[translate]` / `Extra.TRANSLATE`)

| # | pip name | import name | pyproject.toml | registry.py | requirements.txt | template | status |
|---|----------|-------------|----------------|-------------|------------------|----------|--------|
| 35 | pysubtrans | PySubtrans | >=1.5.0 | >=1.5.0 | >=1.5.0 | >=1.5.0 | SYNC |
| 36 | openai | | >=1.35.0 | >=1.35.0 | >=1.35.0 | >=1.35.0 | SYNC |
| 37 | google-genai | | >=1.39.0 | >=1.39.0 | >=1.39.0 | >=1.39.0 | SYNC |
| 38 | google-api-core | google.api_core | >=2.14.0 | >=2.14.0 | N/A | N/A | SYNC |

### LLM Extra (`[llm]` / `Extra.LLM`)

| # | pip name | import name | pyproject.toml | registry.py | requirements.txt | template | status |
|---|----------|-------------|----------------|-------------|------------------|----------|--------|
| 39 | uvicorn | | >=0.22.0 | >=0.22.0 | N/A | N/A | SYNC |
| 40 | fastapi | | >=0.100.0 | >=0.100.0 | N/A | N/A | SYNC |
| 41 | pydantic-settings | | >=2.0.1 | >=2.0.1 | N/A | N/A | SYNC |
| 42 | sse-starlette | | >=1.6.1 | >=1.6.1 | N/A | N/A | SYNC |
| 43 | starlette-context | | >=0.3.6,<0.4 | >=0.3.6,<0.4 | N/A | N/A | SYNC |

### Enhance Extra (`[enhance]` / `Extra.ENHANCE`)

| # | pip name | import name | pyproject.toml | registry.py | requirements.txt | template | status |
|---|----------|-------------|----------------|-------------|------------------|----------|--------|
| 44 | modelscope | | >=1.20 | >=1.20 | >=1.20 | >=1.20 | SYNC |
| 45 | setuptools | pkg_resources | >=61.0,<82 | >=61.0,<82 | N/A | N/A | SYNC |
| 46 | einops | | (none) | (none) | N/A | N/A | SYNC |
| 47 | oss2 | | (none) | (none) | N/A | oss2 | SYNC |
| 48 | addict | | (none) | (none) | (none) | (none) | SYNC |
| 49 | attrs | | (none) | (none) | N/A | N/A | SYNC |
| 50 | datasets | | >=3.0.0,<=3.6.0 | >=3.0.0,<=3.6.0 | >=3.0.0,<=3.6.0 | >=3.0.0,<=3.6.0 | SYNC |
| 51 | simplejson | | (none) | (none) | (none) | (none) | SYNC |
| 52 | sortedcontainers | | (none) | (none) | (none) | (none) | SYNC |
| 53 | packaging | | (none) | (none) | (none) | (none) | SYNC |
| 54 | Pillow | PIL | (none) | (none) | N/A | Pillow | SYNC |
| 55 | clearvoice | | git (meizhong986/ClearerVoice) | git (meizhong986/ClearerVoice) | git (meizhong986/ClearerVoice) | N/A (git phase) | SYNC |
| 56 | gdown | | (none) | (none) | N/A | N/A | SYNC |
| 57 | joblib | | (none) | (none) | N/A | N/A | SYNC |
| 58 | opencv-python | cv2 | >=4.10.0 | >=4.10.0 | N/A | N/A | SYNC |
| 59 | python-speech-features | python_speech_features | >=0.6 | >=0.6 | N/A | N/A | SYNC |
| 60 | rotary-embedding-torch | rotary_embedding_torch | >=0.8 | >=0.8 | N/A | N/A | SYNC |
| 61 | scenedetect | | >=0.6 | >=0.6 | N/A | N/A | SYNC |
| 62 | torchinfo | | (none) | (none) | N/A | N/A | SYNC |
| 63 | yamlargparse | | (none) | (none) | N/A | N/A | SYNC |
| 64 | bs-roformer-infer | bs_roformer | (none) | (none) | (none) | bs-roformer-infer | SYNC |
| 65 | onnxruntime | | >=1.16.0 | >=1.16.0 | >=1.16.0 | >=1.16.0 | SYNC |

### HuggingFace Extra (`[huggingface]` / `Extra.HUGGINGFACE`)

| # | pip name | import name | pyproject.toml | registry.py | requirements.txt | template | status |
|---|----------|-------------|----------------|-------------|------------------|----------|--------|
| 66 | huggingface-hub | | >=0.25.0 | >=0.25.0 | >=0.25.0 | >=0.25.0 | SYNC |
| 67 | transformers | | >=4.40.0 | >=4.40.0 | N/A | >=4.40.0 | SYNC |
| 68 | accelerate | | >=0.26.0 | >=0.26.0 | N/A | >=0.26.0 | SYNC |
| 69 | hf_xet | | (none) | (none) | (none) | hf_xet | SYNC |

### Qwen Extra (`[qwen]` / `Extra.QWEN`)

| # | pip name | import name | pyproject.toml | registry.py | requirements.txt | template | status |
|---|----------|-------------|----------------|-------------|------------------|----------|--------|
| 70 | qwen-asr | qwen_asr | >=0.0.6 | >=0.0.6 | N/A | >=0.0.6 | SYNC |

### Analysis Extra (`[analysis]` / `Extra.ANALYSIS`)

| # | pip name | import name | pyproject.toml | registry.py | requirements.txt | template | status |
|---|----------|-------------|----------------|-------------|------------------|----------|--------|
| 71 | matplotlib | | (none) | (none) | N/A | matplotlib | SYNC |

### Compatibility Extra (`[compatibility]` / `Extra.COMPATIBILITY`)

| # | pip name | import name | pyproject.toml | registry.py | requirements.txt | template | status |
|---|----------|-------------|----------------|-------------|------------------|----------|--------|
| 72 | av | | >=13.0.0 | >=13.0.0 | >=13.0.0 | >=13.0.0 | SYNC |
| 73 | imageio | | >=2.31.0 | >=2.31.0 | >=2.31.0 | >=2.31.0 | SYNC |
| 74 | imageio-ffmpeg | | >=0.4.9 | >=0.4.9 | >=0.4.9 | >=0.4.9 | SYNC |
| 75 | httpx | | >=0.27.0 | >=0.27.0 | >=0.27.0 | >=0.27.0 | SYNC |
| 76 | websockets | | >=13.0 | >=13.0 | >=13.0 | >=13.0 | SYNC |
| 77 | soxr | | >=0.3.0 | >=0.3.0 | >=0.3.0 | >=0.3.0 | SYNC |

### Dev Extra (`[dev]` / `Extra.DEV`)

| # | pip name | import name | pyproject.toml | registry.py | requirements.txt | template | status |
|---|----------|-------------|----------------|-------------|------------------|----------|--------|
| 78 | pytest | | >=7.0 | >=7.0 | N/A | N/A | SYNC |
| 79 | pytest-cov | | (none) | (none) | N/A | N/A | SYNC |
| 80 | ruff | | >=0.1.0 | >=0.1.0 | N/A | N/A | SYNC |
| 81 | pre-commit | | (none) | (none) | N/A | N/A | SYNC |

---

## Undeclared Dependencies (imported but NOT in pyproject.toml/registry)

These are packages imported in the codebase but intentionally NOT declared as dependencies.
All are **conditional imports** (try/except guarded) — they are optional features that
users must install manually if they want to use them.

| # | pip name | import name | used in | why undeclared |
|---|----------|-------------|---------|----------------|
| U1 | llama-cpp-python | llama_cpp | translate/local_backend.py | Installed separately with CUDA detection; listed in requirements.txt only |
| U2 | nemo_toolkit | nemo | modules/speech_segmentation/backends/nemo.py | Optional backend; user installs manually from NVIDIA git |
| U3 | omegaconf | omegaconf | modules/speech_segmentation/backends/nemo.py | Transitive dep of nemo_toolkit |
| U4 | wget | wget | modules/speech_segmentation/backends/nemo.py | Optional dep of nemo; urllib fallback if missing |
| U5 | pyannote-audio | pyannote | utils/japanese_ero_voice_classifier_AkitoP_v1.py | Experimental classifier; not wired into pipelines |
| U6 | flash-attn | flash_attn | modules/qwen_asr.py | Optional GPU optimization for Qwen; try/except guarded |
| U7 | pynvml | pynvml | installer/core/standalone.py | GPU detection fallback; nvidia-smi preferred |
| U8 | tomli | tomli | installer/validation/sync.py | Backport for Python <3.11; stdlib tomllib used on 3.11+ |

---

## Import Name Mismatches (pip name ≠ import name)

| pip name | import name |
|----------|-------------|
| openai-whisper | whisper |
| stable-ts | stable_whisper |
| faster-whisper | faster_whisper |
| ffmpeg-python | ffmpeg |
| silero-vad | silero_vad |
| ten-vad | ten_vad |
| bs-roformer-infer | bs_roformer |
| scikit-learn | sklearn |
| pywebview | webview |
| pythonnet | clr |
| pywin32 | win32com |
| PyYAML | yaml |
| Pillow | PIL |
| opencv-python | cv2 |
| python-speech-features | python_speech_features |
| rotary-embedding-torch | rotary_embedding_torch |
| more-itertools | more_itertools |
| pysubtrans | PySubtrans |
| google-api-core | google.api_core |
| qwen-asr | qwen_asr |
| setuptools | pkg_resources |
| hf_xet | hf_xet |

---

## Summary

- **81 declared packages** across 11 extras (core + 10 optional)
- **8 undeclared packages** (all conditional/optional, by design)
- **22 import name mismatches** (all documented in registry.py)
- **All declared sources are in SYNC** (pyproject.toml = registry.py = requirements.txt where applicable)
