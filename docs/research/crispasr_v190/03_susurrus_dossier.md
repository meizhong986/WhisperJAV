# Susurrus — Phase 0 Dossier

Primary-source research dossier on the Susurrus repository
(`https://github.com/CrispStrobe/Susurrus`). Descriptive only; cites every claim
with file path + line refs or URL. The single highest-value artifact is the
CrispASR backend wrapper (section 12), the only public Python host that already
invokes the `crispasr` binary end-to-end. Citations refer to commit
`7073a77` (default branch `main`, latest commit dated 2026-04-19).

---

## 0. Snapshot

- **Repo URL**: https://github.com/CrispStrobe/Susurrus
- **Default branch**: `main`
- **Latest commit hash (HEAD of `main`)**: `7073a77f298df18b016cdc1bfc24cddac2ad63d1`
- **Latest commit date**: 2026-04-19T08:38:11Z
- **License**: MIT (SPDX: `MIT`, file `LICENSE`, copyright "2026 CrispStrobe")
- **Total commits on `main`**: 7 (`gh api repos/CrispStrobe/Susurrus/commits` Link header
  reports `rel="last"` page=7 at per_page=1)
- **Contributors**: 2 — `CrispStrobe` (1 commit, the initial squash-style commit) and
  `terribleperson` (1 commit, typo fix PR #1)
- **Stars**: 16
- **Forks**: 2
- **Watchers**: 16
- **Subscribers**: 1
- **Open issues**: 0
- **Releases**: 0 (`gh api repos/CrispStrobe/Susurrus/releases` returns `[]`)
- **Tags**: 0 (`gh api repos/CrispStrobe/Susurrus/tags` returns `[]`)
- **`setup.py` version string**: `1.1.0` (`setup.py:5`, also `__init__.py:3`)
- **Topics declared on GitHub**: `ctranslate2`, `diarization`, `pyannote`,
  `pyannote-audio`, `speech-to-text`, `stt`, `voxtral`, `whisper`, `whisper-ai`,
  `whisper-cpp`, `whispercpp`
- **GitHub description**: "speech to text gui for different (mostly Whisper, also
  Voxtral) models and backends, including whisper.cpp, mlx-whisper,
  faster-whisper, ctranslate2; applies pyannote for diarization"

**Stated relationship to CrispASR**: README (`README.md`, "Part of the Crisp
ecosystem" table) lists CrispASR as a separate sibling project ("C++ ASR engine
— 11 backends, ggml inference. Available as a Susurrus backend (auto-downloads
if not found).") Both repos are owned by GitHub user `CrispStrobe` (id
`154636388`). The three Susurrus commits that introduce the CrispASR backend
(`1048d6d`, `0af3e7e`, `7073a77`) are authored by the placeholder identity
`crispasr integration <crispasr-dev@localhost>` and co-authored by
`Claude Opus 4.6 (1M context) <noreply@anthropic.com>`.

---

## 1. Repository structure

Directory tree (verified against `gh api repos/CrispStrobe/Susurrus/contents/...`).

```
Susurrus/
├── .env.example                 (95 bytes)
├── .flake8                      (598 bytes)
├── .gitignore                   (7160 bytes)
├── .pre-commit-config.yaml      (1274 bytes)
├── LICENSE                      (1068 bytes, MIT)
├── README.md                    (14692 bytes)
├── __init__.py                  (120 bytes)
├── __main__.py                  (112 bytes)
├── check_imports.py             (13211 bytes)
├── cli.py                       (6296 bytes)
├── config.py                    (6509 bytes)
├── main.py                      (1027 bytes)         ← GUI entry point
├── mypi.ini                     (786 bytes)          (sic — typo of mypy.ini)
├── pylint.rc                    (939 bytes)
├── pyproject.toml               (725 bytes)
├── requirements-dev.txt         (337 bytes)
├── setup.py                     (386 bytes)
├── test_backends.py             (359 bytes)
├── test_voxtral.py              (3809 bytes)
├── backends/
│   ├── __init__.py              (22 bytes)
│   └── transcription/
│       ├── __init__.py          (106 bytes)
│       ├── voxtral_api.py       (8621 bytes)
│       └── voxtral_local.py     (11353 bytes)
├── gui/
│   ├── __init__.py              (18 bytes)
│   ├── main_window.py           (40439 bytes)
│   ├── dialogs/
│   │   ├── __init__.py          (253 bytes)
│   │   ├── cuda_diagnostics_dialog.py    (6218 bytes)
│   │   ├── dependencies_dialog.py        (3694 bytes)
│   │   └── installer_dialog.py           (12280 bytes)
│   └── widgets/
│       ├── __init__.py          (244 bytes)
│       ├── advanced_options.py           (4703 bytes)
│       ├── collapsible_box.py            (1418 bytes)
│       ├── diarization_settings.py       (4458 bytes)
│       └── voxtral_settings.py           (4162 bytes)
├── models/
│   ├── __init__.py              (233 bytes)
│   └── model_config.py          (16214 bytes)
├── scripts/
│   ├── check_imports.py         (12151 bytes)
│   ├── cleanup_imports.sh       (1229 bytes)
│   ├── fix_all.sh               (565 bytes)
│   ├── fix_all_complete.sh      (1001 bytes)
│   ├── fix_all_imports.sh       (774 bytes)
│   ├── fix_syntax_errors.py     (3918 bytes)
│   ├── fix_unused_imports.py    (5550 bytes)
│   ├── generate_dependency_graph.py      (1226 bytes)
│   ├── lint_all.sh              (2406 bytes)
│   └── pyannote_torch26.py      (18542 bytes)
├── tests/
│   ├── integration/             (empty)
│   └── unit/                    (empty)
├── utils/
│   ├── __init__.py              (192 bytes)
│   ├── audio_utils.py           (7299 bytes)
│   ├── dependency_check.py      (9497 bytes)
│   ├── device_detection.py      (8780 bytes)
│   ├── download_utils.py        (9376 bytes)
│   └── format_utils.py          (812 bytes)
└── workers/
    ├── __init__.py              (241 bytes)
    ├── diarize_audio.py         (3008 bytes)
    ├── diarize_worker.py        (17056 bytes)
    ├── diarize_worker_thin.py   (322 bytes)
    ├── transcribe_worker.py     (5302 bytes)
    ├── transcribe_worker_bkp.py (52009 bytes)  ← backup of earlier monolithic worker
    ├── transcription_thread.py  (15644 bytes)
    └── transcription/
        ├── __init__.py          (144 bytes)
        ├── utils.py             (3991 bytes)
        └── backends/
            ├── __init__.py      (1516 bytes)
            ├── base.py          (880 bytes)
            ├── crispasr_backend.py            (9585 bytes)  ← TARGET
            ├── ctranslate2_backend.py         (5250 bytes)
            ├── faster_whisper_backend.py      (5205 bytes)
            ├── insanely_fast_backend.py       (4585 bytes)
            ├── mlx_backend.py                 (2185 bytes)
            ├── openai_whisper_backend.py      (5378 bytes)
            ├── transformers_backend.py        (2847 bytes)
            ├── voxtral_backend.py             (3273 bytes)
            ├── whisper_cpp_backend.py         (6892 bytes)
            └── whisper_jax_backend.py         (2658 bytes)
```

**Note on README accuracy**: the README's "Architecture Overview" tree
(`README.md` §Development) describes a `backends/diarization/` subdirectory with
`manager.py` and `progress.py`. Those files do not exist on `main` at
`7073a77`. The diarization code lives in `workers/diarize_worker.py` which
imports `from .manager import DiarizationManager`
(`workers/diarize_worker.py:18`), but `workers/manager.py` is also not present
in the tree. This is observable drift between README and source.

**Entry points**:
- `main.py` — GUI entry (boots `PyQt6.QtWidgets.QApplication`, instantiates
  `gui.main_window.MainWindow`). `main.py:30-31`.
- `__main__.py` — package-level entry (delegates to `main.main()`),
  size 112 bytes.
- `cli.py` — headless CLI entry that bypasses PyQt6 entirely.
- `workers/transcribe_worker.py` — subprocess worker invoked by
  `TranscriptionThread` (see §9).
- `workers/diarize_worker.py` — subprocess worker for diarization+ASR.
- `setup.py` `console_scripts` registers `susurrus=main:main`
  (`setup.py:13-15`).

---

## 2. License + redistribution

`LICENSE` is the standard MIT License, "Copyright (c) 2026 CrispStrobe", with
the boilerplate permission/disclaimer paragraphs. `README.md` final section
("License") restates "MIT — see [LICENSE](LICENSE)." and adds the disclaimer:
"Model licenses vary. Most ASR models (Whisper, Parakeet, Canary, Voxtral,
Qwen3-ASR) are permissive (MIT/Apache/CC-BY). Pyannote
speaker-diarization-3.1 is MIT. Check the individual model card on HuggingFace
for the exact terms before commercial deployment."

---

## 3. Build/install

- **`setup.py`** (full text, `setup.py:1-17`):
  ```python
  from setuptools import find_packages, setup
  setup(
      name="susurrus",
      version="1.1.0",
      packages=find_packages(),
      install_requires=[
          "PyQt6>=6.0.0",
          "torch>=1.9.0",
          "torchaudio>=0.9.0",
          "requests>=2.25.1",
          "pydub>=0.25.1",
      ],
      entry_points={
          "console_scripts": [
              "susurrus=main:main",
          ],
      },
  )
  ```
  Runtime dependencies declared in `setup.py` are only PyQt6, torch, torchaudio,
  requests, pydub. No `requirements.txt` file is present in the repo despite
  the README's "`pip install -r requirements.txt`" instruction (`README.md`
  §Quick Start). Backend libraries (faster-whisper, mlx-whisper, ctranslate2,
  whisper-jax, etc.) are not declared as install_requires — README presents
  them as optional manual `pip install` lines per backend (`README.md` §Optional
  Backend Installation).

- **`pyproject.toml`** contains only `[tool.black]`, `[tool.isort]`,
  `[tool.pytest.ini_options]`, `[tool.coverage.run]` — no `[build-system]`,
  no `[project]` table. It is a tool-config file only.

- **`requirements-dev.txt`** (337 bytes): flake8, pylint, mypy, isort, black,
  bandit, pydocstyle, pyflakes, vulture, pydeps, pre-commit, pytest, pytest-cov,
  pytest-mock.

- **Installer mechanism**: `pip` only. There is no `uv.lock`, no
  `[tool.uv.sources]`, no conda recipe. README references `pip install` and
  manual platform tooling (`choco`, `brew`, `apt`).

- **Python version**: README states "Python 3.8+" (`README.md` §Prerequisites).
  `pyproject.toml` `[tool.black]` `target-version` lists `py39`, `py310`, `py311`.

- **Platform-specific install notes** (README §Platform-Specific Setup):
  Windows uses Chocolatey for cmake/ffmpeg/git/python/cuda; macOS uses Homebrew
  plus optional `pip install mlx mlx-whisper`; Linux uses apt for
  ffmpeg/cmake/build-essential/python3/python3-pip/git plus a manual CUDA install.
  Voxtral support requires installing the dev branch of transformers
  (`pip install git+https://github.com/huggingface/transformers.git`) plus
  `mistral-common[audio]` and `soundfile` (`README.md` §Optional Backend
  Installation, last block).

---

## 4. Python orchestration architecture

**Top-level app structure** (verified at `main.py:1-35`):

1. `main.py` configures `logging.basicConfig` at INFO level, imports
   `PyQt6.QtWidgets.QApplication`, calls
   `utils.dependency_check.check_ffmpeg_installation()`, instantiates
   `gui.main_window.MainWindow`, and enters the Qt event loop with
   `sys.exit(app.exec())`.

2. `MainWindow.__init__` (`gui/main_window.py:24-43`) sets up:
   - QSettings via `config.get_settings()` (uses `QSettings("CrispStrobe",
     "Susurrus")` — see `config.py:80-82`)
   - `BACKEND_MODEL_MAP` imported from `config.py`
   - Environment-token detection for `HF_TOKEN` and `MISTRAL_API_KEY`
   - Diagnostics: `_run_diagnostics()` calls `check_ffmpeg_installation`,
     `check_dependencies`, `check_nvidia_installation`
     (`gui/main_window.py:45-60`)

3. **Job flow** (GUI → worker → backend → output):
   - User selects backend, model, audio source in `MainWindow`.
   - On button click, `MainWindow` constructs an `args` dict and starts a
     `TranscriptionThread` (PyQt6 `QThread`) — `workers/transcription_thread.py`.
   - `TranscriptionThread.run()` branches on `args["diarization_enabled"]`:
     - If false, calls `_run_standard_transcription()`
       (`workers/transcription_thread.py:215-371`).
     - If true, calls `_run_diarization()`
       (`workers/transcription_thread.py:52-213`).
   - Both branches build a command-line `cmd` list and launch
     `transcribe_worker.py` or `diarize_worker.py` as a separate Python process
     via `subprocess.Popen` (`workers/transcription_thread.py:303-309`).
   - The worker subprocess imports
     `workers.transcription.backends.get_backend()`, instantiates the chosen
     backend, calls `backend.preprocess_audio()`, then iterates
     `backend.transcribe(audio_path)` and writes each `(start, end, text)`
     tuple to stdout with `print(f"[{start:.3f} --> {end:.3f}] {text}",
     flush=True)` (`workers/transcribe_worker.py:96-105`).
   - The `TranscriptionThread` parses subprocess stdout in real time
     (regex `^\[([^\]]+?)\]\s*(.*)` — `workers/transcription_thread.py:317`),
     emits `progress_signal` / `transcription_replace_signal` /
     `error_signal` / `diarization_signal` PyQt signals back to the GUI.

The pipeline is therefore: **GUI (Qt main thread) → QThread
(`TranscriptionThread`) → subprocess (`transcribe_worker.py` or
`diarize_worker.py`) → in-process backend instance → stdout lines parsed back to
the GUI via Qt signals**.

---

## 5. TranscriptionBackend base class

**File**: `workers/transcription/backends/base.py` (880 bytes — full file
quoted below verbatim).

```python
# workers/transcription/backends/base.py:
"""Base class for transcription backends"""
from abc import ABC, abstractmethod

class TranscriptionBackend(ABC):
    """Base class for all transcription backends"""

    def __init__(self, model_id, device, language=None, **kwargs):
        self.model_id = model_id
        self.device = device
        self.language = language
        self.kwargs = kwargs

    @abstractmethod
    def transcribe(self, audio_path):
        """Transcribe audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Generator yielding (start, end, text) tuples or text lines
        """
        pass

    def preprocess_audio(self, audio_path):
        """Preprocess audio if needed"""
        return audio_path

    def cleanup(self):
        """Cleanup resources"""
        pass
```

- **Constructor contract**: positional `model_id`, `device`; keyword
  `language=None`; `**kwargs` is captured into `self.kwargs` (but most
  subclasses do not rely on `self.kwargs`; they accept their own keyword args
  explicitly).
- **`transcribe(audio_path)`** is the only abstract method. Docstring states it
  is a generator yielding `(start, end, text)` tuples — but the docstring also
  says "or text lines". In practice subclasses (e.g. `CrispasrBackend`,
  `WhisperCppBackend`, `MLXBackend`) sometimes yield `(0.0, 0.0, line)` for
  lines they cannot parse a timestamp from.
- **`preprocess_audio(audio_path)`** is a concrete default that returns the
  input unchanged. Subclasses override (e.g. `CrispasrBackend.preprocess_audio`
  passes through WAV/MP3/FLAC/OGG and converts other formats via
  `utils.audio_utils.convert_audio_to_wav`).
- **`cleanup()`** is a concrete no-op default. Subclasses override to delete
  temporary files (e.g. `CrispasrBackend.cleanup` deletes
  `self.temp_files`).
- **Type hints / return types**: there are no type hints on the base class.
  `audio_path` is documented as "Path to audio file" but typed as plain
  positional. No `typing.Generator` annotation, no `pathlib` types.
- **No lifecycle hooks** beyond `preprocess_audio` and `cleanup`. There is no
  `load_model` / `unload_model` / `__enter__` / `__exit__` / capability
  introspection on the base class.

---

## 6. BACKEND_MODEL_MAP registry

The registry lives in `config.py`. It is a `dict[str, list[tuple[str, str]]]`
mapping a backend name to a list of `(model_id, original_hf_id)` pairs.

```python
# config.py:14-91 (verbatim, abridged structurally to show the pattern)
BACKEND_MODEL_MAP = {
    "mlx-whisper": [
        ("mlx-community/whisper-large-v3-turbo", "openai/whisper-large-v3-turbo"),
        ("mlx-community/whisper-large-v3-turbo-q4", "openai/whisper-large-v3-turbo"),
        ("mlx-community/whisper-tiny-mlx-4bit", "openai/whisper-tiny"),
        ...
    ],
    "faster-batched": [
        ("cstr/whisper-large-v3-turbo-german-int8_float32", "openai/whisper-large-v3-turbo"),
        ...
    ],
    "faster-sequenced": [ ... same models as faster-batched ... ],
    "transformers": [
        ("openai/whisper-large-v3", "openai/whisper-large-v3"),
        ...
    ],
    "OpenAI Whisper": [ ("large-v2", "openai/whisper-large-v2"), ... ],
    "whisper.cpp": [
        ("large-v3-turbo-q5_0", "openai/whisper-large-v3"),
        ("large-v3-turbo", "openai/whisper-large-v3"),
        ...
    ],
    "ctranslate2": [ ... ],
    "whisper-jax": [ ... ],
    "insanely-fast-whisper": [ ... ],
    "voxtral-local": [("mistralai/Voxtral-Mini-3B-2507", "mistralai/Voxtral-Mini-3B-2507")],
    "voxtral-api": [("voxtral-mini-latest", "voxtral-mini-latest")],
}
```

- **Schema**: backend-name string → list of `(model_id, original_hf_id)`. The
  second element is used elsewhere to look up the preprocessor / tokenizer when
  a quantized or community-fork model needs to fall back to the original
  HuggingFace ID. `config.get_default_model_for_backend(backend)` returns a
  hard-coded default per backend (`config.py:113-131`).
- **Notable**: `BACKEND_MODEL_MAP` in `config.py` does **not** include an entry
  for `crispasr`. The CrispASR backend is registered only in the runtime
  backend dispatcher `workers/transcription/backends/__init__.py:18` and in
  the `cli.py` argparse. There is no curated model list for CrispASR in
  `config.py` as of `7073a77`. The README's "Adding a New Backend" steps
  (`README.md` §Adding a New Backend) state step 5 is "Add to config.py
  BACKEND_MODEL_MAP", but this step is not performed for the CrispASR
  integration.
- **How a new backend registers itself** (per README §Adding a New Backend):
  1. Create a file in `workers/transcription/backends/`.
  2. Inherit from `TranscriptionBackend`.
  3. Implement `transcribe`, optional `preprocess_audio`, optional `cleanup`.
  4. Register in `workers/transcription/backends/__init__.py` (the `get_backend`
     function — see §7 below).
  5. Add to `config.py` `BACKEND_MODEL_MAP`.

---

## 7. Per-backend module organization

`workers/transcription/backends/__init__.py` (1516 bytes — full file quoted
verbatim):

```python
# workers/transcription/backends/__init__.py
"""Transcription backends"""
from .base import TranscriptionBackend
from .mlx_backend import MLXBackend
from .faster_whisper_backend import FasterWhisperBatchedBackend, FasterWhisperSequencedBackend
from .transformers_backend import TransformersBackend
from .whisper_cpp_backend import WhisperCppBackend
from .ctranslate2_backend import CTranslate2Backend
from .whisper_jax_backend import WhisperJaxBackend
from .insanely_fast_backend import InsanelyFastBackend
from .openai_whisper_backend import OpenAIWhisperBackend
from .voxtral_backend import VoxtralLocalBackend, VoxtralAPIBackend
from .crispasr_backend import CrispasrBackend

def get_backend(backend_name, **kwargs):
    """Get backend instance by name"""
    backends = {
        'mlx-whisper': MLXBackend,
        'faster-batched': FasterWhisperBatchedBackend,
        'faster-sequenced': FasterWhisperSequencedBackend,
        'transformers': TransformersBackend,
        'whisper.cpp': WhisperCppBackend,
        'ctranslate2': CTranslate2Backend,
        'whisper-jax': WhisperJaxBackend,
        'insanely-fast-whisper': InsanelyFastBackend,
        'openai whisper': OpenAIWhisperBackend,
        'voxtral-local': VoxtralLocalBackend,
        'voxtral-api': VoxtralAPIBackend,
        'crispasr': CrispasrBackend,
    }

    backend_class = backends.get(backend_name.lower())
    if not backend_class:
        raise ValueError(f"Unknown backend: {backend_name}")

    return backend_class(**kwargs)
```

Twelve backends register here (one of them, `voxtral_backend.py`, exports two
classes). Pattern across backends:
- Each backend file is a single module under
  `workers/transcription/backends/` named `<name>_backend.py`.
- The module imports `from .base import TranscriptionBackend`.
- It defines one class (sometimes two) that subclasses `TranscriptionBackend`.
- `transcribe()` is implemented as a generator (`yield`).
- Imports of optional heavy libraries (e.g. `faster_whisper`, `mlx_whisper`,
  `ctranslate2`) are performed **inside** `transcribe()`, not at module top,
  so that importing `__init__.py` does not require all libraries to be
  installed. Example from `MLXBackend.transcribe()`
  (`workers/transcription/backends/mlx_backend.py:17-21`):
  ```python
  try:
      import mlx_whisper
      from huggingface_hub import snapshot_download
  except ImportError:
      raise ImportError("mlx_whisper and huggingface_hub required for mlx-whisper backend")
  ```
  Same pattern at `faster_whisper_backend.py:19-22`,
  `crispasr_backend.py:23-26` (lazy import of `urllib.request`,
  `zipfile`, `tarfile` inside `_download_crispasr()`).

**Two-class mechanism**: `faster_whisper_backend.py` defines
`FasterWhisperBatchedBackend` and `FasterWhisperSequencedBackend` (different
inference paths through the same library); `voxtral_backend.py` defines
`VoxtralLocalBackend` (on-device) and `VoxtralAPIBackend` (Mistral cloud API).

---

## 8. GUI threading isolation (`transcription_thread.py`)

- **Threading model**: PyQt6 `QThread` subclass.
  `workers/transcription_thread.py:21`: `class TranscriptionThread(QThread):`.
- **Signals defined** (`workers/transcription_thread.py:24-27`):
  ```python
  progress_signal = pyqtSignal(str, str)        # (metrics, transcription)
  error_signal = pyqtSignal(str)
  transcription_replace_signal = pyqtSignal(str)
  diarization_signal = pyqtSignal(str)
  ```
- **No tensor work runs inside the QThread.** The QThread immediately spawns
  the actual transcription as a **separate Python subprocess**
  (`subprocess.Popen` on `transcribe_worker.py` or `diarize_worker.py` —
  `workers/transcription_thread.py:303-309` and `:121-126`). All heavy model
  loading and inference happens in that subprocess.
- The QThread's responsibilities are limited to:
  - Building the `cmd` argument list.
  - Spawning the subprocess with stdout/stderr pipes.
  - Spawning a daemon `threading.Thread` to consume stderr in parallel
    (`workers/transcription_thread.py:130-137` and `:307-315`).
  - Iterating stdout line-by-line, regex-matching timestamps, emitting
    PyQt signals.
- **Progress reporting** is per-line: every stdout/stderr line emits
  `progress_signal.emit(line, "")` or, for parsed timestamp lines,
  `progress_signal.emit("", text)`. Worker scripts use `flush=True` on every
  `print` (`workers/transcribe_worker.py:99`).
- **Termination**: `stop()` (`workers/transcription_thread.py:373-377`) sets
  `self._is_running = False` and calls `self.process.terminate(); self.process.wait()`
  on the subprocess.

---

## 9. Multiprocessing isolation (`transcribe_worker.py`)

- Susurrus does **not** use `multiprocessing.Process` or `multiprocessing.Pool`.
  It uses **subprocess** isolation — i.e., a fresh Python interpreter is
  launched via `subprocess.Popen([sys.executable, "-u", worker_script, ...])`.
- The command construction is in
  `workers/transcription_thread.py:285-301`:
  ```python
  python_executable = sys.executable
  worker_dir = os.path.dirname(os.path.abspath(__file__))
  transcribe_worker_path = os.path.join(worker_dir, "transcribe_worker.py")
  ...
  cmd = [
      python_executable,
      "-u",
      transcribe_worker_path,
      "--model-id", model_id,
      "--backend", backend,
      "--device", device_arg,
  ]
  ```
- **IPC mechanism**: stdout/stderr pipes. `subprocess.PIPE` for both, line-buffered
  (`bufsize=1`, `workers/transcription_thread.py:308`). Worker output uses
  ASCII text protocol:
  - `[{start:.3f} --> {end:.3f}] {text}` for segments
    (`workers/transcribe_worker.py:99`).
  - `OUTPUT FILE: <path>` for terminal-state output paths
    (`workers/transcription_thread.py:316`, regex-matched).
  - `OUTPUT FILE ({format}): <path>` for diarization outputs
    (`workers/transcription_thread.py:152-159`).
  - `DIARIZATION JSON: <path>` for diarization JSON
    (`workers/transcription_thread.py:160-162`).
- **Worker entry point** (`workers/transcribe_worker.py:21`):
  ```python
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  ```
  ensures the parent directory is importable so `from
  workers.transcription.backends import get_backend` resolves.
- **UTF-8 encoding** is forced in `workers/transcription/utils.py:5-6`:
  ```python
  sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
  sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
  ```
- **Why isolation matters** (descriptive — not stated in source comments):
  the subprocess pattern means heavy backend libraries (torch, mlx,
  faster-whisper, native binaries like whisper.cpp / crispasr) only ever load
  into a child interpreter and never share an address space with the PyQt6 GUI
  process. When the user clicks "Stop", `terminate()` kills the child cleanly.

---

## 10. VAD / diarization layers

- **pyannote integration**: `workers/diarize_worker.py:18` imports `from
  .manager import DiarizationManager, verify_authentication`. The actual
  `manager` module is **not present** in the repo at `7073a77` (see §1 note
  on README drift). README lists "PyAnnote.audio Integration: State-of-the-art
  diarization engine" (`README.md` §Speaker Diarization) and the Hugging Face
  token setup steps direct users to accept the license at
  `https://huggingface.co/pyannote/speaker-diarization`.
- **Diarization models offered** (`config.py:101`):
  ```python
  DIARIZATION_MODELS = ["Default", "Legacy", "English", "Chinese", "German", "Spanish", "Japanese"]
  ```
- **PyTorch 2.6 compatibility shim**: `scripts/pyannote_torch26.py` (18542
  bytes) — README documents it as the workaround for "PyTorch 2.6+ compatibility
  issues" (`README.md` §Troubleshooting).
- **VAD inside CrispASR**: the CrispASR wrapper exposes a `--vad` CLI flag that
  is passed through to the `crispasr` binary (`crispasr_backend.py:163-164`):
  ```python
  if self.vad:
      cmd.append("--vad")
  ```
  This is delegated to the binary, not implemented in Susurrus. Commit
  `1048d6d` notes the VAD support uses Silero VAD inside the crispasr binary.
- **VAD in faster-whisper backends**: `vad_filter=True` is hard-coded in both
  `FasterWhisperBatchedBackend.transcribe()` (`faster_whisper_backend.py:34`)
  and `FasterWhisperSequencedBackend.transcribe()`
  (`faster_whisper_backend.py:80`).
- No other separate VAD/segmentation tool is observable in the
  `workers/transcription/backends/` modules.

---

## 11. Auto-download mechanism for the CrispASR binary

**Source**: `workers/transcription/backends/crispasr_backend.py:19-128`.

- **Detection / discovery order**, in
  `_find_crispasr()` (`crispasr_backend.py:91-128`):
  1. `os.environ.get("CRISPASR_EXECUTABLE")` — env override
     (`crispasr_backend.py:94-96`).
  2. PATH lookup via `shutil.which("crispasr")` (`crispasr_backend.py:122-124`).
  3. `~/.local/bin/crispasr`, `/usr/local/bin/crispasr`
     (`crispasr_backend.py:99-103`).
  4. `<source_root>/whisper.cpp/build/bin/crispasr` and similar paths under
     `~/whisper.cpp` and `~/CrispASR` (`crispasr_backend.py:107-114`).
  5. The download cache directory `~/.cache/susurrus/crispasr/crispasr[.exe]`
     and the unpacked subdirectories
     `crispasr-linux-x86_64/`, `crispasr-macos/`, `crispasr-windows-x86_64/`
     (`crispasr_backend.py:117-120`).
  6. If nothing was found, `_download_crispasr()` is called
     (`crispasr_backend.py:127-128`).

- **Download source**:
  `_GITHUB_RELEASE_URL = "https://github.com/CrispStrobe/CrispASR/releases/latest/download"`
  (`crispasr_backend.py:21`). The full asset URL pattern is
  `{_GITHUB_RELEASE_URL}/{asset}` — always the **latest** release; no version
  pinning of any kind. The wrapper does not record or assert which release it
  pulled.

- **Per-platform asset selection** (`crispasr_backend.py:33-45`):
  ```python
  if system == "linux" and machine in ("x86_64", "amd64"):
      asset = "crispasr-linux-x86_64.tar.gz"
  elif system == "darwin":
      asset = "crispasr-macos.tar.gz"
  elif system == "windows":
      asset = "crispasr-windows-x86_64.zip"
  else:
      logging.warning(f"No pre-built CrispASR binary for {system}/{machine}")
      return None
  ```
  Linux ARM64 / aarch64 is not supported. macOS does not distinguish arm64 vs.
  x86_64 — a single `crispasr-macos.tar.gz` asset is assumed. Windows ARM64
  is not supported.

- **Storage location**: `_CACHE_DIR = os.path.join(os.path.expanduser("~"),
  ".cache", "susurrus", "crispasr")` (`crispasr_backend.py:22`). Cross-platform
  hard-coded path; no respect for `XDG_CACHE_HOME` env var.

- **Version pinning**: **none**. The URL always points at `.../releases/latest/download/`.

- **Integrity check (checksum / signature)**: **none**. There is no SHA256
  verification, no GPG signature check, no manifest fetch — the binary is
  written straight from `urllib.request.urlretrieve` (`crispasr_backend.py:55`)
  into the cache dir and unpacked.

- **Permissions**: on non-Windows platforms, the extracted binary is chmodded
  `0o755` (`crispasr_backend.py:81-82`):
  ```python
  if system != "windows":
      os.chmod(path, 0o755)
  ```

- **Extraction**: `tar.gz` via `tarfile.open(archive_path, "r:gz")` for
  Linux/macOS, `zipfile.ZipFile(archive_path, "r")` for Windows
  (`crispasr_backend.py:62-72`). After extraction the wrapper walks `_CACHE_DIR`
  to find the binary (`crispasr_backend.py:78-87`).

- **Caching**: a cached binary that already exists and is executable
  (`os.path.isfile(cached_exe) and os.access(cached_exe, os.X_OK)`) is
  returned without re-downloading (`crispasr_backend.py:47-49`).

- **Failure handling**: download failure is logged at WARNING level and
  `_download_crispasr` returns `None`; the calling `transcribe()` then raises
  `FileNotFoundError` with the message
  `"crispasr binary not found. Set CRISPASR_EXECUTABLE or install from
  https://github.com/CrispStrobe/CrispASR"` (`crispasr_backend.py:147-150`).

---

## 12. THE CrispASR-backend wrapper — verbatim invocation pattern

**File**: `workers/transcription/backends/crispasr_backend.py` (9585 bytes,
~240 lines, introduced in commit `1048d6d` 2026-04-19T07:47:57Z, updated in
`7073a77` 2026-04-19T08:38:11Z).

### 12.1 Module docstring and constants

```python
# workers/transcription/backends/crispasr_backend.py:1-13
"""CrispASR backend — unified multi-model ASR via the crispasr binary.

Supports all CrispASR backends (parakeet, canary, cohere, granite,
qwen3, voxtral, voxtral4b, fastconformer-ctc, wav2vec2) through a
single interface. The backend auto-detects from the GGUF file, or
can be forced with the `crispasr_backend` kwarg.

Requires the `crispasr` binary on PATH or at CRISPASR_EXECUTABLE.
Build from https://github.com/CrispStrobe/CrispASR
"""
```

The module-level docstring enumerates **nine** sub-backends: parakeet, canary,
cohere, granite, qwen3, voxtral, voxtral4b, fastconformer-ctc, wav2vec2. The
README §Features expands this to "11 ggml backends — parakeet, canary, qwen3,
granite, voxtral, wav2vec2, etc." and lists "whisper" as well in the
auto-download commit message (`7073a77` commit message says "11 ASR models
(parakeet, canary, cohere, granite, qwen3, voxtral, voxtral4b,
fastconformer-ctc, wav2vec2, whisper)" — that lists 10, not 11). The exact
count and identifiers are not encoded in any Python data structure — the
wrapper passes `--backend` through opaquely to the binary.

### 12.2 Imports

```python
# workers/transcription/backends/crispasr_backend.py:14-18
import logging
import os
import re
import subprocess
import threading
from .base import TranscriptionBackend
```

Standard library only at module load time. Heavy imports
(`urllib.request`, `tarfile`, `zipfile`, `platform`, `shutil`) are pulled in
lazily inside `_download_crispasr()` and `_find_crispasr()`.

### 12.3 Class definition

```python
# workers/transcription/backends/crispasr_backend.py:130-153
class CrispasrBackend(TranscriptionBackend):
    """CrispASR backend — calls the crispasr binary for any supported model.

    model_id should be a path to a GGUF model file. The backend is
    auto-detected from GGUF metadata, or forced via crispasr_backend kwarg.

    Kwargs:
        crispasr_backend: str — force a specific backend (e.g. "parakeet")
        word_timestamps: bool — request word-level timestamps
        vad: bool — enable Silero VAD for long audio
        split_on_punct: bool — split subtitles at sentence boundaries
        temperature: float — sampling temperature (0 = greedy)
        best_of: int — best-of-N candidates with temperature > 0
    """

    def __init__(self, model_id, device, language=None, word_timestamps=False,
                 **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.word_timestamps = word_timestamps
        self.crispasr_backend = kwargs.get("crispasr_backend", None)
        self.vad = kwargs.get("vad", False)
        self.split_on_punct = kwargs.get("split_on_punct", False)
        self.temperature = kwargs.get("temperature", 0.0)
        self.best_of = kwargs.get("best_of", 1)
        self.temp_files = []
```

### 12.4 Preprocess (audio path normalization)

```python
# workers/transcription/backends/crispasr_backend.py:155-164
def preprocess_audio(self, audio_path):
    """Convert to WAV if needed (crispasr handles WAV/MP3/FLAC/OGG)."""
    # CrispASR can decode most formats via miniaudio, but WAV is safest
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in (".wav", ".mp3", ".flac", ".ogg"):
        return audio_path
    from utils.audio_utils import convert_audio_to_wav
    wav_path = convert_audio_to_wav(audio_path)
    if wav_path != audio_path:
        self.temp_files.append(wav_path)
    return wav_path
```

The comment indicates CrispASR decodes via miniaudio internally. Susurrus only
pre-converts when the file extension is not in `{.wav, .mp3, .flac, .ogg}`.
Conversion target is 16-bit mono 16kHz WAV via
`utils/audio_utils.convert_audio_to_wav()` (uses `pydub`/`AudioSegment` and
writes to `tempfile.NamedTemporaryFile(delete=False, suffix=".wav")`).

### 12.5 Invocation mechanism — subprocess

```python
# workers/transcription/backends/crispasr_backend.py:166-186 (cmd construction)
def transcribe(self, audio_path):
    """Transcribe using the crispasr binary."""
    logging.info("=== Starting CrispASR pipeline ===")

    exe = _find_crispasr()
    if not exe:
        raise FileNotFoundError(
            "crispasr binary not found. Set CRISPASR_EXECUTABLE or "
            "install from https://github.com/CrispStrobe/CrispASR"
        )
    logging.info(f"Using crispasr: {exe}")

    if not os.path.isfile(self.model_id):
        raise FileNotFoundError(f"Model not found: {self.model_id}")

    cmd = [
        exe,
        "-m", self.model_id,
        "-f", audio_path,
        "-t", str(min(os.cpu_count() or 4, 8)),
        "-np",  # no progress prints on stderr
    ]
```

**Invocation mechanism: `subprocess.Popen`** spawning the native `crispasr`
executable. Not HTTP, not Python bindings, not ctypes, not gRPC, not stdin/stdout
JSON-RPC.

**Argument schema** observed on the `crispasr` CLI as Susurrus invokes it:

| Flag | Source value | Notes |
|---|---|---|
| `-m <path>` | `self.model_id` | Path to a GGUF file (validated as `os.path.isfile`) |
| `-f <path>` | `audio_path` | Already preprocessed by §12.4 |
| `-t <n>` | `min(os.cpu_count() or 4, 8)` | Thread count, capped at 8 |
| `-np` | (no value) | Suppress progress prints on stderr |
| `--backend <name>` | `self.crispasr_backend` | Optional; forces a specific sub-backend |
| `-l <code>` | `self.language` | Optional language code |
| `--vad` | (no value) | Optional Silero VAD |
| `--split-on-punct` | (no value) | Optional subtitle-friendly splitting |
| `-ml 1` | (no value) | Set when `self.word_timestamps`; "max_len=1 → one word per segment" (comment) |
| `-tp <float>` | `self.temperature` | Only added if `> 0` |
| `--best-of <int>` | `self.best_of` | Only added if `temperature > 0` AND `best_of > 1` |

Flag-construction code:

```python
# workers/transcription/backends/crispasr_backend.py:188-209
if self.crispasr_backend:
    cmd.extend(["--backend", self.crispasr_backend])

if self.language:
    cmd.extend(["-l", self.language])

if self.vad:
    cmd.append("--vad")

if self.split_on_punct:
    cmd.append("--split-on-punct")

if self.word_timestamps:
    cmd.extend(["-ml", "1"])  # max_len=1 → one word per segment

if self.temperature > 0:
    cmd.extend(["-tp", str(self.temperature)])
    if self.best_of > 1:
        cmd.extend(["--best-of", str(self.best_of)])
```

### 12.6 Process spawn + streaming I/O

```python
# workers/transcription/backends/crispasr_backend.py:211-247
logging.info(f"Running: {' '.join(cmd)}")

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)

output_lines = []

def collect_stderr():
    for line in iter(process.stderr.readline, ""):
        line = line.strip()
        if line:
            logging.info(f"crispasr: {line}")

stderr_thread = threading.Thread(target=collect_stderr, daemon=True)
stderr_thread.start()

# Read stdout — crispasr outputs [HH:MM:SS.mmm --> HH:MM:SS.mmm] text
for line in iter(process.stdout.readline, ""):
    line = line.strip()
    if not line:
        continue
    m = re.match(
        r"\[(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)\]\s*(.*)",
        line,
    )
    if m:
        start = self._parse_ts(m.group(1))
        end = self._parse_ts(m.group(2))
        text = m.group(3).strip()
        if text:
            output_lines.append((start, end, text))
    else:
        # Plain text line (no timestamps)
        if line:
            output_lines.append((0.0, 0.0, line))

rc = process.wait()
stderr_thread.join(timeout=2)

if rc != 0:
    logging.error(f"crispasr failed with code {rc}")
    raise RuntimeError(f"crispasr exited with code {rc}")

for result in output_lines:
    yield result
```

**Audio is passed as a file path** (`-f <audio_path>`), not via stdin and not
via memory. The binary opens the file itself.

**Output parsing**: stdout regex `r"\[(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)\]\s*(.*)"`
matching the format `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text`. Lines that do not
match are emitted with `(0.0, 0.0, line)`. This matches the same format
`WhisperCppBackend` parses (`whisper_cpp_backend.py:91`), suggesting the
CrispASR binary deliberately mimics whisper.cpp's stdout convention.

**Note on generator semantics here**: the wrapper does **not** yield streaming.
It accumulates all `(start, end, text)` tuples into `output_lines`, waits for
the process to exit (`process.wait()`), checks the return code, then yields
the buffered tuples at the end (`for result in output_lines: yield result`).
This means downstream consumers see no output until the binary has fully
finished. (Compare `FasterWhisperBatchedBackend.transcribe()`
(`faster_whisper_backend.py:42-60`), which yields each segment as the library
produces it.)

**Stderr handling**: a daemon `threading.Thread` named `collect_stderr` pulls
lines from `process.stderr` and forwards them to Python's `logging.info` with
prefix `"crispasr: "`. Stderr is not parsed for structured data.

### 12.7 Timestamp parsing helper

```python
# workers/transcription/backends/crispasr_backend.py:249-253
@staticmethod
def _parse_ts(ts_str):
    """Parse HH:MM:SS.mmm to seconds."""
    parts = ts_str.split(":")
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
```

### 12.8 Cleanup

```python
# workers/transcription/backends/crispasr_backend.py:255-261
def cleanup(self):
    for f in self.temp_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception as e:
            logging.warning(f"Failed to remove temp file {f}: {e}")
```

`self.temp_files` is only ever appended to when `preprocess_audio` converts to
WAV. There is no model-handle release because there is no in-process model
handle — each `transcribe()` call spawns a fresh subprocess.

### 12.9 Capability handling

The wrapper does **not** introspect any capability flags (no
`CAP_WORD_TIMESTAMPS`, no `--list-backends`, no `--capabilities` probing,
no version check, no GGUF metadata parsing). Capabilities are presumed by
which flags the user asks for:
- `word_timestamps=True` → unconditionally append `-ml 1`.
- `vad=True` → unconditionally append `--vad`.
- `split_on_punct=True` → unconditionally append `--split-on-punct`.
- `crispasr_backend="parakeet"` (or other) → passed through verbatim.

If the binary or a particular sub-backend does not support a given flag, the
subprocess will fail and the failure surfaces as `RuntimeError("crispasr exited
with code {rc}")` (`crispasr_backend.py:243-245`). The wrapper makes no
attempt to negotiate down.

### 12.10 Error handling

- **Missing binary** → `FileNotFoundError` with message pointing at env var
  and source repo (`crispasr_backend.py:172-175`).
- **Missing model file** → `FileNotFoundError(f"Model not found: {self.model_id}")`
  (`crispasr_backend.py:177-178`).
- **Non-zero exit code** → `logging.error` then `RuntimeError(f"crispasr exited
  with code {rc}")` (`crispasr_backend.py:243-245`).
- **Download failure** → logged at WARNING, returns None, eventually escalates
  to the FileNotFoundError above.
- **Extraction failure** → logged at WARNING, returns None
  (`crispasr_backend.py:65-72`).
- No retry logic. No backoff. No stderr parsing for specific error patterns.

### 12.11 Model-path management

- `model_id` is treated as **a local filesystem path to a GGUF file**, validated
  with `os.path.isfile(self.model_id)` before invocation
  (`crispasr_backend.py:177`). It is not a HuggingFace repo ID.
- There is no model auto-download logic for the GGUF — only the *binary* is
  auto-downloaded; the user must supply the GGUF themselves.
- `config.py` `BACKEND_MODEL_MAP` has no `crispasr` entry, so there is no
  curated default model list shown to the user. The CLI requires `--model`
  to be explicitly supplied (`cli.py:127`: `if not args.model or not args.file:
  parser.error("--model and --file are required for transcription")`).

### 12.12 Configuration surface

End-to-end configuration surface exposed by the CrispASR wrapper:
- Environment: `CRISPASR_EXECUTABLE` (binary path override).
- Constructor kwargs (`__init__`): `model_id`, `device`, `language`,
  `word_timestamps`, `crispasr_backend`, `vad`, `split_on_punct`,
  `temperature`, `best_of`. (`device` is accepted but **never used** by this
  backend — the binary chooses its own compute device.)
- CLI flags via `cli.py`: `--model`, `--file`, `--language`, `--device`,
  `--crispasr-backend`, `--vad`, `--split-on-punct` (`cli.py:103-117`).
- Cache dir: `~/.cache/susurrus/crispasr` (hard-coded).
- No JSON / YAML / TOML configuration file.

---

## 13. Other backends — comparison points

### 13.1 `mlx_backend.py` (in-process Python library)

```python
# workers/transcription/backends/mlx_backend.py:17-23, 41-55
import mlx_whisper
from huggingface_hub import snapshot_download
...
model_path = snapshot_download(repo_id=self.model_id)
...
result = mlx_whisper.transcribe(wav_path, **transcribe_options)
if 'segments' in result:
    for segment in result['segments']:
        text = segment['text'].strip()
        start = segment['start']
        end = segment['end']
        yield (start, end, text)
```

**Mechanism**: in-process Python. Loads the model via `mlx_whisper`, calls
`mlx_whisper.transcribe()`, iterates the returned dict's `segments`. No
subprocess. Model files come from `huggingface_hub.snapshot_download`.

### 13.2 `faster_whisper_backend.py` (in-process Python library, streaming)

```python
# workers/transcription/backends/faster_whisper_backend.py:19-31, 42-60
from faster_whisper import WhisperModel, BatchedInferencePipeline
...
model = WhisperModel(self.model_id, device=self.device, compute_type=compute_type)
pipeline = BatchedInferencePipeline(model=model)
segments, info = pipeline.transcribe(audio_path, batch_size=4, language=self.language,
    word_timestamps=self.word_timestamps, vad_filter=True)
...
for segment in segments:
    text = segment.text.strip()
    start = segment.start
    end = segment.end
    yield (start, end, text)
```

**Mechanism**: in-process Python. Generator-driven — `segments` is itself a
generator, so each `yield` happens as soon as `faster_whisper` produces the
segment. Word-level timestamps are emitted as additional `(start, end, text)`
tuples with a two-space prefix on the text.

### 13.3 `whisper_cpp_backend.py` (subprocess, same family as CrispASR)

```python
# workers/transcription/backends/whisper_cpp_backend.py:62-89, 91-94
cmd = [whisper_cpp_executable, '-m', model_path, '-f', audio_path,
       '-t', str(min(os.cpu_count() or 4, 8))]
...
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1)
...
timestamp_match = re.match(r'\[(\d+:\d+:\d+\.\d+) --> (\d+:\d+:\d+\.\d+)\]\s+(.+)', line)
```

**Mechanism**: subprocess. Same thread-count cap (`min(os.cpu_count() or 4, 8)`),
same stdout regex format, same daemon stderr thread, same buffered-then-yield
generator anti-pattern. The CrispASR wrapper is clearly modeled on this
whisper.cpp wrapper.

### 13.4 `voxtral_backend.py` (in-process, two flavors)

```python
# workers/transcription/backends/voxtral_backend.py:17-23, 30-38
from backends.transcription.voxtral_local import VoxtralLocal
self.voxtral = VoxtralLocal(model_id=self.model_id, device=self.device)
segments = self.voxtral.transcribe(audio_path, language=self.language,
    temperature=self.temperature, chunk_length=self.max_chunk_length)
for segment in segments:
    yield (segment['start'], segment['end'], segment['text'].strip())
```

Local flavor delegates to `backends/transcription/voxtral_local.py` (an
in-process helper). API flavor (`VoxtralAPIBackend`) delegates to
`backends/transcription/voxtral_api.py` and requires `MISTRAL_API_KEY`.

### 13.5 What the spread implies about Susurrus's abstraction surface

The `TranscriptionBackend` ABC is minimal enough that it accommodates four
distinct execution mechanisms within the same registry:
1. In-process Python library (mlx, faster-whisper, transformers, openai-whisper,
   insanely-fast-whisper, ctranslate2, whisper-jax, voxtral-local).
2. Subprocess to a native binary (crispasr, whisper.cpp).
3. HTTP API call to a hosted service (voxtral-api).
4. Custom in-process module wrapping a non-trivial loader (voxtral_local).

All four converge on the same generator interface (`yield (start, end, text)`)
and the same `OUTPUT FILE:` / timestamp-line stdout protocol when running
inside the subprocess worker.

---

## 14. GUI framework

- **Framework**: PyQt6 (`PyQt6.QtWidgets`, `PyQt6.QtCore`, `PyQt6.QtGui`).
  `main.py:6`, `config.py:7`, `workers/transcription_thread.py:11`,
  `gui/main_window.py:5-7`.
- **Event loop pattern**: standard Qt event loop. `QApplication(sys.argv)` →
  `MainWindow()` → `window.show()` → `app.exec()` (`main.py:23-26`).
- **Settings store**: `QSettings(APP_ORG, APP_NAME)` = `QSettings("CrispStrobe",
  "Susurrus")` (`config.py:80-82`). README documents the per-platform location:
  - Windows: `%APPDATA%\Susurrus\AudioTranscription.ini`
  - macOS: `~/Library/Preferences/com.Susurrus.AudioTranscription.plist`
  - Linux: `~/.config/Susurrus/AudioTranscription.conf`
- **Widget composition**: `MainWindow` aggregates `CollapsibleBox`,
  `DiarizationSettingsBox`, `VoxtralSettingsBox`, `AdvancedOptionsBox`
  (under `gui/widgets/`) and `DependenciesDialog`, `InstallerDialog`,
  `CUDADiagnosticsDialog` (under `gui/dialogs/`) — see imports at
  `gui/main_window.py:14-19`.

---

## 15. Configuration system

- **App constants**: `config.py:9-11`
  ```python
  APP_NAME = "Susurrus"
  APP_VERSION = "1.1.0"
  APP_ORG = "CrispStrobe"
  ```
- **Backend ↔ model map**: `BACKEND_MODEL_MAP` (see §6).
- **Per-backend defaults**: `get_default_model_for_backend(backend)` returns
  hard-coded defaults from a dict (`config.py:113-131`).
- **Per-backend device fallbacks**: `DEVICE_FALLBACKS` dict
  (`config.py:94-102`):
  ```python
  DEVICE_FALLBACKS = {
      "faster-batched": [("mps", "cpu")],
      "faster-sequenced": [("mps", "cpu")],
      "faster-whisper": [("mps", "cpu")],
      "openai-whisper": [("mps", "cpu")],
      "whisper-jax": [("mps", "cpu")],
      "ctranslate2": [("mps", "cpu"), ("cuda", "cpu")],
      "voxtral-local": [],
      "voxtral-api": [],
  }
  ```
  CrispASR has no entry — it accepts the user's `--device` flag in `cli.py`
  but the wrapper never actually passes a device flag to the binary
  (the binary auto-selects).
- **Default backend selection**: `get_default_backend()` returns
  `mlx-whisper` on macOS, `faster-batched` if CUDA available on Windows/Linux,
  else `whisper.cpp` (`config.py:85-99`).
- **Per-user settings persistence**: handled inside `MainWindow` via QSettings.
- **No `.env` parsing**: `.env.example` exists but no `python-dotenv` is
  imported anywhere.
- **Environment variables read**: `HF_TOKEN`, `MISTRAL_API_KEY`,
  `CRISPASR_EXECUTABLE` (see §11), `PYTORCH_MPS_HIGH_WATERMARK_RATIO`,
  `CUDA_VISIBLE_DEVICES` (latter two documented in `README.md`
  §Configuration § Environment Variables).

---

## 16. Error surface

- **Worker errors → GUI**: any exception in the subprocess prints to stderr,
  which is consumed line-by-line by the daemon stderr thread inside
  `TranscriptionThread` and re-emitted as `progress_signal`. A non-zero exit
  code causes `error_signal.emit("Transcription process failed.")`
  (`workers/transcription_thread.py:370`).
- **Backend errors**: each backend's `transcribe()` may raise `ImportError`
  for missing libraries (e.g.
  `raise ImportError("faster_whisper required for faster-batched backend")` —
  `faster_whisper_backend.py:21`), `FileNotFoundError` for missing
  models/binaries (multiple sites), `RuntimeError` for non-zero subprocess
  exit codes (`crispasr_backend.py:245`).
- **Logging configuration** is set at module import in `main.py:9-14`:
  ```python
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - %(message)s",
      datefmt="%H:%M:%S",
      handlers=[logging.StreamHandler()],
  )
  ```
  And re-set in `workers/transcribe_worker.py:18-22`. There is no file handler
  by default; logs go to stderr only.
- **`CUDADiagnosticsDialog`** (`gui/dialogs/cuda_diagnostics_dialog.py`,
  6218 bytes) plus `utils/dependency_check.py` provide structured
  diagnostics for missing CUDA / PyTorch issues, surfaced at app startup
  (`gui/main_window.py:62-78`).

---

## 17. Strengths and limitations as an integration-pattern reference

**Stated capabilities** (from `README.md` §Features):
- 12 transcription backends in one registry (counting two-class modules
  separately).
- Speaker diarization via pyannote (code on `main` references but does not
  contain the manager module — see §10).
- 8-language Voxtral support.
- Per-backend device fallback rules.
- Auto-download of the CrispASR binary on first use.
- Headless CLI (`cli.py`) that bypasses PyQt6 entirely (commit `0af3e7e`).
- UTF-8 encoding enforcement on stdout/stderr in worker scripts.

**Limitations observable in the source**:
- No `requirements.txt` exists despite README instructing `pip install -r
  requirements.txt`. Only `setup.py install_requires` (5 packages) and
  `requirements-dev.txt` are present.
- `pyproject.toml` has no `[project]` / `[build-system]` tables — `setup.py`
  is the sole build descriptor.
- `BACKEND_MODEL_MAP` is missing a `crispasr` entry, so the README's "step
  5" of the "Adding a New Backend" instructions was not performed for the
  CrispASR integration.
- README's "Architecture Overview" tree describes a `backends/diarization/`
  subpackage with `manager.py` and `progress.py`, but those files are not in
  the repo. The diarization worker imports `from .manager import
  DiarizationManager` which would fail to resolve at runtime against
  `main`@`7073a77`.
- `tests/integration/` and `tests/unit/` directories exist but are empty.
- `transcribe_worker_bkp.py` (52009 bytes) is checked into the repo as a
  backup of an earlier monolithic worker.
- The CrispASR wrapper does **not** stream — it buffers all output and yields
  only after the subprocess exits (§12.6). Other subprocess-based backends
  (e.g. `whisper_cpp_backend.py`) exhibit the same buffering pattern; the
  Python-library backends (faster-whisper, mlx-whisper) stream natively.
- The auto-download mechanism has no integrity checking and always pulls
  `releases/latest` with no version pinning (§11).
- No retry / backoff on download failure.
- Linux ARM64 / Windows ARM64 / non-x86_64-Linux are unsupported by the
  auto-downloader (§11).
- `device` kwarg is accepted by `CrispasrBackend.__init__` but never read
  during `transcribe()` — the wrapper does not pass any device flag to the
  `crispasr` binary.
- All commits introducing the CrispASR backend (`1048d6d`, `0af3e7e`,
  `7073a77`) are authored by a placeholder identity
  `crispasr integration <crispasr-dev@localhost>` and co-authored by
  `Claude Opus 4.6 (1M context)`; they are unsigned (`"verified": false,
  "reason": "unsigned"`).

---

## 18. Maintainer activity

- **Owner**: GitHub user `CrispStrobe` (id `154636388`, joined 2024 based on
  `Susurrus` created_at `2024-10-03T17:14:06Z`). Same owner as `CrispASR`,
  `CrisperWeaver`, `CrispEmbed` (per `README.md` Crisp-ecosystem table).
- **Commit cadence on `main`** (7 commits total):
  1. `80b5c8c` — 2025-11-01 — "Initial commit of Susurrus" (by CrispStrobe).
  2. `6d704feb` — 2025-12-06 — typo fix in voxtral_local.py (by `terribleperson`,
     merged via PR #1).
  3. `69c216da` — 2025-12-07 — merge of PR #1 (by CrispStrobe).
  4. `ef923dda` — 2026-04-19T07:46:02Z — "docs: add Crisp ecosystem section to
     README" (by `crispasr integration`, co-author Claude Opus 4.6).
  5. `1048d6d` — 2026-04-19T07:47:57Z — "feat: add CrispASR as a transcription
     backend" (same placeholder author).
  6. `0af3e7e` — 2026-04-19T07:54:25Z — "feat: add headless CLI for all
     backends" (same).
  7. `7073a77` — 2026-04-19T08:38:11Z — "feat: MIT license, auto-download
     CrispASR, updated README" (same).

  The activity pattern: one initial commit November 2025, one external typo
  fix in early December 2025, then a 4-month silence, then a rapid 4-commit
  burst on 2026-04-19 (~52 minutes elapsed between the first and last commit
  of the burst) introducing the CrispASR backend and the MIT license file.
- **Releases / tags**: none.
- **Same maintainer as CrispASR/CrisperWeaver**: README explicitly groups
  them under "Part of the Crisp ecosystem" and the owner of all four repos is
  the same GitHub user.

---

## 19. Recent issues + PRs

- **`open_issues_count`**: 0 (as of fetch at `7073a77`).
- **Total issues + PRs reachable via `gh api repos/CrispStrobe/Susurrus/issues
  --state=all`**: 1.
  - `#1` [closed] PR by `terribleperson`: "change transcrition to transcription
    in voxtral_local.py" — typo fix, 0 comments, merged 2025-12-07.
- No open WIP signals, no recurring issue themes, no roadmap discussions.
- `has_discussions: false`, `has_wiki: false`, `has_projects: true` but no
  projects are visible.

---

## 20. Could not verify

The following items could not be verified from the primary sources fetched:

1. **Whether `backends/diarization/manager.py` / `progress.py` exist on a
   non-main branch.** Only `main` is published (no other branches present in
   `gh api` listings). `workers/diarize_worker.py:18` references `from
   .manager import DiarizationManager` but `workers/manager.py` is also absent.
   The diarization code path would fail to import on a clean checkout of
   `main`@`7073a77`.
2. **Whether the CrispASR `--backend` argument exhaustive list is 9, 10, or 11**.
   The module docstring lists 9 (parakeet, canary, cohere, granite, qwen3,
   voxtral, voxtral4b, fastconformer-ctc, wav2vec2); the commit message
   adds "whisper" to make 10; the README says "11 ggml backends". The wrapper
   does not enumerate them.
3. **The actual CrispASR release-asset layout** on GitHub Releases. The
   wrapper assumes `crispasr-{linux-x86_64|macos|windows-x86_64}.{tar.gz|zip}`
   but the CrispASR repo's release assets were not fetched as part of this
   research.
4. **Whether `crispasr` binary signals (e.g. SIGTERM handling)** are clean. The
   wrapper relies on `subprocess.terminate()` via
   `TranscriptionThread.stop()` but does not test or document the binary's
   shutdown semantics.
5. **Whether the GGUF-metadata auto-detection of sub-backend** is performed by
   the binary itself or expected of the host. The wrapper docstring says
   "auto-detects from the GGUF file" but Susurrus does not parse GGUF.
6. **The output rate / streaming behavior of the `crispasr` binary itself**.
   The wrapper buffers `output_lines` and yields only after `process.wait()`,
   so even if the binary streams stdout, the Python consumer sees a
   single-shot batch.
7. **The presence of a stable API contract between CrispASR releases**. With
   `releases/latest/download/` as the URL pattern and no version pinning, a
   future asset rename (e.g. `crispasr-linux-arm64.tar.gz`) or a flag rename
   (e.g. `--split-on-punct` → `--split-punct`) would silently break Susurrus
   without any error visible to the auto-downloader.
8. **macOS arm64 vs. x86_64 selection**. The auto-downloader uses a single
   `crispasr-macos.tar.gz` asset on Darwin regardless of `platform.machine()`.
   Whether the binary inside is fat / arm64 / x86_64 was not verified.
9. **The number of contributors on related private branches** (none are
   visible).
10. **The exact line numbers cited above for `crispasr_backend.py`** are
    approximate — they were inferred from the raw file contents fetched
    via `raw.githubusercontent.com`. Reading via a line-numbered fetch could
    refine them.
