# WhisperJAV v1.8.12 — WhisperSeg VAD + tight defaults retune

**New speech segmenter (WhisperSeg) + tightened VAD defaults + sensitivity retune**

The headline feature is **WhisperSeg**, a new ONNX-based speech segmenter
trained on ~500 hours of Japanese ASMR. It outperforms Silero v3.1, Silero
v6.2, and TEN-VAD on a Netflix-quality JAV reference clip (F1 = 0.787 vs
0.625–0.698 for prior backends, aggressive sensitivity).

This release also tightens the default VAD `max_speech_duration_s` and
`max_group_duration_s` across all four in-scope speech segmenters, retunes
the Faster-Whisper / OpenAI Whisper aggressive sensitivity preset based on
forensic acceptance testing, and ships a new offline VAD evaluator tool.
Several bug fixes round out the release including a critical ASR regression
fix that affected aggressive sensitivity on tight VAD groups.

---

## What's new

### WhisperSeg ONNX speech segmenter — new VAD backend (#headline)

A new speech segmenter, `whisperseg`, is now selectable from the GUI Ensemble
tab and the CLI (`--qwen-segmenter whisperseg`, `--pass1/2-speech-segmenter
whisperseg`). It's the strongest VAD backend currently shipped for soft /
whispered Japanese speech.

- **Architecture**: Whisper-base encoder paired with a 2-layer transformer
  decoder, exported to ONNX. Frame-level resolution at 20 ms over 30-second
  input windows. Inherits Whisper's multilingual robustness for non-JA
  content.
- **Training data**: ~500 hours of Japanese ASMR audio with accurate
  timestamps. Strong on whispered/soft speech common in JAV.
- **Performance** on `test_media/283sec-The.Naked.Director.S01E04.Scene4.mkv`
  (Netflix-GT 283 s clip, aggressive sensitivity):

  | VAD backend  | F1    |
  |--------------|------:|
  | **whisperseg** | **0.787** |
  | ten          | 0.698 |
  | silero-v6.2  | 0.654 |
  | silero-v3.1  | 0.625 |

- **Install**: `pip install whisperjav[whisperseg]` (CPU) or
  `whisperjav[whisperseg-gpu]` (CUDA via onnxruntime-gpu). The Windows
  standalone installer ships with onnxruntime already; just select
  WhisperSeg in the GUI.
- **Model**: `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`, MIT-licensed,
  pinned to revision `6ac29e2c`. ~119 MB ONNX file, downloaded once on
  first use.

### Tight VAD defaults (max_speech / max_group)

VAD speech-segment and group durations are tightened across all four
in-scope segmenters (Silero v3.1/v4.0, Silero v6.2, TEN, WhisperSeg) to
match the natural distribution of Japanese subtitles (research shows the
majority of JA subs are <3 s with ~800 ms inter-sub gaps). Sensitivity
gradient is **inverted** from prior versions: aggressive now uses tighter
caps (more sensitive detection benefits from tighter grouping), conservative
uses looser caps.

| Sensitivity   | max_speech_duration_s | max_group_duration_s |
|---------------|----------------------:|---------------------:|
| Aggressive    | 4 (was 7–8)           | 5 (was 9–10)         |
| Balanced      | 5 (was 6–7)           | 6 (was 8–9)          |
| Conservative  | 6 (was 5–6)           | 7 (was 7–8)          |

YAML files updated: `silero-speech-segmentation.yaml`,
`silero-v6-speech-segmentation.yaml`, `ten-speech-segmentation.yaml`,
`whisperseg-speech-segmentation.yaml`. The legacy `SileroVAD` Pydantic
preset is mirrored in lockstep.

### Engine-split sensitivity preset retune (Faster-Whisper / OpenAI Whisper)

Faster-Whisper and OpenAI-Whisper presets diverge per-engine where forensic
evidence warrants. Aggressive `no_speech_threshold` is raised from 0.77 to
0.84 (more permissive, captures more intimate speech). Aggressive
`logprob_threshold` is now `-1.30` for OpenAI Whisper only — Faster-Whisper
keeps `-1.00`. Several other compute knobs are tuned.

### VAD ground-truth analyser (new tool)

`tools/vad_groundtruth_analyser/` — side-by-side VAD evaluator. Runs N VAD
backends on the same media file and produces interactive Plotly HTML +
JSON + CSV showing per-backend speech regions and aggregate stats.
Optional ground-truth SRT enables F1 / IoU / drift / miss-rate / FA-rate
metrics; without GT, an inter-backend agreement matrix is shown instead.

```bash
python -m tools.vad_groundtruth_analyser MEDIA [--ground-truth SRT] \
       [--sensitivity {cons,bal,agg}] \
       [--backends silero-v3.1,silero-v6.2,ten,whisperseg]
```

---

## Critical bug fixes

### Aggressive sensitivity empty-output regression (post-release fix in same train)

A regression introduced earlier in v1.8.12 development (commit `34fa713`,
"engine-split sensitivity preset retune") changed Faster-Whisper aggressive
`best_of: 2 → 1`. On JAV-style content with the new tight VAD groups
(max_group=5 s), this caused Faster-Whisper's temperature 0.17 fallback to
draw a single noisy hypothesis, often degenerate, which then failed the
outer `no_speech` check and was dropped — producing 76.9% empty VAD groups
on the F5 acceptance test.

**Fix**: revert aggressive `best_of: 1 → 2`. Beam search at temperature 0
uses `beam_size`, which is unrelated to `best_of`; only the temperature
fallback (sampling mode) uses `best_of` as `num_hypotheses`. Empirical
F6 acceptance test confirmed: Pass 1 recall jumped from **19.1% → 88.2%**,
F1 from **0.316 → 0.929**, with output produced in 80 s vs 337 s pre-fix.

### Silero version fallback alignment (v4.0 → v3.1)

`faster_whisper_pro_asr.py` and `whisper_pro_asr.py` previously fell back to
`silero-v4.0` when the resolver did not explicitly set
`params.speech_segmenter.backend`. This silently overrode the
`LEGACY_PIPELINES["balanced"|"fidelity"]["vad"] = "silero-v3.1"` declaration
that ships in this release line. Changed to `silero-v3.1` so runtime
matches the declared LEGACY_PIPELINES default.

### Anime-whisper ellipsis-only line filter

The anime-whisper generator backend (Qwen pipeline with
`--qwen-generator anime-whisper`) sometimes produces SRT entries containing
only `…` (or `…?`, `…!`, `…」`, etc.) for short non-speech regions
(breathing, ambient, music). These are now detected and removed at two
layers:

- **Text level** (`AnimeWhisperCleaner.clean()`): returns `""` for any
  string whose stripped form consists only of ellipsis-like chars +
  optional closing punct/quotes + whitespace AND contains at least one
  ellipsis/dot character.
- **SRT level** (`AnimeWhisperCleaner.filter_srt_file()`): defense-in-depth
  pass after stitching that drops empty/ellipsis-only entries and
  renumbers surviving entries 1..N.

Wired into `qwen_pipeline.py` Phase 8 only when
`generator_backend == "anime-whisper"`. Qwen3-ASR's Phase 8 remains
skipped (no behavioral change for that path).

### TEN VAD `max_speech_duration_s` no longer silently stripped

`ten-speech-segmentation.yaml` defines `max_speech_duration_s`, but the
factory `_PARAM_SCHEMAS["ten"]` did not include the key. The factory's
foreign-key defense-in-depth gate (which strips params not in the
backend's schema) was silently dropping this value before it reached the
TEN backend. The schema now includes it. Factory fallback defaults for
`ten`, `silero-v6.2`, and `whisperseg` are also aligned to the new YAML
balanced presets so a cleared GUI input field produces a sensible default
rather than a stale value.

### Anime-mode `chunk_threshold_s` plumbing

`QwenPipeline` Phase 4 now forwards `self.segmenter_chunk_threshold` into
`SpeechSegmenterFactory.create()`. Previously only `max_group_duration_s`
was injected; the chunk_threshold from the orchestrator's segmenter_config
silently won. Anime mode's intended `chunk_threshold_s = 0.5` (set in
`qwen_pipeline.py` for the anime-whisper override) now actually reaches
the factory.

### CLI: `whisperseg` added to `--qwen-segmenter` choices

Argparse choices for `--qwen-segmenter` were missing `whisperseg`, so
selecting it via the Qwen-pipeline CLI was rejected with an
`invalid choice` error. Now accepted. (The `--pass1/2-speech-segmenter`
flag is free-form string and was already working.)

---

## Refactors and cleanup

- **`neg_threshold` removed from speech-segmenter VADs** — the negative
  threshold (used for hysteresis in some Silero versions) is now
  auto-derived internally as `max(threshold − 0.15, 0.01)` and is no
  longer a user-facing parameter on speech segmenters. The factory
  `_sanitize_params` silently strips it from old configs for back-compat.
  Scene-detection-layer `silero_neg_threshold` is a separate namespace
  and is unchanged.
- **GUI Ensemble dropdown simplified** — `nemo`, `whisper-vad-*` and other
  rarely-used speech segmenters are hidden from the GUI Ensemble dropdown
  (CLI access preserved). Removes clutter for the typical user.

---

## How to upgrade or install

**Upgrade from 1.8.11:**

```
pip install -U --no-deps "whisperjav @ git+https://github.com/meizhong986/whisperjav.git@v1.8.12"
```

**Fresh install:**

### Windows — Standalone Installer (.exe)

1. Download **WhisperJAV-1.8.12-Windows-x86_64.exe** from the Assets below
2. Run the installer (no admin rights required)
3. Wait 5–10 minutes for setup to complete
4. Launch from the Desktop shortcut

Installs to `%LOCALAPPDATA%\WhisperJAV`. A desktop shortcut is created
automatically. Your GPU is detected automatically.

### macOS

Requires [Git](https://git-scm.com/downloads). The install script checks
for Xcode CLI Tools, Python, FFmpeg, PortAudio. Open Terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.12
installer/install_mac.sh
```

After installation, open the `whisperjav` folder in Finder and double-click
**WhisperJAV.command** to launch the GUI.

### Linux

Requires Git and Python 3.10–3.12. Open a terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.12
installer/install_linux.sh
```

After installation, launch the GUI with `./WhisperJAV.sh`.

### Windows — Source Install

Requires [Git](https://git-scm.com/downloads) and [Python 3.10–3.12](https://www.python.org/downloads/). Open a terminal and run:

```
cd %USERPROFILE%
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.12
installer\install_windows.bat
```

After installation, double-click **WhisperJAV.bat** to launch the GUI.

---

## Compatibility

Same as v1.8.11 — no dependency changes for the core. WhisperSeg adds
optional extras (`whisperseg`, `whisperseg-gpu`) which are pulled in only
when you select the WhisperSeg backend.

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 – 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |
| onnxruntime | 1.16+ (CPU) / `onnxruntime-gpu` for CUDA WhisperSeg |
| Ollama | 0.3.0+ recommended |

---

## Known issues

- **Windows standalone installer does not add its bundled tools to user
  PATH.** This is the same known issue as v1.8.11 — bundled ffmpeg 7.1
  may be shadowed by a system ffmpeg if the latter is on PATH. Workaround:
  launch via the Desktop shortcut (which activates the environment) or
  manually add `<install-dir>\Library\bin` to your user PATH. A proper fix
  remains planned for a future release.

- **ctranslate2 internal state contamination across `transcribe()` calls.**
  Forensic testing during this release uncovered that faster-whisper /
  ctranslate2 retains internal GPU/CPU state across multiple
  `transcribe()` calls on a single `WhisperModel` instance — and this
  state is *not* freed by `torch.cuda.empty_cache()` or `gc.collect()`.
  Production WhisperJAV uses a Model Reuse Pattern (single WhisperModel
  reused across all files in a batch) that may be silently degrading
  recall on later files in long batch runs. Mitigation candidate: periodic
  `WhisperModel` reload after N transcribes — planned for a future
  release.

- **Apple Silicon MPS + whisper-large-v3-turbo** — produces garbage output
  on MPS for this specific model. Use `--hf-device cpu` or the default
  kotoba model. (#198, #227)

- **Ollama download progress** — the download progress bar in the GUI
  popup is indeterminate (pulsing). Real progress is shown in the
  terminal.

---

## What's next

**v1.8.13 / v1.9.0 candidates** (no firm timeline):

- ctranslate2 state-contamination mitigation in the Model Reuse Pattern
- Installer PATH fix for the bundled ffmpeg (carry-over from v1.8.11
  known issues)
- ZipEnhancer Colab init bug (#290)
- Qwen3-ASR `transformers` version pin (#280)
- Full Ollama migration (remove llama-cpp-python)
- Standalone subtitle merge CLI (#230)
- Chinese GUI partial i18n (#175, #180)
- Speaker diarization (#248, #252)

---

Thanks to everyone who reported issues, ran acceptance tests, and tested
on their own material. The forensic acceptance tests on this release line
(`test_media/1812acceptance/F2` … `F6`) drove most of the preset retune
and the regression fix.
