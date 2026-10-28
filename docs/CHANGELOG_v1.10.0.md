# WhisperJAV v1.10.0 — Development Change Log

> Developer-facing record of what changed on branch `v1.10.0` since the 1.9.5 release (`8917f5a`, tag `v1.9.5`), why, and how it
> was checked. It is the source for the 1.10.0 release notes; it is not itself a user text.
> Ordered by requirement item (`docs/requirements/v1.10.x/2026-10-13-release-1.10.0-requirements.md`), not by date. Each entry says
> what the user sees, the commits, the owner's decisions (W-numbers in `docs/plans/OWNER_WORDS_V110_CYCLE.md`), how it was checked, and
> what was found and left open. Specs: `docs/requirements/v1.10.x/specs/`. Both folders are git-ignored.
>
> **State, 2026-10-27:** items 1, 2a–2f, 3 and 4 (CrispASR, as the GGUF pipelines) done, and the accelerator
> redesign (one accelerator per install, owner W234, design section 22 v2 approved W238); branch at `f9950dd`, 349
> commits ahead of `main`, nothing pushed. No installer has been built since 2026-10-17. The branch has no commits yet for items 12, 5, 10, 11a–c
> and 6 (their place in 1.10.0 is the owner's decision).
>
> **State, 2026-10-17:** items 1, 2a–2f and 3 done; branch at `69c1308`, 128 commits ahead of `origin/main`, nothing pushed.
> Installer built from `69c1308`: `WhisperJAV-1.10.0-Windows-x86_64.exe`, 350,890,544 bytes, SHA-256
> `7B87D977F28C64478FC52C4442E976503CF7AA3654003345C15CB9FBA2245CB7`. Remaining 1.10.0 work in the owner's order (W91): CrispASR
> (item 4) → alignment option (12) → XXL as pass 1 (5) → Tools tab (10) → hardening (11a–c) → Chinese GUI (6). Items 7, 8, 9 moved to
> 1.10.1 (W91).

---

## Release-wide facts

- **Fresh install only.** 1.10.0 needs Python 3.13 (`requires-python = ">=3.13,<3.14"`, `d8618cb`). A 1.9.x install cannot upgrade:
  1.9.5's `whisperjav-upgrade` installs from GitHub `main`, and pip refuses 1.10.0 on its Python 3.10. There is no 1.9.6 (W26).
  The release-note sentence telling 1.9 users this is the owner's wording, not yet written.
- **Colab is not supported in 1.10.0 (W42 (3)).** The expert notebook keeps installing `v1.9.5`; it was not changed.
- **Version 1.10.0:** `installer/VERSION` and `whisperjav/__version__.py` set by the owner (`4d890bc`, `00a2a62`); the 1.9.5 files
  in `installer/generated/` were removed from git (`491dd91`).

---

## Item 1 — 1.9.5 baseline and build hygiene (REQ1, N1)

**What the user sees.** Nothing on screen. The 1.10.0 wheel no longer carries the developer's config backup files: the 1.9.5 wheel
held 50 `asr_config.backup.*.json` files with local program paths.

**Changes.**
- Config backups (written when BYOP or CrispASR choices are saved in the GUI) go to the user settings folder,
  `%APPDATA%\WhisperJAV\config_backups\` on Windows (`~/Library/Application Support/WhisperJAV/…` on macOS, `~/.config/WhisperJAV/…`
  on Linux), no longer into the package folder (`879c000`; test `c8b4854`).
- The wheel names its six config JSON files one by one instead of `config/*.json` (`4eba9ca`).
- `build_release.py` stops on uncommitted or untracked files and on any wheel file git does not track; it deletes stale `build/` and
  `whisperjav.egg-info/` first, which had carried the backups into a fresh wheel by a second route. `--check-tree` and
  `--check-wheel <wheel>` run the checks alone (`9a84a07`).
- Measurement tools committed (W28, W31: tools in git, documents private): `subeval/` (`b85fb2d`); the 1.9.4 scripts moved to
  `scripts/measure/v1.9.4/` (`f2eb87f`); call tracer (`b44d700`); ChronosJAV reference runner options (`1b4bf5d`); the baseline runner
  `run_f0_baseline.py` (`4abace4`) and `summarize_f0.py` (`1f1360a`). `docs/measurements/` is no longer tracked; files kept on disk
  (`780f745`, `840301f`; W30 (2), W40).

**Checked.** Baseline: six pipelines (Balanced, Fast, Faster, Fidelity, ChronosJAV Qwen3-ASR, ChronosJAV anime-whisper) on the two
reference films, run on the owner's fresh 1.9.5 install `D:\195`, scored with subeval: `D:\WJ110_measure\F0_195\`. Build checks H1–H3
passed (spec item 1).

---

## Item 2a/2b — the new stack: Python 3.13, PyTorch 2.11.0 for CUDA 13 (REQ2)

**What the user sees.**
- The Windows installer brings Python 3.13.16 (`00cc4a3`) and PyTorch 2.11.0 for CUDA 13.0, which needs NVIDIA driver 580 or newer
  (`4bf02c8`, `366c64d`, `0321f53`, `ee1d444`).
- **Older cards stop before anything is installed.** A card below compute capability 7.5 (GTX 900/1000 series, V100) gets the owner's
  warning and three typed choices: 1 = card identified wrongly (issues link, then stop), 2 = install for the CPU only, 3 = stop (with
  the 1.9.5 download link). No time limit on the answer (W36 (1), W37, W43).
- **An older driver stops** with the owner's text naming the driver version and the 580 floor, before PyTorch is installed. In 1.9.5
  such a PC silently got the CPU build (W36 (2)).
- **Balanced, Fast and Faster keep the GPU** on the CUDA 13 install: the installer places NVIDIA's CUDA 12 cuBLAS (`nvidia-cublas-cu12
  12.8.4.1`) in the `ctranslate2` folder, because CTranslate2 still needs `cublas64_12.dll` (`a528d23`). If this PC cannot load that
  file, the installer shows a framed warning that waits for Enter and installs the CUDA 12.8 build instead (W37 (3), W42 (4)). The build
  used is written to `whisperjav_stack.txt`; `whisperjav --check` prints one line on a CUDA 12.8 PC and names missing cuBLAS files
  (`df0a44f`, `a3e5f9e`).
- The start-up message for an unsupported card now says 1.10 does not support it and 1.9.5 does, with the link.
- **Linux stays on CUDA 12.8** in 1.10.0, driver floor 570 (W45): on Linux pip-installed cuBLAS 12 is found only with
  `LD_LIBRARY_PATH` set before Python starts. Not tested on a Linux PC.
- Source installers (`install.py`, Linux, macOS and Windows scripts) ask for Python 3.13 (`babbab0`, `544502e`, `c348b6e`, `ca8edd9`).

**Versions.** numpy 2.5.x (W44; the upper bound follows numba). On Python 3.13 several packages move a major step from 1.9.5:
librosa 0.11 → 1.0.0, pandas 2.3 → 3.0, av 17 → 19, scipy 1.15 → 1.18, scikit-learn 1.7 → 1.9, onnxruntime 1.23 → 1.31 (later
1.29.0 in the lock). Every librosa call WhisperJAV makes was run on 1.0.0.

**Checked.** S1 (WJ110 on the stack), S2–S4 (fresh install, clean-environment run, forced cuBLAS failure → CUDA 12.8): the owner,
"All S1 - S4 passed" (W63). S5 (card and driver decisions with fake `nvidia-smi` output) in tests `e60fc47`. One
`assessment-adversary` review (2026-10-14): a failed cuBLAS download now stops the install instead of moving to CUDA 12.8; the fallback
replaces only the PyTorch packages; test switches `WHISPERJAV_TEST_FORCE_COMPUTE_CAP`, `WHISPERJAV_TEST_FORCE_DRIVER`,
`WHISPERJAV_TEST_FORCE_CUBLAS12_FAIL`.

---

## Item 2c — local translation through Ollama only; llama-cpp removed (REQ2; texts W43)

**What the user sees.** The `local` (llama-cpp) translation provider is gone. Ollama is the only local translator; the cloud providers
stay. A saved llama-cpp choice opens as Ollama, with one console line: "Local translation with llama-cpp was removed in 1.10; your
saved choice now uses Ollama." A llama-cpp model saved with it is dropped; Ollama's default is used.
`--provider local` (whisperjav-translate) and `--translate-provider local` (whisperjav) stop with exit 1 and "Error: --provider local
(llama-cpp) was removed in WhisperJAV 1.10. Use --provider ollama for local translation." (the flag named is the one typed).
The GUI's llama-cpp model lists and notices, the installer's `/InstallLocalLLM` option and the `llm` extra are gone.
`--translate-gpu-layers` and `install.py --local-llm*` are still accepted, hidden from `--help`, with no effect.

**Code (2026-10-14).** Translate: `3a2173d`, `4ac2c59`, `b54f756`, `d82abb7`, `7505a47`. Four modules deleted: `69c9293`,
`9a32285`, `47bd5c9`, `15afeff`. CLI `d532a38`. GUI `19426ba`, `6d0e20e`, `6273ff4`, `c60fedc`. Installer `35b23ba`, `ee1d444`,
`babbab0`, `4c36b41`, `3eb1605`. Package docstring `fc97457`.

**Checked.** Tests `6acae80`: a saved `local` becomes Ollama with one line; a typed `local` exits 1 with that error. S7: the owner
opened the GUI with a saved `local` and saw Ollama (W69). Removing llama-cpp also removes one GitHub fetch (JamePeng's wheels).

---

## Item 2d — CTranslate2 4.8.3 (#459)

**What the user sees.** Balanced, Fast and Faster run on CTranslate2 4.8.3 (was 4.8.1), which #459 asked for.
**Checked.** S6 (2026-10-16): the six baseline pipelines rerun on the new stack (`F1_stack`) and compared with `F0_195`. Five
unchanged; Balanced changed on film 2 (11 fewer character errors, 2 more missed lines, 1 more misheard line, start timing closer),
the same in reruns on each stack. The owner accepted it as the effect of the CTranslate2 and onnxruntime versions (W66 I1, I2).
**Noted (W66 I3):** fundamental improvement of Balanced and its ASR (faster-whisper today) is for a later 1.10.x; not scheduled.

---

## Item 2e — `whisperjav-upgrade` stays within 1.10.x

**What the user sees.** `whisperjav-upgrade` installs the newest release of its own series (`v1.10.*`, from GitHub's releases list),
never GitHub `main`, and keeps the PC's PyTorch build (`bb4dee2`; tests `fb2118a`, S8). `WHISPERJAV_UPGRADE_REPO` still overrides.
`tests/test_upgrade_cli.py` no longer runs a real `git pull` in the repository (`ef9083b`).
**Open (owner):** a source install (git clone) still upgrades by `git pull` on its own branch, so the "never main" rule does not
apply to it.

---

## Item 2f — every install gets the tested package versions (W55)

**What the user sees.** Every 1.10.0 install — from the `.exe`, from source, or by `whisperjav-upgrade` — gets exactly the package
versions tested in development, for its PyTorch build (CUDA 13.0, CUDA 12.8 or CPU). In 1.9.5 the installer list had 67 entries, 4
exact. Console lines (owner's wording, W56): "Installing release-pinned dependencies (CUDA 13.0)..." and "Upgrading with
release-pinned dependencies (CUDA 13.0)...", naming the build actually selected.

**Changes.**
- One source of truth: `pyproject.toml` (intent) → `uv.lock` (exact versions), with three conflicting extras `cu130`, `cu128`, `cpu`
  (`a7a25e6`, `11d875f`). `build_release.py` exports the lock per build without hashes to `whisperjav/installer/locked/` (`c655d3b`,
  `2095781`); the files ship in the `.exe` and inside the wheel (C1). The old requirements template is gone (`09ff54e`).
- Installer order: PyTorch build chosen (phase 3) → the locked file minus PyTorch's lines, `--no-deps` (phase 4) → PyTorch (4.5) →
  the wheel (5) (`1a6ea08`). The git packages come from the locked file at fixed commits.
- conda's part of the installer pinned exactly: python 3.13.16, openssl 3.6.2, pip 26.2.1, git 2.56.0, ffmpeg 7.1.1, menuinst 2.5.2
  (`7547445`). A Python package both conda and the lock hold (today only packaging 26.3) is compared by its dist-info name and
  version: same → kept and logged; different → the install stops (W57 D1; `c759541`). Installer-only tools pinned in the installer,
  not the lock: `lief==1.0.0`, `nvidia-ml-py==13.615.71` (W57 D2).
- Source installs: `uv sync --frozen --extra <build>`; `uv.lock` is never rewritten by an install (`e76b536`). Unsupported
  combinations stop with a message: Linux with CUDA 13.0, macOS with a CUDA build, a PC for which the lock has no PyTorch wheel (Intel
  Mac, Windows on ARM) (W57 D3; `f3b5ae6`).
- Upgrades on installer installs: the new release's wheel and its locked file for the recorded build, `--no-deps`; a pre-check
  dry-runs exactly those lines before anything changes; `--extras` applies to source installs only (W57 D4, D5; `3d371e7`, `41db557`).
  The ten console texts were approved as written (W58), seven more (W61).
- Build: environment A, `WJ110_build`, from `installer/build_env.yml` (constructor 3.17.3, uv 0.9.26) (`07b4743`); the build bat runs
  only there and never builds a wheel itself (`8eecd2d`). The NSIS template was ported to constructor 3.17.3 (`2d0b017`); post-install
  runs in its own console window with its exit code passed on.
- **A deliberate stop** (old card answered 1 or 3, old driver) exits cleanly and the installer shows one closing dialog: "WhisperJAV
  was not installed." pointing back to the message the user just read, instead of "Failed to run post install script" and wrong advice
  (`35b23ba`; approach approved W46).
- `installer/README_INSTALLER` names Python 3.13.16 (`79013a1`, W60).

**Checked.** WJ110 recreated from the lock: installed = the lock's 219 entries for this PC + nvidia-cublas-cu12; `uv pip check` clean;
`whisperjav --check` all passed. One `assessment-adversary` review (2026-10-14) refuted "every path installs exactly uv.lock"; its
findings became D1–D5 (W57). A second review on the 3.17.3 port found post-install running hidden; fixed. S2–S4 passed (W63).

---

## Item 3 — transformers 5, Qwen3-ASR through the WhisperJAV fork of `qwen-asr`; ClearVoice and the Kotoba pipeline removed (REQ3)

**What the user sees.**
- Qwen3-ASR and its ForcedAligner run on transformers 5.20.1 with the same results as before (G4). transformers 5.20 also holds what
  later 1.10.x engines need (the jaykwok Qwen3-ASR fine-tune for REQ12, Cohere, VibeVoice ASR, Parakeet, Canary, MOSS-Transcribe).
- `qwen-asr` now comes from the fork `meizhong986/Qwen3-ASR` (branch `whisperjav-tf5`, commit `10cfb32`, version 0.0.6.post1), a
  GitHub fork of QwenLM/Qwen3-ASR with upstream PR #125, a fix for issue #218 and two further fixes found by running both checkpoints
  (sub-configs set before `PretrainedConfig.__init__`; the audio encoder's position table rebuilt in `_init_weights` — without it
  transformers 5 left it empty and a test clip gave 1 character instead of 35). The fork's README is the owner's text M5 (W76).
- **ClearVoice is no longer included** (W73). A ClearVoice value stops the run before any audio is read: on the command line
  (`--pass1-speech-enhancer`, `--pass2-speech-enhancer`, `--qwen-enhancer`, and inside `--passN-*-params`) with text M1 and exit 2; in
  the GUI, a remembered or preset ClearVoice choice marks the pass, the list shows a disabled "ClearVoice (no longer included)" entry,
  and Start shows M2 until another enhancer is chosen (W75, W79, W85).
- The hidden, dormant Kotoba faster-whisper pipeline is removed (W73). Transformers mode and its Kotoba models are unchanged. The two
  Kotoba CTranslate2 models leave the GUI with the pipeline; `--model` stays free text.
- Fewer packages: gradio, flask, sox, qwen-omni-utils (with the fork); clearvoice, gdown, python-speech-features, scenedetect,
  torchinfo, yamlargparse, opencv-python, torchvision (with ClearVoice). `audioop-lts` added to the `cli` extra beside pydub (Python
  3.13 has no `audioop`; it came only through gradio before) (W85). Still five GitHub fetches at install: clearvoice left, the fork came
  in.

**Code (2026-10-17).** Versions: `ffae784`, `24e38f5`, lock `62be5c3`, locked files `4ab8f05`. ClearVoice: `7dc9f0e`, `efc53d8`,
`b52db5c`, `ef3fe8f`, `28b64ee`, `3277aec`, `0b90aec`, `9da9f72`, `32a927f`, `1755fd5`, `aadfc27`, `f2bfdca`, `3bb6e17`, `797ad56`.
Kotoba pipeline: `7b2575e`, `db4d38e`, `6aad815`, `b021c54`, `9e1374e`, `0704bc2`. hub 1.x: `88df648` (stable-ts `resume_download`).
Deleted unused AkitoP classifier: `d94aea3` (W79 C4). Tests: `4d700b3`, `019ea95`, `e02d1a1`, `d729599`, `fcdd3f5`, `69c1308`.
Measurement tools: `a867d37`, `d7b858c`.

**Checked.** G1: both checkpoints on transformers 5.20, text identical to the reference, installed files byte-identical to the tested
fork. G2: new `tests/test_clearvoice_removed_v110.py` (28 tests) and the dependency and installer tests pass; 19 suite files fail with
the same counts on the pre-item-3 code (pre-existing). G3: network faults still counted as network faults (`model_loader.py`
unchanged). G4: six baseline pipelines identical to `F1_stack` line by line; extras identical except htdemucs (random offset, F4).
G5 (GUI) and G6 (fresh install from the 2026-10-17 `.exe`; ChronosJAV Qwen3-ASR, anime-whisper, Transformers mode, ZipEnhancer on one
file): the owner, "Both G5 and G6 passed." (W90). One `assessment-adversary` review of the spec (nine findings, all taken in).

---

## Item 4 — CrispASR: the GGUF pipelines Parakeet-CrispASR and Cohere-CrispASR (owner W118, W176 route B)

**What the user sees.**
- Two Ensemble pass pipelines for Japanese, under **GGUF** in the Pass 1 / Pass 2 menus: **Parakeet-CrispASR**
  (`parakeet-tdt-0.6b-ja`, 674 MB) and **Cohere-CrispASR** (`cohere-asr-ja-v0.1` by default, or `cohere-asr-ja`, 2,424 MB
  each). Experimental in 1.10.0 (early release) — tooltip and `--help` say so (W212 D4 1). (Cohere-CrispASR's line rule
  changed in item 12: aligned, cut at sentence ends.)
- CrispASR comes with WhisperJAV as a Python package (`crispasr 0.8.44`), in the build of the install's accelerator: the CUDA
  build (NVIDIA), the Vulkan build (GPU through Vulkan and Own PyTorch on Windows), the macOS build (Apple GPU), the CPU build
  otherwise. WhisperJAV downloads the model on first use, pinned by commit and SHA-256 (`crispasr/manifest.py`).
- Parakeet-CrispASR cuts subtitles only at natural cues — sentence-end marks, a pause of 0.6 s, or 、/space followed by
  0.4 s — never by characters or duration, with times from the model's tokens (W203, W204). Cohere-CrispASR makes one
  subtitle per VAD group. Parakeet uses fidelity's speech segmenter settings with 6 s groups, Cohere the qwen pipeline's
  (W199 DIR1, W202). Both print the audio analytics with semantic scenes (W211 C2).
- The pass's scene detector (semantic, 12–48 s, by default), speech enhancer, Enhance for VAD only and Customize (Segmenter,
  Enhancer, Scene tabs) work as for the other passes (W212 D3, W214 2, W232 8). With an enhancer, the model is opened per
  file and closed after it, so the two never hold GPU memory together (W217 19).
- A status panel under a GGUF pass: the model (downloaded or not) and where CrispASR runs. `whisperjav --check` lists
  CrispASR's version, build and device and the three models.
- The folder-name note (T24, agreed wording, D-H): "The GGUF (CrispASR) pipelines need folder names in plain Western letters
  (ASCII). When your user folder has Chinese, Japanese or Korean characters, WhisperJAV links or copies the model to a
  plain-named folder (the original stays where it is)." The clause about installing WhisperJAV itself in such a folder was
  dropped after the check of plan step 3 (CrispASR loads and transcribes from a folder with Japanese characters).
- The 1.9.x bring-your-own CrispASR mode is removed: `--mode crispasr`, the pass value `crispasr`, `--crispasr-exe`,
  `--crispasr-backend`, `--crispasr-args` stop with exit 2 and a message naming the new pipelines (W118 3).

**Checked.** End-to-end runs on WJ110 (NVIDIA GPU): Parakeet GGUF on the GPU, 57 subtitles; Parakeet with auditok,
ffmpeg-dsp and Enhance for VAD only, 62; Cohere with `--device cpu`, 76; a GGUF pass with ZipEnhancer and `--device cpu`
loaded the enhancer on the CPU (`D:\WJ110_measure\step3_2026-10-27\e2e\`). The Vulkan build loaded and transcribed on this
PC's NVIDIA card through Vulkan; no AMD or Intel GPU has been tried. Design and decisions:
`docs/plans/design/2026-10-25-crispasr-route-b-design.md` (v3 and section 22 v2), decision record `…-decision-record.md`.

---

## The accelerators — one accelerator per install (owner W212 D1, W217, W232–W234; design section 22 v2, W238)

**What the user sees.**
- Every install is made for one accelerator, chosen once at install and recorded in `whisperjav_stack.txt`: NVIDIA GPU
  (CUDA), GPU through Vulkan (AMD, Intel or older NVIDIA card; Windows), Apple GPU (Metal), Own PyTorch, CPU only (not
  accelerated). Every run and `whisperjav-upgrade` follow the record; a second run of an installer offers it as the default.
- **No run-time question any more.** The GUI box "Accept CPU-only mode" is removed; `--accept-cpu-mode`, `--hf-device` and
  `--qwen-device` are still accepted but hidden (the last two as names for `--device`). `--device auto|cuda|cpu` governs every
  pipeline, the speech enhancers included; `--device cpu` is absolute.
- At the start of a run: an NVIDIA install whose GPU is usable is silent; one whose GPU cannot be used now is the one stop
  (exit 3; the GUI offers "Run on the CPU this time"); every other install shows a box of where each part runs and goes on.
  PyTorch missing or damaged: exit 4 with the command that fixes it. The GUI shows "The run did not start" (exit 2, 4) and
  "The run ended without a run summary" instead of "Finished with failures".
- Windows installer: with no supported NVIDIA GPU, a window asks what the install is made for (CPU only after 20 s without an
  answer; Stop pre-selected with no time limit when an NVIDIA GPU 1.10 cannot use is found, with the reason first). Own
  PyTorch installs everything except PyTorch and shows the one command to add it (also saved in `OWN_PYTORCH_NEXT_STEP.txt`).
- Source installs: `install.py --accelerator nvidia|vulkan|own-pytorch|cpu`; without it, a supported NVIDIA GPU or an Apple
  silicon Mac is chosen by itself, otherwise the same question in the console. `--cpu-only`, `--cuda`, `--force-cuda` still work
  (hidden). A chosen Stop exits 0 (W241 Q3).
- Unsupported computers are refused before anything is installed, in the owner's form: Windows on ARM and before Windows 10;
  Linux with glibc below 2.35; Intel Macs and macOS before 14; Python under Rosetta; Colab and Kaggle (W233 R6, D-F).
- balanced, fast and faster use the GPU only when CTranslate2 itself sees an NVIDIA GPU (a ROCm PyTorch no longer sends them
  to a GPU they cannot use). On an Apple silicon Mac each pipeline says where it runs (T51–T53).
- Lock: a fourth extra `vulkan`; CrispASR on every platform of each extra (PyPI CPU builds, the `+cuda` and `+vulkan` wheels
  by URL and SHA-256). Only two packages were added to `uv.lock` (253 → 255); no other version changed.

**Checked.** Plan step 3 P8 (`D:\WJ110_measure\step3_2026-10-27\`): WJ110 runs above; a CPU-only source install into a scratch
environment (exit 0, record `cpu`, `--check` exit 0); the Vulkan load check; the Japanese install folder. One
`assessment-adversary` review of P2–P6, 17 findings, each checked; fixed with tests or answered (decision record §14.12);
20 test files after the fixes, all passed (`p7_fix_tests.log`). Not checked: any AMD or Intel GPU; a real Own PyTorch install
over another torch; the Windows installer window in a real install (plan step 5); anything on a Mac or a Linux PC.

---

## Item 12 — ChronosJAV timing: the CrispASR aligner and a small Whisper timing model; Cohere-CrispASR aligned (owner W245–W266)

**Why.** "the accuracy of chronosjav has been a big pain point of the users" (W245 C3g): ChronosJAV's subtitles started and
ended at the VAD group's edges, and two sentences (often two speakers) in one VAD group came out as one subtitle.

**What the user sees.**
- **Every ChronosJAV model** (Qwen3-ASR, anime-whisper; also Cohere in ChronosJAV, which 1.10.0 does not offer in the GUI
  or through `--qwen-generator`, and which was not run) now times its text with an **aligner run by
  CrispASR** (`crispasr.align_words`), timestamp mode `aligner_vad_fallback` by default (was `vad_only`). Aligners:
  `wav2vec2-ja` (default, 232 MB), `qwen3-fa-gguf` (529 MB), `parakeet-ctc-ja` (1,137 MB; offered with a warning: its times
  came back evenly spread in WhisperJAV's tests, W255). The PyTorch Qwen3-ForcedAligner stays a choice (its model ID); `vad_only`
  loads no aligner. anime-whisper is no longer forced to `vad_only` with step-down off (W254 Q6).
- **A timing model**, Whisper small through CrispASR's DTW class (488 MB; W255), runs as orchestrator step 8 after the aligner
  is unloaded. Whisper's text is thrown away. It moves a subtitle's start later to the first onset Whisper counts as speech
  (rule T2) and splits a VAD group in two at a mark of the text (。？！… 、 a closing mark, a space) near where Whisper starts
  a new segment (T4); no mark, no split; no number of characters or seconds decides a cut (W203). `--qwen-timing-model none`
  turns it off; without an aligner it moves starts only (T5).
- **Step-down is off by default** for every ChronosJAV model (W266 1). Under `aligner_vad_fallback` with regrouping off, a
  scene the collapse check flags takes the aligner-free times (VAD-group edges, start moved by the timing model, no splits).
  Step-down stays a setting; its retry uses groups up to 6 s (the design note had said "shorter", corrected in v3).
- **Cohere-CrispASR** aligns each VAD group's text with `wav2vec2-ja` after transcription (W248 3) and cuts aligned groups
  only at 。？！…?! (W259, W266 3), each subtitle ending where the next starts; a group the aligner gives nothing keeps one
  subtitle at the VAD group's edges. Parakeet-CrispASR is unchanged.
- **The text is kept** on every failure: an aligner or timing model that cannot be loaded, a VAD group the aligner gives no
  times (now counted and noted), a collapsed scene. Each is one line in the run summary (DRAFT texts).
- New `--qwen-timing-model {whisper-small,none}`; `--qwen-aligner` takes the short names; ensemble `qwen_params` keys
  `aligner_backend` (`crispasr`/`qwen3`/`none`), `aligner_id`, `timing_model`. GUI, Customize → Alignment: Aligner Backend
  (CrispASR / Qwen3 ForcedAligner (PyTorch) / None), Aligner Model following the backend, Timing Model (Whisper). "None"
  switches all three ChronosJAV models to VAD-only with step-down off. The CrispASR status panel also shows under ChronosJAV
  passes (aligner and timing model), and under Cohere-CrispASR passes it lists the aligner.
- **Fixed on the way:** since a0eb82e (2026-04-02; releases 1.8.10.post3 to 1.9.5) a ChronosJAV pass's Customize →
  Alignment values (and a saved preset's) never reached the run; they do now.
- Where it runs: both parts run where the install's CrispASR build runs, so on Vulkan and Apple installs this part of
  ChronosJAV asks for the GPU. `--device cpu` cannot hold them on the CPU; the run-start box and the log say so where the
  build would use a GPU. `--check` lists the three aligner files and the timing-model file. Models are downloaded at first
  use, inside the pass, pinned by commit and SHA-256 (`crispasr/manifest.py`).

**Checked.** Unit tests (new: `test_crispasr_aligner.py`, `test_crispasr_timer.py`, `test_timing_rules.py`,
`test_subtitle_pipeline_timer.py`, `test_chronosjav_timing_wiring.py`) and the affected files, one per process. Real runs on
The Naked Director S02E02 Scene 5 (`D:\WJ110_measure\timing_design_2026-10-27\s5_qwen_default\`, `…\s6\`): Qwen3-ASR with each
aligner and the PyTorch one, timing model off, `vad_only`, anime-whisper — all exit 0; the two-speaker VAD group split at the
speaker change. The owner's GUI click passed (W264). One `assessment-adversary` review; its findings fixed or decided (W266;
design note v3 §15). Not run: a real step-down retry, a real load failure, `--device cpu`, Cohere inside ChronosJAV (gated
model; skipped, W265). Not measured against the reference subtitles; the start rule T2 is to be judged from the owner's
manual tests (W266 2). Design: `docs/plans/design/2026-10-28-chronosjav-timing-c3-c1.md` v3.

---

## Item 5 — Faster Whisper XXL as pass 1; the chosen XXL model reaches XXL (REQ5; #449, #451, #452; owner W19 2, W24 2, W92, W271–W273)

**Why.** #449: "已经设置使用v2模型的，但执行的时候还是会下载v3" — with XXL as pass 2, `--model large-v2` typed into Extra Args was
removed without a word (`byop/xxl_runner.py`, #272) and the GUI sent no model, so XXL ran large-v3. #451 and #452 asked for XXL as
pass 1; the owner said in #449 he would try to add it in the next release.

**What the user sees.**
- XXL Faster Whisper can be pass 1, pass 2 or both (W271 3): `--pass1-pipeline xxl`; in the GUI, the pass 1 menu's group
  External (BYOP). One program (`--xxl-exe`, the Executable field) and one Extra Args box serve both passes; each pass has its model.
- The model reaches XXL: an XXL pass's Model column is no longer greyed and offers Large V2 (default), Large V3, Large V3 Turbo,
  Medium (W271 2); the GUI sends it as `--passN-model`. With no model named, XXL runs `large-v2` (W24 2; was `large-v3`;
  `DEFAULT_XXL_MODEL` in `byop/xxl_runner.py`).
- Extra Args: the box starts as `--verbose True --standard_asia` (was `--model large-v3 --verbose True --standard_asia`).
  A model in the extra arguments (`--model X`, `--model=X`, `-m X`, `-mX`, a bare `--model`) is still removed (#272), now
  argument by argument (a quoted value is never cut; 1.9.5 cut the raw text), also from the undocumented side file
  `whisperjav_xxl.txt` next to the program (which until now could override the chosen model), and reported in the console once
  per pass worker — once per pass, or once per file and pass with `--ensemble-serial` (W271 4; DRAFT text).
- An XXL failure is stated in one console line naming the pass, the file and the reason (XXL's exit code and its last error line)
  (W92; DRAFT text). A failed pass 1 still fails the file (W271 1; the orchestrator is unchanged); for pass 2 the line says the
  file keeps pass 1's subtitles "if pass 1 made them" — pass 2 runs even for a file whose pass 1 failed. The Ensemble summary's
  pass column now begins with XXL's exit code instead of "Traceback" (cut at 37 characters); the RUN SUMMARY still says
  "ensemble pass failed".
- **Fixed on the way:** the shipped `whisperjav/config/asr_config.json` held the developer's XXL path and `--model large-v2` in
  Extra Args (in 1.9.5 too); now only `--verbose True --standard_asia`. The developer's GUI writes into that file (WJ110 runs
  WhisperJAV from the repo), so `installer/build_release.py` now stops a release build when it holds an XXL path or any
  absolute path in `ui_preferences` (step "Check shipped settings"; alone: `--check-settings`; owner W275). The GUI's late reload of the XXL settings wrote into
  `AppState.state`, which does not exist, so it always failed; it now loads into the Ensemble tab's state.
- Docs: the published example `whisperjav video.mp4 --pass2-pipeline xxl --xxl-exe …` lacked `--ensemble` and ran a single
  Balanced pass (checked with `--dump-params`); the examples now carry `--ensemble`. `README.md` said "Bring your own pass 2".

**Checked.** New `tests/test_xxl_pass1.py` (27 passed), including two end-to-end runs through `main.py`, the ensemble and the pass
worker with a stand-in XXL program (a script that records its arguments): XXL as pass 1 gets its model, the extra arguments, and
the side-file warning; a failing XXL in both passes gives both lines. With pass 1's XXL settings removed from `main.py`, the first
fails (`s6_mutation_check.py`). 17 related test files, one per process: all passed except one `test_htdemucs_enhancer.py` check of
the installer build script, which this item does not touch. Real runs on The Naked Director S02E02 Scene 5 with the local XXL
r245.2 (`D:\WJ110_measure\xxl_pass1_2026-10-27\s3\`): XXL large-v2 → Qwen3-ASR, Qwen3-ASR → XXL large-v2, XXL large-v2 → XXL
large-v3, XXL with no model (ran large-v2) — all exit 0, XXL's log naming the model it loaded; a failing XXL as pass 1 (d3) and in
both passes (f) — exit 1 with the plain lines. The owner's GUI click passed (W273). One `assessment-adversary` review; its findings
fixed or brought to the owner. Not run: Large V3 Turbo and Medium with the real XXL (they would download); the Extra Args warning
with the real XXL (end-to-end with the stand-in only).

---

## Item 11c — answer-format labels in translated subtitles (#460; owner W279)

**Why.** #460 (Ollama, `huihui_ai/qwen2.5-abliterate:7b`, Japanese → Chinese): `Translation>` and repeated lines stayed in the
translated SRT — 6 of 397 batches, 45 of 2,596 subtitles; reproduced by the reporter on 1.9.5 / PySubtrans 1.6.0 by replaying
saved answers. The reporter's script, run in WJ110 (PySubtrans 1.6.0): as whole answers, four of the five malformed forms give a
subtitle holding the label and no error from PySubtrans's check, so PySubtrans's own retry (`retry_on_error`, on by default) does
not run. Inside a batch of correct entries (review probe), only the two forms with a trailing `Translation>` stay silent; the
other two leave the line unmatched, which PySubtrans already retried. The fifth form (four numbered lines inside
`<Translation>…</Translation>`) already failed the newline and unmatched-line checks and was retried; the retry's broken text was
kept. The reporter's six batches were not seen, so which of these happened there is not known.

**What the user sees.**
- A translated subtitle whose text holds `Translation>` or `Original>` at the start of a row, a `<Translation>` / `</Translation>`
  tag, or a row that is only `#number` (ASCII digits; Windows line ends allowed), is an error in PySubtrans's check, for every
  provider (W279 1; `core.py: apply_format_marker_check()`, which wraps `client.GetParser`; the label word follows PySubtrans's
  task type, `Translation` unless an instruction file sets another). With PySubtrans's defaults (`RETRY_ON_ERROR` on,
  `AUTOSPLIT_ON_ERROR` off) it then asks once more for that batch, with its retry instructions (its own default text when the
  instruction file has none) and the error text naming the subtitles. Nothing is cut or rewritten by WhisperJAV; PySubtrans's
  own post-processing (line breaks, an unclosed `<summary>` cut) runs as before.
- If the saved SRT still holds labels (the retry failed, or a resumed batch kept them), the subtitles are kept and one console
  line names their entry numbers, read from the file actually written — PySubtrans numbers the SRT again, so its own line
  numbers differ once an earlier line is untranslated (found by the review) (W279 2; text approved by the owner as drafted, W280): "[TRANSLATE] WARNING: {n}
  translated subtitles still contain answer-format labels (Translation>, Original> or a #number line): #21, #34 — check them in
  {output file}" (at most 20 numbers). Not checked when the run stops with an error before the save.
- Not caught: a label in the middle of a row ("Hello. Translation> Hello."), a lowercase `translation>`. A translated row that is
  only `#1` (a ranking, a hashtag) is taken as a label: one extra request and a console line.
- The #455 context-tag filter is unchanged (Ollama only).

**Checked.** New `tests/test_translation_format_markers.py` (7 passed): the reporter's six answers (the normal one gives no error,
each malformed one an error naming its subtitle); ordinary text with `#` or `>` is not flagged; Windows line ends and another
task type; three end-to-end runs of `translate_subtitle()` through the real PySubtrans against a stand-in OpenAI-compatible
server — an answer with labels and then a clean one (2 requests, the second carries the error text, clean SRT); labels twice
(2 requests, text kept, the console line names #1, the SRT entry that holds the label); the first line never translated (the
console line names entry 1 of the saved file, where the model's #2 landed). With the call in `translate_subtitle()` removed, the
first two fail (`s11c_mutation_check.py`). 11 translation test files, one per process: all passed. One `assessment-adversary`
review: the console line's numbers (fixed: read from the saved SRT), the cause stated too broadly, a wrong code comment, Windows
line ends and the task type (fixed), the limits above (written here). Real run: `whisperjav-translate --provider
ollama --model qwen2.5:7b -t chinese` on the seven Netflix reference SRTs (`D:\WJ110_measure\translate_460_2026-10-27\`), twice:
before the review (`run1_before_review\`) 39 batches, 2 retried; on the final code 39 batches, 3 retried — every retry passed
validation (the log does not say which check failed), exit 0, no label in any output, no warning line. Not seen: a real model
writing labels (the reporter's model was not run).

## Item 6 — the GUI in Simplified Chinese (#448; REQ6; owner W283, W286–W288)

Texts approved by the owner as drafted (W290; the release notes with a Key points line).

**Why.** Roadmap R01: "The option for users to switch UI language between English and Chinese… For 1.10.0 the scope can be UI
only. The program outputs and docstrings and such can stay as is." In #448, users translated 1.9.3, 1.9.4 and 1.9.5 by hand
with an AI tool; every update brought the English back.

**What the user sees.**
- A **中文** / **English** button in the header (next to the theme button) switches the window at once; the file list and a
  running job are not touched. In Chinese: the window, dialogs, tooltips, placeholders, status texts, confirm boxes, and the
  Customize Parameters labels and descriptions (W283 2). Stays English: the console (the GUI's own log lines too), file names,
  messages from the program (also inside Chinese pop-up boxes), the Parameter Guide page (`qwen_guide.html`), the Windows file
  dialogs' filter names, the window title.
- The choice is saved in `gui_settings.json` (`ui_language`) and kept whatever **Remember settings** is (W283 3). Until the user
  first switches, the GUI follows the display language: Windows `GetUserDefaultUILanguage()` (any Chinese → Chinese; Traditional
  Chinese Windows gets Simplified), elsewhere the locale (`LANG`); so a 1.9.x user on Chinese Windows sees Chinese at the first
  1.10.0 start. A Mac app started from Finder often has no Chinese locale set and then starts in English.
- Simplified Chinese only (W283 1). Terms, by the owner (W286–W288): Ensemble tab 专家模式, Transcription Mode tab 基础模式,
  the second tab 高级选项, Pass 第一轮 / 第二轮, Aggressive / Balanced / Conservative 高灵敏 / 中灵敏 / 低灵敏, transcription
  转写 (translation 翻译), making subtitles 生成字幕, translating subtitle files AI 字幕翻译 with the provider area naming
  大语言模型. Other terms follow docs/zh; pipeline, model and program names stay as they are. "Recognizer" in an English
  tooltip is "ASR 模型" (W10); the GPU status lines say 请求使用 ("asks for"), as the English does.
- docs/zh takes the same terms (W288): 14 pages, about 175 lines (集成模式 → 专家模式, 通道 → 轮, 激进 / 均衡 / 保守 → 高 / 中 /
  低灵敏, 转录 → 转写, 语音识别 → ASR); the GUI guides (English and Chinese) get a section on the switch, the Chinese one a
  table of the main control names in both languages.

**How it works.** One catalogue, `webview_gui/assets/i18n_zh.js` (818 texts, 78 kept as they are), keyed by the English text.
`I18n` in `app.js` translates what the window shows, and a MutationObserver translates what `app.js` and the Python schemas add
later, so `api.py`, the YAML files and the English HTML stay as they are. A key may hold computed parts (`{name}`); a part that
is itself a catalogue text is translated too, a part named `raw_…` (a preset name, a file, a message from the program) never.
Switching back restores the English kept per node. Code that reads shown text back (`pipelineLabel`, the " (N/A)" menu marks,
the Apply / Test Connection buttons, the saved Balanced segmenter list) reads the English. "Balanced" is translated as a
sensitivity only inside the pass sensitivity menus (it stays as the pipeline name). The ClearVoice warning was rebuilt from
whole sentences; its English is unchanged. Saving `gui_settings.json` now holds a lock (the language switch and the form save
run on separate threads).

**Checked.** New `tests/test_gui_i18n_zh.py` (14 passed): every text from `index.html`, the `app.js` literals that reach the
window or a pop-up box, the Customize schemas and the CrispASR panel lines in `api.py` and `accelerators.py`, the tool YAML
`gui` sections, the Hugging Face model list and the labels `app.js` builds from parameter and group names must have a Chinese
entry or be listed as kept; the catalogue's form; one Chinese text per English text (except AI 字幕翻译, the owner's choice for
both the tab and the tick box); no 识别 and none of the replaced terms; the saved language (first start Chinese / English,
kept with Remember settings off, a bad value refused). `tests/test_gui_settings.py` (56 passed; 36 keys now). The other GUI
test files, one per process: all passed. 18 lookup checks in Node (`node_lookup_check.js`). The real GUI driven by a script
(`drive_gui.py`, settings file in a temporary folder) through the button, every tab, every pass-1 pipeline's Customize window
and tabs, pass 2, Translation Settings and About: no errors; the click saved `zh-CN`, then `en`; left in English only preset
names and model names; after switching back, no Chinese left. One `assessment-adversary` review: the GPU lines said "uses"
(fixed: 请求使用), one 识别 (fixed), a preset named "Default" shown as 默认 (fixed: data parts never translated, the preset
badge skipped), the Apply / Test Connection text in the old language after a quick switch (fixed), a possible lost save
(fixed: lock); it could not refute the rest (no runaway work with 3,000 console lines, values and typed text untouched, the
saved language through restart and Remember settings on/off).

**Limits.** The window shows English for under a second at start-up, before the saved language is applied. Text split by
inline tags is translated in pieces ("将要下载 4.7 GB."). The owner's click test of the final build is still to come.

---

## Found during the work, not fixed in 1.10.0 (requirements §5; places as decided or proposed)

| | Finding | Place |
|---|---|---|
| F1 | BS-RoFormer cannot start (`unexpected keyword argument 'device'`); the run goes on with the original audio and a log WARNING only. CLI only — the GUI shows it as "Coming Soon" | 1.10.1 (W88) |
| F2 | ZipEnhancer fails on the second file of a run (PyTorch thread setting allowed once per process); later files get no enhancement, log WARNING only | 1.10.1 (W88) |
| F3 | `zipenhancer:onnx` memory runaway (62 GB observed) | item 9 spec (1.10.1, W91) |
| F4 | htdemucs output not repeatable (random time offset) | item 9 spec (1.10.1, W91) |
| F5 | an upgrade does not remove packages the lock dropped | 1.10.1 (proposed) |
| F6 | `docs/architecture/DEPENDENCY_CROSS_MATCH.md` out of date | 1.10.1 (proposed) |

F1 and F2 go against the owner's rule of 2026-09-17: "A clean-up the user SELECTED that cannot install or cannot start stops the run."
Also known, not fixed: the launcher-icon step (`embed_icon_in_exe`) fails with lief 1.0.0, as in 1.9.5, not fatal (owner's call);
constructor 3.17.3 removes the Users group's write permission on the install folder (left as stock, noted for the owner).

## For the release notes — still to be written or decided

- Owner's wording: 1.10 is a fresh install for 1.9 users; Colab stays on 1.9.5; ClearVoice removed (M3: "ClearVoice speech
  enhancement is no longer included. If you used it, choose ZipEnhancer, Vocals or None."); the statement for users in China who reach
  GitHub through a VPN (M4).
- Settled 2026-10-17 (W93): the six item 2 console texts drafted while coding (item 2 spec, "Changes found while coding" 11 (a)–(f))
  approved (c1); the closing dialog of a deliberate installer stop reworded, pre-approved (c2): "WhisperJAV was not installed. / The
  installer stopped for the reason shown in its console window. / That reason is saved in this file: / <path>" (template change not
  yet committed; the installer built 2026-10-17 has the earlier wording).
- Public docs: rewritten for 1.10.0 in plan step 4 (2026-10-27) in English — getting started, the four install guides, the
  GGUF page, GUI guide, CLI reference, FAQ, translation, workflows, upgrade pages, `README.md`; Chinese pages to follow.
  Earlier list, for the record: llama-cpp pages (translation, GUI, install guides, en and zh); ClearVoice mentions in `README.md` and
  `docs/en`; `docs/en/guides/installation_windows_standalone.md:177` ("Phase 3.5"); the installer readme template's Local LLM step and
  driver table (listed 2026-10-14, not re-checked since).
- Issues the release touches: #459 (CTranslate2 4.8.3). Whether any reporter is told "fixed" is the owner's decision.
