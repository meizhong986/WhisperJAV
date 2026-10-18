# WhisperJAV v1.10.0 — Development Change Log

> Developer-facing record of what changed on branch `v1.10.0` since the 1.9.5 release (`8917f5a`, tag `v1.9.5`), why, and how it
> was checked. It is the source for the 1.10.0 release notes; it is not itself a user text.
> Ordered by requirement item (`docs/requirements/v1.10.x/2026-10-13-release-1.10.0-requirements.md`), not by date. Each entry says
> what the user sees, the commits, the owner's decisions (W-numbers in `docs/plans/OWNER_WORDS_V110_CYCLE.md`), how it was checked, and
> what was found and left open. Specs: `docs/requirements/v1.10.x/specs/`. Both folders are git-ignored.
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
- Public docs not yet changed: llama-cpp pages (translation, GUI, install guides, en and zh); ClearVoice mentions in `README.md` and
  `docs/en`; `docs/en/guides/installation_windows_standalone.md:177` ("Phase 3.5"); the installer readme template's Local LLM step and
  driver table (listed 2026-10-14, not re-checked since).
- Issues the release touches: #459 (CTranslate2 4.8.3). Whether any reporter is told "fixed" is the owner's decision.
