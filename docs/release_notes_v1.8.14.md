# WhisperJAV v1.8.14 — Quality Hardening + Bug Fixes on top of v1.8.13

A focused stability release. The big items: an automatic safety cap for the ensemble Pass 2 catastrophic-truncation pattern that several users hit in v1.8.13, a comprehensive ASR preset retune across all sensitivities, a fix for Korean transcription dropping nearly all output, and the GUI Check-for-Updates flow now points users to the GitHub Releases page instead of running an in-place pip upgrade.

---

## Quality Improvements

- **ASR preset retune across all 3 sensitivities** — I went through the faster-whisper and openai-whisper presets for Conservative, Balanced, and Aggressive sensitivity and updated several values that were producing fragile behavior in edge cases. Two motivations, kept separate by design:
  - On the **Aggressive** path of faster-whisper specifically, the temperature fallback combined with the high no-speech gate was the underlying mechanism behind the v1.8.13 ensemble catastrophe (see the new safety cap below). Aggressive temperature relaxed from `[0.0, 0.17]` to `[0.0, 0.2]`, no-speech threshold lowered from 0.84 to 0.72, patience tightened from 2.0 to 1.3.
  - **Quality across all three sensitivities, both engines** — `best_of` unified to 2 across all sensitivities (it was inconsistently 1 on Conservative and Balanced), patience tightened on Aggressive for speed, and logprob gates relaxed slightly to recover lines that were being filtered too aggressively. faster-whisper and openai-whisper kept asymmetric where the engine difference is real (different inference paths, different numerical behavior).

  Validated on a representative ensemble fid→bal+aggressive run series (10 runs with the safety cap disabled, target 0/10 catastrophic) — all clean. Output entry counts are within a few percent of the v1.8.13 healthy-run numbers.

- **FAQ accuracy** — I went through the FAQ and corrected several claims that didn't hold up to scrutiny. The transcription-time table was the main one (it didn't even list Fidelity, and the GPU numbers underestimated reality). The new table is anchored on field-observed numbers from RTX 3060 12 GB and Google Colab T4 free tier — Fidelity/aggressive on a 2-hour movie is 40-50 minutes, not 10. ASR-difficult content (ASMR-style whispered audio) can push that past 1 hour. The 4 GB VRAM table now matches the labels users actually see in the GUI dropdown. Several "best/most natural/maximum accuracy" marketing claims removed.

- **Tracker rev49.x cleanup** — eight rounds of triage, response posting, and reconciliation across the 70+ open issues since v1.8.13 shipped. No big news here, but the issue tracker is the cleanest it has been in a while.

---

## New

- **Conditional sensitivity cap (ensemble Pass 2 protection)** — When you run ensemble mode with Pass 1 = Fidelity + Pass 2 = Balanced + sensitivity = Aggressive, WhisperJAV now auto-downgrades the Pass 2 sensitivity to `balanced` and prints a console warning. This is a documented workaround for an intermittent catastrophic truncation pattern in that specific configuration that I observed at roughly a 67% rate during investigation — Pass 2 would drop to ~14 entries instead of ~50 on the same 293-second JAV reference clip. Output quality is not affected by the downgrade — Pass 2 still runs faster-whisper at the `balanced` sensitivity preset; only the temperature fallback path is removed. Other ensemble configurations are unaffected.

  Workarounds for users who explicitly want Aggressive Pass 2 sensitivity:
  - Use Balanced as Pass 1 instead of Fidelity (no cap fires; Aggressive Pass 2 is stable in this combination).
  - Use Fast or Faster as Pass 1 (no cap fires).

  A deeper architectural fix is planned for v1.9.0+. The cap is a documented workaround, not a root-cause fix.

- **Defensive `transformers<5.0` pin** — pinned in `pyproject.toml` and the installer registry. transformers 5.x breaks the bundled Qwen3-ASR fork (decorator API change), so v1.8.14 protects fresh installs from accidentally pulling in a broken combination. See the FAQ for the manual downgrade path if your environment already has transformers 5.x.

---

## Bug Fixes

- **Korean transcription dropped almost all output as "Symbol Only Residue" (#324)** — a subtitle-sanitizer regression that shipped in v1.8.11 silently affected every non-CJK script. The defense-in-depth filter that drops symbol-only lines (`!!`, `???`, `♪♪♪`) was scoped to Hiragana / Katakana / CJK ideographs / Latin only; Korean Hangul, Cyrillic, Arabic, Thai, Hebrew, Devanagari, Greek, and Latin-with-diacritics characters were not recognized as letters and so any line containing only those scripts was dropped as residue. Korean was just the first language community to file it. Fixed by extending the filter to cover all major scripts Whisper produces. 11 new regression tests so this can't recur.

- **GUI Check-for-Updates redirects to the Releases page (was: in-GUI pip install)** — the previous behavior tried to upgrade WhisperJAV in place by running pip from inside the running app. On the standalone Windows installer this was fragile and could leave installs partly upgraded. Replaced with an "Open Release Page" button that takes you to the GitHub Releases page for documented installer/upgrade instructions per platform. Version-check display, release notes preview, and dev-branch commits-ahead view are all unchanged.

- **DeepSeek translation default model bumped to `deepseek-v4-flash` (#325)** — DeepSeek is deprecating `deepseek-chat` and `deepseek-reasoner` on 2026-07-24. Without this update, DeepSeek translation would have started failing for everyone on that date. v1.8.14 ships well before the cutoff. Users who explicitly set `--model deepseek-chat` will continue to work until the deprecation. Users who want the thinking variant can use `--model deepseek-v4-pro`.

- **Kaggle parallel notebook: better Pass 1/2 failure diagnostics (#321)** — the failure preview in the notebook was hardcoded to `log[:200]` characters, which usually only showed the leading "sending unauthenticated requests to HF Hub" warning rather than the actual exception. Now shows the last 1500 characters where Python tracebacks live. HF_TOKEN setup instructions added to the notebook header (Kaggle Secrets / Colab sidebar).

- **Kaggle expert notebook fixes (#315, #318)** — `qwen` was missing from the pass1/pass2 quality dropdowns in cell 1. Cell 3's translation step was passing the multi-token Kaggle command as a single argv token, producing `[Errno 2] No such file or directory: '/usr/bin/python3 -m whisperjav.translate.cli'`. Both fixed by mirroring the split pattern that already worked in cell 2. Notebook installs from `git+...@main`, so re-running Step 2 picks up the fix.

- **`--dump-params` in `--ensemble` mode now produces useful output (#312)** — the v1.8.13 dump_params fix only covered the legacy-resolver path. In ensemble mode, `resolved_config` is intentionally `None` because the EnsembleOrchestrator owns config resolution per-pass downstream. The dump output was a near-empty payload. Now contains an `ensemble_config` block with the full plumb-through for both passes (pipeline, sensitivity, scene_detector, speech_segmenter, speech_enhancer, model, language, device, etc.) plus a `_safety_cap_applied` block when the conditional_sensitivity_cap fires.

---

## Documentation

- **Install troubleshooting FAQ entries** — added entries for installer behind a corporate proxy (#317), macOS SSL CERTIFICATE_VERIFY_FAILED (#320), Phase 3 (PyTorch) appearing silent for several minutes (#314), and Qwen3-ASR `check_model_inputs()` TypeError on transformers 5.x (#280). All four were carry-over user reports from earlier releases.

- **Chinese FAQ parity for the truth-fixes** — the same accuracy cleanup applied to `docs/zh/faq.md` (timing table, scene-detector descriptions, translation provider descriptions, hallucination tips, Linux WebKit, "which pipeline gives best subtitles"). Six of the seven English fixes ported; the seventh (4 GB VRAM section) doesn't have a Chinese counterpart yet — that's part of a wider Chinese-translation gap I'll close in a future release.

---

## Deferred

- **Cohere-Transcribe preview** — was planned to ship in v1.8.14 as a third ChronosJAV generator alongside Qwen3-ASR and Anime-Whisper. The full implementation is committed in the v1.8.14 codebase but is **dormant** — the dropdown entries are greyed out, and the defensive `transformers<5.0` pin blocks the upgrade path that would activate it. Reason: Cohere requires `transformers >= 5.4.0`, but transformers 5.x breaks the bundled Qwen3-ASR fork (decorator API change). Shipping Cohere would have broken Qwen3-ASR for every existing user, which is an unacceptable trade. v1.9.0 will land the Qwen3-ASR fork patch and the transformers upgrade together, then enable Cohere. The upstream issue is being tracked at https://github.com/QwenLM/Qwen3-ASR/issues/138. Sorry to anyone who was looking forward to trying Cohere this release.

---

## How to Upgrade or Install

### Upgrade from v1.8.13 (source install)

```
pip install -U --no-deps "whisperjav @ git+https://github.com/meizhong986/whisperjav.git@v1.8.14"
```

### Upgrade from v1.8.13 (standalone installer)

The in-GUI updater no longer performs pip upgrades. Click **Check for Updates** in the GUI, then **Open Release Page** — download the new `.exe` from the Releases page and run it over the existing install. Your settings and caches are preserved.

### Fresh Install

#### Windows — Standalone Installer (.exe)

1. Download **WhisperJAV-1.8.14-Windows-x86_64.exe** from the Assets below
2. Run the installer (no admin rights required)
3. Wait 10-20 minutes for setup to complete
4. Launch from the Desktop shortcut

Installs to `%LOCALAPPDATA%\WhisperJAV`. A desktop shortcut is created automatically. Your GPU is detected automatically. If you are behind a corporate proxy, set `HTTPS_PROXY` / `HTTP_PROXY` before running the installer (see the FAQ entry).

#### macOS

Requires [Git](https://git-scm.com/downloads). The install script checks for everything else (Xcode CLI Tools, Python, FFmpeg, PortAudio) and tells you what to install if anything is missing. Open Terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.14
python install.py
```

If preflight fails with `SSL: CERTIFICATE_VERIFY_FAILED`, run `/Applications/Python\ 3.12/Install\ Certificates.command` (replace 3.12 with your Python version) and retry. WhisperJAV needs stable Python 3.10-3.12 — avoid alpha builds.

After installation, double-click **WhisperJAV.command** in the project folder to launch the GUI.

#### Linux

Requires Git and Python 3.10-3.12. Open a terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.14
python install.py --cuda cu128   # or cpu / cu118
```

After installation, launch the GUI with `./WhisperJAV.sh`.

#### Windows — Source Install

Requires [Git](https://git-scm.com/downloads) and [Python 3.10-3.12](https://www.python.org/downloads/). Open a terminal and run:

```
cd %USERPROFILE%
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.14
python install.py --cuda cu128
```

After installation, double-click **WhisperJAV.bat** to launch the GUI.

#### Colab / Kaggle

Open the latest notebook from main:
- [Kaggle Parallel Edition](https://github.com/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_kaggle_parallel_edition.ipynb)
- [Colab Expert Edition](https://github.com/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_edition_expert.ipynb)

For qwen ASR or any gated HF model, set `HF_TOKEN` first (Kaggle: Add-ons → Secrets → Add `HF_TOKEN`; Colab: left sidebar key icon → Add `HF_TOKEN`).

---

## Compatibility

Same as v1.8.13 — no dependency changes other than the new `transformers<5.0` pin.

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 - 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |
| transformers | 4.40 - 4.x (5.x not yet supported, see FAQ) |
| Ollama | 0.3.0+ recommended |

---

## Known Issues

- **Cohere preview is greyed out** — see the Deferred section above. v1.9.0 will land the Qwen3-ASR fork patch and the transformers 5.x upgrade together.

- **WhisperSeg in simple Transcription Mode still falls back to silero-v3.1** — carried over from v1.8.13. WhisperSeg is the default in Ensemble / Qwen / Decoupled paths; in simple `--mode balanced` / `--mode fidelity` it is intentionally restricted because of a config-routing bug that produces catastrophic empty output on JAV moaning content. The fix lands in v1.9.0 (unified segmenter param routing). Workaround: use Ensemble mode if you want WhisperSeg.

- **No automatic CPU-offload fallback for low VRAM** — if a transcription mode requests more VRAM than is available, you get a CUDA out-of-memory error rather than a graceful fallback to CPU. On a 4 GB GPU only Fast mode reliably fits. The FAQ has a full per-mode VRAM table.

- **Default Whisper model is still `large-v2`** — carried over from v1.8.13. The aggressive ASR preset values are tuned against large-v2; large-v3 produces catastrophic empty output on JAV moaning content with the current preset. Users who want large-v3 can opt in with `--model large-v3` (CLI) or via the GUI model override checkbox. v1.9.0 will re-tune the aggressive preset for large-v3 so v3 can return as the default.

- **ZH FAQ has gaps relative to EN** — the Chinese FAQ does not yet contain the Models & Cache section, the 4 GB VRAM section, the Why-Fidelity-variance section, or the four install-troubleshooting entries that the English FAQ has. The accuracy fixes that exist in both files are now consistent. Full ZH parity is on the post-v1.8.14 docs queue.

---

## What's Next (v1.9.0)

- **Cohere ship**: coordinated Qwen3-ASR fork patch + transformers 5.x upgrade + Cohere dropdown un-grey
- **Re-tune aggressive ASR preset for `large-v3`** so v3 can return as the default Whisper model
- **Unified segmenter param routing** — eliminate the simple-mode WhisperSeg/TEN restriction; WhisperSeg can become the default everywhere
- **Standalone subtitle merge CLI** (`whisperjav-merge`) (#230)
- **Chinese GUI** (partial i18n) (#175, #180)
- **Speaker diarization** (#248, #252)
- **Settings persistence** for API keys and prompt fields (#96)
- **English-source ASS/SSA support** (#308 part 2)

---

## Issues addressed

| Issue | Reporter | Resolution |
|---|---|---|
| #312 (follow-up) | TinyRick1489 | `--dump-params` ensemble_config block populated |
| #315 | jasial2 | Kaggle expert translate command argv split |
| #318 | jasial2 | qwen added to Kaggle expert pass1/pass2 dropdowns |
| #321 | jasial2 | Kaggle Pass 1/2 failure diagnostics + HF_TOKEN setup doc |
| #324 | micky0526 | Korean / non-CJK transcription not dropped as symbol residue |
| #325 | 1Dreamer666 | DeepSeek default model bumped before deprecation |
| #314 | zoqapopita93 | Phase 3 progress diagnostic in FAQ |
| #317 | zoqapopita93 | Installer proxy setup in FAQ |
| #320 | Madaerpao | macOS SSL cert install in FAQ |
| #280 | zoqapopita93 | Defensive transformers<5.0 pin + downgrade workaround in FAQ |
| (no issue) | (multiple users) | Conditional sensitivity cap for the v1.8.13 ensemble fid→bal+aggressive catastrophe |

---

Thank you to everyone who reported issues, attached log files, and tested the fixes. v1.8.14 is the result of careful triage rather than dramatic new features — that's by design. v1.9.0 is where the bigger changes land.
