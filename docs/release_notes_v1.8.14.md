# WhisperJAV v1.8.14 — Quality Hardening + Bug Fixes on top of v1.8.13

Stability and quality update. An automatic safety cap for the v1.8.13 ensemble Pass 2 catastrophe, an ASR preset retune, several bug fixes, and an FAQ accuracy pass.


CATASTROPHE PROTECTION:
- Auto safety cap for ensemble Pass 1=Fidelity + Pass 2=Balanced + Aggressive (eliminates the ~67% Pass 2 truncation rate)

BUG FIXES (8):
- Korean transcription: lines no longer dropped as "Symbol Only Residue" (#324)
- DeepSeek translation: default bumped to `deepseek-v4-flash` before 2026-07-24 deprecation (#325)
- Kaggle parallel notebook: failed-pass log shows last 1500 chars; HF_TOKEN setup in header (#321)
- Kaggle expert notebook: qwen added to pass1/pass2 dropdowns, translate argv split fixed (#315, #318)
- `--dump-params` in `--ensemble`: now produces populated `ensemble_config` block (#312)
- GUI Check-for-Updates: in-GUI pip-install retired, now opens the Releases page
- Defensive `transformers<5.0` pin: protects fresh installs from breaking Qwen3-ASR (#280)
- Sanitizer regression: 11 new tests cover Korean + 9 other scripts (#324)

QUALITY IMPROVEMENTS:
- ASR preset retune across all 3 sensitivities, both `faster-whisper` and `openai-whisper`
- FAQ accuracy: timing table grounded in field data; VRAM table aligned to GUI labels
- Chinese FAQ parity for the truth-fixes (6 of 7 sections)

DOCS:
- New install-troubleshooting FAQ: proxy (#317), macOS SSL (#320), Phase 3 silence (#314), transformers pin (#280)

DEFERRED:
- Cohere-Transcribe preview deferred to v1.9.0 — code is in the build but greyed out


---


## Conditional Sensitivity Cap (Ensemble Pass 2)

- **What it does** — When ensemble runs Pass 1=Fidelity + Pass 2=Balanced at Aggressive sensitivity, WhisperJAV now auto-downgrades the Pass 2 sensitivity to `balanced` and prints a console warning. I observed this combination drop Pass 2 to ~14 entries instead of ~50 on a 293-second JAV reference clip at roughly 67% rate during investigation.

- **Quality** — Output entry counts are within a few percent of v1.8.13 healthy-run numbers. The downgrade removes only the temperature-fallback path that was the underlying mechanism.

- **Workaround** — If you specifically want Aggressive Pass 2 sensitivity, use Balanced (or Fast/Faster) as Pass 1 instead of Fidelity. The cap doesn't fire in those combinations.


## ASR Preset Retune

- **Across all three sensitivities, both engines** — `best_of` unified to 2 across Conservative/Balanced/Aggressive (was inconsistently 1 on the lower two), patience tightened on Aggressive for speed, logprob gates relaxed slightly to recover lines that were being filtered too aggressively.

- **Aggressive `faster-whisper`** — temperature relaxed `[0.0, 0.17]` → `[0.0, 0.2]`, no-speech threshold lowered `0.84` → `0.72`, patience tightened `2.0` → `1.3`. This addresses the underlying mechanism behind the v1.8.13 ensemble catastrophe (alongside the safety cap above).

- **Validated** — 10 cap-disabled `fid→bal+aggressive` ensemble runs on the same JAV reference clip, all catastrophic-free.


## Korean Transcription Fix (#324)

- **Symptom** — On v1.8.13, Korean source audio produced near-empty SRT output. The `stitched.artifacts.srt` file showed nearly every line flagged `[REMOVED - Symbol Only Residue]`.

- **Cause** — A defense-in-depth filter from v1.8.11 (designed to drop pure-symbol residue like `!!`) only recognized Hiragana, Katakana, CJK ideographs, and Latin as "letters". Korean Hangul wasn't in that class, so any Korean-only line was dropped as symbol residue. The same regression affected Cyrillic, Arabic, Thai, Hebrew, Devanagari, Greek, and diacritic-Latin output silently — Korean was just the first language community to file it.

- **Fix** — Filter extended to cover all major scripts Whisper produces. 11 new regression tests so this can't recur silently for any other language.


## GUI Check-for-Updates Redirect

- **Why** — Field reports showed the in-GUI pip-install upgrade path was fragile, especially on the standalone Windows installer. It could leave installs partly upgraded.

- **What changed** — The "Update to Stable" / "Update Now" buttons now open the GitHub Releases page where you follow the documented installer/upgrade instructions per platform. Version-check display, release notes preview, and dev-branch commits-ahead view are all unchanged.


## FAQ Accuracy Cleanup

- **Timing table was wrong** — The v1.8.13 table claimed Balanced GPU 2hr = ~5 min and didn't list Fidelity at all. Field reports show Fidelity/Aggressive on a 2hr movie is 40-50 min (RTX 3060 12GB / Colab T4). New table is anchored on those numbers and notes ASMR-difficult content can push 1.2-1.5x further.

- **VRAM table aligned to GUI labels** — Numbers now match the dropdown labels users actually see (`fast`=4GB, `faster`=6GB peak, `balanced`=6GB, `fidelity`=10GB). Removed an unfounded "fallback to CPU offload" claim — no such code path exists.

- **Marketing language scrubbed** — "maximum accuracy", "most natural translations", "best for most content" replaced with factual trade-off descriptions.


## DeepSeek Default Model (#325)

- **Why** — DeepSeek deprecates `deepseek-chat` and `deepseek-reasoner` on 2026-07-24. Without this update, default DeepSeek translation would break for everyone on that date.

- **What you'll see** — New invocations of `--provider deepseek` (no model override) now hit `deepseek-v4-flash`. Existing user settings keep working until the cutoff. For the thinking variant: `--model deepseek-v4-pro`.


## Cohere-Transcribe Deferred to v1.9.0

- **What was planned** — Cohere-Transcribe preview as a third ChronosJAV generator alongside Qwen3-ASR and Anime-Whisper.

- **Why deferred** — Cohere needs `transformers >= 5.4.0`, but `transformers` 5.x breaks the bundled Qwen3-ASR fork (decorator API change). Shipping Cohere as-is would have broken Qwen3-ASR for every existing user — an unacceptable trade.

- **What's in v1.8.14** — The Cohere code is in the build but the dropdown entry is greyed out, and a defensive `transformers<5.0` pin protects existing installs. v1.9.0 will land the Qwen3-ASR fork patch and the transformers upgrade together. Sorry to anyone who was looking forward to trying Cohere this release.


---


### How to Upgrade or Install

**Upgrade from 1.8.13:**

```
pip install -U --no-deps "whisperjav @ git+https://github.com/meizhong986/whisperjav.git@v1.8.14"
```

For the standalone Windows installer: in the GUI, click **Check for Updates** → **Open Release Page**, then download the new `.exe` and run it over the existing install.

**Fresh Install:**

### Windows — Standalone Installer (.exe)

1. Download **WhisperJAV-1.8.14-Windows-x86_64.exe** from the Assets below
2. Run the installer (no admin rights required)
3. Wait 10-20 minutes for setup to complete
4. Launch from the Desktop shortcut

Installs to `%LOCALAPPDATA%\WhisperJAV`. A desktop shortcut is created automatically. Your GPU is detected automatically.

### macOS

Requires [Git](https://git-scm.com/downloads). The install script checks for everything else (Xcode CLI Tools, Python, FFmpeg, PortAudio) and tells you what to install if anything is missing. Open Terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.14
python install.py
```

After installation, double-click **WhisperJAV.command** to launch the GUI.

### Linux

Requires Git and Python 3.10-3.12. Open a terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.14
python install.py --cuda cu128
```

After installation, launch the GUI with `./WhisperJAV.sh`.

### Windows — Source Install

Requires [Git](https://git-scm.com/downloads) and [Python 3.10-3.12](https://www.python.org/downloads/). Open a terminal and run:

```
cd %USERPROFILE%
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.14
python install.py --cuda cu128
```

After installation, double-click **WhisperJAV.bat** to launch the GUI.


## Compatibility

Same as v1.8.13. New: defensive `transformers<5.0` pin (see Deferred above).

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 - 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |
| transformers | 4.40 - 4.x |
| Ollama | 0.3.0+ recommended |


## Known Issues

- **Cohere preview is greyed out** — deferred to v1.9.0 (see above).
- **WhisperSeg in simple Transcription Mode falls back to silero-v3.1** — carried from v1.8.13. Use Ensemble mode if you want WhisperSeg. Fix lands in v1.9.0.
- **Default Whisper model is `large-v2`** — carried from v1.8.13. v1.9.0 retunes the aggressive preset for `large-v3` so v3 can return as default.
- **No automatic CPU-offload for low VRAM** — on a 4 GB GPU only Fast mode reliably fits.
- **Chinese FAQ has gaps relative to English** — accuracy fixes are now consistent; some sections (Models & Cache, 4 GB VRAM, install troubleshooting) are English-only.


## What's Next (v1.9.0)

- Cohere ship — Qwen3-ASR fork patch + transformers 5.x upgrade
- Re-tune aggressive preset for `large-v3` so v3 can return as the default Whisper model
- Unified segmenter param routing — eliminates the simple-mode WhisperSeg restriction
- Standalone subtitle merge CLI (`whisperjav-merge`) (#230)
- Chinese GUI partial i18n (#175, #180)
- Speaker diarization (#248, #252)
- Settings persistence for API keys and prompt fields (#96)


---

Thank you to everyone who reported issues, attached log files, and tested the fixes.
