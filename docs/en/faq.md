# Frequently Asked Questions

---

## General

### What video formats does WhisperJAV support?

Any format FFmpeg can read: MP4, MKV, AVI, MOV, WMV, FLV, WAV, MP3, FLAC, M4A, M4B, and many more. If FFmpeg can extract audio from it, WhisperJAV can process it.

### How long does transcription take?

Depends on the mode, GPU tier, and how challenging the audio is. The numbers below are field-observed ranges on a **mid-tier GPU** (RTX 3060 12 GB or Google Colab T4 free tier) for typical JA dialogue content. Higher-end GPUs (RTX 3090/4080/4090) are roughly 1.5-3x faster; CPU-only is significantly slower (often 5-10x the slowest GPU row).

| Mode | 30-min clip | 2-hour movie | Notes |
|------|-------------|--------------|-------|
| **Faster** | 2-4 min | 6-10 min | Single-pass turbo-int8-batch8; fastest mode |
| **Fast** | 3-5 min | 8-15 min | Single-pass turbo-int8 |
| **Balanced** | 4-7 min | 15-25 min | Single-pass large-v2-int8 (recommended default) |
| **Fidelity** | 10-15 min | 40-50 min* | Single-pass large-v2-fp16; high quality, high variance |
| **Ensemble** | 15-25 min | 60-90 min | Two-pass (Pass 1 + Pass 2 sequential, then merged) |

*Fidelity has high run-to-run timing variance (4-5x range on the same input) due to PyTorch cuDNN auto-tune, see "Why does Fidelity pipeline sometimes take much longer than other times?" below. Quality is unaffected, only wall-clock.

**Audio difficulty multiplier**: very dense whispered content (ASMR-style, soft moaning, breathy speech) can push processing time **1.2-1.5x** the values above, especially on Fidelity and Ensemble modes - the model spends more time on close-call frames. A 2-hour ASMR-heavy clip on Fidelity can exceed 1 hour wall-clock.

**CPU-only**: Faster mode is the only realistic option without a GPU. Expect roughly 30-60 minutes for a 30-minute clip; multi-hour content on CPU is impractical for most users.

### Can I use WhisperJAV without a GPU?

Yes. Go to **Advanced** → check **"Accept CPU-only mode"**. Use **Faster** mode for best CPU speed. It works, just significantly slower.

---

## Quality

### Which pipeline gives the best subtitles?

There is no universal "best" - it depends on content style and how much time you can spend. Common recommendations from field reports:

- **Anime / dramatic JAV dialogue**: **ChronosJAV** with anime-whisper.
- **General Japanese dialogue, single-pass**: **Balanced** is the typical default.
- **Highest accuracy at the cost of wall-clock**: **Ensemble** (Pass 1 = Balanced + Pass 2 = Qwen3-ASR with Smart Merge), which combines two model outputs.
- **Whispered / ASMR-heavy content**: try **Fidelity** with **aggressive** sensitivity; it can recover quiet speech that other modes drop.

In all cases, run a 5-10 minute representative clip first and compare - quality on JAV content varies enough by performer voice profile that no single recommendation lands consistently across content libraries.

### The subtitles have hallucinated text (random English phrases, URLs, etc.)

This is a known Whisper behavior on silent or very quiet sections. WhisperJAV includes a hallucination sanitizer that removes most of these. Try:

1. Try a different **sensitivity** setting - in some content profiles **Aggressive** captures real speech that **Conservative** drops as silence (which Whisper then fills with hallucinations); on other content the reverse is true. Run a representative clip on each setting to find the sweet spot for your library.
2. Use **Ensemble** mode - two model outputs typically disagree on different hallucinations, so the merge step removes more than either pass alone.
3. Enable a **Speech Enhancer** (ClearVoice or BS-RoFormer) to clean the audio first

### The timing is off / subtitles appear too early or late

Try a different **Scene Detector**:

- **Semantic** (default in v1.8.13+) - audio-feature-based; reasonable starting choice for most content.
- **Auditok** - energy-based; tends to work better on content with clear silence between dialogue lines.
- **Silero** - VAD-based; produces more frequent splits which can either help or hurt timing depending on content.

Or try **ChronosJAV** pipeline, which uses TEN VAD for tighter timing.

### Why does Fidelity pipeline sometimes take much longer than other times?

Fidelity is built on **openai-whisper**, which uses **PyTorch's default cuDNN auto-tuning**. For each run, cuDNN benchmarks several convolution algorithms and picks one for the current GPU state. The benchmark itself is timing-sensitive — different runs can land on kernels with very different performance characteristics, producing wall-clock variance of roughly 4-5× for the same input audio (e.g. 3 minutes vs 15 minutes on the same clip on the same machine).

This is an inherited characteristic of PyTorch + cuDNN, **not a WhisperJAV bug**. **Output quality and entry counts are unaffected** — only timing varies. Among the pipeline modes:

- **Fidelity** (openai-whisper / PyTorch native): timing varies wildly across runs.
- **Balanced / Fast / Faster** (built on faster-whisper / CTranslate2): timing-stable across runs.

If predictable timing matters more than peak speed, use **Balanced** as Pass 1 instead of Fidelity. A future release may add an option to lock cuDNN kernel selection at the cost of average-case throughput, for users who prefer predictability.

### v1.8.14: Why was my "aggressive" sensitivity downgraded to "balanced" in ensemble mode?

If you run ensemble mode with **Pass 1 = Fidelity + Pass 2 = Balanced + sensitivity = aggressive**, WhisperJAV automatically downgrades the Pass 2 sensitivity to `balanced` and prints a console warning. This is the **conditional_sensitivity_cap** introduced in v1.8.14.

The original combination was empirically observed to produce intermittent catastrophic Pass 2 truncation (output dropping to ~14 entries instead of ~50 on the reference clip) at roughly a 67% rate. The cap removes the unstable fallback code path while preserving full transcription quality — Pass 2 still runs faster-whisper at the `balanced` sensitivity preset (deterministic temperature, slightly tighter no-speech gate). Output entry counts may differ by a few percent vs healthy aggressive runs, but no catastrophic drops.

Workarounds for users who explicitly want aggressive Pass 2 sensitivity:
- Use **Balanced** as Pass 1 instead of Fidelity (no cap fires; aggressive Pass 2 is stable in this combination).
- Use **Fast** or **Faster** as Pass 1 (no cap fires).

A deeper architectural fix is planned for v1.9.0+ pending investigation. The cap is a documented workaround, not a root-cause fix.

---

## Translation

### Which translation provider is best?

No single answer - each has trade-offs and quality varies by content. Field-reported patterns:

- **DeepSeek** (`deepseek-v4-flash` default in v1.8.14): low per-token cost; users frequently report it as a good balance for JA->EN/CN. The thinking variant `deepseek-v4-pro` is available via `--model deepseek-v4-pro`.
- **Claude** (default `claude-3-5-haiku-20241022`) and **OpenAI GPT** (default `gpt-4o-mini`): higher per-token cost; users often prefer them for nuanced or context-dependent passages.
- **Gemini** (`gemini-2.0-flash`): mid-tier cost; quality varies more by content style.
- **Ollama / Local LLM**: runs entirely on your machine - no data leaves your computer. Quality depends on the local model you run; use `qwen2.5:7b-instruct` or `gemma3:12b` for sensible results, avoid thinking models (their chain-of-thought can leak into the SRT output).
- **OpenRouter**: a router for multiple upstream models. Useful if you want to compare providers without maintaining separate API keys.

If you are not sure where to start, run a 30-line sample SRT through DeepSeek and Ollama side-by-side and pick the output that reads better to you.

### Translation fails with "API token limit" error

Your subtitle batches are too large for the model's context window. Try:

1. Reduce **Max Batch Size** in Advanced Settings (try 15 or 10)
2. Use a model with a larger context window
3. WhisperJAV auto-caps batch size for local LLMs

### Translation shows "Unknown provider: Gemini" on Linux

Install the missing dependency: `pip install google-api-core`. Fixed since v1.8.6.

---

## Installation

### The installer is stuck / taking very long

The first install downloads ~3-5 GB of packages. On slow connections this can take 30+ minutes. Check the install log (`install_log_v{VERSION}.txt` in the installation directory) for real-time progress.

**Phase 3 (PyTorch) appears silent for several minutes.** PyTorch is the largest package (~2 GB CUDA-enabled) and downloads as one big stream. The console only updates between download and post-install steps. As long as your network is active and disk usage is increasing under the install directory, the install is making progress. If it stays silent for >20 minutes with no disk activity, your network may be timing out - check `install_log_v{VERSION}.txt` for the underlying error.

### Installer behind a corporate proxy or firewall (HTTP_PROXY / HTTPS_PROXY)

The standalone installer respects standard proxy environment variables. Set them BEFORE launching the installer:

**Windows (current Command Prompt session):**
```cmd
set HTTPS_PROXY=http://user:pass@proxy.example.com:8080
set HTTP_PROXY=http://user:pass@proxy.example.com:8080
WhisperJAV-1.8.x-Windows-x86_64.exe
```

**Windows (persistent, all sessions):**
```cmd
setx HTTPS_PROXY http://user:pass@proxy.example.com:8080
setx HTTP_PROXY http://user:pass@proxy.example.com:8080
```
(Open a new Command Prompt for the variables to take effect, then run the installer.)

**macOS / Linux:**
```bash
export HTTPS_PROXY=http://user:pass@proxy.example.com:8080
export HTTP_PROXY=http://user:pass@proxy.example.com:8080
python install.py
```

`pip`, `uv`, and `git` all read these variables. WhisperJAV does NOT have a separate `--proxy` flag - the env-var approach is the standard pattern across all underlying tools.

### macOS source install fails with `SSL: CERTIFICATE_VERIFY_FAILED`

On macOS, the bundled Python may not trust the system certificate store by default. The pre-flight network check will fail with:

```
<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate>
```

**Fix**: run the official Python "Install Certificates" script that ships in your Python install:

```bash
/Applications/Python\ 3.12/Install\ Certificates.command
```

(Replace `3.12` with your installed Python minor version.) This installs the `certifi` CA bundle into the Python you are running. Re-run `python install.py` after.

**Note**: avoid alpha or pre-release Python builds (e.g. 3.12.0a3) for production installs. WhisperJAV supports stable Python 3.10-3.12.

### Qwen3-ASR fails with `TypeError: check_model_inputs() missing 1 required positional argument`

The bundled Qwen3-ASR fork is incompatible with `transformers` 5.x (the decorator API changed at `modeling_qwen3_asr.py:986`). WhisperJAV v1.8.14 pins `transformers>=4.40.0,<5.0` in `pyproject.toml` and the installer registry, so fresh installs are protected.

**If you hit this error**, your environment has `transformers >=5.0`. Either:

1. Downgrade transformers in your active env:
   ```bash
   <env-pip> install --no-deps "transformers>=4.40.0,<5.0"
   ```
   Replace `<env-pip>` with the exact pip path for your install (e.g. `Scripts\pip.exe` for the Windows standalone installer, or your venv pip for source installs).

2. Or wait for v1.9.0, which will ship a Qwen3-ASR fork patched for transformers 5.x. Tracking: <https://github.com/QwenLM/Qwen3-ASR/issues/138>.

### "CUDA not available" but I have an NVIDIA GPU

1. Verify your NVIDIA driver version: `nvidia-smi`
2. Driver 450+ required for CUDA 11.8, 570+ for CUDA 12.8
3. You do NOT need to install the CUDA Toolkit — PyTorch bundles its own
4. Try reinstalling PyTorch with the correct CUDA version

### How do I upgrade?

```bash
whisperjav-upgrade
```

Or for code-only (faster): `whisperjav-upgrade --wheel-only`

See the [Upgrade Guide](UPGRADE.md) for details.

---

## Models & Cache

### Where are the AI models stored?

WhisperJAV downloads models on first use and caches them on disk. The cache locations follow each library's standard convention — WhisperJAV does not override them:

| Cache | Default path (Linux/macOS) | Default path (Windows) | Used by |
|-------|---------------------------|------------------------|---------|
| HuggingFace | `~/.cache/huggingface/` | `%USERPROFILE%\.cache\huggingface\` | Kotoba, anime-whisper, Qwen3-ASR, WhisperSeg, transformers backend |
| OpenAI Whisper | `~/.cache/whisper/` | `%USERPROFILE%\.cache\whisper\` | Standard `large-v2` / `large-v3` / `turbo` models |
| Faster-Whisper | (uses HuggingFace cache above) | (uses HuggingFace cache above) | `fidelity` / `balanced` / `fast` modes via faster-whisper |
| PyTorch Hub | `~/.cache/torch/hub/` | `%USERPROFILE%\.cache\torch\hub\` | Silero VAD weights |
| ModelScope | `~/.cache/modelscope/` | `%USERPROFILE%\.cache\modelscope\` | ZipEnhancer / ClearVoice (when used) |
| WhisperJAV-internal | `~/.whisperjav_cache/` | `%LOCALAPPDATA%\.whisperjav_cache\` | Update checker cache, numba JIT cache redirect |

A typical WhisperJAV install ends up with **5-15 GB** in HuggingFace + Whisper caches after a few sessions, depending on which models you've used.

### How do I move the cache to a different drive?

Set the relevant environment variable **before** launching WhisperJAV. Each library reads its own variable:

| Cache to relocate | Env variable | Example value |
|-------------------|--------------|---------------|
| HuggingFace (most models) | `HF_HOME` | `D:\AI\hf_cache` |
| OpenAI Whisper | `XDG_CACHE_HOME` | `D:\AI\cache` (Whisper cache becomes `D:\AI\cache\whisper`) |
| PyTorch Hub | `TORCH_HOME` | `D:\AI\torch` |
| ModelScope | `MODELSCOPE_CACHE` | `D:\AI\modelscope` |

**Windows GUI users**: set these via System Properties → Environment Variables → User variables, then restart WhisperJAV. The GUI inherits its environment from your user profile.

**CLI users**: export them in your shell profile (`.bashrc` / `.zshrc`) or pass per-command.

After relocating, your old caches are still on disk in the original locations — copy or move them to the new path so models don't re-download.

### How do I free up disk space after uninstall?

The standalone installer removes the WhisperJAV install directory, but model caches in your home / AppData persist (libraries place them outside the install dir by design). To fully clean up:

1. Delete the cache directories listed in the table above
2. Delete the WhisperJAV-internal cache: `~/.whisperjav_cache` (Linux/macOS) or `%LOCALAPPDATA%\.whisperjav_cache` (Windows)
3. If you set custom `HF_HOME` etc. env vars, clean those paths instead

Total disk reclaimed varies — usually 5-15 GB.

### What if I have only 4 GB VRAM?

VRAM requirements for each transcription mode (matches the labels in the GUI Mode dropdown):

| Mode | Model | VRAM | Fits in 4 GB? |
|------|-------|------|---------------|
| **fast** | `turbo-int8` | ~4 GB | Tight but typically yes; recommended starting point |
| **faster** | `turbo-int8-batch8` | ~6 GB peak | No |
| **balanced** | `large-v2-int8` | ~6 GB | No |
| **fidelity** | `large-v2-fp16` | ~10 GB | No |
| **transformers** | HF `kotoba-bilingual` | ~6 GB | No |

On a 4 GB GPU, **Fast** is the only mode that reliably fits. There is no automatic CPU-offload fallback - if a mode requests more VRAM than is available, you will get a CUDA out-of-memory error rather than a graceful fallback.

Tips:- Avoid running other GPU-heavy applications simultaneously (Chrome, games, video encoding)
- Translation with local LLMs needs separate VRAM (3-8 GB depending on model). On a 4 GB GPU, prefer cloud translation providers (DeepSeek, OpenRouter) for the translation step.
- Speech enhancers (`clearvoice`, `bs-roformer`, `zipenhancer`) each need 1-3 GB extra. Use `none` or `ffmpeg-dsp` (CPU-only) if VRAM is tight.

---

## Cohere-Transcribe (preview — deferred to v1.9.0)

> **Status update (v1.8.14):** The Cohere-Transcribe preview is **deferred to v1.9.0** and the dropdown entry is greyed out. The model requires `transformers ≥ 5.4.0`, but the Qwen3-ASR fork that ships with WhisperJAV breaks on transformers 5.x (`@check_model_inputs()` decorator API change). v1.9.0 will land a coordinated transformers upgrade together with the Qwen3-ASR fork patch and the Cohere ship.

> **Migration note for users who already upgraded transformers:** If you ran a recent `pip install transformers --upgrade` (or otherwise pulled transformers 5.x into your WhisperJAV env), Qwen3-ASR will fail to load. Restore stability with `pip install "transformers>=4.40.0,<5.0"` and then verify Qwen3-ASR loads via `python -c "from qwen_asr import Qwen3ASRModel; print('OK')"`.

The setup notes below remain valid for v1.9.0 resumption — preserved for forward reference.

### Using Cohere-Transcribe (preview, opt-in) — v1.9.0

Cohere Transcribe-03-2026 will be available in v1.9.0 as a third generator under the **ChronosJAV** dropdown, alongside Qwen3-ASR and Anime-Whisper. The model is gated on HuggingFace, so a one-time setup is required. Anime-Whisper remains the default JA-tuned generator; Cohere is opt-in.

**Setup (one-time)**

1. Visit <https://huggingface.co/CohereLabs/cohere-transcribe-03-2026> and click *Agree and access repository*.
2. Create a token at <https://huggingface.co/settings/tokens> (a *Read* token is sufficient).
3. Set `HF_TOKEN` in your environment:
   - **Windows (persistent):** `setx HF_TOKEN hf_xxxxxxxxxxxx` — restart the GUI/terminal after this for the change to take effect.
   - **Windows (current PowerShell only):** `$env:HF_TOKEN = "hf_xxxxxxxxxxxx"`
   - **Linux/macOS:** `export HF_TOKEN=hf_xxxxxxxxxxxx`
4. Pick **ChronosJAV → Cohere-Transcribe (preview)** in the GUI Ensemble dropdown, or run from CLI:
   ```bash
   whisperjav --mode qwen --pass1-qwen-params '{"generator_backend":"cohere"}' video.mp4
   ```

**First-run download**

Cohere weights total ~3.85 GB (`model.safetensors` is ~4.13 GB raw FP16, plus tokenizer and custom code files). The first transcription will spend 10–30 minutes downloading before transcription begins (one-time per machine; cached in your HuggingFace cache afterward). HuggingFace uses Xet (content-addressed) delivery, which streams chunks into a temp area — plan for **at least 5 GB free** on the cache volume to be safe.

**VRAM requirement**

~4–8 GB at FP16. The Qwen3-ForcedAligner used for word-level timestamps is loaded sequentially after Cohere unloads, so peak VRAM is bounded by Cohere alone — Cohere fits comfortably on 8–10 GB cards.

### Why doesn't Cohere produce native word timestamps?

Cohere Transcribe-03-2026 currently emits text only; per-word timing is on the model authors' [roadmap](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/19) but not yet implemented. WhisperJAV pairs Cohere with the Qwen3-ForcedAligner-0.6B (the same aligner used by the Qwen3-ASR pipeline) to derive word-level timestamps from Cohere's output. You can disable the aligner via Customize Parameters → *Aligner Backend → None* to fall back to VAD-derived segment timing only — useful if you only need segment-level subs and want to save the aligner load step.

### Cohere returned an empty transcript / "gated" error

The most common cause is `HF_TOKEN` not being set in the environment that launches WhisperJAV. Re-check the setup steps above and restart the GUI after setting the token (especially on Windows with `setx`, which only affects new shells). If the error persists, confirm at <https://huggingface.co/CohereLabs/cohere-transcribe-03-2026> that you have an *Authorized* status — the model owners gate access manually.

### Common errors during Cohere first-run

WhisperJAV walks the exception chain and tailors the error message — read the `[ERROR]` block in the console for an actionable diagnostic. The most frequent causes:

**Disk ran out of space (`os error 112`, `no space left`, `CAS service error`)**

The Cohere download exhausts the cache volume mid-stream. WhisperJAV pre-flights free space (~5 GB minimum) before download starts in v1.8.14, but if your cache directory is on a near-full system drive you can either free space or redirect the cache to another drive:

- **Windows (persistent)**: `setx HUGGINGFACE_HUB_CACHE D:\hf_cache` — restart the GUI/terminal after this
- **Windows (current session only)**: `$env:HUGGINGFACE_HUB_CACHE = "D:\hf_cache"`
- **macOS/Linux**: `export HUGGINGFACE_HUB_CACHE=/path/with/space`

After redirecting, retry — the download starts fresh in the new cache location.

**Interrupted previous download (`Can't load the model... pytorch_model.bin`)**

A failed prior download left a partial directory. Locate `models--CohereLabs--cohere-transcribe-03-2026/` under your HF cache (default `%USERPROFILE%\.cache\huggingface\hub\` on Windows) and delete it, then retry.

**Network / proxy issues**

If you see connection / timeout / SSL / proxy errors, check your network. Corporate networks may need `HTTPS_PROXY` / `HF_ENDPOINT`. China users: HuggingFace is often throttled — see the dedicated note below.

**Auth failed (401)**

Your `HF_TOKEN` is expired, revoked, or malformed. Recreate a *Read* token at <https://huggingface.co/settings/tokens> and re-set `HF_TOKEN`.

**Wrong loader class on very old transformers**

WhisperJAV uses `AutoModelForSpeechSeq2Seq` (the generate-capable wrapper) per Cohere's `auto_map` metadata. If you're on a transformers version that lacks this class, WhisperJAV falls back to `AutoModel` and logs a warning — `generate()` may then fail at inference time. Upgrade `transformers` (the WhisperJAV environment ships 4.57.6 which has it).

### License and trust_remote_code

- The Cohere model code is **Apache-2.0** licensed.
- The model **weights** are governed by Cohere's gated-access terms, which you accept on HuggingFace. WhisperJAV does not redistribute Cohere weights; they are downloaded from HuggingFace at first use.
- Loading uses `trust_remote_code=True` because `transformers` 4.57.6 (WhisperJAV's pinned version) does not yet expose the Cohere model class natively — the class ships in the model repo itself. This is the same precedent used by ZipEnhancer/ModelScope elsewhere in WhisperJAV.

---

## Troubleshooting

### Processing fails with an error

1. Check the **Console** at the bottom of the GUI for the error message
2. Enable **Debug logging** in Advanced tab and retry — check `whisperjav.log`
3. Try a simpler pipeline (**Faster** mode) to isolate the issue
4. Report the error on [GitHub Issues](https://github.com/meizhong986/whisperjav/issues) with the error message and your system details

### The GUI won't start

- **Windows:** Ensure WebView2 is installed (comes with Windows 10 1803+ and Windows 11)
- **Linux:** Install `libwebkit2gtk-4.0-dev` on Ubuntu (or `libwebkit2gtk-4.1-dev` on Ubuntu 24.04+) or `webkit2gtk4.0-devel` on Fedora
- **macOS:** WebKit is built-in, should work automatically
- Try launching from command line (`whisperjav-gui`) to see error output
