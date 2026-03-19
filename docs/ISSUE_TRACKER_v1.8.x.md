# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-19 (rev15 — 3 fixes coded for post2: #241/#240 + XXL stderr encoding) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **52 open** on GitHub

---

## Status Legend

| Label | Meaning |
|-------|---------|
| `SHIPPED` | Fix released in a stable version. Waiting for user to test/confirm. |
| `FIX VERIFIED` | User confirmed the fix resolves their issue. Safe to close the GitHub issue. |
| `AWAITING CONFIRMATION` | Response given. Waiting for user reply. |
| `NEEDS RESPONSE` | Issue has no response or needs a follow-up reply. |
| `DEFERRED` | Moved to a future release. |

---

## Quick Stats

| Category | Count | Notes |
|----------|------:|-------|
| Total open on GitHub | **52** | +5 since rev12 (#239-#243). |
| New issues since rev12 | 5 | #239 (AMD GPU), #240 (GUI access violation), #241 (CPU float16 crash), #242 (XXL Pass 1 request), #243 (install verification fail) |
| Closed since rev12 | 0 | |
| **FIX CODED (for post2)** | 3 | **#241** (CPU float16 regression), **#240** (private_mode simplification), XXL stderr encoding crash |
| **NEEDS RESPONSE (no reply yet)** | 6 | #231, #233, #234, #240, #241, #243 |
| **NEEDS FOLLOW-UP RESPONSE** | 2 | #225 (new data), #237 (new question from liugngg) |
| **AWAITING CONFIRMATION** | 6 | #200, #204, #207, #209, #210, #212 |
| **SHIPPED (awaiting user test)** | 6 | #132, #218, #223, #235, #236, #238 |
| **FIX VERIFIED (safe to close)** | 2 | #228, #229 |
| Feature requests (open) | 23 | See Cluster J |
| Deferred to v1.9+ | 11 | See v1.9+ Backlog |

---

## v1.8.9.post1 Hotfix — RELEASED (2026-03-19)

Released as follow-up to v1.8.9. All 4 items from rev12 shipped and responded to on GitHub.

| Item | Issues | Fix | Status |
|------|--------|-----|--------|
| Ollama doubled endpoint | #132 | Removed redundant `/v1/chat/completions` append in `translate/service.py` + `translate/cli.py` | `SHIPPED` — meizhong986 responded on #132 |
| WebUI cache on upgrade | #236 | GUI clears WebView2 cache on version change, preserves settings | `SHIPPED` — meizhong986 responded on #236 |
| ctypes hwnd overflow | #235 | Added proper `argtypes`/`restype` for Win32 functions | `SHIPPED` — meizhong986 responded on #235 |
| Portuguese translation target | #238 | Added to SUPPORTED_TARGETS, argparse, both GUI dropdowns | `SHIPPED` — meizhong986 responded on #238 |

### Post-post1 issues (2026-03-19)

| # | Title | Severity | Root Cause | Status |
|---|-------|----------|------------|--------|
| **#243** | Install verification fails (PyTorch) | **MEDIUM — INSTALLER** | RTX 3050 6GB Laptop, driver 595.79. uv installs PyTorch successfully (`SUCCESS`) but verification immediately fails with `No module named 'torch'`. Path mismatch between uv install target and verification Python. Not a v1.8.9 regression. | `NEEDS RESPONSE` — meizhong986 asked for laptop confirmation |
| **#242** | Add XXL Faster Whisper to Pass 1 | **LOW — FEATURE** | yangming2027 requests XXL as Pass 1 option (currently Pass 2 only). Reason: some users' GPUs can't run fidelity mode. | `NEEDS RESPONSE` |
| **#241** | float16 on CPU crash | **CRITICAL — REGRESSION** | v1.8.9 changed default compute type from `int8` to `float16`. CTranslate2 does NOT support `float16` on CPU. ALL CPU-only users crash. | **FIX CODED** for post2: `resolver_v3.py` returns `auto` for non-CUDA; `faster_whisper_pro_asr.py` safety net for MPS→CPU downgrade. |
| **#240** | GUI access violation on Windows 11 | **HIGH — BUG** | WebView2 localhost timeout + access violation in `clr_loader`/`pythonnet`. Likely caused by `private_mode=False` + `storage_path` creating persistent WebView2 data folder. | **FIX CODED** for post2: switch to `private_mode=True`, remove `storage_path` and cache-clearing function. |
| **#239** | AMD GPU support request | **LOW — FEATURE** | User has AMD iGPU, uses Qwen3-ASR-GGUF via llama.cpp+vulkan. Duplicate of #142/#114. | `NEEDS RESPONSE` |
| **—** | XXL stderr UnicodeDecodeError | **MEDIUM — BUG** | `subprocess.run()` in `xxl_runner.py` reads stderr with `encoding="utf-8"` but XXL exe emits Windows system codepage bytes (cp936/GBK) for Chinese filenames. Transcription completes and SRT is written, but Python crashes reading stderr. Fix: add `errors="replace"` to subprocess call. | **FIX CODED** for post2 |

**#241 ROOT CAUSE ANALYSIS (CRITICAL)**: This is a direct regression from the v1.8.9 quality improvement that changed `compute_type` from `int8` to `float16`. The change was intended to improve transcription quality on NVIDIA GPUs, but `float16` is not supported on CPU by CTranslate2. The fix must either:
- (a) Auto-detect device and fall back to `int8` on CPU, OR
- (b) Use `auto` compute type (CTranslate2 picks best for device), OR
- (c) Add explicit CPU guard in `faster_whisper_pro_asr.py` before model load

This breaks: standalone installer users on laptops without NVIDIA GPUs, `--accept-cpu-mode` users, any system where GPU detection fails. **Needs v1.8.9.post2 immediately.**

**#240 analysis**: Complex WebView2 failure. The access violation in `clr_loader`/`pythonnet` is unusual — WhisperJAV doesn't use pythonnet directly, so this might be a system-level conflict. Possible causes: (1) Antivirus intercepting localhost connections, (2) another process binding port 127.0.0.1, (3) corrupted WebView2 user data, (4) pythonnet conflict from another installed app. Need to ask: is there another Python application installed? Is there antivirus blocking localhost? What port does pywebview try to use?

---

## v1.8.9 Stable — RELEASED (2026-03-18)

Released: https://github.com/meizhong986/WhisperJAV/releases/tag/v1.8.9
Merge commit: `eea08a0` (19 commits from `dev_v1.8.9.beta` into main)

### What shipped

| Category | Item | Issues |
|----------|------|--------|
| **Feature** | BYOP Faster Whisper XXL — GUI + CLI + worker integration | #223, #224 |
| **Feature** | OllamaManager — smart Ollama lifecycle (CLI-only, no GUI wiring) | #132, #212, #214, #128 |
| **Quality** | Model large-v2 → large-v3, compute int8 → float16 | #223, #224 |
| **Quality** | All sensitivity presets retuned (ASR + VAD) | #223 |
| **Quality** | CPS threshold 20 → 30, VAD failover | #223 |
| **Quality** | Ensemble degraded status (honest reporting, no false SUCCESS) | — |
| **Cleanup** | Config reduced from ~600 to 19 lines | — |
| **Cleanup** | Dead CLI flags removed | — |
| **Bug fix** | 14+ bugs fixed (see release notes) | — |

### What did NOT ship (deferred)

| Item | Issues | Why |
|------|--------|-----|
| MPS selective policy | #198, #227 | Not started — needs more M3/M4 data |
| OllamaManager GUI wiring | #132, #212 | CLI-only for now |
| Gemma 3 configs | #128 | Awaiting hyiip's PR |

### Known regressions

| # | Regression | Severity | Introduced By | Fix |
|---|-----------|----------|---------------|-----|
| **#241** | float16 compute on CPU crashes | **CRITICAL** | `compute int8 → float16` quality change | Auto-detect device, fall back to int8/auto on CPU |

---

## v1.8.8 Stable — RELEASED (2026-03-13)

All fixes shipped. Commit `b0f9d9b release: v1.8.8 stable`.

| ID | Issue(s) | Description | Status | Notes |
|----|----------|-------------|--------|-------|
| A1 | #212/#214/#132 | Diagnostic token logging (batch stats, "No matches" handler) | `SHIPPED` | #132: `--provider local` confirmed working on Kaggle |
| A2 | #212 | .subtrans stale settings override (version-stamp, auto-delete) | `SHIPPED` | Needs validation from destinyawaits/zhstark |
| A3 | **#218** | cu118 wheel version mismatch (`UV_SKIP_WHEEL_FILENAME_CHECK=1`) | `SHIPPED` | Needs WillChengCN to confirm |
| A4 | #212/#214 | Reduce max_tokens 2x→1x multiplier (2392→1820 for 8K ctx) | `SHIPPED` | Needs real-world validation |
| A5 | #214 | pornify.txt → sectioned format with format example | `SHIPPED` | Internal improvement |
| A6 | — | `--provider ollama` preview (auto-detect, num_ctx=8192, streaming) | `SHIPPED` | #132: confirmed working on Kaggle (after manual ollama install) |
| B1 | **#200** | NVML Optimus laptop fallback (`--force-cuda` flag, guidance text) | `SHIPPED` | Needs Ywocp to confirm |
| C1 | **#209** | Repetition cleaner pattern #8 (`sentence_phrase_repetition`) | `SHIPPED` | Needs weifu8435 to confirm |
| C2 | **#219** | MossFormer2_SS_16K 3D tensor crash | `SHIPPED` | Issue closed 2026-03-13 |
| — | #198 | Force greedy decoding (`num_beams=1`) on MPS | `SHIPPED` | Issue closed. MPS confirmed not accelerating Whisper. |
| — | — | numpy 2.x migration | `SHIPPED` | |
| — | #211 | urllib3/chardet warning suppressed | `SHIPPED` | |
| — | — | Post-install verification timeout 30s→120s | `SHIPPED` | `63936e1` |
| — | — | RequestsDependencyWarning suppression | `SHIPPED` | `168e8b3` |

---

## Cluster Analysis

### Cluster A: Local LLM Translation (4 issues + 1 contributor proposal)

**Issues**: #196 (closed), #212, #214 (closed), #132, #128 (contributor PR), #233
**Status**: v1.8.9.post1 fixed Ollama 404 (#132). #233 is pre-existing llama-cpp-python failure.

| # | Title | Reporter | Platform | State | Status |
|---|-------|----------|----------|-------|--------|
| **#233** | translation error | WillChengCN | Windows, cu118 | **OPEN** | `--provider local`, llama-8b. `n_vocab()` AssertionError. Same as #208. Recommend `--provider ollama`. | `NEEDS RESPONSE` |
| **#214** | 1.8.7 localLLM fail | KenZP12 | Windows, cu128 | **CLOSED** (2026-03-18) | |
| **#212** | Regex Error - Local Translation v1.8.7 | destinyawaits | Linux, 5090 | **OPEN** | v1.8.9 comment posted. | `AWAITING CONFIRMATION` |
| **#196** | Local Translation Errors | zhstark | Ubuntu, 5090 32GB | CLOSED | Related to #212. |
| **#132** | Local LLM on Kaggle | TinyRick1489 | Kaggle | **OPEN** | v1.8.9.post1 fixed doubled endpoint 404 bug. meizhong986 responded 2026-03-19. | `SHIPPED` |
| **#128** | LLM context/batch sizing | hyiip | (contributor) | **OPEN** | Gemma 3 proposal. Redirected to OllamaManager. | `AWAITING CONFIRMATION` |

---

### Cluster B: MPS / Apple Silicon (2 issues)

**Issues**: #198 (closed), #227
**Status**: #198 closed. #227 documented as known issue. MPS selective policy deferred.

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#198** | Transformers MPS crash | francetoastVN | **CLOSED** | MPS fix shipped v1.8.8. MPS confirmed 6x SLOWER for Whisper. |
| **#227** | M1 MAX Transformers hang | dadlaugh | **OPEN** | MPS model-dependent behavior confirmed. | `AWAITING CONFIRMATION` |

---

### Cluster C: Network / Installation (12 issues — 2 new)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#243** | Install verification fails | Trenchcrack | **OPEN (NEW)** | RTX 3050 6GB Laptop. uv installs PyTorch OK but `import torch` fails. Path mismatch? Install dir: `D:\JAV`. | `NEEDS RESPONSE` |
| **#240** | GUI access violation Win11 | m739566004-svg | **OPEN (NEW)** | WebView2 timeout + access violation. Fix coded: switch to `private_mode=True`. | `FIX CODED` for post2 |
| **#234** | CUDA version claim | techguru0 | **OPEN** | CUDA toolkit vs compute capability confusion. | `NEEDS RESPONSE` |
| **#229** | INSTALLATION FAILED (SSL) | WillChengCN | **OPEN** | **SELF-RESOLVED**: "the cause is python environment." | `FIX VERIFIED` — safe to close |
| **#228** | cublas64_12.dll / first run hang | yhxkry | **OPEN** | **SELF-RESOLVED**: installed CUDA DLLs manually. | `FIX VERIFIED` — safe to close |
| **#225** | 白屏 (GUI white screen) | github3C | **OPEN** | **UPDATE 2026-03-19**: Confirmed WebView2 Runtime is latest version (screenshot). Edge latest. Still white screen. Previous analysis (install log clean, RTX 3070, 19/19 checks pass) exhausted. | `NEEDS FOLLOW-UP` — WebView2 confirmed OK, need new hypothesis |
| **#222** | 字幕是日语... (how to get Chinese?) | libinghui20001231-debug | **OPEN** | User confusion. Pointed to docs. | `AWAITING CONFIRMATION` |
| **#221** | 安装完后报错 (cublas64_12.dll missing) | libinghui20001231-debug | **OPEN** | GTX 1650, old driver 462. | `AWAITING LOG` |
| **#220** | 安装卡着不动 (install stalls) | libinghui20001231-debug | **OPEN** | Install stalls during PyTorch download. | `AWAITING LOG` |
| **#218** | 安装错误 (uv cu118 wheel) | WillChengCN | **OPEN** | uv rejects cu118 llama-cpp wheel. | `SHIPPED` (A3). `AWAITING CONFIRMATION` |
| **#217** | 找不到WhisperJAV-GUI.exe | loveGEM + vimbackground | **OPEN** | PyTorch download fails (network). | `AWAITING CONFIRMATION` |
| **#210** | 安装失败 DNS error | iop335577 | **OPEN** | DNS through proxy. | `AWAITING CONFIRMATION` |
| **#204** | VPN/v2rayN SSL failures | yangming2027 | **OPEN** | HF hub SSL errors. | `AWAITING CONFIRMATION` |

**#225 update (2026-03-19)**: github3C confirmed WebView2 Runtime is latest version. Edge is latest. The install log was clean. This rules out the "WebView2 missing/outdated" hypothesis. New possibilities: (1) GPU driver compatibility with WebView2 rendering (RTX 3070 — try disabling hardware acceleration), (2) antivirus/firewall blocking localhost, (3) corrupted WebView2 user data directory, (4) DPI scaling issue. Need to ask user to try `--debug` flag or check if CLI transcription works.

**#240 analysis**: The `pythonnet`/`clr_loader` in the stack trace is suspicious — WhisperJAV doesn't depend on pythonnet. This could be a system-level Python installation conflict. The `access violation` suggests memory corruption. The non-standard install path (`D:\WJ\`) might be relevant if there are path length or permission issues. Need to ask: (1) is there another Python distribution installed? (2) antivirus? (3) does CLI (`whisperjav video.mp4`) work?

---

### Cluster D: GPU Detection (3 issues — 1 new)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#239** | AMD GPU support | bmin1117 | **OPEN (NEW)** | Uses Qwen3-ASR-GGUF via llama.cpp+vulkan on AMD iGPU. Asks about WhisperJAV AMD support. Duplicate of #142/#114. | `NEEDS RESPONSE` |
| **#200** | NVML "Driver Not Loaded" on Optimus laptop | Ywocp | **OPEN** | Dual GPU. | `SHIPPED` (B1: `--force-cuda`). `AWAITING CONFIRMATION` |
| **#213** | Intel GPU support | DDXDB | **OPEN** | Requests `torch.xpu`. | `DEFERRED` to v1.9+ |

---

### Cluster E: GUI / WebUI (5 issues — 1 new)

**Issues**: #96, #207, #235, #236, #240 (NEW)
**Status**: #235 and #236 fixed in v1.8.9.post1. #240 is new complex WebView2 failure.

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#240** | GUI access violation Win11 | m739566004-svg | **OPEN (NEW)** | WebView2 timeout + pythonnet crash. Complex. | `NEEDS RESPONSE` |
| **#236** | WebUI cache prevents v1.8.9 changes | FishYu-OWO | **OPEN** | Fixed in v1.8.9.post1. meizhong986 responded. | `SHIPPED` |
| **#235** | ctypes icon OverflowError on startup | techguru0 | **OPEN** | Fixed in v1.8.9.post1. meizhong986 responded. | `SHIPPED` |
| **#207** | 1.86不能保存设置 | q864310563 | **OPEN** | Dup of #96. | `AWAITING CONFIRMATION` |
| **#96** | Full settings persistence | sky9639 | OPEN | | `DEFERRED` to v1.9 |

---

### Cluster F: Whisper Output Quality (9 issues — 2 new)

**Issues**: #242 (NEW), #241 (NEW REGRESSION), #223, #224, #230, #237, #209, #215, #227
**Severity**: **CRITICAL** — #241 is a regression that crashes ALL CPU-only users. Fix coded.

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#242** | Add XXL to Pass 1 | yangming2027 | **OPEN (NEW)** | Feature request: XXL currently Pass 2 only. Some users' GPUs can't run fidelity mode. | `NEEDS RESPONSE` |
| **#241** | float16 on CPU crash | helphelp7092 | **OPEN (NEW)** | **REGRESSION.** v1.8.9 compute type change (int8→float16) crashes on CPU. | **FIX CODED** for post2. `resolver_v3.py`: non-CUDA → `auto`. Safety net in `faster_whisper_pro_asr.py`. |
| **#237** | XXL model/compute questions | yangming2027 | **OPEN** | **UPDATE 2026-03-19**: meizhong986 responded to all 3 questions (8 comments). New comment from **liugngg** asking whether WhisperJAV post-processes XXL ASR results (optimization aspects). | `RESPONDED` — new question from liugngg |
| **#230** | Subtitle merging module request | weifu8435 | **OPEN** | Feature request. On roadmap for v1.9.0. | `RESPONDED` |
| **#224** | 人声分离分析 (vocal separation) | yangming2027 | **OPEN** | v1.8.9 comment posted. | `AWAITING CONFIRMATION` |
| **#223** | Faster Whisper XXL comparison | weifu8435 | **OPEN** | BYOP XXL + quality tuning shipped in v1.8.9. | `SHIPPED` v1.8.9 |
| **#209** | Single subtitle very long (repetition) | weifu8435 | **OPEN** | Shipped in v1.8.8 (C1) + v1.8.9 quality tuning. | `AWAITING CONFIRMATION` |
| **#215** | Qwen3-ASR subtitle quality | yangming2027 | **OPEN** | Expected behavior. | `AWAITING CONFIRMATION` |
| **#227** | M1 MAX Transformers hang | dadlaugh | **OPEN** | MPS known issue documented. | `AWAITING CONFIRMATION` |

**#241 impact assessment**: The error log shows the user tried multiple approaches (balanced single-pass, ensemble) — all fail with the same `float16` error. The user is on CPU (`--accept-cpu-mode`). The v1.8.9 quality change set `compute_type` to `float16` universally, but CTranslate2 (Faster-Whisper backend) only supports `float16` on CUDA GPUs. **This affects ALL users who:**
- Have no GPU (CPU-only installs)
- Use `--accept-cpu-mode` / `--force-cpu`
- Have GPU detection that falls back to CPU (Optimus, driver issues)
- Use MPS (Apple Silicon) — MPS may also not support float16 via CTranslate2

**#237 liugngg question**: Asks whether WhisperJAV does any post-processing on XXL ASR output. Answer: Yes — Japanese regrouping (stable-ts), hallucination removal, repetition cleaning, timing adjustments, CPS filtering. All post-processing runs on Pass 2 output same as any other pipeline.

---

### Cluster G: Speech Enhancement — RESOLVED

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#219** | MossFormer2_SS_16K 3D tensor crash | anon12642 | **CLOSED** (2026-03-13) | `SHIPPED` in v1.8.8. |

---

### Cluster H: Kaggle / Colab (2 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#231** | Kaggle notebook run error | fzfile | **OPEN** | **Root cause confirmed 2026-03-18**: `llvmlite 0.43.0` too old (Kaggle pre-installed), numba requires >= 0.46.0. Import chain: `stable_whisper → whisper → numba → llvmlite`. Fix: `!pip install -U llvmlite numba`. Not a WhisperJAV bug. | `NEEDS RESPONSE` with fix |
| **#132** | Local LLM on Kaggle | TinyRick1489 | **OPEN** | v1.8.9.post1 fixed doubled endpoint 404. meizhong986 responded 2026-03-19. | `SHIPPED` |

---

### Cluster I: Model Support Requests (1)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#232** | whisper-ja-anime-v0.1 model support | mustssr | **OPEN** | Requests `efwkjn/whisper-ja-anime-v0.1` HuggingFace model. | `AWAITING INFO` — asked for comparison results vs large-v3 |

---

### Cluster J: Feature Requests

| # | Title | Reporter | State | Summary | Target |
|---|-------|----------|-------|---------|--------|
| **#242** | XXL in Pass 1 | yangming2027 | OPEN (NEW) | XXL as Pass 1 option for lower-end GPUs | **Investigate** |
| **#239** | AMD GPU support | bmin1117 | OPEN (NEW) | AMD iGPU via vulkan. Dup of #142/#114. | v1.9+ |
| **#238** | Portuguese/Brazilian translation | SangenBR | OPEN | Translation target language | **SHIPPED v1.8.9.post1** |
| **#232** | whisper-ja-anime-v0.1 model | mustssr | OPEN | HuggingFace anime ASR model | **Investigate** |
| **#230** | Standalone merge module | weifu8435 | OPEN | CLI tool for multi-SRT merging | **v1.9.0** |
| **#224** | Vocal separation (UVR MDX-Net) | yangming2027 | OPEN | Analysis of XXL's vocal separation advantage | **Investigate** |
| **#223** | Faster Whisper XXL comparison | weifu8435 | OPEN | Quality gap with XXL | **SHIPPED v1.8.9** |
| **#213** | Intel GPU (XPU) support | DDXDB | OPEN | torch.xpu via PyTorch XPU wheels | v1.9+ |
| **#206** | Grey out incompatible options | techguru0 | OPEN | Block incompatible GUI choices | v1.9+ |
| **#205** | VibeVoice ASR | kylesskim-sys | OPEN | Microsoft VibeVoice. VRAM too high. | v1.9+ |
| **#181** | Frameless window | QQ804218 | OPEN | Cosmetic | v1.9+ |
| **#180** | Multi-language GUI | QQ804218 | OPEN | Full i18n | v1.9+ |
| **#175** | Chinese GUI | yangming2027 | OPEN | Subset of #180 | v1.9+ |
| **#164** | MPEG-TS + Drive | hosmallming | OPEN | Format + cloud | Backlog |
| **#142** | AMD Radeon | MatthaisUK | OPEN | FishYu-OWO AMD workaround (CTranslate2 ROCm wheels) | v1.9+ |
| **#128** | Gemma 3 model upgrade | hyiip | OPEN | 128K context Gemma 3 4B/12B models. **Contributor offers.** | **v1.9.0** |
| **#126** | Recursive directory | jl6564 | OPEN | Walk subdirs, mirror output | v1.9+ |
| **#114** | DirectML | SingingDalong | OPEN | AMD/Intel GPU via torch-directml | v1.9+ |
| **#99** | 4GB VRAM guidance | hosmallming | OPEN | Log VRAM, recommend settings | Backlog |
| **#71** | Google Translate (free) | x8086 | OPEN | Fragile API | v1.9+ |
| **#59** | Feature plans | meizhong986 | OPEN | Meta roadmap | Keep open |
| **#51** | Batch translate wildcard | lingyunlxh | OPEN | Glob in translate CLI | Backlog |
| **#49** | Output to source folder | meizhong986 | OPEN | Docs gap | Backlog |
| **#44** | GUI drag-drop | lingyunlxh | OPEN | Filename vs path | Backlog |
| **#43** | DeepL provider | teijiIshida | OPEN | Non-LLM adapter | v1.9+ |
| **#33** | Linux pyaudio docs | org0ne | OPEN | Documentation | Backlog |

---

## Tally

### By Resolution Status

| Status | Count | Issues |
|--------|------:|--------|
| **FIX CODED (for post2)** | 3 | **#241** (CPU float16 regression), **#240** (private_mode), XXL stderr encoding crash |
| **NEEDS RESPONSE (new/unresponded)** | 6 | #231, #233, #234, #239, #242, #243 |
| **NEEDS FOLLOW-UP** | 2 | #225 (new data, old hypothesis exhausted), #237 (liugngg question) |
| **AWAITING LOG** | 2 | #220, #221 |
| **AWAITING CONFIRMATION** | 6 | #200, #204, #207, #209, #210, #212 |
| **SHIPPED (awaiting user test)** | 6 | #132, #218, #223, #235, #236, #238 |
| **FIX VERIFIED (safe to close)** | 2 | #228, #229 |
| **Feature requests (open)** | 23 | See Cluster J |
| **DEFERRED to v1.9+** | 11 | #96, #205, #206, #213, #180, #175, #181, #142, #114, #126, #43 |

### By Priority (Active Work)

| Priority | # | Issue | Status | Why |
|----------|---|-------|--------|-----|
| **CRITICAL** | #241 | CPU float16 crash (REGRESSION) | **FIX CODED** for post2 | ALL CPU-only users crash. v1.8.9 regression. |
| **HIGH** | #240 | GUI access violation Win11 | **FIX CODED** for post2 | `private_mode=True` removes storage folder failure mode. |
| **MEDIUM** | — | XXL stderr UnicodeDecodeError | **FIX CODED** for post2 | Chinese filenames cause cp936 bytes on stderr, crashing subprocess.run(). Transcription succeeds but result is lost. |
| **HIGH** | #225 | GUI white screen | `NEEDS FOLLOW-UP` | WebView2 confirmed OK. Need new hypothesis. |
| **MEDIUM** | #243 | Install verification fails | `NEEDS RESPONSE` | RTX 3050 Laptop. uv installs OK but import fails. |
| **MEDIUM** | #231 | Kaggle llvmlite version | `NEEDS RESPONSE` | Kaggle env issue, fix known. |
| **MEDIUM** | #237 | liugngg question on XXL post-processing | `NEEDS FOLLOW-UP` | New question from third user. |
| **MEDIUM** | #233 | Local LLM AssertionError | `NEEDS RESPONSE` | Pre-existing, recommend Ollama. |
| **LOW** | #242 | XXL in Pass 1 | `NEEDS RESPONSE` | Feature request. |
| **LOW** | #234 | CUDA version misunderstanding | `NEEDS RESPONSE` | Polite correction needed. |
| **LOW** | #239 | AMD GPU request | `NEEDS RESPONSE` | Dup of #142/#114, link to existing. |

---

## Pending GitHub Actions

### Issues Needing Response (9)

| # | Action Needed | Priority |
|---|--------------|----------|
| **#241** | **CRITICAL.** Acknowledge regression. Explain: v1.8.9 compute type change breaks CPU. Fix in post2 hotfix. | CRITICAL |
| **#243** | Investigate: PyTorch installs OK via uv but `import torch` fails. Likely path mismatch in conda-constructor env. Ask for full `INSTALLATION_FAILED_v1.8.9.txt`. | MEDIUM |
| **#242** | Respond: XXL in Pass 1 is technically feasible. Evaluate for v1.9.0. XXL doesn't produce timestamps — needs ChronosJAV-style forced alignment in Pass 1. Not trivial. | LOW |
| **#237** | Answer liugngg's new question: Yes, WhisperJAV post-processes XXL output (regrouping, hallucination removal, repetition cleaning, timing, CPS filtering). | MEDIUM |
| **#231** | Respond: Root cause is Kaggle's llvmlite 0.43.0. Fix: `!pip install -U llvmlite numba`. Not a WhisperJAV bug. | MEDIUM |
| **#233** | Respond: Known llama-cpp-python failure. Recommend `--provider ollama` (v1.8.9+). | MEDIUM |
| **#234** | Politely correct: CUDA toolkit version depends on driver, not GPU series. | LOW |
| **#239** | Respond: AMD support tracked in #142 and #114. FishYu-OWO proved ROCm works with CTranslate2. Planned for v1.9+. | LOW |
| **#240** | Fix coded (private_mode=True). After post2 ships, respond: simplified WebView2 startup, should resolve access violation. Ask user to upgrade and test. | POST-RELEASE |

### Issues Needing Follow-Up (1)

| # | Action Needed | Priority |
|---|--------------|----------|
| **#225** | WebView2 is latest, Edge is latest, install log clean. Need new diagnostic approach. Ask: (1) does CLI work? (2) try `whisperjav-gui --debug`? (3) try disabling GPU hardware acceleration in Edge settings? (4) any other Python on system? | HIGH |

### Candidates for Closing

| # | Condition | Notes |
|---|-----------|-------|
| **#229** | Self-resolved: "the cause is python environment." | `FIX VERIFIED` |
| **#228** | Self-resolved: installed CUDA runtime DLLs manually | `FIX VERIFIED` |
| **#222** | Responded with docs + translation instructions | Last activity 2026-03-14, 5 days stale |
| **#211** | urllib3 warning fixed in v1.8.8 | Already closed |

### Decisions Needed

| # | Decision | Recommendation |
|---|----------|----------------|
| **#241/#240** | Release v1.8.9.post2 hotfix? | **YES.** Both fixes coded and tested. #241 breaks all CPU users. #240 simplifies GUI startup. |
| **#243** | Installer verification bug — investigate? | Yes, but not hotfix scope. Investigate path mismatch in conda-constructor post-install. |
| **#242** | XXL in Pass 1? | Evaluate for v1.9.0. Non-trivial: XXL doesn't produce timestamps. |
| **#227** | MPS strategy — selective by model? | Defer to v1.9.0 (needs more data) |
| **#128** | Accept hyiip's Gemma 3 contribution? | Coordinate for v1.9.0 |

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Status |
|---------|--------|---------|--------|
| **Local LLM translation** | #196 (closed), **#212**, #214 (closed), **#132**, **#128**, **#233** | **#132** | v1.8.9.post1 fixed Ollama 404. #233 pre-existing. |
| **MPS/Apple Silicon** | #198 (closed), **#227** | **#227** | Known issue, deferred. |
| **Network/SSL/Install** | **#243** (NEW), **#240** (NEW), **#234**, **#229**, **#228**, **#225**, **#222**, **#221**, **#220**, **#218**, **#217**, #204, #210 | **#243** | #228/#229 self-resolved. #225 needs new approach. #240 fix coded. #243 installer path mismatch. |
| **GPU detection** | **#239** (NEW), **#200**, **#213** | **#200** | #239 is AMD dup of #142/#114. |
| **GUI / WebUI** | **#240** (NEW), **#236**, **#235**, #96, **#207** | **#240** | #235/#236 fixed in post1. #240 fix coded (private_mode). |
| **Whisper quality** | **#242** (NEW), **#241** (REGRESSION), **#237**, **#230**, **#224**, **#223**, **#209**, **#215** | **#241** | Regression fix coded. #242 feature request. |
| **Kaggle/Colab** | **#231**, **#132** | **#132** | #132 fixed. #231 is Kaggle env. |
| **AMD/Intel GPU** | **#239** (NEW), #142, #114, **#213** | Deferred | v1.9+ |
| **Model support** | **#232** | **#232** | whisper-ja-anime-v0.1 request. |
| **Translation providers** | #71, #43, **#233** | Deferred | #233 recommend Ollama |
| **i18n** | **#222**, #180, #175 | **#180** | v1.9+ |

---

## Recently Closed Issues

| # | Title | Closed | Resolution |
|---|-------|--------|------------|
| **#214** | 1.8.7 localLLM fail | 2026-03-18 | Closed |
| **#201** | Install SSL cert error | 2026-03-15 | Self-resolved: `pip install pip-system-certs` |
| **#198** | MPS beam search + detection | 2026-03-15 | Fixed v1.8.8. MPS working but not accelerating Whisper. |
| **#219** | MossFormer2_SS_16K 3D tensor crash | 2026-03-13 | Fixed v1.8.8 |
| #208 | LLM server AssertionError | 2026-03-09 | Self-resolved |
| #197 | Installation problem v1.8.6 | 2026-03-08 | Closed by user |
| #196 | Local Translation Errors | 2026-03-07 | Partial fix. See #212. |
| #195 | UnicodeDecodeError audio extraction | 2026-03-08 | Fixed |
| #194 | M4B file support | 2026-03-08 | Fixed |

---

## Issue Velocity Trend

| Period | New Issues | Closed | Net | Notes |
|--------|-----------|--------|-----|-------|
| 2026-03-08 to 2026-03-10 | 6 (#204-#210) | 5 | +1 | v1.8.7 release cycle |
| 2026-03-11 to 2026-03-13 | 8 (#211-#220) | 3 | +5 | v1.8.8 beta + stable |
| 2026-03-14 | 3 (#221-#223) | 1 | +2 | Post-release influx |
| 2026-03-15 | 3 (#224-#227) | 2 | +1 | Quality + MPS + install |
| 2026-03-16 | 5 (#228-#232) | 0 | +5 | Install, quality, Kaggle, model request |
| 2026-03-18 | 6 (#233-#238) | 1 | +5 | v1.8.9 release day + hotfix items |
| **2026-03-19** | **5 (#239-#243)** | **0** | **+5** | **#241 CPU REGRESSION, #240 GUI crash, #242 feature, #243 installer** |

**Trend**: 52 open issues (was 33 on 2026-03-08). Net +19 in 11 days. Each release generates 3-6 new issues within 24 hours. #241 is the first true regression — previous post-release issues were pre-existing or edge cases. The close rate is too low — 8 issues are safe to close but haven't been.

---

## Release & Roadmap Analysis (rev13 — 2026-03-19)

### v1.8.9.post1 — RELEASED (2026-03-19)

Fixed #132 (Ollama 404), #236 (WebUI cache), #235 (ctypes overflow), #238 (Portuguese). All responded to on GitHub.

### v1.8.9.post2 Hotfix — CODED, READY TO SHIP

| Item | Issues | Root Cause | Fix | Files |
|------|--------|------------|-----|-------|
| **CPU float16 regression** | **#241** | v1.8.9 changed `compute_type` to `float16` unconditionally. CTranslate2 does not support `float16` on CPU. | CTranslate2 branch: `device != "cuda"` → return `"auto"`. Safety net in ASR module for MPS→CPU downgrade. | `config/resolver_v3.py`, `modules/faster_whisper_pro_asr.py` |
| **GUI access violation** | **#240** | `private_mode=False` + `storage_path` creates persistent WebView2 data folder that can corrupt or trigger access violations. | Switch to `private_mode=True`, remove `storage_path` and `_clear_webview_cache_on_upgrade()`. All settings persist via Python backend (asr_config.json), not browser localStorage. | `webview_gui/main.py` |

**Impact assessment:**
- **#241** — CRITICAL. Fixes ALL CPU-only users, Optimus fallback users, MPS/Apple Silicon users.
- **#240** — HIGH. Simplifies GUI startup, removes entire category of WebView2 storage failures. Also retroactively improves the #236 fix (no cache to go stale = no stale UI).

### v1.9.0 — Proposed Scope

**Theme: Platform expansion + UX + close the gap**

| Priority | Item | Issues | Notes |
|----------|------|--------|-------|
| **P0** | Ollama full migration (deprecate llama-cpp-python) | #128, #132, #212, #233 | Remove ~1500 LOC fragile code |
| **P0** | OllamaManager GUI wiring | #132, #212 | CLI-only in v1.8.9 |
| **P0** | Chinese UI (i18n, at least partial) | #175, #180, #222 | Biggest support burden reducer |
| **P1** | Standalone merge CLI tool | #230 | `whisperjav-merge` command |
| **P1** | AMD ROCm support (document + partial) | #142, #114, #213, #239 | FishYu-OWO proved it works |
| **P1** | MPS selective policy | #198, #227 | force CPU for whisper-*, allow MPS for kotoba-* |
| **P2** | GUI settings persistence (pipeline tab) | #96, #207 | Long-standing request |
| **P2** | Vocal separation investigation | #224 | BS-RoFormer or UVR |
| **P2** | WebView2 reliability | #225, #240 | Two separate Win11 failures |
| **P3** | whisper-ja-anime-v0.1 model | #232 | If standard HF format |

---

## v1.9+ Backlog

| # | Issue | Category |
|---|-------|----------|
| #96 | Full GUI settings persistence (pipeline tab) | Enhancement |
| #213 | Intel GPU (XPU) support | Platform |
| #205 | VibeVoice ASR | Feature |
| #206 | Grey out incompatible GUI options | Feature |
| #180/#175 | Multi-language GUI (i18n) | Enhancement |
| #114/#142/#239 | DirectML / ROCm for AMD/Intel GPUs | Platform |
| #126 | Recursive directory + mirror output | Feature |
| #181 | Frameless window | Cosmetic |
| #43 | DeepL translation provider | Feature |
| #71 | Google Translate (no API key) | Feature |

---

## Changelog

| Date | Changes |
|------|---------|
| **2026-03-19** | **rev14.** 2 more new issues: #242 (XXL in Pass 1 feature request), #243 (installer verification fails on RTX 3050 Laptop). **v1.8.9.post2 fixes coded**: #241 (CPU float16 — `resolver_v3.py` returns `auto` for non-CUDA + safety net in `faster_whisper_pro_asr.py`) and #240 (GUI — switch to `private_mode=True`, remove `storage_path` and cache-clearing function). Reviewed all 52 open issues — no other hotfix candidates. Total open: 52. |
| **2026-03-19** | **rev13.** v1.8.9.post1 RELEASED — fixes for #132 (Ollama 404), #236 (WebUI cache), #235 (ctypes hwnd), #238 (Portuguese) all shipped and responded to on GitHub. 3 new issues: **#241 (CRITICAL REGRESSION — CPU float16 crash, breaks ALL CPU-only users)**, #240 (GUI access violation on Win11), #239 (AMD GPU request, dup of #142/#114). #225 new data: WebView2 confirmed latest, still white screen — old hypothesis exhausted. #237 new question from liugngg about XXL post-processing. v1.8.9.post2 recommended immediately for #241. Total open: 50. |
| 2026-03-18 | **rev12.** #238 Portuguese/Brazilian translation target added (4 files: providers.py, main.py, index.html ×2). v1.8.9.1 hotfix scope expanded to 4 items: #236 (WebUI cache), #132 (Ollama 404), #235 (ctypes hwnd), #238 (Portuguese). All coded. Total open: 47. |
| 2026-03-18 | **rev11.1.** Root causes validated for all 3 hotfix candidates. #132: doubled endpoint path confirmed. #236: WebView2 disk cache. #235: Win32 argtypes missing. All 3 fixes coded. |
| 2026-03-18 | **rev11.** v1.8.9 RELEASED. 5 new issues (#233-#237). #214 closed. #228 self-resolved. #231 root cause: Kaggle llvmlite. #132 new Ollama 404. Post-release bugs: #235, #236. Total open: 46. |
| 2026-03-17 | **rev10.** #229 self-resolved. #225 install log analyzed. #231 diagnostics sent. #227 batch_size answered. V189 quality plan verified. |
| 2026-03-16 | **rev9.** 5 new issues (#228-#232). BYOP XXL committed. #227 MPS benchmark. #223 ongoing feedback. Total open: 42. |
| 2026-03-15 | **rev8.1** Responded to #223, #225, #227. |
| 2026-03-15 | **rev8.** 3 new issues (#224-#227). MPS benchmark: 6x slower. #132 local works on Kaggle. |
| 2026-03-14 | **rev7.** v1.8.8 RELEASED. 3 new issues (#221-#223). #217 ESCALATED. |
| 2026-03-13 | rev6. All Track A/B/C code complete. 3 new issues (#218-#220). |
| 2026-03-12 | rev5. v1.8.8b1 pre-release. 5 new issues (#213-#217). |
| 2026-03-11 | rev4. v1.8.7 RELEASED. 3 new issues (#210-#212). |
| 2026-03-09 | Groups B, C, D committed. Fixes shipped. |
| 2026-03-08 | v1.8.7b1 released with China network fixes. |
| 2026-02-27 | v1.8.5-hotfix2 released. |

---

*This tracker is a point-in-time analysis. For live status, see [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues).*
