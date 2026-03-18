# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-18 (rev12 — #238 Portuguese language added, all 4 hotfix items coded) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **47 open** on GitHub

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
| Total open on GitHub | **47** | +6 since rev10 (#233-#238). #214 closed. |
| New issues since rev10 | 6 | #233 (translation error), #234 (CUDA version claim), #235 (ctypes icon bug), #236 (WebUI cache), #237 (XXL model question), #238 (Portuguese language) |
| Closed since rev10 | 1 | #214 (localLLM fail, closed 2026-03-18) |
| **NEEDS RESPONSE (no reply yet)** | 4 | #233, #234, #235, #237 |
| **AWAITING INFO (asked user for details)** | 1 | #232 (comparison results) |
| **AWAITING CONFIRMATION** | 8 | #200, #204, #207, #209, #210, #212, #218, #223 |
| **KEY NEW DATA** | 3 | #132 (Ollama 404 on v1.8.9), #231 (llvmlite version), #228 (self-resolved) |
| Feature requests (open) | 21 | See Cluster J |
| Deferred to v1.9+ | 11 | See v1.9+ Backlog |

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

### Post-release issues (same day)

| # | Title | Severity | Root Cause | Status |
|---|-------|----------|------------|--------|
| **#236** | WebUI cache prevents seeing v1.8.9 changes | **HIGH — BUG** | WebView2 caches old HTML/JS/CSS to disk (`private_mode=False`). After upgrade, stale assets served. | **ROOT CAUSE VALIDATED. FIX CODED.** Version-stamped cache clearing on startup — deletes `Cache/` and `Code Cache/` but preserves `Local Storage/`. |
| **#132** | Ollama 404 error on Kaggle with v1.8.9 | **HIGH — BUG** | **CONFIRMED from user's debug log**: `_api_base_to_custom_server()` returns endpoint `/v1/chat/completions`, then `service.py:393` and `cli.py:596` append `/v1/chat/completions` again → doubled path `/v1/chat/completions/v1/chat/completions` → HTTP 404. User log shows: `POST http://localhost:11434/v1/chat/completions/v1/chat/completions "HTTP/1.1 404 Not Found"`. NOT a custom model name issue. | **ROOT CAUSE VALIDATED. FIX CODED.** Removed redundant append in both files. |
| **#235** | ctypes icon OverflowError on first launch | **MEDIUM — BUG** | Win32 API functions called without `argtypes` declarations. 64-bit `hwnd` values overflow when passed to `IsWindowVisible()`. Non-blocking (exceptions ignored), works on second launch. | **ROOT CAUSE VALIDATED. FIX CODED.** Added proper `argtypes`/`restype` for all 7 Win32 functions. |
| **#237** | XXL can't select model / compute type question | **LOW** | yangming2027 confusion about XXL model selection and compute type. Not a bug. | `NEEDS RESPONSE` |
| **#238** | Add Portuguese/Brazilian translation target | **LOW — FEATURE** | Simple: add to SUPPORTED_TARGETS, argparse, GUI dropdowns. | **FIX CODED** — 4 files updated |
| **#233** | Local LLM translation AssertionError | **LOW** | Same llama-cpp-python `n_vocab()` failure as #208. Pre-existing, not a v1.8.9 regression. | `NEEDS RESPONSE` |

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

### Cluster A: Local LLM Translation — CRITICAL (4 issues + 1 contributor proposal)

**Issues**: #196 (closed), #212, #214 (closed), #132, #128 (contributor PR), #233 (NEW)
**Severity**: CRITICAL — affects ALL local LLM translation users across all platforms
**Status**: v1.8.9 shipped OllamaManager (CLI-only). #214 closed. #233 is same old llama-cpp-python failure. **#132 reports new 404 error with OllamaManager on Kaggle.**

| # | Title | Reporter | Platform | State | Detail |
|---|-------|----------|----------|-------|--------|
| **#233** | translation error | WillChengCN | Windows, cu118 | **OPEN (NEW)** | `--provider local`, llama-8b. `n_vocab()` AssertionError — model fails to load. Same root cause as #208. Pre-existing, not v1.8.9 regression. | `NEEDS RESPONSE` |
| **#214** | 1.8.7 localLLM fail | KenZP12 | Windows, cu128 | **CLOSED** (2026-03-18) | |
| **#212** | Regex Error - Local Translation v1.8.7 | destinyawaits | Linux, 5090 | **OPEN** | v1.8.9 comment posted. `AWAITING CONFIRMATION` |
| **#196** | Local Translation Errors | zhstark | Ubuntu, 5090 32GB | CLOSED | Related to #212. |
| **#132** | Local LLM on Kaggle | TinyRick1489 | Kaggle | **OPEN** | **2026-03-18**: Tried v1.8.9 `--provider ollama` with `translategemma:27b`. Gets `404 page not found`. **ROOT CAUSE FOUND**: doubled endpoint path bug in service.py + cli.py. Fix coded. | `FIX CODED` |
| **#128** | LLM context/batch sizing | hyiip | (contributor) | **OPEN** | Gemma 3 proposal. Redirected to OllamaManager configs. v1.8.9 comment posted. `AWAITING CONFIRMATION` |

**#132 root cause (2026-03-18)**: **VALIDATED from user's debug log.** The 404 is NOT a custom model name issue. It's a doubled endpoint path: `_api_base_to_custom_server('http://localhost:11434')` returns `('/v1/chat/completions')` as the endpoint, then `service.py:393` and `cli.py:596` append `'/v1/chat/completions'` again, producing `POST http://localhost:11434/v1/chat/completions/v1/chat/completions` → 404. Fix: remove the redundant append. This bug affects ALL `--provider ollama` users via both CLI and service paths — not just custom models.

**#233 analysis**: Same `n_vocab()` AssertionError as #208 (closed, self-resolved). llama-cpp-python model loading fails silently then crashes. This is the same fragile llama-cpp-python path that OllamaManager was designed to replace. Recommend: tell user to try `--provider ollama` instead.

---

### Cluster B: MPS / Apple Silicon — CONFIRMED PROBLEMATIC (2 issues)

**Issues**: #198 (closed), #227
**Status**: #198 closed. #227 has benchmark data. MPS selective policy deferred from v1.8.9. Documented as known issue in release notes.

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#198** | Transformers MPS crash | francetoastVN | **CLOSED** | MPS fix shipped v1.8.8. MPS confirmed 6x SLOWER for Whisper. |
| **#227** | M1 MAX Transformers hang | dadlaugh | **OPEN** | MPS model-dependent behavior confirmed. v1.8.9 documents as known issue. `AWAITING CONFIRMATION` |

---

### Cluster C: Network / Installation (10 issues — 1 new)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#234** | CUDA version claim | techguru0 | **OPEN (NEW)** | Claims 1x00 GPUs use CUDA 11.9. Misunderstanding: CUDA toolkit version depends on driver, not GPU series. | `NEEDS RESPONSE` |
| **#229** | INSTALLATION FAILED (SSL) | WillChengCN | **OPEN** | **SELF-RESOLVED**: "the cause is python environment." | `FIX VERIFIED` — safe to close |
| **#228** | cublas64_12.dll not found / first run hang | yhxkry | **OPEN** | **SELF-RESOLVED 2026-03-18**: Installed nvidia-cublas-cu12 and nvidia-cudnn-cu12 via pip, copied DLLs to python.exe dir. Working now. | `FIX VERIFIED` — safe to close |
| **#225** | 白屏 (GUI white screen) | github3C | **OPEN** | Install log clean (RTX 3070, 19/19 checks pass). White screen is runtime issue — likely WebView2 missing/outdated. | `AWAITING CONFIRMATION` |
| **#222** | 字幕是日语... (how to get Chinese?) | libinghui20001231-debug | **OPEN** | User doesn't know about translation feature. Pointed to docs. | `AWAITING CONFIRMATION` |
| **#221** | 安装完后报错 (cublas64_12.dll missing) | libinghui20001231-debug | **OPEN** | GTX 1650, old driver 462. Community helped. | `AWAITING LOG` |
| **#220** | 安装卡着不动 (install stalls) | libinghui20001231-debug | **OPEN** | Install stalls during PyTorch download. | `AWAITING LOG` |
| **#218** | 安装错误 (uv cu118 wheel) | WillChengCN | **OPEN** | uv rejects cu118 llama-cpp wheel | `SHIPPED` (A3). `AWAITING CONFIRMATION` |
| **#217** | 找不到WhisperJAV-GUI.exe | loveGEM + vimbackground | **OPEN** | Root cause: PyTorch download fails (network). Not installer bug. | `RESPONDED` — `AWAITING CONFIRMATION` |
| **#210** | 安装失败 DNS error | iop335577 | **OPEN** | DNS through proxy | `AWAITING CONFIRMATION` |
| **#204** | VPN/v2rayN SSL failures | yangming2027 | **OPEN** | HF hub SSL errors | `AWAITING CONFIRMATION` |

**#228 update (2026-03-18)**: yhxkry self-resolved. Installed `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` via pip into WhisperJAV's Python, then copied the DLLs to the python.exe directory. This is a workaround — the installer should be bundling these CUDA runtime libraries or detecting when they're missing. Worth investigating whether the conda-constructor installer includes them.

**#234 analysis**: techguru0 claims GTX 1x00 GPUs use "CUDA 11.9" and RTX 1600+ use "CUDA 12.9". This confuses CUDA compute capability with CUDA toolkit version. Any GPU supported by the driver can use CUDA 11.8 or 12.x toolkit. Needs a polite correction.

---

### Cluster D: GPU Detection (2 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#200** | NVML "Driver Not Loaded" on Optimus laptop | Ywocp | **OPEN** | Dual GPU. | `SHIPPED` (B1: `--force-cuda`). `AWAITING CONFIRMATION` |
| **#213** | Intel GPU support | DDXDB | **OPEN** | Requests `torch.xpu` | `DEFERRED` to v1.9+ |

---

### Cluster E: GUI / WebUI (4 issues — 2 new)

**Issues**: #96, #207, #235 (NEW), #236 (NEW)
**Status**: Two new v1.8.9 bugs. #236 (WebUI cache) is high priority — users can't see updates.

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#236** | WebUI cache prevents v1.8.9 changes | FishYu-OWO | **OPEN (NEW)** | WebView2 caches old HTML/JS to disk. Version-stamped cache clear on startup. | `FIX CODED` — hotfix candidate |
| **#235** | ctypes icon OverflowError on startup | techguru0 | **OPEN (NEW)** | Win32 API functions missing `argtypes` — 64-bit hwnd overflows. Added proper type declarations. | `FIX CODED` — hotfix candidate |
| **#207** | 1.86不能保存设置 | q864310563 | **OPEN** | Responded (dup of #96). `AWAITING CONFIRMATION` |
| **#96** | Full settings persistence | sky9639 | OPEN | `DEFERRED` to v1.9 |

**#236 analysis (HIGH PRIORITY)**: This affects ALL users upgrading to v1.8.9. WebView2 caches the old `index.html` and `app.js`. Users won't see the new BYOP XXL panel or other UI changes. FishYu-OWO's suggestion of `webview.start(private_mode=True)` is the correct fix — it disables the WebView2 disk cache. This should be a v1.8.9.1 hotfix or addressed immediately.

**#235 analysis**: The `EnumWindows` callback receives window handles as `LPARAM` which can exceed 32-bit int range on 64-bit Windows. The `IsWindowVisible(hwnd)` call fails when `hwnd` is a large 64-bit value. Fix: use `ctypes.c_void_p` or mask to appropriate range. Low priority since it's non-blocking and self-resolves on second launch.

---

### Cluster F: Whisper Output Quality (7 issues — 1 new)

**Issues**: #223, #224, #230, #237 (NEW), #209, #215, #227
**Severity**: HIGH — v1.8.9 shipped quality improvements + BYOP XXL. Awaiting user feedback.
**Status**: v1.8.9 notified on #223, #224, #209. #237 is new question about XXL.

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#237** | XXL can't select model / compute type question | yangming2027 | **OPEN (NEW)** | Asks: (1) is XXL default model large-v2? (2) why can't run XXL? (screenshot attached) (3) is Pass 2 balanced mode int8+fp16? | `NEEDS RESPONSE` |
| **#230** | Subtitle merging module request | weifu8435 | **OPEN** | Feature request. v1.8.9 comment posted (on roadmap for v1.9.0). | `RESPONDED` |
| **#224** | 人声分离分析 (vocal separation analysis) | yangming2027 | **OPEN** | v1.8.9 comment posted. | `AWAITING CONFIRMATION` |
| **#223** | Faster Whisper XXL comparison | weifu8435 | **OPEN** | v1.8.9 comment posted — BYOP XXL + quality tuning shipped. Awaiting feedback. | `SHIPPED` v1.8.9 |
| **#209** | Single subtitle very long (repetition) | weifu8435 | **OPEN** | v1.8.9 comment posted. `SHIPPED` (C1 in v1.8.8 + quality tuning in v1.8.9). | `AWAITING CONFIRMATION` |
| **#215** | Qwen3-ASR subtitle quality | yangming2027 | **OPEN** | Expected behavior. `AWAITING CONFIRMATION` |
| **#227** | M1 MAX Transformers hang | dadlaugh | **OPEN** | MPS known issue documented in v1.8.9. `AWAITING CONFIRMATION` |

**#237 analysis**: yangming2027 has 3 questions: (1) XXL uses whatever model you specify with `--model` flag in XXL's own args, not WhisperJAV's model dropdown. WhisperJAV sends `--model large-v3` by default. (2) Screenshot needed to diagnose why XXL won't run — could be exe path issue. (3) Pass 2 compute type is now float16 on CUDA in v1.8.9 (changed from int8_float16).

---

### Cluster G: Speech Enhancement — RESOLVED

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#219** | MossFormer2_SS_16K 3D tensor crash | anon12642 | **CLOSED** (2026-03-13) | `SHIPPED` in v1.8.8. |

---

### Cluster H: Kaggle / Colab (2 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#231** | Kaggle notebook run error | fzfile | **OPEN** | **UPDATE 2026-03-18**: Diagnostic results in — root cause is `llvmlite 0.43.0` (Kaggle pre-installed) but `numba` requires >= 0.46.0. This breaks `whisper → numba` import chain. | `DATA RECEIVED` — needs response |
| **#132** | Local LLM on Kaggle | TinyRick1489 | **OPEN** | **UPDATE 2026-03-18**: Tried v1.8.9 `--provider ollama`, gets 404 with `translategemma:27b`. Debug log attached. | `NEEDS INVESTIGATION` |

**#231 update (2026-03-18)**: fzfile ran diagnostic code. Full import chain: `stable_whisper → whisper → numba → llvmlite`. Kaggle has llvmlite 0.43.0 but numba requires >= 0.46.0. Fix for user: `!pip install -U llvmlite numba`. This is a Kaggle environment issue, not a WhisperJAV bug.

**#132 update (2026-03-18)**: TinyRick1489 installed Ollama manually on Kaggle, pulled `translategemma:27b`. Command: `whisperjav-translate -i file.srt --provider ollama --model 'translategemma:27b' --yes --debug`. Error: `Client error: 404 404 page not found`. This needs investigation — the 404 could mean OllamaManager is calling the wrong API endpoint for this model, or the model name format is different from what Ollama expects.

---

### Cluster I: Model Support Requests (1)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#232** | whisper-ja-anime-v0.1 model support | mustssr | **OPEN** | Requests `efwkjn/whisper-ja-anime-v0.1` HuggingFace model. | `AWAITING INFO` — asked for comparison results vs large-v3 |

---

### Cluster J: Feature Requests

| # | Title | Reporter | State | Summary | Target |
|---|-------|----------|-------|---------|--------|
| **#238** | Portuguese/Brazilian translation | (request) | OPEN | Add Portuguese as translation target language | **v1.8.9.1** — CODED |
| **#232** | whisper-ja-anime-v0.1 model | mustssr | OPEN | HuggingFace anime ASR model | **Investigate** |
| **#230** | Standalone merge module | weifu8435 | OPEN | CLI tool for multi-SRT merging | **v1.9.0** |
| **#224** | Vocal separation (UVR MDX-Net) | yangming2027 | OPEN | Detailed analysis of XXL's vocal separation advantage | **Investigate** |
| **#223** | Faster Whisper XXL comparison | weifu8435 | OPEN | Quality gap with XXL | **SHIPPED v1.8.9** |
| **#213** | Intel GPU (XPU) support | DDXDB | OPEN | torch.xpu via PyTorch XPU wheels | v1.9+ |
| **#206** | Grey out incompatible options | techguru0 | OPEN | Block incompatible GUI choices | v1.9+ |
| **#205** | VibeVoice ASR | kylesskim-sys | OPEN | Microsoft VibeVoice. VRAM too high. | v1.9+ |
| **#181** | Frameless window | QQ804218 | OPEN | Cosmetic | v1.9+ |
| **#180** | Multi-language GUI | QQ804218 | OPEN | Full i18n | v1.9+ |
| **#175** | Chinese GUI | yangming2027 | OPEN | Subset of #180 | v1.9+ |
| **#164** | MPEG-TS + Drive | hosmallming | OPEN | Format + cloud | Backlog |
| **#142** | AMD Radeon | MatthaisUK | OPEN | FishYu-OWO AMD workaround (CTranslate2 ROCm wheels) | v1.9+ |
| **#128** | Gemma 3 model upgrade | hyiip | OPEN | 128K context Gemma 3 4B/12B models. **Contributor offers to implement.** | **v1.9.0** |
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
| **NEEDS RESPONSE (new/unresponded)** | 4 | #233, #234, #235, #237 |
| **NEEDS INVESTIGATION** | 1 | #132 (Ollama 404 on v1.8.9) |
| **DATA RECEIVED (needs response)** | 1 | #231 (llvmlite version mismatch) |
| **AWAITING LOG (install log requested)** | 2 | #220, #221 |
| **AWAITING CONFIRMATION** | 8 | #200, #204, #207, #209, #210, #212, #218, #223 |
| **FIX VERIFIED (safe to close)** | 2 | #228 (self-resolved), #229 (self-resolved) |
| **Feature requests (open)** | 21 | See Cluster J |
| **DEFERRED to v1.9+** | 11 | #96, #205, #206, #213, #180, #175, #181, #142, #114, #126, #43 |

### By Priority (Active Work)

| Priority | # | Issue | Status | Why |
|----------|---|-------|--------|-----|
| **HIGH** | #236 | WebUI cache after update | `ACKNOWLEDGED` | ALL upgrading users affected. Need hotfix. |
| **HIGH** | #132 | Ollama 404 on Kaggle v1.8.9 | `NEEDS INVESTIGATION` | OllamaManager endpoint bug with custom model names? |
| **MEDIUM** | #235 | ctypes icon overflow | `NEEDS RESPONSE` | v1.8.9 bug, non-blocking |
| **MEDIUM** | #231 | Kaggle llvmlite version | `DATA RECEIVED` | Kaggle env issue, not our bug |
| **MEDIUM** | #237 | XXL model/compute question | `NEEDS RESPONSE` | User confusion, needs explanation |
| **MEDIUM** | #233 | Local LLM AssertionError | `NEEDS RESPONSE` | Pre-existing llama-cpp issue, recommend Ollama |
| **LOW** | #234 | CUDA version misunderstanding | `NEEDS RESPONSE` | Polite correction needed |

---

## Pending GitHub Actions

### Issues Needing Response (6)

| # | Action Needed | Priority |
|---|--------------|----------|
| **#237** | Explain: XXL uses its own model (WhisperJAV sends `--model large-v3`). Ask for screenshot/error details. Explain v1.8.9 compute type is now float16. | MEDIUM |
| **#235** | Acknowledge: known issue with `_set_windows_icon` on first launch. Non-blocking, works on second launch. Fix planned. | MEDIUM |
| **#234** | Politely correct: CUDA toolkit version depends on driver, not GPU series. GTX 1060 can use CUDA 11.8 or 12.x. | LOW |
| **#233** | Respond: This is a known llama-cpp-python model loading failure. Recommend trying `--provider ollama` instead. v1.8.9 has OllamaManager (CLI). | MEDIUM |
| **#231** | Respond: Root cause is Kaggle's outdated llvmlite (0.43.0). Fix: `!pip install -U llvmlite numba` before importing WhisperJAV. Not a WhisperJAV bug. | MEDIUM |
| **#132** | Investigate: Ollama 404 with `translategemma:27b`. Download and examine the debug log. Check if OllamaManager uses wrong endpoint. | HIGH |

### Issues With Coded Fixes (awaiting hotfix release)

| # | Fix | Status |
|---|-----|--------|
| **#236** | Version-stamped WebView2 cache clear on upgrade | **CODED** — in `webview_gui/main.py` |
| **#235** | Proper `argtypes` for Win32 functions | **CODED** — in `webview_gui/main.py` |
| **#132** | Remove doubled `/v1/chat/completions` endpoint | **CODED** — in `translate/service.py` + `translate/cli.py` |
| **#238** | Add Portuguese/Brazilian translation target | **CODED** — in `providers.py`, `main.py`, `index.html` (both dropdowns) |

### Candidates for Closing

| # | Condition | Notes |
|---|-----------|-------|
| **#229** | Self-resolved: "the cause is python environment." | `FIX VERIFIED` |
| **#228** | Self-resolved: installed CUDA runtime DLLs manually | `FIX VERIFIED` |
| **#222** | Responded with docs + translation instructions | Last activity 2026-03-14 |
| **#211** | urllib3 warning fixed in v1.8.8 | Can close |
| **#214** | Already closed 2026-03-18 | Done |

### Decisions Needed

| # | Decision | Recommendation |
|---|----------|----------------|
| **#236/#235/#132** | Release v1.8.9.1 hotfix? | **YES** — all 3 fixes coded with validated root causes. #132 breaks ALL Ollama users, #236 affects ALL upgrading users. |
| **#227** | MPS strategy — selective by model? | Defer to v1.9.0 (needs more M3/M4 data) |
| **#128** | Accept hyiip's Gemma 3 contribution? | Coordinate for v1.9.0 |

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Status |
|---------|--------|---------|--------|
| **Local LLM translation** | #196 (closed), **#212**, #214 (closed), **#132**, **#128**, **#233** (NEW) | **#132** | v1.8.9 shipped OllamaManager (CLI). #132 has new 404 bug. #233 is pre-existing llama-cpp failure. |
| **MPS/Apple Silicon** | #198 (closed), **#227** | **#227** | Documented as known issue in v1.8.9 release notes. |
| **Network/SSL/Install** | **#234** (NEW), **#229**, **#228**, **#225**, **#222**, **#221**, **#220**, **#218**, **#217**, #204, #210 | **#217** | #228 and #229 self-resolved. #234 is CUDA version misunderstanding. |
| **GPU detection** | **#200**, **#213** | **#200** | `SHIPPED` v1.8.8; #213 deferred |
| **GUI / WebUI** | **#236** (NEW), **#235** (NEW), #96, **#207** | **#236** | Two new v1.8.9 bugs. #236 (cache) is high priority. |
| **Whisper quality** | **#237** (NEW), **#230**, **#224**, **#223**, **#209**, **#215** | **#223** | v1.8.9 shipped BYOP XXL + quality tuning. #237 is usage question. |
| **Kaggle/Colab** | **#231**, **#132** | **#132** | #231 is Kaggle env issue (llvmlite). #132 has new Ollama 404. |
| **Model support** | **#232** | **#232** | whisper-ja-anime-v0.1 request. |
| **AMD/Intel GPU** | #142, #114, **#213** | Deferred | v1.9+ |
| **Translation providers** | #71, #43, **#233** (NEW) | Deferred | #233 is llama-cpp failure, recommend Ollama |
| **i18n** | **#222**, #180, #175 | **#180** | v1.9+ |

---

## Recently Closed Issues

| # | Title | Closed | Resolution |
|---|-------|--------|------------|
| **#214** | 1.8.7 localLLM fail | 2026-03-18 | Closed |
| **#201** | Install SSL cert error | 2026-03-15 | Self-resolved: `pip install pip-system-certs` |
| **#198** | MPS beam search + detection | 2026-03-15 | Fixed v1.8.8. MPS working but not accelerating Whisper (benchmark confirmed). |
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
| **2026-03-18** | **5 (#233-#237)** | **1** | **+4** | v1.8.9 release day: translation, CUDA, GUI bugs, WebUI cache |

**Trend**: Consistent 3-5 new issues per day. v1.8.9 release brought immediate post-release issues (#235 icon bug, #236 WebUI cache). #236 is the most urgent — affects all upgrading users.

---

## Release & Roadmap Analysis (rev11 — 2026-03-18)

### v1.8.9 — RELEASED

**Theme: Quality + BYOP + LLM reliability**

All planned items shipped except MPS selective policy (deferred, needs more data).

### v1.8.9.1 Hotfix — RECOMMENDED (all 3 fixes coded, root causes validated)

| Item | Issues | Root Cause | Fix | Files |
|------|--------|------------|-----|-------|
| WebUI cache on upgrade | #236 | WebView2 disk cache serves stale HTML/JS after upgrade | Version-stamped cache clear on startup (preserves Local Storage) | `webview_gui/main.py` |
| Ollama doubled endpoint | #132 | `/v1/chat/completions` appended twice → 404 | Remove redundant append | `translate/service.py`, `translate/cli.py` |
| ctypes hwnd overflow | #235 | Win32 functions missing `argtypes` — 64-bit hwnd overflows | Added proper `argtypes`/`restype` for 7 Win32 functions | `webview_gui/main.py` |
| Portuguese translation target | #238 | Missing language option | Added to SUPPORTED_TARGETS, argparse, both GUI dropdowns | `providers.py`, `main.py`, `index.html` |

**Impact assessment:**
- **#132** — Breaks ALL `--provider ollama` users (CLI and service paths). Not just custom models. Highest urgency.
- **#236** — Affects ALL users upgrading from any previous version. They see old UI with no indication anything changed.
- **#235** — Non-blocking (exceptions ignored, works on retry). Lowest urgency but easy win.

### v1.9.0 — Proposed Scope

**Theme: Platform expansion + UX**

| Priority | Item | Issues | Notes |
|----------|------|--------|-------|
| **P0** | Ollama full migration (deprecate llama-cpp-python) | #128, #132, #212, #233 | Remove ~1500 LOC fragile code |
| **P0** | Chinese UI (i18n, at least partial) | #175, #180, #222 | Biggest support burden reducer |
| **P0** | OllamaManager GUI wiring | #132, #212 | CLI-only in v1.8.9 |
| **P1** | Standalone merge CLI tool | #230 | `whisperjav-merge` command |
| **P1** | AMD ROCm support (document + partial) | #142, #114, #213 | FishYu-OWO proved it works |
| **P1** | MPS selective policy | #198, #227 | force CPU for whisper-*, allow MPS for kotoba-* |
| **P2** | GUI settings persistence (pipeline tab) | #96, #207 | Long-standing request |
| **P2** | Vocal separation investigation | #224 | BS-RoFormer or UVR |
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
| #114/#142 | DirectML / ROCm for AMD/Intel GPUs | Platform |
| #126 | Recursive directory + mirror output | Feature |
| #181 | Frameless window | Cosmetic |
| #43 | DeepL translation provider | Feature |
| #71 | Google Translate (no API key) | Feature |

---

## Changelog

| Date | Changes |
|------|---------|
| **2026-03-18** | **rev12.** #238 Portuguese/Brazilian translation target added (4 files: providers.py, main.py, index.html ×2). v1.8.9.1 hotfix scope expanded to 4 items: #236 (WebUI cache), #132 (Ollama 404), #235 (ctypes hwnd), #238 (Portuguese). All coded. Total open: 47. |
| **2026-03-18** | **rev11.1.** Root causes validated for all 3 hotfix candidates. #132: doubled endpoint path `/v1/chat/completions/v1/chat/completions` confirmed from user's debug log — breaks ALL Ollama users, not just custom models. #236: WebView2 disk cache — fix clears Cache/Code Cache on version change, preserves Local Storage. #235: Win32 argtypes missing — added proper type declarations. All 3 fixes coded. |
| **2026-03-18** | **rev11.** v1.8.9 RELEASED. 5 new issues (#233-#237). #214 closed. #228 self-resolved (installed CUDA runtime DLLs). #231 root cause: Kaggle llvmlite 0.43.0 too old. #132 new: Ollama 404 with translategemma:27b on v1.8.9. **Post-release bugs**: #235 ctypes hwnd overflow, #236 WebUI cache prevents seeing v1.8.9 changes. v1.8.9 release comments posted on #223, #224, #209, #212, #214, #132, #128, #230. Cluster E renamed to GUI/WebUI, expanded with #235/#236. Total open: 46. |
| 2026-03-17 | **rev10.** #229 self-resolved (python env). #225 install log analyzed — clean install, white screen is WebView2 runtime issue. #231 fzfile replied (T4x2, GitHub notebook), sent diagnostic code. #227 answered dadlaugh's batch_size question, asked for M3/M4 testers. All issues responded — 0 NEEDS RESPONSE. V189 quality plan verified complete. #128 Gemma 3 redirected to OllamaManager. |
| 2026-03-16 | **rev9.** 5 new issues (#228-#232). BYOP XXL committed (`aed1af2`). #217 received install log from vimbackground (needs analysis). #227 received MPS benchmark from dadlaugh (model-dependent: kotoba works, whisper-large fails). #223 continued feedback (16 comments). #198 and #201 closed. #230 new feature request for standalone merge tool. #231 Kaggle environment error. #232 whisper-ja-anime model request. Total open: 42. |
| 2026-03-15 | **rev8.1** Responded to #223, #225, #227. |
| 2026-03-15 | **rev8.** 3 new issues (#224-#227). MPS benchmark confirms Whisper 6x SLOWER on MPS than CPU. #132 `--provider local` works on Kaggle. #128 Gemma 3 proposal. |
| 2026-03-14 | **rev7.** v1.8.8 RELEASED. 3 new issues (#221-#223). #217 ESCALATED. |
| 2026-03-13 | rev6. All Track A/B/C code complete. 3 new issues (#218-#220). |
| 2026-03-12 | rev5. v1.8.8b1 pre-release. 5 new issues (#213-#217). |
| 2026-03-11 | rev4. v1.8.7 RELEASED. 3 new issues (#210-#212). |
| 2026-03-09 | Groups B, C, D committed. Fixes shipped. |
| 2026-03-08 | v1.8.7b1 released with China network fixes. |
| 2026-02-27 | v1.8.5-hotfix2 released. |

---

*This tracker is a point-in-time analysis. For live status, see [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues).*
