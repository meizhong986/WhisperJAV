# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-16 (rev9 — 5 new issues, BYOP XXL shipped, #217 install log received) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **42 open** on GitHub

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
| Total open on GitHub | **42** | +5 since rev8.1 (#228, #229, #230, #231, #232) |
| New issues since rev8.1 | 5 | #228 (cublas64_12.dll), #229 (install SSL fail), #230 (merge module request), #231 (Kaggle error), #232 (whisper-ja-anime model) |
| Closed since rev8.1 | 2 | #198 (MPS, closed by user), #201 (SSL, self-resolved) |
| **NEEDS RESPONSE (no reply yet)** | 0 | All responded 2026-03-16 |
| **AWAITING INFO (asked user for details)** | 3 | #228 (nvidia-smi, install method), #231 (full traceback, version), #232 (comparison results) |
| **AWAITING LOG (asked for install log)** | 3 | #220, #221, #225 |
| **AWAITING CONFIRMATION** | 9 | #200, #204, #207, #209, #210, #212, #214, #218 |
| **KEY NEW DATA** | 2 | #217 (install log received from vimbackground), #227 (MPS benchmark from dadlaugh) |
| Feature requests (open) | 20 | See Cluster J |
| Deferred to v1.9+ | 11 | See v1.9+ Backlog |

---

## v1.8.9 Development — IN PROGRESS

### Branch: `dev_v1.8.9.beta` (5 commits ahead of main)

| Commit | Description | Status |
|--------|-------------|--------|
| `aed1af2` | **BYOP Faster Whisper XXL** — full GUI + CLI + worker integration | DONE |
| `6f27629` | Config/settings cleanup (WS1-WS5 workstreams) | DONE |
| `3d4c4da` | Quality/accuracy tuning work (parameter audit findings) | DONE |
| `a94f862` | Planning for v1.8.9 and beyond | DONE |
| `dd56aa4` | OllamaManager smart integration | DONE |

### v1.8.9 Planned Items — Progress

| Priority | Item | Issues | Status | Notes |
|----------|------|--------|--------|-------|
| **P0** | Quality parameter fixes (large-v3, float16, beam_size=5) | #223, #224 | **DONE** | All Q1-Q6 verified in Pydantic presets + resolver_v3.py. See `V189_QUALITY_IMPROVEMENT_PLAN.md`. |
| **P0** | BYOP Faster Whisper XXL ensemble integration | #223 | **DONE** | Committed `aed1af2`. Full GUI + CLI + worker chain. |
| **P0** | MPS → CPU default for Whisper on Apple Silicon | #198, #227 | **NOT STARTED** | #227 benchmark shows kotoba MPS works but whisper-large-v3-turbo produces garbage on MPS |
| **P1** | OllamaManager smart integration | #132, #212, #214 | **CODED** | Needs testing. `ollama_manager.py` committed. |
| **P1** | Gemma 3 model upgrade (hyiip contribution) | #128, #132 | **DECISION NEEDED** | hyiip offers to implement. TinyRick1489 asks about num_ctx tuning (#132). |
| **P2** | Config/settings cleanup (WS1-WS5) | — | **DONE** | Committed `6f27629`. |
| **P2** | Installer investigation (#217 GUI exe) | #217 | **ANALYZED** | Root cause: PyTorch cu118 2.6GB download fails (TLS drops). Not installer bug. Responded on GitHub. |
| **P3** | VAD failover + scene detection fixes (Q4, Q5) | #223 | **DONE** | Q4: `should_force_full_transcribe` in faster_whisper_pro_asr.py. Q5: Pydantic default auto-derives 28.0. |

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

**Issues**: #196 (closed), #212, #214, #132, #128 (contributor PR)
**Severity**: CRITICAL — affects ALL local LLM translation users across all platforms
**Status**: v1.8.8 tactical fixes shipped. #132 partially verified. **#128 has Gemma 3 proposal from hyiip.** **#132 new comment: TinyRick1489 asks about Ollama num_ctx tuning.**

| # | Title | Reporter | Platform | State | Detail |
|---|-------|----------|----------|-------|--------|
| **#214** | 1.8.7 localLLM fail | KenZP12 | Windows, cu128 | **OPEN** | gemma-9b, 502 error. `AWAITING CONFIRMATION` |
| **#212** | Regex Error - Local Translation v1.8.7 | destinyawaits | Linux, 5090 | **OPEN** | llama-8b AND gemma-9b both fail. `AWAITING CONFIRMATION` |
| **#196** | Local Translation Errors | zhstark | Ubuntu, 5090 32GB | CLOSED | Related to #212. |
| **#132** | Local LLM on Kaggle | TinyRick1489 | Kaggle | **OPEN** | **UPDATE 2026-03-15**: `--provider local` works. `--provider ollama` works after manual install. **NEW**: TinyRick1489 asks if there's an arg to change Ollama num_ctx (currently 8192, caps batch_size to 11). `RESPONDED` — settings file override via model_params.num_ctx |
| **#128** | LLM context/batch sizing | hyiip | (contributor) | **OPEN** | hyiip proposes Gemma 3 4B/12B (128K context). **Offers to implement.** `DECISION NEEDED` |

---

### Cluster B: MPS / Apple Silicon — CONFIRMED PROBLEMATIC (2 issues)

**Issues**: #198 (closed), #227
**Status**: #198 closed. #227 has new benchmark data from dadlaugh showing model-dependent MPS behavior.

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#198** | Transformers MPS crash | francetoastVN | **CLOSED** | MPS fix shipped v1.8.8. MPS confirmed 6x SLOWER for Whisper. |
| **#227** | M1 MAX Transformers hang | dadlaugh | **OPEN** | **UPDATE 2026-03-16**: dadlaugh ran MPS test script. Results: kotoba-bilingual **works on MPS** (28s MPS vs 35s CPU = 1.25x speedup), but whisper-large-v3-turbo produces **garbage output on MPS** (42s, wrong language). CPU workaround works but takes 2-3 hours for 120-min video. `AWAITING CONFIRMATION` |

**MPS analysis update (2026-03-16):**
- MPS is **model-dependent**: kotoba-bilingual works, whisper-large-v3-turbo fails
- dadlaugh's benchmark shows MPS CAN provide speedup (1.25x) for some HuggingFace models
- But standard Whisper models produce garbage on MPS — language detection fails
- francetoastVN (M1 Pro) confirms wanting to upgrade to M5 Max, watches progress
- **Recommendation updated**: Don't blanket-disable MPS. Instead: force CPU for `openai/whisper-*` models, allow MPS for `kotoba-*` models

---

### Cluster C: Network / Installation (9 issues — 2 new)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#229** | INSTALLATION FAILED (SSL) | WillChengCN | **OPEN (NEW)** | SSL cert verification failure during preflight. Same user as #218. VPN/proxy. | `AWAITING CONFIRMATION` — suggested antivirus/proxy check |
| **#228** | cublas64_12.dll not found / first run hang | yhxkry | **OPEN (NEW)** | First run hung all day. Second attempt: missing CUDA library. yangming2027 helped (CUDA toolkit not installed). | `AWAITING INFO` — asked for nvidia-smi, install method |
| **#225** | 白屏 (GUI white screen) | github3C | **OPEN** | Program launches but shows white screen. | `AWAITING LOG` |
| **#222** | 字幕是日语... (how to get Chinese?) | libinghui20001231-debug | **OPEN** | User doesn't know about translation feature. Pointed to docs. | `AWAITING CONFIRMATION` |
| **#221** | 安装完后报错 (cublas64_12.dll missing) | libinghui20001231-debug | **OPEN** | GTX 1650, old driver 462. Community helped. | `AWAITING LOG` |
| **#220** | 安装卡着不动 (install stalls) | libinghui20001231-debug | **OPEN** | Install stalls during PyTorch download. | `AWAITING LOG` |
| **#218** | 安装错误 (uv cu118 wheel) | WillChengCN | **OPEN** | uv rejects cu118 llama-cpp wheel | `SHIPPED` (A3). `AWAITING CONFIRMATION` |
| **#217** | 找不到WhisperJAV-GUI.exe | loveGEM + vimbackground | **OPEN** | **ROOT CAUSE FOUND (2026-03-16)**: Install log shows PyTorch download (2.6GB cu118) fails 3/3 with TLS connection drop. Installation never completed — no installer code bug. Network reliability issue for large downloads. Responded with explanation. | `RESPONDED` — `AWAITING CONFIRMATION` |
| **#210** | 安装失败 DNS error | iop335577 | **OPEN** | DNS through proxy | `AWAITING CONFIRMATION` |
| **#204** | VPN/v2rayN SSL failures | yangming2027 | **OPEN** | HF hub SSL errors | `AWAITING CONFIRMATION` |

**#217 update (2026-03-16)**: **ROOT CAUSE FOUND.** vimbackground's install log shows PyTorch cu118 download (2.6GB) failing 3/3 attempts with TLS connection drop (`peer closed connection without sending TLS close_notify`). Installation never completed — Phase 3 failure prevents all subsequent phases including WhisperJAV package install and GUI creation. NOT an installer code bug. Network reliability issue for large downloads in China. Responded 2026-03-16.

**#229**: Same user as #218 (WillChengCN). SSL cert verification failure. Likely VPN/corporate proxy intercepting SSL. User asks if turning off VPN helps.

**#228**: yhxkry's first run hung for a full day (likely model download). Second run: cublas64_12.dll missing — CUDA toolkit not installed or not on PATH. yangming2027 (community) diagnosed it in comments.

---

### Cluster D: GPU Detection (2 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#200** | NVML "Driver Not Loaded" on Optimus laptop | Ywocp | **OPEN** | Dual GPU. | `SHIPPED` (B1: `--force-cuda`). `AWAITING CONFIRMATION` |
| **#213** | Intel GPU support | DDXDB | **OPEN** | Requests `torch.xpu` | `DEFERRED` to v1.9+ |

---

### Cluster E: GUI Settings Persistence (2 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#207** | 1.86不能保存设置 | q864310563 | **OPEN** | Responded (dup of #96). `AWAITING CONFIRMATION` |
| **#96** | Full settings persistence | sky9639 | OPEN | `DEFERRED` to v1.9 |

---

### Cluster F: Whisper Output Quality — EXPANDED (6 issues — 1 new)

**Issues**: #223, #224, #230 (NEW), #209, #215, #227
**Severity**: HIGH — quality is now the #1 user concern, multiple power users reporting
**Status**: Quality improvement plan drafted. BYOP XXL integration done. New #230 requests standalone merge module.

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#230** | Subtitle merging module request | weifu8435 | **OPEN (NEW)** | Wants standalone merge tool for multi-pass merging (pass3, pass4...). 5 comments, active discussion with justantopair-ai. | `AWAITING CONFIRMATION` — acknowledged, noted for roadmap |
| **#224** | 人声分离分析 (vocal separation analysis) | yangming2027 | **OPEN** | Detailed technical breakdown of XXL's vocal separation advantage. cbl19961214-sudo commented agreeing but noting WhisperJAV has basic audio processing. | `AWAITING CONFIRMATION` — corrected: BS-RoFormer exists |
| **#223** | Faster Whisper XXL comparison | weifu8435 | **OPEN** | **UPDATE 2026-03-16**: weifu8435 continues arguing balanced mode is worse than PotPlayer live captions. 16 comments total. We asked for test files to benchmark. BYOP XXL integration now shipped as response. | `RESPONDED` |
| **#209** | Single subtitle very long (repetition) | weifu8435 | **OPEN** | `SHIPPED` (C1: pattern #8). `AWAITING CONFIRMATION` |
| **#215** | Qwen3-ASR subtitle quality | yangming2027 | **OPEN** | Responded — expected behavior. `AWAITING CONFIRMATION` |
| **#227** | M1 MAX Transformers hang | dadlaugh | **OPEN** | See Cluster B. MPS model-dependent behavior confirmed. | `AWAITING CONFIRMATION` |

**#230 analysis**: weifu8435 wants a standalone `whisperjav-merge` command that takes multiple SRT files and merges them iteratively. The existing ensemble merge is internal to 2-pass mode. This is a feature request for a CLI tool. weifu8435 is WhisperJAV's most active quality-focused user (filed #223, #209, #230).

**#223 update**: 16 comments now. weifu8435 continues to push on quality gap. Our response: (1) parameter fixes coming in v1.8.9, (2) BYOP XXL integration shipped so users can use XXL as Pass 2. Asked for test files for benchmarking.

---

### Cluster G: Speech Enhancement — RESOLVED

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#219** | MossFormer2_SS_16K 3D tensor crash | anon12642 | **CLOSED** (2026-03-13) | `SHIPPED` in v1.8.8. |

---

### Cluster H: Kaggle / Colab (2 issues — 1 new)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#231** | Kaggle notebook run error | fzfile | **OPEN (NEW)** | Error: `ffmpeg-dsp not available in single-pass mode` + stable_whisper import error (`whisper.tokenizer`). Running on Kaggle. | `AWAITING INFO` — asked for full traceback, version, install method |
| **#132** | Local LLM on Kaggle | TinyRick1489 | **OPEN** | Partially working. num_ctx override answered. | `AWAITING CONFIRMATION` |

**#231 analysis**: Two errors visible: (1) ffmpeg-dsp enhancer not available in single-pass mode (correct — speech enhancement requires scene detection pipeline), (2) stable_whisper fails to import `whisper.tokenizer` — likely openai-whisper not installed or version mismatch. This may be a Kaggle environment issue.

---

### Cluster I: Model Support Requests (1 new)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#232** | whisper-ja-anime-v0.1 model support | mustssr | **OPEN (NEW)** | Requests `efwkjn/whisper-ja-anime-v0.1` HuggingFace model. Notes it and Anime-Whisper each have strengths. | `AWAITING INFO` — asked for comparison results vs large-v3 |

**#232 analysis**: whisper-ja-anime-v0.1 is a fine-tuned Whisper model for Japanese anime. WhisperJAV already has anime-whisper integration. Adding another anime model would be a Transformers pipeline addition. Low effort if it follows standard HuggingFace format.

---

### Cluster J: Feature Requests

| # | Title | Reporter | State | Summary | Target |
|---|-------|----------|-------|---------|--------|
| **#232** | whisper-ja-anime-v0.1 model | mustssr | OPEN (NEW) | HuggingFace anime ASR model | **Investigate** |
| **#230** | Standalone merge module | weifu8435 | OPEN (NEW) | CLI tool for multi-SRT merging | **Investigate** |
| **#224** | Vocal separation (UVR MDX-Net) | yangming2027 | OPEN | Detailed analysis of XXL's vocal separation advantage | **Investigate** |
| **#223** | Faster Whisper XXL comparison | weifu8435 | OPEN | Quality gap with XXL | **v1.8.9 (params + BYOP XXL)** |
| **#213** | Intel GPU (XPU) support | DDXDB | OPEN | torch.xpu via PyTorch XPU wheels | v1.9+ |
| **#206** | Grey out incompatible options | techguru0 | OPEN | Block incompatible GUI choices | v1.9+ |
| **#205** | VibeVoice ASR | kylesskim-sys | OPEN | Microsoft VibeVoice. VRAM too high. | v1.9+ |
| **#181** | Frameless window | QQ804218 | OPEN | Cosmetic | v1.9+ |
| **#180** | Multi-language GUI | QQ804218 | OPEN | Full i18n | v1.9+ |
| **#175** | Chinese GUI | yangming2027 | OPEN | Subset of #180 | v1.9+ |
| **#164** | MPEG-TS + Drive | hosmallming | OPEN | Format + cloud | Backlog |
| **#142** | AMD Radeon | MatthaisUK | OPEN | FishYu-OWO AMD workaround (CTranslate2 ROCm wheels) | v1.9+ |
| **#128** | Gemma 3 model upgrade | hyiip | OPEN | 128K context Gemma 3 4B/12B models. **Contributor offers to implement.** | **v1.8.9 or v1.9.0** |
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
| **NEEDS RESPONSE (new/unresponded)** | 0 | All responded 2026-03-16 (incl. #132 num_ctx). |
| **AWAITING LOG (install log requested)** | 3 | #220, #221, #225 |
| **AWAITING CONFIRMATION** | 9 | #200, #204, #207, #209, #210, #212, #214, #218 |
| **KEY DATA RECEIVED (action needed)** | 4 | #217 (install log), #227 (MPS benchmark), #132 (num_ctx question), #223 (continued feedback) |
| **Feature requests (open)** | 22 | See Cluster J |
| **DEFERRED to v1.9+** | 11 | #96, #205, #206, #213, #180, #175, #181, #142, #114, #126, #43 |

### By Priority (Active Work)

| Priority | # | Issue | Status | Why |
|----------|---|-------|--------|-----|
| **RESOLVED** | #217 | GUI exe missing after install | `RESPONDED` | Root cause: PyTorch download fails (network). Not an installer bug. |
| **CRITICAL** | #223/#224 | Quality gap vs Faster Whisper XXL | `BYOP SHIPPED + PARAMS DONE` | BYOP XXL done. All Q1-Q6 parameter fixes verified complete. |
| **HIGH** | #230 | Standalone merge module request | `RESPONDED` | Acknowledged, noted for roadmap. |
| **HIGH** | #128 | Gemma 3 model upgrade proposal | `DECISION NEEDED` | hyiip offers to implement. Fixes LLM context root cause. |
| **HIGH** | #227 | MPS model-dependent behavior | `DATA RECEIVED` | kotoba works on MPS, whisper-large fails. Need selective MPS policy. |
| **HIGH** | #132 | Kaggle Ollama num_ctx tuning | `RESPONDED` | Answered: settings file model_params.num_ctx override. |
| **MEDIUM** | #231 | Kaggle notebook error | `AWAITING INFO` | Asked for full traceback, version. |
| **MEDIUM** | #232 | whisper-ja-anime-v0.1 model | `AWAITING INFO` | Asked for comparison results. |
| **MEDIUM** | #229 | Install SSL failure (WillChengCN) | `RESPONDED` | Suggested antivirus/proxy check. |
| **MEDIUM** | #228 | cublas64_12.dll + first run hang | `AWAITING INFO` | Asked for nvidia-smi, install method. |
| **LOW** | #222 | How to get Chinese subs | `AWAITING CONFIRMATION` | Responded with docs. |

---

## Pending GitHub Actions

### Issues Needing Response (6)

| # | Action Needed | Priority |
|---|--------------|----------|
| **#232** | Respond: whisper-ja-anime-v0.1 is HF model, can potentially add to Transformers pipeline. Ask user for benchmark results vs anime-whisper. | MEDIUM |
| **#231** | Respond: ffmpeg-dsp not available in single-pass. Suggest removing `--speech-enhancer` or using ensemble mode. The whisper.tokenizer error is a Kaggle environment issue — openai-whisper may not be installed. | HIGH |
| **#230** | Respond: Acknowledge value. Note ensemble merge exists internally. A standalone `whisperjav-merge` CLI tool is a reasonable feature request. BYOP XXL integration already allows external pass2. | HIGH |
| **#229** | Respond: SSL cert failure = VPN/corporate proxy intercepting HTTPS. Try without VPN, or set `SSL_CERT_FILE` env var. Same root cause as #204. | MEDIUM |
| **#228** | Respond: cublas64_12.dll = CUDA toolkit not installed. yangming2027 already helped. First-run hang = model download (expected, needs better progress indication). | LOW |
| **#224** | Deferred response. User's analysis partially incorrect. Will respond with corrected analysis + mention BS-RoFormer. | MEDIUM |

### Issues Needing Action (not just response)

| # | Action Needed | Priority |
|---|--------------|----------|
| **#217** | **ANALYZE INSTALL LOG** from vimbackground. Determine why GUI exe missing after v1.8.8 install. | CRITICAL |
| **#132** | Respond to TinyRick1489's num_ctx question. OllamaManager already supports `--ollama-num-ctx` via CLI. | HIGH |
| **#223** | Continue quality conversation. BYOP XXL shipped. Await test files from weifu8435 for benchmarking. | MEDIUM |

### Candidates for Closing

| # | Condition | Notes |
|---|-----------|-------|
| **#222** | Responded with docs + translation instructions | Can close if no further questions (last activity 2026-03-14) |
| **#211** | urllib3 warning fixed in v1.8.8 | Can close |

### Decisions Needed

| # | Decision | Options |
|---|----------|---------|
| **#128** | Accept hyiip's Gemma 3 contribution? | (a) Accept PR for v1.8.9, (b) Coordinate for v1.9.0, (c) Do it ourselves |
| **#227** | MPS strategy — selective by model? | (a) Allow MPS for kotoba-*, force CPU for whisper-*, (b) Disable MPS entirely, (c) Let user choose |
| **#230** | Standalone merge CLI tool? | (a) Build for v1.8.9, (b) Defer to v1.9.0, (c) Document existing ensemble merge as workaround |

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Status |
|---------|--------|---------|--------|
| **Local LLM translation** | #196 (closed), **#212**, **#214**, **#132**, **#128** | **#128** | #132 partial fix verified. #128 Gemma 3 proposal pending. |
| **MPS/Apple Silicon** | #198 (closed), **#227** | **#227** | Model-dependent MPS behavior confirmed. |
| **Network/SSL/Install** | **#229** (NEW), **#228** (NEW), **#225**, **#222**, **#221**, **#220**, **#218**, **#217**, #204, #210 | **#217** | #217 ESCALATED — install log received. #229 same user as #218. |
| **GPU detection** | **#200**, **#213** | **#200** | `SHIPPED` v1.8.8; #213 deferred |
| **GUI settings** | #96, **#207** | **#96** | `DEFERRED` (v1.9); #207 responded |
| **Whisper quality** | **#230** (NEW), **#224**, **#223**, **#209**, **#215** | **#223** | BYOP XXL shipped. Quality plan drafted. #230 is new merge request. |
| **Kaggle/Colab** | **#231** (NEW), **#132** | **#132** | #231 is new Kaggle error. |
| **Model support** | **#232** (NEW) | **#232** | whisper-ja-anime-v0.1 request. |
| **AMD/Intel GPU** | #142, #114, **#213** | Deferred | v1.9+ |
| **Translation providers** | #71, #43 | Deferred | v1.9+ |
| **i18n** | **#222**, #180, #175 | **#180** | v1.9+ |

---

## Recently Closed Issues

| # | Title | Closed | Resolution |
|---|-------|--------|------------|
| **#201** | Install SSL cert error | 2026-03-15 | Self-resolved: `pip install pip-system-certs` |
| **#198** | MPS beam search + detection | 2026-03-15 | Fixed v1.8.8. MPS working but not accelerating Whisper (benchmark confirmed). |
| **#219** | MossFormer2_SS_16K 3D tensor crash | 2026-03-13 | Fixed v1.8.8 |
| #208 | LLM server AssertionError | 2026-03-09 | Self-resolved |
| #197 | Installation problem v1.8.6 | 2026-03-08 | Closed by user |
| #196 | Local Translation Errors | 2026-03-07 | Partial fix. See #212/#214. |
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
| **2026-03-16** | **5 (#228-#232)** | **0** | **+5** | Install, quality, Kaggle, model request |

**Trend**: Issue rate accelerating (5 new today). Quality concerns (#223, #224, #230) and installation friction (#228, #229) dominate. Growing user base = growing issue volume.

---

## Release & Roadmap Analysis (rev9 — 2026-03-16)

### v1.8.9 — Proposed Scope (Updated)

**Theme: Quality + BYOP + LLM reliability**

| Priority | Item | Issues | Effort | Status |
|----------|------|--------|--------|--------|
| **P0** | ~~BYOP Faster Whisper XXL~~ | #223 | — | **DONE** (committed `aed1af2`) |
| **P0** | Quality parameter fixes (large-v3, float16, beam_size=5) | #223, #224 | ~45 LOC | **DONE** |
| **P0** | MPS selective policy (CPU for whisper-*, MPS for kotoba-*) | #198, #227 | ~20 LOC | **NOT STARTED** |
| **P1** | OllamaManager testing + num_ctx CLI arg | #132, #212, #214 | Testing | **CODED** |
| **P1** | Gemma 3 model upgrade (if hyiip contributes) | #128 | External PR | **DECISION NEEDED** |
| **P2** | Installer investigation (#217 GUI exe missing) | #217 | Investigation | **LOG RECEIVED** |
| **P2** | VAD failover + scene detection fixes (Q4, Q5) | #223 | ~15 LOC | **NOT STARTED** |
| **P3** | CPS threshold fix (Q6) | #223 | 1 LOC | **NOT STARTED** |

**v1.8.9 status**: ~40% complete. BYOP XXL and config cleanup done. Quality parameter fixes, MPS policy, and OllamaManager testing remain.

### v1.9.0 — Proposed Scope

**Theme: Platform expansion + UX**

| Priority | Item | Issues | Notes |
|----------|------|--------|-------|
| **P0** | Ollama full migration (deprecate llama-cpp-python) | #128, #132, #212, #214 | Remove ~1500 LOC fragile code |
| **P0** | Chinese UI (i18n, at least partial) | #175, #180, #222 | Biggest support burden reducer |
| **P1** | Standalone merge CLI tool | #230 | `whisperjav-merge` command |
| **P1** | AMD ROCm support (document + partial) | #142, #114, #213 | FishYu-OWO proved it works |
| **P1** | First-run setup wizard | #217, #220, #221, #225 | Onboarding quality |
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
| **2026-03-16** | **rev9.** 5 new issues (#228-#232). BYOP XXL committed (`aed1af2`). #217 received install log from vimbackground (needs analysis). #227 received MPS benchmark from dadlaugh (model-dependent: kotoba works, whisper-large fails). #223 continued feedback (16 comments). #198 and #201 closed. #230 new feature request for standalone merge tool. #231 Kaggle environment error. #232 whisper-ja-anime model request. Total open: 42. |
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
