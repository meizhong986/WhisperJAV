# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-28 (rev17 — closed 11, responded 7) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **55 open** on GitHub

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
| Total open on GitHub | **55** | Was 53 at rev16. Closed 11, but 2 older issues not previously counted + accurate recount. |
| Closed in rev17 session | 11 | #228, #229, #235, #222, #220, #200, #204, #207, #209, #210, #212 |
| Responded in rev17 session | 7 | #263, #261, #260, #251, #258, #234, #239 |
| **NEEDS RESPONSE (no reply)** | 7 | #246, #247, #248, #250, #254, #259, #262 |
| **NEEDS RESPONSE (replied, awaiting user)** | 8 | #231, #233, #251, #258, #260, #261, #263, #234 |
| **SHIPPED (awaiting user test)** | 4 | #132, #218, #223, #236 |
| Feature requests (open) | 28 | See Cluster J |
| Deferred to v1.9+ | 11 | See v1.9+ Backlog |

---

## New Issues Since Rev15 (2026-03-19 to 2026-03-28)

| # | Date | Reporter | Title | Category | Status |
|---|------|----------|-------|----------|--------|
| **#263** | 03-28 | herlong6529424-dot | GPU没有利用率 (GPU not utilized) | GPU/Install | `NEEDS RESPONSE` — screenshot shows GUI running, GPU at 0%. Likely not clicking Start, or model not downloaded yet. |
| **#262** | 03-27 | teijiIshida | Cohere Transcribe model | Feature | `NEEDS RESPONSE` — requests Cohere's new ASR model support |
| **#261** | 03-27 | henry99a | Network check failed: unknown url type: https | Install | `NEEDS RESPONSE` — post_install fails with SSL. Conda-constructor env missing `certifi`? |
| **#260** | 03-27 | hawai-stack | Uninstall leaves ~6GB files | Install/UX | Community responded (oceanseamountain): AppData residual. `NEEDS RESPONSE` from dev. |
| **#259** | 03-26 | destinyawaits | Local Translation Issues (v1.8.9-hotfix2) | Translation | `NEEDS RESPONSE` — same user as #212. Persistent local LLM issues. |
| **#258** | 03-26 | Uillike | 我遇到的问题 (my problems) | Quality | `NEEDS RESPONSE` — intermittent issues, vague description |
| **#256** | 03-25 | — | 安装过程报错 (install error) | Install | **CLOSED** 03-26 |
| **#255** | 03-25 | cheny7918 | 如何用ollama进行翻译 (how to use ollama) | Translation/Docs | `NEEDS RESPONSE` — zhifanXU17 also has same problem |
| **#254** | 03-25 | zoqapopita93 | ChronosJAV去掉非语言声音 (remove non-speech sounds) | Feature | `NEEDS RESPONSE` — asks about speech enhancement |
| **#253** | 03-25 | KinhoLeung | Colab运行失败 (Colab run failure) | Colab | `NEEDS RESPONSE` — silero trust_repo issue on Colab |
| **#252** | 03-25 | zoqapopita93 | 是否支持多人对话 (multi-speaker support?) | Feature | Community responded (oceanseamountain → #248). `NEEDS RESPONSE` from dev. |
| **#251** | 03-24 | zoqapopita93 | v1.8.9.post2 启动失败 (launch failure) | Install | `NEEDS RESPONSE` — no details provided |
| **#250** | 03-21 | lcpdguy | 模型文件夹在哪？ (where are model folders?) | Docs | `NEEDS RESPONSE` — asks for documentation of model cache locations |
| **#248** | 03-21 | oceanseamountain | Diarization | Feature | `NEEDS RESPONSE` — requests speaker diarization |
| **#247** | 03-21 | zly19540609 | Docker支持计划？ (Docker support?) | Feature | `NEEDS RESPONSE` |
| **#246** | 03-20 | dadlaugh | Serverless GPU pipeline + anime-whisper hallucination | Bug/Report | `NEEDS RESPONSE` — detailed report on anime-whisper hallucination patterns |
| **#245** | 03-20 | — | Kaggle working setup | Docs | **CLOSED** — community guide |
| **#244** | 03-19 | techguru0 | Search for faster whisper xxl | UX | meizhong986 responded "Good point!" `AWAITING CONFIRMATION` |

---

## Releases Since Rev15

### v1.8.9.post2 — RELEASED

| Item | Issues | Fix | Status |
|------|--------|-----|--------|
| CPU float16 regression | #241 | `resolver_v3.py` returns `auto` for non-CUDA; safety net in ASR module | **#241 CLOSED** |
| GUI access violation | #240 | Switch to `private_mode=True`, remove `storage_path` | `SHIPPED` |
| XXL stderr encoding | — | `errors="replace"` in subprocess call | `SHIPPED` |

### v1.8.10 — IN DEVELOPMENT (current branch: dev_v1.8.10)

| Item | Issues | Description | Status |
|------|--------|-------------|--------|
| Config contamination firewall | — | Prevent Silero vad_params leaking to non-Silero backends | Committed |
| Dead code removal | — | Removed unused config paths | Committed |
| GUI defaults fix | — | Ensemble preset defaults not applied on pipeline switch | Committed |
| **Aggressive sensitivity retune** | — | F7 acceptance test: no_speech, compression_ratio, beam_size, hallucination_silence tuned from ground truth | Committed (6 commits) |
| **Whisper param tuner utility** | — | `scripts/whisper_param_tuner.py` — standalone hypothesis testing | Committed |
| **Diagnostic JSON per scene** | — | Full whisper transcribe() results saved alongside scene SRTs | Committed |

---

## Cluster Analysis

### Cluster A: Local LLM Translation (5 issues)

**Issues**: #259, #255, #233, #212, #132
**Theme**: Users struggling with local translation — Ollama setup, llama-cpp-python failures.

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#259** | Local Translation Issues (v1.8.9-hotfix2) | destinyawaits | OPEN | Same user as #212. Persistent issues. | `NEEDS RESPONSE` |
| **#255** | 如何用ollama进行翻译 | cheny7918 | OPEN | How-to question. zhifanXU17 has same issue. | `NEEDS RESPONSE` |
| **#233** | translation error | WillChengCN | OPEN | `n_vocab()` assertion. llama-cpp-python issue. | `NEEDS RESPONSE` — recommend Ollama |
| **#212** | Regex Error v1.8.7 | destinyawaits | OPEN | v1.8.9 comment posted. | `AWAITING CONFIRMATION` |
| **#132** | Local LLM Kaggle | TinyRick1489 | OPEN | v1.8.9.post1 fixed Ollama 404. | `SHIPPED` |

**Recommendation**: High priority to respond. #255 and #259 are active users blocked on Ollama translation. Write a concise Ollama translation guide and link from all 5 issues. Close #233 with Ollama recommendation.

---

### Cluster B: MPS / Apple Silicon (2 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#227** | M1 MAX Transformers hang | dadlaugh | OPEN | Known MPS issue. | `AWAITING CONFIRMATION` |
| **#246** | Serverless GPU + anime-whisper | dadlaugh | OPEN | Same user. Reports hallucination bug. | `NEEDS RESPONSE` |

**Recommendation**: Respond to #246 (valuable technical report). Keep #227 as known issue. MPS deferred to v1.9+.

---

### Cluster C: Network / Installation (14 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#263** | GPU not utilized | herlong6529424-dot | OPEN | Likely user error (didn't click Start?) | `NEEDS RESPONSE` |
| **#261** | Network check: unknown url type | henry99a | OPEN | SSL missing in conda env | `NEEDS RESPONSE` |
| **#260** | Uninstall leaves 6GB | hawai-stack | OPEN | AppData/model cache residual | `NEEDS RESPONSE` |
| **#253** | Colab silero trust_repo | KinhoLeung | OPEN | PyTorch Hub trust issue | `NEEDS RESPONSE` |
| **#251** | post2 launch failure | zoqapopita93 | OPEN | No details | `NEEDS RESPONSE` |
| **#243** | Install verify fails (RTX 3050) | Trenchcrack | OPEN | Path mismatch in conda-constructor | `NEEDS RESPONSE` |
| **#240** | GUI access violation Win11 | m739566004-svg | OPEN | WebView2 failure. Fix shipped in post2. | `SHIPPED` |
| **#229** | INSTALLATION FAILED (SSL) | WillChengCN | OPEN | Self-resolved | `FIX VERIFIED` — **close** |
| **#228** | cublas64_12.dll / first run hang | yhxkry | OPEN | Self-resolved | `FIX VERIFIED` — **close** |
| **#225** | GUI white screen | github3C | OPEN | WebView2 confirmed OK. Exhausted. | `NEEDS FOLLOW-UP` |
| **#222** | 字幕是日语 (how to get Chinese?) | libinghui20001231 | OPEN | User confusion | `AWAITING CONFIRMATION` |
| **#221** | cublas64_12.dll missing | libinghui20001231 | OPEN | Old driver GTX 1650 | `AWAITING LOG` |
| **#220** | Install stalls | libinghui20001231 | OPEN | Network during PyTorch download | `AWAITING LOG` |
| **#218** | cu118 wheel mismatch | WillChengCN | OPEN | uv rejects cu118 llama-cpp wheel | `SHIPPED` |
| **#217** | GUI.exe not found | loveGEM | OPEN | PyTorch download failed | `AWAITING CONFIRMATION` |
| **#210** | DNS error | iop335577 | OPEN | Proxy DNS | `AWAITING CONFIRMATION` |
| **#204** | VPN/v2rayN SSL | yangming2027 | OPEN | HF hub SSL | `AWAITING CONFIRMATION` |

**Recommendation**: Close #228, #229 immediately (self-resolved). Batch-respond to #261, #253, #251, #263 with standard diagnostics. #225 needs escalation or acceptance as "won't fix" (user-specific WebView2 issue).

---

### Cluster D: GPU Detection (3 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#239** | AMD GPU support | bmin1117 | OPEN | Dup of #142/#114 | `NEEDS RESPONSE` — link to existing |
| **#200** | NVML Optimus laptop | Ywocp | OPEN | `--force-cuda` shipped v1.8.8 | `AWAITING CONFIRMATION` |
| **#213** | Intel GPU (XPU) | DDXDB | OPEN | | `DEFERRED` v1.9+ |

---

### Cluster E: GUI / WebUI (5 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#244** | Search for XXL path | techguru0 | OPEN | UX improvement for XXL browse | `RESPONDED` |
| **#240** | GUI access violation | m739566004-svg | OPEN | Fix shipped post2 | `SHIPPED` |
| **#236** | WebUI cache stale | FishYu-OWO | OPEN | Fixed post1 | `SHIPPED` |
| **#235** | ctypes OverflowError | techguru0 | OPEN | Fixed post1 | `FIX VERIFIED` — **close** |
| **#207** | 1.86 settings not saving | q864310563 | OPEN | Dup of #96 | `AWAITING CONFIRMATION` |

---

### Cluster F: Whisper Output Quality (6 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#258** | 我遇到的问题 (intermittent quality) | Uillike | OPEN | Vague | `NEEDS RESPONSE` |
| **#246** | anime-whisper hallucination | dadlaugh | OPEN | Detailed bug report | `NEEDS RESPONSE` |
| **#242** | XXL in Pass 1 | yangming2027 | OPEN | Feature request | `RESPONDED` (7 comments) |
| **#237** | XXL model questions | yangming2027 | OPEN | liugngg asked about post-processing | `NEEDS FOLLOW-UP` |
| **#230** | Subtitle merging module | weifu8435 | OPEN | Feature → v1.9.0 | `RESPONDED` |
| **#209** | Repetition (single long sub) | weifu8435 | OPEN | Shipped v1.8.8+v1.8.9 | `AWAITING CONFIRMATION` |

---

### Cluster G: Kaggle / Colab (3 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#253** | Colab silero trust_repo | KinhoLeung | OPEN | `pass1_speech_segmenter=automatic/silero` fails with trust issue | `NEEDS RESPONSE` |
| **#231** | Kaggle llvmlite version | fzfile | OPEN | Fix known: `pip install -U llvmlite numba` | `NEEDS RESPONSE` |
| **#132** | Local LLM Kaggle | TinyRick1489 | OPEN | Ollama 404 fixed post1 | `SHIPPED` |

---

### Cluster H: Documentation / How-To (4 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#255** | How to use Ollama for translation | cheny7918 | OPEN | Missing docs | `NEEDS RESPONSE` |
| **#250** | Where are model folders? | lcpdguy | OPEN | Missing docs | `NEEDS RESPONSE` |
| **#222** | How to get Chinese subtitles? | libinghui20001231 | OPEN | User confusion | `AWAITING CONFIRMATION` |
| **#234** | CUDA version confusion | techguru0 | OPEN | Misunderstanding | `NEEDS RESPONSE` |

**Recommendation**: These 4 are documentation gaps, not bugs. Write FAQ entries and respond in bulk.

---

### Cluster J: Feature Requests (28 total)

| # | Title | Reporter | Priority | Target |
|---|-------|----------|----------|--------|
| **#262** | Cohere Transcribe model | teijiIshida | LOW | Evaluate |
| **#254** | Remove non-speech sounds (enhancement) | zoqapopita93 | MEDIUM | v1.9 (speech enhancement exists) |
| **#252** | Multi-speaker / diarization | zoqapopita93 | MEDIUM | v1.9+ |
| **#248** | Diarization | oceanseamountain | MEDIUM | v1.9+ (dup of #252) |
| **#247** | Docker support | zly19540609 | LOW | Backlog |
| **#242** | XXL in Pass 1 | yangming2027 | MEDIUM | v1.9 |
| **#239** | AMD GPU | bmin1117 | MEDIUM | v1.9+ (dup #142/#114) |
| **#232** | whisper-ja-anime model | mustssr | LOW | Evaluate |
| **#230** | Standalone merge module | weifu8435 | HIGH | v1.9.0 |
| **#224** | Vocal separation | yangming2027 | MEDIUM | v1.9 |
| **#213** | Intel GPU (XPU) | DDXDB | LOW | v1.9+ |
| **#206** | Grey out incompatible options | techguru0 | MEDIUM | v1.9+ |
| **#205** | VibeVoice ASR | kylesskim-sys | LOW | v1.9+ |
| **#181** | Frameless window | QQ804218 | LOW | Backlog |
| **#180** | Multi-language GUI (i18n) | QQ804218 | HIGH | v1.9.0 |
| **#175** | Chinese GUI | yangming2027 | HIGH | v1.9.0 (subset #180) |
| **#164** | MPEG-TS + Drive | hosmallming | LOW | Backlog |
| **#142** | AMD Radeon ROCm | MatthaisUK | MEDIUM | v1.9+ |
| **#128** | Gemma 3 models | hyiip | HIGH | v1.9.0 |
| **#126** | Recursive directory | jl6564 | LOW | Backlog |
| **#114** | DirectML | SingingDalong | MEDIUM | v1.9+ |
| **#99** | 4GB VRAM guidance | hosmallming | LOW | Backlog |
| **#96** | Full settings persistence | sky9639 | MEDIUM | v1.9.0 |
| **#71** | Google Translate (free) | x8086 | LOW | Backlog |
| **#59** | Feature plans (meta) | meizhong986 | — | Keep open |
| **#51** | Batch translate wildcard | lingyunlxh | LOW | Backlog |
| **#49** | Output to source folder | meizhong986 | LOW | Backlog |
| **#44** | GUI drag-drop fix | lingyunlxh | LOW | Backlog |
| **#43** | DeepL provider | teijiIshida | LOW | v1.9+ |
| **#33** | Linux pyaudio docs | org0ne | LOW | Backlog |

---

## F2: Issue Group Analysis

### Group 1: URGENT — Needs Response (14 issues, 0 comments from dev)

These issues have **zero developer response**. Users are waiting.

**Batch 1 — Simple responses (can do in 30 min):**
| # | Response Type | Action |
|---|---|---|
| #263 | Diagnosis | Ask: did you click Start? Check if model downloaded. |
| #261 | Known issue | SSL missing in conda env. Provide fix command. |
| #260 | Guidance | Explain AppData/model cache locations for manual cleanup. |
| #251 | Ask for info | Request error logs/screenshots. |
| #258 | Ask for info | Too vague. Request logs and reproduction steps. |
| #234 | Correction | Politely explain CUDA toolkit vs compute capability. |
| #239 | Dup link | Link to #142/#114. AMD tracked for v1.9+. |

**Batch 2 — Needs investigation:**
| # | Response Type | Action |
|---|---|---|
| #253 | Bug fix/workaround | Silero trust_repo on Colab. Provide `torch.hub.set_dir()` workaround. |
| #246 | Technical discussion | Valuable anime-whisper hallucination report. Engage. |
| #255 | Documentation | Write Ollama translation guide. |
| #250 | Documentation | Document model cache paths. |
| #247 | Decision | Docker: yes/no/when? |
| #248/#252 | Roadmap | Diarization: acknowledge as v1.9+ feature. |
| #262 | Evaluate | Cohere Transcribe: research feasibility. |

### Group 2: Stale — Safe to Close — **DONE** (rev17)

All 6 closed: #228, #229, #235, #222, #220 (+ #238 was already closed).

### Group 3: Stale AWAITING — **DONE** (rev17)

All 6 closed for inactivity with "feel free to reopen": #200, #204, #207, #209, #210, #212.

### Group 4: Feature Request Interdependencies

```
Diarization cluster:     #248 ↔ #252 (both ask for multi-speaker)
AMD/Intel GPU cluster:   #114 ↔ #142 ↔ #239 ↔ #213 (all non-NVIDIA GPU)
Translation cluster:     #255 ↔ #259 ↔ #233 ↔ #212 ↔ #132 (all local LLM)
i18n cluster:            #175 ↔ #180 ↔ #222 (all Chinese/multi-language)
Speech quality cluster:  #254 ↔ #224 (both speech enhancement)
XXL cluster:             #242 ↔ #237 ↔ #223 ↔ #244 (all XXL-related)
```

---

## F3: Recommended Steps & Roadmap

### Immediate Actions (this week)

| Action | Issues | Status |
|---|---|---|
| ~~Close 6 stale/verified issues~~ | #228, #229, #235, #238, #222, #220 | **DONE** (rev17) |
| ~~Batch respond to 7 simple issues~~ | #263, #261, #260, #251, #258, #234, #239 | **DONE** (rev17) |
| **Write Ollama translation FAQ** | #255, #259, #233 | TODO |
| **Write model cache paths doc** | #250 | TODO |
| ~~Close 6 stale AWAITING issues~~ | #200, #204, #207, #209, #210, #212 | **DONE** (rev17) |

### v1.8.10 Release Scope (current dev branch)

| Item | Issues | Status |
|---|---|---|
| Aggressive sensitivity retune (F7 validated) | — | Committed (6 commits) |
| Diagnostic JSON per scene | — | Committed |
| Whisper param tuner utility | — | Committed |
| Config contamination firewall | — | Committed |
| GUI ensemble preset fix | — | Committed |
| **Respond to #246 anime-whisper hallucination** | #246 | TODO — valuable bug report |
| **Respond to #253 Colab silero trust** | #253 | TODO — provide workaround |
| **Fix compression_ratio_threshold** | — | Committed (2.2→2.6 tuner-validated) |

### v1.9.0 Roadmap (proposed)

**Theme: Platform + Translation + UX**

| Priority | Item | Issues | Est. Effort |
|---|---|---|---|
| **P0** | Ollama full migration + GUI wiring | #132, #212, #233, #255, #259 | Large |
| **P0** | Chinese UI (partial i18n) | #175, #180, #222 | Medium |
| **P1** | Diarization (speaker ID) | #248, #252 | Large (new feature) |
| **P1** | Standalone merge CLI | #230 | Medium |
| **P1** | AMD ROCm support (document + test) | #142, #114, #239 | Medium |
| **P1** | XXL in Pass 1 | #242 | Medium |
| **P2** | GUI settings persistence | #96, #207 | Medium |
| **P2** | Vocal separation investigation | #224, #254 | Medium |
| **P2** | MPS selective policy | #227 | Small |
| **P2** | Uninstall cleanup | #260 | Small |
| **P3** | Docker support | #247 | Medium |
| **P3** | Gemma 3 model configs | #128 | Small (contributor PR) |
| **P3** | whisper-ja-anime model | #232 | Small |
| **P3** | Grey out incompatible options | #206 | Small |

### Backlog (no target)

#99, #71, #59, #51, #49, #44, #43, #33, #126, #164, #181, #205, #262

---

## Recently Closed Issues

| # | Title | Closed | Resolution |
|---|-------|--------|------------|
| **#256** | 安装过程报错 | 2026-03-26 | Closed by user |
| **#245** | Kaggle working setup | 2026-03-20 | Community guide |
| **#241** | float16 CPU crash (REGRESSION) | 2026-03-19 | Fixed in v1.8.9.post2 |
| **#238** | Portuguese translation | 2026-03-19 | Shipped in v1.8.9.post1 |
| **#219** | MossFormer2 3D tensor crash | 2026-03-13 | Fixed in v1.8.8 |
| **#214** | localLLM fail | 2026-03-18 | Closed |
| **#211** | 启动报错 | 2026-03-13 | Fixed in v1.8.8 |

---

## Issue Velocity Trend

| Period | New Issues | Closed | Net | Notes |
|--------|-----------|--------|-----|-------|
| 2026-03-08 to 2026-03-10 | 6 | 5 | +1 | v1.8.7 release cycle |
| 2026-03-11 to 2026-03-13 | 8 | 3 | +5 | v1.8.8 beta + stable |
| 2026-03-14 to 2026-03-16 | 11 | 3 | +8 | Post-release influx |
| 2026-03-18 | 6 | 1 | +5 | v1.8.9 release day |
| **2026-03-19** | 5 | 0 | +5 | #241 REGRESSION, post1 released |
| **2026-03-20 to 2026-03-28** | 15 | 7 | +8 | Steady influx, post2 released |

**Trend**: 53 open (was 33 on 2026-03-08). Net +20 in 20 days. The close rate is improving (7 closed this week) but needs to accelerate. 6 verified-fixed issues are sitting open and should be closed immediately.

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Action |
|---------|--------|---------|--------|
| **Local LLM** | #259, #255, #233, #212, #132 | #132 | Ollama guide resolves most |
| **Diarization** | #248, #252 | #248 | Merge. Roadmap for v1.9. |
| **AMD/Intel GPU** | #239, #142, #114, #213 | #142 | Link all. v1.9+. |
| **i18n** | #175, #180, #222 | #180 | v1.9.0 scope. |
| **Speech enhancement** | #254, #224 | #224 | Document existing backends. |
| **XXL** | #242, #237, #223, #244 | #242 | v1.9.0 scope. |
| **Install/Network** | #261, #253, #251, #243, #240, #225, #218, #217, #210, #204 | — | Individual fixes. |
| **GUI** | #244, #240, #236, #235, #207, #96 | #96 | #240 fixed, #235/#236 verified. |

---

## Changelog

| Date | Changes |
|------|---------|
| **2026-03-28** | **rev17.** Closed 11 issues: #228, #229, #235, #222, #220 (stale/verified), #200, #204, #207, #209, #210, #212 (inactivity). Responded to 7: #263, #261, #260, #251, #258, #234, #239. Open count: 55. |
| **2026-03-28** | **rev16.** Full refresh from GitHub. Added F2 (group analysis) and F3 (roadmap). |
| 2026-03-19 | **rev15.** 3 fixes coded for post2: #241/#240 + XXL stderr encoding. |
| 2026-03-19 | **rev14.** v1.8.9.post2 fixes coded. |
| 2026-03-19 | **rev13.** v1.8.9.post1 RELEASED. |
| 2026-03-18 | **rev12.** v1.8.9 RELEASED. |
| 2026-03-17 | **rev10.** Pre-release review. |
| 2026-03-16 | **rev9.** 5 new issues. BYOP XXL committed. |
| 2026-03-15 | **rev8.** MPS benchmark: 6x slower. |
| 2026-03-14 | **rev7.** v1.8.8 RELEASED. |
| 2026-03-13 | rev6. All Track A/B/C code complete. |
| 2026-03-12 | rev5. v1.8.8b1 pre-release. |
| 2026-03-11 | rev4. v1.8.7 RELEASED. |

---
