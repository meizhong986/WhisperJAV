# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-30 (rev19) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **54 open** on GitHub

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
| Total open on GitHub | **54** | Was 55 at rev18. #253 closed (Silero trust_repo shipped). |
| **v1.8.10 RELEASED** | — | 39 commits. Aggressive retune, Ollama GUI, bug fixes. |
| **NEW since rev18** | 1 | #264 (model download location) |
| **NEEDS RESPONSE (no reply)** | 8 | #246, #247, #248, #250, #254, #259, #262, #264 |
| **USER REPLIED (needs follow-up)** | 5 | #217, #231, #243, #255, #263 |
| **AWAITING USER REPLY** | 5 | #234, #244, #251, #258, #261 |
| **SHIPPED (awaiting test)** | 4 | #132, #236, #240, #253 |
| Feature requests (open) | 28 | See Feature Requests section |

---

## v1.8.10 — RELEASED (2026-03-30)

### What shipped

| Category | Item | Issues |
|----------|------|--------|
| **Quality** | Aggressive sensitivity retune (F7 ground truth validated, 76.5% → 92.6%) | — |
| **Quality** | compression_ratio=2.6, condition_on_previous_text=True (tuner-validated) | — |
| **Feature** | Ollama first-class GUI integration (provider, onboarding, model download, VRAM cleanup) | #132, #128 |
| **Feature** | Enhance-for-VAD checkbox in ensemble UI, all pipelines | — |
| **Tooling** | Whisper param tuner utility (`scripts/whisper_param_tuner.py`) | — |
| **Tooling** | Diagnostic JSON per scene (full transcribe() results) | — |
| **Bug fix** | Config contamination firewall, dead code removal | — |
| **Bug fix** | GUI ensemble presets not applied on pipeline switch | — |
| **Bug fix** | XXL exe path not restored on app restart | — |
| **Bug fix** | XXL --model moved to user-editable Extra Args | — |
| **Bug fix** | Silero VAD crashes in Colab/Kaggle (trust_repo) | #253 |
| **Bug fix** | Translation diagnostic hardening (12+ fixes) | — |
| **Bug fix** | Colab/Kaggle notebook fixes (llvmlite, trust_repo) | #253, #231 |

---

## v1.8.10 Hotfix Candidates

Issues discovered during or after v1.8.10 release that may warrant a post-release patch.

| Item | Source | Priority | Scope |
|---|---|---|---|
| Colab install: remove venv, reuse system torch | Colab test log analysis | **HIGH** | `install_colab.sh` + notebooks — coded, needs testing |
| Colab install: llama-cpp wheel filename bug (uv rejects generic name) | Colab test log F3 | MEDIUM | `install_colab.sh` — coded |
| Progress counter `[2/1]` — denominator wrong when directory has subdirs | Colab test log F2 | LOW | `main.py` or `unified_progress.py` |
| Notebook file scan doesn't match WhisperJAV's recursive scan | Colab test log F1 | LOW | Notebook cell — cosmetic mismatch |
| #263 follow-up — user shared console log, needs diagnosis | GitHub | MEDIUM | Response needed |
| #255 — 3 users asking how to use Ollama for translation | GitHub | MEDIUM | FAQ or response — v1.8.10 ships the feature |
| #264 — model download location request (new issue) | GitHub | LOW | Response needed |
| #217 — vimbackground still can't install (PyTorch download fails in China) | GitHub | LOW | Network/region issue, hard to fix |

**Recommendation**: Ship hotfix (`v1.8.10.post1`) with the Colab install fix after Colab testing confirms it works. The progress counter and notebook scan mismatch are cosmetic — defer.

---

## Open Issues by Status

### Bugs / Active Issues (14)

| # | Title | Reporter | Status | Notes |
|---|-------|----------|--------|-------|
| **#264** | 模型默认下载位置 + 系列问题 | yuliQAQ | `NEEDS RESPONSE` | NEW. Chinese. Multiple requests: model download path, other. |
| **#263** | GPU not utilized | herlong6529424-dot | `NEEDS FOLLOW-UP` | User replied with console log showing successful processing. GPU may be working but Task Manager misleading. Need to read log carefully. |
| **#261** | Network check: unknown url type https | henry99a | `AWAITING REPLY` | Responded with SSL fix. |
| **#260** | Uninstall leaves 6GB | hawai-stack | `AWAITING REPLY` | Responded with cache paths. |
| **#259** | Local Translation Issues (hotfix2) | destinyawaits | `NEEDS RESPONSE` | Same user as closed #212. Likely needs Ollama FAQ (v1.8.10 ships Ollama GUI). |
| **#258** | 我遇到的问题 (vague) | Uillike | `AWAITING REPLY` | Asked for logs. |
| **#255** | 如何用ollama进行翻译 | cheny7918 | `NEEDS FOLLOW-UP` | 3 users asking. v1.8.10 ships Ollama GUI — respond with upgrade instructions. |
| **#253** | Colab silero trust_repo | KinhoLeung | `SHIPPED` v1.8.10 | **CLOSED** on GitHub. trust_repo=True + env var. |
| **#251** | post2 launch failure | zoqapopita93 | `AWAITING REPLY` | Asked for details. |
| **#246** | anime-whisper hallucination bug | dadlaugh | `NEEDS RESPONSE` | Valuable technical report. Serverless GPU pipeline + hallucination in anime-whisper model. |
| **#243** | Install verify fails (RTX 3050) | Trenchcrack | `NEEDS FOLLOW-UP` | Community member (JiwaniZakir) replied asking for full log. |
| **#240** | GUI access violation Win11 | m739566004-svg | `SHIPPED` post2 | private_mode=True fix shipped. |
| **#237** | XXL model questions | yangming2027 | `NEEDS FOLLOW-UP` | liugngg question about post-processing. Active community discussion. |
| **#234** | CUDA version confusion | techguru0 | `AWAITING REPLY` | Corrected. |
| **#233** | translation error (local LLM) | WillChengCN | `NEEDS RESPONSE` | Recommend Ollama + v1.8.10 upgrade. |
| **#231** | Kaggle notebook error | fzfile | `NEEDS FOLLOW-UP` | We responded with llvmlite fix. User confirmed fix but hit translation error. New user (Liiesl) asking for update. |
| **#225** | GUI white screen | github3C | `STALE` | WebView2 confirmed OK. Exhausted hypotheses. |
| **#217** | GUI.exe not found | loveGEM | `NEEDS FOLLOW-UP` | vimbackground replied again 03-30. Root cause: PyTorch download fails in China (network). Suggested mirror/offline install. |

### Stale / Low Activity (5)

| # | Title | Last Activity | Recommendation |
|---|-------|---------------|----------------|
| #244 | search for XXL | 03-19 | We responded. Close for inactivity if no reply. |
| #236 | WebUI cache stale | 03-19 | `SHIPPED` post1. Close. |
| #232 | whisper-ja-anime model | 03-16 | We responded. Evaluate model. |
| #221 | cublas64_12.dll missing | 03-14 | We responded. Close for inactivity. |
| #218 | cu118 wheel mismatch | 03-14 | `SHIPPED` post1. Close. |

### Feature Requests (28)

| # | Title | Priority | Target |
|---|-------|----------|--------|
| **#264** | Model download location customization | LOW | Evaluate |
| **#262** | Cohere Transcribe model | LOW | Evaluate |
| **#254** | Remove non-speech sounds | MEDIUM | v1.9 |
| **#252** | Multi-speaker / diarization | MEDIUM | v1.9+ |
| **#250** | Model folder documentation | LOW | v1.8.10 hotfix (docs) |
| **#248** | Diarization | MEDIUM | v1.9+ (dup #252) |
| **#247** | Docker support | LOW | Backlog |
| **#242** | XXL in Pass 1 | MEDIUM | v1.9 |
| **#239** | AMD GPU | MEDIUM | v1.9+ |
| **#232** | whisper-ja-anime model | LOW | Evaluate |
| **#230** | Standalone merge module | HIGH | v1.9.0 |
| **#224** | Vocal separation | MEDIUM | v1.9 |
| **#215** | Qwen3-ASR quality | LOW | Expected behavior |
| **#213** | Intel GPU (XPU) | LOW | v1.9+ |
| **#206** | Grey out incompatible options | MEDIUM | v1.9 |
| **#205** | VibeVoice ASR | LOW | v1.9+ |
| **#181** | Frameless window | LOW | Backlog |
| **#180** | Multi-language GUI (i18n) | HIGH | v1.9.0 |
| **#175** | Chinese GUI | HIGH | v1.9.0 |
| **#164** | MPEG-TS + Drive | LOW | Backlog |
| **#142** | AMD Radeon ROCm | MEDIUM | v1.9+ |
| **#128** | Gemma 3 models | HIGH | v1.9.0 |
| **#126** | Recursive directory | LOW | Backlog |
| **#114** | DirectML | MEDIUM | v1.9+ |
| **#99** | 4GB VRAM guidance | LOW | Backlog |
| **#96** | Settings persistence | MEDIUM | v1.9.0 |
| **#71** | Google Translate (free) | LOW | Backlog |
| **#59** | Feature plans (meta) | — | Keep open |
| **#51** | Batch translate | LOW | Backlog |
| **#49** | Output to source folder | LOW | Backlog |
| **#44** | GUI drag-drop | LOW | Backlog |
| **#43** | DeepL provider | LOW | v1.9+ |
| **#33** | Linux pyaudio docs | LOW | Backlog |

---

## Immediate Actions

| Action | Issues | Status |
|---|---|---|
| ~~v1.8.10 released~~ | All fix-coded items | **DONE** (rev19) |
| ~~Close #253~~ | Silero trust_repo | **DONE** (rev19) |
| **Respond to #255** (Ollama FAQ — 3 users waiting) | #255 | TODO |
| **Respond to #264** (model download location) | #264 | TODO |
| **Follow up #263** (read user's console log carefully) | #263 | TODO |
| **Follow up #231** (Kaggle translation error) | #231 | TODO |
| **Close 3 stale issues** | #221, #218, #236 | TODO |
| **Respond to #246** (anime-whisper hallucination) | #246 | TODO |
| **Respond to #259** (local translation → upgrade to v1.8.10 Ollama) | #259 | TODO |
| **Test Colab install fix** and ship v1.8.10.post1 | — | TODO |

---

## Release Roadmap

### v1.8.10.post1 — Hotfix (Colab Install)

**Theme: Colab Installation Fix**

Fix is coded on dev branch, needs Colab testing before release.

| Item | Risk |
|---|---|
| Remove venv, reuse Colab's system torch (~50s faster, ~2GB less bandwidth) | LOW — tested logic, needs Colab verification |
| Fix llama-cpp wheel filename bug (uv rejects generic name) | LOW — filename fix only |
| Notebook updates (remove venv references) | LOW — config changes only |

**Recommendation**: Test on Colab, then release as v1.8.10.post1. No code changes to core pipeline.

---

### v1.8.11 — Optional Maintenance Release

Only needed if post-v1.8.10 bug reports accumulate. Currently not planned.

| Potential item | Trigger |
|---|---|
| #263 GPU utilization fix | If diagnosis reveals a real bug |
| Progress counter `[2/1]` fix | If more users report |
| #217 China PyTorch download mirror | If we add mirror support |
| Customize Parameters UI fixes (F1-F10 from audit) | If user demand emerges |

**Recommendation**: Skip v1.8.11 unless a blocking bug surfaces. Move to v1.9.0 scope.

---

### v1.9.0 — Next Major Release

**Theme: Platform Expansion + Translation Overhaul + UX**

| Priority | Item | Issues | Effort | Notes |
|---|---|---|---|---|
| **P0** | Ollama full migration (deprecate llama-cpp-python) | #132, #233, #255, #259 | Large | Remove ~1500 LOC fragile code. llama-cpp already broken on Colab. |
| **P0** | Chinese UI (partial i18n) | #175, #180 | Medium | Biggest support burden reducer. 40%+ of issues are in Chinese. |
| **P1** | Speaker diarization | #248, #252 | Large | New capability. pyannote-audio or similar. |
| **P1** | Standalone merge CLI | #230 | Medium | `whisperjav-merge` command. 7 comments, active demand. |
| **P1** | XXL in Pass 1 | #242 | Medium | Needs forced alignment since XXL has no timestamps. 7 comments. |
| **P1** | AMD ROCm support | #142, #114, #239 | Medium | Document + test. FishYu-OWO proved it works. |
| **P1** | Full dual-track enhance-for-VAD | — | Medium | ASR module needs separate VAD/ASR audio paths for balanced/fidelity. |
| **P2** | GUI settings persistence | #96 | Medium | Long-standing request. |
| **P2** | Vocal separation investigation | #224, #254 | Medium | BS-RoFormer or UVR integration. |
| **P2** | Grey out incompatible options | #206 | Small | Prevent invalid GUI combinations. |
| **P2** | Uninstall cleanup tool | #260 | Small | Script to find and remove cached models. |
| **P2** | Model cache paths documentation | #250 | Small | FAQ / docs. |
| **P2** | MPS selective policy | #227 | Small | Force CPU for whisper, allow MPS for kotoba. |
| **P3** | Docker support | #247 | Medium | Dockerfile + compose. |
| **P3** | Gemma 3 model configs | #128 | Small | Contributor PR from hyiip. |
| **P3** | whisper-ja-anime model | #232 | Small | If standard HF format. |

**Estimated timeline**: 6-8 weeks for P0+P1 items.

---

### v2.0 — Strategic Vision

**Theme: Architecture + Scale + Ecosystem**

| Area | Item | Why |
|---|---|---|
| **Architecture** | Plugin system for ASR backends | Hardcoded pipeline classes → community contributions without core changes |
| **Architecture** | Separate installer from runtime | Entangled today. Clean separation enables Docker, cloud, headless |
| **Architecture** | Web-based UI (replace pywebview) | PyWebView has persistent issues (#225, #240). Proper web UI eliminates platform bugs |
| **Scale** | Batch processing dashboard | Queue, progress, ETA for large libraries |
| **Scale** | GPU memory management overhaul | Proper memory budget system for multi-model workflows |
| **Ecosystem** | Public API / SDK | REST API for remote processing |
| **Quality** | Ground truth test framework | Formalize F7 pattern. CI/CD with regression detection |
| **Platform** | Linux native installer | AppImage or deb package |
| **Platform** | China mirror support | #217 — PyTorch download fails behind GFW. Mirror URLs or offline bundle |

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Action |
|---------|--------|---------|--------|
| **Local LLM / Ollama** | #259, #255, #233, #132 | #132 | v1.8.10 ships Ollama GUI. Respond with upgrade. |
| **Diarization** | #248, #252 | #248 | Merge. v1.9 roadmap. |
| **AMD/Intel GPU** | #239, #142, #114, #213 | #142 | v1.9+. Link all. |
| **i18n** | #175, #180 | #180 | v1.9.0 P0. |
| **Speech enhancement** | #254, #224 | #224 | v1.9. |
| **XXL** | #242, #237, #223, #244 | #242 | v1.9. |
| **Install/Network** | #261, #253, #251, #243, #240, #225, #217 | — | Individual fixes. |
| **Model management** | #264, #250 | #250 | Docs / FAQ. |

---

## Changelog

| Date | Changes |
|------|---------|
| **2026-03-30** | **rev19.** v1.8.10 released. 54 open. #253 closed. New: #264. Refreshed all issue statuses from GitHub. Added v1.8.10.post1 hotfix plan (Colab install fix). Updated roadmap. |
| 2026-03-29 | **rev18.** Refreshed from GitHub. 55 open. Session work: aggressive retune, Ollama GUI, bug fixes. Added F3 roadmap. |
| 2026-03-28 | **rev17.** Closed 11. Responded 7. |
| 2026-03-28 | **rev16.** Full refresh. |
| 2026-03-19 | **rev15.** 3 fixes coded for post2. |

---
