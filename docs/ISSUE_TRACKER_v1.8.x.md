# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-04-21 (rev41) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **65 open** on GitHub

---

## Status Legend

| Label | Meaning |
|-------|---------|
| `SHIPPED` | Fix released in a stable version. Waiting for user to test/confirm. |
| `FIX CODED` | Fix committed but not yet released. |
| `AWAITING REPLY` | Response given. Waiting for user reply. |
| `NEEDS RESPONSE` | Issue has no response or needs a follow-up reply. |
| `NEEDS FOLLOW-UP` | User replied after our response — needs another look. |
| `DEFERRED` | Moved to a future release. |

---

## Quick Stats

| Category | Count | Notes |
|----------|------:|-------|
| Total open on GitHub | **65** | +1 since rev40: **#298 NEW** (settings persistence, duplicate of #96). |
| **v1.8.11 RELEASED** | — | **2026-04-20**. Tagged, merged to main, pushed. GitHub Release published with `.exe` installer. |
| **v1.8.10.post3 RELEASED** | — | 2026-04-09. Stable (prior release). |
| **NEEDS RESPONSE** | **0** | #298 responded 04-21 (pointed to #96, asked if OK to close). |
| **NEEDS FOLLOW-UP** | **0** | #290 responded 04-21 (ClearVoice to be masked out in v1.8.12, deep dive deferred to 2.x). |
| **AWAITING REPLY** | 23 | Normal turnaround 1-3 days. |
| **SHIPPED in v1.8.11, user notified** | **3** | #287, #271, #291 — reporters notified 2026-04-21 with upgrade commands. |
| **Closed since rev38** | 1 | #295 (closed by reporter). |
| **SHIPPED in post3** | 6 | #272, #269, #265 (filter), #259, #267 (partial), #231. |
| Feature requests (open) | 35 | +1: #298 (settings persistence, ≈#96). |

---

## v1.8.11 — RELEASED (2026-04-20)

Tagged and merged to main on 2026-04-20 (merge commit `97ad9a4`). GitHub Release published with `.exe` installer built from `dev-1.8.11`. Release notes in commit `e4f9b52`.

**User-facing fixes** (from release notes):
- **Subtitle sanitization** (#287): Hardening pass targeting residual `!!` / symbol-only output and SDH-style tokens that escaped previous rules.
- **Ollama** (#271): Curated model list updated — replaced `Shisa-v2.1-Qwen3-8B` (thinking model breaking PySubtrans format) with `gemma3:12b`, `qwen2.5:7b-instruct`, `qwen2.5-abliterate`, `dolphin-llama3` (all instruct-only). Added `--ollama-max-tokens` CLI flag for output-length tuning.
- **Colab** (#291): `install_colab.sh` now pulls in `[llm]` extra so `--provider local` works (starlette-context, sse-starlette, uvicorn, fastapi, pydantic-settings).

**Developer-level changes** (hidden from release notes by user preference):
- `${N:0:M}` slice syntax in `hallucination_remover._apply_regex_replacement_safe` (root cause of #287).
- Symbol-only subtitle drop as defense-in-depth.
- CPS character counting excludes `\n`.
- New SDH patterns: tortoise shell, music symbols, HI keywords.
- Round-2 emoji drop + normalized match + additional escapees.

**Commits merged into main** via `97ad9a4` (16 commits from `dev-1.8.11`): `275adb5`, `eb05858`, `b3499a2`, `a737bdf`, `e224333`, `970af55`, `93bb71a`, `fc5353c`, `3508005`, `783b218`, `96ea413`, `bb2e020`, `10f5b13`, `e4f9b52`, `e459468`, `793d305`.

**Post-release operational steps:**
- ✅ Installer `.exe` built from `dev-1.8.11` and uploaded to GitHub Release.
- ✅ Release notes published on GitHub Release page.
- ✅ Tag `v1.8.11` moved to merge commit (was initially misplaced on old main; force-pushed correction).
- ⏳ **Pending**: Sync `regexp_v09.json` to the gist referenced by `HallucinationConstants.EXACT_LIST_URL` — Fix 5a's new SDH patterns only reach offline users (bundled fallback) otherwise.

---

## v1.8.10.post3 — RELEASED (2026-04-09)

See `installer/RELEASE_NOTES_v1.8.10.post3.md` for full details.

**Key changes**: 3 crash fixes (device="auto", length_penalty, single-segment), full ASR/VAD retune (hf1→hf3), Silero v3.1/v4.0 anti mega-group presets, post-processing overhaul (regex, punct, dedup), GUI customize modal audit (11 fixes), TEN VAD backend hardening, post-model logprob gate toggle, Silero v3.1 as default segmenter for balanced/fidelity, resolver_v3 versioned Silero name support, numba cache redirect, XXL model flag fix, hallucination filter bundling, translation overwrite fix, Colab numpy2 migration.

**Default changes users will notice**:
- Default speech segmenter for balanced/fidelity: silero (v4.0) → **Silero v3.1**
- Default sensitivity in GUI ensemble: aggressive (unchanged but now with retuned params)
- ASR parameters significantly retuned across all 3 sensitivity levels

---

## Open Issues by Status

### Bugs / Active Issues

| # | Title | Reporter | Status | Notes |
|---|-------|----------|--------|-------|
| **#297** | Blank subtitle file after transcription completed | teijiIshida | `AWAITING REPLY` | **NEW 04-19.** Windows 11 25H2, GUI v1.8.10-hotfix3. Multiple files in queue transcribed, only 1 produced subtitles, rest are 0kb. User attached `whisperjav.logs.txt`. **Responded 04-21**: asked for debug-enabled log + batch-vs-separate-runs clarification + segmenter swap (log shows silero-v4.0, default is v3.1). |
| **#296** | Model outputting "thinking or reasoning in the translation" | triatomic | `AWAITING REPLY` | **NEW 04-19.** User on `gemma3-12b` seeing chain-of-thought text in SRT output. Classic thinking-model pattern (same root cause as #271). **Responded 04-21**: explained gemma3 can still mix reasoning; recommended `translategemma:12b` or `qwen2.5:7b-instruct`; noted v1.8.11 curated list fix. |
| **#294** | Doesn't translate the whole file | CaliburnKoko | `AWAITING REPLY` | **NEW 04-17.** Says v1.8.10 hotfix truncates SRT translation to first few minutes. **Responded 04-21**: 4-point info request (provider, length, console, model) + missing-vs-untranslated SRT check. |
| **#291** | Google Colab Step 3 Translation — `ModuleNotFoundError: starlette_context` | ktrankc | `SHIPPED` v1.8.11 | **FIX SHIPPED** (`b3499a2`). User notified 04-21 with same-session pip workaround + recommended migration to `--provider ollama`. Also noted Gemini 2.0 Flash retirement. |
| **#290** | Google Colab Pass 2 Error — "Killed" after MossFormer2 load | ktrankc | `AWAITING REPLY` | **ESCALATED 04-21 02:47**: user tested on **V100 32GB (OOM), RTX 8000 48GB (OOM), RTX PRO 6000 WS 96GB (OOM — PyTorch holding 94GB of 96GB available)**. ClearVoice-FRCRN_SE_16K gives separate numpy error: `Cannot interpret '113705354' as a data type`. OOM at 96GB rules out "heavy model"; points to tensor-retention / numpy-ModelScope interop issue. **Responded 04-21** ([comment-4287955148](https://github.com/meizhong986/WhisperJAV/issues/290#issuecomment-4287955148)): deep dive deferred to 2.x, ClearVoice to be **masked out in v1.8.12**, user invited to share any root-cause findings. |
| **#289** | PyTorch CUDA 12.8 install timeouts | coco7887 | `AWAITING REPLY` | Responded 04-17 with bilingual China-network template (community ask + mirror roadmap + whl workaround). |
| **#287** | All subtitles are "!!" with latest version | zoqapopita93 | `SHIPPED` v1.8.11 | **FIX SHIPPED**: full sanitizer fix chain. User notified 04-21 with root-cause explanation + upgrade paths (installer `.exe` + `Scripts\pip.exe --no-deps`). |
| **#286** | CUDA kernel error on GTX 1050 Ti | techguru0 | `AWAITING REPLY` | User replied 04-14 "will try it" re: PyTorch 2.5.1 downgrade. Waiting on result. |
| **#284** | v1.8.10 install stuck during PyTorch phase | qq73-maker | `AWAITING REPLY` | Follow-up 04-17: posted bilingual China-network template (community ask + mirror roadmap). |
| **#282** | Why does Ollama need GitHub connection? | KenZP12 | `AWAITING REPLY` | Responded 04-09: gist fetch for instructions, bundled fallback works offline. |
| **#280** | Qwen3-ASR TypeError: `check_model_inputs()` | zoqapopita93 | `AWAITING REPLY` | Responded 04-09: upstream `transformers` mismatch, suggested `pip install --no-deps transformers==4.49.0`. |
| **#274** | Pipeline aggregation mode question | cuixiaopi | `AWAITING REPLY` | Responded 04-09: explained 2-pass ensemble mode. |
| **#271** | Ollama translation model issues | justantopair-ai | `SHIPPED` v1.8.11 | **FIX SHIPPED**: curated model list fixed (instruct-only), `--ollama-max-tokens` CLI flag added (`275adb5`). @justantopair-ai and @TinyRick1489 notified 04-21 with upgrade commands + max-tokens usage example. |
| **#268** | Thai + Korean translation targets | yedkung69-ctrl | `AWAITING REPLY` | Responded 04-11: GUI gap noted for v1.9.0. |
| **#265** | Hallucination + post-processing suggestions | yangming2027 | `AWAITING REPLY` | Responded 04-09: answered @zoqapopita93, noted post3 improvements. |
| **#263** | GPU not utilized / stuck at VAD | herlong6529424-dot | `AWAITING REPLY` | Follow-up 04-17: bilingual China-network template. |
| **#261** | Network check: unknown url type https | henry99a | `AWAITING REPLY` | Follow-up 04-17: bilingual China-network template (earlier user found cmd workaround). |
| **#260** | Uninstall leaves 6GB | hawai-stack | `AWAITING REPLY` | Responded with cache paths. |
| **#259** | Local Translation Issues | destinyawaits | `AWAITING REPLY` | Responded 04-09: overwrite fix shipped in post3. |
| **#255** | 如何用ollama进行翻译 | cheny7918 | `RESPONDED` | Pointed to v1.8.10 Ollama GUI. |
| **#243** | Install verify fails (RTX 3050) | Trenchcrack | `AWAITING REPLY` | Community helping. |
| **#240** | GUI access violation Win11 | m739566004-svg | `SHIPPED` post2 | |
| **#237** | XXL model questions | yangming2027 | `COMMUNITY` | |
| **#233** | Translation error (local LLM) | WillChengCN | `RESPONDED` | Ollama recommended. |
| **#231** | Kaggle notebook error | fzfile | `SHIPPED` post3 | Colab fix shipped. Reopened by #290/#291 patterns. |
| **#227** | M1 Max transformer mode issues | dadlaugh | `STALE` | From 2026-03-17. Apple Silicon transformer mode. Not actively tracked. |
| **#225** | GUI white screen | github3C | `STALE` | |
| **#217** | GUI.exe not found (China network) | loveGEM | `AWAITING REPLY` | Follow-up 04-17: bilingual China-network template posted (community ask + mirror roadmap + whl workaround). User vimbackground had given up after whisper install also failed. |

### Feature Requests (35)

| # | Title | Priority | Target |
|---|-------|----------|--------|
| **#298** | Settings persistence for API/prompt fields | MEDIUM | v1.9.0 — **NEW 04-21.** yy739566004: "每次使用都要重新输入api信息和翻译提示". **Duplicate of #96**. **Responded 04-21** ([comment-4287968531](https://github.com/meizhong986/WhisperJAV/issues/298#issuecomment-4287968531)): pointed to #96, asked if OK to close and track there. `AWAITING REPLY`. |
| **#293** | Translation context / Whisper prompt feature | LOW | v1.9+ — **NEW 04-16.** User asks if movie descriptions/names as prompt would improve accuracy. Whisper and Qwen3-ASR both support initial prompt. Valid P2 ask. |
| **#292** | Low GPU utilization with higher-end models | LOW | v1.9+ — **NEW 04-17.** User running supergemma4-26b MoE on RTX 4090. Community (@justantopair-ai) already answered with good technical explanation. Probably just needs a thumbs-up from owner. |
| **#279** | Stash integration | LOW | Backlog — l34240013 provided detailed workflow (filename convention `<VIDEO>.<LANG>.srt`, auto-trigger points). **Responded 04-21**: thanked for integration detail, backlog-confirmed (no v1.9.0 timeline). |
| **#268** | Thai + Korean translation targets | LOW | v1.9+ |
| **#265** | Post-translation hallucination filter (Chinese) | MEDIUM | v1.9 |
| **#264** | Model download location customization | LOW | Responded — starkwsam asked why `HF_HOME` doesn't move whisper cache. **Responded 04-21**: distinguished OpenAI Whisper cache (respects `XDG_CACHE_HOME`) from HuggingFace cache (respects `HF_HOME`). PowerShell examples + manual move instructions. |
| **#262** | Cohere Transcribe model | LOW | v1.9.x |
| **#254** | Remove non-speech sounds | MEDIUM | v1.9 |
| **#252** | Multi-speaker / diarization | MEDIUM | v1.9+ |
| **#250** | Model folder documentation | LOW | Docs / FAQ |
| **#248** | Diarization | MEDIUM | v1.9+ |
| **#247** | Docker support | LOW | Backlog — foxfire881 requested docker-compose for NAS CPU-only 7x24 use case. **Responded 04-21**: acknowledged NAS motivator, backlog. |
| **#246** | Serverless GPU pipeline + anime-whisper hallucination | LOW | v1.9+ — from 2026-03-31. Cluster with #265. |
| **#242** | XXL in Pass 1 | MEDIUM | v1.9 |
| **#239** | AMD GPU | MEDIUM | v1.9+ |
| **#232** | Whisper-ja-anime-v0.1 model support | LOW | v1.9+ — from 2026-03-16. Cluster with #205, #246. |
| **#230** | Standalone merge module | HIGH | v1.9.0 |
| **#224** | Vocal separation | MEDIUM | v1.9 |
| **#223** | Subtitle edit to PurfView Faster Whisper for Pass 2 | LOW | Backlog — from 2026-03-25. Cluster with #230 (merge module). |
| **#215** | Qwen3-ASR quality | LOW | Expected behavior |
| **#213** | Intel GPU (XPU) | LOW | v1.9+ |
| **#206** | Grey out incompatible options | MEDIUM | v1.9 |
| **#205** | VibeVoice ASR | LOW | v1.9+ |
| **#181** | Frameless window | LOW | Backlog |
| **#180** | Multi-language GUI (i18n) | HIGH | v1.9.0 |
| **#175** | Chinese GUI | HIGH | v1.9.0 |
| **#164** | MPEG-TS + Drive | LOW | Backlog |
| **#142** | AMD Radeon ROCm | MEDIUM | v1.9+ |
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

*Note: #128 (Gemma 3 models) was merged as a PR, not an issue — removed from tracker.*

---

## Immediate Actions — Post-Release Priority

### Completed (rev26)

All P0 and P1 responses posted on 2026-04-09:
- #284 (install stuck), #280 (Qwen3-ASR), #267 (numba follow-up)
- #281 (Ollama UX), #282 (Ollama GitHub), #271 (thinking models), #259 (overwrite fix)

### Outstanding Actions (rev40) — All batch replies posted

**2026-04-21 batch: 10 replies posted across P0/P1/P2 queues.** All responded issues flipped to AWAITING REPLY. Comment links:

**P0 — post-release notifications (3):**
- #287 → [comment-4285124363](https://github.com/meizhong986/WhisperJAV/issues/287#issuecomment-4285124363): v1.8.11 sanitizer fix, upgrade paths for installer + in-place pip.
- #271 → [comment-4285124420](https://github.com/meizhong986/WhisperJAV/issues/271#issuecomment-4285124420): curated list + `--ollama-max-tokens` shipped, mentioned @TinyRick1489.
- #291 → [comment-4285124483](https://github.com/meizhong986/WhisperJAV/issues/291#issuecomment-4285124483): Colab `[llm]` fix, Gemini 2.0 Flash retirement noted.

**P0 — new issues (2):**
- #297 → [comment-4285124255](https://github.com/meizhong986/WhisperJAV/issues/297#issuecomment-4285124255): asked for debug log + segmenter swap + batch-vs-separate-runs clarification.
- #296 → [comment-4285124305](https://github.com/meizhong986/WhisperJAV/issues/296#issuecomment-4285124305): thinking-model explanation, recommended `translategemma:12b` / `qwen2.5:7b-instruct`.

**P1 (2):**
- #290 → [comment-4285124773](https://github.com/meizhong986/WhisperJAV/issues/290#issuecomment-4285124773): summarized 3 failure modes, workaround (ffmpeg-dsp/none), v1.8.12 fix commitment.
- #294 → [comment-4285124829](https://github.com/meizhong986/WhisperJAV/issues/294#issuecomment-4285124829): 4-point info-gathering (provider, length, console, model) + missing-vs-untranslated check.

**P2 (3):**
- #264 → [comment-4285124879](https://github.com/meizhong986/WhisperJAV/issues/264#issuecomment-4285124879): distinguished Whisper vs HF cache, `XDG_CACHE_HOME` for Whisper.
- #279 → [comment-4285124938](https://github.com/meizhong986/WhisperJAV/issues/279#issuecomment-4285124938): thanked for workflow detail, backlog confirmation.
- #247 → [comment-4285124985](https://github.com/meizhong986/WhisperJAV/issues/247#issuecomment-4285124985): thanked @foxfire881 for NAS motivator, backlog.

**Board state**: 0 NEEDS RESPONSE, 0 NEEDS FOLLOW-UP, 22 AWAITING REPLY, 3 SHIPPED-awaiting-confirm.

### Operational follow-ups from v1.8.11 release

- ✅ **Regex gist (`EXACT_LIST_URL`, gist `ecca22c8ddb9dcab4f6df7813c275a00`, `regexp_v09.json`)** — verified live 2026-04-21: contains v1.8.11 Fix 5a patterns (tortoise shell `〔[^〕]+〕`, emoji class expanded with `♫♩♬🎵🎶`, bracketed SDH keywords for 歓声/喘息/呼吸/ノイズ/SE/BGM/SFX). All three annotated "added v1.8.11" in the gist.
- ✅ **Filter gist (`FILTER_LIST_URL`, gist `4882bdb3f4f5aa4034a112cebd2e0845`, `filter_sorted_v08.json`)** — manually updated by maintainer 2026-04-21. Separate from Fix 5a but now current.
- ✅ Version bump, release notes, tag, merge, push — all complete.
- ✅ `.exe` uploaded to GitHub Release.
- ⚠️ Tag correction: the v1.8.11 tag was initially created on the wrong commit (old main, 1.8.10.post3 source) and force-moved to the merge commit after the merge. No-one fetched the bad tag in the interim.

**Note on gist architecture**: there are TWO separate hallucination gists. `EXACT_LIST_URL` → regex patterns (v09). `FILTER_LIST_URL` → exact-match phrase list (v08). Fix 5a modified only the regex file. Worth documenting in a developer-facing note for the next maintainer update to avoid confusion.

### Close candidates — re-evaluate on 2026-04-27
- **#217** loveGEM — China-network template posted 04-17. If no reply by 04-27 (10 days), close with "feel free to reopen if you still need help".
- **#251** zoqapopita93 (fastgit SSL) — older AWAITING. Revisit.
- **#261** henry99a — user found cmd workaround long ago. Ask if issue is resolved.

---

### Architectural concerns

1. **#267 RESOLVED — numba JIT cache fix in post3 was insufficient.** OrangeFarmHorse confirms `NUMBA_DISABLE_JIT=1` env var works. So the post3 `ensure_numba_cache_dir()` fix did NOT fully solve the problem — the workaround is to disable JIT entirely. **Action for v1.8.11**: detect admin-installed scenarios and set `NUMBA_DISABLE_JIT=1` automatically, or invoke `setup_console()` even earlier in the import chain so the cache redirect actually takes effect before librosa imports. Need to investigate WHY the redirect didn't work.

2. **#267 and #263 had DIFFERENT root causes** — despite being initially cross-referenced as the same hang. #267 was numba JIT (resolved by NUMBA_DISABLE_JIT=1). #263 is Chinese network/GFW (torch.hub Silero VAD → api.github.com → 403). Two distinct problems that produced similar symptoms.

3. **torch.hub Silero VAD fragility — Chinese network issue (#263)** — silero-v3.1 and silero-v4.0 backends both call `torch.hub.load()` which hits `api.github.com/repos/snakers4/silero-vad/branches/...`. **GitHub API is blocked/throttled from China**, causing HTTP 403. PyTorch's `_validate_not_a_forked_repo` has a latent `KeyError` bug that masks the real network error. Solutions for v1.8.11:
   - Pre-bundle Silero v3.1/v4.0 model files with the installer (no runtime GitHub call)
   - OR switch default to silero-v6.2 (pip-installed `silero-vad` package, no GitHub API)
   - OR catch torch.hub failure and auto-fallback to silero-v6.2
   - Add `skip_validation=True` to torch.hub.load() if PyTorch version supports it
   - OR set HTTP_PROXY/HTTPS_PROXY env vars from Windows system proxy settings (auto-detect)

4. **Default segmenter recently changed to silero-v3.1** — may have **increased Chinese-user exposure** to the torch.hub problem. Worth reconsidering for v1.8.11.

5. **Install hardening track for v1.9.0 must explicitly cover China-network scenarios** — #284, #217, #251, #261, #263 all touch this. It's a coherent cluster that needs systematic treatment, not piecemeal fixes.

6. ~~**Ollama `--max-tokens` not exposed to CLI**~~ — **FIXED in v1.8.11 dev** (commit 275adb5). `--ollama-max-tokens` flag added to both CLI entry points.

7. **Translate tab source language dropdown hard-coded to CJK** — needs GUI fix to pass through all Whisper languages. v1.9.0.

8. **Per-GPU-arch installer variants question (#286)** — techguru0 referenced VideOCR's approach of separate cu118 (10-series) and cu129 (16-50 series) installers. Worth considering for v1.9.0 install hardening, but adds maintenance burden.

9. ~~**Curated Ollama model list is broken (#271)**~~ — **FIXED in v1.8.11 dev** (commit 275adb5). Removed `Shisa-v2.1-Qwen3-8B` (thinking model). New curated list: `gemma3:12b` (#1), `qwen2.5:7b-instruct` (#2), `qwen2.5-abliterate` (#3), `dolphin-llama3` (#4). All instruct-only.

10. **Discussion #257 — community translation knowledge** — referenced by @justantopair-ai in #271. Active community thread about translation models, settings, and findings. Worth mining for v1.9.0 hardening insights and Ollama model recommendations. **Future hardening reference.**

11. **Colab Pass 2 OOM regression (#290)** — ClearVoice 48kHz (MossFormer2_SE_48K) appears to push Colab free-tier memory past the kill threshold. Pass 1 completes 7908s of audio fine; Pass 2 dies during speech enhancement load. Options: (a) lower speech_enhancer default on Colab to `zipenhancer` (16kHz, much smaller), (b) add Colab memory detection and auto-downgrade, (c) document Colab memory constraint in Colab notebook header. Likely regression from post3 ClearVoice-48kHz default.

12. ~~**Colab local-LLM break (#291)**~~ — **FIXED in v1.8.11** (commit `b3499a2`). `install_colab.sh` now pulls the `[llm]` extra, bringing in `starlette-context`, `sse-starlette`, `uvicorn`, `fastapi`, `pydantic-settings`. Still-valid observation: `--provider local` is scheduled for removal in v1.9.0 regardless.

14. **ClearVoice looks like a MEMORY LEAK, not a heavy model (#290)** — updated 2026-04-21 with new data from reporter @ktrankc. Original 3-mode hypothesis was insufficient. New evidence:
    - **V100 32GB** → CUDA OOM during MossFormer2_SE_48K
    - **RTX 8000 48GB** → CUDA OOM
    - **RTX PRO 6000 WS 96GB** → CUDA OOM, **PyTorch already holding 94.28 GiB** when ClearVoice tried to allocate 868 MiB
    - **Colab Pro** → RAM "Killed"
    - **RTX 3090 24GB** → CUDA OOM at 21.8GB
    - **Colab L4 (ZipEnhancer)** → separate ModelScope threading-init race (real bug)
    - **ClearVoice-FRCRN_SE_16K** → separate numpy error: `Cannot interpret '113705354' as a data type`

    OOM on 96GB VRAM rules out the "model is big" interpretation. Either ClearVoice isn't releasing scene tensors between enhancement calls, or there's a numpy / ModelScope binary-compat issue that bleeds memory. Verified 04-21: `clearvoice.py` DOES resample correctly to the model's native SR (so wrong-kHz-to-48K hypothesis is NOT the cause) and DOES call `del` + `gc.collect()` + `torch.cuda.empty_cache()` in `cleanup()` — but `cleanup()` only fires on `__del__`, not per-scene. The enhancer instance is reused across all scenes with no `torch.no_grad()` wrapper and no explicit `.eval()` enforcement in `_process_audio`.

    **Decision 04-21**: do NOT attempt a leak fix in v1.8.12. Mask ClearVoice out in the GUI + CLI, keep backend code in place, defer deep dive to the 2.x series. Rationale: cross-hardware scaling (32GB→96GB all OOM) suggests a fundamental interop issue that needs careful investigation, not a quick patch. Users are directed to `zipenhancer` (owner's own daily driver) or `ffmpeg-dsp`.

13. **#281 closure pattern (uninstall+reinstall with admin)** — wlee15 resolved Ollama detection by reinstalling as all-users with admin. Similar to other "all-users-install fixes it" reports. Worth investigating why user-local installs have more Ollama detection friction. Could be a PATH / env var inheritance issue.

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Status |
|---------|--------|---------|--------|
| **Ollama / Translation** | #296, #293, #292, #282, #271, #259, #255, #233, #132 | #132 | #281 CLOSED. **#271 SHIPPED in v1.8.11** (curated list + max-tokens flag). **#296 NEW**: same thinking-model pattern as #271 with gemma3. Community-saturated cluster. |
| **Install: PyTorch download / China network** | #289, #284, #217, #263, #261, #243 | #284 | **Growing cluster** — 6 issues touching PyTorch download timeouts / TLS resets / GFW. v1.9.0 install hardening imperative. |
| **Install: GPU architecture mismatch** | #286, #243 | #286 | GTX 10-series Pascal kernel dropped from recent torch wheels. Downgrade to 2.5.1+cu118 workaround. |
| **Colab regressions** | #291, #290, #231 | #290 | **#291 SHIPPED in v1.8.11** (`[llm]` extra). **#290 STILL OPEN** — now 3 failure modes confirmed across Colab Pro/L4 and 3090 24GB. ClearVoice memory profile is the thread. v1.8.12. |
| **Semantic hang / numba** | #267 (closed), #263 | #267 | Fix shipped in post3 but reported insufficient. #263 distinct root cause (GFW). |
| **Qwen3-ASR** | #280, #215 | #280 | Upstream transformers mismatch. |
| **Quality / Sanitizer regression** | #287, #297, #265, #246 | #287 | **#287 SHIPPED in v1.8.11** (full sanitizer chain). **#297 NEW**: blank SRT on batch — unknown if related; needs log review. |
| **Diarization** | #248, #252 | #248 | v1.9+. |
| **AMD/Intel GPU** | #239, #142, #114, #213 | #142 | v1.9+. |
| **i18n** | #175, #180 | #180 | v1.9.0 P0. |
| **Merge module** | #230, #223, #265 | #230 | v1.9.0 P1. |
| **Alt ASR models** | #205, #232, #246 | #232 | Anime-specific models, VibeVoice. |

---

## Release Roadmap

### v1.8.12 — Next Point Release (candidate scope)

**Theme: Post-1.8.11 hardening, Colab memory, installer PATH**

| Priority | Item | Issues | Effort | Notes |
|---|---|---|---|---|
| **P0** | Mask ClearVoice out of enhancer options | #290 | Small | **Decision 04-21**: disable (mask out) ClearVoice in GUI dropdown + CLI choices + factory guard (fall back to `zipenhancer` / `ffmpeg-dsp` if old config still requests it). Keep backend code in place for re-enable after 2.x deep dive. Separately: ZipEnhancer ModelScope threading-init race (real bug, still fix in v1.8.12). Separately: ClearVoice leak investigation deferred to **2.x**. |
| **P0** | Installer PATH bug | — | Small | Standalone installer fails to persist user PATH — system `ffmpeg` (potentially 8.x) wins over bundled 7.1.1. Hypothesized link to audio-processing hangs (unverified). See `memory/project_v1812_installer_path_bug.md`. |
| **P1** | Default backend flip: `auditok→semantic`, `silero-v3.1→ten` | — | Small | 8 edits across CLI + GUI Ensemble tab. Notebook already has these defaults on `dev-1.8.11`. See `memory/project_v1812_default_backends_flip.md`. |
| **P1** | Blank SRT on batch triage (#297) | #297 | ? | Triage first — may or may not be bug. Review attached log. |
| **P2** | Qwen3-ASR transformers pin | #280 | Small | One-line pin. Was deferred from v1.8.11 for scope discipline. |
| **P2** | Gist sync operational check | — | Trivial | Confirm `regexp_v09.json` synced to `HallucinationConstants.EXACT_LIST_URL` gist (flagged as pending in rev38 pre-release checklist). |

### v1.9.0 — Next Major Release

**Theme: Platform Expansion + Translation Overhaul + UX**

| Priority | Item | Issues | Effort | Notes |
|---|---|---|---|---|
| **P0** | Ollama full migration (deprecate llama-cpp-python) | #132, #233, #255, #259 | Large | Remove ~1500 LOC. |
| **P0** | Chinese UI (partial i18n) | #175, #180 | Medium | 40%+ issues are Chinese. |
| **P0** | Unified CLI override layer | #269 | Small | Standard/ensemble divergence. |
| **P1** | Ollama model recommendation fix | #271 | Small | Only recommend instruct models in GUI. |
| **P1** | Post-translation hallucination filter (Chinese) | #265 | Medium | User contributed rules + v2 word list. |
| **P1** | Post-processing polish (trailing period, leading dash, line wrapping) | #265 | Small | 3 concrete suggestions. |
| **P1** | Standalone merge CLI | #230 | Medium | `whisperjav-merge`. |
| **P1** | Speaker diarization | #248, #252 | Large | Need quality solution. |
| **P1** | XXL in Pass 1 | #242 | Medium | Needs forced alignment. |
| **P1** | AMD ROCm support | #142, #114, #239 | Medium | Document + test. |
| **P1** | Additional translation targets | #268 | Small | Thai, Korean. |
| **P2** | Semantic scene detection hang (deeper investigation) | #267 | Medium | numba redirect may be insufficient. |
| **P2** | GUI settings persistence | #96 | Medium | Long-standing request. |
| **P2** | Vocal separation | #224, #254 | Medium | BS-RoFormer / UVR. |
| **P2** | Grey out incompatible options | #206 | Small | Prevent invalid GUI combos. |
| **P2** | Qwen3-ASR transformers compatibility | #280 | Medium | Pin or adapt to API change. |
| **P3** | Docker support | #247 | Medium | Dockerfile + compose. |
| **P3** | Gemma 3 model configs | #128 | Small | Contributor PR. |
| **P3** | Stash integration | #279 | Medium | Niche request. |

### v2.0 — Strategic Vision

**Theme: Architecture + Scale + Ecosystem**

| Area | Item | Why |
|---|---|---|
| **Architecture** | Plugin system for ASR backends | Community contributions |
| **Architecture** | Web-based UI (replace pywebview) | Persistent issues (#225, #240) |
| **Translation** | Translation model compatibility layer | #271 — detect thinking models |
| **Scale** | Batch processing dashboard | Queue, progress, ETA |
| **Platform** | China mirror support | #217, #251 — PyTorch behind GFW |
| **Platform** | Linux native installer | AppImage or deb |

---

## Changelog

| Date | Changes |
|------|---------|
| **2026-04-21** | **rev41.** 64→65 open (+1 new: **#298** yy739566004 — settings persistence for API/prompt fields, duplicate of #96, added to Feature Requests as MEDIUM v1.9.0 target). **#290 responded and flipped AWAITING REPLY** ([comment-4287955148](https://github.com/meizhong986/WhisperJAV/issues/290#issuecomment-4287955148)): reporter @ktrankc had escalated 04-21 02:47 with new failure data — ClearVoice OOM on **V100 32GB, RTX 8000 48GB, and RTX PRO 6000 WS 96GB** (PyTorch holding 94GB of 96GB), plus FRCRN_SE_16K numpy dtype error (`Cannot interpret '113705354'`). Verified code-level: sample-rate hypothesis ruled out (backend resamples correctly), but no per-scene cache clear and enhancer instance reused across all scenes. **Decision: mask ClearVoice out in v1.8.12**, defer leak investigation to 2.x. Rationale: cross-hardware scaling (32→96GB all OOM) suggests interop issue that needs a proper deep dive, not a patch. v1.8.12 P0 roadmap row updated from "leak fix + auto-fallback" to "mask out". Architectural concern #14 updated with verified facts and decision. **Board state**: 0 NEEDS RESPONSE, 0 NEEDS FOLLOW-UP, 23 AWAITING REPLY, 3 SHIPPED-awaiting-confirm. #298 responded ([comment-4287968531](https://github.com/meizhong986/WhisperJAV/issues/298#issuecomment-4287968531)) — pointed to #96, asked permission to close. |
| 2026-04-21 | **rev40.** 64 open (no net change). **10 replies posted in one batch** — all P0 / P1 / P2 queues cleared. P0 notifications: #287 (sanitizer fix chain explained + upgrade paths), #271 (curated list + max-tokens, @-mentioned TinyRick1489), #291 (Colab `[llm]` fix + same-session pip workaround + Gemini 2.0 retirement note). P0 new issues: #297 (asked for debug log + segmenter swap + batch-run clarification), #296 (thinking-model explanation + translategemma/qwen2.5-instruct recommendation). P1: #290 (3-mode ClearVoice/ZipEnhancer summary + ffmpeg-dsp workaround + v1.8.12 commitment), #294 (4-point info-gather). P2: #264 (Whisper vs HF cache distinction), #279 (Stash backlog ack), #247 (Docker NAS ack). All 7 NEEDS-RESPONSE/FOLLOW-UP items flipped to AWAITING REPLY. **Board state**: 0 NEEDS RESPONSE, 0 NEEDS FOLLOW-UP, 22 AWAITING REPLY, 3 SHIPPED-awaiting-confirm. Operational follow-up: `regexp_v09.json` gist sync for Fix 5a still pending. |
| **2026-04-20** | **rev39.** 63→64 open. **v1.8.11 RELEASED** (merge commit `97ad9a4`, tag `v1.8.11`, `.exe` + release notes on GitHub). Post-release state flips: **#287, #271, #291 → SHIPPED** (user notifications pending). **2 new issues**: #297 (teijiIshida — blank SRT files on batch, v1.8.10-hotfix3, NEEDS RESPONSE, P0-P1 triage with attached log), #296 (triatomic — gemma3-12b thinking output, classic #271 pattern). **1 closed**: #295 (lukazzz007, by reporter after community pointed to Discussion #257). **#290 updated**: user added 3rd failure mode on RTX 3090 24GB (ClearVoice CUDA OOM) — cross-environment ClearVoice memory issue now clear. **#264, #279, #247 flipped to NEEDS FOLLOW-UP** (user replies after owner's last comment). **Release-state anomaly corrected**: v1.8.11 tag was initially placed on old main (1.8.10.post3 source) due to a missed `git merge` step; force-moved to correct merge commit within ~2hrs, no one fetched the bad tag. **v1.8.12 candidate scope added to roadmap**: Colab memory fix (#290), installer PATH bug, default backend flip (auditok→semantic, silero-v3.1→ten), #297 triage, #280 one-liner. |
| 2026-04-18 | **rev38.** 61→63 open (+2 new: #294 truncated translation, #295 best-settings question). **v1.8.11 sanitizer fix work COMPLETE**: 5 commits landed (`970af55` corpus, `93bb71a` Fix 1 ${N:0:M}, `fc5353c` Fix 2 symbol purge, `3508005` Fix 3 CPS newline, `783b218` Fix 5a SDH patterns). 91/91 unit tests pass. Regression corpus verified (19 identical, 12 Category-A changes). Real JAV forensic run showed zero symbol-only residues leaked. **#287 Tier A blocker RESOLVED.** #290 flipped to NEEDS FOLLOW-UP: user replied with new ZipEnhancer threading bug (v1.8.12 item). v1.8.11 ready to tag pending version bump, gist sync for Fix 5a, and release notes. |
| 2026-04-17 | **rev37.** Commits landed on `dev-1.8.11`: `b3499a2` (Colab `llm` extra fix for #291) + `a737bdf` (docs rev36). Added "Immediate Open Issues for v1.8.11" section with Tier A/B/C/D scoping: #287 tagged as release blocker (quality regression risk from silero-v3.1 default swap), #280 as opportunistic one-line pin, #267/China-network cluster/#265/#286 explicitly deferred to v1.9.0 or v1.9.1. |
| 2026-04-17 | **rev36.** Posted 4 replies: #291 (root cause identified — Colab install missing `[llm]` extra; workaround given + v1.8.11 fix promised), #290 (factual: "Killed" is ambiguous, asked for Colab tier/file size/resource panel, recommended zipenhancer to test ClearVoice hypothesis), #292 (owner ack to community-answered thread, pointed to translategemma:12b / qwen2.5:7b-instruct), #293 (logged Whisper initial_prompt exposure as v1.9+ candidate). **#291 FIX CODED for v1.8.11**: `installer/install_colab.sh` line 176 — added `llm` to extras list (pulls in `starlette-context`, `sse-starlette`, `uvicorn`, `fastapi`, `pydantic-settings`). Both Colab notebooks reference this script, so the fix propagates. NEEDS RESPONSE queue now empty. |
| 2026-04-17 | **rev35.** Posted bilingual China-network template (Chinese + English) to 5 issues: #289, #284, #217, #263, #261. Template asks community to share working VPN / DNS / CUDA+PyTorch combos, flags **v1.9.1 China-mirror exploration** as a goal (without timeline commitment), and repeats the pre-downloaded whl + `Scripts\pip.exe install --no-deps` + silero-v6.2 interim workarounds. All 5 issues now AWAITING REPLY. |
| 2026-04-17 | **rev34.** 54→61 open. **5 new issues**: #293 (translation context), #292 (low GPU util), #291 (Colab local-LLM starlette_context), #290 (Colab Pass 2 OOM), #289 (CUDA 12.8 timeouts). **4 previously-untracked** added: #223, #227, #232, #246. **1 closed**: #281 (user resolved via admin reinstall). **Status flips**: #287 NEEDS RESPONSE → AWAITING REPLY (owner asked about VAD swap). #217 AWAITING REPLY → NEEDS FOLLOW-UP (user gave up after whisper install also failed). **Removed**: #128 (was PR, merged, not issue). **New cluster**: Colab regressions (#290, #291) suggests post3 introduced a regression pair. Install/PyTorch/China cluster grew to 6 issues. |
| 2026-04-12 | **rev33.** 53→54 open (+1 new: #287). **v1.8.11 dev branch started.** Two fixes coded: (1) Ollama curated model list — removed thinking model, new instruct-only defaults. (2) `--ollama-max-tokens` CLI flag added to both entry points. #271 responded with fix confirmation. #287 new bug identified (all subtitles "!!", NEEDS RESPONSE). |
| 2026-04-11 | **rev32.** 54→53 open. **#267 CLOSED** (user confirmed `NUMBA_DISABLE_JIT=1` works). All 6 follow-ups responded: #263 (proxy + silero-v6.2), #286 (humble variant ack), #271 (CLI flag commit + Shisa-Qwen3 curated list bug), #268 (GUI gap), #262 (Cohere thanks). **Found: curated Ollama #1 model is a Qwen3 thinking model — bug for v1.8.11.** Discussion #257 noted as future hardening reference. |
| 2026-04-11 | **rev31.** 54 open. #267 user confirmed fix. #263 user replied with proxy question. 5 new follow-ups identified. |
| 2026-04-10 | **rev30.** 55→54 open. #285 closed by reporter. #263 corrected diagnosis (GFW). Posted silero-v6.2 workaround. |
| 2026-04-10 | **rev29.** 55 open. All P0/P1/P2 responded. Board clean. |
| 2026-04-10 | **rev28.** 53→55 open (+2). 2 new issues: #286, #285. New comments on #271, #268. |
| 2026-04-09 | **rev27.** 57→53 open. All P2 responses posted. Closed 4 stale: #234, #251, #258, #260. |
| 2026-04-09 | **rev26.** All P0/P1 responses posted: #284, #280, #267, #281, #282, #271, #259. Found missed #217 reply (10 days stale). |
| 2026-04-09 | **rev25.** v1.8.10.post3 RELEASED — 25 commits, 3 crash fixes, full ASR/VAD retune, Silero v3.1 default, post-processing overhaul, GUI audit, TEN backend, forensic tools. Tagged, pushed, GitHub release published (stable). |
| 2026-04-09 | **rev24.** 57 open (+2). 7 new issues. 14 unpushed commits identified. |
| 2026-04-01 | **rev23.** 55 open. #267 ROOT CAUSE FOUND. #259 overwrite fix. #271 instruct models work. |
| 2026-04-01 | **rev22.** 55 open. XXL fix. Translation overwrite fix. |
| 2026-03-31 | **rev21.** 54→55 open. New: #271, #272. Closed: #269. |
| 2026-03-31 | **rev20.** 55→54 open. New: #265, #267, #268, #269. Closed: #218, #221, #236, #244. |
| 2026-03-30 | **rev19.** v1.8.10 released. |
| 2026-03-29 | **rev18.** 55 open. |
| 2026-03-28 | **rev17.** Closed 11. Responded 7. |

---
