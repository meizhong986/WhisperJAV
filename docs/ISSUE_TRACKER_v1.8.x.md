# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-04-30 (rev45) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **73 open** on GitHub

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
| Total open on GitHub | **73** | +3 net since rev44: **+3 NEW** (#309 white screen mklink, #311 FireRedVAD feature request, #312 VAD param mismatch report). No closures since rev44. |
| **v1.8.12 RELEASED** | — | **2026-04-30**. Tagged `v1.8.12`, merged `dev_v1.8.12`→main, pushed. GitHub Release published with `.exe` + wheel. **20 commits** including WhisperSeg VAD, tight defaults retune, engine-split presets, F5 regression fix, anime ellipsis filter. |
| **v1.8.11 RELEASED** | — | 2026-04-20. Tagged, merged, pushed. Stable prior release. |
| **v1.8.10.post3 RELEASED** | — | 2026-04-09. Stable (older release). |
| **NEEDS RESPONSE** | **6** | **#300, #304, #305, #309, #311, #312.** Three new issues since rev44; #305 + #306 + #307 + #308 still unresponded from rev44. |
| **NEEDS FOLLOW-UP** | **6** | **#271, #231, #264, #287, #294, #302.** Three flipped from AWAITING REPLY to NEEDS FOLLOW-UP since rev44: **#287** (zoqapopita93 replied 04-25 — sanitizer fix in v1.8.11 didn't help, still `!!` output with default settings); **#294** (CaliburnKoko replied 04-26 — `no_speech_threshold=0.85` didn't help; `fast/aggressive` does work); **#302** (Kukuindi replied 04-22 — same fix didn't help, even WORSE; teijiIshida added comment 04-28 fingering silero-v4.0 as random-fail culprit). |
| **AWAITING REPLY** | 19 | −4 vs rev44: 3 flipped to NEEDS FOLLOW-UP (#287, #294, #302) + 1 net other movement. v1.8.12 retest needed for #294, #302. |
| **SHIPPED in v1.8.12** | **8** | **F5 best_of regression** (#294, #302 cluster — needs user retest), **anime ellipsis filter** (anime-whisper artifact lines), **silero-v3.1 fallback alignment** (#297 cluster — fixes the "silero-v4.0 random fail" teijiIshida found), **TEN max_speech schema fix**, **anime chunk_threshold plumbing**, **`whisperseg` in `--qwen-segmenter` choices**, **engine-split sensitivity retune** (aggressive `no_speech_threshold` 0.77→0.84 — addresses #294/#302/#287 root cause), **WhisperSeg ONNX VAD backend** added. |
| **SHIPPED in v1.8.11, user notified** | **3** | #287 (still NEEDS FOLLOW-UP — see above), #291, #271 (still NEEDS FOLLOW-UP — TinyRick1489 separate ask). |
| Feature requests (open) | 37 | +1 vs rev44: **#311** (FireRedVAD as new VAD backend, kylesskim-sys 04-28). |

---

## v1.8.12 — RELEASED (2026-04-30)

Tagged `v1.8.12` and merged `dev_v1.8.12` to `main` on 2026-04-30. GitHub Release published with `.exe` standalone installer (Windows) and `whisperjav-1.8.12-py3-none-any.whl` (cross-platform). 20 commits on top of v1.8.11. **Theme: WhisperSeg ONNX VAD + tight defaults retune.**

**Headline feature**: WhisperSeg ONNX speech segmenter (`TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`, MIT, ~119MB). Whisper-base encoder + 2-layer decoder, 20ms frame resolution over 30s windows, trained on ~500h Japanese ASMR. Selectable in GUI Ensemble tab + via `--qwen-segmenter whisperseg` / `--pass1/2-speech-segmenter whisperseg`. **F1=0.787** on Netflix-GT 283s JAV clip (aggressive sensitivity) vs `silero-v3.1`=0.625, `silero-v6.2`=0.654, `ten`=0.698. Strongest VAD shipped to date for soft/whispered Japanese speech. Install: `pip install whisperjav[whisperseg]` (CPU) or `[whisperseg-gpu]` (CUDA).

**Critical regression fixes shipped (potential resolution for prior reports):**

- **F5 `best_of` regression fix** (impacts #294 + #302 cluster) — A prior commit `34fa713` (engine-split retune) lowered aggressive `best_of: 2 → 1`, which combined with tight VAD groups caused 76.9% empty VAD groups on JAV-style content. Reverted to `best_of=2` after isolated empirical disambiguation testing (V13 = 0% empty, 29 segs, 29s vs `best_of=1` = 39% empty + 23 segs + 91s; `best_of=3` ALSO regresses to 39% — unique sweet spot at 2). Aggressive `no_speech_threshold` also raised 0.77 → 0.84 in the same retune.
- **Silero-v3.1 fallback alignment** (impacts #297 cluster) — `faster_whisper_pro_asr.py:87` and `whisper_pro_asr.py:59` previously fell back to `silero-v4.0` when the resolver did not explicitly set `params.speech_segmenter.backend`, silently overriding the `LEGACY_PIPELINES["balanced"|"fidelity"]["vad"] = "silero-v3.1"` declaration. teijiIshida's 04-28 comment on #302 fingered silero-v4.0 as a random-fail culprit; this fix should resolve that path.
- **Anime-whisper ellipsis-only line filter** — Anime-whisper produces SRT entries containing only `…` (or `…?`, `…!`, `…」`, etc.) for short non-speech regions. Now detected and removed at two layers: text-level (`AnimeWhisperCleaner.clean()`) and SRT-level (`filter_srt_file()` with renumbering). Wired into `qwen_pipeline.py` Phase 8 only when `generator_backend=="anime-whisper"`.
- **TEN VAD `max_speech_duration_s` no longer silently stripped** — Factory `_PARAM_SCHEMAS["ten"]` previously didn't include this key, causing the foreign-key strip to drop the YAML value before it reached the backend. Schema updated. Factory fallback defaults for `ten`, `silero-v6.2`, and `whisperseg` aligned to YAML balanced presets.
- **Anime-mode `chunk_threshold_s` plumbing** — `QwenPipeline` Phase 4 now forwards `self.segmenter_chunk_threshold` into `SpeechSegmenterFactory.create()`. Anime mode's intended `chunk_threshold_s = 0.5` now actually reaches the factory.
- **CLI `--qwen-segmenter` accepts `whisperseg`** — argparse choices were missing it.

**Default values changed users may notice:**
- Aggressive `no_speech_threshold`: 0.77 → 0.84 (more permissive, captures more intimate speech)
- Aggressive `best_of`: 1 → 2 (revert of regression introduced mid-development)
- Aggressive `beam_size`: 2 → 3
- VAD `max_speech_duration_s` (cons/bal/agg): 6/5/4s — sensitivity gradient INVERTED (aggressive = tightest)
- VAD `max_group_duration_s` (cons/bal/agg): 7/6/5s — same inversion
- Aggressive openai-whisper `logprob_threshold`: -1.00 → -1.30 (engine divergence vs faster-whisper)

**New tool**: `tools/vad_groundtruth_analyser/` — side-by-side VAD evaluator with optional GT SRT, produces interactive Plotly HTML + JSON + CSV.

**Refactors**: `neg_threshold` removed as user-facing speech-segmenter parameter (auto-derived internally). GUI Ensemble dropdown simplified (NeMo, Whisper-VAD hidden, CLI access preserved).

**Operational follow-ups (post-release):**
- ⏳ **Pending user re-test on v1.8.12**: #294, #302, #287 — earlier `no_speech_threshold=0.85` advice ALONE didn't help these users; v1.8.12's combination of `no_speech_threshold=0.84` + `best_of=2` + tight VAD (verified working in F6 acceptance test, Pass 1 recall 19% → 88%) is expected to fix them.
- ✅ Tag `v1.8.12`, merge to main, push, .exe published — all complete per user 04-30.

---

## v1.8.13 candidates (next point release)

| Priority | Item | Issues | Notes |
|---|---|---|---|
| **P1 NEW** | ctranslate2 internal state contamination in Model Reuse Pattern | — | Discovered during F5/F6 forensic. faster-whisper / ctranslate2 retains internal GPU state across `transcribe()` calls that's NOT freed by `torch.cuda.empty_cache()` or `gc.collect()`. Production WhisperJAV's `_ensure_asr` Model Reuse Pattern (single WhisperModel reused across all files in a batch) may silently degrade recall on later files in long batch runs. Mitigation candidate: periodic `WhisperModel` reload after N transcribes. F5 production showed 76.9% empty (Pass 1 first scene partly OK, later scenes degrade) vs F5 isolated test showed 39% empty for same params — the delta is consistent with state accumulation. |
| **P0 carry-over** | Installer PATH fix for bundled ffmpeg | — | Standalone installer fails to persist user PATH; system ffmpeg (potentially 8.x) wins over bundled 7.1. Carry-over from v1.8.11 known issues. Hypothesized link to audio-processing hangs (unverified). See `memory/project_v1812_installer_path_bug.md`. |
| **P1** | ZipEnhancer Colab init bug | #290 | ModelScope threading-init race on Colab L4. Real bug separate from ClearVoice memory issue. |
| **P2** | Qwen3-ASR `transformers` version pin | #280 | One-line pin. Was deferred from v1.8.11 + v1.8.12 for scope discipline. |
| **P2** | `--ollama-num-ctx` CLI flag | #271 (TinyRick1489 follow-up) | Reasonable feature ask. |
| **P2** | Investigate VAD param dump_params artifact | #312 | When `--mode fidelity --speech-segmenter ten`, the `dump_params` shows silero VAD values (because the legacy resolver always emits silero presets via `LEGACY_PIPELINES`) even though the actual TEN backend at runtime gets the right TEN config via the firewall in `whisper_pro_asr.py:68-74`. Misleading metadata. Worth fixing the dump to reflect the actual runtime segmenter config. |

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
| **#312** | VAD Parameters dump_params shows silero values for TEN | TinyRick1489 | `NEEDS RESPONSE` | **NEW 04-28.** Sharp investigation. User ran `--mode fidelity --sensitivity conservative --speech-segmenter ten --debug --dump-params --trace-params`. DEBUG log + dump_params + trace_params ALL show silero conservative VAD values (threshold=0.41, max_speech=6.0, max_group=8.0, speech_pad=500), not TEN conservative YAML (threshold=0.42, max_speech=5, max_group=7, end_pad=300). Asks "Why are TEN vad_params not used?". **Diagnosis (no investigation needed)**: This is the metadata-vs-runtime artifact documented in F5 forensic. The legacy resolver (`resolve_legacy_pipeline → resolve_config_v3`) always emits silero VAD params because `LEGACY_PIPELINES["fidelity"]["vad"] = "silero-v3.1"`. The actual TEN backend at runtime gets the correct TEN YAML config via the firewall in `whisper_pro_asr.py:68-74` (which clears resolver-produced silero params for non-silero backends and uses `speech_segmenter_config` + factory schema). Metadata is misleading; runtime is correct. P2 — needs a friendly explanation reply. v1.8.13 candidate: fix the dump to reflect actual runtime segmenter config. |
| **#311** | Feature Request: Add FireRedVAD as a new VAD backend | kylesskim-sys | `NEEDS RESPONSE` | **NEW 04-28.** Industrial-grade SOTA VAD released 03-2026 by FireRedTeam (HF: `FireRedTeam/FireRedVAD`). Cites FLEURS-VAD-102 benchmark: F1=97.57 vs Silero=95.95 vs TEN=95.19; **False Alarm Rate 2.69% vs 9.41%/15.47%**. Claims dramatic FA reduction is exactly what JAV scenes with BGM/breathing need. Also has Audio Event Detection (AED) feature for speech vs music separation. Output format similar to existing VADs → integration "relatively straightforward". Strong feature request — file as v1.9+ candidate; respond with thanks + benchmark interest + acknowledge VAD evaluator tool can compare it side-by-side with existing backends. Tracked under Feature Requests. |
| **#309** | Persistent White Screen in GUI (mklink + WebView2) | Jerry199022 | `NEEDS RESPONSE` | **NEW 04-26.** Detailed bug report with environment quirks. Win 11, RTX 4070 Ti, **`C:\Users\Jerry` is mklink-junctioned to `D:` drive**. WebView2 sandbox apparently fails on cross-drive symlink when resolving AppData. User performed exhaustive troubleshooting: disabled proxy/VPN (resolved DOM events error), set `WEBVIEW2_USER_DATA_FOLDER`, granted Full Control, disabled GPU accel, forced `gui='edgechromium'`, added Windows Defender exclusions, full reinstall — ALL FAILED. **Critical evidence**: even pure pywebview test (`webview.create_window('Test', html='<h1>Hello World</h1>'); webview.start()`) fails with white screen. So this is an upstream pywebview/WebView2 issue with mklink user profiles, not a WhisperJAV-specific bug. Same symptom as #225 but with deep diagnostic narrative. P1. Response: explain pywebview is the boundary; suggest workaround (avoid mklink for user profile, use a non-symlinked Windows account, or run in a VM); acknowledge the diagnostic depth; track under cluster with #225, #240. |
| **#307** | Question: Principles behind JA SRT sanitization | yedkung69-ctrl | `NEEDS RESPONSE` | **NEW 04-23.** Polite thanks + deep question: how does the sanitization pipeline work under the hood? Asks about regex patterns, NLP filtering, dictionary blacklists, VAD-stage acoustic features. Good opportunity for a documentation-friendly answer. P2. Could become a FAQ entry or a docs page. |
| **#308** | Add English source + ASS/SSA subtitle format | SangenBR | `NEEDS RESPONSE` | **NEW 04-23.** Feature request with screenshot: (1) add "English" to Source Language in AI SRT Translate (currently the target is English but source isn't selectable as English), (2) support .ass / .ssa input in the translate pipeline. P2 feature — tracked under Feature Requests. |
| **#306** | Ensemble Mode with Anime-Whisper truncating subs | ktrankc | `NEEDS RESPONSE` | **NEW 04-23.** RunPod CUDA env, v1.8.11, `--pass1 transformers:kotoba-v2.2` + `--pass2 transformers:litagin/anime-whisper` ensemble. Pass1 produced 355 subs OK; Pass2 produced **1 segment of 13,229 chars** (anime-whisper hallucinated) which sanitizer truncated → **0 subs** from Pass 2. Merge fell back to Pass1 (351 after dedup). Clustered with #246 (anime-whisper hallucination). |
| **#305** | 设置目标语言为中文时会有部分翻译内容变成英文 | ric-reff | `NEEDS RESPONSE` | **NEW 04-22.** CN translation target produces "large chunks of English" in output. Reporter tested `llama-cpp gemma-9b` AND `ollama shisa-v2.1-qwen3-8b` (same root cause as #271 — `shisa-v2.1-qwen3-8b` is exactly the thinking model removed from curated list in v1.8.11). Recommend `qwen2.5:7b-instruct` or `translategemma:12b`. |
| **#304** | Why can't I use Fidelity mode? | wenkine-2026 | `NEEDS RESPONSE` | **NEW 04-22.** Console shows PyTorch incompat: GTX 1050 Ti (CC 6.1) vs. PyTorch built for sm_75+. Log reports "v1.8.10" header. Same pattern as #286. CC 6.x hardware isn't supported by the shipped CUDA wheel. Need compat-matrix FAQ / clear error. |
| **#302** | V1.8.11 do not translate whole file & super slow than 1.8.10 | Kukuindi | `NEEDS FOLLOW-UP` | **NEW 04-21.** **Update 04-22**: user replied — `no_speech_threshold=0.85` advice didn't help, **made it WORSE** (28-min total runtime, only 1m37s translated; v1.8.10 took 6 min and translated whole file). **Update 04-28**: teijiIshida added comment fingering **silero-v4.0** — his testing showed `silero-v3.1` and `silero-v6.2` transcribe to end, but `silero-v4.0` randomly fails at 30m or 1h. Says silero-v4.0 is the GUI default and you have to use Ensemble Mode to change it. **v1.8.12 release likely fixes this**: (a) `daab576` aligned silero fallback v4.0→v3.1; (b) `f1e6510` reverted `best_of: 1→2` for aggressive (F5 fix); (c) aggressive `no_speech_threshold` raised to 0.84 in `34fa713`. Earlier 0.85 advice alone didn't work because v1.8.11 still had `best_of=1`. **Action**: ask user to retest on v1.8.12; if still fails, deeper investigation needed. |
| **#301** | V1.8.11 Does not translate whole file | Kukuindi | `CLOSED` | Closed by reporter 04-21 same day as filing. Likely self-merged into their #302 (same author). |
| **#300** | v1.8.11 install.py issue | ktrankc | `NEEDS RESPONSE` | **NEW 04-21.** uv resolver deadlock: `whisperjav[all]` → `bs-roformer-infer` → `requests>=2.31`, but pytorch index pins `requests==2.28.1`; `--index-strategy unsafe-best-match` not passed. User reverted to 1.8.10.post3 and install succeeded. Install regression specific to `[all]` + uv + pytorch index ordering. |
| **#297** | Blank subtitle file after transcription completed | teijiIshida | `AWAITING REPLY` | **NEW 04-19.** Windows 11 25H2, GUI v1.8.10-hotfix3. Multiple files in queue transcribed, only 1 produced subtitles, rest are 0kb. User attached `whisperjav.logs.txt`. **Responded 04-21**: asked for debug-enabled log + batch-vs-separate-runs clarification + segmenter swap (log shows silero-v4.0, default is v3.1). |
| **#296** | Model outputting "thinking or reasoning in the translation" | triatomic | `AWAITING REPLY` | **NEW 04-19.** User on `gemma3-12b` seeing chain-of-thought text in SRT output. Classic thinking-model pattern (same root cause as #271). **Responded 04-21**: explained gemma3 can still mix reasoning; recommended `translategemma:12b` or `qwen2.5:7b-instruct`; noted v1.8.11 curated list fix. |
| **#294** | Doesn't translate the whole file | CaliburnKoko | `NEEDS FOLLOW-UP` | **NEW 04-17.** **Update 04-26**: user replied — `no_speech_threshold=0.85` with `balanced/aggressive` did NOT work (still ~8 min output for 116-min video). **HOWEVER, switching to `fast/aggressive` with default settings DID work — translated whole video.** User states: "The problem only occur on `balance/aggressive` since I update to 1.8.10 hotfix and 1.8.11. Before update, `balance/aggressive` was pretty good." This points specifically at the v1.8.10.post3 / v1.8.11 retune of balanced+aggressive ASR, not segmenter. **v1.8.12 release likely fixes this**: same fix chain as #302 — best_of revert + no_speech_threshold raise + silero-v3.1 fallback. **Action**: ask user to retest balanced/aggressive on v1.8.12; F6 acceptance test verified Pass 1 recall 19% → 88% on similar JAV audio. |
| **#291** | Google Colab Step 3 Translation — `ModuleNotFoundError: starlette_context` | ktrankc | `SHIPPED` v1.8.11 | **FIX SHIPPED** (`b3499a2`). User notified 04-21 with same-session pip workaround + recommended migration to `--provider ollama`. Also noted Gemini 2.0 Flash retirement. |
| **#290** | Google Colab Pass 2 Error — "Killed" after MossFormer2 load | ktrankc | `AWAITING REPLY` | **ESCALATED 04-21 02:47**: user tested on **V100 32GB (OOM), RTX 8000 48GB (OOM), RTX PRO 6000 WS 96GB (OOM — PyTorch holding 94GB of 96GB available)**. ClearVoice-FRCRN_SE_16K gives separate numpy error: `Cannot interpret '113705354' as a data type`. OOM at 96GB rules out "heavy model"; points to tensor-retention / numpy-ModelScope interop issue. **Responded 04-21** ([comment-4287955148](https://github.com/meizhong986/WhisperJAV/issues/290#issuecomment-4287955148)): deep dive deferred to 2.x, ClearVoice to be **masked out in v1.8.12**, user invited to share any root-cause findings. |
| **#289** | PyTorch CUDA 12.8 install timeouts | coco7887 | `AWAITING REPLY` | Responded 04-17 with bilingual China-network template (community ask + mirror roadmap + whl workaround). |
| **#287** | All subtitles are "!!" with latest version | zoqapopita93 | `NEEDS FOLLOW-UP` | **Update 04-25**: user replied "sorry, the same" + screenshot showing all-`!!` output on v1.8.11 with default settings. v1.8.11 sanitizer fix chain (`${N:0:M}` syntax, symbol-only purge, CPS char count, SDH patterns) DID NOT resolve. User tested both TEN-VAD and Silero v3.1, same result. v1.8.12 did NOT add new sanitizer fixes — likely still broken on v1.8.12. **Action**: needs investigation. Possibly: (a) the user's specific input audio produces a hallucination pattern not caught by current rules; (b) user's pip-upgrade didn't actually take effect (need to verify version display in their GUI); (c) different root cause entirely from #287's original case. Ask for: GUI version display, fresh attached log on v1.8.12, short audio sample if shareable. |
| **#286** | CUDA kernel error on GTX 1050 Ti | techguru0 | `AWAITING REPLY` | User replied 04-14 "will try it" re: PyTorch 2.5.1 downgrade. Waiting on result. |
| **#284** | v1.8.10 install stuck during PyTorch phase | qq73-maker | `AWAITING REPLY` | Follow-up 04-17: posted bilingual China-network template (community ask + mirror roadmap). |
| **#282** | Why does Ollama need GitHub connection? | KenZP12 | `AWAITING REPLY` | Responded 04-09: gist fetch for instructions, bundled fallback works offline. |
| **#280** | Qwen3-ASR TypeError: `check_model_inputs()` | zoqapopita93 | `AWAITING REPLY` | Responded 04-09: upstream `transformers` mismatch, suggested `pip install --no-deps transformers==4.49.0`. |
| **#274** | Pipeline aggregation mode question | cuixiaopi | `AWAITING REPLY` | Responded 04-09: explained 2-pass ensemble mode. |
| **#271** | Ollama translation model issues | justantopair-ai | `NEEDS FOLLOW-UP` | **FIX SHIPPED v1.8.11** (`275adb5`): curated model list + `--ollama-max-tokens`. User notification posted 04-21. **04-22: TinyRick1489 asked for `--ollama-num-ctx` flag** — reports batch auto-reduced to 11 with HF-sourced models. New ask on top of the resolved thread. |
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
| **#231** | Kaggle notebook error | fzfile | `NEEDS FOLLOW-UP` | Kaggle notebook fixes (llvmlite + TORCH_HUB_TRUST_REPO + translate command path) pushed 04-21. **04-22: fzfile replied** "Sorry for the late reply. I haven't used the translation function recently, **and there are still error prompts**." Need to verify whether user tested latest notebook or an older cached one. |
| **#227** | M1 Max transformer mode issues | dadlaugh | `STALE` | From 2026-03-17. Apple Silicon transformer mode. Not actively tracked. |
| **#225** | GUI white screen | github3C | `STALE` | |
| **#217** | GUI.exe not found (China network) | loveGEM | `AWAITING REPLY` | Follow-up 04-17: bilingual China-network template posted (community ask + mirror roadmap + whl workaround). User vimbackground had given up after whisper install also failed. |

### Feature Requests (37)

| # | Title | Priority | Target |
|---|-------|----------|--------|
| **#311** | Add FireRedVAD as a new VAD backend | MEDIUM | v1.9+ — **NEW 04-28.** kylesskim-sys. FLEURS-VAD-102 benchmark cited: F1=97.57 vs Silero=95.95 vs TEN=95.19; **FA rate 2.69% vs 9.41%/15.47%**. Industrial-grade SOTA from FireRedTeam (March 2026). Adds AED for speech-vs-music separation. Architecturally fits alongside silero/ten/whisperseg backends. Worth evaluating — could become the 5th shipped VAD if benchmark replicates on JAV. Use `tools/vad_groundtruth_analyser` to compare on Netflix-GT clip. `NEEDS RESPONSE`. |
| **#308** | Add English as source language for AI SRT Translate + ASS/SSA format support | LOW | v1.9+ — **NEW 04-23.** SangenBR. Two asks: (1) English source language in Translate dropdown (target already supports English), (2) read/write ASS/SSA subtitles (currently SRT-only). `NEEDS RESPONSE`. |
| ~~**#298**~~ | ~~Settings persistence for API/prompt fields~~ | — | **CLOSED 04-23** (yy739566004 agreed duplicate of #96, tracked there). |
| **#293** | Translation context / Whisper prompt feature | LOW | v1.9+ — **NEW 04-16.** User asks if movie descriptions/names as prompt would improve accuracy. Whisper and Qwen3-ASR both support initial prompt. Valid P2 ask. |
| **#292** | Low GPU utilization with higher-end models | LOW | v1.9+ — **NEW 04-17.** User running supergemma4-26b MoE on RTX 4090. Community (@justantopair-ai) already answered with good technical explanation. Probably just needs a thumbs-up from owner. |
| **#279** | Stash integration | LOW | Backlog — l34240013 provided detailed workflow (filename convention `<VIDEO>.<LANG>.srt`, auto-trigger points). **Responded 04-21**: thanked for integration detail, backlog-confirmed (no v1.9.0 timeline). |
| **#268** | Thai + Korean translation targets | LOW | v1.9+ |
| **#265** | Post-translation hallucination filter (Chinese) | MEDIUM | v1.9 |
| **#264** | Model download location customization | LOW | **NEEDS FOLLOW-UP** — **04-22: starkwsam replied** challenging maintainer's answer: "your answer about `HF_HOME=I:\hf_cache` / (e.g., `I:\cache` and `I:\cache\huggingface`) lacks coherence with the earlier detailed explanation — are you a pure AI reply or verified in practice?" Wants concrete set of env vars + folder-creation steps to relocate BOTH OpenAI Whisper cache AND HF cache to I-drive. Needs a humble, verified re-reply. |
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

### Outstanding Actions (rev43)

**Triage queue — 7 NEEDS RESPONSE + 4 NEEDS FOLLOW-UP (updated 2026-04-24 rev43):**

**⚠️ Cluster alert — v1.8.11 translation truncation on long files:**
- **#294** (CaliburnKoko, 116-min video → 8-min output) + **#302** (Kukuindi, truncated at ~31 min) — two independent reports on v1.8.11 with **attached logs on both**. Very likely a real regression. Priority P1 action: read the two log sets side-by-side, run a git-bisect if necessary between v1.8.10.post3 and the v1.8.11 merge commit. Fix candidate for v1.8.12.

**Other rev42 items still outstanding** (see rev42 list below):

**P0 / P1 — install + translation regressions:**
- **#300** (ktrankc, install regression) — reproduce with `pip install "whisperjav[all] @ git+…v1.8.11"` on Python 3.10/3.11 via uv. Probably needs either pinning `requests>=2.31` in the `[torch]` extra's resolution hint or recommending `--index-strategy unsafe-best-match`. User workaround works (checkout post3), so this isn't blocking everyone — but it IS blocking fresh clean installs of v1.8.11 via uv.
- **#294 + #302 CLUSTER** (v1.8.11 long-file translation truncation) — both users confirm on v1.8.11 with attached logs. P1. Diagnose first, reply second; the root-cause finding informs both replies.
- **#306** (ktrankc, anime-whisper ensemble produces 13k-char hallucination) — clustered with #246. Anime-whisper needs explicit `chunk_length_s` + `return_timestamps` tuning when used in Pass 2 of ensemble. Suggest reducing chunk_length or switching Pass 2 to kotoba.
- **#305** (ric-reff, CN target producing English chunks) — SAME root cause as #271. User is running `shisa-v2.1-qwen3-8b` which was blacklisted in v1.8.11 curated list (thinking model). Tell them: upgrade to v1.8.11 and use `qwen2.5:7b-instruct` or `translategemma:12b`.
- **#231** (fzfile follow-up) — ask user which notebook they ran (Kaggle in the repo vs. cached). Fixes are in `notebook/WhisperJAV_kaggle_parallel_edition.ipynb` on main; if they're on an older cached notebook the error persists.
- **#271** (TinyRick1489 follow-up) — `--ollama-num-ctx` is a reasonable one-liner flag. Consider for v1.8.12 or v1.9.0 scope.

**P2 — clarifications + new:**
- **#307** (yedkung69-ctrl, sanitization principles Q) — friendly question. Good reply candidate: short tour of the sanitizer pipeline (regex/exact-match gists, CPS filter, symbol-only drop, hallucination-pattern repo) + pointer to `hallucination_remover.py` in the code. Could become a FAQ / docs section long-term.
- **#308** (SangenBR, English source + ASS/SSA) — feature request. Reply: English source is a small GUI dropdown fix for v1.8.12 or v1.9.0; ASS/SSA format support is a broader subtitle-I/O track (connects to #43 DeepL and #44 GUI drag-drop — bigger work).
- **#304** (wenkine-2026, fidelity on GTX 1050 Ti) — hardware incompat. CC 6.1 not supported by current PyTorch wheel (needs 7.5+). Point to #286 thread or a FAQ. Consider documenting min GPU compute capability.
- **#264** (starkwsam, cache paths challenge) — humble factual re-reply needed. Verify `XDG_CACHE_HOME` actually moves Whisper cache on Windows (check in a clean env). Give exact System Environment Variables steps with folder-create instructions.

---

### Previous batch (rev40-rev41, 2026-04-21) — All replies posted

**10 replies posted across P0/P1/P2 queues.** Comment links below:

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

**Board state after rev41 batch**: 0 NEEDS RESPONSE, 0 NEEDS FOLLOW-UP, 22 AWAITING REPLY, 3 SHIPPED-awaiting-confirm.

**Board state at rev42**: 5 NEEDS RESPONSE, 3 NEEDS FOLLOW-UP, 22 AWAITING REPLY, 2 SHIPPED-awaiting-confirm.

**Board state at rev43**: 7 NEEDS RESPONSE, 4 NEEDS FOLLOW-UP, 21 AWAITING REPLY, 2 SHIPPED-awaiting-confirm.

**Board state at rev44 (04-25 after #294 + #302 forensic + responses)**: **6 NEEDS RESPONSE, 3 NEEDS FOLLOW-UP, 23 AWAITING REPLY, 2 SHIPPED-awaiting-confirm.**

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
| **2026-04-30** | **rev45.** **v1.8.12 RELEASED.** Tagged, merged, pushed. GitHub Release published with .exe installer (Windows) + wheel (cross-platform). 20 commits since v1.8.11. Theme: WhisperSeg ONNX VAD + tight defaults retune. **Headline**: WhisperSeg backend (F1=0.787 on Netflix-GT JAV vs ten=0.698 / silero-v6.2=0.654 / silero-v3.1=0.625, aggressive sensitivity). **Critical fixes**: F5 best_of regression (1→2 for aggressive faster-whisper, verified by F6 acceptance test Pass 1 recall 19%→88%); silero-v4.0→v3.1 fallback alignment in legacy ASR modules; anime-whisper ellipsis-only line filter (text + SRT level with renumbering); TEN max_speech schema fix; anime chunk_threshold plumbing; --qwen-segmenter accepts whisperseg. **Defaults changed**: aggressive no_speech_threshold 0.77→0.84; aggressive best_of 1→2 (regression revert); aggressive beam_size 2→3; VAD max_speech/max_group sensitivity gradient INVERTED (aggressive=tightest at 4/5s, conservative=loosest at 6/7s). **NEW issues since rev44 (+3 net)**: **#309** white screen mklink (Jerry199022 04-26 — exhaustive diagnostic, even pure pywebview fails — likely upstream WebView2/symlink issue), **#311** FireRedVAD feature request (kylesskim-sys 04-28, FLEURS-VAD-102 benchmark cited, P2 v1.9+), **#312** VAD param dump_params artifact (TinyRick1489 04-28, sharp investigation — dump shows silero values for ten because legacy resolver always emits silero presets; runtime is correct via firewall, metadata is misleading). **3 issues flipped AWAITING REPLY → NEEDS FOLLOW-UP**: **#287** (zoqapopita93 04-25 — sanitizer fix didn't help, still all-`!!`); **#294** (CaliburnKoko 04-26 — no_speech=0.85 advice didn't help on balanced/aggressive, but fast/aggressive default DID work; identifies regression specifically in balanced/aggressive since v1.8.10.hotfix); **#302** (Kukuindi 04-22 — no_speech=0.85 advice made it WORSE, only 1m37s translated; teijiIshida 04-28 added comment fingering silero-v4.0 as random-fail culprit). **v1.8.12 expected to fix #294, #302, and possibly #297** via the combined fix chain (best_of + no_speech + silero fallback). #287 needs deeper investigation as v1.8.12 didn't add sanitizer changes. **v1.8.13 candidates section added** with: ctranslate2 state contamination (NEW finding from F5/F6 forensic), installer PATH (carry-over), ZipEnhancer Colab init (#290), Qwen3-ASR transformers pin (#280), --ollama-num-ctx (#271 follow-up), VAD param dump_params artifact fix (#312). **Board state**: **6 NEEDS RESPONSE, 6 NEEDS FOLLOW-UP, 19 AWAITING REPLY, 8 SHIPPED-in-v1.8.12 (mostly awaiting user retest).** |
| **2026-04-25** | **rev44.** Forensic log-read on **#294 + #302** (v1.8.11 "translation truncation" cluster). Result: **not a translation bug, not a regression unique to v1.8.11**. Both runs finished normally — pipeline: full-duration audio extract → auditok scene detect (278 / 203 scenes) → silero-v4.0 VAD finds speech in 94% / 100% → faster-whisper-large-v3 aggressive preset (`no_speech_threshold=0.77`) silently classifies most whispered/breathy scenes as silence. Result: 33% / 11% VAD-segment→subtitle conversion. #294 raw stitched contained real dialogue 0-8 min + 16 `Chili` hallucinations at 50-58 min + empty 42+58 min stretches. Secondary finding: **`BalancedPipeline` creates `silero-v4.0` despite v1.8.10.post3 defaulting to silero-v3.1** — config resolver logs v3.1 but factory call uses v4.0 with `version='v4.0'` param. Requires code-level check. Response sent to both users: raise `no_speech_threshold` from 0.77 → 0.85 via Ensemble tab → Customize Parameters (GUI route). #294 and #302 flipped to AWAITING REPLY. v1.8.12 candidate: if users confirm 0.85 helps, promote aggressive default in `components/asr/faster_whisper.py` and `openai_whisper.py`. Separately, consider one-line CLI flag `--no-speech-threshold` for easier user experimentation. **Board state**: **6 NEEDS RESPONSE, 3 NEEDS FOLLOW-UP, 23 AWAITING REPLY, 2 SHIPPED-awaiting-confirm.** |
| **2026-04-24** | **rev43.** 69→70 open. **+2 new issues** since rev42: **#307** (yedkung69-ctrl — polite Q about JA SRT sanitization pipeline principles; good documentation opportunity), **#308** (SangenBR — feature: add English as source language in AI SRT Translate + support ASS/SSA subtitle format). **1 closed** (#298 yy739566004 closed 04-23 agreeing it's a duplicate of #96, as requested 04-21). **#294 flipped to NEEDS FOLLOW-UP**: CaliburnKoko replied 04-23 with 5 log attachments — reports **116-min video truncated to 8-min output on v1.8.11**. **Cluster alert**: #294 + #302 are two independent v1.8.11 translation-truncation reports with attached logs — probable real regression, warrants git-bisect. **Board state**: **7 NEEDS RESPONSE, 4 NEEDS FOLLOW-UP, 21 AWAITING REPLY, 2 SHIPPED-awaiting-confirm.** |
| **2026-04-24** | **rev42.** 65→69 open. **+5 new issues** since rev41: **#300** (ktrankc — v1.8.11 uv install deadlock, `bs-roformer-infer requests>=2.31` vs pytorch index `requests==2.28.1`; user reverted to post3), **#302** (Kukuindi — v1.8.11 does not translate whole file, slower than v1.8.10, logs attached), **#304** (wenkine-2026 — fidelity mode unusable on GTX 1050 Ti; PyTorch sm_75+ incompat, same pattern as #286), **#305** (ric-reff — CN target producing English chunks when using `shisa-v2.1-qwen3-8b`, exactly the thinking model blacklisted in v1.8.11), **#306** (ktrankc — anime-whisper Pass 2 in ensemble produces 13k-char hallucination → 0 subs after sanitization, clusters with #246). **1 closed** (#301 Kukuindi self-closed, merged into their #302). **3 issues flipped to NEEDS FOLLOW-UP** since rev41: #271 (TinyRick1489 asks for `--ollama-num-ctx` flag on 04-22), #231 (fzfile replied 04-22 "still error prompts" for translation, needs notebook-version clarification), #264 (starkwsam challenges cache explanation 04-22, questions if owner's reply is "pure AI"). **Board state**: **5 NEEDS RESPONSE, 3 NEEDS FOLLOW-UP, 22 AWAITING REPLY, 2 SHIPPED-awaiting-confirm.** Separately: `dev_v1.8.12` has 6 commits on top of v1.8.11 (WhisperSeg VAD backend added, new `tools/vad_groundtruth_analyser` utility, GUI wiring, dropdown cleanup, neg_threshold purge) — NOT released. |
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
