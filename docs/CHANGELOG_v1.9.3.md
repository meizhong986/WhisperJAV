# WhisperJAV v1.9.3 — Development Change Log

> Developer-facing record of what changed on `dev_v1.9.3`, why, and how it was
> verified. `docs/release_notes_v1.9.3.md` (not yet written) will be the user-facing
> text derived from this file. Same conventions as `docs/CHANGELOG_v1.9.2.md`: one
> entry per landed change, newest first; "Decision" lines record who decided what,
> so a later reader can tell policy from mechanism.
>
> **SYNC** — requirements doc `docs/requirements/v1.9.3_vision_mission_highe-leve-features_v1.txt` **v1** (mtime 2026-09-14 14:20, 3,946 bytes, md5 `736f0c97…`) | release plan **`docs/plans/V193_RELEASE_PLAN.md` rev 1.5** (working record; **owner-facing planning moved 2026-09-16 to `docs/plans/V193_MATTERS_MAP.md` rev 2** — the map of matters, his decisions verbatim, the 7 phases; §2.1 and the D1–D43 table are superseded) | tracker rev **53.0** | change log **`docs/CHANGELOG_v1.9.3.md` through 2026-09-16, phases 1–4** (**committed 2026-09-16 on the owner's word**, together with the v1.9.2 log's SYNC edit; the v1.9.2 log is closed at its 2026-09-14 entry) | owner pack: the v1.9.2 pack `73c5c95d` stays at r4.5; **the v1.9.3 pack opens at the Phase 2 scope gate** (owner, 2026-09-14) | branch **`dev_v1.9.3` @ `ed79c1d`** (PRs #388 #363 #376 #364 merged locally 2026-09-16; then phases 2–4: `ac902b4` F1, `7788a2b` F2, `d9afb7e` F3, `58b9f97` F4, `19c20d5` D1, `25863cf` G1, `ed79c1d` C2), `main` @ `e8c1c6e`, `origin/main` @ `63f256e` = tag `v1.9.2`; **nothing pushed — owner decision 2026-09-14: no push until the 1.9.3 release** | GitHub 2026-09-16: **32 comments posted** (25 phase-1 non-G2 + 4 G2 + #366 retest + #305 + #397) and **5 closures** (PRs #365 #362 #361 #360, issue #397); every comment id in `docs/plans/V193_PHASE1_POSTED.log` | **Owner 2026-09-15: F4 (MOSS) dropped from 1.9.3 → next release (perhaps 1.10); step-4 postings APPROVED and POSTED (ten comments + issues #429 #430); replan without MOSS ordered — **scope proposal produced (D2.0) and adversary-gated (25 findings applied); it sits in the release plan §2.1 with 43 numbered decisions for the Phase 2 sitting.** Phase 0 gate (document set, Q5–Q10) still open.**
> Rule: a session that changes the release plan, the tracker, this file or the pack brings the others to the same state before it ends (CLAUDE.md rule A7).

---

## 2026-09-16 — phase 4 continued: three more translation targets; the dead Transcription-tab translation bindings removed

**Decisions (owner, 2026-09-16):** on adding languages, *"yes if minimal work"*; on the Transcription tab,
*"yes the transcription tab needs working translation controls. However it is a bigger job to design GUI and
it should be parked for [1].10.x release"*; on the dangling element ids, *"please correct"*.

- **Italian, Thai and Korean added** (`3ee80a8`). Requested on #351 (Italian) and #268 (Thai); Korean added
  alongside. The work was minimal because **there is no allow-list to satisfy**: PySubtrans substitutes
  `target_language` into the prompt as a free string (`Options.py:301-304`), so a target costs the choice and
  nothing else — per-language quality is the model's, not WhisperJAV's. `SUPPORTED_TARGETS` goes from six to
  nine; `main.py`'s choices and both GUI dropdowns follow. Verified: all three exit 0 through the real CLI
  where they exited 2 before; `tests/test_translate_targets.py` still passes, and it fails if one of the four
  lists is updated without the others.

- **Five dead translation bindings removed from `app.js`** (`5197fd1`). `translateAfterTranscription`,
  `translateQuickSettings`, `quickTranslateProvider`, `quickTranslateTarget` and `ensembleTranslateTarget`
  exist in no asset file, so every binding on them was a silent no-op. The consequential one sat in
  `isEnabled()`, which therefore always evaluated false on the Transcription tab and Advanced Options —
  translation could not be switched on from either. It now returns `false` explicitly, so the behaviour is
  stated rather than reached by accident, and the comments point at the 1.10.x work.
  **No user-visible change: every removed path was already inert.** Giving those tabs real translation
  controls is parked for 1.10.x by the owner's decision above.
  **No reporter exists for this:** 28 threads were searched for anyone describing a missing or unreachable
  translation control and none matched — the gap was found by reading the code, not from a report.

- **D1, the half that needs no GUI: verified.** The page now sends 17 keys for the Ensemble tab (14 controls
  + `rememberSettings` + the two preset names). Driving the real `WhisperJAVAPI.save_gui_settings` /
  `get_gui_settings` through a full save → file → load cycle, **17/17 survive** camelCase → snake_case → disk
  → back. Run against a sandboxed `APPDATA`; the owner's real
  `%APPDATA%/WhisperJAV/gui_settings.json` was verified byte-identical before and after (md5 `0ada9602…`).
  The click-through (set, tick Remember settings, restart, confirm) is still owed and needs the GUI.

## 2026-09-16 — phase 4: the French target defect fixed; C1 was already resolved on 2026-08-29 and the comparison re-measured that fix; one new defect found and left open

**Decision (owner, 2026-09-16):** "go ahead with step 4."

- **C1 step one (map row C1, his #393 note) — done; it closes step one but does NOT identify the cause.**
  At runtime `get_instruction_content()` resolves **Gist → cache → bundled**
  (`whisperjav/translate/instructions.py:170-236`), so a prompt change must be made in the Gist to reach
  users — which is what #393 warns about. As of 2026-09-16 the live Gists match the bundled defaults:
  `pornify` **byte-identical**; `standard` identical after newline normalisation (54 CRLF vs LF, zero
  differing lines — and line endings cannot reach the prompt anyway, `InstructionsHelpers.py:42-43` strips
  every line). Neither Gist nor any bundled file carries a **`### target_language` section** — checked
  explicitly, because `InstructionsHelpers.py:70` feeds `Options.py:326-327`, where one such section would
  force every user of that tone to a fixed language regardless of what they asked for. `contextual` has no
  Gist and is bundled-only, so for that tone the comparison says nothing.
  **Why they match today — the cause was found and fixed on 2026-08-29, by the owner, before this phase began.**
  The GitHub threads (read at the end of this phase, after the code work) record it: on #397 he established that
  the `pornify` Gist was an eight-month-old copy predating `5ada48a`, telling users to write "like a raunchy
  American adult movie" with **worked examples entirely in English**, and he synced the Gist to the repository
  version the same day. The Gist revision history confirms it: `bd34016a…` was rewritten
  **2026-08-29T16:21:47Z, +59/−50**. Because instructions are fetched per run, that fix reached every user
  immediately, with no release. **The comparison ordered in map row C1 therefore measured the owner's own
  repair, three weeks after the fact** — the matching files are the evidence that C1 was already dealt with,
  not evidence that the instruction files were never the cause.
  **Confirmed by the reporter:** yhxkry retested on 2026-08-31 with a local LM Studio model and reported correct
  Chinese output. **C1's reported symptom is resolved on the thread that raised it.** Nothing was changed in
  this phase on C1's account, and no prompt change is proposed; the Gist text remains the owner's to edit.
  **Process note for a later reader:** this session reached "the files match, so the instruction files are not
  the cause" from the code alone, and offered an English-few-shot-exemplar hypothesis that was in fact a
  rediscovery of the already-fixed cause. Reading the two threads first would have prevented both. An
  `assessment-adversary` review caught the overreach (its caveat — "today's Gist is not the Gist those users
  ran" — was exactly right) but could not supply the thread history either.

- **C2 — French was selectable in the GUI and rejected by the CLI.** `SUPPORTED_TARGETS`
  (`translate/providers.py:85`) has six targets including `french`, both GUI dropdowns offer French, and
  `whisperjav-translate --target` derives its choices from that set — but `main.py`'s `--translate-target`
  carried a hand-kept list of five. Choosing French in the GUI therefore built `--translate-target french`
  (`webview_gui/api.py:3355`) and the run died before it started with
  `invalid choice: 'french'` and **exit 2**. Reproduced on that exact command and confirmed fixed (exit 0),
  with `spanish` as the control both times; `--help` now lists `french`.
  Also fixed: `translate/service.py:344`, the hand-kept list used to strip an existing language suffix from
  an output name, was missing `portuguese` and `french`, so re-translating `x.french.srt` produced
  `x.french.french.srt`. It now derives from `SUPPORTED_TARGETS`.
  `tests/test_translate_targets.py` — 5 passed; it holds `main.py`'s choices, both GUI dropdowns and the
  suffix list to `SUPPORTED_TARGETS` so they cannot drift apart again.

- **OPEN, and the one genuinely new finding of this phase: a stale `.subtrans` can return the previous run's
  English.** Unrelated to the instruction files, never previously reported or asked about, and **not fixed**.
  The project file next to the input is deleted **only** when the stamped `whisperjav_version` differs or the
  file is unreadable (`core.py:466-491`; stamped at `:807-818`) — the block contains **no reference to the
  target language at all** (verified: zero occurrences of "target" in it). `core.py:587` then sets
  `translator.resume = True` unconditionally and `SubtitleTranslator.py:124-126` skips every already-translated
  scene. So: translate a file once with the default target (**english** — `translate/settings.py:20`), then
  re-run the same file asking for Chinese on the same WhisperJAV version, and the earlier English lines are
  written to the Chinese output path (`core.py:502-503`). Deleting the `.subtrans` by hand is the way round it.
  This fits #305's wording precisely — he reported "**部分**翻译内容变成英文", *part* of the output in English,
  which is what skipping already-translated scenes produces and which the instruction-file cause does not
  explain. **Whether WhisperJAV should clear the `.subtrans` when the target language changes is an owner
  decision** (it is a behaviour change to resume), so nothing was altered. #305 has been asked the one question
  that decides it; no instrument is needed, since a run already prints "Resuming from existing project file
  (.subtrans)" (`core.py:506-512`).

- **Postings (owner's word, 2026-09-16: "do the postings").** #305 — a follow-up to his own 2026-08-29 retest
  request plus the `.subtrans` question, thread left **open** pending his answer. #397 — closed, with a note
  recording that yhxkry confirmed the fix on 2026-08-31 and that @jasmineamber, asked in the same comment,
  never replied; the note makes **no fix claim** (#397 is on the never-say-fixed list), credits yhxkry with
  finding the cause, and says a reply will reopen it. The owner's instruction was "close with the note that
  closed due to no response"; the note attributes the silence to @jasmineamber rather than to yhxkry, who did
  respond — the instruction stands, the attribution was corrected so the public text is true.
  **No other C1 recipients exist:** 24 translation-adjacent open issues were checked and only #397 and #305
  report the symptom; the map's "7 reporters" counts participants across those two threads (yhxkry,
  loggias06row, jasmineamber, ric-reff), not seven issues.

- **Noted for an upstream report, not C1.** PySubtrans `Instructions.py:89-92` builds tag keys that already
  contain their brackets and `ReplaceTags` (`:104-108`) then searches for `[{name}]`, i.e. `[[ to language]]`
  — so **every tag substitution inside `InitialiseInstructions` is a no-op**. Harmless today, but
  `Instructions.py:94` injects *every* settings key as a substitutable tag, `api_key` included; if upstream
  ever fixes `ReplaceTags`, instruction text containing `[api_key]` would paste the key into the outgoing
  prompt. WhisperJAV is unaffected because `PySubtrans/__init__.py:93-96` re-injects the caller's explicit
  prompt and `Options.BuildUserPrompt` (`Options.py:297-311`) does the real substitution from
  `target_language`, which WhisperJAV sets at `core.py:318`.

- **Found while tracing C1/C3 — the Transcription tab has no working translation control.** `app.js` reads
  five translation element IDs that `index.html` never defines: `translateAfterTranscription` (7436, **7525**),
  `translateQuickSettings` (7437), `quickTranslateProvider` (7457, 7564), `quickTranslateTarget` (7458, 7565)
  and `ensembleTranslateTarget` (7473). Line 7525 is inside `TranslateIntegrationManager.isEnabled()`, so on
  tab 1 and tab 2 that method reads a null checkbox and **can never return true** — translation cannot be
  enabled from the Transcription tab at all. The settings branch behind it
  (`getSettings()`, 7564-7565) would have forced `provider: 'local'` and `target: 'english'` regardless of
  the user's choice, but is unreachable for the same reason. Translation works only from the Ensemble tab,
  where the target is mapped correctly (`target: fullSettings.targetLang`, 7548).
  **Reported, not fixed** — what the Transcription tab should offer is the owner's decision.

## 2026-09-16 — phase 3: the Linux GUI prerequisite is stated where a Linux user will see it (G1)

**Decision (owner, 2026-09-16):** "G1 is a yes"; then "proceed to step 3".

The failure in #366 is that following the distro instructions is not enough: pywebview ships **no Linux
backend by default**, and apt/dnf WebKit packages land system-wide where a plain virtual environment cannot
see them, so `whisperjav-gui` stops with `ModuleNotFoundError: No module named 'gi'`. Both routes out were
already written down (`docs/en/guides/installation_linux.md:180-199` and `:764-772`) but appeared nowhere a
user meets before hitting the error: `installer/install_linux.sh` had no GTK/Qt/WebKit line at all, and
`README.md:399` said only "GUI needs WebKit2GTK".

- **`installer/install_linux.sh`** — the success banner now prints the symptom and the two routes: the Qt
  backend (`pip install "pywebview[qt]"`), or recreating the venv with `--system-site-packages` so it can
  see the distro's bindings, plus a pointer to the full guide. **It prints; it installs nothing** — which
  route is right depends on the user's environment, and the script must not choose for them.
  Verified: `bash -n installer/install_linux.sh` parses; the banner was rendered from the shipped block and
  reads as intended.
- **`README.md`** — the bare "GUI needs WebKit2GTK" sentence is replaced by the reason (system-wide packages
  are invisible to a venv), the symptom, and the same two routes.

No Python changed; no installer behaviour changed.

## 2026-09-16 — phase 2: the four "what the user sees while a run is working" items, and the two-pass settings save

**Decision (owner, 2026-09-16):** "we have done planning of 1.9.3. We now implement the plan step by step…
we proceed with the frozen view and complete step by step." F1–F4 and D1 are map §5 step 2.

- **F1 — progress lines read `[839/1]` in a multi-file run (#429).** `ProgressDisplayAdapter.__init__` set
  `self.total_files = 1` and nothing ever updated it; the real count was set on the *manager*
  (`whisperjav/main.py:1190`), and the adapter is what feeds
  `UnifiedProgressManager.start_file_processing`, which prints the line
  (`whisperjav/utils/unified_progress.py:249`). The adapter now takes the count as a constructor argument,
  falling back to the manager's value and then to 1. `create_progress_adapter` passes it through.
  Verified: `tests/test_progress_file_count.py` — 6 passed (the fallback case fails on the old code);
  and a two-file run driven through the real manager prints `[1/2] Starting: one.mp4`,
  `[2/2] Starting: two.mp4`. `--no-progress` uses `DummyProgressAdapter` and is untouched.
  Files: `whisperjav/utils/progress_adapter.py`, `whisperjav/main.py`.

- **F2 — the English post-processing stage was silent (#372 follow-up).** The three steps that can block —
  loading the hallucination filter (a network fetch), cleaning, writing the final SRT — were `logger.debug`
  and so invisible without `--debug`. Raised to `logger.info` at `srt_postprocessing.py:226,:241,:246`.
  The two cheap steps either side (preparing the work dir, backing up the original) stay at debug.
  Verified: a real English-target post-processing run on a three-line SRT prints all three lines at INFO
  (the logger's default level) and produced the cleaned file.

- **F3 — audio extraction was silent and untimed.** `audio_extraction.py:45` is now INFO, and the success
  line reports how long FFmpeg took. **No timeout was added** (planner + owner): extraction of a long file
  on a slow or sleeping drive legitimately runs for minutes and killing it would lose the run.
  Verified: a real extraction of a synthesised 3-second clip prints
  `Extracting audio from sample.mp4` and `Audio extracted: sample.wav (duration: 3.0s, took 0.1s)`.

- **F4 — a folder of input can quietly include WhisperJAV's own leftovers.** Discovery walks folders
  recursively, so a working folder yields intermediates from an earlier run. The listing now labels files
  shaped like WhisperJAV's own output (`*_extracted.wav`, or under a `scenes` folder — the shapes come from
  `base_pipeline.py:65,:82`) and adds one summary line giving the file and folder counts and saying that
  folders were searched recursively. **Nothing is removed from the list and nothing prompts** — a user's own
  file may carry any name and sit in any folder. Verified: a real CLI run over a folder holding one ordinary
  file and two leftover-shaped files printed the two labels and the summary, and processed **all three**
  (`done 0  empty 3  … total 3`), which is the guarantee that matters.
  Verified also: `tests/test_media_listing_leftovers.py` — 11 passed; `--help` still exits 0.
  Files: `whisperjav/main.py`.

- **D1 step one — the Ensemble tab's two-pass settings were not saved (#381).** The Python side has
  accepted and returned all 16 ensemble keys since 1.9.2 (`webview_gui/api.py` `_GUI_SETTINGS_MAP`); the
  page side listed none, so "Remember settings" lost the whole two-pass setup on restart. Added the 14
  form controls to `SettingsPersistence.FIELDS` (`pass1-*`, `pass2-*`, `merge-strategy`), and the two
  preset *names* — which live on `EnsembleManager.state`, not in the DOM — through `collectAll` and the
  `restorePresets` stub, now implemented and called from `loadFromBackend` after the selectors are back.
  Restoring a preset name only restores the label; it never re-applies a preset over the restored values.
  Verified: `node --check` parses `app.js`; all 16 backend ensemble keys are now covered on the page side
  (checked by comparing `_GUI_SETTINGS_MAP` against `FIELDS` plus the two explicit keys).
  **Not verified by clicking:** the map's named test, `tests/test_gui_refactor.py`, imports
  `whisperjav.gui.whisperjav_gui`, a module removed at `c354d1d` when the webview GUI replaced it — the file
  has been dead since then and cannot check this. The set-toggle-restart click-through is still owed.
  Files: `whisperjav/webview_gui/assets/app.js`.

**Not done:** D1 step two (the Customize modal values — #381's second half) and D2 (what the Customize panel
shows after FireRedVAD) both need the GUI opened and clicked, which this session could not do.

## 2026-09-16 — phase 1 of the matters map: four contributor pull requests merged locally; labels applied

**Decision (owner, 2026-09-16):** the 43-row decision table was rejected ("trees but nothing about forest");
owner-facing planning now lives in `docs/plans/V193_MATTERS_MAP.md` (matters A–J, his decisions verbatim,
priority I → H → E → D → G → F → C → B → A → J, seven phases). "Go on phase 1 confirmed"; "Please process H1
clean PRs"; "I6 is yes" (labels); "G1 is a yes" (Linux prerequisites, phase 3); E1: the merge tool stays 2.x,
weifu8435 is told "let me look into it" and a standalone utility of the owner's is a phase-6 item.

**Merged locally into `dev_v1.9.3`** (every open PR targets `main`, which is not pushed until the release, so
the authors are told at the release push, not now):
- `784696d` **PR #388** (AKB0700) — `.github/workflows/automated-execution.yml`: compile + bare import on
  Python 3.12 on push / pull request. Never run in this repository yet; first run at the release push.
- `6ff5175` **PR #363** (triatomic) — an "Auto-scroll" checkbox in the GUI console header
  (`webview_gui/assets/index.html`, `app.js` `ConsoleManager._autoScroll`): unchecked, new lines no longer
  pull the console to the bottom. **Not verified by clicking it; the diff was read (18/−10 lines).**
- `e626d05` **PR #376** (Mimic-me) — `whisperjav/bench/regression.py`, `regression_cli.py`, `tests/bench/`,
  `bench/` scaffolding, and a new console command `whisperjav-accuracy-gate` in `pyproject.toml`. No runtime
  effect on transcription. Verified: `py_compile` both modules; `tests/bench/test_regression.py` 19 passed;
  `python -m whisperjav.bench.regression_cli --help` prints the three sub-commands. (The console-script entry
  appears on PATH only after the package is reinstalled.)
- `c5405b5` **PR #364** (triatomic) — a `contextual` translation tone: `translate/defaults/contextual.txt`,
  `--tone` / `--translate-tone` choices, the two GUI tone dropdowns. **Also changes behaviour:** a temperature
  saved in the translation settings file now applies only when the run's tone equals the tone the file was
  saved with (before, a saved temperature overrode every tone's default); Ollama's curated temperature no longer
  replaces the contextual/pornify defaults. Verified: `py_compile`; both `--help` outputs list the new tone.
  The online instruction files exist only for `standard` and `pornify` (`translate/instructions.py:15-18`), so a
  contextual run uses the bundled file and logs one line "No URL configured for tone: contextual".

**Labels (owner: "I6 is yes"):** the nine existing labels applied to 139 open issues from the register's kind
(bug 65, enhancement 48, question 22, documentation 4); the eight settings-recipe threads (#410 #265 #374 #386
#387 #390 #413 #423) and #431 left unlabelled — no existing label fits. Nothing else posted.

**Not done:** the phase-1 reply texts (`docs/plans/V193_PHASE1_TEXTS.md`) — drafted from the owner's words,
checked, approved by him 2026-09-17 ("go on all the texts"; the four to weifu8435/yangming2027 held per his grouping), posting pending; the H2 comments to the seven PR
authors (same file, same gate); the #363 GUI click.

## 2026-09-15 — MOSS deferred to the next release; the owed acknowledgements posted; replan ordered (docs only)

**No code change.** **Decision (owner, 2026-09-15):** requirement F4 — the MOSS-Transcribe-Diarize ASR — is
dropped from 1.9.3 and moved to the next release ("perhaps 1.10"). The architect's option record and its
adversary gate (`docs/plans/V193_SESSION_DATA_2026_09_14/D1.5_F4_MOSS_design_options.md`) are kept for that
release; the decisive facts: the package requires `transformers>=5.6.0`, the shipped stack resolves to 4.57.6,
memory for a 65-minute single pass is about 11 GB at the published token budget, and a 3-hour film exceeds the
model's context in one pass.

**Posted, on the owner's approval:** one-line acknowledgements on nine threads that had never been answered
(#424 #417 #422 #418 #423 #420 #410 #425 #426) and the owner's appreciation message on #374; **issues #429**
(progress lines read `[839/1]` — the progress adapter is created with `total_files = 1`,
`whisperjav/utils/progress_adapter.py:21`, and nothing updates it; cosmetic; no fix scheduled) **and #430** (the
clustering crash fixed in `e8c1c6e`; left open until the release) opened. Comment ids in the tracker, rev 53.0.

**Decision (owner, 2026-09-15):** replan v1.9.3 without MOSS with exact PRs, issues and features and their
reasons. Done the same day: the scope proposal was produced by a dedicated planner
(`docs/plans/V193_SESSION_DATA_2026_09_14/D2.0_v193_scope_proposal.md`), adversary-gated (25 findings, applied),
and placed in the release plan §2.1 — 15 work packages, dispositions for all 11 PRs, the issue list, features
in/out, notebook and China-install slices, an ordered sequence, and 43 numbered owner decisions. Nothing in it
is started; it waits for the owner's decision sitting.

## 2026-09-14 — v1.9.3 planning opened; records brought to one state (docs only)

**No code change.** The owner delivered the v1.9.3 requirements document and asked for the
document organisation, a phase-based plan and the immediate steps. The approved plan is
`docs/plans/V193_RELEASE_PLAN.md` (rev 1.1 after the adversary gate). Created today: this file; the release plan; tracker rev 52.9.
`docs/plans/V193_PLAN_PROPOSAL.md` (rev 3, 2026-09-13) is superseded and carries a pointer.

**Decisions (owner, 2026-09-14), verbatim in the release plan §4:** the 1.9.2 VAD / Silero loader /
adapters / Balanced freeze stays for this cycle; the F4 (MOSS-Transcribe-Diarize) tech-stack and design
decision goes to a dedicated planner/architect, not to the orchestrating session; the v1.9.3 owner pack
opens at the scope gate; no push until the 1.9.3 release.

**Facts recorded for Phase 1, verified in this tree (no change made):**
- Both live notebooks fail by default on 1.9.2 code: `whisperjav/main.py:2094-2099` exits 2 for
  `--mode balanced` with any `--speech-segmenter`; `notebook/WhisperJAV_colab_edition_expert.ipynb`
  defaults to `balanced` + `whisperseg` and emits the flag; the Kaggle notebook emits it the same way.
  `installer/install_colab.sh:26,179` installs `@main`.
- Installer: `installer/templates/post_install.py.template:560-571` probes pypi.org only and is fatal at
  `:2668-2670`, before proxy detection at `:2735`; PyTorch index URLs hard-coded at `:197,:203`;
  `docs/en/faq.md:330` points China users to a note that does not exist.
- MOSS-Transcribe-Diarize (F4), read on the owner's instruction from the Hugging Face repository as well as
  GitHub: the package pins `transformers>=5.6.0,<6.0.0` (GitHub `pyproject.toml`); "50+ languages", and the
  HF model card names **Japanese** among the 14 MLC-SLM Challenge languages (an earlier statement this
  day that Japanese was not named was true of the GitHub README only — corrected); whole-file single-pass
  inference, 128k context, up to 90 minutes; 1.82 GB bf16 weights; no VRAM figure published. WhisperJAV's
  `pyproject.toml:241` pins `transformers<5.0` with the comment "Removal deferred to v1.9.x (needs Qwen3-ASR
  fork patch + Cohere ship …)" — whether the cap moves in 1.9.3 is an owner decision not yet taken. Full
  record: release plan §6.1. An adversary gate (g1) ran on the planning records and its 18 findings were
  applied (release plan §7.1).

## Carried from `dev_v1.9.3` before this log existed

- **`e8c1c6e` — keep a clip that is too short to split as one scene, instead of failing it.**
  `whisperjav/vendor/semantic_audio_clustering.py` (`WJAV mod E`, 13 insertions / 4 deletions): fewer
  than two strided feature rows (an input under ~0.48 s — 16 centred frames at hop 512 / 16 kHz) now skips `AgglomerativeClustering`, logs one
  INFO line and uses a single-cluster label array. Ships with 1.9.3 by the owner's decision of 2026-09-14
  (hotfix `v1.9.2.post1` prepared then abandoned). Full mechanism, verification table and the decision:
  `docs/CHANGELOG_v1.9.2.md`, entry "2026-09-14 — the three post-release matters closed"; record
  `docs/plans/V192_HOTFIX_PLAN.md` §4; git operations `docs/plans/V193_BRANCHING_RECORD.md`.
- `7dd00fc`, `3b92e6c`, `920f910`, `8602e8d` — documentation commits of the v1.9.2 post-release sessions
  (`.gitignore` for `docs/requirements/`, the v1.9.2 change-log entries, the rewritten user release note).
