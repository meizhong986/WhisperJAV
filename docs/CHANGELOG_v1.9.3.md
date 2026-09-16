# WhisperJAV v1.9.3 — Development Change Log

> Developer-facing record of what changed on `dev_v1.9.3`, why, and how it was
> verified. `docs/release_notes_v1.9.3.md` (not yet written) will be the user-facing
> text derived from this file. Same conventions as `docs/CHANGELOG_v1.9.2.md`: one
> entry per landed change, newest first; "Decision" lines record who decided what,
> so a later reader can tell policy from mechanism.
>
> **SYNC** — requirements doc `docs/requirements/v1.9.3_vision_mission_highe-leve-features_v1.txt` **v1** (mtime 2026-09-14 14:20, 3,946 bytes, md5 `736f0c97…`) | release plan **`docs/plans/V193_RELEASE_PLAN.md` rev 1.5** (working record; **owner-facing planning moved 2026-09-16 to `docs/plans/V193_MATTERS_MAP.md` rev 2** — the map of matters, his decisions verbatim, the 7 phases; §2.1 and the D1–D43 table are superseded) | tracker rev **53.0** | change log **`docs/CHANGELOG_v1.9.3.md` through 2026-09-16, phases 1–4** (**committed 2026-09-16 on the owner's word**, together with the v1.9.2 log's SYNC edit; the v1.9.2 log is closed at its 2026-09-14 entry) | owner pack: the v1.9.2 pack `73c5c95d` stays at r4.5; **the v1.9.3 pack opens at the Phase 2 scope gate** (owner, 2026-09-14) | branch **`dev_v1.9.3` @ `5147986`** (PRs #388 #363 #376 #364 merged locally 2026-09-16; then phases 2–4: `ac902b4` F1, `7788a2b` F2, `d9afb7e` F3, `58b9f97` F4, `19c20d5` D1, `25863cf` G1, `ed79c1d` C2, `3ee80a8` languages, `5197fd1`+`528e64b`+`7d5406e` GUI, then the three review-remediation commits `8eb8d00` `6393781` `5147986`), `main` @ `e8c1c6e`, `origin/main` @ `63f256e` = tag `v1.9.2`; **nothing pushed — owner decision 2026-09-14: no push until the 1.9.3 release** | GitHub 2026-09-16: **32 comments posted** (25 phase-1 non-G2 + 4 G2 + #366 retest + #305 + #397) and **5 closures** (PRs #365 #362 #361 #360, issue #397); every comment id in `docs/plans/V193_PHASE1_POSTED.log` | **Owner 2026-09-15: F4 (MOSS) dropped from 1.9.3 → next release (perhaps 1.10); step-4 postings APPROVED and POSTED (ten comments + issues #429 #430); replan without MOSS ordered — **scope proposal produced (D2.0) and adversary-gated (25 findings applied); it sits in the release plan §2.1 with 43 numbered decisions for the Phase 2 sitting.** Phase 0 gate (document set, Q5–Q10) still open.**
> Rule: a session that changes the release plan, the tracker, this file or the pack brings the others to the same state before it ends (CLAUDE.md rule A7).

---

## 2026-09-16 — phase 5: the expert Colab notebook, and why local translation returned nothing

**Decision (owner, 2026-09-16):** phase 5 goes ahead; `WhisperJAV_colab_edition_expert.ipynb` is
the notebook to work on; its job is to *"make available the new features and options to the users of
the notebook … whisperseg, fireredVAD, QwenASR, animewhisper and etc. The only items that cannot be
made available to the notebook are BYOP externals."* Install pinned to the 1.9.3 tag; Kaggle left as
it is and marked unmaintained; Python handled per Colab practice.

### The expert notebook now offers what 1.9.2 offers

Before this change the notebook was a v1.8.12 form: five pipelines, four segmenters, four Whisper
models, no FireRedVAD, no Qwen, no anime-whisper, no `--vad-version`, one translation target and
five providers. Worse, it emitted `--speech-segmenter whisperseg` together with `--mode balanced`,
which `validate_balanced_vad_options` has refused since 1.9.2 (exit 2, requirement S2/S9) — so a
user who pressed run on the defaults got a usage error before any audio was read.

Step 1 now offers, per pass: six pipelines (balanced, fidelity, fast, faster, transformers, qwen),
the three sensitivities, the Whisper models, the four Kotoba models for transformers, `qwen3` and
`anime-whisper` for qwen, all ten speech segmenters including `whisperseg` and `firered-vad`, the
three built-in VAD builds for balanced, four scene detectors, four audio clean-up backends, all
seven merge strategies, ten translation providers, nine target languages and three tones.
CrispASR and XXL are not offered: they need an executable the user supplies, which a Colab session
does not have.

- **Balanced and the speech segmenter.** Balanced runs faster-whisper's built-in VAD and does not
  accept an external segmenter; the equivalent control is `--vad-version`. The form therefore has
  two separate controls, and Step 1 *refuses* the combination in the form, naming the two ways out,
  instead of letting the run fail. `main.py` is unchanged: what the program accepts was not touched.
- **Step 2 speaks each pipeline's own flags.** qwen gets `--qwen-*`, transformers gets `--hf-*`,
  the Whisper pipelines get `--model` / `--speech-segmenter` / `--scene-detection-method`, and a
  two-pass run uses the uniform `--passN-*` family, with an anime-whisper pass carried as
  `--passN-qwen-params '{"generator_backend": "anime-whisper"}'` (`pass_worker.prepare_qwen_params`,
  line 434). The command is printed before it runs.
- **The four ffmpeg filter checkboxes were removed.** `--passN-speech-enhancer` takes a plain name;
  the `ffmpeg-dsp:amplify` form the old notebook emitted is rejected by argparse. Nothing on the CLI
  can carry a choice of individual ffmpeg effects, so a control that pretended to was dropped.
  **Owner's to decide** whether the CLI should gain one.
- **Python check corrected.** It read `>= (3,14)` while its message said "3.13+ not supported …
  requires 3.10-3.12". It now reports the supported range 3.10–3.13 truthfully, and only reports:
  Colab's runtime Python is not ours to change.
**Verified:** the notebook's own Step 1 and its two command builders were exec'd out of the shipped
`.ipynb` for 22 settings combinations, and each generated command was handed to
`whisperjav.main --dump-params` (parses and resolves, runs no ASR). 18 combinations resolved with
exit 0; the 4 that must not run (balanced + whisperseg, single pass with scene detection off, cloud
translation with no key, custom provider with no address) were refused by Step 1 itself. 0 failures.
The run that found the `ffmpeg-dsp:amplify` defect is the same harness. Separately, the translate
command Step 3 builds was checked the same way for all ten providers and a spread of target
languages and tones: 11 commands, all accepted by `whisperjav-translate --show-settings`, 0
failures. Nothing was run on Colab —
**no claim is made here about a real Colab session.**

### Local translation returned nothing — the installed llama-cpp-python build rejects its own replies

The owner's screenshot: four `SERVER ERROR` attempts, each `500 2 validation errors`, the first
`'loc': ('response','CreateChatCompletionResponse','choices',0,'message','refusal'), 'msg': 'Field
required'`, raised inside `llama_cpp/server/app.py` line 376, ending
`Failed to communicate with server after 3 retries` and `Translation FAILED after 86.9s`.
The **two** errors are one cause, not two: the route's response model is
`Union[llama_cpp.ChatCompletion, str]`, so Pydantic reports a failure per union member, and
`('response','str')` is just the `str` member refusing a dict.

Provider `local` starts llama-cpp-python's own HTTP server as a subprocess
(`local_backend.py:2196`, reached from `translate/cli.py:718` and `translate/service.py:466`) and
talks to it over the chat-completions API. That route declares
`response_model=Union[llama_cpp.ChatCompletion, str]` (`server/app.py:380`) and the non-streaming
branch returns `run_in_threadpool(llama.create_chat_completion, **kwargs)` (`server/app.py:526`), so
FastAPI validates the server's *own reply*.

In the JamePeng fork of llama-cpp-python, at the tag Colab installs today
(`v0.3.49-cu126-linux-20260831`, `llama_cpp/llama_types.py:109-117`):

```
class ChatCompletionResponseMessage(TypedDict):
    content: Optional[str]
    refusal: Optional[str]          # required: no NotRequired
    role: Literal["assistant"]
```

while that same tag's `_convert_text_completion_to_chat` builds the message as
`{"role": ..., "content": ...}` and nothing else. The reply therefore fails its own validation and
every non-streaming chat completion is an HTTP 500. The key is `Optional[str]`, so supplying `None`
satisfies it. Upstream `abetlen/llama-cpp-python` does not carry `refusal` on that type (checked at
tag `v0.3.21` and at `main`), and neither do this project's own pinned 0.3.21 wheels — so the escape
is the **version**, not the fork.

**Who is affected.** Any install that ends up on a recent fork build. Colab gets there because
`install_colab.sh` looks for the pinned wheel (`LLAMA_CPP_VERSION=0.3.21`) in
`mei986/whisperjav-wheels`, whose `cu126` folder holds one file —
`llama_cpp_python-0.3.21-cp312-cp312-linux_x86_64.whl` — and Colab now runs Python 3.13; with no
match the script falls through to **the newest JamePeng release**, unpinned. The Windows
installer (`installer/templates/post_install.py.template`) and `install.py` have their own wheel
logic and the same fallback, so this is not Colab-only. Every JamePeng release carrying a cp313
cu126 linux wheel (0.3.46 through 0.3.49) has the defect, so pinning to an older one is not a way
out. Provider `custom` pointed at a llama-cpp server **the user started themselves** is not covered
by the fix below — that process is not ours to launch.

**The fix.** `whisperjav/translate/llama_server_shim.py` (new) is started in place of
`-m llama_cpp.server`. It checks whether the installed build's
`ChatCompletionResponseMessage.__required_keys__` actually contains `refusal`; **only then** does it
wrap `Llama.create_chat_completion` to `setdefault("refusal", None)` on each choice's message, then
hands over to `llama_cpp.server.__main__.main()`. `local_backend.py` launches it **by file path, not
with `-m`**, so the server subprocess imports `llama_cpp` and nothing of WhisperJAV — the
translation package and its dependencies are not dragged into that interpreter. Its two stale-server
scans (`:1785`, `:1802`) also match `llama_server_shim`.

The wrapper is called on the streaming path too — the streaming branch passes
`llama_cpp.Llama.create_chat_completion` unbound (`server/app.py:506`, invoked at `:207`) — and
returns the iterator unmodified, which is why streaming is unaffected. Its gate is one key on one
type: the same fork declares `refusal` required on `ChatCompletionLogprobs` while never setting it
there either, so a chat request asking for logprobs would fail the same way and the shim would not
notice. WhisperJAV never asks for logprobs, so nothing user-facing turns on it today.

**What was verified, and what was not.** `tests/test_llama_server_shim.py` (new, 10 tests, all
passing) covers both arms against the llama-cpp-python in the `WJ` environment (upstream 0.3.23,
which does **not** have the defect): the repair declines and leaves the method untouched on a clean
build; with the type made to require `refusal`, it fires, the reply gains `refusal: None`, the reply
text and an existing refusal value are preserved, a streaming reply passes through, and applying it
twice is harmless. Two further tests show the shim runs as a script without importing `whisperjav`,
and that it accepts the server's own options (`--model`, `--n_gpu_layers`, `--n_ctx`, `--host`,
`--port`) and exits 0 — with `PYTHONIOENCODING=utf-8`, because llama-cpp-python's own help text
contains an emoji that a cp1252 console cannot print; `-m llama_cpp.server --help` fails identically
without it, so that is not the shim's doing.

**Not verified:** no build carrying the defect exists on this machine, so the repair has never run
against one, and nothing at run time reports whether it worked. The diagnosis rests on the fork's
source at the tag quoted above plus the owner's error text — not on a reproduction. One supporting
fact: the server readiness probe is `/v1/completions`, not chat (`local_backend.py:1607`), which is
consistent with a server that reports healthy and only fails 86 seconds later at translate time,
and inconsistent with VRAM, model or network explanations.

**Proposed, not done (his call).** Add one non-streaming chat request to the readiness check after
the completions probe. It is the one measurement that separates "this build is broken" from
everything else, and it would turn a 500 at translate time into a clear message at start-up — but it
is a new way for a run to stop early, so it is not being added without his word.

### `pip install whisperjav[local-llm]` installed nothing

`_install_server_deps` asked pip for the extra `local-llm`. No such extra exists; the name in
`pyproject.toml:176` is `llm`. pip warns and exits **0** on an unknown extra, so the function
reported success while installing no server dependency at all, and a first-time local translation
could then fail for want of uvicorn or fastapi with no sign of why. Corrected to `whisperjav[llm]`
in all five places. The Colab path never depended on it: `install_colab.sh` installs the `llm` extra
directly.

### The Colab installer's pinned wheel could never be found

`install_colab.sh` built the wheel filename with the platform tag
`manylinux_2_17_x86_64.manylinux2014_x86_64`, while the file in `mei986/whisperjav-wheels` is
spelled `linux_x86_64`. The pinned source therefore never matched, on any Python, and every install
took the unpinned fallback. It now tries both spellings. **Checked live:** the old spelling returns
404 for cp312 and cp313; the new one finds `llama_cpp_python-0.3.21-cp312-cp312-linux_x86_64.whl`.
On Python 3.13 there is still nothing to find — the dataset has no cp313 wheel — so Colab continues
to take the fallback until one is built. **His call:** whether to build one
(`notebook/build_llama_cpp_wheel.ipynb` exists for it, but it clones the JamePeng fork, so it would
have to check out a tag at or below 0.3.21 or it would ship the same defect on the pinned path).

### The notebooks install a fixed release

`install_colab.sh` now takes `WHISPERJAV_BRANCH="${WHISPERJAV_REF:-main}"`, and the expert notebook
sets `WHISPERJAV_REF=v1.9.3` and clones `--branch v1.9.3`. Other callers are unaffected.
**This notebook cannot install until the tag exists** — that happens in phase 7.

### Kaggle

`WhisperJAV_kaggle_parallel_edition.ipynb` is unchanged apart from a banner at the top saying it is
no longer maintained, that its settings may no longer match what WhisperJAV accepts, that reported
problems are not being fixed, and pointing at the expert Colab notebook. It still emits
`--speech-segmenter` with the balanced pipeline and so still fails on its defaults; that is what
"unmaintained" now tells the reader. `WhisperJAV_colab_edition.ipynb` was already retired and was
not touched.

---

## 2026-09-16 — three adversarial reviews, and the data-loss defect they led to

**Decision (owner, 2026-09-16):** *"I would rather all the deficiencies, errors and bugs are
fixed so we do not leave tech debt to next release."* Three reviews ran: two over the
session's commits (`c5405b5..7d5406e`, disjoint Python and GUI scopes), then one over the
remediation those produced. Commits `8eb8d00`, `6393781`, `5147986`.

- **The cleanup deleted files WhisperJAV never wrote (`5147986`).** `cleanup_temp_directory`
  removed **every** file at the root of the temp directory. Harmless for the default folder
  under the system temp, which WhisperJAV owns — but `--temp-dir` can point anywhere, and
  `whisperjav --temp-dir D:\MyVideos D:\MyVideos` destroyed everything in `D:\MyVideos` at
  the end of the run. It now deletes only the shapes WhisperJAV writes
  (`_extracted.wav`, `_enhanced.wav`, `_resampled.wav`, `_raw.srt`, `_stitched.srt`,
  `_master.json`) and keeps anything it does not recognise — litter is preferable to data
  loss. Demonstrated on a folder holding a video, unrelated media, a text file, a finished
  subtitle and two intermediates: the intermediates and working folders went, **the other
  four survived; all four were deleted before**.
  The behaviour predates this session. It was fixed now because the media-listing text added
  earlier in the session stood in front of it telling users their files "are still going to
  be processed".
- **Six working folders, not four.** `resampled_scenes` and `crispasr_out` were in neither
  the cleanup list nor the labelling list, so they were never tidied up and never labelled.
  Both now read `WHISPERJAV_WORK_DIRS` from `whisperjav/utils/media_leftovers.py`.
- **The preset loader blanked a dropdown instead of restoring it.** `8eb8d00`'s message
  claimed `setSilent` "restores what was there"; it verified and logged but never restored,
  so loading a preset whose model the current pipeline cannot run left the Model dropdown
  empty while the state kept the preset's value — and the next save wrote that blank to the
  settings file. `applyToForm`'s new guard could not catch the blank coming back either
  (`'' !== ''` is false). Both fixed.
- **Legacy passes are now filtered at startup**, so a fresh load on Balanced no longer
  offers turbo from `index.html`'s static list.
- **The recursion clause no longer hides a search that happened** — discovery expands each
  argument with `glob(recursive=True)` and walks any directory it finds, so an unexpanded
  `"D:/Media/*"` does recurse.
- **Tests that could not fail, replaced.** Two asserted against hand-built literals or
  iterated the same tuple the code under test iterates; one had been nested inside another
  by an earlier edit and never ran. Two new files: `tests/test_temp_cleanup_safety.py`
  (27 tests, mostly about the user's files surviving) and `tests/test_gui_model_lists.py`
  (5), which moves the model-list invariant out of a scratchpad file — where nothing would
  have caught the same mistake at the next model addition — and guards against its own
  parser silently failing. **92 tests now pass across five files, run individually.**

**What the reviews say about this session's method, recorded because it recurred four times.**
Work passed its own checks while misreporting what it did: C1 was called a clean negative
result when it was re-measuring the owner's own fix; D1 shipped a badge announcing a preset
that was not in force; the first model-sync attempt would have overwritten the user's model
silently; and `8eb8d00` claimed a restore it had not implemented. Static checks and the
sandboxed round-trip passed every time. **The checks that actually found these were the
owner's click-through and adversarial reviews framed around the code rather than around the
author's claims.** The third review was briefed to read the code first and test the commit
message against it, and it was the most productive of the three.

**Standing correction for later readers:** a comment, docstring or change-log line in this
tree is a claim to verify, not evidence. Verified instances this session: a test docstring
denying it loaded the ASR stack while loading five heavy modules; an invariant falsified by
its own commit; "Standard library only" on a module whose import pulls numpy; a claim about
shell globs that `media_discovery.py` contradicts; and two wrong line citations in this file.

**Confirmed in the GUI by the owner, 2026-09-16:** loading a saved preset whose model the
current pipeline cannot run now keeps a valid model in the row instead of blanking the
dropdown. That was the defect `5147986` fixed and the one check no static test could make —
*"I tested the GUI for that testcase and it worked"*. **The GUI work of this session is
verified.**

**Still open, and the owner's to decide:** the wording of the messages users see that were
added or changed this session — the note beside a file in the listing, the three English
post-processing lines and two audio-extraction lines, the Linux installer's closing text,
the new "cannot use this working folder" error and its exit code, and the two GUI console
warnings. One of those warnings is wrong as written: it prints the internal pipeline id
(`balanced`) rather than the label the row shows (`Balanced`).

## 2026-09-16 — D1 and D2 tested in the GUI by the owner: one defect of ours backed out, one pre-existing defect fixed

**D2 — nothing wrong.** Ensemble tab → a Fidelity pass → segmenter FireRedVAD → Customize renders correctly:
Detection (Speech Threshold 0.3), Processing (Smoothing Window 5 frames), Filters (Min Speech 150 ms,
Min Silence 150 ms, Max Speech 5 s), Grouping (Group Gap 1 s, Max Group Duration 5), Padding (Start 50 ms,
End 100 ms). No change made. This closes D2.

**D1 — the 14 two-pass selectors work.** Verified by the owner across a real restart: Fidelity / Aggressive /
Silero / None / FireRedVAD / Large V2 on pass 1, pass 2 enabled with Transformers / Auditok / Kotoba, and the
merge strategy all came back. The save → disk → load round trip carries 17 of 17 keys.

- **Backed out — a badge that misreported what would run (`528e64b`; ours, introduced in `19c20d5`).**
  `SettingsPersistence.restorePresets()` restored the saved `pass1Preset` / `pass2Preset` onto
  `EnsembleManager`'s state. `updateBadges()` (~`app.js:2256`) derives the green preset badge, the
  "Edit Parameters" button label and the modal's `[Custom]` title from `presetName` **alone**, while the
  parameters a preset names live in `passState.params` — which `restorePresets` could not restore. After a
  restart the GUI therefore announced that preset `MKATEST193` was in force while `params` was `null` and a
  run would have used **defaults**. It also resurrected a cleared preset: Reset to Defaults nulls
  `presetName` and refreshes the badge, but does not rewrite the saved settings, so the old name returned at
  the next restart — the owner's "after I reset to defaults, I still see my saved preset".
  `restorePresets` is now a documented no-op; the badge tells the truth (after a restart the pass is on
  defaults). `collectAll` still saves both names, so the Customize-modal half of D1 has the data when it is
  done. **Restoring the label honestly requires loading the preset through the presets API and applying its
  values — that is D1 step two and was not attempted.**
  **Note for a later reader:** the sandboxed round-trip test passed 17/17 and could not have caught this,
  because the defect was in what the GUI *claimed*, not in what it stored. Only the click-through found it.

- **Fixed — the Customize modal kept the previous pipeline's tab labels (`528e64b`; pre-existing).**
  The modal has one set of tab buttons shared by every pipeline. `generateTransformersTabs` (`app.js:3326`)
  and `generateQwenTabs` (`:3670`) rename them in place — Qwen turns Segmenter into "Generation", Quality
  into "Audio", Enhancer into "Alignment", Scene into "Output", and hides Context — and nothing renamed them
  back. The legacy path used by fidelity and the other classic pipelines generated its panels but never set
  the labels. Opening a Fidelity pass's Customize after a Qwen one therefore showed Qwen's labels over
  Fidelity's content, and **the Segmenter tab appeared to have vanished when it had only been renamed**
  (the owner's screenshot: correct Fidelity content — Whisper model, Turbo, CUDA — under
  Model / Audio / Generation / Alignment / Output). The legacy path now restores `index.html`'s own labels
  and Context's hidden state before filling the panels, as the other two paths already did.
  Still needs a click-through: open a Transformers pass's Customize, close it, open a Fidelity pass's.

**Owner's observation, recorded, no action:** that the saved settings work per pass rather than per tab. That
is the design — `pass1_preset` and `pass2_preset` are separate keys — and he confirmed it was an observation,
not a defect report.

**Decisions (owner, 2026-09-16):** the `.subtrans` target-language fix is **parked for the 1.10.x release**,
not done in 1.9.3 (the recommendation to fix it was accepted, the timing was not). Working translation
controls for the Transcription tab go to **2.x**, pending a possible GUI revamp — further out than the
1.10.x note recorded in the entry below, which this supersedes.

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
  (`whisperjav/main.py:1196`), and the adapter is what feeds
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

- **F3 — audio extraction was silent and untimed.** `audio_extraction.py:48` is now INFO, and the success
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
