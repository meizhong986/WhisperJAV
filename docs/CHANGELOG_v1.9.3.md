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

## 2026-09-17 — a clean-up that fails while it is running now stops the run too

**Owner, 2026-09-17,** answering the open item the audit left him: *"yes stop."* He also confirmed
the new refusal of `--mode balanced --enhance-for-vad` (*"confirm"*).

The rule he set earlier that day — a clean-up the user chose that cannot run stops the run — had
only ever been implemented for a clean-up that cannot *start*: a package that will not import,
caught by the start-up check, plus htdemucs, which raises on everything. The four other backends
(clearvoice, bs-roformer, zipenhancer, ffmpeg-dsp) each return a failed result when the weights
cannot be fetched, the card runs out of memory, FFmpeg exits non-zero or the backend throws — and
`enhance_scenes` logged a warning, put the untouched scene in the list and carried on. The run ended
at 0 with a subtitle file made partly from audio the user had asked to have cleaned up.

`enhance_scenes` now raises `SpeechEnhancerUnavailable` on any scene whose clean-up failed, whether
the backend reported the failure or threw. The message names the clean-up, which scene of how many,
what went wrong, what it means and the way out. Every pipeline enhances its scenes through that one
function, so the rule reaches balanced, fidelity, qwen, transformers and decoupled at once, however
the backend was reached.

**What this changes for users:** a run that used to finish with one or two scenes quietly
un-enhanced now stops. That is the intent — but it does mean a single transient failure on one scene
of a long file fails that file. It is worth watching once real backends are exercised; a tolerance
(stop only when every scene fails) would be a different decision, not a bug fix.

`tests/test_enhancer_failure_stops_run.py` (renamed from the sample-rate test written an hour
earlier, whose premise this replaces): 8 tests — a backend that reports failure, one that throws,
what the message must say, and that no path hands back untouched audio. The sample-rate guarantee it
used to cover is still tested for the scenes that do come back.

---

## 2026-09-17 — the adversary audit of this sitting, and the defects it found

**Owner, 2026-09-17:** *"run the adversary audit. 1b - solve for any issues and bugs with care.
There are no deadlines, take your time. Make sure nothing gets broken."*

Three adversarial reviews, one per piece of the day's work (speech enhancement, translation
start-up, GUI/CLI wiring), each framed around the code rather than around the claims made for it.
Every finding below was re-checked against the code before anything was changed, and every fix has
a command or a test behind it. The items the reviews raised that are the owner's to decide are
listed at the end and were **not** implemented.

### The failure advice never reached the provider it was written for

The advice added earlier the same day keyed on `pysubtrans_name`. That field is the name of
PySubtrans's *client class*, not the provider the user chose: local, ollama, glm, groq and custom
are all `Custom Server` there, and the local run's config is rebuilt as `Custom Server` before it
reaches the translator. So the `local` and `ollama` branches could never run. Every local, Ollama,
GLM, Groq and custom run got the line written for a server the user had named themselves — and
**Ollama users were worse off than before the change**, because they used to be told `ollama logs`.

Each provider config now carries `whisperjav_provider`, the name the user types after
`--translate-provider`, and the advice reads that. A cloud provider redirected to a
`--translate-endpoint` on this machine is no longer called a cloud service either: the address
decides. The advice function was lifted out of `translate_subtitle` so it can be tested at all —
`tests/test_translation_server_log_advice.py`, 14 tests, covering every shipped provider.

### The local server's log was deleted before the user could read it

The advice names a file, and `stop_local_server()` removed that file as the run ended. Naming the
log now marks it to be kept, and the path is logged again when the server stops. An ordinary run
still cleans up after itself.

### The start-up chat check asked in the wrong shape for the translate CLI

The check sent a non-streaming chat request. `whisperjav-translate --provider local` translates
with `stream=True`; the GUI and `--translate` path do not. These are two different paths on the
server — llama-cpp-python validates a single reply against a response model and sends streamed
chunks without validating them, which is exactly where the known defect lives — so the check could
have stopped a CLI run that was going to work, and passed a run that was going to fail. It now asks
in whichever shape the caller will use. A 2xx with an empty body is named instead of surfacing as a
JSON error.

`tests/test_local_chat_check.py` was rewritten: the wiring test was a search of the source file, and
is now the readiness check actually running against a stub that answers all three endpoints. 13
tests, including both "broken only when streaming" and "broken only when not streaming" builds.

### Balanced refused the VAD-only split through one flag and accepted it through another

`--pass1-enhance-for-vad` on a balanced pass was refused, but the pass configuration is built as
*this pass's flag OR the global `--enhance-for-vad`*, and the refusal only read the pass flag.
Verified before the fix: `--ensemble --pass1-pipeline balanced --pass1-speech-enhancer clearvoice
--enhance-for-vad` exited 0 with `enhance_for_vad: true` on a balanced pass — the exact thing the
owner refused. The refusal now reads the same combination the run will use, and names whichever
flag the user actually typed.

`--mode balanced --enhance-for-vad` was likewise accepted and ignored; it is now refused in the same
words. `--pipeline decoupled` is not caught by either rule, because `--mode` still reads `balanced`
from its default on a decoupled run.

### Single-mode fidelity never received the flag at all

`--mode fidelity --enhance-for-vad` was accepted and did nothing: `pipeline_args` had no
`enhance_for_vad` key, so `FidelityPipeline` always saw `False`. The split only ever happened
through `--ensemble` or the GUI. It is now passed. **See the open item below**: single-file mode has
no way to choose a clean-up in the first place, so this is a gap closed rather than a feature
reached.

The flag's help text said "Qwen/Decoupled pipelines only", which was already untrue. It now names
where it applies, where it is refused, and where it is accepted and ignored — the documentation the
owner's 2026-09-17 decision made a condition of that ignoring.

### One scene's hiccup failed the whole file, with a message about the wrong thing

Scenes are extracted at 48 kHz when an enhancer is configured. `enhance_scenes` resampled what it
enhanced down to 16 kHz but handed back the untouched 48 kHz file for any scene whose clean-up
failed. In the VAD-only split that list is paired scene by scene with originals at 16 kHz, and the
recogniser refuses two tracks at different rates. So a transient failure on one scene failed the
entire file with "the audio is 16000 Hz and the cleaned-up copy is 48000 Hz" — a message pointing at
sample rates rather than at the clean-up that actually failed. Every scene returned is now at 16 kHz.
`tests/test_enhance_scene_fallback_rate.py`, 6 tests.

### `--qwen-enhancer` still rejected the form the other flags accept

The same `choices=` list that broke FFmpeg DSP for the two-pass flags was still on `--qwen-enhancer`,
so `--mode qwen --qwen-enhancer ffmpeg-dsp:loudnorm` exited 2 with "invalid choice". It now takes the
same `backend[:detail]` spec, and the detail is separated into the model before it reaches the
factory, which only knows plain backend names. Not GUI-reachable; it bit CLI and notebook users.

### htdemucs: three corrections, none of them run against the model

- Its docstring promised a failed result for a single-scene failure "so the rest of the file can go
  on". No path returns one — every failure stops the run, which is what the owner asked for. The
  words were wrong, not the behaviour.
- Any error whose text merely contained "cuda" was reported as the graphics card running out of
  memory, sending users to free memory they had not run out of. It now matches what running out of
  memory actually says.
- The model was loaded *after* the audio was read, though the reason to load it early is to stop
  before spending time on audio. Reordered.

**Still true: no audio has been through htdemucs on this machine.** `demucs` is not installed.

### The duration fallback answered about the wrong file

With no ffprobe on the machine, the duration came from the extraction log — which is the *source
container's* `Duration:` line, not the extracted stream's, and which gives nothing at all when the
source prints `Duration: N/A`. The extracted file's own WAV header is now read first, which is still
a header read and is about the right file. The log stays as the last resort.

### The GUI showed "off" while sending the flag

The "Enhance for VAD only" tick box and the stored value are two separate things, and only the
balanced rule ever wrote the box. Ticking the option inside the Customize window, or loading a
preset saved with it on, left the row showing unticked while the run still sent the flag. Repainting
the row now always shows what will run. **Not click-confirmed — this needs the owner at the GUI.**

### Verified this sitting

`python -m whisperjav.main --help` exits 0. The nine-command sweep over `--enhance-for-vad` and
`--qwen-enhancer` (refusals exit 2 with the intended message; `--mode fidelity`, `--mode qwen`,
`--pipeline decoupled`, and ensemble fidelity+qwen all exit 0 and still set the flag). Test files run
one at a time: `test_translation_server_log_advice` 14, `test_local_chat_check` 13,
`test_enhance_scene_fallback_rate` 6, `test_llama_server_shim` 10, `test_local_llm_health_check` 33,
`test_enhance_for_vad_dual_track` 17, `test_speech_enhancer_spec` 30, `test_htdemucs_enhancer` 22,
`test_audio_extraction_duration` 9, `test_qwen_customize_model` 6. `node --check` on `app.js`.

**A real transcription with the fidelity VAD-only split, which had never been run.** An 8-second
clip through `--ensemble --pass1-pipeline fidelity --pass1-model tiny --pass1-speech-enhancer
ffmpeg-dsp --pass1-enhance-for-vad`: exit 0, subtitles written, and the temp directory shows the two
tracks the split promises — `enhanced_scenes/clip_scene_0000_enhanced.wav` at 16 kHz for finding the
speech and `resampled_scenes/clip_scene_0000_resampled.wav` at 16 kHz, untouched, for the
recogniser, both from the 48 kHz original. The log confirms which went where. That retires one of
the four things this branch had never proved.

### Left for the owner — not implemented

1. ~~**A clean-up that fails at run time still degrades silently.**~~ **ANSWERED the same day:**
   owner, *"yes stop"* — carried out, see the entry above.
2. **Single-file mode cannot choose a clean-up at all.** There is no top-level `--speech-enhancer`;
   only `--pass1/2-speech-enhancer` and `--qwen-enhancer` exist, and the fidelity preset resolves no
   enhancer. So the VAD-only split in fidelity is reachable only through two-pass/ensemble or the GUI.
3. **The Transformers Customize window** still discards the model picked in it, the same shape as the
   Qwen window fixed earlier that day.
4. **The Customize window's Enhancer tab ticks are collected nowhere**, so DSP effects chosen there
   change nothing; only the inline row panel reaches the command.
5. **htdemucs always receives a mono downmix**, so its stereo model runs in its least favourable
   configuration. Nobody has measured what that costs.

---

## 2026-09-17 — four of the owner's decisions carried out

**Decisions (owner, 2026-09-17):** *"balanced shall not have the VAD only enhancement"*; *"No
external VAD speech segmenter shall be available for balanced"*; *"agree with your recommendation"*
on a start-up translation check; *"I agree"* on the wrong Ollama advice; *"I agree"* on the Qwen
Customize window. Also, on a Python 3.13 translation-helper file: **no** — *"my main strategy is
that I will not fight colab native stack. If python 13 breaks the wheels then the solution pieces
that require new wheels are no longer supported until new precompiled wheels are found and
provided."*

### Balanced no longer offers the VAD-only split

Balanced finds the speech inside faster-whisper's own call, on the very audio it transcribes, so
there is no second track to give the cleaned-up audio to. The flag was being accepted and both
tracks enhanced — the silent difference the other balanced rules exist to prevent. It is now
**refused** for a balanced pass, in the same shape and wording as the segmenter refusal beside it;
the GUI hides the box **and clears the stored value**, so a setting left from another pipeline
cannot travel into a run through a control the user can no longer see; and the GUI's command builder
will not send it for a balanced pass either. Still accepted for fidelity and qwen.

The related confirmation needed no change: no external speech segmenter has been available for
balanced since 1.9.2, and the existing refusal covers it.

### Translation now proves it can chat before the run starts

Everything the local-server readiness check did used `/v1/completions`. Translation sends
`/v1/chat/completions`, and a build can serve one and fail the other every time — which is exactly
what happened on Colab: healthy server, then failure 86.9 s later with nothing translated. A new
phase asks for one short chat completion before the server is reported ready, and a failure there
stops it coming up with a message written for the user. Where the answer mentions a refusal or a
validation error it adds that this build rejects its own chat replies and that WhisperJAV already
repairs the known form, so this is a variant it does not cover.

**One existing test had to change**, and the reason matters: its stub returned a body with no
`choices` for every request, which was fine while only `/v1/completions` was asked for but no longer
represents a **working** server. The stub was corrected; the check was not weakened to fit it.

### Each translation provider gets advice that applies to it

Every server error used to end with *"Check Ollama server logs: ollama logs"* — including the local
provider, which is llama-cpp-python and nothing to do with Ollama. Now: the local provider is
pointed at the log WhisperJAV itself writes and whose path it prints at start-up; Ollama gets
Ollama's; a custom server is the user's own; anything else is a cloud service with no local log.

### The Qwen Customize window and the row agree

The window showed its own model while the row was what ran, so a model chosen there was displayed as
accepted and discarded. It now **shows the row's model** — adding it to the list where the window's
own list does not carry it, rather than substituting, which is how they came to disagree — and
**writes a choice back to the row**, guarded so a value the row cannot show is reported rather than
lost. Transformers has the same shape and is **not** changed; the comment now says so plainly
instead of calling it "a separate matter".

**Verified:** 6 new tests for the chat check against a stub server (including the exact Colab fault),
6 new for the balanced refusal and the GUI guards, 6 new reading the Qwen wiring; plus 33, 30, 27,
22, 17, 10 and 7 across the existing suites, and `node --check` on the GUI script. **Not verified:**
the chat check has not run against a real local server — there is no model on this machine — and a
GUI change is not verified until the owner clicks it.

---

## 2026-09-17 — one rule for a component that cannot run, and the VAD-only split made real in fidelity

**Decision (owner, 2026-09-17):** *"I think all speech segmenters have to have uniform behaviour: if
user selected any but they cannot run then it is a failure and the process shall stop with helpful
communication"*; *"please make the VAD Only Enhancement feature audio separation path to work for
balanced and fidelity"*; and, on the earlier finding, *"it is ok for faster, fast, and transformers
and crispASR to ignore that"*, with *"please rely on the code as comments and docstrings can be out
of date or mistaken."*

### A chosen component that cannot run now stops the run, whichever it is

**Clean-ups** were not uniform: htdemucs stopped, while zipenhancer, clearvoice and bs-roformer
warned, fell back to "none" and produced subtitles from the untouched audio — a poor file from a run
that exited 0. Now every clean-up that has to be installed is fatal. `none` and `ffmpeg-dsp` are
not, because they cannot be missing. The fatal set is **derived** from the dependency table rather
than listed by hand, and a test asserts the two agree.

**Segmenters** already stopped — but inside the recogniser, saying *"Speech Segmenter not configured
- this is an architecture violation"*, which tells a user nothing.
`ensure_segmenter_backend_available()` now checks the chosen segmenter before any audio is read and
prints the same kind of box. It sits beside `ensure_segmenter_model_available()` — that one is about
a **model** that must be downloaded, this one about a **package** that must be installed — and runs
first, because reporting a download failure for a missing package sends the user the wrong way.

Both boxes list the alternatives **actually present on that machine**, computed at the time, rather
than a fixed list that might name something else that is also missing.

### "Enhance for VAD only" now does what its name says — in fidelity

`WhisperProASR.transcribe()` takes an optional second file: the cleaned-up copy of the same audio,
used **only** to find the speech, while the recogniser still hears the original. Omitted, both jobs
read the one file exactly as before. The two tracks must be the same recording at the same rate, or
it refuses — a clock difference would put every subtitle in the wrong place, and a length difference
beyond 0.1 s means it is not the same audio. The fidelity pipeline builds both tracks and pairs them
by position, with a length check, because pairing the wrong scenes would misplace everything.

**Balanced is not done, and not by this route.** Its speech detection happens *inside*
faster-whisper's own `transcribe(vad_filter=True)` call, on the same buffer it recognises — there is
no second track to hand it. Splitting them means switching the built-in VAD off and driving
recognition from an external detector, which is what 1.9.2 deliberately removed (requirement S2/S9),
and the balanced pipeline is on the owner's frozen list. **Left alone; his decision.**

**Verified:** 11 new tests, which prove the split for real — with a second file given, the array
handed to the segmenter is the one read from *that* file, not merely that the argument is accepted.
Plus 22 (3 updated for the new rule), 30, 27, 7, 10 and 33 across the other suites. Both start-up
checks were called with every value an ordinary run passes and let all of them through, and a real
fidelity run on a media file reached the extraction step — so no existing configuration is stopped
by this. **Not verified:** no full transcription has been run with the split active; that needs a
model and real audio.

---

## 2026-09-17 — htdemucs vocal isolation, and the first clean-up that refuses to fail quietly

**Decision (owner, 2026-09-17):** add Demucs v4 for vocal isolation; *"htdemucs for all pipelines to
be available"*; it *"installs only when picked"*; *"there needs to be good checks in place so if the
installation fails or unsuccessful, the program is resilient and informs the users with clear
errors"*; and, asked whether a failure should stop the run or carry on without clean-up,
*"I think it should fail with good user communication."*

### What arrives, and from where

Two things, neither shipped with WhisperJAV. **The package**, `pip install demucs` — checked on
PyPI: 4.1.0 is a pure-Python wheel, and the only two dependencies with compiled code (`lameenc`,
`sphn`) publish prebuilt wheels for Python 3.10–3.14 on Windows x64 and Linux x86_64, so nothing is
built on the user's machine. **The weights**, 84,141,911 bytes confirmed by request today, fetched
once from `dl.fbaipublicfiles.com` and cached. That is **Meta's own site, not Hugging Face** — so
`--offline` and any Hugging Face mirror or endpoint do not cover it. **That matters for phase 6:**
the China work will not help this download.

### It stops the run instead of handing back untouched audio

The enhancement framework degrades quietly by design — a backend that cannot run is warned about,
replaced by "none", and the run finishes. Right for a clean-up that only helps a little; wrong for
one chosen because the audio needs it, where the user would get a poor subtitle file from a run that
exited 0.

So: a new `SpeechEnhancerUnavailable`; a `FATAL_WHEN_UNAVAILABLE` set in the factory derived from a
per-backend flag, containing only htdemucs; both creation paths in `pipeline_helper` raising instead
of falling back for such a backend, and both enhancement loops re-raising instead of swallowing;
and `ensure_speech_enhancer_available()` checking it **before any audio is read**, in the same shape
and with the same exit status as the segmenter-model check approved on 2026-09-12. What the user
sees is a box naming what was asked for, that nothing has been transcribed and why continuing would
be worse, the one install command, the three clean-ups already present, and the Hugging Face note.
**Every other backend's behaviour is unchanged**, and a test pins that.

Offered in both `--passN-speech-enhancer` flags, in `--qwen-enhancer` (which now shares one list
instead of keeping its own copy), in both passes of the GUI's Audio Clean-up dropdown, and in the
notebook, which installs `demucs` when it is chosen. Deliberately **not** in any extra, with a
comment saying why: the program, the GUI, the notebook and the failure message all name one command.

**Verified:** 17 new tests; `test_speech_enhancer_spec.py` 30, `test_v192_small_fixes.py` 27,
`test_audio_extraction_duration.py` 7, `test_llama_server_shim.py` 10; the notebook matrix at 0
failures; both GUI assets parse; `--help` exits 0. Two real runs on a generated media file with
demucs absent — one two-pass, one single-pass qwen — stopped with the box and exit 1 **before the
extraction step**. The start-up check was called with every value an ordinary run passes and let all
of them through.

**Not verified:** htdemucs has never been run. demucs is not installed here, so no audio has gone
through the model — separation quality, speed and memory use are unmeasured. The failure paths are
what has been exercised; the success path has not.

---

## 2026-09-17 — FFmpeg DSP works from the GUI again, and enhance-for-vad says what it does

**Decision (owner, 2026-09-17):** *"I agree with your recommendation about name-with-effects"*;
*"it is ok for faster, fast, and transformers and crispASR to ignore that. please document that"*;
*"please rely on the code as comments and docstrings can be out of date or mistaken"*. He also
confirmed by clicking that the VAD-only row committed earlier today is visible again.

### Choosing FFmpeg DSP in the GUI no longer stops the run

`--passN-speech-enhancer` takes a detail after a colon again. Everything downstream always
understood it -- `pass_worker._parse_speech_enhancer` splits the value, the detail travels as the
enhancer's model, and `FFmpegDSPBackend.__init__` splits a comma-separated model into its effects.
The #306 fix in 1.9.2 (`7934fac`) put a plain `choices=` list on these two flags to catch typos, and
that also rejected every value containing a colon -- which is precisely what the GUI builds for
FFmpeg DSP. **Reproduced before the change:** `--pass1-speech-enhancer ffmpeg-dsp:loudnorm` exited 2
with `invalid choice`.

`main.speech_enhancer_spec()` replaces the `choices=` list on both flags. It keeps #306's protection
in argparse's own wording and extends it to the detail, so a mistyped effect is refused at the
boundary too instead of being dropped mid-run. `FFMPEG_DSP_EFFECTS` is hardcoded beside
`SPEECH_ENHANCER_CHOICES` so `--help` still pays no import cost; a test catches drift from
`AVAILABLE_EFFECTS`. `--qwen-enhancer` is untouched -- it has `--qwen-enhancer-model` already.

The expert notebook's eight FFmpeg filter checkboxes are back, per pass, now that the form is
accepted; they were removed this morning only because it was not.

### What enhance-for-vad actually does, per pipeline

Read from the pipeline sources, not their comments — the `api.py` comment saying "Only effective for
Qwen pipeline" was wrong in both directions:

| Pipelines | What happens |
|---|---|
| qwen, and the decoupled pipeline behind it | The real dual track: cleaned-up audio to the segmenter, original to the recogniser |
| balanced, fidelity | The flag is read and reported in the log, but the cleaned-up audio goes to **both** |
| fast, faster, transformers, kotoba-faster-whisper, crispasr | Never read |

The owner accepted the third group on condition it is documented. `ENHANCE_FOR_VAD_IGNORED_BY` in
`pass_worker.py` names them; the pass worker now logs one line when the flag is set for such a pass,
so a user who asked for it is told rather than left to infer it from silence; and
`--passN-enhance-for-vad`'s help states all three behaviours. A test derives the ignore list from
`inspect.getsource()` of each class in `PIPELINE_CLASSES`, so the documentation cannot drift from
the code. **Balanced and fidelity enhancing both tracks is unchanged and still open** — separating
them needs recogniser-side work.

**Verified:** `tests/test_speech_enhancer_spec.py` (new, 30 tests) — accepted and normalised forms,
every backend and every known effect, the refused forms with #306's wording, no drift, both flags
through argparse, and that the detail reaches the real filter chain
(`loudnorm,denoise` → `loudnorm=I=-16:TP=-1.5:LRA=11,afftdn=nf=-25`). `test_v192_small_fixes.py`
still passes (27), so #306's own tests still hold. The notebook matrix re-ran at 0 failures with the
two-pass case now emitting `--pass1-speech-enhancer ffmpeg-dsp:loudnorm`. **Not verified:** no ASR
run was made; the filter chain was built and inspected, not applied to a recording end to end.

---

## 2026-09-17 — two defects the owner reported: the wasted second read, and the hidden VAD-only control

**Decision (owner, 2026-09-17):** three further requirements for 1.9.3 — the duplicate read in audio
extraction, the missing VAD-only control in the GUI's Ensemble tab, and Demucs (htdemucs) as a
vocal-isolation backend. He gave the go for the first two, and clarified for the third that
**htdemucs is to be available for all pipelines**. Whether it installs by default is still open.

### Audio extraction read the file twice; the second read bought a header field

`extract()` ran FFmpeg, then `_get_audio_duration` ran FFmpeg **again** over the whole extracted
file with `-f null -`, which decodes every sample and throws them away, purely to read the
`Duration:` line FFmpeg prints before it starts decoding. Every pipeline calls `extract()`, so this
happened on every file of every run.

Nothing decodes now: ffprobe reads the header and stops; if ffprobe is absent, the extraction run's
own output is reused, since FFmpeg already printed the duration there; failing both, `0.0`, as
before. The parser is now a static helper and survives `Duration: N/A`, which the old loop would
have raised on.

**Measured** on a 90-minute test video (86 MB source, 173 MB extracted WAV, local SSD):
header read **31 ms**, old full decode **406 ms**, extraction-log fallback **141 ms** — all three
agreeing to within 0.01 s. 13× faster, 0.37 s per file saved here. The absolute saving is small on a
fast local disk and larger wherever the audio is slow to read — a network share, a sleeping drive,
or Google Drive under Colab, where the extracted file actually lives for notebook users.
`tests/test_audio_extraction_duration.py` (new, 7 tests) covers the real extraction, agreement with
the old answer, the no-ffprobe fallback, the failure case, and the three parsing shapes.

### The "Enhance for VAD only" control was not removed — it was hidden almost all the time

Both rows are still in the markup, each with its checkbox, both starting at `display: none`. The
only thing that ever revealed them was `updateEnhanceForVadCheckbox()`, and its only two callers
were the change handlers of the two enhancer dropdowns. Nothing refreshed it when the window opened,
when the tab was shown, when a pipeline changed, or after a preset was loaded — so a pass that
**already** had an enhancer selected kept the checkbox invisible until the user re-picked the
enhancer by hand.

`updateDspPanel` now ends by refreshing that row — both depend on the same one thing, the pass's
enhancer — and `updateRowGreyingState` now ends by calling `updateDspPanel`. Its two early-return
branches already did; the ordinary path, which is what runs at start-up and after a preset load, did
not. That also fixes the same latent gap for Pass 1's FFmpeg DSP panel.

**Checked:** the two functions were pulled out of the shipped `app.js` and run against a stub DOM
for eight cases (no enhancer, each of the four enhancers, second pass off, on, and running an
external tool) — correct in all eight; `node --check` parses the file; the call sites were listed to
show the refresh is now reached from start-up, a preset load and the pass-2 toggle.
**Not verified:** whether this is the fault he saw. A GUI change is not verified until he clicks it.

### Two things found while in that code, both his to decide

- **The checkbox promises more than it delivers on most pipelines.** A comment in `api.py` claimed
  the flag was "Only effective for Qwen pipeline; pass_worker.py silently ignores for others". False
  in both directions. Checked today: **qwen** and the **decoupled** pipeline do the dual track;
  **balanced** and **fidelity** read the flag and say so in an INFO line at run time, but send the
  enhanced audio to **both** VAD and ASR — the separation needs ASR-module changes
  (`balanced_pipeline.py:177`, `fidelity_pipeline.py:111`); **fast**, **faster**, **transformers**
  and **crispasr** never read it at all. The GUI offers the checkbox wherever an enhancer is set.
  The comment now records the truth; what the user is shown is his call.
- **Choosing FFmpeg DSP in the Ensemble tab stops the run.** `api.py` builds
  `--passN-speech-enhancer ffmpeg-dsp:<effects>`, and since the #306 fix in 1.9.2 (`7934fac`,
  2026-08-30) that flag has `choices=SPEECH_ENHANCER_CHOICES`, which does not include the colon
  form. **Reproduced:** `--pass1-speech-enhancer ffmpeg-dsp:loudnorm` exits 2 with
  `invalid choice: 'ffmpeg-dsp:loudnorm'` before any audio is read. So a GUI two-pass run with
  FFmpeg DSP selected fails immediately. The same colon form is why the notebook's four filter
  checkboxes had to go. **Not fixed** — the repair changes what the program accepts, which is his.
  The recommendation is to accept `ffmpeg-dsp:<effects>` again, validating the part before the colon
  against the same list and the effects against the backend's own names, which restores the effect
  choices and keeps the typo protection #306 asked for.

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
