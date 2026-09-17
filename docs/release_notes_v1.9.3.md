# WhisperJAV 1.9.3 — detailed record

> ⚠ **This is NOT the note to publish.** The text for the release page is
> `docs/release_notes_v1.9.3_for_users.md`.
>
> This file is organised by what changed for a user, so a support answer can be found quickly.
> `docs/CHANGELOG_v1.9.3.md` carries the same material in the order it landed, with the reasoning,
> the files and the verification behind each entry.

**Theme:** more bug fixes and overall improvements. 1.9.3 builds on 1.9.2 — where that release was
about performance and the Balanced pipeline, this one is about fixing bugs and making small
improvements across the features. The largest single item is a group of defects in the speech
enhancers.

---

## 1. Speech enhancement — the major fix

### "Enhance for VAD only" did not do what its name says

On the fidelity pipeline the option enhanced **both** tracks: the cleaned-up audio went to the
recogniser as well as to the voice detection. The whole point of the option is that the recogniser
hears the original. It now genuinely splits the two — the cleaned-up scenes drive segmentation, the
originals are resampled and transcribed.

Reachable on **fidelity** and **qwen**. Balanced **refuses** it (see §2) because it detects speech
inside Faster-Whisper's own call, on the very audio it transcribes; there is no second track.

### The control was almost always hidden

The GUI row only refreshed from the enhancer dropdown's own change handler, so a pass that already
had an enhancer — or got one from a preset — kept the tick box hidden until the user re-picked the
enhancer by hand.

### The row and the Customize window disagreed

Ticking the option inside the Customize window set the value but never repainted the row, so the row
showed "off" while the run used "on". Fixed twice: the repaint now sets the box from the stored
value, and Apply now calls the repaint. *(Found by the owner's GUI test A4.)*

### FFmpeg DSP could not be chosen from the GUI

The GUI built `ffmpeg-dsp:<effects>`, a form the CLI had refused since 1.9.2, so the run stopped with
a usage error before reading any audio. The `backend[:detail]` form is accepted again, on the
per-pass flags and on `--qwen-enhancer`, and an unknown effect is still rejected at the boundary.

### A failed scene broke the whole file

When a clean-up failed on one scene, the untouched 48 kHz original was returned while the enhanced
scenes were resampled to 16 kHz. In the dual-track split those lists are paired by position and the
recogniser refuses two tracks at different rates, so one transient failure failed the entire file
with a message about sample rates. Everything returned is now at 16 kHz.

### htdemucs vocal isolation (new)

Demucs v4, separating voice from music and effects. **Installed with WhisperJAV** like the other
enhancers — it is in the `enhance` extra, which is what the Windows installer builds from. Its model
is ~84 MB from `dl.fbaipublicfiles.com` (Meta), **not** Hugging Face, so `--offline`, `--hf-endpoint`
and any Hugging Face mirror do not affect it.

### Failure rules, as agreed with the owner

Recorded in `docs/plans/V193_ERROR_HANDLING_TABLE.md`.

| Situation | Behaviour |
|---|---|
| Chosen clean-up not installed, or cannot start | Run stops before audio is read, with what to do |
| Clean-up fails on **some** scenes | Those scenes go through uncleaned; file reported **suspect** with a count |
| Clean-up fails on **every** scene | That file fails — it ran and achieved nothing |

The shortfall reaches the RUN SUMMARY on every path, including two-pass, where each pass runs in its
own process and the notes are carried back and labelled `pass 1:` / `pass 2:`.

---

## 2. Changed behaviour (breaking)

| Change | Was | Is |
|---|---|---|
| `--mode balanced --enhance-for-vad` | accepted, silently ignored | **exit 2** with an explanation |
| `--pass{1,2}-pipeline balanced --pass{1,2}-enhance-for-vad` | accepted on the per-pass flag; the global flag bypassed the refusal entirely | **exit 2** either way |
| A clean-up failing on every scene | untouched audio transcribed, run exited 0 | that file **fails** |
| `demucs` | not a dependency | an ordinary dependency in the `enhance` extra |

`--enhance-for-vad`'s help text previously read "Qwen/Decoupled pipelines only", which was untrue. It
now names where it applies, where it is refused, and where it is accepted and ignored (fast, faster,
transformers, crispasr — the owner's decision, on condition it is documented).

---

## 3. Installing

- **The PyTorch download shows progress.** Every install ran through
  `subprocess.run(capture_output=True)`, so pip's output was held until the command finished — and
  pip prints nothing at all while a single wheel downloads. Installs now stream: progress lines as
  they happen, and a heartbeat with the elapsed time every 15 s when pip is silent. *(zoqapopita93,
  #314.)*
- **The network check names what it tried** — `pypi.org` and `download.pytorch.org` — and the proxy
  in use, and repeats both on failure. PyPI remains required and fatal; the PyTorch host is reported
  but not required, since the CPU route takes PyTorch from PyPI. A host answering 403 to a bare GET
  counts as reachable, which `download.pytorch.org` does.
- **No more writing to the user's global Git config.** Eight settings were written there
  permanently, including an `http.proxy`. They now go into the install's environment via
  `GIT_CONFIG_COUNT`, which reaches every git the install starts, including the one pip runs for a
  `git+https` install. On git older than 2.31 they are skipped with a plain line rather than written
  to disk.
- **The installer README told users about the wrong release.** Its "WHAT'S NEW IN v{{VERSION}}"
  section listed the v1.9.0 features as NEW under whatever version was built. Rewritten, and
  `build_release.py` now strips lines beginning `{{!` so a template can carry a maintainer note that
  never ships.
- **Linux:** the README and the Linux install script name the pywebview backend packages the GUI
  needs. *(thumper100, #366.)*
- **`--hf-endpoint <url>`** sends Hugging Face downloads to a mirror. Applied from a raw argv scan in
  both entry points, before anything imports `huggingface_hub`, which reads `HF_ENDPOINT` at import
  time. `--dump-params` reports both the value asked for and the endpoint the library ended up with.

---

## 4. Translation

- **French** was accepted and then failed; fixed. **Italian** *(giulub, #351)*, **Thai**
  *(yedkung69-ctrl, #268)* and **Korean** added.
- **Per-provider failure advice.** Everything ended with "check Ollama server logs" — including the
  local llama-cpp server. The advice keyed on `pysubtrans_name`, which is PySubtrans's *client class*
  and is `Custom Server` for local, ollama, glm, groq and custom alike, so the local and ollama
  branches were unreachable. Each config now carries `whisperjav_provider`, the name the user types.
  A cloud provider redirected to a `--translate-endpoint` on this machine is judged by the address.
- **The local server's log survives.** The advice named a file that `stop_local_server()` deleted as
  the run ended.
- **A local server that cannot chat is reported at start-up.** Some prebuilt llama-cpp-python builds
  declare a field on the chat reply their own code never sets, so FastAPI rejects the server's own
  answer and every chat request returns HTTP 500 — which appeared as four retries and nothing
  translated about 90 s in. A shim repairs the known form of that fault. The check **warns and
  continues** rather than stopping (owner's decision for this release): it has never met a real
  llama-cpp server, so a mistake in it would stop runs that were going to work.
- **LM Studio** is covered by the existing custom provider *(teijiIshida, #426)*.

---

## 5. What a run tells you

- Progress counts correctly; it read `[839/1]` in multi-file runs (#429).
- A long file listing ends with a summary line, and WhisperJAV's own leftovers are tagged. Nothing of
  the user's input is filtered out. *(Kukuindi, #302.)*
- Audio extraction announces itself and reports elapsed time. *(teijiIshida, #297.)*
- English post-processing steps appear without `--debug`. *(unretired1516, #372.)*
- A partly-failed clean-up is reported **suspect** with a count; `--fail-on suspect` makes it fatal.
- Two-pass settings are saved and restored in the GUI. *(yangming2027, #381; sky9639, #96.)*
- The Qwen Customize window and the row agree on which model runs.

---

## 6. Under the hood

- **Audio extraction stopped decoding the file twice.** The duration came from a second full decode
  (`-f null -`) of the extracted file. It now reads the header via ffprobe, then the extracted file's
  own WAV header, and only then falls back to the extraction log — which carries the *source*
  container's duration and gives nothing when the source prints `Duration: N/A`.
- **The test suite could not be run at all.** `pytest tests/` aborted during collection: one file ran
  its checks at import and ended in `sys.exit(0)`; four rewrapped `sys.stdout` and killed pytest at
  teardown; one opened five Tk windows and waited for a human. Collection now completes — 2784 tests
  — and `scripts/run_tests.py` runs the suite one file per process.
- `tests/test_gui_refactor.py` removed: it imported `whisperjav.gui.whisperjav_gui`, deleted when the
  Tkinter GUI was replaced.
- The package registry gained `demucs`. It is the single source of truth, and `install.py` installs
  from it, so a source install would otherwise have offered htdemucs with nothing behind it.

---

## 7. Known limitations

- Re-running with a new target language can return the previous run's output; delete the `.subtrans`
  beside the SRT. Scheduled for 1.10.x. *(ric-reff, #305.)*
- The **Transformers** Customize window still discards a model picked in it; the row decides.
- The Customize window's Enhancer tab ticks are collected nowhere — only the inline row panel reaches
  the command.
- Audio is extracted once per pass in a two-pass run.
- `Ensemble summary (Ns)` and `Total processing time` measure different things and can look
  contradictory: the latter is the sum of the passes, the former is wall-clock.
- htdemucs is fed 48 kHz though it wants 44.1 kHz, so the audio is resampled twice.
- The China mirror page, the GUI mirror checkbox and the pip index variable are **not** in this
  release; only `--hf-endpoint` is.

---

## 8. Merged pull requests

| PR | Author | What |
|---|---|---|
| #388 | AKB0700 | GitHub Actions check that compiles and imports the package on Python 3.12 |
| #363 | triatomic | Auto-scroll toggle in the GUI console header |
| #376 | Mimic-me | Accuracy regression scorer and gate; `whisperjav-accuracy-gate` command |
| #364 | triatomic | "contextual" translation tone, with tone-matched temperature defaults |

Accepted for a later release: **zhiyuchen1101**'s role-aware translation proposal (#393), on the
owner's condition that it reads a user-supplied file and fetches nothing.
