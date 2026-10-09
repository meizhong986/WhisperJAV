# WhisperJAV 1.9.4

**Bug fixes, hardening, and a step toward better subtitle timing for ChronosJAV.**

1.9.4 builds on 1.9.3. It changes no library underneath WhisperJAV, so **on 1.9.3 you can upgrade
with one command** — see [Installing and upgrading](#installing-and-upgrading).

Two changes are larger than the rest:

- **Subtitles stay in time in films with holes in their audio.** Some downloaded films have hidden
  holes in their audio track. In those files every subtitle after the first hole came out early — by up to 10 seconds in
  the films we measured. WhisperJAV now fills the holes with silence, so the times stay right, and it
  tells you the file was damaged.
- **An attempt to make ChronosJAV lines end closer to the speech**, for both Qwen3-ASR and
  anime-whisper. On drama scenes, the typical late end shrank by 10 to 24 %, with the text about the
  same.

---

## Contents

- [Damaged audio: found, timed right, reported](#damaged-audio-found-timed-right-reported)
- [ChronosJAV timing](#chronosjav-timing)
- [What will break](#what-will-break) — **read this if you run WhisperJAV from a script**
- [Translation](#translation)
- [Other fixes](#other-fixes)
- [Installer, Colab and PyAV 19](#installer-colab-and-pyav-19)
- [Every change, and who reported it](#every-change-and-who-reported-it)
- [Installing and upgrading](#installing-and-upgrading) — **on 1.9.3? one command**
- [Credits](#credits)

---

## Damaged audio: found, timed right, reported

**What was wrong.** A film can arrive with holes in its audio track: a few audio packets claim to
last about two seconds but carry only a fraction of that. A video player waits through each hole, so
you never notice. WhisperJAV's audio extraction did not wait: it joined the sound on either side.
From the first hole on, the extracted audio ran ahead of the video, and so did every subtitle made
from it. In one film with five such holes, one every 12 minutes, the subtitles were 2 seconds early
after the first hole and 10 seconds early after the last, for the final 38 minutes of the film.

In one library, 4 of the 40 video files checked had this same fault. The check found nothing in
the other 36 (MP4, MKV, AVI and TS).

**What 1.9.4 does.**

- **Keeps the times right.** The extraction now fills each hole with silence, the way a player
  does, so subtitle times stay in step with the video. Any speech that was inside a hole is missing from the file
  itself, and cannot be recovered.
- **Checks every file.** While the audio is being extracted, WhisperJAV also reads the file's
  audio layout and looks for holes, decoding errors, and audio shorter than the track says it is.
  The console says what it found:

  ```
  Audio check: no problems found (1.8 s).
  ```

  The time in brackets is how much the check added to the audio extraction.
- **Marks the file "suspect" and carries on.** A damaged file is transcribed as usual. In the
  summary at the end of the run it is listed as `suspect`, with the reason — for example
  `audio integrity: 1 gap(s) in the audio track, 2.0 s in all (2.0 s at 0:00:53)`.
- **Or stops it, if you prefer.** With `--fail-on suspect`, or the **Treat 'suspect' files as
  failures** box on the GUI's *Transcription Adv. Options* tab, a damaged file is stopped before
  transcription, and the run exits 1. It is the same switch that already makes a run exit 1 when a
  file is `suspect` for another reason, such as subtitles that stop early.

The holes are filled on every pipeline. The check's warning, the `suspect` mark and the stop work
on Balanced, Fidelity, ChronosJAV (Qwen3-ASR and anime-whisper) and two-pass runs.

---

## ChronosJAV timing

In its default timing mode, ChronosJAV (Qwen3-ASR and anime-whisper) takes its subtitle times from
the speech segmenter, not from the words. In 1.9.3 most lines ended late, often running over the
start of the next line. The main cause was a limit on how long one piece of speech could be. In
long stretches of talk, a line ended where that limit fell, not where the speaker paused.

**What 1.9.4 changes:**

- **The longest segment is 4 seconds** for both models at every sensitivity, with the WhisperSeg,
  TEN and FireRedVAD speech segmenters. In 1.9.3 it was 6, 5 and 4 seconds for conservative,
  balanced and aggressive (7, 6 and 5 with FireRedVAD).
- **TEN cuts long speech at the quietest point, and never past the limit.** 1.9.3 cut at the first
  small dip after 80 % of the limit, sometimes inside a word, and some pieces ran up to 2 seconds
  over it. This applies wherever TEN is used.
- **anime-whisper with WhisperSeg, aggressive:** a slightly stricter rule for extending a segment,
  so fewer segments run on into the next line.
- **anime-whisper with WhisperSeg, conservative and balanced:** a segment that reaches the limit is
  now cut where the speech is weakest within its last 70 %. 1.9.3 looked only in the last 40 %, and
  when it found no clear dip there it cut exactly at the limit, often inside a word.
- **Qwen3-ASR with WhisperSeg: no padding after each piece of speech** (it was 100 ms). Lines end a
  little closer to the speech; the text stayed the same.
- **anime-whisper hears 200 ms of silence before each piece of audio**, except with the Silero speech
  segmenters, where it is off. Where each piece starts and
  ends does not change. A new setting controls it: **Silence Before Each Window (ms)**, in the
  *Customize Parameters* window of an anime-whisper pass (0 to 500), or `--qwen-leading-silence MS`
  on the command line (anime-whisper only, despite the flag's name). `0` turns it off.
- **The *Customize Parameters* window shows the values that will actually run** for the pass's speech segmenter,
  and **Apply** sends them. Before, it could show values that the run then replaced.

**Why, and how far it got.** Many of you told us that ChronosJAV's timing was not right: Timi2028
(#433, #421), and weifu8435 — late, trailing ends on short lines even with htdemucs and Enhance for
VAD only (#437), and the wider question of how to get better timing (#417, #427). 1.9.4 is an
attempt to improve this with the settings WhisperJAV already has. It is a step, not a solution.

We measured it on seven TV-drama scenes with reference subtitles (305 lines), comparing the 1.9.3
defaults with 1.9.4's, with WhisperSeg. "How far off a line's end is" is the typical (median) gap
between our line's end and the reference line's end, measured where our line ends a reference line
in both versions. "Text error" is the share of characters wrong, missing or extra (lower is better).

| Model and sensitivity | How far off a line's end is | Ends within 0.5 s (share of those measured) | Text error |
|---|---|---|---|
| Qwen3-ASR, balanced | 0.47 s → 0.36 s (24 % less) | 53 % → 60 % | 0.390 → 0.391 (about the same) |
| anime-whisper, conservative | 0.35 s → 0.32 s (10 % less) | 61 % → 68 % | 0.397 → 0.399 (about the same) |
| anime-whisper, balanced | 0.44 s → 0.36 s (19 % less) | 57 % → 66 % | 0.400 → 0.394 (1.5 % better) |
| anime-whisper, aggressive | 0.43 s → 0.37 s (14 % less) | 54 % → 58 % | 0.392 → 0.387 (1.3 % better) |

Where a line starts barely moved: within 0.05 s either way.

Shorter limits — 3 seconds and below — moved the line ends even closer, but they cost text: the
model wrote less, or wrote it wrong. 4 seconds was the shortest limit that kept the text about the
same.

**Limits — please read these before expecting a big change.** These are drama scenes, not JAV;
scenes full of moans and action sound were not measured, so we do not yet know how much of this
carries over to your films. Qwen3-ASR was measured at balanced only. TEN and FireRedVAD at 4 seconds
were checked at conservative only: lines no longer run to 6 or 7 seconds, line ends did not move,
and about 1 to 2 % less of the text was right. Lines still end late on
average — less late than before. About one line in ten still loses its opening words, because the
model never hears them; 1.9.4 does not change this. A new aligner (#440) or a different pairing of
speech detector and model (#418) is not part of this release. If you try 1.9.4 on the films where
the timing bothered you, please tell us in those issues whether it is better, worse or the same.

The ChronosJAV parameter guide — the **?** button beside a Qwen3-ASR or anime-whisper pass in the
GUI — is updated to match, including recommendations that were out of date. *(Reported by
Angelholl, #436.)*

---

## What will break

**These change what a run produces, or how it ends. If you drive WhisperJAV from a script or a batch
file, read this before you upgrade.**

**1. `--fail-on suspect` now also stops a file with damaged audio** before transcription, and the run
exits with status 1. Without the flag, nothing stops: a damaged file is transcribed and reported as `suspect`.

**2. A ForcedAligner that cannot be loaded no longer fails the run.** With ChronosJAV's aligner
timing, a failed aligner download or load used to end the run with exit 1, and the transcribed text
was lost. Now the text is kept, its times come from the speech segmenter instead, the file is
reported as `suspect` with the reason, and the run exits 0. Use `--fail-on suspect` if you want the
old exit status. *(Reported by Angelholl, #436.)*

**3. A custom translation server now chooses its own temperature.** With the custom
(OpenAI-compatible) provider, WhisperJAV no longer sends a temperature unless you give one
(`--temperature` on the command line). Before, it always sent its tone's default — 0.8 for
contextual — which overrode the server's own setting. In the GUI's translation settings, the
Temperature field is labelled *Custom provider: set by your server (not sent)*. To send one as
before, pass `--temperature` with the value you want (1.9.3 sent 0.5 for the standard tone, 0.8 for
contextual and 1.2 for pornify). Other providers are unchanged. *(Reported by teijiIshida, #444.)*

**4. The Gemini default model is now `gemini-3.6-flash`.** Google has shut down `gemini-2.0-flash`,
the old default. *(#291, #346.)*

**5. Single-pass Fidelity now uses the TEN, WhisperSeg or whisper-vad speech segmenter you choose.**
In 1.9.3 those three were quietly replaced by silero-v3.1, so your output will differ if you chose
one; to get 1.9.3's output, choose silero-v3.1. Any other segmenter that single-pass Fidelity cannot
use is still replaced, as in 1.9.3, and the warning now says which ones it can use. Every segmenter works in a two-pass run. *(#323.)*

**6. Subtitles change.** ChronosJAV output differs from 1.9.3 (see
[ChronosJAV timing](#chronosjav-timing)). On a file with damaged audio, every subtitle after the first
hole moves later, to where the speech actually is. On Balanced and Faster, a speech piece shorter
than 0.1 s is now joined to its neighbour (see [Other fixes](#other-fixes)).

---

## Translation

- **Gemini works again for new users.** The default model is now `gemini-3.6-flash`, and the GUI
  offers `gemini-3.6-flash`, `gemini-3.8-flash` and `gemini-3.5-flash-lite`. *(ktrankc, #291;
  infinitebook, #346.)*
- **Two-pass translation to a custom server sends your API key.** The Ensemble Mode tab left the key
  out for the custom provider, so a server that needs one refused every request. The AI SRT
  Translate tab already sent it. *(Reported by ArenaSora, #420.)*
- **Your translation choices are remembered** between launches. On the Ensemble Mode tab: the
  provider and model. On the AI SRT Translate tab: also the languages, tone, server address and
  batch settings. Not kept: the API key typed into the AI SRT Translate tab, and the per-film details
  (title, names, plot). *(Requested by ArenaSora, #435.)*
- **A custom server keeps its own temperature** — see [What will break](#what-will-break), item 3.
  *(teijiIshida, #444.)*
- **Ollama uses a model you already have, instead of failing.** When you name no model, WhisperJAV
  picks one by your GPU's memory. If that model is not downloaded, and the run can neither download
  it nor ask you, WhisperJAV now uses the largest suitable model you already have, and says so. A
  model you name is never swapped. *(Reported by roninamongus, #356, who also proposed the change.)*
- **The instructions sent to the translation model name your target language**, and ask for the
  summary and scene description in it too. This is aimed at the reports of English lines, or the
  model's own notes, appearing in translations to Chinese, mostly with local models. If you still
  see that in 1.9.4, please tell us, with the model you used. *(bdd-bit2, #347; ric-reff, #305;
  cbl19961214-sudo, #339; yhxkry, #397; triatomic, #296.)*

---

## Other fixes

- **A crash at the edge of a scene, on Balanced and Faster.** A speech piece of a few milliseconds
  at the very end of a scene could crash the whole process (exit code `3221225620`). Pieces shorter
  than 0.1 s are now joined to the nearer neighbour, and any piece still too short is skipped. *(Reported by
  AlanZ-Git, #424, who also proposed the skip.)*
- **The command line finds the bundled FFmpeg** on Windows when you run `whisperjav` without
  activating its environment first. *(Angelholl, #436.)*
- **`whisperjav-upgrade` no longer waits forever.** Its compatibility check now gives up after five
  minutes. This takes effect from your next upgrade *after* this one: upgrading to 1.9.4 still uses
  1.9.3's tool — see [Installing and upgrading](#installing-and-upgrading). *(Angelholl, #436.)*
- **Guides and messages corrected.** The Mac guide used `--model` with transformers mode, which
  ignores it; it now uses `--hf-model-id` *(dadlaugh, #227)*. Cohere-Transcribe was shown as
  "coming in v1.9.0"; it now says "not available yet" *(lesspem, #385)*. `--provider local` said it
  would be removed in v1.9.0; it now says "in a later release" *(teijiIshida, #262)*.

---

## Installer, Colab and PyAV 19

- **Windows installer: a damaged download cache.** PyTorch could "install" in seconds from a damaged
  uv download cache and then fail to start. The installer now checks that PyTorch actually loads.
  If it does not, it downloads PyTorch once more, past the cache. If that also fails, it tells you
  to delete `%LOCALAPPDATA%\uv\cache` and run the installer again. *(Reported by gagagnier67,
  #438.)*
- **PyAV 19.** If you installed WhisperJAV on Python 3.12 or 3.13 on or after 29 September 2026 —
  on Linux, macOS, Colab or from source — 1.9.3 could stop with
  `TypeError: open() got an unexpected keyword argument 'metadata_errors'`. PyAV 19, released that
  day, removed an argument that Faster-Whisper still passes. 1.9.4 works with PyAV 19; upgrading
  to 1.9.4 is all you need to do. Users of the Windows installer were not affected: it uses
  Python 3.10, which cannot install PyAV 19.
- **Colab.** The notebook now installs 1.9.4. When Colab's PyTorch is built for CUDA 13, the
  notebook first switches it to the CUDA 12.6 build, because Faster-Whisper needs CUDA 12; without
  that, every scene failed with `libcublas.so.12`. *(Reported by a-pikachu, #447.)* Full CUDA 13
  support is planned for a later release.

---

## Every change, and who reported it

| What was reported | What 1.9.4 does | Who reported it |
|---|---|---|
| Subtitles drift early in some downloaded films | Fills audio holes with silence; checks every file; reports damaged files | found in the maintainer's testing |
| ChronosJAV timing not accurate; lines end late | An attempt: 4-second longest segment; anime-whisper split settings and 200 ms of silence before each piece (not with Silero) | Timi2028 (#433, #421), weifu8435 (#417, #437, #427) |
| Edge-of-scene crash on Balanced and Faster | Joins very short speech pieces; skips any still too short | AlanZ-Git (#424) |
| FFmpeg not found from the command line; aligner failure loses the work; `whisperjav-upgrade` hangs; ChronosJAV guide out of date | See [Other fixes](#other-fixes) and [What will break](#what-will-break) | Angelholl (#436) |
| Custom model in two-pass translation fails | Sends the API key to the custom server | ArenaSora (#420) |
| Translation settings forgotten | Remembered on both tabs | ArenaSora (#435) |
| Ollama picks a model that is not downloaded | Falls back to a downloaded model | roninamongus (#356) |
| Custom server's temperature overridden | Not sent unless you set it | teijiIshida (#444) |
| Gemini fails for new users | New default model | ktrankc (#291), infinitebook (#346) |
| Installer: PyTorch fails to load after install | Checks, and downloads once more | gagagnier67 (#438) |
| Colab: `libcublas.so.12` on every scene | Notebook switches to the CUDA 12.6 build | a-pikachu (#447) |
| Wrong guidance in guides and messages | Corrected | dadlaugh (#227), lesspem (#385), teijiIshida (#262) |
| `metadata_errors` error on new installs | Works with PyAV 19 | a private report |

**Changed — please tell us whether it helps:**

- Single-pass Fidelity and the TEN, WhisperSeg and whisper-vad segmenters. *(TinyRick1489, #323.)*
- English or notes in Chinese translations. *(bdd-bit2, #347; ric-reff, #305; cbl19961214-sudo,
  #339; yhxkry, #397; triatomic, #296.)*

**Looked at, nothing changed:** Balanced stopping early, or returning empty subtitles, on long runs.
None of the reporters who answered the recheck on 1.9.3 saw it again, so nothing was changed. *(#357, #343, #263, #414.)*

**Still open, not in 1.9.4:**

- GTX 10xx and V100 cards: "no kernel image" errors. The PyTorch that is installed supports newer
  cards only. *(#326, #333, #411.)*
- Saving the values inside the Customize window is planned for 1.10.
- Full CUDA 13 support on Colab. *(#447.)*
- **Kaggle is not maintained.**

---

## Installing and upgrading

What to do depends on the version you are coming from.

### If you are on 1.9.3

**One command:**

```
whisperjav-upgrade --wheel-only
```

1.9.4 changes no library: the packages underneath WhisperJAV are exactly those of 1.9.3. So only
WhisperJAV itself needs replacing, and that is all `--wheel-only` does. It fetches 1.9.4 from GitHub,
skips the compatibility check, and on Windows updates the desktop shortcut. It does not make a
rollback snapshot, so `whisperjav-upgrade --rollback` has no 1.9.3 snapshot to return to afterwards.

We recommend `--wheel-only`. A full `whisperjav-upgrade` has no library changes to make, and it
first runs 1.9.3's compatibility check, which can hang (#436).

### If you are on 1.9.2

`whisperjav-upgrade --wheel-only` works here too. The one package added since 1.9.2 is `demucs`,
for htdemucs vocal isolation, which `--wheel-only` does not install. If you want htdemucs, run this
once afterwards:

```
pip install demucs
```

Without it, nothing breaks quietly: choosing htdemucs stops the run and tells you it is not
installed.

### If you are on anything older than 1.9.2

**Please do a fresh install**, using the route for your platform below. 1.9.2 pinned two libraries
underneath WhisperJAV to exact builds, and an in-place upgrade from an older version can leave your
old versions of those behind.

### Fresh install, by platform

| Platform | How |
|---|---|
| **Windows** (recommended) | Download and run the `.exe` from the [Releases page](https://github.com/meizhong986/WhisperJAV/releases/latest). No Python knowledge needed; it detects your NVIDIA driver and installs the matching CUDA build. |
| **Windows, from source** | `installer\install_windows.bat` (add `--cpu-only` to force CPU) |
| **macOS (Apple Silicon)** | `./installer/install_mac.sh` — M-series chips get MPS acceleration; `--mode transformers` performs best there |
| **Linux** | `./installer/install_linux.sh` — needs the NVIDIA driver (450+), but not the CUDA Toolkit |
| **Google Colab** | Nothing to install. Use the notebook badge in the README. |

> **Do not upgrade with a plain `pip install -U whisperjav...`.** On Windows, PyPI serves the
> **CPU-only** build of PyTorch, so any install that touches PyTorch will silently replace your GPU
> build with a CPU one. Use `whisperjav-upgrade`, or the installer for your platform.

### In every case

Your settings, your output folders and your models stay where they are.

---

## Credits

**Reports and findings that shaped this release:**

- **Angelholl** — a long, careful comparison of pipelines on four full films, which found four of the
  problems dealt with here (#436)
- **AlanZ-Git** — found the edge-of-scene crash, and proposed the skip (#424)
- **Timi2028** and **weifu8435** — kept asking for better ChronosJAV timing, with settings and
  examples (#433, #421, #417, #437, #427, #418, #440)
- **ArenaSora**, **teijiIshida**, **roninamongus**, **gagagnier67**, **a-pikachu**, **ktrankc**,
  **infinitebook**, **bdd-bit2**, **ric-reff**, **cbl19961214-sudo**, **yhxkry**, **triatomic**,
  **dadlaugh**, **lesspem**, **TinyRick1489** — for the reports behind this release
- the sender of a private report on PyAV 19

Thank you. Reports with a command line, a log and a sample are what make these fixable.
