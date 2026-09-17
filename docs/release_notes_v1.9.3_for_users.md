# WhisperJAV 1.9.3

**More bug fixes and overall improvements.**

1.9.3 builds on 1.9.2. Where 1.9.2 was about performance and the Balanced pipeline, 1.9.3 is about
fixing bugs and making small improvements across the features.

One of those fixes is larger than the others: **a major bug in the speech enhancers**, which is worth
reading about even if you have never used one.

---

## Contents

- [The recommended recipe](#the-recommended-recipe) — the settings that give the best results
- [What will break](#what-will-break) — **read this if you run WhisperJAV from a script**
- [The speech-enhancer bug](#the-speech-enhancer-bug) — what was wrong, and what is fixed
- [Installing, proxies and mirrors](#installing-proxies-and-mirrors)
- [Translation](#translation)
- [Progress, messages and saved settings](#progress-messages-and-saved-settings)
- [Every fix, and who reported it](#every-fix-and-who-reported-it)
- [What people contributed](#what-people-contributed) — the pull requests merged
- [Upgrading](#upgrading)
- [Credits](#credits)

---

## The recommended recipe

For more accurate subtitles and more accurate timing, at the cost of a little more time:

1. Choose a **Speech Enhancer** — **htdemucs** (Demucs v4), or any of the stable enhancers that
   suits your machine.
2. Tick **Enhance for VAD only**.

What that does: the enhancer separates the voice from the music and the effects, and produces a
clean, voice-only track (a *vocal stem*). WhisperJAV uses that clean track only to decide where each
piece of speech starts and stops — the step known as VAD, voice activity detection. The words
themselves are still transcribed from the **original, untouched** audio. Preparing one track and
recognising the other — in parallel rather than one after the other — is what improves the accuracy.

The maintainer's own favourite is **Qwen ASR + htdemucs + Enhance-for-VAD-only + TEN VAD**.

On the command line, all on one line:

```
whisperjav video.mkv --ensemble --pass1-pipeline qwen --pass1-model Qwen/Qwen3-ASR-1.7B --pass1-speech-enhancer htdemucs --pass1-enhance-for-vad --pass1-speech-segmenter ten
```

In the GUI: the **Ensemble Mode** tab, Pass 1 row — Pipeline **Qwen3-ASR**, Speech Enhancer
**Vocals** (under *htdemucs (Demucs v4)*), tick **Enhance for VAD only**, Speech Segmenter
**TEN VAD**.

Notes:

- **htdemucs installs with WhisperJAV.** You do not install it separately. Its model is about 84 MB
  and is fetched once, from Meta's own site — not from Hugging Face, so a Hugging Face mirror does
  not redirect it.
- **Enhance for VAD only works on the Fidelity and Qwen pipelines.** Balanced refuses it and says
  so: Balanced finds the speech inside Faster-Whisper itself, on the very audio it transcribes, so
  there is no second track to hand the cleaned-up audio to.
- **It costs time.** Separating the voice out is real work on top of the transcription.

---

## What will break

**Three things behave differently in 1.9.3. Each one can turn a run that used to finish and report
success into a run that stops, or that is reported as failed. If you drive WhisperJAV from a script
or a batch file, read this section before you upgrade.**

**1. Flags that used to be ignored now stop the run.**

`--mode balanced --enhance-for-vad` now exits with status 2 and an explanation, instead of being
accepted and quietly doing nothing. The same applies to a Balanced pass inside a two-pass run
(`--pass1-pipeline balanced --pass1-enhance-for-vad`).

If your script passes that combination, it has never done what it looked like it was doing. Remove
the flag, or change the pipeline to `fidelity` or `qwen`.

**2. A clean-up that fails everywhere in a file now fails that file.**

WhisperJAV cuts each file into sections, which it calls *scenes*, and the enhancer cleans them one at
a time. If the enhancer fails on **every** scene in a file, that file is now reported as **failed**.
Before, it was transcribed from the untouched audio and the run reported success.

A clean-up that fails on *some* scenes only is not fatal: those scenes go through uncleaned, and the
file is reported as **suspect**, with a count, so you can see it happened.

**3. An enhancer that cannot start stops the run.**

If you choose an enhancer that is not installed, or whose model will not load, the run stops before
any audio is read, and says what to do. Before, you could get a subtitle file made from audio you had
asked to have cleaned up, from a run that exited 0.

---

## The speech-enhancer bug

If you used a speech enhancer in 1.9.2 or earlier, it may not have been doing what you asked.

- **"Enhance for VAD only" did not do what its name says.** On the Fidelity pipeline it enhanced
  *both* tracks, so the cleaned-up audio was transcribed as well — exactly what the option exists to
  avoid. It now genuinely splits the two: enhanced audio for finding the speech, original audio for
  transcribing.
- **The control was almost always invisible in the GUI.** The "Enhance for VAD only" row only
  appeared if you re-picked the enhancer by hand after opening the window, so most people never saw
  it.
- **The row and the Customize window disagreed.** Ticking the option inside the Customize window left
  the row showing "off" while the run used "on".
- **FFmpeg DSP could not be chosen from the GUI at all.** Picking it built a value the program had
  refused since 1.9.2, so the run stopped before reading any audio.
- **One failed scene broke the whole file.** When a clean-up failed on a single scene, that scene
  came back at the wrong sample rate, and the run then failed with a message about sample rates
  rather than about the clean-up.

**Also new in this release: htdemucs** vocal isolation, which separates the voice from the music and
the effects.

---

## Installing, proxies and mirrors

**The installer no longer looks frozen.** It has to download several gigabytes — PyTorch, the AI
library WhisperJAV runs on — and it used to show none of that, so a slow connection looked exactly
like a dead installer. It now shows what it is doing as it happens, and when there is nothing new to
report it prints how long it has been working, every fifteen seconds. *(Reported by zoqapopita93,
#314.)*

**If the network check fails, it now tells you what it tried.** It names the addresses — `pypi.org`
and `download.pytorch.org` — and the proxy in use. The failure file repeats them, and says to set
`HTTPS_PROXY` and `HTTP_PROXY` if you reach the internet through a proxy.

**The installer no longer writes into your global Git configuration.** Earlier versions left eight
settings there permanently, including an `http.proxy` pointing at a proxy that might only exist that
day, and nothing ever offered to undo them. Those settings now apply to the installation only.

**Linux:** the GUI needs a pywebview backend in your environment. The README and the Linux install
script now say which packages. *(Reported by thumper100, #366.)*

**Models from a mirror:** `--hf-endpoint <url>` sends Hugging Face downloads to an address you
choose, for machines that cannot reach huggingface.co. It covers what goes through Hugging Face —
not Silero, the openai-whisper weights, the ModelScope enhancers, or htdemucs, which come from
elsewhere.

---

## Translation

- **French works.** Choosing it used to be accepted, and then fail.
- **Italian, Thai and Korean added.** *(Requested by giulub, #351, and yedkung69-ctrl, #268.)*
- **Failure advice now names the right server.** Every failure used to end with "check Ollama server
  logs", including failures of the local llama-cpp server, which has nothing to do with Ollama. Now
  the local provider points at the log WhisperJAV itself writes, Ollama gets Ollama's, a server you
  gave an address for is yours, and a cloud service is told there is no local log to read.
- **A local server that cannot answer is reported at start-up**, with the reason, instead of failing
  ninety seconds into a run with nothing produced. Some prebuilt builds of llama-cpp-python reject
  their own replies. WhisperJAV now starts that server through a shim that repairs the known
  form of this fault. It has been tested against the fault's exact shape, but not yet against a
  build that actually has it, so tell us if you still see it.
- **LM Studio and similar are already supported**, through the existing custom provider:
  `--translate-provider custom --translate-endpoint <url> --translate-model <name>`.
  *(Asked by teijiIshida, #426.)*

---

## Progress, messages and saved settings

- **The progress line counts properly.** It used to read `[839/1]` in a multi-file run. *(#429.)*
- **A long file listing ends with a summary line**, and files left over from an earlier WhisperJAV
  run are tagged as such. Nothing of yours is filtered out. *(Reported by Kukuindi, #302.)*
- **Audio extraction says it is working, and how long it took**, instead of sitting silent.
  *(Reported by teijiIshida, #297.)*
- **The English post-processing steps appear in the normal console**, not only with `--debug`.
  *(Reported by unretired1516, #372.)*
- **A file whose clean-up only partly succeeded is reported as suspect**, with the reason and the
  count, rather than passing silently as done. `--fail-on suspect` makes the run exit non-zero.
- **Two-pass settings are saved and restored** in the GUI. *(Asked by yangming2027, #381, and
  sky9639, #96.)* Saving the values inside the Customize window is a separate piece of work, still
  to come.
- **The Customize window and the row now agree on which model runs.** Picking a model in the window
  used to display as accepted, and then be discarded.

---

## Every fix, and who reported it

| What was wrong | Who reported it |
|---|---|
| Installer stuck at the PyTorch phase with no progress | zoqapopita93 (#314) |
| `ModuleNotFoundError: No module named 'gi'` on Linux | thumper100 (#366) |
| Batch transcription appeared to stop early | unretired1516 (#372) |
| Blank subtitle file after a run reported success | teijiIshida (#297) |
| Whole-folder runs slow, listing scrolled past | Kukuindi (#302) |
| Two-pass settings not kept between sessions | yangming2027 (#381), sky9639 (#96) |
| The Colab notebook failed on its own defaults | jasial2 (#407) |
| Italian translation target | giulub (#351) |
| Thai translation target | yedkung69-ctrl (#268) |
| Progress line `[839/1]` | reported internally (#429) |

**Answered rather than fixed:** LM Studio and similar local servers are already supported through the
existing custom provider — see [Translation](#translation). *(teijiIshida, #426.)*

**Looked at and found working:** timing and omissions across pipeline combinations were investigated
and the reported panel behaviour could not be reproduced; nothing was changed.
*(kylesskim-sys, #425.)*

**Still open, not fixed here:**

- Re-running with a new target language can return the previous run's output. The workaround is to
  delete the `.subtrans` file beside the SRT. *(ric-reff, #305.)*
- A custom translation model being rejected is still being investigated; the path was traced and
  looks correct, so the error text from the thread is needed to go further. *(ArenaSora, #420.)*
- **Kaggle is not maintained.** The Kaggle notebook is left as it is; the Colab notebook is the one
  that was rebuilt. *(fzfile, #231 and #329; jasial2, #330 and #321.)*

---

## What people contributed

Four pull requests were merged into this release. Each was written by someone outside the project and
is in your copy of 1.9.3.

| What it adds | Who wrote it | Pull request |
|---|---|---|
| An **Auto-scroll** toggle in the GUI console header, so the log can follow the run or hold still while you read it | triatomic | #363 |
| A **"contextual" translation tone** — explicit only where the original is explicit — with temperature defaults matched to each tone | triatomic | #364 |
| An **accuracy regression gate**: a scorer, baselines and the `whisperjav-accuracy-gate` command, so a change that quietly makes subtitles worse can be caught | Mimic-me | #376 |
| A **continuous-integration check** that compiles and imports the package on Python 3.12 on every push | AKB0700 | #388 |

Accepted for a later release: **zhiyuchen1101**'s proposal for a role-aware translation layer, with
character anchors read from a file you supply (#393).

---

## Upgrading

- **Windows installer:** download and run the new `.exe`. It installs over the existing copy, and
  you do not have to remove the old one first.
- **Source install:** `whisperjav-upgrade`.
- **Google Colab:** no action required — the notebook installs the release itself.

Everything else is unchanged: your settings, your output folders and your models stay where they
are. The first run after upgrading may fetch the htdemucs model, if you choose that enhancer.

> **If you upgrade manually with pip:** `demucs` is a new dependency in this release, so an upgrade
> with `--no-deps` will leave htdemucs offered in the GUI with nothing behind it. It brings two
> small companions (`lameenc`, `sphn`) and does not change your PyTorch.

---

## Credits

**Code contributed:** **triatomic**, **Mimic-me** and **AKB0700** — see
[What people contributed](#what-people-contributed).

**Reports and findings that shaped this release:**

- **weifu8435** — sustained testing, settings comparisons and detailed reports (#413 and others)
- **yangming2027**, **sky9639**, **teijiIshida**, **Kukuindi**, **unretired1516**, **thumper100**,
  **zoqapopita93**, **kylesskim-sys**, **jasial2**, **fzfile**, **giulub**, **yedkung69-ctrl**,
  **ArenaSora**, **ric-reff** — for the reports behind this release. Some became the fixes above;
  others were investigated, answered, or are still open. All of them were useful.

Thank you. Reports with a command line, a log and a sample are what make these fixable.
