# WhisperJAV 1.9.2

This release is mostly about **failures that used to stay quiet**.

If you have ever started a run, come back half an hour later, and found an empty
subtitle file with a message saying everything went fine — that is the kind of thing
this release is about. In several situations WhisperJAV could finish, report success,
and hand you nothing, or quietly ignore a setting you had chosen. Those situations are
what most of the work went into.

One thing has genuinely got faster, and there is one new thing on screen. Otherwise the
subtitles you get are much the same as before. That is deliberate.

---

## The short version

- **Aggressive is roughly three to four times faster** on Balanced and Fidelity, and
  finishes runs it used to be abandoned on.
- **Empty subtitle files** caused by a decoding setting are fixed. This affected some
  graphics cards and not others, which is why it was hard to pin down.
- **RTX 50 cards (5060, 5070, 5080, 5090)** no longer get a setting that was garbling
  their output.
- **A failed install now says so** instead of finishing and leaving you with a shortcut
  to something broken.
- **New:** after it works out the scenes, WhisperJAV tells you if your audio looks
  difficult, and roughly where.

---

## If you are new here

WhisperJAV turns the speech in a video into a Japanese subtitle file. You point it at a
file, it thinks for a while, and it writes an `.srt` next to it.

**Installing on Windows.** Download `WhisperJAV-1.9.2-Windows-x86_64.exe` from the
release page and run it. It does not need administrator rights and installs into your own
user folder. Budget **10 to 20 minutes** — it downloads a lot, and how long depends
entirely on your internet connection. The first time you actually transcribe something it
downloads the AI model as well, about 3 GB, which adds another 5 to 10 minutes. After
that, models are cached and start-up is quick.

When it finishes you get a desktop shortcut. If something went wrong during installation,
you will be told, and no shortcut is created — that is new in this release, and it exists
because people were getting a shortcut to an installation that could not transcribe.

**Which mode should I use?** Start with **Balanced**. It is the default and it is the one
that gets the most attention. The sensitivity setting (Conservative, Balanced, Aggressive)
controls how eagerly it picks up quiet or unclear speech — Aggressive finds more and
invents more. If you are unsure, leave both alone.

**A realistic expectation.** This is adult video audio: a lot of it is not speech, and
long stretches with no subtitles are normal, not a fault. A two-hour film on a mid-range
graphics card takes roughly 40 minutes.

---

## Faster

**Aggressive no longer runs away with your time.** On a three-hour film, an Aggressive run
used to take more than twice as long as Balanced, and one had to be abandoned before it
finished.

The main cause was that Aggressive was allowed to decode a passage a second time whenever a
quality check tripped — and on continuous, repetitive audio that check trips constantly. It
now decodes once. On a two-hour film that had previously been abandoned, the same run
finished in **41 minutes at 8.8 times real time**, against **2.4 times** before, and
produced 1,961 subtitles covering 99.8% of the film. None of its 89 scenes came back empty;
in the abandoned run, 20 of 26 had.

What you give up: Aggressive explores slightly fewer alternative readings of a difficult
passage. What you gain: it finishes, and it gives the same answer twice.

All the Whisper-based modes now decode once rather than retrying, so **run times are
predictable** and **the same file gives the same subtitles every time**.

---

## Fixed

**Empty subtitle files on some graphics cards.** WhisperJAV was telling the recogniser that
the first subtitle in each chunk had to start at exactly zero seconds. On some hardware
that left the decoder with no legal choice, and it produced a long run of `!` characters,
which a quality check then threw away — leaving you with no subtitles and a run that
reported success. On our test films this produced **zero subtitles across every clip**.
Fixed by letting the first subtitle start up to a second in, which is what Whisper itself
does. On hardware where the fault never triggered, your subtitles are unchanged.
(Related reports: **#414**, **#287**, and probably **#411** and **#326**.)

**RTX 50 cards were being given the wrong precision setting.** These cards were forced to
`float16` to avoid a crash in an underlying library. That crash was fixed upstream some
time ago, and the forced setting had itself become the problem — one user measured it
producing garbled output and almost no subtitles on an RTX 5070, where the automatic
setting worked cleanly. RTX 50 cards now let the library choose. **Other NVIDIA cards are
deliberately unchanged**: we measured the alternative and it was no more accurate while
being 14–32% slower, which is not a trade worth making for you. (**#414**)

**A card this build cannot use now stops the run at the start.** One user with a GTX 1060
watched a twenty-minute run produce nothing and report success. WhisperJAV now stops before
doing anything, names your card, and asks whether to continue on the CPU or stop. It waits
for your answer. (**#411**, **#333**, probably **#326**)

**Installs that claimed to work when they had not.** If a package essential to transcribing
failed to install, the installer used to finish anyway and create a desktop shortcut. It now
tells you the installation failed and creates no shortcut.

**Subtitles that were nothing but punctuation.** On the anime-whisper and Qwen3 modes, a
stretch of sound with no words could produce a subtitle containing only `。` — in one user's
file, a quarter of the second pass. Those lines are now dropped. Punctuation inside real
text is untouched. (**#413**)

**Ten-minute delays when Hugging Face was unreachable.** Every model load checked the site
for a newer version first, and with the site blocked each file was retried five times before
the cached copy was used. Models now load from your cache first. There is also a new
`--offline` switch. (**#415**)

**Settings that were ignored without telling you.** Several options were accepted and then
quietly dropped. They now either work or say they cannot.

---

## New: it tells you when your audio is difficult

After WhisperJAV has worked out the scenes in your file — before it starts the long part —
it now prints something like this:

```
  10 scenes, 25:00 total. Speech detected in 6% of the running time.

  This file appears to be unusually quiet for speech detection.
  Subtitles may be missing or sparse throughout.

  2 scenes appear to be acoustically difficult -
  speech close in level to everything else around it:

     scene 2     3:50 - 7:50       scene 6    18:24 - 21:34

  If the subtitles do look sparse, either of these may help:
     --vad-threshold 0.15             pick up quieter speech
     --speech-enhancement ffmpeg-dsp  raise the level first
```

**If your audio looks normal, none of this appears** — you only get the first line.

Why it exists: a 25-minute file quietly lost two-thirds of its dialogue because it was
recorded very quietly, and nothing in the run said so. You would only have found out by
watching the film against the subtitles.

Two honest caveats. It is a hint, not a verdict — the wording says "appears to be" because
that is what it means. And it does not catch everything: on two of our test clips, speech
buried in noise was lost without this warning firing. Both are the sort of thing you might
check by hand if the subtitles look thin.

This currently runs on the Balanced, Fidelity, Qwen3 and anime-whisper modes, when scene
detection is set to `semantic` (the default).

---

## Changes you may notice

- **Scenes are shorter on Balanced and Fidelity** — between 28 seconds and 4 minutes,
  where the ceiling used to be 20 minutes. You will see more progress lines, and a scene
  that produces nothing now costs at most 4 minutes of audio rather than 20.
- **The terminal says more.** Each scene reports its length, how many subtitles it
  produced and how long it took, and a scene that produced nothing is marked `NO OUTPUT`.
- **A warning on long scenes is gone.** `Speech segmentation produced insufficient
  coverage` was a false alarm. Nothing had failed, and on one test film it accounted for
  every warning printed — which meant a real problem would have looked identical to it.
- **On Fidelity, a rescue behaviour was removed.** If the speech detector reported almost
  no speech in a scene, that scene used to be transcribed end to end anyway. It no longer
  is. This saves a lot of pointless work, and it does mean a scene whose detector genuinely
  fails now yields nothing instead of being salvaged.

---

## Upgrading

**Windows installer:** download and run the new `.exe`. It installs alongside or over your
existing copy.

**Installed from source:** `whisperjav-upgrade`.

**Important if you upgrade by hand:** this release deliberately changes two underlying
libraries (faster-whisper and CTranslate2). If you normally use `pip install --no-deps`,
do not this time, or you will keep the old ones and the RTX 50 fix will not reach you.

---

## Known limitations

- **`--offline` covers Hugging Face downloads only.** Some components fetch from their own
  servers. Models must have been downloaded once while online.
- **Kaggle support has moved to a later release.** Colab is unaffected.
- **The Transcription tab always uses Silero 3.1.** The Ensemble tab and the command line
  let you pick 4.0 or 6.2; the Transcription tab does not.
- **The root cause behind #394 is still open.** The recogniser can enter a state where it
  returns nothing for the rest of a run. This release detects the *result* rather than
  preventing it. Work continues, with useful evidence from **@AlanZ-Git** and **@daoran9**.
- **The "difficult audio" notice does not reach every case**, as described above.

---

## Thanks

To everyone who filed an issue with a log attached, and particularly to the people who went
back and measured things when asked: **@yyyanlei**, whose careful testing on an RTX 5070
found the precision problem and ruled out the explanations that were wrong; **@AlanZ-Git**
and **@daoran9** on #394; and **@skysstst**, who diagnosed a translation fault and supplied
the fix.
