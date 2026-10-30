# Parakeet-CrispASR and Cohere-CrispASR (GGUF)

Parakeet-CrispASR and Cohere-CrispASR are two pipelines that make Japanese subtitles with ASR models in the GGUF file format. They run through CrispASR, which is installed with WhisperJAV in every install; there is no separate program to download.

- **Ensemble Mode only.** They run as Pass 1, Pass 2 or both in the **Ensemble Mode** tab. They have no `--mode` value and are not in the Transcription Mode tab.
- **Japanese only.** Any other source language stops the run with a message.
- **Experimental in 1.10.0 (early release).** Compare their subtitles with those of the pipelines you use now, and tell us how they work for you: https://github.com/meizhong986/WhisperJAV/issues

!!! warning "Folder names"
    The GGUF (CrispASR) pipelines need folder names in plain Western letters (ASCII). When your user folder has Chinese, Japanese or Korean characters, WhisperJAV links or copies the model to a plain-named folder (the original stays where it is).

    This concerns the folder that holds the model file: CrispASR is given no other path (the audio is handed to it in memory). On Windows, WhisperJAV first tries the folder's short (8.3) name; only if that is not available does it put a link or copy in `C:\Users\Public\WhisperJAV\crispasr` (or `C:\ProgramData\WhisperJAV\crispasr`), and the log says so. A copy takes up to 2.4 GB and is kept for later runs. CrispASR's CPU build, installed in a folder with Japanese characters, was checked once: it loaded the model and transcribed from there. WhisperJAV itself, and CrispASR's CUDA and Vulkan builds, have not been checked in such a folder.

!!! note "Coming from 1.9.x?"
    The 1.9.x CrispASR mode, which ran a `crispasr` program you downloaded yourself, is removed. See [Upgrading from 1.9.x](#upgrading-from-19x).

---

## Where they run

Every 1.10.0 install includes CrispASR, in the build that fits the accelerator the install is made for (see [Accelerated or not](../getting-started/index.md#accelerated-or-not)):

| Install made for | CrispASR build | The GGUF pipelines run on |
|---|---|---|
| NVIDIA GPU (CUDA), Windows and Linux x86-64 | CUDA build | the NVIDIA GPU |
| GPU through Vulkan (Windows) | Vulkan build | your graphics card through Vulkan — experimental, not tested |
| Own PyTorch, Windows | Vulkan build | your graphics card through Vulkan — experimental, not tested |
| Apple GPU (Metal), Apple silicon Mac | macOS build | asks for the Apple GPU (Metal) — not yet tested on a Mac |
| Own PyTorch, Apple silicon Mac | macOS build | asks for the Apple GPU (Metal) — not yet tested on a Mac |
| CPU only; Own PyTorch on Linux; Linux on ARM64 | CPU build | the CPU (not accelerated), several times slower |

The processor must support AVX2 (Intel processors since 2013, AMD since 2015). Without it, the GGUF pipelines stop with a message before any audio is read; every other pipeline works.

At the start of each pass, the log says where CrispASR is asked to run, for example:

```
Parakeet-CrispASR asks CrispASR to use the NVIDIA GPU (CUDA) (CrispASR 0.8.44+cuda).
```

With the Vulkan build: "Parakeet-CrispASR asks CrispASR's Vulkan build to use this PC's GPU (CrispASR 0.8.44+vulkan). GPUs through Vulkan are experimental in 1.10.0 and have not been tested on AMD or Intel graphics cards." WhisperJAV cannot yet read which device CrispASR actually used; the line states what it asked for.

---

## The two pipelines

| | Parakeet-CrispASR | Cohere-CrispASR |
|---|---|---|
| **Name in the GUI** | **parakeet** (under **GGUF via crispasr**) | **cohere** (under **GGUF via crispasr**) |
| **Model** | `parakeet-tdt-0.6b-ja` (674 MB) | `cohere-asr-ja-v0.1` (default) or `cohere-asr-ja` (2,424 MB each) |
| **Default speech segmenter** | TEN VAD | FireRedVAD |
| **Default sensitivity** | Aggressive | Balanced |
| **Longest VAD group** | 6 s, at every sensitivity | 3 s |
| **Where a subtitle ends** | only at a natural break in the speech (see below) | only at a sentence end (see below) |
| **Subtitle times** | from the model's own timing of what it wrote | from the aligner `wav2vec2-ja`, which times the model's text |
| **Audio analytics report** | yes | yes |

A **VAD group** is a stretch of speech, a few seconds long, that the speech segmenter finds and hands to the model in one piece.

Parakeet-CrispASR uses the speech segmenter settings of the Fidelity pipeline, with a 6-second limit. Cohere-CrispASR uses those of the Qwen3-ASR pipeline.

---

## How to use

### GUI

1. Open the **Ensemble Mode** tab. It opens with **Pass 1** on **cohere** (Balanced, FireRedVAD) and **Pass 2** on **parakeet** (Aggressive, TEN VAD); Pass 2 is off until you tick it. With **Remember settings** on, the tab opens on the passes you saved instead.
2. To choose another pipeline for a pass, open its **Pipeline** menu: **parakeet** and **cohere** are under **GGUF via crispasr**. Choosing one sets its own sensitivity and speech segmenter, as in the table above.
3. For cohere, choose the model in the **Model** column (`cohere-asr-ja-v0.1` is the default).
4. If you want a second pass, tick **Pass 2**, check its pipeline, and choose how the two are combined under **Merge**.
5. Click **Start**.

When a pass uses one of these pipelines:

- **Sensitivity**, **Speech Segmenter**, **Model**, **Scene Detector**, **Speech Enhancer** and **Enhance for VAD only** can be set, as for the other passes. **None** cannot be chosen as the speech segmenter: these pipelines transcribe only the speech the speech segmenter finds.
- **Customize Parameters** opens three tabs: **Segmenter**, **Enhancer** and **Scene**. They start from the values the pass runs with.
- A status panel under the pass row shows the model and where CrispASR will run, for example:

  ```
  Model: Parakeet TDT 0.6B ja (q8_0) — not downloaded yet (674 MB); downloaded when the run starts
  GPU: asks for the NVIDIA GPU (CUDA)
  ```

  It can also show "CPU (not accelerated), several times slower: <reason>" or "CrispASR cannot run here: <reason>". For Cohere-CrispASR it also lists its aligner, for example "Aligner: wav2vec2 large XLSR-53 ja (q4_k) — not downloaded yet (232 MB); downloaded the first time it is used".

### Command line

One pass:

```bash
whisperjav video.mp4 --ensemble --pass1-pipeline parakeet-crispasr
```

Two passes, with Cohere-CrispASR and its second model as pass 2:

```bash
whisperjav video.mp4 --ensemble --pass1-pipeline parakeet-crispasr --pass2-pipeline cohere-crispasr --pass2-model cohere-asr-ja
```

Without `--merge-strategy`, two passes are combined with `pass1_primary` (in the GUI: "Pass 1 + Fill From Pass 2").

---

## Settings

### What you can set

| Setting | GUI | Command line | Notes |
|---|---|---|---|
| Model | Model | `--pass1-model`, `--pass2-model` | `parakeet-tdt-0.6b-ja`; `cohere-asr-ja-v0.1`, `cohere-asr-ja` |
| Speech segmenter | Speech Segmenter | `--pass1-speech-segmenter` | any except `none` |
| Sensitivity | Sensitivity | `--pass1-sensitivity` | `conservative`, `balanced`, `aggressive`; default `aggressive` for Parakeet-CrispASR, `balanced` for Cohere-CrispASR |
| Scene detector | Scene Detector | `--pass1-scene-detector` | `semantic` (default, scenes of 12 to 48 s), `auditok`, `silero` (scenes of up to 48 s), `none` (one scene per file) |
| Speech enhancer | Speech Enhancer | `--pass1-speech-enhancer` | `none` (default) or the speech enhancers of the other passes, for example `ffmpeg-dsp`, `zipenhancer` or `htdemucs` |
| Enhance for VAD only | Enhance for VAD only | `--pass1-enhance-for-vad` | the cleaned audio goes to the speech segmenter only; CrispASR hears the original |
| Detection threshold | Customize, Segmenter tab | `--pass1-vad-threshold` | 0 to 1; replaces the value the sensitivity sets |
| Padding | Customize, Segmenter tab | `--pass1-speech-pad-ms` | milliseconds added before and after each piece of speech. Cohere-CrispASR: both sides. Parakeet-CrispASR: only with a speech segmenter that has padding, such as WhisperSeg; with FireRedVAD or TEN VAD (its default) it is ignored, with a warning |
| Device | — | `--device auto`, `cuda` or `cpu` | see below |
| Language | Source audio language | `--language japanese` (default) | any other language stops the run |

(The same flags exist for pass 2: `--pass2-…`.)

### Sensitivity

Sensitivity picks the speech segmenter's preset. Aggressive accepts sound the speech segmenter is less sure is speech; Conservative accepts only sound it is surer of. With the default speech segmenters:

| | Conservative | Balanced | Aggressive |
|---|---|---|---|
| Parakeet-CrispASR (TEN VAD): detection threshold | 0.42 | 0.32 | 0.22 |
| Parakeet-CrispASR: padding before / after speech | 0 / 300 ms | 0 / 200 ms | 0 / 100 ms |
| Cohere-CrispASR (FireRedVAD): detection threshold | 0.5 | 0.4 | 0.3 |
| Cohere-CrispASR: padding before and after speech | 100 ms | 100 ms | 100 ms |

The detection threshold is how sure the speech segmenter must be that a sound is speech, from 0 to 1. The longest VAD group stays 6 s (Parakeet-CrispASR) or 3 s (Cohere-CrispASR) at every sensitivity. With another speech segmenter, each pipeline uses the values the Fidelity pipeline (for Parakeet-CrispASR) or the Qwen3-ASR pipeline (for Cohere-CrispASR) gives it; Parakeet-CrispASR keeps its 6-second limit.

To see the values a run will use, without running it:

```bash
whisperjav --ensemble --pass1-pipeline parakeet-crispasr --dump-params settings.json
```

and look under `crispasr_effective` in `settings.json`.

### Device

- `--device auto` (the default, and what the GUI uses): the GPU the install's CrispASR build can use (see [Where they run](#where-they-run)), otherwise the CPU. The log says which, and why.
- `--device cuda`: requires the CUDA build and an NVIDIA GPU it can use; otherwise the run stops before any audio is read.
- `--device cpu`: always the CPU for the ASR model. In the GUI, "Run on the CPU this time" does the same for one run. Cohere-CrispASR's aligner cannot be held on the CPU: where CrispASR's build uses a GPU it still asks for it, and the box at the start of the run says so.

`--device` governs every pipeline of the run, the speech enhancer included.

### Not used by these pipelines

- If you give `--pass1-hf-params` or `--pass1-qwen-params`, one log line names them ("Parakeet-CrispASR does not use: …") and the run goes on.
- `--compute-type` is ignored, with a warning: CrispASR's model files come in one number format (8-bit).
- `--subs-language direct-to-english` still gives Japanese subtitles, with a warning. Translate them afterwards with `--translate` or `whisperjav-translate`.

---

## How the subtitles are made

For each video, both pipelines:

1. extract the audio (16 kHz, mono; 48 kHz when a speech enhancer is chosen, which then gives 16 kHz audio);
2. divide it into scenes with the scene detector (Semantic, 12 to 48 seconds, by default);
3. clean the audio of each scene, if a speech enhancer is chosen; the enhancer is closed before the model opens, so the two never hold GPU memory together;
4. run the speech segmenter on each scene and put the speech it finds into VAD groups;
5. send each VAD group to the model on its own; without a speech enhancer the model is loaded once per pass, with one it is opened for each video's ASR step and closed after it;
6. make subtitles from each VAD group's text, by the pipeline's rule below;
7. join the scenes and apply the same three clean-up filters as the Qwen3-ASR pipeline: they remove lone lines such as 「あ。」 or 「うん。」, lines made only of moaning or breathing sounds, and duplicates and overlaps where two scenes meet;
8. write the pass's subtitle file, which Ensemble Mode combines with the other pass's when there is one.

**Parakeet-CrispASR: where a subtitle ends.** Within a VAD group, a new subtitle starts only at a natural break:

- after a sentence-end mark (。 ？ ！ … and the like);
- at a pause of at least 0.6 seconds;
- at a 、 or a space followed by a pause of at least 0.4 seconds.

It never cuts by number of characters or by length in seconds, so speech with no such break stays one subtitle, as long as its VAD group (up to about 6 seconds). Each subtitle starts where its first word starts and ends where its last spoken word ends, from the times the model gives.

**Cohere-CrispASR: a subtitle ends at a sentence end.** After Cohere writes a VAD group's text, the aligner `wav2vec2-ja` (run by CrispASR) finds when each character of that text is spoken. Within the VAD group, a new subtitle starts only after a sentence-end mark of the text (。 ？ ！ … ? !); a number such as 3.5 is never cut. Each subtitle starts at its first spoken character and ends where the next one starts; the last ends at the end of the VAD group. Pauses do not cut here: the aligner gives the moment each character starts, not where a sound ends, so a held sound would look like a pause. VAD groups are at most 3 seconds, or about 4.2 seconds when one stretch of speech runs on without a break.

If the aligner gives no times for a VAD group, that group stays one subtitle from where the speech segmenter found the speech start to where it found it end, and the run summary counts such groups. If the aligner cannot be loaded at all (for example on the first run without internet), every group is made that way and the run summary says so; the text is kept in both cases. New in 1.10.0; not yet measured against reference subtitles.

**Audio analytics.** As the Qwen3-ASR pipeline does, both pipelines print a short report for each video: which parts may lose speech and which are acoustically hard, with their times, so you can check those places. The report needs the Semantic scene detector; with another one, a log line says so and no report is made.

---

## The model download

- WhisperJAV downloads the model the first time a pipeline needs it, at the start of the run, before any audio is read. It checks first that there is enough free disk space (the model's size plus 10 %).
- The models come from Hugging Face:

| Model | Hugging Face repository | File |
|---|---|---|
| `parakeet-tdt-0.6b-ja` | `cstr/parakeet-tdt-0.6b-ja-GGUF` | `parakeet-tdt-0.6b-ja-q8_0.gguf` |
| `cohere-asr-ja-v0.1` | `TransWithAI/cohere-transcribe-ja-v0.1-GGUF` | `cohere-asr-ja-v0.1-q8_0.gguf` |
| `cohere-asr-ja` | `CKHO/cohere-asr-ja-GGUF` | `cohere-asr-ja-q8_0.gguf` |
| aligner `wav2vec2-ja` (Cohere-CrispASR, and ChronosJAV by default) | `cstr/wav2vec2-large-xlsr-53-japanese-GGUF` | `wav2vec2-large-xlsr-53-japanese-q4_k.gguf` |

Cohere-CrispASR's aligner (232 MB) is downloaded the first time the pass uses it, inside the pass, not before the run. If that download fails, the pass goes on without it: each VAD group becomes one subtitle at the group's edges (see [How the subtitles are made](#how-the-subtitles-are-made)). ChronosJAV's other aligners and its timing model are listed in `whisperjav --check` and on the [ChronosJAV page](chronosjav.md#subtitle-timing-1100).

- The file is kept in your Hugging Face cache folder (by default `C:\Users\<you>\.cache\huggingface\` on Windows, `~/.cache/huggingface/` on macOS and Linux) and reused by later runs.
- Before first use, WhisperJAV checks the file against the version it expects. A damaged download is deleted and downloaded once more.
- If huggingface.co cannot be reached, WhisperJAV tries the mirror hf-mirror.com, as it does for its other models. `--hf-endpoint <address>` names another mirror.
- With `--offline` (GUI: "Offline mode (downloaded Hugging Face models only)") nothing is downloaded. If the model is missing, the run stops before transcribing and says so; run once without `--offline` first.
- If the download or the check fails, the run stops with "CrispASR is not ready for this run" and the reason. Nothing is transcribed.

**Using a model file you downloaded yourself.** Put the file, with its name unchanged, in a plain-named folder and set the environment variable `WHISPERJAV_CRISPASR_MODEL_DIR` to that folder. WhisperJAV uses the file only if it has the expected size, and checks it before first use. A file that fails the check stops the run and is not deleted.

---

## Checking your setup

`whisperjav --check` lists CrispASR and its models. On a PC with the NVIDIA GPU (CUDA) install where everything is ready:

```
✓ CrispASR: 0.8.44+cuda (CUDA build), GGUF pipelines: asks for the NVIDIA GPU (CUDA)
✓ CrispASR model parakeet-tdt-0.6b-ja: Parakeet TDT 0.6B ja (q8_0), for parakeet-crispasr: downloaded
✓ CrispASR model cohere-asr-ja-v0.1: Cohere Transcribe ja v0.1 (q8_0), for cohere-crispasr: downloaded
✓ CrispASR model cohere-asr-ja: Cohere ASR ja (q8_0), for cohere-crispasr: downloaded
✓ CrispASR model wav2vec2-ja: wav2vec2 large XLSR-53 ja (q4_k), for ChronosJAV and Cohere-CrispASR (aligner): downloaded
✓ CrispASR model parakeet-ctc-ja: Parakeet CTC 1.1B ja (q8_0), for ChronosJAV (aligner): downloaded
✓ CrispASR model qwen3-fa-gguf: Qwen3-ForcedAligner 0.6B (q4_k), for ChronosJAV (aligner): downloaded
✓ CrispASR model whisper-small: Whisper small, for ChronosJAV (timing model): downloaded
```

A model that is not downloaded yet shows "not downloaded" with its size. The first run that needs it downloads it.

If CrispASR's Vulkan build cannot load on Windows, the message adds: "With the GPU-through-Vulkan or Own PyTorch install on Windows this usually means the Vulkan runtime (vulkan-1.dll) is missing: update the graphics driver, or run the WhisperJAV installer again and choose CPU only."

---

## When part of a video fails

- If CrispASR fails on some VAD groups of a video, those get no subtitles, and the run summary says how many under that video. If it fails on all of them, that video fails.
- If the speech segmenter fails on a scene, that scene's speech is not transcribed, and the run summary notes it.

---

## Upgrading from 1.9.x

1.9.x had a CrispASR mode that ran a `crispasr` program you downloaded yourself. 1.10.0 removes it:

- `--mode crispasr`, `--pass1-pipeline crispasr`, `--pass2-pipeline crispasr`, `--crispasr-exe`, `--crispasr-backend` and `--crispasr-args` no longer work. The run stops (exit status 2) with:

  ```
  --mode crispasr is no longer available: WhisperJAV 1.10 removed the "bring your own CrispASR" mode.
  CrispASR now runs in Ensemble Mode as two pipelines, Parakeet-CrispASR and Cohere-CrispASR, and WhisperJAV installs it itself.
  Example:
      whisperjav --ensemble --pass1-pipeline parakeet-crispasr <your video>
  In the GUI: open the Ensemble Mode tab and choose parakeet or cohere under GGUF via crispasr in the Pass 1 or Pass 2 menu.
  ```

- In the GUI, the **crispasr** entry of the Mode menu in the Transcription Mode tab is gone, and so is the old **CrispASR** pipeline with its Backend choice in the Ensemble Mode tab.
- The `whispercpp` backend has no replacement among the new pipelines.
- 1.10.0 does not use the `crispasr` program you downloaded, or the models it saved (by default in `~/.cache/whisperjav/crispasr_models`). You can delete them.

What to type instead: `--ensemble --pass1-pipeline parakeet-crispasr` in place of `--mode crispasr --crispasr-backend parakeet`, and `--ensemble --pass1-pipeline cohere-crispasr` in place of `--mode crispasr --crispasr-backend cohere`.

---

## Credits

CrispASR is developed at [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR). Models: NVIDIA's parakeet-tdt_ctc-0.6b-ja (CC-BY-4.0) in GGUF form by cstr; Japanese fine-tunes of Cohere Transcribe by efwkjn, in GGUF form by TransWithAI and CKHO.
