# ChronosJAV Pipeline

ChronosJAV is a pipeline for anime and JAV content. It runs two kinds of ASR model: Qwen3-ASR, whose Model menu also has two Japanese fine-tunes (one tuned on game speech), and anime-whisper, made for anime and visual-novel dialogue.

Inspired by the temporal-awareness approach in [ChronusOmni](https://arxiv.org/abs/2512.09841) (Chen et al., 2025).

---

## Available Models

In the **Ensemble Mode** tab, the **Pipeline** menu's **ChronosJAV** group has two pipelines, **Qwen3-ASR** and **Anime-Whisper**. The pass's **Model** menu then offers:

| Pipeline | Model menu entry | GPU memory, as the menu gives it |
|-------|------|-----------|
| **Qwen3-ASR** | Qwen3-ASR-1.7B (the default) | 8 GB |
| | Qwen3-ASR-0.6B | 4 GB |
| | JA Anime-Galgame 1.7B (jaykwok's fine-tune) | 8 GB |
| | JA-tuned 1.7B (neosophie) | 8 GB |
| **Anime-Whisper** | anime-whisper (the default) | ~4 GB |
| | efwkjn/whisper-ja-anime-v0.3 (experimental, #232) | ~3 GB |
| | Kotoba v2.0 | ~2 GB |
| | Kotoba v2.1 | ~2 GB |

The group's third entry, **Cohere-Transcribe (not available yet)**, is greyed out and cannot be chosen; see the [FAQ](../faq.md#cohere-transcribe-not-available-yet).

!!! tip
    If your graphics card has less than 8 GB of memory, choose a model the menu lists with less: Qwen3-ASR-0.6B (4 GB) or an Anime-Whisper model (2 to 4 GB).

---

## How to Use

### GUI

1. Open the **Ensemble Mode** tab (ChronosJAV is not in the Transcription Mode tab)
2. In the **Pipeline** menu of Pass 1 or Pass 2, choose **Qwen3-ASR** or **Anime-Whisper** (group **ChronosJAV**); the pass's sensitivity and speech segmenter change to the pipeline's own
3. Choose a model in the pass's **Model** menu
4. Click **Start**

On the command line: `whisperjav video.mp4 --mode qwen` (Qwen3-ASR), or `--mode qwen --qwen-generator anime-whisper`.

### As Part of Ensemble

ChronosJAV can run as Pass 1, Pass 2 or both, with any other pipeline as the other pass. With **Pass 2** ticked, the two passes' subtitles are combined by the strategy chosen under **Merge** (default **Pass 1 + Fill From Pass 2**; see [Merge Strategies](ensemble.md#merge-strategies)).

---

## Technical Details

ChronosJAV uses different defaults than the standard Whisper pipelines:

| Setting | ChronosJAV Default | Standard Default |
|---------|-------------------|-----------------|
| **Decoding** | Greedy (beam=1) | differs by pipeline (Balanced: beam search with 2 beams) |
| **Speech Segmenter** | FireRedVAD, TEN VAD or WhisperSeg, by pass (see below) | differs by pipeline |
| **Timestamp Mode** | VAD only (since 1.10.1, as in 1.9.5; 1.10.0: Aligner + VAD fallback) | the Whisper model's own times |
| **Text Cleaner** | Qwen3-ASR: Qwen3 AssemblyTextCleaner; anime-whisper: its own text cleaner | — |
| **Clean-up after the subtitles are made** | ChronosJAV's own: it removes lone lines such as 「あ。」, lines made only of moaning or breathing sounds, and overlaps where two scenes meet, among other steps; no hallucination filter | the sanitizer, with the hallucination filter |

The speech segmenter a ChronosJAV pass starts with: in the **Ensemble Mode** tab, FireRedVAD for Qwen3-ASR as Pass 1, WhisperSeg for Anime-Whisper as Pass 1, and TEN VAD for either as Pass 2. On the command line: WhisperSeg for `--mode qwen`; in an Ensemble run the same as in the GUI. The pass's **Speech Segmenter** menu, `--pass1-speech-segmenter` / `--pass2-speech-segmenter` or `--qwen-segmenter` changes it.

---

## Subtitle timing {#subtitle-timing-1100}

ChronosJAV's ASR models write text without times. Since 1.10.1, by default, Qwen3-ASR and anime-whisper subtitles take their start and end from the piece of speech (VAD group) the speech segmenter found, as in 1.9.5: no aligner and no timing model run, and neither is downloaded. Two sentences in one piece of speech — for example two speakers — come out as one subtitle.

1.10.0 timed the text by default with two more steps; since 1.10.1 they run only when chosen:

1. **The aligner** times each character of the text the ASR model wrote. `wav2vec2-ja` (232 MB) is used when no other is named.
2. **The timing model**, Whisper small (488 MB), runs on each piece of speech; its own text is not used. Where Whisper starts a new segment, the piece of speech is split into separate subtitles at the punctuation mark (such as 。？！…、 or a closing bracket) or space in the text nearest that point. Without such a mark nothing is split. No number of characters or seconds decides a cut, and no split leaves on its own a part that WhisperJAV's clean-up would then delete (such as a lone moan or a lone はい); such a part stays with its neighbour. Each subtitle starts where the aligner puts its first character and ends where the next one starts; the last one ends where the piece of speech ends. Where the aligner gave no times, the timing model moves the start later, to where Whisper hears the speech begin.

Both run in CrispASR, which is installed with WhisperJAV, and are downloaded the first time a ChronosJAV run uses them. They run where the install's CrispASR runs (see [Where they run](crispasr.md#where-they-run)), so on a GPU-through-Vulkan or Apple GPU install they ask for the GPU (not yet tried on either) while the ASR model runs on the CPU.

| Choice | GUI (Customize → Alignment) | Command line |
|---|---|---|
| VAD only (default) | **Timestamp Mode**: VAD Only (no aligner loaded) | `--qwen-timestamp-mode vad_only` (default) |
| Aligner | **Timestamp Mode**: Aligner + VAD Fallback, Aligner + Interpolation or Aligner Only; then **Aligner Backend**: CrispASR (GGUF, recommended) or Qwen3 ForcedAligner (PyTorch), and **Aligner Model** | `--qwen-timestamp-mode aligner_vad_fallback` (or `aligner_interpolation`, `aligner_only`), with `--qwen-aligner wav2vec2-ja` (used when none is named), `qwen3-fa-gguf`, `parakeet-ctc-ja`, or `Qwen/Qwen3-ForcedAligner-0.6B`. `--qwen-aligner` given without `--qwen-timestamp-mode` turns on `aligner_vad_fallback` |
| Timing model | **Timing Model (Whisper)**: Whisper small, None (no timing model; default) | `--qwen-timing-model whisper-small`, or `none` (default) |
| Step-down | **Adaptive Step-Down** (off by default; shown with an aligner mode) | `--qwen-stepdown` (off by default) |

In an Ensemble pass from the command line, the same choices go in `--pass1-qwen-params`, for example `'{"timestamp_mode": "aligner_vad_fallback", "timing_model": "whisper-small"}'` for 1.10.0's timing. An `aligner_id`, or an `aligner_backend` of `crispasr` or `qwen3`, given without `timestamp_mode` also turns on `aligner_vad_fallback`.

- `parakeet-ctc-ja` is not recommended: its times came back evenly spread in our tests.
- Without an aligner, a chosen timing model still moves each subtitle's start, but splits nothing.
- With **Subtitle Regrouping** (Output tab; `--qwen-regroup`) set to Standard or Sentence Only instead of Off (the default), the regrouping makes the splits. In an aligner mode the start comes from the aligner, and a chosen timing model only moves the start of a piece of speech the aligner gave no times, though it still runs. With VAD Only, the regrouping divides each piece of speech's text and shares out its times by the length of the text.
- To get ChronosJAV's times as in 1.10.0 (the aligner and the timing model): `--qwen-timestamp-mode aligner_vad_fallback --qwen-timing-model whisper-small`, or in the GUI Timestamp Mode **Aligner + VAD Fallback** and Timing Model **Whisper small**. A GUI preset saved from a ChronosJAV Customize window in 1.10.0 keeps the Timestamp Mode and Timing Model it was saved with; one saved with 1.10.0's defaults still runs the aligner and the timing model.
- **When something does not work, the text is kept.** If the aligner cannot be loaded (for example on the first run without internet) or gives no times for a piece of speech, those subtitles are not split and take the times of their pieces of speech; the timing model may still move their start. If the timing model cannot be loaded, nothing is split and the times come from the aligner. The run summary says which part and why.
- With **Aligner + VAD Fallback** and the other settings at their defaults, if the aligner piles the words of a scene together (the collapse check), that scene's subtitles take the times of their pieces of speech and are not split; a chosen timing model moves their start. Step-down, which instead retries such a scene with pieces of speech up to 6 seconds long, is off by default.
- With `--device cpu`, the aligner and the timing model, when chosen, cannot be held on the CPU: where CrispASR's build uses a GPU, they still ask for it, and the box at the start of the run says so. On a computer without a GPU they run on the CPU; how long the timing model takes there has not been measured.
- 1.10.0's timing (the aligner and the timing model), measured on 7 scenes of The Naked Director against Netflix's Japanese subtitles: subtitles starting within 0.5 s, Qwen3-ASR 78% (1.9.5: 75%), anime-whisper 77% (1.9.5: 63%); the text scores the same as 1.9.5. A typical subtitle still starts about 0.06 to 0.09 s after the reference; 1.9.5's, which timed each subtitle by its piece of speech as the 1.10.1 default does, started slightly before it; the 1.10.1 default has not been measured.

---

## First Run

On first use, the model is downloaded from HuggingFace (~2-4 GB depending on model). This is a one-time download — subsequent runs use the cached model. The aligner (232 MB) and the timing model (488 MB) are downloaded only when a run chooses them (since 1.10.1; 1.10.0 downloaded both on its first ChronosJAV run).

Models are cached in your HuggingFace cache directory:

- **Windows:** `C:\Users\<you>\.cache\huggingface\`
- **macOS/Linux:** `~/.cache/huggingface/`
