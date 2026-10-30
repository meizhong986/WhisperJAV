# ChronosJAV Pipeline

ChronosJAV is a dedicated pipeline for anime and JAV content, built around speech models specifically trained on Japanese anime and adult content dialogue.

Inspired by the temporal-awareness approach in [ChronusOmni](https://arxiv.org/abs/2512.09841) (Chen et al., 2025).

---

## Available Models

| Model | Size | Strengths |
|-------|------|-----------|
| **anime-whisper** | ~4 GB | Best quality for anime/JAV dialogue. Fine-tuned Whisper large-v3. |
| **Kotoba v2.1** | ~2 GB | Lighter weight with punctuation support. Good balance of speed and quality. |
| **Kotoba v2.0** | ~2 GB | Lighter weight, no punctuation. Fastest of the three. |
| **Cohere-Transcribe** *(deferred to v1.9.0)* | ~4-8 GB | Cohere Labs 2B Conformer ASR. **Currently disabled in the GUI dropdown** — the model requires transformers ≥ 5.4.0 which conflicts with the bundled Qwen3-ASR fork. v1.9.0 will land a coordinated transformers upgrade and ship Cohere alongside the existing generators. See the [FAQ](../faq.md#cohere-transcribe-preview-deferred-to-v190) for the current status. |

!!! tip
    Start with **anime-whisper** for best results. Switch to Kotoba if you need faster processing or have limited GPU memory. Cohere-Transcribe is targeted for v1.9.0; the dropdown entry is visible but greyed out in v1.8.14 as a forward-reference signpost.

---

## How to Use

### GUI

1. Go to the **Ensemble** tab
2. Set **Pipeline** to **ChronosJAV**
3. Select a **Model** from the dropdown
4. Click **Start**

### As Part of Ensemble

For maximum quality, combine ChronosJAV with another pipeline:

1. **Pass 1:** ChronosJAV with anime-whisper
2. **Pass 2:** Qwen3-ASR or Balanced
3. **Merge Strategy:** Smart Merge

---

## Technical Details

ChronosJAV uses different defaults than the standard Whisper pipelines:

| Setting | ChronosJAV Default | Standard Default |
|---------|-------------------|-----------------|
| **Decoding** | Greedy (beam=1) | Beam search (beam=5) |
| **Speech Segmenter** | TEN VAD | Silero v6.2 |
| **Timestamp Mode** | Aligner + VAD fallback (1.10.0; was VAD-only) | Full alignment |
| **Cleaner** | Passthrough | Standard sanitizer |

These defaults are optimized for anime/JAV content. The greedy decoding with TEN VAD segmentation produces tighter subtitle timing and eliminates oversized subtitle blocks.

---

## Subtitle timing (1.10.0)

ChronosJAV's ASR models write text without times. Until 1.10.0, Qwen3-ASR and anime-whisper subtitles took their start and end from the piece of speech (VAD group) the speech segmenter found, and two sentences in one piece of speech — for example two speakers — came out as one subtitle. From 1.10.0, by default:

1. **The aligner** times each character of the text the ASR model wrote. The default is `wav2vec2-ja` (232 MB).
2. **The timing model**, Whisper small (488 MB), runs on each piece of speech; its own text is not used. Where Whisper starts a new segment, the piece of speech is split into separate subtitles at the punctuation mark (such as 。？！…、 or a closing bracket) or space in the text nearest that point. Without such a mark nothing is split. No number of characters or seconds decides a cut, and no split leaves on its own a part that WhisperJAV's clean-up would then delete (such as a lone moan or a lone はい); such a part stays with its neighbour. Each subtitle starts where the aligner puts its first character and ends where the next one starts; the last one ends where the piece of speech ends. Where the aligner gave no times, the timing model moves the start later, to where Whisper hears the speech begin.

Both run in CrispASR, which is installed with WhisperJAV, and are downloaded the first time a ChronosJAV run needs them. They run where the install's CrispASR runs (see [Where they run](crispasr.md#where-they-run)), so on a GPU-through-Vulkan or Apple GPU install they ask for the GPU (not yet tried on either) while the ASR model runs on the CPU.

| Choice | GUI (Customize → Alignment) | Command line |
|---|---|---|
| Aligner | **Aligner Backend**: CrispASR (GGUF, recommended), Qwen3 ForcedAligner (PyTorch), None; **Aligner Model** | `--qwen-aligner wav2vec2-ja` (default), `qwen3-fa-gguf`, `parakeet-ctc-ja`, or `Qwen/Qwen3-ForcedAligner-0.6B` |
| No aligner | Aligner Backend **None** (the run then uses VAD Only with Step-Down off) | `--qwen-timestamp-mode vad_only` |
| Timing model | **Timing Model (Whisper)**: Whisper small (recommended), None (no timing model) | `--qwen-timing-model whisper-small` (default) or `none` |
| Step-down | **Adaptive Step-Down** (off by default) | `--qwen-stepdown` (off by default) |

In an Ensemble pass from the command line, the same choices go in `--pass1-qwen-params`, for example `'{"aligner_backend": "crispasr", "aligner_id": "qwen3-fa-gguf", "timing_model": "none"}'`.

- `parakeet-ctc-ja` is not recommended: its times came back evenly spread in our tests.
- Without an aligner, the timing model still moves each subtitle's start, but splits nothing.
- With **Subtitle Regrouping** (Output tab; `--qwen-regroup`) set to Standard or Sentence Only instead of Off (the default), the regrouping makes the splits and the start comes from the aligner; the timing model then only moves the start of a piece of speech the aligner gave no times, though it still runs.
- To get ChronosJAV's times as in 1.9.5 (from the pieces of speech): `--qwen-timestamp-mode vad_only --qwen-timing-model none`, or Aligner Backend None and Timing Model None in the GUI.
- **When something does not work, the text is kept.** If the aligner cannot be loaded (for example on the first run without internet) or gives no times for a piece of speech, those subtitles are not split and take the times of their pieces of speech; the timing model may still move their start. If the timing model cannot be loaded, nothing is split and the times come from the aligner. The run summary says which part and why.
- If the aligner piles the words of a scene together (the collapse check), with the default settings that scene's subtitles take the times of their pieces of speech, with the start moved by the timing model and no split. Step-down, which instead retries such a scene with pieces of speech up to 6 seconds long, is off by default.
- With `--device cpu`, the aligner and the timing model cannot be held on the CPU: where CrispASR's build uses a GPU, they still ask for it, and the box at the start of the run says so. On a computer without a GPU they run on the CPU; how long the timing model takes there has not been measured.
- Measured on 7 scenes of The Naked Director against Netflix's Japanese subtitles: subtitles starting within 0.5 s, Qwen3-ASR 78% (1.9.5: 75%), anime-whisper 77% (1.9.5: 63%); the text scores the same as 1.9.5. A typical subtitle still starts about 0.06 to 0.09 s after the reference; 1.9.5's started slightly before it.

---

## First Run

On first use, the model is downloaded from HuggingFace (~2-4 GB depending on model). This is a one-time download — subsequent runs use the cached model. The first ChronosJAV run of 1.10.0 also downloads the aligner (232 MB) and the timing model (488 MB).

Models are cached in your HuggingFace cache directory:

- **Windows:** `C:\Users\<you>\.cache\huggingface\`
- **macOS/Linux:** `~/.cache/huggingface/`
