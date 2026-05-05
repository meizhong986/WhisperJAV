# WhisperJAV v1.8.13 — Polish + WhisperSeg Default (Ensemble / Qwen / Decoupled) + Model Default Revert

> Draft — to be finalized before tag.

## Theme

A focused polish release that addresses long-standing paper cuts, fixes
the install regression carried over from v1.8.11, **promotes WhisperSeg
to the default speech segmenter on the pipeline paths that fully support
it** (Ensemble, Qwen, Decoupled), and **reverts the default Whisper model
from `large-v3` back to `large-v2`** for both the FasterWhisper (BalancedPipeline)
and Stable-TS (Fast/Faster pipelines) backends, restoring transcription
quality on continuous-energy non-phonetic JAV content. Simple Transcription
Mode keeps silero-v3.1 in v1.8.13, pending a config-routing refactor
scheduled for v1.9.0. See "Known caveats" for full context on the model
revert.

---

## Headline change: WhisperSeg becomes the default for Ensemble / Qwen / Decoupled

Introduced in v1.8.12 as opt-in, **WhisperSeg ONNX is now the default
speech segmenter** for Ensemble Mode (pass1 and pass2), the Qwen pipeline,
and the Decoupled pipeline. On the Netflix-GT 283s JAV reference
benchmark it scored F1=0.787 — the strongest result among the four
shipped backends:

| Segmenter | F1 (Netflix-GT JAV) |
|---|---:|
| **WhisperSeg (new default for Ensemble/Qwen/Decoupled)** | **0.787** |
| TEN VAD | 0.698 |
| Silero v6.2 | 0.654 |
| Silero v3.1 (default for simple Transcription Mode) | 0.625 |

### Mode-by-mode default policy in v1.8.13

| Path | Default segmenter | Why |
|---|---|---|
| **GUI Ensemble Mode** (pass1 + pass2) | WhisperSeg | Pass-worker routes grouping params correctly |
| **`--ensemble` CLI** | WhisperSeg | Same path as above |
| **`--mode qwen`** (ChronosJAV) | WhisperSeg | QwenPipeline forwards `chunk_threshold_s` / `max_group_duration_s` explicitly |
| **`--pipeline decoupled`** | WhisperSeg | DecoupledPipeline kwargs path |
| **GUI Transcription Mode** | **silero-v3.1** | Routes through legacy ASR constructors with a known param-routing bug; fix is v1.9.0 |
| **`--mode balanced` / `--mode fidelity`** (without `--ensemble`) | **silero-v3.1** | Same — see Known caveats below |

### What changes for you

- **Ensemble Mode users** get WhisperSeg out of the box on first run. First
  transcription downloads ~119 MB WhisperSeg model from HuggingFace
  (one time). Cached at `~/.cache/huggingface/` (Linux/macOS) or
  `%USERPROFILE%\.cache\huggingface\` (Windows).
- **Simple Transcription Mode users** see no behavior change vs v1.8.12 —
  silero-v3.1 stays the default.
- ChronosJAV (`--mode qwen`) segmenter default flipped to WhisperSeg.
- Colab + Kaggle expert notebook defaults flipped to match (Ensemble path).

### How to opt into WhisperSeg in simple mode

Until the v1.9.0 routing fix lands, the recommended path to use WhisperSeg
is **Ensemble Mode** (which also gives you the dual-pass merging that has
become WhisperJAV's strongest result on JAV content). In the GUI, switch
from the "Transcription Mode" tab to the "Ensemble Mode" tab. On the CLI,
add `--ensemble` plus pass1/pass2 settings.

If you explicitly pass `--speech-segmenter whisperseg` (or `ten`, `nemo`,
`whisper-vad`) to a simple `--mode balanced` / `--mode fidelity` run,
v1.8.13 will warn loudly and downgrade to silero-v3.1 to avoid catastrophic
empty output. The warning includes a pointer to `--ensemble`.

### When NOT to use WhisperSeg (Ensemble path)

- **Non-Japanese audio** (Korean, Chinese, English): WhisperSeg is trained
  on Japanese ASMR. For other languages, override:
  - **CLI**: `--pass1-speech-segmenter silero-v3.1 --pass2-speech-segmenter silero-v3.1`
  - **GUI Ensemble tab**: pick "Silero v3.1" from the speech segmenter
    dropdown for each pass
- **CPU-bound users**: WhisperSeg is ~2-3× slower than silero/ten on CPU.
  If you have no GPU (or onnxruntime-gpu isn't installed), silero-v3.1
  may produce faster total runtime.
- **v1.8.12 retest cluster** (#294, #302, #287, #297): if you're validating
  the v1.8.12 ASR fixes, keep silero-v3.1 in both passes for clean
  attribution of which fix helped your case.

---

## Other changes shipped

### Translation

- **`--ollama-num-ctx` CLI flag** (#271 follow-up). Lets you override
  the curated context window size when running HuggingFace-sourced Ollama
  models that need a non-default context. Available in both `whisperjav`
  and `whisperjav-translate` entry points.

- **French target language** (i3) added to AI SRT Translate dropdown +
  CLI `--target` choices.

- **English source language** (#308 part 1) added to AI SRT Translate
  Source dropdown. Useful for translating English SRTs to other languages.

### Diagnostics

- **`dump_params` runtime mirror** (#312). When you select a non-silero
  segmenter (`ten`, `whisperseg`, `silero-v6.2`), `--dump-params` output
  now mirrors the runtime VAD firewall: `params.vad` is removed, with a
  `_dump_note` explaining and `_dump_cleared_vad` preserving the silero
  values that *would* have been emitted (no information loss).

### Install

- **Linux `uv sync` deadlock fixed** (#300, #313). `install.py` now
  auto-passes `--index-strategy unsafe-best-match` whenever the PyTorch
  index is added, resolving the `bs-roformer-infer` requests deadlock that
  blocked fresh installs of v1.8.11/v1.8.12 on Linux.

### Models & cache

- **New FAQ section: Models & Cache** in `docs/en/faq.md` with verified
  paths, env-var relocation recipes (HF_HOME, XDG_CACHE_HOME, TORCH_HOME,
  MODELSCOPE_CACHE), uninstall cleanup guidance, and 4 GB VRAM model
  recommendations. Addresses #99, #250, #264.

### ChronosJAV

- **`efwkjn/whisper-ja-anime-v0.3` selectable** (#232). Community-
  recommended anime-whisper variant added as an experimental opt-in
  alongside `litagin/anime-whisper` default. Visible in both the main
  Ensemble Mode panel and the Customize Parameters modal.

---

## Upgrade

### Standalone installer (Windows)

Download the new .exe from the [Releases page](https://github.com/meizhong986/WhisperJAV/releases).

### Source install

```bash
pip install --no-deps "whisperjav @ git+https://github.com/meizhong986/WhisperJAV.git@v1.8.13"
```

### Linux fresh install (now works without env-var workaround)

```bash
git clone https://github.com/meizhong986/WhisperJAV.git
cd WhisperJAV
python install.py --cuda cu128  # or cpu / cu118 / cu124
```

### Colab / Kaggle

Re-open the notebook from the [latest commit](https://github.com/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_colab_edition_expert.ipynb).

---

## Known caveats

- **Default Whisper model reverted from `large-v3` to `large-v2`.** Late in
  v1.8.13 acceptance testing (F4/F6/F7 vs F8 side-by-side on the same
  293-second JAV reference clip), the v1.8.12 aggressive ASR preset retune
  (`no_speech_threshold=0.84`, `beam_size=3`, `best_of=2`,
  `temperature=[0.0, 0.17]`, `compression_ratio_threshold=2.6`,
  `repetition_penalty=1.3`, `no_repeat_ngram_size=3`, `chunk_length=30`)
  was found to interact pathologically with `large-v3` on continuous-energy
  non-phonetic content (JAV moaning), producing **6–10 SRT entries out of
  68 ground-truth** (≈85–90% loss) under simple Transcription Mode. The
  same audio + same preset values + `large-v2` produced **51 entries**
  (≈75% capture). Root cause: the v1.8.12 retune was tuned against
  large-v2 forensic acceptance data; large-v3's slightly different
  encoder/decoder behavior makes the same gate values too strict on this
  content distribution. Until the preset is re-tuned for large-v3 in
  v1.9.x, v1.8.13 reverts the default to large-v2 across both
  FasterWhisper (`config/components/asr/faster_whisper.py`) and Stable-TS
  (`config/components/asr/stable_ts.py`) backends. Users who want
  large-v3 can opt in with `--model large-v3` (CLI) or via the GUI model
  override checkbox. OpenAI-Whisper backend (FidelityPipeline) was
  already on large-v2 in v1.8.12 and is unaffected.

- **WhisperSeg in simple Transcription Mode is deferred to v1.9.0.**
  During F4/F6 acceptance testing, WhisperSeg invoked through the simple
  `--mode balanced` path produced catastrophic empty output (10 of 68
  ground-truth subtitles) on JAV moaning content, while the same audio +
  same backend through Ensemble Mode produced 52/68. Root cause is a
  config-routing path: BalancedPipeline / FidelityPipeline use a
  CONSTRUCTOR FIREWALL in their ASR modules that strips backend-agnostic
  grouping params (`chunk_threshold_s`, `max_group_duration_s`) for
  non-Silero backends, causing WhisperSeg to fall back to its 29-second
  default group duration and trigger a Whisper repetition pathology.
  v1.8.13 ships the SAFE scope (Ensemble + Qwen + Decoupled keep
  WhisperSeg, simple modes keep silero-v3.1). v1.9.0 lands the proper
  fix (split SileroVADOptions, introduce SegmenterGroupingOptions,
  eliminate the firewall pattern).

- **WhisperSeg field testing is thin**. v1.8.12's WhisperSeg release got
  zero user reports back. v1.8.13's promotion to default for the Ensemble
  path rests on F5 acceptance test (52/68 GT), the Netflix-GT bench, plus
  internal smoke tests. If you observe regressions vs v1.8.12, switch
  back to silero-v3.1 in the Ensemble dropdown and report on the issue
  tracker.

- **Cold-start cost**: first Ensemble transcription after upgrade pauses
  for the 119 MB WhisperSeg model download. Subsequent runs use the
  cached model.

- **Customize Parameters modal**: the Anime-Whisper model dropdown shows
  v0.3 as an option but defaults to litagin/anime-whisper. v0.3 is
  experimental until benchmarked side-by-side in v1.9.x.

---

## Issues addressed

| Issue | Reporter | Resolution |
|---|---|---|
| #99 | (general) | 4 GB VRAM guidance now in FAQ |
| #232 | mustssr | efwkjn/whisper-ja-anime-v0.3 selectable |
| #250 | (general) | Model folder docs in FAQ |
| #264 | starkwsam | Cache relocation recipes in FAQ |
| #271 (follow-up) | TinyRick1489 | `--ollama-num-ctx` CLI flag |
| #292 | leops1984 | Owner ack already posted 04-17 (tracker correction) |
| #300 | ktrankc | Linux uv sync deadlock fixed |
| #308 (part 1) | SangenBR | English source language in dropdown |
| #312 | TinyRick1489 | `dump_params` reflects runtime config |
| #313 | parheliamm | Linux uv sync deadlock fixed (same as #300) |

---

## Internal changes

- WhisperSeg promotion touched 11+ locations across resolver, ASR module
  fallbacks, qwen pipeline, CLI argparse help, GUI HTML, decoupled YAML,
  GUI app.js per-pipeline presets, and Colab + Kaggle notebooks.

- **Default model revert (large-v3 → large-v2)** in three locations:
  `config/components/asr/faster_whisper.py:211`,
  `config/components/asr/stable_ts.py:293`, and
  `webview_gui/assets/index.html:299` (GUI model dropdown default selection).
  Inline comments reference `faster_whisper.py` for full empirical context.
  OpenAI-Whisper backend (`config/components/asr/openai_whisper.py:178`)
  was already on large-v2 and unchanged.

- **v1.8.13 default-flip scope reduction** (post-F4/F6): `main.py`'s
  `--speech-segmenter` default-resolution gained an explicit allow-list of
  paths that route segmenter grouping params correctly to non-Silero
  backends (`--ensemble`, `--pipeline decoupled`, `--mode qwen`). Other
  paths default to silero-v3.1. Explicit `--speech-segmenter whisperseg`
  on a non-allow-listed path emits a hard warning and downgrades to
  silero-v3.1. See inline comment at `main.py:1840-1898`.

- 6 prepared post-release replies in
  `docs/release_v1.8.13_reply_drafts.md` will be posted after the GitHub
  Release publishes.

---

## Next: v1.9.0

Marquee features for v1.9.0 (per `docs/plans/PRODUCT_VISION_AND_ROADMAP_v1.9_v2.md`):

- **Re-tune aggressive sensitivity preset for `large-v3`** (lifts the
  v1.8.13 model revert). The current v1.8.12 aggressive preset values
  (`no_speech_threshold=0.84`, `beam_size=3`, etc.) were tuned against
  large-v2 forensic acceptance data and produce catastrophic empty
  output on JAV content with large-v3. v1.9.x will re-run the forensic
  acceptance suite against large-v3 and produce a per-engine, per-model
  preset variant (e.g., `aggressive_v3` with relaxed gates) so large-v3
  can return as the default. Reference: F4/F6/F7 vs F8 acceptance test
  artifacts in `test_media/1813 acceptance/`.

- **Unified segmenter param routing** (lifts the v1.8.13 simple-mode
  WhisperSeg restriction). Plan: split `SileroVADOptions` into
  `SileroVADOptions` (Silero-specific) + `SegmenterGroupingOptions`
  (backend-agnostic: `chunk_threshold_s`, `max_group_duration_s`,
  `max_speech_duration_s`); resolver places grouping params in a canonical
  location all consumers read from; eliminate the constructor firewall in
  `faster_whisper_pro_asr.py` and `whisper_pro_asr.py`. After this,
  WhisperSeg becomes the default everywhere including simple Transcription
  Mode. Reference: F4/F6 acceptance test artifacts and the diagnostic
  suite runs at `test_media/1813 acceptance/F4/DIAG_FW/`.
- **GUI redesign**: 5→4 tabs, eliminate Advanced, add Utilities tab
- **Standalone Merge utility** (GUI + `whisperjav-merge` CLI) — preview at
  Section 8.5 of the roadmap doc
- **Chinese GUI partial i18n** (#175, #180)
- **ASR backend expansion theme**: FireRedVAD (#311), Cohere Transcribe
  (#262), anime-whisper variants (#232), VibeVoice (#205) — all
  benchmarked side-by-side on the Netflix-GT JAV reference before shipping
- **Settings persistence** (#96)
- **English-source ASS/SSA support** (#308 part 2)
