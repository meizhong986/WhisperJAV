# WhisperJAV v1.8.13 — Polish + WhisperSeg Default

> Draft — to be finalized before tag.

## Theme

A focused polish release that addresses long-standing paper cuts, fixes
the install regression carried over from v1.8.11, and **promotes WhisperSeg
from opt-in to default** speech segmenter across all surfaces.

---

## Headline change: WhisperSeg is now the default speech segmenter

Introduced in v1.8.12 as opt-in, **WhisperSeg ONNX is now the default
speech segmenter** in v1.8.13. On the Netflix-GT 283s JAV reference
benchmark it scored F1=0.787 — the strongest result among the four
shipped backends:

| Segmenter | F1 (Netflix-GT JAV) |
|---|---:|
| **WhisperSeg (new default)** | **0.787** |
| TEN VAD | 0.698 |
| Silero v6.2 | 0.654 |
| Silero v3.1 (previous default) | 0.625 |

### What changes for you

- First transcription after upgrade downloads ~119 MB WhisperSeg model
  from HuggingFace (one time). Cached at `~/.cache/huggingface/` (Linux/macOS)
  or `%USERPROFILE%\.cache\huggingface\` (Windows).
- ChronosJAV pipeline's segmenter default also flipped (was silero-v6.2).
- Colab + Kaggle expert notebook defaults flipped to match.

### When NOT to use WhisperSeg

- **Non-Japanese audio** (Korean, Chinese, English): WhisperSeg is trained
  on Japanese ASMR. For other languages, override:
  - **CLI**: `--speech-segmenter silero-v3.1`
  - **GUI Ensemble tab**: pick "Silero v3.1" from the speech segmenter
    dropdown
- **CPU-bound users**: WhisperSeg is ~2-3× slower than silero/ten on CPU.
  If you have no GPU (or onnxruntime-gpu isn't installed), silero-v3.1
  may produce faster total runtime.
- **v1.8.12 retest cluster** (#294, #302, #287, #297): if you're validating
  the v1.8.12 ASR fixes, keep `--speech-segmenter silero-v3.1` for clean
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

- **Whisperseg field testing is thin**. v1.8.12's WhisperSeg release got
  zero user reports back. v1.8.13's promotion to default rests on the
  Netflix-GT bench plus internal smoke tests. If you observe regressions
  vs v1.8.12, override to silero-v3.1 and report on the issue tracker.

- **Cold-start cost**: first transcription after upgrade pauses for the
  119 MB WhisperSeg model download. Subsequent runs use the cached model.

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

- WhisperSeg promotion touched 11 locations across resolver, ASR module
  fallbacks, qwen pipeline, CLI argparse help, GUI HTML, decoupled YAML,
  and Colab + Kaggle notebooks. Architectural note added inline in
  `legacy.py` explaining why `LEGACY_PIPELINES["balanced"|"fidelity"]["vad"]`
  field stays at "silero-v3.1" (it names a Pydantic VAD *component* that
  defines preset values, not the runtime segmenter backend selector — the
  firewall in `whisper_pro_asr.py:71-77` clears those silero presets at
  runtime when whisperseg is selected).

- 6 prepared post-release replies in
  `docs/release_v1.8.13_reply_drafts.md` will be posted after the GitHub
  Release publishes.

---

## Next: v1.9.0

Marquee features for v1.9.0 (per `docs/plans/PRODUCT_VISION_AND_ROADMAP_v1.9_v2.md`):

- **GUI redesign**: 5→4 tabs, eliminate Advanced, add Utilities tab
- **Standalone Merge utility** (GUI + `whisperjav-merge` CLI) — preview at
  Section 8.5 of the roadmap doc
- **Chinese GUI partial i18n** (#175, #180)
- **ASR backend expansion theme**: FireRedVAD (#311), Cohere Transcribe
  (#262), anime-whisper variants (#232), VibeVoice (#205) — all
  benchmarked side-by-side on the Netflix-GT JAV reference before shipping
- **Settings persistence** (#96)
- **English-source ASS/SSA support** (#308 part 2)
