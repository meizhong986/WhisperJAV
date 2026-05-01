# Cohere Transcribe-03-2026 — Spike Findings (2026-05-01)

**Spike scope**: half-day investigation to determine if Cohere Transcribe is
viable for the v1.9.x ASR expansion roster (alongside FireRedVAD and
anime-whisper variants), per #262 (teijiIshida).

**Outcome**: **BLOCKED — defer until weights are ungated.**

---

## Findings

### 1. Environment is fine

- Python WJ env: transformers 4.57.6 + torch 2.9.1 + CUDA enabled
- RTX 3060 with 10.98 GB free VRAM (model needs ~4-8 GB FP16 — ample headroom)
- No environment-side blockers

### 2. Cohere ASR class is NOT native to transformers 4.57.6

```python
# This fails:
from transformers import CohereAsrForConditionalGeneration
# ImportError: cannot import name 'CohereAsrForConditionalGeneration'
```

The class ships in the model repo itself, requiring `trust_remote_code=True`
at load time. Acceptable but flag-worthy: Cohere integration would carry the
same `trust_remote_code` attack surface as ZipEnhancer/ModelScope.

### 3. The actual blocker — model weights are GATED

`HfApi.model_info()` reports `gated: auto`. Direct download attempt:

```
huggingface_hub.errors.GatedRepoError: 403 Client Error.
Cannot access gated repo for url
  https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/resolve/main/config.json
Access to model CohereLabs/cohere-transcribe-03-2026 is restricted and
you are not in the authorized list.
Visit https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 to ask for access.
```

This is a hard blocker for the spike (we can't measure JAV performance
without weights) AND a hard blocker for shipping (every WhisperJAV user
would face the same gate, defeating the one-click install promise).

### 4. License vs gating — separate concerns

- **Code license**: Apache-2.0 (commercial use OK)
- **Weight access**: gated by Cohere Labs (acceptance criteria not stated
  publicly)

Apache-2.0 on the code doesn't override the access gate on the weights.

---

## Recommendation: drop Cohere from the v1.9.x ASR expansion roster

**Reasoning**:

1. **User friction is decisive.** WhisperJAV's value prop includes one-click /
   one-command install. Forcing every user through HF account + access request +
   token authentication is incompatible with that. Power users can already pull
   Cohere themselves outside WhisperJAV — we wouldn't add value by wiring it in.

2. **No performance signal.** Without weight access we can't measure JAV
   performance to compare against anime-whisper. Shipping integration code
   without that signal would be premature.

3. **JA isn't Cohere's headline strength anyway.** From the model card, Japanese
   is 1 of 14 languages but English is the leaderboard claim. JAV-specific
   content (anime/breathy/whispered Japanese) is exactly what anime-whisper was
   fine-tuned for. Cohere would need to dramatically outperform anime-whisper to
   justify the gating friction — not the expected outcome.

4. **The other v1.9.x candidates remain unblocked.** FireRedVAD, anime-whisper
   variants, and #232 whisper-ja-anime are all freely accessible. We can ship
   the v1.9.x ASR expansion theme without Cohere and still tell a strong story.

---

## Suggested actions

1. **Reply to #262 (teijiIshida)** with the gating finding + the rationale for
   deferring. Keep the issue open as a "watch — revisit if ungated" thread.

2. **Update memory** (`project_v19x_asr_expansion_theme.md`) to remove Cohere
   from the active v1.9.x roster, replace with a "watch list" entry that
   triggers re-evaluation if the gate is removed.

3. **Do NOT request access.** That's a separate decision that requires user
   authorization (it commits the maintainer's HF identity to Cohere's
   acceptance terms, whatever they are).

4. **Spike artifact retention**: keep `_cohere_spike/` directory for now as a
   reference. Delete before committing if not needed.

---

## What we did NOT learn (for the record)

- Whether Cohere Transcribe is competitive on JAV (couldn't load)
- Whether Cohere's "silence hallucination" weakness combines well/poorly with
  WhisperSeg or TEN VAD pre-segmentation (couldn't test)
- Real disk size (model.safetensors metadata reported 0.00 GB — likely because
  HF size metadata is inaccessible while gated, so the metadata probe couldn't
  read it either)
