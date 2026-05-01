# v1.8.13 Post-Release Reply Drafts

> Prepared 2026-05-01 during dev_v1.8.13 work. Post these once v1.8.13 is
> tagged + released on GitHub. Each user has been waiting since their report
> on the upstream issue.

---

## Reply queue

| # | Issue | User | Topic | Status |
|---|---|---|---|---|
| 1 | #271 | TinyRick1489 | `--ollama-num-ctx` flag | Awaiting release |
| 2 | #312 | TinyRick1489 | `dump_params` runtime fix | Awaiting release |
| 3 | #300 | ktrankc | Linux install regression | Awaiting release |
| 4 | #313 | parheliamm | Linux install regression | Awaiting release |
| 5 | #264 | starkwsam | Cache location FAQ | Awaiting release |
| 6 | #308 | SangenBR | English source language (Part 1) | Awaiting release |

---

## Draft 1 — #271 (TinyRick1489)

> Heads up — `--ollama-num-ctx` ships in v1.8.13. You can now pass
> `--ollama-num-ctx 16384` (or whatever fits your model) and it'll override
> the curated default *before* the batch-size cap kicks in, so the batch
> should no longer auto-reduce to 11 on your HF-sourced models. Available on
> both `whisperjav` and `whisperjav-translate` entry points.
>
> Standalone installer: download the new .exe from the [Releases page](https://github.com/meizhong986/WhisperJAV/releases).
> Source install: `pip install --no-deps "whisperjav @ git+https://github.com/meizhong986/WhisperJAV.git@v1.8.13"`.
> Let me know if the override behaves as expected on your end.

---

## Draft 2 — #312 (TinyRick1489)

> The `dump_params` fix shipped in v1.8.13. When you select a non-silero
> segmenter (`ten`, `whisperseg`, `silero-v6.2`), the dump now mirrors the
> runtime VAD firewall: `params.vad` is removed from the dump output, with
> a `_dump_note` explaining the substitution and `_dump_cleared_vad`
> preserving the silero values that would have been emitted — nothing is
> lost, just no longer misleading.
>
> Try the same `--mode fidelity --sensitivity conservative --speech-segmenter ten --dump-params`
> command from your original report — the silero presets should be gone.
> Standalone installer: new .exe from Releases. Source install:
> `pip install --no-deps "whisperjav @ git+https://github.com/meizhong986/WhisperJAV.git@v1.8.13"`.

---

## Draft 3 — #300 (ktrankc)

> v1.8.13 fixes the install deadlock you hit on v1.8.11. `install.py` now
> auto-passes `--index-strategy unsafe-best-match` to `uv` whenever the
> PyTorch index is added, so you no longer need the `UV_INDEX_STRATEGY`
> env-var workaround for fresh installs.
>
> If you're upgrading from v1.8.10.post3 (your current working state), pull
> the latest source and re-run `python install.py --cuda cu128` (or whichever
> GPU flag matches your hardware) and it should resolve cleanly. Standalone
> installer users: download the new .exe from the [Releases page](https://github.com/meizhong986/WhisperJAV/releases).
> Let me know if anything still breaks.

---

## Draft 4 — #313 (parheliamm)

> v1.8.13 ships the `install.py` fix. `--index-strategy unsafe-best-match`
> is now auto-passed whenever the PyTorch index is added, so the
> `bs-roformer-infer` vs `requests` resolver deadlock should be gone for
> fresh installs.
>
> Pull the latest source and re-run `python install.py` (with whichever
> `--cuda` flag matches your hardware). The env-var workaround
> `UV_INDEX_STRATEGY=unsafe-best-match` is no longer needed but doesn't
> hurt if you've already added it to your shell profile. Let me know if the
> install completes for you on v1.8.13.

---

## Draft 5 — #264 (starkwsam)

> Apologies for the earlier reply — your push-back was warranted, and the
> v1.8.13 FAQ now has a verified Models & Cache section that I should have
> written in the first place: https://github.com/meizhong986/WhisperJAV/blob/main/docs/en/faq.md
>
> Quick summary for your I-drive relocation:
> - **HuggingFace cache** (Kotoba, anime-whisper, Qwen3-ASR, WhisperSeg): set `HF_HOME=I:\hf_cache`
> - **OpenAI Whisper cache** (large-v2/v3/turbo): set `XDG_CACHE_HOME=I:\cache` — Whisper cache lands at `I:\cache\whisper\`
> - **PyTorch Hub** (Silero VAD weights): set `TORCH_HOME=I:\torch_cache`
> - **ModelScope** (ZipEnhancer/ClearVoice): set `MODELSCOPE_CACHE=I:\modelscope`
>
> Set these as Windows User environment variables, then restart WhisperJAV.
> If you have existing caches on C:, copy them to the new paths first so
> models don't re-download. Let me know if anything in the FAQ doesn't
> match what you observe.

---

## Draft 6 — #308 (SangenBR)

> Part 1 of your ask shipped in v1.8.13: "English" is now selectable as a
> Source Language in the AI SRT Translate dropdown (and on the CLI via
> `--source english`). You can use this to translate English SRTs to French,
> Spanish, Chinese, etc.
>
> Part 2 (ASS/SSA format support for the translate pipeline) is a broader
> subtitle I/O change and is deferred to the v1.9.0 Translation expansion
> theme, which will also bundle additional providers (DeepL, free Google
> Translate, Cohere) and the new merge utility. Standalone installer: new
> .exe from the [Releases page](https://github.com/meizhong986/WhisperJAV/releases).
> Source install: `pip install --no-deps "whisperjav @ git+https://github.com/meizhong986/WhisperJAV.git@v1.8.13"`.
> Thanks again for both suggestions — they shaped the roadmap.

---

## Posting checklist (when v1.8.13 ships)

After tag + GitHub release publish:

1. Verify each issue is still open
2. For each draft above, run:
   ```bash
   gh issue comment <NUM> --repo meizhong986/WhisperJAV --body "$(cat <<'EOF'
   <draft body>
   EOF
   )"
   ```
3. Update tracker to flip statuses to AWAITING REPLY (currently each row
   already says "Reply pending — will batch with v1.8.13 ship notifications")
4. New tracker rev for the batch
