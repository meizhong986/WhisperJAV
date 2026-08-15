# AGENTS.md — WhisperJAV Team Charter

> Entry point for AI coding agents (Claude Code and any AGENTS.md-aware tool).
> This is the **operating model**. The detailed behavioral rules still live in
> `CLAUDE.md` (call-chain verification, user-instruction supremacy, issue-handling
> discipline) and are **not superseded** — this file references them, it does not
> replace them. Persistent project state lives in `memory/MEMORY.md`.

---

## 1. The operating model (read this first)

WhisperJAV is maintained by a **single orchestrator** (the main Claude Code session,
working with the owner) that dispatches **ephemeral, task-scoped subagents**. There are
**no standing per-module agents** — this is a deliberate, evidence-based choice (mid-2026
consensus: persistent module-owner fleets cost ~15× tokens, drift out of sync, and
underperform a strong orchestrator + good context for a solo-maintained mature codebase).

**Where expertise lives:** not in agents, but in **files** —
- per-area dossiers (`<dir>/AGENTS.md`, loaded only when that subtree is touched),
- the cross-area interface contracts (`docs/architecture/MODULE_CONTRACTS.md`),
- volatile "what I did / what's next" ledgers (`memory/`).

An agent does not *remember*; it **re-reads its dossier**. Dossiers therefore store
**decisions, rationale, contracts, and gotchas** — the things not recoverable from the
code — and **point to tests/code** for mechanics. Do not duplicate code into a dossier.

**Escalation rule (the coordination mechanism):** a task-scoped agent works **within its
area's scope**. If a task requires **changing a shared interface** documented in
`MODULE_CONTRACTS.md`, the agent **STOPS and escalates to the orchestrator** — it does not
unilaterally change a contract. The orchestrator coordinates the affected areas.

---

## 2. Build / test / run (verified commands)

**Dev environment is the `WJ` Anaconda env** (`C:\Users\MK\anaconda3\envs\WJ`). Activate it
first; run all commands inside it. The repo-local `.venv` is an unused leftover — ignore it.

```bash
conda activate WJ

# Editable (re)install (used by owner after the folder move):
uv pip install -e . --no-deps

# Tests
python -m pytest tests/                      # all
python -m pytest tests/test_config_v4.py     # one file

# Lint / format
python -m ruff check whisperjav/
python -m ruff format whisperjav/

# Inspect resolved config WITHOUT running ASR (the safe config probe)
python -m whisperjav.main --dump-params /dev/null --mode balanced --sensitivity aggressive

# CLI-flag completeness gate (MANDATORY for any new main.py flag)
python -m whisperjav.main --help | grep <flag>
```

---

## 3. Architecture map → area dossiers

| Area | Lives in | Dossier | Status |
|------|----------|---------|--------|
| **ASR / engines** | `whisperjav/modules/*asr*.py`, `segment_filters.py` | `whisperjav/modules/AGENTS.md` | ✅ inducted |
| **Pipeline / orchestration** | `whisperjav/pipelines/`, `whisperjav/ensemble/pass_worker.py`, `main.py` | `whisperjav/pipelines/AGENTS.md` | ✅ inducted |
| **Config** | `whisperjav/config/` (Pydantic `components/`, `v4/`, `legacy.py`) | `whisperjav/config/AGENTS.md` | ✅ inducted |
| Audio front-end | `modules/scene_detection.py`, `modules/speech_segmentation/`, `audio_preprocessing.py`, `modules/speech_enhancement/` | _pending_ | ⬜ |
| Postprocessing / subtitle | `modules/subtitle_sanitizer.py`, `srt_postprocessing.py` | _pending_ | ⬜ |
| Translation | `whisperjav/translate/` | _pending_ | ⬜ |
| GUI | `whisperjav/webview_gui/` | _pending_ | ⬜ |
| Install / release | `whisperjav/installer/`, `installer/`, `build_release.py` | _pending_ | ⬜ |

The **interface contracts** between these areas: `docs/architecture/MODULE_CONTRACTS.md`.

---

## 4. Red-lines (summary — full detail in `CLAUDE.md` + `memory/MEMORY.md`)

- **Primary sources only.** Read the actual file/log/comment. Never decide from a summary or memory. (`memory/feedback_primary_sources_only.md`)
- **NEVER do a heavy import to "verify".** Importing `pass_worker`/ASR/torch/transformers/faster-whisper **hangs for hours**. Use `py_compile` + Grep + `--help` + `--dump-params`. (`memory/feedback_no_heavy_import_for_verification.md`)
- **Separate observed facts from inferences.** Label hypotheses as hypotheses.
- **Call-chain verification is mandatory** before declaring any GUI→CLI feature done (`CLAUDE.md`).
- **No parallel heavy local processes** (multiple concurrent GPU/ASR runs freeze the machine). Parallel *reasoning* subagents are fine.
- **Do not push or commit without explicit owner authorization.**
- Active branch is **`dev_v1.9.0`** (CrispASR WIP); `whisperjav/config/asr_config.json` is intentionally modified — do not commit it. See `memory/project_v190_crispasr_test_loop_resume_pointer.md`.

---

## 5. Environment notes (post-folder-move, 2026-06)

The project moved `C:\bin\git\…` → `D:\Git\WhisperJav_V1_Minami_Edition`. The **`WJ` conda env
is the dev env and is healthy** (reports current version 1.8.14). Other conda envs exist
(`WJ2`, `WJ_TEST`, `HF`, `QWEN`) for testing/scratch. The repo-local **`.venv` is a stale
leftover** whose editable finder still points at the old path — it is NOT used; do not judge
the environment by it. The old `C:\bin\git\…` folder also still exists on disk.

---

## 6. Subagent disciplines

Every subagent definition under `.claude/agents/` must include the shared discipline
snippet `.claude/agents/_disciplines.md`. Existing task agents:
`config-tracer`, `call-chain-verifier`, plus the slash-command services
`/whisperjav_forensic_analyst` and `/whisperjav_issue_tracker_maintainer`.
