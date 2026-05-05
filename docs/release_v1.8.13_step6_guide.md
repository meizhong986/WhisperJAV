# v1.8.13 Step 6 Manual Release Guide

> **You are here:** code committed (4 commits on `dev_v1.8.13`), version
> bumped to 1.8.13 (`installer/VERSION`), installer files regenerated
> (`installer/generated/`), wheel built, .exe built. This guide walks
> you through the final manual steps: merge, tag, push, publish.

---

## Pre-flight checklist

Before doing anything destructive, verify the local state is what you expect:

```powershell
# Working tree clean?
git status                                   # → "nothing to commit, working tree clean"

# Right branch + ahead of main as expected?
git rev-parse --abbrev-ref HEAD              # → dev_v1.8.13
git log --oneline main..HEAD | wc -l         # → ~27 (28 after VERSION commit, see below)

# The 4 fix commits from this session present?
git log --oneline -5                         # → cb9d8c3 docs..., 745f970 tools..., 42e7415 feat(main)..., ad22a6e fix(asr)...

# Wheel + .exe built?
ls installer/generated/whisperjav-1.8.13-py3-none-any.whl
ls installer/generated/WhisperJAV-1.8.13-Windows-x86_64.exe
```

If any of these fail, **stop** and reconcile before proceeding.

---

## Step 6.0 — Commit the version bump (if not already)

The `installer/VERSION` change (1.8.12 → 1.8.13) was made but not yet
committed. Commit it as the final pre-tag commit:

```powershell
git add installer/VERSION
git commit -m @'
chore(release): bump version 1.8.12 → 1.8.13

Triggers build_release.py to regenerate all 11 templated installer
files at installer/generated/, builds whisperjav-1.8.13-py3-none-any.whl
wheel, and produces WhisperJAV-1.8.13-Windows-x86_64.exe via
conda-constructor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Step 6.1 — Merge `dev_v1.8.13` → `main`

Two merge options. Pick whichever your repo convention prefers:

### Option A — Merge commit (preserves dev_v1.8.13 history)

```powershell
git checkout main
git pull --ff-only origin main                # safety: ensure local main is up to date
git merge --no-ff dev_v1.8.13 -m @'
Merge branch 'dev_v1.8.13' for v1.8.13 release

v1.8.13 — Polish + WhisperSeg Default (Ensemble/Qwen/Decoupled) + Model Default Revert

Headline changes:
- Default Whisper model reverted from large-v3 to large-v2
  (BalancedPipeline + Stable-TS backends; OpenAI-Whisper already on v2).
  Empirical regression: F4/F6/F7 acceptance testing showed 6-10/68 GT
  entries with v3 vs 51/68 with v2 on JAV reference clip.
  v1.9.x will re-tune for v3.

- WhisperSeg becomes default for Ensemble Mode, Qwen pipeline, and
  Decoupled pipeline. Simple Transcription Mode keeps silero-v3.1
  pending v1.9.0 unified segmenter param routing fix.

- main.py adds explicit allow-list of "safe paths" for non-Silero
  segmenter defaults; warns + downgrades on explicit non-Silero
  override in unsafe simple-mode paths.

Polish: --ollama-num-ctx flag (#271), French target + English source
languages (#308), --dump-params runtime mirror (#312), Linux uv install
fix (#300, #313), Models & Cache FAQ (#99, #250, #264), efwkjn/whisper-
ja-anime-v0.3 selectable (#232), assorted GUI/qwen/decoupled bug fixes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

### Option B — Fast-forward (only if main has no commits ahead of where dev_v1.8.13 diverged)

```powershell
git checkout main
git pull --ff-only origin main
git merge --ff-only dev_v1.8.13
```

If `--ff-only` errors, main has commits dev_v1.8.13 doesn't. Use Option A.

---

## Step 6.2 — Tag the release

```powershell
git tag -a v1.8.13 -m @'
WhisperJAV v1.8.13

Polish + WhisperSeg Default (Ensemble/Qwen/Decoupled) + Model Default
Revert (large-v3 → large-v2 due to v1.8.12 ASR preset interaction).

See docs/release_notes_v1.8.13.md for full details.
'@

# Verify the tag
git show v1.8.13 --stat | head -20
```

---

## Step 6.3 — Push to GitHub

```powershell
# Push main with tag in one step
git push origin main --follow-tags

# OR if you prefer separately:
git push origin main
git push origin v1.8.13
```

If `git push` fails due to upstream protection rules, push `dev_v1.8.13`
first and let the merge happen via PR:

```powershell
git push origin dev_v1.8.13
# Then open a PR from dev_v1.8.13 → main on github.com
```

---

## Step 6.4 — Publish GitHub Release

1. Go to: https://github.com/meizhong986/whisperjav/releases/new

2. Fill in:
   - **Tag**: `v1.8.13` (select the tag you just pushed)
   - **Target**: `main`
   - **Title**: `WhisperJAV v1.8.13 — Polish + Model Default Revert`
   - **Description**: paste the body of `docs/release_notes_v1.8.13.md`
     (omit the `> Draft — to be finalized before tag.` line)

3. **Attach binaries** (drag-drop into the release):
   - `installer/generated/WhisperJAV-1.8.13-Windows-x86_64.exe`
   - (Optional but useful) `installer/generated/whisperjav-1.8.13-py3-none-any.whl`

4. **CRITICAL — Pre-release checkbox:**
   - **☐ Set as a pre-release** ← **LEAVE UNCHECKED**
   - This is a stable release. Per CLAUDE.md / repo memory:
     - Unchecked → all users get upgrade notification via the GUI's
       UpdateCheckManager (which queries `/releases/latest`)
     - Checked → ONLY testers who manually visit Releases page see it
   - Misclicking this is the #1 release-day mistake.

5. **☐ Set as the latest release** — leave checked (default behavior).

6. **Publish release** (the green button).

---

## Step 6.5 — Post-release smoke checks (recommended, ~10 min)

Once the release is live:

```powershell
# Test the upgrade prompt fires (pretend you're an old user):
# 1. Launch a v1.8.12 install of WhisperJAV
# 2. GUI's "Check for updates" should now offer v1.8.13
# 3. Verify the .exe downloads + installs cleanly
```

Or simply visit the release page and confirm:
- The .exe is downloadable
- File size is ~150MB (constructor builds)
- The release does NOT have a "Pre-release" tag in the UI

---

## Step 6.6 — Post-release replies

Per the resume pointer: 6 prepared replies in
`docs/release_v1.8.13_reply_drafts.md` should be posted to the relevant
GitHub issues after the release publishes. Open each linked issue and
paste the corresponding draft.

---

## Step 6.7 — Tracker update

After everything is live, add a tracker entry to
`docs/ISSUE_TRACKER_v1.8.x.md` (rev49) documenting:
- v1.8.13 shipped
- Headline changes (model revert, WhisperSeg scope)
- Open follow-ups for v1.9.x (re-tune for large-v3, unified segmenter
  param routing)

---

## If something goes wrong

| Symptom | Action |
|---|---|
| `git push` rejected | Pull first: `git pull --rebase origin main` |
| Wrong tag pushed | `git tag -d v1.8.13 && git push origin :refs/tags/v1.8.13`, then re-tag and push |
| Pre-release checkbox accidentally checked | Edit the release on GitHub → uncheck → save (users will get the notification on the next /releases/latest poll) |
| .exe upload fails (size > 2GB) | This shouldn't happen (~150MB) but if it does, host externally and link in the release notes |
| Pre-flight checklist fails | Don't proceed. Re-run `git status`, verify the 4 fix commits, verify the wheel + .exe exist. If anything is wrong, ask Claude to investigate before continuing |

---

## Reference: what was committed in this release session

```
cb9d8c3  docs: v1.8.13 release notes — model revert + scoped WhisperSeg default
745f970  tools(fw-diag): add G_PROD_CL30 + H_PROD_NOCL chunk_length isolation variants
42e7415  feat(main): scope v1.8.13 segmenter default flip to safe paths
ad22a6e  fix(asr): revert default Whisper model large-v3 → large-v2 (regression)
```

Plus the version-bump commit (Step 6.0 above) makes 5 new commits for
v1.8.13.

Test artifacts referenced in commit messages:
- `test_media/1813 acceptance/F4/` (catastrophic baseline, large-v3)
- `test_media/1813 acceptance/F4/DIAG_FW/chunk_length_test/` (G_PROD_CL30 vs H_PROD_NOCL)
- `test_media/1813 acceptance/F8/` (verified fix, large-v2, 51/68 entries)
