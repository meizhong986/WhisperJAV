# v1.8.12 Release — Manual Merge / Tag / Push Guide

Step-by-step instructions for shipping `v1.8.12`. Run from the repo root
(`C:\BIN\git\WhisperJav_V1_Minami_Edition`).

**Pre-flight assumptions:**
- You are on branch `dev_v1.8.12` with a clean working tree.
- Version files already bumped (`whisperjav/__version__.py = "1.8.12"`,
  `installer/VERSION patch = 12`) — committed in `1d6e241`.
- 18 commits ahead of `origin/main`.
- Wheel + installer artifacts already generated under `installer/generated/`
  (gitignored, do not commit).
- Release notes written: `docs/release_notes_v1.8.12.md` — pending commit.

---

## Step 0 — Sanity check the state

```bash
git status                                  # clean tree expected
git log --oneline origin/main..dev_v1.8.12  # should list ~18 commits
python -c "from whisperjav.__version__ import __version__; print(__version__)"
# expected: 1.8.12
```

If anything looks off, stop and re-evaluate. Do not proceed until clean.

---

## Step 1 — Commit the release notes (if not already)

```bash
git add docs/release_notes_v1.8.12.md docs/release_v1.8.12_merge_guide.md
git commit -m "docs: v1.8.12 release notes + merge guide"
git log --oneline -3
```

You should see the new docs commit at HEAD.

---

## Step 2 — Build the .exe installer (Windows-only step)

This is done in a fresh terminal so the conda environment is activated
correctly. Skip this step if you only want to publish a source-only release.

```bash
cd installer\generated
build_installer_v1.8.12.bat
```

Build takes ~5–15 minutes. Output: `WhisperJAV-1.8.12-Windows-x86_64.exe`
(~150 MB) in `installer/generated/`. Logs go to
`build_log_v1.8.12.txt` in the same folder.

Verify the .exe exists:
```bash
dir WhisperJAV-1.8.12-Windows-x86_64.exe
```

If conda-constructor isn't installed locally, install it via:
```bash
conda install -c conda-forge constructor
```

---

## Step 3 — Push the dev branch to origin

```bash
git push origin dev_v1.8.12
```

This makes the dev branch (and all 18+ commits) visible on GitHub. It
does NOT yet release anything to users.

---

## Step 4 — Merge `dev_v1.8.12` into `main`

You have two options. Pick one.

### Option A — Fast-forward merge (preserves linear history)

If `main` hasn't moved since `dev_v1.8.12` was branched off, this gives
clean linear history:

```bash
git checkout main
git pull origin main         # make sure local main is current
git merge --ff-only dev_v1.8.12
```

If this errors with "Not possible to fast-forward, aborting", switch to
Option B.

### Option B — Merge commit (always safe)

```bash
git checkout main
git pull origin main
git merge --no-ff dev_v1.8.12 -m "Merge dev_v1.8.12: WhisperSeg VAD + tight defaults retune"
```

Resolve any conflicts that arise (unlikely on this release line — dev has
been fast-forwarded throughout).

### Verify the merge

```bash
git log --oneline -5
```

You should see your merge commit (or fast-forwarded HEAD) at the tip of
`main` with all the v1.8.12 commits in history.

---

## Step 5 — Tag the release

```bash
git tag -a v1.8.12 -m "v1.8.12 — WhisperSeg VAD + tight defaults retune"
git tag -l v1.8.12          # confirm tag exists
git show v1.8.12             # confirm tag points to the right commit
```

---

## Step 6 — Push main + tag

```bash
git push origin main
git push origin v1.8.12
```

Or do both at once:
```bash
git push origin main --tags
```

---

## Step 7 — Create the GitHub release

### Option A — via the GitHub web UI

1. Open https://github.com/meizhong986/whisperjav/releases/new
2. Tag: choose `v1.8.12` (just-pushed)
3. Target: `main`
4. Release title: **WhisperJAV v1.8.12 : WhisperSeg VAD + tight defaults retune**
5. Body: paste the contents of `docs/release_notes_v1.8.12.md` starting
   from the second line ("**New speech segmenter…**") — do NOT include
   the H1 title since the GitHub UI separates title and body.
6. Attach binaries (drag & drop into "Attach binaries"):
   - `installer/generated/WhisperJAV-1.8.12-Windows-x86_64.exe`
   - `installer/generated/whisperjav-1.8.12-py3-none-any.whl`
7. **CRITICAL**: Leave **"Set as a pre-release"** UNCHECKED — this is a
   stable release. The GUI's update-check API queries
   `/releases/latest` which only returns non-pre-release entries; if you
   tick this box, no users will see the upgrade prompt.
8. Click **Publish release**.

### Option B — via the `gh` CLI (faster, scriptable)

```bash
# Extract body (skip H1 line) into a temp file
cd C:\BIN\git\WhisperJav_V1_Minami_Edition
tail -n +3 docs/release_notes_v1.8.12.md > /tmp/release_body.md

gh release create v1.8.12 \
  --title "WhisperJAV v1.8.12 : WhisperSeg VAD + tight defaults retune" \
  --notes-file /tmp/release_body.md \
  installer/generated/WhisperJAV-1.8.12-Windows-x86_64.exe \
  installer/generated/whisperjav-1.8.12-py3-none-any.whl
```

Confirm published:
```bash
gh release view v1.8.12
```

---

## Step 8 — Smoke-test the published release

A few minutes after publishing:

1. **Pip install path**: in a clean Python 3.11 env:
   ```bash
   pip install --no-deps "whisperjav @ git+https://github.com/meizhong986/whisperjav.git@v1.8.12"
   python -c "from whisperjav.__version__ import __version_display__; print(__version_display__)"
   # expected: 1.8.12
   ```

2. **GitHub UI check**:
   - Visit https://github.com/meizhong986/whisperjav/releases/latest
   - Confirm it points to `v1.8.12` (not v1.8.11)
   - Confirm both assets are listed and downloadable

3. **GUI update prompt**: in an existing v1.8.11 install, the in-app
   update check should show v1.8.12 as available within the next refresh
   interval.

---

## Step 9 — Tracker / memory updates (optional but recommended)

Update the issue tracker to reflect the release:

```bash
# Edit docs/ISSUE_TRACKER_v1.8.x.md
#  - Move v1.8.12 items from "Awaiting release" to "Shipped in v1.8.12"
#  - Add v1.8.13 candidates section if applicable
```

Update the project memory (`memory/MEMORY.md`):
- Note v1.8.12 release date and tag SHA
- Update "Current state" pointer to v1.8.13 dev (if next dev branch exists)
- Add the ctranslate2 state-contamination finding (from F5/F6 forensic) as
  a v1.8.13 candidate

---

## Rollback procedure (only if release goes wrong)

If you publish v1.8.12 and discover a critical issue within minutes:

### Rollback the GitHub release (un-publish)

```bash
gh release delete v1.8.12 --yes
```

This makes v1.8.11 the latest release again. The tag stays in the repo
(don't delete the tag — re-tagging the same name later confuses people).

### Rollback the merge (only if NOT pushed to origin)

If you've merged but not yet pushed:
```bash
git reset --hard ORIG_HEAD     # or to the last pre-merge commit SHA
```

If you've pushed and need to back out, do NOT force-push to main. Instead,
create a revert commit:
```bash
git checkout main
git revert -m 1 <merge_commit_sha>
git push origin main
```

Then ship a v1.8.12.post1 with the fix.

---

## Quick reference — full command sequence (happy path)

```bash
# Step 1: commit docs
git add docs/release_notes_v1.8.12.md docs/release_v1.8.12_merge_guide.md
git commit -m "docs: v1.8.12 release notes + merge guide"

# Step 2: build .exe (Windows only)
cd installer\generated
build_installer_v1.8.12.bat
cd ..\..

# Step 3: push dev branch
git push origin dev_v1.8.12

# Step 4: merge to main
git checkout main
git pull origin main
git merge --no-ff dev_v1.8.12 -m "Merge dev_v1.8.12: WhisperSeg VAD + tight defaults retune"

# Step 5-6: tag and push
git tag -a v1.8.12 -m "v1.8.12 — WhisperSeg VAD + tight defaults retune"
git push origin main --tags

# Step 7: create GitHub release (gh CLI)
tail -n +3 docs/release_notes_v1.8.12.md > /tmp/release_body.md
gh release create v1.8.12 \
  --title "WhisperJAV v1.8.12 : WhisperSeg VAD + tight defaults retune" \
  --notes-file /tmp/release_body.md \
  installer/generated/WhisperJAV-1.8.12-Windows-x86_64.exe \
  installer/generated/whisperjav-1.8.12-py3-none-any.whl

# Step 8: verify
gh release view v1.8.12
```
