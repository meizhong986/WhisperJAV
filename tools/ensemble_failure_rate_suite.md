# Ensemble Failure-Rate Test Suite

A diagnostic test suite for investigating the WhisperJAV v1.8.13 ensemble-mode catastrophe where **pass 1 = fidelity + pass 2 = balanced** intermittently produces catastrophically truncated pass 2 SRT output. The same configuration produces correct output on rerun, so the bug is **non-deterministic**.

This suite runs N iterations of each test configuration as independent subprocesses and reports the catastrophic-failure rate. Designed for unattended overnight runs and for distribution to other users running on different machines so we can collect cross-environment data.

---

## Background: What's the Bug?

Multiple users have reported that running WhisperJAV v1.8.13 in ensemble mode with **pass 1 = fidelity** and **pass 2 = balanced** (default GUI selections) sometimes produces a pass 2 SRT file that is catastrophically short — e.g., 14 entries spanning 1:28 of audio when 50 entries spanning the full 4:53 are expected. The same configuration on the same media, rerun, often produces the correct output. We've confirmed:

- The bug is **non-deterministic** (rerun of identical config produces correct output)
- The bug is at the **ASR layer** — whisperseg correctly finds VAD segments, but faster-whisper returns 0 raw segments for the heavier scenes despite running for 33-67 seconds per scene
- The bug specifically affects **fidelity → balanced**; same audio with **balanced → balanced** or **fast → balanced** produces correct output
- The bug is not deterministic state contamination, not a config-routing bug, and not a worker crash

Working hypothesis: probabilistic numerical drift in ctranslate2 inference, with the heavier pass 1 mode (fidelity) increasing the trigger probability via cuDNN auto-tune nondeterminism, GPU thermal accumulation, allocator state, or driver state. **This is a hypothesis, not a confirmed cause.**

This test suite quantifies the failure rate empirically.

---

## Requirements

- WhisperJAV installed (`pip install whisperjav` or via the standalone .exe — the suite invokes it as a subprocess)
- An NVIDIA GPU with CUDA (this bug is GPU-side; the suite is most useful on the same hardware where the bug appears)
- An audio/video file to test against. The reference clip used in the original investigation is a 293-second JAV scene; any media of similar length and content density should work.
- ~1 hour of unattended GPU time for the default test plan (10 fidelity→balanced runs at ~6 min each)

The suite uses only the Python standard library; no `pip install` needed for the suite itself.

---

## Quick Start

### Default plan (10 primary + 9 baseline runs, ~80-100 minutes on RTX 3060)

```bash
python tools/ensemble_failure_rate_suite.py --media path/to/your/clip.mkv
```

This runs:
- 10 iterations of pass1=fidelity + pass2=balanced (the suspect configuration)
- 3 iterations each of pass1=balanced, fast, faster + pass2=balanced (baselines for context)

### Primary-only (faster, just measures the failure rate)

```bash
python tools/ensemble_failure_rate_suite.py --media clip.mkv --primary-only --runs 10
```

### Overnight (20 runs, ~2 hours unattended)

```bash
python tools/ensemble_failure_rate_suite.py --media clip.mkv --primary-only --runs 20
```

### Single configuration run (e.g., baseline only)

```bash
python tools/ensemble_failure_rate_suite.py --media clip.mkv --config B_bal_bal --runs 5
```

---

## Output

The suite writes to `test_media/ensemble_failure_rate_results/` by default (override with `--output-root`):

```
ensemble_failure_rate_results/
├── results.jsonl          # one JSON line per run, append-only (resume-safe)
├── summary.txt            # human-readable aggregate stats per config
├── system_info.json       # GPU model, driver version, OS, Python, faster-whisper info
└── runs/
    ├── 001_A_fid_bal_20260507_140523/
    │   ├── command.txt    # exact whisperjav invocation
    │   ├── log.txt        # full whisperjav stdout/stderr (with --debug)
    │   ├── temp/          # whisperjav --keep-temp output (scene WAVs, intermediate SRTs)
    │   ├── *.pass1.srt    # pass 1 output
    │   ├── *.pass2.srt    # pass 2 output (the one we're measuring)
    │   └── *.merged.whisperjav.srt
    └── ...
```

`results.jsonl` is appended after every run, so a partial run is recoverable. If you Ctrl-C mid-suite, you can analyze whatever finished.

`summary.txt` (rewritten at suite end) shows per-config aggregates, including the failure rate for the primary configuration:

```
Config: A_fid_bal
  Runs:                  10
  Successful:            7
  Catastrophic:          3  (30.0%)
  Errored / timed out:   0
  pass2 entries  min/p50/max:  14 / 50 / 52
  Elapsed s     min/p50/max:   62 / 75 / 387
  Per-run:
    iter   1  pass1= 58 pass2= 50    72s  [OK]
    iter   2  pass1= 56 pass2= 14   381s  [CATASTROPHIC]
    iter   3  pass1= 58 pass2= 52    68s  [OK]
    ...
```

---

## What "Catastrophic" Means

A run is flagged `CATASTROPHIC` when its pass 2 SRT has **fewer than 25 entries** (configurable via `--catastrophic-threshold`). On the reference 293-second JAV clip a healthy pass 2 produces around 50 entries; the buggy runs produce around 14. The threshold is set deliberately conservatively at 25 (half of healthy) to avoid false positives from variations in normal audio coverage.

If you're testing on a different clip, calibrate by first running `--config B_bal_bal --runs 3` (the always-healthy baseline) and noting the pass2 entry count. Set `--catastrophic-threshold` to roughly half of that.

---

## Cross-Machine Distribution

We're collecting failure-rate data from multiple environments to determine whether the bug is GPU-model-specific, driver-version-specific, OS-specific, or universal. If you're running the suite on a non-RTX-3060 system or a non-Windows-11 system, we'd appreciate you sharing:

1. The full `output-root/system_info.json` file
2. The `output-root/summary.txt` file
3. (Optional) A few representative `runs/NNN_A_fid_bal_*/log.txt` files — at least one CATASTROPHIC and one OK if both occurred

Share via attachment in the relevant GitHub issue (TBD link) or email to the project maintainer.

---

## Default Test Matrix

| Config name | Pass 1 | Pass 2 | Runs (default) | Purpose |
|-------------|--------|--------|---------------|---------|
| `A_fid_bal` | fidelity | balanced | 10 | **Primary** — the configuration with the suspected bug |
| `B_bal_bal` | balanced | balanced | 3 | Control — should always succeed |
| `C_fast_bal` | fast | balanced | 3 | Cross-engine but lighter than fidelity |
| `D_faster_bal` | faster | balanced | 3 | Lightest pass 1, ctranslate2-based |

All configurations use the GUI ensemble defaults: `aggressive` sensitivity for both passes, `whisperseg` segmenter, `large-v2` model, `auditok` scene detector for pass 2, `semantic` scene detector for pass 1, `pass1_primary` merge strategy, Japanese language.

---

## Failure Rate Interpretation

- **0% catastrophic across 10 primary runs**: Bug not reproducible on this machine. Useful to know.
- **10-30% catastrophic**: Confirms intermittent bug. Establishes a baseline for measuring fixes.
- **>50% catastrophic**: Bug is highly likely on this configuration; useful for targeted debugging since reproduction is reliable.
- **Errored / timed out**: Configuration error or environmental issue. Check the `log.txt` for the failing run.

---

## Limitations / Caveats

- **GPU thermal state matters**: This suite includes a 5-second cool-down between runs by default (`--cooldown-seconds`). If your GPU's clocks throttle aggressively, results may vary based on ambient temperature and the duration of the run.
- **Non-determinism is the whole point**: A 0%-catastrophic result on 10 runs does not prove the bug doesn't exist on your machine — only that it didn't appear in those 10 runs. The user-facing test that triggered the original investigation took 1 of 1 attempts to surface; statistical confidence requires more runs.
- **Disk space**: Each run with `--keep-temp` produces ~30-50 MB of artifacts (extracted WAV, scene WAVs, intermediate SRTs). 10 runs = ~500 MB. Plan accordingly.
- **The catastrophic threshold is calibrated for one specific reference clip** (293 seconds, ~50 entries baseline). Other clips will need a different threshold.

---

## Running the Suite Without WhisperJAV's Repo

The suite is a single Python file with stdlib-only imports. You can drop it anywhere on a machine that has `whisperjav` installed. To distribute:

```bash
# Just copy the script
cp tools/ensemble_failure_rate_suite.py ~/test_run.py
cd ~
python test_run.py --media ~/your_clip.mkv
```

Or share with another user as `ensemble_failure_rate_suite.py` standalone — they need only Python 3.10+ and a working `whisperjav` install.

---

## Related Files

- `docs/plans/V1814_T142_NONDETERMINISM_INVESTIGATION.md` — the investigation memo this suite was built for
- `test_media/T142/` — original catastrophic test artifacts
- `test_media/T143/` — original good-baseline test artifacts
- `test_media/T144/` — TEST_α/β/δ artifacts (from the rerun investigation)
