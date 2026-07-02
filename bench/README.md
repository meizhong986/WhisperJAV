# Accuracy Regression Gate

Prevents ASR-quality regressions (preset retunes, model default changes,
routing bugs) from shipping unnoticed. Motivated by three incidents that
reached users: the v1.8.12 silero grouping retune, the large-v3 aggressive
preset incompatibility (F4/F6 catastrophic output), and the YAML/Pydantic
sensitivity mismatch.

## Layout

    bench/
      corpus.example.yaml   # manifest template - copy to corpus.yaml
      corpus.yaml           # your real manifest (GT paths committed)
      thresholds.yaml       # gate tolerances
      gt/                   # human-verified ground-truth SRTs (committed)
      media/                # source clips (NOT committed - gitignored)
      baselines/            # per-pipeline baseline metrics (committed)

## Workflow

1. Build the corpus once: pick 5-10 clips (3-5 min each) covering clean
   dialogue, whisper/ASMR, moaning-heavy (repetition-risk), dialect, and
   BGM-heavy audio. Write human-verified GT SRTs into `bench/gt/`.

2. Record a baseline on a known-good release (GPU machine):

       whisperjav bench/media/*.m4a --mode balanced --output-dir out/balanced
       # rename/copy outputs to out/balanced/<clip_id>.srt, then:
       python -m whisperjav.bench.regression_cli baseline \
           --manifest bench/corpus.yaml --hyp-dir out/balanced \
           --label "v1.8.14 balanced" -o bench/baselines/balanced.json

3. Gate any candidate change (new preset values, model default flip,
   segmenter default change) by re-running the corpus and:

       python -m whisperjav.bench.regression_cli gate \
           --manifest bench/corpus.yaml --hyp-dir out/candidate \
           --baseline bench/baselines/balanced.json \
           --thresholds bench/thresholds.yaml

   Exit code 1 + a violation report means the change regressed.

## Metrics

Per clip: CER (global, char-level, NFKC-normalized), segment precision /
recall / F1 (temporal-overlap + text-similarity matching), mean IoU of
matched pairs, GT time coverage, false-alarm duration ratio, repetition
ratio (longest identical-text run / total - the Whisper loop signature),
and subtitle count ratio (hard-bounded: the F4/F6 collapse produced ~0.1).

## CI

`.github/workflows/accuracy-gate.yml` runs the scorer's unit tests plus the
config-resolution suite on every PR (CPU-only, no media needed). Full
corpus gating needs a GPU and is run manually before releases; if stored
candidate outputs are committed or artifact-uploaded, the same gate command
runs in CI unchanged.

## Use cases queued for this gate

- Re-tune aggressive preset for large-v3 (v1.9.0 P0): sweep decode params,
  gate each operating point, restore large-v3 as default when it passes.
- Flip simple-mode segmenter default silero-v3.1 -> whisperseg (needs the
  nvv_heavy / F4-style clips to pass with the v1.9.0 unified routing).
- Any silero/whisperseg/TEN preset value change.
