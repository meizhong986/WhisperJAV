# subeval

subeval compares a test subtitle file with a reference subtitle file. It is a standalone research and measurement
utility: Python 3.10 or later, standard library only, and nothing imported from WhisperJAV. It lives in the
WhisperJAV repository but is not part of the WhisperJAV package.

It reports:
- **text accuracy**: the character error rate, split into wrong, missing and extra characters;
- **missed lines**: reference lines of which no character was transcribed correctly;
- **suspected hallucination lines**: test lines none of whose characters is paired with any reference character,
  kept apart from **misheard lines** (paired, but no character right);
- **line openings and endings**: whether each reference line's opening and ending were captured, lost, got wrong,
  or fell into another test line;
- **timing**: start and end errors, measured only where a test line opens or closes a reference line.

All of these come from one time-aware character alignment. `METHOD.md` defines every step and indicator exactly,
and every report carries the same definitions. `PRIOR_ART.md` explains why it replaces WhisperJAV's earlier
evaluation code for measurement.

## Use
Run from the folder that contains `subeval/` (in WhisperJAV, the repository root):

```
python -m subeval score REF.srt TEST.srt
python -m subeval score --ref-dir REFS --test-dir RUN --out OUT
python -m subeval compare --ref-dir REFS --base-dir RUN_A --cand-dir RUN_B --out OUT
python -m subeval --help
```

**Folder mode pairs files by clip id.** The clip id is the reference file name without `.ja.srt` or `.srt`. The test
folder must hold exactly one `.srt` whose name starts with `<clip id>.`; anything else is an error.

**Options:**
- `--window S`: the largest time gap between two lines whose characters may be paired (default 1.0 s).
- `--profile`: the text normalisation (default `ja-v1`).
- `--tolerance S` (compare only): the timing difference below which a clip counts as the same (default 0.02 s).

**What `--out` writes:**
- `report.md`: the parameters, the inputs with their SHA-256, reading problems, results for all clips together and
  per clip, and the definitions.
- `result.json`: every number, down to each line.
- `pairs.txt`: for each reference line, the test lines its characters went to and how it was classified.

`compare` writes a `base/` and a `cand/` report, plus `compare.json` with the clip-by-clip differences.

From Python:
`from subeval.cli import pair_folder, score_files`, `from subeval.evaluate import summarize`,
`from subeval.compare import compare`.

## Tests
```
python -m pytest subeval/tests/test_subeval.py
```

## Version
0.1.0 (2026-10-08): first version, written for the WhisperJAV 1.9.4 REQ2 measurements. Any change to a definition
must raise the version, because reports name the version they were made with.
