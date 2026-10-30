# Earlier evaluation code in WhisperJAV, and why subeval replaces it for measurement

Examined on 2026-10-08, on 163 runs × 7 Netflix clips (305 reference lines each): 49,715 reference lines, 38,205 test lines and 28,933
accepted pairs. Scripts: `scripts/measure/v1.9.4/subeval_step1_prior_art.py` and `subeval_step1_wrong_partner.py` (they import WhisperJAV, being about its code).

## The earlier code
- **Line pairing:** `whisperjav.bench.matcher.match_subtitles`. It is used by `whisperjav.bench.timing` (the timing
  figures in the 1.9.4 measurements) and by `whisperjav.bench.regression` (`whisperjav-accuracy-gate`). Each
  reference line, in file order, takes the one free test line with the highest text similarity
  (`difflib.SequenceMatcher.ratio` on `normalize_text`). That line must overlap the reference line by ≥ 0.1 s and
  have a similarity of ≥ 0.2. Each test line can be taken only once.
- **Character alignment:** the 1.9.4 measurement scripts use `rapidfuzz.distance.Levenshtein.editops` on the whole
  clip's text. It returns one of the optimal alignments, chosen by an internal tie order.
- **Two different normalisations:** `whisperjav.bench.metrics.normalize_text` removes a fixed list of punctuation,
  while the measurement scripts keep letters and digits only. So `whisperjav-accuracy-gate` and the measurement
  scripts compute different CERs from the same files.

## Findings
| | Finding | Count | Effect |
|---|---|---|---|
| A | Reference lines left unpaired although an overlapping test line holds their whole text, because that test line was already taken by an earlier reference line (one test line, one reference line) | 1,555 | Such lines drop out of the timing figures, and the test line counts as matched to the earlier line only |
| A2 | The same, because a short reference line inside a long test line falls below similarity 0.2 (for example, 2 characters in a 20-character line give 2×2/22 = 0.18) | 12 | Short lines inside long lines can never be paired |
| B | Test lines left unpaired (the matcher's "hallucinated" list) | 9,272 | Of these, 192 hold at least half of an overlapping reference line's text. The list is not a hallucination count. |
| C | Accepted pairs where another overlapping test line holds at least 2 more of the reference line's characters (wrong partner) | 717 (2.5%) | Their median end error is 1.40 s, against 0.54 s for the other pairs, so they inflate the end error |
| D | "Lines with no text" (all characters deleted) under the forward and under the mirror alignment | 6,974 vs 6,629; 3,417 lines classed differently | The count depends on an arbitrary tie order, by up to 9 lines per clip |
| E | Characters the plain alignment calls equal, by the time gap between their two lines | 96.9% overlap; 1.72% more than 5 s apart; at most 73.6 s apart | Without a time window, identical words far apart in time are paired |
| F | Each run's own end-error median is taken over the pairs that run happened to match | — | Two runs are measured on different line sets. `compare_clip` fixes this, but the per-run medians were also quoted. |
| G | When one test line holds several reference lines, the pair's end error includes the gap to the later reference lines. 30–45% of test lines overlap two or more reference lines. | — | The end error partly measures line splitting, not where speech ends |

## How subeval answers each finding
- **A, A2, B, C:** there is no line pairing. Lines are related through their characters, so a test line can hold
  any number of reference lines, and the reverse.
- **D:** both tie orders are computed. Start and end classes are reported as uncertain where the two disagree.
  For every other indicator (missed lines, hallucination and misheard lines, timing), the number of lines that differ
  is reported under *Ambiguity*, and the value is taken from the forward alignment. On the run checked, no missed or
  hallucination status differed.
- **E:** the time window.
- **F:** `compare` uses only boundaries measured in both runs.
- **G:** a test-line start or end is timed only when it opens or closes a reference line, judged by the first and
  last correctly heard characters. Test-line edges that fall inside a reference line are not timed; they are counted
  as *misplaced*.
