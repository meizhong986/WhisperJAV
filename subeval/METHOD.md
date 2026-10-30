# subeval — method

subeval compares a **test** subtitle file with a **reference** subtitle file. For a whole set of clips, it reports
how accurate the text is, which reference lines were missed, which test lines have no counterpart, whether line
openings and endings were captured, and how far line starts and ends are from the reference.

This document defines each step and each indicator exactly. The code files implement them one-to-one, so an
independent reader can check, reproduce or improve the method. The utility uses the Python standard library only.

| Step | Code |
|---|---|
| 1. Read the SRT files | `srt.py` |
| 2. Normalise the text | `normalize.py` |
| 3. Align the characters | `align.py` |
| 4. Classify lines and characters | `evaluate.py` (`_classify`, `evaluate_clip`) |
| 5. Pool the clips | `evaluate.py` (`summarize`) |
| 6. Compare two runs | `compare.py` |
| 7. Write the reports | `report.py`, `definitions.py` (the definitions printed in every report), `cli.py` |

## 1. Reading
An SRT *block* is a group of non-blank lines, separated from the next block by blank lines.
- **Timing line.** `H:MM:SS,mmm --> H:MM:SS,mmm`. A `.` is accepted instead of the `,`, and the hours may have any
  number of digits.
- **Block number.** It may come before the timing line.
- **Text.** Every line after the timing line is text, joined with a line break.
- **Problems are reported, never skipped silently.** A block with no readable timing line, or whose end comes before
  its start, is left out and listed under *Reading problems*.
- **Byte-order mark and CRLF line endings** are accepted.
- **A missing blank line is reported.** A timing line inside a block starts a new block there, together with the
  number line just before it, if any.
- **Lines keep the file's own block numbers.** The pairing record names lines by these numbers (R12, T7), so they
  match the file even when a block was skipped.
- **Reference notes that may be damaged are flagged.** A reference line with a bracket that has no partner is listed
  too: its note may be damaged, and its text is then compared as speech.

Each block is one **line**: start, end, text.

## 2. Normalisation (profile `ja-v1`)
Applied to each line, in this order:
1. **Remove bracketed notes.** This means every `(` or `（`, the characters up to the next `)` or `）`, and that closing
   bracket. It removes speaker names and sound notes such as （村西） or (笑).
2. **Unicode NFKC.** Full-width and half-width forms become one form.
3. **Keep letters and digits only.** Only characters whose Unicode category starts with L (letter) or N (number) are
   kept. Punctuation, symbols such as ♪, spaces and line breaks go. Kana marks such as ー and 々 are letters and stay.

A line with no characters left is *empty*. It is counted, but it takes no part in the line indicators.

## 3. Alignment (`align.py`)
### Building the streams
For each clip, the lines of each file are sorted by start time, then by end time. The normalised texts are joined
into a **reference stream** and a **test stream**. Each character keeps the line it came from, and that line's start
and end time. No timing finer than the line exists, so every character of a line carries the times of the whole line.

### The edit distance
An alignment maps the two streams onto each other, keeping their order. Each reference character is one of:
- *hit*: paired with an identical test character;
- *wrong*: paired with a different test character;
- *missing*: not paired.

A test character that is not paired is *extra*. The number of **edits** is wrong + missing + extra. This is the
usual Levenshtein edit distance, in which substitution, deletion and insertion each cost 1.

### Constraint: the time window
A reference character and a test character may be paired (as a hit or as wrong) only when the **gap** between their
two lines is at most the window *W* (default 1.0 s).
- **The gap.** For the time spans [a1, a2] and [b1, b2], gap = max(0, max(a1, b1) − min(a2, b2)). It is 0 when the two
  lines overlap.
- **Why.** Without the window, a word could be paired with an identical word said a minute later, or a missed
  reference line could be paired, character by character, with an unrelated test line elsewhere. Both happen in
  practice; see `PRIOR_ART.md`, findings D and E.

### Choosing among equally good alignments
Many alignments can have the same smallest number of edits. subeval picks the one that is smallest in this order:
1. number of edits;
2. sum, over paired characters, of the gap in milliseconds;
3. sum, over paired characters, of the distance between the two lines' midpoints in milliseconds.

This is computed exactly in one dynamic-programming pass. Each step's cost is one integer:
- edits × *E* + gap × *M* + midpoint distance;
- *M* is larger than any possible total midpoint distance;
- *E* is larger than any possible total of gap × *M* + midpoint distance.

So a lower level can never outweigh a higher one. The table has (n+1) × (m+1) cells for n reference and m test
characters. A cell may take a pairing step only when the window allows it.

### Remaining ties
Ties that remain are broken by a fixed move order:
- **forward:** pair, then *missing*, then *extra*, applied while tracing back from the end of the streams;
- **mirror:** the same order applied to both streams reversed.

Where the cost cannot tell two alignments apart, for example reference ああそう against test あそう (which あ was
dropped?), the forward order tends to place the loss towards the start of a line and the mirror order towards the
end. Either alone would bias the start and end indicators, so subeval computes both alignments. It reads every
indicator from the forward alignment, except two:
- **Start and end classes** (section 4) are kept only where both alignments agree; otherwise they are *uncertain*.
- **Lost-character counts** take the lower of the two values.

How many lines differ between the two alignments is reported for each indicator, under *Ambiguity*. Two alignments
can agree where a third equally good alignment would not, so these counts are a **lower bound** on the ambiguity. A
brute-force check over every optimal alignment of small cases found such cases; on three real runs they affected
about 1 line edge in 450.

### Plain alignment
The **plain** edit distance is also computed, with no window and no tie rules. It is the textbook Levenshtein
distance, and it gives *CER (plain)*, which can be compared with earlier measurements.

## 4. Indicators for one clip (`evaluate.py`)
Let *R* be the number of reference characters.

### Text
- **CER (plain)** = plain edit distance / *R*.
- **CER** = (wrong + missing + extra) / *R*, from the time-aware alignment. It is never lower than CER (plain). The
  difference counts text that exists but sits more than *W* from its reference line: it is counted once as missing
  and once as extra.
- **Wrong, Missing, Extra**: each count divided by *R*. Together they add up to CER.

### Missed lines, suspected hallucination lines, misheard lines
- **Missed line.** A non-empty reference line none of whose characters is a hit, so the viewer gets nothing right
  for that line. It is split into two kinds:
  - *nothing*: every character is missing;
  - *wrong only*: characters are paired, but none is right.

  Also reported: how many missed lines have only 1–2 characters.
- **Suspected hallucination line.** A non-empty test line none of whose characters is paired with a reference
  character (all are *extra*). It has no counterpart in the reference within *W*. It is "suspected" because it may be
  real speech that the reference does not subtitle.
- **Misheard test line.** A non-empty test line whose characters are paired with reference characters, but none of
  them is right. This is a mishearing, not a hallucination, and its reference line is a missed line (*wrong only*).
- **Lines decided by a single hit.** Reference lines and test lines of 4 or more characters with exactly one hit. One
  shared character, possibly by chance, then decides whether the line is missed, or a hallucination or misheard line.
  This count shows how much those two indicators rest on single characters.

### Line openings and endings
These are computed for each reference line that is not missed.
- **Home line.** The test line holding the most hit characters of the reference line. If two test lines hold
  equally many, the earlier one is the home line.
- **Start class.** Read from the line's first character, under each of the two alignments:
  - *ok*: a hit, in the home line;
  - *misplaced*: a hit in a test line other than the home line, meaning our line boundary cuts through the reference
    line;
  - *wrong*: substituted, in whichever test line;
  - *lost*: missing;
  - *uncertain*: the two alignments give different classes.
- **Start lost characters.** The number of missing characters before the first character that is not missing. Per
  line it is the lower of the two alignments' values, summed over lines.
- **End class** and **end lost characters**: the same, read from the line's last character backwards.

What the classes show about the segmenter:
- A lost start or end means the text was not delivered: the line was cut too early at that edge, or not heard.
- A misplaced start or end means the text was delivered but split differently: the boundary falls inside the
  reference line.
- Text placed far from where it belongs shows up as missing plus extra in CER, but not in CER (plain).

### Timing
A test-line edge is measured only when it corresponds to a reference-line edge:
- **start error** = test line start − reference line start. It is measured when the test line's first hit is the
  reference line's first hit, so the test line opens that reference line.
- **end error** = test line end − reference line end. It is measured when the test line's last hit is the reference
  line's last hit, so the test line closes that reference line.

What this gives:
- Each test line gives at most one start and one end, and each reference line receives at most one of each.
- When one test line holds several reference lines, its start is measured against the first of them and its end
  against the last, so a reference line inside such a line is not charged for the gap to its neighbours.
- When a reference line continues across two test lines, the test-line edges between them fall inside the reference
  line. They have no reference edge to be measured against and are not timed; they are counted under the *misplaced*
  start and end classes.
- If a line's first characters were lost, its start error compares the test line's start with the reference start,
  and so includes the lateness that lost them.

Reported for starts and for ends:
- the number measured;
- the median of |error|;
- the signed mean and the signed median (+ means our line is later);
- the share with |error| ≤ 0.5 s.

### Ambiguity
The forward and mirror alignments are compared, and for each indicator subeval reports the number of reference
lines whose value differs. The indicators are missed status; start class and lost count; end class and lost count;
and timing (start error, end error). For test lines it reports how many differ in hallucination or misheard
status. Only the start and end classes are resolved, as *uncertain*; the other indicators are taken from the forward
alignment.

## 5. Pooling clips (`summarize`)
- **Counts are added up** over clips.
- **CER and its parts** are total edits / total *R*.
- **Medians, means and shares** are taken over all measured boundaries of all clips together.

## 6. Comparing two runs (`compare.py`)
Both runs must cover the same clips and must have been scored against the same reference (the same lines, times
and texts). This is checked, and anything else stops with an error.
- **Text, per clip:** the change in edits, in plain edit distance, in missed lines, in suspected hallucination
  lines and in misheard test lines.
- **Timing, per clip and pooled:** only on the reference lines whose start (or end) error was measured in **both**
  runs. Lines measured in only one run are counted but not compared. A clip is *better* or *worse* when the
  candidate's median |error| is lower or higher than the base's by more than 0.02 s (`--tolerance`).
- **Clip counts:** the number of clips with fewer edits and the number with more are reported. Because only
  consistent changes are meaningful on a small set of clips, these counts belong next to every pooled difference.

## 7. Outputs
- `report.md`:
  - the parameters and the subeval version;
  - every input file with its SHA-256;
  - reading problems;
  - the pooled indicators, the indicators per clip, and the definitions.
- `result.json`: every number, down to each line.
- `pairs.txt`: the pairing record. For every reference line it gives the test lines holding its characters, the
  counts, the classes and the timing errors. For every test line it gives the reference lines it holds and whether
  it is a suspected hallucination. A person can check any decision against the two subtitle files.

## 8. Validation (subeval 0.1.0, 2026-10-08)
- **Hand-built cases.** `tests/test_subeval.py` has 28 cases, each pairing a situation with the answer a person
  would give:
  - a missed line, and a mishearing, which is a missed line and a misheard test line but not a hallucination;
  - a hallucinated line, and a line decided by one shared character;
  - an opening cut off, an ending cut off, a wrong first character;
  - a boundary inside a reference line, with both of the reference line's edges timed where they are;
  - one test line holding two reference lines;
  - a repeated word at the wrong time;
  - equal-cost alternatives, and empty lines;
  - damaged SRT blocks, a missing blank line, and the file's own block numbers;
  - comparing two runs, including different clip sets or references, which are refused.
- **Exactness of the alignment.** On more than 500 random small cases with several lines per stream, the forward and
  mirror alignments are checked against an enumeration of every possible alignment. Both must reach the true minimum
  of (edits, total gap, total midpoint distance).
- **Plain edit distance.** It is checked against a brute-force recursive definition on 200 random string pairs. The
  time-aware alignment with an unlimited window is checked to give the plain edit count on 100 random pairs.
- **An independent review** (one adversarial reviewer, 2026-10-08) brute-forced the alignment on 3,000 cases and
  found it exact. It also found faults in the first draft, all corrected in 0.1.0 before any result was reported:
  - timing measured across line splits;
  - mishearings counted as hallucinations;
  - several statements in this document;
  - a missing blank line in an SRT swallowing a block;
  - block numbers not matching the file.
- **Regression against the WhisperJAV 1.9.4 measurement scripts.** On 161 runs of 7 clips each:
  - CER (plain) equals `score_segmenters.py`'s CER exactly in every run;
  - the number of lines read from every SRT equals pysrt's.
- **Window sensitivity.** Four runs (Qwen3-ASR with TEN at balanced; anime-whisper with FireRedVAD at aggressive;
  anime-whisper with Silero at aggressive; Qwen3-ASR with WhisperSeg at conservative) were scored at *W* = 0.5, 1, 2,
  3, 5 s and with no window.
  - For TEN, FireRedVAD and WhisperSeg, between 0.5 and 2 s: CER changes by at most 0.008, missed lines by at most 4,
    and the end median by at most 0.006 s.
  - For Silero, CER at *W* = 1 s is 1.00 against 0.54 plain. In that run, one test line from 42 to 52 s holds the
    speech from 20 to 44 s, and the window counts that text as missing where it was spoken and extra where it was
    shown.

## 9. Known limits
- **No timing finer than the line.** Within a line, characters are not timed. So a line opening is judged by its
  text (start class), not by its time.
- **The reference is a set of subtitles, not a transcript.** Its times are display times, and its text may leave out
  or add words. Absolute values therefore reflect the reference as well. Use differences between runs scored against
  the same reference.
- **Short lines are fragile.** A 1–2-character line (うん, ハァ) is missed if its one or two characters are not hits.
  Their share of missed lines is reported.
- **One character can decide a line.** Missed, hallucination and misheard status depend on whether a line has at
  least one hit. A single shared character, such as う, makes an unrelated line look delivered. These cases are
  counted ("lines decided by a single hit"), not corrected.
- **Overlapping reference lines in a different order.** When two reference lines overlap in time and the test gives
  them in the opposite order, the order-keeping alignment can pair only one of them. The other comes out as missed,
  and its test line as a hallucination, even if both were transcribed perfectly.
- **Uncertain classes are not resolved.** They are reported as uncertain, and no tie order is presumed right.
- **The window is a parameter.** It is printed in every report. A run whose lines are badly displaced in time
  depends on it strongly; see the Silero case above.
- **Size.** Memory and time grow with n × m per clip. 7 clips of about 450 characters each take about 1–2 seconds
  for both alignments.
