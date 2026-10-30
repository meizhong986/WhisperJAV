"""The exact definition of every indicator, as printed in every report (one source; METHOD.md explains the method)."""

DEFINITIONS = """\
## Definitions (subeval {version}; normalisation profile {profile}; time window {window} s)

**Inputs.** A reference SRT and a test SRT per clip. Each SRT block is one *line* (start time, end time, text).
Lines are put in time order (start, then end). Each line's text is normalised (profile {profile}: bracketed notes
removed, NFKC, only letters and digits kept). The normalised texts of a clip are joined in time order into one
*reference stream* and one *test stream*; every character remembers the line it came from. A line with no
character left is *empty*: counted, but not used by the line indicators.

**Alignment.** An edit-distance alignment of the test stream to the reference stream (substitution, deletion,
insertion each cost 1). A reference character may be paired with a test character only if the time gap between
their lines is at most {window} s (gap = 0 when the lines overlap). Among the alignments with the fewest edits, the
one with the smallest total gap, then the smallest total distance between line midpoints, is used. Ties that remain
are broken by a fixed order (pair, then missing, then extra) while tracing back from the end of the streams
(*forward*); the *mirror* alignment applies the same order to both streams reversed. Each reference character is
*hit* (paired with the same character), *wrong* (paired with a different character) or *missing* (no partner);
each unpaired test character is *extra*. All indicators are read from the forward alignment, except CER (plain)
and the start and end classes (see below).

**Text**
- **CER (plain)**: plain edit distance between the two streams, no time window, divided by the number of reference
  characters; summed over clips before dividing. Comparable with earlier measurements.
- **CER**: (wrong + missing + extra) / reference characters, from the time-aware alignment. Never lower than CER
  (plain); the difference is text that exists but was paired more than {window} s from its reference line.
- **Wrong / Missing / Extra**: each count / reference characters (they add up to CER).
- **Missed lines**: non-empty reference lines with no hit - the viewer gets nothing right for that line. Split
  into *nothing* (every character missing) and *wrong only* (characters paired, none right); also counted: those
  of 1-2 characters.
- **Suspected hallucination lines**: non-empty test lines none of whose characters is paired with a reference
  character (all extra): text with no counterpart in the reference within {window} s. "Suspected": it may be real
  speech the reference does not subtitle.
- **Misheard test lines**: non-empty test lines whose characters are paired with reference characters but none is
  right (a mishearing, not a hallucination; its reference line is a missed line, *wrong only*).
- **Lines decided by a single hit**: reference lines and test lines of 4 or more characters with exactly one hit -
  one accidental shared character decides whether such a line is missed or a hallucination.

**Line openings and endings** (for each reference line that is not missed)
- *Home line*: the test line holding the most hit characters of the reference line (tie: the earliest).
- **Start**: the class of the reference line's first character - *ok* (hit, in the home line), *misplaced* (hit, in
  another test line: our line boundary cuts through the reference line), *wrong* (substituted, in any test line),
  *lost* (missing). The class is computed from the forward and the mirror alignment; where the two differ it is
  *uncertain*. **Start lost characters**: the number of missing characters before the first non-missing one - per
  line the lower of the two alignments' values - summed.
- **End**: the same for the last character of the reference line. **End lost characters**: likewise, from the end.

**Timing** (seconds; + means our line is later)
- **Start error** = start of a test line - start of a reference line, measured when the test line's first hit is
  the reference line's first hit (the test line opens that reference line). **End error** = end of a test line - end
  of a reference line, measured when the test line's last hit is the reference line's last hit. A test-line edge
  that falls inside a reference line (the reference line continues in another test line) has no reference edge to
  be measured against and is not timed; such edges are counted as *misplaced* above.
- **Starts/Ends measured**: how many errors were measured. **median |error|**, **signed mean**, and **within
  {within} s**: the share of measured errors with |error| <= {within} s.

**Ambiguity.** For each indicator, the number of lines whose value differs between the forward and the mirror
alignment: missed status, start class and lost count, end class and lost count, timing (start and end error), and
for test lines hallucination/misheard status. Only start and end classes are resolved (as *uncertain*); the others
are taken from the forward alignment. Two alignments can agree where a third equally good one would not, so these
counts are a lower bound. A difference between two runs smaller than its ambiguity is not meaningful.
"""
