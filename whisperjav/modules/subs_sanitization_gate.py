"""Subtitle sanitization gate (1.10.3; owner W753 CL2, W754, default strict W756 5, name W772; decided W789-W803).

The user decides how strictly the subtitle clean-up treats the owner's pass-through words (--subs-sanitization-gate):
  strict   (default) the clean-up as before. The pass-through file is not read.
  relaxed  lets more terms through: a subtitle line made only of the pass-through words of its language, that a
           clean-up step would remove, is not removed by that step and leaves it as it came in. A line the step
           keeps, or only shortens or edits, gets the step's own result, as in strict; so does a loop of the words
           (a repeated word or phrase found by the repetition rule, owner W803 N14). The CPS check (characters per
           second) and the repetition rule are not asked: they run as in strict (owner W802 2, W803 N11).

The one test (spares), the same for every step (owner W799 1, W802 1): the text and each word are normalized the same
way -- NFKC, case folding, then every space, punctuation mark, symbol, separator, control and format character
removed, and the emoji variation selectors and the keycap mark -- and the text must be wholly a sequence of the words,
each optionally followed by the elongation marks っ ッ ー ぅ ゥ; words may repeat and mix (owner W799 5). So イク,
イクッイクッ, 気持ちいい…, 気持ちいい♡, 「イク」, 気持ち いい are made only of the words; あっ、イク, もうイク, いくら,
イクラ are not.

Where it is asked (each step keeps its own rule; strict's result is never changed for a line the step keeps): the
hallucination filter (HallucinationRemover.remove_hallucinations: its word list, bracket, emoji and regex rules), the
text cleaner's word list (assembly_text_cleaner), the English clean-up's word list and artefact strips, the
non-linguistic (sound-line) filter, the nonverbal (lone-token) filter, MOSS-CrispASR's interjection rules; ChronosJAV's
split check uses the two filters' own tests. Not asked: the CPS checks, the repetition cleaners, fast and faster's
filler-word removal at ASR time (owner W790 5), the Whisper modules' ASR-time suppress list.

The words are data, defined and extended by the owner per language key, like the hallucination filter list: the file
WhisperJAV_sanitization_passthrough_v01.json on his gist (HallucinationConstants.PASSTHROUGH_LIST_URL), read in
relaxed only, from the cache (under 7 days), the gist, the stale cache, or the copy shipped in
whisperjav/data/hallucination_filters. A copy that is not valid JSON or not of the file's form is refused with one
line, and the next source is read; with no usable copy, relaxed runs as strict and says so once.

Relaxed: one log line per file with the lines each step kept for the words (report_file); in an Ensemble pass process
the gate's lines name the pass (set_pass, owner W799 6). The run summary is the same as in strict (owner W790 7). The
setting reaches every process of a run through the environment: main.py sets ENV from --subs-sanitization-gate and
the Ensemble's pass processes inherit it.
"""
import json
import os
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

from whisperjav.utils.logger import logger

ENV = "WHISPERJAV_SUBS_SANITIZATION_GATE"
CHOICES = ("strict", "relaxed")
DEFAULT = "strict"

# The hallucination filter list's keys for the language codes (HallucinationRemover.LANGUAGE_CODE_MAP); a language
# without a key of its own takes the Japanese one, as the hallucination filter's lookup falls back to it.
LANGUAGE_KEYS = {
    "ja": "japanese", "jp": "japanese", "japanese": "japanese",
    "ko": "korean", "korean": "korean",
    "zh": "chinese", "zh-cn": "chinese", "zh-tw": "chinese", "chinese": "chinese",
    "en": "english", "english": "english",
}
JAPANESE = "japanese"
ENGLISH = "english"

# After a word, inside the normalized text (NFKC turns half-width ｯ into ッ; 〜 and ～ are symbols and are removed).
ELONGATION = "っッーぅゥ"
# Marks NFKC keeps that belong to an emoji, not to a word: the variation selectors and the combining keycap.
_EMOJI_MARKS = frozenset("︎️⃣")

# The repetition rule's findings that make a line a loop (repetition_cleaner.RepetitionCleaner): a repeated word or
# phrase, its repeated-text check and its length limit. Its character patterns (a drawn-out sound such as イク〜〜〜〜
# or イクーーーー) are not a loop: the elongation marks count as part of the word (owner W799 5).
_LOOP_FINDINGS = frozenset({
    "phrase_with_separator", "multi_char_word", "phrase_with_comma", "wavedash_comma_phrase",
    "sentence_phrase_repetition", "generic_repetition_safety_net", "length_limit_truncation",
})

# Step names in the per-file line (the console's own words for the steps).
HALLUCINATION_FILTER = "hallucination filter"
TEXT_CLEANER = "text cleaner"
NONLINGUISTIC = "non-linguistic filter"
NONVERBAL = "nonverbal filter"
INTERJECTIONS = "interjection rules"
ENGLISH_CLEANUP = "English clean-up"

PASSTHROUGH_CACHE = "sanitization_passthrough.json"

_words: Optional[Dict[str, Tuple[str, ...]]] = None   # key -> words, normalized; read once per process
_refused = False                                       # no usable copy: relaxed runs as strict
_kept: Counter = Counter()
_pass: Optional[int] = None                            # the Ensemble pass of this process, or None
_repetition = None                                     # the repetition rule, built on first use


def current() -> str:
    """The run's setting: 'strict' unless the environment says 'relaxed'."""
    value = (os.environ.get(ENV) or DEFAULT).strip().lower()
    return value if value in CHOICES else DEFAULT


def relaxed() -> bool:
    return current() == "relaxed"


def key_for(language: Optional[str]) -> str:
    """The list key of a language code or key ('ja' -> 'japanese'); Japanese when the language has none."""
    return LANGUAGE_KEYS.get((language or "").strip().lower(), JAPANESE)


def set_pass(number: Optional[int]) -> None:
    """The Ensemble pass this process runs (the pass worker says it when it starts); its lines name it."""
    global _pass
    _pass = number


def _name(relaxed_tag: bool = False) -> str:
    """The start of the gate's console lines, naming the pass in an Ensemble pass process."""
    if relaxed_tag:
        return f"Subtitle sanitization gate (relaxed, pass {_pass})" if _pass else \
            "Subtitle sanitization gate (relaxed)"
    return f"Subtitle sanitization gate, pass {_pass}" if _pass else "Subtitle sanitization gate"


def normalize(text: str) -> str:
    """The one normalization for a text and a word: NFKC, case folding, then every space, punctuation mark,
    symbol, separator, control and format character removed, and the emoji variation selectors and keycap."""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(ch for ch in folded
                   if not (ch.isspace() or ch in _EMOJI_MARKS or unicodedata.category(ch)[0] in "PSZC"))


def validate(data) -> Optional[str]:
    """Why a pass-through file is refused, or None: it must be an object whose keys other than "meta" each hold a list
    of strings that are words (not empty, and not empty once normalized, such as "♡")."""
    if not isinstance(data, dict):
        return "the file is not a JSON object"
    for key, value in data.items():
        if key == "meta":
            continue
        if not isinstance(value, list):
            return f'"{key}" is not a list'
        for item in value:
            if not isinstance(item, str) or not normalize(item):
                return f'"{key}" holds an entry that is not a word: {item!r}'
    return None


def _refuse(where: str, why: str) -> None:
    logger.warning(f"{_name()}: the pass-through words from the {where} are refused ({why})")


def _parse(raw: str, where: str) -> Optional[dict]:
    """A copy's data, or None (refused with one line when it is not valid JSON or not of the file's form)."""
    try:
        data = json.loads(raw)
    except ValueError:
        _refuse(where, "not valid JSON")
        return None
    why = validate(data)
    if why:
        _refuse(where, why)
        return None
    return data


def _read_copy(path: Path, where: str) -> Optional[dict]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.debug(f"Subtitle sanitization gate: no {where} ({e})")
        return None
    return _parse(raw, where)


def load_passthrough() -> Optional[dict]:
    """The pass-through file, as the hallucination lists are read: a fresh cache (under 7 days), the gist (the cache
    is refreshed), the stale cache, the shipped copy. An empty address reads the shipped copy only; an address that is
    not http(s) is read as a local file. None when no copy is usable."""
    from whisperjav.config.sanitization_constants import HallucinationConstants
    from whisperjav.data.hallucination_filters import get_bundled_passthrough_path
    from whisperjav.modules.hallucination_remover import CACHE_DIR, CACHE_MAX_AGE_DAYS, DOWNLOAD_TIMEOUT

    url = HallucinationConstants().PASSTHROUGH_LIST_URL
    if url and not url.startswith(("http://", "https://")):
        return _read_copy(Path(url), f"file {url}")
    if url:
        cache = CACHE_DIR / PASSTHROUGH_CACHE
        try:
            fresh = cache.exists() and (time.time() - cache.stat().st_mtime) / 86400 < CACHE_MAX_AGE_DAYS
        except OSError:
            fresh = False
        if fresh:
            data = _read_copy(cache, "cache")
            if data is not None:
                logger.debug("Subtitle sanitization gate: pass-through words: cache")
                return data
        raw = None
        try:
            import requests
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            raw = response.content.decode("utf-8")
        except Exception as e:                      # network, HTTP status, decoding: try the next source
            logger.debug(f"Subtitle sanitization gate: pass-through words not downloaded ({e})")
        if raw is not None:
            data = _parse(raw, "gist")
            if data is not None:
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                except OSError as e:
                    logger.debug(f"Subtitle sanitization gate: cache not written ({e})")
                logger.debug("Subtitle sanitization gate: pass-through words: gist (fresh download)")
                return data
        if cache.exists() and not fresh:            # a fresh cache already read (and refused) is not read again
            data = _read_copy(cache, "stale cache")
            if data is not None:
                logger.debug("Subtitle sanitization gate: pass-through words: stale cache (network unavailable)")
                return data
    data = _read_copy(get_bundled_passthrough_path(), "shipped copy")
    if data is not None:
        logger.debug("Subtitle sanitization gate: pass-through words: bundled package data")
    return data


def _load() -> Dict[str, Tuple[str, ...]]:
    """The pass-through words of every key, normalized, read once per process (relaxed only)."""
    global _words, _refused
    if _words is None:
        data = load_passthrough()
        words: Dict[str, Tuple[str, ...]] = {}
        _refused = data is None
        if data is None:
            logger.warning(f"{_name()}: the pass-through words could not be read; relaxed runs as strict in this run.")
        else:
            for key, items in data.items():
                if key == "meta":
                    continue
                normalized = {normalize(w) for w in items}
                words[key] = tuple(sorted((w for w in normalized if w), key=len, reverse=True))
            logger.debug("Subtitle sanitization gate: pass-through words loaded: %s",
                         ", ".join(f"{k} {len(v)}" for k, v in words.items()))
        _words = words
    return _words


def words(key: str) -> Tuple[str, ...]:
    """The pass-through words of a list key (normalized); none in strict, where the file is not read."""
    if not relaxed():
        return ()
    return _load().get(key, ())


def _made_of(normalized: str, ws: Tuple[str, ...]) -> bool:
    """True when the whole normalized text is a sequence of the words, each optionally followed by elongation marks.
    Worked out from the end of the text to its start (ok[i]: text[i:] is such words), so its time grows with the
    text's length times the number of words (a regex of overlapping words can take exponential time)."""
    n = len(normalized)
    if not ws or not n:
        return False
    ok = [False] * (n + 1)
    ok[n] = True
    for i in range(n - 1, -1, -1):
        for w in ws:
            if not normalized.startswith(w, i):
                continue
            j = i + len(w)
            while True:
                if ok[j]:
                    ok[i] = True
                    break
                if j < n and normalized[j] in ELONGATION:
                    j += 1
                else:
                    break
            if ok[i]:
                break
    return ok[0]


def is_words(text: str, key: str = JAPANESE) -> bool:
    """True when the setting is relaxed and the text is made only of the pass-through words of ``key``."""
    ws = words(key)
    return bool(ws) and _made_of(normalize(text), ws)


def is_loop(text: str) -> bool:
    """True when the repetition rule finds a repeated word or phrase in the text, or the text is over its length
    limit (the same rule and limits as the clean-up's repetition step)."""
    global _repetition
    from whisperjav.config.sanitization_constants import RepetitionConstants
    constants = RepetitionConstants()
    if len(text or "") > constants.MAX_SUBTITLE_TEXT_LENGTH:
        return True                    # decided here, so the rule's own truncation warning is not printed for it
    if _repetition is None:
        from whisperjav.modules.repetition_cleaner import RepetitionCleaner
        _repetition = RepetitionCleaner(constants)
    _, findings = _repetition.clean_repetitions(text)
    return any(f.get("type") in _LOOP_FINDINGS for f in findings)


def spares(text: str, key: str = JAPANESE) -> bool:
    """Relaxed: True when a step that would remove this line keeps it instead: the line is made only of the
    pass-through words of ``key`` and is not a loop. Always False in strict."""
    return is_words(text, key) and not is_loop(text)


def note(step: str) -> None:
    """A step kept one line for its pass-through words (counted for the file's log line)."""
    _kept[step] += 1


def reset_file() -> None:
    """A new file begins: the counts start at 0."""
    _kept.clear()


def report_file() -> None:
    """Relaxed: the file's one log line, with the lines each step kept for the pass-through words; the counts are
    then reset. Strict: nothing is said."""
    if relaxed():
        _load()                         # a refused file is said once, at its reading; no per-file line then
    if relaxed() and not _refused:
        if _kept:
            detail = ", ".join(f"{step} {n}" for step, n in sorted(_kept.items(), key=lambda kv: (-kv[1], kv[0])))
        else:
            detail = "none"
        logger.info(f"{_name(relaxed_tag=True)}: lines kept for their pass-through words: {detail}")
    _kept.clear()


def _reset_for_tests() -> None:
    """Forget the loaded words, counts and pass (tests change the setting and the file within one process)."""
    global _words, _refused, _pass
    _words = None
    _refused = False
    _pass = None
    _kept.clear()
