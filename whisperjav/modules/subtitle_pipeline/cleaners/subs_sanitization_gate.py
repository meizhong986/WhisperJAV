"""Subtitle sanitization gate (1.10.3, owner W753 CL2, W754; default strict, W756 5; named W772).

One setting decides how strictly subtitle sanitization removes lines that carry words:
  strict   today's behaviour, unchanged (the default);
  relaxed  a line that holds one of RELAXED_TERMS is kept where the hallucination filter or the non-linguistic
           filter would remove it.
Lines of sounds only stay removed in both states, and so do the lone tokens of the non-verbal filter (あ, え, ん, ...,
a lone はい or うん: owner CFF5 and the standing decision of 2026-09-26), which this setting does not reach.

Where it acts: every pipeline's clean-up goes through these filters --
  hallucination filter (modules/hallucination_remover.py remove_hallucinations, before all its stages, the bracket
    and emoji ones included, so 「気持ちいい♡」 is kept too): balanced, fast, faster, fidelity, transformers and the GGUF
    pipelines but MOSS-CrispASR (crispasr/cleanup.py SRT_STAGES);
  the hallucination list in the assembly cleaner (modules/assembly_text_cleaner.py, stage 3): ChronosJAV with
    Qwen3-ASR, and Cohere-CrispASR's text cleaner;
  non-linguistic filter (cleaners/nonlinguistic_utterance_filter.py): the same pipelines' post-processing, ChronosJAV's
    Phase 8, the GGUF pipelines' clean-up (MOSS-CrispASR included), and the orchestrator's check that a split leaves no
    part Phase 8 would remove (orchestrator.py line_filter_test), so split and clean-up agree.
Decoupled runs no post-processing; its text cleaner (the assembly cleaner, by default) and the split check follow the
setting. XXL has no WhisperJAV clean-up. Not reached: the legacy sanitizer's reading-speed (CPS) rule, the ASR-time
suppress list of the Whisper modules, the English-output cleaner.

The setting reaches every process of a run through the environment, as --fail-on suspect does: main.py sets ENV from
--subs-sanitization-gate, and the Ensemble's pass processes inherit it.

RELAXED_TERMS is a first list (W769: Claude's proposal, for the owner to settle, W754 "we shall work together to
define the relaxed terms"): his examples (kimochi 気持ちいい, iku iku イクイク) and the lines with words that
ChronosJAV's Phase 8 removed in his anime-whisper test of 2026-11-17 (じゃあ, こっち, すご..., いっちゃ...), plus やば
from the hallucination list. See holds_relaxed_term for where in a line a term counts.
"""
import os

ENV = "WHISPERJAV_SUBS_SANITIZATION_GATE"
CHOICES = ("strict", "relaxed")
DEFAULT = "strict"

RELAXED_TERMS = (
    # kimochi (feels good)
    "気持ちいい", "きもちいい", "気持ち良い", "気持ちい", "きもちい", "キモチイイ",
    # iku (coming)
    "イク", "いく", "イッちゃ", "いっちゃ", "イキそう", "いきそう", "イっちゃ",
    # dialogue words Phase 8 removed in the owner's anime-whisper test (2026-11-17); やば from the hallucination list
    "じゃあ", "こっち", "そっち", "あっち", "すご", "やば",
)


def current() -> str:
    """The run's setting: 'strict' unless the environment says 'relaxed'."""
    value = (os.environ.get(ENV) or DEFAULT).strip().lower()
    return value if value in CHOICES else DEFAULT


def _chunks(text: str) -> list:
    """The line cut at every punctuation mark and space (PUNCTUATION_CHARS of the non-linguistic filter)."""
    from whisperjav.modules.subtitle_pipeline.cleaners.nonlinguistic_utterance_filter import PUNCTUATION_CHARS
    chunks, current = [], []
    for ch in text or "":
        if ch in PUNCTUATION_CHARS or ch.isspace():
            if current:
                chunks.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        chunks.append("".join(current))
    return chunks


def holds_relaxed_term(text: str) -> bool:
    """True when the line starts with one of RELAXED_TERMS, or has one right after a punctuation mark or a space
    (whatever the setting). A term inside a run of other kana does not count, so a loop such as くいくい (which holds
    いく) is not taken for words (review of 2026-11-24); イクイク, い、いっちゃ…, あ、こっち。 do count."""
    return any(chunk.startswith(term) for chunk in _chunks(text) for term in RELAXED_TERMS)


def relaxed_keeps(text: str) -> bool:
    """True when the setting is relaxed and the line holds one of RELAXED_TERMS: the filter keeps the line."""
    return current() == "relaxed" and holds_relaxed_term(text)
