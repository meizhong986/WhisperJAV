"""Non-linguistic utterance filter for Japanese SRT output (v1.9.0).

Drops subtitle entries whose Japanese text consists solely of sound-effect
kana (moaning, breathing, sucking, slurping, etc.) with no evidence of real
dialogue.  Applied as the final content-level filter in both legacy pipelines
(via SRTPostProcessor) and Qwen pipeline (Phase 8).

Two-layer approach on each Japanese line:
  1. EVIDENCE CHECK — contains kanji, particles, verb endings, or common
     words?  If yes -> KEEP unconditionally.
  2. SOUND-KANA CHECK — are ALL remaining kana from a defined sound alphabet?
     If yes -> REMOVE.

Ported from the owner's proven standalone utility
(moan_noise_sound_remover_utility_v3.py), adapted to operate on pysrt
subtitle entries within the existing cleaner architecture.
"""

import re
from pathlib import Path
from typing import Dict, Union

import pysrt

from whisperjav.utils.logger import logger


# ── Character-set regexes ───────────────────────────────────────────────

_HIRAGANA = r"぀-ゟ"
_KATAKANA = r"゠-ヿ"
_KANJI = r"一-鿿"
_CJK_EXT = r"㐀-䶿豈-﫿"

_JP_RE = re.compile(f"[{_HIRAGANA}{_KATAKANA}]")
_CJK_RE = re.compile(f"[{_KANJI}{_CJK_EXT}]")


# ── Sound-kana alphabet ────────────────────────────────────────────────
#
# Characters that sound-effect / vocalization lines are composed of.
# After confirming NO linguistic evidence, every kana in the line must
# belong to this set for the line to be classified as non-linguistic.

SOUND_KANA: set = set(
    # Vowels & small vowels (moaning / breathing core)
    "あぁいぃうぅえぇおぉ"
    "アァイィウゥエェオォ"
    # Nasal / geminate / prolongation
    "んンっッーゝゞ"
    # Breathing (はぁ, ふぅ, ひぃ)
    "はひふへほ"
    "ハヒフヘホ"
    # Kissing / sucking (ちゅ, ぢゅ, つ)
    "ちぢつづ"
    "チヂツヅ"
    # Slurping / sighing (ずず, すぅ)
    "すず"
    "スズ"
    # Swallowing (ごくっ, くちゅ)
    "くぐこご"
    "クグコゴ"
    # Licking (ぺろ, れろ, べろ)
    "ぺべれめ"
    "ペベレメ"
    # Popping / puffing (ちゅぱ, ぷはぁ)
    "ぷぱぶば"
    "プパブバ"
    # Nasal closure (んむ, んぬ)
    "むぬ"
    "ムヌ"
    # Rolling / liquid (ちゅる, れろ)
    "ろる"
    "ロル"
    # Voiced fricative (じゅる)
    "じ"
    "ジ"
    # Small ya/yu/yo (ちゅ, じゃ)
    "ゃゅょ"
    "ャュョ"
)


# ── Punctuation (stripped before the sound-kana check) ──────────────────

PUNCTUATION_CHARS: set = set(
    " \t　"
    "、。，．,.!?！？"
    "…‥・~〜"
    "「」『』【】()（）<>〈〉《》"
    "\"'“”‘’«»"
    "─—–-"
    ":;：；"
    "♪♡♥★☆●○◎"
    "*＊"
)


# ── Language evidence patterns ──────────────────────────────────────────

_MULTI_CHAR_PARTICLES = [
    "から", "まで", "より", "ほど", "くらい", "ぐらい",
    "だけ", "しか", "ばかり", "なんて", "なんか", "って",
    "では", "には", "とは", "とも", "への", "での",
]

_SINGLE_CHAR_PARTICLES = list("をがでとものにへ")

_VERB_ENDINGS = [
    "ます", "ません", "ました", "ましょう",
    "です", "でした", "でしょう",
    "ない", "なかった", "なくて",
    "たい", "たくない", "たがる",
    "ちゃう", "じゃう", "てしまう", "でしまう",
    "てる", "でる", "ている", "でいる",
    "ちゃった", "じゃった",
    "なきゃ", "なくちゃ", "なければ",
    "られる", "させる", "される",
    "だろう", "だった",
    "ている", "ておく", "てあげる", "てくれる", "てもらう",
]

_QUESTION_MARKERS = [
    "の？", "のか", "ですか", "ますか", "ましたか",
    "だっけ", "っけ", "かな", "かしら",
]

_COMMON_WORDS = [
    "お願い", "すみません", "ごめん", "ありがとう", "どうぞ",
    "はい", "いいえ", "こんにちは", "こんばんは", "おはよう",
    "あれ", "これ", "それ", "どれ",
    "ここ", "そこ", "あそこ", "どこ",
    "こう", "そう", "ああ", "どう",
    "なん", "なに", "だれ", "いつ", "なぜ", "どうして",
    "私", "わたし", "あたし", "あなた", "おまえ", "お前",
    "君", "きみ", "彼", "彼女", "みんな", "皆",
    "僕", "ぼく", "俺", "おれ", "あいつ", "こいつ",
    "先生", "せんせい", "先輩", "せんぱい",
    "お兄", "お姉", "おにい", "おねえ",
    "いい", "よい", "悪い", "わるい",
    "やばい", "すごい", "凄い",
    "可愛い", "かわいい", "綺麗", "きれい",
    "美しい", "うつくしい", "素晴らしい",
    "ひどい", "おかしい", "嬉しい", "うれしい",
    "痛い", "いたい", "熱い", "あつい",
    "冷たい", "つめたい", "温かい", "あたたかい",
    "気持ちいい", "きもちいい",
    "大きい", "おおきい", "小さい", "ちいさい",
    "欲しい", "ほしい", "怖い", "こわい",
    "好き", "嫌い", "きらい", "大丈夫",
    "ずるい", "ふるい", "ぬるい", "えぐい",
    # False-positive guards: all-SOUND_KANA verbs (dictionary form)
    "ふる", "ぬる", "ぬぐ", "つぐ", "ふく", "つく", "むく",
    "くぐる", "ふれる", "つれる", "くれる", "ぐれる",
    "ふるえる", "ふくれる", "くずれる", "つぶれる",
    "うごめく", "くずす", "ふくむ", "ふるう", "つぶす",
    # False-positive guards: all-SOUND_KANA nouns
    "ちち", "うち", "おく", "こえ", "ふじ", "おふろ", "ふくろ",
    "ちず", "こぶ", "こめ", "おこめ", "くず", "ふち", "つち",
    "いぬ", "ちえ", "ぬい", "えこ", "すじ", "うず",
    "こじ", "ふぐ", "めす", "つえ", "ふるえ",
    # False-positive guards: all-SOUND_KANA adverbs
    "すぐ", "ぐっ", "じっ",
    # False-positive guards: all-SOUND_KANA mimetics used in dialogue
    "めちゃめちゃ", "ごちゃごちゃ", "ぐちゃぐちゃ", "ごろごろ", "ぐるぐる",
    "ぶつぶつ", "ぷちぷち", "じめじめ", "ぬめぬめ",
    "ぷるぷる", "ぶるぶる",
    "おちんちん", "ちんちん", "おまんこ", "まんこ",
    "おっぱい", "乳首", "クリトリス",
    "精液", "中", "奥", "膣",
    "行く", "いく", "来る", "くる", "帰る", "かえる",
    "見る", "みる", "聞く", "きく", "言う", "いう",
    "思う", "おもう", "知る", "しる", "分かる", "わかる",
    "できる", "ある", "いる",
    "入れる", "いれる", "出す", "だす",
    "舐める", "なめる", "吸う", "すう",
    "触る", "さわる", "動く", "うごく",
    "やる", "する", "なる",
    "止める", "やめる", "やめて",
    "見て", "来て", "して", "出して", "入れて",
    "待って", "まって",
    "もっと", "ちょっと", "すごく", "とても",
    "まだ", "もう", "ずっと", "やっぱり",
    "そして", "でも", "けど", "だから",
    "本当", "ほんとう", "ほんと",
    "ダメ", "だめ", "駄目",
    "嫌", "イヤ",
    "うそ", "嘘",
    "バカ", "ばか", "馬鹿",
    "変態", "へんたい",
    "何", "なに",
]

_DIALOGUE_INTERJECTIONS = [
    "えっ", "あっ", "おっ", "わっ", "やっ",
    "おい", "ねえ", "ちょっと", "まあ",
    "さあ", "うん", "ううん", "いや", "へえ",
    "よし", "よっし", "やった", "しまった",
    "なるほど",
]

_ALL_EVIDENCE = (
    _MULTI_CHAR_PARTICLES + _VERB_ENDINGS + _QUESTION_MARKERS
    + _COMMON_WORDS + _DIALOGUE_INTERJECTIONS
)
_EVIDENCE_RE = re.compile(
    "|".join(re.escape(w) for w in sorted(set(_ALL_EVIDENCE), key=len, reverse=True))
)

_SINGLE_PARTICLE_RE = re.compile(
    "[" + "".join(_SINGLE_CHAR_PARTICLES) + "]"
)

_SENTENCE_END_PARTICLE_RE = re.compile(r"[よねさわ](?=$|[、。！？…\s])")


# ── Helper functions ────────────────────────────────────────────────────

def _contains_japanese(text: str) -> bool:
    return bool(_JP_RE.search(text))


def _contains_kanji(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _has_ha_particle(text: str) -> bool:
    """Return True if は appears as a grammatical particle, not breathing."""
    ha_sound_followers = set("ぁあぃいぅうぇえぉおーっはひふへほ")
    for m in re.finditer("は", text):
        pos = m.end()
        if pos >= len(text):
            if pos >= 2:
                prev = text[pos - 2]
                if prev not in SOUND_KANA and prev not in PUNCTUATION_CHARS:
                    return True
            continue
        next_ch = text[pos]
        if next_ch in ha_sound_followers or next_ch in PUNCTUATION_CHARS:
            continue
        return True
    return False


def _has_evidence_of_language(text: str) -> bool:
    """Return True if the text shows any sign of real dialogue."""
    if _contains_kanji(text):
        return True
    if _EVIDENCE_RE.search(text):
        return True
    if _has_ha_particle(text):
        return True
    if _SINGLE_PARTICLE_RE.search(text):
        return True
    if re.search(r"[?？]", text):
        return True
    if _SENTENCE_END_PARTICLE_RE.search(text):
        return True
    kata_chars = [c for c in text if "ァ" <= c <= "ヶ"]
    if len(set(kata_chars)) >= 2:
        return True
    return False


def _is_japanese_sound_line(text: str) -> bool:
    """Return True if the line is purely sound-effect kana."""
    if not _contains_japanese(text):
        return False
    if _has_evidence_of_language(text):
        return False
    for ch in text:
        if ch in PUNCTUATION_CHARS:
            continue
        if ch in SOUND_KANA:
            continue
        return False
    return True


def _should_remove_entry(text: str) -> bool:
    """Decide whether a subtitle entry should be removed.

    For bilingual entries (Japanese + English lines), the decision is based
    solely on the Japanese line.  If the Japanese line is sound-only, the
    entire entry (including any translation) is removed.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    for line in lines:
        if _contains_japanese(line):
            return _is_japanese_sound_line(line)
    return False


# ── Filter class ────────────────────────────────────────────────────────

class NonlinguisticUtteranceFilter:
    """Drops subtitle entries consisting solely of non-linguistic vocal sounds."""

    @staticmethod
    def is_nonlinguistic(text: str) -> bool:
        """Return True if the subtitle text is a non-linguistic utterance."""
        return _should_remove_entry(text)

    def filter_srt_file(self, srt_path: Union[str, Path]) -> Dict[str, int]:
        """Drop non-linguistic entries in place, renumber, write back.

        Returns stats dict: original_count, dropped_nonlinguistic,
        dropped_empty, final_count.
        """
        path = Path(srt_path)
        stats = {
            "original_count": 0,
            "dropped_nonlinguistic": 0,
            "dropped_empty": 0,
            "final_count": 0,
        }
        if not path.exists() or path.stat().st_size == 0:
            return stats

        try:
            subs = pysrt.open(str(path), encoding="utf-8")
        except Exception as e:
            logger.warning(
                "[NonlinguisticUtteranceFilter] Failed to parse %s: %s", path, e
            )
            return stats

        stats["original_count"] = len(subs)
        kept: list = []
        for sub in subs:
            text = (sub.text or "").strip()
            if not text:
                stats["dropped_empty"] += 1
                continue
            if _should_remove_entry(text):
                stats["dropped_nonlinguistic"] += 1
                continue
            kept.append(sub)

        for new_idx, sub in enumerate(kept, start=1):
            sub.index = new_idx

        pysrt.SubRipFile(items=kept).save(str(path), encoding="utf-8")

        stats["final_count"] = len(kept)
        if stats["dropped_nonlinguistic"] or stats["dropped_empty"]:
            logger.info(
                "[NonlinguisticUtteranceFilter] %s: %d -> %d entries "
                "(-%d non-linguistic, -%d empty)",
                path.name,
                stats["original_count"],
                stats["final_count"],
                stats["dropped_nonlinguistic"],
                stats["dropped_empty"],
            )
        return stats
