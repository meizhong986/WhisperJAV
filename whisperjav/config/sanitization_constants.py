# whisperjav/config/sanitization_constants.py

from dataclasses import dataclass, field
from typing import Dict, List, Set

@dataclass
class HallucinationConstants:
    """Constants for hallucination detection"""
    EXACT_LIST_URL: str = "https://gist.githubusercontent.com/meizhong986/ecca22c8ddb9dcab4f6df7813c275a00/raw/WhisperJAV_hallucination_regexp_v09.json"
    FILTER_LIST_URL: str = "https://gist.githubusercontent.com/meizhong986/4882bdb3f4f5aa4034a112cebd2e0845/raw/9e78020b9f85cb7aa3d7004d477353adbfe60ee9/WhisperJAV_hallucination_filter_sorted_v08.json"
    FUZZY_MATCH_THRESHOLD: float = 0.9
    MIN_CONFIDENCE_THRESHOLD: float = 0.5



@dataclass
class RepetitionConstants:
    """Constants for repetition detection"""
    DEFAULT_THRESHOLD: int = 2
    MIN_THRESHOLD: int = 1
    MAX_THRESHOLD: int = 800
    
    # --- Specific, configurable thresholds  ---
    MIN_CHAR_REPETITION_THRESHOLD: int = 3 # Total occurrences (e.g., ううう)
    MIN_PHRASE_REPETITION_THRESHOLD: int = 3 # Total occurrences (e.g., すごい、すごい、すごい、)
    # ---------------------------------------------------------

    # These base patterns are used to dynamically build the full regex in the cleaner module.
    # CORRECTED: Simplified to a single capture group for robustness.
    CHAR_REPETITION_PATTERN: str = r'([ぁ-んァ-ン])' 

    # Base pattern for a phrase + comma.
    COMMA_PHRASE_PATTERN: str = r'((?:[^,、]{1,10}[,、]\s*))'

    # Base pattern for a word + optional space.
    WORD_REPETITION_PATTERN: str = r'((\p{{L}}{2,})\s*)'

    # These patterns are complete and do not need dynamic thresholds.
    PREFIX_PLUS_CHAR_FLOOD_PATTERN: str = r'(\p{{L}}{1,3})([ぁ-んァ-ン])\2{5,}'
    SMALL_VOWEL_FLOOD_PATTERN: str = r'([ぁ-んァ-ン])([ぁぃぅぇぉァィゥェォ])\2{3,}'
    VOWEL_EXTENSION_PATTERN: str = r'([ぁ-んァ-ン])([〜ー])'
    
    LEGITIMATE_REPETITIONS: Set[str] = field(default_factory=lambda: {
        "そこそこ", "まあまあ", "どきどき", "わくわく", "いらいら",
        "うきうき", "きらきら", "パンパン", "ドキドキ", "バンバン"
    })
        
    HIGH_DENSITY_MIN_LENGTH: int = 30
    HIGH_DENSITY_MIN_OCCURRENCES: int = 5
    HIGH_DENSITY_RATIO: float = 0.3

@dataclass
class CrossSubtitleConstants:
    """Constants for cross-subtitle processing"""
    MERGE_SIMILARITY_THRESHOLD: float = 0.9
    MAX_GAP_MS: int = 600
    MIN_GAP_MS: int = 0
    FIT_TOLERANCE_S: float = 0.1
    
    # Reading speed by language
    CHARS_PER_SECOND: Dict[str, float] = field(default_factory=lambda: {
        "ja": 7.0,
        "zh": 6.0,
        "ko": 6.5,
        "en": 15.0,
        "default": 10.0
    })
    
    DEDUP_THRESHOLD: int = 3  # Minimum consecutive similar subtitles before merging

# In whisperjav/config/sanitization_constants.py

@dataclass
class TimingConstants:
    """Constants for timing adjustments"""
    MIN_SUBTITLE_DURATION: float = 0.3
    MAX_SUBTITLE_DURATION: float = 12.0
    DEFAULT_DURATION: float = 2.0
    
    
    # The minimum reading speed in Characters Per Second.
    # Subtitles below this will be sent to the TimingAdjuster for recalculation.
    MIN_SAFE_CPS: float = 1.0 

    # The maximum reading speed in Characters Per Second.
    # Subtitles above this will be completely removed as artifacts.
    MAX_SAFE_CPS: float = 20.0

    # The minimum number of characters a subtitle needs to have to be checked.
    MIN_TEXT_LENGTH_FOR_CPS_CHECK: int = 3
    # ----------------------------------------------------

    MIN_READING_TIME_RATIO: float = 0.8
    IDEAL_DURATION_BUFFER: float = 1.0

@dataclass
class ProcessingConstants:
    """General processing constants"""
    # Aggressiveness presets
    AGGRESSIVENESS_PRESETS: Dict[str, float] = field(default_factory=lambda: {
        "conservative": 0.7,
        "balanced": 1.0,
        "aggressive": 1.3
    })
    
    # File naming
    SANITIZED_SUFFIX: str = "sanitized"
    ORIGINAL_BACKUP_SUFFIX: str = "original"
    ARTIFACTS_SUFFIX: str = "artifacts"
    RAW_SUBS_FOLDER: str = "raw_subs"
    
    # Processing limits
    MAX_FILE_SIZE_MB: int = 100
    MAX_SUBTITLES: int = 50000
    
    # Artifact detail levels
    ARTIFACT_DETAIL_LEVELS: Set[str] = field(default_factory=lambda: {"full", "summary", "minimal"})