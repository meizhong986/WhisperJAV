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
    
    # Character repetition patterns
    #CHAR_REPETITION_PATTERN: str = r'(([あアぁァいイぃィうウぅゥえエぇェおオぉォはハぱパんンっッ])\2{{{threshold},}})'
    # CORRECTED: Use the full Unicode range for Hiragana and Katakana.
    CHAR_REPETITION_PATTERN: str = r'(([ぁ-んァ-ン])\2{{{threshold},}})'
    VOICED_CHAR_PATTERN: str = r'(([あアいイうウえエおオ][゛゙])\2{{{threshold},}})'
    
    # Legitimate repetitions to preserve
    LEGITIMATE_REPETITIONS: Set[str] = field(default_factory=lambda: {
        "そこそこ", "まあまあ", "どきどき", "わくわく", "いらいら",
        "うきうき", "きらきら", "パンパン", "ドキドキ", "バンバン"
    })
    
    # Complex pattern detection - FIXED: Properly escaped patterns
    COMMA_PATTERN: str = r'(([あアぁァいイぃィうウぅゥえエぇェおオぉォはハぱパんン]、){{{threshold_plus_10},}})'
    MULTI_CHAR_COMMA_PATTERN: str = r'((\p{{Hiragana}}{{2,5}}、){{{threshold_plus_2},}})'
    COMPLEX_PHRASE_PATTERN: str = r'(([\p{{Hiragana}}\p{{Katakana}}]{{3,10}}、){{{threshold_plus_2},}})'
    
    VOWEL_EXTENSION_PATTERN: str = r'([あアぁァいイぃィうウぅゥえエぇェおオぉォはハぱパ])[〜～ー]{{{threshold_plus_5},}}'
    STANDALONE_EXTENSION_PATTERN: str = r'[〜～ー]{{{threshold_plus_10},}}'
    DAKUTEN_PATTERN: str = r'([あアいイうウえエおオ][゛゙]){{{threshold_plus_5},}}'
       
    
       
    # High density thresholds
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

@dataclass
class TimingConstants:
    """Constants for timing adjustments"""
    MIN_SUBTITLE_DURATION: float = 0.5
    MAX_SUBTITLE_DURATION: float = 12.0
    DEFAULT_DURATION: float = 2.0
    MIN_READING_TIME_RATIO: float = 0.8
    IDEAL_DURATION_BUFFER: float = 1.0  # Min duration in seconds

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