from typing import Union, Optional, Tuple, Dict
from pathlib import Path
import shutil

from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer
from whisperjav.modules.subtitle_sanitizer_english import EnglishSubtitleCleaner
from whisperjav.config.sanitization_config import SanitizationConfig
from whisperjav.utils.logger import logger


def normalize_language_code(language: str) -> str:
    """
    Normalize language names/codes to ISO 639-1 codes.

    This function handles the common case where ASR models return full language
    names (e.g., "Japanese") but downstream components expect ISO codes ("ja").

    JP-004 fix: Qwen-ASR returns "Japanese" but SRTPostProcessor expects "ja".

    Args:
        language: Language name or code (e.g., "Japanese", "ja", "english", "en")

    Returns:
        Normalized ISO 639-1 language code (e.g., "ja", "en", "zh")

    Examples:
        >>> normalize_language_code("Japanese")
        'ja'
        >>> normalize_language_code("ja")
        'ja'
        >>> normalize_language_code("ENGLISH")
        'en'
    """
    if not language:
        return 'ja'  # Default to Japanese for this project

    # Mapping of full names and variants to ISO codes
    language_map = {
        # Japanese
        'japanese': 'ja',
        'jpn': 'ja',
        'jp': 'ja',
        # English
        'english': 'en',
        'eng': 'en',
        # Chinese
        'chinese': 'zh',
        'mandarin': 'zh',
        'zho': 'zh',
        'cmn': 'zh',
        # Cantonese
        'cantonese': 'yue',
        # Korean
        'korean': 'ko',
        'kor': 'ko',
        # Other common languages
        'french': 'fr',
        'fra': 'fr',
        'german': 'de',
        'deu': 'de',
        'italian': 'it',
        'ita': 'it',
        'spanish': 'es',
        'spa': 'es',
        'portuguese': 'pt',
        'por': 'pt',
        'russian': 'ru',
        'rus': 'ru',
    }

    lang_lower = language.lower().strip()

    # Check if it's already a 2-letter code
    if len(lang_lower) == 2:
        return lang_lower

    # Look up in mapping
    if lang_lower in language_map:
        return language_map[lang_lower]

    # If it's a 3-letter code or unknown, try to extract first 2 chars
    # But only if it looks like a valid code (lowercase letters only)
    if len(lang_lower) >= 2 and lang_lower[:2].isalpha():
        return lang_lower[:2]

    # Fallback: return as-is (let downstream handle it)
    logger.debug(f"Unknown language '{language}', returning as-is")
    return language


class SRTPostProcessor:
    """Post-processor that routes to appropriate language-specific sanitizer"""

    def __init__(self, language: str = 'ja', **kwargs):
        """
        Initialize post-processor with language selection.

        Args:
            language: Language code ('ja' for Japanese, 'en' for English)
                     Also accepts full names like 'Japanese', 'English' (JP-004 fix)
            **kwargs: Additional parameters passed to sanitizers
        """
        # Normalize language code (JP-004 fix)
        # Accepts both ISO codes ('ja') and full names ('Japanese')
        self.language = normalize_language_code(language)
        self.config = kwargs

        logger.debug(f"SRTPostProcessor: normalized language '{language}' -> '{self.language}'")

        # For CJK languages (Japanese, Korean, Chinese), create sanitizer once
        # Korean and Chinese use the same sanitizer - Japanese-specific patterns
        # simply won't match Hangul/Han characters and pass through unchanged
        if language in ('ja', 'ko', 'zh'):
            config = SanitizationConfig(
                enable_exact_matching=kwargs.get('remove_hallucinations', True),
                enable_repetition_cleaning=kwargs.get('remove_repetitions', True),
                repetition_threshold=kwargs.get('repetition_threshold', 2),
                min_subtitle_duration=kwargs.get('min_subtitle_duration', 0.5),
                max_subtitle_duration=kwargs.get('max_subtitle_duration', 7.0),
                primary_language=language  # Pass language for hallucination filter
            )
            self.cjk_sanitizer = SubtitleSanitizer(config)
            logger.debug(f"Initialized CJK subtitle sanitizer for language: {language}")
        else:
            # For English and other languages, we'll create cleaner per file
            logger.debug(f"Configured for English subtitle cleaning (language: {language})")
            
    def process(self, srt_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Tuple[Path, Dict]:
        """
        Process SRT file using appropriate language-specific sanitizer.
        
        Args:
            srt_path: Path to input SRT file
            output_path: Optional output path (used for determining target directory)
            
        Returns:
            Tuple of (processed_file_path, statistics_dict)
        """
        srt_path = Path(srt_path)

        if self.language in ('ja', 'ko', 'zh'):
            return self._process_cjk(srt_path, output_path)
        elif self.language == 'en':
            return self._process_english(srt_path, output_path)
        else:
            logger.warning(f"Unknown language '{self.language}', defaulting to English sanitizer")
            return self._process_english(srt_path, output_path)
    
    
    def _process_cjk(self, srt_path: Path, output_path: Optional[Path]) -> Tuple[Path, Dict]:
        """Process using CJK sanitizer (Japanese, Korean, Chinese)"""
        # Configure to match old behavior
        self.cjk_sanitizer.config.preserve_original_file = output_path is not None
        self.cjk_sanitizer.config.save_original = True
        self.cjk_sanitizer.config.save_artifacts = True

        # Process
        result = self.cjk_sanitizer.process(srt_path)
        
        # Return in expected format
        stats = result.statistics
        
        # Handle empty statistics (when no subtitles found) - THIS IS THE ONLY CHANGE
        if not stats:
            old_stats = {
                'total_subtitles': 0,
                'removed_hallucinations': 0,
                'removed_repetitions': 0,
                'duration_adjustments': 0,
                'empty_removed': 0
            }
        else:
            old_stats = {
                'total_subtitles': stats.get('original_subtitle_count', 0),
                'removed_hallucinations': stats.get('modifications_by_category', {}).get('hallucination', 0),
                'removed_repetitions': stats.get('modifications_by_category', {}).get('repetition_cleaning', 0),
                'duration_adjustments': stats.get('modifications_by_category', {}).get('timing_adjustment', 0),
                'cps_filtered': stats.get('modifications_by_category', {}).get('content_cleaning_cps', 0),
                'empty_removed': stats.get('removals', 0)
            }
        
        return result.sanitized_path, old_stats
        
    
    def _process_english(self, srt_path: Path, output_path: Optional[Path]) -> Tuple[Path, Dict]:
        """Process using English sanitizer"""
        # Determine target directory
        if output_path:
            target_dir = output_path.parent
            final_name = output_path.name
        else:
            target_dir = srt_path.parent
            final_name = srt_path.name
        
        # Create temporary working directory for EnglishSubtitleCleaner
        temp_dir = target_dir / "temp_english_clean"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Initialize English cleaner with extracted parameters
            cleaner = EnglishSubtitleCleaner(
                source_file=str(srt_path),
                target_dir=str(temp_dir),
                hallucination_list_url=self.config.get('hallucination_list_url', 
                    "https://gist.githubusercontent.com/meizhong986/4882bdb3f4f5aa4034a112cebd2e0845/raw/9e78020b9f85cb7aa3d7004d477353adbfe60ee9/WhisperJAV_hallucination_filter_sorted_v08.json"),
                cps_slow_threshold=self.config.get('cps_slow_threshold', 6.0),
                cps_fast_threshold=self.config.get('cps_fast_threshold', 60.22),
                max_merge_gap_sec=self.config.get('max_merge_gap_sec', 0.4),
                min_duration=self.config.get('min_subtitle_duration', 0.5),
                max_duration=self.config.get('max_subtitle_duration', 8.0)
            )
            
            # Process
            clean_path, log_path = cleaner.clean()
            
            # Move cleaned file to final destination
            final_clean_path = target_dir / final_name
            shutil.move(clean_path, final_clean_path)
            
            # Move log file to raw_subs folder
            raw_subs_dir = target_dir / "raw_subs"
            raw_subs_dir.mkdir(exist_ok=True)
            
            # Copy original to raw_subs as backup
            original_backup = raw_subs_dir / f"{srt_path.stem}.original{srt_path.suffix}"
            shutil.copy2(srt_path, original_backup)
            
            # Move log file to raw_subs
            log_name = Path(log_path).name
            final_log_path = raw_subs_dir / log_name
            shutil.move(log_path, final_log_path)
            
            # Create statistics (approximate based on log entries)
            # Since EnglishSubtitleCleaner doesn't return stats, we'll provide basic ones
            stats = {
                'total_subtitles': len(cleaner.subs) if hasattr(cleaner, 'subs') else 0,
                'removed_hallucinations': 0,  # Would need to parse log for exact count
                'removed_repetitions': 0,
                'duration_adjustments': 0,
                'empty_removed': 0
            }
            
            logger.debug(f"English subtitle cleaning complete: {final_clean_path}")
            logger.debug(f"Log saved to: {final_log_path}")
            
            return final_clean_path, stats
            
        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)