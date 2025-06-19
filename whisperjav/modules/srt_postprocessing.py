from typing import Union, Optional, Tuple, Dict
from pathlib import Path

from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer
from whisperjav.config.sanitization_config import SanitizationConfig

class SRTPostProcessor:
    """Legacy post-processor - redirects to new SubtitleSanitizer"""
    
    def __init__(self, **kwargs):
        # Map old parameters to new config
        config = SanitizationConfig(
            enable_exact_matching=kwargs.get('remove_hallucinations', True),
            enable_repetition_cleaning=kwargs.get('remove_repetitions', True),
            repetition_threshold=kwargs.get('repetition_threshold', 2),
            min_subtitle_duration=kwargs.get('min_subtitle_duration', 0.5),
            max_subtitle_duration=kwargs.get('max_subtitle_duration', 7.0)
        )
        
        # Use new sanitizer
        self.sanitizer = SubtitleSanitizer(config)
        
    def process(self, srt_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Tuple[Path, Dict]:
        """Process SRT file using new sanitizer system"""
        srt_path = Path(srt_path)
        
        # Configure to match old behavior
        self.sanitizer.config.preserve_original_file = output_path is not None
        self.sanitizer.config.save_original = True
        self.sanitizer.config.save_artifacts = True
        
        # Process
        result = self.sanitizer.process(srt_path)
        
        # Return in old format
        stats = result.statistics
        old_stats = {
            'total_subtitles': stats['original_subtitle_count'],
            'removed_hallucinations': stats['modifications_by_category'].get('hallucination', 0),
            'removed_repetitions': stats['modifications_by_category'].get('repetition', 0),
            'duration_adjustments': stats['modifications_by_category'].get('timing', 0),
            'empty_removed': stats['removals']
        }
        
        return result.sanitized_path, old_stats