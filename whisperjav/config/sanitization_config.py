# whisperjav/config/sanitization_config.py

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import os

@dataclass
class SanitizationConfig:
    """Main configuration for subtitle sanitization"""
    
    # Sensitivity/aggressiveness
    sensitivity_mode: str = "balanced"  # conservative/balanced/aggressive
    custom_aggressiveness: Optional[float] = None  # Override preset
    
    # Feature toggles
    enable_exact_matching: bool = True
    enable_regex_matching: bool = True
    enable_fuzzy_matching: bool = True
    enable_repetition_cleaning: bool = True
    enable_cross_subtitle: bool = True
    
    # Customizable thresholds
    repetition_threshold: Optional[int] = None
    fuzzy_match_threshold: Optional[float] = None
    merge_similarity_threshold: Optional[float] = None
    max_gap_ms: Optional[int] = None
    min_subtitle_duration: Optional[float] = None
    max_subtitle_duration: Optional[float] = None
    
    # Pattern sources
    hallucination_exact_list_url: Optional[str] = None
    hallucination_regex_patterns_url: Optional[str] = None
    user_blacklist_patterns: List[str] = field(default_factory=list)
    
    # Language settings
    primary_language: Optional[str] = None  # For language-specific processing
    language_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Output options
    save_original: bool = True
    save_artifacts: bool = True
    preserve_original_file: bool = False
    artifact_detail_level: str = "full"  # full/summary/minimal
    create_raw_subs_folder: bool = True
    
    # Processing options
    verbose: bool = False
    debug: bool = False
    exit_on_error: bool = False
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'SanitizationConfig':
        """Load configuration from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_file(self, config_path: Path):
        """Save configuration to JSON file"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    def get_effective_constants(self) -> Dict[str, Any]:
        """Get constants with overrides applied"""
        from .sanitization_constants import (
            HallucinationConstants, RepetitionConstants,
            CrossSubtitleConstants, TimingConstants, ProcessingConstants
        )
        
        # Start with defaults
        constants = {
            'hallucination': HallucinationConstants(),
            'repetition': RepetitionConstants(),
            'cross_subtitle': CrossSubtitleConstants(),
            'timing': TimingConstants(),
            'processing': ProcessingConstants()
        }
        
        # Apply aggressiveness scaling
        aggressiveness = self.custom_aggressiveness or \
                        constants['processing'].AGGRESSIVENESS_PRESETS.get(
                            self.sensitivity_mode, 1.0)
        
        # Scale thresholds based on aggressiveness
        if self.repetition_threshold is None:
            constants['repetition'].DEFAULT_THRESHOLD = max(
                constants['repetition'].MIN_THRESHOLD,
                int(constants['repetition'].DEFAULT_THRESHOLD * (2.0 - aggressiveness))
            )
        else:
            constants['repetition'].DEFAULT_THRESHOLD = self.repetition_threshold
            
        if self.fuzzy_match_threshold is None:
            constants['hallucination'].FUZZY_MATCH_THRESHOLD = min(
                1.0, constants['hallucination'].FUZZY_MATCH_THRESHOLD * aggressiveness
            )
        else:
            constants['hallucination'].FUZZY_MATCH_THRESHOLD = self.fuzzy_match_threshold
        
        # Apply other overrides
        if self.merge_similarity_threshold is not None:
            constants['cross_subtitle'].MERGE_SIMILARITY_THRESHOLD = self.merge_similarity_threshold
        if self.max_gap_ms is not None:
            constants['cross_subtitle'].MAX_GAP_MS = self.max_gap_ms
        if self.min_subtitle_duration is not None:
            constants['timing'].MIN_SUBTITLE_DURATION = self.min_subtitle_duration
        if self.max_subtitle_duration is not None:
            constants['timing'].MAX_SUBTITLE_DURATION = self.max_subtitle_duration
            
        # Apply pattern URL overrides
        if self.hallucination_exact_list_url:
            constants['hallucination'].FILTER_LIST_URL = self.hallucination_exact_list_url
        if self.hallucination_regex_patterns_url:
            constants['hallucination'].EXACT_LIST_URL = self.hallucination_regex_patterns_url
            
        # Apply language-specific overrides
        if self.primary_language and self.primary_language in self.language_overrides:
            lang_overrides = self.language_overrides[self.primary_language]
            if 'repetition_threshold' in lang_overrides:
                constants['repetition'].DEFAULT_THRESHOLD = lang_overrides['repetition_threshold']
            if 'chars_per_second' in lang_overrides:
                constants['cross_subtitle'].CHARS_PER_SECOND[self.primary_language] = lang_overrides['chars_per_second']
            if 'legitimate_repetitions' in lang_overrides:
                constants['repetition'].LEGITIMATE_REPETITIONS.update(lang_overrides['legitimate_repetitions'])
                
        return constants

    @classmethod
    def from_env(cls) -> 'SanitizationConfig':
        """Create config from environment variables"""
        config = cls()
        
        # String values
        for key in ['sensitivity_mode', 'artifact_detail_level', 'primary_language']:
            env_val = os.environ.get(f'WHISPERJAV_SANITIZE_{key.upper()}')
            if env_val:
                setattr(config, key, env_val)
        
        # Boolean values
        for key in ['enable_exact_matching', 'enable_regex_matching', 'enable_fuzzy_matching',
                   'enable_repetition_cleaning', 'enable_cross_subtitle', 'save_original',
                   'save_artifacts', 'preserve_original_file', 'verbose', 'debug']:
            env_val = os.environ.get(f'WHISPERJAV_SANITIZE_{key.upper()}')
            if env_val:
                setattr(config, key, env_val.lower() in ('true', '1', 'yes', 'on'))
        
        # Numeric values
        for key, converter in [
            ('repetition_threshold', int),
            ('fuzzy_match_threshold', float),
            ('merge_similarity_threshold', float),
            ('max_gap_ms', int),
            ('custom_aggressiveness', float)
        ]:
            env_val = os.environ.get(f'WHISPERJAV_SANITIZE_{key.upper()}')
            if env_val:
                try:
                    setattr(config, key, converter(env_val))
                except ValueError:
                    pass
                    
        return config

def get_testing_profile(profile_name: str) -> SanitizationConfig:
    """Get predefined testing profiles"""
    
    profiles = {
        "minimal": SanitizationConfig(
            sensitivity_mode="conservative",
            enable_fuzzy_matching=False,
            enable_cross_subtitle=False,
            artifact_detail_level="minimal"
        ),
        
        "strict": SanitizationConfig(
            sensitivity_mode="aggressive",
            repetition_threshold=1,
            fuzzy_match_threshold=0.8,
            merge_similarity_threshold=0.85,
            artifact_detail_level="full"
        ),
        
        "japanese_optimized": SanitizationConfig(
            sensitivity_mode="balanced",
            primary_language="ja",
            language_overrides={
                "ja": {
                    "repetition_threshold": 2,
                    "chars_per_second": 7.0,
                    "legitimate_repetitions": [
                        "そこそこ", "まあまあ", "どきどき", "わくわく"
                    ]
                }
            }
        ),
        
        "debug": SanitizationConfig(
            sensitivity_mode="balanced",
            artifact_detail_level="full",
            save_original=True,
            save_artifacts=True,
            verbose=True,
            debug=True,
            enable_exact_matching=True,
            enable_regex_matching=True,
            enable_fuzzy_matching=True,
            enable_repetition_cleaning=True,
            enable_cross_subtitle=True
        )
    }
    
    return profiles.get(profile_name, SanitizationConfig())