# whisperjav/modules/hallucination_remover.py
#V12


import re
import json
import requests
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from difflib import SequenceMatcher
from whisperjav.utils.logger import logger
from whisperjav.config.sanitization_constants import HallucinationConstants

class HallucinationRemover:
    """Handles exact, regex, and fuzzy hallucination detection with improved debugging"""
    
    BRACKET_PAIRS: Tuple[Tuple[str, str], ...] = (
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
        ("（", "）"),
        ("［", "］"),
        ("｛", "｝"),
        ("【", "】"),
        ("『", "』"),
        ("「", "」"),
        ("《", "》"),
    )

    def __init__(self, constants: HallucinationConstants, 
                 primary_language: Optional[str] = None,
                 user_blacklist: Optional[List[str]] = None):
        self.constants = constants
        self.primary_language = primary_language
        self.user_blacklist = user_blacklist or []
        
        # Caches for loaded patterns
        self._exact_lists: Optional[Dict[str, Set[str]]] = None
        self._regex_patterns: Optional[List[Dict[str, Any]]] = None
        self._blacklist_phrases: Optional[List[str]] = None
        
        # Load patterns on init with fallback
        self._load_patterns_safe()
        
    def _load_patterns_safe(self):
        """QUICK FIX: Load patterns with fallback on any failure"""
        try:
            self._load_patterns()
            if not self._exact_lists and not self._regex_patterns:
                raise Exception("No patterns loaded from external sources")
        except Exception as e:
            logger.warning(f"External pattern loading failed: {e}")
            logger.debug("Using built-in fallback patterns")
            self._load_fallback_patterns()
    
    def _load_fallback_patterns(self):
        """QUICK FIX: Minimal built-in patterns when external sources fail"""
        self._exact_lists = {
            'ja': {'www', 'ok', '笑', 'wwwww', 'ｗｗｗ'},
            'japanese': {'www', 'ok', '笑', 'wwwww', 'ｗｗｗ'},
            'jp': {'www', 'ok', '笑', 'wwwww', 'ｗｗｗ'}
        }
        
        self._regex_patterns = [
            {
                'pattern': r'^(OK|www|笑|W+|ｗ+)$',
                'category': 'common_hallucination',
                'confidence': 1.0,
                'replacement': ''
            },
            {
                'pattern': r'^ご視聴.*ありがとう.*$',
                'category': 'closing_phrase',
                'confidence': 0.95,
                'replacement': ''
            }
        ]
        
        self._blacklist_phrases = ['www', 'ok', '笑', 'wwwww']
        logger.debug(f"Loaded fallback patterns: {len(self._exact_lists)} exact lists, {len(self._regex_patterns)} regex patterns")
        
    def _load_patterns(self):
        """Load all hallucination patterns with improved error handling"""
        # Load exact match list
        try:
            self._exact_lists = self._load_json_from_url(self.constants.FILTER_LIST_URL)
            if self._exact_lists:
                # Convert lists to sets for faster lookup
                self._exact_lists = {
                    lang: set(phrases) if isinstance(phrases, list) else phrases
                    for lang, phrases in self._exact_lists.items()
                }
                total_phrases = sum(len(phrases) for phrases in self._exact_lists.values() if isinstance(phrases, (list, set)))
                logger.debug(f"Loaded exact hallucination lists for {len(self._exact_lists)} languages ({total_phrases} total phrases)")
            else:
                logger.warning("No exact hallucination lists loaded - hallucination removal may be ineffective")
                self._exact_lists = {}
        except Exception as e:
            logger.error(f"Failed to load exact hallucination list: {e}")
            self._exact_lists = {}
        
        # Load regex patterns
        try:
            pattern_data = self._load_json_from_url(self.constants.EXACT_LIST_URL)
            if pattern_data and 'patterns' in pattern_data:
                self._regex_patterns = pattern_data['patterns']
                logger.debug(f"Loaded {len(self._regex_patterns)} regex patterns")
                
                # Extract phrases for fuzzy matching
                self._blacklist_phrases = self._extract_blacklist_phrases(self._regex_patterns)
                logger.debug(f"Extracted {len(self._blacklist_phrases)} phrases for fuzzy matching")
            else:
                logger.warning("No regex patterns loaded from URL")
                self._regex_patterns = []
                self._blacklist_phrases = []
        except Exception as e:
            logger.error(f"Failed to load regex patterns: {e}")
            self._regex_patterns = []
            self._blacklist_phrases = []

    def _load_json_from_url(self, url: str) -> Optional[Dict]:
        """QUICK FIX: Add timeout and better error handling"""
        try:
            if url.startswith(('http://', 'https://')):
                logger.debug(f"Loading hallucination patterns from URL: {url}")
                response = requests.get(url, timeout=5)  # Quick timeout
                response.raise_for_status()
                return response.json()
            else:
                logger.debug(f"Loading hallucination patterns from file: {url}")
                with open(url, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except requests.Timeout:
            logger.warning(f"Timeout loading from {url}")
            return None
        except requests.RequestException as e:
            logger.warning(f"Network error loading from {url}: {e}")
            return None
        except FileNotFoundError:
            logger.warning(f"File not found: {url}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error loading from {url}: {e}")
            return None
            
    def _extract_blacklist_phrases(self, patterns: List[Dict]) -> List[str]:
        """Extract phrases for fuzzy matching from patterns"""
        blacklist_phrases = []
        fuzzy_categories = {
            "meta_reference", "media_reference", "closing_phrase", 
            "nonsensical", "user_defined"
        }
        
        for pattern_info in patterns:
            category = pattern_info.get('category', '')
            pattern = pattern_info.get('pattern', '')
            
            if category not in fuzzy_categories:
                continue
                
            # Count special regex characters
            special_chars = r'.*+?^${}()|[]\\<>'
            special_count = sum(1 for char in pattern if char in special_chars)
            
            # If pattern has few special characters, it might be a literal phrase
            if special_count <= 2 and len(pattern) > 5:
                # Clean basic regex elements
                cleaned = pattern
                for char in r'\^$.*+?()[]{}|':
                    cleaned = cleaned.replace(char, '')
                    
                if cleaned and len(cleaned) >= 3:
                    blacklist_phrases.append(cleaned)
                    
        # Add user blacklist
        blacklist_phrases.extend(self.user_blacklist)
        
        return blacklist_phrases
        
    def remove_hallucinations(self, text: str, language: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        REVISED: Performs strict, full-line, exact-match removal.
        No partial matching, no regex, no fuzzy logic.
        """
        if not text or not text.strip() or not self._exact_lists:
            return text, []
        
        effective_language = language or self.primary_language
        
        # Get the correct list for the language, with fallbacks
        lang_list = self._exact_lists.get(effective_language)
        if not lang_list:
            for fallback in ['ja', 'japanese', 'jp']:
                if fallback in self._exact_lists:
                    lang_list = self._exact_lists[fallback]
                    break
        if not lang_list:
            return text, []

        # Normalize text for comparison: lowercase and remove leading/trailing whitespace
        normalized_text = text.strip().lower()

        bracket_info = self._is_bracketed_context(text)
        if bracket_info:
            modifications = [{
                'type': 'bracketed_context',
                'pattern': bracket_info['wrapper_sequence'],
                'category': 'context_caption',
                'confidence': 1.0,
                'original': text,
                'modified': '',
                'language': effective_language,
            }]
            return '', modifications

        # Strict, full-line, exact matching
        if normalized_text in lang_list:
            modifications = [{
                'type': 'exact_match_full_line',
                'pattern': normalized_text,
                'category': 'hallucination',
                'confidence': 1.0,
                'original': text,
                'modified': '',
                'language': effective_language,
            }]
            return '', modifications
        
        return text, []

    def _is_bracketed_context(self, text: str) -> Optional[Dict[str, Any]]:
        stripped = text.strip()
        if len(stripped) < 3:
            return None

        inner = stripped
        wrappers: List[str] = []

        while True:
            matched = False
            for left, right in self.BRACKET_PAIRS:
                if inner.startswith(left) and inner.endswith(right) and len(inner) > len(left) + len(right):
                    inner = inner[len(left):-len(right)].strip()
                    wrappers.append(f"{left}{right}")
                    matched = True
                    break
            if not matched:
                break

        if wrappers and inner:
            return {
                'wrapper_sequence': wrappers,
                'inner_text': inner,
            }

        return None

    def _looks_like_valid_japanese_expression(self, text: str) -> bool:
        """NEW: Check if text looks like valid Japanese expression to prevent false positives"""
        text = text.strip()
        
        # Very short expressions with punctuation are likely valid
        if len(text) <= 10 and any(p in text for p in ['、', '。', 'です', 'だ', 'である']):
            return True
        
        # Mixed content (hiragana + something else) is likely valid
        import regex
        has_hiragana = bool(regex.search(r'[\p{Hiragana}]', text))
        has_katakana = bool(regex.search(r'[\p{Katakana}]', text))
        has_kanji = bool(regex.search(r'[\p{Han}]', text))
        
        script_count = sum([has_hiragana, has_katakana, has_kanji])
        if script_count >= 2:
            return True
        
        # Don't remove expressions with numbers or currency
        if regex.search(r'[\d¥$€£円]', text):
            return True
        
        # Expressions with common Japanese particles/endings are likely valid
        japanese_indicators = ['です', 'だ', 'である', 'ます', 'でした', 'いる', 'ある', 'する', 'した']
        if any(indicator in text for indicator in japanese_indicators):
            return True
            
        return False
        
    def _apply_exact_matching(self, text: str, language: str) -> Tuple[str, List[Dict]]:
        """Apply exact COMPLETE LINE matching with improved patterns"""
        modifications = []
        
        # FIRST: Check local hallucination patterns
        hallucination_pattern = re.compile(r'^(OK|www|笑|W+)$', re.IGNORECASE)
        if hallucination_pattern.match(text.strip()):
            logger.debug(f"Local hallucination pattern match: '{text.strip()}'")
            modifications.append({
                'type': 'exact_match_local_hallucination',
                'pattern': 'local_hallucination_patterns',
                'category': 'hallucination',
                'confidence': 1.0,
                'original': text,
                'modified': '',
                'language': 'local',
                'match_type': 'complete_line_local'
            })
            return '', modifications
        
        # Closing phrase patterns (complete line removal)
        closing_phrase_pattern = re.compile(r'^(ご|お)?視聴(して)?[いただきくれて]?(ありがとう|ございました)$')
        if closing_phrase_pattern.match(text.strip()):
            logger.debug(f"Closing phrase pattern match: '{text.strip()}'")
            modifications.append({
                'type': 'exact_match_closing_phrase',
                'pattern': 'closing_phrase_complete',
                'category': 'closing_phrase',
                'confidence': 1.0,
                'original': text,
                'modified': '',
                'language': 'ja',
                'match_type': 'complete_line_closing'
            })
            return '', modifications
        
        # Check external database patterns (from JSON files)
        lang_list = self._exact_lists.get(language, set())
        if not lang_list:
            # Try fallback languages
            fallback_langs = ['ja', 'japanese', 'jp'] if language not in ['ja', 'japanese', 'jp'] else []
            for fallback in fallback_langs:
                if fallback in self._exact_lists:
                    lang_list = self._exact_lists[fallback]
                    logger.debug(f"Using fallback language '{fallback}' for exact matching")
                    break
                    
        if not lang_list:
            logger.debug(f"No exact match list available for language '{language}'")
            return text, []
            
        # Normalize COMPLETE text for comparison
        normalized_text = text.strip().lower()
        
        # CRITICAL: Only match COMPLETE lines, no substring removal
        if normalized_text in lang_list:
            logger.debug(f"COMPLETE LINE exact hallucination match found: '{normalized_text}'")
            modifications.append({
                'type': 'exact_match_complete_line',
                'pattern': normalized_text,
                'category': 'hallucination',
                'confidence': 1.0,
                'original': text,
                'modified': '',
                'language': language,
                'match_type': 'complete_line_database'
            })
            return '', modifications
        else:
            logger.debug(f"No complete line exact match for: '{normalized_text[:50]}...'")
            
        return text, modifications
        
    def _apply_regex_matching(self, text: str) -> Tuple[str, List[Dict]]:
        """Apply regex pattern matching with improved error handling"""
        modifications = []
        current_text = text
        
        logger.debug(f"Applying regex matching to: '{text[:50]}...'")
        patterns_tested = 0
        patterns_matched = 0
        
        for pattern_info in self._regex_patterns:
            pattern = pattern_info.get('pattern', '')
            category = pattern_info.get('category', '')
            confidence = pattern_info.get('confidence', 0.9)
            replacement = pattern_info.get('replacement', '')
            
            patterns_tested += 1
            
            # Skip if confidence too low
            if confidence < self.constants.MIN_CONFIDENCE_THRESHOLD:
                continue
                
            try:
                # Check if pattern matches
                if re.search(pattern, current_text):
                    logger.debug(f"Regex match found - Category: {category}, Pattern: {pattern[:50]}...")
                    
                    # Apply the replacement
                    new_text = self._apply_regex_replacement_safe(pattern, replacement, current_text)
                    
                    if new_text != current_text:
                        modifications.append({
                            'type': 'regex_match',
                            'pattern': pattern,
                            'category': category,
                            'confidence': confidence,
                            'original': current_text,
                            'modified': new_text
                        })
                        current_text = new_text.strip()
                        patterns_matched += 1
                        
            except re.error as e:
                logger.warning(f"Regex error for pattern '{pattern[:30]}...': {e}")
                continue
                
        logger.debug(f"Regex matching complete: {patterns_tested} patterns tested, {patterns_matched} matched")
        return current_text, modifications

    def _apply_regex_replacement_safe(self, pattern: str, replacement: str, text: str) -> str:
        """QUICK FIX: Safe regex replacement handling"""
        try:
            if not replacement or replacement in ['', 'null', 'None']:
                return re.sub(pattern, '', text)
            
            # Handle malformed replacement strings
            if replacement.startswith('${') and '}' not in replacement:
                logger.debug(f"Malformed replacement '{replacement}' - using empty replacement")
                return re.sub(pattern, '', text)
            
            # For now, just use simple replacements
            if replacement.startswith('${'):
                return re.sub(pattern, '', text)  # QUICK FIX: Remove complex replacements
            
            return re.sub(pattern, replacement, text)
            
        except re.error as e:
            logger.warning(f"Regex error in pattern '{pattern[:30]}...': {e}")
            return text
        except Exception as e:
            logger.warning(f"Unexpected error in regex replacement: {e}")
            return text
        
    def _apply_fuzzy_matching(self, text: str) -> Tuple[str, List[Dict]]:
        """Apply fuzzy string matching"""
        if not text or not self._blacklist_phrases:
            return text, []
            
        modifications = []
        normalized_text = text.strip().lower()
        
        # Skip very short texts
        if len(normalized_text) < 3:
            return text, []
            
        best_match = None
        best_score = 0
        best_phrase = None
        
        for phrase in self._blacklist_phrases:
            normalized_phrase = phrase.strip().lower()
            
            # Skip very short phrases
            if len(normalized_phrase) < 3:
                continue
                
            # Skip if phrase is much shorter than text
            if len(normalized_phrase) < len(normalized_text) * 0.3:
                continue
                
            # Calculate similarity
            similarity = SequenceMatcher(None, normalized_text, normalized_phrase).ratio()
            
            if similarity > best_score and similarity >= self.constants.FUZZY_MATCH_THRESHOLD:
                best_score = similarity
                best_match = normalized_text
                best_phrase = phrase
                
        if best_match and best_phrase:
            logger.debug(f"Fuzzy hallucination match: '{text}' matches '{best_phrase}' with {best_score:.2f} similarity")
            modifications.append({
                'type': 'fuzzy_match',
                'pattern': f"fuzzy:{best_phrase}",
                'category': 'blacklisted_phrase',
                'confidence': best_score,
                'original': text,
                'modified': '',
                'matched_phrase': best_phrase
            })
            return '', modifications
            
        return text, modifications
        
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded hallucination databases"""
        stats = {
            'exact_lists': {},
            'regex_patterns_count': len(self._regex_patterns) if self._regex_patterns else 0,
            'blacklist_phrases_count': len(self._blacklist_phrases) if self._blacklist_phrases else 0
        }
        
        if self._exact_lists:
            for lang, phrases in self._exact_lists.items():
                stats['exact_lists'][lang] = len(phrases) if isinstance(phrases, (list, set)) else 0
                
        return stats