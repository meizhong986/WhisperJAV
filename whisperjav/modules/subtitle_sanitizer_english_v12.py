# whisperjav/modules/simple_english_sanitizer.py
# FINAL VERSION: A self-contained English sanitizer with advanced timing and logging.

import pysrt
import re
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any
import copy
import requests
import json
import logging

# --- English Sanitization Constants ---
ENGLISH_CONSTANTS = {
    "HALLUCINATION_LIST_URL": "https://gist.githubusercontent.com/meizhong986/4882bdb3f4f5aa4034a112cebd2e0845/raw/9e78020b9f85cb7aa3d7004d477353adbfe60ee9/WhisperJAV_hallucination_filter_sorted_v08.json",
    "FALLBACK_HALLUCINATION_PHRASES": {
        'um', 'uh', 'hmm', 'mhm', 'ah', 'oh', 'eh', 'uh-huh',
        'yeah', 'yep', 'yup', 'okay', 'ok', 'bye', '...',
        '[music]', '[applause]', '[laughter]', '[cheering]',
        'thanks for watching', 'like and subscribe', 'subtitles by'
    },
    "MIN_SAFE_CPS": 2.0, "MAX_SAFE_CPS": 25.0, "MIN_TEXT_LENGTH_FOR_CPS_CHECK": 5, "TARGET_CPS": 15.0,
    "MIN_SUBTITLE_DURATION_S": 0.2, "MAX_SUBTITLE_DURATION_S": 10.0, "MERGE_MIN_SEQUENCE": 3,
    "MERGE_MAX_GAP_MS": 1000, "REPETITION_THRESHOLD": 2,
}
# -----------------------------------------

class SimpleEnglishSanitizer:
    """A self-contained English subtitle sanitizer with advanced features."""

    def __init__(self, constants: Dict[str, Any] = None):
        self.constants = constants or ENGLISH_CONSTANTS
        self.hallucination_phrases = self._load_hallucination_phrases()
        artifact_patterns = [r'^\[.*\]$', r'^♪+$', r'^\.+$', r'^-+$', r'^\W+$']
        self.artifact_regex = [re.compile(p, re.IGNORECASE) for p in artifact_patterns]

    def _load_hallucination_phrases(self) -> Set[str]:
        url = self.constants.get("HALLUCINATION_LIST_URL")
        fallback_phrases = self.constants.get("FALLBACK_HALLUCINATION_PHRASES", set())
        if not url: return fallback_phrases
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            phrases = set()
            for lang_key in ['en', 'eng', 'english', 'English']:
                if lang_key in data:
                    lang_phrases = data[lang_key]
                    if isinstance(lang_phrases, list):
                        phrases.update(p.lower().strip() for p in lang_phrases if p)
                    break
            if not phrases:
                logging.warning("Could not find English phrases in loaded list. Using fallback.")
                return fallback_phrases
            logging.info(f"Successfully loaded {len(phrases)} external English hallucination phrases.")
            return phrases
        except (requests.RequestException, json.JSONDecodeError) as e:
            logging.warning(f"Failed to load external hallucination list ({e}). Using fallback.")
            return fallback_phrases

    def sanitize(self, srt_path: Path) -> Path:
        """Main sanitization method orchestrating all steps."""
        try:
            subtitles = list(pysrt.open(str(srt_path), encoding='utf-8'))
        except Exception as e:
            logging.error(f"Error loading SRT file {srt_path}: {e}")
            return srt_path
        if not subtitles: return srt_path

        original_subs_map = {sub.index: copy.deepcopy(sub) for sub in subtitles}
        artifacts = []

        content_cleaned_subs, content_artifacts = self._clean_content(subtitles)
        artifacts.extend(content_artifacts)
        merged_subs, merge_artifacts = self._merge_consecutive_duplicates(content_cleaned_subs)
        artifacts.extend(merge_artifacts)
        final_subs, timing_artifacts = self._fix_timing_issues(original_subs_map, merged_subs)
        artifacts.extend(timing_artifacts)
        
        for i, sub in enumerate(final_subs, 1):
            sub.index = i
        
        output_path = srt_path.parent / f"{srt_path.stem}_sanitized.srt"
        pysrt.SubRipFile(final_subs).save(str(output_path), encoding='utf-8')
        logging.info(f"Sanitized SRT saved to: {output_path}")
        self._save_artifacts(artifacts, output_path)
        return output_path

    def _clean_content(self, subtitles: List[pysrt.SubRipItem]) -> Tuple[List[pysrt.SubRipItem], List[str]]:
        kept_subs, artifacts = [], []
        max_cps, min_len = self.constants["MAX_SAFE_CPS"], self.constants["MIN_TEXT_LENGTH_FOR_CPS_CHECK"]
        for sub in subtitles:
            original_text = sub.text
            if not sub.text.strip():
                artifacts.append(f"[REMOVED] Sub #{sub.index}: Line was empty.")
                continue
            if self._is_hallucination(original_text):
                artifacts.append(f"[REMOVED] Sub #{sub.index}: Matched hallucination list. Text: '{original_text}'")
                continue
            text_len, duration_s = len(original_text.strip()), sub.duration.ordinal / 1000.0
            if text_len >= min_len and duration_s > 0:
                actual_cps = text_len / duration_s
                if actual_cps > max_cps:
                    artifacts.append(f"[REMOVED] Sub #{sub.index}: High CPS detected ({actual_cps:.1f} > {max_cps}). Text: '{original_text}'")
                    continue
            cleaned_text = self._clean_repetitions(original_text)
            if cleaned_text != original_text:
                artifacts.append(f"[CLEANED] Sub #{sub.index}: Repetitions cleaned.\n  - Original: '{original_text}'\n  - Cleaned:  '{cleaned_text}'")
                sub.text = cleaned_text
            if sub.text.strip():
                kept_subs.append(sub)
        return kept_subs, artifacts

    def _is_hallucination(self, text: str) -> bool:
        normalized_text = text.strip().lower()
        if not normalized_text: return False
        if normalized_text in self.hallucination_phrases: return True
        normalized_no_punct = re.sub(r'[^\w\s]+$', '', normalized_text).strip()
        if normalized_no_punct in self.hallucination_phrases: return True
        for pattern in self.artifact_regex:
            if pattern.match(text.strip()): return True
        return False

    def _clean_repetitions(self, text: str) -> str:
        cleaned = text
        rep_threshold = self.constants["REPETITION_THRESHOLD"]
        patterns = [
            (r'\b(\w+)\s+(\1\s*){' + str(rep_threshold) + r',}', r'\1 ' * rep_threshold),
            (r'(\w)\1{2,}', r'\1' * rep_threshold),
            (r'([.!?])\1+', r'\1'),
        ]
        for pattern, replacement in patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def _merge_consecutive_duplicates(self, subtitles: List[pysrt.SubRipItem]) -> Tuple[List[pysrt.SubRipItem], List[str]]:
        if not subtitles: return [], []
        merged_subs, artifacts = [], []
        min_sequence, max_gap_ms = self.constants["MERGE_MIN_SEQUENCE"], self.constants["MERGE_MAX_GAP_MS"]
        i = 0
        while i < len(subtitles):
            current_sub = subtitles[i]
            sequence_end_j = i
            for j in range(i + 1, len(subtitles)):
                next_sub = subtitles[j]
                gap_ms = next_sub.start.ordinal - subtitles[j-1].end.ordinal
                if current_sub.text.strip().lower() == next_sub.text.strip().lower() and 0 <= gap_ms < max_gap_ms:
                    sequence_end_j = j
                else: break
            num_in_sequence = (sequence_end_j - i) + 1
            if num_in_sequence >= min_sequence:
                first_sub, last_sub = subtitles[i], subtitles[sequence_end_j]
                merged_subs.append(pysrt.SubRipItem(index=first_sub.index, start=first_sub.start, end=last_sub.end, text=first_sub.text))
                artifacts.append(f"[MERGED] Subs #{first_sub.index} through #{last_sub.index} into one line.")
                i = sequence_end_j + 1
            else:
                merged_subs.append(current_sub)
                i += 1
        return merged_subs, artifacts

    def _fix_timing_issues(self, original_map: Dict[int, pysrt.SubRipItem], modified_subs: List[pysrt.SubRipItem]) -> Tuple[List[pysrt.SubRipItem], List[str]]:
        final_subs, artifacts = [], []
        min_cps, min_len = self.constants["MIN_SAFE_CPS"], self.constants["MIN_TEXT_LENGTH_FOR_CPS_CHECK"]
        target_cps, min_dur, max_dur = self.constants["TARGET_CPS"], self.constants["MIN_SUBTITLE_DURATION_S"], self.constants["MAX_SUBTITLE_DURATION_S"]
        for sub in modified_subs:
            original_sub, should_adjust, reason = original_map.get(sub.index), False, ""
            if original_sub:
                if sub.text.strip() != original_sub.text.strip(): should_adjust, reason = True, "content changed"
                elif abs(sub.duration.ordinal - original_sub.duration.ordinal) > 500: should_adjust, reason = True, "line merged"
            text_len, duration_s = len(sub.text.strip()), sub.duration.ordinal / 1000.0
            if not should_adjust and text_len >= min_len and duration_s > 0 and (text_len / duration_s) < min_cps:
                should_adjust, reason = True, f"low CPS ({text_len / duration_s:.1f})"
            if should_adjust:
                original_duration_s = sub.duration.ordinal / 1000.0
                ideal_duration_s = min(max(text_len / target_cps, min_dur), max_dur)
                new_start_ms = sub.end.ordinal - int(ideal_duration_s * 1000)
                if final_subs: new_start_ms = max(new_start_ms, final_subs[-1].end.ordinal + 50)
                sub.start = pysrt.SubRipTime(milliseconds=max(0, new_start_ms))
                artifacts.append(f"[TIMING ADJUSTED] Sub #{sub.index} ({reason}): Duration {original_duration_s:.2f}s -> {ideal_duration_s:.2f}s. Text: '{sub.text}'")
            final_subs.append(sub)
        return final_subs, artifacts

    def _save_artifacts(self, artifacts: List[str], srt_path: Path):
        if not artifacts: return
        log_path = srt_path.parent / f"{srt_path.stem.replace('_sanitized', '')}_sanitizer.log"
        header = f"Sanitization Report for: {srt_path.name}\n" + "=" * 40 + "\n\n"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(header)
                f.write("\n\n".join(artifacts))
            logging.info(f"Detailed log saved to: {log_path}")
        except Exception as e:
            logging.warning(f"Error saving artifact log: {e}")