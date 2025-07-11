# whisperjav/modules/subtitle_sanitizer.py

# V12.1 - Refactored for logical correctness and simplicity



import pysrt

import shutil

import copy 

from pathlib import Path

from typing import List, Dict, Tuple, Optional, Any

from datetime import datetime

from dataclasses import dataclass



from whisperjav.utils.logger import logger

from whisperjav.config.sanitization_config import SanitizationConfig

from whisperjav.modules.hallucination_remover import HallucinationRemover

from whisperjav.modules.repetition_cleaner import RepetitionCleaner

from whisperjav.modules.cross_subtitle_processor import CrossSubtitleProcessor

from whisperjav.modules.timing_adjuster import TimingAdjuster



@dataclass

class SanitizationResult:

    """Result of sanitization process"""

    sanitized_path: Path

    original_backup_path: Optional[Path]

    artifacts_path: Optional[Path]

    statistics: Dict[str, Any]

    processing_time: float



@dataclass

class ArtifactEntry:

    """Represents a removed or modified subtitle entry"""

    index: int

    start_time: str

    end_time: str

    original_text: str

    modified_text: Optional[str]  # None if completely removed

    reason: str

    category: str

    confidence: float

    pattern: Optional[str]

    step: str

    additional_info: Dict[str, Any]



@dataclass

class Phase1Stats:

    """Statistics for Phase 1 processing"""

    original_count: int = 0

    # Simplified stats for the new refactored process

    hallucinations_removed: int = 0

    repetitions_cleaned: int = 0

    empty_purged: int = 0

    final_count: int = 0

    hallucination_phrases_loaded: Optional[Dict[str, int]] = None





class SubtitleSanitizer:

    """Comprehensive subtitle sanitization system with a refactored and validated pipeline"""

    

    def __init__(self, config: Optional[SanitizationConfig] = None,

                 config_file: Optional[Path] = None,

                 testing_profile: Optional[str] = None):

        # Constructor is unchanged for backward compatibility

        if config:

            self.config = config

        elif config_file:

            self.config = SanitizationConfig.from_file(config_file)

        elif testing_profile:

            from whisperjav.config.sanitization_config import get_testing_profile

            self.config = get_testing_profile(testing_profile)

        else:

            self.config = SanitizationConfig()

            

        if not config:

            self.config = SanitizationConfig.from_env()

            

        self.constants = self.config.get_effective_constants()

        self._init_processors()

        self.artifact_entries: List[ArtifactEntry] = []

        self.phase1_stats = Phase1Stats()

        logger.debug(f"SubtitleSanitizer initialized with sensitivity: {self.config.sensitivity_mode}")





    def _init_processors(self):

        # This is simplified as CrossSubtitleProcessor is now handled internally

        self.hallucination_remover = HallucinationRemover(

            self.constants['hallucination'],

            self.config.primary_language,

            self.config.user_blacklist_patterns

        )

        self.repetition_cleaner = RepetitionCleaner(self.constants['repetition'])

        self.timing_adjuster = TimingAdjuster(

            self.constants['timing'],

            self.constants['cross_subtitle'],

            self.config.primary_language

        )

        logger.debug("Initialized processors for rule-based cleaning.")





    def process(self, input_srt_path: Path) -> SanitizationResult:

        """Main processing function, orchestrates the new rule-based workflow."""

        start_time = datetime.now()

        logger.debug(f"Starting sanitization of: {input_srt_path}")



        try:

            original_subtitles = list(pysrt.open(str(input_srt_path), encoding='utf-8'))

            if not original_subtitles:

                logger.warning(f"No subtitles found in {input_srt_path}, exiting.")

                return SanitizationResult(input_srt_path, None, None, {}, 0)



            # FIX 1: Create a deep copy to work on, preserving the original list.

            subtitles_to_process = copy.deepcopy(original_subtitles)



            self.artifact_entries.clear()

            self.phase1_stats = Phase1Stats(original_count=len(original_subtitles))



            # === NEW RULE-BASED WORKFLOW ===

            

            # 1. Content Cleaning (now works on the copy)

            content_cleaned_subs = self._process_content_cleaning(subtitles_to_process)

            

            # 2. Timing Adjustment

            # CRITICAL: Pass the pristine original list and the cleaned list

            timed_subs, timing_mods = self.timing_adjuster.adjust_timings_content_aware(

                original_subtitles, content_cleaned_subs

            )

            # Re-enabled artifact recording for timing modifications

            for mod in timing_mods:

                self._record_timing_modification(mod)

            

            # 3. Final Renumbering

            final_subtitles = self._renumber_subtitles(timed_subs)

            

            # === END OF WORKFLOW ===



            # Save outputs

            paths = self._setup_output_paths(input_srt_path)

            if self.config.save_original:

                shutil.copy2(input_srt_path, paths['original_backup'])

            

            output_path = paths['original'] if not self.config.preserve_original_file else paths['sanitized']

            self._save_srt(final_subtitles, output_path)

            

            # FIX 2: Added the call to save the artifacts file

            if self.config.save_artifacts and self.artifact_entries:

                self._save_artifacts_srt(paths['artifacts'])

            

            processing_time = (datetime.now() - start_time).total_seconds()

            statistics = self._calculate_statistics(len(original_subtitles), len(final_subtitles))

            logger.debug(f"Sanitization complete in {processing_time:.2f}s. Subtitles: {len(original_subtitles)} → {len(final_subtitles)}")

            

            # FIX 2 (cont.): Return the correct artifacts_path in the result

            return SanitizationResult(

                sanitized_path=output_path,

                original_backup_path=paths.get('original_backup'),

                artifacts_path=paths.get('artifacts') if self.config.save_artifacts and self.artifact_entries else None,

                statistics=statistics,

                processing_time=processing_time

            )



        except Exception as e:

            logger.error(f"Critical sanitization failure for {input_srt_path}: {e}", exc_info=True)

            raise

            

            

    def _process_with_validation(self, input_srt_path: Path, start_time: datetime) -> SanitizationResult:

        """Internal process with validation and error handling."""

        if not input_srt_path.exists():

            raise FileNotFoundError(f"Input SRT file not found: {input_srt_path}")



        paths = self._setup_output_paths(input_srt_path)

        if self.config.save_original:

            shutil.copy2(input_srt_path, paths['original_backup'])

            logger.debug(f"Saved original backup to: {paths['original_backup']}")



        original_subtitles = list(pysrt.open(str(input_srt_path), encoding='utf-8'))

        logger.debug(f"Loaded {len(original_subtitles)} subtitles")



        self.artifact_entries.clear()

        self.phase1_stats = Phase1Stats()

        self.phase1_stats.original_count = len(original_subtitles)



        self._log_hallucination_database_info()



        # PHASE 1: Refactored Content Cleaning

        logger.debug("=== STARTING PHASE 1: Unified Content Cleaning ===")

        content_cleaned_subtitles = self._process_phase1_refactored(original_subtitles)



        # PHASE 2: Strict Timing Adjustment (Unchanged)

        logger.debug("=== STARTING PHASE 2: Strict Timing Adjustment ===")

        final_subtitles = self._process_phase2_with_validation(original_subtitles, content_cleaned_subtitles)



        self._log_phase1_results()



        output_path = paths['original'] if not self.config.preserve_original_file else paths['sanitized']

        self._save_srt(final_subtitles, output_path)



        if self.config.save_artifacts and self.artifact_entries:

            self._save_artifacts_srt(paths['artifacts'])



        processing_time = (datetime.now() - start_time).total_seconds()

        statistics = self._calculate_statistics(len(original_subtitles), len(final_subtitles))



        result = SanitizationResult(

            sanitized_path=output_path,

            original_backup_path=paths.get('original_backup'),

            artifacts_path=paths.get('artifacts'),

            statistics=statistics,

            processing_time=processing_time

        )

        

        logger.debug(f"Sanitization complete in {processing_time:.2f}s. Subtitles: {len(original_subtitles)} → {len(final_subtitles)}")

        return result



    # ===================================================================

    # REFACTORED PHASE 1 LOGIC

    # ===================================================================



    def _clean_subtitle_content(self, subtitle: pysrt.SubRipItem) -> Tuple[str, List[Dict]]:

        """

        NEW: Applies the full, ordered cleaning pipeline to a single subtitle's text.

        1. Full Hallucination Removal (Exact, Regex, Fuzzy)

        2. Repetition Cleaning

        """

        current_text = subtitle.text

        all_modifications = []



        # Step 1: Full-suite Hallucination Removal

        if self.hallucination_remover and (self.config.enable_exact_matching or self.config.enable_regex_matching or self.config.enable_fuzzy_matching):

            try:

                # CRITICAL FIX: Call the main public method to use all matching types

                cleaned_text, hall_mods = self.hallucination_remover.remove_hallucinations(

                    current_text, self.config.primary_language

                )

                if hall_mods:

                    all_modifications.extend(hall_mods)

                    current_text = cleaned_text

            except Exception as e:

                logger.warning(f"Hallucination remover failed for sub {subtitle.index}: {e}")



        # Step 2: Repetition Cleaning (on the already-cleaned text)

        if self.repetition_cleaner and self.config.enable_repetition_cleaning and current_text.strip():

            try:

                cleaned_text, rep_mods = self.repetition_cleaner.clean_repetitions(current_text)

                if rep_mods:

                    all_modifications.extend(rep_mods)

                    current_text = cleaned_text

            except Exception as e:

                logger.warning(f"Repetition cleaner failed for sub {subtitle.index}: {e}")

            

        return current_text.strip(), all_modifications



    def _process_phase1_refactored(self, subtitles: List[pysrt.SubRipItem]) -> List[pysrt.SubRipItem]:

        """

        REFACTORED: A single, unified content cleaning pass that replaces the old multi-pass system.

        """

        processed_subs = []

        

        for i, sub in enumerate(subtitles):

            original_text = sub.text

            if not original_text or not original_text.strip():

                self._record_removal(sub, i, "empty_text_initial", "validation")

                continue



            # Use the new unified cleaning method

            cleaned_text, modifications = self._clean_subtitle_content(sub)



            if not cleaned_text:

                # Subtitle text was removed entirely

                self._record_removal(sub, i, "content_cleaned_to_empty", "cleaning", modifications)

            elif cleaned_text != original_text:

                # Subtitle was modified

                modified_sub = pysrt.SubRipItem(index=sub.index, start=sub.start, end=sub.end, text=cleaned_text)

                self._record_modification(sub, modified_sub, i, modifications, "phase1_unified_cleaning")

                processed_subs.append(modified_sub)

            else:

                # Subtitle was unchanged

                processed_subs.append(sub)



        # Update stats based on recorded artifacts

        self.phase1_stats.hallucinations_removed = sum(1 for e in self.artifact_entries if 'hallucination' in e.category)

        self.phase1_stats.repetitions_cleaned = sum(1 for e in self.artifact_entries if 'repetition' in e.category)

        

        logger.debug(f"Phase 1 unified cleaning complete. Modifications recorded: {len(self.artifact_entries)}")



        final_subs = self._phase1_purge_empty(processed_subs)

        final_subs = self._phase1_renumber(final_subs)

        

        self.phase1_stats.final_count = len(final_subs)

        return final_subs



    # NOTE: The old methods `_phase1_exact_hallucination_pass`, `_phase1_repetition_cleaning_with_validation`,

    # and `_validate_repetition_cleaning` have been removed as they are now obsolete.



    # ===================================================================

    # PHASE 2 AND HELPER METHODS (Largely Unchanged)

    # ===================================================================



    def _process_phase2_with_validation(self, 

                                      original_subtitles: List[pysrt.SubRipItem],

                                      content_cleaned_subtitles: List[pysrt.SubRipItem]) -> List[pysrt.SubRipItem]:

        """Phase 2 timing adjustment with strict validation (Unchanged)."""

        if self.timing_adjuster:

            timing_adjusted_subs, timing_mods = self.timing_adjuster.adjust_timings_content_aware(

                original_subtitles, content_cleaned_subtitles

            )

            for mod in timing_mods:

                self._record_timing_modification(mod)

            logger.debug(f"✓ Strict timing adjustment complete: {len(timing_mods)} adjustments made")

            return timing_adjusted_subs

        else:

            logger.debug("⏭ Timing adjustment skipped (disabled)")

            return content_cleaned_subtitles



    def _log_hallucination_database_info(self):

        """Log information about loaded hallucination databases."""

        if not self.hallucination_remover: return

        stats = self.hallucination_remover.get_database_stats()

        logger.debug("=== HALLUCINATION DATABASE INFO ===")

        if stats['exact_lists']:

            total_exact = sum(stats['exact_lists'].values())

            logger.debug(f"Exact match lists loaded for {len(stats['exact_lists'])} languages, total phrases: {total_exact}")

            self.phase1_stats.hallucination_phrases_loaded = stats['exact_lists']

        if stats['regex_patterns_count']:

            logger.debug(f"Regex patterns loaded: {stats['regex_patterns_count']}")

        if stats['blacklist_phrases_count']:

            logger.debug(f"Fuzzy match blacklist phrases: {stats['blacklist_phrases_count']}")

        logger.debug("=====================================")



    def _phase1_purge_empty(self, subtitles: List[pysrt.SubRipItem]) -> List[pysrt.SubRipItem]:

        """Remove any subtitles that became empty during Phase 1."""

        filtered = [sub for sub in subtitles if sub.text and sub.text.strip()]

        purged_count = len(subtitles) - len(filtered)

        self.phase1_stats.empty_purged = purged_count

        if purged_count > 0:

            logger.debug(f"  Purged {purged_count} empty subtitles created during cleaning")

        return filtered



    def _phase1_renumber(self, subtitles: List[pysrt.SubRipItem]) -> List[pysrt.SubRipItem]:

        """Renumber subtitles sequentially."""

        for i, sub in enumerate(subtitles, 1):

            sub.index = i

        return subtitles



    def _log_phase1_results(self):

        """Log comprehensive Phase 1 results."""

        stats = self.phase1_stats

        logger.debug("=== PHASE 1 RESULTS SUMMARY ===")

        logger.debug(f"Original subtitles: {stats.original_count}")

        logger.debug(f"Hallucinations modified/removed: {stats.hallucinations_removed}")

        logger.debug(f"Repetitions modified/removed: {stats.repetitions_cleaned}")

        logger.debug(f"Empty subtitles purged: {stats.empty_purged}")

        logger.debug(f"Final subtitles after Phase 1: {stats.final_count}")

        logger.debug("===============================")



    def _record_removal(self, subtitle: pysrt.SubRipItem, original_index: int, reason: str, category: str, modifications: List[Dict] = []):

        """Record a completely removed subtitle."""

        entry_category = category

        confidence = 1.0

        if modifications:

            entry_category = ','.join(set(m.get('category', 'unknown') for m in modifications))

            confidence = min(m.get('confidence', 1.0) for m in modifications)



        entry = ArtifactEntry(

            index=original_index + 1,

            start_time=str(subtitle.start), end_time=str(subtitle.end),

            original_text=subtitle.text, modified_text=None,

            reason=reason, category=entry_category, confidence=confidence,

            pattern=None, step="phase1_unified_cleaning", additional_info={}

        )

        self.artifact_entries.append(entry)



    def _record_modification(self, original: pysrt.SubRipItem, modified: pysrt.SubRipItem, 

                           original_index: int, modifications: List[Dict], step_name: str):

        """Record a modified subtitle."""

        if not modifications: return



        categories = set(mod.get('category', mod.get('type', 'unknown')) for mod in modifications)

        patterns = [mod.get('pattern') for mod in modifications if mod.get('pattern')]

        min_confidence = min(mod.get('confidence', 1.0) for mod in modifications)

                

        entry = ArtifactEntry(

            index=original_index + 1,

            start_time=str(original.start), end_time=str(original.end),

            original_text=original.text, modified_text=modified.text,

            reason="multiple_modifications" if len(modifications) > 1 else modifications[0].get('type', 'unknown'),

            category=','.join(categories), confidence=min_confidence,

            pattern=patterns[0] if len(patterns) == 1 else "multiple",

            step=step_name,

            additional_info={'modifications': modifications} if self.config.artifact_detail_level == "full" else {}

        )

        self.artifact_entries.append(entry)



    def _record_timing_modification(self, modification: Dict):

        """Record timing adjustment in artifacts."""

        entry = ArtifactEntry(

            index=modification.get('subtitle_index', 0),

            start_time=str(modification.get('original_start', '')), end_time=str(modification.get('end_timestamp', '')),

            original_text=modification.get('original_text', 'N/A'), modified_text=modification.get('modified_text', 'N/A'),

            reason=modification.get('reason', 'timing_adjustment'), category="timing_adjustment",

            confidence=1.0, pattern=None, step="phase2_timing_adjustment",

            additional_info=modification if self.config.artifact_detail_level == "full" else {}

        )

        self.artifact_entries.append(entry)





    def _process_content_cleaning(self, subtitles: List[pysrt.SubRipItem]) -> List[pysrt.SubRipItem]:

        """

        REVISED: Runs the full content cleaning pipeline AND records artifacts for each step.

        """

        logger.debug("Step 1/3 (Content Cleaning) starting...")

        

        # First, handle deduplication, as it changes the number of subtitles

        subtitles_after_dedup = self._deduplicate_sequential_lines(subtitles)

        # Note: A full implementation would record merge artifacts here. Keeping it simple for now.



        cleaned_subs = []

        for sub in subtitles_after_dedup:

            original_sub_for_comparison = copy.deepcopy(sub)

            modified_text = sub.text

            all_mods = []

            final_reason = ""



            # --- Apply cleaning steps sequentially ---



            # Hallucination Removal

            modified_text, hall_mods = self.hallucination_remover.remove_hallucinations(modified_text, self.config.primary_language)

            if hall_mods:

                all_mods.extend(hall_mods)

                final_reason = "hallucination"



            # Repetition Cleaning

            if modified_text.strip(): # Only process if there's text left

                modified_text, rep_mods = self.repetition_cleaner.clean_repetitions(modified_text)

                if rep_mods:

                    all_mods.extend(rep_mods)

                    final_reason = "repetition"

            

            # --- Final Decision ---



            # If text is now empty, record it as a removal

            if not modified_text.strip():

                self._record_removal(original_sub_for_comparison, original_sub_for_comparison.index, final_reason, "content_cleaning", all_mods)

                continue # Skip to the next subtitle



            # If text was changed, record the modification

            if modified_text != original_sub_for_comparison.text:

                sub.text = modified_text

                self._record_modification(original_sub_for_comparison, sub, original_sub_for_comparison.index, all_mods, "content_cleaning")



            cleaned_subs.append(sub)



        # Now, run the High CPS removal on the already cleaned text

        final_subs = self._remove_abnormally_fast_subs(cleaned_subs)

        logger.debug(f"Content Cleaning complete. Subtitles remaining: {len(final_subs)}")

        return final_subs



    def _deduplicate_sequential_lines(self, subtitles: List[pysrt.SubRipItem]) -> List[pysrt.SubRipItem]:

        """

        NEW: Implements strict, exact-match deduplication based on user directive.

        """

        if not subtitles:

            return []



        dedup_threshold = self.constants['cross_subtitle'].DEDUP_THRESHOLD # e.g., 3

        max_gap_ms = self.constants['cross_subtitle'].MAX_GAP_MS # e.g., 600



        merged_subs = []

        i = 0

        while i < len(subtitles):

            current_sub = subtitles[i]

            

            # Find a sequence of identical subtitles

            sequence = [current_sub]

            j = i + 1

            while j < len(subtitles):

                next_sub = subtitles[j]

                # Condition 1: Exact text match (normalized)

                is_text_match = current_sub.text.strip() == next_sub.text.strip()

                # Condition 2: Time gap is small

                gap = next_sub.start.ordinal - sequence[-1].end.ordinal

                is_gap_small = gap >= 0 and gap < max_gap_ms

                

                if is_text_match and is_gap_small:

                    sequence.append(next_sub)

                    j += 1

                else:

                    break

            

            # If the sequence is long enough, merge it

            if len(sequence) >= dedup_threshold:

                first_sub = sequence[0]

                last_sub = sequence[-1]

                

                # Create a new merged subtitle. Keep the index of the first sub in the sequence.

                merged_sub = pysrt.SubRipItem(

                    index=first_sub.index,

                    start=first_sub.start,

                    end=last_sub.end,

                    text=first_sub.text # Text is just the single instance

                )

                merged_subs.append(merged_sub)

                i = j # Move pointer past the processed sequence

            else:

                # Sequence not long enough, just add the current sub and move on

                merged_subs.append(current_sub)

                i += 1

        

        return merged_subs







    def _remove_abnormally_fast_subs(self, subtitles: List[pysrt.SubRipItem]) -> List[pysrt.SubRipItem]:

        """Removes subtitles with a CPS higher than MAX_SAFE_CPS and records an artifact."""

        timing_consts = self.constants['timing']

        max_cps = timing_consts.MAX_SAFE_CPS

        min_len = timing_consts.MIN_TEXT_LENGTH_FOR_CPS_CHECK



        kept_subs = []

        for sub in subtitles:

            text_len = len(sub.text_without_tags.strip())

            duration_s = sub.duration.ordinal / 1000.0



            if text_len >= min_len and duration_s > 0:

                actual_cps = text_len / duration_s

                if actual_cps > max_cps:

                    reason = f"abnormally_fast_cps_{actual_cps:.1f}"

                    logger.debug(f"Removing Abnormally Fast Subtitle #{sub.index} ({actual_cps:.1f} CPS). Text: '{sub.text}'")

                    # Record the removal before deleting

                    self._record_removal(sub, sub.index, reason, "content_cleaning_cps")

                    continue

            

            kept_subs.append(sub)

        

        return kept_subs











    def _renumber_subtitles(self, subtitles: List[pysrt.SubRipItem]) -> List[pysrt.SubRipItem]:

        """Renumber subtitles sequentially. This should be the final step."""

        for i, sub in enumerate(subtitles, 1):

            sub.index = i

        return subtitles





    def _setup_output_paths(self, input_path: Path) -> Dict[str, Path]:

        """Setup all output file paths"""

        paths = {}

        raw_subs_dir = input_path.parent / self.constants['processing'].RAW_SUBS_FOLDER

        raw_subs_dir.mkdir(exist_ok=True)

        paths['original'] = input_path

        if self.config.save_original:

            backup_name = f"{input_path.stem}.{self.constants['processing'].ORIGINAL_BACKUP_SUFFIX}{input_path.suffix}"

            paths['original_backup'] = raw_subs_dir / backup_name

        if self.config.preserve_original_file:

            sanitized_name = f"{input_path.stem}.{self.constants['processing'].SANITIZED_SUFFIX}{input_path.suffix}"

            paths['sanitized'] = input_path.parent / sanitized_name

        else:

            sanitized_name = f"{input_path.stem}.{self.constants['processing'].SANITIZED_SUFFIX}{input_path.suffix}"

            paths['sanitized'] = raw_subs_dir / sanitized_name

        if self.config.save_artifacts:

            artifacts_name = f"{input_path.stem}.{self.constants['processing'].ARTIFACTS_SUFFIX}{input_path.suffix}"

            paths['artifacts'] = raw_subs_dir / artifacts_name

        return paths



    def _save_srt(self, subtitles: List[pysrt.SubRipItem], output_path: Path):

        """Save subtitles to SRT file"""

        output_path.parent.mkdir(parents=True, exist_ok=True)

        pysrt.SubRipFile(subtitles).save(str(output_path), encoding='utf-8')

        logger.debug(f"Saved sanitized SRT to: {output_path}")



    def _save_artifacts_srt(self, artifacts_path: Path):

        """Save artifacts as SRT file with detailed information"""

        if not self.artifact_entries: return

        # Implementation remains the same...

        artifacts_subs = []

        if self.config.artifact_detail_level in ["full", "summary"]:

            artifacts_subs.append(self._create_summary_subtitle())

        for entry in sorted(self.artifact_entries, key=lambda e: e.index):

            sub = self._create_artifact_subtitle(entry, len(artifacts_subs) + 1)

            artifacts_subs.append(sub)

        if artifacts_subs:

            artifacts_path.parent.mkdir(parents=True, exist_ok=True)

            pysrt.SubRipFile(artifacts_subs).save(str(artifacts_path), encoding='utf-8')

            logger.debug(f"Saved artifacts to: {artifacts_path}")



    def _create_summary_subtitle(self) -> pysrt.SubRipItem:

        # Implementation remains the same...

        stats = self.phase1_stats

        summary_text = f"""[SANITIZATION SUMMARY]

Original subtitles: {stats.original_count}

Hallucinations modified/removed: {stats.hallucinations_removed}

Repetitions modified/removed: {stats.repetitions_cleaned}

Final subtitles: {stats.final_count}

Config: {self.config.sensitivity_mode}"""

        return pysrt.SubRipItem(index=1, start=pysrt.SubRipTime(0), end=pysrt.SubRipTime(seconds=5), text=summary_text)





# In whisperjav/modules/subtitle_sanitizer.py



    def _create_artifact_subtitle(self, entry: ArtifactEntry, index: int) -> pysrt.SubRipItem:

        """Create subtitle entry for artifact with more detail."""

        reason = entry.reason.replace('_', ' ').title()

        

        # --- FIX: Special formatting now includes the final modified text ---

        if entry.category == "timing_adjustment":

            info = entry.additional_info

            original_duration = info.get('original_duration', 0)

            new_duration = info.get('new_duration', 0)

            text = (

                f"[TIMING MODIFIED - {reason}]\n"

                f"Original: {entry.original_text}\n"

                f"Modified: {entry.modified_text}\n"  # This line is new

                f"Duration: {original_duration:.2f}s → {new_duration:.2f}s"

            )

        # ---------------------------------------------------

        elif entry.modified_text is None:

            text = f"[REMOVED - {reason}]\nOriginal: {entry.original_text}"

        else:

            text = f"[MODIFIED - {reason}]\nOriginal: {entry.original_text}\nModified: {entry.modified_text}"



        text += f"\nCategory: {entry.category} | Confidence: {entry.confidence:.2f}"

            

        try:

            start = pysrt.SubRipTime.from_string(entry.start_time)

            end = pysrt.SubRipTime.from_string(entry.end_time)

        except:

            start = pysrt.SubRipTime(milliseconds=index * 1000)

            end = pysrt.SubRipTime(milliseconds=index * 1000 + 1000)

            

        return pysrt.SubRipItem(index=index, start=start, end=end, text=text)



        

    def _calculate_statistics(self, original_count: int, final_count: int) -> Dict[str, Any]:

        """Calculate processing statistics"""

        stats = {

            'original_subtitle_count': original_count,

            'final_subtitle_count': final_count,

            'total_modifications': len(self.artifact_entries),

            'removals': sum(1 for e in self.artifact_entries if e.modified_text is None),

            'modifications': sum(1 for e in self.artifact_entries if e.modified_text is not None),

            'reduction_percentage': ((original_count - final_count) / original_count * 100) if original_count > 0 else 0

        }

        category_counts = {}

        for entry in self.artifact_entries:

            for cat in entry.category.split(','):

                category_counts[cat] = category_counts.get(cat, 0) + 1

        stats['modifications_by_category'] = category_counts

        return stats

