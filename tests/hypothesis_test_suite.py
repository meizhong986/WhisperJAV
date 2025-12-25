#!/usr/bin/env python3
"""
Hypothesis Testing Suite for ASR Regression Investigation

This suite tests specific parameter hypotheses to investigate why v1.7.3
produces ~20% fewer subtitles than v1.7.1.

Usage:
    # Run all hypothesis tests
    python tests/hypothesis_test_suite.py --audio subset.wav --reference v1.7.1_subset.srt

    # Run specific hypothesis
    python tests/hypothesis_test_suite.py --audio subset.wav --hypothesis vad_params

    # Quick mode (fewer variations)
    python tests/hypothesis_test_suite.py --audio subset.wav --quick

    # Skip baseline (if already run)
    python tests/hypothesis_test_suite.py --audio subset.wav --skip-baseline

Author: Claude Code (Automated Regression Investigation)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hypothesis_configs import (
    HypothesisConfigs,
    TestConfig,
    V173Defaults,
    list_all_configs
)

# WhisperJAV imports
from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
from whisperjav.config.transcription_tuner import TranscriptionTuner
from whisperjav.utils.logger import logger

# SRT parsing
import srt as srt_module


@dataclass
class TestResult:
    """Results from a single hypothesis test run."""
    config_name: str
    hypothesis: str
    total_subtitles: int
    total_speech_duration_sec: float
    avg_subtitle_duration_sec: float
    very_short_subtitles: int  # < 1s
    long_subtitles: int  # > 5s
    processing_time_sec: float
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ComparisonMetrics:
    """Comparison metrics against baseline and reference."""
    subtitle_delta_vs_baseline: int
    subtitle_delta_vs_reference: int
    duration_delta_vs_baseline: float
    duration_delta_vs_reference: float
    improvement_vs_baseline_pct: float
    improvement_vs_reference_pct: float


class SubtitleMetrics:
    """Calculate metrics from SRT subtitle data."""

    @staticmethod
    def parse_srt_file(srt_path: Path) -> List[srt_module.Subtitle]:
        """Parse SRT file and return list of subtitles."""
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return list(srt_module.parse(content))
        except Exception as e:
            logger.error(f"Failed to parse SRT file {srt_path}: {e}")
            return []

    @staticmethod
    def calculate_metrics(subtitles: List[srt_module.Subtitle]) -> Dict[str, Any]:
        """
        Calculate metrics from subtitle list.

        Returns:
            Dict with keys: total_count, total_duration_sec, avg_duration_sec,
            very_short_count, long_count
        """
        if not subtitles:
            return {
                "total_count": 0,
                "total_duration_sec": 0.0,
                "avg_duration_sec": 0.0,
                "very_short_count": 0,
                "long_count": 0
            }

        durations = []
        very_short = 0
        long_subs = 0

        for sub in subtitles:
            duration = (sub.end - sub.start).total_seconds()
            durations.append(duration)

            if duration < 1.0:
                very_short += 1
            elif duration > 5.0:
                long_subs += 1

        total_duration = sum(durations)
        avg_duration = total_duration / len(durations) if durations else 0.0

        return {
            "total_count": len(subtitles),
            "total_duration_sec": total_duration,
            "avg_duration_sec": avg_duration,
            "very_short_count": very_short,
            "long_count": long_subs
        }


class HypothesisTester:
    """Orchestrates hypothesis testing across configurations."""

    def __init__(
        self,
        audio_path: Path,
        reference_srt_path: Optional[Path] = None,
        output_dir: Path = None,
        temp_dir: Path = None,
    ):
        """
        Initialize the hypothesis tester.

        Args:
            audio_path: Path to test audio file (25-minute subset)
            reference_srt_path: Optional path to v1.7.1 reference SRT
            output_dir: Directory for output SRT files
            temp_dir: Directory for temporary files
        """
        self.audio_path = audio_path
        self.reference_srt_path = reference_srt_path
        self.output_dir = output_dir or Path("./hypothesis_outputs")
        self.temp_dir = temp_dir or Path("./hypothesis_temp")

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Load reference SRT if provided
        self.reference_metrics = None
        if reference_srt_path and reference_srt_path.exists():
            logger.info(f"Loading reference SRT: {reference_srt_path}")
            ref_subs = SubtitleMetrics.parse_srt_file(reference_srt_path)
            self.reference_metrics = SubtitleMetrics.calculate_metrics(ref_subs)
            logger.info(
                f"Reference has {self.reference_metrics['total_count']} subtitles, "
                f"{self.reference_metrics['total_duration_sec']:.1f}s total speech"
            )

        # Results storage
        self.results: List[TestResult] = []
        self.baseline_result: Optional[TestResult] = None

    def _build_resolved_config(self, test_config: TestConfig) -> Dict[str, Any]:
        """
        Build a resolved config by merging test overrides with v1.7.3 defaults.

        Args:
            test_config: Test configuration with parameter overrides

        Returns:
            Resolved configuration dict suitable for BalancedPipeline
        """
        # Start with v1.7.3 aggressive defaults
        tuner = TranscriptionTuner()
        base_config = tuner.resolve_params(
            pipeline_name="balanced",
            sensitivity="aggressive",
            task="transcribe"
        )

        # CRITICAL: Set speech_segmenter backend to silero-v3.1 to match v1.7.1 behavior
        # The resolver doesn't set this, so FasterWhisperProASR defaults to v4.0
        # v1.7.1 used silero-v3.1, so we must explicitly set it here for accurate testing
        if "speech_segmenter" not in base_config["params"]:
            base_config["params"]["speech_segmenter"] = {}
        base_config["params"]["speech_segmenter"]["backend"] = "silero-v3.1"

        # Apply parameter overrides
        for section, overrides in test_config.params_override.items():
            if section == "asr":
                # ASR parameters go into params.decoder and params.provider
                for key, value in overrides.items():
                    # Map to correct location
                    if key in ["temperature", "fp16"]:
                        base_config["params"]["provider"][key] = value
                    else:
                        base_config["params"]["decoder"][key] = value

            elif section == "vad":
                # VAD parameters go into params.vad
                base_config["params"]["vad"].update(overrides)

            elif section == "model":
                # Model parameters (compute_type, device, etc.) go into model section
                base_config["model"].update(overrides)

            else:
                # Unknown section - log warning but continue
                logger.warning(f"Unknown config section: {section}")

        return base_config

    def run_test(self, test_config: TestConfig) -> TestResult:
        """
        Run a single hypothesis test.

        Args:
            test_config: Test configuration to run

        Returns:
            TestResult with metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {test_config.name}")
        logger.info(f"Description: {test_config.description}")
        logger.info(f"Hypothesis: {test_config.hypothesis}")
        logger.info(f"{'='*70}")

        start_time = time.time()

        try:
            # Build resolved config
            resolved_config = self._build_resolved_config(test_config)

            # Create pipeline
            pipeline = BalancedPipeline(
                output_dir=str(self.output_dir),
                temp_dir=str(self.temp_dir),
                keep_temp_files=False,
                subs_language="native",
                resolved_config=resolved_config
            )

            # Run transcription
            logger.info(f"Running transcription for {test_config.name}...")

            # Prepare media_info dict for pipeline
            media_info = {
                'path': str(self.audio_path),
                'basename': self.audio_path.stem,
                'file_number': 1,
                'total_files': 1
            }

            # Process and get result
            result_dict = pipeline.process(media_info)
            srt_path = result_dict.get('output_files', {}).get('final_srt')

            # Parse results
            subtitles = SubtitleMetrics.parse_srt_file(Path(srt_path))
            metrics = SubtitleMetrics.calculate_metrics(subtitles)

            # Copy output file for clarity (delete existing if present)
            output_srt = self.output_dir / f"{test_config.name}.srt"
            if output_srt.exists():
                output_srt.unlink()
            import shutil
            shutil.copy2(srt_path, output_srt)

            processing_time = time.time() - start_time

            result = TestResult(
                config_name=test_config.name,
                hypothesis=test_config.hypothesis,
                total_subtitles=metrics["total_count"],
                total_speech_duration_sec=metrics["total_duration_sec"],
                avg_subtitle_duration_sec=metrics["avg_duration_sec"],
                very_short_subtitles=metrics["very_short_count"],
                long_subtitles=metrics["long_count"],
                processing_time_sec=processing_time
            )

            logger.info(f"\n{'='*70}")
            logger.info(f"RESULTS for {test_config.name}:")
            logger.info(f"  Total subtitles: {result.total_subtitles}")
            logger.info(f"  Total speech: {result.total_speech_duration_sec:.1f}s")
            logger.info(f"  Avg duration: {result.avg_subtitle_duration_sec:.2f}s")
            logger.info(f"  Very short (<1s): {result.very_short_subtitles}")
            logger.info(f"  Long (>5s): {result.long_subtitles}")
            logger.info(f"  Processing time: {processing_time:.1f}s")
            logger.info(f"{'='*70}\n")

            return result

        except Exception as e:
            logger.error(f"Test failed: {test_config.name}", exc_info=True)
            processing_time = time.time() - start_time

            return TestResult(
                config_name=test_config.name,
                hypothesis=test_config.hypothesis,
                total_subtitles=0,
                total_speech_duration_sec=0.0,
                avg_subtitle_duration_sec=0.0,
                very_short_subtitles=0,
                long_subtitles=0,
                processing_time_sec=processing_time,
                error=str(e)
            )

    def run_all_tests(
        self,
        configs: List[TestConfig],
        skip_baseline: bool = False
    ) -> List[TestResult]:
        """
        Run all test configurations.

        Args:
            configs: List of test configurations to run
            skip_baseline: If True, skip baseline test (assumes already run)

        Returns:
            List of test results
        """
        results = []

        for i, config in enumerate(configs, 1):
            # Skip baseline if requested
            if skip_baseline and config.hypothesis == "baseline":
                logger.info(f"Skipping baseline test (--skip-baseline)")
                continue

            logger.info(f"\n\n### Test {i}/{len(configs)} ###\n")

            result = self.run_test(config)
            results.append(result)

            # Store baseline for comparison
            if config.hypothesis == "baseline":
                self.baseline_result = result

        self.results = results
        return results

    def compare_to_baseline(self, result: TestResult) -> Optional[ComparisonMetrics]:
        """Compare a result to baseline."""
        if not self.baseline_result or result.hypothesis == "baseline":
            return None

        baseline = self.baseline_result

        subtitle_delta_baseline = result.total_subtitles - baseline.total_subtitles
        duration_delta_baseline = result.total_speech_duration_sec - baseline.total_speech_duration_sec

        improvement_baseline_pct = (
            (subtitle_delta_baseline / baseline.total_subtitles * 100)
            if baseline.total_subtitles > 0 else 0.0
        )

        # Reference comparison
        subtitle_delta_ref = 0
        duration_delta_ref = 0.0
        improvement_ref_pct = 0.0

        if self.reference_metrics:
            ref_count = self.reference_metrics["total_count"]
            ref_duration = self.reference_metrics["total_duration_sec"]

            subtitle_delta_ref = result.total_subtitles - ref_count
            duration_delta_ref = result.total_speech_duration_sec - ref_duration
            improvement_ref_pct = (
                (subtitle_delta_ref / ref_count * 100)
                if ref_count > 0 else 0.0
            )

        return ComparisonMetrics(
            subtitle_delta_vs_baseline=subtitle_delta_baseline,
            subtitle_delta_vs_reference=subtitle_delta_ref,
            duration_delta_vs_baseline=duration_delta_baseline,
            duration_delta_vs_reference=duration_delta_ref,
            improvement_vs_baseline_pct=improvement_baseline_pct,
            improvement_vs_reference_pct=improvement_ref_pct
        )

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all test results.

        Returns:
            Dictionary with summary data
        """
        summary = {
            "audio_file": str(self.audio_path),
            "reference_srt": str(self.reference_srt_path) if self.reference_srt_path else None,
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r.success),
            "failed_tests": sum(1 for r in self.results if not r.success),
            "baseline": self.baseline_result.to_dict() if self.baseline_result else None,
            "reference_metrics": self.reference_metrics,
            "results": [],
            "recommendations": []
        }

        # Add all results with comparisons
        for result in self.results:
            result_dict = result.to_dict()

            # Add comparison metrics
            comparison = self.compare_to_baseline(result)
            if comparison:
                result_dict["comparison"] = asdict(comparison)

            summary["results"].append(result_dict)

        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations()

        return summary

    def _generate_recommendations(self) -> List[str]:
        """
        Analyze results and generate recommendations.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if not self.baseline_result or not self.results:
            return ["Not enough data to generate recommendations"]

        baseline_count = self.baseline_result.total_subtitles

        # Find configs that improved over baseline
        improvements = []
        for result in self.results:
            if result.hypothesis == "baseline" or not result.success:
                continue

            delta = result.total_subtitles - baseline_count
            if delta > 0:
                improvement_pct = (delta / baseline_count) * 100
                improvements.append((result, delta, improvement_pct))

        # Sort by improvement
        improvements.sort(key=lambda x: x[1], reverse=True)

        if not improvements:
            recommendations.append(
                "No configuration improved over baseline. "
                "The missing subtitles may be caused by factors not tested."
            )
        else:
            recommendations.append(f"Found {len(improvements)} configurations that improved over baseline:")

            for i, (result, delta, pct) in enumerate(improvements[:5], 1):
                recommendations.append(
                    f"  {i}. {result.config_name}: +{delta} subtitles (+{pct:.1f}%) - {result.hypothesis}"
                )

        # Hypothesis-specific analysis
        by_hypothesis = {}
        for result in self.results:
            if result.hypothesis == "baseline" or not result.success:
                continue

            if result.hypothesis not in by_hypothesis:
                by_hypothesis[result.hypothesis] = []

            delta = result.total_subtitles - baseline_count
            by_hypothesis[result.hypothesis].append((result, delta))

        # Find most impactful hypothesis
        hypothesis_impacts = {}
        for hyp, results_list in by_hypothesis.items():
            max_delta = max(delta for _, delta in results_list)
            hypothesis_impacts[hyp] = max_delta

        if hypothesis_impacts:
            best_hypothesis = max(hypothesis_impacts.items(), key=lambda x: x[1])
            if best_hypothesis[1] > 0:
                recommendations.append(
                    f"\nMost impactful hypothesis: {best_hypothesis[0]} "
                    f"(max improvement: +{best_hypothesis[1]} subtitles)"
                )

        return recommendations

    def save_results(self, output_file: Path):
        """Save results to JSON file."""
        summary = self.generate_summary()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResults saved to: {output_file}")

    def print_summary_table(self):
        """Print a summary table of all results."""
        if not self.results:
            logger.info("No results to display")
            return

        print("\n" + "="*100)
        print("HYPOTHESIS TEST SUMMARY")
        print("="*100)

        # Header
        print(f"{'Config':<30} {'Hypothesis':<20} {'Subs':>6} {'Duration':>10} {'Avg':>8} {'<1s':>5} {'>5s':>5}")
        print("-"*100)

        # Baseline first
        if self.baseline_result:
            r = self.baseline_result
            print(
                f"{r.config_name:<30} {r.hypothesis:<20} {r.total_subtitles:>6} "
                f"{r.total_speech_duration_sec:>9.1f}s {r.avg_subtitle_duration_sec:>7.2f}s "
                f"{r.very_short_subtitles:>5} {r.long_subtitles:>5}"
            )
            print("-"*100)

        # All other results
        for result in self.results:
            if result.hypothesis == "baseline":
                continue

            comparison = self.compare_to_baseline(result)
            delta_str = ""
            if comparison:
                delta = comparison.subtitle_delta_vs_baseline
                if delta > 0:
                    delta_str = f" (+{delta})"
                elif delta < 0:
                    delta_str = f" ({delta})"

            print(
                f"{result.config_name:<30} {result.hypothesis:<20} {result.total_subtitles:>6}{delta_str:<8} "
                f"{result.total_speech_duration_sec:>9.1f}s {result.avg_subtitle_duration_sec:>7.2f}s "
                f"{result.very_short_subtitles:>5} {result.long_subtitles:>5}"
            )

        print("="*100)

        # Reference comparison if available
        if self.reference_metrics:
            print(f"\nReference (v1.7.1): {self.reference_metrics['total_count']} subtitles, "
                  f"{self.reference_metrics['total_duration_sec']:.1f}s total speech")

        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            print("\nRECOMMENDATIONS:")
            print("-"*100)
            for rec in recommendations:
                print(rec)
            print()


def main():
    """Main entry point for hypothesis testing."""
    parser = argparse.ArgumentParser(
        description="Hypothesis testing suite for ASR regression investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all hypothesis tests
  python hypothesis_test_suite.py --audio subset.wav --reference v1.7.1.srt

  # Run specific hypothesis
  python hypothesis_test_suite.py --audio subset.wav --hypothesis vad_params

  # Quick mode (fewer tests)
  python hypothesis_test_suite.py --audio subset.wav --quick

  # Skip baseline (if already run)
  python hypothesis_test_suite.py --audio subset.wav --skip-baseline
        """
    )

    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to test audio file (25-minute subset)"
    )

    parser.add_argument(
        "--reference",
        type=Path,
        help="Path to reference v1.7.1 SRT file for comparison"
    )

    parser.add_argument(
        "--hypothesis",
        choices=["vad_params", "asr_duration_filter", "temperature_fallback", "patience_beam", "compute_type", "all"],
        default="all",
        help="Specific hypothesis to test (default: all)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test suite (fewer variations)"
    )

    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline test (use if already run)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./hypothesis_outputs"),
        help="Output directory for SRT files (default: ./hypothesis_outputs)"
    )

    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("./hypothesis_temp"),
        help="Temporary file directory (default: ./hypothesis_temp)"
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("./hypothesis_results.json"),
        help="Output file for JSON results (default: ./hypothesis_results.json)"
    )

    args = parser.parse_args()

    # Validate audio file exists
    if not args.audio.exists():
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    # Validate reference if provided
    if args.reference and not args.reference.exists():
        logger.warning(f"Reference SRT not found: {args.reference}")
        args.reference = None

    # Build test configuration list
    if args.quick:
        logger.info("Running quick test suite")
        configs = HypothesisConfigs.get_quick_suite()
    elif args.hypothesis == "all":
        logger.info("Running all hypothesis tests")
        configs = list_all_configs()
    else:
        logger.info(f"Running hypothesis: {args.hypothesis}")
        all_hypotheses = HypothesisConfigs.get_all_hypotheses()
        configs = [HypothesisConfigs.get_baseline()] + all_hypotheses[args.hypothesis]

    logger.info(f"Total tests to run: {len(configs)}")

    # Create tester
    tester = HypothesisTester(
        audio_path=args.audio,
        reference_srt_path=args.reference,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir
    )

    # Run tests
    try:
        tester.run_all_tests(configs, skip_baseline=args.skip_baseline)

        # Print summary
        tester.print_summary_table()

        # Save results
        tester.save_results(args.results_file)

        logger.info("\n✓ Hypothesis testing complete!")

    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user. Saving partial results...")
        tester.save_results(args.results_file)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
