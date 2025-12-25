#!/usr/bin/env python3
"""
Inspect Hypothesis Configurations

This utility script lets you inspect test configurations without running tests.
Useful for understanding what parameters each test will modify.

Usage:
    # List all configurations
    python inspect_hypothesis_configs.py

    # List specific hypothesis
    python inspect_hypothesis_configs.py --hypothesis vad_params

    # Show detailed parameter differences
    python inspect_hypothesis_configs.py --detailed

    # Export to JSON
    python inspect_hypothesis_configs.py --json configs.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hypothesis_configs import (
    HypothesisConfigs,
    TestConfig,
    V173Defaults,
    list_all_configs
)


def print_config_summary(config: TestConfig, detailed: bool = False):
    """Print a summary of a test configuration."""
    print(f"\n{'='*70}")
    print(f"Config: {config.name}")
    print(f"{'='*70}")
    print(f"Hypothesis: {config.hypothesis}")
    print(f"Description: {config.description}")

    if config.params_override:
        print(f"\nParameter Overrides:")
        for section, params in config.params_override.items():
            print(f"  [{section}]")
            for key, value in params.items():
                print(f"    {key} = {value}")
    else:
        print(f"\nParameter Overrides: None (baseline)")

    if detailed:
        print(f"\nFull Configuration (with defaults):")
        full_config = build_full_config(config)
        print(json.dumps(full_config, indent=2))


def build_full_config(config: TestConfig) -> Dict[str, Any]:
    """Build full configuration by merging overrides with defaults."""
    full_config = {
        "asr": dict(V173Defaults.ASR),
        "vad": dict(V173Defaults.VAD)
    }

    # Apply overrides
    for section, overrides in config.params_override.items():
        if section in full_config:
            full_config[section].update(overrides)

    return full_config


def list_by_hypothesis(hypothesis: str = None, detailed: bool = False):
    """List configurations grouped by hypothesis."""
    all_hypotheses = HypothesisConfigs.get_all_hypotheses()

    if hypothesis:
        if hypothesis not in all_hypotheses:
            print(f"ERROR: Unknown hypothesis '{hypothesis}'")
            print(f"Available: {', '.join(all_hypotheses.keys())}")
            sys.exit(1)

        hypotheses_to_show = {hypothesis: all_hypotheses[hypothesis]}
    else:
        hypotheses_to_show = all_hypotheses

    for hyp_name, configs in hypotheses_to_show.items():
        print(f"\n{'#'*70}")
        print(f"# HYPOTHESIS: {hyp_name.upper()}")
        print(f"{'#'*70}")

        for config in configs:
            print_config_summary(config, detailed)

        print(f"\nTotal configs in '{hyp_name}': {len(configs)}")


def export_to_json(output_file: Path):
    """Export all configurations to JSON."""
    all_configs = list_all_configs()

    export_data = {
        "total_configs": len(all_configs),
        "defaults": {
            "asr": V173Defaults.ASR,
            "vad": V173Defaults.VAD
        },
        "configurations": []
    }

    for config in all_configs:
        config_data = {
            "name": config.name,
            "description": config.description,
            "hypothesis": config.hypothesis,
            "params_override": config.params_override,
            "full_config": build_full_config(config)
        }
        export_data["configurations"].append(config_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Exported {len(all_configs)} configurations to {output_file}")


def print_statistics():
    """Print statistics about test configurations."""
    all_hypotheses = HypothesisConfigs.get_all_hypotheses()
    quick_suite = HypothesisConfigs.get_quick_suite()

    print("\n" + "="*70)
    print("HYPOTHESIS TEST SUITE STATISTICS")
    print("="*70)

    print(f"\nTotal Hypotheses: {len(all_hypotheses)}")
    for hyp_name, configs in all_hypotheses.items():
        print(f"  {hyp_name}: {len(configs)} configs")

    print(f"\nTotal Configurations: {len(list_all_configs())}")
    print(f"Quick Suite Size: {len(quick_suite)}")

    print(f"\nEstimated Time (@ 3 min/test):")
    print(f"  Full suite: {len(list_all_configs()) * 3} minutes")
    print(f"  Quick suite: {len(quick_suite) * 3} minutes")


def compare_configs(config1_name: str, config2_name: str):
    """Compare two configurations side-by-side."""
    all_configs = {c.name: c for c in list_all_configs()}

    if config1_name not in all_configs:
        print(f"ERROR: Config '{config1_name}' not found")
        sys.exit(1)
    if config2_name not in all_configs:
        print(f"ERROR: Config '{config2_name}' not found")
        sys.exit(1)

    config1 = all_configs[config1_name]
    config2 = all_configs[config2_name]

    full1 = build_full_config(config1)
    full2 = build_full_config(config2)

    print(f"\n{'='*70}")
    print(f"COMPARING: {config1_name} vs {config2_name}")
    print(f"{'='*70}")

    # Find differences
    all_sections = set(full1.keys()) | set(full2.keys())

    differences_found = False

    for section in sorted(all_sections):
        params1 = full1.get(section, {})
        params2 = full2.get(section, {})

        all_params = set(params1.keys()) | set(params2.keys())

        section_diffs = []
        for param in sorted(all_params):
            val1 = params1.get(param)
            val2 = params2.get(param)

            if val1 != val2:
                section_diffs.append((param, val1, val2))

        if section_diffs:
            differences_found = True
            print(f"\n[{section}]")
            print(f"{'Parameter':<30} {config1_name:<20} {config2_name:<20}")
            print("-"*70)
            for param, val1, val2 in section_diffs:
                print(f"{param:<30} {str(val1):<20} {str(val2):<20}")

    if not differences_found:
        print("\nNo differences found between these configurations.")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect hypothesis test configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--hypothesis",
        choices=["baseline", "vad_params", "asr_duration_filter", "temperature_fallback", "patience_beam"],
        help="Show only specific hypothesis"
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed configuration including all parameters"
    )

    parser.add_argument(
        "--json",
        type=Path,
        metavar="FILE",
        help="Export configurations to JSON file"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics about test suite"
    )

    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("CONFIG1", "CONFIG2"),
        help="Compare two configurations"
    )

    args = parser.parse_args()

    # Handle different modes
    if args.stats:
        print_statistics()

    elif args.compare:
        compare_configs(args.compare[0], args.compare[1])

    elif args.json:
        export_to_json(args.json)

    else:
        list_by_hypothesis(args.hypothesis, args.detailed)


if __name__ == "__main__":
    main()
