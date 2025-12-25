"""
Generate ASCII visualizations for the forensic analysis
"""

import sys

# Force UTF-8 encoding for Windows console output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def print_bar_chart(data, title, max_width=60):
    """Print a simple ASCII bar chart"""
    print(f"\n{title}")
    print("=" * (max_width + 20))

    if not data:
        print("No data")
        return

    max_value = max(value for _, value in data)

    for label, value in data:
        bar_width = int((value / max_value) * max_width) if max_value > 0 else 0
        bar = "█" * bar_width
        print(f"{label:<20} {bar} {value:.1f}%")


def print_timeline(v171_start, v173_start, halluc_end):
    """Print timeline comparison"""
    print("\nTIMELINE COMPARISON: First Real Speech Detection")
    print("=" * 80)
    print()
    print("v1.7.1:  [START]──────2:28────────5:53 (First Real Speech)")
    print("         |         (halluc)    |")
    print("         |                     ↓")
    print("         0:00                  Real dialogue begins")
    print()
    print("v1.7.3:  [START]─────────────────────9:35 (First Real Speech)")
    print("         |                           |")
    print("         |                           ↓")
    print("         0:00                        Real dialogue begins")
    print()
    print("         ╔═══════════════════════════════════════╗")
    print("         ║  3.7 MINUTE GAP in speech detection   ║")
    print("         ╚═══════════════════════════════════════╝")
    print()


def print_segment_map():
    """Print visual map of missing rate by segment"""
    print("\nMISSING RATE HEATMAP BY VIDEO SEGMENT")
    print("=" * 80)
    print()

    segments = [
        ("Seg 1  (0:00-13:43)", 95.2),
        ("Seg 2  (13:43-27:26)", 17.9),
        ("Seg 3  (27:26-41:09)", 13.9),
        ("Seg 4  (41:09-54:52)", 11.2),
        ("Seg 5  (54:52-68:35)", 29.8),
        ("Seg 6  (68:35-82:18)", 11.8),
        ("Seg 7  (82:18-96:01)", 25.5),
        ("Seg 8  (96:01-109:44)", 9.1),
        ("Seg 9  (109:44-123:27)", 36.8),
        ("Seg 10 (123:27-137:10)", 67.2),
    ]

    print("Segment              Missing Rate")
    print("-" * 50)

    for seg_name, missing_pct in segments:
        # Visual bar
        max_width = 30
        bar_width = int((missing_pct / 100) * max_width)
        bar = "█" * bar_width

        # Color coding (for terminals that support it)
        if missing_pct >= 60:
            marker = "🔴"
        elif missing_pct >= 30:
            marker = "🟡"
        else:
            marker = "🟢"

        print(f"{seg_name:<20} {bar:<30} {missing_pct:5.1f}% {marker}")

    print()
    print("Legend: 🔴 High Loss (≥60%)  🟡 Medium (30-60%)  🟢 Low (<30%)")


def print_duration_analysis():
    """Print duration-based missing rate analysis"""
    print("\nMISSING RATE BY SUBTITLE DURATION")
    print("=" * 80)
    print()

    categories = [
        ("Very Short (<1s)", 33.8, 44, 86),
        ("Short (1-2s)", 21.4, 62, 228),
        ("Medium (2-5s)", 28.0, 60, 154),
        ("Long (≥5s)", 13.0, 3, 20),
    ]

    print("Category              Missing   Matched   Total    Missing %")
    print("-" * 65)

    for cat_name, missing_pct, missing_count, matched_count in categories:
        total = missing_count + matched_count
        bar_width = int((missing_pct / 40) * 20)  # Scale to 40% max
        bar = "█" * bar_width
        print(f"{cat_name:<20}  {missing_count:<8}  {matched_count:<8}  {total:<7}  {bar:<20} {missing_pct:.1f}%")

    print()
    print("Key Finding: Very short segments (<1s) are 2.6x more likely to be missing")
    print("             than long segments (≥5s)")


def print_hallucination_summary():
    """Print hallucination removal summary"""
    print("\nHALLUCINATION REMOVAL EFFECTIVENESS")
    print("=" * 80)
    print()

    print("Overall Statistics:")
    print("-" * 50)
    print(f"v1.7.1:  35 hallucinations / 657 total  =  5.3%")
    print(f"v1.7.3:  11 hallucinations / 529 total  =  2.1%")
    print()
    print(f"Reduction: 24 hallucinations removed (68.6% reduction in halluc rate)")
    print()

    print("Top Hallucination Patterns Removed:")
    print("-" * 50)

    patterns = [
        ("私たちの家族 (Our family)", 10, 1, 90),
        ("あなた.*愛して (Love you)", 7, 0, 100),
        ("私.*あなた.*大好き (I love you)", 6, 0, 100),
        ("電子レンジ.* (Microwave...)", 5, 1, 80),
        ("次回.*ビデオ.* (Next video)", 3, 0, 100),
    ]

    for pattern, v171, v173, reduction_pct in patterns:
        reduction = v171 - v173
        arrow = "→"
        print(f"{pattern:<40} {v171} {arrow} {v173}  ({reduction_pct}% reduction)")

    print()
    print("✓ Hallucination removal is working correctly")
    print("✓ This accounts for 18.8% of total subtitle reduction")


def print_recommendation_box():
    """Print recommendation summary"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "RECOMMENDATION SUMMARY" + " " * 36 + "║")
    print("╠" + "═" * 78 + "╣")
    print("║                                                                              ║")
    print("║  v1.7.3 is LIKELY MORE ACCURATE overall due to improved hallucination       ║")
    print("║  removal, but two areas need investigation:                                 ║")
    print("║                                                                              ║")
    print("║  1. Opening Segment (0:00-13:43): 95% loss rate                             ║")
    print("║     → Manually verify if v1.7.3 is correctly skipping intro music/credits   ║")
    print("║                                                                              ║")
    print("║  2. Very Short Segments (<1s): 34% loss rate                                ║")
    print("║     → Investigate if real speech is being lost or just better merged        ║")
    print("║                                                                              ║")
    print("║  Recommended 25-minute test subset: 112:00 - 137:00                         ║")
    print("║     → Contains 56% missing rate, good for targeted debugging                ║")
    print("║                                                                              ║")
    print("╚" + "═" * 78 + "╝")
    print()


def print_breakdown_pie():
    """Print breakdown of the 128 missing subtitles"""
    print("\nBREAKDOWN OF 128 MISSING SUBTITLES")
    print("=" * 80)
    print()

    components = [
        ("Hallucinations (correctly removed)", 24, 18.8),
        ("Segment 1 loss (opening)", 19, 14.8),
        ("Segment 10 loss (ending)", 20, 15.6),
        ("Duration bias + other factors", 65, 50.8),
    ]

    print("Component                              Count    Percentage")
    print("-" * 65)

    total = 128
    for desc, count, pct in components:
        bar_width = int((pct / 100) * 30)
        bar = "█" * bar_width
        print(f"{desc:<38} {count:<8} {bar:<30} {pct:.1f}%")

    print()
    print("Key Insight: Only 18.8% is explained by hallucination removal.")
    print("            The remaining 81.2% needs further investigation.")


def main():
    print("\n" + "=" * 80)
    print(" " * 20 + "FORENSIC ANALYSIS VISUAL SUMMARY")
    print(" " * 25 + "v1.7.1 vs v1.7.3")
    print("=" * 80)

    print_timeline(148, 575, 353)
    print_segment_map()
    print_duration_analysis()
    print_hallucination_summary()
    print_breakdown_pie()
    print_recommendation_box()

    print("\nFor detailed analysis, see: scripts/FORENSIC_ANALYSIS_REPORT.md")
    print("=" * 80)


if __name__ == '__main__':
    main()
