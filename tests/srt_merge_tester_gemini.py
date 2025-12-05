import argparse
import sys
import logging
from typing import List, Optional, Tuple
from datetime import timedelta

# Third-party libraries
try:
    import pysrt
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich.logging import RichHandler
except ImportError as e:
    sys.exit(f"Critical Error: Missing dependency. Please run: pip install pysrt rich\nDetails: {e}")

# Initialize Rich Console
console = Console()

# Configure Logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)
log = logging.getLogger("SRTVerifier")

class SRTVerifier:
    """
    A robust utility to verify the correctness of merged SRT files.
    """

    def __init__(self, path_srt1: str, path_srt2: str, path_merged: str):
        self.path_srt1 = path_srt1
        self.path_srt2 = path_srt2
        self.path_merged = path_merged
        
        self.subs1 = self._load_file(path_srt1)
        self.subs2 = self._load_file(path_srt2)
        self.subs_merged = self._load_file(path_merged)

    def _load_file(self, path: str) -> pysrt.SubRipFile:
        """Loads an SRT file with UTF-8 encoding handling."""
        try:
            # encoding='utf-8-sig' handles UTF-8 with or without BOM (common in Windows)
            subs = pysrt.open(path, encoding='utf-8-sig')
            log.info(f"Loaded [bold cyan]{path}[/]: {len(subs)} subtitles.")
            return subs
        except Exception as e:
            console.print(f"[bold red]Failed to load file:[/bold red] {path}")
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    def verify(self, mode: int):
        """Dispatches the verification logic based on the selected mode."""
        console.print(Panel.fit(f"[bold yellow]Starting Verification - Mode {mode}[/bold yellow]"))
        
        expected_subs = []

        if mode == 3:
            expected_subs = self._strategy_combine()
        elif mode in [1, 2, 4, 5]:
            base_subs = self.subs1 if mode in [1, 4] else self.subs2
            fill_subs = self.subs2 if mode in [1, 4] else self.subs1
            allow_overlap = True if mode in [4, 5] else False
            
            expected_subs = self._strategy_fill_gaps(base_subs, fill_subs, allow_overlap)
        else:
            log.error(f"Unknown mode: {mode}")
            sys.exit(1)

        # Post-processing: Sort and Re-index expected output
        expected_subs.sort(key=lambda x: x.start)
        self._reindex(expected_subs)

        # Final Comparison
        self._compare_results(expected_subs, self.subs_merged)

    def _strategy_combine(self) -> List[pysrt.SubRipItem]:
        """Mode 3: Simple combination of both files."""
        log.info("Calculating expected result: [bold]Combine All[/bold]")
        combined = list(self.subs1) + list(self.subs2)
        return combined

    def _strategy_fill_gaps(self, base: pysrt.SubRipFile, fill: pysrt.SubRipFile, allow_partial_overlap: bool) -> List[pysrt.SubRipItem]:
        """
        Modes 1, 2, 4, 5: Base file + Fill gaps from second file.
        allow_partial_overlap: If True, allows overlap < 30%.
        """
        strategy_name = "Base + Fill (Partial Overlap)" if allow_partial_overlap else "Base + Fill (Strict)"
        log.info(f"Calculating expected result: [bold]{strategy_name}[/bold]")

        result = list(base)
        
        # We need to check every candidate in the 'fill' file against the 'base' file
        # This is O(N*M), but for SRT files (usually < 2000 lines) it's acceptable for a test utility.
        # Ideally, an interval tree would be used for massive datasets, but lists are fine here.
        
        added_count = 0
        rejected_count = 0

        for candidate in track(fill, description="Processing Fill Candidates..."):
            is_conflicting = False
            
            for existing in base:
                if self._check_overlap(existing, candidate, allow_partial_overlap):
                    is_conflicting = True
                    break
            
            if not is_conflicting:
                result.append(candidate)
                added_count += 1
            else:
                rejected_count += 1

        log.info(f"Stats: Base Size: {len(base)} | Added from Fill: {added_count} | Rejected: {rejected_count}")
        return result

    def _check_overlap(self, s1: pysrt.SubRipItem, s2: pysrt.SubRipItem, allow_partial: bool) -> bool:
        """
        Determines if two subtitles conflict based on the overlap rules.
        Returns True if there is a 'conflict' (meaning we should NOT add s2).
        """
        # Convert to absolute milliseconds for easier math
        start1, end1 = s1.start.ordinal, s1.end.ordinal
        start2, end2 = s2.start.ordinal, s2.end.ordinal

        # Check if they actually overlap in time
        # Logic: max(starts) < min(ends)
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return False # No physical overlap

        # If we are here, there is an overlap.
        
        # Strict mode (Modes 1 & 2): Any overlap is a conflict.
        if not allow_partial:
            return True

        # Partial mode (Modes 4 & 5): 
        # "Allow 30% overlap (if two subtitles have overlap which is less than 30% of duration)"
        # We calculate overlap duration against the duration of the candidate subtitle (s2)
        # or the existing one? Usually, strictly speaking, if the interference is minor 
        # relative to the subtitle being inserted, it might be acceptable.
        # However, to be robust, we usually check against the *minimum* duration of the two involved.
        
        overlap_duration = overlap_end - overlap_start
        duration1 = end1 - start1
        duration2 = end2 - start2
        
        # Using the smaller duration ensures the overlap isn't significant for EITHER subtitle.
        min_duration = min(duration1, duration2)
        
        if min_duration == 0: return True # Avoid div by zero, treat as conflict

        overlap_percentage = (overlap_duration / min_duration) * 100

        # If overlap is less than 30%, it is ALLOWED.
        # Therefore, if overlap >= 30%, it is a CONFLICT.
        if overlap_percentage >= 30.0:
            return True # Conflict
        
        return False # Acceptable overlap

    def _reindex(self, subs: List[pysrt.SubRipItem]):
        """Re-indexes the list sequentially starting from 1."""
        for index, sub in enumerate(subs, 1):
            sub.index = index

    def _compare_results(self, expected: List[pysrt.SubRipItem], actual: pysrt.SubRipFile):
        """
        Deep comparison of Expected vs Actual.
        Checks: Total count, Start Time, End Time, and Text content.
        """
        console.print(Panel("Comparing Expected Result vs Actual Output", style="blue"))

        errors = 0
        
        # 1. Count Check
        if len(expected) != len(actual):
            console.print(f"[bold red]FAIL:[/bold red] Count Mismatch. Expected {len(expected)}, Got {len(actual)}")
            errors += 1
        else:
            console.print(f"[green]PASS:[/green] Subtitle counts match ({len(actual)}).")

        # 2. Content Check
        # We iterate up to the shorter length to avoid IndexErrors, though count check handles major diffs.
        limit = min(len(expected), len(actual))
        
        table = Table(title="Discrepancies Found", show_header=True, header_style="bold magenta")
        table.add_column("Index", style="dim")
        table.add_column("Type")
        table.add_column("Expected")
        table.add_column("Actual")

        discrepancy_found = False

        for i in range(limit):
            exp_sub = expected[i]
            act_sub = actual[i]

            # Compare Times (ordinal converts to total milliseconds)
            if exp_sub.start.ordinal != act_sub.start.ordinal:
                table.add_row(str(i+1), "Start Time", str(exp_sub.start), str(act_sub.start))
                discrepancy_found = True
                errors += 1
            
            if exp_sub.end.ordinal != act_sub.end.ordinal:
                table.add_row(str(i+1), "End Time", str(exp_sub.end), str(act_sub.end))
                discrepancy_found = True
                errors += 1

            # Compare Text (strip whitespace to be lenient on trailing newlines)
            if exp_sub.text.strip() != act_sub.text.strip():
                # Show first 20 chars for brevity if needed
                table.add_row(str(i+1), "Text", exp_sub.text[:30]+"...", act_sub.text[:30]+"...")
                discrepancy_found = True
                errors += 1

        if discrepancy_found:
            console.print(table)
            console.print(f"[bold red]VERIFICATION FAILED with {errors} errors.[/bold red]")
            sys.exit(1)
        elif errors == 0:
            console.print(Panel(f"[bold green]SUCCESS[/bold green]\nThe merged file '{self.path_merged}' is exactly as expected.", border_style="green"))
            sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="SRT Merge Verification Utility")
    parser.add_argument("srt1", help="Path to first SRT file")
    parser.add_argument("srt2", help="Path to second SRT file")
    parser.add_argument("merged", help="Path to the merged SRT file to verify")
    parser.add_argument("mode", type=int, choices=[1, 2, 3, 4, 5], 
                        help="Merge Mode Used: \n"
                             "1: Base SRT1, Fill SRT2 (No Overlap)\n"
                             "2: Base SRT2, Fill SRT1 (No Overlap)\n"
                             "3: Combine All\n"
                             "4: Base SRT1, Fill SRT2 (Allow <30% Overlap)\n"
                             "5: Base SRT2, Fill SRT1 (Allow <30% Overlap)")

    args = parser.parse_args()

    verifier = SRTVerifier(args.srt1, args.srt2, args.merged)
    verifier.verify(args.mode)

if __name__ == "__main__":
    main()