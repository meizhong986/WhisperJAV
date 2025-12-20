import argparse
import json
import re
import sys
from pathlib import Path

class SRTStitcher:
    def __init__(self, manifest_path, srt_folder, output_folder=None, video_name=None):
        self.manifest_path = Path(manifest_path)
        self.srt_folder = Path(srt_folder)
        
        # Load Manifest
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.chunks = data['chunks']
            # Fallback to manifest video name if not provided in CLI
            self.source_video_name = video_name or Path(data.get('source_video', 'output')).stem

        self.output_folder = Path(output_folder) if output_folder else self.srt_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def run(self):
        print(f"--- Stitching Subtitles for: {self.source_video_name} ---")
        
        # 1. Detect Variants based on chunk_0000
        variants = self._detect_variants()
        if not variants:
            print("Error: No SRT files found for chunk_0000 in the specified folder.")
            return

        print(f"Detected {len(variants)} subtitle variants: {', '.join(variants)}")

        # 2. Process each variant
        for suffix in variants:
            self._stitch_variant(suffix)

    def _detect_variants(self):
        """
        Scans the folder for files matching chunk_0000*.srt to determine naming patterns.
        Returns a list of suffixes (e.g., '.ja.pass1.srt')
        """
        # We assume the chunks follow the ID format from the manifest (usually 0)
        first_chunk_id = self.chunks[0]['id']
        pattern = f"chunk_{first_chunk_id:04d}*.srt"
        
        files = list(self.srt_folder.glob(pattern))
        variants = []
        
        for f in files:
            # Extract everything after "chunk_0000"
            # e.g., "chunk_0000.ja.pass1.srt" -> ".ja.pass1.srt"
            suffix = f.name.replace(f"chunk_{first_chunk_id:04d}", "")
            variants.append(suffix)
            
        return variants

    def _stitch_variant(self, suffix):
        """
        Stitches all chunks for a specific variant suffix into one master SRT.
        """
        output_filename = f"{self.source_video_name}{suffix}"
        output_path = self.output_folder / output_filename
        
        print(f"Generating: {output_filename}...")
        
        global_counter = 1
        all_lines = []

        for chunk in self.chunks:
            chunk_id = chunk['id']
            offset = chunk['global_start']
            
            # Construct expected filename for this chunk
            chunk_filename = f"chunk_{chunk_id:04d}{suffix}"
            chunk_path = self.srt_folder / chunk_filename
            
            if not chunk_path.exists():
                print(f"  [Warning] Missing file: {chunk_filename}. Skipping this segment.")
                continue
            
            # Parse and Shift
            srt_blocks = self._parse_and_shift(chunk_path, offset)
            
            # Re-serialize with new index
            for start_str, end_str, text in srt_blocks:
                all_lines.append(f"{global_counter}\n")
                all_lines.append(f"{start_str} --> {end_str}\n")
                all_lines.append(f"{text}\n\n")
                global_counter += 1

        # Write to disk
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(all_lines)
        
        print(f"  -> Saved {global_counter-1} subtitles to {output_path}")

    def _parse_and_shift(self, filepath, offset):
        """
        Reads an SRT, adds 'offset' to timestamps, returns list of (start, end, text).
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Regex to find SRT blocks. 
        # Group 1: Index (ignored)
        # Group 2: Start Time
        # Group 3: End Time
        # Group 4: Text Content
        pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2}[,.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,.]\d{3})\n((?:.|\n)*?)(?=\n\d+\n|\Z)', re.MULTILINE)
        
        matches = pattern.findall(content)
        result = []
        
        for _, start_raw, end_raw, text in matches:
            # Shift Timestamps
            new_start = self._add_offset(start_raw, offset)
            new_end = self._add_offset(end_raw, offset)
            result.append((new_start, new_end, text.strip()))
            
        return result

    def _add_offset(self, time_str, offset_seconds):
        """
        Parses HH:MM:SS,mmm string, adds offset seconds, returns new string.
        """
        # Handle both comma (standard) and dot (some ASRs)
        time_str = time_str.replace('.', ',')
        hours, minutes, seconds_part = time_str.split(':')
        seconds, millis = seconds_part.split(',')
        
        total_seconds = (int(hours) * 3600) + (int(minutes) * 60) + int(seconds) + (int(millis) / 1000.0)
        
        new_total = total_seconds + offset_seconds
        
        # Convert back
        new_h = int(new_total // 3600)
        rem = new_total % 3600
        new_m = int(rem // 60)
        new_s_float = rem % 60
        new_s = int(new_s_float)
        new_ms = int((new_s_float - new_s) * 1000)
        
        return f"{new_h:02d}:{new_m:02d}:{new_s:02d},{new_ms:03d}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch split SRT chunks back into a single synchronized file.")
    
    parser.add_argument("manifest", help="Path to the manifest.json file")
    parser.add_argument("srt_folder", help="Folder containing the chunk SRT files")
    parser.add_argument("-o", "--output", help="Folder to save merged SRTs (default: same as srt_folder)")
    parser.add_argument("-n", "--name", help="Original video name (optional, overrides manifest)")

    args = parser.parse_args()

    if not Path(args.manifest).exists():
        print("Error: Manifest file not found.")
        sys.exit(1)
        
    stitcher = SRTStitcher(args.manifest, args.srt_folder, args.output, args.name)
    stitcher.run()