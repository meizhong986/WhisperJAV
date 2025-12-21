import subprocess
import json
import re
import os
import shutil
import argparse
import sys
from pathlib import Path

# ==========================================
# PART 1: The Chunker (Video -> Audio Chunks)
# ==========================================
class RobustSceneChunker:
    def __init__(self, input_video, output_dir="chunks", min_scene_len=5.0, max_scene_len=30.0, threshold=0.35):
        self.input_video = Path(input_video)
        self.output_dir = Path(output_dir)
        self.min_scene_len = min_scene_len
        self.max_scene_len = max_scene_len
        self.threshold = threshold
        self.wav_path = self.output_dir / "temp_locked_audio.wav"
        
        if self.output_dir.exists():
            print(f"[Chunker] Warning: Cleaning output directory '{self.output_dir}'...")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

    def run(self):
        print(f"--- [Mode: Chunk] Processing: {self.input_video.name} ---")
        print(f"Config: Min={self.min_scene_len}s, Max={self.max_scene_len}s, Threshold={self.threshold}")
        
        # 1. Extract PTS-Locked Audio
        print("[1/4] Extracting PTS-locked audio (CPU bound)...")
        self._extract_wav_locked()

        # 2. Detect Scenes
        print("[2/4] Detecting video scenes...")
        scenes = self._detect_scenes()
        
        # 3. Optimize Segments
        print(f"[3/4] Optimizing {len(scenes)} scenes...")
        segments = self._optimize_segments(scenes)

        # 4. Slice Audio
        print(f"[4/4] Slicing audio into {len(segments)} chunks...")
        manifest = self._slice_audio(segments)

        # Write Manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"--- Done! Manifest saved to {manifest_path} ---")

    def _extract_wav_locked(self):
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-fflags", "+genpts", "-copyts", "-start_at_zero",
            "-i", str(self.input_video),
            "-vn", "-af", "aresample=async=1:first_pts=0",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(self.wav_path)
        ]
        subprocess.run(cmd, check=True)

    def _detect_scenes(self):
        cmd = [
            "ffmpeg", "-v", "error", "-i", str(self.input_video),
            "-filter_complex", f"select='gt(scene,{self.threshold})',metadata=print:file=-",
            "-f", "null", "-"
        ]
        # Capture stdout and stderr because ffmpeg metadata printing varies by version
        result = subprocess.run(cmd, capture_output=True, text=True)
        timestamps = [0.0]
        matches = re.findall(r'pts_time:([0-9.]+)', result.stdout + result.stderr)
        for t in matches:
            timestamps.append(float(t))
        return sorted(list(set(timestamps)))

    def _optimize_segments(self, scene_points):
        segments = []
        # Get duration
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
               "-of", "default=noprint_wrappers=1:nokey=1", str(self.wav_path)]
        total_duration = float(subprocess.check_output(cmd).strip())

        if not scene_points or scene_points[-1] < total_duration:
            scene_points.append(total_duration)

        current_start = scene_points[0]
        for i in range(1, len(scene_points)):
            current_end = scene_points[i]
            segment_len = current_end - current_start
            
            # Logic: Merge short, Split long
            if segment_len >= self.min_scene_len:
                # Good length. Commit it.
                # (Note: strictly enforcing max_scene_len via forced splits is complex;
                # here we prioritize scene integrity but you can add forced splits if needed)
                segments.append((current_start, current_end))
                current_start = current_end
            else:
                # Too short. Check if merging exceeds max
                next_boundary = scene_points[i+1] if i+1 < len(scene_points) else total_duration
                if (next_boundary - current_start) > self.max_scene_len:
                    # Merging makes it too huge. Force the short cut.
                    segments.append((current_start, current_end))
                    current_start = current_end
                else:
                    # Merge allowed (skip updating current_start)
                    pass
        
        if current_start < total_duration:
             segments.append((current_start, total_duration))
        return segments

    def _slice_audio(self, segments):
        manifest = {
            "source_video": str(self.input_video.absolute()),
            "chunks": []
        }
        for i, (start, end) in enumerate(segments):
            duration = end - start
            filename = f"chunk_{i:04d}.wav"
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(self.wav_path),
                "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
                str(self.output_dir / filename)
            ]
            subprocess.run(cmd, check=True)
            manifest["chunks"].append({
                "id": i, "file": filename,
                "global_start": round(start, 3), "global_end": round(end, 3),
                "duration": round(duration, 3)
            })
        return manifest

# ==========================================
# PART 2: The Stitcher (SRT Chunks -> Master SRT)
# ==========================================
class SRTStitcher:
    def __init__(self, manifest_path, srt_folder, output_folder=None, video_name=None):
        self.manifest_path = Path(manifest_path)
        self.srt_folder = Path(srt_folder)
        
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.chunks = data['chunks']
            self.source_video_name = video_name or Path(data.get('source_video', 'output')).stem

        self.output_folder = Path(output_folder) if output_folder else self.srt_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def run(self):
        print(f"--- [Mode: Stitch] Video: {self.source_video_name} ---")
        variants = self._detect_variants()
        
        if not variants:
            print(f"Error: No SRT files found matching 'chunk_0000*.srt' in {self.srt_folder}")
            return

        print(f"Detected variants: {', '.join(variants)}")
        for suffix in variants:
            self._stitch_variant(suffix)

    def _detect_variants(self):
        first_id = self.chunks[0]['id']
        pattern = f"chunk_{first_id:04d}*.srt"
        files = list(self.srt_folder.glob(pattern))
        return [f.name.replace(f"chunk_{first_id:04d}", "") for f in files]

    def _stitch_variant(self, suffix):
        output_filename = f"{self.source_video_name}{suffix}"
        output_path = self.output_folder / output_filename
        print(f"Generating: {output_filename}...")
        
        global_counter = 1
        all_lines = []

        for chunk in self.chunks:
            chunk_filename = f"chunk_{chunk['id']:04d}{suffix}"
            chunk_path = self.srt_folder / chunk_filename
            
            if not chunk_path.exists():
                print(f"  [Warning] Missing {chunk_filename}")
                continue
            
            srt_blocks = self._parse_and_shift(chunk_path, chunk['global_start'])
            for start, end, text in srt_blocks:
                all_lines.append(f"{global_counter}\n{start} --> {end}\n{text}\n\n")
                global_counter += 1

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(all_lines)
        print(f"  -> Saved {global_counter-1} subtitles.")

    def _parse_and_shift(self, filepath, offset):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2}[,.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,.]\d{3})\n((?:.|\n)*?)(?=\n\d+\n|\Z)', re.MULTILINE)
        matches = pattern.findall(content)
        result = []
        for _, s_raw, e_raw, text in matches:
            result.append((self._add_offset(s_raw, offset), self._add_offset(e_raw, offset), text.strip()))
        return result

    def _add_offset(self, time_str, offset_seconds):
        time_str = time_str.replace('.', ',')
        h, m, s_part = time_str.split(':')
        s, ms = s_part.split(',')
        total = (int(h)*3600) + (int(m)*60) + int(s) + (int(ms)/1000.0)
        new_total = total + offset_seconds
        
        new_h = int(new_total // 3600)
        rem = new_total % 3600
        new_m = int(rem // 60)
        s_float = rem % 60
        return f"{new_h:02d}:{new_m:02d}:{int(s_float):02d},{int((s_float-int(s_float))*1000):03d}"

# ==========================================
# PART 3: The CLI Controller
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VBR Audio Drift Fixer Suite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Subcommand: CHUNK ---
    chunk_parser = subparsers.add_parser("chunk", help="Split video into audio chunks based on scenes")
    chunk_parser.add_argument("input_video", help="Input video file")
    chunk_parser.add_argument("-o", "--output", default="chunks", help="Output directory (default: chunks)")
    chunk_parser.add_argument("--min", type=float, default=5.0, help="Min chunk length (seconds)")
    chunk_parser.add_argument("--max", type=float, default=30.0, help="Max chunk length (seconds)")
    chunk_parser.add_argument("--thresh", type=float, default=0.35, help="Scene detection threshold (0.0-1.0)")

    # --- Subcommand: STITCH ---
    stitch_parser = subparsers.add_parser("stitch", help="Merge SRT chunks back into master file")
    stitch_parser.add_argument("manifest", help="Path to manifest.json")
    stitch_parser.add_argument("srt_folder", help="Folder containing chunk SRT files")
    stitch_parser.add_argument("-o", "--output", help="Output folder for master SRTs")
    stitch_parser.add_argument("-n", "--name", help="Override video name for output filename")

    args = parser.parse_args()

    try:
        if args.command == "chunk":
            if not os.path.exists(args.input_video):
                print(f"Error: Video '{args.input_video}' not found.")
                sys.exit(1)
            chunker = RobustSceneChunker(
                args.input_video, args.output, args.min, args.max, args.thresh
            )
            chunker.run()

        elif args.command == "stitch":
            if not os.path.exists(args.manifest):
                print(f"Error: Manifest '{args.manifest}' not found.")
                sys.exit(1)
            stitcher = SRTStitcher(
                args.manifest, args.srt_folder, args.output, args.name
            )
            stitcher.run()

    except KeyboardInterrupt:
        print("\nAborted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")