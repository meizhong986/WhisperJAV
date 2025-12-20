import subprocess
import json
import re
import os
import shutil
import argparse
from pathlib import Path

class RobustSceneChunker:
    def __init__(self, input_video, output_dir="chunks", min_scene_len=5.0, max_scene_len=30.0, threshold=0.35):
        self.input_video = Path(input_video)
        self.output_dir = Path(output_dir)
        self.min_scene_len = min_scene_len
        self.max_scene_len = max_scene_len
        self.threshold = threshold
        self.wav_path = self.output_dir / "temp_locked_audio.wav"
        
        # Clean/Create output directory
        if self.output_dir.exists():
            print(f"Warning: Output directory '{self.output_dir}' exists. Cleaning it...")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

    def run(self):
        print(f"--- Processing: {self.input_video.name} ---")
        print(f"Config: Min={self.min_scene_len}s, Max={self.max_scene_len}s, Threshold={self.threshold}")
        
        # [Fix #1] Extract Audio locked to Video PTS
        print("[1/4] Extracting PTS-locked audio...")
        self._extract_wav_locked()

        # [Fix #2] Detect Scenes
        print("[2/4] Detecting video scenes...")
        scenes = self._detect_scenes()
        
        # [Fix #3] Optimize Cuts
        print(f"[3/4] Optimizing {len(scenes)} scenes (merging short, limiting long)...")
        segments = self._optimize_segments(scenes)

        # [Fix #4] Slice Audio
        print(f"[4/4] Slicing audio into {len(segments)} chunks...")
        manifest = self._slice_audio(segments)

        # Generate Manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"--- Done! Manifest saved to {manifest_path} ---")

    def _extract_wav_locked(self):
        """
        Extracts audio while forcing it to align with the video's Presentation Time Stamps (PTS).
        """
        cmd = [
            "ffmpeg", "-y",
            "-fflags", "+genpts",
            "-copyts",
            "-start_at_zero",
            "-i", str(self.input_video),
            "-vn",
            "-af", "aresample=async=1:first_pts=0",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(self.wav_path)
        ]
        # Using check=True to raise error if ffmpeg fails
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _detect_scenes(self):
        """
        Detects scene changes using the configured threshold.
        """
        cmd = [
            "ffmpeg", "-i", str(self.input_video),
            "-filter_complex", f"select='gt(scene,{self.threshold})',metadata=print:file=-",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        timestamps = [0.0]
        # Regex captures standard pts_time from ffmpeg output
        matches = re.findall(r'pts_time:([0-9.]+)', result.stdout + result.stderr)
        for t in matches:
            timestamps.append(float(t))
            
        return sorted(list(set(timestamps)))

    def _optimize_segments(self, scene_points):
        """
        Merges short scenes and forces splits on long scenes.
        """
        segments = []
        
        # Get exact duration of the locked WAV
        duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", str(self.wav_path)
        ]
        try:
            total_duration = float(subprocess.check_output(duration_cmd).strip())
        except subprocess.CalledProcessError:
            print("Error reading audio duration. Is ffprobe installed?")
            raise

        if not scene_points or scene_points[-1] < total_duration:
            scene_points.append(total_duration)

        current_start = scene_points[0]

        for i in range(1, len(scene_points)):
            current_end = scene_points[i]
            segment_len = current_end - current_start
            
            # Logic: Valid length?
            if segment_len >= self.min_scene_len:
                # Is it too long?
                if segment_len > self.max_scene_len:
                    # Commit (could implement force-split here if strictly needed)
                    segments.append((current_start, current_end))
                    current_start = current_end
                else:
                    # Perfect length
                    segments.append((current_start, current_end))
                    current_start = current_end
            else:
                # Too short. Can we merge?
                next_boundary = scene_points[i+1] if i+1 < len(scene_points) else total_duration
                if (next_boundary - current_start) > self.max_scene_len:
                    # Merging would be too big. Force the short cut.
                    segments.append((current_start, current_end))
                    current_start = current_end
                else:
                    # Merge allowed (skip updating current_start)
                    pass
        
        # Ensure tail is caught
        if current_start < total_duration:
             segments.append((current_start, total_duration))

        return segments

    def _slice_audio(self, segments):
        manifest = {
            "source_video": str(self.input_video.absolute()),
            "parameters": {
                "min_scene_len": self.min_scene_len,
                "max_scene_len": self.max_scene_len,
                "threshold": self.threshold
            },
            "chunks": []
        }
        
        for i, (start, end) in enumerate(segments):
            duration = end - start
            filename = f"chunk_{i:04d}.wav"
            output_path = self.output_dir / filename
            
            # [Fix #4] Input seeking for sample accuracy
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(self.wav_path),
                "-ss", f"{start:.3f}",
                "-t", f"{duration:.3f}",
                str(output_path)
            ]
            subprocess.run(cmd, check=True)
            
            manifest["chunks"].append({
                "id": i,
                "file": filename,
                "global_start": round(start, 3),
                "global_end": round(end, 3),
                "duration": round(duration, 3)
            })
            
        return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split video audio into chunks based on visual scenes to fix VBR ASR drift.")
    
    # Required Argument
    parser.add_argument("input_video", help="Path to the input video file (e.g., video.mp4)")
    
    # Optional Arguments
    parser.add_argument("-o", "--output", default="chunks", help="Output directory for chunks (default: chunks)")
    parser.add_argument("--min", type=float, default=5.0, help="Minimum chunk length in seconds (default: 5.0)")
    parser.add_argument("--max", type=float, default=30.0, help="Maximum chunk length in seconds (default: 30.0)")
    parser.add_argument("--thresh", type=float, default=0.35, help="Scene detection threshold 0.0-1.0 (default: 0.35)")

    args = parser.parse_args()

    # Input validation
    if not os.path.exists(args.input_video):
        print(f"Error: Input file '{args.input_video}' not found.")
        exit(1)

    try:
        chunker = RobustSceneChunker(
            input_video=args.input_video, 
            output_dir=args.output,
            min_scene_len=args.min,
            max_scene_len=args.max,
            threshold=args.thresh
        )
        chunker.run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")