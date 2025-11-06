#!/usr/bin/env python3
"""Media discovery module for analyzing input media files."""

from pathlib import Path
from typing import Dict, Optional, List
import subprocess
import json
import glob

from whisperjav.utils.logger import logger


class MediaDiscovery:
    """Discovers and analyzes media file properties."""
    
    def __init__(self):
        """Initialize media discovery module."""
        self.ffprobe_available = self._check_ffprobe()
        
        # Supported extensions
        self.video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}
        self.audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus'}
        self.supported_extensions = self.video_extensions | self.audio_extensions
    
    def _check_ffprobe(self) -> bool:
        """Check if ffprobe is available."""
        try:
            subprocess.run(['ffprobe', '-version'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("ffprobe not found. Media discovery will be limited.")
            return False
    
    def discover(self, input_paths: List[str]) -> List[Dict[str, Optional[str]]]:
        """
        Discover all valid media files from a list of paths, directories, or glob patterns.
        
        Args:
            input_paths: List of input paths (files, directories, or wildcards)
            
        Returns:
            List of media file information dictionaries
        """
        processed_paths = set()  # Track processed files to avoid duplicates
        final_media_list = []
        
        for path_pattern in input_paths:
            # Use glob to expand the pattern (handles files, dirs, and wildcards)
            # recursive=True allows patterns like **/*.mp4
            expanded_paths = glob.glob(path_pattern, recursive=True)
            
            # If glob didn't find anything, try treating it as a literal path
            if not expanded_paths and Path(path_pattern).exists():
                expanded_paths = [path_pattern]
            
            for path_str in expanded_paths:
                path = Path(path_str)
                
                if path.is_dir():
                    # If it's a directory, recursively find all media files in it
                    for item in path.rglob('*'):
                        if item.is_file() and item.suffix.lower() in self.supported_extensions:
                            # Use resolve() to get canonical path and prevent duplicates
                            canonical_path = item.resolve()
                            if canonical_path not in processed_paths:
                                media_info = self._analyze_file(str(item))
                                if media_info['type'] in ['video', 'audio']:
                                    final_media_list.append(media_info)
                                    processed_paths.add(canonical_path)
                
                elif path.is_file():
                    # If it's a file, check if it's a supported media file
                    if path.suffix.lower() in self.supported_extensions:
                        canonical_path = path.resolve()
                        if canonical_path not in processed_paths:
                            media_info = self._analyze_file(str(path))
                            if media_info['type'] in ['video', 'audio']:
                                final_media_list.append(media_info)
                                processed_paths.add(canonical_path)
                    else:
                        logger.warning(f"Unsupported file extension: {path}")
                        
        if not final_media_list:
            logger.warning(f"No valid media files found for input patterns: {input_paths}")
            
        return final_media_list
    
    def _analyze_file(self, file_path: str) -> Dict[str, Optional[str]]:
        """
        Analyze a single media file's properties.
        
        Args:
            file_path: Path to media file
            
        Returns:
            Dictionary with media information
        """
        path = Path(file_path)
        
        # Basic info
        info = {
            'path': str(path.absolute()),
            'basename': path.stem,
            'extension': path.suffix.lower(),
            'exists': path.exists(),
            'size_bytes': path.stat().st_size if path.exists() else 0
        }
        
        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            info['type'] = 'unknown'
            info['error'] = 'File not found'
            return info
        
        # Determine media type from extension
        if info['extension'] in self.video_extensions:
            info['type'] = 'video'
        elif info['extension'] in self.audio_extensions:
            info['type'] = 'audio'
        else:
            info['type'] = 'unknown'
        
        # Get detailed info using ffprobe if available
        if self.ffprobe_available:
            detailed_info = self._get_ffprobe_info(str(path))
            if detailed_info:
                info.update(detailed_info)
        
        return info
    
    def _get_ffprobe_info(self, file_path: str) -> Optional[Dict]:
        """Get detailed media info using ffprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', check=True)
            data = json.loads(result.stdout)
            
            info = {}
            
            # Get format info
            if 'format' in data:
                fmt = data['format']
                info['format'] = fmt.get('format_name', 'unknown')
                info['duration'] = float(fmt.get('duration', 0))
                info['bit_rate'] = int(fmt.get('bit_rate', 0))
            
            # Get stream info
            if 'streams' in data:
                audio_streams = [s for s in data['streams'] if s.get('codec_type') == 'audio']
                video_streams = [s for s in data['streams'] if s.get('codec_type') == 'video']
                
                info['audio_streams'] = len(audio_streams)
                info['video_streams'] = len(video_streams)
                
                # Get first audio stream info
                if audio_streams:
                    audio = audio_streams[0]
                    info['audio_codec'] = audio.get('codec_name', 'unknown')
                    info['audio_sample_rate'] = int(audio.get('sample_rate', 0))
                    info['audio_channels'] = int(audio.get('channels', 0))
                
                # Get first video stream info
                if video_streams:
                    video = video_streams[0]
                    info['video_codec'] = video.get('codec_name', 'unknown')
                    info['video_width'] = int(video.get('width', 0))
                    info['video_height'] = int(video.get('height', 0))
                    info['video_fps'] = eval(video.get('r_frame_rate', '0/1'))
            
            return info
            
        except Exception as e:
            logger.debug(f"ffprobe failed for {file_path}: {e}")
            return None