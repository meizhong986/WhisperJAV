#!/usr/bin/env python3
"""Media file discovery and handling for WhisperJAV."""

from pathlib import Path
from typing import List, Union
import glob
from whisperjav.utils.logger import logger

class MediaDiscovery:
    """Handle media file discovery with wildcard support."""
    
    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', 
                          '.webm', '.m4v', '.mpg', '.mpeg', '.3gp', '.mp3', 
                          '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
    
    def __init__(self):
        pass
        
    def discover_media_files(self, input_path: Union[str, List[str]]) -> List[Path]:
        """Discover media files from input path(s) with wildcard support."""
        if isinstance(input_path, str):
            input_paths = [input_path]
        else:
            input_paths = input_path
            
        discovered_files = []
        
        for path_pattern in input_paths:
            # Handle wildcards
            if '*' in path_pattern or '?' in path_pattern:
                matched_files = glob.glob(path_pattern, recursive=True)
                for file_path in matched_files:
                    if self._is_media_file(file_path):
                        discovered_files.append(Path(file_path))
            else:
                path = Path(path_pattern)
                if path.is_file() and self._is_media_file(path):
                    discovered_files.append(path)
                elif path.is_dir():
                    # Search directory for media files
                    for ext in self.SUPPORTED_EXTENSIONS:
                        discovered_files.extend(path.glob(f"*{ext}"))
                        discovered_files.extend(path.glob(f"*{ext.upper()}"))
                        
        # Remove duplicates and sort
        discovered_files = sorted(list(set(discovered_files)))
        
        logger.info(f"Discovered {len(discovered_files)} media files")
        for file in discovered_files:
            logger.debug(f"  - {file}")
            
        return discovered_files
        
    def _is_media_file(self, file_path: Union[str, Path]) -> bool:
        """Check if a file is a supported media file."""
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS