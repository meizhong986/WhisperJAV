#!/usr/bin/env python3
"""Metadata manager for WhisperJAV - handles master and chunk JSON files."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import numpy as np

def convert_numpy_for_json(obj):
    """Convert numpy objects to JSON-serializable types."""
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None 
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, Path):
        return str(obj)
    return str(obj)

class MetadataManager:
    """Manages master and chunk metadata for WhisperJAV processing."""
    
    def __init__(self, temp_dir: Path = Path("./temp"), output_dir: Path = Path("./output")):
        self.temp_dir = Path(temp_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_master_metadata(self, 
                             input_file: str,
                             mode: str,
                             job_id: Optional[str] = None) -> Dict:
        """Create initial master metadata structure."""
        if job_id is None:
            job_id = f"whisperjav_{int(time.time())}"
            
        return {
            "metadata_master": {
                "structure_version": "1.0.0",
                "whisperjav_version": "1.0.0",
                "job_id": job_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z"
            },
            "input_info": {
                "original_input_file": input_file,
                "processed_audio_file": None,
                "audio_duration_seconds": None
            },
            "config": {
                "mode": mode,
                "chunking_params": {
                    "min_dur": 0.1,
                    "max_dur": 900.0,
                    "max_silence": 2.0,
                    "energy": 50
                },
                "pipeline_options": {}
            },
            "chunks_generated": [],
            "output_files": {},
            "processing_stages": {},
            "summary": {
                "total_processing_time_seconds": 0,
                "total_chunks_created": 0,
                "chunks_processed_successfully": 0,
                "final_subtitles_refined": 0,
                "final_subtitles_raw": 0,
                "classification_distribution": {},
                "quality_metrics": {}
            }
        }
    
    def create_chunk_metadata(self,
                            chunk_index: int,
                            chunk_filename: str,
                            start_seconds: float,
                            end_seconds: float) -> Dict:
        """Create initial chunk metadata structure."""
        return {
            "metadata_chunk": {
                "structure_version": "1.0.0",
                "analyzer_version": "1.0.0",
                "analysis_timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "chunk_info": {
                "chunk_id": f"chunk_{chunk_index:04d}",
                "chunk_index": chunk_index,
                "start_time_seconds": start_seconds,
                "end_time_seconds": end_seconds,
                "duration_seconds": end_seconds - start_seconds,
                "audio_path": chunk_filename
            },
            "acoustic_analysis": {},
            "processing_pipeline": {
                "preprocessing_applied": [],
                "asr_parameters": {},
                "asr_output": {}
            }
        }
    
    def save_master_metadata(self, metadata: Dict, media_basename: str):
        """Save master metadata to JSON file."""
        file_path = self.temp_dir / f"{media_basename}_master.json"
        metadata["metadata_master"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=convert_numpy_for_json)
            
    def load_master_metadata(self, media_basename: str) -> Optional[Dict]:
        """Load master metadata from JSON file."""
        file_path = self.temp_dir / f"{media_basename}_master.json"
        if not file_path.exists():
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_chunk_metadata(self, metadata: Dict, media_basename: str, chunk_index: int):
        """Save chunk metadata to JSON file."""
        file_path = self.temp_dir / f"{media_basename}_chunk_{chunk_index:04d}_profile.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=convert_numpy_for_json)
    
    def load_chunk_metadata(self, media_basename: str, chunk_index: int) -> Optional[Dict]:
        """Load chunk metadata from JSON file."""
        file_path = self.temp_dir / f"{media_basename}_chunk_{chunk_index:04d}_profile.json"
        if not file_path.exists():
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_processing_stage(self, 
                              master_metadata: Dict,
                              stage_name: str,
                              status: str,
                              **kwargs):
        """Update processing stage in master metadata."""
        if "processing_stages" not in master_metadata:
            master_metadata["processing_stages"] = {}
            
        master_metadata["processing_stages"][stage_name] = {
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs
        }
        
    def add_chunk_to_master(self,
                          master_metadata: Dict,
                          chunk_info: Dict):
        """Add chunk information to master metadata."""
        master_metadata["chunks_generated"].append(chunk_info)
        master_metadata["summary"]["total_chunks_created"] = len(master_metadata["chunks_generated"])