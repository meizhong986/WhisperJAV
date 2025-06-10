#!/usr/bin/env python3
"""Balanced pipeline implementation."""

from typing import Dict
from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.utils.logger import logger


class BalancedPipeline(BasePipeline):
    """Balanced pipeline with full preprocessing."""
    
    def get_mode_name(self) -> str:
        return "balanced"
    
    def process(self, media_info: Dict) -> Dict:
        """
        Process media file through balanced pipeline.
        
        Args:
            media_info: Dictionary containing discovered media information
            
        Returns:
            Processing metadata dictionary
        """
        # Extract info from the media_info dictionary
        input_file = media_info['path']
        media_basename = media_info['basename']
        
        logger.info(f"BALANCED pipeline not yet implemented for: {input_file}")
        raise NotImplementedError("Balanced pipeline is coming soon")