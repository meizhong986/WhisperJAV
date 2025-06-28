#!/usr/bin/env python3
"""
Configuration Integration for WhisperJAV GUI
This module helps connect the GUI to actual WhisperJAV configuration
"""

import sys
from pathlib import Path

# Add WhisperJAV to path if needed
whisperjav_path = Path(__file__).parent.parent
if str(whisperjav_path) not in sys.path:
    sys.path.insert(0, str(whisperjav_path))

try:
    from whisperjav.config.transcription_tuner import TranscriptionTuner
    WHISPERJAV_AVAILABLE = True
except ImportError:
    WHISPERJAV_AVAILABLE = False
    print("Warning: WhisperJAV not found. Using mock configuration.")

class ConfigurationProvider:
    """Provides real configuration values to the GUI"""
    
    def __init__(self):
        self.tuner = TranscriptionTuner() if WHISPERJAV_AVAILABLE else None
        
    def get_config_for_selections(self, mode, sensitivity):
        """Get actual configuration based on user selections"""
        if self.tuner:
            try:
                params = self.tuner.get_resolved_params(mode, sensitivity)
                return {
                    'vad_options': params.get('vad_options', {}),
                    'transcribe_options': params.get('transcribe_options', {}),
                    'decode_options': params.get('decode_options', {})
                }
            except Exception as e:
                print(f"Error getting config: {e}")
                
        # Return mock data if WhisperJAV not available
        return self._get_mock_config(mode, sensitivity)
        
    def _get_mock_config(self, mode, sensitivity):
        """Mock configuration for testing"""
        return {
            'vad_options': {
                'threshold': 0.5,
                'min_speech_duration_ms': 250,
                'max_speech_duration_s': float('inf'),
                'min_silence_duration_ms': 2000,
                'speech_pad_ms': 400
            },
            'transcribe_options': {
                'temperature': 0.0,
                'compression_ratio_threshold': 2.4,
                'logprob_threshold': -1.0,
                'no_speech_threshold': 0.6,
                'condition_on_previous_text': True,
                'initial_prompt': None
            },
            'decode_options': {
                'task': 'transcribe',
                'language': 'ja',
                'beam_size': 5,
                'best_of': 5,
                'patience': 1.0,
                'length_penalty': 1.0,
                'suppress_tokens': '-1',
                'suppress_blank': True,
                'without_timestamps': False
            }
        }

# Enhanced GUI methods to use real configuration
def enhance_gui_with_real_config(gui_class):
    """Enhance the GUI class to use real WhisperJAV configuration"""
    
    # Store original methods
    original_init = gui_class.__init__
    original_get_vad = gui_class.get_vad_settings
    original_get_transcribe = gui_class.get_transcribe_settings
    original_get_decode = gui_class.get_decode_settings
    
    def new_init(self, root):
        original_init(self, root)
        # Add configuration provider
        self.config_provider = ConfigurationProvider()
        
    def new_get_vad_settings(self, mode, sensitivity):
        if hasattr(self, 'config_provider'):
            config = self.config_provider.get_config_for_selections(mode, sensitivity)
            return config['vad_options']
        return original_get_vad(self, mode, sensitivity)
        
    def new_get_transcribe_settings(self, mode, sensitivity):
        if hasattr(self, 'config_provider'):
            config = self.config_provider.get_config_for_selections(mode, sensitivity)
            return config['transcribe_options']
        return original_get_transcribe(self, mode, sensitivity)
        
    def new_get_decode_settings(self, mode, sensitivity):
        if hasattr(self, 'config_provider'):
            config = self.config_provider.get_config_for_selections(mode, sensitivity)
            return config['decode_options']
        return original_get_decode(self, mode, sensitivity)
    
    # Replace methods
    gui_class.__init__ = new_init
    gui_class.get_vad_settings = new_get_vad_settings
    gui_class.get_transcribe_settings = new_get_transcribe_settings
    gui_class.get_decode_settings = new_get_decode_settings
    
    return gui_class

# Example usage in whisperjav_gui.py:
"""
# At the top of whisperjav_gui.py, after imports:
try:
    from gui_config_integration import enhance_gui_with_real_config
    WhisperJAVGUI = enhance_gui_with_real_config(WhisperJAVGUI)
except ImportError:
    pass  # Use mock config if integration not available
"""

if __name__ == "__main__":
    # Test configuration provider
    provider = ConfigurationProvider()
    
    print("Testing Configuration Provider")
    print("=" * 50)
    
    for mode in ['faster', 'fast', 'balanced']:
        for sensitivity in ['conservative', 'balanced', 'aggressive']:
            print(f"\nMode: {mode}, Sensitivity: {sensitivity}")
            config = provider.get_config_for_selections(mode, sensitivity)
            
            # Show sample values
            vad = config['vad_options']
            print(f"  VAD Threshold: {vad.get('threshold', 'N/A')}")
            
            decode = config['decode_options']
            print(f"  Beam Size: {decode.get('beam_size', 'N/A')}")
            print(f"  Best Of: {decode.get('best_of', 'N/A')}")
            
    print("\n" + "=" * 50)
    print("WhisperJAV Available:", WHISPERJAV_AVAILABLE)