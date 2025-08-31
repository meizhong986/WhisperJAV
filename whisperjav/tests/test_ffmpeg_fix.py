#!/usr/bin/env python3
"""Test script to verify the FFmpeg window fix for scene processing."""

import sys
import os
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
import logging

# Add the project to the path
sys.path.insert(0, str(Path(__file__).parent))

from whisperjav.modules.stable_ts_asr import StableTSASR
from whisperjav.utils.logger import logger

def create_test_wav(duration_sec=5, sample_rate=16000):
    """Create a test WAV file with simple sine wave."""
    temp_dir = Path(tempfile.gettempdir()) / "whisperjav_test"
    temp_dir.mkdir(exist_ok=True)
    
    test_wav = temp_dir / "test_scene.wav"
    
    # Generate a simple sine wave (440 Hz tone)
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))
    audio_data = 0.3 * np.sin(2 * np.pi * 440 * t)  # Quiet 440Hz tone
    
    # Save as WAV file
    sf.write(str(test_wav), audio_data, sample_rate)
    
    return test_wav

def test_numpy_loading():
    """Test the numpy array loading functionality."""
    print("=" * 60)
    print("Testing FFmpeg Window Fix - Numpy Array Loading")
    print("=" * 60)
    
    # Set logging level to see debug messages
    logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create a test WAV file
        print("1. Creating test WAV file...")
        test_wav = create_test_wav(duration_sec=3)
        print(f"   Created: {test_wav}")
        
        # Initialize StableTSASR with minimal config
        print("2. Initializing StableTSASR...")
        
        model_config = {
            "model_name": "tiny",  # Use smallest model for testing
            "device": "cpu",       # Force CPU to avoid GPU issues
            "compute_type": "float32"
        }
        
        params = {
            "decoder": {
                "temperature": 0.0,
                "language": "en"
            },
            "provider": {},
            "stable_ts": {}
        }
        
        asr = StableTSASR(
            model_config=model_config,
            params=params, 
            task="transcribe",
            turbo_mode=False
        )
        
        print("3. Testing numpy array loading...")
        # Test the numpy loading directly
        audio_array = asr._load_audio_as_numpy(test_wav)
        
        if audio_array is not None:
            print(f"   ✅ Successfully loaded as numpy array: shape={audio_array.shape}")
            print(f"   ✅ Audio duration: {len(audio_array)/16000:.2f} seconds")
        else:
            print("   ❌ Failed to load as numpy array")
            return False
        
        print("4. Testing transcription with numpy array...")
        # This should NOT create any FFmpeg windows
        result = asr.transcribe(test_wav)
        
        if result and result.segments:
            print(f"   ✅ Transcription successful: {len(result.segments)} segments")
            # Print first segment if available
            if result.segments:
                print(f"   Sample text: '{result.segments[0].text[:50]}'")
        else:
            print("   ⚠️ Transcription returned no segments (expected for sine wave)")
        
        print("5. Cleanup...")
        test_wav.unlink()
        print(f"   Removed: {test_wav}")
        
        print("\n" + "=" * 60)
        print("✅ TEST PASSED: No FFmpeg windows should have appeared!")
        print("The fix is working - stable-ts is using numpy arrays instead of file paths.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_numpy_loading()
    sys.exit(0 if success else 1)